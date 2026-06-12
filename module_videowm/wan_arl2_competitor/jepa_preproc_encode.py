"""§3a REPA-loss preprocessing STEP 2: encode each decoded target mp4 with the FROZEN V-JEPA2 ViT-g
ENCODER (not the AC predictor) -> per-patch target features for REPA alignment.

Why these choices (see JEPA_LOSS_NOTES.md):
  * Encoder = vjepa2_ac_vit_giant's ENCODER (vit_giant_xformers_rope, embed_dim=1408, patch16/tubelet2, RoPE).
    Weights come from the AC ckpt's ["encoder"] (the M1 asset). use_sdpa=True -> no xformers needed.
  * Spatial: feed the FULL 480x832 frame (NO resize/crop) so the patch grid is 30x52, which EXACTLY matches
    Wan's DiT spatial grid (60/2=30, 104/2=52). RoPE + handle_nonsquare_inputs=True let the ViT take this
    out-of-train-resolution input; spatial alignment with Wan is then 1:1 (no spatial interpolation needed).
  * Temporal: uniformly subsample the 81 decoded frames -> 64 (the encoder's native num_frames) so the temporal
    RoPE stays IN-distribution (64 frames / tubelet 2 = 32 tubelets). We then mean-pool the 32 tubelets into
    T_VJEPA temporal groups (start..end dynamics anchors) before storing, to fit disk.
  * Spatial: after encoding at the full 30x52 patch grid, AVG-POOL 2x2 -> 15x26 before storing. This trades
    a little spatial resolution for 4x temporal headroom on disk, letting us raise T_VJEPA to 8. The Wan DiT
    hidden is pooled the SAME 2x2 way in the REPA loss, so spatial alignment stays 1:1 at the 15x26 grid.
  * Storage: [T_VJEPA, 15, 26, 1408] fp16 per clip. T_VJEPA=8 -> 8.79 MB/clip -> ~52.7 GB for 6000 clips
    (8*15*26 == 2*30*52, so T=8@15x26 costs the SAME disk as the old T=2@30x52 — but 4x the temporal slices).
    RUN-2 FIX: run-1 used T_VJEPA=2 @ full 30x52; interpolating 2->21 made the target temporally SMOOTH, which
    pushed the student's per-frame features alike across time and SUPPRESSED motion (VBench dynamic_degree -5.55).
    More temporal slices (T=8) keep the target's frame-to-frame variation so motion is no longer suppressed.
    The REPA loss later F.interpolate's the stored T_VJEPA -> 21 (Wan's frame count) on the time axis only.

2-GPU sharded:  CUDA_VISIBLE_DEVICES=0 python jepa_preproc_encode.py 0 2   (and `1 2` on the other GPU).
Self-test one clip + report shapes/mem:  CUDA_VISIBLE_DEVICES=0 python jepa_preproc_encode.py 0 1 --probe
"""
import os, sys, glob, time, argparse
sys.path.insert(0, "/home/wm-vla-rl/_ext/vjepa2")
import torch
import torch.nn.functional as F
import numpy as np
import imageio.v2 as imageio
from src.models.vision_transformer import vit_giant_xformers_rope

SRC = "/home/wan_distill_data_mp4"
OUT = "/home/wan_jepa_feats"
CKPT = "/home/.cache/torch/hub/checkpoints/vjepa2-ac-vitg.pt"   # full AC ckpt; we load ["encoder"]
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1, 1)
NUM_FRAMES = int(os.environ.get("JEPA_NUM_FRAMES", 64))   # encoder native temporal length (64 -> 32 tubelets)
T_VJEPA = int(os.environ.get("JEPA_T", 8))                # stored temporal groups (mean-pool of the tubelets); run-2: 8
H, W = 480, 832        # full res -> 30x52 patch grid == Wan spatial grid
GRID_H, GRID_W = H // 16, W // 16   # 30, 52
POOL = int(os.environ.get("JEPA_SPATIAL_POOL", 2))        # spatial 2x2 avg-pool of the patch grid: 30x52 -> 15x26
POOL_H, POOL_W = GRID_H // POOL, GRID_W // POOL            # 15, 26 (Wan DiT hidden pooled the same way in the loss)


def ts():
    return time.strftime("%H:%M:%S")


def load_encoder(device):
    enc = vit_giant_xformers_rope(img_size=(256, 256), num_frames=NUM_FRAMES, tubelet_size=2, patch_size=16)
    blob = torch.load(CKPT, map_location="cpu", weights_only=True)
    sd = blob["encoder"]
    sd = {k.replace("module.", "").replace("backbone.", ""): v for k, v in sd.items()}
    msg = enc.load_state_dict(sd, strict=False)
    miss = [k for k in msg.missing_keys if "pos_embed" not in k]   # pos_embed unused on the RoPE path
    print(f"[enc] loaded encoder: missing={len(msg.missing_keys)} (non-posembed {len(miss)}) "
          f"unexpected={len(msg.unexpected_keys)}", flush=True)
    assert len(miss) == 0, f"unexpected MISSING encoder keys: {miss[:6]}"
    enc = enc.to(device).to(torch.bfloat16).eval()
    for p in enc.parameters():
        p.requires_grad_(False)
    return enc


def read_clip(path):
    """mp4 -> float tensor [1,3,64,480,832], imagenet-normalized (uniform 81->64 temporal subsample)."""
    rdr = imageio.get_reader(path, "ffmpeg")
    frames = [f for f in rdr]
    rdr.close()
    arr = np.stack(frames)                       # [T0,H,W,3] uint8
    T0 = arr.shape[0]
    idx = torch.linspace(0, T0 - 1, NUM_FRAMES).round().long().clamp(0, T0 - 1)
    v = torch.from_numpy(arr).float()[idx] / 255.0            # [64,H,W,3]
    v = v.permute(3, 0, 1, 2).unsqueeze(0)                    # [1,3,64,H,W]
    if v.shape[-2:] != (H, W):
        v = F.interpolate(v.flatten(0, 1), size=(H, W), mode="bilinear", align_corners=False).unflatten(0, (1, 3))
    v = (v - IMAGENET_MEAN) / IMAGENET_STD
    return v


@torch.inference_mode()
def encode(enc, v, device):
    """[1,3,64,480,832] -> stored target [T_VJEPA,15,26,1408] fp16 (cpu).
    Pipeline: ViT-g encode @30x52 -> spatial 2x2 avg-pool to 15x26 -> temporal mean-pool 32 tubelets -> T_VJEPA."""
    z = enc(v.to(device, torch.bfloat16))                    # [1, 32*30*52, 1408]
    N = z.shape[1]
    Tt = N // (GRID_H * GRID_W)
    assert Tt * GRID_H * GRID_W == N, f"token count {N} not divisible by {GRID_H}x{GRID_W}"
    z = z.reshape(Tt, GRID_H, GRID_W, -1).float()            # [32,30,52,1408] (T,H,W row-major == Wan order)
    # spatial 2x2 avg-pool: [Tt,30,52,D] -> [Tt,15,26,D] (same pooling applied to the Wan hidden in repa_align_loss)
    z = F.avg_pool2d(z.permute(0, 3, 1, 2), kernel_size=POOL, stride=POOL).permute(0, 2, 3, 1).contiguous()
    assert Tt % T_VJEPA == 0, f"tubelets {Tt} not divisible by T_VJEPA {T_VJEPA}"
    pooled = z.reshape(T_VJEPA, Tt // T_VJEPA, POOL_H, POOL_W, z.shape[-1]).mean(1)   # [T_VJEPA,15,26,1408]
    return pooled.half().contiguous().cpu()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("shard", type=int)
    ap.add_argument("nshard", type=int)
    ap.add_argument("--probe", action="store_true", help="encode ONE clip, print shapes/mem, do not write")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    device = "cuda"
    enc = load_encoder(device)

    files = sorted(glob.glob(os.path.join(SRC, "clip_*.mp4")))
    mine = files[args.shard::args.nshard]
    print(f"[enc shard {args.shard}/{args.nshard} {ts()}] {len(mine)}/{len(files)} clips -> {OUT}", flush=True)

    if args.probe:
        f = mine[0]
        v = read_clip(f)
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        feat = encode(enc, v, device)
        dt = time.time() - t0
        print(f"[probe] {os.path.basename(f)} in {v.shape} -> feat {tuple(feat.shape)} {feat.dtype} "
              f"{feat.numel()*2/1e6:.2f} MB  {dt:.2f}s  peak {torch.cuda.max_memory_allocated()/1e9:.1f} GB",
              flush=True)
        print(f"[probe] est total = {len(files)*feat.numel()*2/1e9:.1f} GB for {len(files)} clips", flush=True)
        return

    done = 0; t0 = time.time()
    for i, f in enumerate(mine):
        name = os.path.splitext(os.path.basename(f))[0]
        outp = os.path.join(OUT, name + ".pt")
        if os.path.exists(outp):
            done += 1; continue
        try:
            feat = encode(enc, read_clip(f), device)
            tmp = outp + ".tmp"
            torch.save(feat, tmp)
            os.replace(tmp, outp)                            # atomic -> safe to resume on crash
            done += 1
        except Exception as e:
            print(f"[enc shard {args.shard} {ts()}] ERR {name}: {repr(e)}", flush=True)
        if (i + 1) % 50 == 0:
            rate = (i + 1) / (time.time() - t0)
            print(f"[enc shard {args.shard} {ts()}] {i+1}/{len(mine)}  {rate:.2f} clip/s  "
                  f"~{(len(mine)-i-1)/rate/60:.0f}min left", flush=True)
    print(f"[enc shard {args.shard} {ts()}] DONE {done}/{len(mine)} feats in {OUT}", flush=True)


if __name__ == "__main__":
    main()
