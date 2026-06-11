"""§3a JEPA-loss preprocessing STEP 1: VAE-decode the 6000 teacher clips -> RGB video -> mp4.
The mp4s are the fixed distillation TARGET frames; STEP 2 (separate vjepa2 venv) encodes them to
V-JEPA2 features. Decoding here (cosmos_predict2 venv, Wan VAE) avoids running the VAE in the train loop.
clip latent [21,16,60,104]=[T,C,H,W] -> permute [16,21,60,104] -> WanVAE.decode -> [3,81,480,832] in [-1,1].
2-GPU sharded:  CUDA_VISIBLE_DEVICES=0 python jepa_preproc_decode.py 0 2   (and `1 2` on the other GPU).
"""
import os, sys, glob, time
sys.path.insert(0, "/home/wm-vla-rl/_ext/Causal-Forcing")
import torch, numpy as np, imageio.v2 as imageio
from wan.modules.vae import WanVAE

SHARD = int(sys.argv[1]); NSHARD = int(sys.argv[2])
SRC = "/home/wan_distill_data"
OUT = "/home/wan_distill_data_mp4"          # ~9-18 GB total for 6000 clips
os.makedirs(OUT, exist_ok=True)
VAE_PTH = "/home/wan21_1.3b/Wan2.1_VAE.pth"

def ts(): return time.strftime("%H:%M:%S")
fs = sorted(glob.glob(os.path.join(SRC, "clip_*.pt")))
mine = fs[SHARD::NSHARD]
print(f"[decode shard {SHARD}/{NSHARD} {ts()}] {len(mine)}/{len(fs)} clips -> {OUT}", flush=True)
vae = WanVAE(z_dim=16, vae_pth=VAE_PTH, device="cuda")

done = 0; t0 = time.time()
for i, f in enumerate(mine):
    name = os.path.splitext(os.path.basename(f))[0]
    outp = os.path.join(OUT, name + ".mp4")
    if os.path.exists(outp):
        done += 1; continue
    try:
        d = torch.load(f, map_location="cpu", weights_only=False)
        z = d["latent"].to("cuda", torch.float32).permute(1, 0, 2, 3).contiguous()  # [16,21,60,104]
        with torch.no_grad():
            vid = vae.decode([z])[0]                      # [3,81,480,832] in [-1,1]
        vid = ((vid.clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8).cpu()         # [3,T,H,W]
        frames = vid.permute(1, 2, 3, 0).numpy()         # [T,H,W,3]
        imageio.mimwrite(outp, list(frames), fps=16, codec="libx264", quality=8, macro_block_size=8)
        done += 1
    except Exception as e:
        print(f"[decode shard {SHARD} {ts()}] ERR {name}: {e}", flush=True)
    if (i + 1) % 100 == 0:
        rate = (i + 1) / (time.time() - t0)
        print(f"[decode shard {SHARD} {ts()}] {i+1}/{len(mine)}  {rate:.1f} clip/s  ~{(len(mine)-i-1)/rate/60:.0f}min left", flush=True)
print(f"[decode shard {SHARD} {ts()}] DONE {done}/{len(mine)} mp4 in {OUT}", flush=True)
