"""cosmos_data.py — encode LIBERO video clips into Cosmos-Predict2-2B-Video2World latents + T5 text embeds
+ conditioning/padding masks, saved as .pt batches for linearization distillation:
  Stage-1 (attn-transfer): only needs realistic DiT-input activations (real latents + masks + text).
  Stage-2 (flow-matching finetune): needs clean latents as the regression target + cond frames.

Faithful to Cosmos2VideoToWorldPipeline:
  - VAE = AutoencoderKLWan (z_dim=16, temporal /4, spatial /8). latent = (z - latents_mean)/latents_std * sigma_data.
  - text = T5 (d_model=1024), tokenize pad->512, encoder.last_hidden_state, zero padded positions -> [B,512,1024].
  - cond on first `num_cond_frames` pixel frames -> num_cond_latent_frames = (n-1)//4+1 latent frames.
  - cond_mask [1,1,Tl,Hl,Wl] = 1 on cond latent frames; padding_mask [1,1,H,W] zeros.

Each .pt (one clip): {clean_latents[1,16,Tl,Hl,Wl], encoder_hidden_states[1,512,1024],
                      cond_indicator[1,1,Tl,1,1], cond_mask[1,1,Tl,Hl,Wl], padding_mask[1,1,H,W], prompt}
Run: CUDA_VISIBLE_DEVICES=0 /home/venv/cosmos_predict2/bin/python cosmos_data.py \
        --model <cosmos_snapshot> --out /home/cosmos_lin_data --n_clips 256 --frames 29 --size 256
Smoke (CPU, tiny): python cosmos_data.py --smoke
"""
import os, sys, glob, json, argparse, random
import torch

LIBERO_IMG = "/home/wm-vla-rl/_ext/WorldVLA/worldvla/processed_data/libero_10_img_action_256"


def humanize_task(traj_dir):
    """'.../KITCHEN_SCENE8_put_both_moka_pots_on_the_stove/trj_15' -> 'put both moka pots on the stove'."""
    name = os.path.basename(os.path.dirname(traj_dir))     # KITCHEN_SCENE8_put_both_...
    parts = name.split("_")
    # drop the leading SCENE tag tokens (KITCHEN, SCENE8) until first lowercase verb-ish token
    while parts and (parts[0].isupper() or parts[0][:5].upper() == "SCENE" or parts[0].isdigit()
                     or parts[0][-1].isdigit()):
        parts.pop(0)
    desc = " ".join(parts).strip()
    return f"A robot arm performs a manipulation task: {desc}." if desc else "A robot arm manipulation video."


def list_trajs(split, n_eval=32):
    trajs = sorted(glob.glob(os.path.join(LIBERO_IMG, "*", "trj_*")))
    trajs = [t for t in trajs if os.path.isdir(os.path.join(t, "imgs"))]
    return trajs[:-n_eval] if split == "train" else trajs[-n_eval:]


def load_clip(traj, frames, size, rng):
    """Load `frames` consecutive frames (random start) -> [1,3,T,H,W] in [-1,1]."""
    from PIL import Image
    fs = sorted(glob.glob(os.path.join(traj, "imgs", "image_*.png")),
                key=lambda p: int(p.split("_")[-1].split(".")[0]))
    if len(fs) < frames:
        return None
    start = rng.randint(0, len(fs) - frames)
    sel = fs[start:start + frames]
    arr = []
    for p in sel:
        im = Image.open(p).convert("RGB").resize((size, size))
        arr.append(torch.from_numpy(__import__("numpy").asarray(im)).float())
    v = torch.stack(arr).permute(3, 0, 1, 2).unsqueeze(0)   # [1,3,T,H,W]
    return v / 127.5 - 1.0


def build_masks(Tl, Hl, Wl, H, W, num_cond_latent_frames, device, dtype):
    cond_indicator = torch.zeros(1, 1, Tl, 1, 1, device=device, dtype=dtype)
    cond_indicator[:, :, :num_cond_latent_frames] = 1.0
    ones_pad = torch.ones(1, 1, 1, Hl, Wl, device=device, dtype=dtype)
    cond_mask = cond_indicator * ones_pad                  # 1 on cond frames, 0 elsewhere [1,1,Tl,Hl,Wl]
    padding_mask = torch.zeros(1, 1, H, W, device=device, dtype=dtype)
    return cond_indicator, cond_mask, padding_mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default=None, help="Cosmos snapshot dir (has vae/, text_encoder/, tokenizer/)")
    ap.add_argument("--out", type=str, default="/home/cosmos_lin_data")
    ap.add_argument("--split", type=str, default="train", choices=["train", "eval"])
    ap.add_argument("--n_clips", type=int, default=256)
    ap.add_argument("--frames", type=int, default=29, help="pixel frames per clip -> (frames-1)//4+1 latent frames")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--num_cond_frames", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    if args.smoke:
        # CPU smoke of the mask/normalize logic only (no VAE/T5 weights)
        Tl = (5 - 1) // 4 + 1
        ci, cm, pm = build_masks(Tl, 4, 4, 32, 32, 1, "cpu", torch.float32)
        print(f"[smoke] Tl={Tl} cond_indicator{tuple(ci.shape)} cond_mask{tuple(cm.shape)} padding_mask{tuple(pm.shape)}")
        print("[smoke] humanize:", humanize_task(os.path.join(LIBERO_IMG, "KITCHEN_SCENE8_put_both_moka_pots_on_the_stove", "trj_15")))
        ts = list_trajs("train"); print(f"[smoke] {len(ts)} train trajs, e.g. {os.path.basename(os.path.dirname(ts[0])) if ts else 'NONE'}")
        print("=== cosmos_data SMOKE PASS (mask/normalize/humanize logic) ===")
        return

    assert args.model, "--model required (Cosmos snapshot dir)"
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dt = torch.bfloat16
    from diffusers import AutoencoderKLWan
    from transformers import T5EncoderModel, T5TokenizerFast
    print(f"[data] loading VAE + T5 from {args.model}", flush=True)
    vae = AutoencoderKLWan.from_pretrained(args.model, subfolder="vae", torch_dtype=torch.float32).to(dev).eval()
    tok = T5TokenizerFast.from_pretrained(args.model, subfolder="tokenizer")
    txt = T5EncoderModel.from_pretrained(args.model, subfolder="text_encoder", torch_dtype=dt).to(dev).eval()

    lm = torch.tensor(vae.config.latents_mean, device=dev).view(1, vae.config.z_dim, 1, 1, 1)
    ls = torch.tensor(vae.config.latents_std, device=dev).view(1, vae.config.z_dim, 1, 1, 1)
    sigma_data = 1.0
    vsf_t = 2 ** sum(1 for x in vae.config.temperal_downsample if x)   # =4
    vsf_s = 2 ** (len(vae.config.dim_mult) - 1)                         # =8
    num_cond_lat = (args.num_cond_frames - 1) // vsf_t + 1
    print(f"[data] vsf_t={vsf_t} vsf_s={vsf_s} num_cond_latent_frames={num_cond_lat}", flush=True)

    os.makedirs(args.out, exist_ok=True)
    rng = random.Random(args.seed)
    trajs = list_trajs(args.split)
    print(f"[data] {len(trajs)} {args.split} trajs -> encoding {args.n_clips} clips to {args.out}", flush=True)

    @torch.no_grad()
    def encode_text(prompt):
        ti = tok([prompt], padding="max_length", max_length=512, truncation=True, return_tensors="pt", return_length=True)
        ids = ti.input_ids.to(dev); am = ti.attention_mask.bool().to(dev)
        emb = txt(ids, attention_mask=am).last_hidden_state.to(dt)
        L = am.sum(1)
        for i, l in enumerate(L):
            emb[i, l:] = 0
        return emb.cpu()

    n = 0; tries = 0
    while n < args.n_clips and tries < args.n_clips * 4:
        traj = trajs[tries % len(trajs)]; tries += 1
        v = load_clip(traj, args.frames, args.size, rng)
        if v is None:
            continue
        v = v.to(dev, torch.float32)
        with torch.no_grad():
            enc = vae.encode(v)
            z = enc.latent_dist.mode() if hasattr(enc, "latent_dist") else enc.latents
            z = (z - lm) / ls * sigma_data                     # normalized clean latents [1,16,Tl,Hl,Wl]
        B, C, Tl, Hl, Wl = z.shape
        H, W = args.size, args.size
        ci, cm, pm = build_masks(Tl, Hl, Wl, H, W, num_cond_lat, "cpu", dt)
        prompt = humanize_task(traj)
        emb = encode_text(prompt)
        torch.save({"clean_latents": z.to(dt).cpu(), "encoder_hidden_states": emb,
                    "cond_indicator": ci, "cond_mask": cm, "padding_mask": pm,
                    "prompt": prompt, "traj": traj}, os.path.join(args.out, f"clip_{n:05d}.pt"))
        n += 1
        if n % 20 == 0:
            print(f"[data] {n}/{args.n_clips}  latent{tuple(z.shape)}  '{prompt[:50]}'", flush=True)
    print(f"=== cosmos_data DONE: {n} clips -> {args.out} (latent {tuple(z.shape)}) ===")


if __name__ == "__main__":
    main()
