"""Phase-2 (cosmos_predict2): encode rendered nav clips into the wan_distill .pt format.

For each frames/clip_XXXXX/: load the PNG frames, resize to the WM I2V native 480x832 (the EXACT
transform the eval wrapper uses), VAE-encode to Wan latents [F,16,60,104], and T5-encode the nav
action text (built EXACTLY like downstream/api_models.process_b_action_nav) to prompt_embeds[512,4096].

Saves latents/clip_XXXXX.pt = {latent[F,16,60,104] f32, prompt_embeds[512,4096] bf16, prompt, b_action,
scene} -- the SAME schema wan_distill / nav_posttrain consume.

Run (cosmos_predict2):
  /home/venv/cosmos_predict2/bin/python navgen_encode.py --frames /root/nav_posttrain/frames \
      --out /root/nav_posttrain/latents --device cuda:0
"""
import argparse, json, os, sys
import torch
from PIL import Image
from torchvision import transforms

CF_DIR = "/home/wm-vla-rl/_ext/Causal-Forcing"
os.chdir(CF_DIR)
sys.path.insert(0, CF_DIR)
sys.path.insert(0, "/home/wm-vla-rl/_ext")
import wan_kda_surgery  # noqa: F401  (SDPA-flash patch on import; harmless here)
from utils.wan_wrapper import WanVAEWrapper, WanTextEncoder

GEN_H, GEN_W = 480, 832

# --- EXACT replica of downstream/api_models.process_b_action_nav (frozen in the repo) ---
IGENEX_ACTION_STRS = {1: "forward", 2: "turn_left", 3: "turn_right", 4: "stop", 0: "placeholder"}


def action_text(b_action):
    """b_action = [placeholder] + motion_ids ; matches process_b_action_nav (drops [0])."""
    seq = []
    for a in b_action[1:]:
        s = IGENEX_ACTION_STRS[a]
        if s == "forward":
            s = "forward 0.2m"
        elif "turn" in s:
            s = f"{s} 22.5°"  # 22.5°
        seq.append(s)
    return f"Follow this sequence of camera motions: {str(seq)}."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", default="/root/nav_posttrain/frames")
    ap.add_argument("--out", default="/root/nav_posttrain/latents")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--expect_flat", type=int, default=5, help="assert latent frame count (5 for 17 px)")
    ap.add_argument("--limit", type=int, default=0, help="encode at most N clips (0=all)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    dev = torch.device(args.device)

    tfm = transforms.Compose([
        transforms.Resize((GEN_H, GEN_W)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    print("[encode] loading VAE + T5 ...", flush=True)
    vae = WanVAEWrapper().to(dev).to(torch.bfloat16)
    txt = WanTextEncoder().to(dev)
    torch.set_grad_enabled(False)

    clips = sorted(d for d in os.listdir(args.frames) if d.startswith("clip_"))
    if args.limit:
        clips = clips[: args.limit]
    print(f"[encode] {len(clips)} clips -> {args.out}", flush=True)
    n = 0
    for c in clips:
        cdir = os.path.join(args.frames, c)
        outp = os.path.join(args.out, c + ".pt")
        if os.path.exists(outp):
            continue
        meta = json.load(open(os.path.join(cdir, "meta.json")))
        pngs = sorted(p for p in os.listdir(cdir) if p.endswith(".png"))
        imgs = [tfm(Image.open(os.path.join(cdir, p)).convert("RGB")) for p in pngs]  # each [3,H,W]
        pixel = torch.stack(imgs, dim=1).unsqueeze(0).to(dev, torch.bfloat16)  # [1,3,F,H,W]
        lat = vae.encode_to_latent(pixel)  # [1,Flat,16,60,104]
        Flat = lat.shape[1]
        if args.expect_flat and Flat != args.expect_flat:
            print(f"[encode][WARN] {c}: latent F={Flat} != expected {args.expect_flat} "
                  f"(px frames {len(pngs)})", flush=True)
        prompt = action_text(meta["b_action"])
        pe = txt(text_prompts=[prompt])["prompt_embeds"][0].detach().cpu()  # [L,4096]
        torch.save({
            "latent": lat[0].detach().float().cpu(),     # [F,16,60,104]
            "prompt_embeds": pe,                          # [512,4096] bf16
            "prompt": prompt,
            "b_action": meta["b_action"],
            "scene": meta["scene"],
        }, outp)
        n += 1
        if n <= 3 or n % 25 == 0:
            print(f"[encode] {c}: latent {tuple(lat[0].shape)} pe {tuple(pe.shape)} | {prompt}", flush=True)
    print(f"=== [encode] DONE: {n} clips encoded to {args.out} ===", flush=True)


if __name__ == "__main__":
    main()
