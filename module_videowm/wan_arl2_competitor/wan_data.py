"""Teacher-synthesized distillation data for the ARL2-competitor run. Runs the pristine Causal Forcing AR
pipeline (CausalInferencePipeline, 4-step denoising) on text prompts and saves, per clip:
  {latent: [F,C,H,W] clean latent, prompt_embeds: [L,C] T5 embedding}
This is the ARL2 Stage-2 data recipe (teacher-synthesized videos) and also exercises the teacher generation.

Run: CUDA_VISIBLE_DEVICES=0 python wan_data.py --config configs/causal_forcing_dmd_framewise.yaml \
        --prompts prompts/demos.txt --n 256 --frames 21 --out /home/wan_distill_data --seeds 3
"""
import os, sys, argparse, glob
sys.path.insert(0, "/home/wm-vla-rl/_ext")
sys.path.insert(0, "/home/wm-vla-rl/_ext/Causal-Forcing")
import torch
import wan_kda_surgery  # SDPA-flash patch (no flash-attn pkg needed)

CF_CKPT = "/home/causal_forcing_ckpt/framewise/causal_forcing.pt"


def build_pipeline(config_path, device):
    from omegaconf import OmegaConf
    from pipeline import CausalInferencePipeline
    config = OmegaConf.merge(OmegaConf.load("configs/default_config.yaml"), OmegaConf.load(config_path))
    pipe = CausalInferencePipeline(config, device=device)
    sd = torch.load(CF_CKPT, map_location="cpu", weights_only=False)["generator_ema"]
    fixed = {(k.replace("model._fsdp_wrapped_module.", "model.", 1) if k.startswith("model._fsdp_wrapped_module.") else k): v
             for k, v in sd.items()}
    pipe.generator.load_state_dict(fixed, strict=False)
    pipe = pipe.to(dtype=torch.bfloat16)
    pipe.text_encoder.to(device); pipe.generator.to(device); pipe.vae.to(device)
    return pipe


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/causal_forcing_dmd_framewise.yaml")
    ap.add_argument("--prompts", default="prompts/demos.txt")
    ap.add_argument("--out", default="/home/wan_distill_data")
    ap.add_argument("--n", type=int, default=256, help="number of clips to generate")
    ap.add_argument("--frames", type=int, default=21, help="latent frames (21 -> 81 px frames)")
    ap.add_argument("--seeds", type=int, default=3, help="seeds per prompt (cycles prompts to reach --n)")
    ap.add_argument("--start", type=int, default=None, help="start clip index (for sharding two generators)")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    dev = "cuda"

    prompts = [l.strip() for l in open(args.prompts) if l.strip()]
    print(f"[data] {len(prompts)} prompts; generating {args.n} clips @ {args.frames} latent frames", flush=True)
    pipe = build_pipeline(args.config, dev)

    done = len(glob.glob(os.path.join(args.out, "clip_*.pt")))
    print(f"[data] {done} clips already present, resuming", flush=True)
    j = args.start if args.start is not None else done    # --start lets a 2nd generator shard the upper range
    made = j
    while made < (1 if args.smoke else args.n):
        prompt = prompts[(j // 1) % len(prompts)]
        seed = j // len(prompts)
        path = os.path.join(args.out, f"clip_{j:05d}.pt")
        if os.path.exists(path):
            j += 1; made += 1; continue
        g = torch.Generator(device=dev).manual_seed(seed * 100003 + j)
        noise = torch.randn([1, args.frames, 16, 60, 104], device=dev, dtype=torch.bfloat16, generator=g)
        with torch.no_grad():
            _video, latents = pipe.inference(noise=noise, text_prompts=[prompt], return_latents=True)
            cond = pipe.text_encoder(text_prompts=[prompt])
        pe = cond["prompt_embeds"][0].detach().cpu()           # [L,C]
        lat = latents[0].detach().float().cpu()                # [F,C,H,W]
        torch.save({"latent": lat, "prompt_embeds": pe, "prompt": prompt, "seed": seed}, path)
        if j % 10 == 0 or args.smoke:
            print(f"[data] clip {j}: latent {tuple(lat.shape)} pe {tuple(pe.shape)}  '{prompt[:50]}...'", flush=True)
        j += 1; made += 1
    print(f"=== [data] done: {made} clips in {args.out} ===", flush=True)


if __name__ == "__main__":
    main()
