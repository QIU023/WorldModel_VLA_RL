"""Generate teacher OR distilled-student videos for VBench, via the repo's BidirectionalDiffusionInferencePipeline
(non-streaming full-sequence unipc denoising). Both teacher and student use the SAME pipeline/mode => fair
comparison, and the student runs through WanDiffusionWrapper.forward(kv_cache=None) = the KDA-validated training
path (no KDA streaming needed). Saves a [N,T,H,W,C] uint8 tensor (VBench-ready) + per-prompt mp4 optional.

Run teacher:  CUDA_VISIBLE_DEVICES=0 python wan_generate.py --mode teacher --n 32 --out /home/wan_eval_gen
Run student:  CUDA_VISIBLE_DEVICES=0 python wan_generate.py --mode student --resume <s2.safetensors> --n 32 --out ...
"""
import os, sys, argparse
sys.path.insert(0, "/home/wm-vla-rl/_ext"); sys.path.insert(0, "/home/wm-vla-rl/_ext/Causal-Forcing")
import torch
import wan_kda_surgery  # SDPA-flash patch
from wan_distill import load_wrapper, load_linstate


def build_pipeline(config_path, mode, resume, ratio, groups, dev):
    from omegaconf import OmegaConf
    from pipeline import BidirectionalDiffusionInferencePipeline
    args = OmegaConf.merge(OmegaConf.load("configs/default_config.yaml"), OmegaConf.load(config_path))
    surgery = (mode == "student")
    gen = load_wrapper(dev, surgery=surgery, ratio=ratio, groups=groups)   # loads causal_forcing.pt (+surgery)
    if surgery and resume:
        load_linstate(gen, resume)
    gen.model.eval()
    pipe = BidirectionalDiffusionInferencePipeline(args, generator=gen)
    pipe = pipe.to(dtype=torch.bfloat16)
    pipe.text_encoder.to(dev); pipe.generator.to(dev); pipe.vae.to(dev)
    return pipe


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["teacher", "student"], required=True)
    ap.add_argument("--config", default="configs/causal_forcing_dmd_framewise.yaml")
    ap.add_argument("--prompts", default="prompts/demos.txt")
    ap.add_argument("--resume", default=None, help="student linstate (S2 ckpt)")
    ap.add_argument("--ratio", type=float, default=0.75)
    ap.add_argument("--groups", type=int, default=4)
    ap.add_argument("--frames", type=int, default=21, help="latent frames (21 -> 81 px)")
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="/home/wan_eval_gen")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    dev = "cuda"
    prompts = [l.strip() for l in open(args.prompts) if l.strip()][:args.n]
    pipe = build_pipeline(args.config, args.mode, args.resume, args.ratio, args.groups, dev)

    vids = []
    for i, prompt in enumerate(prompts):
        g = torch.Generator(device=dev).manual_seed(args.seed * 9973 + i)
        noise = torch.randn([1, args.frames, 16, 60, 104], device=dev, dtype=torch.bfloat16, generator=g)
        with torch.no_grad():
            video = pipe.inference(noise=noise, text_prompts=[prompt])   # [B,T,C,H,W] pixel in [0,1]-ish
        v = video[0].float().clamp(0, 1).mul(255).byte().cpu()           # [T,C,H,W]
        v = v.permute(0, 2, 3, 1).contiguous()                          # [T,H,W,C] (VBench layout)
        vids.append(v)
        if i % 5 == 0:
            print(f"[gen-{args.mode}] {i+1}/{len(prompts)}  vid {tuple(v.shape)}  '{prompt[:48]}'", flush=True)
    out = os.path.join(args.out, f"vids_{args.mode}.pt")
    torch.save({"videos": torch.stack(vids)}, out)   # {"videos":[N,T,H,W,3] uint8} -> reuse cosmos_vbench.py as-is
    print(f"=== [gen-{args.mode}] saved {len(vids)} videos -> {out} ===", flush=True)


if __name__ == "__main__":
    main()
