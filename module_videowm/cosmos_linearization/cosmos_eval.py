"""cosmos_eval.py — evaluate the linearized Cosmos-Predict2-2B WM vs the SDPA teacher.

Two modes (the two columns of the SANA-competitor story):
  efficiency : full-model forward latency + peak mem vs video length T (latent frames). Softmax self-attn is
               O(N^2) in tokens (N = T*Hl*Wl), Hedgehog linear is O(N). Sweep T until softmax OOMs; linear
               keeps going. This is the headline efficiency win.
  quality    : on held-out LIBERO clips (cosmos_data.py --split eval), velocity-prediction error of the
               linearized model vs the softmax teacher's velocity field (flow-matching target). Low gap =
               linearization preserved the world model. (+ optional latent-recon PSNR.)

Run:
  CUDA_VISIBLE_DEVICES=0 python cosmos_eval.py --mode efficiency --transformer <path> --ratio 0.75 \
        [--resume <linstate.safetensors>] --frames 4,8,16,32,48,64
  CUDA_VISIBLE_DEVICES=0 python cosmos_eval.py --mode quality --transformer <path> --ratio 0.75 \
        --resume <linstate.safetensors> --latents /home/cosmos_lin_data_eval
"""
import os, sys, time, glob, json, argparse
sys.path.insert(0, "/home/wm-vla-rl/_ext")
import torch
from cosmos_surgery import linearize_cosmos
from cosmos_hedgehog import get_dit_blocks, get_self_attn
from cosmos_distill import load_transformer, make_noisy_input, _load_linstate, latent_batches

Hl = Wl = 32   # 256px / 8


def set_all_softmax(model, linear_layers):
    """Restore CosmosAttnProcessor2_0 on the linearized attn1 (the original softmax teacher)."""
    from diffusers.models.transformers.transformer_cosmos import CosmosAttnProcessor2_0
    blocks, _ = get_dit_blocks(model)
    saved = {}
    for i in linear_layers:
        blk = blocks[i].inner if hasattr(blocks[i], "inner") else blocks[i]
        a, _ = get_self_attn(blk)
        saved[i] = a.processor
        a.set_processor(CosmosAttnProcessor2_0())
    return saved


def restore_procs(model, linear_layers, saved):
    blocks, _ = get_dit_blocks(model)
    for i in linear_layers:
        blk = blocks[i].inner if hasattr(blocks[i], "inner") else blocks[i]
        a, _ = get_self_attn(blk)
        a.set_processor(saved[i])


def synth_input(model, T):
    dev = next(model.parameters()).device; dt = next(model.parameters()).dtype
    C = model.config.in_channels - 1
    x = torch.randn(1, C, T, Hl, Wl, device=dev, dtype=dt)
    ci = torch.zeros(1, 1, T, 1, 1, device=dev, dtype=dt); ci[:, :, :1] = 1.0
    return dict(hidden_states=x, timestep=(torch.ones(1, 1, T, 1, 1, device=dev, dtype=dt) * 0.5),
                encoder_hidden_states=torch.randn(1, 512, model.config.text_embed_dim, device=dev, dtype=dt),
                condition_mask=ci.expand(1, 1, T, Hl, Wl).contiguous(),
                padding_mask=torch.zeros(1, 1, 256, 256, device=dev, dtype=dt), fps=16)


@torch.no_grad()
def time_forward(model, fwd, iters=3):
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    model(**fwd, return_dict=False); torch.cuda.synchronize()       # warmup
    t0 = time.time()
    for _ in range(iters):
        model(**fwd, return_dict=False)
    torch.cuda.synchronize()
    return (time.time() - t0) / iters, torch.cuda.max_memory_allocated() / 1e9


def efficiency(args):
    model = load_transformer(args.transformer)
    model, linear_layers, softmax_layers = linearize_cosmos(model, ratio_linear=args.ratio)
    if args.resume:
        _load_linstate(model, args.resume)
    model.eval()
    frames = [int(x) for x in args.frames.split(",")]
    rows = []
    for T in frames:
        N = T * Hl * Wl
        rec = {"T": T, "N_tokens": N}
        fwd = synth_input(model, T)
        # linear (linearized model as-is)
        try:
            lt, lm = time_forward(model, fwd)
            rec["linear_s"], rec["linear_gb"] = round(lt, 4), round(lm, 2)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache(); rec["linear_s"] = rec["linear_gb"] = "OOM"
        # softmax teacher (toggle processors)
        saved = set_all_softmax(model, linear_layers)
        try:
            st, sm = time_forward(model, fwd)
            rec["softmax_s"], rec["softmax_gb"] = round(st, 4), round(sm, 2)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache(); rec["softmax_s"] = rec["softmax_gb"] = "OOM"
        restore_procs(model, linear_layers, saved)
        if isinstance(rec.get("softmax_s"), float) and isinstance(rec.get("linear_s"), float):
            rec["speedup"] = round(rec["softmax_s"] / rec["linear_s"], 2)
        rows.append(rec)
        print(f"[eff] T={T:3d} N={N:6d} | linear {rec['linear_s']}s/{rec['linear_gb']}GB | "
              f"softmax {rec['softmax_s']}s/{rec['softmax_gb']}GB | x{rec.get('speedup','-')}", flush=True)
    os.makedirs("cosmos_eval_out", exist_ok=True)
    json.dump(rows, open("cosmos_eval_out/efficiency.json", "w"), indent=2)
    print("-> cosmos_eval_out/efficiency.json")


@torch.no_grad()
def quality(args):
    model = load_transformer(args.transformer)
    model, linear_layers, softmax_layers = linearize_cosmos(model, ratio_linear=args.ratio)
    if args.resume:
        _load_linstate(model, args.resume)
    model.eval()
    files = sorted(glob.glob(os.path.join(args.latents, "*.pt")))[: args.n_clips]
    assert files, f"no eval latents in {args.latents}"
    torch.manual_seed(0)
    lin_err, tea_err, gap = [], [], []
    for f in files:
        batch = torch.load(f, map_location="cpu")
        fwd, v, gm = make_noisy_input(batch, model, args.p_mean, args.p_std)
        gmf = gm.float()
        # linear prediction
        pl = model(**fwd, return_dict=False)[0].float()
        e_lin = (((pl - v.float()) ** 2) * gmf).sum() / gmf.sum().clamp_min(1) / pl.shape[1]
        # softmax teacher on identical input
        saved = set_all_softmax(model, linear_layers)
        pt = model(**fwd, return_dict=False)[0].float()
        restore_procs(model, linear_layers, saved)
        e_tea = (((pt - v.float()) ** 2) * gmf).sum() / gmf.sum().clamp_min(1) / pt.shape[1]
        g = (((pl - pt) ** 2) * gmf).sum() / gmf.sum().clamp_min(1) / pl.shape[1]   # lin vs teacher velocity gap
        lin_err.append(e_lin.item()); tea_err.append(e_tea.item()); gap.append(g.item())
    import numpy as np
    out = {"n_clips": len(files), "ratio": args.ratio, "resume": args.resume,
           "lin_velocity_mse": float(np.mean(lin_err)), "teacher_velocity_mse": float(np.mean(tea_err)),
           "lin_vs_teacher_gap": float(np.mean(gap))}
    print(f"[quality] n={len(files)} | teacher vMSE {out['teacher_velocity_mse']:.4f} | "
          f"linear vMSE {out['lin_velocity_mse']:.4f} | lin-vs-teacher gap {out['lin_vs_teacher_gap']:.4f}", flush=True)
    os.makedirs("cosmos_eval_out", exist_ok=True)
    json.dump(out, open("cosmos_eval_out/quality.json", "w"), indent=2)
    print("-> cosmos_eval_out/quality.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["efficiency", "quality"], required=True)
    ap.add_argument("--transformer", required=True)
    ap.add_argument("--ratio", type=float, default=0.75)
    ap.add_argument("--resume", type=str, default=None)
    ap.add_argument("--latents", type=str, default="/home/cosmos_lin_data_eval")
    ap.add_argument("--frames", type=str, default="4,8,16,32,48,64")
    ap.add_argument("--n_clips", type=int, default=32)
    ap.add_argument("--p_mean", type=float, default=0.0)
    ap.add_argument("--p_std", type=float, default=1.0)
    args = ap.parse_args()
    (efficiency if args.mode == "efficiency" else quality)(args)


if __name__ == "__main__":
    main()
