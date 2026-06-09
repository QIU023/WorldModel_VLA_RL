"""ARL2-competitor eval. Two parts:
  --mode efficiency : forward-pass micro-benchmark of the KDA-linearized student vs the softmax teacher across
                      increasing frame counts (latency + peak memory) -> the O(F) long-video advantage. Does NOT
                      need the trained ckpt (measures the architecture). Mirrors the Cosmos efficiency table.
  --mode quality    : non-streaming generation @81 frames (teacher vs distilled student), dump mp4s for VBench.
                      (built after Stage-2 finishes; needs --resume <student linstate>.)

Run efficiency: CUDA_VISIBLE_DEVICES=0 python wan_eval.py --mode efficiency
"""
import os, sys, time, argparse, json
sys.path.insert(0, "/home/wm-vla-rl/_ext"); sys.path.insert(0, "/home/wm-vla-rl/_ext/Causal-Forcing")
import torch
import wan_distill as WD


def _rand_input(wrapper, Fr, B=1):
    dev = next(wrapper.parameters()).device; dt = next(wrapper.parameters()).dtype
    x0 = torch.randn(B, Fr, 16, 60, 104, device=dev, dtype=dt)
    # one real prompt-embed, reused (timing is content-independent)
    import glob
    pe_file = sorted(glob.glob("/home/wan_distill_data/clip_*.pt"))[0]
    pe = torch.load(pe_file, map_location="cpu")["prompt_embeds"].to(dev, dt)
    cond = {"prompt_embeds": pe.unsqueeze(0).expand(B, -1, -1).contiguous()}
    ts = wrapper.scheduler.timesteps.to(dev)
    timestep = ts[torch.randint(0, len(ts), (B, Fr), device=dev)]
    return x0, cond, timestep


def _bench(wrapper, Fr, iters=3):
    x0, cond, timestep = _rand_input(wrapper, Fr)
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        wrapper(x0, cond, timestep)                      # warmup (kernel autotune)
    torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        for _ in range(iters):
            wrapper(x0, cond, timestep)
    torch.cuda.synchronize()
    dt = (time.time() - t0) / iters
    mem = torch.cuda.max_memory_allocated() / 1e9
    return dt, mem


def efficiency(ratio=0.75, groups=4):
    dev = "cuda:0"
    print("[eff] loading softmax teacher + KDA student (no ckpt needed; measuring architecture)", flush=True)
    teacher = WD.load_wrapper(dev, surgery=False)
    student = WD.load_wrapper(dev, surgery=True, ratio=ratio, groups=groups)
    rows = []
    for Fr in [21, 41, 81, 121, 161, 231]:               # latent frames; px frames ~= (Fr-1)*4+1 (81..921)
        px = (Fr - 1) * 4 + 1
        try:
            ts, ms = _bench(student, Fr); tt, mt = _bench(teacher, Fr)
            sp = tt / ts if ts > 0 else 0
            rows.append(dict(latent_F=Fr, px_F=px, kda_s=round(ts, 3), softmax_s=round(tt, 3),
                             speedup=round(sp, 2), kda_GB=round(ms, 1), softmax_GB=round(mt, 1)))
            print(f"[eff] F={Fr:3d} (px {px:3d})  KDA {ts:.3f}s/{ms:.1f}GB  softmax {tt:.3f}s/{mt:.1f}GB  "
                  f"speedup {sp:.2f}x", flush=True)
        except torch.cuda.OutOfMemoryError:
            print(f"[eff] F={Fr} softmax OOM (linear still fits = the point)", flush=True)
            torch.cuda.empty_cache(); rows.append(dict(latent_F=Fr, px_F=px, softmax="OOM"))
    out = "/home/wm-vla-rl/_ext/wan_eval_efficiency.json"
    json.dump(rows, open(out, "w"), indent=2)
    print(f"=== efficiency table -> {out} ===", flush=True)
    return rows


def velocity_match(ratio=0.75, groups=4, resume=None, n=32):
    """逼近教师: on held-out teacher latents, compare the student's predicted velocity to the pristine teacher's
    on IDENTICAL noised inputs (non-streaming training-path forward — KDA-supported, fair same-mode). Reports
    relative-MSE and cosine sim, averaged over clips x random timesteps. This is the S2 objective measured on
    held-out data: 1.0 cosine / 0.0 relMSE = perfect teacher reproduction."""
    dev = "cuda:0"
    teacher = WD.load_wrapper(dev, surgery=False)
    student = WD.load_wrapper(dev, surgery=True, ratio=ratio, groups=groups)
    if resume:
        WD.load_linstate(student, resume)
        tag = os.path.basename(resume)
    else:
        tag = "UNDISTILLED (harness check)"
    import glob
    files = sorted(glob.glob("/home/wan_distill_data/clip_*.pt"))[-n:]   # held-out tail
    rel, cos = [], []
    for f in files:
        d = torch.load(f, map_location="cpu")
        batch = {"latent": d["latent"].unsqueeze(0), "prompt_embeds": d["prompt_embeds"].unsqueeze(0)}
        noisy, cond, timestep, _v = WD.make_noisy(batch, teacher)
        with torch.no_grad():
            tv, _ = teacher(noisy, cond, timestep)
            sv, _ = student(noisy, cond, timestep)
        tvf, svf = tv.float().flatten(), sv.float().flatten()
        rel.append(((svf - tvf).pow(2).sum() / tvf.pow(2).sum().clamp_min(1e-8)).item())
        cos.append(torch.nn.functional.cosine_similarity(svf, tvf, dim=0).item())
    import statistics as st
    print(f"[velocity-match] {tag}  n={len(files)}", flush=True)
    print(f"  rel_MSE(student||teacher) = {st.mean(rel):.4f} +- {st.pstdev(rel):.4f}  (0=perfect)", flush=True)
    print(f"  cosine(student,teacher)   = {st.mean(cos):.4f} +- {st.pstdev(cos):.4f}  (1=perfect)", flush=True)
    out = "/home/wm-vla-rl/_ext/wan_eval_velocity.json"
    json.dump({"tag": tag, "rel_mse": st.mean(rel), "cosine": st.mean(cos), "n": len(files)}, open(out, "w"), indent=2)
    print(f"=== velocity-match -> {out} ===", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["efficiency", "velocity", "quality"], default="efficiency")
    ap.add_argument("--ratio", type=float, default=0.75)
    ap.add_argument("--groups", type=int, default=4)
    ap.add_argument("--resume", type=str, default=None)
    ap.add_argument("--n", type=int, default=32)
    args = ap.parse_args()
    if args.mode == "efficiency":
        efficiency(args.ratio, args.groups)
    elif args.mode == "velocity":
        velocity_match(args.ratio, args.groups, args.resume, args.n)
    else:
        raise SystemExit("quality(VBench-video) mode wired after Stage-2 completes")


if __name__ == "__main__":
    main()
