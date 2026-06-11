"""§3a REPA smoke GATE (run BEFORE the full overnight run). Single GPU, batch=1 (the real Stage-2 regime).
Reuses the EXACT helpers from wan_distill.py (load_wrapper / make_noisy / data_batches / RepaProjector /
load_jepa_target / repa_align_loss) so we test the same code path the full run uses.

Checks (ALL must pass):
  (a) both loss terms finite, non-NaN, no shape error;
  (b) proj output [B,21,30,52,1408] matches the temporally-interpolated target;
  (c) magnitude balance: lam*l_align is within ~0.3-3x of distill_mse;
  (d) matched-vs-shuffled: l_align with the CORRECTLY-matched clip target MUST be clearly lower than with a
      MISMATCHED clip's target (here: the previous step's target, a different clip). If matched ~= shuffled the
      clip->feature mapping is broken or features are uninformative -> STOP.

Prints SMOKE_PASS / SMOKE_FAIL.  Run:
  CUDA_VISIBLE_DEVICES=0 python jepa_smoke.py --steps 100
"""
import os, sys, argparse, time
sys.path.insert(0, "/home/wm-vla-rl/_ext")
import torch
import torch.nn.functional as F
import wan_distill as WD


def mean(x):
    return sum(x) / max(len(x), 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--resume", default="/home/wm-vla-rl/_ext/wan_lin_out/wanlin_s1_step4000.safetensors")
    ap.add_argument("--latents", default="/home/wan_distill_data")
    ap.add_argument("--jepa_feats", default="/home/wan_jepa_feats")
    ap.add_argument("--jepa_block", type=int, default=15)
    ap.add_argument("--lam", type=float, default=0.25)
    ap.add_argument("--lr", type=float, default=1e-4)
    args = ap.parse_args()
    dev = "cuda"
    torch.manual_seed(0)

    student = WD.load_wrapper(dev, surgery=True, ratio=0.75, groups=4, lora=True, lora_r=8)
    teacher = WD.load_wrapper(dev, surgery=False)
    WD.load_linstate(student, args.resume)
    base = getattr(student.model, "_orig_mod", student.model)

    params = [p for _, p in WD.surgery_trainable_params(base)]
    for p in teacher.parameters():
        p.requires_grad_(False)
    projector = WD.RepaProjector(base.dim).to(dev).float()
    params = params + list(projector.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.95))
    print(f"[smoke] proj {base.dim}->{WD.DIM_VJEPA} ({sum(p.numel() for p in projector.parameters())/1e6:.2f}M); "
          f"tap block {args.jepa_block}/{len(base.blocks)}; lam={args.lam}", flush=True)

    cache = {}
    base.blocks[args.jepa_block].register_forward_hook(
        lambda m, i, o: cache.__setitem__("h", o[0] if isinstance(o, tuple) else o))
    di = WD.data_batches(args.latents, batch_size=1, rank=0, world_size=1)

    student.model.train()
    mses, als, ratios, matched, shuffled = [], [], [], [], []
    shape_ok = True; nan = False; proj_shape = None; tgt_shape = None
    prev_tg = None
    t0 = time.time()
    for step in range(args.steps):
        batch = next(di)
        noisy, cond, timestep, _v = WD.make_noisy(batch, teacher)
        with torch.no_grad():
            tgt, _ = teacher(noisy, cond, timestep)
        cache.clear()
        pred, _ = student(noisy, cond, timestep)
        loss_mse = F.mse_loss(pred.float(), tgt.float())

        hidden = cache["h"]                                    # [B, 32760, 1536]
        B = hidden.shape[0]
        tg = torch.stack([WD.load_jepa_target(c, args.jepa_feats, hidden.device)
                          for c in batch["clip_ids"]], 0)      # [B,21,30,52,1408]
        h5 = hidden.float().reshape(B, *WD.WAN_GRID, -1)
        p = projector(h5)                                      # [B,21,30,52,1408]
        proj_shape, tgt_shape = tuple(p.shape), tuple(tg.shape)
        if proj_shape != tgt_shape:
            shape_ok = False
        l_align = (1.0 - F.cosine_similarity(p, tg, dim=-1)).mean()

        # (d) matched vs shuffled: compare the SAME projected hidden against a DIFFERENT clip's target
        if prev_tg is not None and prev_tg.shape == tg.shape:
            with torch.no_grad():
                l_sh = (1.0 - F.cosine_similarity(p.detach(), prev_tg, dim=-1)).mean().item()
            matched.append(l_align.item()); shuffled.append(l_sh)
        prev_tg = tg.detach()

        loss = loss_mse + args.lam * l_align
        if not (torch.isfinite(loss_mse) and torch.isfinite(l_align)):
            nan = True
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()

        m, a = loss_mse.item(), l_align.item()
        r = (args.lam * a) / max(m, 1e-9)
        mses.append(m); als.append(a); ratios.append(r)
        if step % 10 == 0 or step == args.steps - 1:
            sh = shuffled[-1] if shuffled else float("nan")
            print(f"[smoke] step {step:3d}  mse {m:.4f}  l_align {a:.4f}  {args.lam:g}*la/mse {r:.2f}  "
                  f"| matched {a:.4f} shuffled {sh:.4f}  {time.time()-t0:.0f}s", flush=True)

    L = slice(-30, None)
    mm, ms = mean(matched[L]), mean(shuffled[L])
    ratio_last = mean(ratios[L])
    gate_a = (not nan)
    gate_b = shape_ok
    gate_c = 0.3 <= ratio_last <= 3.0
    gate_d = mm < ms - 0.01                      # matched clearly lower than shuffled
    allpass = gate_a and gate_b and gate_c and gate_d
    print("=" * 70, flush=True)
    print(f"proj shape {proj_shape}  target shape {tgt_shape}  -> match={shape_ok}", flush=True)
    print(f"(a) finite/non-NaN          : {gate_a}", flush=True)
    print(f"(b) shapes match            : {gate_b}", flush=True)
    print(f"(c) lam*l_align/mse ratio   : {ratio_last:.2f}  in[0.3,3.0]={gate_c}   "
          f"(mse {mean(mses[L]):.4f}, l_align {mean(als[L]):.4f}, lam*l_align {args.lam*mean(als[L]):.4f})", flush=True)
    print(f"(d) matched {mm:.4f} < shuffled {ms:.4f}  (gap {ms-mm:+.4f}, "
          f"shuffled/matched {ms/max(mm,1e-9):.2f}x) -> {gate_d}", flush=True)
    print("SMOKE_PASS" if allpass else "SMOKE_FAIL", flush=True)
    sys.exit(0 if allpass else 1)


if __name__ == "__main__":
    main()
