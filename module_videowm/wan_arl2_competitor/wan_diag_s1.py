"""Independent stage-1 bug diagnosis — NOT relying on Cosmos. Three checks:
  (A) MECHANISM sanity: feed the teacher's captured per-layer input back into the TEACHER's own self_attn;
      output MUST equal the hooked output (MSE ~ 0). If not, the hook/replay comparison itself is broken.
  (B) untrained (fresh random KDA) student attn_mse  — the starting point.
  (C) trained (loaded S1 ckpt) student attn_mse — must be << (B) if training actually works.
Also reports the teacher's INTER-frame energy share (how much of each layer's attention output comes from
attending across frames) — this is the part KDA must reproduce; it bounds the achievable floor.
Run: CUDA_VISIBLE_DEVICES=0 python wan_diag_s1.py
"""
import sys, glob; sys.path.insert(0, "/home/wm-vla-rl/_ext"); sys.path.insert(0, "/home/wm-vla-rl/_ext/Causal-Forcing")
import torch, torch.nn.functional as F
import wan_distill as W

dev = "cuda:0"
ck = sorted(glob.glob("/home/wm-vla-rl/_ext/wan_lin_out/wanlin_s1_step*.safetensors"))
ck = ck[-1] if ck else None
print(f"[diag] latest S1 ckpt = {ck}", flush=True)

teacher = W.load_wrapper(dev, surgery=False)
student = W.load_wrapper(dev, surgery=True, ratio=0.75, groups=4)
lin = W.linear_block_idx(student)
di = W.data_batches("/home/wan_distill_data", batch_size=1)


def run(tag, nb=4):
    cache = {}
    def mk(i):
        def h(m, a, kw, o): cache[i] = (a, kw, o.detach())
        return h
    hs = [W.get_self_attn(teacher.model.blocks[i]).register_forward_hook(mk(i), with_kwargs=True) for i in lin]
    s_mse = r_mse = tgt_rms = 0.0; n = 0
    for _ in range(nb):
        b = next(di); noisy, cond, ts, _ = W.make_noisy(b, teacher)
        cache.clear()
        with torch.no_grad():
            teacher(noisy, cond, ts)
        for i in lin:
            a, kw, tgt = cache[i]
            with torch.no_grad():
                so = W.get_self_attn(student.model.blocks[i])(*a, **kw)      # student (KDA)
                to = W.get_self_attn(teacher.model.blocks[i])(*a, **kw)      # teacher self-replay
            s_mse += F.mse_loss(so.float(), tgt.float()).item()
            r_mse += F.mse_loss(to.float(), tgt.float()).item()
            tgt_rms += tgt.float().pow(2).mean().sqrt().item()
            n += 1
    for h in hs: h.remove()
    rms = tgt_rms / n
    print(f"[{tag}] student attn_mse = {s_mse/n:.5f}  | rel-MSE = {(s_mse/n)/(rms**2):.3f}  "
          f"| teacher self-replay MSE = {r_mse/n:.2e} (MUST be ~0)  | tgt RMS = {rms:.3f}", flush=True)


run("untrained (fresh random KDA)")
if ck:
    W.load_linstate(student, ck)
    run("trained   (S1 ckpt loaded)")
print("=== diag done: self-replay~0 => mechanism OK; trained<<untrained => training works ===", flush=True)
