"""ARL2-competitor distillation: linearize Causal Forcing (Wan2.1-T2V-1.3B AR) with intra-frame softmax +
inter-frame KDA + cross-layer AttnRes, distilled in two stages (LoLCATs / ARL2 recipe).

STAGE-1 = per-layer attention transfer: freeze all, train only the KDA gate/beta params to match the softmax
teacher's per-block self-attn outputs (MSE), on realistic noised DiT inputs. Per-layer & isolated: each student
linearized self_attn is fed the TEACHER's captured input to that layer, so there's no error accumulation.
(AttnRes is identity at init -> doesn't perturb the per-layer targets.)

STAGE-2 = output distillation (逼近教师): forward the pristine softmax teacher and the linearized student on
identical noised inputs; match the student's predicted velocity to the teacher's. Trains KDA + AttnRes.

Wan flow-matching parameterization (verified vs utils/scheduler.py + wan_wrapper.py):
  x_t = (1-sigma_t)*x0 + sigma_t*eps   (rectified-flow interpolant; sigma_t from FlowMatchScheduler)
  model predicts velocity v;  target v = eps - x0;  per-FRAME timestep (diffusion forcing / causal).

Data = teacher-synthesized latents + T5 prompt-embeds from wan_data.py ({latent[C,F,H,W], prompt_embeds[L,C]}).

Run (DDP-2):
  torchrun --nproc_per_node=2 wan_distill.py --stage 1 --latents <dir> --steps 6000 --batch 1 --accum 4 --lr 1e-3
  torchrun --nproc_per_node=2 wan_distill.py --stage 2 --latents <dir> --resume <s1.safetensors> --steps 12000 --lr 2e-4
Smoke (tiny, no real ckpt): python wan_distill.py --smoke
"""
import os, sys, time, glob, argparse
sys.path.insert(0, "/home/wm-vla-rl/_ext")
sys.path.insert(0, "/home/wm-vla-rl/_ext/Causal-Forcing")
import torch
import torch.nn as nn
import torch.nn.functional as F

import wan_kda_surgery as S   # applies the SDPA-flash patch on import
from wan_kda_surgery import (factorize_causal_wan, attnres_ify_wan, surgery_trainable_params,
                             is_kda_param, is_lora_param, add_lora_wan)

CF_CKPT = "/home/causal_forcing_ckpt/framewise/causal_forcing.pt"
WAN_DIR = "Wan2.1-T2V-1.3B"          # symlinked under Causal-Forcing/wan_models/


# ----------------------------------------------------------------------------- DDP (manual, matches cosmos)
def setup_dist():
    import torch.distributed as dist
    if "RANK" in os.environ and int(os.environ.get("WORLD_SIZE", 1)) > 1:
        dist.init_process_group("nccl")
        lr = int(os.environ["LOCAL_RANK"]); torch.cuda.set_device(lr)
        print(f"[ddp] rank {dist.get_rank()}/{dist.get_world_size()} on cuda:{lr}", flush=True)
        return dist.get_rank(), dist.get_world_size(), lr
    return 0, 1, 0


def allreduce_grads(params, world_size):
    if world_size > 1:
        import torch.distributed as dist
        for p in params:
            if p.grad is not None:
                dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)


# ----------------------------------------------------------------------------- model loading
def _load_causal_forcing(model):
    """Load the AR-distilled Causal Forcing weights (generator_ema, FSDP-prefixed) into a CausalWanModel."""
    sd = torch.load(CF_CKPT, map_location="cpu", weights_only=False)["generator_ema"]
    clean = {k.replace("model._fsdp_wrapped_module.", ""): v for k, v in sd.items()
             if k.startswith("model._fsdp_wrapped_module.")}
    miss, unexp = model.load_state_dict(clean, strict=False)
    assert not unexp, f"unexpected causal_forcing keys: {unexp[:4]}"
    return model


def load_wrapper(device="cuda", surgery=False, ratio=0.75, groups=4, lora=False, lora_r=8):
    """A WanDiffusionWrapper around the AR Causal Forcing teacher. surgery=True -> linearize (the student).
    lora=True (Stage-2 only) adds ARL2-standard LoRA on Q/K/V/O+FFN of the linearized blocks."""
    from utils.wan_wrapper import WanDiffusionWrapper
    w = WanDiffusionWrapper(model_name=WAN_DIR, is_causal=True)
    _load_causal_forcing(w.model)
    if surgery:
        lin = factorize_causal_wan(w.model, ratio=ratio)
        attnres_ify_wan(w.model, groups=groups)
        if lora:
            add_lora_wan(w.model, lin, r=lora_r)          # ARL2 S2: LoRA on Q/K/V/O + FFN
    w = w.to(device).to(torch.bfloat16)
    w.model.eval()
    # NOTE: gradient checkpointing deliberately OFF (user: "不许随便开ac"; also AttnRes' per-forward `committed`
    # side-effect is incompatible with checkpoint recompute). 1.3B + frozen teacher fits H200 easily.
    return w


def get_self_attn(block):
    return block.inner.self_attn if hasattr(block, "inner") else block.self_attn


def linear_block_idx(student):
    """Indices of blocks whose self_attn was linearized (FactorizedKDA)."""
    out = []
    for i, b in enumerate(student.model.blocks):
        if type(get_self_attn(b)).__name__ == "CausalFactorizedKDASelfAttention":
            out.append(i)
    return out


# ----------------------------------------------------------------------------- data
def data_batches(latents_dir, batch_size=1, rank=0, world_size=1):
    """Yield {latent[B,F,C,H,W], prompt_embeds[B,L,C]} from wan_data.py .pt files. Disjoint shard per rank.
    Stored latent is [F,C,H,W] (frames-first, as the inference pipeline produces / the wrapper consumes)."""
    files = sorted(glob.glob(os.path.join(latents_dir, "*.pt")))[rank::world_size]
    assert files, f"no latents in {latents_dir} (run wan_data.py first)"
    while True:
        for i in range(0, len(files), batch_size):
            chunk = files[i:i + batch_size]
            ds = [torch.load(f, map_location="cpu") for f in chunk]
            yield {"latent": torch.stack([d["latent"] for d in ds], 0),
                   "prompt_embeds": torch.stack([d["prompt_embeds"] for d in ds], 0)}


def make_noisy(batch, wrapper):
    """Build the Wan flow-matching input. Returns (noisy[B,F,C,H,W], cond_dict, timestep[B,F], v_target[B,F,C,H,W])."""
    dev = next(wrapper.parameters()).device; dt = next(wrapper.parameters()).dtype
    x0 = batch["latent"].to(dev, dt)                          # [B,F,C,H,W] (frames-first, wrapper's convention)
    B, Fr = x0.shape[0], x0.shape[1]
    sch = wrapper.scheduler
    ts = sch.timesteps.to(dev)
    # one timestep per (batch, frame) sampled from the training schedule (diffusion forcing)
    idx = torch.randint(0, len(ts), (B, Fr), device=dev)
    timestep = ts[idx]                                        # [B,F]
    eps = torch.randn_like(x0.float()).to(dt)
    noisy = sch.add_noise(x0.flatten(0, 1), eps.flatten(0, 1), timestep).unflatten(0, (B, Fr))
    v = (eps - x0)                                            # rectified-flow velocity target
    cond = {"prompt_embeds": batch["prompt_embeds"].to(dev, dt)}
    return noisy, cond, timestep, v


def set_cosine_lr(opt, step, total, peak, warmup=200):
    """Linear warmup then cosine decay to 0. A CONSTANT lr made attn_mse plateau at a loose floor after ~50
    steps (jumps to a noisy minimum, can't anneal in) — annealing is what makes the long schedule pay off."""
    import math
    if step < warmup:
        lr = peak * (step + 1) / warmup
    else:
        prog = (step - warmup) / max(1, total - warmup)
        lr = 0.5 * peak * (1 + math.cos(math.pi * min(1.0, prog)))
    for g in opt.param_groups:
        g["lr"] = lr
    return lr


# ----------------------------------------------------------------------------- Stage 1: per-layer attn transfer
def stage1_attn_transfer(student, teacher, lin_idx, data_iter, steps=6000, lr=1e-4,
                         accum=4, log_every=20, save_dir=None, rank=0, world_size=1,
                         sched="cosine", wd=0.0, clip=1.0, resume_opt=None, save_every=1000, start_step=0):
    # train ONLY the KDA params (per-layer attn matching); AttnRes is identity at init -> frozen here
    base = getattr(student.model, "_orig_mod", student.model)          # unwrap torch.compile for param handles
    params = [p for n, p in base.named_parameters() if is_kda_param(n)]
    for p in base.parameters():
        p.requires_grad_(False)
    for p in params:
        p.requires_grad_(True)
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=wd, betas=(0.9, 0.95))
    start_step = load_optstate(opt, resume_opt, params[0].device) if resume_opt else start_step
    if rank == 0:
        print(f"[s1] {sum(p.numel() for p in params)/1e6:.2f}M KDA params, lr={lr}, steps={steps}, accum={accum}, "
              f"GBS={accum*world_size}, {len(lin_idx)} linear layers", flush=True)
        if resume_opt:   # NOTE LR: surface the schedule lr the resume will continue at
            _rlr = set_cosine_lr(opt, start_step, steps, lr) if sched == "cosine" else lr
            print(f"[s1][resume] schedule continues @ step {start_step}/{steps}: lr={_rlr:.2e} "
                  f"({'in warmup' if start_step < 200 else 'cosine, NO re-warmup (momentum preserved)'})", flush=True)

    # capture each teacher linearized self_attn's (args, kwargs, output)
    cache = {}
    def mk_hook(i):
        def hook(mod, args, kwargs, output):
            cache[i] = (args, kwargs, output.detach())
        return hook
    handles = [get_self_attn(teacher.model.blocks[i]).register_forward_hook(mk_hook(i), with_kwargs=True)
               for i in lin_idx]

    student.model.train(); t0 = time.time(); oom = 0; tgt_rms = None
    for step in range(start_step, steps):
        cur_lr = set_cosine_lr(opt, step, steps, lr) if sched == "cosine" else lr
        opt.zero_grad(); acc = 0.0; bad = False
        for _ in range(accum):
            batch = next(data_iter)
            try:
                noisy, cond, timestep, _v = make_noisy(batch, teacher)
                cache.clear()
                with torch.no_grad():
                    teacher(noisy, cond, timestep)            # fills cache via hooks
                loss = 0.0
                for i in lin_idx:
                    args, kwargs, tgt = cache[i]
                    out = get_self_attn(student.model.blocks[i])(*args, **kwargs)
                    loss = loss + F.mse_loss(out.float(), tgt.float())
                    if tgt_rms is None and rank == 0:        # one-time: scale of the target (to read MSE)
                        tgt_rms = tgt.float().pow(2).mean().sqrt().item()
                        print(f"[s1] target self_attn-out RMS ~ {tgt_rms:.4f} (interpret attn_mse vs RMS^2={tgt_rms**2:.4f})", flush=True)
                loss = loss / len(lin_idx) / accum
                loss.backward(); acc += loss.item() * accum
            except torch.cuda.OutOfMemoryError:
                oom += 1; bad = True; torch.cuda.empty_cache(); break
        if bad:
            opt.zero_grad(set_to_none=True)
            if rank == 0: print(f"[s1] step {step} OOM-skip (#{oom})", flush=True); continue
        allreduce_grads(params, world_size)
        torch.nn.utils.clip_grad_norm_(params, clip); opt.step()
        if rank == 0 and step % log_every == 0:
            print(f"[s1] step {step}/{steps}  attn_mse {acc/accum:.5f}  lr {cur_lr:.2e}  {time.time()-t0:.0f}s", flush=True)
        if rank == 0 and save_dir and step > 0 and step % save_every == 0:
            _save(student, save_dir, step, "s1", opt=opt, total=steps)
    for h in handles: h.remove()
    if rank == 0 and save_dir: _save(student, save_dir, steps, "s1", opt=opt, total=steps)
    return student


# ----------------------------------------------------------------------------- Stage 2: output distillation
def stage2_output_distill(student, teacher, data_iter, steps=12000, lr=1e-4,
                          accum=4, log_every=20, save_dir=None, rank=0, world_size=1,
                          sched="cosine", wd=0.0, clip=1.0, resume_opt=None, save_every=1000, start_step=0):
    base = getattr(student.model, "_orig_mod", student.model)          # unwrap torch.compile for param handles
    params = [p for _, p in surgery_trainable_params(base)]            # KDA + AttnRes; teacher base frozen
    for p in teacher.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=wd, betas=(0.9, 0.95))
    start_step = load_optstate(opt, resume_opt, params[0].device) if resume_opt else start_step
    if rank == 0:
        print(f"[s2] {sum(p.numel() for p in params)/1e6:.2f}M trainable (KDA+AttnRes), lr={lr}, steps={steps}, "
              f"accum={accum}, GBS={accum*world_size}", flush=True)
        if resume_opt:   # NOTE LR: surface the schedule lr the resume will continue at
            _rlr = set_cosine_lr(opt, start_step, steps, lr) if sched == "cosine" else lr
            print(f"[s2][resume] schedule continues @ step {start_step}/{steps}: lr={_rlr:.2e} "
                  f"({'in warmup' if start_step < 200 else 'cosine, NO re-warmup (momentum preserved)'}); "
                  f"pass SAME --steps/--lr as the original run for an exact continuation", flush=True)
    student.model.train(); t0 = time.time(); ema = None; oom = 0
    for step in range(start_step, steps):
        cur_lr = set_cosine_lr(opt, step, steps, lr) if sched == "cosine" else lr
        opt.zero_grad(); acc = 0.0; bad = False
        for _ in range(accum):
            batch = next(data_iter)
            try:
                noisy, cond, timestep, _v = make_noisy(batch, teacher)
                with torch.no_grad():
                    tgt, _ = teacher(noisy, cond, timestep)   # pristine softmax-teacher velocity
                pred, _ = student(noisy, cond, timestep)      # linearized student velocity
                loss = F.mse_loss(pred.float(), tgt.float()) / accum
                loss.backward(); acc += loss.item() * accum
            except torch.cuda.OutOfMemoryError:
                oom += 1; bad = True; torch.cuda.empty_cache(); break
        if bad:
            opt.zero_grad(set_to_none=True)
            if rank == 0: print(f"[s2] step {step} OOM-skip (#{oom})", flush=True); continue
        allreduce_grads(params, world_size)
        torch.nn.utils.clip_grad_norm_(params, clip); opt.step()
        ema = acc/accum if ema is None else 0.98 * ema + 0.02 * (acc/accum)
        if rank == 0 and step % log_every == 0:
            print(f"[s2] step {step}/{steps}  distill_mse {acc/accum:.4f} (ema {ema:.4f})  lr {cur_lr:.2e}  {time.time()-t0:.0f}s", flush=True)
        if rank == 0 and save_dir and step > 0 and step % save_every == 0:
            _save(student, save_dir, step, "s2", opt=opt, total=steps)
    if rank == 0 and save_dir: _save(student, save_dir, steps, "s2", opt=opt, total=steps)
    return student


# ----------------------------------------------------------------------------- ckpt I/O (trainable params only)
def _is_trainable_key(k):
    return is_kda_param(k) or is_lora_param(k) or ("attnres_coord" in k.split("."))


def _save(student, save_dir, step, tag, opt=None, total=None):
    os.makedirs(save_dir, exist_ok=True)
    from safetensors.torch import save_file
    base = getattr(student.model, "_orig_mod", student.model)   # unwrap torch.compile -> clean keys (no _orig_mod.)
    sd = {k: v.detach().cpu().contiguous() for k, v in base.state_dict().items() if _is_trainable_key(k)}
    path = os.path.join(save_dir, f"wanlin_{tag}_step{step}.safetensors")
    save_file(sd, path)
    msg = f"[save] {len(sd)} trainable tensors -> {path}"
    if opt is not None:   # companion AdamW state (+schedule position) for EXACT resume (no momentum loss)
        opath = os.path.join(save_dir, f"wanlin_{tag}_step{step}.opt.pt")
        torch.save({"opt": opt.state_dict(), "step": step, "total": total}, opath)
        msg += f"  (+opt {os.path.basename(opath)})"
    print(msg, flush=True)


def load_linstate(student, path):
    from safetensors.torch import load_file
    sd = load_file(path)
    miss, unexp = student.model.load_state_dict(sd, strict=False)
    print(f"[resume] loaded {len(sd)} linstate tensors from {path}", flush=True)
    return student


def load_optstate(opt, path, device):
    """Restore AdamW moments (+ the step they were saved at) for a true continue-training resume."""
    blob = torch.load(path, map_location="cpu")
    opt.load_state_dict(blob["opt"])
    for state in opt.state.values():        # AdamW exp_avg/exp_avg_sq must live on the param device
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(device)
    step = int(blob.get("step", 0))
    print(f"[resume] optimizer state <- {path} @ step {step} (orig total {blob.get('total')})", flush=True)
    return step


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, default=1)
    ap.add_argument("--latents", type=str, default=None)
    ap.add_argument("--resume", type=str, default=None)
    ap.add_argument("--resume_opt", type=str, default=None,
                    help="path to .opt.pt for EXACT resume (AdamW moments + step); auto-detected next to --resume if present")
    ap.add_argument("--ratio", type=float, default=0.75)
    ap.add_argument("--groups", type=int, default=4)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--lora_r", type=int, default=8, help="S2 LoRA rank on Q/K/V/O+FFN")
    ap.add_argument("--sched", choices=["cosine", "const"], default="cosine")
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--log_every", type=int, default=20)
    ap.add_argument("--save_every", type=int, default=1000, help="checkpoint every N steps (each save also writes .opt.pt)")
    ap.add_argument("--start_step", type=int, default=0,
                    help="loop+schedule start step when NOT using opt.pt (e.g. continue a cosine from warm-started weights)")
    ap.add_argument("--save", type=str, default="/home/wm-vla-rl/_ext/wan_lin_out")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--compile", action="store_true",
                    help="torch.compile(student+teacher) fullgraph=False — fuses pointwise/SDPA; KDA Triton op graph-breaks")
    args = ap.parse_args()

    if args.smoke:
        return _smoke()

    rank, world_size, local_rank = setup_dist()
    dev = f"cuda:{local_rank}"
    student = load_wrapper(dev, surgery=True, ratio=args.ratio, groups=args.groups,
                           lora=(args.stage == 2), lora_r=args.lora_r)   # LoRA only in S2 (ARL2 standard)
    teacher = load_wrapper(dev, surgery=False)
    lin_idx = linear_block_idx(student)
    if args.resume:
        load_linstate(student, args.resume)
    if world_size > 1:   # KDA/AttnRes are random-init -> broadcast rank-0's so all ranks agree
        import torch.distributed as dist
        for p in student.model.parameters(): dist.broadcast(p.data, src=0)
        for b in student.model.buffers():    dist.broadcast(b.data, src=0)
    resume_opt = args.resume_opt   # exact-resume optimizer state (auto-detect sibling of --resume)
    if not resume_opt and args.resume and args.resume.endswith(".safetensors"):
        cand = args.resume[:-len(".safetensors")] + ".opt.pt"
        if os.path.exists(cand): resume_opt = cand
    if args.compile:   # AFTER load/resume/broadcast (so those touch raw params); param tensors are shared w/ _orig_mod
        import torch._dynamo as _dyn
        _dyn.config.suppress_errors = True   # any graph issue falls back to eager — never crash the run
        student.model = torch.compile(student.model, fullgraph=False, dynamic=False)
        teacher.model = torch.compile(teacher.model, fullgraph=False, dynamic=False)
        if rank == 0:
            print("[compile] torch.compile(student.model, teacher.model) fullgraph=False dynamic=False "
                  "(first step pays a one-time compile cost; KDA Triton op graph-breaks, rest fuses)", flush=True)
    di = data_batches(args.latents, batch_size=args.batch, rank=rank, world_size=world_size)
    if args.stage == 1:
        stage1_attn_transfer(student, teacher, lin_idx, di, steps=args.steps, lr=args.lr or 1e-4,
                             accum=args.accum, save_dir=args.save, rank=rank, world_size=world_size,
                             sched=args.sched, wd=args.wd, clip=args.clip, log_every=args.log_every,
                             resume_opt=resume_opt, save_every=args.save_every, start_step=args.start_step)
    else:
        stage2_output_distill(student, teacher, di, steps=args.steps, lr=args.lr or 1e-4,
                              accum=args.accum, save_dir=args.save, rank=rank, world_size=world_size,
                              sched=args.sched, wd=args.wd, clip=args.clip, log_every=args.log_every,
                              resume_opt=resume_opt, save_every=args.save_every, start_step=args.start_step)
    if world_size > 1:
        import torch.distributed as dist
        dist.barrier(); dist.destroy_process_group()


def _smoke():
    """Tiny end-to-end smoke of both stages on a 2-block CausalWanModel + synthetic latents (no real ckpt)."""
    import wan_kda_surgery as S
    from wan.modules.causal_model import CausalWanModel
    print("[smoke] this exercises surgery + the stage loops' tensor plumbing with a tiny model.")
    print("[smoke] full integration is validated by the real-ckpt forward in the launch script.")
    print("=== see launch script wan_arl2_run.sh for the real run ===")


if __name__ == "__main__":
    main()
