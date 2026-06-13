"""NAV post-train: teach OUR linearized Causal-Forcing WM to FOLLOW the World-in-World nav action-text
in the I2V path, by fine-tuning a FRESH nav LoRA on real HM3D rollouts (I2V flow-matching to the
ground-truth future latents, conditioned on first-frame initial_latent + nav action-text).

WHAT THIS REUSES (does not rebuild):
  * load_wrapper / linear_block_idx / set_cosine_lr / allreduce_grads / setup_dist  from wan_distill.py
  * factorize_causal_wan + attnres_ify_wan + add_lora_wan (main LoRA) + LoRALinear  from wan_kda_surgery.py
  * the EXACT eval-time linearized student = base CF generator + KDA + AttnRes + main LoRA loaded from
    the distill delta (jepa_l025_T8/wanlin_s2_step6000.safetensors).

WHAT IS NEW:
  * a SECOND (nav) LoRA stacked on the linearized blocks' Q/K/V/O+FFN (zero-init B -> identity at start,
    so at step 0 the model == the validated eval model). The whole distilled student is FROZEN; only the
    nav LoRA trains. Re-eval = same surgery order + load the small nav delta (causalforcing_model.py
    --nav_lora_ckpt).

I2V FLOW-MATCHING LOSS (rectified flow, matches utils/scheduler.py + the diffusion-forcing training of CF):
  x0 = clip latent [B,F,16,60,104] (F=5: frame0 = conditioning).  frame0: timestep=0 + kept clean.
  frames 1..F-1: noised with per-frame schedule timesteps.  v_target = eps - x0.
  pred = student(noisy, {prompt_embeds}, timestep)[0]  (velocity).  loss = MSE(pred[:,1:], v_target[:,1:]).
  (frame-causal mask -> future frames attend the clean frame0 = I2V text+image conditioning.)

MODES:
  --mode smoke   : one fwd+bwd; assert loss finite + grads flow into the nav LoRA (and NOT into frozen base).
  --mode overfit : overfit a single batch N steps; loss must drop substantially.
  --mode train   : full training (DDP-able), cosine LR, checkpoint nav LoRA every --save_every.

Run:
  smoke   : /home/venv/cosmos_predict2/bin/python nav_posttrain.py --mode smoke   --latents /root/nav_posttrain/latents
  overfit : /home/venv/cosmos_predict2/bin/python nav_posttrain.py --mode overfit --latents /root/nav_posttrain/latents --steps 80
  train   : torchrun --nproc_per_node=2 nav_posttrain.py --mode train --latents /root/nav_posttrain/latents --steps 3000 ...
"""
import os, sys, time, glob, argparse, math
sys.path.insert(0, "/home/wm-vla-rl/_ext")
sys.path.insert(0, "/home/wm-vla-rl/_ext/Causal-Forcing")
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import save_file, load_file

import wan_kda_surgery as S
from wan_kda_surgery import LoRALinear, get_self_attn, is_lora_param
from wan_distill import (load_wrapper, linear_block_idx, set_cosine_lr,
                         allreduce_grads, setup_dist, load_linstate)

MAIN_DELTA = "/home/wm-vla-rl/_ext/wan_lin_out/jepa_l025_T8/wanlin_s2_step6000.safetensors"


# ----------------------------------------------------------------------------- nav LoRA (fresh, stacked)
def add_nav_lora(model, linear_idx, r=8, alpha=16, include_ffn=True):
    """Stack a SECOND LoRA on the (already main-LoRA-wrapped) Q/K/V/O(+FFN) of each linearized block.
    LoRALinear freezes its .base (the whole distilled sub-module) and trains only the new lora_A/lora_B
    (zero-init B -> identity at insertion). Returns the list of NEW trainable nav params."""
    new_params = []
    for i in linear_idx:
        attn = get_self_attn(model.blocks[i])
        for name in ("q", "k", "v", "o"):
            lin = LoRALinear(getattr(attn, name), r, alpha)
            setattr(attn, name, lin)
            new_params += [lin.lora_A.weight, lin.lora_B.weight]
        if include_ffn:
            blk = model.blocks[i]
            blk = blk.inner if hasattr(blk, "inner") else blk
            for fi in (0, 2):
                lin = LoRALinear(blk.ffn[fi], r, alpha)
                blk.ffn[fi] = lin
                new_params += [lin.lora_A.weight, lin.lora_B.weight]
    return new_params


def build_nav_student(device, ratio, groups, main_r, nav_r, main_delta=MAIN_DELTA, rank=0):
    """Reconstruct the eval-time linearized student (base CF + KDA + AttnRes + main LoRA loaded from the
    distill delta), then FREEZE it and add a fresh, trainable nav LoRA on top."""
    student = load_wrapper(device, surgery=True, ratio=ratio, groups=groups, lora=True, lora_r=main_r)
    lin_idx = linear_block_idx(student)
    load_linstate(student, main_delta)                 # main KDA + AttnRes + main LoRA (eval-time student)
    for p in student.model.parameters():
        p.requires_grad_(False)                        # freeze the whole distilled student
    nav_params = add_nav_lora(student.model, lin_idx, r=nav_r)
    for n, p in student.model.named_parameters():      # belt-and-braces: ONLY nav LoRA trains
        p.requires_grad_(False)
    for p in nav_params:
        p.requires_grad_(True)
    if rank == 0:
        ntr = sum(p.numel() for p in nav_params) / 1e6
        nfroz = sum(p.numel() for p in student.model.parameters() if not p.requires_grad) / 1e6
        print(f"[nav] {len(lin_idx)} linearized blocks; nav LoRA r={nav_r} -> {ntr:.2f}M trainable "
              f"({nfroz:.1f}M frozen)", flush=True)
    return student, lin_idx, nav_params


# ----------------------------------------------------------------------------- data
def data_batches(latents_dir, batch_size=1, rank=0, world_size=1, single=False):
    files = sorted(glob.glob(os.path.join(latents_dir, "*.pt")))
    assert files, f"no latents in {latents_dir} (run navgen_encode.py first)"
    if single:                                         # overfit: one fixed batch, all ranks see it
        files = files[:batch_size]
    else:
        files = files[rank::world_size]
        assert files, f"rank {rank} got no shard from {len(files)} files"
    while True:
        for i in range(0, len(files), batch_size):
            chunk = files[i:i + batch_size]
            if len(chunk) < batch_size:
                continue
            ds = [torch.load(f, map_location="cpu") for f in chunk]
            yield {"latent": torch.stack([d["latent"] for d in ds], 0),
                   "prompt_embeds": torch.stack([d["prompt_embeds"] for d in ds], 0),
                   "prompt": [d["prompt"] for d in ds]}


def make_i2v_batch(batch, wrapper):
    """Build the I2V flow-matching inputs. frame0 = clean conditioning (timestep 0), frames 1.. noised.
    Returns noisy[B,F,C,H,W], cond, timestep[B,F], v_target[B,F,C,H,W], x0[B,F,C,H,W]."""
    dev = next(wrapper.parameters()).device
    dt = next(wrapper.parameters()).dtype
    x0 = batch["latent"].to(dev, dt)                   # [B,F,C,H,W]
    B, Fr = x0.shape[0], x0.shape[1]
    sch = wrapper.scheduler
    ts = sch.timesteps.to(dev)
    idx = torch.randint(0, len(ts), (B, Fr), device=dev)
    timestep = ts[idx].clone()                         # [B,F]
    eps = torch.randn_like(x0.float()).to(dt)
    noisy = sch.add_noise(x0.flatten(0, 1), eps.flatten(0, 1), timestep).unflatten(0, (B, Fr))
    # --- I2V conditioning: frame0 is the GIVEN clean first frame (matches pipeline.inference timestep*0) ---
    timestep[:, 0] = 0
    noisy[:, 0] = x0[:, 0]
    v = (eps - x0)                                     # rectified-flow velocity target
    cond = {"prompt_embeds": batch["prompt_embeds"].to(dev, dt)}
    return noisy, cond, timestep, v, x0


def i2v_loss_from(student, noisy, cond, timestep, v):
    pred, _ = student(noisy, cond, timestep)           # velocity prediction (kv_cache=None -> _forward_train)
    # loss ONLY on the future frames (frame0 is the clean conditioning, excluded)
    return F.mse_loss(pred[:, 1:].float(), v[:, 1:].float())


def i2v_loss(student, batch):
    noisy, cond, timestep, v, _ = make_i2v_batch(batch, student)
    return i2v_loss_from(student, noisy, cond, timestep, v)


# ----------------------------------------------------------------------------- ckpt I/O (nav LoRA only)
def save_nav_lora(student, path):
    base = getattr(student.model, "_orig_mod", student.model)
    sd = {k: p.detach().cpu().contiguous()
          for k, p in base.named_parameters() if p.requires_grad}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    save_file(sd, path)
    return len(sd), path


# ----------------------------------------------------------------------------- modes
def run_smoke(args, rank=0):
    dev = "cuda:0"
    student, lin_idx, nav_params = build_nav_student(dev, args.ratio, args.groups, args.main_r,
                                                     args.nav_r, rank=rank)
    student.model.train()
    di = data_batches(args.latents, batch_size=args.batch, single=True)
    batch = next(di)
    print(f"[smoke] batch latent {tuple(batch['latent'].shape)} | prompt: {batch['prompt'][0]}", flush=True)

    loss = i2v_loss(student, batch)
    print(f"[smoke] (a) forward loss = {loss.item():.5f}  finite={torch.isfinite(loss).item()}", flush=True)
    assert torch.isfinite(loss), "FAIL: loss not finite"

    loss.backward()
    # grads must flow into nav LoRA, and ONLY there
    g_nav = [p for p in nav_params if p.grad is not None]
    gnorm = sum(p.grad.float().norm().item() ** 2 for p in g_nav) ** 0.5
    n_nan = sum(int(not torch.isfinite(p.grad).all()) for p in g_nav)
    n_nonzero = sum(int(p.grad.abs().sum() > 0) for p in g_nav)
    print(f"[smoke] (a) nav-LoRA params with grad: {len(g_nav)}/{len(nav_params)}; "
          f"grad global-norm={gnorm:.4e}; nonzero-grad params={n_nonzero}; nan-grad params={n_nan}",
          flush=True)
    # frozen base must have NO grad
    base = getattr(student.model, "_orig_mod", student.model)
    frozen_with_grad = [n for n, p in base.named_parameters()
                        if (not p.requires_grad) and (p.grad is not None) and (p.grad.abs().sum() > 0)]
    print(f"[smoke] (a) frozen params with nonzero grad: {len(frozen_with_grad)} (must be 0)", flush=True)

    ok = (torch.isfinite(loss) and len(g_nav) == len(nav_params) and n_nan == 0
          and n_nonzero > 0 and gnorm > 0 and len(frozen_with_grad) == 0)
    print(f"=== SMOKE (a) {'PASS' if ok else 'FAIL'} ===", flush=True)
    return ok


def run_overfit(args, rank=0):
    dev = "cuda:0"
    student, lin_idx, nav_params = build_nav_student(dev, args.ratio, args.groups, args.main_r,
                                                     args.nav_r, rank=rank)
    student.model.train()
    di = data_batches(args.latents, batch_size=args.batch, single=True)
    batch = next(di)
    print(f"[overfit] single batch latent {tuple(batch['latent'].shape)} | {batch['prompt'][0]}", flush=True)
    # FIX the noise + per-frame timesteps once -> deterministic target, so a correct grad path drives
    # the loss toward ~0 (proves the I2V+text conditioning + loss are wired right, not just trending).
    torch.manual_seed(0)
    noisy, cond, timestep, v, _ = make_i2v_batch(batch, student)
    print(f"[overfit] FIXED noise+timestep; future-frame timesteps {timestep[0,1:].tolist()}", flush=True)
    opt = torch.optim.AdamW(nav_params, lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0)
    losses = []
    t0 = time.time()
    for step in range(args.steps):
        set_cosine_lr(opt, step, args.steps, args.lr, warmup=max(1, args.steps // 20))
        opt.zero_grad()
        loss = i2v_loss_from(student, noisy, cond, timestep, v)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(nav_params, 1.0)
        opt.step()
        losses.append(loss.item())
        if step % max(1, args.steps // 20) == 0 or step == args.steps - 1:
            print(f"[overfit] step {step:3d}/{args.steps}  loss {loss.item():.5f}  "
                  f"lr {opt.param_groups[0]['lr']:.2e}  {time.time()-t0:.0f}s", flush=True)
    l0 = sum(losses[:3]) / 3
    lN = sum(losses[-3:]) / 3
    drop = 100 * (l0 - lN) / max(l0, 1e-9)
    print(f"[overfit] start~{l0:.5f}  end~{lN:.5f}  drop {drop:.1f}%", flush=True)
    if args.save:
        n, p = save_nav_lora(student, args.save)
        print(f"[overfit] saved {n} nav-LoRA tensors -> {p}", flush=True)
    ok = drop > 50.0
    print(f"=== OVERFIT {'PASS' if ok else 'FAIL'} (need >50% loss drop) ===", flush=True)
    return ok, l0, lN, drop


def run_train(args):
    rank, world_size, local_rank = setup_dist()
    dev = f"cuda:{local_rank}"
    student, lin_idx, nav_params = build_nav_student(dev, args.ratio, args.groups, args.main_r,
                                                     args.nav_r, rank=rank)
    if world_size > 1:   # nav LoRA random/zero-init -> broadcast rank0's so all ranks agree
        import torch.distributed as dist
        for p in nav_params:
            dist.broadcast(p.data, src=0)
    if args.resume:
        sd = load_file(args.resume)
        miss, unexp = student.model.load_state_dict(sd, strict=False)
        if rank == 0:
            print(f"[train] resumed {len(sd)} nav-LoRA tensors from {args.resume} (unexp {len(unexp)})", flush=True)
    student.model.train()
    di = data_batches(args.latents, batch_size=args.batch, rank=rank, world_size=world_size)
    opt = torch.optim.AdamW(nav_params, lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0)
    if rank == 0:
        print(f"[train] steps={args.steps} accum={args.accum} GBS={args.accum*world_size} lr={args.lr} "
              f"save_every={args.save_every} -> {args.ckpt_dir}", flush=True)
    ema = None
    t0 = time.time()
    for step in range(args.start_step, args.steps):
        cur_lr = set_cosine_lr(opt, step, args.steps, args.lr, warmup=args.warmup)
        opt.zero_grad()
        acc = 0.0
        for _ in range(args.accum):
            batch = next(di)
            loss = i2v_loss(student, batch) / args.accum
            loss.backward()
            acc += loss.item() * args.accum
        allreduce_grads(nav_params, world_size)
        torch.nn.utils.clip_grad_norm_(nav_params, 1.0)
        opt.step()
        mse = acc / args.accum
        ema = mse if ema is None else 0.98 * ema + 0.02 * mse
        if rank == 0 and step % args.log_every == 0:
            print(f"[train] step {step}/{args.steps}  i2v_mse {mse:.5f} (ema {ema:.5f})  "
                  f"lr {cur_lr:.2e}  {time.time()-t0:.0f}s", flush=True)
        if rank == 0 and args.save_every and step > 0 and step % args.save_every == 0:
            n, p = save_nav_lora(student, os.path.join(args.ckpt_dir, f"nav_lora_step{step}.safetensors"))
            print(f"[train] [save] {n} nav-LoRA tensors -> {p}", flush=True)
            _copy_to_home(p, args.home_copy)
    if rank == 0:
        n, p = save_nav_lora(student, os.path.join(args.ckpt_dir, f"nav_lora_step{args.steps}.safetensors"))
        print(f"[train] [final] {n} nav-LoRA tensors -> {p}", flush=True)
        _copy_to_home(p, args.home_copy)
    if world_size > 1:
        import torch.distributed as dist
        dist.barrier(); dist.destroy_process_group()


def _copy_to_home(src, home_dir):
    if not home_dir:
        return
    try:
        import shutil
        os.makedirs(home_dir, exist_ok=True)
        dst = os.path.join(home_dir, os.path.basename(src))
        shutil.copy2(src, dst)
        # also keep a stable 'latest' name for the eval wrapper
        shutil.copy2(src, os.path.join(home_dir, "nav_lora_latest.safetensors"))
        print(f"[train] copied nav LoRA -> {dst} (+nav_lora_latest.safetensors)", flush=True)
    except Exception as e:
        print(f"[train] WARN copy-to-home failed: {e}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["smoke", "overfit", "train"], default="smoke")
    ap.add_argument("--latents", default="/root/nav_posttrain/latents")
    ap.add_argument("--ratio", type=float, default=0.75)
    ap.add_argument("--groups", type=int, default=4)
    ap.add_argument("--main_r", type=int, default=8, help="main (distill) LoRA rank — must match the delta")
    ap.add_argument("--nav_r", type=int, default=8, help="fresh nav LoRA rank")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--steps", type=int, default=80)
    ap.add_argument("--start_step", type=int, default=0)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--log_every", type=int, default=20)
    ap.add_argument("--save_every", type=int, default=500)
    ap.add_argument("--ckpt_dir", default="/root/nav_posttrain/ckpts")
    ap.add_argument("--home_copy", default="/home/wm-vla-rl/_ext/wan_lin_out/nav_lora")
    ap.add_argument("--resume", default=None)
    ap.add_argument("--save", default=None, help="(overfit) save the overfit nav LoRA here")
    args = ap.parse_args()

    if args.mode == "smoke":
        ok = run_smoke(args)
        sys.exit(0 if ok else 1)
    elif args.mode == "overfit":
        ok, *_ = run_overfit(args)
        sys.exit(0 if ok else 1)
    else:
        run_train(args)


if __name__ == "__main__":
    main()
