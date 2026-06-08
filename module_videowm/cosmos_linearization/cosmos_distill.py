"""Cosmos-Predict2-2B-Video2World linearization training (LoLCATs 2-stage), on REAL LIBERO video latents
(produced by cosmos_data.py).

STAGE-1 = attention transfer: freeze all, train only Hedgehog phi_q/phi_k to match the softmax teacher's
attn1 outputs (MSE), on realistic noised DiT inputs. (LoRA/AttnRes stay frozen.)
STAGE-2 = flow-matching finetune: unfreeze feat_map + LoRA + AttnRes, train end-to-end on the Cosmos
velocity objective.

Cosmos-Predict2 parameterization (derived from Cosmos2VideoToWorldPipeline, verified vs the sampler):
  s ~ flow time in (0,1) via s = sigma/(sigma+1), ln(sigma) ~ N(p_mean, p_std).
  z_s = (1-s)*x_clean + s*eps         <- transformer INPUT on generated frames (the rectified-flow interpolant)
  cond frames                         <- clean x_clean, timestep t_conditioning = 1e-4/(1+1e-4)
  transformer predicts F,  target velocity v = eps - x_clean   (=> denoised D = z_s - s*F = x_clean)
  loss = MSE(F, v) on NON-conditioning frames only.

Run Stage-1: CUDA_VISIBLE_DEVICES=0 python cosmos_distill.py --stage 1 --transformer <path> --latents <dir> --steps 2000
Run Stage-2: CUDA_VISIBLE_DEVICES=0 python cosmos_distill.py --stage 2 --transformer <path> --latents <dir> \
                 --resume <stage1_linstate.safetensors> --steps 3000 --lr 2e-4
Smoke (no GPU/data): python cosmos_distill.py --smoke
"""
import os, sys, time, glob, math, argparse
sys.path.insert(0, "/home/wm-vla-rl/_ext")
import torch
import torch.nn as nn
from cosmos_surgery import linearize_cosmos, attn_transfer_loss
from cosmos_hedgehog import get_dit_blocks

T_CONDITIONING = 0.0001 / (0.0001 + 1.0)   # cond frames sit at ~0 flow time (sigma_conditioning=1e-4)
FPS = 16


def load_transformer(path, device="cuda", dtype=torch.bfloat16):
    from diffusers import CosmosTransformer3DModel
    m = CosmosTransformer3DModel.from_pretrained(path, torch_dtype=dtype).to(device).eval()
    return m


# padding_mask stays [1,1,H,W]: the transformer broadcasts it via .repeat(batch_size,...), so it must NOT
# be stacked to B (else B*B blowup). Everything else is per-sample and concatenated along dim 0.
_BATCH_KEYS = ("clean_latents", "encoder_hidden_states", "cond_indicator", "cond_mask")


def latent_batches(latents_dir, batch_size=1):
    """Yield (possibly stacked) clip batches from cosmos_data.py .pt files. All clips share shape, so we
    concat batch_size of them along dim 0 -> uses the H200 instead of one-clip-at-a-time. Loops forever."""
    files = sorted(glob.glob(os.path.join(latents_dir, "*.pt")))
    assert files, f"no latents in {latents_dir} (run cosmos_data.py first)"
    while True:
        for i in range(0, len(files), batch_size):
            dicts = [torch.load(f, map_location="cpu") for f in files[i:i + batch_size]]
            batch = {k: torch.cat([d[k] for d in dicts], dim=0) for k in _BATCH_KEYS}
            batch["padding_mask"] = dicts[0]["padding_mask"]   # [1,1,H,W], broadcast over batch by the DiT
            yield batch


def make_noisy_input(batch, model, p_mean=0.0, p_std=1.0):
    """Build the Cosmos flow-matching transformer input from a clean-latent clip. Returns
    (fwd_kwargs, target_velocity, gen_mask). gen_mask is 1 on generated (loss) frames."""
    dev = next(model.parameters()).device
    dt = next(model.parameters()).dtype
    x = batch["clean_latents"].to(dev, dt)                 # [B,16,T,Hl,Wl]
    ci = batch["cond_indicator"].to(dev, dt)               # [B,1,T,1,1]
    cond_mask = batch["cond_mask"].to(dev, dt)             # [B,1,T,Hl,Wl]
    padding_mask = batch["padding_mask"].to(dev, dt)       # [B,1,H,W]
    enc = batch["encoder_hidden_states"].to(dev, dt)       # [B,512,1024]
    B, C, T, Hl, Wl = x.shape

    # sample one flow time s per (batch, frame): ln(sigma)~N(p_mean,p_std) -> s = sigma/(sigma+1)
    sigma = torch.exp(p_mean + p_std * torch.randn(B, 1, T, 1, 1, device=dev, dtype=torch.float32))
    s = (sigma / (sigma + 1.0)).to(dt)                     # [B,1,T,1,1] in (0,1)
    eps = torch.randn_like(x.float()).to(dt)
    z_s = (1 - s) * x + s * eps                            # rectified-flow interpolant on every frame
    # generated frames use z_s; conditioning frames use clean x at t_conditioning
    hidden = ci * x + (1 - ci) * z_s
    timestep = ci * dt_const(T_CONDITIONING, s) + (1 - ci) * s   # [B,1,T,1,1]
    v = (eps - x)                                          # velocity target (supervised on gen frames)
    gen_mask = (1 - ci).expand(B, C, T, Hl, Wl)            # [B,16,T,Hl,Wl]

    fwd = dict(hidden_states=hidden, timestep=timestep, encoder_hidden_states=enc,
               condition_mask=cond_mask, padding_mask=padding_mask, fps=FPS)
    return fwd, v, gen_mask


def dt_const(val, like):
    return torch.full_like(like, val)


def _trainable(model, groups=("feat_map",)):
    for p in model.parameters():
        p.requires_grad_(False)
    params = [p for n, p in model.named_parameters() if any(g in n.lower() for g in groups)]
    for p in params:
        p.requires_grad_(True)
    return params


def stage1_attn_transfer(model, linear_layers, data_iter, steps=2000, lr=1e-3, p_mean=0.0, p_std=1.0,
                         accum=1, log_every=20, save_dir=None):
    """Train ONLY Hedgehog phi to match softmax teacher attn1 outputs, on real noised inputs. GBS = micro*accum."""
    feat_params = _trainable(model, ("feat_map",))
    opt = torch.optim.AdamW(feat_params, lr=lr, weight_decay=0.0, betas=(0.9, 0.95))
    print(f"[stage1] {sum(p.numel() for p in feat_params)/1e6:.2f}M feat_map params, lr={lr}, steps={steps}, accum={accum}", flush=True)
    model.train(); t0 = time.time(); oom = 0
    for step in range(steps):
        opt.zero_grad(); acc = 0.0; bad = False
        for _ in range(accum):
            batch = next(data_iter)
            try:
                fwd, _v, _m = make_noisy_input(batch, model, p_mean, p_std)
                loss = attn_transfer_loss(model, linear_layers, fwd) / accum
                loss.backward(); acc += loss.item() * accum
            except torch.cuda.OutOfMemoryError:
                oom += 1; bad = True; torch.cuda.empty_cache(); break
        if bad:
            opt.zero_grad(set_to_none=True); print(f"[stage1] step {step} OOM-skip (#{oom})", flush=True); continue
        torch.nn.utils.clip_grad_norm_(feat_params, 1.0); opt.step()
        if step % log_every == 0:
            print(f"[stage1] step {step}/{steps}  attn_mse {acc/accum:.5f}  {time.time()-t0:.0f}s", flush=True)
        if save_dir and step > 0 and step % 500 == 0:
            _save(model, save_dir, step, tag="s1")
    if save_dir:
        _save(model, save_dir, steps, tag="s1")
    return model


def stage2_flow_finetune(model, data_iter, steps=3000, lr=2e-4, p_mean=0.0, p_std=1.0,
                         log_every=20, save_dir=None):
    """End-to-end flow-matching finetune: predict velocity v=eps-x_clean on generated frames. Trains
    feat_map + LoRA + AttnRes (everything the surgery left trainable)."""
    params = _trainable(model, ("feat_map", "lora", "attnres_coord"))
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0, betas=(0.9, 0.95))
    print(f"[stage2] {sum(p.numel() for p in params)/1e6:.2f}M trainable, lr={lr}, steps={steps}", flush=True)
    model.train(); t0 = time.time(); ema = None; oom = 0
    for step in range(steps):
        batch = next(data_iter)
        try:
            fwd, v, gen_mask = make_noisy_input(batch, model, p_mean, p_std)
            pred = model(**fwd, return_dict=False)[0]                   # [B,16,T,Hl,Wl] = F
            diff = (pred.float() - v.float()) * gen_mask.float()
            loss = (diff ** 2).sum() / gen_mask.float().sum().clamp_min(1.0) / pred.shape[1]
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
        except torch.cuda.OutOfMemoryError:
            oom += 1; opt.zero_grad(set_to_none=True); torch.cuda.empty_cache()
            print(f"[stage2] step {step} OOM-skip (#{oom})", flush=True); continue
        ema = loss.item() if ema is None else 0.98 * ema + 0.02 * loss.item()
        if step % log_every == 0:
            print(f"[stage2] step {step}/{steps}  flow_mse {loss.item():.4f} (ema {ema:.4f})  {time.time()-t0:.0f}s", flush=True)
        if save_dir and step > 0 and step % 500 == 0:
            _save(model, save_dir, step, tag="s2")
    if save_dir:
        _save(model, save_dir, steps, tag="s2")
    return model


def stage2_output_distill(model, teacher, data_iter, steps=2000, lr=2e-4, p_mean=0.0, p_std=1.0,
                          accum=1, log_every=20, save_dir=None):
    """OUTPUT distillation (the direct 逼近教师 objective): match the linearized student's velocity to the
    PRISTINE softmax teacher's velocity on identical noised inputs. Trains feat_map+LoRA+AttnRes. GBS=micro*accum."""
    params = _trainable(model, ("feat_map", "lora", "attnres_coord"))
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0, betas=(0.9, 0.95))
    print(f"[s2-distill] {sum(p.numel() for p in params)/1e6:.2f}M trainable, lr={lr}, steps={steps}, accum={accum}", flush=True)
    model.train(); t0 = time.time(); ema = None; oom = 0
    for step in range(steps):
        opt.zero_grad(); acc = 0.0; bad = False
        for _ in range(accum):
            batch = next(data_iter)
            try:
                fwd, _v, gen_mask = make_noisy_input(batch, model, p_mean, p_std)
                with torch.no_grad():
                    tgt = teacher(**fwd, return_dict=False)[0]          # pristine teacher velocity
                pred = model(**fwd, return_dict=False)[0]               # linearized student velocity
                diff = (pred.float() - tgt.float()) * gen_mask.float()
                loss = (diff ** 2).sum() / gen_mask.float().sum().clamp_min(1.0) / pred.shape[1] / accum
                loss.backward(); acc += loss.item() * accum
            except torch.cuda.OutOfMemoryError:
                oom += 1; bad = True; torch.cuda.empty_cache(); break
        if bad:
            opt.zero_grad(set_to_none=True); print(f"[s2-distill] step {step} OOM-skip (#{oom})", flush=True); continue
        torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
        ema = acc/accum if ema is None else 0.98 * ema + 0.02 * (acc/accum)
        if step % log_every == 0:
            print(f"[s2-distill] step {step}/{steps}  distill_mse {acc/accum:.4f} (ema {ema:.4f})  {time.time()-t0:.0f}s", flush=True)
        if save_dir and step > 0 and step % 500 == 0:
            _save(model, save_dir, step, tag="s2d")
    if save_dir:
        _save(model, save_dir, steps, tag="s2d")
    return model


def _save(model, save_dir, step, tag="s1"):
    os.makedirs(save_dir, exist_ok=True)
    sd = {k: v.detach().cpu() for k, v in model.state_dict().items()
          if ("feat_map" in k) or ("lora" in k.lower()) or ("attnres_coord" in k)}
    from safetensors.torch import save_file
    save_file(sd, os.path.join(save_dir, f"linstate_{tag}_step{step}.safetensors"))
    print(f"[save] {len(sd)} trainable tensors -> {save_dir}/linstate_{tag}_step{step}.safetensors", flush=True)


def _load_linstate(model, path):
    from safetensors.torch import load_file
    sd = load_file(path)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    loaded = len(sd)
    print(f"[resume] loaded {loaded} linstate tensors from {path} (missing-in-ckpt ok)", flush=True)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, default=1)
    ap.add_argument("--transformer", type=str, default=None)
    ap.add_argument("--latents", type=str, default=None)
    ap.add_argument("--resume", type=str, default=None, help="linstate safetensors to warm-start (stage-1 -> stage-2)")
    ap.add_argument("--ratio", type=float, default=0.75)
    ap.add_argument("--method", choices=["flow", "distill"], default="flow",
                    help="stage-2 objective: flow=flow-matching on data, distill=match pristine teacher velocity (逼近教师)")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=1, help="micro-batch per GPU")
    ap.add_argument("--accum", type=int, default=1, help="grad-accum steps -> GBS = batch * accum")
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--p_mean", type=float, default=0.0)
    ap.add_argument("--p_std", type=float, default=1.0)
    ap.add_argument("--save", type=str, default="/home/wm-vla-rl/_ext/cosmos_lin_out")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    if args.smoke:
        from diffusers import CosmosTransformer3DModel
        torch.manual_seed(0)
        m = CosmosTransformer3DModel(num_layers=6, num_attention_heads=4, attention_head_dim=32,
                                     text_embed_dim=1024, max_size=(16, 16, 16), patch_size=(1, 2, 2),
                                     in_channels=17, out_channels=16, concat_padding_mask=True)
        m, lin, sm = linearize_cosmos(m, ratio_linear=0.75, lora_r=8, attnres_groups=3)
        # synth clip dict matching cosmos_data.py output (tiny), B=2 to exercise the batched path
        B, C, T, Hl, Wl = 2, 16, 2, 16, 16
        ci = torch.zeros(B, 1, T, 1, 1); ci[:, :, :1] = 1.0
        clip = dict(clean_latents=torch.randn(B, C, T, Hl, Wl),
                    encoder_hidden_states=torch.randn(B, 16, 1024),
                    cond_indicator=ci, cond_mask=ci.expand(B, 1, T, Hl, Wl).contiguous(),
                    padding_mask=torch.zeros(1, 1, 32, 32))   # [1,1,H,W], broadcast over batch
        di = (clip for _ in iter(int, 1))
        print("--- stage1 smoke ---")
        stage1_attn_transfer(m, lin, di, steps=12, lr=1e-2, log_every=4, save_dir=None)
        print("--- stage2 flow smoke ---")
        di = (clip for _ in iter(int, 1))
        stage2_flow_finetune(m, di, steps=8, lr=1e-3, log_every=4, save_dir=None)
        print("--- stage2 output-distill smoke ---")
        teacher = CosmosTransformer3DModel(num_layers=6, num_attention_heads=4, attention_head_dim=32,
                                           text_embed_dim=1024, max_size=(16, 16, 16), patch_size=(1, 2, 2),
                                           in_channels=17, out_channels=16, concat_padding_mask=True)
        di = (clip for _ in iter(int, 1))
        stage2_output_distill(m, teacher, di, steps=8, lr=1e-3, log_every=4, save_dir=None)
        print("=== SMOKE PASS: stage1 attn-transfer + stage2 flow + stage2 output-distill loops run ===")
        return

    assert args.transformer, "--transformer required"
    model = load_transformer(args.transformer)
    model, linear_layers, softmax_layers = linearize_cosmos(model, ratio_linear=args.ratio)
    if args.resume:
        _load_linstate(model, args.resume)
    di = latent_batches(args.latents, batch_size=args.batch)
    if args.stage == 1:
        stage1_attn_transfer(model, linear_layers, di, steps=args.steps, lr=args.lr or 1e-3,
                             p_mean=args.p_mean, p_std=args.p_std, accum=args.accum, save_dir=args.save)
    elif args.stage == 2 and args.method == "flow":
        stage2_flow_finetune(model, di, steps=args.steps, lr=args.lr or 2e-4,
                             p_mean=args.p_mean, p_std=args.p_std, save_dir=args.save)
    elif args.stage == 2 and args.method == "distill":
        teacher = load_transformer(args.transformer)   # pristine softmax teacher (no surgery)
        stage2_output_distill(model, teacher, di, steps=args.steps, lr=args.lr or 2e-4,
                              p_mean=args.p_mean, p_std=args.p_std, accum=args.accum, save_dir=args.save)
    else:
        raise ValueError(args.stage)


if __name__ == "__main__":
    main()
