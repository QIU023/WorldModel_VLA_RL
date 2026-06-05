# WorldVLA-7B × KDA(75% Kimi)+AttnRes — time-axis linearization surgery

Heavy WAM experiment: take a WAM SOTA (WorldVLA, Chameleon-7B autoregressive Action World Model)
and **surgically replace 75% of its self-attention with KDA (Kimi Delta Attention, O(N) gated
delta-rule linear attention)** + **cross-layer Block-AttnRes**, then SFT-adapt from the released
SOTA checkpoint on LIBERO (no pretrain-from-scratch — attention-transfer / linearization, à la
LoLCATs / MOHAWK / Mamba-in-Llama).

## Why adapt-from-SOTA (not reproduce)
SDPA baseline == the original WorldVLA paper (LIBERO-Long 59.0% @256px). Reproducing it from
scratch is ~21 days on 2×H200; pointless. We fine-tune the **modified (KDA) version** directly
from the SOTA ckpt to recover performance, which is the legitimate, expected approach for
architecture-surgery papers.

## Surgery (code: `../kdaify_chameleon.py`)
- `kimi_kda_layers(n)` → KDA on layers where `(i+1)%4 != 0` ⇒ 24/32 KDA, 8 full-softmax (3:1 hybrid).
- `kdaify_chameleon(model, layers, init_from_softmax)` — weight-only copy of pretrained q/k/v/o into
  KDA modules (gated-delta params `A_log, dt_bias, f_a/f_b, g_a/g_b, b_proj, o_norm` fresh-init).
- `attnres_ify_chameleon(model, groups=8)` — wraps each decoder layer → `.inner.` + cross-layer
  grouped residual on the attention path.
- Injected in `pretrain_solver_awm_w_ck.py::_model_func` after `del model.model.vqmodel`
  (rank0 copies softmax weights; other FSDP ranks empty-init + `sync_module_states` broadcast).

## Training (`train_wvla_kda.sh`)
- Full-FT (NOT LoRA — we changed the architecture). 2×H200 FSDP, effective batch 32 (bs8×accum2),
  lr 5e-6, wd 0.1, clip 4, dropout 0.05, z_loss 1e-5, NO activation-checkpointing (143GB ample).
- closs 15.5 → **1.16** over 1 full epoch (12166 steps), ~3.8h/epoch, steady ~130GB/GPU.
- ⚠️ Known: CUDA-OOM at the epoch0→epoch1 boundary (the FSDP full-state-dict save spikes >139.8GB).
  epoch0 ckpt saved cleanly first; resume epoch1 as a fresh 1-epoch run from `epoch0/`.

## Eval (`eval_libero_kda_10.sh`, solver patch in `worldvla_solver.patch`)
Eval must rebuild the **exact surgery module tree before loading weights** (the saved state-dict has
`model.layers.N.inner.self_attn.*` KDA keys that won't bind to a vanilla SDPA arch):
`from_pretrained → kdaify → attnres_ify → load_state_dict(strict=False)`.
Validated: **loaded 523/523 tensors, missing(non-vq)=0, unexpected=0**; KDA triton kernel executes;
LIBERO-Long rollouts run. Compared against SDPA SOTA (`model_256/libero_10`, paper 59% @256px).

## Env fixes (gitignored `_ext`, recorded here for reproducibility)
- `LIBERO/.../benchmark/__init__.py::get_task_init_states` → `torch.load(..., weights_only=False)`
  (torch≥2.6 default `weights_only=True` blocks the numpy init_states).
- tokenizer paths → local `worldvla_ckpts/combined_tokenizer/`; `ck_5`→`ck_10` for libero_10 (Long).
- numpy pinned 1.26.4 (metaworld pulled 2.2.6 → broke matplotlib/LIBERO).
- launch via `/home/venv/libero/bin/torchrun`; `PYTHONPATH` includes LIBERO repo + worldvla + libero_util.
