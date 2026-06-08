# Cosmos-Predict2-2B Video World-Model Linearization (LoLCATs + Block-AttnRes)

Linearize NVIDIA **Cosmos-Predict2-2B-Video2World** (a 2B diffusion video DiT, the scale-credible
SDPA teacher in the SANA-WM weight class) from O(N²) softmax self-attention to **O(N) linear attention**,
via distillation (no pretraining), and recover generation quality. Target competitor: **SANA-WM** (born-linear,
64×H100×15d). Our differentiation: *distillation instead of pretraining* + cross-layer **AttnRes**.

## Why Hedgehog (not KDA)
Cosmos `attn1` is **bidirectional** 3-D self-attention (`is_causal=False`) over all spatio-temporal tokens.
KDA / gated-delta is causal-only → does not fit. We use **Hedgehog** learnable feature maps
(`φ(x)=softmax([Wx,−Wx])`, ICLR'24) — the LoLCATs operator, bidirectional, trained to mimic softmax.

## Architecture (surgery)
- `cosmos_hedgehog.py` — `HedgehogFeatureMap` + `HedgehogCosmosProcessor` (bidirectional linear attn,
  mirrors the original q/k/v→RMSNorm→RoPE then replaces softmax with `φ(q)(φ(k)ᵀv)/φ(q)Σφ(k)`).
  `cosmosify_hedgehog` linearizes 75% of blocks; softmax kept on evenly-spaced anchors (SANA-style hybrid).
  Unified DiT adapters (`get_dit_blocks`/`get_self_attn`) so the surgery generalizes to other diffusers DiTs.
- `cosmos_surgery.py` — full surgery: Hedgehog(attn1) + **LoRA**(q/k/v/out) + **Block-AttnRes** (Kimi 2603.15031,
  cross-layer depth aggregation, zero-init → starts as no-op). Freezes base; ~0.6% trainable.
  `attn_transfer_loss` = MSE(student-Hedgehog attn1 out, softmax-teacher attn1 out) on identical input.

## Data + training (LoLCATs 2-stage)
- `cosmos_data.py` — encode LIBERO video clips → `AutoencoderKLWan` latents (`(z−mean)/std·σ_data`,
  temporal /4, spatial /8) + T5 (d_model 1024) text embeds + cond/padding masks. Faithful to
  `Cosmos2VideoToWorldPipeline`.
- `cosmos_distill.py`
  - **Stage-1 attn-transfer**: freeze all, train only Hedgehog φ to match the softmax teacher (MSE).
  - **Stage-2 flow-matching finetune**: unfreeze φ+LoRA+AttnRes, train the Cosmos velocity objective.
    Parameterization **derived from and algebraically verified against the sampler**:
    input `z_s=(1−s)x+sε`, timestep `s=σ/(σ+1)`, model predicts velocity **`v=ε−x`**,
    cond frames clean at `t≈1e-4`, loss only on generated frames.
- `cosmos_eval.py`
  - `efficiency`: full-model latency + peak-mem vs video length T (softmax O(N²) OOMs; linear O(N) continues).
  - `quality`: held-out velocity-prediction MSE, linearized vs softmax teacher (gap = linearization cost).

## Run
```bash
# orchestrated overnight chain (data → S1 → S2 → eval), GPU0:
CUDA_VISIBLE_DEVICES=0 nohup bash cosmos_overnight.sh > cosmos_overnight.log 2>&1 &
# or fire automatically when another GPU job frees the card:
nohup bash cosmos_wait_launch.sh &
```
venv: `/home/venv/cosmos_predict2` (torch2.12/cu130, diffusers0.38). Model snapshot under
`/home/cosmos_predict2_model/...Cosmos-Predict2-2B-Video2World/`.

> These are tracked snapshots of the live scripts in `_ext/` (the ignored scratch/run dir).
