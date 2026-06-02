# AttnRes / KDA × WM-VLA-RL — integration proposal

> **Audience**: the Claude/engineer working on this `wm-vla-rl` repo.
> **Author**: the agent on the sibling project `/home/torchtitan_attention_residual`
> (Kimi-Linear AttnRes VLM: torchtitan training + sglang KDA inference + GRPO/seq-KD).
> **Status**: proposal — evaluate it, don't take it as gospel. Dates: drafted 2026-06-02.

This is a cross-project design note: how the **Block-AttnRes + KDA linear-attention**
stack (built & validated on the sibling project) can become the **efficient
long-horizon temporal backbone of this project's latent world model**, and how the
two projects' infra/lessons compose.

---

## TL;DR thesis

**"Long-video AttnRes" in this repo's correct form = "long-horizon latent-world-model
AttnRes."** A robot episode (LIBERO = 512 steps) is a long temporal sequence. The
V-JEPA2-AC predictor already does autoregressive next-latent prediction under a
**block-causal** mask — that is exactly the place a **linear-attention (KDA) +
Block-AttnRes** backbone replaces O(N²) attention with **O(N)** and unlocks much
longer rollout horizons. The world model's "compressed history" **is** KDA's
recurrent SSM state; rollout KV-caching becomes KDA recurrent-state caching (for
which we just built & validated the cache machinery — see §"Infra reuse").

This sidesteps the capacity trap of training a small video model from scratch: you
keep the **pretrained, frozen V-JEPA2-AC encoder** and only swap the **305M
predictor's** temporal attention. Robotics/WM tasks are structured with **dense**
reward (unlike prior-saturated VQA), so a small efficient predictor is viable here.

---

## What AttnRes and KDA are (self-contained)

Both live in `/home/torchtitan_attention_residual` (training: `torchtitan/`;
inference: `sglang/` on branch `attention_residual_inference`).

- **KDA (Kimi Delta Attention)** = a **linear / state-space attention** (gated delta
  rule, fla-style triton kernels). Cost is **O(N)** in sequence length; it carries a
  **recurrent SSM state** (a fixed-size summary of all past tokens) instead of an
  O(N) KV cache. sglang backend: `sglang/python/sglang/srt/layers/attention/linear/kda_backend.py`.
- **Block-AttnRes (Block Attention Residual, Kimi paper §5)** = a learned
  **aggregation across a set of units** (currently transformer layer-blocks): per-unit
  scalar gate + a **two-phase online-softmax merge** (Phase-1 inter-block attention,
  Phase-2 streaming online-softmax combine). The two-phase merge is **streamable** —
  you can fold new units into a running state. Overlay:
  `sglang/python/sglang/srt/models/attn_res_overlay.py`; training model:
  `torchtitan/torchtitan/experiments/attention_residual/`.
- **The generalization that matters here**: AttnRes is "online-softmax-merge
  aggregation across a set of units." The units don't have to be layer-blocks — they
  can be **frames / timesteps**. That is the temporal world model.

### What's validated on the sibling project (so you can trust the pieces)
- **KDA recurrent-state + prefix/radix caching works** (sibling "Fix A", upstream PR
  candidate #15): we wired Kimi-Linear into sglang's `MambaRadixCache` so the KDA
  **recurrent state is checkpointed/restored across a prefix** correctly — verified on
  the official `Kimi-Linear-48B-A3B` (radix-on vs off greedy **4/4 identical**, incl.
  shared-long-prefix). **This is the exact primitive a WM latent rollout needs**
  (reuse/cache the recurrent state across timesteps). Doc:
  `/home/torchtitan_attention_residual/Raising_PRs/PR15_sglang_kimi_linear_mamba_radix_cache/`.
- **2-GPU GRPO infra works** (8→2 migration done): `phase11_rlhf_grpo_infra/rlhf/run_grpo_llava_kimi_2gpu.py`
  (trainer FSDP=1 on GPU0 + sglang generator on GPU1, save_dcp, periodic ckpt). This
  is precisely the migration your Module-2 PLAN flags as a risk.
- **Hard-won RL lessons** (see §"Lessons").

---

## Why this repo is the right vehicle (not from-scratch 447M video)

- Training a tiny video model from scratch hits capacity + data + a real capability
  ceiling (we proved a 447M VQA model plateaus and GRPO can't push it). **Don't.**
- This repo uses a **frozen pretrained V-JEPA2-AC encoder (ViT-G, ~1B)** + a **305M
  predictor**. AttnRes only touches the predictor's temporal attention → no
  from-scratch capacity fight; you stand on strong pretrained latents.
- Robot/VLA tasks are **lower-dimensional and densely-rewarded** (success/world-model
  reward, not prior-saturated multiple-choice) → a small efficient temporal predictor
  is genuinely viable here, unlike open-ended VQA.

---

## Integration plan — 3 levels

### Level 1 — Architecture: AttnRes+KDA predictor (core contribution)
Replace the predictor's block-causal **O(N²) SDPA** with **KDA linear attention** so
the latent world model is **O(N)** and can roll out longer horizons.

- **Where**: `_ext/vjepa2/src/models/utils/modules.py:114` `ACRoPEAttention`
  (currently `F.scaled_dot_product_attention(..., attn_mask=block_causal)`), driven by
  `_ext/vjepa2/src/models/utils/modules.py:12` `build_action_block_causal_attention_mask`
  and `_ext/vjepa2/src/models/ac_predictor.py:17` `VisionTransformerPredictorAC`.
- **How**: behind a flag (`use_linear_attn`), route the **temporal axis** through a
  KDA-style linear-attention kernel; the block-causal-by-frame structure is exactly
  the recurrence linear attention expresses natively (frame `t` only attends ≤ `t`).
- **Reuse**: the KDA kernel + the AttnRes training model from the sibling repo
  (`torchtitan/.../experiments/attention_residual/` + `kda_backend.py`).

### Level 2 — Rollout = KDA recurrent-state rollout (the cache win)
Implement the latent rollout as a **recurrent-state** loop, not full re-attention.

- **Where**: `module1_vjepa2ac/rollout.py:15` (stub `rollout(predictor, z0, actions,
  ee_states, use_cache=True)`).
- **How**: carry KDA's recurrent SSM state across timesteps; each step folds the new
  frame's (latent, action, state) into the running state in **O(1)** rather than
  re-attending the whole prefix. This is the world-model analogue of KV-cache, but
  the cache is a fixed-size **state**, not a growing KV — so memory is flat over the
  horizon.
- **Reuse**: the **MambaRadixCache state-checkpoint/restore** primitive we just
  validated (PR #15) — same mechanism (snapshot/fork the recurrent state at a node,
  `copy_from` to restore). For CEM/MPC (`cem_planner.py:18`) this lets you fork the
  state at the planning root and roll out N candidate action sequences cheaply from a
  shared history.

### Level 3 — RL + reward (reuse, and apply our lessons)
- **2-GPU GRPO**: reuse the sibling's `run_grpo_llava_kimi_2gpu.py` 8→2 layout
  (trainer + sglang generator colocated on 2 GPUs, save_dcp). Maps onto your
  `_ext/SimpleVLA-RL/.../rob_rollout.py:424` + `verl/trainer/main_ppo.py`.
- **Reward**: your `module2_vla_rl/reward_latent_wm.py:20` (latent-WM dense reward) is
  the right call — see the zero-variance lesson below; binary success reward
  (`reward_binary.py`) collapses GRPO.
- **Note on the policy side**: OpenVLA-7B-OFT uses **parallel decoding**, so an
  sglang/KDA backend gives **limited** rollout speedup there. **Put AttnRes/KDA on the
  world model (sequential/recurrent), not on the VLA policy.**

---

## Infra reuse — cross-repo pointers

| Need (this repo) | Reuse from `/home/torchtitan_attention_residual` |
|---|---|
| Linear-attn temporal kernel | `sglang/.../layers/attention/linear/kda_backend.py`; AttnRes training model `torchtitan/.../experiments/attention_residual/` |
| Recurrent-state rollout cache | sglang `MambaRadixCache` wiring (PR #15) — state checkpoint/fork/restore; see `Raising_PRs/PR15_sglang_kimi_linear_mamba_radix_cache/PR.md` |
| 2-GPU RL (FSDP trainer + generator) | `phase11_rlhf_grpo_infra/rlhf/run_grpo_llava_kimi_2gpu.py` (+ `PolicyTrainer.save_dcp` in `torchtitan/.../experiments/rl/actors/trainer.py`) |
| AttnRes training harness | `torchtitan` AttnRes flavors + `parallelize` |

---

## Main technical challenges (be honest)

1. **3D RoPE × linear attention** (the one real hurdle). `ACRoPEAttention` rotates
   q/k separately over **(T, H, W)** (`modules.py:168-250`). KDA's recurrence is
   along **one** axis. Options: (a) put **time** on the KDA recurrence and keep
   spatial (H,W) as intra-frame full/structured attention (hybrid — clean, recommended
   first); (b) flatten to 1D with a fused positional scheme. Start with (a).
2. **Quality parity vs SDPA**. Linear attention is an approximation; validate the
   AttnRes predictor matches the SDPA baseline's next-latent prediction (MSE / probe)
   **before** claiming the long-horizon win.
3. **Action/state token interleaving** (`ac_predictor.py`) must be preserved in the
   recurrent formulation (the action/state tokens are part of the per-step input).

---

## Unified narrative (why this is one system, not two projects)

- This repo's thesis: **reverse-KL unifies RL (anti policy-drift) + distillation
  (anti mode-collapse)** — the *learning-objective* layer.
- AttnRes/KDA thesis: **linear-attention + block-residual = efficient long-context
  aggregation** — the *architecture/substrate* layer.
- **Together**: *an efficient (linear-attention) long-horizon latent world model that
  makes RL planning + on-policy distillation tractable over long robot episodes.*
  AttnRes/KDA is the substrate that makes long-horizon WM + RL affordable; reverse-KL
  is the objective on top. Complementary layers of the same system.

---

## Recommended first milestone (self-contained, demonstrable)

**Do Level 1 + Level 2 on the V-JEPA2-AC predictor**, then measure three things:
1. **Quality parity**: AttnRes/KDA predictor next-latent MSE ≈ SDPA baseline (on the
   same LIBERO clips / probe).
2. **Long-horizon scaling**: roll out **horizons the O(N²) baseline can't afford**
   (memory flat, latency linear) — this is the headline AttnRes result.
3. **Latency**: per-step rollout time (target < the current 0.93s/64-frame forward;
   recurrent-state should help most at long horizons).

If parity holds and long-horizon scales → that alone is a publishable/demonstrable
AttnRes contribution **on a real downstream (planning/RL)**, and Level 3 (RL with the
sibling's 2-GPU GRPO infra) follows.

---

## Lessons from the sibling project's GRPO (directly relevant to Module-2)

- **Zero-variance advantage trap** (you already flag this): with a **binary** success
  reward, groups that are all-success or all-fail have ~0 within-group reward variance
  → GRPO advantage ≈ 0 → no gradient. **Your `reward_latent_wm.py` dense reward is the
  fix.** (We hit this and confirmed: dense, varied reward is essential.)
- **Validate the rollout is actually grounded before trusting reward.** On the sibling
  project a *radix-cache state bug* made rollouts silently ignore the image (fluent
  but blind) → reward looked "flat/low" and we nearly mis-diagnosed it as a model
  ceiling. **Always eyeball real rollout samples + check reward+ rate vs a known
  offline metric before concluding "the model can't learn."**
- **Capability ceiling is real at tiny scale.** A 447M VQA model genuinely plateaued
  (strong advantage, ~120 steps, no gain) — RL can't add capacity. For this repo the
  implication is: lean on the **pretrained encoder + structured dense-reward task**;
  don't expect RL to manufacture capability the backbone lacks.

---

## Pointers to read on the sibling side
- `/home/torchtitan_attention_residual/Raising_PRs/PR15_sglang_kimi_linear_mamba_radix_cache/PR.md`
  (KDA recurrent-state radix caching — the rollout-cache primitive).
- `/home/torchtitan_attention_residual/UPSTREAM_PR_LIST.md` (#5 AttnRes overlay, #13/#15 radix, KDA fixes).
- `sglang/.../models/attn_res_overlay.py` (two-phase AttnRes merge), `.../linear/kda_backend.py` (KDA).
- `phase11_rlhf_grpo_infra/rlhf/run_grpo_llava_kimi_2gpu.py` (2-GPU GRPO reference).
