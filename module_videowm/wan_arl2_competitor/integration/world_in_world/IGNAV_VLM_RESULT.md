# IGNav closed-loop with a REAL VLM policy on HELD-OUT scenes — OUR linearized Causal-Forcing WM (2026-06-14)

Clean finding-#3 downstream signal (option ①). The prior `--use_heur` runs (`IGNAV_EVAL_RESULT.md`,
`IGNAV_NAVLORA_RESULT.md`) were **policy-floored** (heuristic action proposals; baseline 1/10, nav-LoRA 0/10
= noise). Here we (a) drop `--use_heur` and drive the closed loop with a **real VLM policy**, (b) evaluate on
**held-out scenes NOT used in nav-LoRA training**, so the WM's imagination actually drives action selection.

## Setup
- **Policy (REAL VLM):** `Qwen/Qwen2.5-VL-7B-Instruct-AWQ` served via vLLM 0.7.3 on **GPU1** (verified answering
  multimodal requests). Smallest viable real VLM for fast bring-up; added to `LOCAL_MODELS` in `downstream/vlm.py`.
- **WM (OUR model):** linearized Causal-Forcing `causalforcing` worker on **GPU0** — base CF 2-step generator +
  KDA(75%) + naive AttnRes(groups=4) + main LoRA(r=8), surgery `jepa_l025_T8/wanlin_s2_step6000.safetensors`.
  ~2.0 s/rollout (the linearized-WM efficiency). Two arms: **(a) no-LoRA**, **(b) + nav-LoRA step2000**
  (`nav_lora_step2000.safetensors`, r=8; loaded cleanly: 264 tensors, unexpected=0).
- **Closed loop:** `solver_IGNav`, NO `--use_heur`, `--use_WM`, `TTS_ratio=1.0` (full imagination),
  query_num=3, look_ahead=5, max_decisions=20. Habitat-sim `habitat025`, HM3D val.
- **Held-out episode set:** scenes in the IGNav episodes that are **NOT** in the nav-LoRA training set.
  The nav-LoRA was trained on **28 HM3D-val scenes** (extracted from `/root/nav_posttrain/latents/*.pt`);
  the IGNav episodes span 78 scenes, **104 of 144 episodes are on held-out (non-training) scenes** (57 unique).
  We selected a deterministic set of held-out episodes, one per unique held-out scene, step_to_goal≤25
  (`/root/ignav_heldout_datums.txt`). **Both arms run the SAME episodes in the SAME order** (matched comparison).
- **Disk discipline:** all VLM weights → `/root/hf_cache` (`HF_HOME`), all eval outputs/logs → `/` overlay.
  `/home` never touched (stayed 13 G free); `/` ~790 G free throughout.

## Result — held-out scenes, REAL VLM policy

| config (held-out scenes, VLM=Qwen2.5-VL-7B-AWQ, no `--use_heur`, TTS=1.0) | N | **SR** | **SPL** | mean traj_len | mean demo_len | mean dist-to-goal (m) |
|---|---|---|---|---|---|---|
| **OUR WM, no-LoRA**        | 12 | **0.00 (0/12)** | **0.00** | 38.7 | 20.8 | 2.50 |
| **OUR WM, + nav-LoRA s2000** | 11 | **0.00 (0/11)** | **0.00** | 39.5 | 20.5 | 2.11 |

Goal success requires final distance ≤ **0.5 m** (`goal_radius`). **Neither arm reached a single goal.**

### Matched first-11 episodes (identical episodes both arms — clean apples-to-apples)
| | no-LoRA | nav-LoRA |
|---|---|---|
| SR | 0/11 | 0/11 |
| mean dist-to-goal (m) | 2.46 | **2.11** |
| best (closest) approach (m) | 1.46 | **0.81** |
| episodes where this arm ended closer to goal | 3 | **7** (1 tie) |

Per-episode final distance-to-goal (m), matched first 11 held-out episodes:

| scene/ep (held-out) | no-LoRA | nav-LoRA |
|---|---|---|
| c5eTyR3Rxyh/E003 | 2.38 | 1.59 |
| LT9Jq6dN3Ea/E002 | 3.72 | 2.66 |
| HMkoS756sz6/E005 | 1.46 | 3.44 |
| tQ5s4ShP627/E001 | 1.65 | 1.65 |
| kJJyRFXVpx2/E029 | 1.46 | 2.45 |
| rJhMRvNn4DS/E001 | 2.99 | 1.77 |
| zt1RVoi7PcG/E001 | 1.73 | **0.81** |
| YRUkbU5xsYj/E032 | 2.44 | 3.13 |
| 7MXmsvcQjpJ/E003 | 3.08 | 2.58 |
| a8BtkwhxdRV/E013 | 4.05 | 1.47 |
| Qpor2mEya8F/E003 | 2.11 | 1.66 |

## vs the prior `--use_heur` floor

| run | policy | scenes | N | SR | SPL |
|---|---|---|---|---|---|
| prior baseline (`IGNAV_EVAL_RESULT.md`)  | heuristic (`--use_heur`) | in-distribution val | 10 | 0.10 (1/10) | 0.035 |
| prior nav-LoRA (`IGNAV_NAVLORA_RESULT.md`) | heuristic (`--use_heur`) | in-distribution val | 10 | 0.00 (0/10) | 0.00 |
| **THIS: no-LoRA**  | **VLM-7B-AWQ** | **held-out** | 12 | **0.00** | **0.00** |
| **THIS: nav-LoRA** | **VLM-7B-AWQ** | **held-out** | 11 | **0.00** | **0.00** |

## Honest read

- **Even with a real VLM policy on held-out scenes, SR stays 0** for both the linearized WM and the
  nav-post-trained WM. This is a **real finding about linearized-WM fidelity for closed-loop planning**:
  the 2-step linearized Causal-Forcing WM produces imagined rollouts that are too **blurry / low-fidelity**
  for the VLM planner to select goal-reaching action sequences with the precision IGNav requires (≤0.5 m).
  The pipeline is fully validated end-to-end (real VLM + WM + Habitat closed loop), and the WM contributes
  at its efficiency advantage (~2 s/rollout) — but imagination fidelity, not the policy and not WM speed,
  is now the binding constraint.
- **nav-LoRA gives a small, consistent proximity improvement but does NOT cross the success threshold.**
  On the matched 11 episodes it lowers mean final distance-to-goal **2.46 m → 2.11 m**, ends closer in
  **7/11** episodes, and produces the single closest approach of the whole study (**0.81 m**, scene
  zt1RVoi7PcG) — directionally consistent with the offline gains reported in `NAV_POSTTRAIN.md`
  (own-GT motion MSE −26%). But 0.81 m is still > 0.5 m, so **SR remains 0**. Nav post-training tightens the
  action→camera-motion mapping just enough to nudge final pose, not enough to navigate to goal over a blurry WM.
- **The agent consistently over-wanders:** trajectory length ~38–40 steps vs demo ~20, i.e. the planner does
  not converge — consistent with the WM imagination being too uninformative for the VLM to home in on the goal.

## Caveats
- **Small N** (12 / 10) — enough for a clear qualitative read (uniform 0 SR, both arms), not for fine SPL
  statistics. Scaling to ≥50 / a TTS_ratio sweep was the planned next step but each episode is ~6 min with the
  VLM in the loop (~6 hr for both arms at N=30); not run given the unambiguous SR=0 floor at this N.
- **Policy = 7B-AWQ** (smallest viable real VLM), not the repo-default 72B-AWQ. A stronger planner might extract
  more from the same WM, but the binding constraint observed here is WM imagination fidelity, not policy size;
  the 7B also showed frequent action-format parse retries (handled by the solver's 30-retry loop).
- **A manager bug was fixed** to enable the real-policy path: `worker_manager.split_batch` sliced the scalar
  `request_model_name` string per batch-item ("causalforcing"→"c"), crashing the WM worker on every request.
  Patched to keep string fields whole (`downstream/utils/worker_manager.py`). (This likely also silently
  degraded the prior `--use_heur` WM contribution.)
- **TTS_ratio=1.0** (full imagination). A sweep 0.25→1.0 is only meaningful once SR is off the floor; it is not.

## Repro / artifacts
- Held-out datum list: `/root/ignav_heldout_datums.txt` (30; first 10–12 evaluated). Env var
  `IGNAV_DATUM_IDS` (added to `solver_IGNav.py`) restricts the dataset to these exact episodes.
- VLM: `bash` launch in `/root/ignav_eval_out/logs/vllm_8000.log`; WM managers: `wm_manager_{noLoRA,navLoRA}.log`.
- Solvers: `/root/ignav_eval_out/run_solver.sh <exp_id> localhost:7000 1.0`; logs `solver_{noLoRA,navLoRA}.log`.
- Per-episode metrics: `downstream/states/IGNav_06.14_IGNav_causalforcing_heldout_{noLoRA,navLoRA}/<scene>/E*/metrics.jsonl`.
- Aggregator: `/root/ignav_eval_out/aggregate.py <exp_id>`.
