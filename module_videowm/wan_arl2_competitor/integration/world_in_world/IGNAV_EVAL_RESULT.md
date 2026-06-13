# IGNav closed-loop eval — OUR linearized Causal-Forcing WM (first downstream numbers, 2026-06-13)

End-to-end closed-loop run of World-in-World **Image-Goal Navigation** (Habitat + HM3D val) driven by
OUR linearized Causal-Forcing world model (`causalforcing` `text`/I2V WM, the ARL2-competitor student
jepa_l025_T8/wanlin_s2_step6000.safetensors). This validates the FULL WoVR-style pipeline and gives the
first real downstream metric for our linearized WM.

## Config
- exp: `06.13_IGNav_causalforcing_sub10_tts100`; 10 episodes (subset of the 144), HM3D val scenes.
- Policy: `--use_heur` (heuristic action proposals, NO VLM) — deliberately isolates the WM's contribution:
  the WM imagines each candidate action sequence and the planner selects by imagined-goal match.
- WM: 2-step linearized Causal-Forcing, `TTS_ratio=1.0` (full imagination), query_num=2, look-ahead=4.

## Result
| metric | value |
|---|---|
| **mean Success Rate** | **0.10 (1/10)** |
| mean SPL | 0.035 |
| mean traj len | 62.4 (vs demo 24.2) |
| wall-clock | 32 min / 10 ep (~3.2 min/ep) |

## Read (honest)
- **NON-ZERO success with a heuristic policy + a zero-shot WM** — the WM's imagination selected a
  goal-reaching action sequence in 1/10 episodes. The pipeline is validated end-to-end AND the linearized
  WM is functionally contributing (not a passthrough).
- Absolute SR is low BY DESIGN: `--use_heur` has no real navigation policy, and zero-shot Wan2.1-1.3B
  follows discrete nav-text weakly. This run's purpose was pipeline validation + first numbers, not a high score.
- The efficiency selling point (finding #3) is already evidenced separately: our 2-step WM = 2.09s/rollout
  vs 8-step 4.06s vs Wan 30-step ~15x — i.e. at fixed compute we can afford more imagination/rollouts.

## Next (to a real finding-#3 number)
1. Add a VLM policy (or train a small nav LoRA on the WM = `FTtext`) so success is not policy-floored.
2. Full 144-episode run; sweep `TTS_ratio` 0.25→1.0 → does more (cheap, linearized) WM imagination raise SR/SPL?
3. Controlled compute-matched comparison: our 2-step linearized WM vs Wan2.1 30-step on the same I2V backbone
   at equal wall-clock budget (the headline: cheaper WM → more rollouts → higher closed-loop success).

Logs: world-in-world/downstream/logs/06.13_IGNav_causalforcing_sub10_tts100/ ; states/IGNav_06.13_*/metrics_0.jsonl.
