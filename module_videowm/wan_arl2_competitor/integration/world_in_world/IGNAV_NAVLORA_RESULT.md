# IGNav re-eval: nav-LoRA vs baseline — HONEST NEGATIVE (2026-06-13)

Re-ran the World-in-World IGNav closed-loop eval with OUR nav-post-trained WM (nav LoRA step2000,
confirmed loaded: manager forwarded `--nav_lora_ckpt …nav_lora_step2000.safetensors --nav_lora_r=8`),
same 10-episode subset / same config as the baseline.

| config (10 ep, --use_heur, TTS=1.0) | Success Rate | SPL | wall-clock |
|---|---|---|---|
| baseline (no LoRA) | 0.10 (1/10) | 0.035 | 32 min |
| **nav-LoRA (step2000)** | **0.00 (0/10)** | **0.00** | 33 min |

## Verdict: nav post-training did NOT lift closed-loop IGNav success.
- 0/10 vs 1/10 at n=10 with `--use_heur` is **statistical noise** — the baseline's single success was almost
  certainly luck, not a real 10% capability. Both runs are at the policy floor.
- The nav LoRA DID improve the WM at the offline level (smoke gates all passed; own-GT motion MSE −26%;
  overfit −85.6%), but **that did not translate to navigation success**.
- **Root cause = the policy, not the WM.** `--use_heur` proposes heuristic/uninformed action candidates;
  the WM only *scores* them. A better WM cannot navigate if no candidate is good. The closed-loop bottleneck
  here is the absence of an intelligent policy, which WM post-training cannot fix.

## What this honestly establishes (and doesn't)
- ✅ Full WoVR-style pipeline works end-to-end (WM server + Habitat/HM3D + closed-loop eval), and OUR
  linearized WM + nav LoRA plug in cleanly and run at the efficiency advantage (2-step ~2s/rollout).
- ✅ Nav post-training of the linearized WM is mechanically valid and improves offline motion fidelity.
- ❌ It does NOT, on its own (heuristic policy, n=10, in-distribution val scenes), raise IGNav SR.

## To get a real finding-#3 signal (next, not done)
1. **Add a real VLM policy** (drop `--use_heur`) so the WM's imagination actually drives action selection —
   this is the missing piece; only then can "better/faster WM → higher SR" show.
2. Larger N (≥50 episodes) for statistical power; **train/val scene split** (nav data here used HM3D val =
   in-distribution) for a clean claim.
3. TTS_ratio sweep (0.25→1.0) once SR is off the floor.

Logs: world-in-world/downstream/logs/06.13_IGNav_causalforcing_navlora_sub10_tts100/.
