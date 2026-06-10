# Causal-Forcing/inference.py — additive patches for ARL2-comparable VBench eval

Standard CF generation reused as-is; three additive, backward-compatible changes (teacher path unaffected):

1. **args**: `--surgery_ckpt --surgery_ratio(0.75) --surgery_groups(4) --surgery_lora_r(8) --vbench_naming`.
2. **surgery branch** (after base `--checkpoint_path` load, before `.to(bfloat16)`): import `wan_kda_surgery`,
   `patch_wan_flash_sdpa()`, `factorize_causal_wan(model,ratio)`, `attnres_ify_wan(model,groups)`,
   `add_lora_wan(model,lin,r)`, then `load_file(surgery_ckpt)` into `pipeline.generator.model` (strict=False).
   → builds OUR linearized student in-place on CF's STANDARD generator.
3. **output naming**: when `--vbench_naming`, write `'{full_prompt}-0.mp4'` (vbench_standard convention) instead
   of `'{prompt[:100]}.mp4'` (3 sites). Needed because 72/944 prompts exceed 100 chars and VBench standard mode
   matches `f'{prompt}-{i}.mp4'`.

Eval = run_vbench_eval.sh: gen teacher + student (DDP-2, 81 pixel frames) → official VBench.evaluate(vbench_standard).
