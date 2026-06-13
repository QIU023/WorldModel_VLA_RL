# Plugging OUR World Model into World-in-World (closed-loop eval)

ICLR 2026 Oral — "World Models in a Closed-Loop World" (arXiv 2510.18135).
Repo cloned to `/home/wm-vla-rl/_ext/world-in-world` (URL `https://github.com/World-In-World/world-in-world` is correct; matches the project page).
All claims below cite actual code I read. Compute on this box: 2x H200 (143 GB each), conda envs `base` + `py3.10`.

---

## TL;DR (honest fit verdict, read this first)

The task's premise — *"action-conditioned closed-loop almost certainly needs OUR-B (iVideoGPT), not OUR-A (text-only Causal-Forcing)"* — is **only half right, and the more useful half is the opposite.**

World-in-World's "unified action API" **turns actions into a TEXT prompt** for the entire diffusion-WM family (wan, ltx, cosmos, hunyuan, svd, wan22). Those WMs are **first-frame(image) + text** conditioned, *not* native-action conditioned. See `downstream/api_models/__init__.py:51-98` (`process_b_action_nav/manip/freetext`) and the I2V pipelines in `wan_model.py` / `svd_model.py`. The genuinely action-(pose/vector)-conditioned path exists but is a minority (NWM pose-deltas, `igen`/FTsvd `action_input_channel`), and the manipulation `action` mode is **currently disabled** in the release ("use `zero_shot_text`", `wiw_manip/planner/igenex_planner.py:129-130`).

Consequence:
- **OUR-A (Causal-Forcing, Wan2.1-1.3B, linearized) FITS the dominant `text`/`FTtext` WM API and is the BETTER demonstrator of the paper's finding #3** ("a faster WM lets you spend more inference-time compute → higher closed-loop success"), *because OUR-A's entire value proposition is inference speed.* Critically, Causal-Forcing's **frame-wise** model **natively supports I2V** (first-frame conditioning) — verified in `/home/wm-vla-rl/_ext/Causal-Forcing/inference.py:134-200` and the README ("frame-wise … natively unifying T2V and I2V"). So OUR-A satisfies the required image-conditioning. Caveat: zero-shot Wan2.1-1.3B obeying discrete nav text ("turn_left 22.5°") is weak (the paper itself shows zero-shot WMs underperform and they post-train = `FTwan21`), so for *real* success numbers you'd add a small navigation LoRA — but it is **API-compatible today**.
- **OUR-B (iVideoGPT + KDA, native action-conditioned) fits the minority `action` WM type** and is architecturally ideal for manipulation closed-loop (obs+action→next obs). BUT it is **heavier to land**: (i) iVideoGPT ships no action-conditioned OXE checkpoint (README: "we currently do not have an action-conditioned prediction model on OXE") — you must fine-tune on the target sim's action space (LIBERO/RLBench), and (ii) the repo's manip `action` WM mode is disabled for LIBERO in this release.

**Recommendation:** demonstrate finding #3 with **OUR-A as a `text`/`FTtext`-type WM on IGNav** (cheapest Habitat task, no detection model). Use OUR-B only if you specifically target RLBench manipulation via the `genex_manip`/`action` pattern and after training an action-cond checkpoint.

---

## STEP 2(a) — The WM plug-in API (what your WM must implement)

### The contract (one function)
Every WM worker is launched as a **long-lived subprocess** and ends with:
```python
worker_main(pipe_fd, task_fn)   # downstream/utils/worker_manager.py:660
```
`task_fn(input_dict) -> output_dict` is the *entire* interface. The manager talks to it over a length-prefixed-pickle protocol on stdin (in) and an inherited pipe FD (out) — `worker_main` at `worker_manager.py:660-699`, framing in `read_pickled_data_non_blocking`/`write_pickled_data` (`worker_manager.py:186-231`), receiver loop `receiver_for_worker` (`worker_manager.py:391-423`).

### Input dict (validated at `worker_manager.py:106-134`, `check_inputdict`, `server_type="world_model"`)
Required keys: `b_action`, `save_dirs`, `request_model_name`.
- `b_action`: `np.int64` array `[B, T]` of action IDs (nav), **or** `list` (manip = list of text instructions / pose lists). Action IDs map: `{forward:1, turn_left:2, turn_right:3, stop:4, placeholder:0}` (`api_models/__init__.py:2`).
- `save_dirs`: `list[str]` — one per batch item; the worker reads conditioning frame `cond_rgb.png` from here if `b_image` absent (`api_models/__init__.py:100-110`).
- `request_model_name`: `str` — must equal the worker's `args.world_model_name` (asserted in `process_input_dict`, `api_models/__init__.py:189`).
- `b_image` (optional): `np.uint8` `[B, C, H, W]` — the current observation frame, sent directly by the solver (`solver_base.py:659-660`). This is the **first-frame conditioning** for I2V.
- `return_objects` (optional): `list[bool]` — if true, return frames in-memory instead of writing to disk.

### Output dict (validated at `worker_manager.py:138-147`, `check_outputdict`)
- `save_dirs`: `list[str]` (required, echo input).
- `pred_frames` (optional): `np.uint8` `[B, T, C, H, W]` predicted future frames (T≈14). Required when `return_objects` (remote mode). Otherwise write frames to `save_dirs` via `save_predict` (`api_models/__init__.py:226-237`).

### Is it action-conditioned? — the "unified action API"
Conditioning is on **(first frame image) + (action rendered as text)** for the diffusion family. `process_b_action` (`api_models/__init__.py:88-98`) dispatches by `task_type`:
- `navigation`: IDs → `"Follow this sequence of camera motions: ['forward 0.2m', 'turn_left 22.5°', ...]."`
- `manipulation`: `"Follow the instruction to move the robotic arm: {instruction}."`
- `freetext`: passthrough.
The base class `DiffuserModel.inference_batch` (`api_models/__init__.py:266-276`) then calls `generate_pipe_images(txt_list, img_list)` — i.e. image+text→video. Native-action WMs bypass this: NWM converts IDs→continuous pose deltas (`nwm_model.py:284-431`), FTsvd/`igen` converts IDs→action-channel embeddings (`FTsvd/eval_inference.py`, `--action_input_channel`).

### How a new WM is registered (4 edits — `docs/05_add_new_WM.md`)
1. **Wrapper file** `downstream/api_models/<your_wm>_model.py` — subclass `DiffuserModel` (override `_load_pipe`, `_load_pipe_args`, optionally `generate_pipe_images`/`inference_batch`) and a `__main__` that calls `worker_main(pipe_fd, do_some_tasks)`. (Template-equivalent: `wan_model_diffsynth.py` is the closest example to OUR-A.)
2. **`downstream/vlm.py:27-33`** — add the symbolic `wm_type` to `WORLD_MODEL_TYPES` (`text` / `FTtext` / `camera` / `action`).
3. **`downstream/utils/worker_manager.py:752-758`** — add the name to the `worker_type` list so the manager launches it via `get_worldmodel_workers_cmd`.
4. **`downstream/utils/workers_cfg.py`** — add a `COMMON_ARGS[PLATFORM]["<name>_worker"]` entry = `[python_path, script_path, ...args]`. NOTE: `PLATFORM` defaults to `"Host3"` on any host whose name isn't `Host1/Host2` (`workers_cfg.py:5-11`) — so on **this** box add the entry under the `Host3` dict.

(Env spec file `downstream/api_models/env_config/<name>.txt` is optional/documentation.)

---

## STEP 2(b) — Task suite

| Task | Sim / dataset | Metric (code) | Extra models needed | 2-GPU feasible here? |
|------|---------------|---------------|---------------------|----------------------|
| **AR** (Active Recognition) | Habitat-sim + **MP3D** scenes; `episodes_AR.json.gz` | accuracy `int(target_categrory==pred)` + mean traj len (`evaluator.py:236-238`, `18-22`) | **SAM2** server + VLM + WM | Yes compute-wise; MP3D download is large/gated |
| **A-EQA** (Active Embodied QA) | Habitat-sim + **HM3D**; `episodes_AEQA.json.gz` | LLM-match score `(sigma-1)*25` via GPT-4o + efficiency (`evaluator.py:24-38`) | **Grounding-SAM2** + VLM + WM + **OpenAI key** | Yes; needs HM3D + OpenAI key |
| **IGNav** (Image-Goal Nav) | Habitat-sim + **HM3D**; `episodes_IGNav.json.gz` + `igdataset_goal_imgs.zip` | **success rate + SPL** `s*l/max(p,l)` (`evaluator.py:40-50`) | VLM + WM only (**no detection model** — README step 5) | **Yes — lightest Habitat task** |
| **Manipulation** | **RLBench** (CoppeliaSim/PyRep) *or* **LIBERO** backend; data ships in `downstream/world-in-world-manip/data` | success rate, aggregated by `wiw_manip/aggregate_results.py` | proposer (VLM planner / 3D-Diffuser-Actor / OpenPI) + WM (optional) | LIBERO path avoids the big scene download + CoppeliaSim → lightest to *bring up*, but WM `action` mode disabled (text-only) |

Compute on 2x H200 (143 GB) is not a constraint (the default 72B-AWQ VLM fits on one H200, or swap to `Qwen/Qwen2.5-VL-3B-Instruct-AWQ` listed in `vlm.py:42-47`, or skip the VLM proposer entirely with `--use_heur`). The real blockers are **(1) Habitat-sim 0.2.5 install (py3.9)** and **(2) multi-GB gated scene downloads (HM3D/MP3D)**.

---

## STEP 2(c) — Online planning strategy (how the WM drives the loop)

This is a VLM-proposes / WM-imagines / VLM-selects MPC-style look-ahead, gated by a test-time-compute knob. Per step (AR shown; IGNav/AEQA analogous):

1. **Stochastic imagination gate** — `inference_ar` (`solver_AR.py:501-520`): with probability `TTS_ratio` run `forward_with_WM`. `TTS_ratio` default **1.0** (`solver_AR.py:827`, `solver_IGNav.py:844`); docstring: "Probability of running forward_with_WM". **This is the inference-time-compute dial for finding #3.**
2. **Propose N candidate action sequences** — `forward_with_WM` (`solver_AR.py:526-547`): VLM planner `planner_N_action` proposes `query_num` (=2, `solver_AR.py:125`) candidate sequences of length `look_ahead_action_num` (=4, `solver_AR.py:126`); or heuristic sampling with `--use_heur`.
3. **WM rolls out each candidate** — `imagine_by_model_type` (`solver_base.py:691-738`) dispatches by `world_model_type`: `text`→`look_ahead_explore_non_pano` (perspective first frame + actions, `solver_base.py:517-525`), `action`/`FTtext`→`look_ahead_explore` (pano), `camera`→rgb+depth, `GTsim`→real sim rollout. All converge on `send_batch_to_server` (`solver_base.py:645-673`) which ships `{b_action, save_dirs, request_model_name, b_image}` to the WM manager and gets back `pred_frames`.
4. **Ground the imagined frames** — SAM2/Grounding-SAM2 draws bboxes on predicted frames (`solver_AR.py:570-572`).
5. **Inject imagined rollouts into the VLM context** — predicted obs + action sequences are appended (`imagine_obs_key`, `imagine_action_key`; `solver_AR.py:593, 598`); the planner/answerer VLM then picks the real next action and answers using these imagined futures (`fetch_action_decision_vlm` `solver_AR.py:445-477`).

So the WM is used as a **forward simulator for candidate action sequences**; selection is done by the VLM conditioned on the imagined outcomes. More imagination (higher `TTS_ratio`, more `query_num`) = more compute = (paper's claim) higher success — but only affordable if the WM is fast.

---

## STEP 2(d) — Env / deps / launch / compute

**Envs** (each component a separate conda env — `docs/01_setup_env.md`):
- `habitat025` (py3.9, habitat-sim 0.2.5 + habitat-lab pinned commit + open-eqa subtree) — runs the **solver**.
- `vllm` (vllm==0.7.3) — VLM policy server **and** the WM **manager** (manager kept separate from worker env).
- `sam2` (py3.10) — SAM2 / Grounding-SAM2.
- One env **per WM** (see table `docs/01_setup_env.md:305-324`). OUR-A would reuse the Causal-Forcing env.
- `wow-manip` (py3.9) + CoppeliaSim/PyRep/AMSolver for RLBench; or `uv` `.venv-libero` for LIBERO.

**Launch sequence** (`docs/03_run_commands.md`), all from repo root, ports auto-increment:
```bash
# 1. VLM policy (port 8000)
CUDA_VISIBLE_DEVICES=0 bash downstream/scripts/vllm_dsai.sh 8000 &
# 2. (AR) SAM2 (6001) / (AEQA) Grounding-SAM2 (6002)
CUDA_VISIBLE_DEVICES=1 bash downstream/scripts/init_sam2_manager.sh <exp> 1 &
# 3. WM manager (7000) — worker_type must be registered
CUDA_VISIBLE_DEVICES=1 bash downstream/scripts/init_worldmodel_manager.sh <exp> 1 <wm_type> &
# 4. Solver (IGNav example)
CUDA_VISIBLE_DEVICES=0 bash downstream/scripts/init_solvers.sh \
   downstream.solver_IGNav <exp> 4 "localhost:8000" localhost:7000 --use_WM &
# 5. Aggregate
PYTHONPATH=. python downstream/evaluator.py <exp> --task IGNav
```
Manager (`init_worldmodel_manager.sh`) just runs `worker_manager.py --worker_type <wm_type>`; it spawns `num_workers` worker subprocesses on the listed GPUs.

**Compute/time** (paper context): the WM is the bottleneck — a 30-step diffusion I2V rollout per candidate × `query_num` candidates × steps × episodes. With `TTS_ratio=1.0`, `query_num=2`, ~10 steps/episode that's ~20 WM rollouts/episode; a 30-step Wan I2V at 480×480, 14 frames is seconds each on H200 → minutes per episode, hours for a full split. **This is exactly why a linearized few-step WM (OUR-A) is the point.**

---

## STEP 3 — Fit assessment for OUR models (detail)

### OUR-A = Causal-Forcing (linearized, Wan2.1-1.3B AR, "TEXT-to-video, no action cond")
- **Image conditioning: SATISFIED.** Frame-wise Causal-Forcing supports I2V: `inference.py:134-200` encodes the first image to `initial_latent` and feeds it to `pipeline.inference(..., initial_latent=...)`; constraint `config.num_frame_per_block==1` (frame-wise model) and `batch_size==1` (`inference.py:184-185`). README confirms native T2V+I2V.
- **Action conditioning: SATISFIED via text** — World-in-World renders actions as text; OUR-A is text-conditioned. No native action input required for the `text`/`FTtext` type.
- **Speed: the whole point.** Linearized (KDA inter-frame + intra-frame softmax + AttnRes, per `wan_kda_surgery.py`) + few-step Causal-Forcing++ → the cheap WM that makes high `TTS_ratio` affordable. Best vehicle for **finding #3**.
- **Honest caveat:** zero-shot Wan2.1-1.3B following discrete navigation text is weak (paper shows zero-shot WMs lose to post-trained `FTwan`). For real success deltas, add a navigation LoRA mirroring `FTwan21` (the harness already supports LoRA: `wan_model.py:81-120`, `wan_model_diffsynth.py:67-74`). API-compatibility, however, exists *now*.
- **Verdict: PRIMARY FIT** as a `text` (zero-shot) or `FTtext` (LoRA) WM. Maps cleanly onto `wan_model_diffsynth.py`.

### OUR-B = iVideoGPT + KDA (native action-conditioned interactive WM)
- **Architecturally ideal** for the `action` WM type and for manipulation (obs+action→next obs autoregressive; `inference/predict.py:36-122`, `HeadModelWithAction`, `--action_conditioned --action_dim`).
- **Blockers:** (i) iVideoGPT has **no action-conditioned OXE checkpoint** (README) — you must fine-tune on the target sim's action space (LIBERO 7-DoF / RLBench 10-ch like `genex_manip`'s `action_input_channel=10`); (ii) the manip `action` WM mode is **disabled** in this release (`wiw_manip/planner/igenex_planner.py:129-130`) — only `zero_shot_text` is active for LIBERO; for RLBench the `genex_manip` action path (`FTsvd/eval_inference.py`) is the supported native-action template to mirror.
- **Verdict: SECONDARY FIT.** Use only for RLBench manip with a trained action-cond checkpoint; not the cheapest demo and partially blocked by the current release.

---

## STEP 4 — Integration plan (minimal) + cheapest finding-#3 demo

### Cheapest demo of finding #3
**IGNav with OUR-A as a `text`-type WM, sweeping `TTS_ratio`.** Rationale: IGNav needs **no detection server** (README step 5), metric is automatic geometric **success + SPL** (no GPT-4o/OpenAI key, unlike AEQA; no SAM2, unlike AR). Demonstrate: hold the WM fixed (OUR-A linearized), increase `TTS_ratio` (e.g. 0.25→0.5→1.0) and/or `query_num`, show success/SPL rising — and that the *same* compute budget is unaffordable with the 30-step Wan2.1 teacher. The clean controlled comparison is **OUR-A few-step vs Wan2.1 30-step on the same I2V backbone**.

### Minimal code to register OUR-A (4 edits, all under the `Host3` platform on this box)

1. **New wrapper** `downstream/api_models/ourA_cf_model.py` (skeleton — mirror `wan_model_diffsynth.py`):
```python
import sys, os.path as osp
from downstream.api_models import DiffuserModel, process_input_dict, process_output_dict, images_to_tensor
from downstream.utils.worker_manager import worker_main
from utils.logger import setup_logger
# + import Causal-Forcing pipeline/surgery from /home/wm-vla-rl/_ext/Causal-Forcing

class OurACausalForcing(DiffuserModel):
    def _load_pipe(self):
        # build Wan2.1-1.3B frame-wise pipeline, apply KDA surgery
        # (wan_kda_surgery.py), load --checkpoint_path (+ optional nav LoRA),
        # set config.num_frame_per_block == 1 (I2V requires frame-wise).
        ...
        return pipe
    def _load_pipe_args(self):
        return dict(num_inference_steps=self.args.num_inference_steps,
                    height=self.args.height, width=self.args.width,
                    num_frames=self.args.num_frames)
    def inference_batch(self, input_dict):
        b_action, save_dirs, return_objects, txt_list, img_list = \
            process_input_dict(input_dict, self.args.task_type, self.args.world_model_name)
        pipe_images = []
        for prompt, img in zip(txt_list, img_list):     # B==1 per call (I2V constraint)
            initial_latent = self.pipe.vae.encode_to_latent(<img→tensor>)
            frames = self.pipe.inference(text_prompts=[prompt],
                                         initial_latent=initial_latent, **self.pipe_args)
            pipe_images.append(frames)                  # list[PIL] per item
        video = images_to_tensor(pipe_images, uni_samp=self.args.num_output_frames,
                                 save_size=(self.args.width, self.args.height))
        return process_output_dict(b_action, save_dirs, return_objects, video)

if __name__ == "__main__":
    # argparse incl. --checkpoint_path --config_path --surgery_ckpt --num_inference_steps
    # --task_type --device --num_frames --num_output_frames ; set args.world_model_name="ourA"
    pipe_fd = int(sys.argv[-1]); args = parser.parse_args(sys.argv[1:-1])
    setup_logger(osp.join(args.log_dir, args.exp_id, "ourA_worker", f"worker{args.device}.log"))
    nav = OurACausalForcing(args=args)
    worker_main(pipe_fd, lambda d: nav.inference_batch(d))
```
2. **`downstream/vlm.py:27-33`** — add `"ourA"` to `WORLD_MODEL_TYPES["text"]` (or `["FTtext"]` if LoRA'd).
3. **`downstream/utils/worker_manager.py:752`** — add `"ourA"` to the `worker_type in [...]` list.
4. **`downstream/utils/workers_cfg.py`** — under `Host3`, add:
```python
"ourA_worker": [
    "/root/miniconda3/envs/<cf_env>/bin/python",
    "downstream/api_models/ourA_cf_model.py",
    "--config_path=/home/wm-vla-rl/_ext/Causal-Forcing/configs/causal_forcing_dmd_framewise.yaml",
    "--checkpoint_path=<cf_ckpt>",
    "--surgery_ckpt=<kda_surgery_ckpt>",   # the linearized weights
    "--num_inference_steps=2",             # few-step = the speed win
    "--num_frames=14", "--width=480", "--height=480",
],
```
Then launch: `init_worldmodel_manager.sh <exp> 1 ourA` and the IGNav solver with `--use_WM`.

### Blockers (ranked)
1. **Habitat-sim 0.2.5 (py3.9) + HM3D download** — required for IGNav/AR/AEQA; HM3D is multi-GB and gated (Matterport agreement). Largest setup cost.
2. **Causal-Forcing I2V wrapper details** — `pipeline.inference` I2V path is `batch_size==1`, frame-wise only; the harness sends batches, so loop per-item (as in `wan_model_diffsynth.py`). Confirm `vae.encode_to_latent` accepts the harness's resized PIL→tensor and output frame count ≈ 14.
3. **Zero-shot text-following quality** — for *good* success numbers OUR-A likely needs a navigation LoRA (treat as `FTtext`); otherwise the demo shows the *compute-scaling trend* but with modest absolute success.
4. **VLM policy** — default 72B-AWQ fits one H200; or use `Qwen2.5-VL-3B-AWQ` / `--use_heur` to drop the VLM and isolate the WM-speed effect (cleanest for finding #3).
5. **OUR-B only:** no action-cond iVideoGPT ckpt + manip `action` mode disabled (LIBERO) → train a ckpt and target RLBench `genex_manip` pattern. Significant.

---

## STEP 5 — Smoke test (ran; harness confirmed runnable)

Used `py3.10` (torch 2.11/cu130, torchvision 0.26, diffusers 0.38 present; installed only `imageio jaxtyping json_repair`). No model downloads, no full benchmark.

1. **WM-side harness imports** — `import downstream.api_models` OK (after the 3 light deps).
2. **Unified action API** (`process_b_action`) produces the exact prompts the WM receives:
   - nav `[[0,1,2,1,3]]` → `"Follow this sequence of camera motions: ['forward 0.2m', 'turn_left 22.5°', 'forward 0.2m', 'turn_right 22.5°']."`
   - manip `['pick up the red block']` → `"Follow the instruction to move the robotic arm: pick up the red block."`
   - freetext passthrough OK.
3. **I/O contract validators** — `check_inputdict({b_action, save_dirs, request_model_name})` and `check_outputdict({save_dirs, uint8 pred_frames})` both pass.
4. **Full worker protocol end-to-end** — wrote a stub WM (`/tmp/ourA_stub_model.py`) calling the real `worker_main(pipe_fd, task_fn)`, drove it with a framed `(client_id, task_id, payload)` over stdin and read back the framed result over the inherited pipe FD. Output:
   ```
   [worker] Processing task (client=0, task=0). Remaining queue size: 0
   [stub] prompts=["Follow this sequence of camera motions: ['turn_left 22.5°', 'forward 0.2m', 'turn_right 22.5°']."]
   RESULT: 0 0 {'save_dirs': ['/tmp/stubout'], 'pred_frames': (1, 14, 3, 64, 64)}
   WORKER EXIT CODE: 0
   ```
   This confirms the **entire plug-in mechanism** (length-prefixed pickle framing, action→text conversion, uint8 `[B,T,C,H,W]` return) works on this box — exactly the path OUR-A's wrapper will sit in.

**Not tested** (would need heavy downloads / py3.9 Habitat env): Habitat-sim, HM3D/MP3D, the real VLM policy, a real WM checkpoint, and a closed-loop episode. Those are the STEP-4 blockers, not harness bugs.

---

## Key file map (absolute paths)
- WM plug-in base + unified action API: `/home/wm-vla-rl/_ext/world-in-world/downstream/api_models/__init__.py`
- Worker protocol / manager / `worker_main` / registry L752: `/home/wm-vla-rl/_ext/world-in-world/downstream/utils/worker_manager.py`
- Worker command/env config (PLATFORM→Host3): `/home/wm-vla-rl/_ext/world-in-world/downstream/utils/workers_cfg.py`
- `wm_type` registry: `/home/wm-vla-rl/_ext/world-in-world/downstream/vlm.py` (L27-33)
- Closest wrapper template for OUR-A: `/home/wm-vla-rl/_ext/world-in-world/downstream/api_models/wan_model_diffsynth.py`
- Native-action WM templates (OUR-B): `/home/wm-vla-rl/_ext/world-in-world/downstream/api_models/nwm_model.py`, `/home/wm-vla-rl/_ext/world-in-world/FTsvd/eval_inference.py`
- Planning loop / TTS knob: `/home/wm-vla-rl/_ext/world-in-world/downstream/solver_AR.py` (L501-600), `/home/wm-vla-rl/_ext/world-in-world/downstream/solver_base.py` (L645-738)
- Metrics: `/home/wm-vla-rl/_ext/world-in-world/downstream/evaluator.py` (L18-50, 236-238)
- Docs: `/home/wm-vla-rl/_ext/world-in-world/docs/{01_setup_env,02_evaluation_datasets,03_run_commands,05_add_new_WM,09_WM_server_details}.md`
- OUR-A I2V proof: `/home/wm-vla-rl/_ext/Causal-Forcing/inference.py` (L134-200); KDA surgery: `/home/wm-vla-rl/_ext/wan_kda_surgery.py`
- OUR-B action-cond: `/home/wm-vla-rl/_ext/iVideoGPT/inference/predict.py` (L36-122)
