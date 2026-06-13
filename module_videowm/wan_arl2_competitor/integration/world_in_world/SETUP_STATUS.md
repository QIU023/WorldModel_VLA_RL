# World-in-World IGNav — Setup Status (data + sim ready)

**Date:** 2026-06-13
**Box:** 2x H200, NVIDIA driver 580.126.20
**Repo:** `/home/wm-vla-rl/_ext/world-in-world`
**Status:** ✅ **READY** (data + sim verified end-to-end). No gating or disk blockers hit.
Closed-loop eval not run (out of scope); remaining steps to a full run listed below.

---

## TL;DR

- The Habitat-sim env (`habitat025`) is installed and **IGNav verified headless**: an HM3D val
  scene loads via EGL and an IGNav episode instantiates (agent placed, RGB rendered, geodesic
  path found). See "Verification" below.
- HM3D **val** split (the split IGNav needs) + the IGNav episode dataset are downloaded and in place.
- **HM3D was NOT gated for us:** the repo ships the val split as an *ungated Google Drive bundle*
  (not an HF gated dataset), so the HF-gating concern did not apply. The HF token was only used for
  the (public) episode files. **The HF token still works (user `yq023`, role `read`) — please rotate it.**
- `/home` disk never dropped below ~7.5 G; final free **15 G**. The heavy Python stack (torch + CUDA +
  habitat-lab) installed onto the **root overlay fs** (`/`, 815 G free), NOT `/home`.

---

## 1. What IGNav needs (exact) + sizes

IGNav (Habitat-sim) requires the **HM3D val** scene split + a custom episode dataset. Determined from:
- `docs/02_evaluation_datasets.md` (download instructions)
- `downstream/downstream_datasets.py:154` `IGDataset(hm3d_root="data/scene_datasets/hm3d/val")`
- `downstream/simulator.py:200-214` `get_simulator()` →
  `scene_dataset_config_file = data/scene_datasets/hm3d/hm3d_annotated_basis.scene_dataset_config.json`
- `habitat_data/config_utils.py` `hm3d_config(stage='val')`

| Item | Source | Size (download) | Size (on disk) | Path |
|------|--------|-----------------|----------------|------|
| HM3D **val** scenes (100 scenes, all assets + configs) | Google Drive bundle `hm3d_val_with_configs.tar.gz` (id `1UsxvJpp7A4Byrd6E3Cjx6EoHcAph_rz5`) | **7.84 GB** tar.gz (deleted after extract) | **9.3 GB** | `data/scene_datasets/hm3d/val/` |
| IGNav episodes | HF `zonszer/WIW_datasets` (public) | 18 KB | 82 KB | `data/WIW_datasets/eval_datasets/IGNav/episodes_IGNav.json.gz` |
| IGNav goal images | HF `zonszer/WIW_datasets` (public) | **300 MB** | 287 MB (zip) | `data/WIW_datasets/eval_datasets/IGNav/igdataset_goal_imgs.zip` |

**Episode/scene coverage:** `episodes_IGNav.json.gz` = **144 episodes** referencing **78 unique HM3D val
scenes** (scene IDs in the 00803–00899 range = the HM3D val split; minival would NOT cover these — the
full val split is genuinely required). Each episode has start/goal pose, `goal_radius=0.5`,
`scene_relpath` (e.g. `00824-Dd4bFSTQ8gi/Dd4bFSTQ8gi.basis.glb`), and a matching goal image.
The goal-images zip is auto-unzipped to `goal_imgs_unzipped/` at runtime (no manual unzip needed).

**Note on HM3D distribution / gating:**
- The official HM3D is gated via the **Matterport data portal** (Matterport agreement) — that route
  is *not* a standard HF dataset (no `ai-habitat/HM3D` on HF; all guessed names 404 with the token).
  So an HF token alone would **not** fetch official HM3D.
- The repo sidesteps this with the ungated **Google Drive bundle** above. That is the route used here,
  so **no gating blocker was encountered**. The bundle's non-annotated config
  `hm3d_basis.scene_dataset_config.json` is a **broken symlink** (points to a `/weka/...` path), but
  IGNav uses `hm3d_annotated_basis.scene_dataset_config.json` (a real file), so this is harmless.

---

## 2. Disk

- Before: `/home` ≈ 25 G free (500 G vol, 96 % used). Constraint: keep ≥ 4 G, ≥ 5 G headroom for downloads.
- Strategy that avoided filling `/home`:
  - HM3D downloaded to `/home` then **tar deleted immediately after extract** (transient peak ~7.5 G free, monitored by a disk-guard that would `pkill` at < 4 G — never triggered).
  - The conda env + torch/CUDA/habitat-lab wheels live on `/root/miniconda3` = **root overlay `/`** (815 G free), not `/home`. pip cache + TMPDIR were redirected to `/root` so `/home/.cache/pip` stayed empty.
- **After: `/home` = 15 G free; `/` = 815 G free.** `/home/wan_distill_data_mp4` and `/home/wan_*data` untouched.

---

## 3. Environment

Conda env **`habitat025`** at `/root/miniconda3/envs/habitat025` (Python 3.9.23). Other envs
(`/home/venv/cosmos_predict2`, `/home/venv/libero`, conda `py3.10`) were not touched.

Install commands (exact, as run):
```bash
# conda channel ToS (required on this box before any conda create)
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# habitat-sim 0.2.5 (headless/EGL + bullet)
conda create -n habitat025 python=3.9 cmake=3.14 -y
conda install -n habitat025 -y -c conda-forge -c aihabitat habitat-sim=0.2.5 headless withbullet

# Missing system EGL/GLVND libs (habitat-sim import failed without these). Installed via apt (land on / not /home):
apt-get install -y libegl1 libgles2 libgbm1 libopengl0 libglx0 libglvnd0
# (box had libEGL_nvidia.so.0 but not the glvnd dispatch libEGL.so.1 / libOpenGL.so.0)

# torch + python deps (PIP_CACHE_DIR/TMPDIR redirected to /root to spare /home)
pip install "numpy>=1.20,<1.24" "torch==2.5.1" "torchvision==0.20.1" "torchaudio==2.5.1" \
  --extra-index-url https://download.pytorch.org/whl/cu121
pip install --upgrade "imageio[ffmpeg]" "exceptiongroup>=1.2" "pydantic>=2.7,<3" pandas==2.2.3 \
  jaxtyping==0.2.36 tiktoken==0.9.0 anthropic==0.49.0 json-repair tabulate==0.9.0 tenacity==9.0.0 \
  pyequilib==0.5.8 open3d==0.18.0 openai gdown

# habitat-lab + habitat-baselines at the repo's pinned commit
pip install -e "git+https://github.com/facebookresearch/habitat-lab.git@094d6be2f9d057e4781a68ae792132895fd4d3d0#egg=habitat_lab&subdirectory=habitat-lab" \
            -e "git+https://github.com/facebookresearch/habitat-lab.git@094d6be2f9d057e4781a68ae792132895fd4d3d0#egg=habitat_baselines&subdirectory=habitat-baselines"

# Pin numpy back down (habitat_baselines pulled numpy 1.26.4, breaks habitat-sim 0.2.5)
pip install "numpy==1.23.5"
```

Installed versions (verified importable together):
- habitat-sim **0.2.5**, habitat-lab **0.3.3** (commit `094d6be2`), habitat_baselines 0.3.3
- torch **2.5.1+cu121** (`torch.cuda.is_available() == True`), numpy **1.23.5**, python 3.9.23

**Known dependency conflict (benign for IGNav):** habitat-lab 0.3.3 / moviepy declare `numpy>=1.25`,
but habitat-sim 0.2.5 (compiled) needs `numpy<1.24`. We kept **numpy 1.23.5** because habitat-sim is
the component that renders HM3D; all four packages still import and IGNav verifies. moviepy (video
export only) may warn but is not on the IGNav core path. If a downstream util breaks on numpy<1.25,
isolate moviepy or run video export elsewhere.

src checkouts (on `/home`, 366 M): `src/habitat-lab/` (the pip `-e` git clone).

---

## 4. Verification (no full eval)

Script `/home/wm-vla-rl/_ext/world-in-world/verify_ignav.py` (uses `habitat_sim` directly, headless
EGL — no habitat-lab/torch needed) loads the first available scene + episode:
```
scene_dataset_config: .../hm3d/hm3d_annotated_basis.scene_dataset_config.json
episode datum_id: Dd4bFSTQ8gi_13  scene: 00824-Dd4bFSTQ8gi/Dd4bFSTQ8gi.basis.glb (23 MB)
Simulator created OK. pathfinder loaded: True
rgb obs shape: (256, 256, 4) dtype: uint8
geodesic path found: True  geodesic_distance: 3.17  (episode step_to_goal=21)
VERIFY_OK
```
Re-run: `MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet /root/miniconda3/envs/habitat025/bin/python verify_ignav.py`

---

## 5. Remaining steps to a full IGNav closed-loop run

Data + sim are done. To run the benchmark (per `docs/03_run_commands.md`):

1. **VLM policy server** (or skip): create the `vllm` env (`vllm==0.7.3`, py3.9) and launch
   `downstream/scripts/vllm_dsai.sh 8000` with Qwen2.5-VL-72B-AWQ (fits one H200) or the 3B-AWQ;
   **or** pass `--use_heur` to the solver to drop the VLM (cleanest for isolating the WM-speed effect).
   *Not yet created.*
2. **WM server** (our model): register OUR-A Causal-Forcing as a `text`-type WM. A prototype wrapper
   and the 4 registration edits already exist in this dir: `causalforcing_model.py`,
   `REGISTRATION_EDITS.txt`, `SMOKE_RESULT.md`. Needs the Causal-Forcing conda/venv env + the IGNav
   nav-text rendering. Launch via `downstream/scripts/init_worldmodel_manager.sh <exp> 1 <wm_type>`.
3. **Solver:** `CUDA_VISIBLE_DEVICES=0 bash downstream/scripts/init_solvers.sh downstream.solver_IGNav <exp> 4 "localhost:8000" localhost:7000 --use_WM` (from repo root, in `habitat025`).
4. **Aggregate:** `PYTHONPATH=. python downstream/evaluator.py <exp> --task IGNav` (metric = success rate + SPL).
5. **Optional / only if hit at runtime:**
   - habitat-baselines cubemap fix (`docs/01_setup_env.md` note: `get_cubemap_projections` `img_h/img_w` 256→512). Likely only needed for the panoramic WM path; IGNav `text` WM uses `look_ahead_explore_non_pano`.
   - `subtrees/open-eqa` `pip install -e .` (docs list it for the Habitat env; AEQA needs it, IGNav may import through the evaluator chain — install if an `open_eqa`/`openeqa` ImportError appears).
   - IGNav also reads `downstream/process_IGnav_dataset/inputs.json` (present in repo) via `IGDatasetPortable`.

---

## 6. Blockers

**None for data + sim.** All of: HM3D val download, episode download, habitat-sim/lab install, and a
headless scene+episode instantiation succeeded.

Operational notes:
- **Rotate the HF token** `hf_xje…cXvm` (used here for the public episode files; still valid, role `read`). We cannot rotate it from this box — please rotate it in the HF account settings.
- `/home` is at 15 G free — fine for IGNav, but the VLM/WM model weights (Qwen-VL-AWQ ~40 G, or our CF/Wan ckpts) should be cached on `/` (815 G free), not `/home`.
