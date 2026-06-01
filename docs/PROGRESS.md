# 进度日志

## 2026-06-01（Day 0：搭架子 + Module-1 环境打通）

### 环境
- 2×H200 (sm_90, 各 ~140GB)，状态全落 `/home`（持久）。
- 约定：共享机器时按需 `CUDA_VISIBLE_DEVICES=1` 只用 GPU1，GPU0 留给其他项目。
- `vjepa2` 独立 venv：`/home/venv/vjepa2`（torch 2.12.0+cu130 + timm/einops），不污染 `vllm`/`py3.10`。

### 已完成
- ✅ 仓库骨架 + push 到 `github.com/QIU023/WorldModel_VLA_RL`（main）。
- ✅ V-JEPA2-AC ckpt（`vjepa2-ac-vitg.pt`, 11.76GB）下载到 hub 缓存 `/home/.cache/torch/hub/checkpoints/`。
- ✅ **Module-1 smoke 通过**（GPU1）：encoder **1012.2M**（ViT-giant ~1B）/ predictor **305.2M**（AC ~300M）/ encoder 前向输出 **(1, 8192, 1408)** / dt 0.93s / 峰值 6.04GB。脚本 `scripts/smoke_vjepa2ac.py` + `scripts/run_week1_smoke.sh`，sentinel `module1_vjepa2ac/results/SMOKE_VJEPA2AC_DONE`。
- ✅ OpenVLA-OFT ckpt（`moojink/openvla-7b-oft-finetuned-libero-spatial`）下载到 `_ext/ckpts/`（Week2 用）。
- ✅ SimpleVLA-RL clone 到 `_ext/`（仅用于源码核查）。

### 关键发现（写进 writeup）
1. **vjepa2 仓库坑**：`src/hub/backbones.py` 的 `VJEPA_BASE_URL` 被写死成 `http://localhost:8300`（"for testing"，公开 URL 被注释）。绕过：把公开 ckpt 预下到 hub 缓存，`load_state_dict_from_url` 按文件名命中、不碰 localhost。公开 URL = `https://dl.fbaipublicfiles.com/vjepa2/vjepa2-ac-vitg.pt`。→ 可顺手提个 trivial PR。
2. **SimpleVLA-RL 引擎真相**：VLA rollout（`verl/workers/rollout/rob_rollout.py`）是 **HF transformers + FSDP 裸前向**跑 OpenVLA-OFT，**不经 vLLM/sglang**。仓库里的 vLLM 全在 `verl/third_party/vllm/`（veRL 给通用 LLM 用的老版 0.3.1/0.4.2），VLA 路没调；全树**无 sglang**。→ 不被 vLLM 锁死；接 sglang 优化 rollout 是 infra-PR 机会。
3. **τ0-WM 定位**：生成式视频扩散世界模型（Wan2.2-5B），与 V-JEPA 隐空间路线对立。**只作叙事对照（latent vs 生成式）+ reward-as-critic 设计参照**；不复现、不当 live reward 引擎（5B 扩散采样算力上不可行，且违背"廉价 dense latent reward"主线）。

### 下一步（暂停中——机器交还其他项目）
- Phase A（GPU1）：`rollout.py`（block-causal latent rollout + KV-cache 副产品）→ `cem_planner.py`（CEM，能量曲线图）。
- Phase B（CPU/网络）：LIBERO 装环境（robosuite+MuJoCo，headless EGL）+ smoke；`vla-rl` env（老 veRL，timebox）。
- Phase C（GPU1，依赖 B）：`probe.py`（V-JEPA2 vs SigLIP-224 状态回归，GO/NO-GO 闸门）。
