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

## 2026-06-02（P0 完成 + P1 centerpiece 完成 overnight）

### P0（foundation，合成验证）
- E0.1 `rollout.py` / E0.2 `cem_planner.py`：合成验证通过（CEM 找回 a*，能量↓5×）。
- E0.3 SDPA scaling：O(N²) 墙量化（T=512 75s/96GB，T=768 OOM）。
- E0.4 probe：待 LIBERO sim（deferred）。

### P1 — KDA-AttnRes predictor（CENTERPIECE，py3.10 fla 环境）
- 环境：复用 conda `py3.10`（torch2.11+fla0.5.0），`--no-deps` 补 timm/scipy/h5py/opencv。
- E1.1 `kda_acblock.py` `DividedKDAACBlock`（帧内 attn ⊕ 时间轴 KDA）：drop-in，shape==SDPA，+107M。
- E1.4 **headline**：KDA O(N) vs SDPA O(N²)。**T=512 KDA 2.9s/10GB vs SDPA 75s/96GB（26×快/9.6×省显存）；SDPA T=768 OOM，KDA 到 1024**。图 `results/e14_kda_vs_sdpa_scaling.png`。
- E1.5 recurrent rollout（`forward_step` + `fused_recurrent_kda`）：**O(1)/步**（77ms 平、3.4GB 平 @horizon512）。full-vs-recurrent ~2e-3/block（bf16 seqlen 级，部署用递归路径自洽）。
- **E1.2/E1.3 真数据微调 + parity（LIBERO libero_spatial，40 demos 编码成 latent 序列）**：
  - SDPA zero-shot（域外）0.820 → **SDPA fine-tuned（305M）0.102** / **KDA fine-tuned（107M adapter，冻 backbone）0.127**。
  - **KDA 在 SDPA 的 1.24× 以内、仅 1/3 训练参数 + O(N) 效率** = 近 parity + 大效率（divided 线性付 ~24% 质量换效率，符合预期）。
- 定位（SOTA 核查后）：主卖点=效率；能力探针=context 长度 ablation。生成派旗舰（Cosmos 3 16-64B / τ0-WM 5B）作 foil 强化效率论点，不当 live 组件。

### 下一步
- P1 收尾：context 长度 4→32→128 ablation（能力探针）；可选 KDA 容量/步数扫提升 parity。
- P2（WM-native 重构）：CEM/MPC 规划成功率 + latent-WM-reward RL（砍 OpenVLA-GRPO-leaderboard）。
- 脚本：`scripts/exp_e1*`、`kda_acblock.py`、`prep_libero_latents.py`、`exp_e12_finetune_real.py`。
