# WM × VLA-RL × AttnRes — 完整计划 v2（2026-06-02，AttnRes 集成）

> v1（latent-WM + VLA-RL + 蒸馏，见 git 历史）的升级。引入 sibling 项目
> `/home/torchtitan_attention_residual` 的 **KDA 线性注意力 + Block-AttnRes** 作为隐空间
> 世界模型的**高效长时程时间 backbone**。依据：`docs/ATTNRES_INTEGRATION.md`。两张 H200 均可用。

## 核心重定性（相对 v1 的变化）
1. **Level-1 是蒸馏任务，不是换 kernel**：V-JEPA2-AC 的 305M predictor 用 full SDPA+3D RoPE 预训练，直接换 KDA 套旧权重=垃圾。要 parity 必须把 **SDPA predictor 蒸馏成 KDA-AttnRes predictor**（frozen encoder，teacher=SDPA 的 next-latent）。→ 架构层(AttnRes) × 目标层(reverse-KL 蒸馏) 合一。
2. **双 hero**：(B) 高效长时程 KDA 隐空间 WM；(A) 用它当 RL dense reward。
3. **AttnRes/KDA 只放世界模型（时序/recurrent），不放 OpenVLA policy**（parallel decode 收益有限）。

## 统一叙事
*高效(线性注意力 O(N))长时程隐空间世界模型，让 RL 规划 + on-policy 蒸馏在长机器人 episode(LIBERO 512 步)上可负担；reverse-KL 是其上的学习目标。* 架构 substrate + 学习目标 = 一个系统。

---

## Phases & 实验清单（"所有需要的实验"= 下列 E*）

### P0 — SDPA 基线 + 问题量化（foundation；teacher + baseline）
- **E0.1** `module1_vjepa2ac/rollout.py`：用真 `VisionTransformerPredictorAC.forward(x,actions,states)` 实现 block-causal 自回归 latent rollout。复用 notebook 的 `forward_target`/`forward_actions`/`loss_fn`（`notebooks/utils`）。franka_example_traj 验证。
- **E0.2** `cem_planner.py`：latent 能量 = `loss_fn(z_hat, target)`；CEM over actions（notebook 的 5³ action-grid 能量是参照）→ 能量曲面/收敛图。
- **E0.3** **长时程 scaling harness**：SDPA predictor 显存/延迟 vs 帧数 T（**合成输入即可，无需真数据**）→ 定位 O(N²) 崩点。量化 AttnRes 要解决的问题。
- **E0.4** `probe.py`：frozen V-JEPA2 vs SigLIP-224 状态回归（需 LIBERO，见 Infra）。GO/NO-GO 闸门。
- 交付：rollout 通 + 能量图 + SDPA scaling 曲线 + probe 图。

### P1 — KDA-AttnRes predictor（🌟 CENTERPIECE）
- **E1.1** KDA-attn 版 `ACBlock`（flag `use_linear_attn`）：**时间轴→KDA recurrence，空间 H/W→帧内 attn**（3D RoPE hybrid，doc 推荐先行方案）。复用 sibling `fla chunk_kda`（训练）/ `fused_sigmoid_gating_delta_rule_update`（step）。换点 `_ext/vjepa2/src/models/utils/modules.py:114 ACRoPEAttention`。
- **E1.2** **蒸馏**：teacher=SDPA predictor，student=KDA predictor，在 Droid/LIBERO latent 序列 match next-latent（MSE+feature，可加 reverse-KL 风格）。frozen encoder；student 的 predictor_embed/proj/action·state encoder 可 init from teacher。
- **E1.3** **parity gate**：KDA next-latent MSE ≈ SDPA（同 clips/probe）。**不过则不进 E1.4**。
- **E1.4** **长时程 scaling**：KDA recurrent-state rollout 显存平/延迟线性，跑 SDPA 跑不动的 horizon（headline 图）。
- **E1.5** recurrent-state rollout + CEM state-forking（CEM 单序列用简化 per-layer state dict deepcopy/restore，思路源自 PR#15 fork/copy_from）。
- 交付：parity 表 + KDA-vs-SDPA scaling 对比图（headline）+ rollout latency。

### P2 — RL（2-GPU GRPO on LIBERO）
- **E2.1** LIBERO 装环境 + OpenVLA-OFT（ckpt 已下 `_ext/ckpts/`）+ vanilla GRPO baseline（binary reward）→ 兜底 banked。
- **E2.2** latent-WM dense reward（用 P1 的高效 WM）vs binary 对比；**ablation**：KL 系数 / rollout budget / shaping 开关；reward-hacking 定位。
- 复用 sibling `run_grpo_llava_kimi_2gpu.py` 的 8→2 layout（trainer FSDP GPU0 + sglang generator GPU1）+ `save_dcp`。
- 交付：成功率曲线 + dense-vs-binary 对比 + ablation。

### P3 — RL→蒸馏 + 打包
- **E3.1** RL policy 自蒸馏（reverse-KL）→ 前后 perf/stability/sample-eff/推理成本。
- **E3.2** README headline 表 + writeup + figures + push。

### Infra（横切，Module-4 副产品）
- KDA recurrent-state rollout cache（P1，接 PR#15 fork/restore）。
- 2-GPU GRPO 迁移笔记（P2，复用 sibling）。
- 可选 upstream PR：vjepa2 `VJEPA_BASE_URL` localhost fix（trivial）；KDA-on-WM 长时程 backbone。

---

## 复用源（sibling `/home/torchtitan_attention_residual`）
| 需要 | 复用 |
|---|---|
| KDA 线性注意力 kernel | `sglang/.../attention/linear/kernels/kda_triton.py`（`chunk_kda` 训练 / `fused_sigmoid_gating_delta_rule_update` step）；训练模型 `torchtitan/.../experiments/attention_residual/` |
| Block-AttnRes merge（纯 pytorch，可直接抄） | 训练 `torchtitan/.../attention_residual/attn_res.py:48 block_attn_res(blocks,partial,proj,norm)`；推理 phase1/2 `sglang/.../attn_res.py` |
| recurrent-state fork/restore | `mamba_radix_cache.py` `fork_from`/`copy_from`（CEM 单序列简化为 per-layer state deepcopy） |
| 2-GPU GRPO | `phase11_rlhf_grpo_infra/rlhf/run_grpo_llava_kimi_2gpu.py` + `PolicyTrainer.save_dcp` |
| env pin | 两侧 `torch==2.11.0` + fla（Kimi fork）；FLA_CHUNK_SIZE=64 |

## 风险（诚实）
1. **parity（最大）**：线性注意力是近似 + 蒸馏能否追平 SDPA next-latent。E1.3 设硬 gate；过不了则退回"长时程效率 + 可接受质量损失"叙事。
2. **3D RoPE × 线性注意力**：hybrid(time→KDA recurrence, space H/W→帧内 attn) 先行；不行再试 1D 融合位置。
3. **env**：KDA 需 fla + torch2.11；vjepa2 venv 是 torch2.12 → P1 可能要单独 env（conda py3.10 有 fla+torch2.11）或在 vjepa2 venv 装 fla（torch2.12 兼容性待验）。
4. **数据**：长时程真数据需 LIBERO/Droid；E0.3/E1.4 scaling 先用合成输入解耦。
5. **action/state token 交织**（`ac_predictor.py` 每帧 [a,s,patches]）必须在 recurrent 形式里保留。

## 执行顺序
P0(E0.1→0.3 先；E0.4 待 LIBERO) → **P1 centerpiece** → P2 → P3。Infra 随 P1/P2 收割。两卡：长跑（蒸馏/GRPO）占一卡，scaling/eval 占另一卡。
