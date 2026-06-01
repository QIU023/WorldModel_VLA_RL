# WM × VLA-RL — World-Model Representations meet VLA Reinforcement Learning

复现并机制级理解机器人学习的两大前沿范式 —— **latent world model 视觉表征 (V-JEPA 2)** 与
**VLA 上的 on-policy RL (GRPO)** —— 并用 **knowledge distillation / KL-anchoring** 把两者串起来。

> 核心论点：*reverse-KL 在 RL 里抗 policy 漂移、在蒸馏里抗 mode-collapse，是同一套数学。*
> 本项目用一个把世界模型表征当 RL reward 的实验，把这条连接做实。

硬件：单节点 **2×H200 (sm_90, 各 ~140GB)**。周期：~4 周 MVP。

---

## Headline results (填充中)

| # | Module | 结果 | 图 |
|---|--------|------|----|
| 1 | V-JEPA2-AC 表征 | latent rollout + CEM planner 跑通；probe 显示 V-JEPA2 状态回归 MSE < SigLIP 基线 | `module1_vjepa2ac/results/` |
| 2 | VLA GRPO (2-GPU) | 8-GPU SimpleVLA-RL 移植到 2×H200；LIBERO-Spatial 成功率 ↑ | `module2_vla_rl/results/` |
| 3 | **latent-WM reward (HERO)** | 世界模型 latent reward vs 稀疏 binary reward：样本效率/稳定性对比 + reward-hacking 分析 | `module2_vla_rl/results/` |
| 4 | RL→蒸馏 | RL policy 自蒸馏前后 perf/stability/推理成本 | `module3_distill/results/` |
| infra | 副产品 | rollout KV-cache / 8→2 卡移植 / batched latent infer 加速比 | `infra_notes/` |

---

## 结构

```
module1_vjepa2ac/   V-JEPA2-AC: rollout / CEM planner / probe   (Week 1)
module2_vla_rl/     SimpleVLA-RL 2-GPU + binary & latent-WM reward + ablations  (Week 2-3)
module3_distill/    RL'd policy self-distillation (KL anchoring)  (Week 4)
infra_notes/        Module-4 机会主义副产品
docs/writeup.md     机制级讲解（面试时"讲到别人讲不出"的部分）
env/SETUP.md        三套隔离 python 的安装配方
configs/  scripts/  launch .sh（沿用 smoke + sentinel 习惯）
```

完整方案见 `docs/PLAN.md`（从 `~/.claude/plans/` 同步）。

## 执行状态

- [ ] **CP1** Week1 — V-JEPA2-AC rollout + CEM + probe（含 GO/NO-GO 闸门）
- [ ] **CP2** Week2 — SimpleVLA-RL 2-GPU 移植 + vanilla GRPO baseline
- [ ] **CP3** Week3 — latent-WM reward + 强制 ablations 🌟
- [ ] **CP4** Week4 — RL'd policy 蒸馏 + 打包

## 复用的已有资产（本机）

- SigLIP-224 probe 基线：`/home/.hf_home/models--google--siglip-base-patch16-224`
- rollout engine 可复用：`/home/venv/vllm`
- AttnRes 的 2-GPU GRPO / veRL mesh 改造经验：`/home/torchtitan_attention_residual`、`/home/seqkd_overnight`
