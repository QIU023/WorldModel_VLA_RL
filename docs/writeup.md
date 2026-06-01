# 机制级讲解 (writeup)

> 面试时"讲到别人讲不出"的部分。边做边填，每个结论配一张图/一个数。

## 1. V-JEPA 2-AC 世界模型
- block-causal attention 怎么 autoregressively 预测下一帧 latent？（mask 结构、action/EE 条件注入）
- latent rollout 的 KV-cache：为什么每步只需算新时刻 K/V（接 AttnRes cross-stage cache 的同构）。
- probe 结论：V-JEPA2 latent 比普通 video encoder 多"懂"了哪些物理量（task-relevant state vs task-irrelevant 扰动）。

## 2. on-policy GRPO on VLA
- advantage = group-normalized reward（无 critic）；group sampling 怎么造出 within-group 方差。
- KL-to-ref（reverse KL）抗 policy 漂移 —— **与蒸馏里 reverse-KL 抗 mode-collapse 是同一套数学**（核心连接点）。
- 为什么 on-policy 在 VLA 比 off-policy 稳（分布匹配、reward 稀疏且噪声）。
- **zero-variance 陷阱**（AttnRes 实测教训）：binary reward 整组全对/全错 → advantage≈0 → 无梯度。dense latent-WM reward 如何缓解。

## 3. latent-WM reward (HERO)
- reward 设计：latent goal-distance（CEM 能量改作 reward） vs world-model imagination reward。
- reward hacking 出现在哪：policy 是否压低 latent 距离但任务没成功。
- ablation：KL 系数、rollout budget、shaping 开关下 policy 怎么崩。

## 4. RL → 蒸馏
- 把 RL 探索出的收益蒸回更稳/更小 policy 保留/丢失了什么。
- forward-KL（覆盖全 support）vs reverse-KL（mode-seeking）在 policy 蒸馏里的取舍。
