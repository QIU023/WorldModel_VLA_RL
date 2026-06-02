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

## 5. 框架现状与 infra 落点（2026-06-01 核查）
- **SimpleVLA-RL 的 VLA rollout 不走推理引擎**：`rob_rollout.py` 用 HF transformers + FSDP 裸前向跑 OpenVLA-OFT、与 LIBERO 交互；仓库里的 vLLM 是 veRL 给通用 LLM 文本生成用的 vendored 老版（0.3.1/0.4.2），VLA 路没调；全树无 sglang。
- **infra-PR 机会**：rollout 前向未优化 → 把 sglang 接成 OpenVLA-OFT 优化后端（接 AttnRes sglang overlay 经验）。前提：OpenVLA-**OFT** 是 parallel decoding（一次前向），引擎收益有限；原版自回归 OpenVLA 才是 sglang 显著加速场景。Week2 profile rollout 瓶颈后再定。
- veRL mainline 支持 `rollout.name=sglang`，但 SimpleVLA-RL 是老 fork 没带。

## 6. latent vs 生成式世界模型（项目定位轴）
- 两条对立路线：**V-JEPA2-AC = 隐空间预测**（轻、CEM ~16s/action、能量=latent 距离）；**τ0-WM = 生成式视频扩散**（Wan2.2-5B，显式想象未来帧给动作打分）。
- 本项目选隐空间做 RL reward 的理由（面试弹药）：成本、采样吞吐、reward-hacking 面更小。
- τ0-WM 的"想象→评估(打分)→执行"是 world-model-as-critic，与本项目 M2 latent-WM reward 同家族——借其思路、用 V-JEPA 隐空间廉价实现。

## 7. P1 实测结论（2026-06-02 overnight）
- **效率（主卖点，成立）**：KDA divided predictor O(N) vs SDPA O(N²)。T=512: 2.9s/10GB vs 75s/96GB（26×快/9.6×省显存）；SDPA T=768 OOM，KDA 跑到 1024。recurrent rollout O(1)/步。
- **parity（近 parity）**：LIBERO 真数据微调，KDA-ft(107M adapter) next-latent MSE 0.127 在 SDPA-ft(305M) 0.102 的 **1.24×** 内（1/3 训练参数）。divided 线性付 ~24% 质量换大效率，符合预期。
- **能力探针（诚实负结果）**：context 长度 16/32/64/96 → MSE 0.155/0.144/0.153/0.149 **基本持平**，长上下文在 LIBERO 上无清晰收益——**操作类任务近 Markov，长历史不太帮**。⇒ 不要硬卖"长历史必然更好"；效率赢面才是 P1 的硬通货。真正能展现长上下文价值要找**长程记忆依赖/部分可观测**的任务（future work）。
- **定位**：生成派旗舰（Cosmos 3 16-64B / τ0-WM 5B）作 foil 强化"小而高效 latent WM"的效率论点；不当 live 组件。

## 8. 公平 parity 修正 + 跨帧 Block-AttnRes 消融（2026-06-02 P1.5）
之前"KDA 比 SDPA 差 25%"是 confound（KDA 只训 107M adapter vs SDPA 305M 全参）。**公平对比（同 LIBERO 200-demo split / 160 训 / 1500 步 / 全参微调）**：
| 模型 | held-out next-latent MSE | vs SDPA | 代价 |
|---|---|---|---|
| SDPA-full | **0.1007** | — | O(N²) |
| **KDA⊕AttnRes（novel）** | **0.1031** | **+2.4%** | O(N) |
| KDA-full | 0.1050 | +4.3% | O(N) |
- **能力差距实为 ~4%（非 25%）**：公平训练下 divided-KDA 几乎不牺牲能力，O(N) 效率近乎白送。
- **跨帧 Block-AttnRes（novel）把差距砍半（4.3%→2.4%）且保持 O(N)** —— 验证了"AttnRes 作为 WM 时间聚合原语"这个新结构**确实有效且补能力**。
- 这是 P1 的核心 ablation：纯 KDA / KDA⊕AttnRes / SDPA 三方，讲清线性化的代价、novel 结构的增益、与 softmax 上限的距离。图 `results/e12_parity_3way.png`。
- gap 仅 2.4% → 大概率无需再加 full-SDPA 层（hybrid ②）；若要彻底闭合是后续可选项。

### 8b. novel 结构的长时程增益（最强证据）
AttnRes 是 O(N)（cumsum 形，scaling 实测 T→1024、~10% over KDA）。更关键——**开环 rollout 保真度（LIBERO held-out n=40），跨帧 AttnRes 的优势随 horizon 增大**：
| horizon | KDA-full | KDA⊕AttnRes | 改善 |
|---|---|---|---|
| h1 | 0.059 | 0.060 | ~平 |
| h4 | 0.132 | 0.125 | -5% |
| h8 | 0.237 | 0.213 | -10% |
| **h16** | **0.467** | **0.370** | **-21%** |
→ 短程持平、**长程明显更好**（h16 漂移 -21%）。这正是 cross-frame 聚合该赢的地方（比纯递归更能压长程开环漂移），且仍 O(N)。**比 next-latent@T=64 的 2.4% 更有说服力的 novel-结构卖点**。图 `results/e1z_rollout_compare.png`。

## 9. (B) learned latent-WM reward — 诊断确认 + 下一层（2026-06-02）
progress head（mean-pool latent→MLP→t/T；held progress-MSE 0.0047、monotonicity 0.60）替手工 L1 当 CEM reward：
- progress 前 20 步真涨 2×（0.073→0.141）—— **CEM 确实在跟 learned 信号优化**，对比 L1 的持平 ⇒ **坐实"手工 L1 太弱"的诊断**。
- 但 t20 后回落、未 success。下一层问题：head 仅 60% 单调（噪声）；model-based reward-hacking（CEM 钻 WM 想象的空子）；head goal-agnostic。
- 下一步：goal-conditioned + 更单调 head / 抗 hacking（ensemble、短 horizon 重规划）/ 换 CEM→GRPO 用 progress reward 训 policy。
