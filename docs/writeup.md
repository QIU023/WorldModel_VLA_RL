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

### 9b. sim-GRPO with learned reward — 核心负结果：hacking 在 reward 层不在 optimizer 层（2026-06-02）
把 CEM 换成 GRPO：小 Gaussian policy π(a|z) 在冻结 V-JEPA2 latent 上，真 LIBERO sim rollout，reward=progress head，group-normalized advantage + reverse-KL to ref（`exp_p2c_grpo_sim.py`，group8/H40/60步）。
| | R_mean (learned progress reward) | held-out success |
|---|---|---|
| 起始 (first-5 avg) | 0.471 | **0.00** |
| 峰值 (step15) | 0.864 | **0.00** |
| 末尾 (last-5 avg) | **0.833 (+77%)** | **0.00** |
- **GRPO 机制健康**：R_std 全程 0.04–0.13（组内有方差 → 有梯度，**没踩 zero-variance 坑**，正面兑现了 §2 的风险预案）；reward 被狠狠优化（+77%，~step15 饱和）。
- **但 480 次 rollout 零 success** → 教科书级 Goodhart：**policy 把"让 mean-pooled latent 读起来像 demo 后期帧"这件事学满，却不做任务**。
- **反直觉的关键结论**：我们原假设"GRPO 比 CEM 更抗单步 hacking"——**实测证伪**。GRPO 把这个有缺陷的 reward 优化得**比 CEM 更猛更彻底**，hacking 因此**更刺眼**。⇒ **瓶颈是 reward model，不是 optimizer**；换 optimizer 救不了可被钻空子的 reward。
- **根因**：progress head 只在 demo（in-distribution 成功轨迹）上训 → 没见过 policy 生成的失败态 → 外推时被白嫖；mean-pool 丢 grounding、goal-agnostic。
- **下一步（指向 reward model 自身）**：① **off-policy/对抗式负样本**——把 policy 自己 hack 出的状态当低-progress 负例回灌重训 head（reward 从自己的 hacking 中学，discriminator 式，最高杠杆）；② goal-conditioned head；③ ensemble + 不确定性惩罚（离分布检测）；④ 掺少量真 sparse success 把 reward grounding 住。
- **定位（降级为 §10 的动机，不当门面）**：这是个 naive **WM-as-reward** 玩具（自制 progress 标量、goal-agnostic、只见正样本）→ 必然被 hack。它的价值仅是**干净地暴露了"为什么必须用接地的、含失败态训练的标准 reward model + value"** → 直接导出 §10 的标准范式。图/日志 `results/p2c_grpo.{json,log}`。

## 10. M2 重定义 — imagination RL in latent WM（TD-MPC2/Dreamer 标准范式，2026-06-03）
**项目定调（user）**：除 efficient 长程 WM + 预训练 ckpt 结构微调（P1/P1.5，我们的差异化创新）外，**其余 WM-VLA-RL 一律对标算力允许下的大规模、标准 setup，不允许 toy / reward-hacking 式实验**。据此 M2 从"手搓 progress-head GRPO"重定义为 **paradigm (A) imagination RL in latent WM**：policy 在学习到的 latent 世界模型**想象**里训，标准方法 = **DreamerV3 / TD-MPC2**。
- **为什么是 (A) 而非 model-free（SimpleVLA-RL/RIPT-VLA）或 WM-as-reward**：SimpleVLA-RL/RIPT-VLA 是 model-free 真-sim RL、根本不含 WM（只能当兜底 baseline）；而 (A) 让 **P1 的 O(N) 高效 WM 直接变成使能器**——softmax WM 想象海量长轨迹 O(N²) 烧不起，我们 O(N) 能廉价想象 → "高效 WM → 海量想象 RL" 是比"WM 当 reward"紧得多的故事。
- **组件映射（标准 TD-MPC2/Dreamer ← 我们的资产）**：

| 标准组件 | 我们的实现 | 状态 |
|---|---|---|
| latent dynamics d(z,a)→z′ | V-JEPA2-AC + KDA/AttnRes predictor（预训练 + LIBERO 微调，**冻结/轻调** = RL-from-pretrained-WM / offline-MBRL） | ✅ P1 |
| planner（MPPI/CEM） | 现有 CEM = **无 value 的 TD-MPC** | ✅ P2 |
| **reward model R(z)** | 接地到**真 LIBERO 成功**、训练**含失败态**的 head（§10a） | ✅ M2.1 |
| value V(z)（TD） | TD 学习、真 reward bootstrap → 稀疏 success 稠密化 | 🔨 M2.2 |
| actor π(z) | 想象里 actor-critic（Dreamer）/ MPPI prior（TD-MPC2） | 🔨 M2.3 |

- **反 hacking = 标准机制**（非补丁）：reward 接地真 success + value 用真 reward bootstrap + 短想象 horizon + WM 不确定性/保守惩罚（offline-MBRL：MOPO/COMBO 思路）。
- benchmark = libero-spatial 单 suite 先通；held-out **success-rate**（真指标，非 reward）作 gate。

### 10a. M2.1 grounded reward model — 修掉 §9b 的 hacking 根因
标准 reward modeling（Dreamer 的 reward predictor）：r̂(z) 接地到**真实成功结局**，训练数据**含失败态**（随机策略 rollout = 负例，γ-return-to-success 给正例 demo）→ 不像 demo-only progress head 那样可被钻空子。`exp_m2a_grounded_reward.py`。
- **接地性（full：14.6k pos / 2.4k neg）**：held-out AUC(成功导向 vs 失败)=**1.000**，pos r̂ 0.680 / neg 0.002。**诚实 caveat**：AUC=1.0 是因为**随机负例太易分**（乱挥 vs 有目的操作）——这不是卖点。
- **真正信息在硬负例（unseen 被-hack 态）**：new grounded r̂ **0.258** vs old progress head **0.559**——接地 reward 对没训过的 hacked 态明显更不易被骗（老 head 虚高 2.2×），但 0.258 仍高于纯失败档 0.002 → **硬负例鲁棒性只是部分**。这恰好诚实地导出下两步：①把 hacked 态当**对抗负例**回灌重训（adversarial reward modeling）；②**value 学习**（TD bootstrap 真 success）才是最终接地"行为"的关键，reward head 单独不够。
- 这是 imagination-RL 的地基（reward → value → actor）。图/日志 `results/m2a_grounded_reward.{json,log}`。

### 10b. grounded-CEM 闭环 + imagination actor-critic scaffold（2026-06-03）
**(i) grounded reward 闭环（M2.1b，CEM）**：把 grounded r̂ 当 CEM objective 闭环跑（`exp_p2_cem REWARD=grounded`，40 步）。r̂ **0.004→0.001**（诚实地≈0）、energy 0.548→0.518 几乎不动、**no success 且 no hacking**。→ 坐实"grounded reward 诚实，但在 2 步 MPC 前瞻下太稀疏、无 planning 信号"。对比 §9b 的 progress head（被 hack 到 0.14+）：**grounding 用诚实换掉了 hacking，代价是稀疏**。

**(ii) imagination actor-critic（M2C，Dreamer-style scaffold）**：actor π(z)+critic V(z) 在 latent WM **想象 rollout**（可微穿过冻结 WM，math-SDPA）里一起学，reward=grounded r̂，λ-return。`exp_m2c_imagine_ac.py`。
- **机器跑通且稳定**。**关键 bug 抓到并修**：缺 target critic → **value 爆炸**（actor 不拉 reward 而 inflate critic：imag_return 0.22→2.07、r̂ 平 → **value 层的 hacking**）。标准修法 = **target critic（EMA、detached bootstrap）+ actor 梯度只走 reward→dynamics（value 项 detach）** → imag_return 稳定 0.1–0.24、critic_loss 平。
- **诚实信号（印证主线）**：H=4 短 horizon、随机 demo 起点 → actor **够不到高-r̂（近 success）态**，imag_reward 全程 ~0.05 不涨。
- **→ 主线闭环**：grounded reward 修 hacking ✓ → AC 机器稳定 ✓ → **短 horizon 想象够不到 success（reward 诚实稀疏）→ 需长 horizon → 长想象 SDPA O(N²) 烧不起 → 必须换 O(N) KDA WM（finale）**。scaffold 故意用 SDPA、到长 H 自然撞墙，给 KDA 换入制造干净动机。**efficiency 使能长想象、长想象使能能力**——即北极星。
- infra note：长-H 想象的 autograd 需 contiguous-mask + flash（现 math backend 为正确性，长 H 慢/吃显存）；KDA recurrent 想象是 O(1)/步，正是长想象的解。图/日志 `results/m2c_imagine_ac.{json,log}`。

### 10c. WM imagination reward-fidelity 诊断 —— 诚实的 inconclusive（2026-06-03）
动机：actor-in-loop 下 KDA 想象 imag_reward~0.5 vs SDPA~0.05（随机未训 actor），疑似 WM-exploitation。建 teacher-forced 诊断（真 demo action 开环 roll WM，比 r̂(预测) vs r̂(真) vs horizon），`exp_m2d_wm_reward_fidelity.py`。h16 |err|：SDPA 0.612 / KDA-full 0.119 / KDA⊕AttnRes 0.213。
- **此 run 有方法学 confound，不直接下结论**：① SDPA 用 attn_mask=zeros(1,1) hack → 破坏 block-causal 训练配置（train/inference mismatch）→ SDPA 长 rollout 退化（0.612 多半 artifact）；KDA 时间混合线性递归、因果内生、不受 hack 影响 → **SDPA 与 KDA 不可比**。② KDA-full(0.119) < AttnRes(0.213) 与 P1.5 latent-MSE rollout（AttnRes 0.370 < KDA-full 0.467）**相反** → reward-error 与 latent-MSE 是不同度量（reward 经 mean-pool+sigmoid 非线性），**reward-fidelity 是 metric-dependent 噪声代理**。
- **诚实结论**：「跨帧 AttnRes 是想象 enabler」假设**未证实**，不 spin。两个真东西保留：**(a) WM-exploitation 是真风险**（随机 actor 下 KDA 想象 reward 虚高）→ finale 必须靠标准 MBRL grounding（真 reward bootstrap value + 短 horizon + 真环境校正）压，不假设 WM 完美；**(b) KDA 因果内生、不依赖 attn_mask 处理 → rollout/想象里比 SDPA 鲁棒**（SDPA 的 zeros-mask hack 破坏其因果）——KDA 当想象 WM 的一个真实工程优势。
- → 不追 reward-fidelity，转 finale 正题：imagination RL（KDA WM + grounding）on libero_10 长任务。日志 `results/m2d_wm_reward_fidelity.json`。
