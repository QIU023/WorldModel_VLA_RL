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

### 10c. WM imagination reward-fidelity 诊断 + 一个真 mask bug 的修复（2026-06-03）
动机：actor-in-loop 下随机未训 actor 使 KDA 想象 imag_reward~0.5（疑 WM-exploitation）。建 teacher-forced 诊断（真 demo action 开环 roll WM，比 r̂(预测) vs r̂(真) vs horizon），`exp_m2d_wm_reward_fidelity.py`。
- **先修一个真 bug（rigor）**：rollout 一直把 predictor **自带的 block-causal mask 覆盖成 `attn_mask=zeros(1,1)`**（沿 exp_p2_cem）→ 破坏 SDPA 因果（train/inference mask mismatch）。改用按 rollout 长度建的正确 `build_action_block_causal_attention_mask`。**效果：SDPA h16 reward-error 0.612（artifact）→ 0.176**，坐实是 mask bug。（KDA 因果内生、不受影响。）
- **有效三方对比（h16 |err|，已修正 mask）**：SDPA **0.176** / KDA-full **0.119** / KDA⊕AttnRes **0.213**（h4 都 ~0.01–0.02）。
- **诚实结论（valid 数据，不 spin）**：① 原假设「跨帧 AttnRes 是想象 enabler/降 exploitation」**不成立**——这里 KDA-full reward-error 反而最低，且与 P1.5 latent-MSE（AttnRes 漂移更低）**不一致** → reward-error 是 mean-pool+sigmoid 粗代理、两度量测不同东西、**真判据是 success-rate**。② **但有干净正结果支撑 finale**：**高效 O(N) KDA WM 在 teacher-forced 真 action、长 horizon(h16) 的 imagination reward-fidelity 与 O(N²) SDPA 相当（甚至更好）→ 高效 WM 是有效的想象环境**（想象 rollout 跟踪真 reward 不逊 softmax），且不依赖"AttnRes 特殊"。③ WM-exploitation 是 **off-distribution（随机 actor）** 现象 → 靠标准 grounding 压。
- **finale WM 选型**：用 **KDA⊕AttnRes**（novel 结构 + P1.5 长程 latent 保真最好；reward-error 0.213 与他者同量级、在噪声内）。mask bug 修复回灌 exp_m2c/exp_p2_cem。日志 `results/m2d_fixed.log`。

## 11. Track A — P1/P1.5 大规模化：长序列 WM 微调（2026-06-03）
user 定调"P1/P1.5 也转大规模、不允许 toy"。把 efficient WM 从 200-demo/T=64/adapter 升级到**全参微调、长序列 T=128、libero_10(LIBERO-LONG, T~265)+spatial、effective batch 8**（`exp_e12` 加 FT_DATA 多集拼接 / FT_BS / FT_ACCUM / FT_ACT_CKPT / ckpt-resume）。
| WM @ T=128 全参微调 | held-out next-latent MSE | cost |
|---|---|---|
| **SDPA-full** | **OOM**（140GB H200 训不起） | O(N²) |
| **KDA⊕AttnRes（finale WM）** | **0.13748** | O(N) |
| KDA-full | 0.13886 | O(N) |
| _SDPA 零样本参考_ | _0.901_ | — |
- **效率墙在训练级（比 P1 forward-scaling 更进一步）**：长序列 T=128 下 **softmax WM 连微调都 OOM**（O(N²) 注意力 33k×33k），**O(N) KDA 变体训得动**且 held-out 0.137（6.5× 优于 SDPA 零样本）。
- **AttnRes ≥ full**（0.1375<0.1389，且 eval@50→100 gap 扩大 0.0007→0.0015）—— 与 P1.5"AttnRes 在更难/更长场景增益更大"一致。**finale WM = `kda_attnres_long`**。
- **infra 副产品（Module 4）**：① ckpt/resume（防长训 OOM 丢进度）；② **KDA fla 内核 T=128 micro-batch 上限 ~8**（bs16/48→Triton grid 溢出），用**梯度累积**拿 effective batch；③ act-ckpt 在重前向上 2× 重算→病态慢（12min/步），改 **bs1-无ckpt 本就填满 134GB VRAM + 快**。日志 `results/trackA_kda_{attnres,full}_long.log`、`trackA_sdpa_long.log`(OOM)。

### 11b. 公平 SDPA baseline（修正）+ "toy 微调测不了新架构"的方法论转向（2026-06-03）
**修正前述失误**：之前让 SDPA-full 在 T=128 OOM 就不比 = 逃避。**开 activation checkpointing（"降 config"），SDPA-full 在 T=128 训得动（19GB，之前没开 act-ckpt 才 137GB OOM）**。
- **公平 toy-scale（150步）三方质量**：SDPA-full **0.13079**（上限）/ KDA⊕AttnRes-v1 0.13748 / KDA-full 0.13886 → **线性化付 ~5-6% 质量代价**（诚实数）。**效率论点（Track A SDPA-OOM 是训练级、KDA 训得动）+ 质量论点（KDA 付 ~5%）现在都有数。**
- **AttnRes 涨点失败的根因（关键方法论）**：试了 v2-AttnRes（multi-head+value+可学温度），toy-scale 下 ≈ KDA-full（eval@100 0.1416），**没涨**。诊断：**AttnRes 是全新架构原语、zero-init 螺栓在"没有它也已预训练完备"的 backbone 上、只训 150 步** → 结构上长不出互补能力（新组件要在**预训练/大规模训练**里让网络学会用它、协同适配）。**当前微调能公平测"效率/适配"，但测不了"新架构的质量价值"。**
- **转向 → 大规模 SFT 公平架构对比**：T=96、**4000 步（27× toy）**、KDA-full vs KDA⊕AttnRes-v2 vs SDPA-ref **同 scale 充分训练** → 才能公平看 AttnRes 是否真涨点 + 是否逼近 SDPA 上限。下一级（若 SFT 仍不分）= **predictor 从随机初始化预训练**（带架构、大量 latent；305M 中等规模可负担；encoder 全 V-JEPA 预训练 2 卡不可行，诚实标注）。日志 `results/scale_kda_{full,attnres}.log`。

## 12. Finale — imagination RL on LIBERO-LONG：诚实的完整结论（2026-06-03）
把 finale WM(`kda_attnres_long`)+ libero_10 grounded reward 接进 imagination actor-critic（H=8）训 actor，再在**真 libero_10 sim** 测 success-rate（`exp_m2c`/`exp_m2e`）。
| 测法 | 任务 | 阶段 | success-rate |
|---|---|---|---|
| imagination-RL actor（from scratch） | libero_10 | 多阶段长 | **0.00** (0/9) |
| BC warm-start actor | libero_10 | 多阶段长 | **0.00** (0/12) |
| BC warm-start actor | libero_spatial | **单阶段** | **8.3%** (1/12，task2 33%) |
- **imag_reward 没起来**（0.17→0.20）+ from-scratch 0% → 加 **BC warm-start**（VLA-RL 标准从 BC 起步；`exp_m2f`，held-MSE 0.051）。BC 在 LIBERO-LONG 仍 0%，但**在更易的单阶段 spatial 拿到非零 8.3%**。
- **诚实根因 = policy class，不是训练法/WM**：actor 是**微型 reactive MLP + mean-pool latent、无语言条件、无记忆**。LIBERO-LONG 是多阶段（"把两个X都放进篮子"/"开炉灶 AND 放壶"），需要知道"在哪个子目标"（记忆/语言）+ 长程规划 + 精细操作——单帧 mean-pool latent → 同一个 MLP（跨任务共享、不知道在做哪个任务）做不到。spatial 单阶段非零 = **管线在 tractable 任务上可行**，印证是**任务难度/policy 容量**而非 WM 或 imagination 机制。
- **效率论点（项目真贡献）成立且未受影响**：Track A 证明长序列下 **softmax WM 连训都 OOM、O(N) KDA⊕AttnRes 训得动** → 高效 WM **能**当 softmax 烧不起的长程想象环境。imagination-RL 全管线（grounded reward→稳定 actor-critic→真 sim eval）端到端打通。
- **诚实 scope**：在 LIBERO-LONG 拿到 success **需要真 VLA policy**（OpenVLA-OFT 量级、语言条件、全 token），用我们的高效 WM 当想象/reward 环境——即 SimpleVLA-RL 量级的下一步大工程，超出当前 2-GPU/时限。**不伪称"解决了 LIBERO-LONG"**。日志 `results/finale_eval{,_bc,_bc_spatial}.log`。
