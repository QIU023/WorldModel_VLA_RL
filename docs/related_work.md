# Related Work — WM/JEPA × VLA × (long-horizon) RL（2026-06 核实版）

> 三路并行联网调研综合。每个 arXiv ID 均经 fetch 核实。标 ⚠️ 的为未从一手页面核实的数字/ID。
> arXiv YYMM:25xx=2025、26xx=2026。

## 0. WM 怎么用进 VLA —— 四种范式（关键分类）

| 范式 | 含义 | 代表作 | 用 RL 吗 |
|---|---|---|---|
| **(a) 测试时规划** | WM + CEM/MPC,在 latent 里搜动作序列、最小化到目标 latent 距离 | **V-JEPA2-AC**、**DINO-WM**、TD-MPC2 | ❌ 无 RL/reward |
| **(b) 想象 RL** | policy 在 WM 的 imagined rollout 里用 RL 训(Dreamer/GRPO) | **WMPO**、VLA-RFT、DreamerV3 | ✅ 核心 |
| **(c) WM 当 reward/verifier** | WM 给 dense reward 或验证动作,喂给 model-free RL | **VLA-RFT**、**WoVR**、IRL-VLA、AtomVLA | ✅ |
| **(d) WM 当辅助表征** | JEPA 预测未来 latent 仅作 BC 辅助 loss(无 rollout) | **VLA-JEPA**、**JEPA-VLA**、FLARE | ❌ 纯 BC |

**本项目(KDA+AttnRes V-JEPA latent WM)目标 = (b) 想象 RL,但用 JEPA-latent(非像素)。这正是空白(见 §5)。**

## 1. 想象 RL / WM-RL for VLA（问题核心）

| 论文 | arXiv | WM | VLA/policy | RL 方法 | reward | benchmark | 结果 |
|---|---|---|---|---|---|---|---|
| **WMPO** | 2511.09515 | **像素**视频 WM(刻意非 latent,对齐 VLA 预训练特征) | **OpenVLA-OFT** | **GRPO 全程在想象 rollout 里**(零真机交互) | sparse | MimicGen + 真机 | 真机 insert 53%→**70%**(>DPO 60%);涌现自纠正 |
| **VLA-RFT** | 2510.00406 | 动作条件视频 WM 当 simulator | VLA-Adapter | **GRPO** | **WM 验证 dense reward**(到目标帧 L1+LPIPS) | LIBERO 全 4 suite | LIBERO-Long 77.2→**80.2**;均值 86.6→91.1,<400 步 |
| **WoVR** | 2602.13977 | 动作条件视频 WM 当可靠 simulator | VLA | RL(WM-policy 协同进化、关键帧初始化抗幻觉) | sparse | LIBERO + 真机 | LIBERO 39.95→**69.2**;真机 61.7→**91.7** |
| **AtomVLA** | 2603.08519 | **预测 latent WM** 给子任务**打分**(非 rollout) | VLA+LLM 分解原子子任务 | 非 RL(latent 打分/verifier) | — | LIBERO 97.0% / LIBERO-PRO 48.0% | latent-WM 当 critic + 层级 |
| **iVideoGPT** | 2405.15223 (NeurIPS24) | 自回归 token 视频 WM(150万轨迹预训练) | 通用 MBRL(非 VLA) | model-based RL + 规划 | — | RoboNet/操作 | 可扩展 WM 基座 |

## 2. JEPA / latent-WM + VLA policy（与 task#12 最相关）

| 论文 | arXiv | 用法 | base | benchmark | 结果 | 备注 |
|---|---|---|---|---|---|---|
| **VLA-JEPA** | 2602.10098 | JEPA 未来 latent **仅作监督**(leakage-free,未来不进 input) | Qwen3-VL + SigLIP-2 | LIBERO/LIBERO-Plus/SimplerEnv/真机 | LIBERO ⚠️97.2%、**LIBERO-Plus 79.5%**(鲁棒性是卖点) | 最明确"JEPA 融进 VLA",但 **(d) 表征非想象** |
| **JEPA-VLA** | 2602.11832 | 把 **V-JEPA2 预测 embedding** 注入 VLA | (未公开细节) | LIBERO/Plus/RoboTwin2/真机 | ⚠️数字未核实 | 第二个独立 datapoint:V-JEPA2 emb 提升 VLA |
| **Intercepting the Future (AHEAD)** | 2606.02486 | latent 预测 WM 包住**冻结 OpenVLA-7B**(+4.9M) | OpenVLA-7B | 20 个**动态**场景+真机 | 79–97%(最强 baseline 31–58%) | 动态/移动物体,非长程 |
| **V-JEPA2-AC** | 2506.09985 (Meta) | 动作条件 JEPA latent WM + **CEM 规划** | 非 VLA | 真机 Franka 零样本 | 抓取 65%/放置 80%,16s/动作 | **(a) 规划,无 RL**;<62h Droid 视频训 predictor |
| **DINO-WM** | 2411.04983 (ICML25) | 冻结 DINOv2 + causal ViT predictor | 非 VLA | PushT/Maze/Rope 等 sim | +45% vs 旧 latent WM | **(a) 零样本规划** |

## 3. VLA RL 方法/框架（不依赖 WM 的主线,task#12 的 RL 选型）

| 论文 | arXiv/GitHub | base VLA | RL 算法 | reward | **LIBERO-Long SFT→RL** |
|---|---|---|---|---|---|
| **SimpleVLA-RL** | 2509.09674 / PRIME-RL | **OpenVLA-OFT** | **GRPO**(去 KL、改 clip) | sparse binary | **86.5 → 98.5**(π0=85.2)⭐最高 |
| **RIPT-VLA** | 2505.17016 | **OpenVLA-OFT** | RLOO+PPO(LOOP,动态弃 zero-var 组) | binary | **92.9 → 93.8**;1-demo 4%→97% |
| **VLA-RFT** | 2510.00406 | VLA-Adapter | GRPO | **WM 验证 reward** | 77.2 → 80.2 |
| **π_RL** | 2510.25889 | **π0 / π0.5**(也测 GR00T) | Flow-SDE/Flow-Noise(解流模型 log-lik) | sparse | LIBERO 均值 π0 57.6→97.6 / π0.5 77.1→98.3(Long 未单列) |
| **RLinf-VLA** | 2510.06710 | OpenVLA(-OFT)、π0.5 | 统一 PPO+GRPO | sparse | LIBERO-130 **98.11%**;框架/infra 参考 |
| **TGRPO** | 2506.08440 | OpenVLA-7B | Trajectory-wise GRPO(步级+轨迹级 advantage) | LLM dense | 10 LIBERO ~91% |
| **VLA-RL** | 2505.18719 | OpenVLA-7B | 轨迹级在线 RL(PPO 风) | **学习式 process reward** | 40 LIBERO +4.5% |
| **RLVLA** | 2505.19789 (NeurIPS25) | OpenVLA | **PPO vs GRPO vs DPO 对打** | sparse | **结论:PPO > GRPO/DPO** for VLA(重要反论) |
| **ConRFT** | 2502.05450 (RSS25) | consistency policy | offline(CalQL)+online consistency RL+人介入 | sparse+HIL | 8 真机任务 96.3%(无 LIBERO) |
| **iRe-VLA** | 2501.16664 | OpenVLA 风 | 在线 RL(冻 VLM 训 head)↔SFT 成功轨迹 交替 | sparse | 稳定大 VLA RL(与 task#12 设计相关) |

## 4. 长程专用方法 + LIBERO-Long 记分牌

**LIBERO-Long(libero_10,10 个多阶段长任务)成功率(一手核实):**
| 方法 | VLA | 机制 | LIBERO-Long |
|---|---|---|---|
| **SimpleVLA-RL** | OpenVLA-OFT | GRPO+binary | **98.5%** ⭐ RL 后最高 |
| **OpenVLA-OFT** | — | 并行解码+action chunking+连续动作+L1 | **94.5%**(SFT SOTA 参照;π0 85.2、OpenVLA 53.7) |
| **RIPT-VLA** | OpenVLA-OFT | RLOO+PPO | 93.8% |
| **MemoryVLA** | — | 双记忆(工作+情景) | 93.4% |
| **VLA-RFT** | VLA-Adapter | WM reward GRPO | 80.2% |

**长程能力的杠杆(文献共识):** ① 记忆/历史(MemoryVLA)② 预测未来 latent(FLARE/VLA-JEPA/AtomVLA)③ 层级/子目标分解(LiLo-VLA、AtomVLA)④ 语言 grounding(OpenVLA-OFT 缺语言时长程崩)⑤ action chunking+连续动作。

**⚠️ benchmark 饱和警告:** libero_10 已接近饱和+记忆化;LIBERO-Plus/PRO 鲁棒性测试下 95%+ 会崩(AtomVLA 97→48)。更诚实的长程信号是 **LiLo-VLA 在 Long++/Ultra-Long(21任务)69%**(2602.21531,物体中心层级)。

**长程相关(非 WM):** Long-VLA(2508.19958,CoRL25,phase-aware masking,纯模仿);LAPA(2410.11758,ICLR25,latent action 预训练);Moto(2412.04445)、GR-2(2410.06158)、LAWM(2509.18428)。

**WM 生成"梦"做 BC(非 RL):** DreamGen/GR00T-Dreams(2505.12705,NVIDIA,WAN2.1 视频 WM 生成 neural trajectory→BC,+LAPA/IDM 伪动作;新行为 0%→43%)。

## 5. ⭐ 空白 = 本项目白区(三 agent 一致结论)

**没有任何已发表工作在 JEPA/V-JEPA latent WM 的想象里用 RL 训 VLA policy。** 拼图都在,但没人做并集:
- V-JEPA2-AC = 动作条件 **latent WM**,但用 CEM **规划**、非 RL、非 VLA。
- WMPO = **想象 RL for VLA** 的配方,但**像素空间**(刻意不用 latent)。
- VLA-JEPA = JEPA 进 VLA,但**仅监督**、无 rollout、无 RL。

→ **{JEPA-latent O(N) WM} × {VLA policy} × {想象 RL} × {超长程}** 是 2026-06 的开放空白。本项目的 KDA+AttnRes 高效 latent WM 正落在此。**差异化叙事:WMPO 证明像素 WM 想象-RL 能涨 VLA;但像素 WM 在超长 horizon 上 O(N²)/烧不起——我们的 O(N) JEPA-latent WM 正是把它推到 LIBERO-Long/100 的使能器。**

## 6. task#12 建议选型(基于上表)
- **VLA:** **OpenVLA-OFT**(LIBERO-Long SFT 94.5% 最强参照、SimpleVLA-RL/RIPT/WMPO 都用它、ckpt 现成 `moojink/openvla-7b-oft-finetuned-libero-*`)。
- **RL:** 主线 **GRPO**(SimpleVLA-RL 配方,LIBERO-Long 98.5%,2 卡可移植);但记 **RLVLA 警告 PPO>GRPO/DPO**,可留 PPO 备选。
- **WM 接入:** 我们的 latent WM 当 **(b)想象环境** 或 **(c)dense reward/verifier**(VLA-RFT/WoVR 范式),让 GRPO 在我们的 O(N) 想象里 rollout——而非真 sim(超长 horizon 真 sim 太贵)。
- **诚实参照系:** 报 libero_10 时同时报 LIBERO-Plus/PRO 或 Long++(避免饱和/记忆化误导)。

### 核实 flags
- 一手核实:WMPO/VLA-RFT/WoVR/SimpleVLA-RL/RIPT-VLA/V-JEPA2-AC/VLA-JEPA/DINO-WM/OpenVLA-OFT 94.5%/MemoryVLA/LiLo-VLA/LAPA/DreamGen/AtomVLA。
- ⚠️未一手核实数字:JEPA-VLA 全部数字、VLA-JEPA 97.2%、π_RL 的 Long 单列、"Long-Lived Robots"(2602.10503)的 WM 角色。
- ⚠️ 搜索中出现但**未逐一核实、勿引用**:Causal-JEPA(2602.11389)、MIND-V、ThinkJEPA、Dream-MPC、MoLA、"DreamerV3-on-LIBERO"。
- 无公开 arXiv:**ProgVLA / Qwen-VLA** 未找到专门 RL 论文(可能是内部名);Genie 2 仅博客。
