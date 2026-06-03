# task#12 实施方案 — 真 VLA × O(N) 隐空间 WM × 超长程(Ultra-Long)

> 状态:2026-06-03 立项。承接 P1/P1.5(架构定论)+ related_work.md(空白确认)。
> 战场从饱和的 libero_10 换到 **LiLo-VLA Ultra-Long**(有真 headroom)。

## 1. 命题 & 框架转变

- **不追"刷爆 libero_10"**(已饱和 94-98%,WM 证明不了价值)。
- **改证"效率使能能力"**:在**超长 horizon** 上,softmax(O(N²)) WM 当想象环境烧不起/OOM;我们的 **O(N) KDA(+AttnRes) WM 能学**,把 VLA 从 baseline 抬起来。
- benchmark:**LiLo-VLA Ultra-Long**(arXiv 2602.21531,LIBERO-native,MIT,已 load 验证):
  | 任务 | 步数 | baseline(OpenVLA-OFT/π0.5) | SOTA(LiLo-VLA) |
  |---|---|---|---|
  | complete_kitchen_organization | 9 | **0%** | 44% |
  | cooking_preparation_setup | 10 | **0%** | 44% |
  | organize_table | 16 | **0%** | 43% |
  - 巨大 headroom;baseline 0% → 任何有意义提升都可证明。

## 2. 组件(大多已就绪)

| 组件 | 状态 |
|---|---|
| **3 个 WM**(SDPA / KDA / KDA+AttnRes,全 matched recipe) | ✅ 已训(`sdpa_sdpaH10` / `kda_kdaonlyH10` / `kda_arlayerH10`)|
| **Ultra-Long benchmark** | ✅ 已装载(`get_benchmark("ultra_long")`,3 任务)|
| **V-JEPA2 encoder(1B,冻结)** | ✅ |
| **OpenVLA-OFT(7B 策略)** | ⏳ 需建 env(veRL/openvla-oft/flash-attn);仅 spatial ckpt 已缓存,需下 libero-10/goal ckpt |
| **GRPO RL(2 卡)** | ⏳ SimpleVLA-RL/veRL 8→2 卡移植(时间黑洞) |

## 3. ⚠️ 根本架构决策(必须 owner 拍板)—— latent WM ↔ pixel VLA 接口缺口

**问题**:V-JEPA2 是 JEPA,**没有像素 decoder**(只在 latent 里预测,不重建图像)。OpenVLA-OFT 吃**像素**。
→ **无法把像素 VLA 直接 roll 进我们的 latent WM 想象**(想象出的是 latent,VLA 要图像,中间没 decoder)。这正是 WMPO 为何选**像素** WM。

三条解法:

| 设计 | 做法 | 优 | 劣 |
|---|---|---|---|
| **A1 latent 策略 + 想象** | 训一个 latent-acting 策略(V-JEPA2 latent+语言→动作),BC 起步,在 WM 想象里 GRPO | thesis 最干净;**SDPA WM O(N²) 在超长 rollout 会 OOM/慢→正是要的证据** | 不是字面 OpenVLA-OFT |
| **A2 蒸 OFT→latent 策略 + 想象** | OpenVLA-OFT(像素教师)蒸馏成 latent-acting 学生,再想象 GRPO | 保留 OFT 知识 + 接项目 KD 主线 + 能想象 | 多一步蒸馏工程 |
| **B WM 当 dense reward** | OpenVLA-OFT 仍在**真 sim** 走像素;WM 给 latent-progress dense reward 喂 GRPO | 字面用真 OFT | 仍需真 sim;SDPA-vs-KDA 只差在 reward 打分成本(OOM 证据弱) |

**推荐:A2(蒸 OFT→latent 策略)或 A1。** 理由:① 用户框架"SDPA OOM/慢=好证据、KDA 能学就行"**只在 A(想象 rollout)成立**;② A2 顺带把"真 VLA"的知识蒸进来 + 串起项目蒸馏强项;③ 保住 O(N) WM 当想象环境的核心叙事。
**代价**:"真 VLA"从"字面 OpenVLA-OFT 跑 RL"变成"**OFT 蒸馏出的 latent 策略**在我们 WM 里想象 RL"。需 owner 确认这个解释。

## 4. 核心实验矩阵(对比 SDPA vs KDA vs KDA+AttnRes 当想象 WM)

固定策略 + benchmark,只换想象 WM:
| 想象 WM | Ultra-Long success(抬过 0%?) | 想象**显存** | 想象**时间/step** | 超长 horizon 是否 OOM/超时 |
|---|---|---|---|---|
| SDPA(O(N²)) | ? | 高 | 慢 | **预期 OOM/超时 = 正证据** |
| KDA(O(N)) | ? | 低 | 快 | 不 OOM |
| **KDA+AttnRes(O(N))** | **目标:最高、能学** | 低 | 快 | 不 OOM |

→ headline 两图:① success vs WM(KDA 能学、SDPA 学不动/跑不起);② 想象成本(显存/时间)vs horizon,SDPA O(N²) 爆、KDA O(N) 平。

## 5. 步骤(1-4)

1. ✅ **装 + load Ultra-Long**(done;init states 用 `weights_only=False`)。
2. ⏳ **OpenVLA-OFT baseline**:建 openvla-oft 推理 env + 下合适 ckpt(libero-10/goal)→ Ultra-Long 上跑(design-independent,预期 ~0%,坐实 headroom)。**长 pole / 第一个真风险**。
3. ⏳ **WM 接想象**(按 §3 决策):latent 策略(A1/A2)或 dense reward(B);3 个 WM 各接一遍。
4. ⏳ **想象 GRPO**(2 卡 LoRA)→ 抬 success + 出 SDPA-vs-KDA 成本/上限对比。

## 6. 风险
- **env 时间黑洞**(veRL/openvla-oft/flash-attn/2 卡移植)——SETUP 已知。先只建**推理 env** 拿 baseline,RL 栈后置。
- **7B on 2×H200**:LoRA + colocated 可行(plan 已算)。
- **接口缺口**(§3):未拍板会卡住 3-4。
- **诚实底线**:KDA WM 能把 Ultra-Long 抬过 0% 即成立;若也很难(SOTA 才 44%),退一步报"SDPA 跑不起 / KDA 跑得起 + 部分提升"也是有效的效率叙事。**不伪称解决 Ultra-Long。**

## 7. 成功判据(诚实)
- **主**:KDA(+AttnRes) WM 想象 RL 把 Ultra-Long success 抬过 OpenVLA-OFT 的 0% baseline。
- **次(同等重要)**:SDPA WM 在超长 horizon 想象上 **OOM/显著更慢**,KDA(+AttnRes) 跑得起 → 坐实"O(N) WM 是超长程想象的使能器"。
- 报告同时给 LiLo-VLA(44%)作天花板参照,不与之争 SOTA。
