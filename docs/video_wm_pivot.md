# 方向转向:KDA-时间轴 + AttnRes-深度轴 的高效 action-conditioned 视频世界模型

日期:2026-06-04。决策人:用户。状态:已确认"一步到位 —— 核心 + 下游 RL"。

## 0. 为什么从 JEPA-WM 规划转出来(double-check 结论)

JEPA latent WM(V-JEPA2-AC / dino_wm)做下游对展示我们的 **KDA+AttnRes predictor** 是死胡同,三个结构性死穴(均已核实):

1. **无长上下文**:dino_wm `rollout` 滑动 `num_hist=3` 帧窗口,predictor 永远吃 3 帧 → SDPA 注意力固定 O((3·P)²),**永不因 horizon 爆炸** → 我们的效率前沿(核心卖点)在标准 JEPA 规划里**结构性无法显现**。只能证 parity,且任务饱和(point_maze 0.98)。
2. **VLA 模态不对齐**:困在 predictor latent;LatentVLA 闭环 0%;唯一 work 的 JEPA-VLA(41%)注入的是 **encoder 特征,不碰 predictor** → 我们的贡献在 VLA 路上不出场。
3. **无 RL = toy**:CEM 规划饱和 toy;RL(LIBERO 多阶段)是文档化 0%-失败区。

已有 JEPA 结果(V-JEPA2-AC:MSE all-10-held arlayer 0.370<kda 0.413<sdpa 0.416;imagination-cost SDPA T=768 OOM、KDA 13×)作为"高效 JEPA WM 架构"独立成立,但**下游无高价值落点**。保留,不再投入。

## 1. 为什么 Cosmos-like 像素空间视频 WM 是真命主场

| JEPA 死穴 | 像素空间视频 WM |
|---|---|
| 无长上下文 | 视频 = 长 token 序列(Cosmos DiT ~56k tokens;AR rollout 帧数增长)。**Wan-2.1:attention 占 ~95% 算力**(2509.24006)→ O(N²) 是真瓶颈 → **KDA O(N) 效率前沿真实显现** |
| 模态不对齐 | action-conditioned 像素生成天然连动作/策略;像素可被 VLM 读 |
| 无 RL=toy | **2025-26 RL-in-video-WM 集群**有可测"更好WM→更高success" |

## 2. Novelty —— 邻域拥挤但这个 cell 是空的(可辩护)

- 已做:线性/SSM 时间轴 + 空间注意力 for 视频 —— ARL²(2605.16579,gated-linear 时间轴+空间 softmax,2.26× 加速)、SS-WM(2505.20171,Mamba 时间轴 action-conditioned WM)、TTT-Video(2504.05298,TTT 层时间轴,**ablation 里 Gated DeltaNet 输给 TTT**)。
- **KDA(delta-rule,channel-wise gating)**:2510.26692,**纯语言**,无视频/时间轴应用。**Block-AttnRes(深度轴跨层注意力残差)**:2603.15031,**纯语言,从未碰视觉/视频**。
- **空 cell**:KDA(delta-rule,非 Mamba/GLA)时间轴 ⊕ 跨层 Block-AttnRes 深度轴,**在 action-conditioned 视频 WM 里**。无单一工作组合这两轴。
- 定位:对 ARL²/SS-WM 换时间轴算子为 KDA + 加深度轴;对 TTT-Video 用调好的 KDA(GDN 后继)打平/反超其 TTT-vs-GatedDeltaNet 的 gap。**不过度宣称"首个视频时间轴线性注意力"**(已被占),宣称**具体算子 + 深度轴组合 + WM 设定**。
- parity:线性注意力在视频生成已近 parity(SANA-Video 2509.24695 VBench 83.71;LinVideo 2510.08318 <3% 降),但靠 **hybrid**(留 ~25% full attn)—— 正合 KDA+MLA hybrid 哲学。

## 3. 下游:"更好 WM → 更高 success" 在这里可测(JEPA 给不了)

- **WoVR(2602.13977)**:GRPO RL 后训练于 Wan2.2-5B WM → LIBERO 39.95→**69.2%**、真 Franka 61.7→**91.7%**;**LIBERO-Long 上超过 WMPO**(WMPO 长任务零增益)→ **长 horizon 任务里 WM 质量 = 策略提升瓶颈**。
- **World-Gymnast(2602.02454)**:GRPO in 600M video WM,**就跑在 4×H200(140GB)、1-2 天**;真机 eggplant 4→**72%**。
- **WMPO(2511.09515)**:像素空间 on-policy GRPO;Mimicgen 33.6→47.1、真 ALOHA 53→70。
- 最干净可测链:policy-eval 相关性 Pearson **0.83-0.88,WM 变差→0.50-0.74**(2511.11520)。
- 长 horizon drift 是公认 live 失败模式(LIVE 2602.03747、MIND-V、Say-Dream-Act)→ **正是 KDA(O(N) 长 rollout)+ AttnRes(跨帧一致性抗 drift)瞄准点**。

## 4. 脚手架:iVideoGPT(thuml,2405.15223)

- 138M/436M,开源 code+weights,**VQGAN 压缩 tokenizer(256 ctx + 16/future 帧)+ HF LLaMA AR transformer**,已有 (frame, action, reward) token 接口,`mbrl/` 有下游 MBRL(Meta-World ≈ DreamerV3)。2×H200 可从零训。
- **架构注入点**:`AutoModelForCausalLM`(configs/llama)→ `HeadModelWithAction` 包动作。LLaMA 在扁平因果 token 序列上注意力 → **KDA 天然设定**(因果序列,fla `chunk_kda` 直接用)。替换 LlamaAttention→KDA + 跨层 Block-AttnRes。长 rollout→序列长→O(N²) 爆→KDA 赢。
- 我的 KDA(fla 版,`module1_vjepa2ac/kda_acblock.py`)+ AttnRes(`block_attn_res`)predictor-agnostic,可移植。**用 libero venv(torch2.12 + fla0.5.0 + KDA)**,视频长序列需 fla 快核(不像 dino_wm T=3 用纯-torch)。

## 5. 分阶段计划

- **Stage 1(核心)**:KDA+AttnRes 移植进 iVideoGPT LLaMA;训练 vs SDPA baseline;证 (a) 视频预测 parity(FVD),(b) **长 rollout 效率前沿**(序列长→SDPA O(N²) OOM/慢,KDA O(N) 撑住)。直接复用 KDA+AttnRes 代码,showcase 架构在真正吃长序列的场景。
- **Stage 2(下游 RL)**:接 WoVR/World-Gymnast 式 GRPO-in-WM loop(LIBERO-Long 或 iVideoGPT MBRL Meta-World),固定 VLA/reward/GRPO,swap WM = {SDPA vs KDA+AttnRes},测最终 success + rollout/sec + 长 horizon 保真度。目标:**更好/更长/更省 WM → 更高 success**。

## 6. 纪律(吸取本 session 教训)

- 不乱开 venv(复用 libero);不用 AC/grad-accum/乱压 bs 省显存(143GB 充裕,大 bs 吃满);串行实验避免 CPU 争抢;预载入用 3TB RAM;清理死产品看磁盘。
