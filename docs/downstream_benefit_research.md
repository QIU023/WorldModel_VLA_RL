# JEPA-WM 下游收益:为何 RL/规划失败 + 正路(2026-06-04,核实版)

## 为何我的 RL/规划没收益(不是 WM bug)
- **LIBERO 多阶段 = flat JEPA 规划的已知 0% 失败区**。无任何已发表 JEPA-WM 用 flat CEM 解多阶段(HWM 2604.03208 实测 0%);V-JEPA2-AC 靠人喂 3 子目标图+脚本切步才"解"pick-place。
- energy 弱:L1(应 L2)、单 demo 末帧(无子目标)、可能池化(应 patch grid)。
- imagination RL(TD-MPC2/Dreamer 范式)需**学习式 value/reward**,不是 L1-to-goal。
→ 我的 0% 是预期。

## 标准 JEPA-WM 下游(两条)
1. **短程单目标 goal-reaching 规划**(DINO-WM):PushT 0.90/Maze 0.98/Wall 0.96/Reach 0.92,**L2-patch 距离 + CEM(300/30/30/H5)**,冻结编码器,单目标图。harness=`facebookresearch/jepa-wms`。"What Drives Success"(2512.24497):L2>L1、proprio、AdaLN+RoPE、2步rollout(sim)/6步(DROID)。
2. **JEPA 当 VLA 辅助表征/loss**(非规划/RL)。

## JEPA-VLA(2602.11832)= 最干净可复用模板
**Table 7(LIBERO-Long,gated cross-attn 注入冻结特征)**:baseline 25.8 → +DINOv2 26 → +SigLIP 33.6 → **+V-JEPA2 41.0**。**V-JEPA2 +15pp 且赢 SigLIP/DINOv2 = 干净 with-vs-without**。2卡可做(batch16)。recipe:VLA lr 5e-4,fusion 层 lr 1e-5~1e-4。
（VLA-JEPA 2602.10098 反而无 β=0 ablation,收益混淆在数据;LIBERO 97.2% 但 WM-loss 是否有用未隔离。）

## 推荐(改:augment 不 swap)
**线1 下游收益正路 = 把 KDA+AttnRes WM 的特征/预测经 gated cross-attn 注入小 VLA(OpenVLA-OFT)→ 复刻 JEPA-VLA Table 7 干净 ablation**:baseline vs +我的WM vs +SigLIP vs +DINOv2,LIBERO(-Long/-Plus)success。
- 比 B-proper(swap 编码器,掉点风险)**低风险**——augment 只加不换,只会涨。
- 我的 WM 的 **predictor 预测未来 = anticipation 特征**(AHEAD 路线),可比纯编码器更强。
- 退而求其次:DINO-WM 式 PushT/Maze goal-reaching 规划(换 benchmark,latent 规划真能 work 的地方),比 KDA+AttnRes vs SDPA 规划 success。
