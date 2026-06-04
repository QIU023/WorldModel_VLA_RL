# 方向决策 — LatentVLA via 对齐(2026-06-04,核实版)

## 接口缺口(为何 latent WM 不能直接接像素 VLA)
OpenVLA-OFT 的 LLM 预训练绑定它自己的视觉编码器(SigLIP/DINO)→ 期待该特征,非 JEPA latent;且 WM 想象 latent、VLA 要像素、无 decoder。**WMPO(2511.09515)明确据此选像素**("VLA 视觉理解无法用在不匹配 latent dynamics 里")。

## 解法:LatentVLA via 对齐(owner 洞察,改进我之前的 A2)
冻结 V-JEPA2 编码器 → projector → VLA 的 LLM + action head,在机器人 demo 上**对齐预训练**(LLaVA 范式)→ 读 JEPA latent 的 LatentVLA → 在 JEPA WM 想象里 imagination-RL,**无 decoder + 保 LLM 容量**(A2 是蒸进 tiny MLP,丢容量)。

**核实(逐一 fetch)**:
- V-JEPA2 本身已把 JEPA latent 对齐到 LLM(projector)做视频QA → **对齐步可行的直接证明**。
- JEPA-VLA(2602.11832):冻结 V-JEPA2 特征经 projector 融进 VLA(augment 非 swap)。
- AHEAD(2606.02486,极新待核实):VLA 读 WM 预测 latent(但用 VLA 自身特征空间、planning 非 RL)。
- **组合(VLA-LLM 对齐到 JEPA-WM latent + 在该 WM imagination-RL)= 空白未占**。
- 可行性:对齐(projector+LoRA)= 个位数 GPU-时(2卡足够);RL 阶段重,但无 decoder 省算力。
- 风险:WMPO 是反方(对齐质量是 crux);RL 阶段算力。

## LDA-1B(2602.12215,PKU-EPIC,RSS2026)判断
- 是 **DINO-latent 扩散 VLA 策略**(MM-DiT 并行去噪 action+未来DINO特征),冻结 Qwen3-VL+DINOv3;**非 latent-action 模型**(用真动作);**非可规划 WM**(dynamics 只是 co-training 正则,无 MPC/想象)。
- 与我们 JEPA WM:**互补**。最有价值=**实证"DINO-latent dynamics 共训 >> 像素/VAE"**(UWM VAE→DINO,RoboCasa 20%→49%)→ 支撑我们 latent-预测器论点。可作行为先验/动作接口(但 DINOv3 vs V-JEPA2 编码器要选 canonical)。

## 推荐
**新北极星 = LatentVLA via 对齐**(on-thesis、新颖、统一我们 O(N) WM):
> V-JEPA2 编码器 → projector 对齐到 VLA-LLM → 在 O(N) KDA+AttnRes V-JEPA2-AC WM 想象里 imagination-RL(无 decoder);O(N) WM = 长想象使能器。
**方向1(latent CEM 规划)= 快速中间结果**(证明 WM 可被 roll/control,是 LatentVLA RL 的前置)。
