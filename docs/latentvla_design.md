# LatentVLA 设计(LLaVA 范式,2026-06-04)

## 目标
冻结 V-JEPA2 编码器 → projector → LLM + action head,对齐到读 JEPA latent 的策略 →
在 O(N) KDA+AttnRes V-JEPA2-AC WM 想象里 imagination-RL(无 decoder)。

## 模块(对照 LLaVA:CLIP→MLP projector→Vicuna)
| 模块 | 内容 | 维度 | 训练? |
|---|---|---|---|
| **视觉编码器** | 冻结 V-JEPA2 enc(kda_arlayerH10 的 encoder) | obs→[256 tok, 1408] | ❄️ 冻结(训练时用预编码 latent,免跑) |
| **Projector** | MLP 1408→896(LLaVA 的 MLP) | [256,1408]→[256,896] | ✅ 训(stage1) |
| **LLM** | Qwen2-0.5B-Instruct(hidden 896) | — | ❄️stage1 / LoRA stage2 |
| **Action head** | 末 token hidden→MLP→7 维动作 | 896→7 | ✅ 训 |
| 语言 | LIBERO task.language → Qwen2 tokenizer | — | — |

**LLM 输入序列** = [256 projected latent embeds] + [language token embeds] → Qwen2 → 末 token hidden → action head → 7 维动作。

## 数据(跨模态对齐 = BC)
LIBERO demos(libero_10_arlayer_latents.pt,120 demo):每帧 (V-JEPA2 latent[256,1408], task.language, action[7])。
- **正样本对** = (latent_t, language) → action_t。训练时直接读预编码 latent(已有)。
- held-out:留 task8/9 测 action MSE。

## 训练(两阶段,LLaVA)
- **Stage 1 对齐**:冻结 enc+LLM,**只训 projector + action head**,BC L1(pred, demo action)。目标:LLM 能"读懂"JEPA latent + 语言 → 预测动作(held-out MSE 合理)。
- **Stage 2 微调**:训 projector + action head + **LLM LoRA**,更多 BC。
- **Stage 3(下一步)**:LatentVLA 在 V-JEPA2-AC WM 想象里 GRPO imagination-RL(无 decoder)。

## 验收(1-day)
Stage1 对齐:held-out action MSE 显著低于随机/常数基线 → 证明"对齐这步 work、LLM 读得动 JEPA latent"。

## Day-1 结果(2026-06-04)—— 对齐成功
scaled 数据(libero_90+10,870 demo,77 distinct languages,池化 64-token V-JEPA2 latent)+ Qwen3-1.7B + LoRA(10.8M 可训):
**held-out action MSE 0.0753 vs 常数 baseline 0.196(−62%)** → LLM 读得动 JEPA latent + 语言 → 预测动作。
- 对照:初版 8 语言 + Qwen2-0.5B(冻结)= FAILS(0.244>0.198)→ 数据规模 + 模型是瓶颈,scale 后翻成 BEATS。
- **接口缺口被证明可解**(像素 VLA 不能进 latent 想象的难题,用 VLM 式对齐绕过 + 保 LLM 容量)。
- 下一步:LatentVLA 在 O(N) KDA+AttnRes V-JEPA2-AC WM 想象里 imagination-RL(GRPO,无 decoder)。
  注:当前 ckpt 只存了 projector+action_head;RL 步要补存 LoRA 权重。

## 严格对齐判据(2026-06-04,回应"对齐够好吗")
按研究(LLaVA/OFT/VLA-JEPA recipe + "Hidden in Plain Sight" 2506.08008)修正:分组 lr(proj 1e-3/head 1e-4/LoRA 2e-5)、rank32 全模块、加 proprio、ablation 判据。
| 判据 | 值 | 判定 |
|---|---|---|
| **vision-zeroed gap**(决定性:清零 V-JEPA2 latent) | full 0.11 → 0.26(**+0.15**) | 🟢 **视觉强用**(即便有 proprio)→ 编码器对齐真实 |
| lang-zeroed gap | +0.009 | 🟡 语言弱用(单帧动作视觉主导)|
| full vs last-action-repeat | 0.11 vs 0.018 | 🔴 高于(但 last-action 特权;offline MSE 弱 gate)|
| full vs const(mean) | 0.11 vs 0.196 | 🟢 |
**结论**:编码器对齐**经决定性 vision-ablation 验证为真**(非"垃圾编码器")→ RL 有依据。caveat:BC 策略绝对质量中等(单动作难)+ 语言条件弱;**闭环 sim success 是真 gate**(待建,offline MSE 不预测闭环)。
