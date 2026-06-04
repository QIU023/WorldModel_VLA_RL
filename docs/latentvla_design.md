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
