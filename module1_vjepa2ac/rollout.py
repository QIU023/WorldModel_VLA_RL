"""block-causal latent rollout for V-JEPA 2-AC.

给定初始帧 embedding + 动作序列，自回归预测未来 latent 表征。
predictor 来自 facebookresearch/vjepa2 app/vjepa_droid/ (VisionTransformerPredictorAC)。

Module-4 副产品: KV-cache —— rollout 每步只算新时刻的 K/V，而非整段重算
(接 AttnRes cross-stage/block cache 的同构)。用 --no-cache 对照测加速比。

interface (待实现):
    encoder, predictor = load_vjepa2_ac()           # torch.hub vjepa2_ac_vit_giant
    z0 = encoder(frames)                             # 初始 latent
    zs = rollout(predictor, z0, actions, ee_states, use_cache=True)
    # zs: [T, ...] 预测的未来 latent 序列
"""
# TODO(week1): load encoder + AC predictor; 实现 rollout(); 加 KVCache; benchmark cache on/off
