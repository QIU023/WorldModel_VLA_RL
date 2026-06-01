"""Frozen-feature state-regression probe —— GO/NO-GO 闸门.

冻结 V-JEPA2 编码器, 训小 linear/MLP head 回归 LIBERO 轨迹的 task-relevant 状态
(EE 位姿 7D, 物体位姿), 对比基线编码器:
  - 基线: SigLIP-224 (已缓存 /home/.hf_home/models--google--siglip-base-patch16-224)
  - 可选再加 DINOv2

产出: probe MSE 对比图 → results/probe_mse.png

🚦 GO/NO-GO 闸门 (决定 Week3 latent reward 路线):
  - V-JEPA2 特征能干净线性解码 LIBERO EE/物体位姿 → latent reward 走 AC predictor (Droid 真机训练, sim gap 需验证)
  - sim-domain gap 太大 → 退回通用 V-JEPA2 编码器做 latent-distance reward, 或在 LIBERO 帧上轻量 probe-adapt

interface (待实现):
    feats = extract_frozen_feats(encoder, libero_frames)
    mse = train_probe(feats, gt_states)   # linear & MLP head
    compare(['vjepa2', 'siglip'])
"""
# TODO(week1): 抽 LIBERO 轨迹帧+状态; 冻结特征; 训 probe; 画对比图; 记 GO/NO-GO 结论到 docs/writeup.md
