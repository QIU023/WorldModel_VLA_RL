"""latent-WM reward (HERO) —— V-JEPA2-AC 世界模型导出的 dense RL reward.

接 veRL 的 reward hook, 替换/增强 LIBERO 稀疏 binary success reward。
复用 module1 的 encoder / energy 函数。

设计 (首选 latent goal-distance):
    z_goal = encoder(goal_image)                    # 任务目标 latent (一次性算好)
    r_t = - dist(encoder(obs_t), z_goal)            # dense, grounded in WM 表征

备选 (world-model imagination reward):
    用 AC predictor 想象动作序列在 latent 空间能否到目标, reward = 预测能量。

dense reward 的好处: 天然有 within-group 方差 → 缓解 AttnRes 踩过的 GRPO zero-variance 坑。
Module-4: encoder forward 是 RL 采样瓶颈 → batched latent inference + 编码器输出缓存。

interface (待实现):
    reward_fn = LatentWMReward(encoder, goal_specs, mode='goal_dist'|'imagination')
    r = reward_fn(obs_batch, success_flags)
"""
# TODO(week3): 实现 reward; 接 veRL hook; vs reward_binary 对比; KL/budget/shaping ablation; reward-hacking 监测
