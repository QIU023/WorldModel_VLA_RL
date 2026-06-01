"""vanilla LIBERO binary success reward (baseline / 兜底).

SimpleVLA-RL 默认的 outcome-level 0/1 reward。作为 latent-WM reward 的对照 + banked 结果。
注意 zero-variance 风险 (整组全对/全错 → advantage≈0): 单 suite 上确认 group 内有方差。

interface (待实现):
    reward_fn = BinarySuccessReward(env)
    r = reward_fn(episode_outcome)   # 1.0 success / 0.0 fail
"""
# TODO(week2): 接 SimpleVLA-RL 现成 reward 即可, 这里只是包一层便于和 latent-WM 同接口对比
