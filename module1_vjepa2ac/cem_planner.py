"""CEM / MPC planner over actions in V-JEPA 2 latent space.

仓库没有打包的 planner —— 从零写。参考 notebooks/energy_landscape_example.ipynb 的能量计算。

算法:
  1. 从高斯采样 N 条动作序列
  2. 每条经 rollout() 在 latent 空间展开
  3. 用 latent L1/L2 到目标 embedding 的能量打分
  4. 取 top-k elites 重拟合高斯 (mean/var)
  5. 迭代 M 次, 返回最优首动作 (MPC: 执行首动作后重规划)

interface (待实现):
    action = plan(predictor, z_cur, z_goal, horizon, n_samples, n_iters, n_elites)

验证: 在 Droid/goal 例子上确认 latent 能量随 CEM 迭代下降 (画能量曲线 → results/)。
这个能量函数 Week3 会被 reward_latent_wm.py 复用为 RL reward。
"""
# TODO(week1): 实现 CEM; energy(z_pred, z_goal); MPC 闭环; 存 planning demo + 能量曲线
