"""CEM / MPC planner over actions in V-JEPA 2 latent space.  (E0.2)

复用 vjepa2 notebooks/utils/mpc_utils.cem（完整 CEM：samples/topk/cem_steps/momentum）+
我们的 rollout.step_predictor。能量函数 latent_goal_energy = l1(z, goal)（latent 空间 L1 距离）
—— 同时是 E2.2 的 RL dense reward（一鱼两吃）。

P1：world_model step 走 KDA recurrent 路径时，CEM 在 planning root fork SSM 状态、
N 个候选共享历史（PR#15 fork/restore 思路）→ rollout 摆脱 O(N³)。
"""
import os, sys

_VJEPA = "/home/wm-vla-rl/_ext/vjepa2"
_HERE = os.path.dirname(os.path.abspath(__file__))
for p in (_HERE, _VJEPA, _VJEPA + "/notebooks"):
    if p not in sys.path:
        sys.path.insert(0, p)

from utils.mpc_utils import cem, compute_new_pose, l1   # noqa: E402
from rollout import step_predictor as _step             # noqa: E402


def latent_goal_energy(z, goal):
    """latent 空间 L1 距离（越小越接近目标）。E0.2 能量 / E2.2 dense reward 共用。"""
    return l1(z.flatten(1), goal.flatten(1))


def make_world_model_step(predictor, tokens_per_frame, normalize_reps=True):
    """构造 cem() 需要的 world_model step：f(frame_traj, action_traj, pose_traj)->(next_frame,next_pose)."""
    def step(frame_traj, action_traj, pose_traj):
        nxt = _step(predictor, frame_traj, action_traj, pose_traj, tokens_per_frame, normalize_reps)
        next_pose = compute_new_pose(pose_traj[:, -1:], action_traj[:, -1:])
        return nxt, next_pose
    return step


def plan(predictor, context_rep, context_pose, goal_rep, tokens_per_frame, **mpc_args):
    """CEM 规划：返回动作轨迹 [1, rollout, 7]；[0,0] 即下一步最优动作。"""
    wm_step = make_world_model_step(predictor, tokens_per_frame)
    return cem(context_frame=context_rep, context_pose=context_pose,
               goal_frame=goal_rep, world_model=wm_step, **mpc_args)
