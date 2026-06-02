"""block-causal latent rollout for V-JEPA 2-AC.  (E0.1)

语义来自 vjepa2 notebooks/utils/world_model_wrapper.py 的 step_predictor：
每步把 growing context 喂回 predictor，取最后一帧 tokens 作为新 latent，更新 pose。

⚠️ SDPA 路径（use_cache=False）= 每步在整段 context 上重跑 predictor → 每步 O(N²)、
整条 horizon O(N³)。这正是 P1 的 KDA recurrent-state rollout(Level-2)要消灭的低效：
use_cache=True 时携带固定大小的 SSM 状态，每步 O(1)、显存随 horizon 平。

predictor.forward(x, actions, states):
  x: [B, T*tokens_per_frame, enc_dim]（context tokens, 不含 action/state，内部再交织）
  actions/states: [B, T, 7]
"""
import torch
import torch.nn.functional as F


@torch.inference_mode()
def step_predictor(predictor, reps, actions, poses, tokens_per_frame, normalize_reps=True):
    """单步：reps [B,T,N_T,D] → 预测下一帧 latent [B,1,N_T,D]."""
    B, T, N_T, D = reps.size()
    flat = reps.flatten(1, 2)                                  # [B, T*N_T, D]
    nxt = predictor(flat, actions, poses)[:, -tokens_per_frame:]
    if normalize_reps:
        nxt = F.layer_norm(nxt, (nxt.size(-1),))
    return nxt.view(B, 1, N_T, D)


@torch.inference_mode()
def rollout(predictor, z0, actions, poses, tokens_per_frame,
            compute_new_pose=None, normalize_reps=True, use_cache=False):
    """自回归 block-causal latent rollout.

    z0: [B,1,N_T,D] 初始帧 latent；actions/poses: [B,H,7] / [B,1,7]（初始 pose）。
    返回预测 latent 序列 [B, H+1, N_T, D]（含 z0）。
    use_cache: P1 KDA recurrent-state 路径（Level-2，尚未实现）。
    """
    if use_cache:
        raise NotImplementedError(
            "P1/Level-2: KDA recurrent-state rollout — carry SSM state, O(1)/step. TODO.")
    frame_traj, pose_traj, action_traj = z0, poses, None
    outs = [z0]
    for t in range(actions.size(1)):
        a_t = actions[:, t:t + 1]
        action_traj = a_t if action_traj is None else torch.cat([action_traj, a_t], dim=1)
        nxt = step_predictor(predictor, frame_traj, action_traj, pose_traj, tokens_per_frame, normalize_reps)
        frame_traj = torch.cat([frame_traj, nxt], dim=1)
        if compute_new_pose is not None:
            pose_traj = torch.cat([pose_traj, compute_new_pose(pose_traj[:, -1:], a_t)], dim=1)
        outs.append(nxt)
    return torch.cat(outs, dim=1)
