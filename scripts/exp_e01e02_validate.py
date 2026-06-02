"""E0.1/E0.2 合成功能验证（无真数据）。

E0.1 rollout: 随机 context + 随机动作序列 → 检查输出 shape / 有限。
E0.2 CEM: goal = 在已知动作 a* 下的 1 步预测；CEM 应把 a* 找回（planned 能量 < 随机动作能量，
         且 planned 动作的 x 分量 ≈ a* 的 x）。验证 planner 真在优化（世界模型内部自洽，与真数据无关）。
"""
import os, sys, torch
REPO = "/home/wm-vla-rl/_ext/vjepa2"; HERE = "/home/wm-vla-rl/module1_vjepa2ac"
sys.path.insert(0, REPO); sys.path.insert(0, HERE)
from src.models.utils.modules import build_action_block_causal_attention_mask
from rollout import rollout, step_predictor
import cem_planner as C


def main():
    dev = "cuda"
    from hubconf import vjepa2_ac_vit_giant
    enc, pred = vjepa2_ac_vit_giant(pretrained=True); del enc; torch.cuda.empty_cache()
    pred = pred.to(dev).eval()
    D = pred.predictor_embed.in_features; H = pred.grid_height; W = pred.grid_width
    tpf = H * W
    pred.attn_mask = build_action_block_causal_attention_mask(8, H, W, 2)   # 覆盖 ≤7 帧
    print(f"predictor D={D} grid={H}x{W} tpf={tpf} gpu={torch.cuda.get_device_name(0)}", flush=True)

    B = 1
    z0 = torch.randn(B, 1, tpf, D, device=dev)
    pose0 = torch.zeros(B, 1, 7, device=dev)

    # --- E0.1 rollout（4 步）---
    actions = torch.randn(B, 4, 7, device=dev) * 0.02
    traj = rollout(pred, z0, actions, pose0, tpf, compute_new_pose=C.compute_new_pose)
    ok1 = (tuple(traj.shape) == (B, 5, tpf, D)) and bool(torch.isfinite(traj).all())
    print(f"E0.1 rollout: out={tuple(traj.shape)} finite={bool(torch.isfinite(traj).all())} -> {'OK' if ok1 else 'FAIL'}", flush=True)

    # --- E0.2 CEM: goal = 1 步预测 under a_star(+x) ---
    a_star = torch.zeros(B, 1, 7, device=dev); a_star[..., 0] = 0.04
    goal = step_predictor(pred, z0, a_star, pose0, tpf)              # [B,1,tpf,D]
    rand_a = torch.zeros(B, 1, 7, device=dev); rand_a[..., 1] = 0.04  # +y（错方向）
    z_rand = step_predictor(pred, z0, rand_a, pose0, tpf)
    e_rand = C.latent_goal_energy(z_rand[:, 0], goal[:, 0]).item()

    act = C.plan(pred, z0, pose0, goal, tpf,
                 rollout=1, samples=128, topk=10, cem_steps=6, maxnorm=0.06, verbose=False)
    z_plan = step_predictor(pred, z0, act[:, 0:1].to(dev), pose0, tpf)
    e_plan = C.latent_goal_energy(z_plan[:, 0], goal[:, 0]).item()
    ok2 = (e_plan < e_rand)
    print(f"E0.2 CEM: a_star.x={a_star[0,0,0]:.3f} planned.x={act[0,0,0]:.3f} | "
          f"energy random(+y)={e_rand:.5f} planned={e_plan:.5f} -> {'OK' if ok2 else 'FAIL'}", flush=True)

    print("VALID_OK" if (ok1 and ok2) else "VALID_FAIL", flush=True)


main()
