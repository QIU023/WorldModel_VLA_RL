"""P2 — CEM/MPC 规划 in LIBERO sim，用高效 KDA WM（V-JEPA2-AC 同款 WM-native eval）。
单进程：LIBERO sim + V-JEPA2 encoder + 微调 KDA predictor + CEM。
超参起点 = vjepa2 repo WorldModel 默认（论文实现）。action scale 可 env 覆盖（ACT_MAXNORM）。
目标：闭环不崩 + 开始探索（success-rate 调参留后）。
"""
import os, sys, glob, h5py, numpy as np, torch
import torch.nn.functional as F
os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, "/home/wm-vla-rl/_ext/vjepa2")
sys.path.insert(0, "/home/wm-vla-rl/module1_vjepa2ac")

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from app.vjepa_droid.transforms import make_transforms
from hubconf import vjepa2_ac_vit_giant
from kda_acblock import kda_ify_predictor
import cem_planner as C

import logging
logging.getLogger("utils.mpc_utils").setLevel(logging.WARNING)   # 静音 cem 每步 INFO

DEV = "cuda"
TASK = int(os.environ.get("P2_TASK", "0"))
MAX_STEPS = int(os.environ.get("P2_MAXSTEPS", "120"))
MAXNORM = float(os.environ.get("ACT_MAXNORM", "0.05"))     # 论文 0.05(Droid)；LIBERO xyz~[-0.9,0.9] 探索调大
MPC = dict(rollout=2, samples=int(os.environ.get("P2_SAMPLES", "400")), topk=10,
           cem_steps=int(os.environ.get("P2_CEMSTEPS", "10")),
           momentum_mean=0.15, momentum_std=0.15, maxnorm=MAXNORM, verbose=False)


def main():
    enc, pred = vjepa2_ac_vit_giant(pretrained=True)
    kda_ify_predictor(pred)
    pred.load_state_dict(torch.load("/home/wm-vla-rl/_ext/ckpts/kda_predictor_libero.pt", map_location="cpu"))
    enc = enc.to(DEV).eval(); pred = pred.to(DEV).eval()
    pred.attn_mask = torch.zeros(1, 1, dtype=torch.bool, device=DEV)
    tok = pred.grid_height * pred.grid_width
    transform = make_transforms(crop_size=256, random_horizontal_flip=False,
                                random_resize_aspect_ratio=(1., 1.), random_resize_scale=(1., 1.),
                                reprob=0., auto_augment=False, motion_shift=False)

    @torch.inference_mode()
    def encode(img):  # img [H,W,3] uint8 -> [1,1,tok,dim]
        clip = transform(img[None]).unsqueeze(0).to(DEV)        # [1,C,1,256,256]
        B, Cc, T, H, W = clip.size()
        c = clip.permute(0, 2, 1, 3, 4).flatten(0, 1).unsqueeze(2).repeat(1, 1, 2, 1, 1)
        h = enc(c)                                              # [1, tok, dim]
        h = h.view(1, 1, h.shape[1], h.shape[2])               # [1,1,tok,dim]
        return F.layer_norm(h.float(), (h.size(-1),))

    # task + env
    bench = benchmark.get_benchmark_dict()["libero_spatial"]()
    task = bench.get_task(TASK)
    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=128, camera_widths=128)
    env.seed(0)

    # goal = 该 task 某 demo 的末帧 latent
    demo_f = sorted(glob.glob(f"/home/wm-vla-rl/_ext/datasets/LIBERO/libero_spatial/*{task.problem_folder}*"))
    demo_f = demo_f or sorted(glob.glob("/home/wm-vla-rl/_ext/datasets/LIBERO/libero_spatial/*.hdf5"))
    with h5py.File(demo_f[0], "r") as hf:
        k0 = list(hf["data"].keys())[0]
        goal_img = hf["data"][k0]["obs"]["agentview_rgb"][-1]   # 末帧
    goal_z = encode(np.asarray(goal_img))
    print(f"task={task.name} | maxnorm={MAXNORM} | goal_z {tuple(goal_z.shape)}", flush=True)

    obs = env.reset()
    print("obs keys:", [k for k in obs if "eef" in k or "gripper" in k or "image" in k][:8], flush=True)
    success = False
    for t in range(MAX_STEPS):
        img = obs["agentview_image"]
        z = encode(np.asarray(img))
        pos = obs.get("robot0_eef_pos", np.zeros(3))
        quat = obs.get("robot0_eef_quat", np.array([0, 0, 0, 1.]))
        from scipy.spatial.transform import Rotation
        euler = Rotation.from_quat(quat).as_euler("xyz")
        grip = obs.get("robot0_gripper_qpos", np.zeros(2))[:1]
        pose = torch.tensor(np.concatenate([pos, euler, grip]), dtype=torch.float32, device=DEV).view(1, 1, 7)
        act_traj = C.plan(pred, z, pose, goal_z, tok, **MPC)        # [1, rollout, 7]
        a = act_traj[0, 0].detach().cpu().numpy().astype(np.float32)
        obs, r, done, info = env.step(a)
        if env.check_success():
            success = True
        e = C.latent_goal_energy(z[:, 0], goal_z[:, 0]).item()
        if t % 10 == 0 or success:
            print(f"  t={t:3d} energy={e:.4f} success={success} act_xyz={np.round(a[:3],3)}", flush=True)
        if success or done:
            break
    env.close()
    print(f"P2_DONE task={TASK} success={success} steps={t+1} final_energy={e:.4f}", flush=True)


main()
