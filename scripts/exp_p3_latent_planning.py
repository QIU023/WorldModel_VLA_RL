"""(方向1 centerpiece) Latent CEM 规划 on LIBERO，用验证好的 O(N) WM —— efficiency-at-lookahead 为核心贡献。
对比 SDPA / KDA / KDA+AttnRes 当规划 WM:扫 lookahead horizon,测 success + 每-plan 时间/显存 + OOM。
结论目标:**KDA O(N) 撑得起长 lookahead 规划(softmax 在长 lookahead OOM/超时),且长 lookahead 不掉点**。
复用 cem_planner + V-JEPA2 enc + 验证 WM(enc+pred sftfull)+ LIBERO sim。
"""
import os, sys, glob, time, json, h5py, numpy as np, torch
import torch.nn.functional as F
os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, "/home/wm-vla-rl/_ext/vjepa2")
sys.path.insert(0, "/home/wm-vla-rl/module1_vjepa2ac")
sys.path.insert(0, "/home/wm-vla-rl/scripts")
import logging; logging.getLogger("utils.mpc_utils").setLevel(logging.WARNING)
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from app.vjepa_droid.transforms import make_transforms
from hubconf import vjepa2_ac_vit_giant
from kda_acblock import kda_ify_predictor, attnres_ify_predictor
from src.models.utils.modules import build_action_block_causal_attention_mask
import cem_planner as C

DEV = "cuda"; CKPT = "/home/wm-vla-rl/_ext/ckpts"; RES = "/home/wm-vla-rl/module1_vjepa2ac/results"
BENCH = os.environ.get("P3_BENCH", "libero_spatial")
NTASK = int(os.environ.get("P3_NTASK", "4")); NEP = int(os.environ.get("P3_EP", "1"))
MAX_STEPS = int(os.environ.get("P3_MAXSTEPS", "100"))
ROLLOUT = int(os.environ.get("P3_ROLLOUT", "4"))            # ★ lookahead horizon（扫描变量）
SAMPLES = int(os.environ.get("P3_SAMPLES", "200")); CEMSTEPS = int(os.environ.get("P3_CEMSTEPS", "6"))
MAXNORM = float(os.environ.get("ACT_MAXNORM", "0.3"))
WM = os.environ.get("P3_WM", "kda_arlayerH10_sftfull.pt")
KDA = int(os.environ.get("P3_KDA", "1")); AR = int(os.environ.get("P3_AR", "1")); NAME = os.environ.get("P3_NAME", "kda_attnres")


def main():
    enc, pred = vjepa2_ac_vit_giant(pretrained=True)
    if KDA:
        kda_ify_predictor(pred, use_attnres=False)
        if AR: attnres_ify_predictor(pred, num_blocks=8)
    sd = torch.load(f"{CKPT}/{WM}", map_location="cpu")
    enc.load_state_dict(sd["encoder"]); pred.load_state_dict(sd["predictor"])
    enc = enc.to(DEV).eval(); pred = pred.to(DEV).eval()
    tok = pred.grid_height * pred.grid_width
    # SDPA 需 block-causal mask（按 lookahead+context 建，合理 O(N²)）；KDA 时间轴因果线性、tiny 占位
    if KDA:
        pred.attn_mask = torch.zeros(2, 2, dtype=torch.bool, device=DEV)
    else:
        pred.attn_mask = build_action_block_causal_attention_mask(ROLLOUT + 3, pred.grid_height, pred.grid_width, add_tokens=2).to(DEV).contiguous()
    transform = make_transforms(crop_size=256, random_horizontal_flip=False, random_resize_aspect_ratio=(1., 1.),
                                random_resize_scale=(1., 1.), reprob=0., auto_augment=False, motion_shift=False)

    @torch.inference_mode()
    def encode(img):
        clip = transform(img[None]).unsqueeze(0).to(DEV)
        B, Cc, T, H, W = clip.size()
        c = clip.permute(0, 2, 1, 3, 4).flatten(0, 1).unsqueeze(2).repeat(1, 1, 2, 1, 1)
        h = enc(c).view(1, 1, -1, enc.embed_dim)
        return F.layer_norm(h.float(), (h.size(-1),))

    MPC = dict(rollout=ROLLOUT, samples=SAMPLES, topk=10, cem_steps=CEMSTEPS,
               momentum_mean=0.15, momentum_std=0.15, maxnorm=MAXNORM, verbose=False)
    bench = benchmark.get_benchmark_dict()[BENCH]()
    print(f"P3 [{NAME}] WM={WM} | lookahead={ROLLOUT} samples={SAMPLES} cem={CEMSTEPS} | {BENCH} ntask={NTASK}", flush=True)

    succ_all = 0; n_ep = 0; plan_times = []; status = "ok"; peak = 0.0
    try:
        for ti in range(NTASK):
            task = bench.get_task(ti)
            bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
            env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=128, camera_widths=128); env.seed(0)
            demo = sorted(glob.glob(f"/home/wm-vla-rl/_ext/datasets/LIBERO/{BENCH}/*.hdf5"))
            with h5py.File(demo[ti % len(demo)], "r") as hf:
                k0 = list(hf["data"].keys())[0]; goal_img = hf["data"][k0]["obs"]["agentview_rgb"][-1]
            goal_z = encode(np.asarray(goal_img))
            for ep in range(NEP):
                obs = env.reset(); success = False
                for t in range(MAX_STEPS):
                    z = encode(np.asarray(obs["agentview_image"]))
                    pos = obs.get("robot0_eef_pos", np.zeros(3)); quat = obs.get("robot0_eef_quat", np.array([0, 0, 0, 1.]))
                    from scipy.spatial.transform import Rotation
                    euler = Rotation.from_quat(quat).as_euler("xyz"); grip = obs.get("robot0_gripper_qpos", np.zeros(2))[:1]
                    pose = torch.tensor(np.concatenate([pos, euler, grip]), dtype=torch.float32, device=DEV).view(1, 1, 7)
                    torch.cuda.reset_peak_memory_stats(); t0 = time.time()
                    act_traj = C.plan(pred, z, pose, goal_z, tok, **MPC)
                    torch.cuda.synchronize(); plan_times.append(time.time() - t0)
                    peak = max(peak, torch.cuda.max_memory_allocated() / 1e9)
                    a = act_traj[0, 0].detach().cpu().numpy().astype(np.float32)
                    obs, r, done, info = env.step(a)
                    if env.check_success(): success = True; break
                    if done: break
                succ_all += int(success); n_ep += 1
                print(f"  task{ti} ep{ep}: success={success} steps={t+1} | plan {np.mean(plan_times)*1000:.0f}ms peak {peak:.1f}GB", flush=True)
            env.close()
    except RuntimeError as e:
        status = "OOM" if "out of memory" in str(e).lower() else f"ERR:{str(e)[:30]}"; torch.cuda.empty_cache()
    sr = succ_all / max(1, n_ep); mpt = float(np.mean(plan_times)) if plan_times else None
    out = dict(name=NAME, wm=WM, lookahead=ROLLOUT, samples=SAMPLES, status=status, success_rate=sr,
               n_ep=n_ep, plan_ms=round(mpt * 1000, 1) if mpt else None, peak_gb=round(peak, 1))
    json.dump(out, open(f"{RES}/p3_{NAME}_h{ROLLOUT}.json", "w"))
    print(f"P3_DONE [{NAME}] lookahead={ROLLOUT} status={status} success={sr:.2f}({succ_all}/{n_ep}) "
          f"plan={out['plan_ms']}ms peak={peak:.1f}GB", flush=True)


if __name__ == "__main__":
    main()
