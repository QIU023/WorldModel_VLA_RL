"""确定性大样本 held-out 评估 —— 修掉 exp_e14 里 eval_held 随机小样本的噪声。
对 2 个 held libero_10 任务的**所有 demo × 所有 stride 窗口**算 teacher-force jloss(L1)，确定性。
对每个 final ckpt(KDA-only / KDA+跨层AttnRes / SDPA)分别评 → 干净对比。
"""
import os, sys, glob, numpy as np, h5py, torch
import torch.nn.functional as F
from scipy.spatial.transform import Rotation
sys.path.insert(0, "/home/wm-vla-rl/_ext/vjepa2"); sys.path.insert(0, "/home/wm-vla-rl/module1_vjepa2ac")
from hubconf import vjepa2_ac_vit_giant
from kda_acblock import kda_ify_predictor, attnres_ify_predictor
from app.vjepa_droid.transforms import make_transforms
from src.models.utils.modules import build_action_block_causal_attention_mask

DEV = "cuda"; CKPT = "/home/wm-vla-rl/_ext/ckpts"
FPC = 8; FSTP = 5
ROOT = "/home/wm-vla-rl/_ext/datasets/LIBERO"
# (ckpt 文件, kda?, attnres?)
RUNS = [("sdpa_sdpa_big_sftfull.pt", 0, 0), ("kda_full_big_sftfull.pt", 1, 0), ("kda_arlayer_big_sftfull.pt", 1, 1)]


def poses_to_diffs(p):
    xyz, th = p[:, :3], p[:, 3:6]
    m = [Rotation.from_euler("xyz", t).as_matrix() for t in th]
    ang = np.stack([Rotation.from_matrix(m[i + 1] @ m[i].T).as_euler("xyz") for i in range(len(m) - 1)], 0)
    return np.concatenate([xyz[1:] - xyz[:-1], ang, p[1:, -1:] - p[:-1, -1:]], 1)


def main():
    transform = make_transforms(random_horizontal_flip=False, random_resize_aspect_ratio=[0.75, 1.35],
                                random_resize_scale=[1.777, 1.777], reprob=0., auto_augment=False, motion_shift=False, crop_size=256)
    held = sorted(glob.glob(f"{ROOT}/libero_10/*.hdf5"))[-2:]                 # 与训练同 held 划分
    # 预收集所有确定性窗口（每 demo 取不重叠 stride 窗口）
    windows = []
    for f in held:
        with h5py.File(f, "r") as hf:
            for k in hf["data"].keys():
                T = hf["data"][k]["obs"]["agentview_rgb"].shape[0]
                n = FPC * FSTP
                for sf in range(0, T - n, n):                                # 不重叠
                    windows.append((f, k, sf))
    print(f"held: {len(held)} tasks, {len(windows)} 确定性窗口", flush=True)

    @torch.no_grad()
    def load_window(f, k, sf):
        with h5py.File(f, "r") as hf:
            d = hf["data"][k]
            idx = np.arange(sf, sf + FPC * FSTP, FSTP).astype(np.int64)
            buf = np.asarray(d["obs"]["agentview_rgb"][:])[idx]
            st = np.concatenate([np.asarray(d["obs"]["ee_pos"][:]), np.asarray(d["obs"]["ee_ori"][:]), np.asarray(d["obs"]["gripper_states"][:])[:, :1]], 1)[idx]
        a = torch.tensor(poses_to_diffs(st).astype(np.float32))[None]
        s = torch.tensor(st.astype(np.float32))[None]
        clip = transform(buf)[None]
        return clip, a, s

    for ckname, kda, ar in RUNS:
        path = f"{CKPT}/{ckname}"
        if not os.path.exists(path):
            print(f"  [skip] {ckname} 不存在", flush=True); continue
        enc, pred = vjepa2_ac_vit_giant(pretrained=True)
        if kda:
            kda_ify_predictor(pred, use_attnres=False)
            if ar:
                attnres_ify_predictor(pred, num_blocks=8)
        tok = pred.grid_height * pred.grid_width
        pred.attn_mask = build_action_block_causal_attention_mask(FPC + 2, pred.grid_height, pred.grid_width, add_tokens=2).to(DEV).contiguous()
        sd = torch.load(path, map_location="cpu")
        enc.load_state_dict(sd["encoder"]); pred.load_state_dict(sd["predictor"])
        enc = enc.to(DEV).eval(); pred = pred.to(DEV).eval()

        def fwd_t(c):
            B, C, T, H, W = c.shape
            c = c.permute(0, 2, 1, 3, 4).flatten(0, 1).unsqueeze(2).repeat(1, 1, 2, 1, 1)
            h = enc(c).view(B, T, -1, enc.embed_dim).flatten(1, 2)
            return F.layer_norm(h, (h.size(-1),))

        def step(z, a, s):
            z = pred(z, a, s); return F.layer_norm(z, (z.size(-1),))

        tot = 0.0; n = 0
        with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
            for f, k, sf in windows:
                clip, a, s = load_window(f, k, sf)
                clip, a, s = clip.to(DEV), a.to(DEV), s.to(DEV)
                h = fwd_t(clip)
                z_tf = step(h[:, :-tok], a, s[:, :-1])
                _h = h[:, tok:z_tf.size(1) + tok]
                tot += torch.mean(torch.abs(z_tf - _h)).item(); n += 1
        print(f"  {ckname:32s} kda={kda} attnres={ar}  CLEAN held jloss(L1) = {tot/n:.5f}  (over {n} 窗口)", flush=True)
        del enc, pred; torch.cuda.empty_cache()
    print("CLEAN_EVAL_DONE", flush=True)


if __name__ == "__main__":
    main()
