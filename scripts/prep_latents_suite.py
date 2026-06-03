"""通用 suite latent 编码（prep_libero_latents 的参数化版）。disk-aware：每任务取前 PREP_PER_TASK 条。
PREP_SUITE=libero_10（LIBERO-LONG，T~265 长程）→ finale 的 grounded-reward 正例 + Track A 长序列 WM 数据。
同 prep 归一化（per-token layer_norm）→ 与 reward model / WM 输入一致。
"""
import sys, glob, os, h5py, numpy as np, torch
import torch.nn.functional as F
REPO = "/home/wm-vla-rl/_ext/vjepa2"; sys.path.insert(0, REPO)
from app.vjepa_droid.transforms import make_transforms

SUITE = os.environ.get("PREP_SUITE", "libero_10")
PER_TASK = int(os.environ.get("PREP_PER_TASK", "12"))
OUT = os.environ.get("PREP_OUT", f"/home/wm-vla-rl/_ext/datasets/{SUITE}_latents.pt")
CHUNK = 16


@torch.inference_mode()
def encode_demo(encoder, transform, frames, dev):
    clip = transform(frames).unsqueeze(0).to(dev)
    B, C, T, H, W = clip.size()
    c = clip.permute(0, 2, 1, 3, 4).flatten(0, 1).unsqueeze(2).repeat(1, 1, 2, 1, 1)
    outs = [encoder(c[i:i + CHUNK]).cpu() for i in range(0, T, CHUNK)]
    h = torch.cat(outs, 0).float()
    return F.layer_norm(h, (h.size(-1),))


def main():
    dev = "cuda"
    from hubconf import vjepa2_ac_vit_giant
    enc, pred = vjepa2_ac_vit_giant(pretrained=True); del pred; torch.cuda.empty_cache()
    _wm = os.environ.get("PREP_WM_ENC", "")               # 从验证 WM ckpt 载训练好的 encoder（一致性）
    if _wm:
        sd = torch.load(_wm, map_location="cpu"); enc.load_state_dict(sd["encoder"])
        print(f"[enc] loaded trained encoder from {_wm}", flush=True)
    enc = enc.to(dev).eval()
    transform = make_transforms(crop_size=256, random_horizontal_flip=False,
                                random_resize_aspect_ratio=(1., 1.), random_resize_scale=(1., 1.),
                                reprob=0., auto_augment=False, motion_shift=False)
    files = sorted(glob.glob(f"/home/wm-vla-rl/_ext/datasets/LIBERO/{SUITE}/*.hdf5"))
    print(f"PREP suite={SUITE} files={len(files)} per_task={PER_TASK} -> {OUT}", flush=True)
    out = []; cnt = 0
    for ti, f in enumerate(files):
        with h5py.File(f, "r") as hf:
            for j, k in enumerate(hf["data"].keys()):
                if j >= PER_TASK:
                    break
                demo = hf["data"][k]
                frames = demo["obs"]["agentview_rgb"][:]
                actions = np.asarray(demo["actions"][:], dtype=np.float32)
                states = np.concatenate([demo["obs"]["ee_pos"][:], demo["obs"]["ee_ori"][:],
                                         demo["obs"]["gripper_states"][:][:, :1]], axis=-1).astype(np.float32)
                lat = encode_demo(enc, transform, frames, dev)
                out.append(dict(latent=lat.to(torch.bfloat16), actions=torch.tensor(actions),
                                states=torch.tensor(states), T=int(frames.shape[0]), task=ti))
                cnt += 1
        print(f"  task {ti+1}/{len(files)} done; total demos={cnt}; last T={out[-1]['T']}", flush=True)
    torch.save(out, OUT)
    sz = os.path.getsize(OUT) / 1e9
    print(f"PREP_SUITE_OK suite={SUITE} demos={len(out)} size={sz:.1f}GB latent={tuple(out[0]['latent'].shape)} -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
