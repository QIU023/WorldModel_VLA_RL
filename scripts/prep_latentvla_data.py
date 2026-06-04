"""(LatentVLA 数据) 编码 libero_90+libero_10 → 池化 latent + 语言 + 动作,给跨模态对齐用。
空间池化 256(16x16)→64(8x8) 省盘;帧 /STRIDE;每 demo 存 (latent[T,64,1408], actions, language, task)。
用验证 WM 的 encoder(一致性)。
"""
import os, sys, glob, h5py, numpy as np, torch
import torch.nn.functional as F
os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, "/home/wm-vla-rl/_ext/LIBERO"); sys.path.insert(0, "/home/wm-vla-rl/_ext/vjepa2"); sys.path.insert(0, "/home/wm-vla-rl/module1_vjepa2ac")
from app.vjepa_droid.transforms import make_transforms
from hubconf import vjepa2_ac_vit_giant
from libero.libero import benchmark
DEV = "cuda"; CHUNK = 16
PER_TASK = int(os.environ.get("PREP_PER_TASK", "10")); STRIDE = int(os.environ.get("PREP_STRIDE", "2"))
POOL = int(os.environ.get("PREP_POOL", "8"))           # 8x8=64 token
WM_ENC = os.environ.get("PREP_WM_ENC", "kda_arlayer_big_sftfull.pt")
OUT = os.environ.get("PREP_OUT", "/home/wm-vla-rl/_ext/datasets/latentvla_align_data.pt")
SUITES = os.environ.get("PREP_SUITES", "libero_90,libero_10").split(",")


def main():
    enc, pred = vjepa2_ac_vit_giant(pretrained=True); del pred; torch.cuda.empty_cache()
    sd = torch.load(f"/home/wm-vla-rl/_ext/ckpts/{WM_ENC}", map_location="cpu"); enc.load_state_dict(sd["encoder"])
    enc = enc.to(DEV).eval()
    transform = make_transforms(crop_size=256, random_horizontal_flip=False, random_resize_aspect_ratio=(1., 1.),
                                random_resize_scale=(1., 1.), reprob=0., auto_augment=False, motion_shift=False)

    @torch.inference_mode()
    def encode_demo(frames):
        clip = transform(frames).unsqueeze(0).to(DEV)
        B, C, T, H, W = clip.size()
        c = clip.permute(0, 2, 1, 3, 4).flatten(0, 1).unsqueeze(2).repeat(1, 1, 2, 1, 1)
        outs = [enc(c[i:i + CHUNK]).cpu() for i in range(0, T, CHUNK)]
        h = torch.cat(outs, 0).float()                  # [T,256,1408]
        h = F.layer_norm(h, (h.size(-1),))
        g = int(h.shape[1] ** 0.5)                       # 16
        h = h.view(h.shape[0], g, g, h.shape[2]).permute(0, 3, 1, 2)   # [T,1408,16,16]
        h = F.adaptive_avg_pool2d(h, (POOL, POOL))       # [T,1408,8,8]
        return h.flatten(2).permute(0, 2, 1)             # [T,64,1408]

    out = []; cnt = 0; gid = 0
    for suite in SUITES:
        bench = benchmark.get_benchmark_dict()[suite]()
        name2lang = {bench.get_task(i).name: bench.get_task(i).language for i in range(bench.n_tasks)}
        files = sorted(glob.glob(f"/home/wm-vla-rl/_ext/datasets/LIBERO/{suite}/*.hdf5"))
        print(f"PREP {suite}: {len(files)} files, per_task={PER_TASK} stride={STRIDE} pool={POOL}x{POOL}", flush=True)
        for f in files:
            stem = os.path.basename(f).replace("_demo.hdf5", "").replace(".hdf5", "")
            lang = name2lang.get(stem, None)
            if lang is None:                              # 文件名→benchmark task 匹配（容错）
                lang = next((v for k, v in name2lang.items() if k in stem or stem in k), stem.replace("_", " "))
            with h5py.File(f, "r") as hf:
                for j, k in enumerate(list(hf["data"].keys())[:PER_TASK]):
                    d = hf["data"][k]
                    frames = d["obs"]["agentview_rgb"][::STRIDE]
                    actions = np.asarray(d["actions"][::STRIDE], dtype=np.float32)
                    lat = encode_demo(frames)
                    n = min(lat.shape[0], actions.shape[0])
                    out.append(dict(latent=lat[:n].to(torch.bfloat16), actions=torch.tensor(actions[:n]),
                                    language=lang, task=gid, T=n)); cnt += 1
            gid += 1
        print(f"  {suite} done; total demos={cnt} tasks={gid}", flush=True)
    torch.save(out, OUT)
    langs = set(d["language"] for d in out)
    print(f"PREP_DONE demos={cnt} tasks={gid} distinct_languages={len(langs)} -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
