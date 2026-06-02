"""Phase 2 — 编码 LIBERO demos → latent 序列（给 E1.2 真数据微调）。
follow notebook forward_target：每帧→2帧 tubelet→frozen V-JEPA2 encoder→latent。
state = [ee_pos(3), ee_ori(3), gripper(1)] = 7 维（对齐 predictor）。
"""
import sys, glob, os, h5py, numpy as np, torch
import torch.nn.functional as F
REPO = "/home/wm-vla-rl/_ext/vjepa2"; sys.path.insert(0, REPO)
from app.vjepa_droid.transforms import make_transforms

N_DEMOS = int(os.environ.get("PREP_DEMOS", "40"))
CHUNK = 16


@torch.inference_mode()
def encode_demo(encoder, transform, frames, dev):
    clip = transform(frames).unsqueeze(0).to(dev)              # [1,C,T,256,256]
    B, C, T, H, W = clip.size()
    c = clip.permute(0, 2, 1, 3, 4).flatten(0, 1).unsqueeze(2).repeat(1, 1, 2, 1, 1)  # [T,C,2,H,W]
    outs = []
    for i in range(0, T, CHUNK):
        outs.append(encoder(c[i:i + CHUNK]).cpu())
    h = torch.cat(outs, 0).float()                            # [T, tokens, dim]
    h = F.layer_norm(h, (h.size(-1),))
    return h


def main():
    dev = "cuda"
    from hubconf import vjepa2_ac_vit_giant
    enc, pred = vjepa2_ac_vit_giant(pretrained=True); del pred; torch.cuda.empty_cache()
    enc = enc.to(dev).eval()
    transform = make_transforms(crop_size=256, random_horizontal_flip=False,
                                random_resize_aspect_ratio=(1., 1.), random_resize_scale=(1., 1.),
                                reprob=0., auto_augment=False, motion_shift=False)
    files = sorted(glob.glob("/home/wm-vla-rl/_ext/datasets/LIBERO/libero_spatial/*.hdf5"))
    out = []; cnt = 0
    for f in files:
        if cnt >= N_DEMOS:
            break
        with h5py.File(f, "r") as hf:
            for k in hf["data"].keys():
                if cnt >= N_DEMOS:
                    break
                demo = hf["data"][k]
                frames = demo["obs"]["agentview_rgb"][:]                       # [T,128,128,3] uint8
                actions = np.asarray(demo["actions"][:], dtype=np.float32)     # [T,7]
                states = np.concatenate([demo["obs"]["ee_pos"][:], demo["obs"]["ee_ori"][:],
                                         demo["obs"]["gripper_states"][:][:, :1]], axis=-1).astype(np.float32)
                lat = encode_demo(enc, transform, frames, dev)                 # [T, tokens, dim]
                out.append(dict(latent=lat.to(torch.bfloat16), actions=torch.tensor(actions),
                                states=torch.tensor(states), T=int(frames.shape[0])))
                cnt += 1
                if cnt % 10 == 0:
                    print(f"encoded {cnt} demos; last latent={tuple(lat.shape)}", flush=True)
    path = "/home/wm-vla-rl/_ext/datasets/libero_latents.pt"
    torch.save(out, path)
    print(f"PREP_OK demos={len(out)} latent={tuple(out[0]['latent'].shape)} -> {path}", flush=True)


main()
