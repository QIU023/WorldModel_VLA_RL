"""多步自回归 rollout-fidelity —— WM 真正该被评的指标（比 exp_e15 的 1-步 teacher-force 更狠）。
从 C=4 帧 context 出发，自由自回归 roll H=8 步（真 action/state 作条件，喂回**预测** latent），
逐 horizon h 测 L1(pred_h, true_h)。AttnRes 若有用，应在长 horizon 上发散更慢。
对 3 个 matched-recipe ckpt（KDA / KDA+跨层AttnRes / SDPA）各评一遍 → 干净 3-way。
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
FPC = 14; FSTP = 5; CTX = 4; HOR = 8                      # 窗口14帧：context 4 + roll 8（留余量）
ROOT = "/home/wm-vla-rl/_ext/datasets/LIBERO"
# (ckpt, kda?, attnres?) —— 后缀走 env（默认 matched "2"；all-10-held 用 "H10"）
SUF = os.environ.get("EVAL_SUF", "2")
RUNS = [(f"sdpa_sdpa{SUF}_sftfull.pt", 0, 0), (f"kda_kdaonly{SUF}_sftfull.pt", 1, 0), (f"kda_arlayer{SUF}_sftfull.pt", 1, 1)]


def poses_to_diffs(p):
    xyz, th = p[:, :3], p[:, 3:6]
    m = [Rotation.from_euler("xyz", t).as_matrix() for t in th]
    ang = np.stack([Rotation.from_matrix(m[i + 1] @ m[i].T).as_euler("xyz") for i in range(len(m) - 1)], 0)
    return np.concatenate([xyz[1:] - xyz[:-1], ang, p[1:, -1:] - p[:-1, -1:]], 1)


def main():
    transform = make_transforms(random_horizontal_flip=False, random_resize_aspect_ratio=[0.75, 1.35],
                                random_resize_scale=[1.777, 1.777], reprob=0., auto_augment=False, motion_shift=False, crop_size=256)
    _all = sorted(glob.glob(f"{ROOT}/libero_10/*.hdf5"))
    held = _all if int(os.environ.get("EVAL_HOLD_ALL10", "0")) else _all[-2:]
    windows = []
    for f in held:
        with h5py.File(f, "r") as hf:
            for k in hf["data"].keys():
                T = hf["data"][k]["obs"]["agentview_rgb"].shape[0]
                n = FPC * FSTP
                for sf in range(0, T - n, n):
                    windows.append((f, k, sf))
    print(f"held: {len(held)} tasks, {len(windows)} 窗口  | ctx={CTX} roll={HOR}", flush=True)

    @torch.no_grad()
    def load_window(f, k, sf):
        with h5py.File(f, "r") as hf:
            d = hf["data"][k]
            idx = np.arange(sf, sf + FPC * FSTP, FSTP).astype(np.int64)
            buf = np.asarray(d["obs"]["agentview_rgb"][:])[idx]
            st = np.concatenate([np.asarray(d["obs"]["ee_pos"][:]), np.asarray(d["obs"]["ee_ori"][:]),
                                 np.asarray(d["obs"]["gripper_states"][:])[:, :1]], 1)[idx]
        a = torch.tensor(poses_to_diffs(st).astype(np.float32))[None]   # [1,T-1,7]
        s = torch.tensor(st.astype(np.float32))[None]                  # [1,T,7]
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

        def fwd_t(c):                                         # teacher encode 整窗 -> [1,T,tok,D]（layer_norm）
            B, C, T, H, W = c.shape
            cc = c.permute(0, 2, 1, 3, 4).flatten(0, 1).unsqueeze(2).repeat(1, 1, 2, 1, 1)
            h = enc(cc).view(B, T, -1, enc.embed_dim)
            return F.layer_norm(h, (h.size(-1),))

        def step(frames, a_ctx, s_ctx):                       # frames:[1,t,tok,D] -> pred 下一帧 [1,tok,D]
            flat = frames.flatten(1, 2)
            nxt = pred(flat, a_ctx, s_ctx)[:, -tok:]
            return F.layer_norm(nxt, (nxt.size(-1),))

        per_h = np.zeros(HOR); nseen = 0
        with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
            for f, k, sf in windows:
                clip, a, s = load_window(f, k, sf)
                clip, a, s = clip.to(DEV), a.to(DEV), s.to(DEV)
                z_true = fwd_t(clip)                          # [1,T,tok,D]
                frames = z_true[:, :CTX].clone()             # 自由 rollout 缓冲（喂回预测）
                for h in range(HOR):
                    Tc = CTX + h                              # 当前 context 帧数
                    nxt = step(frames, a[:, :Tc], s[:, :Tc]) # 预测帧 Tc（0-based）
                    per_h[h] += torch.mean(torch.abs(nxt[:, 0] - z_true[:, Tc])).item()
                    frames = torch.cat([frames, nxt.unsqueeze(1)], dim=1)
                nseen += 1
        per_h /= nseen
        line = "  ".join([f"h{h+1}={per_h[h]:.4f}" for h in range(HOR)])
        print(f"  {ckname:28s} kda={kda} ar={ar}  rollout-L1: {line}  | mean={per_h.mean():.4f}", flush=True)
        del enc, pred; torch.cuda.empty_cache()
    print("ROLLOUT_FIDELITY_DONE", flush=True)


if __name__ == "__main__":
    main()
