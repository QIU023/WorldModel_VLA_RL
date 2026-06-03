"""(B 全标准) V-JEPA2-AC SFT —— on-the-fly 编码 + 训 encoder + augs，严格按官方 train.py + droid 配方。
LIBERODataset 复刻 DROIDVideoDataset 接口（fps 抽帧、tubelet frameskip、actions=poses_to_diffs(states)、transform augs）。
唯一 deviation（标注）：① 数据=LIBERO(libero_10) 非 Droid；② 2 卡/单卡 scale（batch 砍小 + act-ckpt）；③ 架构换 KDA(/⊕AttnRes)；④ 无相机 extrinsics(LIBERO 无→zeros, use_extrinsics=false 本就不用)。
其余全按官方：L1(loss_exp=1) jloss+sloss(auto_steps), AdamW+WSD+CosineWD lr4.25e-4 wd0.04, bf16, target_encoder=frozen copy。
"""
import os, sys, copy, glob, math, random, json
from math import ceil
import numpy as np, h5py, torch
import torch.nn as nn, torch.nn.functional as F
from scipy.spatial.transform import Rotation
sys.path.insert(0, "/home/wm-vla-rl/_ext/vjepa2")
sys.path.insert(0, "/home/wm-vla-rl/module1_vjepa2ac")
from hubconf import vjepa2_ac_vit_giant
from kda_acblock import kda_ify_predictor
from app.vjepa_droid.transforms import make_transforms
from src.models.utils.modules import build_action_block_causal_attention_mask
from src.utils.schedulers import WSDSchedule, CosineWDSchedule

DEV = "cuda"
CKPT = "/home/wm-vla-rl/_ext/ckpts"; RES = "/home/wm-vla-rl/module1_vjepa2ac/results"
# 标准超参（droid-256px-8f.yaml）
FPC = int(os.environ.get("SFT_FPC", "8"))            # frames/clip
FSTP = int(os.environ.get("SFT_FSTP", "5"))          # 抽帧步长（LIBERO ~20Hz / fps4 ≈ 5）
TUBELET = 2
AUTO = int(os.environ.get("SFT_AUTO_STEPS", "2"))
LOSS_EXP = float(os.environ.get("SFT_LOSS_EXP", "1.0"))
BATCH = int(os.environ.get("SFT_BATCH", "2"))
TOTAL = int(os.environ.get("SFT_ITERS", "4000"))
WARM = int(0.05 * TOTAL); ANNEAL = int(0.05 * TOTAL)
LR = float(os.environ.get("SFT_LR", "4.25e-4")); START_LR = 7.5e-5
WD = float(os.environ.get("SFT_WD", "0.04"))
CROP = 256; PATCH = 16
MODE = os.environ.get("FT_MODE", "kda"); USE_ATTNRES = bool(int(os.environ.get("USE_ATTNRES", "0")))
TAG = os.environ.get("FT_TAG", "")
LIBERO_DIR = os.environ.get("SFT_LIBERO", "/home/wm-vla-rl/_ext/datasets/LIBERO/libero_10")


class LIBERODataset(torch.utils.data.Dataset):
    """复刻 DROIDVideoDataset：LIBERO hdf5 → (buffer[C,T,H,W], actions[T'-1,7], states[T',7], extrinsics, indices)。"""
    def __init__(self, libero_dir, transform, split="train", held_files=2):
        self.transform = transform
        self.samples = []                                # (file, demo_key)
        files = sorted(glob.glob(f"{libero_dir}/*.hdf5"))
        use = files[:-held_files] if split == "train" else files[-held_files:]   # 按 task 文件留 held（unseen 任务）
        for f in use:
            with h5py.File(f, "r") as hf:
                for k in hf["data"].keys():
                    self.samples.append((f, k))

    def __len__(self): return len(self.samples)

    def poses_to_diffs(self, poses):                     # 官方逻辑：state 差分 → action
        xyz, thetas = poses[:, :3], poses[:, 3:6]
        mats = [Rotation.from_euler("xyz", t, degrees=False).as_matrix() for t in thetas]
        xyz_d = xyz[1:] - xyz[:-1]
        ang_d = np.stack([Rotation.from_matrix(mats[t + 1] @ mats[t].T).as_euler("xyz") for t in range(len(mats) - 1)], 0)
        clo_d = poses[1:, -1:] - poses[:-1, -1:]
        return np.concatenate([xyz_d, ang_d, clo_d], axis=1)

    def __getitem__(self, idx):
        while True:
            try:
                f, k = self.samples[idx]
                with h5py.File(f, "r") as hf:
                    d = hf["data"][k]
                    rgb = d["obs"]["agentview_rgb"]; T = rgb.shape[0]
                    nframes = FPC * FSTP
                    if T < nframes + 1:
                        idx = random.randint(0, len(self.samples) - 1); continue
                    ef = random.randint(nframes, T - 1); sf = ef - nframes
                    indices = np.arange(sf, sf + nframes, FSTP).astype(np.int64)        # fpc 帧
                    buffer = np.asarray(d["obs"]["agentview_rgb"][:])[indices]          # [FPC,H,W,3]
                    ee_pos = np.asarray(d["obs"]["ee_pos"][:]); ee_ori = np.asarray(d["obs"]["ee_ori"][:])
                    grip = np.asarray(d["obs"]["gripper_states"][:])[:, :1]
                    states_full = np.concatenate([ee_pos, ee_ori, grip], axis=1)        # [T,7]
                    states = states_full[indices]                                       # FPC 帧（每 latent 帧一个 state，匹配 per-frame 编码）
                    actions = self.poses_to_diffs(states).astype(np.float32)
                    extr = np.zeros_like(states, dtype=np.float32)                      # LIBERO 无 extrinsics
                buffer = self.transform(buffer)                                         # → [C,FPC,H,W]
                return buffer, actions.astype(np.float32), states.astype(np.float32), extr, indices
            except Exception:
                idx = random.randint(0, len(self.samples) - 1)


def main():
    transform = make_transforms(random_horizontal_flip=False, random_resize_aspect_ratio=[0.75, 1.35],
                                random_resize_scale=[1.777, 1.777], reprob=0.0, auto_augment=False,
                                motion_shift=False, crop_size=CROP)
    ds = LIBERODataset(LIBERO_DIR, transform, split="train")
    held_ds = LIBERODataset(LIBERO_DIR, transform, split="held")
    loader = torch.utils.data.DataLoader(ds, batch_size=BATCH, shuffle=True, num_workers=4,
                                         drop_last=True, pin_memory=True, persistent_workers=True)

    enc, pred = vjepa2_ac_vit_giant(pretrained=True)
    if MODE == "kda":
        kda_ify_predictor(pred, use_attnres=USE_ATTNRES)
    enc = enc.to(DEV).train(); pred = pred.to(DEV).train()
    ACT_CKPT = bool(int(os.environ.get("SFT_ACT_CKPT", "0")))     # 默认关（显存够，关掉更快）
    enc.use_activation_checkpointing = ACT_CKPT; pred.use_activation_checkpointing = ACT_CKPT
    target_encoder = copy.deepcopy(enc).to(DEV).eval()                                  # 官方：frozen copy（无 EMA 更新）
    for p in target_encoder.parameters(): p.requires_grad_(False)
    tok = pred.grid_height * pred.grid_width
    pred.attn_mask = build_action_block_causal_attention_mask(FPC + 2, pred.grid_height, pred.grid_width, add_tokens=2).to(DEV).contiguous()

    # optimizer：官方 param-group（enc+pred 都训，bias/1d 排除 wd）+ WSD/CosineWD
    decay, no_decay = [], []
    for m in (enc, pred):
        for n, p in m.named_parameters():
            if not p.requires_grad: continue
            (no_decay if (p.ndim == 1 or "bias" in n) else decay).append(p)
    opt = torch.optim.AdamW([{"params": decay}, {"params": no_decay, "weight_decay": 0.0}], betas=(0.9, 0.999), eps=1e-8)
    sched = WSDSchedule(opt, warmup_steps=WARM, anneal_steps=ANNEAL, start_lr=START_LR, ref_lr=LR, final_lr=0.0, T_max=TOTAL)
    wd_sched = CosineWDSchedule(opt, ref_wd=WD, final_wd=WD, T_max=TOTAL)
    scaler = torch.cuda.amp.GradScaler()
    print(f"SFT-FULL(B) | {MODE}{'+attnres' if USE_ATTNRES else ''} FPC={FPC} fstp={FSTP} auto={AUTO} "
          f"lr={LR} wd={WD} batch={BATCH} iters={TOTAL} | demos={len(ds)} train enc+pred", flush=True)

    def forward_target(c):                               # 官方 forward_target（on-the-fly 编码 + layer_norm）
        with torch.no_grad():
            B, C, T, H, W = c.shape
            c = c.permute(0, 2, 1, 3, 4).flatten(0, 1).unsqueeze(2).repeat(1, 1, 2, 1, 1)
            h = target_encoder(c)
            h = h.view(B, T, -1, h.size(-1)).flatten(1, 2)
            return F.layer_norm(h, (h.size(-1),))

    def forward_context(c):                              # context 用可训 encoder（与 target 同输入）
        B, C, T, H, W = c.shape
        c = c.permute(0, 2, 1, 3, 4).flatten(0, 1).unsqueeze(2).repeat(1, 1, 2, 1, 1)
        h = enc(c).view(B, T, -1, enc.embed_dim if hasattr(enc, "embed_dim") else 1408).flatten(1, 2)
        return F.layer_norm(h, (h.size(-1),))

    def step_pred(z, a, s):
        z = pred(z, a, s); return F.layer_norm(z, (z.size(-1),))

    def forward_predictions(z, a, s):                    # 官方
        z_tf = step_pred(z[:, :-tok], a, s[:, :-1])
        _z = torch.cat([z[:, :tok], z_tf[:, :tok]], dim=1)
        for n in range(1, AUTO):
            _z_nxt = step_pred(_z, a[:, :n + 1], s[:, :n + 1])[:, -tok:]
            _z = torch.cat([_z, _z_nxt], dim=1)
        return z_tf, _z[:, tok:]

    def loss_fn(z, h):
        _h = h[:, tok:z.size(1) + tok]
        return torch.mean(torch.abs(z - _h) ** LOSS_EXP) / LOSS_EXP

    @torch.no_grad()
    def eval_held(n=16):                                 # held-out jloss(L1)（teacher-force），unseen 任务
        enc.eval(); pred.eval(); tot = 0.0
        for i in range(n):
            buffer, actions, states, _, _ = held_ds[random.randint(0, len(held_ds) - 1)]
            clips = buffer[None].to(DEV); a = torch.tensor(actions)[None].to(DEV); s = torch.tensor(states)[None].to(DEV)
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                h = forward_target(clips); z_tf, _ = forward_predictions(h, a, s)
                tot += loss_fn(z_tf, h).item()
        enc.train(); pred.train(); return tot / n

    it = 0; losses = []
    data_iter = iter(loader)
    while it < TOTAL:
        try:
            buffer, actions, states, extr, _ = next(data_iter)
        except StopIteration:
            data_iter = iter(loader); continue
        clips = buffer.to(DEV, non_blocking=True)
        a = actions.to(DEV, dtype=torch.float, non_blocking=True)
        s = states.to(DEV, dtype=torch.float, non_blocking=True)
        lr_now = sched.step(); wd_sched.step()
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            h = forward_target(clips)                    # target（frozen encoder）
            zc = forward_context(clips)                  # context（trainable encoder）
            z_tf, z_ar = forward_predictions(zc, a, s)
            jloss = loss_fn(z_tf, h); sloss = loss_fn(z_ar, h)
            loss = jloss + sloss
        opt.zero_grad(); scaler.scale(loss).backward()
        scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(list(enc.parameters()) + list(pred.parameters()), 1.0)
        scaler.step(opt); scaler.update()
        losses.append(loss.item())
        if it % 100 == 0 or it == TOTAL - 1:
            print(f"it {it:5d} loss {loss.item():.4f} [jloss {jloss.item():.4f} sloss {sloss.item():.4f}] "
                  f"lr {lr_now:.2e} mem {torch.cuda.max_memory_allocated()/1e9:.0f}GB", flush=True)
        if it and it % 1000 == 0:
            print(f"  [eval@{it}] held jloss(L1) = {eval_held():.4f}", flush=True)
        it += 1
    held = eval_held(32)
    torch.save(dict(encoder=enc.state_dict(), predictor=pred.state_dict()), f"{CKPT}/{MODE}{TAG}_sftfull.pt")
    json.dump(dict(mode=MODE, attnres=USE_ATTNRES, fpc=FPC, auto=AUTO, iters=TOTAL,
                   held_jloss=held, final_loss=float(np.mean(losses[-50:]))), open(f"{RES}/sftfull{TAG}.json", "w"))
    print(f"SFT_FULL_DONE held jloss(L1)={held:.4f} final_loss={np.mean(losses[-50:]):.4f} -> {MODE}{TAG}_sftfull.pt", flush=True)


if __name__ == "__main__":
    main()
