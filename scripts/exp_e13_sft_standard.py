"""(标准 recipe) V-JEPA2-AC predictor SFT —— 严格按官方 app/vjepa_droid/train.py + droid-256px-8f.yaml：
  • loss = L1（loss_exp=1.0）, normalize_reps（layer_norm）
  • objective = jloss(teacher-force 1步) + sloss(auto_steps 自回归 rollout)   ← 之前漏了 rollout
  • optimizer = AdamW + WSDSchedule(warmup-stable-decay) + CosineWDSchedule, lr 4.25e-4, wd 0.04
  • dtype = bf16
被迫偏离（标注，非随意）：① 冻结 encoder（受控架构对比，predictor 才是 O(N)）；② 用预编码 latent
（→ 无图像级 aug[官方本就极少]、fps=native 非官方 fps=4，时间尺度偏短，记为已知 deviation）；
③ LIBERO 数据；④ 2 卡 scale（batch/iters 砍小）；⑤ 架构换 KDA(/⊕AttnRes)。
"""
import os, sys, math, random, json, numpy as np, torch
import torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, "/home/wm-vla-rl/_ext/vjepa2")
sys.path.insert(0, "/home/wm-vla-rl/module1_vjepa2ac")
from hubconf import vjepa2_ac_vit_giant
from kda_acblock import kda_ify_predictor
from src.models.utils.modules import build_action_block_causal_attention_mask
from src.utils.schedulers import WSDSchedule, CosineWDSchedule

DEV = "cuda"
CKPT = "/home/wm-vla-rl/_ext/ckpts"; RES = "/home/wm-vla-rl/module1_vjepa2ac/results"
# -- 标准超参（droid-256px-8f.yaml）；可 env 覆盖
FPC = int(os.environ.get("SFT_FPC", "8"))                 # frames/clip（官方 8）
AUTO = min(int(os.environ.get("SFT_AUTO_STEPS", "2")), FPC - 1)   # 官方 2
LOSS_EXP = float(os.environ.get("SFT_LOSS_EXP", "1.0"))   # L1
BATCH = int(os.environ.get("SFT_BATCH", "8"))             # 官方 8/GPU
TOTAL = int(os.environ.get("SFT_ITERS", "6000"))         # 2卡 scale（官方 94.5k×256 全局）
WARM = int(0.05 * TOTAL); ANNEAL = int(0.05 * TOTAL)
LR = float(os.environ.get("SFT_LR", "4.25e-4")); START_LR = 7.5e-5; FINAL_LR = 0.0
WD = float(os.environ.get("SFT_WD", "0.04"))
MODE = os.environ.get("FT_MODE", "kda")
USE_ATTNRES = bool(int(os.environ.get("USE_ATTNRES", "0")))
TAG = os.environ.get("FT_TAG", "")
DATA = os.environ.get("FT_DATA", "/home/wm-vla-rl/_ext/datasets/libero_10_latents.pt").split(",")


def build_opt(predictor):                                 # 复刻官方 init_opt 的 param-group（bias/1d 排除 wd）
    pg = [{"params": [p for n, p in predictor.named_parameters() if p.requires_grad and "bias" not in n and p.ndim != 1]},
          {"params": [p for n, p in predictor.named_parameters() if p.requires_grad and ("bias" in n or p.ndim == 1)],
           "WD_exclude": True, "weight_decay": 0.0}]
    opt = torch.optim.AdamW(pg, betas=(0.9, 0.999), eps=1e-8)
    sched = WSDSchedule(opt, warmup_steps=WARM, anneal_steps=ANNEAL, start_lr=START_LR, ref_lr=LR, final_lr=FINAL_LR, T_max=TOTAL)
    wd_sched = CosineWDSchedule(opt, ref_wd=WD, final_wd=WD, T_max=TOTAL)
    return opt, sched, wd_sched


def main():
    enc, pred = vjepa2_ac_vit_giant(pretrained=True)
    if MODE == "kda":
        kda_ify_predictor(pred, use_attnres=USE_ATTNRES)
    pred = pred.to(DEV).train()
    enc = enc.to(DEV).eval()
    for p in enc.parameters(): p.requires_grad_(False)    # 冻结 encoder（受控；标注偏离）
    tok = pred.grid_height * pred.grid_width
    pred.attn_mask = build_action_block_causal_attention_mask(FPC + 2, pred.grid_height, pred.grid_width, add_tokens=2).to(DEV).contiguous()

    # 数据：预编码 latent（target = latent 本身，因 encoder 冻结）
    data = []
    for p in DATA:
        data += torch.load(p.strip())
    data = [d for d in data if d["T"] > FPC + 1]
    random.seed(0); random.shuffle(data)
    nh = max(8, len(data) // 5); train, held = data[:-nh], data[-nh:]
    print(f"SFT-standard | {MODE}{'+attnres' if USE_ATTNRES else ''} FPC={FPC} auto={AUTO} L{LOSS_EXP} "
          f"lr={LR} wd={WD} iters={TOTAL} batch={BATCH} | demos train/held={len(train)}/{len(held)}", flush=True)

    def sample(ds, bs):                                    # bs 个 FPC-帧窗口 -> (z[B,FPC*tok,dim], a[B,FPC-1,7], s[B,FPC,7])
        zs, as_, ss = [], [], []
        for _ in range(bs):
            d = random.choice(ds); st = random.randint(0, d["T"] - FPC - 1)
            lat = d["latent"][st:st + FPC].float()        # [FPC,tok,dim]
            zs.append(lat.reshape(FPC * tok, -1))
            as_.append(d["actions"][st:st + FPC - 1].float())     # [FPC-1,7]
            ss.append(d["states"][st:st + FPC].float())           # [FPC,7]
        return (torch.stack(zs).to(DEV), torch.stack(as_).to(DEV), torch.stack(ss).to(DEV))

    def step_pred(z, a, s):                               # 官方 _step_predictor + normalize_reps
        z = pred(z, a, s)
        return F.layer_norm(z, (z.size(-1),))

    def forward_predictions(h, a, s):                     # 复刻官方 forward_predictions
        # teacher forcing（1 步全预测）
        z_tf = step_pred(h[:, :-tok], a, s[:, :-1])
        # 自回归 rollout auto 步
        _z = torch.cat([h[:, :tok], z_tf[:, :tok]], dim=1)
        for n in range(1, AUTO):
            _z_nxt = step_pred(_z, a[:, :n + 1], s[:, :n + 1])[:, -tok:]
            _z = torch.cat([_z, _z_nxt], dim=1)
        z_ar = _z[:, tok:]
        return z_tf, z_ar

    def loss_fn(z, h):                                    # 官方 L_{loss_exp}
        _h = h[:, tok:z.size(1) + tok]
        return torch.mean(torch.abs(z - _h) ** LOSS_EXP) / LOSS_EXP

    @torch.no_grad()
    def eval_held():
        pred.eval(); tot = 0.0; n = 0
        for d in held:
            T = d["T"]
            for st in range(0, T - FPC, FPC):
                h = d["latent"][st:st + FPC].float().reshape(1, FPC * tok, -1).to(DEV)
                a = d["actions"][st:st + FPC - 1].float()[None].to(DEV); s = d["states"][st:st + FPC].float()[None].to(DEV)
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    z_tf, _ = forward_predictions(h, a, s)
                    tot += loss_fn(z_tf, h).item(); n += 1
        pred.train(); return tot / max(n, 1)

    opt, sched, wd_sched = build_opt(pred)
    losses = []
    for it in range(TOTAL):
        lr_now = sched.step(); wd_sched.step()            # 官方：scheduler.step() 返回新 lr
        h, a, s = sample(train, BATCH)
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            z_tf, z_ar = forward_predictions(h, a, s)
            jloss = loss_fn(z_tf, h); sloss = loss_fn(z_ar, h)
            loss = jloss + sloss
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(pred.parameters(), 1.0); opt.step()
        losses.append(loss.item())
        if it % 200 == 0 or it == TOTAL - 1:
            print(f"it {it:5d} loss {loss.item():.4f} [jloss {jloss.item():.4f} sloss {sloss.item():.4f}] lr {lr_now:.2e}", flush=True)
        if it and it % 1000 == 0:
            print(f"  [eval@{it}] held jloss(L1) = {eval_held():.4f}", flush=True)
    held_mse = eval_held()
    torch.save(pred.state_dict(), f"{CKPT}/{MODE}{TAG}_sft_std.pt")
    json.dump(dict(mode=MODE, attnres=USE_ATTNRES, fpc=FPC, auto=AUTO, loss_exp=LOSS_EXP, held_L1=held_mse, iters=TOTAL),
              open(f"{RES}/sft_std{TAG}.json", "w"))
    print(f"SFT_STD_DONE held jloss(L1)={held_mse:.4f} -> {MODE}{TAG}_sft_std.pt", flush=True)


if __name__ == "__main__":
    main()
