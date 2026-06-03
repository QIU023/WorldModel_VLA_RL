"""E1.2 真数据微调 + E1.3 parity（LIBERO latents）。
student = kda_ify predictor，冻结 pretrained backbone，只训 temporal-KDA(107M)。
目标：next-latent 预测（pred frame t ↔ 真实 latent[t+1]），不需 teacher。
parity：held-out 上 KDA-finetuned 的 next-latent MSE vs SDPA-pretrained（zero-shot）。
"""
import sys, os, json, copy, random, torch
import torch.nn.functional as F
REPO = "/home/wm-vla-rl/_ext/vjepa2"; HERE = "/home/wm-vla-rl/module1_vjepa2ac"
sys.path.insert(0, REPO); sys.path.insert(0, HERE)
from src.models.utils.modules import build_action_block_causal_attention_mask
from kda_acblock import kda_ify_predictor

DISK_STOP = "/home/wm-vla-rl/scripts/DISK_STOP"
RES = "/home/wm-vla-rl/module1_vjepa2ac/results"
STEPS = int(os.environ.get("FT_STEPS", "600"))
TW = int(os.environ.get("FT_T", "64"))            # 训练窗口帧数


def sample_window(demos, dev, T):
    while True:
        d = random.choice(demos)
        if d["T"] >= T + 1:
            break
    st = random.randint(0, d["T"] - T - 1)
    lat = d["latent"][st:st + T + 1].float().to(dev)          # [T+1, tok, dim]
    tok = lat.shape[1]; dim = lat.shape[2]
    x = lat[:T].reshape(1, T * tok, dim)                      # context frames 0..T-1
    tgt = lat[1:T + 1].reshape(1, T * tok, dim)               # next latents 1..T
    a = d["actions"][st:st + T].float().unsqueeze(0).to(dev)  # [1,T,7]
    s = d["states"][st:st + T].float().unsqueeze(0).to(dev)
    return x, a, s, tgt


@torch.no_grad()
def eval_mse(model, demos, dev, T, n=8):
    tot, cnt = 0.0, 0
    for d in demos[:n]:
        if d["T"] < T + 1:
            continue
        lat = d["latent"][:T + 1].float().to(dev)
        tok = lat.shape[1]; dim = lat.shape[2]
        x = lat[:T].reshape(1, T * tok, dim); tgt = lat[1:T + 1].reshape(1, T * tok, dim)
        a = d["actions"][:T].float().unsqueeze(0).to(dev); s = d["states"][:T].float().unsqueeze(0).to(dev)
        tot += F.mse_loss(model(x, a, s), tgt).item(); cnt += 1
    return tot / max(cnt, 1)


def main():
    dev = "cuda"
    from hubconf import vjepa2_ac_vit_giant
    enc, pred = vjepa2_ac_vit_giant(pretrained=True); del enc; torch.cuda.empty_cache()
    H = W = pred.grid_height
    sdpa = copy.deepcopy(pred).to(dev).eval()                 # parity 基线（SDPA, pretrained）
    sdpa.attn_mask = build_action_block_causal_attention_mask(TW, H, W, 2).to(dev)
    for p in sdpa.parameters():
        p.requires_grad_(False)

    MODE = os.environ.get("FT_MODE", "kda")
    FREEZE = int(os.environ.get("FT_FREEZE", "1"))   # 1=冻 backbone(adapter); 0=全参微调
    TAG = os.environ.get("FT_TAG", "")               # ckpt/json 后缀，避免覆盖
    if MODE == "kda":
        kda_ify_predictor(pred, use_attnres=bool(int(os.environ.get("USE_ATTNRES", "0")))); student = pred.to(dev)
        student.attn_mask = torch.zeros(1, 1, dtype=torch.bool, device=dev)
        if FREEZE:   # 冻结 backbone，只训 temporal-KDA + norm_t（adapter）
            for blk in student.predictor_blocks:
                for m in (blk.norm1, blk.attn, blk.norm2, blk.mlp):
                    for p in m.parameters():
                        p.requires_grad_(False)
            for p in student.predictor_embed.parameters(): p.requires_grad_(False)
            for p in student.predictor_proj.parameters(): p.requires_grad_(False)
    else:  # sdpa：公平 parity 参考——同样在 LIBERO 上微调（全参数）
        student = pred.to(dev)
        student.attn_mask = build_action_block_causal_attention_mask(TW, H, W, 2).to(dev)
    train_params = [p for p in student.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(train_params, lr=1e-4)
    print(f"[{MODE}] trainable={sum(p.numel() for p in train_params)/1e6:.0f}M T={TW} steps={STEPS} "
          f"gpu={torch.cuda.get_device_name(0)}", flush=True)

    _paths = os.environ.get("FT_DATA", "/home/wm-vla-rl/_ext/datasets/libero_latents.pt").split(",")
    data = []
    for _p in _paths:
        data += torch.load(_p.strip())                    # 多数据集拼接（Track A: libero_10 长序列 + spatial breadth）
    data = [d for d in data if d["T"] > TW + 1]           # 只留够长的 demo（长 T 窗口）
    random.seed(0); random.shuffle(data)
    nh = max(8, int(len(data) * 0.2)); train, held = data[:-nh], data[-nh:]   # 80/20 split
    sdpa_mse = eval_mse(sdpa, held, dev, TW)
    print(f"[parity baseline] SDPA-pretrained held-out next-latent MSE = {sdpa_mse:.5f}", flush=True)

    CKPT_EVERY = int(os.environ.get("FT_CKPT_EVERY", "200"))
    RESUME = f"/home/wm-vla-rl/_ext/ckpts/{MODE}{TAG}_resume.pt"
    start_step, losses = 0, []
    if os.path.exists(RESUME):                            # resume：中途崩/OOM 可续训
        _ck = torch.load(RESUME, map_location="cpu")
        student.load_state_dict(_ck["model"]); opt.load_state_dict(_ck["opt"])
        start_step, losses = _ck["step"] + 1, _ck.get("losses", [])
        print(f"[resume] {RESUME} @ step {start_step}", flush=True)
    student.train()
    for step in range(start_step, STEPS):
        if os.path.exists(DISK_STOP):
            print("DISK_STOP -> stop", flush=True); break
        x, a, s, tgt = sample_window(train, dev, TW)
        out = student(x, a, s)
        loss = F.mse_loss(out, tgt)
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(loss.item())
        if step % 50 == 0 or step == STEPS - 1:
            print(f"step {step:4d} loss {loss.item():.5f}", flush=True)
        if step and step % 200 == 0:
            student.eval(); m = eval_mse(student, held, dev, TW); student.train()
            print(f"  [eval@{step}] KDA held-out MSE = {m:.5f}", flush=True)
        if step and step % CKPT_EVERY == 0:              # 周期存 resume（防 OOM/崩丢进度）
            torch.save(dict(model=student.state_dict(), opt=opt.state_dict(),
                            step=step, losses=losses), RESUME)

    student.eval()
    ft_mse = eval_mse(student, held, dev, TW)
    torch.save(student.state_dict(), f"/home/wm-vla-rl/_ext/ckpts/{MODE}{TAG}_predictor_libero.pt")
    json.dump({"mode": MODE, "tag": TAG, "freeze": FREEZE, "n_train": len(train),
               "train_losses": losses, "sdpa_zeroshot_mse": sdpa_mse, "finetuned_mse": ft_mse},
              open(f"{RES}/e12_{MODE}{TAG}_finetune.json", "w"))
    print(f"[{MODE}] finetuned held-out MSE={ft_mse:.5f} (SDPA-zeroshot baseline={sdpa_mse:.5f})", flush=True)
    print("FINETUNE_REAL_OK", flush=True)


main()
