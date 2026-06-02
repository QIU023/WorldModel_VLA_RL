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
    if MODE == "kda":
        kda_ify_predictor(pred); student = pred.to(dev)
        student.attn_mask = torch.zeros(1, 1, dtype=torch.bool, device=dev)
        # 冻结 backbone（复用的 norm1/attn/norm2/mlp），只训 temporal-KDA + norm_t
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

    data = torch.load("/home/wm-vla-rl/_ext/datasets/libero_latents.pt")
    random.seed(0); random.shuffle(data)
    train, held = data[:32], data[32:]
    sdpa_mse = eval_mse(sdpa, held, dev, TW)
    print(f"[parity baseline] SDPA-pretrained held-out next-latent MSE = {sdpa_mse:.5f}", flush=True)

    student.train(); losses = []
    for step in range(STEPS):
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

    student.eval()
    ft_mse = eval_mse(student, held, dev, TW)
    torch.save(student.state_dict(), f"/home/wm-vla-rl/_ext/ckpts/{MODE}_predictor_libero.pt")
    json.dump({"mode": MODE, "train_losses": losses, "sdpa_zeroshot_mse": sdpa_mse, "finetuned_mse": ft_mse},
              open(f"{RES}/e12_{MODE}_finetune.json", "w"))
    print(f"[{MODE}] finetuned held-out MSE={ft_mse:.5f} (SDPA-zeroshot baseline={sdpa_mse:.5f})", flush=True)
    print("FINETUNE_REAL_OK", flush=True)


main()
