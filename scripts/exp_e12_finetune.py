"""E1.2/E1.3 — KDA predictor 微调。
今晚模式 = teacher-matching 验证训练环（无需 LIBERO 数据）：
  teacher = 原 SDPA predictor（冻结）；student = kda_ify 后的 copy；
  在合成 latent 序列上让 student 匹配 teacher 的 next-latent。T=128 → chunk_kda 走可微路径。
验证：loss 下降（KDA 可训）、chunk_kda backward 通、watchdog/ckpt 工作。
（LIBERO 真数据微调路径见 exp_e12_finetune 的 --data 扩展，明天数据到位后上。）
"""
import sys, os, time, json, copy, torch
REPO = "/home/wm-vla-rl/_ext/vjepa2"; HERE = "/home/wm-vla-rl/module1_vjepa2ac"
sys.path.insert(0, REPO); sys.path.insert(0, HERE)
from src.models.utils.modules import build_action_block_causal_attention_mask
from kda_acblock import kda_ify_predictor

DISK_STOP = "/home/wm-vla-rl/scripts/DISK_STOP"
RES = "/home/wm-vla-rl/module1_vjepa2ac/results"
STEPS = int(os.environ.get("FT_STEPS", "150"))
T = int(os.environ.get("FT_T", "128"))


def main():
    dev = "cuda"
    from hubconf import vjepa2_ac_vit_giant
    enc, pred = vjepa2_ac_vit_giant(pretrained=True); del enc; torch.cuda.empty_cache()
    H = W = pred.grid_height; D = pred.predictor_embed.in_features; A = 2
    teacher = pred.to(dev).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    teacher.attn_mask = build_action_block_causal_attention_mask(T, H, W, A).to(dev)

    student = copy.deepcopy(pred)
    kda_ify_predictor(student); student = student.to(dev).train()
    student.attn_mask = torch.zeros(1, 1, dtype=torch.bool, device=dev)
    opt = torch.optim.AdamW([p for p in student.parameters() if p.requires_grad], lr=1e-4)
    ntr = sum(p.numel() for p in student.parameters() if p.requires_grad)
    print(f"teacher(SDPA) frozen; student(KDA) trainable={ntr/1e6:.0f}M; T={T} steps={STEPS} "
          f"gpu={torch.cuda.get_device_name(0)}", flush=True)

    losses = []
    for step in range(STEPS):
        if os.path.exists(DISK_STOP):
            print("DISK_STOP detected -> graceful stop", flush=True); break
        x = torch.randn(1, T * H * W, D, device=dev)
        a = torch.randn(1, T, 7, device=dev) * 0.05
        s = torch.randn(1, T, 7, device=dev) * 0.05
        with torch.no_grad():
            tgt = teacher(x, a, s)
        out = student(x, a, s)
        loss = torch.nn.functional.mse_loss(out, tgt)
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(loss.item())
        if step % 20 == 0 or step == STEPS - 1:
            print(f"step {step:4d} loss {loss.item():.5f}", flush=True)

    os.makedirs("/home/wm-vla-rl/_ext/ckpts", exist_ok=True)
    torch.save(student.state_dict(), "/home/wm-vla-rl/_ext/ckpts/kda_student_loopval.pt")
    json.dump(losses, open(f"{RES}/e12_loopval_losses.json", "w"))
    print(f"FINETUNE_LOOP_OK first={losses[0]:.5f} last={losses[-1]:.5f} "
          f"drop={(1-losses[-1]/losses[0])*100:.1f}%", flush=True)


main()
