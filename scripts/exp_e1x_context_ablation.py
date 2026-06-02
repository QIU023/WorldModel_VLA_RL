"""context 长度 ablation（能力探针）：KDA 微调@不同窗口 T，看长历史是否帮"预测下一帧"。
对每个 T：fresh kda_ify predictor，冻 backbone 训 KDA adapter，eval held-out 最后一帧 next-latent MSE。
长历史有用 ⇒ 最后一帧 MSE 随 T 下降（"保留长上下文"派的能力证据）。
"""
import sys, os, json, copy, random, torch
import torch.nn.functional as F
REPO = "/home/wm-vla-rl/_ext/vjepa2"; HERE = "/home/wm-vla-rl/module1_vjepa2ac"
sys.path.insert(0, REPO); sys.path.insert(0, HERE)
from kda_acblock import kda_ify_predictor

DISK_STOP = "/home/wm-vla-rl/scripts/DISK_STOP"
RES = "/home/wm-vla-rl/module1_vjepa2ac/results"
STEPS = int(os.environ.get("ABL_STEPS", "250"))
TS = [16, 32, 64, 96]


def make_student(tmpl, dev):
    st = copy.deepcopy(tmpl); kda_ify_predictor(st); st = st.to(dev)
    st.attn_mask = torch.zeros(1, 1, dtype=torch.bool, device=dev)
    for blk in st.predictor_blocks:
        for m in (blk.norm1, blk.attn, blk.norm2, blk.mlp):
            for p in m.parameters(): p.requires_grad_(False)
    for p in st.predictor_embed.parameters(): p.requires_grad_(False)
    for p in st.predictor_proj.parameters(): p.requires_grad_(False)
    return st


def win(demos, dev, T):
    while True:
        d = random.choice(demos)
        if d["T"] >= T + 1: break
    s0 = random.randint(0, d["T"] - T - 1)
    lat = d["latent"][s0:s0 + T + 1].float().to(dev); tok, dim = lat.shape[1], lat.shape[2]
    x = lat[:T].reshape(1, T * tok, dim); tgt = lat[1:T + 1].reshape(1, T * tok, dim)
    a = d["actions"][s0:s0 + T].float().unsqueeze(0).to(dev); s = d["states"][s0:s0 + T].float().unsqueeze(0).to(dev)
    return x, a, s, tgt, tok, dim


@torch.no_grad()
def eval_last(model, demos, dev, T, n=8):
    tot, cnt = 0., 0
    for d in demos[:n]:
        if d["T"] < T + 1: continue
        lat = d["latent"][:T + 1].float().to(dev); tok, dim = lat.shape[1], lat.shape[2]
        x = lat[:T].reshape(1, T * tok, dim)
        a = d["actions"][:T].float().unsqueeze(0).to(dev); s = d["states"][:T].float().unsqueeze(0).to(dev)
        out = model(x, a, s).view(1, T, tok, dim)[:, -1]          # 预测 frame T（给 0..T-1）
        tot += F.mse_loss(out, lat[T].unsqueeze(0)).item(); cnt += 1
    return tot / max(cnt, 1)


def main():
    dev = "cuda"
    from hubconf import vjepa2_ac_vit_giant
    enc, pred = vjepa2_ac_vit_giant(pretrained=True); del enc; torch.cuda.empty_cache()
    data = torch.load("/home/wm-vla-rl/_ext/datasets/libero_latents.pt")
    random.seed(0); random.shuffle(data); train, held = data[:32], data[32:]
    results = {}
    for T in TS:
        if os.path.exists(DISK_STOP): print("DISK_STOP", flush=True); break
        random.seed(T)
        student = make_student(pred, dev).train()
        opt = torch.optim.AdamW([p for p in student.parameters() if p.requires_grad], lr=1e-4)
        for step in range(STEPS):
            x, a, s, tgt, _, _ = win(train, dev, T)
            loss = F.mse_loss(student(x, a, s), tgt)
            opt.zero_grad(); loss.backward(); opt.step()
        student.eval()
        m = eval_last(student, held, dev, T)
        results[T] = m
        print(f"T={T:3d}: held-out last-frame next-latent MSE = {m:.5f}", flush=True)
        del student, opt; torch.cuda.empty_cache()
    json.dump(results, open(f"{RES}/e1x_context_ablation.json", "w"))
    print("ABLATION_OK", results, flush=True)


main()
