"""(LatentVLA stage1) 跨模态对齐 —— LLaVA 范式:冻结 Qwen2 + 训 projector(V-JEPA2 latent→LLM)+ action head。
BC:(V-JEPA2 latent[256,1408], task language) → action[7]。验证"LLM 读得动 JEPA latent + 语言 → 预测动作"。
训练只读预编码 latent(libero_10_arlayer_latents.pt),不跑 V-JEPA2 编码器。
"""
import os, sys, json, random, numpy as np, torch
import torch.nn as nn, torch.nn.functional as F
os.environ.setdefault("HF_HOME", "/home/.hf_home")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
from transformers import AutoModel, AutoTokenizer

DEV = "cuda"; CKPT = "/home/wm-vla-rl/_ext/ckpts"; RES = "/home/wm-vla-rl/module1_vjepa2ac/results"
DATA = os.environ.get("L1_DATA", "/home/wm-vla-rl/_ext/datasets/latentvla_align_data.pt")
LLM = os.environ.get("L1_LLM", "Qwen/Qwen3-1.7B")
STEPS = int(os.environ.get("L1_STEPS", "3000")); BS = int(os.environ.get("L1_BS", "32"))
USE_LORA = int(os.environ.get("L1_LORA", "1"))        # LoRA the LLM(stage2 风格)
LATENT_DIM = 1408


class LatentVLA(nn.Module):
    def __init__(self, llm, hid):
        super().__init__()
        self.llm = llm                                   # 冻结
        self.projector = nn.Sequential(nn.Linear(LATENT_DIM, hid), nn.GELU(), nn.Linear(hid, hid))
        self.proprio_proj = nn.Sequential(nn.Linear(7, hid), nn.GELU(), nn.Linear(hid, hid))   # proprio token
        self.action_head = nn.Sequential(nn.LayerNorm(hid), nn.Linear(hid, 256), nn.GELU(), nn.Linear(256, 7))

    def forward(self, latent, proprio, lang_embeds, lang_mask):
        vis = self.projector(latent)                     # [B,64,hid]
        pro = self.proprio_proj(proprio).unsqueeze(1)    # [B,1,hid]
        seq = torch.cat([vis, pro, lang_embeds], dim=1)
        pre = torch.ones(vis.shape[0], vis.shape[1] + 1, device=vis.device, dtype=lang_mask.dtype)
        mask = torch.cat([pre, lang_mask], dim=1)
        out = self.llm(inputs_embeds=seq, attention_mask=mask).last_hidden_state
        return self.action_head(out[:, -1])              # [B,7]


def main():
    tok = AutoTokenizer.from_pretrained(LLM)
    llm = AutoModel.from_pretrained(LLM, torch_dtype=torch.bfloat16).to(DEV)
    for p in llm.parameters(): p.requires_grad_(False)
    hid = llm.config.hidden_size
    if USE_LORA:
        from peft import LoraConfig, get_peft_model
        llm = get_peft_model(llm, LoraConfig(r=32, lora_alpha=32, lora_dropout=0.05,
              target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    model = LatentVLA(llm, hid).to(DEV)
    model.projector = model.projector.to(torch.bfloat16); model.proprio_proj = model.proprio_proj.to(torch.bfloat16); model.action_head = model.action_head.to(torch.bfloat16)
    ntr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"LatentVLA align | LLM={LLM} hid={hid} lora={USE_LORA} | trainable={ntr/1e6:.1f}M", flush=True)

    data = torch.load(DATA, map_location="cpu")
    langs = sorted(set(d["language"] for d in data)); l2i = {s: i for i, s in enumerate(langs)}
    emb = (llm.get_input_embeddings() if not USE_LORA else llm.base_model.get_input_embeddings())
    lang_emb, lang_msk = [], []
    for s in langs:
        t = tok(s, return_tensors="pt").to(DEV)
        with torch.no_grad(): e = emb(t.input_ids)[0]
        lang_emb.append(e.to(torch.bfloat16)); lang_msk.append(t.attention_mask[0])
    Ls = max(m.shape[0] for m in lang_msk)
    lang_emb_pad = torch.stack([F.pad(e, (0, 0, 0, Ls - e.shape[0])) for e in lang_emb])
    lang_msk_pad = torch.stack([F.pad(m, (0, Ls - m.shape[0])) for m in lang_msk])

    random.seed(0); random.shuffle(data)
    nh = max(40, len(data) // 7); train = data[:-nh]; held = data[-nh:]    # held-out by demo
    print(f"demos train={len(train)} held={len(held)} | distinct languages={len(langs)}", flush=True)
    allA = torch.cat([d["actions"].float() for d in train]); amean = allA.mean(0)
    heldA = torch.cat([d["actions"].float() for d in held])
    base_mse = ((heldA - amean) ** 2).mean().item()
    # last-action-repeat baseline（更强：动作自相关）
    lar = []
    for d in held:
        a = d["actions"].float()
        if a.shape[0] > 1: lar.append(((a[1:] - a[:-1]) ** 2).mean().item())
    lar_mse = float(np.mean(lar)) if lar else float("nan")
    # 分组 lr（projector 1e-3 / action_head 1e-4 / LoRA 2e-5）—— VLA-JEPA/OFT recipe
    pg = [{"params": model.projector.parameters(), "lr": 1e-3},
          {"params": model.proprio_proj.parameters(), "lr": 1e-3},
          {"params": model.action_head.parameters(), "lr": 1e-4}]
    lora_p = [p for n, p in model.named_parameters() if "lora" in n.lower() and p.requires_grad]
    if lora_p: pg.append({"params": lora_p, "lr": 2e-5})
    opt = torch.optim.AdamW(pg)

    def make(pool, zero_vis=False, zero_lang=False):
        lat, pro, act, ti = [], [], [], []
        for _ in range(BS):
            d = random.choice(pool); t = random.randint(0, d["T"] - 1)
            lat.append(d["latent"][t].float()); pro.append(d["states"][t].float())
            act.append(d["actions"][t].float()); ti.append(l2i[d["language"]])
        lat = torch.stack(lat).to(DEV, torch.bfloat16); pro = torch.stack(pro).to(DEV, torch.bfloat16)
        act = torch.stack(act).to(DEV); ti = torch.tensor(ti, device=DEV)
        le = lang_emb_pad[ti].clone(); lm = lang_msk_pad[ti]
        if zero_vis: lat = torch.zeros_like(lat)        # ablation: 清零 V-JEPA2 latent（测视觉是否超过 proprio+lang）
        if zero_lang: le = torch.zeros_like(le)         # ablation: 清零语言
        return lat, pro, act, le, lm

    @torch.no_grad()
    def eval_mse(zv=False, zl=False, n=16):
        model.eval(); ms = []
        for _ in range(n):
            lat, pro, act, le, lm = make(held, zero_vis=zv, zero_lang=zl)
            ms.append(F.mse_loss(model(lat, pro, le, lm).float(), act).item())
        model.train(); return float(np.mean(ms))

    print(f"baselines: const(mean)={base_mse:.4f} last-action-repeat={lar_mse:.4f}", flush=True)
    hist = []
    for step in range(STEPS):
        lat, pro, act, le, lm = make(train)
        loss = F.l1_loss(model(lat, pro, le, lm).float(), act)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 200 == 0 or step == STEPS - 1:
            full = eval_mse(); zv = eval_mse(zv=True)      # full vs vision-zeroed（决定性 ablation）
            hist.append(dict(step=step, train_l1=loss.item(), held_mse=full, vis_zeroed=zv))
            print(f"  step {step:5d} train_L1={loss.item():.4f} held_MSE={full:.4f} | vision-zeroed={zv:.4f} "
                  f"(gap {zv-full:+.4f} = vision {'USED' if zv-full>0.005 else 'IGNORED'})", flush=True)

    # 最终 ablation 套件
    full = eval_mse(n=40); zv = eval_mse(zv=True, n=40); zl = eval_mse(zl=True, n=40)
    # 存:projector + action_head + LoRA(给 RL/闭环用)
    sd = dict(projector=model.projector.state_dict(), action_head=model.action_head.state_dict())
    if USE_LORA:
        sd["lora"] = {k: v.cpu() for k, v in model.named_parameters() if "lora" in k.lower()}
    torch.save(sd, f"{CKPT}/latentvla_align.pt")
    out = dict(hist=hist, const_baseline=base_mse, last_action_baseline=lar_mse,
               full_mse=full, vision_zeroed_mse=zv, lang_zeroed_mse=zl,
               vision_gap=zv - full, lang_gap=zl - full)
    json.dump(out, open(f"{RES}/latentvla_align.json", "w"))
    verdict = "ALIGNED" if (full < lar_mse and zv - full > 0.01) else "WEAK(vision underused or below last-action)"
    print(f"L1_DONE full={full:.4f} | vs const {base_mse:.4f} last-action {lar_mse:.4f} | "
          f"vision-zeroed {zv:.4f}(gap {zv-full:+.4f}) lang-zeroed {zl:.4f}(gap {zl-full:+.4f}) → {verdict}", flush=True)


if __name__ == "__main__":
    main()
