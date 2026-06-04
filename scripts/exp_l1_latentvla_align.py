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
        self.action_head = nn.Sequential(nn.LayerNorm(hid), nn.Linear(hid, 256), nn.GELU(), nn.Linear(256, 7))

    def forward(self, latent, lang_embeds, lang_mask):
        # latent [B,256,1408] -> [B,256,hid]; concat lang_embeds [B,L,hid]
        vis = self.projector(latent)
        seq = torch.cat([vis, lang_embeds], dim=1)
        vmask = torch.ones(vis.shape[0], vis.shape[1], device=vis.device, dtype=lang_mask.dtype)
        mask = torch.cat([vmask, lang_mask], dim=1)
        out = self.llm(inputs_embeds=seq, attention_mask=mask).last_hidden_state   # [B,S,hid]
        last = out[:, -1]                                # 末 token hidden
        return self.action_head(last)                    # [B,7]


def main():
    tok = AutoTokenizer.from_pretrained(LLM)
    llm = AutoModel.from_pretrained(LLM, torch_dtype=torch.bfloat16).to(DEV)
    for p in llm.parameters(): p.requires_grad_(False)
    hid = llm.config.hidden_size
    if USE_LORA:
        from peft import LoraConfig, get_peft_model
        llm = get_peft_model(llm, LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"], lora_dropout=0.05))
    model = LatentVLA(llm, hid).to(DEV)
    model.projector = model.projector.to(torch.bfloat16); model.action_head = model.action_head.to(torch.bfloat16)
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
    base_mse = ((torch.cat([d["actions"].float() for d in held]) - amean) ** 2).mean().item()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=3e-4)

    def make(pool):
        lat, act, ti = [], [], []
        for _ in range(BS):
            d = random.choice(pool); t = random.randint(0, d["T"] - 1)
            lat.append(d["latent"][t].float()); act.append(d["actions"][t].float()); ti.append(l2i[d["language"]])
        lat = torch.stack(lat).to(DEV, torch.bfloat16); act = torch.stack(act).to(DEV)
        ti = torch.tensor(ti, device=DEV)
        return lat, act, lang_emb_pad[ti], lang_msk_pad[ti]

    hist = []
    for step in range(STEPS):
        lat, act, le, lm = make(train)
        pred = model(lat, le, lm).float()
        loss = F.l1_loss(pred, act)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 100 == 0 or step == STEPS - 1:
            model.eval()
            with torch.no_grad():
                mses = []
                for _ in range(8):
                    lat, act, le, lm = make(held)
                    mses.append(F.mse_loss(model(lat, le, lm).float(), act).item())
                hmse = float(np.mean(mses))
            model.train()
            hist.append(dict(step=step, train_l1=loss.item(), held_mse=hmse))
            print(f"  step {step:4d} train_L1={loss.item():.4f} held_MSE={hmse:.4f} (const baseline {base_mse:.4f})", flush=True)

    torch.save(dict(projector=model.projector.state_dict(), action_head=model.action_head.state_dict()), f"{CKPT}/latentvla_align.pt")
    fin = hist[-1]["held_mse"]
    json.dump(dict(hist=hist, base_mse=base_mse, final_held_mse=fin), open(f"{RES}/latentvla_align.json", "w"))
    print(f"L1_DONE held_MSE {fin:.4f} vs const baseline {base_mse:.4f} "
          f"({'BEATS' if fin < base_mse else 'FAILS'} baseline → LLM {'reads' if fin<base_mse else 'cannot read'} JEPA latent)", flush=True)


if __name__ == "__main__":
    main()
