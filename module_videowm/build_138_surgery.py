"""从论文 BAIR benchmark 模型 ivideogpt-bair-64-act-free(138M/12L,FVD 60.8)做 attention surgery。
两个变体:
  surgery_138_kimi  = 75% Kimi-Linear 3:1(9 KDA + 3 full@{3,7,11})+ AttnRes
  surgery_138_allkda= 100% KDA(12 层)+ AttnRes(效率上界,= Stage-1 结构,换更大数据重训)
init_from_softmax:拷贝 60.8 模型的 q/k/v/o(linearize 强模型)→ 短 adapt 测 FVD 保留率。
baseline K=0 = 该模型本身(eval-only,论文 60.8)。
"""
import os, sys, glob, torch
sys.path.insert(0, "/home/wm-vla-rl/module_videowm")
from transformers import LlamaForCausalLM, AutoConfig
from safetensors.torch import load_file, save_file
from kda_llama import kdaify_llama, attnres_ify_llama

BSNAP = glob.glob("/home/.hf_home/models--thuml--ivideogpt-bair-64-act-free/snapshots/*")[0]
BD = os.path.join(BSNAP, "transformer")
L = 12
FULL = {3, 7, 11}                                  # Kimi-Linear 3:1:每4层留第4层 full
KIMI_KDA = [i for i in range(L) if i not in FULL]  # 9 层 KDA(75%)
GROUPS = 4                                          # 与 Stage-1 一致(12 层)

cfg = AutoConfig.from_pretrained(BD)
print(f"bair-64 base: vocab {cfg.vocab_size}, {cfg.num_hidden_layers}L/{cfg.hidden_size}d")
sd0 = load_file(os.path.join(BD, "model.safetensors"))

for tag, kda_layers in [("kimi", set(KIMI_KDA)), ("allkda", set(range(L)))]:
    model = LlamaForCausalLM(cfg)
    miss, unexp = model.load_state_dict(sd0, strict=False)
    assert len(unexp) == 0, f"unexpected keys: {unexp[:5]}"
    kdaify_llama(model, layers=kda_layers, init_from_softmax=True)
    attnres_ify_llama(model, groups=GROUPS)
    out = f"/home/wm-vla-rl/_ext/videowm_runs/surgery_138_{tag}"
    os.makedirs(out, exist_ok=True)
    save_file(model.state_dict(), os.path.join(out, "model.safetensors"))
    n = sum(p.numel() for p in model.parameters())
    print(f"SURGERY_138_{tag.upper()}_DONE {out} | {n/1e6:.1f}M | KDA {len(kda_layers)}/{L} layers {sorted(kda_layers)}")
print("BUILD_138_SURGERY_ALL_DONE")
