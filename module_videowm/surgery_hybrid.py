"""Hybrid attention surgery on iVideoGPT medium(24层/436M),参数化 KDA 层数 K(env KDA_K)。
换 K 层时间注意力为 KDA(均匀分布;其余保 softmax,Kimi-Linear hybrid)+ 跨层 AttnRes。
surgery-init(拷 q/k/v/o)→ 存 ckpt(surgery_k{K}),train_gpt 可 load 续训。
扫 K∈{2,12,24} 给 FVD-vs-效率 Pareto。
"""
import os, sys, glob, torch
sys.path.insert(0, "/home/wm-vla-rl/module_videowm")
from transformers import LlamaForCausalLM, AutoConfig
from safetensors.torch import load_file, save_file
from kda_llama import kdaify_llama, attnres_ify_llama

K = int(os.environ.get("KDA_K", "2"))
L = 24
KDA_LAYERS = set(int(round(i * (L - 1) / max(1, K - 1))) for i in range(K)) if K > 1 else {L // 2}
GROUPS = 6
BD = glob.glob("/home/.hf_home/models--thuml--ivideogpt-oxe-64-act-free-medium/snapshots/*/transformer")[0]
OUT = f"/home/wm-vla-rl/_ext/videowm_runs/surgery_k{K}"

config = AutoConfig.from_pretrained(BD)
print(f"medium: vocab {config.vocab_size}, {config.num_hidden_layers}L/{config.hidden_size}d | K={K}")
model = LlamaForCausalLM(config)
sd = load_file(os.path.join(BD, "model.safetensors"))
miss, unexp = model.load_state_dict(sd, strict=False)
print(f"loaded pretrained medium SDPA: missing {len(miss)}, unexpected {len(unexp)}")
assert len(unexp) == 0
kdaify_llama(model, layers=KDA_LAYERS, init_from_softmax=True)
attnres_ify_llama(model, groups=GROUPS)
os.makedirs(OUT, exist_ok=True)
save_file(model.state_dict(), os.path.join(OUT, "model.safetensors"))
n = sum(p.numel() for p in model.parameters())
print(f"HYBRID_SURGERY_DONE {OUT} | {n/1e6:.1f}M | KDA layers {sorted(KDA_LAYERS)} | AttnRes groups {GROUPS}")
