"""Hybrid attention surgery on iVideoGPT medium(24层/436M):
只换 2 层时间注意力为 KDA(其余 22 层保 softmax,Kimi-Linear hybrid 哲学,保质量)+ 跨层 AttnRes。
surgery-init(拷 q/k/v/o)→ 存 ckpt,train_gpt 可 --pretrained_transformer_path load 续训。
"""
import os, sys, glob, torch
sys.path.insert(0, "/home/wm-vla-rl/module_videowm")
from transformers import LlamaForCausalLM, AutoConfig
from safetensors.torch import load_file, save_file
from kda_llama import kdaify_llama, attnres_ify_llama

BD = glob.glob("/home/.hf_home/models--thuml--ivideogpt-oxe-64-act-free-medium/snapshots/*/transformer")[0]
OUT = "/home/wm-vla-rl/_ext/videowm_runs/hybrid_surgery_init"
KDA_LAYERS = {8, 16}                                   # 24 层里换 2 层(均匀分布)为 KDA
GROUPS = 6                                              # 24 层 → 6 组 AttnRes(深)

config = AutoConfig.from_pretrained(BD)
print(f"medium: vocab {config.vocab_size}, {config.num_hidden_layers}L/{config.hidden_size}d")
model = LlamaForCausalLM(config)
sd = load_file(os.path.join(BD, "model.safetensors"))
miss, unexp = model.load_state_dict(sd, strict=False)
print(f"loaded pretrained medium SDPA: missing {len(miss)}, unexpected {len(unexp)}")
assert len(unexp) == 0
kdaify_llama(model, layers=KDA_LAYERS, init_from_softmax=True)   # 只换 2 层
attnres_ify_llama(model, groups=GROUPS)
os.makedirs(OUT, exist_ok=True)
save_file(model.state_dict(), os.path.join(OUT, "model.safetensors"))
n = sum(p.numel() for p in model.parameters())
print(f"HYBRID_SURGERY_DONE {OUT} | {n/1e6:.1f}M | KDA layers {sorted(KDA_LAYERS)} | AttnRes groups {GROUPS}")
