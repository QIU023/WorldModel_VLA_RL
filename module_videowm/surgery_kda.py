"""Attention surgery:把预训练 iVideoGPT-bair-64 SDPA transformer 线性化成 KDA+AttnRes。
build 138M(vocab 16386)→ load 预训练 SDPA → kdaify(init_from_softmax,拷 q/k/v/o)+ attnres →
存 ckpt(结构与 train_gpt 的 kdaify+attnres 一致,可被 --pretrained_transformer_path strict load)。
之后短 adapt finetune 恢复质量。Attention Surgery 2509.24899 / LinVideo 2510.08318 范式。
"""
import os, sys, glob, torch
sys.path.insert(0, "/home/wm-vla-rl/module_videowm")
from transformers import LlamaForCausalLM, AutoConfig
from safetensors.torch import load_file, save_file
from kda_llama import kdaify_llama, attnres_ify_llama

BD = glob.glob("/home/.hf_home/models--thuml--ivideogpt-bair-64-act-free/snapshots/*/transformer")[0]
OUT = "/home/wm-vla-rl/_ext/videowm_runs/kda_surgery_init"
GROUPS = int(os.environ.get("ATTNRES_GROUPS", "4"))

config = AutoConfig.from_pretrained(BD)
print(f"pretrained config: vocab {config.vocab_size}, {config.num_hidden_layers}L/{config.hidden_size}d")
model = LlamaForCausalLM(config)
sd = load_file(os.path.join(BD, "model.safetensors"))
miss, unexp = model.load_state_dict(sd, strict=False)
print(f"loaded pretrained SDPA: missing {len(miss)}, unexpected {len(unexp)}")
assert len(unexp) == 0, f"unexpected keys: {unexp[:5]}"

kdaify_llama(model, init_from_softmax=True)       # 手术:拷 q/k/v/o,gating 新初始化
attnres_ify_llama(model, groups=GROUPS)            # 跨层 AttnRes(zero-init no-op)

os.makedirs(OUT, exist_ok=True)
save_file(model.state_dict(), os.path.join(OUT, "model.safetensors"))
n = sum(p.numel() for p in model.parameters())
print(f"SURGERY_DONE saved {OUT}/model.safetensors | {n/1e6:.1f}M params | groups={GROUPS}")
