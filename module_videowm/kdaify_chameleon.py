"""KDA(时间轴)+ 跨层 AttnRes(深度轴)surgery for WorldVLA Chameleon-7B —— WAM 重实验。
复用 kda_llama 的 KDAttention + AttnRes(已验证)。Chameleon 适配点:
  - 权重-only 拷贝(Chameleon attention 可能带 bias;KDA 不需要 → 只拷 q/k/v/o weight)。
  - KDA 层丢弃训练好的 q_norm/k_norm(KDA 用自带 qk-l2norm 替代;8 个 full 层保留 q_norm/k_norm)。
  - head_dim=128 干净,无需 down-projection。
32 层 → Kimi-Linear 3:1 = 24 KDA + 8 full@{3,7,11,15,19,23,27,31}。
"""
import sys
import torch
sys.path.insert(0, "/home/wm-vla-rl/module_videowm")
from kda_llama import KDAttention, attnres_ify_llama


def kimi_kda_layers(n_layers):
    """Kimi-Linear 3:1:每第4层留 full(MLA 位),其余 KDA。返回 KDA 层 set。"""
    return set(i for i in range(n_layers) if (i + 1) % 4 != 0)


def kdaify_chameleon(model, layers=None, init_from_softmax=False):
    """把 Chameleon decoder layer 的 self_attn 原地换成 KDAttention。
    layers: None=全替;set/list=hybrid(传 kimi_kda_layers(32) 得 24 层)。
    init_from_softmax: attention surgery,拷预训练 q/k/v/o weight(继承 SOTA 知识,短 adapt)。"""
    base = model.model if hasattr(model, "model") else model
    config = base.config
    n = len(base.layers)
    target = set(range(n)) if layers is None else set(layers)
    copied = 0
    for i, layer in enumerate(base.layers):
        if i not in target:
            continue
        old = layer.self_attn
        dev = next(old.parameters()).device
        dtype = next(old.parameters()).dtype
        new = KDAttention(config, layer_idx=i).to(dev, dtype)
        if init_from_softmax:                              # 手术:拷 q/k/v/o weight(只 weight,跳 bias/q_norm)
            for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
                if hasattr(old, proj):
                    ow = getattr(old, proj).weight
                    nw = getattr(new, proj).weight
                    if ow.shape == nw.shape:
                        nw.data.copy_(ow.data.to(nw.dtype))
                        copied += 1
        layer.self_attn = new
    config.use_cache = False
    if hasattr(model, "config"):
        model.config.use_cache = False
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.use_cache = False
    print(f"kdaify_chameleon: replaced {len(target)}/{n} attn layers with KDA"
          f"{f' (surgery: copied {copied} q/k/v/o)' if init_from_softmax else ''}; "
          f"full-attn kept @ {sorted(set(range(n)) - target)}")
    return model


def attnres_ify_chameleon(model, groups=8):
    """跨层 Block-AttnRes(深度轴)—— 复用 llama 版(对 base.layers 通用)。32 层 groups=8 → lpb=4。"""
    return attnres_ify_llama(model, groups=groups)


import math


class LoRALinear(torch.nn.Module):
    """Wrap a frozen nn.Linear with a low-rank adapter: y = base(x) + (B(A(x)))*scale.
    LoLCATs/ARL2 attention-transfer needs q/k flexibility (gating alone can't match softmax outputs).
    base.weight is frozen; lora_A/lora_B are trainable (names contain 'lora' -> picked up by the
    solver's trainable filter). lora_B zero-init => starts as identity (no perturbation)."""
    def __init__(self, base, r=32, alpha=64):
        super().__init__()
        self.base = base
        self.base.weight.requires_grad_(False)
        din, dout = base.in_features, base.out_features
        self.lora_A = torch.nn.Linear(din, r, bias=False)
        self.lora_B = torch.nn.Linear(r, dout, bias=False)
        torch.nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        torch.nn.init.zeros_(self.lora_B.weight)
        self.scale = alpha / r

    def forward(self, x):
        return self.base(x) + self.lora_B(self.lora_A(x)) * self.scale


def lora_ify_kda(model, r=32, alpha=64):
    """Inject LoRA on q/k/v/o of every KDAttention module (the linearized layers). Run on ALL ranks
    (structure must match for FSDP); rank0's lora init is broadcast via sync_module_states."""
    n = 0
    for mod in model.modules():
        if isinstance(mod, KDAttention):
            for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
                if hasattr(mod, proj) and not isinstance(getattr(mod, proj), LoRALinear):
                    setattr(mod, proj, LoRALinear(getattr(mod, proj), r=r, alpha=alpha))
                    n += 1
    print(f"[lora_ify] wrapped {n} q/k/v/o projections with LoRA (r={r}, alpha={alpha})")
    return model
