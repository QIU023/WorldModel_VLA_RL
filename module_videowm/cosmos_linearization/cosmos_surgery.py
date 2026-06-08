"""Cosmos-Predict2 (CosmosTransformer3DModel) full linearization surgery = Hedgehog(attn1) + LoRA + AttnRes,
+ LoLCATs attention-transfer distillation loss. This is the diffusers-DiT ADAPTER; the reusable operators
(HedgehogFeatureMap, block_attn_res/_AttnResCoord) are the basis of a future unified `linattn/` lib.

Pipeline (LoLCATs 2-stage):
  surgery: cosmosify_hedgehog (φ_q/φ_k on 75% attn1) + add_lora_cosmos (LoRA q/k/v/out) + attnres_ify_cosmos
  stage-1: attn_transfer_loss — MSE(student hedgehog attn1 out, softmax teacher attn1 out) on shared input
  stage-2: + LoRA finetune end-to-end (diffusion loss)
"""
import math
import torch
import torch.nn as nn
from cosmos_hedgehog import (HedgehogFeatureMap, HedgehogCosmosProcessor,
                             cosmosify_hedgehog, pick_softmax_layers, get_dit_blocks, get_self_attn)


# ---------------------------------------------------------------- LoRA (reuse the chameleon design)
class LoRALinear(nn.Module):
    """Frozen base + low-rank adapter: y = base(x) + B(A(x))*scale. lora_B zero-init => starts identity."""
    def __init__(self, base, r=32, alpha=64):
        super().__init__()
        self.base = base
        self.base.weight.requires_grad_(False)
        if getattr(base, "bias", None) is not None:
            self.base.bias.requires_grad_(False)
        self.lora_A = nn.Linear(base.in_features, r, bias=False)
        self.lora_B = nn.Linear(r, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
        self.lora_A = self.lora_A.to(dtype=base.weight.dtype, device=base.weight.device)  # match base (bf16)
        self.lora_B = self.lora_B.to(dtype=base.weight.dtype, device=base.weight.device)
        self.scale = alpha / r

    def forward(self, x):
        return self.base(x) + self.lora_B(self.lora_A(x)) * self.scale


def add_lora_cosmos(model, linear_layers, r=32, alpha=64):
    """LoRA on attn1 q/k/v/out[0] of the linearized layers (LoLCATs stage-2 flexibility)."""
    blocks, _ = get_dit_blocks(model)
    n = 0
    for i in linear_layers:
        a, _ = get_self_attn(blocks[i])
        a.to_q = LoRALinear(a.to_q, r, alpha); a.to_k = LoRALinear(a.to_k, r, alpha)
        a.to_v = LoRALinear(a.to_v, r, alpha); a.to_out[0] = LoRALinear(a.to_out[0], r, alpha)
        n += 4
    print(f"[lora] wrapped {n} attn1 q/k/v/out Linears (r={r}) on {len(linear_layers)} layers")
    return model


# ---------------------------------------------------------------- AttnRes (port from kda_llama, DiT-generic)
def block_attn_res(blocks, partial, proj, norm):
    """Cross-layer (depth) Block-AttnRes (Kimi 2603.15031): learned-pseudo-query softmax aggregation over
    committed block outputs + current partial. zero-init proj -> starts uniform (stable no-op)."""
    V = torch.stack(blocks + [partial], dim=0)            # [N+1,B,T,D]
    K = norm(V)
    q = proj.weight.squeeze(0)                            # [D]
    w = torch.softmax(torch.einsum("d,nbtd->nbt", q, K), dim=0)
    return torch.einsum("nbt,nbtd->btd", w, V)


class _AttnResCoord(nn.Module):
    def __init__(self, dim, n_commit):
        super().__init__()
        self.proj = nn.ModuleList([nn.Linear(dim, 1, bias=False) for _ in range(max(1, n_commit))])
        for p in self.proj:
            nn.init.zeros_(p.weight)
        self.norm = nn.RMSNorm(dim)
        self.committed = []

    def reset(self):
        self.committed = []


class _CosmosBlockAttnRes(nn.Module):
    """Wrap a CosmosTransformerBlock: commit output at group boundaries; replace next-group input with the
    AttnRes aggregation. Passthrough forward (DiT block returns a tensor)."""
    def __init__(self, inner, coord, idx, lpb, n_layers):
        super().__init__()
        self.inner = inner
        object.__setattr__(self, "coord", coord)   # ref only, not a submodule (avoid shared-param save error)
        self.idx, self.lpb = idx, lpb
        self.is_boundary = (idx + 1) % lpb == 0
        self.is_last = idx == n_layers - 1

    def forward(self, hidden_states, *args, **kwargs):
        if self.idx == 0:
            self.coord.reset()
        out = self.inner(hidden_states, *args, **kwargs)
        hs = out[0] if isinstance(out, tuple) else out
        if self.is_boundary:
            self.coord.committed.append(hs)
            if not self.is_last:
                g = len(self.coord.committed) - 1
                hs = block_attn_res(self.coord.committed[:-1], self.coord.committed[-1],
                                    self.coord.proj[g], self.coord.norm)
            return (hs,) + out[1:] if isinstance(out, tuple) else hs
        return out


def attnres_ify_cosmos(model, groups=7):
    """Cross-layer AttnRes over the 28 DiT blocks (depth axis), orthogonal to Hedgehog (token axis)."""
    blocks, battr = get_dit_blocks(model)
    L = len(blocks)
    lpb = max(1, L // groups)
    n_agg = sum(1 for i in range(L) if (i + 1) % lpb == 0 and i != L - 1)
    dim = model.config.num_attention_heads * model.config.attention_head_dim
    dev, dt = next(model.parameters()).device, next(model.parameters()).dtype
    coord = _AttnResCoord(dim, n_agg).to(dev, dt)   # model dtype (bf16); matches hidden states
    model.attnres_coord = coord
    setattr(model, battr, nn.ModuleList([_CosmosBlockAttnRes(b, coord, i, lpb, L)
                                         for i, b in enumerate(blocks)]))
    print(f"[attnres] {L} blocks, lpb={lpb}, {n_agg} cross-layer AttnRes points")
    return model


# ---------------------------------------------------------------- full surgery + freeze
def linearize_cosmos(model, ratio_linear=0.75, lora_r=32, attnres_groups=7, feat_dim=None):
    """Hedgehog(attn1) + LoRA(q/k/v/out) + AttnRes. Freezes base; trainable = feat_map + lora + attnres."""
    model, linear_layers, softmax_layers = cosmosify_hedgehog(model, ratio_linear, feat_dim)
    add_lora_cosmos(model, linear_layers, r=lora_r, alpha=2 * lora_r)
    attnres_ify_cosmos(model, groups=attnres_groups)
    # freeze everything except feat_map / lora / attnres
    n_tr = n_fr = 0
    for name, p in model.named_parameters():
        train = ("feat_map" in name) or ("lora" in name.lower()) or ("attnres_coord" in name)
        p.requires_grad_(train)
        n_tr += p.numel() if train else 0; n_fr += 0 if train else p.numel()
    print(f"[linearize] trainable {n_tr/1e6:.1f}M / frozen {n_fr/1e9:.2f}B "
          f"({100*n_tr/(n_tr+n_fr):.2f}% trainable)")
    return model, linear_layers, softmax_layers


# ---------------------------------------------------------------- LoLCATs stage-1 attn-transfer loss
def attn_transfer_loss(student_model, linear_layers, fwd_kwargs):
    """MSE(student hedgehog attn1 out, softmax-teacher attn1 out) on the SAME input, per linearized layer.
    Teacher = same module with CosmosAttnProcessor2_0 (no_grad); student = HedgehogProcessor (grad).
    Toggling the processor per layer gives both outputs from identical inputs/weights."""
    from diffusers.models.transformers.transformer_cosmos import CosmosAttnProcessor2_0
    attns = [student_model.transformer_blocks[i].inner.attn1 if hasattr(student_model.transformer_blocks[i], "inner")
             else student_model.transformer_blocks[i].attn1 for i in linear_layers]
    cap = {}
    def mk_hook(j, store):
        def h(mod, inp, out): store[j] = out
        return h

    # teacher pass (softmax), no grad
    teach = {}
    saved = [a.processor for a in attns]
    for a in attns:
        a.set_processor(CosmosAttnProcessor2_0())
    hooks = [a.register_forward_hook(mk_hook(j, teach)) for j, a in enumerate(attns)]
    with torch.no_grad():
        student_model(**fwd_kwargs, return_dict=False)
    for hk in hooks: hk.remove()
    for a, p in zip(attns, saved): a.set_processor(p)   # restore hedgehog

    # student pass (hedgehog), with grad
    stud = {}
    hooks = [a.register_forward_hook(mk_hook(j, stud)) for j, a in enumerate(attns)]
    student_model(**fwd_kwargs, return_dict=False)
    for hk in hooks: hk.remove()

    loss = sum(torch.nn.functional.mse_loss(stud[j].float(), teach[j].float()) for j in range(len(attns))) / len(attns)
    return loss


# ---------------------------------------------------------------- smoke
if __name__ == "__main__":
    from diffusers import CosmosTransformer3DModel
    torch.manual_seed(0)
    m = CosmosTransformer3DModel(num_layers=8, num_attention_heads=4, attention_head_dim=32,
                                 text_embed_dim=1024, max_size=(16, 16, 16), patch_size=(1, 2, 2),
                                 concat_padding_mask=False)
    fwd = dict(hidden_states=torch.randn(1, 16, 2, 16, 16), timestep=torch.tensor([500.0]),
               encoder_hidden_states=torch.randn(1, 16, 1024))
    with torch.no_grad():
        o0 = m(**fwd, return_dict=False)[0]
    m, lin, sm = linearize_cosmos(m, ratio_linear=0.75, lora_r=8, attnres_groups=4)
    with torch.no_grad():
        o1 = m(**fwd, return_dict=False)[0]
    print(f"[smoke] forward after full surgery OK, shape {tuple(o1.shape)}, |Δ|={abs((o1-o0)).mean():.4f}")
    # attn-transfer loss + backward
    loss = attn_transfer_loss(m, lin, fwd)
    loss.backward()
    g_feat = sum(p.grad.abs().sum().item() for n,p in m.named_parameters() if "feat_map" in n and p.grad is not None)
    g_lora = sum(p.grad.abs().sum().item() for n,p in m.named_parameters() if "lora" in n.lower() and p.grad is not None)
    print(f"[smoke] attn_transfer_loss={loss.item():.4f}  grad(feat_map)={g_feat:.3f} grad(lora)={g_lora:.3f}")
    tr = [n for n,p in m.named_parameters() if p.requires_grad]
    print(f"[smoke] trainable groups: feat_map={sum('feat_map' in n for n in tr)} lora={sum('lora' in n.lower() for n in tr)} attnres={sum('attnres' in n for n in tr)}")
    assert g_feat > 0, "feat_map must get attn-transfer gradient"
    print("=== SMOKE PASS: full surgery forwards + attn-transfer loss flows to feat_map+lora ===")
