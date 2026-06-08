"""LoLCATs-style linearization of Cosmos-Predict2 (CosmosTransformer3DModel) — bidirectional Hedgehog
linear attention + cross-layer AttnRes, on the SELF-attention (attn1) of 75% of blocks. Cross-attention
(attn2, video->T5 text) and the kept-softmax blocks are untouched.

Cosmos attn1 is BIDIRECTIONAL self-attn (dispatch_attention_fn is_causal=False) over the full video token
sequence → causal KDA does NOT fit; we use Hedgehog (learnable softmax-mimicking feature maps φ_q,φ_k, ICLR24
2402.04347, the LoLCATs operator). Trained via attention-output transfer (MSE to the softmax teacher) + LoRA.

Feature maps stored AS submodules of the Attention (attn.feat_map_q/k) → in model.parameters()+state_dict
(trainable + saved); the processor reads them via `attn`.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class HedgehogFeatureMap(nn.Module):
    """Hedgehog learnable feature map: φ(x)=softmax([Wx, -Wx]) → positive, spiky, ~softmax-like.
    head_dim -> 2*feat_dim. Default feat_dim=head_dim."""
    def __init__(self, head_dim, feat_dim=None):
        super().__init__()
        feat_dim = feat_dim or head_dim
        self.proj = nn.Linear(head_dim, feat_dim, bias=False)
        # near-identity init so early-on φ is informative (not collapsed)
        nn.init.normal_(self.proj.weight, std=head_dim ** -0.5)

    def forward(self, x):  # x: [B,H,T,head_dim] -> [B,H,T,2*feat_dim]
        w = self.proj(x.float())
        return torch.softmax(torch.cat([w, -w], dim=-1), dim=-1).to(x.dtype)


class HedgehogCosmosProcessor(nn.Module):
    """Drop-in for CosmosAttnProcessor2_0 (self-attn only) but BIDIRECTIONAL LINEAR attention via φ_q/φ_k.
    Mirrors the original q/k/v -> reshape -> RMSNorm -> RoPE, then replaces softmax with linear:
        out = φ(q) @ (φ(k)^T @ v) / (φ(q) @ φ(k)^T 1)   (full sum over keys = bidirectional)."""
    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, image_rotary_emb=None):
        assert encoder_hidden_states is None, "Hedgehog processor is for SELF-attn (attn1) only"
        q = attn.to_q(hidden_states); k = attn.to_k(hidden_states); v = attn.to_v(hidden_states)
        q = q.unflatten(2, (attn.heads, -1)).transpose(1, 2)   # [B,H,T,hd]
        k = k.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        v = v.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        q = attn.norm_q(q); k = attn.norm_k(k)
        if image_rotary_emb is not None:
            from diffusers.models.embeddings import apply_rotary_emb
            q = apply_rotary_emb(q, image_rotary_emb, use_real=True, use_real_unbind_dim=-2)
            k = apply_rotary_emb(k, image_rotary_emb, use_real=True, use_real_unbind_dim=-2)
        # GQA (no-op when heads==kv_heads)
        if k.size(1) != q.size(1):
            k = k.repeat_interleave(q.size(1) // k.size(1), dim=1)
            v = v.repeat_interleave(q.size(1) // v.size(1), dim=1)
        # bidirectional linear attention with Hedgehog feature maps
        phi_q = attn.feat_map_q(q)        # [B,H,T,2F]
        phi_k = attn.feat_map_k(k)        # [B,H,T,2F]
        vf = v.float()
        kv = torch.einsum("bhtf,bhtd->bhfd", phi_k.float(), vf)        # [B,H,2F,hd]  sum over T (bidir)
        z = phi_k.float().sum(dim=2)                                    # [B,H,2F]
        num = torch.einsum("bhtf,bhfd->bhtd", phi_q.float(), kv)        # [B,H,T,hd]
        den = torch.einsum("bhtf,bhf->bht", phi_q.float(), z).unsqueeze(-1).clamp_min(1e-6)
        out = (num / den).to(q.dtype)                                   # [B,H,T,hd]
        out = out.transpose(1, 2).flatten(2, 3).type_as(q)             # [B,T,H*hd]
        out = attn.to_out[0](out); out = attn.to_out[1](out)
        return out


def get_dit_blocks(model):
    """Unified: return the transformer block ModuleList of a diffusers DiT (Cosmos2/2.5/Wan/CogVideoX...).
    Tries common attr names so the surgery isn't hardcoded to one model."""
    for attr in ("transformer_blocks", "blocks", "layers", "single_transformer_blocks"):
        b = getattr(model, attr, None)
        if b is not None and len(b) > 0:
            return b, attr
    raise AttributeError("no transformer block list found (tried transformer_blocks/blocks/layers)")


def get_self_attn(block):
    """Unified: return the SELF-attention module of a DiT block (has to_q/to_k/to_v; not the cross-attn).
    diffusers DiT blocks: self-attn is usually `attn1`, cross-attn `attn2`; some have a single `attn`."""
    for attr in ("attn1", "attn", "self_attn"):
        a = getattr(block, attr, None)
        if a is not None and hasattr(a, "to_q") and hasattr(a, "to_k") and hasattr(a, "to_v"):
            return a, attr
    raise AttributeError(f"no self-attn (to_q/k/v) found on block {type(block).__name__}")


def pick_softmax_layers(n_layers, ratio_linear=0.75):
    """Keep softmax on evenly-spaced layers (SANA-style hybrid anchors); linearize the rest.
    n_keep_softmax = round(n_layers*(1-ratio)). Returns (linear_set, softmax_set)."""
    n_keep = max(1, round(n_layers * (1.0 - ratio_linear)))
    # evenly spaced softmax anchors
    softmax = set(int(round(i * (n_layers - 1) / (n_keep - 1))) if n_keep > 1 else n_layers // 2
                  for i in range(n_keep))
    linear = [i for i in range(n_layers) if i not in softmax]
    return linear, sorted(softmax)


def cosmosify_hedgehog(model, ratio_linear=0.75, feat_dim=None):
    """Replace attn1 (self-attn) processor with HedgehogCosmosProcessor on `ratio_linear` of blocks;
    add feat_map_q/k submodules. Cross-attn (attn2) + softmax-anchor blocks untouched."""
    blocks, _ba = get_dit_blocks(model)
    n = len(blocks)
    linear_layers, softmax_layers = pick_softmax_layers(n, ratio_linear)
    dt = next(model.parameters()).dtype
    dev = next(model.parameters()).device
    for i in linear_layers:
        attn, _aa = get_self_attn(blocks[i])
        hd = attn.to_q.out_features // attn.heads
        # keep feat maps in fp32 (small; softmax stable; forward casts input to float -> output to x.dtype)
        attn.feat_map_q = HedgehogFeatureMap(hd, feat_dim).to(dev)
        attn.feat_map_k = HedgehogFeatureMap(hd, feat_dim).to(dev)
        attn.set_processor(HedgehogCosmosProcessor())
    print(f"[cosmosify] {len(linear_layers)}/{n} blocks Hedgehog-linearized (self-attn); "
          f"softmax kept @ {softmax_layers}; feat_dim={feat_dim or 'head_dim'}")
    return model, linear_layers, softmax_layers


# ----------------------------------------------------------------- smoke test (no ckpt, random weights)
if __name__ == "__main__":
    from diffusers import CosmosTransformer3DModel
    torch.manual_seed(0)
    print("=== build tiny CosmosTransformer3DModel ===")
    m = CosmosTransformer3DModel(num_layers=4, num_attention_heads=4, attention_head_dim=32,
                                 text_embed_dim=1024, max_size=(16, 16, 16), patch_size=(1, 2, 2),
                                 concat_padding_mask=False).eval()
    # tiny inputs: latent video [B,C,T,H,W], text embed, timestep
    B, C, T, H, W = 1, 16, 2, 16, 16
    hs = torch.randn(B, C, T, H, W)
    enc = torch.randn(B, 16, 1024)
    ts = torch.tensor([500.0])
    with torch.no_grad():
        out_sdpa = m(hidden_states=hs, timestep=ts, encoder_hidden_states=enc, return_dict=False)[0]
    print("[smoke] SDPA forward OK, out shape", tuple(out_sdpa.shape))
    m2, lin, sm = cosmosify_hedgehog(m, ratio_linear=0.75)
    with torch.no_grad():
        out_hh = m2(hidden_states=hs, timestep=ts, encoder_hidden_states=enc, return_dict=False)[0]
    print("[smoke] Hedgehog forward OK, out shape", tuple(out_hh.shape))
    print(f"[smoke] |out_hh - out_sdpa| mean = {(out_hh-out_sdpa).abs().mean().item():.4f} "
          f"(nonzero = surgery changed attention; trained later to match)")
    # trainable feat maps present?
    nfm = sum('feat_map' in n for n, _ in m2.named_parameters())
    print(f"[smoke] feat_map params registered: {nfm} (trainable, in state_dict)")
    assert out_hh.shape == out_sdpa.shape and nfm > 0
    print("=== SMOKE PASS: Hedgehog surgery forwards + feat maps registered ===")
