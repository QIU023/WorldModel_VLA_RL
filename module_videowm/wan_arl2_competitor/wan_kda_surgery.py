"""ARL2 competitor: linearize Causal-Forcing (Wan2.1-1.3B AR) with INTRA-frame softmax + INTER-frame KDA
(gated-delta) + naive cross-layer AttnRes. Drop-in replaces WanSelfAttention on 75% (23/30) of blocks.

Decomposition (ARL2 recipe, our operator = KDA not GDN):
  full-3D attn over F*H*W  ->  two branches on shared q,k,v:
    intra-frame: softmax over the H*W tokens within each frame (preserves spatial detail + dynamics).
                 3D RoPE applied; the F-phase cancels within a frame, so it = spatial-only RoPE.
    inter-frame: KDA causal gated-delta over the F frames at each spatial position (the O(F) win for long
                 video; fixed-size recurrent state). No RoPE (KDA gating encodes recency, like GDN).
  out = o_proj( intra_out + inter_out ).  Distilled to match the teacher's full-3D output (attn-transfer
  per-layer MSE -> output velocity MSE, ARL2's two stages = our Cosmos recipe).

Teacher q/k/v/o weights are inherited; KDA gate/beta + AttnRes are the new ~2% trainable params.
"""
import sys, math
sys.path.insert(0, "/home/wm-vla-rl/_ext/Causal-Forcing")
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from fla.ops.kda import chunk_kda
from fla.ops.kda.gate import fused_kda_gate
from wan.modules.model import WanSelfAttention, rope_apply
import wan.modules.model as _wmod
import wan.modules.attention as _watt


def _sdpa_flash_attention(q, k, v, q_lens=None, k_lens=None, dropout_p=0., softmax_scale=None,
                          q_scale=None, causal=False, window_size=(-1, -1), deterministic=False,
                          dtype=torch.bfloat16, version=None):
    """Drop-in for the repo's flash_attention using torch SDPA — on H200 SDPA auto-selects the FLASH kernel
    (~flash-attn speed + O(N) memory), so no flash-attn package / CUDA-version compile needed. q,k,v: [B,L,n,d]."""
    if q_scale is not None:
        q = q * q_scale
    qt, kt, vt = (x.transpose(1, 2).to(dtype) for x in (q, k, v))   # [B,n,L,d]
    o = F.scaled_dot_product_attention(qt, kt, vt, dropout_p=dropout_p, is_causal=causal, scale=softmax_scale)
    return o.transpose(1, 2).contiguous()                            # [B,L,n,d]


def patch_wan_flash_sdpa():
    """Replace the repo's flash_attention (needs the flash-attn package + matching CUDA) with the SDPA-flash
    version everywhere WanSelfAttention/attention use it."""
    _wmod.flash_attention = _sdpa_flash_attention
    _watt.flash_attention = _sdpa_flash_attention
    _watt.FLASH_ATTN_2_AVAILABLE = True   # let attention() take the (now-SDPA) flash path
    print("[patch] flash_attention -> torch SDPA (H200 flash backend; no flash-attn pkg)")


patch_wan_flash_sdpa()   # apply on import so the teacher WanSelfAttention runs


def _add_kda_params(m, dim, nh, hd):
    """Add KDA inter-frame gate/beta params (the new ~2% trainable) to a self-attn module m."""
    m.b_proj = nn.Linear(dim, nh, bias=False)              # per-head delta beta
    m.f_a = nn.Linear(dim, hd, bias=False); m.f_b = nn.Linear(hd, dim, bias=False)   # forget gate
    m.g_a = nn.Linear(dim, hd, bias=False); m.g_b = nn.Linear(hd, dim, bias=False)   # output gate
    m.A_log = nn.Parameter(torch.log(torch.empty(nh).uniform_(1, 16)))
    dt = torch.exp(torch.rand(dim) * (math.log(0.1) - math.log(1e-3)) + math.log(1e-3)).clamp(min=1e-4)
    m.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))   # mild-decay init ~ keeps memory ~ softmax
    nn.init.zeros_(m.f_b.weight)
    m.kda_o_norm = nn.RMSNorm(hd)


class FactorizedKDASelfAttention(WanSelfAttention):
    """BIDIRECTIONAL teacher (Cosmos/Wan bidir). Inherits q/k/v/o/norm from base; adds KDA params."""
    def __init__(self, base: WanSelfAttention):
        super().__init__(base.dim, base.num_heads, base.window_size, base.qk_norm, base.eps)
        self.load_state_dict(base.state_dict())
        _add_kda_params(self, self.dim, self.num_heads, self.head_dim)

    def forward(self, x, seq_lens, grid_sizes, freqs):
        return self.o(_factorized_attn(self, x, grid_sizes, freqs).flatten(2))


def _factorized_attn(attn, x, grid_sizes, freqs):
    """Shared compute for both bidir + causal variants: intra-frame softmax (bidir over H*W) + inter-frame
    causal KDA (over F). KDA is inherently causal, so this is the SAME for the AR teacher. Returns [B,L,n,d]
    (pre-o-proj)."""
    b, s, n, d = *x.shape[:2], attn.num_heads, attn.head_dim
    Fr, H, W = grid_sizes[0].tolist()
    q = attn.norm_q(attn.q(x)).view(b, s, n, d)
    k = attn.norm_k(attn.k(x)).view(b, s, n, d)
    v = attn.v(x).view(b, s, n, d)
    # intra-frame softmax (spatial): 3D-RoPE q,k; the F-phase cancels within a frame -> = spatial RoPE
    qr = rope_apply(q, grid_sizes, freqs); kr = rope_apply(k, grid_sizes, freqs)
    qs = rearrange(qr, "b (f hw) n d -> (b f) n hw d", f=Fr)
    ks = rearrange(kr, "b (f hw) n d -> (b f) n hw d", f=Fr)
    vs = rearrange(v,  "b (f hw) n d -> (b f) n hw d", f=Fr)
    o_intra = F.scaled_dot_product_attention(qs, ks, vs)
    o_intra = rearrange(o_intra, "(b f) n hw d -> b (f hw) n d", b=b)
    # inter-frame causal KDA (temporal): gated-delta over F at each spatial position
    qf = rearrange(q, "b (f hw) n d -> (b hw) f n d", f=Fr).to(torch.bfloat16)
    kf = rearrange(k, "b (f hw) n d -> (b hw) f n d", f=Fr).to(torch.bfloat16)
    vf = rearrange(v, "b (f hw) n d -> (b hw) f n d", f=Fr).to(torch.bfloat16)
    xf = rearrange(x, "b (f hw) c -> (b hw) f c", f=Fr)
    g_raw = rearrange(attn.f_b(attn.f_a(xf)), "p f (n d) -> p f n d", n=n).float()
    g = fused_kda_gate(g_raw, attn.A_log.float(), dt_bias=attn.dt_bias.float())
    beta = attn.b_proj(xf).float().sigmoid()
    o_t, _ = chunk_kda(q=qf, k=kf, v=vf, g=g, beta=beta, initial_state=None,
                       output_final_state=False, use_qk_l2norm_in_kernel=True, cu_seqlens=None)
    g_out = rearrange(attn.g_b(attn.g_a(xf)), "p f (n d) -> p f n d", n=n).float()
    o_t = attn.kda_o_norm(o_t.float()) * torch.sigmoid(g_out)
    o_inter = rearrange(o_t.to(x.dtype), "(b hw) f n d -> b (f hw) n d", b=b)
    return o_intra + o_inter


def kda_trainable_names():
    return ("b_proj", "f_a", "f_b", "g_a", "g_b", "A_log", "dt_bias", "kda_o_norm")


def is_kda_param(name):
    """True iff `name` is one of the added KDA gate/beta params. EXACT dotted-component match —
    NOT substring: "f_a" is a substring of "self_attn" (se|lf_a|ttn), so `"f_a" in name` would
    wrongly flag every q/k/v/o weight. Split on '.' and require an exact leaf-module match."""
    parts = set(name.split("."))
    return any(kk in parts for kk in kda_trainable_names())


def factorize_wan(model, ratio=0.75):
    """Replace self_attn with FactorizedKDASelfAttention on `ratio` of the 30 WanAttentionBlocks
    (evenly-spaced softmax anchors kept). Returns the list of linearized block indices."""
    blocks = model.blocks
    L = len(blocks)
    n_keep = max(1, round(L * (1 - ratio)))
    anchors = set(int(round(i * (L - 1) / (n_keep - 1))) if n_keep > 1 else L // 2 for i in range(n_keep))
    linear_idx = [i for i in range(L) if i not in anchors]
    dev = next(model.parameters()).device; dt = next(model.parameters()).dtype
    for i in linear_idx:
        blocks[i].self_attn = FactorizedKDASelfAttention(blocks[i].self_attn).to(dev, dt)
    print(f"[factorize] {len(linear_idx)}/{L} blocks -> intra-softmax+inter-KDA; softmax anchors @ {sorted(anchors)}")
    return linear_idx


# ---------------- CAUSAL variant (the real ARL2 competitor: Causal Forcing / CausalWanModel) ----------------
from wan.modules.causal_model import CausalWanSelfAttention


class CausalFactorizedKDASelfAttention(CausalWanSelfAttention):
    """CAUSAL teacher (Causal Forcing AR). intra-frame softmax (bidir over H*W) + inter-frame CAUSAL KDA over F.
    KDA is inherently causal -> matches frame-causality. Training path (kv_cache=None) uses the shared compute;
    streaming kv_cache (frame-by-frame gen) carries the KDA recurrent state per spatial position (TODO for eval)."""
    def __init__(self, base: CausalWanSelfAttention):
        super().__init__(base.dim, base.num_heads, base.local_attn_size, base.sink_size, base.qk_norm, base.eps)
        self.load_state_dict(base.state_dict())
        _add_kda_params(self, self.dim, self.num_heads, self.head_dim)

    def forward(self, x, seq_lens, grid_sizes, freqs, block_mask=None, kv_cache=None,
                current_start=0, cache_start=None):
        # block_mask is the teacher's frame-causal mask — not needed: intra is within-frame, inter-KDA is causal.
        # kv_cache (streaming gen) -> recurrent KDA state TODO; training (kv_cache=None) is the immediate path.
        return self.o(_factorized_attn(self, x, grid_sizes, freqs).flatten(2))


def factorize_causal_wan(model, ratio=0.75):
    """Replace self_attn with CausalFactorizedKDASelfAttention on `ratio` of the CausalWanModel blocks."""
    blocks = model.blocks
    L = len(blocks)
    n_keep = max(1, round(L * (1 - ratio)))
    anchors = set(int(round(i * (L - 1) / (n_keep - 1))) if n_keep > 1 else L // 2 for i in range(n_keep))
    linear_idx = [i for i in range(L) if i not in anchors]
    dev = next(model.parameters()).device; dt = next(model.parameters()).dtype
    for i in linear_idx:
        blocks[i].self_attn = CausalFactorizedKDASelfAttention(blocks[i].self_attn).to(dev, dt)
    print(f"[factorize-causal] {len(linear_idx)}/{L} blocks -> intra-softmax+inter-causalKDA; anchors @ {sorted(anchors)}")
    return linear_idx


if __name__ == "__main__":
    # smoke: build a tiny Wan-like self-attn, factorize, forward
    torch.manual_seed(0)
    dim, nh = 128, 4
    base = WanSelfAttention(dim, nh).cuda().to(torch.bfloat16)
    fac = FactorizedKDASelfAttention(base).cuda().to(torch.bfloat16)
    Fr, Hh, Ww = 4, 8, 8
    B, L = 1, Fr * Hh * Ww
    x = torch.randn(B, L, dim, device="cuda", dtype=torch.bfloat16)
    grid = torch.tensor([[Fr, Hh, Ww]], device="cuda")
    seq_lens = torch.tensor([L], device="cuda")
    from wan.modules.model import rope_params
    freqs = rope_params(1024, dim // nh).cuda()
    with torch.no_grad():
        o_base = base(x, seq_lens, grid, freqs)
        o_fac = fac(x, seq_lens, grid, freqs)
    print(f"[smoke] base {tuple(o_base.shape)}  factorized {tuple(o_fac.shape)}  |Δ|={(o_fac-o_base).abs().float().mean():.3f}")
    print("=== wan_kda_surgery SMOKE PASS ===")
