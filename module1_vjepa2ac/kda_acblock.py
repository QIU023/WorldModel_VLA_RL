"""E1.1 — Divided KDA-ACBlock：把 V-JEPA2-AC predictor 的 block 改成
  空间(帧内 full attn, 复用预训练 ACRoPEAttention) ⊕ 时间(跨帧 KDA 线性递归)。

监 patch 思路：kda_ify_predictor(pred) 把每个 ACBlock 原地替换为 DividedKDAACBlock，
复用其 norm1/attn/norm2/mlp（预训练权重），新增 norm_t + KDATemporal（fresh）。
predictor.forward 不变（它仍按 [B, T*P, D] 调 block，P=action_tokens+H*W）。

KDA 用法照搬 sibling KimiDeltaAttention：fused_kda_gate(g_raw,A_log,dt_bias)+chunk_kda(use_qk_l2norm)。
简化：v1 不含 ShortConvolution（留作 parity tuning 的 TODO）。
环境：py3.10 (torch2.11 + fla0.5.0)。
"""
import torch
import torch.nn as nn
from einops import rearrange
from fla.ops.kda import chunk_kda, fused_recurrent_kda
from fla.ops.kda.gate import fused_kda_gate


class KDATemporal(nn.Module):
    """跨帧 KDA 线性注意力。输入 [N, T, D]（N=B*P，每个空间位置一条长度 T 序列）→ [N, T, D]。
    返回 (out, final_state)；传 initial_state 走单步递归（rollout）。"""

    def __init__(self, dim, num_heads):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = hd = dim // num_heads
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.b_proj = nn.Linear(dim, num_heads, bias=False)          # beta per head
        self.f_a = nn.Linear(dim, hd, bias=False)                    # 低秩 forget-gate
        self.f_b = nn.Linear(hd, dim, bias=False)
        self.g_a = nn.Linear(dim, hd, bias=False)                    # 低秩 output-gate
        self.g_b = nn.Linear(hd, dim, bias=False)
        self.A_log = nn.Parameter(torch.log(torch.empty(num_heads).uniform_(1, 16)))
        self.dt_bias = nn.Parameter(torch.zeros(dim))
        self.o_norm = nn.RMSNorm(hd)
        self.o_proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x, initial_state=None, output_final_state=False):
        N, T, D = x.shape
        H = self.num_heads
        q = rearrange(self.q_proj(x), "n t (h d) -> n t h d", h=H).to(torch.bfloat16)
        k = rearrange(self.k_proj(x), "n t (h d) -> n t h d", h=H).to(torch.bfloat16)
        v = rearrange(self.v_proj(x), "n t (h d) -> n t h d", h=H).to(torch.bfloat16)
        g_raw = rearrange(self.f_b(self.f_a(x)), "n t (h d) -> n t h d", h=H).float()
        g = fused_kda_gate(g_raw, self.A_log.float(), dt_bias=self.dt_bias.float())
        beta = self.b_proj(x).float().sigmoid()
        use_recurrent = (initial_state is not None) or (T <= 64)
        fn = fused_recurrent_kda if use_recurrent else chunk_kda
        o, final = fn(q=q, k=k, v=v, g=g, beta=beta,
                      initial_state=initial_state, output_final_state=output_final_state,
                      use_qk_l2norm_in_kernel=True, cu_seqlens=None)
        g_out = rearrange(self.g_b(self.g_a(x)), "n t (h d) -> n t h d", h=H).float()
        o = self.o_norm(o.float()) * torch.sigmoid(g_out)
        o = rearrange(o, "n t h d -> n t (h d)").to(x.dtype)
        return self.o_proj(o), final


class CrossFrameAttnRes(nn.Module):
    """跨帧 Block-AttnRes（Kimi AttnRes §5 的 units=帧 版本，arxiv 2603.15031）。
    对每个空间位置的帧序列，用学习式 pseudo-query w_l 做 **causal online-softmax 聚合**
    （= cumulative-softmax，O(N) streamable）。输入 [N,T,D]（N=B*P）→ [N,T,D]。
    novel 点：把 AttnRes 当世界模型的"时间聚合原语"，与 KDA 递归互补。
    """

    def __init__(self, dim, norm_eps=1e-5):
        super().__init__()
        self.norm = nn.RMSNorm(dim, eps=norm_eps)
        self.proj = nn.Linear(dim, 1, bias=False)        # pseudo-query w_l；zero-init→初始均匀
        self.o_proj = nn.Linear(dim, dim, bias=False)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.o_proj.weight)               # 起步为 no-op，稳定接入

    def forward(self, x):                                 # [N, T, D]
        # causal softmax-weighted 帧聚合 == cumulative-softmax（O(N) streamable）。
        # 与 O(T²) causal-mask 形数学等价（已训 ckpt 直接适用）。
        logit = self.proj(self.norm(x)).squeeze(-1)       # [N, T] 每帧标量 gate
        logit = logit - logit.max(dim=1, keepdim=True).values   # per-pos 常数移位（稳定；常数在 softmax 中抵消，不破因果）
        e = logit.exp().unsqueeze(-1)                     # [N, T, 1] ≤1
        num = torch.cumsum(e * x.float(), dim=1)          # [N, T, D]  O(N)
        den = torch.cumsum(e, dim=1).clamp_min(1e-6)      # [N, T, 1]
        agg = (num / den).to(x.dtype)
        return self.o_proj(agg)


class DividedKDAACBlock(nn.Module):
    """复用预训练 ACBlock 的 norm1/attn/norm2/mlp/drop_path；插入 norm_t + KDATemporal
    （可选 ⊕ 跨帧 CrossFrameAttnRes）。forward 签名与 ACBlock 一致（predictor 不感知差异）。"""

    def __init__(self, dim, num_heads, norm1, attn, norm2, mlp, drop_path, norm_layer=nn.LayerNorm, use_attnres=False):
        super().__init__()
        self.norm1, self.attn, self.norm2, self.mlp, self.drop_path = norm1, attn, norm2, mlp, drop_path
        self.norm_t = norm_layer(dim)
        self.kda = KDATemporal(dim, num_heads)
        self.use_attnres = use_attnres
        if use_attnres:
            self.norm_ar = norm_layer(dim)
            self.attnres = CrossFrameAttnRes(dim)

    @classmethod
    def from_acblock(cls, blk, use_attnres=False):
        dim = blk.norm1.normalized_shape[0]
        nh = blk.attn.num_heads
        return cls(dim, nh, blk.norm1, blk.attn, blk.norm2, blk.mlp, blk.drop_path, use_attnres=use_attnres)

    def forward(self, x, mask=None, attn_mask=None, T=None, H=None, W=None, action_tokens=0):
        B, N, D = x.shape
        P = action_tokens + H * W                       # tokens/帧
        # ① 空间：帧内 full attn（复用 ACRoPEAttention，按帧 T=1 → 帧内、忽略 attn_mask）
        y = self.norm1(x).view(B * T, P, D)
        y = self.attn(y, mask=None, attn_mask=None, T=1, H=H, W=W, action_tokens=action_tokens)
        x = x + self.drop_path(y.view(B, N, D))
        # ② 时间：每个空间位置跨帧 KDA 递归
        y = self.norm_t(x).view(B, T, P, D).permute(0, 2, 1, 3).reshape(B * P, T, D)
        o, _ = self.kda(y)
        o = o.reshape(B, P, T, D).permute(0, 2, 1, 3).reshape(B, N, D)
        x = x + self.drop_path(o)
        # ②b 时间：跨帧 Block-AttnRes（units=帧，learned causal-softmax 聚合）⊕ KDA
        if self.use_attnres:
            y2 = self.norm_ar(x).view(B, T, P, D).permute(0, 2, 1, 3).reshape(B * P, T, D)
            o2 = self.attnres(y2).reshape(B, P, T, D).permute(0, 2, 1, 3).reshape(B, N, D)
            x = x + self.drop_path(o2)
        # ③ MLP
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

    def forward_step(self, x_frame, H, W, action_tokens, kda_state=None):
        """E1.5 单步：处理一个新帧 [B, P, D] + 携带的 KDA state → (out_frame, new_state)。
        时间轴走 fused_recurrent_kda（O(1)/步、state 固定大小）。"""
        B, P, D = x_frame.shape
        y = self.attn(self.norm1(x_frame), mask=None, attn_mask=None, T=1, H=H, W=W, action_tokens=action_tokens)
        x = x_frame + self.drop_path(y)
        y = self.norm_t(x).reshape(B * P, 1, D)
        o, new_state = self.kda(y, initial_state=kda_state, output_final_state=True)
        o = o.reshape(B, P, D)
        x = x + self.drop_path(o)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x, new_state


def kda_ify_predictor(pred, use_attnres=False):
    """把 predictor 的每个 ACBlock 原地换成 DividedKDAACBlock（复用预训练权重 + fresh KDA
    [+可选跨帧 Block-AttnRes]）。"""
    for i in range(len(pred.predictor_blocks)):
        pred.predictor_blocks[i] = DividedKDAACBlock.from_acblock(pred.predictor_blocks[i], use_attnres=use_attnres)
    return pred
