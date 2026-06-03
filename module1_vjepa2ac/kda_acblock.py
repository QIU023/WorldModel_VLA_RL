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

    def __init__(self, dim, norm_eps=1e-5, n_heads=None):
        super().__init__()
        import os
        n_heads = n_heads or int(os.environ.get("ATTNRES_HEADS", "4"))
        while dim % n_heads:                              # 保证整除
            n_heads -= 1
        self.n_heads, self.hd = n_heads, dim // n_heads
        self.norm = nn.RMSNorm(dim, eps=norm_eps)
        self.k = nn.Linear(dim, n_heads, bias=False)     # multi-head per-frame gate logit（v1 是 1）
        self.v = nn.Linear(dim, dim, bias=False)         # value 投影（v1 直接聚合 raw x）
        self.tau = nn.Parameter(torch.zeros(n_heads))    # per-head 可学温度（log scale）→ 尖锐帧选择
        self.o_proj = nn.Linear(dim, dim, bias=False)
        nn.init.zeros_(self.o_proj.weight)               # 起步为 no-op，稳定接入

    def forward(self, x):                                 # [N, T, D]
        # multi-head content-weighted causal online-softmax 聚合（v2，O(N) streamable）。
        N, T, D = x.shape; Hh, hd = self.n_heads, self.hd
        xn = self.norm(x)
        logit = self.k(xn) * self.tau.exp()               # [N, T, Hh] 温度缩放
        logit = logit - logit.max(dim=1, keepdim=True).values   # per-pos 常数移位（softmax 不变、不破因果）
        e = logit.exp().float()                           # [N, T, Hh]
        v = self.v(xn).view(N, T, Hh, hd).float()         # [N, T, Hh, hd]
        num = torch.cumsum(e.unsqueeze(-1) * v, dim=1)    # [N, T, Hh, hd]  O(N)
        den = torch.cumsum(e, dim=1).clamp_min(1e-6).unsqueeze(-1)   # [N, T, Hh, 1]
        agg = (num / den).reshape(N, T, D).to(x.dtype)
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
    """把 predictor 的每个 ACBlock 原地换成 DividedKDAACBlock（复用预训练权重 + fresh KDA）。
    注意：use_attnres=True 是早期**误用**（跨帧 AttnRes，与 KDA 冗余）；正确的 AttnRes 用
    attnres_ify_predictor（跨层）。这里默认 False。"""
    for i in range(len(pred.predictor_blocks)):
        pred.predictor_blocks[i] = DividedKDAACBlock.from_acblock(pred.predictor_blocks[i], use_attnres=use_attnres)
    return pred


def block_attn_res(blocks, partial, proj, norm):
    """真 Block-AttnRes（Kimi 2603.15031）—— 跨 block(深度) 的 softmax 注意力聚合。
    blocks: 已完成各 block 输出 list，每个 [B, Ntok, D]；partial: 当前 block [B, Ntok, D]。
    proj: Linear(D,1,bias=False) 的 weight = 该边界的 pseudo-query（zero-init→起步均匀）。
    """
    V = torch.stack(blocks + [partial], dim=0)          # [N+1, B, Ntok, D]  (values = 各 block 输出)
    K = norm(V)                                          # keys = RMSNorm(values)
    q = proj.weight.squeeze(0)                           # [D]  pseudo-query
    logits = torch.einsum("d,nbtd->nbt", q, K)           # [N+1, B, Ntok] over block(深度)轴
    w = torch.softmax(logits, dim=0)
    return torch.einsum("nbt,nbtd->btd", w, V)           # [B, Ntok, D]


def attnres_ify_predictor(pred, num_blocks=8):
    """正确的 Block-AttnRes：在 predictor 的 L 个 block 上、沿**深度轴**做跨层注意力残差。
    把 L 个 block 分成 num_blocks 组（组内标准残差），**组边界**用 block_attn_res 把"下一组输入"
    换成对所有已完成组输出的注意力加权。每个边界一个 zero-init pseudo-query + 共享 RMSNorm。
    与 KDA（跨帧）正交。monkeypatch pred.forward。"""
    import types
    L = len(pred.predictor_blocks)
    lpb = max(1, L // num_blocks)                        # layers_per_block
    n_commit = L // lpb                                  # 提交次数
    dim = pred.predictor_blocks[0].norm1.normalized_shape[0]
    dev = pred.predictor_norm.weight.device if hasattr(pred.predictor_norm, "weight") else next(pred.parameters()).device
    pred.ar_proj = nn.ModuleList([nn.Linear(dim, 1, bias=False) for _ in range(n_commit)]).to(dev)
    for p in pred.ar_proj:
        nn.init.zeros_(p.weight)                         # zero-init → softmax 起步均匀（稳定接入）
    pred.ar_norm = nn.RMSNorm(dim, eps=1e-5).to(dev)
    pred.ar_lpb = lpb

    def new_forward(self, x, actions, states, extrinsics=None):
        x = self.predictor_embed(x)
        B, N_ctxt, D = x.size()
        T = N_ctxt // (self.grid_height * self.grid_width)
        s = self.state_encoder(states).unsqueeze(2)
        a = self.action_encoder(actions).unsqueeze(2)
        x = x.view(B, T, self.grid_height * self.grid_width, D)
        cond = 3 if self.use_extrinsics else 2
        if self.use_extrinsics:
            e = self.extrinsics_encoder(extrinsics).unsqueeze(2)
            x = torch.cat([a, s, e, x], dim=2).flatten(1, 2)
        else:
            x = torch.cat([a, s, x], dim=2).flatten(1, 2)
        attn_mask = self.attn_mask[: x.size(1), : x.size(1)].to(x.device, non_blocking=True)
        blocks, g = [], 0
        for i, blk in enumerate(self.predictor_blocks):
            x = blk(x, mask=None, attn_mask=attn_mask, T=T, H=self.grid_height, W=self.grid_width, action_tokens=cond)
            if (i + 1) % self.ar_lpb == 0:               # 组边界：提交 + 跨层 AttnRes
                blocks.append(x)
                if i < len(self.predictor_blocks) - 1:
                    x = block_attn_res(blocks[:-1], blocks[-1], self.ar_proj[g], self.ar_norm)
                    g += 1
        x = x.view(B, T, cond + self.grid_height * self.grid_width, D)
        x = x[:, :, cond:, :].flatten(1, 2)
        x = self.predictor_norm(x)
        x = self.predictor_proj(x)
        return x

    pred.forward = types.MethodType(new_forward, pred)
    return pred
