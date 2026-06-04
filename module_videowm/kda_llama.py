"""KDA(Kimi Delta Attention,门控 delta-rule 线性注意力)drop-in 替换 HF LlamaAttention,
给 iVideoGPT 的视频 token AR transformer 用。

为什么这里用真 fla 快核(不像 dino_wm 的纯-torch):iVideoGPT 是**扁平因果 token 序列**
(256 ctx + 16/帧),长 rollout 时序列变长(T=100 帧 → ~1840 tokens,T=1000 → ~16k)→
O(N²) softmax 真爆 → KDA O(N) 的 chunk_kda 快核正是为长因果序列设计的。

复用 module1_vjepa2ac/kda_acblock.py 里验证过的 fla-KDA 配方:
  q/k/v + per-head beta + 低秩 forget-gate(fused_kda_gate)+ qk L2norm + output gate + chunk_kda。
接口匹配 transformers 4.46 LlamaAttention.forward;KDA 因果自带、无需 mask/RoPE(纯-KDA 层)。

env:libero venv(torch2.12 + fla0.5.0 + KDA kernels)。
"""
import torch
import torch.nn as nn
from einops import rearrange
from fla.ops.kda import chunk_kda, fused_recurrent_kda
from fla.ops.kda.gate import fused_kda_gate


class KDAttention(nn.Module):
    """门控 delta-rule 线性注意力,drop-in 替换 LlamaAttention。
    forward 签名兼容 transformers 4.46:(hidden_states, attention_mask, position_ids,
    past_key_value, output_attentions, use_cache, cache_position, position_embeddings, **kw)。
    KDA 因果自带 → 忽略 attention_mask/RoPE(position_embeddings)。
    use_cache 走 fused_recurrent_kda 携带固定大小 recurrent state(O(1)/步,长 rollout 的关键)。"""

    def __init__(self, config, layer_idx=None):
        super().__init__()
        self.layer_idx = layer_idx
        dim = config.hidden_size
        nh = config.num_attention_heads
        while dim % nh:
            nh -= 1
        self.num_heads = nh
        self.head_dim = hd = dim // nh
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.b_proj = nn.Linear(dim, nh, bias=False)              # per-head beta(delta 强度)
        self.f_a = nn.Linear(dim, hd, bias=False)                 # 低秩 forget-gate
        self.f_b = nn.Linear(hd, dim, bias=False)
        self.g_a = nn.Linear(dim, hd, bias=False)                 # 低秩 output-gate
        self.g_b = nn.Linear(hd, dim, bias=False)
        self.A_log = nn.Parameter(torch.log(torch.empty(nh).uniform_(1, 16)))
        self.dt_bias = nn.Parameter(torch.zeros(dim))
        self.o_norm = nn.RMSNorm(hd)
        self.o_proj = nn.Linear(dim, dim, bias=False)

    def forward(self, hidden_states, attention_mask=None, position_ids=None,
                past_key_value=None, output_attentions=False, use_cache=False,
                cache_position=None, position_embeddings=None, **kwargs):
        x = hidden_states
        B, T, D = x.shape
        H = self.num_heads
        q = rearrange(self.q_proj(x), "b t (h d) -> b t h d", h=H).to(torch.bfloat16)
        k = rearrange(self.k_proj(x), "b t (h d) -> b t h d", h=H).to(torch.bfloat16)
        v = rearrange(self.v_proj(x), "b t (h d) -> b t h d", h=H).to(torch.bfloat16)
        g_raw = rearrange(self.f_b(self.f_a(x)), "b t (h d) -> b t h d", h=H).float()
        g = fused_kda_gate(g_raw, self.A_log.float(), dt_bias=self.dt_bias.float())
        beta = self.b_proj(x).float().sigmoid()

        # past_key_value: 用 dict 存 KDA recurrent state(非标准 HF Cache，自管理）
        init_state = None
        if use_cache and isinstance(past_key_value, dict):
            init_state = past_key_value.get(self.layer_idx)
        recurrent = (init_state is not None) or (T <= 64)
        fn = fused_recurrent_kda if recurrent else chunk_kda
        o, final = fn(q=q, k=k, v=v, g=g, beta=beta, initial_state=init_state,
                      output_final_state=use_cache, use_qk_l2norm_in_kernel=True, cu_seqlens=None)
        if use_cache and isinstance(past_key_value, dict):
            past_key_value[self.layer_idx] = final

        g_out = rearrange(self.g_b(self.g_a(x)), "b t (h d) -> b t h d", h=H).float()
        o = self.o_norm(o.float()) * torch.sigmoid(g_out)
        o = rearrange(o, "b t h d -> b t (h d)").to(x.dtype)
        return self.o_proj(o), None, past_key_value


def block_attn_res(blocks, partial, proj, norm):
    """跨层(深度轴)Block-AttnRes(Kimi 2603.15031)—— 对所有已完成 block 输出 + 当前 partial
    做 learned pseudo-query 的 softmax 聚合。zero-init proj → 起步均匀(no-op)稳定接入。
    blocks/partial: 每个 [B, T, D]。"""
    V = torch.stack(blocks + [partial], dim=0)            # [N+1, B, T, D]
    K = norm(V)
    q = proj.weight.squeeze(0)                            # [D]
    logits = torch.einsum("d,nbtd->nbt", q, K)
    w = torch.softmax(logits, dim=0)
    return torch.einsum("nbt,nbtd->btd", w, V)


class _AttnResCoord(nn.Module):
    """跨层 AttnRes 协调器:持 zero-init pseudo-query(每组边界一个)+ 共享 RMSNorm。
    committed 是 runtime list(每次 forward 由 layer0 清空)。"""

    def __init__(self, dim, n_commit):
        super().__init__()
        self.proj = nn.ModuleList([nn.Linear(dim, 1, bias=False) for _ in range(max(1, n_commit))])
        for p in self.proj:
            nn.init.zeros_(p.weight)
        self.norm = nn.RMSNorm(dim)
        self.committed = []

    def reset(self):
        self.committed = []


class _LayerAttnRes(nn.Module):
    """包一个 decoder layer:组边界(每 lpb 层)提交输出,并把"下一组输入"换成对已提交组的
    AttnRes 聚合。forward 签名透传(兼容 HF layer loop)。layer0 负责每次 forward 清空 committed。"""

    def __init__(self, inner, coord, idx, lpb, n_layers):
        super().__init__()
        self.inner = inner
        # coord 只在 base.attnres_coord 注册一次;这里用 object.__setattr__ 存引用，
        # 不让 nn.Module 把它注册成本层子模块(否则参数在多层名字下共享 → safetensors 拒存)。
        object.__setattr__(self, 'coord', coord)
        self.idx = idx
        self.lpb = lpb
        self.is_boundary = (idx + 1) % lpb == 0
        self.is_last = idx == n_layers - 1

    def forward(self, hidden_states, *args, **kwargs):
        if self.idx == 0:
            self.coord.reset()
        out = self.inner(hidden_states, *args, **kwargs)
        hs = out[0] if isinstance(out, tuple) else out
        if self.is_boundary:
            self.coord.committed.append(hs)
            if not self.is_last:                          # 组边界:下一组输入 = AttnRes 聚合
                g = len(self.coord.committed) - 1         # 第几个边界
                hs = block_attn_res(self.coord.committed[:-1], self.coord.committed[-1],
                                    self.coord.proj[g], self.coord.norm)
            if isinstance(out, tuple):
                return (hs,) + out[1:]
            return hs
        return out


def attnres_ify_llama(model, groups=4):
    """跨层 Block-AttnRes:把 L 层分成 groups 组,组边界用 block_attn_res 聚合已完成组输出。
    与 KDA(时间轴)正交。先 kdaify_llama 再 attnres_ify_llama。"""
    base = model.model if hasattr(model, "model") else model
    dim = base.config.hidden_size
    L = len(base.layers)
    lpb = max(1, L // groups)
    n_agg = sum(1 for i in range(L) if (i + 1) % lpb == 0 and i != L - 1)  # 实际聚合边界数(避免 DDP 未用参数)
    dev = next(base.parameters()).device
    dtype = next(base.parameters()).dtype
    coord = _AttnResCoord(dim, n_agg).to(dev, dtype)
    base.attnres_coord = coord                            # 挂住,随模型保存/移动
    base.layers = nn.ModuleList([_LayerAttnRes(l, coord, i, lpb, L) for i, l in enumerate(base.layers)])
    print(f"attnres_ify_llama: {L} layers, lpb={lpb}, {n_agg} cross-layer AttnRes aggregation points")
    return model


def kdaify_llama(model, layers=None, init_from_softmax=False):
    """把 LlamaModel 每个 decoder layer 的 self_attn 原地换成 KDAttention。
    layers: None=全替(纯-KDA);或传 set/list 只替部分层(hybrid)。
    init_from_softmax=True: **attention surgery** —— 把预训练 softmax attn 的 q/k/v/o 权重拷进
      新 KDAttention(同 dim、无 GQA 时直接拷),gating(beta/forget/output)新初始化 → 继承预训练知识,
      只需短 adapt(Attention Surgery 2509.24899 / LinVideo 2510.08318 范式),不从零。
    model 可以是 LlamaForCausalLM(取 .model)或 LlamaModel。"""
    base = model.model if hasattr(model, "model") else model
    config = base.config
    n = len(base.layers)
    target = set(range(n)) if layers is None else set(layers)
    copied = 0
    for i, layer in enumerate(base.layers):
        if i in target:
            old = layer.self_attn
            dev = next(old.parameters()).device
            dtype = next(old.parameters()).dtype
            new = KDAttention(config, layer_idx=i).to(dev, dtype)
            if init_from_softmax:                         # 手术:拷 q/k/v/o
                for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
                    if hasattr(old, proj) and getattr(old, proj).weight.shape == getattr(new, proj).weight.shape:
                        getattr(new, proj).load_state_dict(getattr(old, proj).state_dict())
                        copied += 1
            layer.self_attn = new
    # 关键:KDA 的 recurrent state 与 HF DynamicCache 不兼容 —— HF generate 默认 use_cache=True
    # 传增量 token,KDA 会丢上下文 → 生成乱码。强制 use_cache=False:生成时每步喂全序列、
    # KDA 走 chunk_kda 全序列因果(正确,O(N²) decode 慢但对)。
    # TODO(decode 效率):写 KDA 专用 Cache 存 recurrent state → O(1)/步(那才是 KDA 的 decode 卖点)。
    config.use_cache = False
    if hasattr(model, "config"):
        model.config.use_cache = False
    if hasattr(model, "generation_config") and model.generation_config is not None:
        model.generation_config.use_cache = False
    print(f"kdaify_llama: replaced {len(target)}/{n} attention layers with KDA"
          f"{f' (surgery: copied {copied} q/k/v/o projections)' if init_from_softmax else ''}; use_cache=False (KDA full-seq decode)")
    return model
