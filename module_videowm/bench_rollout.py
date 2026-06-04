"""Rollout 效率 benchmark:自回归 decode 时 KDA(O(1) recurrent state)vs SDPA(KV cache O(N) 增长)。
测 N 步 rollout 的 decode 延迟 + 峰值显存 vs N。这是 KDA 的 decode-效率 superiority 轴
(Kimi-Linear:−75% KV、6× decode @1M)—— WM imagination rollout 越长,KDA 优势越大。

KDA decode:每步用 fused_recurrent_kda 携带固定 state(O(1)/步)。
SDPA decode:HF DynamicCache,KV 随步增长(O(N)/步,O(N²) 总)。
"""
import os, sys, time, json, torch
sys.path.insert(0, "/home/wm-vla-rl/module_videowm")
from transformers import LlamaConfig, LlamaForCausalLM
from transformers.cache_utils import DynamicCache
from kda_llama import KDAttention

DEV = "cuda"


@torch.no_grad()
def sdpa_rollout(model, ctx_len, steps):
    """SDPA KV-cache 自回归 decode：prefill ctx_len，再逐 token 生成 steps 个。"""
    ids = torch.randint(0, model.config.vocab_size, (1, ctx_len), device=DEV)
    cache = DynamicCache()
    torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); t0 = time.time()
    out = model(input_ids=ids, past_key_values=cache, use_cache=True)
    nxt = out.logits[:, -1:].argmax(-1)
    for _ in range(steps):
        out = model(input_ids=nxt, past_key_values=cache, use_cache=True)
        nxt = out.logits[:, -1:].argmax(-1)
    torch.cuda.synchronize()
    return time.time() - t0, torch.cuda.max_memory_allocated() / 1e9


@torch.no_grad()
def kda_rollout(model, ctx_len, steps):
    """KDA recurrent 自回归 decode：每层携带固定 state，每步 O(1)。手动跨层 forward。"""
    base = model.model
    emb = base.embed_tokens
    layers = base.layers
    norm = base.norm
    states = [None] * len(layers)
    ids = torch.randint(0, model.config.vocab_size, (1, ctx_len), device=DEV)
    torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); t0 = time.time()

    def step(tok_ids, init):  # 跑一遍所有层，KDA 层携带/更新 state
        h = emb(tok_ids)
        new_states = []
        for li, layer in enumerate(layers):
            attn = layer.self_attn
            res = h
            hs = layer.input_layernorm(h)
            if isinstance(attn, KDAttention):           # KDA：recurrent decode（O(1) state）
                from einops import rearrange
                from fla.ops.kda import fused_recurrent_kda
                from fla.ops.kda.gate import fused_kda_gate
                B, T, D = hs.shape; H = attn.num_heads
                q = rearrange(attn.q_proj(hs), "b t (h d) -> b t h d", h=H).to(torch.bfloat16)
                k = rearrange(attn.k_proj(hs), "b t (h d) -> b t h d", h=H).to(torch.bfloat16)
                v = rearrange(attn.v_proj(hs), "b t (h d) -> b t h d", h=H).to(torch.bfloat16)
                g_raw = rearrange(attn.f_b(attn.f_a(hs)), "b t (h d) -> b t h d", h=H).float()
                gate = fused_kda_gate(g_raw, attn.A_log.float(), dt_bias=attn.dt_bias.float())
                beta = attn.b_proj(hs).float().sigmoid()
                o, fin = fused_recurrent_kda(q=q, k=k, v=v, g=gate, beta=beta, initial_state=init[li],
                                             output_final_state=True, use_qk_l2norm_in_kernel=True, cu_seqlens=None)
                g_out = rearrange(attn.g_b(attn.g_a(hs)), "b t (h d) -> b t h d", h=H).float()
                o = attn.o_norm(o.float()) * torch.sigmoid(g_out)
                o = rearrange(o, "b t h d -> b t (h d)").to(hs.dtype)
                ao = attn.o_proj(o); new_states.append(fin)
            else:                                        # softmax 层：单步只看自己（近似；hybrid 里少数层）
                ao = attn(hs, position_embeddings=base.rotary_emb(hs, torch.arange(T, device=DEV)[None]))[0]
                new_states.append(init[li])
            h = res + ao
            h = h + layer.mlp(layer.post_attention_layernorm(h))
        return h, new_states

    h, states = step(ids, states)                        # prefill
    nxt = model.lm_head(norm(h))[:, -1:].argmax(-1)
    for _ in range(steps):
        h, states = step(nxt, states)
        nxt = model.lm_head(norm(h))[:, -1:].argmax(-1)
    torch.cuda.synchronize()
    return time.time() - t0, torch.cuda.max_memory_allocated() / 1e9


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--layers', type=int, default=24)
    ap.add_argument('--dim', type=int, default=1024)
    ap.add_argument('--heads', type=int, default=16)
    ap.add_argument('--kda_layers', type=int, default=2)   # hybrid:换几层为 KDA
    ap.add_argument('--ctx', type=int, default=256)
    args = ap.parse_args()
    cfg = LlamaConfig(vocab_size=8192, hidden_size=args.dim, intermediate_size=args.dim * 4,
                      num_hidden_layers=args.layers, num_attention_heads=args.heads,
                      num_key_value_heads=args.heads, max_position_embeddings=131072, attn_implementation='sdpa')
    res = {"config": vars(args), "rollout": {}}
    for steps in [256, 1024, 4096, 16384]:
        row = {}
        torch.manual_seed(0); m = LlamaForCausalLM(cfg).cuda().bfloat16().eval()
        try:
            t, mem = sdpa_rollout(m, args.ctx, steps); row['sdpa'] = {'ms': round(t * 1000), 'gb': round(mem, 2)}
        except Exception as e:
            row['sdpa'] = 'OOM/' + str(e)[:30]
        del m; torch.cuda.empty_cache()
        # 全-KDA（纯 O(1) 上界对比）
        torch.manual_seed(0); m = LlamaForCausalLM(cfg).cuda().bfloat16().eval()
        from kda_llama import kdaify_llama
        kdaify_llama(m); m.config.use_cache = True
        try:
            t, mem = kda_rollout(m, args.ctx, steps); row['kda'] = {'ms': round(t * 1000), 'gb': round(mem, 2)}
        except Exception as e:
            row['kda'] = 'ERR/' + str(e)[:40]
        del m; torch.cuda.empty_cache()
        res['rollout'][steps] = row
        print(f"steps={steps:6d} | SDPA {row['sdpa']} | KDA {row['kda']}", flush=True)
    json.dump(res, open('/home/wm-vla-rl/module_videowm/results/rollout_efficiency.json', 'w'), indent=1)
    print("ROLLOUT_BENCH_DONE", flush=True)


if __name__ == '__main__':
    main()
