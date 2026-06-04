"""138M(12L/768d/12head)rollout 效率 benchmark —— Pareto 的效率轴。
WM imagination rollout 到长度 N 时,decode 显存 = (12-K)层 softmax 的 KV cache(O(N)) + K 层 KDA state(O(1))。
扫 K(KDA 层数){0, 9(75% Kimi), 12(all)} × N(rollout 长度)→ 峰值显存 + 单 decode 步延迟。
K=9 用 Kimi-Linear 3:1 pattern(full@{3,7,11}),与训练一致。
"""
import os, sys, time, json, torch
sys.path.insert(0, "/home/wm-vla-rl/module_videowm")
from transformers import LlamaConfig, LlamaForCausalLM
from transformers.cache_utils import DynamicCache
from kda_llama import kdaify_llama
DEV = "cuda"
L, DIM, H = 12, 768, 12   # 138M bair-64 架构


@torch.no_grad()
def measure(K, N):
    cfg = LlamaConfig(vocab_size=16386, hidden_size=DIM, intermediate_size=3072, num_hidden_layers=L,
                      num_attention_heads=H, num_key_value_heads=H, max_position_embeddings=N + 16, attn_implementation='sdpa')
    torch.manual_seed(0)
    m = LlamaForCausalLM(cfg).cuda().bfloat16().eval()
    if K == 9:
        kda_idx = set(i for i in range(L) if (i + 1) % 4 != 0)   # Kimi 3:1:full@{3,7,11}
    elif K == 12:
        kda_idx = set(range(L))
    else:
        kda_idx = set()
    if kda_idx:
        kdaify_llama(m, layers=kda_idx)
        m.config.use_cache = False
    ids = torch.randint(0, 16386, (1, N), device=DEV)
    torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
    cache = DynamicCache()
    out = m(input_ids=ids, past_key_values=cache, use_cache=True)   # prefill
    nxt = out.logits[:, -1:].argmax(-1)
    torch.cuda.synchronize(); t0 = time.time()
    for _ in range(8):
        out = m(input_ids=nxt, past_key_values=cache, use_cache=True)
        nxt = out.logits[:, -1:].argmax(-1)
    torch.cuda.synchronize()
    step_ms = (time.time() - t0) / 8 * 1000
    peak = torch.cuda.max_memory_allocated() / 1e9
    del m, cache; torch.cuda.empty_cache()
    return round(peak, 2), round(step_ms, 1)


def main():
    res = {}
    print(f"{'N':>7} | " + " | ".join(f"K={k} (GB/ms)" for k in [0, 9, 12]), flush=True)
    for N in [512, 2048, 8192, 32768]:
        row = {}; cells = []
        for K in [0, 9, 12]:
            try:
                gb, ms = measure(K, N); row[K] = {'gb': gb, 'step_ms': ms}; cells.append(f"{gb}/{ms}")
            except Exception as e:
                row[K] = 'OOM/' + str(e)[:20]; cells.append('OOM')
        res[N] = row
        print(f"{N:>7} | " + " | ".join(cells), flush=True)
    json.dump({'arch': '138M 12L/768d', 'note': 'K=KDA layers (9=75% Kimi 3:1); cell=peak GB / decode step ms',
               'data': res}, open('/home/wm-vla-rl/module_videowm/results/rollout_efficiency_138.json', 'w'), indent=1)
    print("ROLLOUT138_DONE", flush=True)


if __name__ == '__main__':
    main()
