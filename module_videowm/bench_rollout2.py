"""干净的 rollout 效率 benchmark(纯 memory + 单步延迟,快)。
WM imagination rollout 到长度 N 时,decode 显存 = (24-K)层 softmax 的 KV cache(O(N)) + K 层 KDA state(O(1))。
扫 K(KDA 层数){0,2,12,24} × N(rollout 长度)→ 显存 + 单 decode 步延迟。
显示 hybrid 的"层数→效率"权衡(Pareto 的效率轴)。
"""
import os, sys, time, json, torch
sys.path.insert(0, "/home/wm-vla-rl/module_videowm")
from transformers import LlamaConfig, LlamaForCausalLM
from transformers.cache_utils import DynamicCache
from kda_llama import kdaify_llama
DEV = "cuda"
L, DIM, H = 24, 1024, 16   # medium 架构


@torch.no_grad()
def measure(K, N):
    """K 层 KDA(其余 softmax)。prefill N token + 1 decode 步,测峰值显存 + 单步延迟。
    softmax 层走 KV cache(use_cache);KDA 层 use_cache=False(O(1) state，不进 KV cache)。"""
    cfg = LlamaConfig(vocab_size=8192, hidden_size=DIM, intermediate_size=DIM * 4, num_hidden_layers=L,
                      num_attention_heads=H, num_key_value_heads=H, max_position_embeddings=N + 16, attn_implementation='sdpa')
    torch.manual_seed(0)
    m = LlamaForCausalLM(cfg).cuda().bfloat16().eval()
    if K > 0:
        # 均匀分布 K 层为 KDA
        kda_idx = set(int(round(i * (L - 1) / max(1, K - 1))) for i in range(K)) if K > 1 else {L // 2}
        kdaify_llama(m, layers=kda_idx)
        m.config.use_cache = False  # 全局；但 softmax 层仍会建 KV cache（HF 行为）
    ids = torch.randint(0, 8192, (1, N), device=DEV)
    torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
    cache = DynamicCache()
    out = m(input_ids=ids, past_key_values=cache, use_cache=True)   # prefill
    nxt = out.logits[:, -1:].argmax(-1)
    torch.cuda.synchronize(); t0 = time.time()
    for _ in range(8):                                              # 8 个 decode 步测单步延迟
        out = m(input_ids=nxt, past_key_values=cache, use_cache=True)
        nxt = out.logits[:, -1:].argmax(-1)
    torch.cuda.synchronize()
    step_ms = (time.time() - t0) / 8 * 1000
    peak = torch.cuda.max_memory_allocated() / 1e9
    del m, cache; torch.cuda.empty_cache()
    return round(peak, 2), round(step_ms, 1)


def main():
    res = {}
    print(f"{'N':>7} | " + " | ".join(f"K={k} (GB/ms)" for k in [0, 2, 12, 24]), flush=True)
    for N in [512, 2048, 8192, 32768]:
        row = {}
        cells = []
        for K in [0, 2, 12, 24]:
            try:
                gb, ms = measure(K, N); row[K] = {'gb': gb, 'step_ms': ms}; cells.append(f"{gb}/{ms}")
            except Exception as e:
                row[K] = 'OOM/' + str(e)[:20]; cells.append('OOM')
        res[N] = row
        print(f"{N:>7} | " + " | ".join(cells), flush=True)
    json.dump({'arch': 'medium 24L/1024d', 'note': 'K=KDA layers; cell=peak GB / decode step ms', 'data': res},
              open('/home/wm-vla-rl/module_videowm/results/rollout_efficiency.json', 'w'), indent=1)
    print("ROLLOUT2_DONE", flush=True)


if __name__ == '__main__':
    main()
