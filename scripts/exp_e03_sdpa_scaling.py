"""E0.3 — SDPA predictor 长时程 scaling harness.

测 V-JEPA2-AC predictor 的前向延迟/显存峰值 vs 时间长度 T（合成输入，无需真数据）。
量化 block-causal O(N²) 注意力的崩点 —— 即 KDA 线性注意力(P1)要解决的问题。
输出 JSON 供后续与 KDA overlay 同轴作图。
"""
import os, sys, time, json
REPO = "/home/wm-vla-rl/_ext/vjepa2"; sys.path.insert(0, REPO)
import torch
from src.models.utils.modules import build_action_block_causal_attention_mask

def main():
    dev = "cuda"
    from hubconf import vjepa2_ac_vit_giant
    enc, pred = vjepa2_ac_vit_giant(pretrained=True)
    del enc; torch.cuda.empty_cache()           # scaling 只需 predictor
    pred = pred.to(dev).eval()
    D = pred.predictor_embed.in_features
    H, W = pred.grid_height, pred.grid_width
    addt = 3 if pred.use_extrinsics else 2
    print(f"predictor: D_in={D} grid={H}x{W} add_tokens={addt} blocks={len(pred.predictor_blocks)} "
          f"gpu={torch.cuda.get_device_name(0)}", flush=True)
    results = []
    for T in [2, 4, 8, 16, 32, 48, 64, 96, 128, 192, 256, 384, 512]:
        try:
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
            pred.attn_mask = build_action_block_causal_attention_mask(T, H, W, addt)
            x = torch.randn(1, T * H * W, D, device=dev)
            actions = torch.randn(1, T, 7, device=dev); states = torch.randn(1, T, 7, device=dev)
            with torch.inference_mode():
                _ = pred(x, actions, states); torch.cuda.synchronize()
                t0 = time.time()
                for _ in range(3): _ = pred(x, actions, states)
                torch.cuda.synchronize(); dt = (time.time() - t0) / 3
            mem = torch.cuda.max_memory_allocated() / 1e9
            ntok = T * (H * W + addt)
            print(f"T={T:4d} ntok={ntok:6d} dt={dt*1000:8.1f}ms mem={mem:6.2f}GB", flush=True)
            results.append(dict(T=T, ntok=ntok, dt_ms=round(dt*1000, 2), mem_gb=round(mem, 3)))
        except RuntimeError as e:
            print(f"T={T} OOM/err: {str(e)[:140]}", flush=True)
            results.append(dict(T=T, oom=True, err=str(e)[:140])); break
    out = "/home/wm-vla-rl/module1_vjepa2ac/results/e03_sdpa_scaling.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(results, open(out, "w"), indent=2)
    print("WROTE", out); print("E03_OK", flush=True)

main()
