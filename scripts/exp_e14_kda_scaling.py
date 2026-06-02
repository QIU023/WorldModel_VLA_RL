"""E1.4 — KDA-ified predictor 长时程 scaling（与 E0.3 SDPA 同条件，合成输入）。
trick：KDA block 忽略 attn_mask，故把 pred.attn_mask 设 1×1 dummy，避免预测器构建 T×258 的
O(N²) mask（那本身就会 OOM）。于是能跑到 SDPA OOM 的 T=512/768/1024。
"""
import sys, time, json, torch
REPO = "/home/wm-vla-rl/_ext/vjepa2"; HERE = "/home/wm-vla-rl/module1_vjepa2ac"
sys.path.insert(0, REPO); sys.path.insert(0, HERE)
from kda_acblock import kda_ify_predictor


def main():
    dev = "cuda"
    from hubconf import vjepa2_ac_vit_giant
    enc, pred = vjepa2_ac_vit_giant(pretrained=True); del enc; torch.cuda.empty_cache()
    kda_ify_predictor(pred); pred = pred.to(dev).eval()
    D = pred.predictor_embed.in_features
    H, W = pred.grid_height, pred.grid_width
    addt = 3 if pred.use_extrinsics else 2
    pred.attn_mask = torch.zeros(1, 1, dtype=torch.bool, device=dev)   # dummy（KDA 忽略；切片不 OOM）
    print(f"KDA predictor: D={D} grid={H}x{W} blocks={len(pred.predictor_blocks)} gpu={torch.cuda.get_device_name(0)}", flush=True)
    results = []
    for T in [2, 4, 8, 16, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024]:
        try:
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
            x = torch.randn(1, T * H * W, D, device=dev)
            a = torch.randn(1, T, 7, device=dev); s = torch.randn(1, T, 7, device=dev)
            with torch.inference_mode():
                _ = pred(x, a, s); torch.cuda.synchronize()
                t0 = time.time()
                for _ in range(3): _ = pred(x, a, s)
                torch.cuda.synchronize(); dt = (time.time() - t0) / 3
            mem = torch.cuda.max_memory_allocated() / 1e9
            print(f"T={T:5d} dt={dt*1000:8.1f}ms mem={mem:6.2f}GB", flush=True)
            results.append(dict(T=T, dt_ms=round(dt * 1000, 2), mem_gb=round(mem, 3)))
        except RuntimeError as e:
            print(f"T={T} OOM/err: {str(e)[:120]}", flush=True)
            results.append(dict(T=T, oom=True)); break
    out = "/home/wm-vla-rl/module1_vjepa2ac/results/e14_kda_scaling.json"
    json.dump(results, open(out, "w"), indent=2)
    print("WROTE", out); print("E14_OK", flush=True)


main()
