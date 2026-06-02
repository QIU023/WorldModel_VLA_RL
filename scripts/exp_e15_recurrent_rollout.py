"""E1.5 — KDA recurrent rollout（携带 SSM state，O(1)/步）。
A) 正确性：frame-by-frame 递归（threaded state）vs full divided forward 末帧一致。
B) 效率：递归 rollout 的 per-step dt + peak mem 随 horizon 是否持平（O(1)/步、显存平）。
"""
import sys, time, statistics, torch
REPO = "/home/wm-vla-rl/_ext/vjepa2"; HERE = "/home/wm-vla-rl/module1_vjepa2ac"
sys.path.insert(0, REPO); sys.path.insert(0, HERE)
from kda_acblock import kda_ify_predictor


def main():
    dev = "cuda"
    from hubconf import vjepa2_ac_vit_giant
    enc, pred = vjepa2_ac_vit_giant(pretrained=True); del enc; torch.cuda.empty_cache()
    kda_ify_predictor(pred); pred = pred.to(dev).eval()
    blocks = pred.predictor_blocks
    Dblk = blocks[0].norm1.normalized_shape[0]
    H = W = pred.grid_height; A = 2; P = A + H * W
    torch.manual_seed(0)
    print(f"blocks={len(blocks)} Dblk={Dblk} P={P} gpu={torch.cuda.get_device_name(0)}", flush=True)

    # A) 正确性
    T = 8
    xseq = torch.randn(1, T, P, Dblk, device=dev)
    with torch.inference_mode():
        xf = xseq.reshape(1, T * P, Dblk).clone()
        for blk in blocks:
            xf = blk(xf, T=T, H=H, W=W, action_tokens=A)
        full_last = xf.view(1, T, P, Dblk)[:, -1]
        states = [None] * len(blocks); rec_last = None
        for t in range(T):
            x = xseq[:, t].clone()
            for i, blk in enumerate(blocks):
                x, states[i] = blk.forward_step(x, H, W, A, states[i])
            rec_last = x
    diff = (full_last.float() - rec_last.float()).abs().max().item()
    print(f"[A] recurrent vs full divided (last frame, T={T}): maxdiff={diff:.4e}", flush=True)

    # B) 效率
    horizon = 512
    states = [None] * len(blocks)
    torch.cuda.reset_peak_memory_stats(); dts = []
    with torch.inference_mode():
        for t in range(horizon):
            x = torch.randn(1, P, Dblk, device=dev)
            torch.cuda.synchronize(); t0 = time.time()
            for i, blk in enumerate(blocks):
                x, states[i] = blk.forward_step(x, H, W, A, states[i])
            torch.cuda.synchronize(); dts.append(time.time() - t0)
    mem = torch.cuda.max_memory_allocated() / 1e9
    early = statistics.mean(dts[:10]) * 1000; late = statistics.mean(dts[-10:]) * 1000
    print(f"[B] recurrent rollout horizon={horizon}: per-step first10={early:.1f}ms last10={late:.1f}ms "
          f"peak_mem={mem:.2f}GB (flat ⇒ O(1)/step)", flush=True)
    print("E15_OK", flush=True)


main()
