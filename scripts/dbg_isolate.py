"""分离 spatial vs temporal 的 full-vs-step diff（frame 0）。"""
import sys, torch
REPO = "/home/wm-vla-rl/_ext/vjepa2"; HERE = "/home/wm-vla-rl/module1_vjepa2ac"
sys.path.insert(0, REPO); sys.path.insert(0, HERE)
from kda_acblock import kda_ify_predictor

dev = "cuda"
from hubconf import vjepa2_ac_vit_giant
enc, pred = vjepa2_ac_vit_giant(pretrained=True); del enc; torch.cuda.empty_cache()
kda_ify_predictor(pred); pred = pred.to(dev).eval()
blk = pred.predictor_blocks[0]
H = W = pred.grid_height; A = 2; P = A + H * W
D = blk.norm1.normalized_shape[0]
T = 2
torch.manual_seed(0)
xseq = torch.randn(1, T, P, D, device=dev)
with torch.inference_mode():
    # spatial：full（batch=T）vs step（batch=1）frame 0
    xfull = xseq.reshape(1, T * P, D)
    ys_full = blk.attn(blk.norm1(xfull).view(T, P, D), T=1, H=H, W=W, action_tokens=A)  # [T,P,D]
    ys_step = blk.attn(blk.norm1(xseq[:, 0]), T=1, H=H, W=W, action_tokens=A)            # [1,P,D]
    sp = (ys_full[0].float() - ys_step[0].float()).abs().max().item()
    print(f"spatial frame0 diff: {sp:.4e}")
    # temporal：在相同输入上比 full(T) 位置0 vs step(1)
    inp = torch.randn(1, T, P, D, device=dev)
    yf = blk.norm_t(inp.reshape(1, T * P, D)).view(1, T, P, D).permute(0, 2, 1, 3).reshape(P, T, D)
    of, _ = blk.kda(yf)
    of0 = of.reshape(1, P, T, D).permute(0, 2, 1, 3)[:, 0]      # frame0 temporal (full)
    ys = blk.norm_t(inp[:, 0]).reshape(P, 1, D)
    os_, _ = blk.kda(ys)
    os0 = os_.reshape(1, P, D)
    tp = (of0.float() - os0.float()).abs().max().item()
    print(f"temporal frame0 diff: {tp:.4e}")
print("DBG_DONE")
