"""定位 E1.5 block 层 full-vs-step diff：逐帧比对单个 DividedKDAACBlock。"""
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
T = 3
torch.manual_seed(0)
xseq = torch.randn(1, T, P, D, device=dev)
with torch.inference_mode():
    full = blk(xseq.reshape(1, T * P, D), T=T, H=H, W=W, action_tokens=A).view(1, T, P, D)
    st = None; steps = []
    for t in range(T):
        o, st = blk.forward_step(xseq[:, t].clone(), H, W, A, st)
        steps.append(o)
    step = torch.stack(steps, dim=1)  # [1, T, P, D]
for t in range(T):
    d = (full[:, t].float() - step[:, t].float()).abs().max().item()
    print(f"frame {t}: full-vs-step maxdiff={d:.4e}")
print("DBG_DONE")
