"""E1.1 数值 smoke：KDA-ified predictor 是否 shape==SDPA、有限、可替换。"""
import sys, torch
REPO = "/home/wm-vla-rl/_ext/vjepa2"; HERE = "/home/wm-vla-rl/module1_vjepa2ac"
sys.path.insert(0, REPO); sys.path.insert(0, HERE)
from src.models.utils.modules import build_action_block_causal_attention_mask
from kda_acblock import kda_ify_predictor


def main():
    dev = "cuda"
    from hubconf import vjepa2_ac_vit_giant
    enc, pred = vjepa2_ac_vit_giant(pretrained=True); del enc; torch.cuda.empty_cache()
    pred = pred.to(dev).eval()
    H, W = pred.grid_height, pred.grid_width
    D = pred.predictor_embed.in_features
    nh = pred.predictor_blocks[0].attn.num_heads
    T = 4
    pred.attn_mask = build_action_block_causal_attention_mask(T, H, W, 2)
    x = torch.randn(1, T * H * W, D, device=dev)
    a = torch.randn(1, T, 7, device=dev); s = torch.randn(1, T, 7, device=dev)
    with torch.inference_mode():
        o_sdpa = pred(x, a, s)
    n_sdpa = sum(p.numel() for p in pred.parameters())
    print(f"SDPA  out={tuple(o_sdpa.shape)} params={n_sdpa/1e6:.1f}M heads={nh} D={D}", flush=True)

    kda_ify_predictor(pred); pred = pred.to(dev).eval()
    n_kda = sum(p.numel() for p in pred.parameters())
    with torch.inference_mode():
        o_kda = pred(x, a, s)
    same = (o_kda.shape == o_sdpa.shape)
    finite = bool(torch.isfinite(o_kda).all())
    print(f"KDA   out={tuple(o_kda.shape)} params={n_kda/1e6:.1f}M (+{(n_kda-n_sdpa)/1e6:.1f}M temporal-KDA)", flush=True)
    print(f"shape==SDPA: {same} | finite: {finite}", flush=True)
    print("E1.1_OK" if (same and finite) else "E1.1_FAIL", flush=True)


main()
