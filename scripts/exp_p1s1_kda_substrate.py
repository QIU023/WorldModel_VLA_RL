"""P1 S1 — substrate smoke（py3.10: torch2.11 + fla0.5.0）。
A) vjepa2 AC predictor 在该 env 能 load+forward。
B) KDA kernel（chunk_kda / fused_recurrent_kda）在本机 GPU 跑通，chunk≈recurrent 一致。
KDA 用法照搬 sibling KimiDeltaAttention：fused_kda_gate(g_raw,A_log,dt_bias) + use_qk_l2norm_in_kernel。
"""
import sys, torch
REPO = "/home/wm-vla-rl/_ext/vjepa2"; sys.path.insert(0, REPO)


def part_A():
    from hubconf import vjepa2_ac_vit_giant
    from src.models.utils.modules import build_action_block_causal_attention_mask
    enc, pred = vjepa2_ac_vit_giant(pretrained=True); del enc; torch.cuda.empty_cache()
    pred = pred.to("cuda").eval()
    H, W = pred.grid_height, pred.grid_width
    D = pred.predictor_embed.in_features
    pred.attn_mask = build_action_block_causal_attention_mask(3, H, W, 2)
    x = torch.randn(1, 2 * H * W, D, device="cuda")
    a = torch.randn(1, 2, 7, device="cuda"); s = torch.randn(1, 2, 7, device="cuda")
    with torch.inference_mode():
        o = pred(x, a, s)
    print(f"[A] vjepa2 AC predictor @py3.10: out={tuple(o.shape)} finite={bool(torch.isfinite(o).all())}", flush=True)


def part_B():
    from fla.ops.kda import chunk_kda, fused_recurrent_kda
    from fla.ops.kda.gate import fused_kda_gate
    dev = "cuda"; B, T, Hh, Dh = 1, 128, 4, 64
    torch.manual_seed(0)
    q = torch.randn(B, T, Hh, Dh, device=dev, dtype=torch.bfloat16)
    k = torch.randn(B, T, Hh, Dh, device=dev, dtype=torch.bfloat16)
    v = torch.randn(B, T, Hh, Dh, device=dev, dtype=torch.bfloat16)
    g_raw = torch.randn(B, T, Hh, Dh, device=dev, dtype=torch.float32)
    A_log = torch.log(torch.empty(Hh, device=dev).uniform_(1, 16))
    dt_bias = torch.zeros(Hh * Dh, device=dev)
    g = fused_kda_gate(g_raw, A_log, dt_bias=dt_bias)
    beta = torch.rand(B, T, Hh, device=dev, dtype=torch.float32)
    common = dict(q=q, k=k, v=v, g=g, beta=beta, initial_state=None,
                  output_final_state=True, use_qk_l2norm_in_kernel=True, cu_seqlens=None)
    oc, sc = chunk_kda(**common)
    orr, sr = fused_recurrent_kda(**common)
    diff = (oc.float() - orr.float()).abs().max().item()
    print(f"[B] KDA chunk out={tuple(oc.shape)} finite={bool(torch.isfinite(oc).all())} "
          f"| chunk-vs-recurrent maxdiff={diff:.4e}", flush=True)


if __name__ == "__main__":
    part_A()
    part_B()
    print("P1S1_OK", flush=True)
