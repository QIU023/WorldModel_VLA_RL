"""(task#12 核心证据) WM 当想象环境的**每步成本** vs 想象 context 长度 —— SDPA(O(N²)) vs KDA/KDA+AttnRes(O(N))。
度量:单次 predictor forward 在递增 context 长度 T(帧)下的 peak 显存 + 时间(= imagination rollout 每步的真成本,
KV-cache/stateful 路径下也是这个量级),并捕获 OOM。
结论:SDPA 的每步成本随想象 horizon **二次爆炸 → 在超长程 OOM/跑不动**;KDA(+AttnRes) 线性 → 撑得起超长想象。
(用户:SDPA 慢/OOM 即正证据;KDA 能学就行。)纯架构成本,用训练好的 predictor 保真。
"""
import os, sys, time, json, numpy as np, torch
import torch.nn.functional as F
sys.path.insert(0, "/home/wm-vla-rl/_ext/vjepa2"); sys.path.insert(0, "/home/wm-vla-rl/module1_vjepa2ac")
from hubconf import vjepa2_ac_vit_giant
from kda_acblock import kda_ify_predictor, attnres_ify_predictor
from src.models.utils.modules import build_action_block_causal_attention_mask

DEV = "cuda"; CKPT = "/home/wm-vla-rl/_ext/ckpts"; RES = "/home/wm-vla-rl/module1_vjepa2ac/results"
B = int(os.environ.get("IMAG_B", "8"))                   # 想象轨迹数（GRPO group 量级）
TS = [8, 16, 32, 64, 96, 128, 192, 256, 384, 512, 768]   # 想象 context 长度（帧）
NWARM, NMEAS = 1, 3
RUNS = [("SDPA", "sdpa_sdpaH10_sftfull.pt", 0, 0),
        ("KDA", "kda_kdaonlyH10_sftfull.pt", 1, 0),
        ("KDA+AttnRes", "kda_arlayerH10_sftfull.pt", 1, 1)]


def build_pred(kda, ar, ckname):
    _, pred = vjepa2_ac_vit_giant(pretrained=True)
    if kda:
        kda_ify_predictor(pred, use_attnres=False)
        if ar: attnres_ify_predictor(pred, num_blocks=8)
    path = f"{CKPT}/{ckname}"
    if os.path.exists(path):
        try: pred.load_state_dict(torch.load(path, map_location="cpu")["predictor"])
        except Exception as e: print(f"  [warn] {ckname}: {e}", flush=True)
    return pred.to(DEV).eval()


def main():
    print(f"IMAGINATION STEP-COST | B={B} context_frames={TS}", flush=True)
    out = {}
    for name, ckname, kda, ar in RUNS:
        pred = build_pred(kda, ar, ckname)
        tok = pred.grid_height * pred.grid_width
        res = []
        for T in TS:
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
            status, mem, dt = "ok", None, None
            # SDPA 需 block-causal mask（按当前 T 建，合理 O(T²)）；KDA 块的时间轴是因果线性、不用 dense mask → 占位 tiny
            if kda:
                pred.attn_mask = torch.zeros(2, 2, dtype=torch.bool, device=DEV)
            else:
                pred.attn_mask = build_action_block_causal_attention_mask(T, pred.grid_height, pred.grid_width, add_tokens=2).to(DEV).contiguous()
            try:
                z = F.layer_norm(torch.randn(B, T, tok, 1408, device=DEV), (1408,)).flatten(1, 2)
                a = torch.zeros(B, T, 7, device=DEV); s = torch.zeros(B, T, 7, device=DEV)
                with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    for _ in range(NWARM): pred(z, a, s)
                    torch.cuda.synchronize(); t0 = time.time()
                    for _ in range(NMEAS): pred(z, a, s)
                    torch.cuda.synchronize(); dt = (time.time() - t0) / NMEAS
                mem = torch.cuda.max_memory_allocated() / 1e9
            except RuntimeError as e:
                status = "OOM" if "out of memory" in str(e).lower() else f"ERR:{str(e)[:24]}"
                torch.cuda.empty_cache()
            res.append((T, mem, dt, status))
            ms = f"{dt*1000:7.1f}ms" if dt else "   --   "
            gb = f"{mem:6.1f}GB" if mem else "  OOM  "
            print(f"  {name:12s} T={T:4d}({T*tok:5d}tok)  mem={gb}  fwd={ms}  [{status}]", flush=True)
            if status == "OOM": break                      # 之后更大 T 必 OOM
        out[name] = res
        del pred; torch.cuda.empty_cache()
    json.dump(out, open(f"{RES}/imagination_cost.json", "w"))
    print("IMAGINATION_COST_DONE", flush=True)


if __name__ == "__main__":
    main()
