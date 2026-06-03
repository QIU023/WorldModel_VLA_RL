"""(finale 核心诊断) WM imagination reward-fidelity vs horizon —— 解耦 actor。
teacher-force 真 demo action 开环 roll WM，比 r̂(预测 latent) vs r̂(真 latent) 随 horizon 的偏差。
假设：WM 开环漂移 → 预测 latent 离真实流形 → grounded reward 被误读（inflation）= model-exploitation 的根。
三方 WM：SDPA(O(N²) 准) / KDA-full(O(N) 漂移大) / KDA⊕AttnRes(O(N) 且 novel 结构降漂移)。
→ 若 AttnRes 的 reward-error 远低于 KDA-full（接近 SDPA），则**跨帧 AttnRes 正是让高效 WM 可当想象环境的关键**。
"""
import os, sys, json, numpy as np, torch
import torch.nn as nn, torch.nn.functional as F
import torch.backends.cuda as _cb                          # SDPA predictor 时间轴 mask 广播 → 强制 math backend
_orig_sdp = _cb.sdp_kernel
_cb.sdp_kernel = lambda *a, **k: _orig_sdp(enable_flash=False, enable_mem_efficient=False, enable_math=True)
sys.path.insert(0, "/home/wm-vla-rl/_ext/vjepa2")
sys.path.insert(0, "/home/wm-vla-rl/_ext/vjepa2/notebooks")
sys.path.insert(0, "/home/wm-vla-rl/module1_vjepa2ac")
from hubconf import vjepa2_ac_vit_giant
from rollout import rollout as wm_rollout
from utils.mpc_utils import compute_new_pose
from kda_acblock import kda_ify_predictor
from src.models.utils.modules import build_action_block_causal_attention_mask

DEV = "cuda"
RES = "/home/wm-vla-rl/module1_vjepa2ac/results"
CKPT = "/home/wm-vla-rl/_ext/ckpts"
MAXH = int(os.environ.get("M2D_MAXH", "24"))
NDEMO = int(os.environ.get("M2D_NDEMO", "40"))
DATA = os.environ.get("M2D_DATA", "/home/wm-vla-rl/_ext/datasets/libero_latents.pt")
WMS = [("SDPA", "sdpa_full_predictor_libero.pt", 0, 0),
       ("KDA-full", "kda_full_predictor_libero.pt", 1, 0),
       ("KDA+AttnRes", "kda_attnres_predictor_libero.pt", 1, 1)]


class RewardModel(nn.Module):
    def __init__(self, dim, h=256):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, h), nn.GELU(),
                                 nn.Linear(h, h), nn.GELU(), nn.Linear(h, 1))

    def forward(self, z):
        return self.net(z).sigmoid()


@torch.inference_mode()
def main():
    enc, _ = vjepa2_ac_vit_giant(pretrained=True); torch.cuda.empty_cache()
    dim = enc.embed_dim if hasattr(enc, "embed_dim") else 1408
    del enc; torch.cuda.empty_cache()
    reward = RewardModel(dim).to(DEV).eval()
    reward.load_state_dict(torch.load(f"{CKPT}/reward_model.pt", map_location="cpu"))

    data = torch.load(DATA)
    demos = data[-NDEMO:]                                  # held-out
    print(f"WM reward-fidelity | data={DATA.split('/')[-1]} demos={len(demos)} MAXH={MAXH}", flush=True)

    results = {}
    for name, ck, kda, attnres in WMS:
        _, pred = vjepa2_ac_vit_giant(pretrained=True)
        if kda:
            kda_ify_predictor(pred, use_attnres=bool(attnres))
        pred.load_state_dict(torch.load(f"{CKPT}/{ck}", map_location="cpu"))
        pred = pred.to(DEV).eval()
        # 正确 block-causal mask（按 rollout 长度建，别用 zeros(1,1) 破坏因果）
        pred.attn_mask = build_action_block_causal_attention_mask(
            MAXH + 2, pred.grid_height, pred.grid_width, add_tokens=2).to(DEV).contiguous()
        tok = pred.grid_height * pred.grid_width

        err = np.zeros(MAXH); infl = np.zeros(MAXH); cnt = np.zeros(MAXH)
        for d in demos:
            lat = d["latent"].float().to(DEV)              # [T,tok,dim]
            T = d["T"]; H = min(MAXH, T - 1)
            if H < 1:
                continue
            z0 = lat[0][None, None]                        # [1,1,tok,dim]
            acts = d["actions"][:H].float().to(DEV)[None]  # [1,H,7]
            pose0 = d["states"][0].float().to(DEV)[None, None]   # [1,1,7]
            zhat = wm_rollout(pred, z0, acts, pose0, tok, compute_new_pose=compute_new_pose,
                              normalize_reps=True)         # [1,H,tok,dim]
            r_hat = reward(zhat[0].mean(1)).squeeze(-1)    # [H_hat]
            r_real = reward(lat[1:H + 1].mean(1)).squeeze(-1)   # [H]
            hh = min(r_hat.shape[0], r_real.shape[0], H)   # robust 对齐（rollout 可能 off-by-one）
            e = (r_hat[:hh] - r_real[:hh]).cpu().numpy()
            err[:hh] += np.abs(e); infl[:hh] += e; cnt[:hh] += 1
        err = err / np.maximum(cnt, 1); infl = infl / np.maximum(cnt, 1)
        results[name] = dict(reward_abs_err=err.tolist(), reward_inflation=infl.tolist())
        hs = [1, 4, 8, 16, min(MAXH, 24)]
        msg = "  ".join(f"h{h}:|err|={err[h-1]:.3f}/infl={infl[h-1]:+.3f}" for h in hs if h <= MAXH)
        print(f"  {name:12s} {msg}", flush=True)
        del pred; torch.cuda.empty_cache()

    json.dump(results, open(f"{RES}/m2d_wm_reward_fidelity.json", "w"))
    print("M2D_DONE wm reward-fidelity saved", flush=True)


if __name__ == "__main__":
    main()
