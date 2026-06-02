"""P1 capstone — 微调 KDA WM 的开环多步 rollout 保真度（held-out LIBERO）。
从真实首帧 latent 出发、喂真实动作开环 rollout H 步（预测 latent 反馈），测各 horizon 的 latent MSE。
量化"高效 WM 能准确预测多远"——P2 规划的基础。无需 sim。
"""
import sys, json, random, torch
import torch.nn.functional as F
REPO = "/home/wm-vla-rl/_ext/vjepa2"; HERE = "/home/wm-vla-rl/module1_vjepa2ac"
sys.path.insert(0, REPO); sys.path.insert(0, HERE)
from kda_acblock import kda_ify_predictor
from rollout import rollout
import cem_planner as C
RES = "/home/wm-vla-rl/module1_vjepa2ac/results"
H = 16


def main():
    dev = "cuda"
    from hubconf import vjepa2_ac_vit_giant
    enc, pred = vjepa2_ac_vit_giant(pretrained=True); del enc; torch.cuda.empty_cache()
    kda_ify_predictor(pred)
    pred.load_state_dict(torch.load("/home/wm-vla-rl/_ext/ckpts/kda_predictor_libero.pt", map_location="cpu"))
    pred = pred.to(dev).eval()
    pred.attn_mask = torch.zeros(1, 1, dtype=torch.bool, device=dev)
    tok = pred.grid_height * pred.grid_width
    data = torch.load("/home/wm-vla-rl/_ext/datasets/libero_latents.pt")
    random.seed(0); random.shuffle(data); held = data[32:]
    hm = [0.0] * H; cnt = 0
    for d in held:
        if d["T"] < H + 1:
            continue
        lat = d["latent"].float().to(dev)
        z0 = lat[0:1].unsqueeze(0)
        actions = d["actions"][:H].float().unsqueeze(0).to(dev)
        pose0 = d["states"][0:1].float().unsqueeze(0).to(dev)
        with torch.inference_mode():
            traj = rollout(pred, z0, actions, pose0, tok, compute_new_pose=C.compute_new_pose)
        for h in range(1, H + 1):
            hm[h - 1] += F.mse_loss(traj[0, h], lat[h]).item()
        cnt += 1
    hm = [m / cnt for m in hm]
    json.dump({"horizon_mse": hm, "n_demos": cnt}, open(f"{RES}/e1y_rollout_fidelity.json", "w"))
    print(f"ROLLOUT_FIDELITY_OK n={cnt} h1={hm[0]:.4f} h4={hm[3]:.4f} h8={hm[7]:.4f} h16={hm[15]:.4f}", flush=True)


main()
