"""(b) 长时程 rollout 保真度对比：KDA-full vs KDA⊕AttnRes（同 held-out, 开环 H 步）。
novel 跨帧 AttnRes 该在长程多步预测上优于纯 KDA 递归（这是它 next-latent@T=64 之外的卖点）。
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
CONFIGS = [("KDA-full", "kda_full_predictor_libero.pt", False),
           ("KDA+AttnRes", "kda_attnres_predictor_libero.pt", True)]


@torch.inference_mode()
def run(ckpt, use_ar, held, dev):
    from hubconf import vjepa2_ac_vit_giant
    enc, pred = vjepa2_ac_vit_giant(pretrained=True); del enc; torch.cuda.empty_cache()
    kda_ify_predictor(pred, use_attnres=use_ar)
    pred.load_state_dict(torch.load(f"/home/wm-vla-rl/_ext/ckpts/{ckpt}", map_location="cpu"))
    pred = pred.to(dev).eval(); pred.attn_mask = torch.zeros(1, 1, dtype=torch.bool, device=dev)
    tok = pred.grid_height * pred.grid_width
    hm = [0.0] * H; cnt = 0
    for d in held:
        if d["T"] < H + 1:
            continue
        lat = d["latent"].float().to(dev)
        z0 = lat[0:1].unsqueeze(0)
        actions = d["actions"][:H].float().unsqueeze(0).to(dev)
        pose0 = d["states"][0:1].float().unsqueeze(0).to(dev)
        traj = rollout(pred, z0, actions, pose0, tok, compute_new_pose=C.compute_new_pose)
        for h in range(1, H + 1):
            hm[h - 1] += F.mse_loss(traj[0, h], lat[h]).item()
        cnt += 1
    del pred; torch.cuda.empty_cache()
    return [m / cnt for m in hm], cnt


def main():
    dev = "cuda"
    data = torch.load("/home/wm-vla-rl/_ext/datasets/libero_latents.pt")
    random.seed(0); random.shuffle(data); held = data[-40:]   # 同 finetune 的 held-out
    res = {}
    for name, ckpt, ar in CONFIGS:
        hm, cnt = run(ckpt, ar, held, dev)
        res[name] = hm
        print(f"{name:12s}: h1={hm[0]:.4f} h2={hm[1]:.4f} h4={hm[3]:.4f} h8={hm[7]:.4f} h16={hm[15]:.4f} (n={cnt})", flush=True)
    json.dump(res, open(f"{RES}/e1z_rollout_fidelity_compare.json", "w"))
    print("COMPARE_OK", flush=True)


main()
