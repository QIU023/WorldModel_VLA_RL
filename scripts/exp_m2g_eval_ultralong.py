"""(task#12 step1.5) Ultra-Long sim eval harness —— 所有设计都要用它量 success。
LiLo-VLA Ultra-Long(3 任务,9/10/16 步),用 init_states 重置,闭环 roll 策略,测 success-rate。
策略可插拔:现在用我们现成 latent actor 做 smoke(确认 harness 在 ultra_long 上端到端跑通);
后续换 OFT-蒸馏 latent 策略 / OFT。
"""
import os, sys, json, numpy as np, torch
import torch.nn as nn, torch.nn.functional as F
os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, "/home/wm-vla-rl/_ext/LIBERO")
sys.path.insert(0, "/home/wm-vla-rl/_ext/vjepa2")
sys.path.insert(0, "/home/wm-vla-rl/module1_vjepa2ac")

# --- 修 LiLo init_states 的 torch.load weights_only 默认（新版 torch=True 读不了 pickled state）---
_orig_load = torch.load
def _patched_load(*a, **k):
    k.setdefault("weights_only", False)
    return _orig_load(*a, **k)
torch.load = _patched_load

import lilo_vla.benchmark                                  # 注册 ultra_long / libero_long_plus_plus 套件
# --- 补注册 Ultra-Long goal 用到的 `Closed` predicate（LIBERO 只有 `close`=is_close()，语义同）---
from libero.libero.envs.predicates import VALIDATE_PREDICATE_FN_DICT
from libero.libero.envs.predicates.base_predicates import Close
VALIDATE_PREDICATE_FN_DICT.setdefault("closed", Close())
from libero.libero.benchmark import get_benchmark
from libero.libero.envs import OffScreenRenderEnv
from app.vjepa_droid.transforms import make_transforms
from hubconf import vjepa2_ac_vit_giant

DEV = "cuda"
CKPT = "/home/wm-vla-rl/_ext/ckpts"
RES = "/home/wm-vla-rl/module1_vjepa2ac/results"
BENCH = os.environ.get("EVAL_BENCH", "ultra_long")
NEP = int(os.environ.get("EVAL_EP", "5"))                 # 每任务 trial 数（init_states 有 50）
HMAX = int(os.environ.get("EVAL_H", "600"))               # 超长任务需要很多步（16 步任务）
MAXN = float(os.environ.get("ACT_MAXNORM", "1.0"))
ACTOR = os.environ.get("EVAL_ACTOR", "actor_bc_arlayer.pt")
WM_ENC = os.environ.get("M2G_WM_ENC", "kda_arlayer_big_sftfull.pt")


class Actor(nn.Module):                                    # 复刻 exp_m2c 的 latent actor（smoke 用）
    def __init__(self, dim, act=7):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 256), nn.GELU(),
                                 nn.Linear(256, 256), nn.GELU())
        self.mean = nn.Linear(256, act); self.logstd = nn.Parameter(torch.full((act,), -1.0))


def main():
    enc, _ = vjepa2_ac_vit_giant(pretrained=True)
    if WM_ENC:
        sd = torch.load(f"{CKPT}/{WM_ENC}", map_location="cpu")
        enc.load_state_dict(sd["encoder"]); print(f"[enc] validated {WM_ENC}", flush=True)
    enc = enc.to(DEV).eval()
    for p in enc.parameters(): p.requires_grad_(False)
    dim = enc.embed_dim if hasattr(enc, "embed_dim") else 1408
    actor = Actor(dim).to(DEV).eval()
    actor.load_state_dict(torch.load(f"{CKPT}/{ACTOR}", map_location="cpu")["actor"])
    transform = make_transforms(crop_size=256, random_horizontal_flip=False,
                                random_resize_aspect_ratio=(1., 1.), random_resize_scale=(1., 1.),
                                reprob=0., auto_augment=False, motion_shift=False)

    @torch.no_grad()
    def encode(img):
        clip = transform(img[None]).unsqueeze(0).to(DEV)
        B, C, T, Hh, Ww = clip.size()
        c = clip.permute(0, 2, 1, 3, 4).flatten(0, 1).unsqueeze(2).repeat(1, 1, 2, 1, 1)
        h = enc(c)[0]
        return F.layer_norm(h.float(), (h.size(-1),)).mean(0)

    bench = get_benchmark(BENCH)()
    n = bench.n_tasks
    print(f"ULTRALONG eval | {BENCH} ntask={n} ep={NEP} Hmax={HMAX} actor={ACTOR}", flush=True)
    results = {}
    for ti in range(n):
        task = bench.get_task(ti)
        bddl = bench.get_task_bddl_file_path(ti)
        init_states = bench.get_task_init_states(ti)        # (50, state_dim)（torch.load 已 patch）
        env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=128, camera_widths=128)
        succ = 0
        for ep in range(NEP):
            env.reset()
            obs = env.set_init_state(init_states[ep % len(init_states)])
            done = False
            for t in range(HMAX):
                z = encode(np.asarray(obs["agentview_image"]))
                with torch.no_grad():
                    a = actor.mean(actor.net(z[None])).clamp(-MAXN, MAXN)
                obs, _, done, _ = env.step(a[0].cpu().numpy().astype(np.float32))
                if env.check_success():
                    succ += 1; break
                if done: break
        env.close()
        sr = succ / NEP; results[f"task{ti}"] = sr
        print(f"  task{ti} {task.name[:48]:48s} success={sr:.2f} ({succ}/{NEP})", flush=True)
    overall = float(np.mean(list(results.values())))
    json.dump(dict(overall=overall, bench=BENCH, ep=NEP, **results), open(f"{RES}/ultralong_eval.json", "w"))
    print(f"ULTRALONG_EVAL_DONE overall={overall:.3f} ({BENCH}, {NEP}ep/task, {n}tasks)", flush=True)


if __name__ == "__main__":
    main()
