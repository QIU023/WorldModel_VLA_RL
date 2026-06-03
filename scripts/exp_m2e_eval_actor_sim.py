"""(finale 真指标) LIBERO-LONG sim success-rate —— imagination-trained actor 的真能力测。
load actor(imagine_ac_libero10.pt 里的 actor）+ 冻结 V-JEPA2 encoder；在真 libero_10 sim 里
闭环 roll（encode obs → actor mean action → step），测 success-rate。这是 finale 的真判据
（非想象里的 reward）。actor 是 reactive policy（无 WM，纯 encoder latent → action）。
"""
import os, sys, json, numpy as np, torch
import torch.nn as nn, torch.nn.functional as F
os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, "/home/wm-vla-rl/_ext/vjepa2")
sys.path.insert(0, "/home/wm-vla-rl/module1_vjepa2ac")
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from app.vjepa_droid.transforms import make_transforms
from hubconf import vjepa2_ac_vit_giant

DEV = "cuda"
CKPT = "/home/wm-vla-rl/_ext/ckpts"
RES = "/home/wm-vla-rl/module1_vjepa2ac/results"
BENCH = os.environ.get("EVAL_BENCH", "libero_10")
NTASK = int(os.environ.get("EVAL_NTASK", "4"))
NEP = int(os.environ.get("EVAL_EP", "3"))
HMAX = int(os.environ.get("EVAL_H", "180"))
MAXN = float(os.environ.get("ACT_MAXNORM", "1.0"))
ACTOR = os.environ.get("EVAL_ACTOR", "imagine_ac_libero10.pt")


class Actor(nn.Module):                                    # 复刻 exp_m2c 的 Actor
    def __init__(self, dim, act=7):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 256), nn.GELU(),
                                 nn.Linear(256, 256), nn.GELU())
        self.mean = nn.Linear(256, act); self.logstd = nn.Parameter(torch.full((act,), -1.0))


def main():
    enc, _ = vjepa2_ac_vit_giant(pretrained=True)
    _wm = os.environ.get("M2E_WM_ENC", "")               # 验证 WM 的训练 encoder（与 actor 训练空间一致）
    if _wm:
        enc.load_state_dict(torch.load(_wm, map_location="cpu")["encoder"]); print(f"[enc] {_wm}", flush=True)
    enc = enc.to(DEV).eval()
    for p in enc.parameters(): p.requires_grad_(False)
    dim = enc.embed_dim if hasattr(enc, "embed_dim") else 1408
    actor = Actor(dim).to(DEV).eval()
    ck = torch.load(f"{CKPT}/{ACTOR}", map_location="cpu")
    actor.load_state_dict(ck["actor"])
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

    bench = benchmark.get_benchmark_dict()[BENCH]()
    print(f"FINALE eval actor={ACTOR} | {BENCH} ntask={NTASK} ep={NEP} Hmax={HMAX}", flush=True)
    results = {}
    for ti in range(NTASK):
        task = bench.get_task(ti)
        bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
        env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=128, camera_widths=128)
        succ = 0
        for ep in range(NEP):
            env.seed(7000 + ep); obs = env.reset(); done = False
            for t in range(HMAX):
                z = encode(np.asarray(obs["agentview_image"]))
                with torch.no_grad():
                    a = actor.mean(actor.net(z[None])).clamp(-MAXN, MAXN)
                obs, _, done, _ = env.step(a[0].cpu().numpy().astype(np.float32))
                if env.check_success():
                    succ += 1; break
                if done:
                    break
        env.close()
        sr = succ / NEP; results[f"task{ti}"] = sr
        print(f"  task{ti} {task.name[:45]} success={sr:.2f} ({succ}/{NEP})", flush=True)
    overall = float(np.mean(list(results.values())))
    json.dump(dict(overall=overall, bench=BENCH, ep=NEP, **results), open(f"{RES}/finale_eval.json", "w"))
    print(f"FINALE_EVAL_DONE overall success-rate={overall:.3f} ({BENCH}, {NEP}ep/task, {NTASK}tasks)", flush=True)


if __name__ == "__main__":
    main()
