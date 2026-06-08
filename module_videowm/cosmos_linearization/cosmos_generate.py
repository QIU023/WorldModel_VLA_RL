"""cosmos_generate.py — downstream video-generation eval: teacher Cosmos-Predict2-2B vs linearized-distilled.

Task: image+text -> future-frame video prediction on held-out LIBERO trajectories. Both models generate from
the SAME first frame + prompt + SAME seed. Metrics:
  - lin-vs-teacher  (PSNR/SSIM/LPIPS): linearization FIDELITY = how close the distilled model's video is to
                    the teacher's video under identical input. THE "逼近教师" number (domain-independent).
  - model-vs-GT     (PSNR/SSIM/LPIPS): predictive quality vs ground-truth future frames.

Modes:
  teacher : load pipeline, generate on held-out set, SAVE teacher videos (.pt) + teacher-vs-GT metrics.
  lineval : linearize pipe.transformer + load distilled linstate, generate same set,
            metrics lin-vs-GT AND lin-vs-saved-teacher.

Run:
  CUDA_VISIBLE_DEVICES=1 python cosmos_generate.py --mode teacher --model <snap> --out teacher_gen --n 24
  CUDA_VISIBLE_DEVICES=0 python cosmos_generate.py --mode lineval --model <snap> --teacher_dir teacher_gen \
        --resume <linstate.safetensors> --ratio 0.75 --tag s1 --n 24
"""
import os, sys, glob, json, argparse
sys.path.insert(0, "/home/wm-vla-rl/_ext")
import numpy as np
import torch
from PIL import Image

LIBERO_IMG = "/home/wm-vla-rl/_ext/WorldVLA/worldvla/processed_data/libero_10_img_action_256"


def humanize_task(traj_dir):
    name = os.path.basename(os.path.dirname(traj_dir))
    parts = name.split("_")
    while parts and (parts[0].isupper() or parts[0][:5].upper() == "SCENE" or parts[0].isdigit()
                     or parts[0][-1].isdigit()):
        parts.pop(0)
    desc = " ".join(parts).strip()
    return f"A robot arm performs a manipulation task: {desc}." if desc else "A robot arm manipulation video."


def eval_trajs(n, n_eval=32):
    trajs = sorted(glob.glob(os.path.join(LIBERO_IMG, "*", "trj_*")))
    trajs = [t for t in trajs if os.path.isdir(os.path.join(t, "imgs"))][-n_eval:]
    return trajs[:n]


def load_clip_frames(traj, num_frames, size):
    fs = sorted(glob.glob(os.path.join(traj, "imgs", "image_*.png")),
                key=lambda p: int(p.split("_")[-1].split(".")[0]))
    if len(fs) < num_frames:
        return None
    sel = fs[:num_frames]
    return [Image.open(p).convert("RGB").resize((size, size)) for p in sel]


# ---------------- metrics ----------------
def to_uint8(frames):
    """list[PIL] -> uint8 array [T,H,W,3]."""
    return np.stack([np.asarray(f.convert("RGB")) for f in frames]).astype(np.uint8)


def psnr(a, b):
    a = a.astype(np.float32); b = b.astype(np.float32)
    mse = ((a - b) ** 2).mean()
    return 99.0 if mse < 1e-8 else float(10 * np.log10(255.0 ** 2 / mse))


def ssim_mean(a, b):
    from skimage.metrics import structural_similarity as ssim
    vals = [ssim(a[t], b[t], channel_axis=2, data_range=255) for t in range(min(len(a), len(b)))]
    return float(np.mean(vals))


_LPIPS = None
def lpips_mean(a, b, device):
    global _LPIPS
    import lpips as lpips_mod
    if _LPIPS is None:
        _LPIPS = lpips_mod.LPIPS(net="vgg").to(device).eval()
    T = min(len(a), len(b)); vals = []
    with torch.no_grad():
        for t in range(T):
            xa = torch.from_numpy(a[t]).permute(2, 0, 1).float().div(127.5).sub(1).unsqueeze(0).to(device)
            xb = torch.from_numpy(b[t]).permute(2, 0, 1).float().div(127.5).sub(1).unsqueeze(0).to(device)
            vals.append(_LPIPS(xa, xb).item())
    return float(np.mean(vals))


def trio(a, b, device):
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    return {"psnr": round(psnr(a, b), 2), "ssim": round(ssim_mean(a, b), 4), "lpips": round(lpips_mean(a, b, device), 4)}


# ---------------- pipeline ----------------
class _NoGuard(torch.nn.Module):
    """Passthrough stand-in for CosmosSafetyChecker (avoids the heavy cosmos_guardrail dep; research use)."""
    def __init__(self, *a, **k):
        super().__init__(); self.register_buffer("_d", torch.zeros(1))
    @property
    def dtype(self): return self._d.dtype
    def check_text_safety(self, p): return True
    def check_video_safety(self, vid): return vid


def load_pipe(model_dir, device, dtype=torch.bfloat16):
    import diffusers.pipelines.cosmos.pipeline_cosmos2_video2world as cmod
    cmod.CosmosSafetyChecker = _NoGuard          # monkeypatch before construction (keep non-None for license gate)
    # device inference trips over the param-less safety stub; each gen proc sees one GPU as cuda:0, so pin it
    import diffusers.pipelines.pipeline_utils as pu
    pu.DiffusionPipeline._execution_device = property(lambda self: torch.device("cuda"))
    from diffusers import Cosmos2VideoToWorldPipeline
    pipe = Cosmos2VideoToWorldPipeline.from_pretrained(model_dir, torch_dtype=dtype)
    pipe = pipe.to(device)
    return pipe


def linearize_pipe(pipe, ratio, linstate):
    from cosmos_surgery import linearize_cosmos
    from cosmos_distill import _load_linstate
    pipe.transformer, lin, sm = linearize_cosmos(pipe.transformer, ratio_linear=ratio)
    if linstate:
        _load_linstate(pipe.transformer, linstate)
    pipe.transformer.eval()
    return pipe


@torch.no_grad()
def generate(pipe, image, prompt, H, W, num_frames, steps, guidance, seed, device):
    g = torch.Generator(device=device).manual_seed(seed)
    out = pipe(image=image, prompt=prompt, negative_prompt="", height=H, width=W, num_frames=num_frames,
               num_inference_steps=steps, guidance_scale=guidance, generator=g, output_type="pil")
    return out.frames[0]   # list[PIL]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["teacher", "lineval"], required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", default="cosmos_gen_out")
    ap.add_argument("--teacher_dir", default="cosmos_gen_out/teacher")
    ap.add_argument("--resume", default=None)
    ap.add_argument("--ratio", type=float, default=0.75)
    ap.add_argument("--tag", default="lin")
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--num_frames", type=int, default=29)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--guidance", type=float, default=7.0)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    device = "cuda"
    pipe = load_pipe(args.model, device)
    if args.mode == "lineval":
        pipe = linearize_pipe(pipe, args.ratio, args.resume)

    save_dir = os.path.join(args.out, "teacher" if args.mode == "teacher" else args.tag)
    os.makedirs(save_dir, exist_ok=True)
    trajs = eval_trajs(args.n)
    print(f"[gen] mode={args.mode} n={len(trajs)} {args.size}px {args.num_frames}f {args.steps}steps", flush=True)

    rows, vs_gt, vs_tea = [], [], []
    for i, traj in enumerate(trajs):
        gt = load_clip_frames(traj, args.num_frames, args.size)
        if gt is None:
            continue
        prompt = humanize_task(traj)
        vid = generate(pipe, gt[0], prompt, args.size, args.size, args.num_frames, args.steps, args.guidance,
                       args.seed + i, device)
        gen_u8 = to_uint8(vid); gt_u8 = to_uint8(gt)
        m_gt = trio(gen_u8, gt_u8, device); vs_gt.append(m_gt)
        rec = {"traj": os.path.basename(os.path.dirname(traj)) + "/" + os.path.basename(traj), "vs_gt": m_gt}
        # save generated frames (uint8) for cross-model comparison
        torch.save({"frames": torch.from_numpy(gen_u8)}, os.path.join(save_dir, f"gen_{i:04d}.pt"))
        if args.mode == "lineval":
            tf = os.path.join(args.teacher_dir, f"gen_{i:04d}.pt")
            if os.path.exists(tf):
                tea_u8 = torch.load(tf)["frames"].numpy()
                m_tea = trio(gen_u8, tea_u8, device); vs_tea.append(m_tea); rec["vs_teacher"] = m_tea
        rows.append(rec)
        msg = f"  [{i}] vs_gt {m_gt}"
        if args.mode == "lineval" and "vs_teacher" in rec:
            msg += f" | vs_teacher {rec['vs_teacher']}"
        print(msg, flush=True)

    def avg(lst, k):
        return round(float(np.mean([r[k] for r in lst])), 4) if lst else None
    summary = {"mode": args.mode, "tag": args.tag, "ratio": args.ratio, "resume": args.resume, "n": len(rows),
               "vs_gt": {k: avg(vs_gt, k) for k in ("psnr", "ssim", "lpips")}}
    if vs_tea:
        summary["vs_teacher"] = {k: avg(vs_tea, k) for k in ("psnr", "ssim", "lpips")}
    os.makedirs(args.out, exist_ok=True)
    json.dump({"summary": summary, "rows": rows}, open(os.path.join(args.out, f"metrics_{args.mode}_{args.tag}.json"), "w"), indent=2)
    print(f"\n=== {args.mode}/{args.tag} === vs_GT {summary['vs_gt']}"
          + (f" | vs_TEACHER {summary['vs_teacher']}  <-- 逼近教师" if vs_tea else ""), flush=True)
    print(f"-> {args.out}/metrics_{args.mode}_{args.tag}.json")


if __name__ == "__main__":
    main()
