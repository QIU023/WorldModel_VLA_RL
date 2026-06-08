"""cosmos_fvd.py — Fréchet Video Distance (the metric SANA-WM & Cosmos-Predict report) for teacher vs
linearized-distilled Cosmos-2B on DAVIS. Generates N videos (reusing the pipeline), extracts video features
with torchvision S3D (Kinetics-400 pretrained — a standard 3D-CNN video backbone, in lieu of the exact
I3D-torchscript), and computes the Fréchet distance vs a real-DAVIS reference set.

Reports:  FVD(teacher_gen ‖ real) = teacher quality   |   FVD(lin_gen ‖ real) = linearized quality
          FVD(lin_gen ‖ teacher_gen) = linearization gap in FVD terms.
Honest: small-N (≈96) FVD is noisier than the 2048-sample standard, and S3D≠exact-I3D, so this is a
RELATIVE teacher-vs-lin comparison (same backbone+N+protocol), not a cross-paper absolute number.

Run: CUDA_VISIBLE_DEVICES=0 python cosmos_fvd.py --model <snap> --resume <linstate> --ratio 0.75 --n 96
"""
import os, sys, glob, json, argparse, random
sys.path.insert(0, "/home/wm-vla-rl/_ext")
import numpy as np
import torch
from cosmos_frames import list_sequences, load_frames, num_frames_avail, prompt_for
from cosmos_generate import load_pipe, linearize_pipe, generate, to_uint8


# ---------------- S3D feature extractor ----------------
def load_s3d(device):
    from torchvision.models.video import s3d, S3D_Weights
    m = s3d(weights=S3D_Weights.KINETICS400_V1)
    m.classifier = torch.nn.Identity()          # features before the classifier head
    return m.to(device).eval()


@torch.no_grad()
def video_feats(videos_u8, model, device):
    """videos_u8: list of [T,H,W,3] uint8 arrays -> [N, D] S3D features. S3D wants [B,3,T,224,224] in [0,1]."""
    import torch.nn.functional as F
    feats = []
    for v in videos_u8:
        x = torch.from_numpy(v).float().div(255.0).permute(3, 0, 1, 2).unsqueeze(0).to(device)  # [1,3,T,H,W]
        x = F.interpolate(x, size=(x.shape[2], 224, 224), mode="trilinear", align_corners=False)
        feats.append(model(x).flatten(1).cpu())
    return torch.cat(feats, 0).numpy()


def frechet(f1, f2):
    from scipy import linalg
    mu1, mu2 = f1.mean(0), f2.mean(0)
    s1 = np.cov(f1, rowvar=False); s2 = np.cov(f2, rowvar=False)
    covmean = linalg.sqrtm(s1.dot(s2))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(((mu1 - mu2) ** 2).sum() + np.trace(s1 + s2 - 2 * covmean))


# ---------------- sampling real clips ----------------
def real_clips(dataset, n, num_frames, size, seed=0):
    rng = random.Random(seed)
    seqs = list_sequences(dataset, "train") + list_sequences(dataset, "eval")
    out = []
    tries = 0
    while len(out) < n and tries < n * 20:
        seq = rng.choice(seqs); tries += 1
        navail = num_frames_avail(dataset, seq)
        if navail < num_frames:
            continue
        start = rng.randint(0, navail - num_frames)
        fr = load_frames(dataset, seq, num_frames, size, start=start)
        if fr is not None:
            out.append(to_uint8(fr))
    return out


def gen_videos(pipe, dataset, n, num_frames, size, steps, guidance, seed):
    """Generate n videos conditioned on sampled real first-frames + prompts (window-sampled for diversity)."""
    rng = random.Random(seed)
    seqs = list_sequences(dataset, "train") + list_sequences(dataset, "eval")
    vids = []; i = 0; tries = 0
    while len(vids) < n and tries < n * 20:
        seq = rng.choice(seqs); tries += 1
        navail = num_frames_avail(dataset, seq)
        if navail < num_frames:
            continue
        start = rng.randint(0, navail - num_frames)
        fr = load_frames(dataset, seq, num_frames, size, start=start)
        if fr is None:
            continue
        v = generate(pipe, fr[0], prompt_for(dataset, seq), size, size, num_frames, steps, guidance, seed + i, "cuda")
        vids.append(to_uint8(v)); i += 1
        if i % 16 == 0:
            print(f"[fvd-gen] {i}/{n}", flush=True)
    return vids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--mode", choices=["teacher", "lin", "score"], required=True)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--ratio", type=float, default=0.75)
    ap.add_argument("--dataset", default="davis")
    ap.add_argument("--out", default="cosmos_fvd_out")
    ap.add_argument("--tag", default="teacher")
    ap.add_argument("--n", type=int, default=96)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--num_frames", type=int, default=29)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--guidance", type=float, default=7.0)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    if args.mode == "score":
        dev = "cuda"
        s3 = load_s3d(dev)
        def loadset(tag):
            d = torch.load(os.path.join(args.out, f"vids_{tag}.pt"))
            return [v.numpy() for v in d["videos"]]
        tea = loadset("teacher"); lin = loadset(args.tag)
        real = real_clips(args.dataset, max(len(tea), len(lin)), args.num_frames, args.size, seed=99)
        print(f"[fvd] feats: teacher={len(tea)} lin={len(lin)} real={len(real)}", flush=True)
        f_tea = video_feats(tea, s3, dev); f_lin = video_feats(lin, s3, dev); f_real = video_feats(real, s3, dev)
        out = {"n_teacher": len(tea), "n_lin": len(lin), "n_real": len(real),
               "FVD_teacher_vs_real": frechet(f_tea, f_real),
               "FVD_lin_vs_real": frechet(f_lin, f_real),
               "FVD_lin_vs_teacher": frechet(f_lin, f_tea)}
        print(f"\n=== FVD (S3D-K400, n≈{len(tea)}) ===")
        print(f"  teacher vs real : {out['FVD_teacher_vs_real']:.1f}")
        print(f"  lin     vs real : {out['FVD_lin_vs_real']:.1f}   <- linearized quality")
        print(f"  lin vs teacher  : {out['FVD_lin_vs_teacher']:.1f}   <- linearization gap")
        json.dump(out, open(os.path.join(args.out, f"fvd_{args.tag}.json"), "w"), indent=2)
        return

    pipe = load_pipe(args.model, "cuda")
    if args.mode == "lin":
        pipe = linearize_pipe(pipe, args.ratio, args.resume)
    print(f"[fvd] generating {args.n} {args.mode} videos on {args.dataset}", flush=True)
    vids = gen_videos(pipe, args.dataset, args.n, args.num_frames, args.size, args.steps, args.guidance, args.seed)
    # save uint8 videos for cross-set FVD later
    torch.save({"videos": [torch.from_numpy(v) for v in vids]}, os.path.join(args.out, f"vids_{args.tag}.pt"))
    print(f"[fvd] saved {len(vids)} videos -> {args.out}/vids_{args.tag}.pt", flush=True)


if __name__ == "__main__":
    main()
