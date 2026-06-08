"""cosmos_vbench.py — VBench (SANA-WM's headline metric) on our generated videos. Converts the FVD-generated
teacher/lin video sets (.pt frames) to mp4 and runs VBench custom_input on reference-free dimensions, comparing
teacher vs linearized. These dimensions score the video itself (no prompt/GT needed): subject/background
consistency, motion smoothness, dynamic degree, aesthetic & imaging quality, temporal flickering.

Run: CUDA_VISIBLE_DEVICES=0 python cosmos_vbench.py --vids cosmos_fvd_out/vids_teacher.pt --name teacher \
        --dims subject_consistency,motion_smoothness,aesthetic_quality,imaging_quality
"""
import os, sys, glob, json, argparse
sys.path.insert(0, "/home/wm-vla-rl/_ext")
import torch, numpy as np, imageio

# VBench's dimension checkpoints (AMT/RAFT/MUSIQ...) predate torch 2.6's weights_only=True default and contain
# typing.OrderedDict etc. They are trusted (official VBench releases) -> force weights_only=False.
_orig_load = torch.load
torch.load = lambda *a, **k: _orig_load(*a, **{**k, "weights_only": False})


def write_mp4s(pt_path, out_dir, fps=12, limit=None):
    os.makedirs(out_dir, exist_ok=True)
    vids = torch.load(pt_path)["videos"]
    if limit:
        vids = vids[:limit]
    for i, v in enumerate(vids):
        frames = v.numpy().astype("uint8")              # [T,H,W,3]
        imageio.mimwrite(os.path.join(out_dir, f"vid_{i:03d}.mp4"), frames, fps=fps,
                         codec="libx264", quality=8, macro_block_size=1)
    return len(vids), out_dir


def run_vbench(videos_dir, name, dims, out="vbench_out"):
    import vbench
    from vbench import VBench
    os.makedirs(out, exist_ok=True)
    full_info = os.path.join(os.path.dirname(vbench.__file__), "VBench_full_info.json")
    vb = VBench(torch.device("cuda"), full_info, out)
    vb.evaluate(videos_path=os.path.abspath(videos_dir), name=name, dimension_list=dims, mode="custom_input")
    res = os.path.join(out, f"{name}_eval_results.json")
    d = json.load(open(res))
    # VBench result format: {dim: [score, [per-video...]]}
    return {k: (v[0] if isinstance(v, list) else v) for k, v in d.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vids", required=True, help="vids_*.pt from cosmos_fvd.py")
    ap.add_argument("--name", required=True)
    ap.add_argument("--dims", default="subject_consistency,motion_smoothness,aesthetic_quality,imaging_quality")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--fps", type=int, default=12)
    args = ap.parse_args()
    mp4_dir = f"vbench_mp4/{args.name}"
    n, _ = write_mp4s(args.vids, mp4_dir, fps=args.fps, limit=args.limit)
    print(f"[vbench] wrote {n} mp4 -> {mp4_dir}", flush=True)
    dims = args.dims.split(",")
    scores = run_vbench(mp4_dir, args.name, dims)
    print(f"\n=== VBench {args.name} (n={n}) ===")
    for k, v in scores.items():
        print(f"  {k}: {v:.4f}")
    json.dump(scores, open(f"vbench_out/scores_{args.name}.json", "w"), indent=2)
    print(f"-> vbench_out/scores_{args.name}.json")


if __name__ == "__main__":
    main()
