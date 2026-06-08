"""Unified frame-sequence dataset interface for the Cosmos linearization pipeline. Lets cosmos_data.py /
cosmos_generate.py read EITHER LIBERO robot video OR DAVIS general natural video via a --dataset flag, without
duplicating glob/prompt logic. DAVIS = the general-domain track (Cosmos's native strength) for SANA-comparable
quality; LIBERO = the robotics application track.
"""
import os, glob

LIBERO_IMG = "/home/wm-vla-rl/_ext/WorldVLA/worldvla/processed_data/libero_10_img_action_256"
DAVIS_ROOT = "/home/DAVIS/JPEGImages/480p"


def _libero_humanize(seq):
    name = os.path.basename(os.path.dirname(seq))            # KITCHEN_SCENE8_put_both_...
    parts = name.split("_")
    while parts and (parts[0].isupper() or parts[0][:5].upper() == "SCENE" or parts[0].isdigit()
                     or parts[0][-1].isdigit()):
        parts.pop(0)
    desc = " ".join(parts).strip()
    return f"A robot arm performs a manipulation task: {desc}." if desc else "A robot arm manipulation video."


def _davis_humanize(seq):
    name = os.path.basename(seq.rstrip("/")).replace("-", " ").replace("_", " ").strip()
    return f"A natural video of {name}." if name else "A natural video scene."


def list_sequences(dataset, split="train", n_eval=16):
    """Return sequence dirs. LIBERO: .../trj_* with imgs/. DAVIS: JPEGImages/480p/<seq>/."""
    if dataset == "libero":
        seqs = sorted(glob.glob(os.path.join(LIBERO_IMG, "*", "trj_*")))
        seqs = [s for s in seqs if os.path.isdir(os.path.join(s, "imgs"))]
        n_eval = 32
    elif dataset == "davis":
        seqs = sorted(d for d in glob.glob(os.path.join(DAVIS_ROOT, "*")) if os.path.isdir(d))
    else:
        raise ValueError(dataset)
    return seqs[:-n_eval] if split == "train" else seqs[-n_eval:]


def _frame_paths(dataset, seq):
    if dataset == "libero":
        return sorted(glob.glob(os.path.join(seq, "imgs", "image_*.png")),
                      key=lambda p: int(p.split("_")[-1].split(".")[0]))
    return sorted(glob.glob(os.path.join(seq, "*.jpg")))   # DAVIS 00000.jpg... already zero-padded


def num_frames_avail(dataset, seq):
    return len(_frame_paths(dataset, seq))


def load_frames(dataset, seq, num_frames, size, start=0):
    """Load `num_frames` frames from `start` -> list[PIL RGB size×size]. LIBERO is already square; DAVIS 480p
    is center-cropped to square then resized. Returns None if too short. start=0 = deterministic (eval)."""
    from PIL import Image
    fs = _frame_paths(dataset, seq)
    if len(fs) < start + num_frames:
        return None
    out = []
    for p in fs[start:start + num_frames]:
        im = Image.open(p).convert("RGB")
        if dataset == "davis":
            w, h = im.size; s = min(w, h)
            im = im.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))   # center square
        out.append(im.resize((size, size)))
    return out


def prompt_for(dataset, seq):
    return _libero_humanize(seq) if dataset == "libero" else _davis_humanize(seq)
