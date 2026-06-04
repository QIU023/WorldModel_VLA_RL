"""BAIR tfrecord → iVideoGPT .npz 预处理 —— 用轻量 tfrecord 包(纯 python,无 tensorflow,
不破坏 libero venv 的 fla/torch)。复刻 iVideoGPT/datasets/preprocess_bair.py 的逻辑:
每条 example 30 帧,{i}/image_main/encoded = raw RGB bytes(64x64x3,非JPEG),{i}/action = 4 floats。
存 npz: image[30,64,64,3] uint8 + action[30,4]。--max_traj 限子集(parity demo 不需全 43k)。
"""
import os, sys, glob, argparse, numpy as np
from tfrecord.reader import example_loader

ap = argparse.ArgumentParser()
ap.add_argument('--input_path', required=True)    # .../softmotion30_44k
ap.add_argument('--save_path', required=True)
ap.add_argument('--max_traj', type=int, default=0)  # 0=全部
ap.add_argument('--splits', default='train,test')
args = ap.parse_args()
T = 30


def feat_bytes(ex, name):
    v = ex[name]
    return v[0] if isinstance(v, (list, tuple)) else (v.tobytes() if hasattr(v, 'tobytes') else bytes(v))


for split in args.splits.split(','):
    data_dir = os.path.join(args.input_path, split)
    files = sorted(glob.glob(os.path.join(data_dir, '*.tfrecords')))
    out_dir = os.path.join(args.save_path, split)
    os.makedirs(out_dir, exist_ok=True)
    print(f"[{split}] {len(files)} tfrecord files -> {out_dir}", flush=True)
    idx = 0
    for fi, f in enumerate(files):
        try:
            it = example_loader(f, None)
        except Exception as e:
            print(f"  skip {f}: {e}", flush=True); continue
        while True:                                   # 截断 tar 的末个 tfrecord 不完整 → 优雅跳过
            try:
                ex = next(it)
            except StopIteration:
                break
            except Exception as e:
                print(f"  truncated record in {os.path.basename(f)}, stop file: {e}", flush=True); break
            imgs, acts = [], []
            ok = True
            for i in range(T):
                try:
                    b = ex[f'{i}/image_main/encoded']
                    b = b.tobytes() if hasattr(b, 'tobytes') else (b[0] if isinstance(b, (list, tuple)) else bytes(b))
                    img = np.frombuffer(b, dtype=np.uint8)
                    if img.size != 64 * 64 * 3:
                        ok = False; break
                    imgs.append(img.reshape(64, 64, 3))
                    a = ex[f'{i}/action']
                    a = np.asarray(a, dtype=np.float32).reshape(-1)[:4]
                    acts.append(a if a.size == 4 else np.zeros(4, np.float32))
                except Exception:
                    ok = False; break
            if not ok:
                continue
            np.savez_compressed(os.path.join(out_dir, f"traj_{idx:06d}.npz"),
                                image=np.stack(imgs).astype(np.uint8), action=np.stack(acts).astype(np.float32))
            idx += 1
            if idx % 500 == 0:
                print(f"  [{split}] {idx} trajectories", flush=True)
            if args.max_traj and idx >= args.max_traj:
                break
        if args.max_traj and idx >= args.max_traj:
            break
    print(f"[{split}] DONE {idx} npz", flush=True)
print("PREP_BAIR_DONE", flush=True)
