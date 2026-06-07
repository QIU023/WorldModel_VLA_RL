"""Pre-bake RAW continuous action targets for the continuous-action-head Stage-2 (clean path).
For the his_2_img_only_ck_10 world-model source, each conv id predicts 10 future actions; conv['action']
is the list of 10 action npy paths. We load + normalize (same norm as item_processor) -> [N,10,7] in [-1,1],
indexed by the source pkl id (== conv index). Saved per split. The dataset looks these up by (source,pkl_id);
a2i source samples get no cont target (mask=0).
"""
import json, os, sys
import numpy as np

MIN = np.array([-0.9375, -0.9375, -0.9375, -0.32571429, -0.375, -0.375, -1.0])
MAX = np.array([0.9375, 0.9375, 0.9375, 0.375, 0.375, 0.375, 1.0])

def norm(a):
    n = 2 * (a - MIN) / (MAX - MIN + 1e-8) - 1
    return np.clip(n, -1, 1)

CONV_DIR = "/home/wm-vla-rl/_ext/WorldVLA/worldvla/processed_data/convs"
OUT_DIR = "/home/wm-vla-rl/_ext/WorldVLA/worldvla/processed_data/cont_actions"
os.makedirs(OUT_DIR, exist_ok=True)
SPLITS = {
    "train": "libero_10_his_2_train_img_only_ck_10_256.json",
    "val_ind": "libero_10_his_2_val_ind_img_only_ck_10_256.json",
    "val_ood": "libero_10_his_2_val_ood_img_only_ck_10_256.json",
}
CK = 10

for split, fn in SPLITS.items():
    path = os.path.join(CONV_DIR, fn)
    if not os.path.exists(path):
        print(f"[skip] {split}: {path} missing"); continue
    convs = json.load(open(path))
    N = len(convs)
    out = np.zeros((N, CK, 7), dtype=np.float32)
    bad = 0
    for i, c in enumerate(convs):
        paths = c["action"]
        if len(paths) != CK:
            bad += 1
        for j in range(min(CK, len(paths))):
            try:
                a = np.load(paths[j]).astype(np.float64).reshape(-1)[:7]
                out[i, j] = norm(a).astype(np.float32)
            except Exception as e:
                bad += 1
        if i % 10000 == 0:
            print(f"[{split}] {i}/{N}")
    np.save(os.path.join(OUT_DIR, f"cont_actions_{split}.npy"), out)
    print(f"[{split}] saved {out.shape} -> cont_actions_{split}.npy | bad={bad} | "
          f"range[{out.min():.3f},{out.max():.3f}] mean|x|={np.abs(out).mean():.3f}")
print("DONE prebake")
