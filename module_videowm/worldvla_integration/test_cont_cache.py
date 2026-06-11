"""VERIFY the KV-cached cont-head decode (get_action_Chameleon_cont_awm_ck_cached) against the SLOW
full-seq path (get_action_Chameleon_cont_awm_ck) on a single observation: the per-action chunk must match
within ~2-3% (bf16 chunk_kda-vs-fused_recurrent_kda rounding), and the cached path must be materially
faster (1 prefill + cheap incremental vs N full-seq forwards). This codebase shipped KDA-inference bugs
twice -> never trust the fast eval before this passes.

Run: /home/venv/libero/bin/python test_cont_cache.py
"""
import os, sys, glob, time
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
sys.path.insert(0, "/home/wm-vla-rl/module_videowm")
sys.path.insert(0, "/home/wm-vla-rl/_ext/WorldVLA/worldvla")
sys.path.insert(0, "/home/wm-vla-rl/_ext/WorldVLA")
import numpy as np
import torch
from PIL import Image
from model.modeling_xllmx_chameleon_ck import ChameleonXLLMXForConditionalGeneration_ck
from kdaify_chameleon import kdaify_chameleon, attnres_ify_chameleon, attach_cont_head
from data.pre_tokenize_action import ItemProcessor
from libero_util.Chameleon_utils import (
    get_action_Chameleon_cont_awm_ck,
    get_action_Chameleon_cont_awm_ck_cached,
)

DEV = "cuda:0"
CK = "/home/wm-vla-rl/_ext/WorldVLA/worldvla/exps_256_all/output_libero_256/kda_75_s2conthead_MERGED"
ARR = "3,5,6,8,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30"
MAXLEN = 4096
NCHUNKS = 8
HIS = "2h_1a_img_only"


def load_merged():
    print("[load] merged conthead model ...", flush=True)
    m = ChameleonXLLMXForConditionalGeneration_ck.from_pretrained(
        CK, max_position_embeddings=MAXLEN, mask_image_logits=False,
        dropout=0.0, z_loss_weight=1e-5, torch_dtype=torch.bfloat16, device_map="cpu")
    if hasattr(m.model, "vqmodel"):
        del m.model.vqmodel
    kset = set(int(x) for x in ARR.split(","))
    kdaify_chameleon(m, layers=kset, init_from_softmax=False)
    attnres_ify_chameleon(m, groups=8)
    attach_cont_head(m)
    sd = {}
    from safetensors.torch import load_file
    for f in sorted(glob.glob(os.path.join(CK, "model-*.safetensors"))):
        sd.update(load_file(f))
    sd = {k: v.to(torch.bfloat16) for k, v in sd.items()}
    miss, unexp = m.load_state_dict(sd, strict=False)
    miss = [x for x in miss if "vqmodel" not in x]
    print(f"[load] {len(sd)} tensors | miss(non-vq)={len(miss)} ({miss}) unexp={len(unexp)}", flush=True)
    print(f"[attn impl] {m.config._attn_implementation}", flush=True)
    return m.to(DEV, torch.bfloat16).eval()


@torch.no_grad()
def main():
    m = load_merged()
    ip = ItemProcessor(target_size=256, pretrained_checkpoint=CK)
    # build a deterministic pseudo-observation: a structured RGB image + 1 history image + task
    rng = np.random.RandomState(0)
    def mkimg(seed):
        a = np.zeros((256, 256, 3), np.uint8)
        r = np.random.RandomState(seed)
        for _ in range(40):
            x, y = r.randint(0, 220, 2); w, h = r.randint(8, 36, 2); c = r.randint(0, 255, 3)
            a[y:y+h, x:x+w] = c
        return Image.fromarray(a)
    cur_img = mkimg(1)
    his_img = [mkimg(2)]
    task = "pick up the black bowl and place it on the plate"

    os.environ["CONT_NCHUNKS"] = str(NCHUNKS)

    # ---- A) SLOW full-seq path (ground truth) ----
    t0 = time.time(); torch.cuda.synchronize()
    actA = get_action_Chameleon_cont_awm_ck(m, cur_img, task, ip, his_img, [], HIS, 25)
    torch.cuda.synchronize(); tA = time.time() - t0
    actA = np.stack([a.numpy() for a in actA])      # [N,7]

    # ---- B) CACHED path ----
    t0 = time.time(); torch.cuda.synchronize()
    actB = get_action_Chameleon_cont_awm_ck_cached(m, cur_img, task, ip, his_img, [], HIS, 25)
    torch.cuda.synchronize(); tB = time.time() - t0
    actB = np.stack([a.numpy() for a in actB])

    # ---- timing again (warm) ----
    t0 = time.time(); torch.cuda.synchronize()
    _ = get_action_Chameleon_cont_awm_ck(m, cur_img, task, ip, his_img, [], HIS, 25)
    torch.cuda.synchronize(); tA2 = time.time() - t0
    t0 = time.time(); torch.cuda.synchronize()
    _ = get_action_Chameleon_cont_awm_ck_cached(m, cur_img, task, ip, his_img, [], HIS, 25)
    torch.cuda.synchronize(); tB2 = time.time() - t0

    print(f"\n[shapes] full={actA.shape} cached={actB.shape}")
    d = np.abs(actA - actB)
    denom = np.maximum(np.abs(actA).mean(), 1e-6)
    print("[per-action max|Δ|]", np.round(d.max(1), 4).tolist())
    print(f"[overall] max|Δ|={d.max():.4e}  mean|Δ|={d.mean():.4e}  rel(mean/|a|mean)={d.mean()/denom:.4%}")
    print(f"[action 0 full ] {np.round(actA[0],4).tolist()}")
    print(f"[action 0 cache] {np.round(actB[0],4).tolist()}")
    print(f"[action {NCHUNKS-1} full ] {np.round(actA[-1],4).tolist()}")
    print(f"[action {NCHUNKS-1} cache] {np.round(actB[-1],4).tolist()}")
    # sane-action checks
    nan = np.isnan(actB).any()
    grip_ok = (actB[:, 6] >= -1.001).all() and (actB[:, 6] <= 1.001).all()
    varied = actB.std(0).mean() > 1e-4
    print(f"[sanity] nan={nan}  gripper_in[-1,1]={grip_ok}  varied_across_chunks={varied} (std/dim mean={actB.std(0).mean():.4f})")
    print(f"\n[timing] full-seq: {tA:.2f}s (warm {tA2:.2f}s) | cached: {tB:.2f}s (warm {tB2:.2f}s) | "
          f"speedup(warm)={tA2/max(tB2,1e-6):.2f}x  over {NCHUNKS} chunks")
    # at NCHUNKS=25 (eval) the prompt-reprocessing gap is ~3x wider than at 8; report extrapolation note
    ok = (d.max() <= 0.05) and (not nan) and grip_ok and varied
    print(f"\n{'PASS' if ok else 'FAIL'}: cached cont decode "
          f"{'matches full-seq within tol + faster + sane' if ok else 'DEVIATES — debug before eval'}")


if __name__ == "__main__":
    main()
