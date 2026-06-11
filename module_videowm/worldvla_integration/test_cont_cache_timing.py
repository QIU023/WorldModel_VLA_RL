"""At the REAL eval setting (NCHUNKS=25): (1) measure cached vs full-seq wall-clock speedup, and
(2) establish the noise floor — run the full-seq path TWICE and the cached path TWICE end-to-end, to see
how much the open-loop discretized-feedback rollout diverges from ITSELF (bf16 chunk_kda nondeterminism),
vs cached-vs-full. If cached-vs-full ~ full-vs-full, the cached rollout is within the model's own
determinism floor and is safe to use.

Run: /home/venv/libero/bin/python test_cont_cache_timing.py
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
    get_action_Chameleon_cont_awm_ck, get_action_Chameleon_cont_awm_ck_cached)

DEV = "cuda:0"
CK = "/home/wm-vla-rl/_ext/WorldVLA/worldvla/exps_256_all/output_libero_256/kda_75_s2conthead_MERGED"
ARR = "3,5,6,8,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30"
HIS = "2h_1a_img_only"
NCHUNKS = 25


def load_merged():
    m = ChameleonXLLMXForConditionalGeneration_ck.from_pretrained(
        CK, max_position_embeddings=4096, mask_image_logits=False,
        dropout=0.0, z_loss_weight=1e-5, torch_dtype=torch.bfloat16, device_map="cpu")
    if hasattr(m.model, "vqmodel"):
        del m.model.vqmodel
    kdaify_chameleon(m, layers=set(int(x) for x in ARR.split(",")), init_from_softmax=False)
    attnres_ify_chameleon(m, groups=8)
    attach_cont_head(m)
    from safetensors.torch import load_file
    sd = {}
    for f in sorted(glob.glob(os.path.join(CK, "model-*.safetensors"))):
        sd.update(load_file(f))
    m.load_state_dict({k: v.to(torch.bfloat16) for k, v in sd.items()}, strict=False)
    return m.to(DEV, torch.bfloat16).eval()


@torch.no_grad()
def main():
    m = load_merged()
    ip = ItemProcessor(target_size=256, pretrained_checkpoint=CK)
    def mkimg(seed):
        a = np.zeros((256, 256, 3), np.uint8); r = np.random.RandomState(seed)
        for _ in range(40):
            x, y = r.randint(0, 220, 2); w, h = r.randint(8, 36, 2); a[y:y+h, x:x+w] = r.randint(0, 255, 3)
        return Image.fromarray(a)
    cur_img, his_img = mkimg(1), [mkimg(2)]
    task = "pick up the black bowl and place it on the plate"
    os.environ["CONT_NCHUNKS"] = str(NCHUNKS)

    def run(fn):
        torch.cuda.synchronize(); t = time.time()
        a = fn(m, cur_img, task, ip, his_img, [], HIS, 25)
        torch.cuda.synchronize()
        return np.stack([x.numpy() for x in a]), time.time() - t

    # warm
    run(get_action_Chameleon_cont_awm_ck); run(get_action_Chameleon_cont_awm_ck_cached)
    fullA, tf1 = run(get_action_Chameleon_cont_awm_ck)
    fullB, tf2 = run(get_action_Chameleon_cont_awm_ck)        # full-seq self-consistency
    cachA, tc1 = run(get_action_Chameleon_cont_awm_ck_cached)
    cachB, tc2 = run(get_action_Chameleon_cont_awm_ck_cached)  # cached self-consistency

    def stats(a, b):
        d = np.abs(a - b)
        return d.max(), d.mean(), d.mean() / max(np.abs(a).mean(), 1e-6), d.max(1)

    print(f"\n=== NCHUNKS={NCHUNKS}, P~566, his=2 imgs ===")
    print(f"[timing] full-seq {tf1:.2f}s, {tf2:.2f}s | cached {tc1:.2f}s, {tc2:.2f}s | "
          f"speedup={ (tf1+tf2)/max(tc1+tc2,1e-9):.1f}x")
    mx, mn, rel, perc = stats(fullA, fullB)
    print(f"[full-vs-full   ] max={mx:.4f} mean={mn:.4f} rel={rel:.3%}  (model bf16 determinism floor)")
    mx, mn, rel, perc = stats(cachA, cachB)
    print(f"[cache-vs-cache ] max={mx:.4f} mean={mn:.4f} rel={rel:.3%}")
    mx, mn, rel, perc = stats(fullA, cachA)
    print(f"[cache-vs-full  ] max={mx:.4f} mean={mn:.4f} rel={rel:.3%}")
    print(f"[cache-vs-full per-chunk max|Δ|] {np.round(perc,3).tolist()}")
    print(f"[full bf16 nondeterministic?] {'YES' if (np.abs(fullA-fullB).max()>1e-5) else 'NO (deterministic)'}")


if __name__ == "__main__":
    main()
