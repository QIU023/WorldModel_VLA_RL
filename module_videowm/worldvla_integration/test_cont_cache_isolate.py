"""ISOLATE cache+mask correctness from open-loop feedback divergence. Build ONE fixed token sequence
(prompt + N action blocks with FIXED dummy dim tokens) and compare the cont-head action read at every
action_start position via:
  (GT)  a single full-seq forward over the WHOLE sequence with generate_att_mask_3 (training=True path)
        — causal => reading action_start_k here == the per-chunk slow decode (future tokens don't matter).
  (CACHE) prefill + incremental cached decode feeding the SAME fixed dims (teacher-forced, no feedback).
If these match within ~bf16 noise at EVERY block, the masking replicates generate_att_mask_3 and the KDA
cache is correct; any eval-time drift is then purely open-loop feedback amplification, not a bug.

Run: /home/venv/libero/bin/python test_cont_cache_isolate.py
"""
import os, sys, glob
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
sys.path.insert(0, "/home/wm-vla-rl/module_videowm")
sys.path.insert(0, "/home/wm-vla-rl/_ext/WorldVLA/worldvla")
sys.path.insert(0, "/home/wm-vla-rl/_ext/WorldVLA")
import numpy as np
import torch
from PIL import Image
from transformers import DynamicCache
from model.modeling_xllmx_chameleon_ck import ChameleonXLLMXForConditionalGeneration_ck
from kdaify_chameleon import kdaify_chameleon, attnres_ify_chameleon, attach_cont_head
from data.pre_tokenize_action import ItemProcessor

DEV = "cuda:0"
CK = "/home/wm-vla-rl/_ext/WorldVLA/worldvla/exps_256_all/output_libero_256/kda_75_s2conthead_MERGED"
ARR = "3,5,6,8,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30"
N = 10


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
    sd = {k: v.to(torch.bfloat16) for k, v in sd.items()}
    m.load_state_dict(sd, strict=False)
    return m.to(DEV, torch.bfloat16).eval()


@torch.no_grad()
def main():
    m = load_merged()
    ip = ItemProcessor(target_size=256, pretrained_checkpoint=CK)
    base, head = m.model, m.cont_action_head
    _hd = next(head.parameters()).dtype
    ACT_START = ip.token2id(ip.action_start_token)
    ACT_END = ip.token2id(ip.action_end_token)

    def mkimg(seed):
        a = np.zeros((256, 256, 3), np.uint8); r = np.random.RandomState(seed)
        for _ in range(40):
            x, y = r.randint(0, 220, 2); w, h = r.randint(8, 36, 2); a[y:y+h, x:x+w] = r.randint(0, 255, 3)
        return Image.fromarray(a)
    val = "What action should the robot take to pick up the bowl?" + "<|image|>" + "<|image|>"
    conv = {"conversations": [{"from": "human", "value": val}], "image": [mkimg(2), mkimg(1)], "action": []}
    prompt = ip.process_item(conv, training_mode=False)
    P = len(prompt)

    # fixed dummy dims per block (valid action-token range = ACT_START+1 .. ACT_START+256)
    rng = np.random.RandomState(7)
    dims_per_block = [(ACT_START + 1 + rng.randint(0, 256, size=7)).tolist() for _ in range(N)]

    # ---- build full sequence + record action_start positions ----
    seq = list(prompt); act_pos = []
    for k in range(N):
        act_pos.append(len(seq)); seq.append(ACT_START)
        seq += dims_per_block[k] + [ACT_END]
    # drop trailing dims+end after the last action_start is unnecessary; keep full structure (causal => fine)

    # ---- GT: single full-seq forward (training=True uses generate_att_mask_3 + use_cache=False) ----
    for mod in m.modules():
        if hasattr(mod, "_kda_state"):
            mod._kda_state = None
    m(input_ids=[seq], labels=[[-100] * len(seq)], output_hidden_states=True, training=True, att_mask=True)
    gt = m._cont_preds[0, act_pos].float().cpu().numpy()    # [N,7] cont head at each action_start

    # ---- CACHE: prefill + incremental, teacher-forced identical dims, replicate generate_att_mask_3 ----
    for mod in m.modules():
        if hasattr(mod, "_kda_state"):
            mod._kda_state = None
    pkv = DynamicCache()
    dtype = next(base.parameters()).dtype; min_dtype = torch.finfo(dtype).min

    def mask4d(q_pos, q_blk, all_blk):
        kv = all_blk.shape[0]; key_idx = torch.arange(kv, device=DEV)
        attend = (key_idx[None, :] <= q_pos[:, None]) & ((all_blk[None, :] == -1) | (all_blk[None, :] == q_blk[:, None]))
        mm = torch.zeros((q_pos.shape[0], kv), dtype=dtype, device=DEV); mm.masked_fill_(~attend, min_dtype)
        return mm[None, None]

    ch = []
    seq0 = list(prompt) + [ACT_START]
    all_blk = torch.tensor([-1] * P + [0], device=DEV); cur = len(seq0)
    qp = torch.arange(cur, device=DEV)
    out = base(input_ids=torch.tensor([seq0], device=DEV), attention_mask=mask4d(qp, all_blk, all_blk),
               past_key_values=pkv, use_cache=True, cache_position=qp, output_hidden_states=True)
    ch.append(head(out.last_hidden_state[:, -1].to(_hd))[0].float().cpu().numpy())
    for k in range(1, N):
        appended = dims_per_block[k - 1] + [ACT_END, ACT_START]
        q_blk = torch.tensor([k - 1] * 8 + [k], device=DEV); all_blk = torch.cat([all_blk, q_blk])
        qp = torch.arange(cur, cur + 9, device=DEV)
        out = base(input_ids=torch.tensor([appended], device=DEV), attention_mask=mask4d(qp, q_blk, all_blk),
                   past_key_values=pkv, use_cache=True, cache_position=qp, output_hidden_states=True)
        ch.append(head(out.last_hidden_state[:, -1].to(_hd))[0].float().cpu().numpy()); cur += 9
    ch = np.stack(ch)

    d = np.abs(gt - ch)
    print(f"\n[isolate: TEACHER-FORCED identical tokens, N={N} blocks, P={P}]")
    print("[per-block max|Δ| over 7 dims]", np.round(d.max(1), 4).tolist())
    print(f"[overall] max|Δ|={d.max():.4e} mean|Δ|={d.mean():.4e} rel={d.mean()/max(np.abs(gt).mean(),1e-6):.4%}")
    print(f"[block 0 gt ] {np.round(gt[0],4).tolist()}")
    print(f"[block 0 ch ] {np.round(ch[0],4).tolist()}")
    print(f"[block {N-1} gt ] {np.round(gt[-1],4).tolist()}")
    print(f"[block {N-1} ch ] {np.round(ch[-1],4).tolist()}")
    ok = d.max() <= 0.02
    print(f"\n{'PASS' if ok else 'FAIL'}: mask+cache {'replicate generate_att_mask_3 within noise (drift is feedback-only)' if ok else 'DIVERGE on identical tokens => mask/cache BUG'}")


if __name__ == "__main__":
    main()
