"""ARR-style probe (ARL2): on the frozen SDPA teacher, measure per-layer how much the ACTION-token
query positions attend to the IMAGE/observation key positions. Layers where action attends strongly
to the observation = doing the observation->action retrieval = action-critical = should stay softmax
(NOT linearized). This is the ARL2 layer-selection principle (linearize only recoverable layers).
Output: per-layer action->image attention mass, ranked. We keep the top-K as softmax.
"""
import os, sys, pickle, glob
import torch

sys.path.insert(0, "/home/wm-vla-rl/_ext/WorldVLA/worldvla")
sys.path.insert(0, "/home/wm-vla-rl/_ext/WorldVLA")
from model import ChameleonXLLMXForConditionalGeneration_ck

SOTA = "/home/wm-vla-rl/_ext/worldvla_ckpts/model_256/libero_10"
ACTION_START, ACTION_END = 10004, 15004
dev = "cuda:0"

# eager attention so output_attentions returns the [B,H,T,T] weights (sdpa is fused, no weights)
model = ChameleonXLLMXForConditionalGeneration_ck.from_pretrained(
    SOTA, torch_dtype=torch.bfloat16, device_map="cpu", attn_implementation="eager",
    mask_image_logits=False,
)
if hasattr(model.model, "vqmodel"):
    del model.model.vqmodel
model = model.to(dev).eval()
image_tok = set(model.model.vocabulary_mapping.image_tokens)
nl = model.config.num_hidden_layers
print(f"layers={nl}, #image-vocab={len(image_tok)}")

# collect a few tokenized records
recs = sorted(glob.glob("/home/wm-vla-rl/_ext/WorldVLA/worldvla/processed_data/libero_10_his_1_train_a2i_256/files/*.pkl"))[:8]
print(f"probing {len(recs)} records")

act0_to_img = torch.zeros(nl)        # action_0 query -> image keys, per layer
actall_to_img = torch.zeros(nl)      # all action queries -> image keys, per layer
n = 0
for rp in recs:
    try:
        d = pickle.load(open(rp, "rb"))
        toks = torch.tensor(d["token"], dtype=torch.long, device=dev).unsqueeze(0)
        T = toks.shape[1]
        is_img = torch.tensor([int(t) in image_tok for t in d["token"]], device=dev)
        # action value tokens are strictly between the 10004/15004 markers
        tl = d["token"]
        act_pos = []
        in_act = False
        for i, t in enumerate(tl):
            if t == ACTION_START: in_act = True; continue
            if t == ACTION_END: in_act = False; continue
            if in_act: act_pos.append(i)
        if not act_pos or is_img.sum() == 0:
            continue
        act0 = act_pos[0]
        with torch.no_grad():
            out = model.model(input_ids=toks, output_attentions=True, use_cache=False)
        atts = out.attentions  # tuple len nl of [1, H, T, T]
        img_idx = torch.where(is_img)[0]
        for l in range(nl):
            a = atts[l][0].float().mean(0)        # [T,T] head-avg attention
            act0_to_img[l] += a[act0, img_idx].sum().item()
            actall_to_img[l] += a[act_pos][:, img_idx].sum(-1).mean().item()
        n += 1
    except Exception as e:
        print("skip", rp, e)

act0_to_img /= max(n, 1); actall_to_img /= max(n, 1)
print(f"\n=== per-layer action->image attention mass (n={n} records) ===")
order = torch.argsort(actall_to_img, descending=True)
for rank, l in enumerate(order.tolist()):
    print(f"layer {l:2d}: act0->img {act0_to_img[l]:.4f}  actALL->img {actall_to_img[l]:.4f}  {'<-- top' if rank < 8 else ''}")
# Kimi 3:1 currently linearizes layers where (i+1)%4 != 0 (keeps 3,7,11,...,31 as softmax)
kept_softmax = sorted([i for i in range(nl) if (i + 1) % 4 == 0])
top_critical = sorted(order[:8].tolist())
print(f"\nKimi3:1 keeps softmax: {kept_softmax}")
print(f"Top-8 action-critical layers (by actALL->img): {top_critical}")
print(f"Action-critical layers CURRENTLY linearized (should stay softmax): {sorted(set(top_critical)-set(kept_softmax))}")

print("\n=== ARR-style selection: linearize ONLY the lowest action->image layers ===")
asc = torch.argsort(actall_to_img)  # ascending = most linearizable first
for keep_soft in [12, 16]:
    lin = sorted(asc[:nl-keep_soft].tolist())
    soft = sorted(asc[nl-keep_soft:].tolist())
    print(f"keep {keep_soft} softmax (action-critical): {soft}")
    print(f"  -> linearize (KDA) {len(lin)} layers: {','.join(map(str,lin))}")
