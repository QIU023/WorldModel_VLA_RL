"""Merge LoRA (base + B@A*scale) into base weights -> plain surgery-model ckpt the eval harness loads.
S2/S3 ckpts wrap KDA q/k/v/o + FFN in LoRALinear: keys X.base.weight / X.lora_A.weight / X.lora_B.weight.
eval_solver builds kdaify+attnres (NO LoRA) then load_state_dict(strict=False) over model-*.safetensors.
We fold LoRA in (scale=alpha/r=2.0) and rename X.base.weight -> X.weight; drop lora_A/lora_B.
Result: 859 -> 523 keys = the exact plain surgery tree (matches '523 tensors' the harness expects).
"""
import sys, os, glob, json
import torch
from safetensors.torch import load_file, save_file

SRC = sys.argv[1]
DST = sys.argv[2]
SCALE = float(sys.argv[3]) if len(sys.argv) > 3 else 2.0  # alpha/r = 64/32
os.makedirs(DST, exist_ok=True)

sd = {}
for f in sorted(glob.glob(os.path.join(SRC, "model-*.safetensors"))):
    sd.update(load_file(f))
print(f"loaded {len(sd)} keys from {SRC}")

out = {}
merged = 0
for k in list(sd.keys()):
    if k.endswith(".lora_A.weight") or k.endswith(".lora_B.weight"):
        continue  # consumed via their .base sibling
    if k.endswith(".base.weight"):
        pfx = k[: -len(".base.weight")]
        A = sd[pfx + ".lora_A.weight"].float()   # [r, in]
        B = sd[pfx + ".lora_B.weight"].float()   # [out, r]
        base = sd[k].float()                      # [out, in]
        w = base + (B @ A) * SCALE
        out[pfx + ".weight"] = w.to(torch.bfloat16)
        merged += 1
    else:
        out[k] = sd[k]
print(f"merged {merged} LoRA modules | out keys = {len(out)}")

# save as single shard matching glob model-*-of-*.safetensors + index.json
shard = "model-00001-of-00001.safetensors"
save_file(out, os.path.join(DST, shard), metadata={"format": "pt"})
total = sum(v.numel() * v.element_size() for v in out.values())
idx = {"metadata": {"total_size": total}, "weight_map": {k: shard for k in out}}
json.dump(idx, open(os.path.join(DST, "model.safetensors.index.json"), "w"))

# copy config + generation_config so from_pretrained works
import shutil
for fn in ("config.json", "generation_config.json"):
    src = os.path.join(SRC, fn)
    if os.path.exists(src):
        shutil.copy(src, os.path.join(DST, fn))
print(f"wrote merged ckpt -> {DST} ({total/1e9:.1f} GB)")
