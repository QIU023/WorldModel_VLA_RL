"""Thin driver around the OFFICIAL VBench package (vbench 0.1.5) — no custom metrics.
Scores a folder of generated videos with `VBench.evaluate(..., mode='vbench_standard')`,
the exact protocol ARL2 used (full 946-prompt official benchmark), so numbers are comparable.

Videos must be named '{full_prompt}-0.mp4' (vbench_standard convention) — produced by
Causal-Forcing/inference.py --vbench_naming.

Usage:
  python vbench_score.py <videos_dir> <name> <output_dir> [dim1,dim2,...|all]
The official VBench_full_info.json is auto-located inside the installed vbench package.
"""
import sys, os, json, torch
import vbench
from vbench import VBench

videos_dir = sys.argv[1]
name       = sys.argv[2]
output_dir = sys.argv[3]
dims_arg   = sys.argv[4] if len(sys.argv) > 4 else "all"

FULL_INFO = os.path.join(os.path.dirname(vbench.__file__), "VBench_full_info.json")
os.makedirs(output_dir, exist_ok=True)

device = torch.device("cuda")
vb = VBench(device, FULL_INFO, output_dir)

# dimension_list=None -> VBench.build_full_dimension_list() -> all 16 standard dimensions
dimension_list = None if dims_arg == "all" else dims_arg.split(",")

print(f"[vbench] scoring '{name}'  videos={videos_dir}  dims={dims_arg}  mode=vbench_standard", flush=True)
vb.evaluate(
    videos_path=videos_dir,
    name=name,
    dimension_list=dimension_list,
    mode="vbench_standard",   # official: uses VBench_full_info.json, expects '{prompt}-{i}.mp4'
)
# VBench writes <output_dir>/<name>_eval_results.json
res_path = os.path.join(output_dir, f"{name}_eval_results.json")
if os.path.exists(res_path):
    r = json.load(open(res_path))
    print(f"[vbench] {name} per-dimension:")
    for k, v in r.items():
        val = v[0] if isinstance(v, list) else v
        print(f"    {k:28} {val}")
print(f"[vbench] done -> {res_path}", flush=True)
