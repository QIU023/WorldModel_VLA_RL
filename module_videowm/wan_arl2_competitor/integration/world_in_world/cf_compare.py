"""Warm-timing comparison: 2-step vs 4-step vs 8-step denoising, one model load.
Swaps pipeline.denoising_step_list in place (replicating warp_denoising_step=true)."""
import sys, os, time
sys.argv = ["cf", "--smoke"]  # so module-level argv parse path is harmless; we call build_parser ourselves
import importlib.util
spec = importlib.util.spec_from_file_location(
    "cfmod", "/home/wm-vla-rl/_ext/world-in-world/downstream/api_models/causalforcing_model.py")
# avoid running __main__
import types
cf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cf)  # this will hit __main__? No: __name__ != '__main__' for imported module

import torch, numpy as np

parser = cf.build_parser()
args = parser.parse_args([])
args.world_model_name = "causalforcing"
args.device = "cuda:0"

t0 = time.time()
model = cf.CausalForcingModel(args=args)
print(f"[compare] model loaded in {time.time()-t0:.1f}s", flush=True)

# real first frame
import imageio.v2 as iio
rdr = iio.get_reader("/home/wan_distill_data_mp4/clip_00000.mp4")
frame0 = rdr.get_data(0); rdr.close()
b_image = np.transpose(frame0, (2,0,1))[None].astype(np.uint8)
input_dict = {
    "b_action": np.array([[0,1,2,1,3]], dtype=np.int64),
    "save_dirs": ["/tmp/causalforcing_smoke"],
    "request_model_name": "causalforcing",
    "b_image": b_image,
    "return_objects": [True],
}

def set_steps(pipe, steps_list, first_chunk_list=None):
    dsl = torch.tensor(steps_list, dtype=torch.long)
    ts = torch.cat((pipe.scheduler.timesteps.cpu(), torch.tensor([0], dtype=torch.float32)))
    pipe.denoising_step_list = ts[1000 - dsl]
    if first_chunk_list is not None:
        fcl = torch.tensor(first_chunk_list, dtype=torch.long)
        pipe.denoising_step_list_first_chunk = ts[1000 - fcl]
    else:
        pipe.denoising_step_list_first_chunk = None

def bench(label, n=4):
    times = []
    for i in range(n):
        out = model.inference_batch(input_dict)
        times.append(model.last_rollout_time)
    warm = times[1:]  # drop first (warmup)
    print(f"[compare] {label}: per-rollout (gpu) warm mean={sum(warm)/len(warm):.3f}s  "
          f"all={['%.3f'%t for t in times]}  pred={out['pred_frames'].shape}", flush=True)
    return sum(warm)/len(warm)

# 2-step (the shipped 2step config: 2 steps/block, first chunk 4 steps)
set_steps(model.pipe, [1000,500], [1000,750,500,250])
t2 = bench("2-step (first-chunk 4)")
# 4-step uniform (the dmd_framewise config)
set_steps(model.pipe, [1000,750,500,250], None)
t4 = bench("4-step uniform")
# 8-step uniform
set_steps(model.pipe, [1000,875,750,625,500,375,250,125], None)
t8 = bench("8-step uniform")

print(f"\n[compare] SPEEDUP 2-step vs 4-step: {t4/t2:.2f}x | 2-step vs 8-step: {t8/t2:.2f}x", flush=True)
print("=== COMPARE DONE ===", flush=True)
