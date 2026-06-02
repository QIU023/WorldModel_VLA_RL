"""LIBERO 真·环境 smoke：headless EGL 建 env + reset + step + 取图。通则 sim 完全可用。"""
import os
os.environ.setdefault("MUJOCO_GL", "egl")
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

bench = benchmark.get_benchmark_dict()["libero_spatial"]()
task = bench.get_task(0)
bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
print("task:", task.name, "| bddl:", os.path.basename(bddl))
env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=128, camera_widths=128)
env.seed(0); env.reset()
import numpy as np
obs, r, done, info = env.step(np.zeros(7))
print("agentview_image:", obs["agentview_image"].shape, obs["agentview_image"].dtype,
      "| reward:", r, "| keys:", [k for k in obs if "image" in k or "state" in k][:4])
env.close()
print("LIBERO_ENV_OK")
