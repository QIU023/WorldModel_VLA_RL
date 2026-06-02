"""LIBERO import smoke：预写 ~/.libero/config.yaml（绕过首次 input() 提示）后 import。"""
import os, yaml, importlib.util
os.environ.setdefault("MUJOCO_GL", "egl")
cfg = os.path.expanduser("~/.libero/config.yaml")
if not os.path.exists(cfg):
    spec = importlib.util.find_spec("libero.libero")     # 不执行 libero.libero.__init__（避开 prompt）
    root = os.path.dirname(spec.origin)
    d = {"benchmark_root": root,
         "bddl_files": os.path.join(root, "bddl_files"),
         "init_states": os.path.join(root, "init_files"),
         "datasets": os.path.join(root, "../datasets"),
         "assets": os.path.join(root, "assets")}
    os.makedirs(os.path.dirname(cfg), exist_ok=True)
    yaml.dump(d, open(cfg, "w"))
    print("wrote config:", d)
from libero.libero import benchmark, get_libero_path
print("benchmarks:", list(benchmark.get_benchmark_dict().keys())[:6])
print("bddl exists:", os.path.exists(get_libero_path("bddl_files")))
print("init_states exists:", os.path.exists(get_libero_path("init_states")))
print("LIBERO_IMPORT_OK")
