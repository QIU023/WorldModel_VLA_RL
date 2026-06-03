"""(task#12 step2) OpenVLA-OFT 在 Ultra-Long 上的 baseline —— 坐实 0% headroom + 作 A2 蒸馏教师。
复用 OFT 官方推理(initialize_model/get_action/prepare_observation/process_action)+ 我们跑通的
ultra_long harness(LiLo bddl + closed predicate + init_states)。unnorm_key=libero_10(模型训练统计)。
"""
import os, sys, json, numpy as np, torch
import faulthandler; faulthandler.enable()
from collections import deque
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("HF_HOME", "/home/.hf_home")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")  # protobuf 4.x + tf2.15 兼容
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
# tf(被 OFT 拉进来做图像预处理)别抢 GPU —— 否则和 torch 7B + mujoco EGL 争 GPU → core dump
try:
    import tensorflow as _tf
    _tf.config.set_visible_devices([], "GPU")
except Exception:
    pass
sys.path.insert(0, "/home/wm-vla-rl/_ext/LIBERO")
sys.path.insert(0, "/home/wm-vla-rl/_ext/openvla-oft")

# --- patches: torch.load weights_only（LiLo init_states）+ closed predicate ---
_ol = torch.load
def _pl(*a, **k): k.setdefault("weights_only", False); return _ol(*a, **k)
torch.load = _pl
import lilo_vla.benchmark
from libero.libero.envs.predicates import VALIDATE_PREDICATE_FN_DICT
from libero.libero.envs.predicates.base_predicates import Close
VALIDATE_PREDICATE_FN_DICT.setdefault("closed", Close())

from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv
from experiments.robot.libero.run_libero_eval import GenerateConfig, initialize_model, prepare_observation, process_action
from experiments.robot.libero.libero_utils import get_libero_dummy_action
from experiments.robot.robot_utils import get_action, get_image_resize_size

BENCH = os.environ.get("EVAL_BENCH", "ultra_long")
NEP = int(os.environ.get("EVAL_EP", "5"))
HMAX = int(os.environ.get("EVAL_H", "1000"))             # 超长任务需很多步
RES = "/home/wm-vla-rl/module1_vjepa2ac/results"
CKPT_HF = os.environ.get("OFT_CKPT", "moojink/openvla-7b-oft-finetuned-libero-10")


def main():
    cfg = GenerateConfig(
        pretrained_checkpoint=CKPT_HF, model_family="openvla",
        task_suite_name="libero_10", unnorm_key="libero_10",   # 模型训练统计
        center_crop=True, use_proprio=True, num_images_in_input=2,
        use_l1_regression=True, num_open_loop_steps=8, num_steps_wait=10,
    )
    print(f"[OFT] loading {CKPT_HF} ...", flush=True)
    model, action_head, proprio_projector, noisy_action_projector, processor = initialize_model(cfg)
    resize_size = get_image_resize_size(cfg)
    print("[OFT] loaded.", flush=True)

    bench = benchmark.get_benchmark_dict()[BENCH]()
    n = bench.n_tasks
    print(f"OFT ULTRALONG | {BENCH} ntask={n} ep={NEP} Hmax={HMAX}", flush=True)
    results = {}
    for ti in range(n):
        task = bench.get_task(ti)
        bddl = bench.get_task_bddl_file_path(ti)
        init_states = bench.get_task_init_states(ti)
        print(f"  [task{ti}] creating env...", flush=True)
        env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256)
        env.seed(0)
        succ = 0
        for ep in range(NEP):
            env.reset()
            obs = env.set_init_state(init_states[ep % len(init_states)])
            for _ in range(cfg.num_steps_wait):
                obs, _, _, _ = env.step(get_libero_dummy_action(cfg.model_family))
            q = deque(maxlen=cfg.num_open_loop_steps)
            ok = False
            for t in range(HMAX):
                observation, _ = prepare_observation(obs, resize_size)
                if len(q) == 0:
                    acts = get_action(cfg, model, observation, task.language, processor=processor,
                                      action_head=action_head, proprio_projector=proprio_projector,
                                      noisy_action_projector=noisy_action_projector, use_film=cfg.use_film)
                    q.extend(acts)
                action = process_action(q.popleft(), cfg.model_family)
                obs, _, done, _ = env.step(action.tolist())
                if env.check_success(): ok = True; break
                if done: break
            succ += int(ok)
            print(f"    task{ti} ep{ep}: {'SUCCESS' if ok else 'fail'}", flush=True)
        env.close()
        sr = succ / NEP; results[f"task{ti}"] = sr
        print(f"  task{ti} {task.name[:46]:46s} success={sr:.2f} ({succ}/{NEP})", flush=True)
    overall = float(np.mean(list(results.values())))
    json.dump(dict(overall=overall, bench=BENCH, ep=NEP, ckpt=CKPT_HF, **results), open(f"{RES}/oft_ultralong.json", "w"))
    print(f"OFT_ULTRALONG_DONE overall={overall:.3f} ({BENCH}, {NEP}ep, {n}tasks)", flush=True)


if __name__ == "__main__":
    main()
