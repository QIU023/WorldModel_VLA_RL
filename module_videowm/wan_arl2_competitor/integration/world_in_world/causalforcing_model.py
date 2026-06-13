"""
OUR linearized Causal-Forcing world model, wrapped as a World-in-World `text` / I2V WM.

What this is
------------
Causal-Forcing (Wan2.1-1.3B, autoregressive, frame-wise) made into OUR linearized student:
  * INTER-frame KDA (gated-delta) + INTRA-frame softmax on 75% of blocks (ratio=0.75)
  * naive cross-LAYER AttnRes (groups=4)
  * LoRA r=8 on Q/K/V/O(+FFN) of the linearized blocks
applied in-place to the standard generator, then OUR trained delta safetensors are loaded.
This is the exact surgery used in Causal-Forcing/inference.py:107-123, here driven through the
frame-wise I2V path (inference.py:134-200): first-frame PIL -> initial_latent -> pipeline.inference.

World-in-World contract
-----------------------
Subclasses `DiffuserModel`. The harness renders the action sequence to a TEXT prompt
(`process_b_action`), sends `{b_action, save_dirs, request_model_name, b_image}` to this worker,
and expects back `pred_frames` uint8 [B, T, C, H, W] (T == num_output_frames, ~14).
We override `generate_pipe_images(txt_list, img_list)` to do I2V per item (batch_size==1, the
frame-wise I2V constraint) and let the base class subsample/resize to (num_output_frames, H, W).

Env: runs in /home/venv/cosmos_predict2 (the env that already runs Causal-Forcing). The only
world-in-world deps added were `jaxtyping json_repair` (and `imageio`, already present).

Import ordering note
--------------------
Both repos ship a top-level `utils` package. world-in-world's `utils` has __init__.py (a REGULAR
package, which always wins over a namespace package regardless of sys.path order), while
Causal-Forcing's `utils` is a namespace package. So we MUST import Causal-Forcing's pipeline
(claiming the `utils` name) BEFORE importing world-in-world's `downstream`. We also do NOT import
world-in-world's `utils.logger` (would shadow CF's utils); a tiny local logger is used instead.
"""

import os
import os.path as osp
import sys
import argparse
import time

# --- paths ---
CF_DIR = os.environ.get("CF_DIR", "/home/wm-vla-rl/_ext/Causal-Forcing")
EXT_DIR = os.environ.get("EXT_DIR", "/home/wm-vla-rl/_ext")
WIW_DIR = osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))  # .../world-in-world

# Causal-Forcing's WanTextEncoder / WanVAEWrapper load RELATIVE paths (wan_models/..., configs/...),
# so the worker must run with cwd == CF_DIR.
os.chdir(CF_DIR)
for _p in (CF_DIR, EXT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image
from omegaconf import OmegaConf
from safetensors.torch import load_file

# Claim `utils` == Causal-Forcing's BEFORE world-in-world is imported (see module docstring).
from pipeline import CausalInferencePipeline           # noqa: E402  (imports CF's utils.wan_wrapper)
import wan_kda_surgery as kda_surgery                   # noqa: E402  (KDA + AttnRes + LoRA surgery)

# Now world-in-world (its `downstream` package; NOT its top-level `utils`).
if WIW_DIR not in sys.path:
    sys.path.insert(0, WIW_DIR)
from downstream.api_models import (                     # noqa: E402
    DiffuserModel,
    process_input_dict,
    process_output_dict,
    images_to_tensor,
)
from downstream.utils.worker_manager import worker_main  # noqa: E402

# Native generation resolution is FIXED: the pipeline hardcodes frame_seq_length=1560 (the
# patchified seq-len of a 60x104 latent == 480x832 pixels). Changing it would break the KV cache.
GEN_HEIGHT = 480
GEN_WIDTH = 832
LATENT_H = GEN_HEIGHT // 8   # 60
LATENT_W = GEN_WIDTH // 8    # 104
LATENT_C = 16


class CausalForcingModel(DiffuserModel):
    """OUR linearized Causal-Forcing student as a World-in-World text/I2V world model."""

    def _load_pipe(self):
        a = self.args
        device = torch.device(a.device)
        torch.set_grad_enabled(False)

        # Merge default + chosen config (cwd is CF_DIR; use absolute config dir to be safe).
        config = OmegaConf.load(a.config_path)
        default_config = OmegaConf.load(osp.join(CF_DIR, "configs", "default_config.yaml"))
        config = OmegaConf.merge(default_config, config)

        kda_surgery.patch_wan_flash_sdpa()  # route CF flash_attention -> torch SDPA (no flash-attn pkg)

        assert hasattr(config, "denoising_step_list"), \
            "config must define denoising_step_list (few-step Causal-Forcing)."
        assert getattr(config, "num_frame_per_block", 1) == 1, \
            "I2V requires the frame-wise model (num_frame_per_block == 1)."

        pipe = CausalInferencePipeline(config, device=device)

        # 1) base trained generator weights (causal_forcing.pt)
        if a.checkpoint_path:
            sd = torch.load(a.checkpoint_path, map_location="cpu")
            want = "generator_ema" if a.use_ema else "generator"
            if want not in sd:  # this base ckpt is EMA-only (only `generator_ema`); fall back
                alt = "generator_ema" if "generator_ema" in sd else ("generator" if "generator" in sd else None)
                assert alt is not None, f"checkpoint has no generator weights; keys={list(sd.keys())}"
                print(f"[causalforcing] key '{want}' absent; using '{alt}'", flush=True)
                want = alt
            gen_sd = sd[want]
            try:
                pipe.generator.load_state_dict(gen_sd)
            except RuntimeError:
                fixed = {}
                for k, v in gen_sd.items():
                    if k.startswith("model._fsdp_wrapped_module."):
                        k = k.replace("model._fsdp_wrapped_module.", "model.", 1)
                    fixed[k] = v
                pipe.generator.load_state_dict(fixed, strict=False)
            print(f"[causalforcing] loaded base generator from {a.checkpoint_path}", flush=True)

        # 2) OUR surgery on the standard generator, then load trained deltas
        if a.surgery_ckpt:
            m = pipe.generator.model  # CausalWanModel
            lin = kda_surgery.factorize_causal_wan(m, ratio=a.surgery_ratio)
            kda_surgery.attnres_ify_wan(m, groups=a.surgery_groups)
            kda_surgery.add_lora_wan(m, lin, r=a.surgery_lora_r, include_ffn=True)
            delta = load_file(a.surgery_ckpt)
            miss, unexp = m.load_state_dict(delta, strict=False)
            print(f"[causalforcing] surgery: {len(delta)} trained tensors -> generator "
                  f"(unexpected={len(unexp)}, missing-base={len(miss)}); "
                  f"ratio={a.surgery_ratio} groups={a.surgery_groups} lora_r={a.surgery_lora_r}",
                  flush=True)

        pipe = pipe.to(dtype=torch.bfloat16)
        pipe.text_encoder.to(device)
        pipe.generator.to(device)
        pipe.vae.to(device)

        self.device = device
        self._transform = transforms.Compose([
            transforms.Resize((GEN_HEIGHT, GEN_WIDTH)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        # last-call timing (seconds), populated per generate
        self.last_rollout_time = None
        return pipe

    def _load_pipe_args(self):
        # generate_pipe_images uses self.args directly; nothing extra needed here.
        return {}

    @torch.no_grad()
    def generate_pipe_images(self, txt_list, img_list):
        """I2V per item (batch_size == 1, the frame-wise constraint).
        Returns list[list[PIL.Image]]: outer over batch, inner over predicted frames."""
        a = self.args
        pipe_images = []
        t0 = time.time()
        for prompt, img in zip(txt_list, img_list):
            if img.mode != "RGB":
                img = img.convert("RGB")
            px = self._transform(img)                                  # [3, H, W]
            image = px.unsqueeze(0).unsqueeze(2).to(self.device, torch.bfloat16)  # [1, 3, 1, H, W]
            initial_latent = self.pipe.vae.encode_to_latent(image).to(self.device, torch.bfloat16)
            noise = torch.randn(
                [1, a.num_latent_frames - 1, LATENT_C, LATENT_H, LATENT_W],
                device=self.device, dtype=torch.bfloat16,
            )
            video, _ = self.pipe.inference(
                noise=noise, text_prompts=[prompt],
                initial_latent=initial_latent, return_latents=True,
            )
            self.pipe.vae.model.clear_cache()
            vid = video[0].float().clamp(0, 1).cpu()                   # [T, C, H, W] in [0,1]
            frames = [to_pil_image(vid[t]) for t in range(vid.shape[0])]
            pipe_images.append(frames)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.last_rollout_time = time.time() - t0
        n_px = len(pipe_images[0]) if pipe_images else 0
        print(f"[causalforcing] {len(pipe_images)} rollout(s), {n_px} pixel frames each, "
              f"{self.last_rollout_time:.2f}s total ({self.last_rollout_time / max(1, len(pipe_images)):.2f}s/rollout)",
              flush=True)
        return pipe_images


# --------------------------------------------------------------------------- #
def build_parser():
    p = argparse.ArgumentParser(description="OUR linearized Causal-Forcing WM worker (World-in-World)")
    p.add_argument("--config_path", type=str,
                   default=osp.join(CF_DIR, "configs", "causal_forcing_dmd_framewise_2step.yaml"))
    p.add_argument("--checkpoint_path", type=str, default="/home/causal_forcing_ckpt/framewise/causal_forcing.pt")
    p.add_argument("--surgery_ckpt", type=str,
                   default="/home/wm-vla-rl/_ext/wan_lin_out/jepa_l025_T8/wanlin_s2_step6000.safetensors")
    p.add_argument("--surgery_ratio", type=float, default=0.75)
    p.add_argument("--surgery_groups", type=int, default=4)
    p.add_argument("--surgery_lora_r", type=int, default=8)
    p.add_argument("--use_ema", action="store_true")
    # I2V / output
    p.add_argument("--num_latent_frames", type=int, default=5,
                   help="latent frames incl. the conditioning frame; pixel frames = (n-1)*4+1 "
                        "(Wan VAE 4x temporal). 5 -> 17 px frames -> subsampled to num_output_frames.")
    p.add_argument("--num_output_frames", type=int, default=14, help="frames returned to the harness")
    p.add_argument("--width", type=int, default=480, help="output frame width sent to harness")
    p.add_argument("--height", type=int, default=480, help="output frame height sent to harness")
    p.add_argument("--task_type", type=str, default="navigation",
                   choices=["navigation", "manipulation", "freetext"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--log_dir", type=str, default="downstream/logs")
    p.add_argument("--exp_id", type=str, default="causalforcing_worker")
    p.add_argument("--smoke", action="store_true",
                   help="run one real-ckpt I2V rollout and print shapes/timing, then exit (no pipe_fd)")
    return p


def _simple_logger(args):
    """Minimal stdout->file tee (we avoid world-in-world's utils.logger to not shadow CF's utils)."""
    log_path = osp.join(WIW_DIR, args.log_dir, args.exp_id, "causalforcing_worker", f"worker{args.device}.log")
    os.makedirs(osp.dirname(log_path), exist_ok=True)
    return log_path


def run_smoke(args):
    """Drive the exact task_fn worker_main would call, with a real uint8 frame + action text."""
    import imageio.v2 as iio
    args.world_model_name = "causalforcing"
    print(f"[smoke] args:\n{args}", flush=True)

    t_load = time.time()
    model = CausalForcingModel(args=args)
    print(f"[smoke] model loaded in {time.time() - t_load:.1f}s", flush=True)

    # --- a real first-frame from the distill data (fall back to random if missing) ---
    mp4 = "/home/wan_distill_data_mp4/clip_00000.mp4"
    if osp.exists(mp4):
        rdr = iio.get_reader(mp4)
        frame0 = rdr.get_data(0)            # H, W, 3 uint8
        rdr.close()
        b_image = np.transpose(frame0, (2, 0, 1))[None].astype(np.uint8)  # [1, 3, H, W]
        print(f"[smoke] first frame from {mp4}: {b_image.shape} {b_image.dtype}", flush=True)
    else:
        b_image = (np.random.rand(1, 3, 256, 256) * 255).astype(np.uint8)
        print(f"[smoke] random first frame: {b_image.shape}", flush=True)

    save_dir = "/tmp/causalforcing_smoke"
    os.makedirs(save_dir, exist_ok=True)
    input_dict = {
        "b_action": np.array([[0, 1, 2, 1, 3]], dtype=np.int64),  # placeholder,fwd,turn_left,fwd,turn_right
        "save_dirs": [save_dir],
        "request_model_name": "causalforcing",
        "b_image": b_image,
        "return_objects": [True],
    }

    # this is EXACTLY task_fn(payload) inside worker_main
    t0 = time.time()
    out = model.inference_batch(input_dict)
    dt = time.time() - t0

    from downstream.utils.worker_manager import check_outputdict
    check_outputdict(out)  # the same validation worker_main runs on the result
    pf = out["pred_frames"]
    print("=" * 60, flush=True)
    print(f"[smoke] RESULT pred_frames: shape={pf.shape} dtype={pf.dtype} "
          f"min={pf.min()} max={pf.max()}", flush=True)
    print(f"[smoke] inference_batch wall-clock: {dt:.2f}s", flush=True)
    print(f"[smoke] generate-only (gpu): {model.last_rollout_time:.2f}s", flush=True)
    print(f"[smoke] save_dirs={out['save_dirs']}", flush=True)
    assert pf.dtype == np.uint8 and pf.ndim == 5 and pf.shape[0] == 1, "bad pred_frames"
    print("=== causalforcing real-ckpt SMOKE PASS ===", flush=True)
    return out


if __name__ == "__main__":
    parser = build_parser()
    if "--smoke" in sys.argv:
        args = parser.parse_args()
        run_smoke(args)
        sys.exit(0)

    # normal worker: last argv is the inherited pipe write-fd
    pipe_fd = int(sys.argv[-1])
    args = parser.parse_args(sys.argv[1:-1])
    args.world_model_name = "causalforcing"

    log_path = _simple_logger(args)
    print(f"[causalforcing_worker] log: {log_path}", flush=True)
    print(f"[causalforcing_worker] All Args:\n{args}", flush=True)

    navigator = CausalForcingModel(args=args)
    print("[causalforcing_worker] linearized Causal-Forcing loaded successfully!", flush=True)

    def do_some_tasks(input_dict):
        return navigator.inference_batch(input_dict)

    worker_main(pipe_fd, do_some_tasks)
