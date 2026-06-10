import argparse
import argparse
import torch
import os
from omegaconf import OmegaConf
from tqdm import tqdm
from torchvision import transforms
def write_video(filename, video_array, fps):   # torchvision 0.27 removed io.write_video -> imageio+ffmpeg shim (same sig: [T,H,W,C])
    import imageio.v2 as _iio, numpy as _np, torch as _torch
    v = video_array.detach().cpu() if isinstance(video_array, _torch.Tensor) else video_array
    v = _np.asarray(v, dtype=_np.float32)
    if v.max() <= 1.01:                          # [0,1] convention -> scale (CF passes [0,255] already)
        v = v * 255.0
    v = _np.clip(v, 0, 255).round().astype(_np.uint8)
    _iio.mimwrite(filename, [f for f in v], fps=int(fps), codec="libx264", quality=8)
from einops import rearrange
import torch.distributed as dist
from torch.utils.data import DataLoader, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
import json

from pipeline import (
    CausalDiffusionInferencePipeline,
    CausalInferencePipeline,
)
from utils.dataset import TextDataset, TextImagePairDataset
from utils.misc import set_seed

from demo_utils.memory import gpu, get_cuda_free_memory_gb, DynamicSwapInstaller

parser = argparse.ArgumentParser()
parser.add_argument("--config_path", type=str, help="Path to the config file")
parser.add_argument("--checkpoint_path", type=str, help="Path to the checkpoint folder")
parser.add_argument("--data_path", type=str, help="Path to the dataset")
parser.add_argument("--output_folder", type=str, help="Output folder")
parser.add_argument("--num_output_frames", type=int, default=21, help="Number of overlap frames between sliding windows")
parser.add_argument("--use_ema", action="store_true", help="Whether to use EMA parameters")
parser.add_argument("--seed", type=int, default=0, help="Random seed")
parser.add_argument("--i2v", action="store_true", help="Whether to perform I2V (or T2V by default)")
parser.add_argument("--report_timing", action="store_true",
                    help="Only tested on A800, for the Causal Forcing++ latency. Not make claims for other hardware like H100. For the result on H100, refer to the reported results in the Self Forcing paper.")
# --- linearized-student injection (our KDA+AttnRes+LoRA surgery on the STANDARD generator; teacher path unaffected) ---
parser.add_argument("--surgery_ckpt", type=str, default=None,
                    help="our trained KDA+LoRA+AttnRes safetensors; applies surgery to pipeline.generator.model then loads these deltas (load AFTER --checkpoint_path base)")
parser.add_argument("--surgery_ratio", type=float, default=0.75)
parser.add_argument("--surgery_groups", type=int, default=4)
parser.add_argument("--surgery_lora_r", type=int, default=8)
parser.add_argument("--vbench_naming", action="store_true",
                    help="write '{full_prompt}-0.mp4' (VBench vbench_standard convention) instead of '{prompt[:100]}.mp4'")
parser.add_argument("--batch_size", type=int, default=1,
                    help="batched T2V generation — fills GPU memory, far higher throughput than batch=1")
args = parser.parse_args()

# Initialize distributed inference
if "LOCAL_RANK" in os.environ:
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    world_size = dist.get_world_size()
    
else:
    device = torch.device("cuda")
    local_rank = 0
    world_size = 1

set_seed(args.seed)

print(f'Free VRAM {get_cuda_free_memory_gb(gpu)} GB')
low_memory = get_cuda_free_memory_gb(gpu) < 40

torch.set_grad_enabled(False)

config = OmegaConf.load(args.config_path)
default_config = OmegaConf.load("configs/default_config.yaml")
config = OmegaConf.merge(default_config, config)

# This env has NO flash-attn-2 -> route CF's flash_attention to torch SDPA (flash backend) for BOTH teacher & student
# (teacher path doesn't go through the surgery branch, so it would hit `assert FLASH_ATTN_2_AVAILABLE` without this).
import sys as _sys0
if "/home/wm-vla-rl/_ext" not in _sys0.path: _sys0.path.insert(0, "/home/wm-vla-rl/_ext")
import wan_kda_surgery as _S0; _S0.patch_wan_flash_sdpa()

# Initialize pipeline
if hasattr(config, 'denoising_step_list'):
    # Few-step inference
    pipeline = CausalInferencePipeline(config, device=device)
else:
    # Multi-step diffusion inference
    pipeline = CausalDiffusionInferencePipeline(config, device=device)

if args.checkpoint_path:
    state_dict = torch.load(args.checkpoint_path, map_location="cpu")
    key = 'generator_ema' if args.use_ema else 'generator'
    gen_sd = state_dict[key]

    try:
        pipeline.generator.load_state_dict(gen_sd)
    except RuntimeError:
        fixed = {}
        for k, v in gen_sd.items():
            if k.startswith("model._fsdp_wrapped_module."):
                k = k.replace("model._fsdp_wrapped_module.", "model.", 1)
            fixed[k] = v
        pipeline.generator.load_state_dict(fixed, strict=False)

if args.surgery_ckpt:   # build OUR linearized student in-place on the standard generator, then load trained deltas
    import sys as _sys
    if "/home/wm-vla-rl/_ext" not in _sys.path:
        _sys.path.insert(0, "/home/wm-vla-rl/_ext")
    import wan_kda_surgery as _S
    _S.patch_wan_flash_sdpa()                                   # match the training attention backend
    _m = pipeline.generator.model                              # CausalWanModel (base teacher weights already loaded)
    _lin = _S.factorize_causal_wan(_m, ratio=args.surgery_ratio)
    _S.attnres_ify_wan(_m, groups=args.surgery_groups)
    _S.add_lora_wan(_m, _lin, r=args.surgery_lora_r, include_ffn=True)
    from safetensors.torch import load_file as _load_file
    _sd = _load_file(args.surgery_ckpt)
    _miss, _unexp = _m.load_state_dict(_sd, strict=False)
    print(f"[surgery] {len(_sd)} trained tensors -> standard generator "
          f"(unexpected={len(_unexp)}, missing-base={len(_miss)}); ratio={args.surgery_ratio} "
          f"groups={args.surgery_groups} lora_r={args.surgery_lora_r}", flush=True)

pipeline = pipeline.to(dtype=torch.bfloat16)
if low_memory:
    DynamicSwapInstaller.install_model(pipeline.text_encoder, device=gpu)
else:
    pipeline.text_encoder.to(device=gpu)
pipeline.generator.to(device=gpu)
pipeline.vae.to(device=gpu)


# Create dataset
if args.i2v:
    assert not dist.is_initialized(), "I2V does not support distributed inference yet"
    transform = transforms.Compose([
        transforms.Resize((480, 832)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    dataset = TextImagePairDataset(args.data_path, transform=transform)
else:
    dataset = TextDataset(prompt_path=args.data_path)
num_prompts = len(dataset)
print(f"Number of prompts: {num_prompts}")

if args.report_timing and num_prompts < 2:
    print(f"[WARN] --report_timing requires at least 2 prompts "
          f"(got {num_prompts}); timing disabled.")
    args.report_timing = False

if dist.is_initialized():
    sampler = DistributedSampler(dataset, shuffle=False, drop_last=True)
else:
    sampler = SequentialSampler(dataset)
dataloader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler, num_workers=0, drop_last=False)

# Create output directory (only on main process to avoid race conditions)
if local_rank == 0:
    os.makedirs(args.output_folder, exist_ok=True)

if dist.is_initialized():
    dist.barrier()

def encode(self, videos: torch.Tensor) -> torch.Tensor:
    device, dtype = videos[0].device, videos[0].dtype
    scale = [self.mean.to(device=device, dtype=dtype),
             1.0 / self.std.to(device=device, dtype=dtype)]
    output = [
        self.model.encode(u.unsqueeze(0), scale).float().squeeze(0)
        for u in videos
    ]

    output = torch.stack(output, dim=0)
    return output


for i, batch_data in tqdm(enumerate(dataloader), disable=(local_rank != 0)):
    prompts = batch_data['prompts']
    if isinstance(prompts, str):
        prompts = [prompts]
    B = len(prompts)
    if args.i2v:
        assert B == 1, "I2V path uses batch_size=1"
        assert config.num_frame_per_block == 1, "Current I2V only supports the frame-wise model."
        image = batch_data['image'].squeeze(0).unsqueeze(0).unsqueeze(2).to(device=device, dtype=torch.bfloat16)
        initial_latent = pipeline.vae.encode_to_latent(image).to(device=device, dtype=torch.bfloat16)
        text_prompts = [prompts[0]]
        sampled_noise = torch.randn([1, args.num_output_frames - 1, 16, 60, 104], device=device, dtype=torch.bfloat16)
    else:                                                              # batched T2V (fills GPU, high throughput)
        text_prompts = list(prompts)
        if 'extended_prompts' in batch_data:
            ep = batch_data['extended_prompts']
            text_prompts = [(e if e else p) for e, p in zip(ep, prompts)]
        initial_latent = None
        sampled_noise = torch.randn([B, args.num_output_frames, 16, 60, 104], device=device, dtype=torch.bfloat16)

    video, latents = pipeline.inference(
        noise=sampled_noise, text_prompts=text_prompts,
        return_latents=True, initial_latent=initial_latent,
    )
    video = 255.0 * rearrange(video, 'b t c h w -> b t h w c').cpu()   # [B,T,H,W,C] float in [0,255]
    pipeline.vae.model.clear_cache()
    for j, prompt in enumerate(prompts):
        output_path = os.path.join(args.output_folder, (f'{prompt}-0.mp4' if args.vbench_naming else f'{prompt[:100]}.mp4'))
        write_video(output_path, video[j], fps=16)

       
