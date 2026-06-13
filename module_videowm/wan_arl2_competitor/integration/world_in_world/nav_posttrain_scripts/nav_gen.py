"""SMOKE gate (c): sanity I2V generation with the nav LoRA, via the REAL streaming inference path
(the same path IGNav eval uses). Loads the pipeline fresh, applies OUR surgery + the distill main delta
+ (optionally) a fresh nav LoRA + its delta, then I2V-generates from a trained-on (first_frame, action_text)
and reports degeneracy + whether the generation is steered toward the ground-truth trajectory.

Checks:
  * frames finite, not uniform (per-frame std > 0), and dynamic (inter-frame motion > 0).
  * pixel MSE(gen, OWN-clip GT future) vs MSE(gen, OTHER-clip GT future): conditioning works if own < other.
  * if --compare_base, also gens with the nav LoRA OFF (identity) and reports the same, so we can see
    whether nav post-training moved the generation closer to GT.

Run (cosmos_predict2):
  /home/venv/cosmos_predict2/bin/python nav_gen.py --clip clip_00000 --nav_lora_ckpt <nav.safetensors> \
      --compare_base --out /root/nav_posttrain/gen
"""
import argparse, json, os, os.path as osp, sys
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image
from omegaconf import OmegaConf
from safetensors.torch import load_file

CF_DIR = "/home/wm-vla-rl/_ext/Causal-Forcing"
os.chdir(CF_DIR)
sys.path.insert(0, CF_DIR)
sys.path.insert(0, "/home/wm-vla-rl/_ext")
from pipeline import CausalInferencePipeline
import wan_kda_surgery as KS
from nav_posttrain import add_nav_lora  # reuse the EXACT nav-LoRA stacking used in training

GEN_H, GEN_W, LC, LH, LW = 480, 832, 16, 60, 104
IGENEX_ACTION_STRS = {1: "forward", 2: "turn_left", 3: "turn_right", 4: "stop", 0: "placeholder"}


def action_text(b_action):
    seq = []
    for a in b_action[1:]:
        s = IGENEX_ACTION_STRS[a]
        if s == "forward":
            s = "forward 0.2m"
        elif "turn" in s:
            s = f"{s} 22.5°"
        seq.append(s)
    return f"Follow this sequence of camera motions: {str(seq)}."


def build_pipe(args, with_nav):
    config = OmegaConf.merge(OmegaConf.load(osp.join(CF_DIR, "configs", "default_config.yaml")),
                             OmegaConf.load(args.config_path))
    KS.patch_wan_flash_sdpa()
    pipe = CausalInferencePipeline(config, device=torch.device(args.device))
    sd = torch.load(args.checkpoint_path, map_location="cpu")
    want = "generator_ema" if "generator_ema" in sd else "generator"
    gen_sd = {(k.replace("model._fsdp_wrapped_module.", "model.", 1)): v for k, v in sd[want].items()}
    pipe.generator.load_state_dict(gen_sd, strict=False)
    m = pipe.generator.model
    lin = KS.factorize_causal_wan(m, ratio=args.ratio)
    KS.attnres_ify_wan(m, groups=args.groups)
    KS.add_lora_wan(m, lin, r=args.main_r, include_ffn=True)
    miss, unexp = m.load_state_dict(load_file(args.main_delta), strict=False)
    print(f"[gen] main delta loaded (unexp={len(unexp)})", flush=True)
    if with_nav and args.nav_lora_ckpt:
        add_nav_lora(m, lin, r=args.nav_r)             # SAME stacking + order as training
        miss, unexp = m.load_state_dict(load_file(args.nav_lora_ckpt), strict=False)
        nload = len(load_file(args.nav_lora_ckpt))
        print(f"[gen] nav LoRA loaded: {nload} tensors (unexp={len(unexp)})", flush=True)
    pipe = pipe.to(dtype=torch.bfloat16)
    pipe.text_encoder.to(args.device); pipe.generator.to(args.device); pipe.vae.to(args.device)
    return pipe


@torch.no_grad()
def gen(pipe, img_pil, prompt, num_latent_frames, device, seed=0):
    tfm = transforms.Compose([transforms.Resize((GEN_H, GEN_W)),
                              transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
    px = tfm(img_pil.convert("RGB")).unsqueeze(0).unsqueeze(2).to(device, torch.bfloat16)  # [1,3,1,H,W]
    initial_latent = pipe.vae.encode_to_latent(px).to(device, torch.bfloat16)
    torch.manual_seed(seed)
    noise = torch.randn([1, num_latent_frames - 1, LC, LH, LW], device=device, dtype=torch.bfloat16)
    video, _ = pipe.inference(noise=noise, text_prompts=[prompt],
                              initial_latent=initial_latent, return_latents=True)
    pipe.vae.model.clear_cache()
    return video[0].float().clamp(0, 1).cpu()          # [T,C,H,W] in [0,1]


def gt_frames(clip_dir, T):
    pngs = sorted(p for p in os.listdir(clip_dir) if p.endswith(".png"))
    imgs = [np.asarray(Image.open(osp.join(clip_dir, p)).convert("RGB").resize((GEN_W, GEN_H))) / 255.0
            for p in pngs]
    arr = np.stack(imgs, 0)                              # [Tg,H,W,3]
    return arr[:T] if arr.shape[0] >= T else arr


def report(tag, vid):
    v = vid.numpy()                                     # [T,C,H,W]
    finite = bool(np.isfinite(v).all())
    per_frame_std = float(v.std(axis=(1, 2, 3)).mean())
    motion = float(np.abs(np.diff(v, axis=0)).mean())
    print(f"[gen][{tag}] T={v.shape[0]} finite={finite} mean={v.mean():.3f} "
          f"per-frame-std={per_frame_std:.4f} inter-frame-motion={motion:.4f}", flush=True)
    return finite, per_frame_std, motion


def mse_to_gt(vid, gt):
    v = vid.permute(0, 2, 3, 1).numpy()                 # [T,H,W,3]
    T = min(v.shape[0], gt.shape[0])
    return float(np.mean((v[:T] - gt[:T]) ** 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config_path", default=osp.join(CF_DIR, "configs", "causal_forcing_dmd_framewise_2step.yaml"))
    ap.add_argument("--checkpoint_path", default="/home/causal_forcing_ckpt/framewise/causal_forcing.pt")
    ap.add_argument("--main_delta", default="/home/wm-vla-rl/_ext/wan_lin_out/jepa_l025_T8/wanlin_s2_step6000.safetensors")
    ap.add_argument("--nav_lora_ckpt", default=None)
    ap.add_argument("--ratio", type=float, default=0.75)
    ap.add_argument("--groups", type=int, default=4)
    ap.add_argument("--main_r", type=int, default=8)
    ap.add_argument("--nav_r", type=int, default=8)
    ap.add_argument("--frames_dir", default="/root/nav_posttrain/frames")
    ap.add_argument("--clip", default="clip_00000")
    ap.add_argument("--other_clip", default="clip_00010")
    ap.add_argument("--num_latent_frames", type=int, default=5)
    ap.add_argument("--out", default="/root/nav_posttrain/gen")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--compare_base", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    cdir = osp.join(args.frames_dir, args.clip)
    meta = json.load(open(osp.join(cdir, "meta.json")))
    prompt = action_text(meta["b_action"])
    img0 = Image.open(osp.join(cdir, "frame_00.png"))
    print(f"[gen] clip {args.clip} | {prompt}", flush=True)

    T = (args.num_latent_frames - 1) * 4 + 1
    gt = gt_frames(cdir, T)
    gt_other = gt_frames(osp.join(args.frames_dir, args.other_clip), T)

    def run(with_nav, name):
        pipe = build_pipe(args, with_nav)
        vid = gen(pipe, img0, prompt, args.num_latent_frames, args.device)
        fin, std, mot = report(name, vid)
        own = mse_to_gt(vid, gt); oth = mse_to_gt(vid, gt_other)
        print(f"[gen][{name}] MSE(gen, OWN-GT)={own:.5f}  MSE(gen, OTHER-GT)={oth:.5f}  "
              f"-> conditioning {'OK (own<other)' if own < oth else 'WEAK (own>=other)'}", flush=True)
        # save a contact sheet (first/mid/last gen vs GT)
        for i in [0, T // 2, T - 1]:
            to_pil_image(vid[i]).save(osp.join(args.out, f"{name}_f{i:02d}.png"))
        del pipe
        torch.cuda.empty_cache()
        return dict(finite=fin, std=std, motion=mot, own=own, other=oth)

    res = {}
    if args.compare_base:
        res["base"] = run(False, "base")
    res["nav"] = run(True, "nav") if args.nav_lora_ckpt else {}
    # GT save for reference
    for i in [0, T // 2, T - 1]:
        Image.fromarray((gt[i] * 255).astype(np.uint8)).save(osp.join(args.out, f"GT_f{i:02d}.png"))
    print(f"[gen] SUMMARY: {json.dumps(res)}", flush=True)


if __name__ == "__main__":
    main()
