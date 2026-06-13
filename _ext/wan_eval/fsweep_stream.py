"""Efficiency F-sweep — STREAMING / autoregressive deployment path (the mode where linearization is actually sold).

WHY a second sweep: the full-seq training-path forward (wan_eval.py --mode efficiency) measures KDA's parallel
chunk form vs softmax's flex_attention — both OOM beyond F=41 and show memory parity, so it CANNOT reach the
requested 321/501/921px and is NOT the deployment mode. Real video models are deployed AR with a KV-cache: the
softmax teacher keeps a bounded rolling KV-cache (local_attn_size=-1 -> max_attention_size=32760 = 21 frames),
the KDA student keeps a fixed-size recurrent state. This driver times THAT path at increasing rollout length.

Reuses the repo's validated CausalInferencePipeline (no hand-wired caches). VAE decode is bypassed (we only need
denoiser timing + cache/activation memory; last_generation_time already excludes VAE) so 921px doesn't OOM in VAE.

Run (one mode per process, isolated memory):
  CUDA_VISIBLE_DEVICES=0 python fsweep_stream.py teacher  > stream_teacher.json.log
  CUDA_VISIBLE_DEVICES=0 python fsweep_stream.py student  > stream_student.json.log
"""
import os, sys, time, json, argparse
sys.path.insert(0, "/home/wm-vla-rl/_ext")
sys.path.insert(0, "/home/wm-vla-rl/_ext/Causal-Forcing")
os.chdir("/home/wm-vla-rl/_ext/Causal-Forcing")
import torch
from omegaconf import OmegaConf
from pipeline import CausalInferencePipeline
import wan_kda_surgery as S

CONFIG = "configs/causal_forcing_dmd_framewise_2step.yaml"
BASE = "/home/causal_forcing_ckpt/framewise/causal_forcing.pt"
FRS = [21, 41, 81, 126, 231]                 # latent frames -> px (F-1)*4+1 = 81 / 161 / 321 / 501 / 921
OUT = "/home/wm-vla-rl/_ext/wan_eval"


def build(mode, device):
    config = OmegaConf.merge(OmegaConf.load("configs/default_config.yaml"), OmegaConf.load(CONFIG))
    S.patch_wan_flash_sdpa()                  # BOTH paths use the sdpa backend (inference.py does this unconditionally)
    pipe = CausalInferencePipeline(config, device=device)
    sd = torch.load(BASE, map_location="cpu")
    gen = sd["generator_ema"] if "generator_ema" in sd else sd["generator"]
    try:
        pipe.generator.load_state_dict(gen)
    except RuntimeError:
        pipe.generator.load_state_dict({k.replace("model._fsdp_wrapped_module.", "model.", 1): v
                                        for k, v in gen.items()}, strict=False)
    if mode == "student":                     # KDA(75%)+AttnRes+LoRA architecture (timing is weight-independent)
        m = pipe.generator.model
        lin = S.factorize_causal_wan(m, ratio=0.75)
        S.attnres_ify_wan(m, groups=4)
        S.add_lora_wan(m, lin, r=8, include_ffn=True)
    pipe = pipe.to(device).to(torch.bfloat16)
    pipe.generator.model.eval()
    pipe.vae.decode_to_pixel = lambda *a, **k: torch.zeros(1, device=device)   # bypass VAE (denoiser-only timing)
    return pipe


@torch.no_grad()
def bench(pipe, Fr, device, prompt):
    pipe.kv_cache1 = None                      # force a fresh KV-cache sized for this rollout
    noise = torch.randn([1, Fr, 16, 60, 104], device=device, dtype=torch.bfloat16)
    pipe.inference(noise=noise, text_prompts=[prompt], return_latents=True, report_timing=True)  # warmup
    pipe.kv_cache1 = None
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    noise = torch.randn([1, Fr, 16, 60, 104], device=device, dtype=torch.bfloat16)
    t0 = time.time()
    pipe.inference(noise=noise, text_prompts=[prompt], return_latents=True, report_timing=True)
    torch.cuda.synchronize()
    wall = time.time() - t0
    gen_t = float(getattr(pipe, "last_generation_time", wall) or wall)   # denoiser streaming time (excl VAE)
    first = float(getattr(pipe, "first_chunk_time", 0.0) or 0.0)
    mem = torch.cuda.max_memory_allocated() / 1e9
    perblk = gen_t / Fr * 1000.0               # ms per latent frame (steady-state proxy)
    return dict(latent_F=Fr, px_F=(Fr - 1) * 4 + 1, gen_s=round(gen_t, 3),
                per_frame_ms=round(perblk, 1), first_chunk_s=round(first, 3), peak_GB=round(mem, 1))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("mode", choices=["teacher", "student"]); a = ap.parse_args()
    device = torch.device("cuda")
    prompt = "a cinematic shot of a coastal town at sunset, gentle waves, warm light"
    pipe = build(a.mode, device)
    print(f"[stream-{a.mode}] built; sweeping {FRS}", flush=True)
    rows = []
    for Fr in FRS:
        try:
            r = bench(pipe, Fr, device, prompt)
            rows.append(r)
            print(f"[stream-{a.mode}] F={Fr:3d} (px {r['px_F']:3d})  gen {r['gen_s']:.2f}s  "
                  f"{r['per_frame_ms']:.1f}ms/frame  TTFC {r['first_chunk_s']:.2f}s  peak {r['peak_GB']:.1f}GB",
                  flush=True)
        except torch.cuda.OutOfMemoryError:
            rows.append(dict(latent_F=Fr, px_F=(Fr - 1) * 4 + 1, oom=True))
            torch.cuda.empty_cache()
            print(f"[stream-{a.mode}] F={Fr:3d} OOM", flush=True)
    json.dump(rows, open(os.path.join(OUT, f"stream_{a.mode}.json"), "w"), indent=2)
    print(f"[stream-{a.mode}] saved -> {OUT}/stream_{a.mode}.json", flush=True)


if __name__ == "__main__":
    main()
