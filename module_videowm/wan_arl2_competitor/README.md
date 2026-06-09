# ARL2 Competitor — Causal Forcing (Wan2.1-1.3B AR) linearization with KDA + AttnRes

Direct competitor to ARL2 (Attend Locally Remember Linearly, 2605.16579). Same teacher (**Causal Forcing
1.3B**, thu-ml, = AR distill of Wan2.1-T2V-1.3B), same benchmark (VBench 946 prompts, 832×480, 81-921 frames),
same two-stage distill recipe (per-layer attn-transfer → joint velocity output-distill). Our differentiation:
inter-frame operator = **KDA** (gated-delta, not ARL2's GDN) + **naive cross-layer AttnRes** (ARL2 has none).

`wan_kda_surgery.py`: FactorizedKDASelfAttention drop-in for WanSelfAttention —
- intra-frame **softmax** over H·W (torch SDPA = flash kernel on H200; preserves spatial detail/dynamics)
- inter-frame **KDA** causal gated-delta over F frames (the O(F) long-video win, fixed-size state)
- 3D RoPE: F-phase cancels within frame (intra = spatial RoPE); KDA gating handles temporal position
- `factorize_wan(model, ratio=0.75)` replaces 23/30 blocks; teacher q/k/v/o inherited, KDA gate/beta new (~2%)
- `patch_wan_flash_sdpa()`: teacher flash_attention -> torch SDPA (no flash-attn pkg; CUDA 12.6 toolkit vs
  torch cu130 mismatch blocks flash-attn compile, but SDPA on H200 already uses the FLASH kernel: 8192 tok 2.5ms/0.1GB)

venv: /home/venv/cosmos_predict2 (torch2.12/cu130, diffusers, **fla** for KDA). Teacher: /home/causal_forcing_ckpt
(framewise/causal_forcing.pt) + /home/wan21_1.3b (Wan2.1 VAE/T5/DiT). Repo: _ext/Causal-Forcing.
