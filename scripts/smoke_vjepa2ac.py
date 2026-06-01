"""Week-1 smoke: 验证 V-JEPA2-AC ckpt 加载 + encoder 前向能在 GPU1 上跑通。
注意: ckpt 已预下载到 $TORCH_HOME/hub/checkpoints/vjepa2-ac-vitg.pt,
load_state_dict_from_url 按文件名命中缓存, 不会去碰仓库里写死的 localhost:8300。
"""
import os, sys, time, traceback

REPO = "/home/wm-vla-rl/_ext/vjepa2"
sys.path.insert(0, REPO)

try:
    import torch
    print("torch", torch.__version__, "cuda_avail", torch.cuda.is_available(),
          "ndev", torch.cuda.device_count())
    # CUDA_VISIBLE_DEVICES=1 → 这里的 device 0 物理上就是 GPU1
    print("device0(物理GPU1?):", torch.cuda.get_device_name(0))

    from hubconf import vjepa2_ac_vit_giant
    t = time.time()
    enc, pred = vjepa2_ac_vit_giant(pretrained=True)   # 命中本地 11.76GB ckpt
    print("build+load %.1fs" % (time.time() - t))

    enc = enc.cuda().eval()
    pred = pred.cuda().eval()
    print("encoder params M", round(sum(p.numel() for p in enc.parameters()) / 1e6, 1))
    print("predictor params M", round(sum(p.numel() for p in pred.parameters()) / 1e6, 1))

    x = torch.randn(1, 3, 64, 256, 256, device="cuda")  # (B,C,T,H,W), 同 demo 64 帧
    with torch.inference_mode():
        t = time.time()
        z = enc(x)
        dt = time.time() - t
    print("encoder forward out shape", tuple(z.shape), "dt %.3fs" % dt)
    print("max_mem_GB", round(torch.cuda.max_memory_allocated() / 1e9, 2))
    print("SMOKE_OK")
except Exception:
    print("SMOKE_FAIL")
    traceback.print_exc()
    sys.exit(1)
