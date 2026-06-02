# 环境安装配方（三套隔离 python）

> 沿用 AttnRes 教训：torch 版本冲突时各组件分 env；`--no-deps` 装外围包别让它拖动 torch；
> **所有 ckpt/数据/日志写 `/home`**（`/workspace` 易失）。`export HF_HOME=/home/.hf_home`。

## env `vjepa2` — V-JEPA 2 推理 + planning（Module 1）
```bash
# python 3.10/3.11, torch 最新
git clone https://github.com/facebookresearch/vjepa2 /home/wm-vla-rl/_ext/vjepa2
# pip install -r requirements；einops timm decord scipy（CEM 用）matplotlib
# AC predictor: torch.hub.load('facebookresearch/vjepa2','vjepa2_ac_vit_giant')
# encoder:      facebook/vjepa2-vitl-fpc64-256 (300M) 或 vit-g (1B)
# 关键文件参考: app/vjepa_droid/ (predictor 类), notebooks/energy_landscape_example.ipynb (能量)
```

## env `vla-rl` — SimpleVLA-RL / veRL / vLLM（Module 2-3）
```bash
git clone https://github.com/PRIME-RL/SimpleVLA-RL /home/wm-vla-rl/_ext/SimpleVLA-RL
git clone https://github.com/moojink/openvla-oft  /home/wm-vla-rl/_ext/openvla-oft
# veRL + vLLM + flash-attn；OpenVLA-OFT 依赖（transformers 特定版本，按其 README 锁）
# ckpt: moojink/openvla-7b-oft-finetuned-libero-spatial (先单 suite)
# ⚠️ SimpleVLA-RL 默认 8-GPU colocated → 降配见 module2_vla_rl/simplevla_2gpu/
```

## env `libero` — 仿真（Module 2-3）✅ 已跑通（2026-06-02，headless EGL）
独立 venv `/home/venv/libero`（robosuite 1.4.0 + mujoco 3.9.0 + bddl + robomimic + gym 0.25，老 pin 隔离）。
踩过的 3 个坑 + 解法（见 scripts/install_libero_*.log）：
1. `egl_probe` 编译需 CMake → `apt-get update && apt-get install -y cmake libegl1-mesa-dev libgl1-mesa-dev`。
2. `-e` editable 没注册到 venv → 运行时加 `PYTHONPATH=/home/wm-vla-rl/_ext/LIBERO`。
3. 首次 import `input()` 问 dataset 路径会卡 stdin → 预写 `~/.libero/config.yaml`（见 scripts/libero_smoke.py）。
```bash
# 可复现运行约定：
PYTHONPATH=/home/wm-vla-rl/_ext/LIBERO MUJOCO_GL=egl /home/venv/libero/bin/python scripts/libero_env_smoke.py
# 验证：建 OffScreenRenderEnv + reset + step → agentview_image (128,128,3) ✅
# benchmarks: libero_spatial/object/goal/90/10/100；bddl_files + init_files 内置；demo hdf5 在 _ext/datasets/LIBERO/
```

### ⭐ env 统一（P2 单进程闭环，2026-06-02）
libero venv（torch2.12）可同时跑 **sim + V-JEPA2 encoder + KDA WM**，避免 server/client：
```bash
# 1) 把 py3.10 能用的 fla 0.5.0 复制进 libero venv（torch2.12+triton3.7 实测 chunk_kda forward OK）：
cp -rn /root/miniconda3/envs/py3.10/lib/python3.10/site-packages/fla /home/venv/libero/lib/python3.10/site-packages/
# 2) timm：
/home/venv/libero/bin/pip install --no-deps timm
# 3) 运行 P2 闭环（单进程）：
PYTHONPATH=/home/wm-vla-rl/_ext/LIBERO MUJOCO_GL=egl TORCH_HOME=/home/.cache/torch \
  /home/venv/libero/bin/python <p2_script>   # sys.path 加 _ext/vjepa2 + module1_vjepa2ac
# 实测：vjepa2 encoder(1012M)+KDA predictor 在 libero venv 加载+forward 通。
```

## (备选, Module-3 stretch) LAPA
```bash
git clone https://github.com/LatentActionPretraining/LAPA /home/wm-vla-rl/_ext/LAPA
```
