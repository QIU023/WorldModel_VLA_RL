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

## env `libero` — 仿真（Module 2-3）
```bash
git clone https://github.com/Lifelong-Robot-Learning/LIBERO /home/wm-vla-rl/_ext/LIBERO
# pip install -e LIBERO；robosuite + mujoco
# ⚠️ headless 服务器需 EGL: export MUJOCO_GL=egl (早装早排，OFT 与 LIBERO 的 robosuite 版本要对齐)
```

## (备选, Module-3 stretch) LAPA
```bash
git clone https://github.com/LatentActionPretraining/LAPA /home/wm-vla-rl/_ext/LAPA
```
