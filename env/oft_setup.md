# OpenVLA-OFT 推理 env 配方(task#12,2026-06-03 实测打通)

独立 venv `/home/venv/oft`(torch 2.2,与 libero/vjepa2 venv 隔离)。inference-only(不需 flash-attn)。

## 关键依赖解析(踩过的坑)
| 包 | 版本 | 备注 |
|---|---|---|
| torch / torchvision | 2.2.0 / 0.17.0 (+cu121) | OFT pin |
| transformers | **4.40.1**(moojink fork `git+.../transformers-openvla-oft`)| 双向 attn/并行解码;**LIBERO req 会把它降到 4.21.1 → 必须最后重装盖回** |
| tokenizers | 0.19.1 | 同上,LIBERO 降到 0.12.1 要盖回 |
| numpy | 1.26.4 | torch2.2 期望 <2;tf2.15 需 ≥1.23;LIBERO req 想 1.22.4(实测 1.26.4 也跑) |
| protobuf | 4.25.9 + **`PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python`** | tfds 要 4.x,但 4.x 会"Descriptors cannot be created directly" → 用 python impl 绕过 |
| tensorflow-cpu | 2.15.0 | **推理路径真用**(`openvla_utils.py` 用 `tf.image.resize` lanczos3 做与训练一致的预处理),不能 stub |
| tensorflow_datasets / tensorflow_graphics | 4.9.3 / 2021.12.3 | dlimp 链需要 |
| dlimp | `git+.../dlimp_openvla` | import 链需要(非仅训练) |
| robosuite / mujoco / bddl / gym | 1.4.0 / 3.9.0 / 1.0.1 / 0.25.2 | LIBERO sim;装 `LIBERO/requirements.txt` |
| prismatic(openvla-oft) | -e --no-deps | 本体 |
| libero / lilo-vla | -e | sim + ultra_long benchmark |

## 安装顺序(关键:OFT 版本最后盖)
1. `pip install torch==2.2.0 torchvision==0.17.0 --index-url .../cu121`
2. OFT PyPI deps(accelerate draccus einops peft==0.11.1 timm==0.9.10 diffusers==0.30.3 wandb ...)
3. `pip install git+.../transformers-openvla-oft`(fork)
4. `pip install -e openvla-oft --no-deps`
5. `pip install -r LIBERO/requirements.txt` + `pip install -e LIBERO -e LiLo-VLA`
6. `pip install tensorflow-cpu==2.15.0 tensorflow_datasets==4.9.3 tensorflow_graphics==2021.12.3 git+.../dlimp_openvla`
7. **重装盖回**:`pip install git+.../transformers-openvla-oft tokenizers==0.19.1 numpy==1.26.4`(步骤5会降级)

## 运行(eval/蒸馏教师)
```
MUJOCO_GL=egl PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
PYTHONPATH=/home/wm-vla-rl/_ext/openvla-oft:/home/wm-vla-rl/_ext/LIBERO \
/home/venv/oft/bin/python scripts/exp_m2i_oft_ultralong.py
```
- ckpt:`moojink/openvla-7b-oft-finetuned-libero-10`(HF_HOME=/home/.hf_home);unnorm_key=libero_10。
- ultra_long bddl 用 `bench.get_task_bddl_file_path`(在 LiLo configs,非标准 LIBERO 路径);`Closed` predicate 补 `Close()` 别名;init_states 用 `torch.load(weights_only=False)`。
- 已验证:import 链 + ultra_long env reset/step + agentview/wrist 256×256 obs ✅(OFT_ENV_SMOKE_OK)。
