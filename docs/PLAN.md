# WM × VLA-RL 简历项目 — 本地结构与执行流程

## Context（为什么做、做成什么样）

求职用的算法简历项目。主线是**机器人学习两大前沿范式的复现与机制级理解**：
(1) latent world model 视觉表征（V-JEPA 2 / JEPA），(2) VLA 上的 on-policy RL（GRPO）。
辅线（顺手 / 复用强项）：(3) 蒸馏作为 RL 的下游，(4) 2×H200 上的分布式/推理 infra 优化作为副产品。

**已确认的三个决策（2026-06-01）：**
- 总周期 **~4 周**（均衡 MVP，战线不拖长）。
- VLA RL 走 **改造 SimpleVLA-RL 到 2 卡**（保留 GRPO+KL vanilla loop，2-GPU 移植本身即 Module-4 infra 交付物，复用 AttnRes 的 veRL/FSDP/2 卡经验）。
- Module 2 以 **latent-WM reward 路线为主角**（用 V-JEPA2-AC 世界模型给 RL 当 reward；vanilla binary-reward GRPO 只作 baseline + 兜底）。

**最终交付 = 一个能写进简历的 repo**：README 里 3-4 个 headline 结果 + 图，每个模块对应 1 个 checkpoint。
核心叙事：*"把世界模型表征（WM）和 VLA RL 用我的蒸馏/KL 老本行串起来 —— reverse-KL 在 RL 抗 policy 漂移、在蒸馏抗 mode-collapse 是同一套数学，我能讲出别人讲不出的连接。"*

---

## 硬件与环境约束

- **2×H200 sm_90 单节点**，状态全部落 `/home`（持久；`/workspace` 易失 —— 沿用 AttnRes 的教训，所有 ckpt/数据/日志写 `/home`）。
- **三套隔离 python**（torch 版本冲突，沿用 AttnRes 模式）：
  - `vjepa2` env：facebookresearch/vjepa2 推理 + planning（torch 最新）。
  - `vla-rl` env：SimpleVLA-RL / veRL / vLLM rollout。
  - `libero` env：robosuite + MuJoCo（headless EGL，注意远程服务器配置）。
- 7B OpenVLA-OFT 用 **LoRA**：policy + ref + vLLM rollout 在 2×141GB 上 colocated 可行（LoRA 共享 base、切 adapter）。

---

## 仓库结构（建议 `/home/wm-vla-rl/`，git init）

```
wm-vla-rl/
├── README.md                    # 简历门面：叙事 + headline 结果 + 图
├── docs/writeup.md              # 机制级讲解（"讲到别人讲不出"的部分）
├── env/{vjepa2,vla-rl,libero}.yml
├── module1_vjepa2ac/
│   ├── rollout.py               # block-causal latent rollout（+ KV-cache 副产品）
│   ├── cem_planner.py           # CEM/MPC planner（仓库没有，从零写）
│   ├── probe.py                 # frozen-feature 状态回归 probe（GO/NO-GO 闸门）
│   ├── plan_demo.ipynb
│   └── results/                 # planning demo、probe 图、cache 加速数
├── module2_vla_rl/
│   ├── simplevla_2gpu/          # 降配 veRL configs + mesh 改造 patch
│   ├── reward_binary.py         # vanilla LIBERO success reward（baseline/兜底）
│   ├── reward_latent_wm.py      # V-JEPA2-AC latent-WM reward（主角）
│   ├── ablations/               # KL 系数 / rollout budget / shaping 开关 / reward-hacking
│   └── results/
├── module3_distill/
│   ├── onpolicy_distill.py      # RL'd policy self-distillation（KL anchoring）
│   └── results/
├── infra_notes/                 # Module-4 副产品：rollout cache / 2-GPU 移植 / batched latent infer
├── configs/  scripts/           # launch .sh（沿用 AttnRes 的 smoke + sentinel 习惯）
```

外部关键资产（已调研确认开源）：
- `facebookresearch/vjepa2`：AC predictor 经 `torch.hub.load('facebookresearch/vjepa2','vjepa2_ac_vit_giant')`（~300M，Droid 训练，block-causal attn）。⚠️ 仓库**只有 energy-landscape notebook，没有打包 CEM planner，也没有 rollout KV-cache** → 这两个是亲手补的贡献点。
- `PRIME-RL/SimpleVLA-RL`（基于 veRL；GRPO-style；OpenVLA-OFT + LIBERO；默认 8-GPU colocated → 需改 2 卡）。
- `moojink/openvla-oft`（7B）+ HF ckpt `moojink/openvla-7b-oft-finetuned-libero-{spatial,object,goal,10}`。
- `Lifelong-Robot-Learning/LIBERO`（robosuite/MuJoCo，CPU-bound）。
- `LatentActionPretraining/LAPA`（latent action 蒸馏，Module-3 备选）。
- ⚠️ **待核实再引用**：调研给的 JEPA-VLA / VLA-JEPA 的 arXiv ID（2602.x）可能不准 —— 复现 probe 时先核实确切论文与 probing 方法，方案里只复用"方法论"，不硬绑某个 ID。

---

## 执行流程（4 周，每周一个 checkpoint）

### Week 1 — Module 1：V-JEPA2-AC 表征 hands-on（低风险，先拿 win）
1. 装 `vjepa2` env，load 编码器（ViT-L/16 300M 或 ViT-g）+ AC predictor。
2. `rollout.py`：实现 block-causal 自回归 latent rollout —— 给初始帧 embedding + 动作序列，逐步预测未来 latent（predictor 类在 `app/vjepa_droid/`）。
3. `cem_planner.py`（从零写）：CEM —— 高斯采样 N 条动作序列 → latent rollout → 用 latent L1/L2 到目标 embedding 的能量打分 → 取 top-k elites 重拟合高斯 → 迭代。参考 `energy_landscape_example.ipynb` 的能量计算。
4. `probe.py`：冻结 V-JEPA2 编码器，小 linear/MLP head 回归 LIBERO 轨迹的 task-relevant 状态（EE 位姿、物体位姿），对比基线编码器（**用你 AttnRes 里现成的 SigLIP** 当对照最省事，可再加 DINOv2）。
5. **infra 副产品（Module 4 埋点）**：给 block-causal rollout 加 **KV-cache**（每步只算新时刻的 K/V，而非整段重算）—— 这正是你 AttnRes cross-stage/block cache 的同构落点。测 rollout 加速比。
- **Checkpoint 1**：planning demo（rollout/goal-reaching 视频）+ 1 张 probe 图（V-JEPA2 状态回归 MSE < 基线）+ rollout-cache 加速数。
- **🚦 GO/NO-GO 闸门**：probe 若显示 V-JEPA2 特征能干净线性解码 LIBERO 的 EE/物体位姿 → Week-3 latent reward 可行；若 sim-domain gap 太大（V-JEPA2-AC 在 Droid 真机训练）→ 退路：用通用 V-JEPA2 **编码器**（非 AC predictor）做 latent-distance reward，或对编码器在 LIBERO 帧上轻量 probe-adapt。

### Week 2 — Module 2a：vanilla GRPO baseline + 2 卡移植（安全网 + Module-4 infra）
1. 装 SimpleVLA-RL + OpenVLA-OFT + LIBERO；取 OFT-LIBERO ckpt（先 `libero-spatial` 单 suite）。
2. **降配 veRL 到 2-GPU colocated**（复用你 AttnRes 的 8→2 mesh sed 经验）：total_gpus 8→2、dp/tp degree、rollout(vLLM) colocated、LoRA 让 policy+ref+rollout 挤进 2×141GB。
3. 跑 vanilla GRPO-on-LIBERO-Spatial（binary success reward）→ reward 曲线上升 + held-out success-rate eval。**这是 banked 的兜底结果。**
4. `docs/writeup.md` 记机制：advantage = group-normalized reward、group sampling、KL-to-ref（reverse KL → 接 SATS KL anchoring）、为何 on-policy 在 VLA 比 off-policy 稳（分布匹配、reward 稀疏噪声）。
- **Checkpoint 2**：2 卡 GRPO 跑通 + reward 曲线 + LIBERO 成功率 + "8-GPU 框架移植到 2 卡"的 infra 笔记。

### Week 3 — Module 2b：latent-WM reward（🌟 HERO）
核心：在 veRL 的 reward hook 里，把 LIBERO 稀疏 binary reward **替换/增强**为 V-JEPA2-AC 世界模型导出的 dense reward。选一个最干净的设计：
- **(首选) latent goal-distance reward**：把目标态编进 V-JEPA2 latent，reward = 当前观测 embedding 与目标 embedding 的负 latent 距离（即 CEM 的能量改作 RL reward，dense 且 grounded）。
- (备选) **world-model rollout/imagination reward**：用 AC predictor 想象动作序列在 latent 空间能否到目标，reward = 预测能量（WoVR/VLA-RFT 范式）。
- 实现上接 Week-1 的 `rollout.py` / 能量函数；reward 推理会成为 RL 采样瓶颈 → **batched latent inference + 编码器 forward 缓存**（Module-4 第二个副产品）。

**强制 ablation（"跑过"与"理解"的分水岭）：**
- 扫 **KL 系数 β** → 看 policy 漂移/崩溃（直接接 KL anchoring 叙事）。
- **rollout budget**（group size / 采样数）→ within-group 方差 vs 成本。
- **关掉 latent-WM shaping** → 学习速度/稳定性退化多少。
- **reward hacking 定位**：policy 是否靠移动相机/EE 压低 latent 距离但任务没成功？记录 reward hacking 出现在哪 —— 正是你要的"reward hacking 在哪发生"。
- **Checkpoint 3（HERO）**：vanilla-binary vs latent-WM reward 对比图（样本效率/最终成功率/稳定性）+ reward-hacking case study + KL 系数 ablation 曲线。
- **dense reward 的隐性好处**：你 AttnRes 踩过 GRPO zero-variance 坑（binary reward 整组全对/全错 → advantage≈0 → 无梯度）。latent-WM reward 是 dense 的，天然有 within-group 方差 → 这恰好正面回应你之前的失败教训，是个好叙事点。

### Week 4 — Module 3（蒸馏，轻量收尾）+ 打包
- **首选（最贴叙事）**：on-policy / self-distillation —— 把 latent-WM-reward 训出的 RL policy 蒸回更稳/更小的 student（KL 目标）。三合一你的 SATS 表征-KD + GRPO 经验 + DriveLM 蒸馏工程。对比蒸馏前后 performance/stability/sample-efficiency/推理成本，讲清楚蒸馏保留/丢失了什么（reverse-KL mode-seeking vs forward-KL 的取舍 = 你强项）。
- **备选/stretch**：LAPA 式 latent action 蒸馏（universal latent action，~10 条真机轨迹 few-shot），把蒸馏和表征学习绑更紧。
- **打包**：README headline 结果 + 图、`docs/writeup.md` 机制讲解、`infra_notes/` 汇总三个 Module-4 副产品。
- **Checkpoint 4**：RL'd policy 蒸馏前后对比表 + 1 张图。

---

## Module 4（分布式/infra，不单列阶段，全程机会主义收割）
1. V-JEPA2-AC block-causal rollout **KV-cache**（Week 1，接 AttnRes cache）。
2. SimpleVLA-RL **8-GPU→2-GPU 移植**（Week 2，FSDP/colocated mesh 改造 —— 真正的 infra 故事）。
3. RL rollout 的 **batched latent inference / 编码器 forward 缓存**（Week 3，latent-WM reward 的采样瓶颈驱动）。
- 交付：`infra_notes/` 里 2-3 个实测加速比/配置。不占独立工期。

---

## 风险登记（含 AttnRes 已交过的学费）
- **reward↑ ≠ task-success↑**（AttnRes GRPO 教训）→ 永远用 held-out 成功率而非 reward 作真能力 gate。
- **zero-variance group 杀死 GRPO**（AttnRes POPE-RL 教训）→ latent-WM dense reward 正好缓解；vanilla binary 单 suite 要确认有方差。
- **2 卡 veRL infra 是时间黑洞**（monarch/torchstore/mesh 那套）→ 给 Week-2 移植设硬 timebox；若 >3 天卡死，**RIPT-VLA 作 fallback**（RLOO≈无 critic GRPO，2 卡复现性最好）。
- **V-JEPA2-AC 的 sim/real domain gap**（Droid 真机 vs LIBERO 仿真）→ Week-1 probe 闸门先验证；退路见 Week-1。
- **latent-WM reward hacking** = feature 不是 bug，记录即交付。
- **LIBERO headless/EGL** 安装摩擦 → 早装早排。
- **论文 ID 待核实**（JEPA-VLA / VLA-JEPA）→ 引用前 WebFetch 核对。

---

## 验证（端到端怎么测）
- **Module 1**：`cem_planner.py` 在 Droid/goal 例子上跑，确认 latent 能量随 CEM 迭代下降；`probe.py` 确认 V-JEPA2 probe MSE < 基线；rollout 开/关 KV-cache benchmark 加速。
- **Module 2**：先 8-step GRPO smoke（沿用 AttnRes smoke + sentinel 习惯）确认 reward 非零方差，再全量跑 → held-out LIBERO 成功率 eval（**不只看 reward**）。binary vs latent-WM reward 同条件对比。
- **Module 3**：蒸馏后 student vs RL'd teacher 在 held-out LIBERO suite 上比成功率/稳定性/推理延迟。

---

## 简历产出（每模块一条 bullet）
- M1：复现并理解 V-JEPA 2-AC 世界模型，实现 latent rollout + CEM planner，probe 量化"该表征比普通 video encoder 多懂了哪些物理量"。
- M2：在 2×H200 复现 GRPO-on-VLA，并**用世界模型 latent reward 替代稀疏 reward**，对比样本效率/稳定性 + reward-hacking 分析。
- M3：把 RL 探索出的收益自蒸馏回更稳/更小 policy，量化保留/丢失（reverse-KL anchoring）。
- M4：3 个 infra 副产品 —— rollout KV-cache、8→2 卡 RL 框架移植、batched latent inference 加速。
</content>
</invoke>
