# 项目收尾总结 — WM/VLA 线性化蒸馏全线回顾

> v2 (2026-06-13)。run-2(T=8)=84.00 / F-sweep=诚实负 / conthead=partial 已填入。

## 0. 一句话

把 Wan2.1-AR 视频世界模型 75% 线性化(KDA 时序 + AttnRes 深度 + intra-softmax 分解),
**可复现 teacher−4.1(84.0)、无维度坍塌**;逐项归因 ablation(算子 → 补训 → REPA-JEPA 损失);
诊断并修复两类 KDA 推理态 bug(其一值 +15pt);REPA 式表征蒸馏用于线性化是空白点,首个数据点已拿到。
**效率是诚实负结果**:此教师本就有界 21 帧注意力,线性化无 headroom。

## 0.5 跨 paradigm 总结(全线一句话:质量可逼近、分解是必须、效率看 base、VLA 瓶颈在接口)

| # | Paradigm | 判决 | 核心数 |
|---|---|---|---|
| ① | JEPA-WM 隐空间(M1/M2) | 🔴 dead-end(遗产复用) | 隐空间太小,KDA 长序列优势出不来;V-JEPA2 编码器后变 REPA 目标 |
| ② | dino_wm / point_maze | 🟢 正 | AttnRes 0.370 < SDPA 0.416,KDA **13×** 加速 |
| ③ | iVideoGPT / KDA-LLaMA | 🟢 正 | **2.47×** @65k token,长序列效率 thesis 在 LM 侧成立 |
| ④ | Cosmos 线性化 | 🟠 诚实负(错 base) | 输出蒸馏逼近有效(LPIPS 0.85→0.50),但 full-3D 时空耦合 → dynamic 0.92→0.49,效率无赢(<66k) |
| ⑤ | WorldVLA-7B KDA(VLA) | 🔴 算子非瓶颈 | 75% 离散闭环 **0%**(离散动作误差复合)、50%=2%;conthead 修法 eval partial 偏低 |
| ⑥ | ARL² 竞品 / Causal Forcing(headline) | 🟡 质量成立、效率负 | Quality **84.0** vs teacher 88.09 无坍塌;REPA 空间 +1.9 / dynamic −4.2;效率负 |

**跨线规律(真 take-aways):**
1. **质量逼近可行,但效率 win 只在"长序列 × 无界 softmax"成立。** ②③(纯时序/LM,序列真长)有 win;④⑥(生产级 video teacher,时空耦合或**有界局部注意力**)无 headroom。线性化效率卖点高度依赖教师是否真有"长无界注意力可省"——最反直觉、最重要。
2. **分解(intra-softmax + inter-linear)是必须的**:uniform 把空间轴也线性化 → 动态崩(Cosmos)→ 催生 ⑥ 正确设计。
3. **VLA 瓶颈是离散动作接口,不是算子**(WorldVLA 离散 0%)→ 连续头才是对的修法。
4. **REPA 表征蒸馏有效但与动态有张力**:空间 +1.9,代价 dynamic −4.2(对齐平滑 latent 轨迹压制运动)。
5. **推理递归态是纪律级的坑**:两个 KDA 推理 bug(decode-cache +15pt、streaming-state)→ 先验证递归态再信指标。
6. **loss floor ≠ 质量;不可复现他报数只能软参照**(ARL² 86.71 无 code/data/lr)。

## 1. 各线结论

### ① JEPA-WM 隐空间线(M1/M2) — 已收,dead-end 但遗产被复用
- V-JEPA2-AC rollout / CEM planner / probe + TD-MPC2/Dreamer imagination RL 全部跑通。
- 结论:范式复现成立;但作为"线性化主战场"价值低(隐空间太小,KDA 长序列优势出不来)→ pivot。
- **遗产:V-JEPA2 编码器在 §3a 变成 REPA 蒸馏目标 —— 资产复用。**

### ② dino_wm / point_maze — 已收(正)
- AttnRes 0.370 < SDPA 0.416;KDA 13× 加速。算子组合在小 WM 成立,是后续手术的信心来源。

### ③ iVideoGPT / KDA-LLaMA — 已收(正)
- KDA 2.47× @65k token。长序列效率 thesis 在 LM 侧成立。

### ④ Cosmos 线性化 — 已收(诚实负结果,催生正确设计)
- 输出蒸馏逼近教师有效(teacher-student LPIPS 0.85→0.50)。
- 但 Cosmos 是双向 + 时空耦合 full-3D = **错误 base**:uniform 线性化把空间轴也线性化 →
  dynamic_degree 0.92→0.49(480p)。效率无赢(<66k token)。
- **确立 "intra-frame softmax + inter-frame linear" 分解的必要性** → 直接催生 ⑥。

### ⑤ WorldVLA-7B KDA 手术(VLA 侧) — 基本收
- 75% 离散头闭环 **0% 是真实的**(逐 token 离散动作误差复合,经 use_cache=False 全序列验证非 eval bug;50% 也仅 2%)。
- **结论:线性化 VLA 的瓶颈在"离散动作接口",不在算子** → 连续头(conthead, OFT/VITA 式)是对的修法。
- conthead 已训(kda_75_s2conthead_MERGED, 14G)+ decode 验证 sane(varied 7-dim, gripper 正常开合)。
- 缓存 decode 已实现+验证(teacher-forced 0.76% rel),实测仅 1.6×(瓶颈是 osmesa CPU 渲染,非 decode)。
- **公平开环 eval(CONT_NCHUNKS=25,= 离散头协议)代码就绪(commit d9a988c),pending GPU。**

### ⑥ ARL² competitor(headline) — 90% 收
可复现核心数(our-teacher vs our-student, VBench Quality-7, 944 prompts, 832×480, 81f):

| 配置 | Quality | 备注 |
|---|---|---|
| teacher(Causal Forcing) | **88.09** | |
| 75% KDA+AttnRes(S2 3k) | 83.48 | KDA 流式态 bug 修复后(修复值 +15.35) |
| + 补训(9k total) | 83.84 | 预算杠杆耗尽证明(+0.36,全在 dynamic) |
| + §3a JEPA T=2(run-1) | 83.90 | 净平;空间 4 维 +5.8 ✓,dynamic −5.55 ✗(T=2 病) |
| **+ §3a JEPA T=8(run-2, FINAL)** | **84.00** | **最终 headline;空间 subject/imaging +1.9 ✓,dynamic −4.17(REPA 本质抑制动态,T=8 仅救回 +1.39)** |
| ARL²(GDN, 自报) | 86.71 | 软参照:无 code/data/lr/loss,不可复现 |

- 已证:① loss floor(0.065)与质量弱耦合,补训/重退火无用;② ARL² 的唯一 recipe 差异是调度完整性,
  lr 论文不公布;③ REPA 式对齐在线性化蒸馏上**方法有效但与动态有张力**:空间稳定 +1.9,代价是 dynamic_degree
  系统性下降(对齐"平滑 latent 轨迹"的归纳偏置压制运动),T 分辨率只能部分缓解 → 净 +0.16 边际胜;
- 两类 KDA 推理态 bug(WorldVLA decode-cache、Wan streaming-state)→ **纪律:先验证推理递归态再信指标**。

#### ⑥-eff 效率结论(诚实负结果,见 _ext/wan_eval/EFFICIENCY_FINDINGS.md)
- **全序列前向**:KDA 1.16–1.49× 快、**显存持平**,F>41 两者皆 OOM(flex_attention 本就省显存,且到不了长序列)。
- **streaming 部署路径(真卖点测试)**:F=21 原生长度 KDA 学生 **慢 2.5×(6.33s vs 2.50s)、重 2.4×(50.2 vs 20.8GB)**;
  且 `local_attn_size=-1` → KV-cache 21 帧硬上限无滚动,F=41 越界崩溃,进不了长序列区间。
- **根因**:Causal Forcing 本身是有界 21 帧注意力,softmax 在该长度已便宜,KDA-chunk + 朴素 AttnRes(4 taps)+ LoRA
  算子开销成净亏损;线性化的"线性 vs 二次"win 只存在于无界长序列(此教师不进入)→ **ARL² 2.26×/54% 在此教师上不可复现**。
- → 本工作可辩护贡献 = **质量(84.0 不坍塌)+ REPA-线性化数据点 + 两个 KDA 推理 bug 修复**,而非效率。

## 2. 收尾执行结果(user-locked: 只做 1-3)

1. **run-2 §3a T=8 — 完成 ✓**:V-JEPA2 重编码 [8,15,26,1408]→ smoke → S2 重训(jepa_l025_T8/)→ VBench。
   结果 **Quality 84.00**:dynamic 从 T2 的 77.78 救回 79.17(+1.39)但仍 −4.17 vs MSE;空间保住 +1.9。
   判据部分达成(dynamic 部分恢复,未到 85-86)→ REPA 与动态张力是方法本质,非配置问题。ckpt 已 commit。
2. **效率 F-sweep — 完成,诚实负结果**:见上 ⑥-eff。全序列 1.16–1.49×/显存持平;streaming F=21 学生慢 2.5×/重 2.4×;
   `local_attn=-1` 21 帧硬上限进不了长序列。**结论:此有界注意力教师上线性化不赢效率**(代码+数据已 commit)。
3. **WorldVLA conthead 开环 eval — 已启动,部分**:CONT_NCHUNKS=25 开环,2-GPU sharded(HALF2@GPU1 + HALF1@GPU0)。
   rendering-bound(osmesa CPU,分钟/集),机器释放窗口内仅跑完部分;按 mp4 文件名聚合 partial success(填见下)。
   完整 250 集/shard 需数小时,**留 pending**(启动器 commit d9a988c,可直接续跑)。

## 3. 未来 wins(写进 future work,不在本次收尾)

1. **§1b 多时间尺度/复数模 KDA state** — 唯一直接打"算子容量 gap(−4.2 根源)"的招;
   慢衰减通道保长程身份。最大潜在 win + 真正区分于 ARL²-GDN 的算子级创新。
2. **§2b 多步 rollout-consistency loss** — 打 AR 漂移(dynamic gap),与 JEPA latent 可组合,便宜。
3. **§1c 学习式稀疏 keyframe(DeepSeek NSA/DSA 视频化)** — 架构新意最大;视频的局部性+稀疏远程召回比文本更适配。
4. JEPA 损失微调空间:centering(已 opt-in)、更深/多 block tap、token-relation(VideoREPA 式)。
5. MSE-only matched 控制组(S1→6000 同 lr 无 JEPA)— 让 §3a ablation 完全干净。
6. 50% KDA Wan 点(补全 ratio 表)。
7. repo 延伸:P3 RL→policy 自蒸馏、真 VLA(OpenVLA-OFT)+ WM 交互训练、Module4 infra notes 汇总。

## 4. 教训(已固化为纪律)

- **推理递归态必须先验证再信 eval**(两次 KDA 推理 bug:WorldVLA cache、Wan streaming;光流 vs teacher 是廉价验证)。
- 改协议/换 chunk 前**先实测吞吐**,别凭理论拍加速比(conthead NCHUNKS=25 反而 5× 慢的教训)。
- steps 跟数据规模走(epochs),不抄绝对步数;bash 保留变量名(GROUPS)不要用。
- loss floor ≠ 质量:velocity-MSE 与 VBench 弱耦合,verdict 永远用端到端 eval。
- 不可复现的他报数(ARL² 86.71)只能当软参照;verdict = 自己可复现的 teacher-vs-student。

## 6. WoVR 下游扩展(2026-06-13→14 overnight)—— 管线全通,下游成功未达成(诚实)

**转向**:WAM 直接线性化失败后,改用 linearized-WM 做下游(WoVR 流派:World-in-World ICLR'26 Oral / VLA-RFT / RLVR-World)。

**已落地(全 commit)**:
- World-in-World(ICLR'26 Oral)整合:**我们的线性化 Causal-Forcing WM 端到端跑进闭环 worker**(smoke PASS,2-step 2.09s/rollout)。Habitat-sim + HM3D(非门控)+ IGNav 就绪。
- **IGNav 闭环数(诚实负)**:
  | 配置 | SR | 备注 |
  |---|---|---|
  | 启发式策略, no-LoRA | 0.10 (1/10) | 策略地板(噪声) |
  | 启发式策略, nav-LoRA | 0.00 | 噪声 |
  | **真 VLM 策略(Qwen2.5-VL-7B)+ held-out, no-LoRA** | **0.00 (0/12)** | 真策略也救不起 |
  | **真 VLM 策略 + held-out, nav-LoRA s2000** | **0.00 (0/11)** | 终距 2.46→2.11m(7/11 更近,best 0.81m)= 后训在动但不跨 0.5m 阈 |
- **nav 后训**:smoke 3/3 过,2000 步 DDP,WM 离线 motion MSE −26%,但**不传导到闭环成功**。
- VLA-RFT:代码骨架 + de-risk(RT1 base 公开),按用户 B 不做归档。

**结论**:线性化 WM 在闭环下游**机制上全通**,但**下游成功 ≈ 0**。根因不是单一线性化:
1. **可控性 / 基模型**:Wan2.1-1.3B 是通用 T2V,零样本跟随离散导航文本("turn left 22.5°")很弱;World-in-World 自己的发现 = **画质 ≠ 任务成功,可控性(action-obs 后训)>> 升级视频生成器**。
2. **线性化保真**:2-step 线性化 WM 更糊,对 VLM 规划是额外不利(幅度未单独隔离)。

→ **关键判断:换"原始二次方 WM"不会解决这个问题**(见下方问答);真正的修法 = **足量 action-observation 可控性后训**(我们的 nav-LoRA 太小:612 clips/r8)。一旦 WM 可控,**线性化的效率(2-step ~2s vs 教师 30-step ~15×)才成为卖点 = 更便宜的可控模拟器**。

**未做(下次)—— headline 干净实验 = FTcausalforcing vs FTwan21 on held-out IGNav:**
- 核心对照表:`wan21`(零样本二次方)/`causalforcing`(零样本,我们)都≈0(已部分验证);**`FTwan21`(后训二次方,WiW 已公开可下)vs `FTcausalforcing`(后训,我们)** = 若匹配成功率则线性化靠效率赢(2-step ~2s vs 30-step ~15×)。
- WiW 后训配方(docs/04_post_training.md):`scripts/data_collect.sh`(habitat032 env,`--trajs_num_per_scene 1`,~HM3D 1k 场景=~1k trajs)采 (obs,action,future) → 全量 FT 视频生成器(他们用 4×H100)。我们玩具版只有 612 clip/r8。
- **省时杠杆**:① FTwan21 大概率从 `hf:zonszer/WIW_ckpts` 直接下(免训二次方,省 1-2 天);② WiW 有小型 nav 后训数据集可下(数据从半天降到几小时);③ eval 慢瓶颈是 VLM ~3min/集(先 N=30 拿信号再扩 144)。
- **时间估**:A(最小可行信号,2×H200)≈ **1.5-2 天**(下/采小数据 + LoRA-r32 FT 我们的 WM + 下 FTwan21 + IGNav N=30 held-out);B(完整可发表)≈ **1 周**(全量采数据 + 全量 FT + N=144 + TTS 扫)。先 A 出正信号再上 B。
