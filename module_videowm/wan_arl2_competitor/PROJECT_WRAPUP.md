# 项目收尾总结 — WM/VLA 线性化蒸馏全线回顾

> v1 (2026-06-12 morning)。run-2(T=8)/ F-sweep / conthead eval 出数后更新 v2。

## 0. 一句话

把 Wan2.1-AR 视频世界模型 75% 线性化(KDA 时序 + AttnRes 深度 + intra-softmax 分解),
**可复现 teacher−4.2、无维度坍塌**;逐项归因 ablation(算子 → 补训 → REPA-JEPA 损失);
诊断并修复两类 KDA 推理态 bug(其一值 +15pt);REPA 式表征蒸馏用于线性化是空白点,首个数据点已拿到。

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
