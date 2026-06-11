# 线性化视频 WM/VLA 蒸馏 — 下一步改造方向

> 状态(2026-06-11):75% KDA+AttnRes 线性化 Causal-Forcing(Wan2.1-1.3B AR)已跑通。
> 最终可复现结果(our-student vs our-teacher,VBench Quality-7,944 prompts):
> **teacher 88.09 / student(cont-9k) 83.84 (−4.26)**;ARL²(GDN) 自报 86.71 为不可复现软参照。

## 0. 经验动机(为什么需要改造,而不是再训)

补训实验(5e-5×6000 SGDR warm-restart,3K→9K 步)给出关键事实:

- distill_mse floor:0.065 → 0.0625(深退火 lr→1e-9,**仅 ~5% 边际**)。
- VBench:83.48 → 83.84(**仅 +0.36**,且几乎全在 dynamic_degree 79.17→83.33)。

**结论:预算杠杆已耗尽,且 velocity-MSE floor 与质量弱耦合。** −4.26 的 gap 只能靠
**算子 / 损失 / 架构创新** close,不是更多训练、也不是更容易的 50% ratio(同算子同 recipe)。

gap 结构(指明该上哪些招):
| 维度 | Δ(student−teacher) | 归因 |
|---|---|---|
| dynamic_degree | −12.50 | AR rollout 相干性 / 时序算子容量 |
| subject_consistency | −5.10 | 长程身份保持(物体恒存) |
| background_consistency | −4.14 | 长程场景保持 |
| imaging / aesthetic | −3.73 / −3.19 | 感知细节(高频,velocity floor 所在) |
| temporal_flicker / motion_smooth | −0.33 / −0.82 | 已基本追平 |

## 1. 真正的瓶颈定义

"丢失长程相关性"的根因 = **纯固定尺寸线性递归 + 快速衰减门**:状态矩阵 S(d×d)要把所有
过去帧压成定长,衰减一旦偏快,远处帧信息被指数抹掉。所有"不丢长程"的改造,本质都在回答:
**远程信息要么被慢速保留、要么被稀疏直取、要么被显式监督。**

视频 ≠ LM 长上下文:LM 是从基本唯一的 token 历史里**检索**(大海捞针);视频帧间是**压缩
冗余、平滑演化的观测**(信息率低、相关历史≈近期+少数 anchor)。→ 要吃冗余红利,别照搬 LM 的
KV 压缩范式。

---

## 2. 算子轴(operator)

**a) 状态容量解耦 / 深度记忆。** KDA 每步 rank-1 delta 更新,矩阵 state 秩受限。视频每帧信息量
远大于 LM 一个 token。
- **DeltaProduct**(Siems 2025):每 token n 次 delta = 每步有效秩 n,代价线性。
- **Titans / TTT**:递归 state 是推理时用梯度下降更新的 MLP(非线性深度记忆),容量远超线性矩阵。

**b) 多时间尺度 / 复数模 state(直接对症丢长程)。** 不要全层同一衰减门:一部分通道快衰减(抓
运动)、一部分慢衰减或**复数振荡模**(抓场景身份 / 周期运动 — 实指数必然遗忘,复数模能记周期结构)。
极小代价显式保长程。Clockwork-RNN / 多尺度 SSM 思路。

**c) 学习式稀疏帧选取(= DeepSeek 机制视频化,见 §5)。** 线性递归扛稠密近程,叠一个**学习的
top-k 帧选择器做稀疏全注意力**直取远程。代价 O(F·k)。视频局部性强+稀疏远程召回,天然适配。

**d) 层级化(chunk-local 线性 + chunk-global softmax)。** 帧分块,块内线性递归 + 每块蒸出几个
summary token,块间 summary 全 softmax。O(F)+O((F/c)²),F/c 小。长程靠 summary 保。

---

## 3. 损失轴(loss)— ROI 最高

**a) ★ JEPA / 表征空间蒸馏(首选).** velocity-MSE 在扩散噪声空间,被高频细节主导(线性算子最难
匹配又最不影响动力学)。改成在**冻结自监督隐空间**(V-JEPA2 / DINOv2 / VideoMAE)匹配 student vs
teacher 的 rollout 隐轨迹:
- 直接优化 VBench 真正奖励的语义+动态保真,**绕开与质量脱钩的 velocity floor**。
- V-JEPA2 隐空间是**预测性训练**的 → 在其空间匹配 = 显式监督时序预测结构 = **保长程**。
- 复用 M1 V-JEPA2-AC 资产,几乎零额外成本。精准打 subject/background-consistency + dynamic 三个最大 gap。

**b) 多步 rollout-consistency 损失(对症 AR 漂移 = dynamic_degree gap).** 现在 teacher-forced 逐步
匹配;改成 student 自回归 unroll K 帧(不喂 teacher)再匹配 teacher rollout(scheduled-sampling /
DAgger for video)。训练递归 state 长 horizon 保持相干。

**c) 注意力图结构蒸馏.** Stage-1 只匹配输出 y;teacher softmax 矩阵编码"哪些帧 attend 哪些帧"。蒸
attn 路由结构(top-k 被关注帧 / 相对熵),让线性算子学会像 teacher 路由 → 泛化到更长视频;还能反过来
监督 §2c 的帧选择器(teacher attention mass 当 label)。

**d) 频率分解 / 混真数据目标.** velocity 拆低频(运动结构,可匹配)+ 高频(纹理,floor 所在且最不
重要),低频加权;混一点真数据 flow-matching(不锁死在 teacher 精确 velocity,可能跳更好盆地)。

---

## 4. 架构轴 / 视频原生(吃帧冗余,LM 无对应)

**a) ★ 时序 token 合并 / delta-frame.** 相邻帧高度冗余。时序算子**之前**跨帧合并/剪枝冗余 token
(ToMe / 静态区 run-length),**只让变化的 token 进递归**,静态背景由廉价持久 state 兜。等效 F 对静态
区大幅缩小,省成本不丢信息(静态=可预测)。= 视频编解码 I/P/B 帧直觉,LM 无对应物。

**b) 记忆库 / landmark(I 帧化).** 周期性把一段帧压成几个全精度 memory token 存库;当前帧对小
memory 库+近期帧做全 softmax。O(F)写、O(库)读,长程靠库保,廉价因库小(Gist / AutoCompressor)。

**c) Slow-Fast 双流.** 慢流(低帧率、语义、长记忆、帧少能上更全注意力)+ 快流(高帧率、局部、线性)。
长程由慢流廉价承载,和深度轴 AttnRes 正交可组合。

---

## 5. DeepSeek-V4 / 超长上下文 LM 相关性判定

- **稀疏选择(NSA / DSA lightning-indexer+top-k / MoBA 块路由):高度可搬,且视频比文本更适配** —
  就是"关键帧选取"的可微学习版。**从 LM 能搬的最强一招**(= §2c)。
- **MLA 低秩 KV 压缩:部分相关** — 压 per-token KV(省显存),仍对全部帧注意力,**不降 F² 计算**。次要。
- **LM 上下文压缩范式:不可直接搬** — 视频是压冗余平滑观测,非检索唯一 token。→ §4 的 token-merge /
  delta-frame / slow-fast 这些吃冗余的招 LM 根本没有。

---

## 6. 优先级与论点

**论点(相对 ARL² naive GDN-swap 的真差异):**
> "线性化视频 WM 不该只换时序算子,要利用帧冗余+平滑演化:(1) delta-frame/ToMe 压冗余 token,
> (2) 门控线性递归扛稠密近程,(3) 学习式稀疏 keyframe(NSA/DSA)直取远程,(4) 在 JEPA 预测性隐
> 空间蒸馏(而非噪声空间 velocity)。"

**执行优先级(ROI / 风险):**
1. **§3a JEPA 隐空间蒸馏损失** — 最便宜(改一个 loss)、复用 M1 资产、对症三大 gap。**首测。**
2. **§4a 时序 token-merge / delta-frame** — 纯效率,视频专属,与 KDA+AttnRes 正交。
3. **§2c/§5 学习式稀疏 keyframe 选择器(DeepSeek NSA/DSA 视频化)** — 工程量中等,架构新意最大。

**保长程判据(选改造时盯):** 纯快衰减定长 state 必丢长程;真正保住的是 — 慢衰减/多时间尺度通道
(§2b)、稀疏全局帧直取(§2c)、memory 库(§4b)、JEPA 预测性损失显式监督(§3a)。

---

## 7. 决策记录(2026-06-11)

- **不重训干净 12K**:前段震荡(梯度噪声平台 std 18%)、后段学不动(容量地板,深退火仅挪 5%),
  调度非瓶颈;ARL² 不公开 code/data/lr/loss → 86.71 不可复现,不作硬指标。
- **不自动跑 50% KDA**:同算子同 recipe,不会 close −4.26 的 gap。
- **verdict 指标 = 可复现 our-student-vs-our-teacher**;ARL² −0.38 仅软参照。
- **headline 重定位**:(a) 可复现 teacher-vs-student 线性化 gap(无维度坍塌)+ (b) 真卖点 = 效率
  (KDA 定长 state + AttnRes 长度泛化:训 81f 跑 921f 不重训,2.26×/54% F-sweep)。
</content>
