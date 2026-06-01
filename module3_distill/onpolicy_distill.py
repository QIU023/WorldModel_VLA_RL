"""RL'd policy self-distillation (KL anchoring) —— Module 3.

把 latent-WM-reward 训出的 RL policy 蒸回更稳/更小的 student。
三合一: SATS 表征-KD + GRPO 经验 + DriveLM 蒸馏工程。

KL 取舍 (你的强项):
  - reverse-KL (mode-seeking): 匹配 teacher 高概率区, 防 hallucination, policy 蒸馏首选
  - forward-KL: 覆盖全 support, 削弱专精

对比蒸馏前后: success rate / stability / sample-efficiency / 推理延迟 → results/。
讲清楚蒸馏保留/丢失了什么。

备选 stretch: LAPA 式 latent action 蒸馏 (universal latent action, ~10 条真机轨迹 few-shot)。

interface (待实现):
    student = distill(teacher=rl_policy, student_init, kl='reverse', ...)
"""
# TODO(week4): 实现 self-distill loop; 蒸馏前后对比表+图
