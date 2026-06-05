"""Distillation losses for linearizing WorldVLA (SDPA teacher -> KDA+AttnRes student).

Paradigm = the standard WM/VLA linearization-transfer recipe (not full-FT-from-scratch):
  - ARL2 "Attend Locally, Remember Linearly" (2605.16579): per-layer attention-output MSE +
    joint output matching, frozen teacher, 50/75% layers linearized.
  - VLA distillation (VITA-VLA 2510.09607, ActDistill 2511.18082): distill the ACTION head/tokens.

Our fitting-gap diagnosis showed the loss gap is concentrated in ACTION tokens (l1 0.018 vs 0.14)
while image/world-model tokens are already at parity — because action tokens are ~7/263 of the
sequence and get drowned out in the image-dominated CE. So we (a) weight the LM CE on action
positions and (b) add logit-KD focused on action positions, plus optional ARL2-style attention MSE.

All functions are pure (no model deps) so they unit-test on CPU.
"""

import torch
import torch.nn.functional as F

# WorldVLA action tokens: decoded via (id - 1 - 10004), delimited by start=10004 / end=15004.
# The action VALUE tokens therefore live strictly between the markers.
ACTION_START_ID = 10004
ACTION_END_ID = 15004


def action_token_mask(labels: torch.Tensor, lo: int = ACTION_START_ID + 1, hi: int = ACTION_END_ID - 1) -> torch.Tensor:
    """Boolean mask [.. ] of positions whose target token is an action VALUE token."""
    return (labels >= lo) & (labels <= hi)


def _shift(logits: torch.Tensor, labels: torch.Tensor):
    """Standard causal-LM shift: logits[t] predicts labels[t+1]."""
    return logits[..., :-1, :].contiguous(), labels[..., 1:].contiguous()


def logit_kd_loss(
    student_logits: torch.Tensor,   # [B, T, V]
    teacher_logits: torch.Tensor,   # [B, T, V]
    labels: torch.Tensor,           # [B, T]
    T: float = 1.0,
    action_only: bool = True,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Forward-KL distillation KL(teacher || student) at temperature T, averaged over positions.

    action_only=True restricts to action-token positions (the control-relevant gap). Returns a
    scalar; 0 if no valid positions in the batch.
    """
    s_logits, s_labels = _shift(student_logits, labels)
    t_logits, _ = _shift(teacher_logits, labels)

    valid = s_labels != ignore_index
    if action_only:
        valid = valid & action_token_mask(s_labels)
    if valid.sum() == 0:
        return student_logits.new_zeros(())

    s = s_logits[valid]                       # [N, V]
    t = t_logits[valid]
    s_logp = F.log_softmax(s.float() / T, dim=-1)
    t_p = F.softmax(t.float() / T, dim=-1)
    # batchmean KL(teacher || student); *T^2 keeps gradient scale ~constant across temperatures.
    return F.kl_div(s_logp, t_p, reduction="batchmean") * (T * T)


def action_weighted_ce(
    logits: torch.Tensor,           # [B, T, V]
    labels: torch.Tensor,           # [B, T]
    action_weight: float = 5.0,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Cross-entropy with action-token positions up-weighted (cheap alternative/companion to KD).

    Counteracts action tokens being drowned by the 256 image tokens in the standard CE.
    """
    s_logits, s_labels = _shift(logits, labels)
    V = s_logits.size(-1)
    flat_logits = s_logits.view(-1, V)
    flat_labels = s_labels.view(-1)
    per_tok = F.cross_entropy(flat_logits.float(), flat_labels, ignore_index=ignore_index, reduction="none")
    valid = flat_labels != ignore_index
    w = torch.ones_like(per_tok)
    w[action_token_mask(flat_labels)] = action_weight
    w = w * valid.float()
    denom = w.sum().clamp_min(1.0)
    return (per_tok * w).sum() / denom


def attn_output_mse(student_outs: dict, teacher_outs: dict, layer_ids) -> torch.Tensor:
    """ARL2/LoLCATs-style: mean ||student_o - teacher_o||^2 over the linearized (KDA) layers.

    student_outs/teacher_outs: {layer_idx: [B, T, D]} attention-block outputs captured via hooks.
    """
    losses = []
    for l in layer_ids:
        if l in student_outs and l in teacher_outs:
            losses.append(F.mse_loss(student_outs[l].float(), teacher_outs[l].float()))
    if not losses:
        dev = next(iter(student_outs.values())).device if student_outs else "cpu"
        return torch.zeros((), device=dev)
    return torch.stack(losses).mean()
