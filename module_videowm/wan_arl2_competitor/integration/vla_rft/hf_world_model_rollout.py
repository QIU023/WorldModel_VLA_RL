"""HF interact rollout for the VLA-RFT *world model* (KDA backbone).

Drop-in replacement for the vLLM WM rollout (vllm_rollout.py:159-308) when the WM has
been linearised with KDA. KDA uses custom ``fla`` Triton kernels + a module-level
recurrent state (kda_llama.py:77-88) that vLLM's paged-attention executor cannot run,
so the action-conditioned imagined rollout must move to plain HF generation.

This mirrors the vLLM interact loop *exactly* (vllm_rollout.py:231-242):

    for t in range(action_ids.shape[1] - 1):        # segment_length-1 - 1 = 7 steps
        frame = generate(seq, max_new_tokens=tokens_per_frame=64, use_cache=True)
        seq  += frame                               # 64 predicted dynamics tokens
        seq  += action_ids[:, t+1]                  # 7 discretised action ids

and returns the identical DataProto contract consumed downstream:
    - msp_reward_fn reshapes ``responses`` to [B, segment_length-1, tokens_per_frame+action_dim]
      (ray_trainer.py:1306) -> response_length = 8 * (64+7) = 568 (padded).
    - keys: prompts, responses, input_ids, attention_mask, position_ids [, gt_responses].

Place this file at:  train/verl/verl/workers/rollout/hf_world_model_rollout.py
and register it in   train/verl/verl/workers/rollout/__init__.py (see patches/).

KDA + HF generate correctness (the key reason this works at all):
    KDAttention.forward (kda_llama.py:77-88) keeps its fixed-size recurrent state on the
    MODULE. On prefill (cache_position[0]==0) it resets; on each subsequent 1-token decode
    step (cp0>0) it carries the state forward -> O(1)/step *within* a generate() call. So
    each per-frame ``generate(max_new_tokens=64, use_cache=True)`` is already recurrent for
    the 64 dynamics tokens. Between frames we re-prefill the grown prefix (option (b) in the
    integration plan): simple, robust, and N(~600) is tiny. For true end-to-end O(N) set
    ``rollout.hf_recurrent=True`` to use the manual recurrent decoder below, which threads
    _kda_state across the interleaved action tokens (option (a)) and never re-prefills.

We process per-sample (ragged), matching vLLM's ``_pre_process_inputs`` left-pad strip.
This is also the safe choice for KDA, which ignores ``attention_mask`` (it is causal by
construction) -- so feeding left-pad tokens would corrupt the recurrent state. In the
LIBERO config the WM prompt is uniform length (gen_input_length == prompt_length == 1095,
no real left padding), so this is equivalent to vLLM but robust if that ever changes.
"""
import torch
import torch.distributed
from tensordict import TensorDict
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from transformers import GenerationConfig

from verl import DataProto
from verl.utils.torch_functional import get_response_mask, pad_sequence_to_length
from .base import BaseRollout

__all__ = ["HFWorldModelRollout"]


def _strip_left_pad(pad_token_id, row: torch.Tensor):
    """Remove left padding from a 1-D prompt id tensor (mirror vllm _pre_process_inputs)."""
    nz = torch.nonzero(row != pad_token_id, as_tuple=False)
    if nz.numel() == 0:
        return row.tolist()
    return row[nz[0, 0]:].tolist()


class HFWorldModelRollout(BaseRollout):

    def __init__(self, world_module: nn.Module, config):
        super().__init__()
        self.config = config
        self.module = world_module                       # FSDP-wrapped LlamaForCausalLM (KDA)
        self.pad_token_id = config.pad_token_id if hasattr(config, "pad_token_id") else None
        if self.pad_token_id is None:
            # WorldModelRolloutWorker injects eos/pad into meta_info; fall back at call time.
            self.pad_token_id = config.get("pad_token_id", None)

    # ---- sampling config helper (mirror vllm_rollout do_sample / is_validate / interact) ----
    def _gen_config(self, eos_token_id, pad_token_id):
        do_sample = self.config.get("do_sample", True)
        is_validate = self.config.get("is_validate", False)
        gc = dict(
            max_new_tokens=int(self.config.get("interact_max_tokens", 64)),
            use_cache=True,                              # <-- KDA recurrent decode within a frame
            do_sample=True,
            num_return_sequences=1,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
        )
        if not do_sample:                               # greedy
            gc.update(do_sample=False, temperature=1.0, top_p=1.0, top_k=0)
        elif is_validate:
            vk = self.config.val_kwargs
            gc.update(
                temperature=float(vk.temperature),
                top_p=float(vk.top_p),
                top_k=(0 if int(vk.top_k) < 0 else int(vk.top_k)),  # hf: 0 disables, vllm: -1
            )
        else:                                           # training rollout sampling
            tk = int(self.config.get("top_k", -1))
            gc.update(
                temperature=float(self.config.get("temperature", 1.0)),
                top_p=float(self.config.get("top_p", 1.0)),
                top_k=(0 if tk < 0 else tk),
            )
        # WM never stops early during interact: ignore eos, always emit exactly 64 tokens.
        if self.config.get("ignore_eos", True):
            gc["eos_token_id"] = None
        return GenerationConfig(**gc)

    def generate_actions(self, prompts):
        raise NotImplementedError("HFWorldModelRollout only supports generate_sequences (WM rollout).")

    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        assert self.config.interact, "HFWorldModelRollout only supports interact mode."
        device = prompts.batch["input_ids"].device
        idx = prompts.batch["input_ids"]                # [B, P]  (uniform, gen_input_length)
        attention_mask = prompts.batch["attention_mask"]
        position_ids = prompts.batch["position_ids"]
        actions = prompts.batch["action_ids"]           # [B, T-1, 7]  discretised action ids
        w_gt_ac = self.config.get("w_gt_ac", False)
        gt_actions = prompts.batch["gt_action_ids"] if w_gt_ac else None

        eos_token_id = prompts.meta_info.get("eos_token_id", None)
        pad_token_id = prompts.meta_info.get("pad_token_id", self.pad_token_id)
        gen_config = self._gen_config(eos_token_id, pad_token_id)
        n_steps = actions.shape[1] - 1                  # mirror vllm: range(actions.shape[1]-1)

        # summon full (sharded) params for generation, like the policy HFRollout does.
        ctx = (FSDP.summon_full_params(self.module, writeback=False)
               if isinstance(self.module, FSDP) else torch.autocast("cuda", dtype=torch.bfloat16))
        model = self.module
        model.eval()

        batch_size = idx.size(0)
        responses = []                                  # per-sample appended tokens (no prompt)
        gt_responses = [] if w_gt_ac else None

        recurrent = bool(self.config.get("hf_recurrent", False))

        with ctx:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                for j in range(batch_size):
                    prompt_ids = _strip_left_pad(pad_token_id, idx[j])
                    act_j = actions[j]                  # [T-1, 7]
                    if recurrent:
                        resp = self._rollout_recurrent(model, prompt_ids, act_j, n_steps, gen_config)
                    else:
                        resp = self._rollout_generate(model, prompt_ids, act_j, n_steps, gen_config, device)
                    responses.append(resp)

                    if w_gt_ac:
                        # FAITHFUL MIRROR of vllm_rollout.py:216-229: the shipped gt loop
                        # regenerates each frame from the *original* prompt (it never grows the
                        # gt sequence). We replicate that exactly for numerical parity. See
                        # RUN_REQUIREMENTS.md (B7) -- this looks like an upstream quirk; flip
                        # `hf_gt_grows=True` to instead roll out gt autoregressively (growing).
                        if self.config.get("hf_gt_grows", False):
                            gresp = (self._rollout_recurrent if recurrent else
                                     (lambda *a: self._rollout_generate(*a, device)))(
                                         model, prompt_ids, gt_actions[j], n_steps, gen_config)
                        else:
                            gresp = self._rollout_gt_static(model, prompt_ids, gt_actions[j],
                                                            n_steps, gen_config, device)
                        gt_responses.append(gresp)

        response = self._stack_pad(responses, pad_token_id, device)
        out = self._assemble(idx, response, attention_mask, position_ids,
                             eos_token_id, pad_token_id, batch_size)
        if w_gt_ac:
            gt_response = self._stack_pad(gt_responses, pad_token_id, device)
            out.batch["gt_responses"] = gt_response
        return out

    # ---------- rollout variant (b): per-frame HF generate, re-prefill the grown prefix ----------
    def _rollout_generate(self, model, prompt_ids, act_j, n_steps, gen_config, device):
        tpf = int(self.config.get("interact_max_tokens", 64))
        seq = list(prompt_ids)
        resp = []
        for t in range(n_steps):
            inp = torch.tensor([seq], device=device, dtype=torch.long)
            am = torch.ones_like(inp)
            out = model.generate(input_ids=inp, attention_mask=am, generation_config=gen_config)
            frame = out[0, inp.shape[1]:inp.shape[1] + tpf].tolist()
            act = act_j[t + 1].detach().cpu().tolist()      # mirror vllm: actions[j, t+1]
            seq += frame + act
            resp += frame + act
        return resp

    # ---------- rollout variant (a): manual recurrent decode, true O(N), no re-prefill ----------
    def _rollout_recurrent(self, model, prompt_ids, act_j, n_steps, gen_config):
        """Thread KDA's module-level _kda_state across the whole imagined trajectory.

        Prefill the prompt once (cache_position[0]==0 -> KDA resets). Then every token --
        sampled frame tokens AND the injected action tokens -- is fed as a single-step
        forward with cache_position>0, so KDA carries its recurrent state forward and never
        re-prefills. This is the O(1)/step decode that is KDA's selling point.

        Hybrid (Kimi N:1) note: any *softmax* layers left in the backbone DO need a KV cache
        threaded across the manual steps (KDA layers ignore it and use _kda_state instead).
        We pass a single growing ``DynamicCache`` for that; KDA layers leave it untouched. For
        pure-KDA (kda_layers=null) the cache stays empty and adds no cost. We pass an explicit
        cache_position every step so the model never relies on DynamicCache.get_seq_length()
        (which would be 0 when layer 0 is KDA and holds no cache entry).
        """
        from transformers.cache_utils import DynamicCache
        tpf = int(self.config.get("interact_max_tokens", 64))
        dev = next(model.parameters()).device
        do_sample = gen_config.do_sample
        temperature = float(getattr(gen_config, "temperature", 1.0) or 1.0)
        top_p = float(getattr(gen_config, "top_p", 1.0) or 1.0)
        top_k = int(getattr(gen_config, "top_k", 0) or 0)
        kv = DynamicCache()                                  # for residual softmax layers (hybrid)

        def step(token_ids_1d, pos0):
            inp = torch.tensor([token_ids_1d], device=dev, dtype=torch.long)
            cache_position = torch.arange(pos0, pos0 + inp.shape[1], device=dev)
            position_ids = cache_position.unsqueeze(0)
            out = model(input_ids=inp, use_cache=True, past_key_values=kv,
                        cache_position=cache_position, position_ids=position_ids)
            return out.logits[:, -1, :]                     # [1, vocab]

        def sample(logits):
            if not do_sample:
                return int(logits.argmax(-1))
            logits = logits.float() / max(temperature, 1e-6)
            if top_k and top_k > 0:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float("inf")
            if top_p and top_p < 1.0:
                s, si = torch.sort(logits, descending=True)
                cum = torch.softmax(s, dim=-1).cumsum(-1)
                mask = cum - torch.softmax(s, dim=-1) > top_p
                s[mask] = -float("inf")
                probs = torch.softmax(s, dim=-1)
                return int(si[0, torch.multinomial(probs[0], 1)])
            probs = torch.softmax(logits, dim=-1)
            return int(torch.multinomial(probs[0], 1))

        seq = list(prompt_ids)
        logits = step(seq, 0)                               # prefill (resets _kda_state)
        pos = len(seq)
        resp = []
        for t in range(n_steps):
            for _ in range(tpf):                            # 64 dynamics tokens
                tok = sample(logits)
                resp.append(tok)
                logits = step([tok], pos)                   # cp0>0 -> carry KDA state
                pos += 1
            act = act_j[t + 1].detach().cpu().tolist()      # 7 action tokens
            resp += act
            logits = step(act, pos)                         # advance state over action tokens
            pos += len(act)
        return resp

    # ---------- faithful mirror of vllm gt loop (regenerate each frame from original prompt) ----------
    def _rollout_gt_static(self, model, prompt_ids, gt_act_j, n_steps, gen_config, device):
        tpf = int(self.config.get("interact_max_tokens", 64))
        resp = []
        inp = torch.tensor([list(prompt_ids)], device=device, dtype=torch.long)
        am = torch.ones_like(inp)
        for t in range(n_steps):
            out = model.generate(input_ids=inp, attention_mask=am, generation_config=gen_config)
            frame = out[0, inp.shape[1]:inp.shape[1] + tpf].tolist()
            act = gt_act_j[t + 1].detach().cpu().tolist()
            resp += frame + act                             # NB: inp is NOT grown (matches vllm)
        return resp

    # ---------- output assembly (identical to vllm_rollout.py:273-308) ----------
    def _stack_pad(self, rows, pad_token_id, device):
        maxlen = max(len(r) for r in rows)
        t = torch.full((len(rows), maxlen), pad_token_id, dtype=torch.long, device=device)
        for i, r in enumerate(rows):
            t[i, :len(r)] = torch.tensor(r, dtype=torch.long, device=device)
        if t.shape[1] < self.config.response_length:
            t = pad_sequence_to_length(t, self.config.response_length, pad_token_id)
        return t

    def _assemble(self, idx, response, attention_mask, position_ids,
                  eos_token_id, pad_token_id, batch_size):
        seq = torch.cat([idx, response], dim=-1)
        response_length = response.size(1)
        delta = torch.arange(1, response_length + 1, device=position_ids.device)
        delta = delta.unsqueeze(0).repeat(batch_size, 1)
        response_position_ids = position_ids[:, -1:] + delta
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        response_attention_mask = get_response_mask(
            response_id=response,
            eos_token=(eos_token_id if eos_token_id is not None else pad_token_id),
            dtype=attention_mask.dtype,
        )
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)
        batch = TensorDict(
            {
                "prompts": idx,
                "responses": response,
                "input_ids": seq,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            },
            batch_size=batch_size,
        )
        return DataProto(batch=batch)
