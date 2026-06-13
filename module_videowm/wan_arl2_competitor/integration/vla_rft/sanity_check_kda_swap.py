"""Mechanical sanity check for the VLA-RFT KDA world-model integration.

Validates the load-bearing pieces WITHOUT verl / vLLM / checkpoints:
  1. apply_wm_kda_surgery() (the exact hook the patched fsdp_workers.py calls) converts a
     9008-vocab LlamaForCausalLM with a Kimi N:1 hybrid layer selection + AttnRes.
  2. forward() produces well-shaped logits.
  3. HF generate(use_cache=True) works (the per-frame decode primitive used by the default
     HFWorldModelRollout path).
  4. The manual recurrent interact loop (prefill -> per-token decode threading KDA's
     _kda_state, interleaving 7 action tokens between 64-token frames) runs and yields a
     response of the expected length -> reshapeable to [segment_length-1, tokens_per_frame+action_dim].

Run:  /home/venv/libero/bin/python sanity_check_kda_swap.py
"""
import os
import sys

import torch
from transformers import LlamaConfig, LlamaForCausalLM

# make the integration hook + kda_llama importable
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, "/home/wm-vla-rl/module_videowm")
from wm_kda_surgery import apply_wm_kda_surgery, _resolve_kda_layers  # noqa: E402


class Cfg:
    """Minimal stand-in for the world_model DictConfig (only .get is used by the hook)."""
    def __init__(self, **kw):
        self._d = kw

    def get(self, k, default=None):
        return self._d.get(k, default)


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)

    # VLA-RFT LIBERO WM shape (shrunk depth for a fast smoke; vocab/heads/hidden faithful).
    VOCAB = 9008
    cfg = LlamaConfig(
        vocab_size=VOCAB, hidden_size=1024, intermediate_size=4096,
        num_hidden_layers=8, num_attention_heads=16, num_key_value_heads=16,
        max_position_embeddings=8192, tie_word_embeddings=False,
    )
    model = LlamaForCausalLM(cfg).to(dev).to(torch.bfloat16)

    # sanity on layer-selection logic: Kimi 3:1 over 8 layers -> keep softmax at 3,7
    sel = _resolve_kda_layers(3, 8)
    assert sel == {0, 1, 2, 4, 5, 6}, sel
    print("[1] kda_layers=3 (Kimi 3:1) over 8 layers -> KDA layers", sorted(sel), "OK")

    # === the actual integration hook ===
    wm_cfg = Cfg(attn_type="kda", kda_layers=3, init_from_softmax=True, attnres_groups=4)
    apply_wm_kda_surgery(model, wm_cfg, rank=0)

    # verify the right layers are now KDA and the others are still softmax
    from kda_llama import KDAttention
    base = model.model
    inner = lambda L: getattr(L, "inner", L)            # unwrap _LayerAttnRes if present
    kinds = [type(inner(L).self_attn).__name__ for L in base.layers]
    print("[2] per-layer attn after surgery:", kinds)
    assert sum(t == "KDAttention" for t in kinds) == 6, kinds

    B, P = 2, 40
    ids = torch.randint(0, VOCAB, (B, P), device=dev)

    # === forward ===
    with torch.no_grad():
        out = model(input_ids=ids)
    print("[3] forward logits:", tuple(out.logits.shape))
    assert out.logits.shape == (B, P, VOCAB)

    # === HF generate(use_cache=True): the per-frame decode primitive ===
    model.config.use_cache = True
    if model.generation_config is not None:
        model.generation_config.use_cache = True
    with torch.no_grad():
        gen = model.generate(input_ids=ids[:1], attention_mask=torch.ones_like(ids[:1]),
                             max_new_tokens=64, do_sample=True, top_p=0.8, temperature=1.0,
                             use_cache=True, pad_token_id=9007)
    print("[4] generate(use_cache=True) ->", tuple(gen.shape), "(prompt 40 + 64)")
    assert gen.shape[1] == P + 64

    # === manual recurrent interact loop (mirrors HFWorldModelRollout._rollout_recurrent) ===
    TPF, ADIM, N_STEPS = 64, 7, 7        # tokens_per_frame, action_dim, segment_length-1 - 1
    prompt = ids[0].tolist()
    action_ids = torch.randint(8750, 9006, (N_STEPS + 1, ADIM), device=dev)  # discretised-action band

    from transformers.cache_utils import DynamicCache
    kv = DynamicCache()                  # threads KV for the 2 residual softmax layers (hybrid)

    def step(tok_list, pos0):
        inp = torch.tensor([tok_list], device=dev, dtype=torch.long)
        cp = torch.arange(pos0, pos0 + inp.shape[1], device=dev)
        with torch.no_grad():
            o = model(input_ids=inp, use_cache=True, past_key_values=kv,
                      cache_position=cp, position_ids=cp.unsqueeze(0))
        return o.logits[:, -1, :]

    logits = step(prompt, 0)             # prefill resets KDA _kda_state
    pos, resp = len(prompt), []
    for t in range(N_STEPS):
        for _ in range(TPF):
            tok = int(torch.softmax(logits.float(), -1).multinomial(1))
            resp.append(tok)
            logits = step([tok], pos); pos += 1
        act = action_ids[t + 1].tolist()
        resp += act
        logits = step(act, pos); pos += len(act)        # advance state over action tokens
    expected = N_STEPS * (TPF + ADIM)
    print("[5] recurrent interact loop -> response len", len(resp), "expected", expected)
    assert len(resp) == expected
    r = torch.tensor(resp).reshape(N_STEPS, TPF + ADIM)
    print("    reshape -> [segment_steps, tpf+adim] =", tuple(r.shape), "OK")

    # confirm KDA module state was actually threaded (not None) after the rollout
    kda_layer = next(inner(L).self_attn for L in base.layers
                     if type(inner(L).self_attn).__name__ == "KDAttention")
    assert getattr(kda_layer, "_kda_state", None) is not None
    print("[6] KDA _kda_state is populated after recurrent decode -> O(1)/step path live")

    print("\nALL SANITY CHECKS PASSED")


if __name__ == "__main__":
    main()
