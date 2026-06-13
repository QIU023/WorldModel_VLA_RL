"""KDA / AttnRes surgery hook for the VLA-RFT world model.

This is the single function the patched ``WorldModelRolloutWorker._build_model``
calls right after ``AutoModelForCausalLM.from_pretrained(...)`` (fsdp_workers.py:984)
and *before* the model is wrapped in FSDP. It optionally linearises the WM's LLaMA
backbone with our KDA (Kimi Delta Attention) drop-in and adds cross-layer AttnRes,
gated entirely by config flags so the stock softmax path is untouched when disabled.

Why here (ordering): the surgery swaps ``nn.Module`` layers in place, so it MUST run
on the raw module before FSDP shards/flattens parameters. It is therefore inserted
inside the ``with init_context()`` block, after ``world_module.to(torch_dtype)`` and
before ``gradient_checkpointing_enable`` / the FSDP wrap.

Meta-tensor / multi-rank note (verl uses ``use_meta_tensor=True`` for the WM because
``tie_word_embeddings=False``): on rank!=0 the module lives on the meta device and
real weights arrive later via FSDP ``sync_module_states`` (broadcast from rank 0).
The surgery is purely structural on meta ranks (it replaces modules and creates new
KDA params, which broadcast from rank 0 like everything else). ``init_from_softmax``
only meaningfully copies q/k/v/o on rank 0 (the only rank holding real weights); the
copied + freshly-initialised KDA params are then broadcast to all ranks. So running
this on every rank is correct. If you observe init hangs with meta tensors, set
``world_model.fsdp_config`` so the WM is materialised on all ranks before surgery,
or run with ``tie_word_embeddings`` left at its default (already False here).

Config flags (added under ``world_model_rollout.world_model`` in the trainer yaml):
    attn_type:       'softmax' (default, no-op) | 'kda'
    kda_layers:      null  -> all layers become KDA (pure-KDA)
                     int N -> Kimi N:1 hybrid, i.e. every layer except 1-in-(N+1)
                              stays softmax (we keep the LAST of each (N+1) block as
                              softmax, matching the iVideoGPT 3:1 recipe);
                     list  -> explicit set of layer indices to convert.
    init_from_softmax: bool, default True  -> attention-surgery weight copy.
    attnres_groups:  int, default 0 (off). >0 -> cross-layer Block-AttnRes with that
                     many groups (applied AFTER kdaify; orthogonal time vs depth axis).

The actual primitives live in /home/wm-vla-rl/module_videowm/kda_llama.py
(``kdaify_llama`` / ``attnres_ify_llama``). We add that dir to ``sys.path`` so the
hook works regardless of where verl is installed. Set ``KDA_LLAMA_DIR`` to override.
"""
import os
import sys


def _import_kda():
    """Locate and import kda_llama (kdaify_llama / attnres_ify_llama)."""
    candidates = [
        os.environ.get("KDA_LLAMA_DIR"),
        # the canonical location of the surgery module in this repo
        "/home/wm-vla-rl/module_videowm",
        # co-located copy shipped in this integration dir (fallback)
        os.path.dirname(os.path.abspath(__file__)),
    ]
    for d in candidates:
        if d and os.path.isfile(os.path.join(d, "kda_llama.py")):
            if d not in sys.path:
                sys.path.insert(0, d)
            break
    import kda_llama  # noqa: E402
    return kda_llama


def _resolve_kda_layers(kda_layers, n_layers):
    """Turn the ``kda_layers`` flag into an explicit set of indices (or None=all)."""
    if kda_layers is None:
        return None  # pure-KDA: convert every layer
    # explicit list / ListConfig of indices
    if isinstance(kda_layers, (list, tuple)) or (
        hasattr(kda_layers, "__iter__") and not isinstance(kda_layers, (int, str))
    ):
        return set(int(i) for i in kda_layers)
    # integer N -> Kimi N:1 hybrid. Keep 1 softmax layer per (N+1) block; convert the
    # other N. We keep the LAST layer of each block as softmax (it sees the deepest
    # features), matching iVideoGPT's train_gpt.py 3:1 recipe.
    n = int(kda_layers)
    if n <= 0:
        return None
    period = n + 1
    keep_softmax = {i for i in range(n_layers) if (i + 1) % period == 0}
    return set(range(n_layers)) - keep_softmax


def apply_wm_kda_surgery(world_module, wm_cfg, rank=0):
    """Apply KDA + AttnRes surgery to the WM backbone in place, gated by ``wm_cfg``.

    Args:
        world_module: the raw (un-wrapped) HF ``LlamaForCausalLM`` returned by
                      ``from_pretrained`` (NOT the FSDP wrapper).
        wm_cfg:       ``self.config.world_model`` DictConfig (has .get()).
        rank:         process rank, only used for logging.

    Returns:
        world_module (mutated in place; returned for convenience).
    """
    attn_type = str(wm_cfg.get("attn_type", "softmax")).lower()
    if attn_type in ("softmax", "", "none", "default"):
        if rank == 0:
            print("[wm_kda_surgery] attn_type=softmax -> no surgery (stock WM).")
        return world_module

    if attn_type != "kda":
        raise ValueError(
            f"[wm_kda_surgery] unknown world_model.attn_type={attn_type!r} "
            f"(expected 'softmax' or 'kda')"
        )

    kda = _import_kda()

    base = world_module.model if hasattr(world_module, "model") else world_module
    n_layers = len(base.layers)
    kda_layers = wm_cfg.get("kda_layers", None)
    init_from_softmax = bool(wm_cfg.get("init_from_softmax", True))
    attnres_groups = int(wm_cfg.get("attnres_groups", 0))

    target = _resolve_kda_layers(kda_layers, n_layers)
    if rank == 0:
        which = "ALL" if target is None else sorted(target)
        print(
            f"[wm_kda_surgery] attn_type=kda | layers={which} | "
            f"init_from_softmax={init_from_softmax} | attnres_groups={attnres_groups}"
        )

    # 1) time-axis: replace softmax attention with KDA (linear, O(1)/step recurrent decode)
    kda.kdaify_llama(world_module, layers=target, init_from_softmax=init_from_softmax)

    # 2) depth-axis (optional, orthogonal): cross-layer Block-AttnRes aggregation
    if attnres_groups and attnres_groups > 0:
        kda.attnres_ify_llama(world_module, groups=attnres_groups)

    # NOTE: kdaify_llama sets config.use_cache=False (its default safety for naive HF
    # generate). The HF *world-model* rollout we ship (hf_world_model_rollout.py)
    # passes use_cache=True explicitly and threads KDA's module-level _kda_state, so it
    # gets O(1)/step recurrent decode regardless of this flag. compute_log_prob() calls
    # forward with use_cache=False explicitly, so it is also unaffected.
    return world_module
