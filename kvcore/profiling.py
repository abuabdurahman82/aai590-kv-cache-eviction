"""Two-pass attention profiling utilities.

Profiling runs at short context with eager attention and `output_attentions=True`.
Long-context evaluation runs with attention outputs disabled.  The generated
profile is therefore an artifact consumed by evaluation, not a hidden
quadratic-cost operation in the long-context benchmark path.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .cache import prefill
from .features import safe_norms
from .runtime import write_json


def _torch():
    import torch
    return torch


def force_eager_attention(model) -> dict[str, Any]:
    """Best-effort eager-attention request with transparent reporting."""
    prior = getattr(model.config, "_attn_implementation", None)
    try:
        model.config._attn_implementation = "eager"
        applied = True
    except Exception:
        applied = False
    return {"requested": "eager", "prior": prior, "applied": applied}


def disable_attention_outputs(model) -> dict[str, Any]:
    prior = getattr(model.config, "_attn_implementation", None)
    # Do not hard-code FlashAttention; SDPA is generally available in Colab.
    try:
        model.config._attn_implementation = "sdpa"
        applied = True
    except Exception:
        applied = False
    return {"requested": "sdpa", "prior": prior, "applied": applied}


def _aggregate_attention(attentions, key_length: int):
    torch = _torch()
    sums = torch.zeros(key_length)
    maxima = torch.zeros(key_length)
    hits = torch.zeros(key_length)
    observed = 0
    if attentions is None:
        return sums, maxima, hits, observed
    for layer_attention in attentions:
        if layer_attention is None:
            continue
        # [batch, heads, query, key], possibly keys shorter than requested in
        # chunked inference. Align to the current right edge.
        tensor = layer_attention.detach().float().cpu()
        per_key_sum = tensor.sum(dim=(0, 1, 2))
        per_key_max = tensor.amax(dim=(0, 1, 2))
        per_key_hits = (tensor > 0).sum(dim=(0, 1, 2)).float()
        offset = max(0, key_length - per_key_sum.numel())
        sums[offset:offset + per_key_sum.numel()] += per_key_sum
        maxima[offset:offset + per_key_max.numel()] = torch.maximum(maxima[offset:offset + per_key_max.numel()], per_key_max)
        hits[offset:offset + per_key_hits.numel()] += per_key_hits
        observed += 1
    return sums, maxima, hits, observed


def profile_short_context(model, input_ids, max_tokens: int, run_root: str | Path, tag: str,
                          prefill_chunk_tokens: int = 512) -> dict[str, Any]:
    """Collect a bounded eager-attention profile and release GPU tensors afterward.

    Attention tensors are only needed long enough to aggregate CPU-side summary
    statistics. Explicit cleanup prevents repeated oracle-label documents from
    accumulating cache, logits, and per-layer attention allocations.
    """
    torch = _torch()
    if input_ids.ndim == 1:
        input_ids = input_ids.unsqueeze(0)
    x = input_ids[:, :max_tokens]
    eager_info = force_eager_attention(model)
    cache = out = None
    try:
        cache, out = prefill(model, x, chunk_size=min(prefill_chunk_tokens, x.shape[-1]), output_attentions=True)
        length = cache.physical_length
        cumulative, maximum, hits, layer_count = _aggregate_attention(getattr(out, "attentions", None), length)
        key_norm, value_norm = safe_norms(cache)
        path = Path(run_root) / "attention_profiles" / f"{tag}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            positions=cache.physical_positions.detach().cpu().numpy(),
            cumulative_attention=cumulative.numpy(), max_attention=maximum.numpy(), attention_hits=hits.numpy(),
            key_norm=key_norm.detach().cpu().numpy(), value_norm=value_norm.detach().cpu().numpy(),
        )
        metadata = {
            "tag": tag, "context_tokens": int(x.shape[-1]), "observed_attention_layers": layer_count,
            "eager": eager_info, "profile_path": str(path), "cache_snapshot": cache.snapshot(f"profile_{tag}", run_root),
            "attention_available": bool(layer_count > 0),
            "rule": "eager short-context profiling only; no attention outputs in long-context evaluation",
        }
        write_json(Path(run_root) / "attention_profiles" / f"{tag}.json", metadata)
        return metadata
    finally:
        # Restore a non-eager backend before subsequent cache ablation/decoding.
        disable_attention_outputs(model)
        # Break output/cache references before returning to the next label document.
        if out is not None:
            attentions = getattr(out, "attentions", None)
            if attentions is not None:
                del attentions
            del out
        if cache is not None:
            del cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def load_profile(path: str | Path, device=None) -> dict[str, Any]:
    torch = _torch()
    data = np.load(path)
    target = device or "cpu"
    return {key: torch.from_numpy(data[key]).to(target) for key in data.files}


def long_context_attention_guard(model) -> dict[str, Any]:
    """Set a non-eager implementation and return an auditable guard record."""
    guard = disable_attention_outputs(model)
    guard["output_attentions"] = False
    guard["rule"] = "long_context_eval_must_not_request_attentions"
    return guard
