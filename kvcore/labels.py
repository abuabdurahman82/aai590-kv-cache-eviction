"""Self-generated importance labels for learned KV eviction.

Labels do not require external annotation. They combine model-observed attention
statistics with a controlled drop-and-measure change in gold-answer negative
log likelihood. This expensive operation is intentionally capped to the
short-context eager profiling partition.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .cache import EvictingCache, make_attention_mask, prefill
from .features import aggregate_blocks, token_features
from .profiling import load_profile
from .runtime import write_json


def _clone_cache(cache: EvictingCache) -> EvictingCache:
    """Clone K/V tensors while preserving the source DynamicCache layer layout.

    In recent Transformers releases, an empty ``DynamicCache()`` lazily creates
    layers as model blocks update it. Writing directly to ``clone.layers[i]``
    before those layers exist fails for a multi-layer Qwen cache. Rebuilding from
    DDP-style key/value pairs creates the full layer container before ablation.
    """
    if hasattr(cache, "key_cache"):
        clone = EvictingCache()
        # Preserve the legacy sparse list layout, including empty layers.
        clone.key_cache = []
        clone.value_cache = []
        for key, value in zip(cache.key_cache, cache.value_cache):
            clone.key_cache.append(None if key is None else key.detach().clone())
            clone.value_cache.append(None if value is None else value.detach().clone())
    elif hasattr(cache, "layers"):
        pairs = []
        for index, layer in enumerate(cache.layers):
            key, value = getattr(layer, "keys", None), getattr(layer, "values", None)
            if key is None or value is None:
                raise RuntimeError(f"Cannot clone sparse modern cache: layer {index} has no key/value tensors")
            pairs.append((key.detach().clone(), value.detach().clone()))
        clone = EvictingCache(ddp_cache_data=pairs)
    else:
        raise RuntimeError("Unsupported DynamicCache layout for cloning")
    clone.logical_length = int(cache.logical_length)
    clone.physical_positions = cache.physical_positions.detach().clone()
    return clone


def gold_continuation_nll(model, cache: EvictingCache, prefill_logits, answer_ids) -> float:
    """Teacher-force the gold answer after a compacted prefix cache.

    ``prefill_logits`` may be the final-token vector ``[batch, vocab]`` so
    label ablation does not retain a full ``[batch, context, vocab]`` tensor.
    """
    if answer_ids.ndim == 1:
        answer_ids = answer_ids.unsqueeze(0)
    if answer_ids.numel() == 0:
        return 0.0
    running = _clone_cache(cache)
    logits = prefill_logits if prefill_logits.ndim == 2 else prefill_logits[:, -1, :]
    nll_sum, count = 0.0, 0
    for offset in range(answer_ids.shape[-1]):
        gold = answer_ids[:, offset]
        log_probs = torch.log_softmax(logits.float(), dim=-1)
        nll_sum += float((-log_probs.gather(1, gold.unsqueeze(1)).squeeze(1)).sum().item())
        count += int(gold.numel())
        token = gold.unsqueeze(1)
        logical_pos = running.logical_length
        mask = make_attention_mask(running, 1, device=token.device)
        pos = torch.tensor([logical_pos], device=token.device, dtype=torch.long)
        # Oracle labels are an inference measurement, not a differentiable
        # training step. Avoid retaining a growing decode graph per ablation.
        with torch.no_grad():
            out = model(
                input_ids=token, attention_mask=mask, past_key_values=running, use_cache=True,
                cache_position=pos, position_ids=pos.unsqueeze(0), output_attentions=False,
            )
        running = out.past_key_values
        logits = out.logits[:, -1, :]
    return nll_sum / max(count, 1)


def label_blocks(model, prefix_ids, answer_ids, profile_npz: str | Path, block_size: int,
                 max_counterfactual_blocks: int = 6, attention_weight: float = 0.5,
                 prefill_chunk_tokens: int = 512) -> list[dict[str, Any]]:
    """Build short-context labels and release all GPU-only tensors per document."""
    cache = prefill_out = profile = features = x = attention = None
    ablated = None
    try:
        cache, prefill_out = prefill(model, prefix_ids, chunk_size=prefill_chunk_tokens, output_attentions=False)
        profile = load_profile(profile_npz, device=prefix_ids.device)
        features = token_features(
            cache.physical_positions, profile.get("cumulative_attention"), profile.get("max_attention"),
            profile.get("attention_hits"), profile.get("key_norm"), profile.get("value_norm"), step=1,
        )
        x, spans = aggregate_blocks(features, block_size)
        # The label objective only needs the final prefix-token distribution.
        # Drop the much larger full-context logits tensor before ablations.
        initial_logits = prefill_out.logits[:, -1, :].detach()
        del prefill_out
        prefill_out = None
        base_nll = gold_continuation_nll(model, cache, initial_logits, answer_ids)
        # Candidate prioritization limits ablation work while preserving a
        # reproducible record of sampled candidate positions.
        attention = x[:, len(x[0]) // 2 - 1] if x.shape[0] else torch.empty(0, device=x.device)
        candidate_order = torch.argsort(attention, descending=True).tolist()[:max_counterfactual_blocks]
        labels = []
        for block_index in candidate_order:
            start, end = spans[block_index]
            keep = torch.tensor([index for index in range(cache.physical_length) if not (start <= index < end)], device=prefix_ids.device)
            if keep.numel() == 0:
                continue
            ablated = _clone_cache(cache)
            ablated.compact(keep, reason="oracle_single_block_ablation")
            ablated_nll = gold_continuation_nll(model, ablated, initial_logits, answer_ids)
            nll_impact = max(0.0, ablated_nll - base_nll)
            attn_score = float(attention[block_index].item()) if attention.numel() else 0.0
            labels.append({
                "block_index": int(block_index), "start": int(start), "end": int(end),
                "feature": x[block_index].detach().cpu(), "attention_score": attn_score,
                "baseline_answer_nll": float(base_nll), "ablated_answer_nll": float(ablated_nll),
                "nll_impact": float(nll_impact),
            })
            # Each ablation owns a cloned K/V cache. Release it before the next
            # counterfactual so GPU usage stays bounded by one clone.
            del ablated
            ablated = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        if labels:
            max_attention = max(row["attention_score"] for row in labels) or 1.0
            max_impact = max(row["nll_impact"] for row in labels) or 1.0
            for row in labels:
                row["target"] = float(attention_weight * row["attention_score"] / max_attention + (1 - attention_weight) * row["nll_impact"] / max_impact)
        return labels
    finally:
        # Returned features are explicitly CPU tensors. Remove all device-resident
        # cache, logits, profile, and intermediate tensors before the next row.
        if ablated is not None:
            del ablated
        for name in ("attention", "x", "features", "profile", "prefill_out", "initial_logits", "cache"):
            value = locals().get(name)
            if value is not None:
                del value
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def save_labels(label_rows: list[dict[str, Any]], path: str | Path, metadata: dict[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    serializable = []
    for row in label_rows:
        converted = dict(row)
        if isinstance(converted.get("feature"), torch.Tensor):
            converted["feature"] = converted["feature"].tolist()
        serializable.append(converted)
    torch.save({"labels": serializable, "metadata": metadata}, path)
    write_json(path.with_suffix(".json"), {"count": len(serializable), **metadata})
    return path
