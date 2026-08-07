"""Reusable test assertions for the five GPU-profiled notebooks."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .accounting import formula_gate
from .cache import greedy_decode_step, prefill
from .policies import CacheState, FullCachePolicy
from .runtime import read_json


def assert_formula_gate(model_config_or_gate, cache=None, dtype=None, tolerance: float = 0.02):
    """Assert a formula-gate result in current or legacy notebook form.

    Prior notebook 00 cells supplied a precomputed gate dictionary. Current
    callers may instead provide ``model_config``, ``cache``, and ``dtype``.
    """
    if isinstance(model_config_or_gate, dict) and cache is None and dtype is None:
        result = model_config_or_gate
    else:
        if cache is None or dtype is None:
            raise TypeError("assert_formula_gate() requires cache and dtype with model_config")
        result = formula_gate(model_config_or_gate, cache, dtype=dtype, tolerance=tolerance)
    assert result["passed"], f"KV formula mismatch: {result}"
    return result


def assert_policy_respects_budget(cache, policy, state: CacheState, budget: int):
    selection = policy.select(state, budget)
    assert selection.numel() <= budget or policy.name == "full", (policy.name, selection.numel(), budget)
    assert selection.numel() >= 1, f"{policy.name} returned an empty retention set"
    return int(selection.numel())


def assert_full_cache_equivalence(model, input_ids, decode_token_id, atol: float = 1e-4):
    """Full policy must leave cache state/output equivalent to the non-evicted path."""
    import torch
    cache_a, initial_a = prefill(model, input_ids, output_attentions=False)
    cache_b, initial_b = prefill(model, input_ids, output_attentions=False)
    state = CacheState(positions=cache_b.physical_positions)
    cache_b.apply_policy(FullCachePolicy(), state, budget=cache_b.physical_length, reason="full_equivalence_test")
    out_a, token_a, logits_a = greedy_decode_step(model, cache_a, decode_token_id, logical_position=cache_a.logical_length)
    out_b, token_b, logits_b = greedy_decode_step(model, cache_b, decode_token_id, logical_position=cache_b.logical_length)
    assert torch.allclose(initial_a.logits, initial_b.logits, atol=atol, rtol=0), "Prefill outputs differ before any eviction"
    assert torch.equal(token_a, token_b), "Full cache policy changed greedy next token"
    assert torch.allclose(logits_a.logits, logits_b.logits, atol=atol, rtol=1e-4), "Full cache policy changed decode logits"
    return {"passed": True, "next_token": int(token_a.item()), "retained": cache_b.physical_length}


def assert_checkpoint_reload(model, normaliser, checkpoint_path, loader):
    loaded_model, loaded_normaliser, _ = loader(checkpoint_path)
    import torch
    x = torch.randn(3, model.config.input_dim, device=next(model.parameters()).device)
    model.eval(); loaded_model.eval()
    with torch.no_grad():
        expected = model(normaliser(x))
        observed = loaded_model(loaded_normaliser(x.to(next(loaded_model.parameters()).device))).to(expected.device)
    assert torch.allclose(expected, observed, atol=1e-5, rtol=1e-4), "Reloaded scorer predictions differ"
    return {"passed": True}


def assert_frontier_coverage(records: list[dict], expected_policies: list[str], required_contexts: list[int] | None = None):
    observed = {row.get("policy") for row in records if row.get("status") == "observed"}
    missing = set(expected_policies) - observed
    assert not missing, f"Missing observed policy rows: {sorted(missing)}"
    if required_contexts:
        contexts = {int(row.get("prompt_tokens", 0)) for row in records if row.get("status") == "observed"}
        for value in required_contexts:
            assert any(context >= value for context in contexts), f"No frontier observation at requested context {value}"
    return {"observed_policies": sorted(observed), "count": len(records)}


def assert_equal_memory_granularity(records: list[dict], tolerance_ratio: float = 0.01):
    groups: dict[tuple[Any, ...], list[int]] = {}
    for row in records:
        if row.get("status") != "observed":
            continue
        key = (row.get("benchmark"), row.get("task"), row.get("row_id"), row.get("budget"))
        groups.setdefault(key, []).append(int(row.get("physical_kv_bytes_post_evict", 0)))
    offending = {key: values for key, values in groups.items() if len(values) > 1 and max(values) - min(values) > max(values) * tolerance_ratio}
    assert not offending, f"Granularity variants were not measured at equal memory: {offending}"
    return {"groups_checked": len(groups), "passed": True}


def assert_fp8_byte_ratio(record: dict, tolerance: float = 0.05):
    ratio = float(record["analytic_ratio"] if "analytic_ratio" in record else record["fp8_payload_bytes"] / record["fp16_payload_bytes"])
    assert abs(ratio - 0.5) <= tolerance, f"FP8 payload ratio {ratio:.3f} is not approximately 0.5"
    return {"ratio": ratio, "passed": True}


def assert_nonfabricated(records: list[dict]):
    allowed = {"observed", "requires_larger_gpu_or_smaller_context", "error"}
    unexpected = [row for row in records if row.get("status") not in allowed]
    assert not unexpected, f"Unknown/unverifiable record statuses: {unexpected[:3]}"
    return {"checked": len(records), "passed": True}


def assert_artifact(path: str | Path):
    path = Path(path)
    assert path.exists() and path.stat().st_size > 0, f"Missing or empty artifact: {path}"
    return {"artifact": str(path), "bytes": path.stat().st_size}
