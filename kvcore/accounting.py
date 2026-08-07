"""Exact, GQA-aware KV-cache payload accounting."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


def bytes_per_dtype(dtype) -> int:
    text = str(dtype).replace("torch.", "").lower()
    mapping = {
        "float16": 2, "half": 2, "bfloat16": 2, "float32": 4, "float": 4,
        "float8_e4m3fn": 1, "float8_e5m2": 1, "int8": 1,
    }
    if text not in mapping:
        raise ValueError(f"Unsupported KV dtype {dtype!r} for accounting")
    return mapping[text]


def kv_formula_bytes(num_layers: int, num_kv_heads: int, head_dim: int, sequence_length: int,
                     batch_size: int, bytes_per_element: int) -> int:
    """`2 × L × H_kv × d_head × S × B × bytes` (GQA-aware)."""
    return int(2 * num_layers * num_kv_heads * head_dim * sequence_length * batch_size * bytes_per_element)


def model_kv_spec(model_config, dtype) -> dict[str, int]:
    layers = int(getattr(model_config, "num_hidden_layers"))
    attention_heads = int(getattr(model_config, "num_attention_heads"))
    kv_heads = int(getattr(model_config, "num_key_value_heads", attention_heads))
    hidden = int(getattr(model_config, "hidden_size"))
    head_dim = int(getattr(model_config, "head_dim", hidden // attention_heads))
    return {
        "num_layers": layers,
        "num_attention_heads": attention_heads,
        "num_key_value_heads": kv_heads,
        "head_dim": head_dim,
        "bytes_per_element": bytes_per_dtype(dtype),
    }


def formula_for_model(model_config, dtype, sequence_length: int | None = None, batch_size: int = 1,
                      *, retained_tokens: int | None = None) -> dict[str, Any]:
    """Return GQA-aware KV-cache accounting with legacy keyword compatibility.

    ``retained_tokens`` is retained as an alias for prior notebook builds that
    used that name.  Both names may be supplied only when they agree.
    """
    if sequence_length is None:
        if retained_tokens is None:
            raise TypeError("formula_for_model() requires sequence_length or retained_tokens")
        sequence_length = retained_tokens
    elif retained_tokens is not None and int(sequence_length) != int(retained_tokens):
        raise ValueError("sequence_length and retained_tokens disagree")
    spec = model_kv_spec(model_config, dtype)
    result = dict(spec)
    result.update({
        "sequence_length": int(sequence_length), "batch_size": int(batch_size),
        "formula": "2 * L * H_kv * d_head * S * B * bytes",
    })
    result["bytes"] = kv_formula_bytes(
        spec["num_layers"], spec["num_key_value_heads"], spec["head_dim"], sequence_length, batch_size, spec["bytes_per_element"],
    )
    # Stable legacy alias used by prior notebook 00 builds.
    result["formula_bytes"] = result["bytes"]
    return result


def measured_cache_bytes(cache) -> int:
    if hasattr(cache, "payload_bytes"):
        return int(cache.payload_bytes())
    total = 0
    if hasattr(cache, "key_cache"):
        for key, value in zip(cache.key_cache, cache.value_cache):
            if key is not None:
                total += key.numel() * key.element_size() + value.numel() * value.element_size()
    return int(total)


def formula_gate(model_config_or_cache, cache_or_expected=None, dtype=None, batch_size: int = 1,
                 tolerance: float = 0.02) -> dict[str, Any]:
    """Compare measured cache payload with an expected GQA-aware payload.

    Supports both the current ``(model_config, cache, dtype=...)`` form and the
    legacy notebook form ``(cache, expected_formula_bytes)``.
    """
    if hasattr(model_config_or_cache, "physical_length") and isinstance(cache_or_expected, (int, float)):
        cache = model_config_or_cache
        expected_bytes = int(cache_or_expected)
        expected = {"formula_bytes": expected_bytes, "bytes": expected_bytes, "source": "caller_supplied_expected_bytes"}
    else:
        model_config = model_config_or_cache
        cache = cache_or_expected
        if cache is None:
            raise TypeError("formula_gate() requires a cache object")
        if dtype is None:
            raise TypeError("formula_gate() requires dtype when called with model_config")
        sequence_length = int(getattr(cache, "physical_length", cache.get_seq_length()))
        expected = formula_for_model(model_config, dtype, sequence_length, batch_size)
        expected_bytes = int(expected["bytes"])
    actual = measured_cache_bytes(cache)
    relative_error = abs(actual - expected_bytes) / max(expected_bytes, 1)
    return {
        "formula_bytes": expected_bytes, "measured_payload_bytes": actual,
        "relative_error": relative_error, "tolerance": tolerance,
        "passed": bool(relative_error <= tolerance), "details": expected,
    }
