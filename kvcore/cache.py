"""Physical KV-cache compaction for reproducible eviction experiments.

`EvictingCache` subclasses Transformers `DynamicCache` and deliberately keeps
logical token positions separate from compact physical cache indices.  It calls
`super().update()` exactly once per model cache update; eviction happens only
in the explicit `compact()`/`apply_policy()` phase after all decoder layers
have been updated, avoiding recursive cache-update behavior.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .runtime import cuda_memory_snapshot, write_json


def _torch():
    import torch
    return torch


def _dynamic_cache_cls():
    try:
        from transformers import DynamicCache
    except ImportError:
        from transformers.cache_utils import DynamicCache
    return DynamicCache


DynamicCache = _dynamic_cache_cls()


@dataclass
class CacheSnapshot:
    label: str
    logical_length: int
    physical_length: int
    retained_positions: list[int]
    payload_bytes: int
    cuda: dict[str, Any]
    timestamp_unix: float


class EvictingCache(DynamicCache):
    """A `DynamicCache` with explicit physical compaction and position maps.

    The logical index map tracks original sequence positions.  The physical
    cache remains sorted in logical order, so RoPE-rotated K/V vectors retain
    their original absolute positions while the next decode token receives its
    own original logical `cache_position`.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        torch = _torch()
        self.physical_positions = torch.empty(0, dtype=torch.long)
        self.logical_length = 0
        self._update_depth = 0
        self._update_calls = 0
        self._last_compaction: dict[str, Any] | None = None

    @property
    def physical_length(self) -> int:
        return int(self.physical_positions.numel())

    def _get_pairs(self):
        """Return `[(K,V), ...]` across supported DynamicCache layouts."""
        if hasattr(self, "key_cache"):
            return [(k, v) for k, v in zip(self.key_cache, self.value_cache) if k is not None and v is not None]
        if hasattr(self, "layers"):
            return [(layer.keys, layer.values) for layer in self.layers if getattr(layer, "keys", None) is not None]
        return []

    def _set_layer_pair(self, index: int, key, value) -> None:
        if hasattr(self, "key_cache"):
            self.key_cache[index] = key
            self.value_cache[index] = value
            return
        if hasattr(self, "layers"):
            self.layers[index].keys = key
            self.layers[index].values = value
            return
        raise RuntimeError("Unsupported DynamicCache tensor layout")

    def _positions_from_kwargs(self, key_states, cache_kwargs):
        torch = _torch()
        new_tokens = int(key_states.shape[-2])
        device = key_states.device
        supplied = (cache_kwargs or {}).get("cache_position")
        if supplied is not None:
            supplied = supplied.to(device=device, dtype=torch.long).flatten()
            if supplied.numel() >= new_tokens:
                return supplied[-new_tokens:]
        start = self.logical_length
        return torch.arange(start, start + new_tokens, device=device, dtype=torch.long)

    def update(self, key_states, value_states, layer_idx: int, cache_kwargs: dict | None = None):
        """Delegate tensor growth to DynamicCache once; never call `self.update` recursively."""
        if self._update_depth:
            raise RuntimeError("EvictingCache.update recursion detected")
        self._update_depth += 1
        try:
            updated = super().update(key_states, value_states, layer_idx, cache_kwargs)
            self._update_calls += 1
            # DynamicCache invokes update once for each decoder layer. Only the
            # first layer advances the shared sequence-position map.
            if layer_idx == 0:
                positions = self._positions_from_kwargs(key_states, cache_kwargs)
                if self.physical_positions.numel() == 0:
                    self.physical_positions = positions.detach().clone()
                else:
                    self.physical_positions = _torch().cat([self.physical_positions.to(positions.device), positions.detach().clone()])
                self.logical_length = max(self.logical_length, int(positions.max().item()) + 1 if positions.numel() else self.logical_length)
            return updated
        finally:
            self._update_depth -= 1

    def payload_bytes(self) -> int:
        return int(sum(k.numel() * k.element_size() + v.numel() * v.element_size() for k, v in self._get_pairs()))

    def cache_shape_report(self) -> list[dict[str, Any]]:
        output = []
        for layer, (key, value) in enumerate(self._get_pairs()):
            output.append({
                "layer": layer,
                "key_shape": list(key.shape),
                "value_shape": list(value.shape),
                "dtype": str(key.dtype).replace("torch.", ""),
                "bytes": int(key.numel() * key.element_size() + value.numel() * value.element_size()),
            })
        return output

    def snapshot(self, label: str, run_root: str | Path | None = None) -> dict[str, Any]:
        payload = CacheSnapshot(
            label=label,
            logical_length=int(self.logical_length),
            physical_length=self.physical_length,
            retained_positions=[int(x) for x in self.physical_positions.detach().cpu().tolist()],
            payload_bytes=self.payload_bytes(),
            cuda=cuda_memory_snapshot(label),
            timestamp_unix=time.time(),
        ).__dict__
        payload["layer_shapes"] = self.cache_shape_report()
        payload["measurement_class"] = "physical_kv_tensor_payload_and_cuda_allocator_snapshot"
        if run_root is not None:
            safe_label = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in label)
            write_json(Path(run_root) / "cache_snapshots" / f"{safe_label}.json", payload)
        return payload

    def _validate_keep_indices(self, keep_indices, budget: int | None = None):
        torch = _torch()
        keep = keep_indices.to(self.physical_positions.device, dtype=torch.long).flatten()
        if keep.numel() == 0:
            raise ValueError("Eviction would remove all KV tokens")
        if keep.min().item() < 0 or keep.max().item() >= self.physical_length:
            raise ValueError("Keep indices are outside physical cache bounds")
        if not torch.equal(keep, torch.unique(keep, sorted=True)):
            raise ValueError("Keep indices must be sorted and unique")
        if budget is not None and keep.numel() > budget:
            raise ValueError("Keep set exceeds requested cache budget")
        return keep

    def compact(self, keep_indices, budget: int | None = None, reason: str = "policy") -> dict[str, Any]:
        """Physically index-select all K/V layers and update the position map."""
        torch = _torch()
        keep = self._validate_keep_indices(keep_indices, budget)
        before = self.physical_length
        if keep.numel() == before:
            self._last_compaction = {"reason": reason, "before": before, "after": before, "evicted": 0, "no_op": True}
            return dict(self._last_compaction)
        # Iterate actual cache-layer indexes rather than filtered pairs so a
        # sparse DynamicCache layout still receives its tensor replacement.
        if hasattr(self, "key_cache"):
            layer_count = len(self.key_cache)
            pairs = [(index, self.key_cache[index], self.value_cache[index]) for index in range(layer_count)]
        elif hasattr(self, "layers"):
            pairs = [(index, layer.keys, layer.values) for index, layer in enumerate(self.layers)]
        else:
            raise RuntimeError("Unsupported DynamicCache layout for compaction")
        for index, key, value in pairs:
            if key is None or value is None:
                continue
            local = keep.to(key.device)
            self._set_layer_pair(index, key.index_select(-2, local).contiguous(), value.index_select(-2, local).contiguous())
        self.physical_positions = self.physical_positions.index_select(0, keep.to(self.physical_positions.device)).contiguous()
        after = self.physical_length
        self._last_compaction = {
            "reason": reason, "before": before, "after": after, "evicted": before - after,
            "retained_logical_positions": [int(x) for x in self.physical_positions.detach().cpu().tolist()], "no_op": False,
        }
        return dict(self._last_compaction)

    def apply_policy(self, policy, state, budget: int, reason: str = "policy") -> dict[str, Any]:
        """Map a policy's logical-position selection to physical cache rows."""
        torch = _torch()
        if self.physical_length == 0:
            return {"reason": reason, "before": 0, "after": 0, "evicted": 0, "no_op": True}
        if getattr(policy, "name", "") == "full":
            return self.compact(torch.arange(self.physical_length, device=self.physical_positions.device), reason="full_baseline")
        selected_logical = policy.select(state, budget)
        selected_logical = torch.unique(selected_logical.to(self.physical_positions.device), sorted=True)
        lookup = {int(position): index for index, position in enumerate(self.physical_positions.detach().cpu().tolist())}
        physical = [lookup[int(position)] for position in selected_logical.detach().cpu().tolist() if int(position) in lookup]
        if not physical:
            # Fail closed by retaining the newest physical token rather than
            # silently producing an unusable empty cache.
            physical = [self.physical_length - 1]
        keep = torch.tensor(sorted(set(physical)), device=self.physical_positions.device, dtype=torch.long)
        return self.compact(keep, budget=budget, reason=reason)


def make_attention_mask(cache: EvictingCache, current_tokens: int, device=None):
    """Create a physical-cache-length mask for manual cached decode calls."""
    torch = _torch()
    device = device or (cache.physical_positions.device if cache.physical_length else "cpu")
    return torch.ones((1, cache.physical_length + int(current_tokens)), dtype=torch.long, device=device)


def prefill(model, input_ids, chunk_size: int | None = None, output_attentions: bool = False):
    """Prefill a cache in chunks while retaining original logical positions."""
    torch = _torch()
    if input_ids.ndim == 1:
        input_ids = input_ids.unsqueeze(0)
    cache = EvictingCache()
    total = int(input_ids.shape[-1])
    chunk_size = chunk_size or total
    outputs = None
    for start in range(0, total, chunk_size):
        end = min(start + chunk_size, total)
        piece = input_ids[:, start:end]
        position = torch.arange(start, end, device=piece.device, dtype=torch.long)
        attention_mask = make_attention_mask(cache, piece.shape[-1], device=piece.device)
        # This suite performs inference-only cache measurements. Avoid retaining
        # an autograd graph for every chunk, which otherwise accumulates across
        # repeated oracle-label profiles on large GPUs.
        with torch.no_grad():
            outputs = model(
                input_ids=piece, attention_mask=attention_mask, past_key_values=cache, use_cache=True,
                cache_position=position, position_ids=position.unsqueeze(0), output_attentions=output_attentions,
            )
        cache = outputs.past_key_values
        if not isinstance(cache, EvictingCache):
            raise RuntimeError("Model replaced EvictingCache with an unsupported cache type; use an eager-compatible model/runtime.")
    return cache, outputs


def greedy_decode_step(model, cache: EvictingCache, token_id, logical_position: int, output_attentions: bool = False):
    """Decode one token using physical cache length and original logical position."""
    torch = _torch()
    token = token_id if getattr(token_id, "ndim", 0) == 2 else torch.tensor([[int(token_id)]], device=cache.physical_positions.device)
    position = torch.tensor([logical_position], device=token.device, dtype=torch.long)
    mask = make_attention_mask(cache, token.shape[-1], device=token.device)
    with torch.no_grad():
        out = model(
            input_ids=token, attention_mask=mask, past_key_values=cache, use_cache=True,
            cache_position=position, position_ids=position.unsqueeze(0), output_attentions=output_attentions,
        )
    next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    return out.past_key_values, next_token, out
