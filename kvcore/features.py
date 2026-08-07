"""Feature extraction and train-only normalization for learned block eviction."""
from __future__ import annotations

from dataclasses import dataclass


def _torch():
    import torch
    return torch

FEATURE_VERSION = 1
TOKEN_FEATURE_NAMES = (
    "relative_position", "recency", "is_sink_candidate", "cumulative_attention",
    "max_attention", "attention_hits", "attention_density", "key_norm",
    "value_norm", "key_value_norm_product", "attention_x_recency", "position_x_attention",
    "log_position", "constant",
)
BLOCK_FEATURE_NAMES = tuple(f"mean_{name}" for name in TOKEN_FEATURE_NAMES) + tuple(f"max_{name}" for name in TOKEN_FEATURE_NAMES)


@dataclass
class FeatureNormaliser:
    mean: object | None = None
    std: object | None = None
    eps: float = 1e-6

    def fit(self, x):
        torch = _torch()
        if x.ndim != 2 or x.shape[0] == 0:
            raise ValueError("Feature normaliser requires a nonempty [N, D] training tensor")
        self.mean = x.mean(dim=0)
        self.std = x.std(dim=0, unbiased=False).clamp_min(self.eps)
        return self

    def transform(self, x):
        if self.mean is None or self.std is None:
            raise RuntimeError("Fit FeatureNormaliser on training features before transform")
        return (x - self.mean.to(x.device)) / self.std.to(x.device)

    def __call__(self, x):
        return self.transform(x)

    def state_dict(self):
        if self.mean is None or self.std is None:
            raise RuntimeError("Cannot save an unfitted FeatureNormaliser")
        return {"mean": self.mean.detach().cpu(), "std": self.std.detach().cpu(), "eps": self.eps, "feature_version": FEATURE_VERSION}

    @classmethod
    def from_state_dict(cls, state):
        return cls(mean=state["mean"], std=state["std"], eps=float(state.get("eps", 1e-6)))


def safe_norms(cache) -> tuple[object, object]:
    """Compute aggregate K/V norms by logical token without creating labels.

    This only observes cache tensors. Values are features, not ground-truth
    importance annotations.
    """
    torch = _torch()
    if cache is None or not getattr(cache, "key_cache", None):
        empty = torch.empty(0)
        return empty, empty
    keys, values = [], []
    for key, value in zip(cache.key_cache, cache.value_cache):
        if key is None or value is None:
            continue
        keys.append(key.detach().float().pow(2).mean(dim=(0, 1, 3)).sqrt())
        values.append(value.detach().float().pow(2).mean(dim=(0, 1, 3)).sqrt())
    if not keys:
        return torch.empty(0), torch.empty(0)
    return torch.stack(keys).mean(dim=0), torch.stack(values).mean(dim=0)


def token_features(positions, cumulative_attention=None, max_attention=None, attention_hits=None,
                   key_norm=None, value_norm=None, step: int | None = None, sink_tokens: int = 4):
    """Return the 14-dimensional per-token feature matrix for one cache state."""
    torch = _torch()
    if positions.ndim != 1:
        raise ValueError("positions must be one dimensional")
    n = int(positions.numel())
    device = positions.device
    if n == 0:
        return torch.empty((0, len(TOKEN_FEATURE_NAMES)), device=device, dtype=torch.float32)
    last = float(max(n - 1, 1))
    rel = positions.float() / last
    recency = 1.0 - rel
    sink = (positions < sink_tokens).float()

    def vector(value, default=0.0):
        if value is None or value.numel() == 0:
            return torch.full((n,), float(default), device=device, dtype=torch.float32)
        value = value.to(device=device, dtype=torch.float32).flatten()
        if value.numel() < n:
            pad = torch.full((n - value.numel(),), float(default), device=device)
            value = torch.cat([value, pad])
        return value[:n]

    attn_sum = vector(cumulative_attention)
    attn_max = vector(max_attention)
    hits = vector(attention_hits)
    denom = float(max(step or 1, 1))
    density = hits / denom
    k = vector(key_norm)
    v = vector(value_norm)
    product = k * v
    log_position = torch.log1p(positions.float()) / torch.log1p(torch.tensor(last, device=device))
    feats = torch.stack((
        rel, recency, sink, attn_sum, attn_max, hits, density, k, v, product,
        attn_sum * recency, rel * attn_sum, log_position, torch.ones(n, device=device),
    ), dim=-1)
    return feats.nan_to_num(0.0, 0.0, 0.0)


def aggregate_blocks(token_feature_matrix, block_size: int = 16):
    """Aggregate each contiguous candidate block to mean+max (28 dimensions)."""
    torch = _torch()
    if token_feature_matrix.ndim != 2 or token_feature_matrix.shape[1] != len(TOKEN_FEATURE_NAMES):
        raise ValueError("Expected [tokens, 14] token feature matrix")
    n = token_feature_matrix.shape[0]
    rows, spans = [], []
    for start in range(0, n, block_size):
        end = min(start + block_size, n)
        block = token_feature_matrix[start:end]
        rows.append(torch.cat([block.mean(dim=0), block.max(dim=0).values]))
        spans.append((start, end))
    if not rows:
        return torch.empty((0, len(BLOCK_FEATURE_NAMES)), device=token_feature_matrix.device), []
    return torch.stack(rows), spans
