"""KV-cache eviction policy implementations with explicit fallbacks."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .features import aggregate_blocks, token_features


def _torch():
    import torch
    return torch


@dataclass
class CacheState:
    positions: object
    cumulative_attention: object | None = None
    max_attention: object | None = None
    attention_hits: object | None = None
    key_norm: object | None = None
    value_norm: object | None = None
    step: int = 1

    @property
    def length(self) -> int:
        return int(self.positions.numel())


class Policy:
    name = "policy"
    needs_attention = False

    def select(self, state: CacheState, budget: int):
        raise NotImplementedError

    @staticmethod
    def _validate(state: CacheState, budget: int):
        if budget < 1:
            raise ValueError("cache budget must be ≥ 1")
        if state.length == 0:
            return

    @staticmethod
    def _finish(indices, state: CacheState, budget: int):
        torch = _torch()
        if state.length == 0:
            return torch.empty(0, dtype=torch.long, device=state.positions.device)
        indices = torch.unique(indices.to(state.positions.device, dtype=torch.long), sorted=True)
        if indices.numel() > budget:
            indices = indices[-budget:]
        if indices.numel() == 0:
            indices = state.positions[-1:]
        return indices


class FullCachePolicy(Policy):
    name = "full"

    def select(self, state, budget):
        # Full cache deliberately ignores a budget; it is the uncompressed reference.
        return state.positions.clone()


class FIFOPolicy(Policy):
    name = "fifo"

    def __init__(self, min_recent: int = 32):
        self.min_recent = min_recent

    def select(self, state, budget):
        self._validate(state, budget)
        keep = max(1, min(state.length, max(budget, self.min_recent)))
        return self._finish(state.positions[-keep:], state, keep)


class RandomPolicy(Policy):
    name = "random"

    def __init__(self, seed: int = 590, min_recent: int = 32):
        self.seed, self.min_recent = seed, min_recent

    def select(self, state, budget):
        torch = _torch()
        self._validate(state, budget)
        keep = min(state.length, budget)
        recent = state.positions[-min(self.min_recent, keep):]
        room = max(0, keep - recent.numel())
        pool = state.positions[:max(0, state.length - recent.numel())]
        gen = torch.Generator(device=pool.device).manual_seed(self.seed + state.step)
        chosen = pool[torch.randperm(pool.numel(), generator=gen, device=pool.device)[:room]] if room and pool.numel() else pool[:0]
        return self._finish(torch.cat([chosen, recent]), state, keep)


class UniformPolicy(Policy):
    name = "uniform"

    def __init__(self, min_recent: int = 32):
        self.min_recent = min_recent

    def select(self, state, budget):
        torch = _torch()
        self._validate(state, budget)
        keep = min(state.length, budget)
        recent_n = min(self.min_recent, keep)
        recent = state.positions[-recent_n:]
        room = keep - recent_n
        prefix = state.positions[:state.length - recent_n]
        if room <= 0 or prefix.numel() == 0:
            return self._finish(recent, state, keep)
        sample = torch.linspace(0, max(prefix.numel() - 1, 0), room, device=prefix.device).round().long()
        return self._finish(torch.cat([prefix[sample], recent]), state, keep)


class SinkRecentPolicy(Policy):
    name = "sink_recent"

    def __init__(self, sink_tokens: int = 4, min_recent: int = 32):
        self.sink_tokens, self.min_recent = sink_tokens, min_recent

    def select(self, state, budget):
        torch = _torch()
        self._validate(state, budget)
        keep = min(state.length, budget)
        sink = state.positions[:min(self.sink_tokens, keep)]
        recent_room = max(0, keep - sink.numel())
        recent = state.positions[-min(self.min_recent, recent_room):]
        room = max(0, keep - torch.unique(torch.cat([sink, recent])).numel())
        mid_start, mid_end = sink.numel(), max(sink.numel(), state.length - recent.numel())
        mid = state.positions[mid_start:mid_end]
        if room and mid.numel():
            sampled = mid[torch.linspace(0, mid.numel() - 1, room, device=mid.device).round().long()]
        else:
            sampled = torch.empty(0, dtype=torch.long, device=state.positions.device)
        return self._finish(torch.cat([sink, sampled, recent]), state, keep)


class AttentionTopKPolicy(Policy):
    name = "attention_topk"
    needs_attention = True

    def __init__(self, sink_tokens: int = 4, min_recent: int = 32, fallback: Policy | None = None):
        self.sink_tokens, self.min_recent = sink_tokens, min_recent
        self.fallback = fallback or SinkRecentPolicy(sink_tokens=sink_tokens, min_recent=min_recent)

    def select(self, state, budget):
        torch = _torch()
        if state.cumulative_attention is None or state.cumulative_attention.numel() < state.length:
            return self.fallback.select(state, budget)
        keep = min(state.length, budget)
        sink = state.positions[:min(self.sink_tokens, keep)]
        recent = state.positions[-min(self.min_recent, max(0, keep - sink.numel())):]
        already = torch.unique(torch.cat([sink, recent]))
        room = max(0, keep - already.numel())
        if not room:
            return self._finish(already, state, keep)
        score = state.cumulative_attention.float().clone()
        score[:sink.numel()] = float("inf")
        score[-recent.numel():] = float("inf") if recent.numel() else score[-0:]
        top_indexes = torch.topk(score, k=min(room, score.numel())).indices
        return self._finish(torch.cat([already, state.positions.index_select(0, top_indexes)]), state, keep)


class H2OPolicy(AttentionTopKPolicy):
    name = "h2o"

    def __init__(self, recent_fraction: float = 0.50, sink_tokens: int = 4, min_recent: int = 32, fallback: Policy | None = None):
        super().__init__(sink_tokens=sink_tokens, min_recent=min_recent, fallback=fallback)
        self.recent_fraction = float(recent_fraction)

    def select(self, state, budget):
        torch = _torch()
        if state.cumulative_attention is None or state.cumulative_attention.numel() < state.length:
            return self.fallback.select(state, budget)
        keep = min(state.length, budget)
        sink = state.positions[:min(self.sink_tokens, keep)]
        recent_n = min(max(self.min_recent, int(round(keep * self.recent_fraction))), max(0, keep - sink.numel()))
        recent = state.positions[-recent_n:]
        protected = torch.unique(torch.cat([sink, recent]))
        room = max(0, keep - protected.numel())
        if room <= 0:
            return self._finish(protected, state, keep)
        score = state.cumulative_attention.float().clone()
        if protected.numel():
            # `protected` stores logical positions, while score is indexed by
            # physical cache row after previous compaction.
            protected_rows = torch.tensor([
                index for index, position in enumerate(state.positions.detach().cpu().tolist())
                if int(position) in set(int(x) for x in protected.detach().cpu().tolist())
            ], device=state.positions.device, dtype=torch.long)
            score[protected_rows] = float("-inf")
        top_indexes = torch.topk(score, k=min(room, int((score > float("-inf")).sum().item()))).indices
        return self._finish(torch.cat([protected, state.positions.index_select(0, top_indexes)]), state, keep)


class LearnedBlockPolicy(Policy):
    name = "learned_block"

    def __init__(self, scorer, normaliser=None, block_size: int = 16, sink_tokens: int = 4, min_recent: int = 32,
                 fallback: Policy | None = None):
        self.scorer, self.normaliser = scorer, normaliser
        self.block_size, self.sink_tokens, self.min_recent = block_size, sink_tokens, min_recent
        self.fallback = fallback or H2OPolicy(sink_tokens=sink_tokens, min_recent=min_recent)

    def select(self, state, budget):
        torch = _torch()
        if self.scorer is None or state.length < self.block_size:
            return self.fallback.select(state, budget)
        features = token_features(
            state.positions, state.cumulative_attention, state.max_attention, state.attention_hits,
            state.key_norm, state.value_norm, state.step, self.sink_tokens,
        )
        block_x, spans = aggregate_blocks(features, self.block_size)
        if block_x.numel() == 0:
            return self.fallback.select(state, budget)
        model_device = next(self.scorer.parameters()).device
        with torch.no_grad():
            x = block_x.to(model_device)
            if self.normaliser is not None:
                x = self.normaliser(x)
            block_scores = self.scorer(x).flatten().to(state.positions.device)
        keep = min(state.length, budget)
        sink = state.positions[:min(self.sink_tokens, keep)]
        recent = state.positions[-min(self.min_recent, max(0, keep - sink.numel())):]
        protected = torch.unique(torch.cat([sink, recent]))
        room = max(0, keep - protected.numel())
        selected = [protected]
        for block_index in torch.argsort(block_scores, descending=True).tolist():
            start, end = spans[block_index]
            candidate = state.positions[start:end]
            available = room - sum(int(piece.numel()) for piece in selected[1:])
            if available <= 0:
                break
            selected.append(candidate[:available])
        return self._finish(torch.cat(selected), state, keep)


def policy_from_spec(spec: dict, learned_model=None, normaliser=None) -> Policy:
    kind = spec["type"]
    if kind == "full": return FullCachePolicy()
    if kind == "fifo": return FIFOPolicy(**{k: v for k, v in spec.items() if k != "type"})
    if kind == "random": return RandomPolicy(**{k: v for k, v in spec.items() if k != "type"})
    if kind == "uniform": return UniformPolicy(**{k: v for k, v in spec.items() if k != "type"})
    if kind == "sink_recent": return SinkRecentPolicy(**{k: v for k, v in spec.items() if k != "type"})
    if kind == "attention_topk": return AttentionTopKPolicy(**{k: v for k, v in spec.items() if k != "type"})
    if kind == "h2o": return H2OPolicy(**{k: v for k, v in spec.items() if k != "type"})
    if kind == "learned_block": return LearnedBlockPolicy(scorer=learned_model, normaliser=normaliser, **{k: v for k, v in spec.items() if k != "type"})
    raise ValueError(f"Unknown policy {kind!r}")


def default_policy_specs(include_learned: bool = False, seed: int = 590) -> list[dict]:
    specs = [
        {"type": "full"}, {"type": "fifo", "min_recent": 32},
        {"type": "random", "seed": seed, "min_recent": 32}, {"type": "uniform", "min_recent": 32},
        {"type": "sink_recent", "sink_tokens": 4, "min_recent": 32},
        {"type": "attention_topk", "sink_tokens": 4, "min_recent": 32},
        {"type": "h2o", "recent_fraction": 0.50, "sink_tokens": 4, "min_recent": 32},
    ]
    if include_learned:
        specs.append({"type": "learned_block", "block_size": 16, "sink_tokens": 4, "min_recent": 32})
    return specs
