"""Lightweight learned KV-importance scorer and reproducible ranking training."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .features import BLOCK_FEATURE_NAMES, FEATURE_VERSION, FeatureNormaliser
from .runtime import DEFAULT_SEED, stable_hash, write_json


@dataclass
class ScorerConfig:
    input_dim: int = 28
    hidden_1: int = 64
    hidden_2: int = 32
    dropout: float = 0.10
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 128
    max_epochs: int = 30
    patience: int = 5
    loss: str = "pairwise_logistic"
    seed: int = DEFAULT_SEED


class KVImportanceMLP(nn.Module):
    """LayerNorm -> 64 GELU -> Dropout -> 32 GELU -> Dropout -> score."""
    def __init__(self, config: ScorerConfig | None = None):
        super().__init__()
        self.config = config or ScorerConfig()
        if self.config.input_dim != len(BLOCK_FEATURE_NAMES):
            raise ValueError(f"Expected {len(BLOCK_FEATURE_NAMES)} block features")
        self.network = nn.Sequential(
            nn.LayerNorm(self.config.input_dim),
            nn.Linear(self.config.input_dim, self.config.hidden_1), nn.GELU(), nn.Dropout(self.config.dropout),
            nn.Linear(self.config.hidden_1, self.config.hidden_2), nn.GELU(), nn.Dropout(self.config.dropout),
            nn.Linear(self.config.hidden_2, 1),
        )

    def forward(self, x):
        return self.network(x).squeeze(-1)


def split_documents(label_rows: list[dict], seed: int = DEFAULT_SEED) -> dict[str, list[dict]]:
    """Return source-assigned splits when present, otherwise a stable 60/20/20 split.

    Controlled-corpus labels carry ``source_partition`` copied from the canonical
    30/10/10-per-length design. Preserving it prevents a later scorer stage from
    silently reshuffling records across the fixed train/validation/test boundary.
    """
    groups = {"train": [], "validation": [], "test": []}
    for row in label_rows:
        supplied = str(row.get("source_partition", "")).strip().lower()
        if supplied in groups:
            partition = supplied
        else:
            doc_id = str(row.get("example_id", row.get("document_id", "unlabeled-document")))
            bucket = stable_hash(doc_id, seed) % 10_000
            partition = "train" if bucket < 6000 else "validation" if bucket < 8000 else "test"
        groups[partition].append(row)
    return groups


def _tensorize(rows: list[dict], device):
    """Stack saved label features without re-wrapping tensor objects.

    Oracle labels persist each feature as a CPU ``torch.Tensor``. ``torch.tensor``
    cannot safely rebuild a two-dimensional matrix from a list of such tensors
    on all supported PyTorch releases, so normalize each row and use ``stack``.
    """
    if not rows:
        raise ValueError("No rows supplied for scorer training/evaluation")
    features = []
    width = None
    for index, row in enumerate(rows):
        feature = row.get("feature")
        if feature is None:
            raise KeyError(f"Label row {index} is missing 'feature'")
        vector = torch.as_tensor(feature, dtype=torch.float32).detach().cpu().reshape(-1)
        if width is None:
            width = int(vector.numel())
        elif int(vector.numel()) != width:
            raise ValueError(f"Inconsistent feature width at row {index}: expected {width}, got {vector.numel()}")
        features.append(vector)
    x = torch.stack(features, dim=0).to(device)
    y = torch.as_tensor([float(row["target"]) for row in rows], dtype=torch.float32, device=device)
    docs = [str(row.get("example_id", row.get("document_id", "unlabeled-document"))) for row in rows]
    return x, y, docs


def pairwise_logistic_loss(scores, target, docs) -> torch.Tensor:
    """Pairwise logistic loss only for pairs from the same logical document."""
    losses = []
    by_doc: dict[str, list[int]] = {}
    for idx, doc in enumerate(docs):
        by_doc.setdefault(doc, []).append(idx)
    for indexes in by_doc.values():
        if len(indexes) < 2:
            continue
        idx = torch.tensor(indexes, device=scores.device)
        s, y = scores.index_select(0, idx), target.index_select(0, idx)
        diffs = y[:, None] - y[None, :]
        mask = diffs > 1e-8
        if mask.any():
            score_diffs = s[:, None] - s[None, :]
            losses.append(F.softplus(-score_diffs[mask]).mean())
    if not losses:
        # A single-document/single-block smoke data set should not silently
        # train arbitrary weights; it is an explicit error.
        raise ValueError("Pairwise loss requires at least one within-document pair with unequal targets")
    return torch.stack(losses).mean()


def ranking_metrics(rows: list[dict], scores: list[float] | torch.Tensor, k: int = 3) -> dict[str, float]:
    """Document-averaged NDCG@k and pairwise agreement, deterministic and transparent."""
    import math
    if isinstance(scores, torch.Tensor):
        scores = scores.detach().cpu().tolist()
    grouped: dict[str, list[tuple[float, float]]] = {}
    for row, score in zip(rows, scores):
        doc = str(row.get("example_id", row.get("document_id", "unlabeled-document")))
        grouped.setdefault(doc, []).append((float(score), float(row["target"])))
    ndcgs, agreements = [], []
    for values in grouped.values():
        ordered = sorted(values, key=lambda pair: pair[0], reverse=True)
        ideal = sorted(values, key=lambda pair: pair[1], reverse=True)
        def dcg(seq): return sum((2 ** rel - 1) / math.log2(rank + 2) for rank, (_, rel) in enumerate(seq[:k]))
        ideal_dcg = dcg(ideal)
        if ideal_dcg > 0:
            ndcgs.append(dcg(ordered) / ideal_dcg)
        for i in range(len(values)):
            for j in range(i + 1, len(values)):
                if values[i][1] == values[j][1]:
                    continue
                agreements.append(float((values[i][0] - values[j][0]) * (values[i][1] - values[j][1]) > 0))
    return {"ndcg_at_k": float(sum(ndcgs) / max(len(ndcgs), 1)), "pairwise_agreement": float(sum(agreements) / max(len(agreements), 1)), "documents": len(grouped)}


def train_scorer(label_rows: list[dict], config: ScorerConfig | None = None, device: str | None = None,
                 run_root: str | Path | None = None) -> tuple[KVImportanceMLP, FeatureNormaliser, dict[str, Any]]:
    config = config or ScorerConfig()
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(config.seed)
    splits = split_documents(label_rows, config.seed)
    if not splits["train"] or not splits["validation"]:
        raise ValueError("Need nonempty train and validation document partitions; increase profiling examples")
    x_train, y_train, docs_train = _tensorize(splits["train"], device)
    x_val, y_val, docs_val = _tensorize(splits["validation"], device)
    normaliser = FeatureNormaliser().fit(x_train)
    x_train, x_val = normaliser.transform(x_train), normaliser.transform(x_val)
    model = KVImportanceMLP(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    best_state, best_val, stalled = None, float("-inf"), 0
    history = []
    for epoch in range(1, config.max_epochs + 1):
        model.train()
        order = torch.randperm(x_train.shape[0], device=device)
        epoch_loss = 0.0
        batches = 0
        for start in range(0, order.numel(), config.batch_size):
            index = order[start:start + config.batch_size]
            if index.numel() < 2:
                continue
            # Keep doc-level pairs by using full batch documents where possible.
            batch_scores = model(x_train.index_select(0, index))
            batch_target = y_train.index_select(0, index)
            batch_docs = [docs_train[int(i)] for i in index.detach().cpu().tolist()]
            try:
                loss = pairwise_logistic_loss(batch_scores, batch_target, batch_docs)
            except ValueError:
                continue
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += float(loss.item()); batches += 1
        model.eval()
        with torch.no_grad():
            val_scores = model(x_val)
        val_metrics = ranking_metrics(splits["validation"], val_scores)
        current = val_metrics["ndcg_at_k"]
        history.append({"epoch": epoch, "train_pairwise_loss": epoch_loss / max(batches, 1), **val_metrics})
        if current > best_val + 1e-6:
            best_val, stalled = current, 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stalled += 1
        if stalled >= config.patience:
            break
    if best_state is None:
        raise RuntimeError("No valid scorer training update occurred; inspect label diversity and batch size")
    model.load_state_dict(best_state)
    model.eval()
    summary = {
        "config": asdict(config), "feature_version": FEATURE_VERSION, "feature_names": list(BLOCK_FEATURE_NAMES),
        "splits": {name: len(rows) for name, rows in splits.items()}, "history": history, "best_validation_ndcg_at_3": best_val,
    }
    if run_root is not None:
        run_root = Path(run_root)
        checkpoints_dir = run_root / "checkpoints"
        checkpoints_dir.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": model.state_dict(), "normaliser": normaliser.state_dict(), "config": asdict(config), "feature_names": BLOCK_FEATURE_NAMES}, checkpoints_dir / "learned_kv_scorer.pt")
        write_json(checkpoints_dir / "learned_kv_scorer_manifest.json", summary)
    return model, normaliser, summary


def load_scorer(checkpoint_path: str | Path, device: str | None = None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    payload = torch.load(checkpoint_path, map_location=device)
    config = ScorerConfig(**payload["config"])
    model = KVImportanceMLP(config).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    normaliser = FeatureNormaliser.from_state_dict(payload["normaliser"])
    return model, normaliser, payload
