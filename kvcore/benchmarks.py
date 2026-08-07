"""Benchmark adapters and the canonical controlled retrieval corpus.

The active capstone data protocol is a deterministic RULER-like single-needle
retrieval corpus. Public RULER and LongBench adapters remain available for
future controls and external-validity work, but are not active sources in the
current Colab execution path.
"""
from __future__ import annotations

import hashlib
import io
import json
import zipfile
from pathlib import Path
from typing import Any, Iterable

from .config import BENCHMARKS, CONTROLLED_RETRIEVAL, DEFAULT_SEED
from .runtime import stable_hash, write_json


def project_partition(benchmark: str, task: str, row_id: str, seed: int = DEFAULT_SEED,
                      train_fraction: float = 0.60, validation_fraction: float = 0.20) -> str:
    """Assign a stable generic project partition at the example/document level."""
    value = stable_hash(f"{benchmark}:{task}:{row_id}", seed) % 10_000
    boundary_train = int(train_fraction * 10_000)
    boundary_val = int((train_fraction + validation_fraction) * 10_000)
    return "train" if value < boundary_train else "validation" if value < boundary_val else "test"


def deterministic_sample(rows: Iterable[dict], n: int | None, seed: int = DEFAULT_SEED) -> list[dict]:
    ranked = sorted((dict(row) for row in rows), key=lambda row: stable_hash(str(row["row_id"]), seed))
    return ranked if n is None else ranked[:n]


def _partition_for_fixed_index(index: int) -> str:
    """Return the canonical 30/10/10 split within a 50-record length stratum."""
    if index < 30:
        return "train"
    if index < 40:
        return "validation"
    return "test"


def _position_band(index: int) -> tuple[str, float]:
    """Cycle deterministic early/middle/late answer locations across records."""
    options = (("early", 0.10), ("middle", 0.50), ("late", 0.90))
    return options[index % len(options)]


def _token_count(text: str, tokenizer: Any | None) -> int | None:
    if tokenizer is None:
        return None
    encoded = tokenizer(text, add_special_tokens=False)
    ids = encoded.input_ids if hasattr(encoded, "input_ids") else encoded["input_ids"]
    return int(len(ids))


def _controlled_filler(tokenizer: Any | None, token_count: int) -> str:
    """Create deterministic neutral filler with approximately the requested token count."""
    if token_count <= 0:
        return ""
    if tokenizer is None:
        return " ".join(["context"] * token_count)
    candidates = tokenizer.encode(" evidence", add_special_tokens=False)
    if not candidates:
        return " ".join(["context"] * token_count)
    # Decoding a repeated ordinary token produces a deterministic neutral context
    # and usually re-encodes to the same or nearly the same count.
    return tokenizer.decode(
        [int(candidates[-1])] * token_count,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )


def _controlled_prompt(answer: str, target_tokens: int, answer_fraction: float,
                       tokenizer: Any | None) -> tuple[str, str, int | None, float]:
    """Construct one deterministic context at a planned total prompt length.

    When a tokenizer is available, a short correction loop adjusts the suffix
    so the stored measured length is close to the planned length rather than
    merely trusting whitespace approximation.
    """
    question = "What is the retrieval key? Reply with the exact key only."
    needle = f" The retrieval key is {answer}. "
    question_tokens = _token_count(f"\n\nQuestion: {question}\nAnswer:", tokenizer) or 14
    needle_tokens = _token_count(needle, tokenizer) or 10
    available = max(target_tokens - question_tokens - needle_tokens, 32)
    prefix_tokens = max(1, int(available * answer_fraction))
    suffix_tokens = max(1, available - prefix_tokens)
    measured: int | None = None
    for _ in range(4):
        prefix = _controlled_filler(tokenizer, prefix_tokens)
        suffix = _controlled_filler(tokenizer, suffix_tokens)
        passage = f"{prefix}{needle}{suffix}".strip()
        prompt = f"{passage}\n\nQuestion: {question}\nAnswer:"
        measured = _token_count(prompt, tokenizer)
        if measured is None:
            break
        delta = target_tokens - measured
        if abs(delta) <= 1:
            break
        suffix_tokens = max(1, suffix_tokens + delta)
    position_pct = round(100.0 * prefix_tokens / max(prefix_tokens + needle_tokens + suffix_tokens, 1), 4)
    return passage, prompt, measured, position_pct


def generate_controlled_retrieval_corpus(tokenizer: Any | None = None, seed: int = DEFAULT_SEED) -> list[dict]:
    """Generate the canonical balanced 200-record RULER-like retrieval corpus.

    The corpus contains four planned prompt lengths, 50 records per length, and
    an exact 30/10/10 train/validation/test allocation within every length
    stratum. Each record has one answer-bearing needle placed in an early,
    middle, or late location. The token count is measured with the supplied
    tokenizer and stored for later cache accounting.
    """
    spec = CONTROLLED_RETRIEVAL
    rows: list[dict] = []
    for planned_length in spec["context_lengths"]:
        for index in range(spec["records_per_length"]):
            partition = _partition_for_fixed_index(index)
            position_band, fraction = _position_band(index)
            answer = f"KEY-{planned_length}-{index:02d}-{stable_hash(f'{planned_length}:{index}', seed) % 10_000:04d}"
            passage, prompt, measured, position_pct = _controlled_prompt(answer, int(planned_length), fraction, tokenizer)
            row_id = f"controlled-{planned_length}-{index:03d}"
            content_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
            rows.append({
                "benchmark": spec["name"],
                "task": spec["task"],
                "row_id": row_id,
                "original_source_id": row_id,
                "source_collection": "controlled_corpus",
                "source_subset": "single_needle",
                "source_revision": spec["source_revision"],
                "generator_version": spec["generator_version"],
                "generation_seed": int(seed),
                "partition": partition,
                "planned_context_length": int(planned_length),
                "measured_context_length": measured,
                "declared_length": int(planned_length),
                "answer_location_band": position_band,
                "answer_location_pct": position_pct,
                "num_hidden_targets": 1,
                "passage": passage,
                "question": "What is the retrieval key? Reply with the exact key only.",
                "prompt": prompt,
                "answers": [answer],
                "token_counting_method": spec["token_counting_method"] if tokenizer is not None else "whitespace_fallback",
                "content_hash": content_hash,
            })
    return rows


def audit_controlled_retrieval_corpus(rows: list[dict], max_length_drift: float = 0.05) -> dict[str, Any]:
    """Validate schema, balance, content integrity, and token-length drift."""
    spec = CONTROLLED_RETRIEVAL
    required = {
        "benchmark", "task", "row_id", "partition", "source_revision", "planned_context_length",
        "passage", "question", "prompt", "answers", "answer_location_pct", "content_hash",
    }
    missing = [row["row_id"] for row in rows if not required.issubset(row)]
    duplicates = len({row.get("content_hash") for row in rows}) != len(rows)
    missing_answers = [row["row_id"] for row in rows if not row.get("answers") or str(row["answers"][0]) not in str(row.get("passage", ""))]
    invalid_positions = [row["row_id"] for row in rows if not 0.0 <= float(row.get("answer_location_pct", -1.0)) <= 100.0]
    drift_rows = []
    for row in rows:
        measured = row.get("measured_context_length")
        planned = int(row.get("planned_context_length", 0))
        if measured is None or planned <= 0:
            continue
        drift = abs(int(measured) - planned) / planned
        if drift > max_length_drift:
            drift_rows.append({"row_id": row["row_id"], "planned": planned, "measured": int(measured), "drift": drift})
    strata: dict[str, dict[str, int]] = {}
    for length in spec["context_lengths"]:
        subset = [row for row in rows if int(row.get("planned_context_length", -1)) == int(length)]
        strata[str(length)] = {partition: sum(row.get("partition") == partition for row in subset) for partition in ("train", "validation", "test")}
        strata[str(length)]["total"] = len(subset)
    expected = spec["partition_counts"]
    balance_errors = {
        length: counts
        for length, counts in strata.items()
        if counts["total"] != spec["records_per_length"]
        or counts["train"] != expected["train"]
        or counts["validation"] != expected["validation"]
        or counts["test"] != expected["test"]
    }
    issues = {
        "missing_required_fields": missing,
        "duplicate_content_hashes": duplicates,
        "answers_absent_from_passage": missing_answers,
        "invalid_answer_locations": invalid_positions,
        "length_drift_above_threshold": drift_rows,
        "balance_errors": balance_errors,
    }
    passed = not any(bool(value) for value in issues.values()) and len(rows) == len(spec["context_lengths"]) * spec["records_per_length"]
    return {
        "dataset": spec["name"],
        "source_revision": spec["source_revision"],
        "generator_version": spec["generator_version"],
        "row_count": len(rows),
        "expected_row_count": len(spec["context_lengths"]) * spec["records_per_length"],
        "context_lengths": list(spec["context_lengths"]),
        "partition_counts_by_length": strata,
        "max_length_drift": max_length_drift,
        "passed": passed,
        "issues": issues,
    }


def load_controlled_retrieval(tokenizer: Any | None = None, seed: int = DEFAULT_SEED,
                              max_examples: int | None = None, partitions: tuple[str, ...] | None = None) -> list[dict]:
    """Return the canonical controlled corpus or a deterministic filtered subset."""
    rows = generate_controlled_retrieval_corpus(tokenizer=tokenizer, seed=seed)
    if partitions is not None:
        allowed = set(partitions)
        rows = [row for row in rows if row["partition"] in allowed]
    if max_examples is None:
        return rows
    # Preserve length strata under a cap by deterministic ranking within each stratum.
    selected: list[dict] = []
    for length in CONTROLLED_RETRIEVAL["context_lengths"]:
        subset = [row for row in rows if row["planned_context_length"] == length]
        selected.extend(deterministic_sample(subset, max_examples, seed))
    return selected


def load_ruler(config: str, max_examples: int | None = None, seed: int = DEFAULT_SEED) -> list[dict]:
    """Load optional public RULER controls at a pinned revision."""
    spec = BENCHMARKS["ruler"]
    from datasets import get_dataset_config_names, load_dataset
    allowed = set(spec["configs"])
    if config not in allowed:
        try:
            allowed.update(get_dataset_config_names(spec["repo_id"], revision=spec["revision"]))
        except Exception as error:
            raise ValueError(f"Cannot verify non-default RULER config {config!r} at pinned revision") from error
    if config not in allowed:
        raise ValueError(f"Unsupported RULER config {config!r}; discovered configurations do not include it")
    ds = load_dataset(spec["repo_id"], config, split=spec["split"], revision=spec["revision"])
    rows = []
    for item in ds:
        row_id = str(item["index"])
        rows.append({
            "benchmark": "ruler",
            "task": config,
            "row_id": row_id,
            "prompt": str(item["input"]),
            "answers": [str(value) for value in item["outputs"]],
            "declared_length": int(item.get("length", 0)),
            "source_revision": spec["revision"],
            "partition": project_partition("ruler", config, row_id, seed),
        })
    return deterministic_sample(rows, max_examples, seed)


def _longbench_archive_items(task: str) -> list[dict]:
    spec = BENCHMARKS["longbench"]
    from huggingface_hub import hf_hub_download
    archive_path = hf_hub_download(
        repo_id=spec["repo_id"], repo_type="dataset", filename=spec["archive_filename"], revision=spec["revision"],
    )
    member = f"data/{task}.jsonl"
    with zipfile.ZipFile(archive_path) as archive:
        if member not in archive.namelist():
            available = sorted(Path(name).stem for name in archive.namelist() if name.startswith("data/") and name.endswith(".jsonl"))
            raise ValueError(f"LongBench task {task!r} missing from official archive. Available examples: {available[:12]}")
        content = archive.read(member).decode("utf-8")
    return [json.loads(line) for line in io.StringIO(content) if line.strip()]


def load_longbench(task: str, max_examples: int | None = None, seed: int = DEFAULT_SEED) -> list[dict]:
    """Load optional LongBench external-validity data at a pinned revision."""
    spec = BENCHMARKS["longbench"]
    if task not in spec["tasks"]:
        raise ValueError(f"Unsupported LongBench task {task!r}; choose from {spec['tasks']}")
    loader = "official_hub_archive"
    try:
        from datasets import load_dataset
        ds = load_dataset(spec["repo_id"], task, split=spec["split"], revision=spec["revision"])
        items = [dict(item) for item in ds]
        loader = "datasets"
    except Exception:
        items = _longbench_archive_items(task)
    rows = []
    for i, item in enumerate(items):
        row_id = str(item.get("_id", item.get("id", f"{task}:{i}")))
        context = str(item.get("context", "")).strip()
        question = str(item.get("input", "")).strip()
        prompt = f"{context}\n\nQuestion: {question}\nAnswer:" if context else question
        answers = item.get("answers", [])
        if isinstance(answers, str):
            answers = [answers]
        rows.append({
            "benchmark": "longbench_v1",
            "task": task,
            "row_id": row_id,
            "prompt": prompt,
            "context": context,
            "question": question,
            "answers": [str(value) for value in answers],
            "all_classes": list(item.get("all_classes") or []),
            "declared_length": item.get("length"),
            "source_revision": spec["revision"],
            "loader": loader,
            "partition": project_partition("longbench_v1", task, row_id, seed),
        })
    return deterministic_sample(rows, max_examples, seed)


def build_benchmark_manifest(rows: list[dict], run_root: str | Path, benchmark_name: str | None = None,
                             seed: int = DEFAULT_SEED, tokenizer_id: str | None = None,
                             tokenizer_revision: str | None = None, model_id: str | None = None,
                             model_revision: str | None = None) -> Path:
    """Write an auditable manifest, including controlled-corpus fields when present."""
    if not rows:
        raise ValueError("Cannot write a benchmark manifest for zero records")
    required = {"benchmark", "task", "row_id", "partition", "source_revision"}
    missing = [index for index, row in enumerate(rows) if not required.issubset(row)]
    if missing:
        raise ValueError(f"Rows at positions {missing[:5]} are missing benchmark-manifest fields")
    target = Path(run_root)
    if target.suffix.lower() == ".json":
        output_path = target
        resolved_name = benchmark_name or target.stem
    else:
        resolved_name = benchmark_name or "benchmark"
        output_path = target / "benchmark_manifests" / f"{resolved_name}_manifest.json"
    base_keys = ("benchmark", "task", "row_id", "partition", "source_revision")
    controlled_keys = (
        "original_source_id", "source_collection", "source_subset", "planned_context_length",
        "measured_context_length", "answer_location_band", "answer_location_pct", "num_hidden_targets",
        "generation_seed", "generator_version", "token_counting_method", "content_hash",
    )
    manifest_rows = []
    for row in rows:
        item = {key: row[key] for key in base_keys}
        item.update({key: row[key] for key in controlled_keys if key in row})
        manifest_rows.append(item)
    payload = {
        "benchmark_name": resolved_name,
        "seed": seed,
        "tokenizer_id": tokenizer_id,
        "tokenizer_revision": tokenizer_revision,
        "model_id": model_id,
        "model_revision": model_revision,
        "row_count": len(rows),
        "rows": manifest_rows,
    }
    return write_json(output_path, payload)
