"""Pinned-model inference and policy-sweep evaluation helpers.

These routines record observed measurements only. A skipped or unsupported
runtime yields an explicit `requires <GPU> — not run here` status rather than a
synthetic latency or quality value.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from .accounting import formula_gate, measured_cache_bytes
from .cache import greedy_decode_step, prefill
from .features import safe_norms
from .policies import CacheState, policy_from_spec
from .profiling import long_context_attention_guard
from .runtime import append_jsonl, cuda_memory_snapshot, reset_cuda_peak_memory, write_json, write_table_csv


def _torch():
    import torch
    return torch


def load_model_and_tokenizer(model_spec: dict[str, str], load_mode: str = "bf16", attn_implementation: str = "sdpa"):
    """Load a model at immutable Hub revisions with an auditable configuration."""
    torch = _torch()
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    tokenizer = AutoTokenizer.from_pretrained(model_spec["tokenizer_id"], revision=model_spec["tokenizer_revision"], use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    kwargs: dict[str, Any] = {"revision": model_spec["revision"], "attn_implementation": attn_implementation, "low_cpu_mem_usage": True}
    if load_mode == "4bit":
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_type="nf4")
        kwargs["device_map"] = "auto"
    else:
        kwargs["torch_dtype"] = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
        kwargs["device_map"] = "auto" if torch.cuda.is_available() else None
    model = AutoModelForCausalLM.from_pretrained(model_spec["model_id"], **kwargs)
    model.eval()
    return model, tokenizer


def tokenise_prompt(tokenizer, prompt: str, max_context_tokens: int, device=None):
    torch = _torch()
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ids = tokenizer(prompt, return_tensors="pt", truncation=False).input_ids
    if ids.shape[-1] > max_context_tokens:
        # Explicit tail preservation is recorded in trace metadata. It prevents
        # accidental silent overrun on small GPUs.
        ids = ids[:, -max_context_tokens:]
    return ids.to(device)


def answer_ids(tokenizer, answers: list[str], device):
    text = str(answers[0]) if answers else ""
    ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids
    return ids.to(device)


def simple_quality(generated: str, answers: list[str]) -> dict[str, float]:
    """Transparent exact/substring comparison, not a replacement for task-specific LongBench scoring."""
    norm = lambda value: " ".join(str(value).strip().lower().split())
    pred = norm(generated)
    refs = [norm(value) for value in answers if norm(value)]
    exact = float(any(pred == reference for reference in refs)) if refs else float("nan")
    contains = float(any(reference in pred or pred in reference for reference in refs)) if refs and pred else float("nan")
    return {"exact_match": exact, "substring_match": contains}


def _state_from_cache(cache, profile=None):
    torch = _torch()
    positions = cache.physical_positions
    n = positions.numel()
    def aligned(name):
        if profile is None or name not in profile:
            return torch.zeros(n, device=positions.device)
        values = profile[name].to(positions.device).flatten()
        if values.numel() >= n:
            return values[-n:]
        return torch.cat([values, torch.zeros(n - values.numel(), device=positions.device)])
    key_norm, value_norm = safe_norms(cache)
    if key_norm.numel() != n: key_norm = torch.zeros(n, device=positions.device)
    if value_norm.numel() != n: value_norm = torch.zeros(n, device=positions.device)
    return CacheState(
        positions=positions, cumulative_attention=aligned("cumulative_attention"), max_attention=aligned("max_attention"),
        attention_hits=aligned("attention_hits"), key_norm=key_norm, value_norm=value_norm, step=max(n, 1),
    )


def generate_with_policy(model, tokenizer, prompt_ids, policy, budget: int, max_new_tokens: int,
                         profile=None, prefill_chunk_tokens: int = 512, run_root: str | Path | None = None,
                         trace_tag: str = "run") -> dict[str, Any]:
    """Run one long-context attention-free evaluation with one cache policy."""
    torch = _torch()
    guard = long_context_attention_guard(model)
    reset_cuda_peak_memory()
    start = time.perf_counter()
    cache, initial = prefill(model, prompt_ids, chunk_size=prefill_chunk_tokens, output_attentions=False)
    before = cache.snapshot(f"{trace_tag}_prefill", run_root)
    state = _state_from_cache(cache, profile)
    compact_info = cache.apply_policy(policy, state, budget, reason=f"{policy.name}_post_prefill")
    after = cache.snapshot(f"{trace_tag}_post_evict", run_root)
    token = initial.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    generated = []
    policy_overhead_seconds = 0.0
    for decode_index in range(max_new_tokens):
        evict_start = time.perf_counter()
        state = _state_from_cache(cache, profile)
        cache.apply_policy(policy, state, budget, reason=f"{policy.name}_decode_{decode_index}")
        policy_overhead_seconds += time.perf_counter() - evict_start
        cache, token, _ = greedy_decode_step(model, cache, token, logical_position=cache.logical_length, output_attentions=False)
        generated.append(int(token.item()))
        if tokenizer.eos_token_id is not None and int(token.item()) == int(tokenizer.eos_token_id):
            break
    elapsed = time.perf_counter() - start
    text = tokenizer.decode(generated, skip_special_tokens=True)
    trace = {
        "policy": policy.name, "budget": int(budget), "prompt_tokens": int(prompt_ids.shape[-1]), "generated_tokens": len(generated),
        "generated_text": text, "prefill_snapshot": before, "post_evict_snapshot": after,
        "compaction": compact_info, "elapsed_seconds": elapsed, "policy_overhead_seconds": policy_overhead_seconds,
        "long_context_attention_guard": guard, "measurement_class": "observed_model_run",
    }
    if run_root is not None:
        append_jsonl(Path(run_root) / "traces" / f"{trace_tag}.jsonl", trace)
    return trace


def _resume_key(row: dict, policy_name: str, budget: int) -> str:
    """Deterministic identity for one row-policy-budget evaluation."""
    payload = {
        "benchmark": row.get("benchmark"), "task": row.get("task"), "row_id": row.get("row_id"),
        "partition": row.get("partition"), "policy": policy_name, "budget": int(budget),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _resume_fingerprint(artifact_stem: str, policy_specs: list[dict], budgets, max_context_tokens: int,
                        decode_tokens: int, metadata: dict | None) -> str:
    payload = {
        "schema": "kv_eval_resume_v1", "artifact_stem": artifact_stem, "policy_specs": policy_specs,
        "budgets": list(budgets), "max_context_tokens": int(max_context_tokens),
        "decode_tokens": int(decode_tokens), "metadata": metadata or {},
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def _load_resume_records(progress_path: Path) -> tuple[list[dict], set[str]]:
    records, completed = [], set()
    if not progress_path.exists():
        return records, completed
    for line in progress_path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        key = row.pop("_resume_key", None)
        if key:
            completed.add(key)
            records.append(row)
    return records, completed


def evaluate_rows(model, tokenizer, rows: list[dict], policy_specs: list[dict], budgets: list[int] | tuple[int, ...],
                  max_context_tokens: int, decode_tokens: int, prefill_chunk_tokens: int, run_root: str | Path,
                  profile=None, learned_model=None, normaliser=None, artifact_stem: str = "results",
                  resume: bool = False, resume_metadata: dict | None = None) -> list[dict]:
    """Evaluate observed benchmark rows with optional deterministic per-record resume.

    When ``resume`` is enabled, every completed row-policy-budget measurement is
    immediately appended to a JSONL journal and the CSV is refreshed. Re-entry
    skips only records whose immutable identity already appears in the matching
    journal. The metadata fingerprint prevents resuming after a checkpoint,
    policy-grid, budget, or test-manifest change.
    """
    torch = _torch()
    run_root = Path(run_root)
    result_path = run_root / "results" / f"{artifact_stem}.csv"
    progress_path = run_root / "results" / f"{artifact_stem}_progress.jsonl"
    resume_path = run_root / "manifests" / f"{artifact_stem}_resume_manifest.json"
    expected_records = sum(len(budgets) if spec.get("type") != "full" else 1 for spec in policy_specs) * len(rows)
    fingerprint = _resume_fingerprint(artifact_stem, policy_specs, budgets, max_context_tokens, decode_tokens, resume_metadata)
    records, completed = ([], set())
    if resume:
        if resume_path.exists():
            prior = json.loads(resume_path.read_text())
            if prior.get("fingerprint") != fingerprint:
                raise RuntimeError("Refusing to resume: locked-test configuration fingerprint differs from the existing progress manifest.")
        else:
            write_json(resume_path, {
                "schema": "kv_eval_resume_v1", "artifact_stem": artifact_stem, "fingerprint": fingerprint,
                "expected_records": expected_records, "resume_metadata": resume_metadata or {}, "status": "in_progress",
            })
        records, completed = _load_resume_records(progress_path)
        if len(records) != len(completed):
            raise RuntimeError("Resume journal contains duplicate or malformed records; do not continue the locked test.")
    device = next(model.parameters()).device
    for row in rows:
        prompt_ids = tokenise_prompt(tokenizer, row["prompt"], max_context_tokens, device=device)
        for policy_spec in policy_specs:
            policy = policy_from_spec(policy_spec, learned_model=learned_model, normaliser=normaliser)
            current_budgets = budgets if policy.name != "full" else [int(prompt_ids.shape[-1])]
            for budget in current_budgets:
                key = _resume_key(row, policy.name, int(budget))
                if key in completed:
                    continue
                tag = f"{artifact_stem}_{row['benchmark']}_{row['task']}_{row['row_id']}_{policy.name}_{budget}"
                try:
                    trace = generate_with_policy(model, tokenizer, prompt_ids, policy, int(budget), decode_tokens, profile, prefill_chunk_tokens, run_root, tag)
                    quality = simple_quality(trace["generated_text"], row.get("answers", []))
                    record = {
                        "status": "observed", "benchmark": row["benchmark"], "task": row["task"], "row_id": row["row_id"],
                        "partition": row.get("partition"), "policy": policy.name, "budget": int(budget),
                        "prompt_tokens": trace["prompt_tokens"], "generated_tokens": trace["generated_tokens"],
                        "physical_kv_bytes_prefill": trace["prefill_snapshot"]["payload_bytes"],
                        "physical_kv_bytes_post_evict": trace["post_evict_snapshot"]["payload_bytes"],
                        "effective_retained_tokens": trace["post_evict_snapshot"]["physical_length"],
                        "elapsed_seconds": trace["elapsed_seconds"], "policy_overhead_seconds": trace["policy_overhead_seconds"],
                        **quality,
                    }
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    record = {"status": "requires_larger_gpu_or_smaller_context", "benchmark": row["benchmark"], "task": row["task"], "row_id": row["row_id"], "policy": policy.name, "budget": int(budget)}
                except Exception as error:
                    record = {"status": "error", "benchmark": row["benchmark"], "task": row["task"], "row_id": row["row_id"], "policy": policy.name, "budget": int(budget), "error": repr(error)}
                records.append(record)
                completed.add(key)
                if resume:
                    append_jsonl(progress_path, {**record, "_resume_key": key})
                    write_table_csv(records, result_path)
                    write_json(resume_path, {
                        "schema": "kv_eval_resume_v1", "artifact_stem": artifact_stem, "fingerprint": fingerprint,
                        "expected_records": expected_records, "completed_records": len(records),
                        "resume_metadata": resume_metadata or {},
                        "status": "complete" if len(records) == expected_records else "in_progress",
                    })
                    # One progress line per completed test row (not every policy)
                    # provides a durable, human-readable heartbeat in notebook logs.
                    if len(records) % max(1, expected_records // max(1, len(rows))) == 0 or len(records) == expected_records:
                        print(f"{artifact_stem}: {len(records)}/{expected_records} policy-budget records checkpointed")
        del prompt_ids
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    write_table_csv(records, result_path)
    if resume:
        write_json(resume_path, {
            "schema": "kv_eval_resume_v1", "artifact_stem": artifact_stem, "fingerprint": fingerprint,
            "expected_records": expected_records, "completed_records": len(records),
            "resume_metadata": resume_metadata or {},
            "status": "complete" if len(records) == expected_records else "in_progress",
        })
    return records


def requires_gpu_record(required_gpu: str, command: str, reason: str) -> dict[str, Any]:
    return {"status": f"requires {required_gpu} — not run here", "required_gpu": required_gpu, "command": command, "reason": reason}


def fp8_payload_accounting(cache, quality_rows: list[dict] | None = None) -> dict[str, Any]:
    """Analytic byte accounting; actual FP8 model cache support is runtime-dependent.

    This is not reported as an observed compression result unless a runtime
    provides an FP8 cache implementation. The notebook labels it clearly.
    """
    fp16 = measured_cache_bytes(cache)
    return {"fp16_payload_bytes": fp16, "analytic_fp8_payload_bytes": fp16 // 2, "analytic_ratio": 0.5,
            "observed_fp8_cache": False, "note": "requires an FP8-capable cache backend for observed FP8 results"}
