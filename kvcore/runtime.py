"""Runtime, reproducibility, and artifact helpers for the Colab suite."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import subprocess
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .config import DEFAULT_SEED, RESULT_SCHEMA_VERSION, SUITE_VERSION, profile_spec


def _torch():
    import torch
    return torch


def json_default(value: Any):
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    try:
        torch = _torch()
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().tolist()
    except Exception:
        pass
    raise TypeError(f"Cannot serialize {type(value)!r}")


def write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=json_default) + "\n")
    return path


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def append_jsonl(path: str | Path, row: dict[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as out:
        out.write(json.dumps(row, sort_keys=True, default=json_default) + "\n")
    return path


def stable_hash(value: str, seed: int = DEFAULT_SEED) -> int:
    return int(hashlib.sha256(f"{seed}:{value}".encode("utf-8")).hexdigest()[:16], 16)


def set_all_seeds(seed: int = DEFAULT_SEED, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch = _torch()
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def nvidia_smi() -> list[dict[str, Any]]:
    try:
        rows = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.free,driver_version", "--format=csv,noheader,nounits"],
            text=True,
        ).strip().splitlines()
    except Exception:
        return []
    parsed = []
    for row in rows:
        fields = [field.strip() for field in row.split(",")]
        if len(fields) == 4:
            parsed.append({"name": fields[0], "total_mib": int(fields[1]), "free_mib": int(fields[2]), "driver": fields[3]})
    return parsed


def detect_runtime() -> dict[str, Any]:
    torch = _torch()
    gpus = nvidia_smi()
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": getattr(torch, "__version__", None),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda": getattr(torch.version, "cuda", None),
        "gpus": gpus,
    }


def select_profile(requested: str, allow_fallback: bool = True):
    """Return ``(active_profile_spec, runtime_status)`` without relabeling hardware.

    The two-value return contract is consumed by every delivered notebook.  The
    status record preserves the requested tier and the actual detected runtime,
    so a fallback execution is never presented as an H100/RTX measurement.
    """
    requested_spec = profile_spec(requested)
    gpus = nvidia_smi()
    if not gpus:
        if not allow_fallback:
            raise RuntimeError("No NVIDIA GPU is visible; select a GPU Colab runtime.")
        status = {"requested_profile": requested, "active_profile": "cpu_smoke", "fallback": True, "reason": "no_gpu", "gpu": None}
        # The notebooks halt before model execution when CUDA is absent.  Return
        # the requested specification only to keep the status/manifest path
        # type-stable until that clear error is raised.
        return requested_spec, status
    gpu = gpus[0]
    name = gpu["name"].lower()
    total_gib = gpu["total_mib"] / 1024
    matches_name = any(token in name for token in requested_spec.accepted_name_tokens)
    has_memory = total_gib >= requested_spec.min_vram_gib
    if matches_name and has_memory:
        status = {"requested_profile": requested, "active_profile": requested, "fallback": False, "reason": "matched", "gpu": gpu}
        return requested_spec, status
    if not allow_fallback:
        raise RuntimeError(f"Requested {requested}, detected {gpu['name']} with {total_gib:.1f} GiB.")
    # Conservative fallback: a low-VRAM device runs T4 controls, whereas a
    # larger unrecognized GPU runs RTX development controls.  The actual GPU
    # name remains in ``status`` and is written into every manifest.
    active = "t4" if total_gib < 24 else "rtx_pro_6000"
    status = {"requested_profile": requested, "active_profile": active, "fallback": True, "reason": "gpu_mismatch_or_low_vram", "gpu": gpu}
    return profile_spec(active), status


def ensure_run_root(root: str | Path) -> Path:
    root = Path(root)
    for rel in ("manifests", "benchmark_manifests", "cache_snapshots", "attention_profiles", "labels", "checkpoints", "traces", "results", "figures", "logs", "tests"):
        (root / rel).mkdir(parents=True, exist_ok=True)
    return root


def run_manifest(root: str | Path, *legacy_args: Any, notebook: str | None = None,
                 requested_profile: str | None = None, active_profile: str | None = None,
                 model: dict[str, str] | None = None, seed: int = DEFAULT_SEED,
                 extra: dict[str, Any] | None = None, runtime_status: dict[str, Any] | None = None) -> Path:
    """Write a runtime manifest while accepting legacy notebook call forms.

    Older notebook cells passed ``notebook`` and a profile positionally before
    supplying profile values by keyword.  Capturing those legacy values through
    ``*legacy_args`` prevents Python from raising a duplicate-argument error
    and retains the supplied keyword values as the source of truth.
    """
    unused_legacy = list(legacy_args)
    if notebook is None and unused_legacy:
        notebook = str(unused_legacy.pop(0))
    if active_profile is None and unused_legacy:
        active_profile = str(unused_legacy.pop(0))
    notebook = notebook or "runtime"
    requested_profile = requested_profile or active_profile or "unknown"
    active_profile = active_profile or requested_profile
    merged_extra = dict(extra or {})
    if runtime_status is not None:
        merged_extra["runtime_status"] = runtime_status
    if unused_legacy:
        merged_extra["ignored_legacy_manifest_args"] = unused_legacy
    root = ensure_run_root(root)
    payload = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "suite_version": SUITE_VERSION,
        "created_at_unix": time.time(),
        "notebook": notebook,
        "requested_profile": requested_profile,
        "active_profile": active_profile,
        "seed": seed,
        "runtime": detect_runtime(),
        "model": model,
        "extra": merged_extra,
    }
    return write_json(root / "manifests" / f"{notebook}_runtime_manifest.json", payload)


def cuda_memory_snapshot(label: str, device: int = 0) -> dict[str, Any]:
    torch = _torch()
    if not torch.cuda.is_available():
        return {"label": label, "cuda_available": False}
    torch.cuda.synchronize(device)
    return {
        "label": label,
        "cuda_available": True,
        "allocated_bytes": int(torch.cuda.memory_allocated(device)),
        "reserved_bytes": int(torch.cuda.memory_reserved(device)),
        "max_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "max_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
    }


def reset_cuda_peak_memory(device: int = 0) -> None:
    torch = _torch()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)


def write_table_csv(rows: list[dict[str, Any]], path: str | Path) -> Path:
    import pandas as pd
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    return path
