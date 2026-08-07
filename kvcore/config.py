"""Versioned configuration for the KV-cache eviction Colab capstone suite.

All public model and dataset revisions are pinned deliberately.  A notebook may
only change them through an explicit configuration override that is recorded in
its runtime manifest.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

SUITE_VERSION = "2.1.0"
RESULT_SCHEMA_VERSION = "2.1.0"
DEFAULT_SEED = 590

# Canonical controlled corpus for the current capstone snapshot. LongBench is
# retained as a future external-validity extension, not an active evaluation
# source in this version of the Colab workflow.
CONTROLLED_RETRIEVAL = {
    "name": "controlled_ruler_like_single_needle",
    "generator_version": "controlled_niah_v1",
    "source_revision": "controlled_niah_v1_seed_590",
    "task": "single_needle_retrieval",
    "context_lengths": (1024, 2048, 4096, 8192),
    "records_per_length": 50,
    "partition_counts": {"train": 30, "validation": 10, "test": 10},
    "answer_position_bands": ("early", "middle", "late"),
    "token_counting_method": "configured_tokenizer",
}

PACKAGE_PINS = {
    "transformers": "4.56.2",
    "accelerate": "1.10.1",
    "datasets": "4.0.0",
    "huggingface_hub": "0.34.4",
    "bitsandbytes": "0.47.0",
    "safetensors": "0.6.2",
    "sentencepiece": "0.2.1",
    "scipy": "1.16.1",
    "matplotlib": "3.10.6",
    "seaborn": "0.13.2",
    "pandas": "2.3.2",
}

MODELS = {
    "t4": {
        "model_id": "Qwen/Qwen2.5-1.5B-Instruct",
        "revision": "989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
        "tokenizer_id": "Qwen/Qwen2.5-1.5B-Instruct",
        "tokenizer_revision": "989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
    },
    "rtx_pro_6000": {
        "model_id": "Qwen/Qwen2.5-7B-Instruct",
        "revision": "a09a35458c702b33eeacc393d103063234e8bc28",
        "tokenizer_id": "Qwen/Qwen2.5-7B-Instruct",
        "tokenizer_revision": "a09a35458c702b33eeacc393d103063234e8bc28",
    },
    "h100": {
        "model_id": "Qwen/Qwen2.5-7B-Instruct-1M",
        "revision": "e28526f7bb80e2a9c8af03b831a9af3812f18fba",
        "tokenizer_id": "Qwen/Qwen2.5-7B-Instruct-1M",
        "tokenizer_revision": "e28526f7bb80e2a9c8af03b831a9af3812f18fba",
    },
    # A100 uses the identical frozen locked-test model revision. The distinct
    # profile exists solely to record the actual hardware faithfully.
    "a100": {
        "model_id": "Qwen/Qwen2.5-7B-Instruct-1M",
        "revision": "e28526f7bb80e2a9c8af03b831a9af3812f18fba",
        "tokenizer_id": "Qwen/Qwen2.5-7B-Instruct-1M",
        "tokenizer_revision": "e28526f7bb80e2a9c8af03b831a9af3812f18fba",
    },
}

BENCHMARKS = {
    "ruler": {
        "repo_id": "rbiswasfc/ruler",
        "revision": "af56d31d700e6450edb8895278c7b6842fc5c505",
        "configs": ("cwe_4k", "cwe_8k", "niah_multikey_1_4k", "niah_multikey_1_8k", "qa_2_4k", "qa_2_8k"),
        "split": "validation",
    },
    "longbench": {
        "repo_id": "zai-org/LongBench",
        "revision": "5e628be450b7e67fb7ae6e201bd6d8f7056f7672",
        "tasks": ("qasper", "gov_report", "hotpotqa", "2wikimqa"),
        "split": "test",
        "archive_filename": "data.zip",
    },
}

@dataclass(frozen=True)
class GPUProfile:
    name: str
    accepted_name_tokens: tuple[str, ...]
    recommended_vram_gib: int
    min_vram_gib: int
    model_tier: str
    load_mode: str
    dtype: str
    max_context_tokens: int
    prefill_chunk_tokens: int
    decode_tokens: int
    ruler_examples: int
    longbench_examples: int
    budgets: tuple[int, ...]
    attention_profile_context_cap: int
    strict_gpu_name: bool

    def asdict(self) -> dict[str, Any]:
        return asdict(self)


PROFILES = {
    "t4": GPUProfile(
        name="t4",
        accepted_name_tokens=("t4",),
        recommended_vram_gib=16,
        min_vram_gib=12,
        model_tier="t4",
        load_mode="4bit",
        dtype="float16",
        max_context_tokens=8192,
        prefill_chunk_tokens=512,
        decode_tokens=12,
        ruler_examples=8,
        longbench_examples=0,
        budgets=(512, 1024, 2048),
        attention_profile_context_cap=2048,
        strict_gpu_name=False,
    ),
    "rtx_pro_6000": GPUProfile(
        name="rtx_pro_6000",
        accepted_name_tokens=("rtx pro 6000", "rtx_pro_6000", "rtx 6000"),
        recommended_vram_gib=48,
        min_vram_gib=24,
        model_tier="rtx_pro_6000",
        load_mode="bf16_or_4bit",
        dtype="bfloat16",
        max_context_tokens=32768,
        prefill_chunk_tokens=1024,
        decode_tokens=24,
        ruler_examples=32,
        longbench_examples=32,
        budgets=(1024, 2048, 4096, 8192),
        attention_profile_context_cap=4096,
        strict_gpu_name=False,
    ),
    "h100": GPUProfile(
        name="h100",
        accepted_name_tokens=("h100",),
        recommended_vram_gib=80,
        min_vram_gib=40,
        model_tier="h100",
        load_mode="bf16",
        dtype="bfloat16",
        max_context_tokens=131072,
        prefill_chunk_tokens=4096,
        decode_tokens=32,
        ruler_examples=96,
        longbench_examples=96,
        budgets=(2048, 4096, 8192, 16384),
        attention_profile_context_cap=8192,
        strict_gpu_name=False,
    ),
    "a100": GPUProfile(
        name="a100",
        accepted_name_tokens=("a100",),
        recommended_vram_gib=40,
        min_vram_gib=30,
        model_tier="a100",
        load_mode="bf16",
        dtype="bfloat16",
        # The controlled locked corpus reaches 8,192 tokens; a conservative
        # 32k ceiling fits both 40 GB and 80 GB A100 variants without making
        # a misleading H100-scale context claim.
        max_context_tokens=32768,
        prefill_chunk_tokens=2048,
        decode_tokens=32,
        ruler_examples=40,
        longbench_examples=0,
        budgets=(2048, 4096, 8192, 16384),
        attention_profile_context_cap=4096,
        strict_gpu_name=False,
    ),
}

POLICY_DEFAULTS = {
    "full": {},
    "fifo": {"min_recent": 32},
    "random": {"seed": DEFAULT_SEED, "min_recent": 32},
    "uniform": {"min_recent": 32},
    "sink_recent": {"sink_tokens": 4, "min_recent": 32},
    "attention_topk": {"sink_tokens": 4, "min_recent": 32},
    "h2o": {"recent_fraction": 0.50, "sink_tokens": 4, "min_recent": 32},
    "learned_block": {"block_size": 16, "sink_tokens": 4, "min_recent": 32},
}


def model_spec(profile_name: str) -> dict[str, str]:
    if profile_name not in MODELS:
        raise KeyError(f"Unknown model tier {profile_name!r}")
    return dict(MODELS[profile_name])


def profile_spec(profile_name: str) -> GPUProfile:
    if profile_name not in PROFILES:
        raise KeyError(f"Unknown GPU profile {profile_name!r}; expected one of {sorted(PROFILES)}")
    return PROFILES[profile_name]
