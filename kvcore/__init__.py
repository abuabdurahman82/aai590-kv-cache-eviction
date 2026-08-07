"""Shared KV-cache eviction core used by every GPU-profiled Colab notebook."""
from .accounting import formula_for_model, formula_gate, kv_formula_bytes, measured_cache_bytes
from .benchmarks import audit_controlled_retrieval_corpus, build_benchmark_manifest, generate_controlled_retrieval_corpus, load_controlled_retrieval, load_longbench, load_ruler, project_partition
from .cache import EvictingCache, greedy_decode_step, prefill
from .config import BENCHMARKS, CONTROLLED_RETRIEVAL, DEFAULT_SEED, MODELS, PACKAGE_PINS, POLICY_DEFAULTS, PROFILES, RESULT_SCHEMA_VERSION, SUITE_VERSION, model_spec, profile_spec
from .evaluation import evaluate_rows, fp8_payload_accounting, load_model_and_tokenizer, requires_gpu_record, tokenise_prompt
from .features import BLOCK_FEATURE_NAMES, FEATURE_VERSION, FeatureNormaliser, aggregate_blocks, token_features
from .labels import label_blocks, save_labels
from .policies import CacheState, default_policy_specs, policy_from_spec
from .profiling import load_profile, profile_short_context
from .runtime import ensure_run_root, run_manifest, select_profile, set_all_seeds, write_json
from .scorer import KVImportanceMLP, ScorerConfig, load_scorer, ranking_metrics, train_scorer
from .tests import assert_artifact, assert_checkpoint_reload, assert_equal_memory_granularity, assert_formula_gate, assert_fp8_byte_ratio, assert_frontier_coverage, assert_full_cache_equivalence, assert_nonfabricated, assert_policy_respects_budget

# Stable concise aliases retained for notebooks and downstream validation code.
make_policy = policy_from_spec
ImportanceScorer = KVImportanceMLP

__all__ = [
    "EvictingCache", "formula_for_model", "formula_gate", "kv_formula_bytes", "measured_cache_bytes",
    "load_controlled_retrieval", "generate_controlled_retrieval_corpus", "audit_controlled_retrieval_corpus", "load_longbench", "load_ruler", "build_benchmark_manifest", "project_partition", "prefill", "greedy_decode_step",
    "default_policy_specs", "policy_from_spec", "make_policy", "CacheState", "profile_short_context", "load_profile",
    "label_blocks", "save_labels", "KVImportanceMLP", "ImportanceScorer", "ScorerConfig", "train_scorer", "load_scorer",
    "ranking_metrics", "evaluate_rows", "load_model_and_tokenizer", "fp8_payload_accounting", "requires_gpu_record", "tokenise_prompt",
    "ensure_run_root", "run_manifest", "select_profile", "set_all_seeds", "write_json", "profile_spec", "model_spec", "CONTROLLED_RETRIEVAL",
    "assert_artifact", "assert_checkpoint_reload", "assert_equal_memory_granularity", "assert_formula_gate",
    "assert_fp8_byte_ratio", "assert_frontier_coverage", "assert_full_cache_equivalence", "assert_nonfabricated",
    "assert_policy_respects_budget",
]
