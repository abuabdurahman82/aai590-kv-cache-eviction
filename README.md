# Learned KV-Cache Eviction for Long-Context LLM Serving

**AAI-590 Capstone Project**  
**Team:** Faisal Abdul Gaffoor and Preetish Dash

## Project Objective

Long-context language-model applications such as coding assistants and agentic workflows repeatedly process conversation history, retrieved documents, files, and tool outputs. The model retains a Key–Value (KV) cache to avoid recomputing prior attention states, but that cache consumes GPU memory as context grows. This project evaluates fixed and learned policies that decide which cached information to retain under a controlled memory budget.

The project uses a controlled RULER-like single-needle retrieval corpus so that the effect of cache retention can be measured without mixing in unrelated task variability. The completed workflow preserves strict train/validation/test separation and labels every output as observed, gated, or diagnostic rather than fabricating unsupported metrics.

> **Evidence boundary.** The completed A100 locked test successfully recorded cache and timing behavior, but the answer-quality proxy was inadequate even for the full-cache reference. The repository therefore does not claim learned-policy superiority or deployment readiness from this run. See [`results/A100_LOCKED_TEST_EVIDENCE_AUDIT.md`](results/A100_LOCKED_TEST_EVIDENCE_AUDIT.md).

## Repository Structure

| Folder or file | Purpose |
|---|---|
| [`notebooks/`](notebooks) | Ordered Google Colab workflow from foundations through reporting. |
| [`kvcore/`](kvcore) | Reusable cache, accounting, corpus, evaluation, policy, profiling, scorer, and test code. |
| [`kvcore_bundle.zip`](kvcore_bundle.zip) | Colab upload bundle containing the same shared `kvcore` package. |
| [`scripts/`](scripts) | Suite-generation and validation utilities. |
| [`docs/`](docs) | Capstone report and reproducible Colab/A100 execution guidance. |
| [`results/`](results) | Approved observed-only summary tables and figures. |
| [`artifacts/`](artifacts) | Inventory and access policy for excluded large or sensitive runtime artifacts. |

## Notebook Execution Order

| Order | Notebook | Purpose | Hardware evidence status |
|---:|---|---|---|
| 00 | `00_foundations_environment_eda.ipynb` | Environment manifest, GQA-aware KV formula gate, 200-record corpus intake, and report-aligned EDA. | Observed on T4. |
| 01 | `01_evicting_cache_core_tests.ipynb` | Cache-core correctness, budget-respect, and decode-after-compaction tests. | Observed. |
| 02 | `02_t4_ruler_baseline_sweeps.ipynb` | Controlled-corpus intake and short-context profiling. | Infrastructure observed; broad T4 sweep intentionally gated for memory safety. |
| 03 | `03_rtx_pro6000_controlled_learned_eviction.ipynb` | Oracle labels, 28-input learned block-ranker training, validation-only selection, and validation evaluation. | Observed on RTX PRO 6000. |
| 04 | `04_a100_locked_test_serving.ipynb` | Resumable frozen-checkpoint A100 locked test with Drive journaling. | Observed on A100 80 GB. |
| 05 | `05_results_pareto_and_reporting.ipynb` | Artifact validation, A100-scoped aggregation, and reporting figures. | Observed-only reporting; Pareto interpretation withheld by quality guard. |

## Code Elements and Reproducibility

The repository includes the functional code required to reproduce the project pipeline:

1. The `kvcore` package implements GQA-aware cache accounting, cache cloning/compaction, baseline policies, learned-policy features, oracle label generation, scorer training/loading, controlled-corpus manifests, profiling, and resumable evaluation.
2. Notebook 00 validates the expected 200-record corpus (four context strata; 30/10/10 train/validation/test records per stratum) before later code can use it.
3. Notebook 03 selects the learned scorer using validation-only evidence; Notebook 04 consumes the frozen selected checkpoint and the preassigned 40-record test partition.
4. Notebook 04 records one JSONL journal entry per policy-budget measurement and protects resume compatibility with a fingerprinted manifest.
5. Notebook 05 rejects non-observed artifacts and scopes final A100 reporting to deduplicated, verified exports.

## Running in Google Colab

1. Open the desired notebook from `notebooks/` in Google Colab.
2. Upload the included `kvcore_bundle.zip` when the bootstrap cell requests it.
3. Select the required GPU tier for that notebook and run cells in order.
4. Read [`docs/COLAB_EXECUTION_PLAN.md`](docs/COLAB_EXECUTION_PLAN.md) before attempting Notebook 03 or 04.
5. For the A100 locked test, follow [`docs/A100_LOCKED_TEST_GUIDE.md`](docs/A100_LOCKED_TEST_GUIDE.md) exactly and preserve the Drive journal directory during a resume.

The pinned dependencies are recorded in [`requirements.txt`](requirements.txt). `bitsandbytes` and 4-bit loading are used for the T4 tier, while development and locked-test tiers use the hardware/model profiles defined in `kvcore/config.py`.

## Observed Results and Interpretation

The current A100 locked test completed 1,000 policy-budget measurements across 40 held-out records. The observed forced-eviction subset reduced physical KV payload by 29.45% on average. However, exact match was 0.0% and substring match was 2.1% across the locked run; the full-cache reference also showed inadequate retrieval quality. These outputs are retained as instrumentation and protocol-diagnostic evidence, not as a quality-preservation claim.

The RTX PRO 6000 development stage selected Trial 0 from validation-only evidence with NDCG@3 = 0.9780. This score describes the block-ranking development objective; it is not an A100 locked-test answer-quality result.

## Artifact and Data Policy

Large checkpoints, raw per-record JSONL journals, trace/cache dumps, locally mounted Drive folders, credentials, and browser-generated notebook outputs are intentionally excluded. [`artifacts/README.md`](artifacts/README.md) documents the excluded artifact classes and how reviewers can verify evidence without them.

## Team

- Faisal Abdul Gaffoor
- Preetish Dash
