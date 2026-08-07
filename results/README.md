# Approved Result Summaries

This directory contains **small, traceable summaries only**. It intentionally excludes raw A100 journal rows, per-record traces, model checkpoints, KV-cache snapshots, and Drive paths.

| File | Meaning |
|---|---|
| `A100_LOCKED_TEST_EVIDENCE_AUDIT.md` | Provenance and completion audit for the A100 locked test. |
| `a100_forced_eviction_policy_summary.csv` | Observed cache-reduction and time summary for forced-eviction records. |
| `a100_context_summary.csv` | Context-stratified A100 summary. |
| `notebook03_validation_*` | RTX PRO 6000 validation-only scorer and policy summaries. |
| `figures/` | Report-ready plots generated from approved summary inputs. |

> The answer-quality proxy was inadequate even under full cache. Treat these files as cache-instrumentation and protocol-diagnostic evidence, not proof of learned-policy superiority.
