# A100 Notebook 04 Resumable Locked-Test Guide

## Purpose

`04_a100_locked_test_serving.ipynb` performs the 40-record controlled locked test on an **NVIDIA A100**. This resumable version uses the same frozen Notebook 03 checkpoint, model revision, policy grid, budgets, and held-out partition as the non-resumable A100 notebook. It writes progress directly to Google Drive after every policy-budget measurement.

> Report the resulting measurements as **A100 locked-test evidence**, not as H100 evidence. The 8-bit cache estimate remains analytic and is not an observed A100 serving metric.

## Before starting

Upload the current `04_a100_locked_test_serving.ipynb` and current `kvcore_bundle.zip` to a fresh Google Colab Pro A100 runtime. You need both frozen Notebook 03 files:

```text
learned_kv_scorer.pt
selected_by_validation_only.json
```

Set the two variables to their actual Google Drive paths:

```python
CHECKPOINT_PATH = '/content/drive/.../learned_kv_scorer.pt'
SELECTION_RECORD_PATH = '/content/drive/.../selected_by_validation_only.json'
```

Notebook 03 stores the checkpoint path relative to its run root, whereas `CHECKPOINT_PATH` is an absolute Google Drive path. The notebook safely verifies that the selected relative path exactly matches the tail of the supplied Drive path, and refuses to run if they differ.

## Start or resume the locked test

1. Use **Runtime → Change runtime type → A100 GPU**, then start a clean runtime.
2. Run bootstrap, runtime, model-load, and preflight cells. Confirm `active_profile: a100` and `status: eligible_for_a100_run`.
3. Keep the following fixed for both first execution and any resume:

```python
RUN_A100_TEST = True
RESUME_LOCKED_TEST = True
RUN_ANALYTIC_8BIT_ACCOUNTING = True
RUN_SERVING_MEASUREMENT = False
LOCKED_TEST_RUN_ROOT = (
    '/content/drive/MyDrive/KV_Eviction_Capstone/02_notebook_runs/'
    '04_a100_locked_test_run/kv_eviction_04_a100_locked_test_serving'
)
```

4. Run the locked-test cell. It checkpoints the CSV and journal every completed policy-budget measurement, and prints a progress line after each completed test row.

## If Colab disconnects

Do not change the checkpoint, selection record, model revision, policy grid, budgets, test manifest, or `LOCKED_TEST_RUN_ROOT`. Start a new A100 session, upload the **same current bundle**, rerun setup/preflight, set the same variables, and rerun the locked-test cell. It will verify the immutable resume fingerprint and skip completed measurements automatically.

The persistent Drive directory contains:

```text
results/a100_controlled_locked_test.csv
results/a100_controlled_locked_test_progress.jsonl
manifests/a100_controlled_locked_test_resume_manifest.json
manifests/a100_locked_test_rows.json
```

A completed run has `status: complete` in `a100_controlled_locked_test_resume_manifest.json`. The final result copy at `results/a100_locked_test.csv` is written by the notebook cell after the evaluation returns. If a disconnect occurs before that final copy, Notebook 05 may instead read the durable `a100_controlled_locked_test.csv` or rebuild the final copy from the JSONL journal.

## Reporting boundary

Do not retrain, retune, or select a new scorer from locked-test results. Do not treat analytic 8-bit accounting as serving throughput/latency. Once the resume manifest is complete, copy the entire Drive run directory and use it as the Notebook 05 input.
