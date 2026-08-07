# KV-Cache Eviction Capstone

## Step-by-Step Execution Plan and Google Drive Workflow

**Applies to:** `KV_Eviction_Colab_Suite_Styled_v3.zip` and the six ordered Colab notebooks.

**Purpose.** This plan provides one repeatable workflow for storing the project, running the notebooks in the correct order, preserving complete experiment artifacts, and preparing evidence for the final report. Follow the stages in order. The notebook suite produces valid results only when the relevant runtime, validation gate, and dataset/model provenance are present.

> **Evidence rule.** A result may be reported as an experimental measurement only when the relevant output row has `status = observed`. A `requires_larger_gpu_or_smaller_context` row, a skipped run, an error record, or an analytic FP8 calculation is valuable evidence, but it is **not** a measured benchmark result.

## 1. Recommended Google Drive Folder Structure

Create one top-level folder in **My Drive** named `KV_Eviction_Capstone`. It separates unmodified source files, complete notebook outputs, final reporting inputs, and submission-ready items. Keep every notebook output in its own folder; do not mix rows or checkpoints from different runs.

```
My Drive/
└── KV_Eviction_Capstone/
    ├── 00_source_suite/
    │   ├── notebooks/
    │   │   ├── 00_foundations_environment_eda.ipynb
    │   │   ├── 01_evicting_cache_core_tests.ipynb
    │   │   ├── 02_t4_ruler_baseline_sweeps.ipynb
    │   │   ├── 03_rtx_pro6000_longbench_learned_eviction.ipynb
    │   │   ├── 04_h100_locked_test_serving.ipynb
    │   │   └── 05_results_pareto_and_reporting.ipynb
    │   ├── shared_bundle/
    │   │   ├── kvcore_bundle.zip
    │   │   └── requirements.txt
    │   └── documentation/
    │       ├── README.md
    │       ├── KV_Eviction_Colab_Suite_v2_Step_by_Step_Guide.md
    │       ├── KV_Eviction_Colab_Execution_Plan_with_Google_Drive.md
    │       ├── Colab_Style_Alignment_Summary.md
    │       └── validation_report_v2.json
    ├── 01_experiment_log/
    │   ├── run_log.xlsx                         # or a Google Sheet
    │   └── decisions_and_deviations.md
    ├── 02_notebook_runs/
    │   ├── 00_foundations_run/
    │   ├── 01_cache_core_run/
    │   ├── 02_t4_ruler_run/
    │   ├── 03_rtx_longbench_run/
    │   ├── 04_h100_locked_test_run/
    │   └── 05_reporting_run/
    ├── 03_final_reporting_inputs/
    │   ├── 02_t4_ruler_export.zip
    │   ├── 03_rtx_longbench_export.zip
    │   ├── 04_h100_locked_test_export.zip
    │   └── selected_by_validation_only.json
    └── 04_submission/
        ├── final_tables/
        ├── final_figures/
        ├── final_report/
        └── github_release_copy/
```

### Create the folders manually in Google Drive

Open [Google Drive](https://drive.google.com/), select **New → New folder**, and create  `KV_Eviction_Capstone`. Open that folder, then create the four first-level folders: `00_source_suite`, `01_experiment_log`, `02_notebook_runs`, `03_final_reporting_inputs`, and `04_submission`. Continue opening each relevant folder and create its subfolders exactly as shown in the tree.

Google Drive does not need placeholder files when folders are created through the website. If you create the same structure on GitHub later, add a `README.md` file to otherwise empty folders because GitHub does not preserve empty directories.

| Drive location | What belongs there | Do not put here |
| --- | --- | --- |
| `00_source_suite/` | The unchanged restyled notebooks, the current `kvcore_bundle.zip`, validation report, and guides. | Generated benchmark outputs or edited checkpoints. |
| `01_experiment_log/` | A log of dates, runtime hardware, run owner, status, and deviations. | Raw run artifacts; those belong in `02_notebook_runs/`. |
| `02_notebook_runs/` | A complete exported `run_root` directory or ZIP from each notebook execution. | Mixed files from more than one notebook/run. |
| `03_final_reporting_inputs/` | Copies of the completed ZIP exports from notebooks 02–04 and the frozen checkpoint-selection record. | Draft or manually edited result CSVs. |
| `04_submission/` | Final approved figures, tables, report copies, and the final GitHub upload package. | Intermediate raw outputs. |

## 2. One-Time Setup

### Step 1 — Download, extract, and place the source suite

Download `KV_Eviction_Colab_Suite_Styled_v3.zip` to your computer and extract it. Upload the six `.ipynb` notebooks to `00_source_suite/notebooks/` in Google Drive. Upload `kvcore_bundle.zip` and `requirements.txt` to `00_source_suite/shared_bundle/`. Upload the README, validation report, execution guides, and style summary to `00_source_suite/documentation/`.

Before running anything, open `validation_report_v2.json` and confirm that the top-level field is:

```json
"pass": true
```

### Step 2 — Create the experiment log

Create `01_experiment_log/run_log.xlsx` or a Google Sheet with the following columns. Create a new row before and after every notebook execution.

| Run date/time | Notebook | Requested profile | Detected GPU | Active profile | Run status | Output folder/ZIP | Notes or deviation |
| --- | --- | --- | --- | --- | --- | --- | --- |
| YYYY-MM-DD HH:MM | `00_foundations...` | `t4` | e.g., Tesla T4 | `t4` | passed / observed / status | Drive link | Any error, restart, or config change |

Use `decisions_and_deviations.md` to document any approved departure from the written plan, such as a delayed H100 run or a separately labelled reduced-scale pilot. Do not overwrite an old record; add a dated entry.

### Step 3 — Understand the hardware decision points

The suite is intentionally sequenced from low-cost correctness checks to higher-cost evaluation. Select the hardware based on the notebook’s purpose, not merely on what happens to be available in Colab.

| Notebook | Primary purpose | Minimum appropriate runtime | Decision before continuing |
| --- | --- | --- | --- |
| 00 | Environment, GQA formula gate, EDA | CUDA GPU; T4 is sufficient | Formula gate passes. |
| 01 | Cache correctness and streaming decode | T4 or stronger CUDA GPU | Core checks pass. |
| 02 | Matched-budget baseline sweep | T4 or stronger; reduce scope only as a separately labelled pilot | Enable sweep only after memory preflight. |
| 03 | Label generation, training, validation-only selection | RTX PRO 6000 or equivalent VRAM | Save a frozen validation-selected checkpoint. |
| 04 | Locked 64k/128k-scale test | Actual H100 | Preflight confirms H100 eligibility and frozen checkpoint is available. |
| 05 | Aggregation and Pareto analysis | Any runtime capable of reading the artifacts | Upload only completed artifact ZIPs. |

> **Important for a 15 GB T4.** If the loaded model leaves little free GPU memory, keep `RUN_T4_SWEEP = False`. The output should be an honest status record rather than an out-of-memory crash or an invented baseline result.

## 3. Standard Colab Procedure for Every Notebook

### Step 4 — Start a clean runtime

Open [Google Colab](https://colab.research.google.com/), select **File → Upload notebook**, and upload the next notebook from `00_source_suite/notebooks/`. Select **Runtime → Change runtime type → GPU**. Use a clean runtime for each notebook, especially after updating the shared bundle or recovering from an out-of-memory error.

Google Colab runtimes are transient: files in `/content` and installed packages are not a durable experiment record. Export finished output before you disconnect or allow the runtime to expire.[1]

### Step 5 — Run the bootstrap cell and upload the bundle

Run the first code cell in the notebook. When it prompts for a shared package archive, upload the current file:

```
kvcore_bundle.zip
```

from `00_source_suite/shared_bundle/`. Do not reuse an older local copy of the bundle after a compatibility update. Wait for the pinned package installation and version information to print before continuing.

### Step 6 — Record the runtime manifest

Run the runtime/setup cell. It prints the requested profile, active profile, runtime status, `run_root`, GPU name, and available memory. Copy these details into the experiment log.

Continue only when the active profile and runtime capability are appropriate for the specific notebook. If the notebook writes a structured hardware requirement or fallback status, save it to Drive and record it honestly; do not relabel it as a target-tier execution.

### Step 7 — Run each notebook top-to-bottom

Do not execute cells out of order. Read the markdown before each code section. The restyled notebooks use **Step 1**, **Step 2**, and “What to Look For” markers to identify the purpose and expected outcome of each stage.

### Step 8 — Export the complete `run_root` after each notebook

After the notebook finishes, run this *new final cell* to copy the complete run directory to the correct Google Drive folder. This is safer than saving individual CSV files because it preserves manifests, figures, checks, errors, source metadata, and result tables together.

```python
from google.colab import drive
from pathlib import Path
import shutil

# Mount only after you have reviewed the notebook and are ready to copy your own artifacts.
drive.mount('/content/drive')

DRIVE_BASE = Path('/content/drive/MyDrive/KV_Eviction_Capstone/02_notebook_runs')
NOTEBOOK_DRIVE_FOLDER = '00_foundations_run'  # Change for each notebook.

target = DRIVE_BASE / NOTEBOOK_DRIVE_FOLDER / run_root.name
if target.exists():
    raise FileExistsError(f'Refusing to overwrite an existing run: {target}')

target.parent.mkdir(parents=True, exist_ok=True)
shutil.copytree(run_root, target)
print(f'Copied complete run to: {target}')
```

Change `NOTEBOOK_DRIVE_FOLDER` to the correct Drive folder for the current notebook:

| Notebook | `NOTEBOOK_DRIVE_FOLDER` value |
| --- | --- |
| 00 | `00_foundations_run` |
| 01 | `01_cache_core_run` |
| 02 | `02_t4_ruler_run` |
| 03 | `03_rtx_longbench_run` |
| 04 | `04_h100_locked_test_run` |
| 05 | `05_reporting_run` |

## 4. Notebook-by-Notebook Execution Plan

### Step 9 — Notebook 00: Foundations, environment, and EDA

Open `00_foundations_environment_eda.ipynb`. Leave the requested profile at `t4` unless an instructor-approved plan specifies another tier. Run all cells from top to bottom.

| Required check | Expected artifact | Continue only if |
| --- | --- | --- |
| GQA-aware formula gate | `checks/kv_formula_gate.json` | The gate reports a pass. |
| Deterministic EDA intake | `manifests/eda_rows.json` and `results/eda_prompt_inventory.csv` | Rows retain benchmark/task/row provenance. |
| Cache growth | `results/observed_cache_growth.csv` and `figures/eda_prompt_tokens.png` | Artifacts are present and clearly labelled as EDA/capacity planning. |

Copy the complete `run_root` to `02_notebook_runs/00_foundations_run/`. A formula-gate failure is a blocker: preserve the run, stop, and resolve the configuration issue before starting Notebook 01.

### Step 10 — Notebook 01: Cache-core correctness tests

Open `01_evicting_cache_core_tests.ipynb` in a clean GPU runtime. Run it top-to-bottom. This notebook has no experimental performance switch; it is a required correctness gate.

Verify these artifacts before proceeding:

```
results/policy_budget_checks.csv
results/streaming_decode_after_compaction.csv
checks/cache_core_complete.json
```

Every compressed policy must respect its cache budget, logical positions must remain monotonic, full-cache behavior must satisfy the equivalence check, and the decode trace must continue after compaction. Copy the complete `run_root` to `02_notebook_runs/01_cache_core_run/`.

### Step 11 — Notebook 02: T4 RULER baseline sweep

Open `02_t4_ruler_baseline_sweeps.ipynb` on a T4 or stronger GPU. Run all cells with the default safety guard first:

```python
RUN_T4_SWEEP = False
```

Inspect the runtime manifest, model load, RULER row manifest, and short attention-profile artifacts. Only if the GPU is healthy and the memory preflight is satisfactory should you change the single guard line to:

```python
RUN_T4_SWEEP = True
```

Then rerun the guard-definition cell and all downstream evaluation/reporting cells. Do not change the policy list, seeds, or budgets after the sweep starts.

| Expected output | Interpretation |
| --- | --- |
| `results/t4_ruler_baselines.csv` | Valid baseline rows only when `status = observed`; hardware-limit rows remain auditable but are not measurements. |
| `manifests/t4_ruler_rows.json` | Exact RULER rows used. |
| `attention_profiles/t4_profile_protocol.json` | Short-pass attention-profile metadata. |
| `figures/t4_quality_budget.png` | Created only if observed rows exist. |

Copy the full `run_root` to `02_notebook_runs/02_t4_ruler_run/`. Also create a ZIP copy for notebook 05:

```python
import shutil
from pathlib import Path
from google.colab import drive

drive.mount('/content/drive')
zip_path = shutil.make_archive(str(run_root), 'zip', root_dir=str(run_root))
target = Path('/content/drive/MyDrive/KV_Eviction_Capstone/03_final_reporting_inputs/02_t4_ruler_export.zip')
shutil.copy2(zip_path, target)
print(target)
```

### Step 12 — Notebook 03: RTX LongBench development and validation-only selection

Open `03_rtx_pro6000_longbench_learned_eviction.ipynb` on an RTX PRO 6000 or a genuinely equivalent GPU with sufficient memory. Begin with all execution gates set to `False`, run the setup and partition cells, and review the deterministic 60/20/20 document-level split.

Run each gated stage separately and preserve its output before enabling the next stage.

| Stage | Single setting to enable | Required evidence |
| --- | --- | --- |
| Oracle labels | `RUN_LABEL_GENERATION = True` | `labels/oracle_block_labels.pt` and attention/profile metadata. |
| Training and model selection | `RUN_TRAINING = True` | `results/validation_model_selection.csv` and `checkpoints/selected_by_validation_only.json`. |
| Validation-only comparison | `RUN_VALIDATION_EVAL = True` | `results/rtx_validation_only.csv`. |

The selected checkpoint must be chosen using validation information only. Once `selected_by_validation_only.json` exists, treat it as frozen. Do not use the later H100 test output to alter the checkpoint, dropout, learning rate, budgets, or selection rule.

Copy the complete run directory to `02_notebook_runs/03_rtx_longbench_run/`. Create two reporting hand-off files:

```
03_final_reporting_inputs/03_rtx_longbench_export.zip
03_final_reporting_inputs/selected_by_validation_only.json
```

### Step 13 — Notebook 04: H100 locked test

Open `04_h100_locked_test_serving.ipynb` only on an actual H100 if you intend to claim H100-scale 64k/128k results. Copy the exact selected checkpoint from Notebook 03 into the new runtime and set `CHECKPOINT_PATH` to that file.

Run the preflight and configuration-discovery cells before enabling the locked test. Continue only if the preflight identifies the active profile as H100 and confirms that verified pinned RULER configurations match the requested scale.

```python
RUN_H100_TEST = True
```

Run the remaining locked-test cells. Save the complete `run_root` to `02_notebook_runs/04_h100_locked_test_run/` and make a reporting copy at:

```
03_final_reporting_inputs/04_h100_locked_test_export.zip
```

Treat FP8 accounting and serving outputs correctly. `RUN_FP8_ANALYTIC_ACCOUNTING = True` yields analytic capacity estimates unless an observed FP8 cache backend is present. Keep `RUN_SERVING_MEASUREMENT = False` unless you have separately provisioned a serving harness and a defined measurement protocol.

### Step 14 — Notebook 05: Final reporting and Pareto analysis

Open `05_results_pareto_and_reporting.ipynb` in any suitable runtime. Its `artifacts/` directory should contain separate extractions of the completed ZIP files from notebooks 02–04.

Upload these Drive files when prompted:

```
03_final_reporting_inputs/02_t4_ruler_export.zip
03_final_reporting_inputs/03_rtx_longbench_export.zip
03_final_reporting_inputs/04_h100_locked_test_export.zip
```

Run the artifact-discovery cell, then inspect `aggregated_results.csv` before creating any charts. The notebook generates quality, memory, latency, and Pareto outputs only when the required values are observed and comparable.

Save the complete reporting run to `02_notebook_runs/05_reporting_run/`. Copy approved tables and figures—not draft raw files—to:

```
04_submission/final_tables/
04_submission/final_figures/
```

## 5. Final Reporting Rules

Use the following labels consistently in the capstone report, GitHub README, and presentation.

| Label | Meaning | May it be plotted as benchmark performance? |
| --- | --- | --- |
| **Observed** | The notebook executed and recorded the metric with provenance. | Yes, if comparison conditions match. |
| **Analytic** | A formula-derived capacity estimate, such as FP8 payload accounting. | No; keep separate from empirical curves. |
| **Hardware/status** | The intended runtime or context could not be executed. | No; retain in tables and limitations. |
| **Error** | A failed execution requiring diagnosis. | No. |

Before final submission, verify that every reported table or figure can be traced to a Drive artifact folder, an exact manifest, a model/dataset revision, an active GPU profile, a budget, a seed, a partition, and a measurement class.

## 6. Final Completion Checklist

| Checkpoint | Completion condition |
| --- | --- |
| Source suite secured | The restyled notebooks and current `kvcore_bundle.zip` are stored unchanged in `00_source_suite/`. |
| Logging started | Every execution has a row in `01_experiment_log/run_log.xlsx` or the Google Sheet. |
| Formula verified | Notebook 00 gate passed and its artifact is saved. |
| Cache verified | Notebook 01 pass record is saved. |
| T4 baseline handled honestly | Notebook 02 produced observed rows or explicit hardware/status rows. |
| No development/test leakage | Notebook 03 checkpoint was selected only by validation results. |
| Locked test preserved | Notebook 04 used the frozen selected checkpoint and retained its preflight record. |
| Artifacts separated | ZIP exports from notebooks 02, 03, and 04 are separate under `03_final_reporting_inputs/`. |
| Report is evidence-based | Notebook 05 uses only observed compatible rows for figures and Pareto analysis. |
| Submission is clean | Only approved tables, figures, report files, and intended GitHub items are copied to `04_submission/`. |

## References

[1]: https://research.google.com/colaboratory/faq.html "Google Colaboratory, Frequently Asked Questions"

[2]: https://huggingface.co/docs/transformers/en/models "Hugging Face, Transformers Documentation: Loading Models"

[3]: https://huggingface.co/docs/datasets/en/loading "Hugging Face, Datasets Documentation: Loading"

