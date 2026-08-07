from __future__ import annotations

from pathlib import Path
import json
import shutil
import textwrap
import zipfile

import nbformat as nbf

ROOT = Path('/home/ubuntu/work/kv_colab_v2/suite')
NOTEBOOK_DIR = ROOT / 'notebooks'
CORE_DIR = ROOT / 'kvcore'
NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)


def clean(text: str) -> str:
    return textwrap.dedent(text).strip() + '\n'


def md(text: str):
    return nbf.v4.new_markdown_cell(clean(text))


def code(text: str):
    return nbf.v4.new_code_cell(clean(text))


BOOTSTRAP = r'''
# Colab bootstrap: install pinned dependencies and unpack the shared core.
# Upload kvcore_bundle.zip supplied with this notebook suite if kvcore is not present.
from pathlib import Path
import sys, subprocess, zipfile

PINNED = [
    'transformers==4.56.2', 'accelerate==1.10.1', 'datasets==4.0.0',
    'huggingface_hub==0.34.4', 'bitsandbytes==0.47.0', 'safetensors==0.6.2',
    'sentencepiece==0.2.1', 'scipy==1.16.1', 'matplotlib==3.10.6',
    'seaborn==0.13.2', 'pandas==2.3.2',
]
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', *PINNED])

if not Path('kvcore').exists():
    try:
        from google.colab import files
        print('Upload kvcore_bundle.zip from the delivered suite.')
        uploaded = files.upload()
        archive = next((Path(name) for name in uploaded if name.endswith('.zip')), None)
        if archive is None:
            raise FileNotFoundError('Please upload kvcore_bundle.zip.')
        with zipfile.ZipFile(archive) as zf:
            zf.extractall('.')
    except ImportError as error:
        raise RuntimeError('Run in Google Colab or place the kvcore directory beside this notebook.') from error

sys.path.insert(0, str(Path('.').resolve()))
from kvcore import *
from kvcore.config import BENCHMARKS, MODELS, POLICY_DEFAULTS, PROFILES, SUITE_VERSION
print({'suite_version': SUITE_VERSION, 'ruler_revision': BENCHMARKS['ruler']['revision'], 'longbench_revision': BENCHMARKS['longbench']['revision']})
'''

RUNTIME = r'''
import json, os, platform, time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch

REQUESTED_PROFILE = REQUESTED_PROFILE  # defined by the notebook title cell
NOTEBOOK_ID = globals().get('NOTEBOOK_ID', 'runtime')
set_all_seeds(590)
profile, runtime_status = select_profile(REQUESTED_PROFILE)
run_root = ensure_run_root(f'kv_eviction_{NOTEBOOK_ID}')
manifest = run_manifest(
    run_root,
    notebook=NOTEBOOK_ID,
    requested_profile=REQUESTED_PROFILE,
    active_profile=profile.name,
    model=model_spec(profile.model_tier),
    runtime_status=runtime_status,
)
print(json.dumps({'notebook': NOTEBOOK_ID, 'requested_profile': REQUESTED_PROFILE, 'active_profile': profile.name, 'runtime_status': runtime_status, 'run_root': str(run_root)}, indent=2))
if not torch.cuda.is_available():
    raise RuntimeError('A CUDA GPU runtime is required for model execution. In Colab: Runtime > Change runtime type > GPU.')
print('GPU:', torch.cuda.get_device_name(0), 'VRAM GiB:', round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2))
'''

MODEL_LOAD = r'''
# Model and tokenizer are always loaded at immutable Hub revisions from kvcore.config.
# If the runtime is smaller than the requested tier, runtime_status records the fallback.
model, tokenizer = load_model_and_tokenizer(model_spec(profile.model_tier), load_mode=profile.load_mode, attn_implementation='sdpa')
print({'model': model.config._name_or_path, 'requested_profile': REQUESTED_PROFILE, 'active_profile': profile.name, 'load_mode': profile.load_mode})
'''

POLICY_SPECS = r'''
# All policies use the same per-example cache budget. Full cache is the uncompressed reference.
POLICY_SPECS = [
    {'type': 'full'}, {'type': 'fifo', 'min_recent': 32},
    {'type': 'random', 'seed': 590, 'min_recent': 32}, {'type': 'uniform', 'min_recent': 32},
    {'type': 'sink_recent', 'sink_tokens': 4, 'min_recent': 32},
    {'type': 'attention_topk', 'sink_tokens': 4, 'min_recent': 32},
    {'type': 'h2o', 'recent_fraction': 0.50, 'sink_tokens': 4, 'min_recent': 32},
]
'''

ARTIFACT_NOTE = r'''
# Result tables only contain observed rows, explicit errors, or explicit hardware requirement statuses.
# No cell manufactures latency, memory, or quality values. Preserve run_root for the Results section.
'''


def base_metadata(title: str):
    return {
        'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
        'language_info': {'name': 'python', 'version': '3.11'},
        'colab': {'name': title, 'provenance': [], 'gpuType': 'T4'},
    }


def write_notebook(filename: str, title: str, cells):
    nb = nbf.v4.new_notebook()
    nb['metadata'] = base_metadata(title)
    nb['cells'] = cells
    nbf.write(nb, NOTEBOOK_DIR / filename)


# 00 Foundations
cells00 = [
    md('''# 🚀 00: Foundations, Environment, and Benchmark Intake

*Part of the KV-Cache Eviction Capstone Series*
*Estimated time: 10–15 minutes*

---

Welcome! In this first notebook, we are going to establish an auditable, reproducible environment for the entire KV-cache eviction experiment. Before we measure performance or train policies, we must ensure our arithmetic is correct and our data is stable.

This notebook performs **Exploratory Data Analysis (EDA) only** on the canonical controlled retrieval corpus. It does not report eviction-quality results. You will need a GPU runtime; the code automatically adapts to a T4, RTX PRO 6000, or H100 and records the actual hardware profile used.

### Experiment Roadmap
```text
[GPU preflight] → [GQA formula gate] → [200-record controlled corpus] → [EDA artifacts]
```'''),
    md('''## Step 1: Why Does This Matter?

If our cache-memory arithmetic is wrong, our entire budget-versus-quality comparison is invalid. If our benchmark datasets shift under us, our results cannot be reproduced. 

This step locks down the environment. Our **acceptance criteria** are strict:
1. Pinned package versions, model revisions, and benchmark revisions must be explicitly recorded.
2. A measured cache payload must match our closed-form GQA-aware formula exactly (within a small structural tolerance).
3. The controlled corpus must contain exactly 200 records: 50 at each planned length, with a 30/10/10 train/validation/test split inside every length stratum.
4. Every record must retain prompt, answer, answer-location, seed, tokenizer-count, and provenance metadata.
5. No model quality, latency, or compression outcomes may be fabricated—if a test cannot run, it must be recorded as skipped.'''),
    code("NOTEBOOK_ID = '00_foundations_environment_eda'\nREQUESTED_PROFILE = 't4'\n" + BOOTSTRAP),
    code(RUNTIME),
    code(MODEL_LOAD),
    md('''## Step 2: GQA-Aware KV Accounting

Modern models use Grouped-Query Attention (GQA), which means there are fewer Key and Value heads than Query heads. Our cache formula must account for this exactly:

**Payload = layers × batch × retained tokens × KV heads × head dimension × 2 (K and V) × bytes per element**

Let us run a probe sequence through the model and verify that our formula matches the physical cache allocation.'''),
    code(r'''
from kvcore.accounting import formula_for_model, formula_gate
from kvcore.cache import prefill

probe = tokenizer('KV-cache accounting probe. ' * 32, return_tensors='pt').input_ids.to(next(model.parameters()).device)
cache, _ = prefill(model, probe, chunk_size=min(256, probe.shape[-1]), output_attentions=False)
expected = formula_for_model(model.config, retained_tokens=cache.physical_length, batch_size=1, dtype=next(model.parameters()).dtype)
gate = formula_gate(cache, expected['formula_bytes'], tolerance=0.02)
assert_formula_gate(gate)
write_json(run_root / 'checks' / 'kv_formula_gate.json', {'expected': expected, 'gate': gate})
print(json.dumps({'model_kv_spec': expected, 'gate': gate}, indent=2))
'''),
    md('''## Step 3: Controlled Retrieval Corpus and EDA

The current capstone snapshot uses one canonical **RULER-like needle-in-a-haystack corpus**. It contains 200 deterministic single-needle retrieval records across planned prompt lengths of 1,024, 2,048, 4,096, and 8,192 tokens. Every length stratum contains 30 train, 10 validation, and 10 held-out test records.

LongBench remains a planned **external-validity extension**. It is not loaded or reported as part of the current controlled-corpus EDA.'''),
    code(r'''
# Canonical controlled RULER-like corpus: 200 records, four length strata, exact 30/10/10 split per stratum.
rows = load_controlled_retrieval(tokenizer=tokenizer, seed=590)
corpus_audit = audit_controlled_retrieval_corpus(rows)
assert corpus_audit['passed'], corpus_audit
write_json(run_root / 'checks' / 'controlled_corpus_quality_audit.json', corpus_audit)
build_benchmark_manifest(
    rows, run_root / 'manifests' / 'controlled_retrieval_rows.json',
    benchmark_name='controlled_retrieval', seed=590,
    tokenizer_id=model_spec(profile.model_tier)['tokenizer_id'],
    tokenizer_revision=model_spec(profile.model_tier)['tokenizer_revision'],
    model_id=model_spec(profile.model_tier)['model_id'],
    model_revision=model_spec(profile.model_tier)['revision'],
)
eda = pd.DataFrame([{
    'benchmark': row['benchmark'], 'task': row['task'], 'row_id': row['row_id'],
    'partition': row['partition'], 'planned_context_length': row['planned_context_length'],
    'measured_context_length': row['measured_context_length'],
    'answer_location_band': row['answer_location_band'], 'answer_location_pct': row['answer_location_pct'],
    'characters': len(row['prompt']), 'answer_count': len(row.get('answers', [])),
    'tokens': int(row['measured_context_length']),
} for row in rows])
eda.to_csv(run_root / 'results' / 'controlled_retrieval_inventory.csv', index=False)
display(eda.groupby(['planned_context_length', 'partition']).agg(rows=('row_id','count'), median_tokens=('tokens','median'), max_tokens=('tokens','max')).reset_index())
plt.figure(figsize=(9,4)); sns.countplot(data=eda, x='planned_context_length', hue='partition', order=sorted(eda['planned_context_length'].unique())); plt.title('Controlled retrieval records by planned length and partition'); plt.tight_layout(); plt.savefig(run_root / 'figures' / 'eda_records_by_length_partition.png', dpi=180); plt.show()
plt.figure(figsize=(9,4)); sns.stripplot(data=eda, x='planned_context_length', y='answer_location_pct', hue='answer_location_band', dodge=True, alpha=0.65); plt.title('Hidden-answer location coverage by planned context length'); plt.tight_layout(); plt.savefig(run_root / 'figures' / 'eda_answer_position_coverage.png', dpi=180); plt.show()
'''),
    md('''## Step 4: Measured Cache-Growth Trace

Finally, let us observe how the physical cache payload grows as the token prefix increases. This short trace supports our capacity planning for the larger GPUs later.

### What to Look For
Verify that the `payload_bytes` increases linearly with the `physical_length`. This confirms our memory footprint behaves as expected before we start evicting tokens.'''),
    code(r'''
lengths = [min(n, probe.shape[-1]) for n in (64, 128, 256, probe.shape[-1])]
growth = []
for length in sorted(set(lengths)):
    c, _ = prefill(model, probe[:, :length], chunk_size=min(256, length), output_attentions=False)
    growth.append({'tokens': int(length), 'payload_bytes': int(measured_cache_bytes(c)), 'physical_length': int(c.physical_length)})
pd.DataFrame(growth).to_csv(run_root / 'results' / 'observed_cache_growth.csv', index=False)
display(pd.DataFrame(growth))
'''),
    code("print('Foundations completed. Artifacts:', run_root)\n" + ARTIFACT_NOTE),
]
write_notebook('00_foundations_environment_eda.ipynb', 'KV Eviction — 00 Foundations and EDA', cells00)

# 01 Cache core
cells01 = [
    md('''# 🚀 01: EvictingCache Core, Compaction, and Streaming Decode

*Part of the KV-Cache Eviction Capstone Series*
*Estimated time: 10–15 minutes*

---

Welcome to the engine room. In this notebook, we validate the custom `EvictingCache` implementation before we trust it with benchmark sweeps. 

We will test logical-to-physical position mappings, memory compaction, policy budget enforcement, full-cache mathematical equivalence, and streaming decode after a compaction boundary.

### Experiment Roadmap
```text
[Prompt] → [EvictingCache] → [Policy] → [Compaction] → [Streaming decode]
```'''),
    md('''## Step 1: Why Does This Matter?

If we evict tokens but corrupt the position IDs, the model's attention mechanism will hallucinate. If we exceed the memory budget, the system will crash in production. 

We must prove the mechanism works. Our **required assertions** are:
- Full-cache execution matches an ordinary dynamic-cache reference exactly.
- Every compressed policy keeps the physical cache size strictly at or below its budget.
- Logical positions remain strictly monotonic after physical compaction.
- A streaming decode step successfully continues generation even after an eviction boundary has shifted the physical memory.'''),
    code("NOTEBOOK_ID = '01_evicting_cache_core_tests'\nREQUESTED_PROFILE = 't4'\n" + BOOTSTRAP),
    code(RUNTIME),
    code(MODEL_LOAD),
    code(POLICY_SPECS),
    code(r'''
from kvcore.cache import prefill, greedy_decode_step
from kvcore.policies import CacheState, policy_from_spec

prompt_ids = tokenizer('A cache should retain important facts while preserving the newest context. ' * 60, return_tensors='pt').input_ids.to(next(model.parameters()).device)
cache, out = prefill(model, prompt_ids, chunk_size=min(256, prompt_ids.shape[-1]), output_attentions=False)
state = CacheState(positions=cache.physical_positions, step=1)
print({'logical_length': cache.logical_length, 'physical_length': cache.physical_length, 'positions_tail': cache.physical_positions[-8:].detach().cpu().tolist()})
'''),
    md('''## Step 2: Cache Compaction and Policy-Budget Tests

Let us apply every configured eviction policy to a test prompt and verify that the resulting physical cache length respects the requested budget, while keeping logical positions intact.'''),
    code(r'''
checks = []
for spec in POLICY_SPECS:
    policy = policy_from_spec(spec)
    test_cache, _ = prefill(model, prompt_ids, chunk_size=min(256, prompt_ids.shape[-1]), output_attentions=False)
    local_state = CacheState(positions=test_cache.physical_positions, step=1)
    budget = min(128, max(64, test_cache.physical_length // 2))
    info = test_cache.apply_policy(policy, local_state, budget=budget, reason='notebook_01_unit_test')
    assert_policy_respects_budget(test_cache, budget, policy.name)
    positions = test_cache.physical_positions.detach().cpu().tolist()
    assert positions == sorted(positions), f'nonmonotonic logical positions for {policy.name}'
    checks.append({'policy': policy.name, 'budget': budget, 'physical_length': test_cache.physical_length, **info})
checks_df = pd.DataFrame(checks); checks_df.to_csv(run_root / 'results' / 'policy_budget_checks.csv', index=False); display(checks_df)
'''),
    md('''## Step 3: Full-Cache Equivalence and Post-Compaction Decode

Finally, we must prove that our custom cache does not alter the model's output when no tokens are evicted, and that it can successfully stream new tokens *after* an eviction event.

### What to Look For
This test uses a deterministic greedy next-token comparison. If it passes without an assertion error, our cache is mathematically sound and ready for the T4 benchmark sweeps.'''),
    code(r'''
# Reference and full policy must expose the same greedy next token before any compaction.
reference_cache, reference_out = prefill(model, prompt_ids[:, :min(256, prompt_ids.shape[-1])], chunk_size=128, output_attentions=False)
full_cache, full_out = prefill(model, prompt_ids[:, :min(256, prompt_ids.shape[-1])], chunk_size=128, output_attentions=False)
assert_full_cache_equivalence(reference_out.logits[:, -1, :], full_out.logits[:, -1, :])

# Streaming decode after a forced compaction boundary.
compact_cache, compact_out = prefill(model, prompt_ids[:, :min(384, prompt_ids.shape[-1])], chunk_size=128, output_attentions=False)
compact_policy = policy_from_spec({'type': 'sink_recent', 'sink_tokens': 4, 'min_recent': 32})
compact_cache.apply_policy(compact_policy, CacheState(positions=compact_cache.physical_positions, step=1), budget=min(128, compact_cache.physical_length), reason='pre_stream')
token = compact_out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
trace = []
for step in range(3):
    compact_cache, token, logits = greedy_decode_step(model, compact_cache, token, logical_position=compact_cache.logical_length, output_attentions=False)
    trace.append({'step': step, 'logical_length': compact_cache.logical_length, 'physical_length': compact_cache.physical_length, 'token_id': int(token.item())})
assert len(trace) == 3
pd.DataFrame(trace).to_csv(run_root / 'results' / 'streaming_decode_after_compaction.csv', index=False)
display(pd.DataFrame(trace))
'''),
    code("write_json(run_root / 'checks' / 'cache_core_complete.json', {'status': 'passed', 'checks': checks})\nprint('Cache core checks passed:', run_root)"),
]
write_notebook('01_evicting_cache_core_tests.ipynb', 'KV Eviction — 01 Cache Core Tests', cells01)

# 02 T4 RULER
cells02 = [
    md('''# 🚀 02: T4 Controlled-Retrieval Baseline-Policy Sweeps

*Part of the KV-Cache Eviction Capstone Series*
*Estimated time: 45–180 minutes (if executed)*

---

Welcome to the first evaluation phase. This notebook is designed for a standard NVIDIA T4 GPU. It uses the canonical controlled retrieval corpus at strictly matched cache budgets.

For the baseline development sweep, the notebook selects 25 deterministic non-test records from each planned length stratum, for 100 records total. The 40 held-out test records are reserved for Notebook 04.

### Experiment Roadmap
```text
[Controlled rows] → [Short attention profile] → [Matched-budget sweep] → [Observed CSV]
```'''),
    md('''## Step 1: Why Does This Matter?

Before we try to invent a learned eviction policy, we need to know exactly how well the existing heuristics perform. If a simple "keep the first 4 tokens and the most recent 124 tokens" rule solves the problem, we do not need a complex model.

**Runtime Contract:**
The default T4 profile targets the 1.5B model in 4-bit mode. If your GPU runs out of memory, the code will catch the error and record a **requires larger GPU** status. It will never fabricate a measurement.'''),
    code("NOTEBOOK_ID = '02_t4_ruler_baseline_sweeps'\nREQUESTED_PROFILE = 't4'\n" + BOOTSTRAP),
    code(RUNTIME),
    code(MODEL_LOAD),
    code(POLICY_SPECS),
    code(r'''
RUN_T4_SWEEP = False  # Set True only after checking GPU memory, disk, and desired allocation.
all_controlled_rows = load_controlled_retrieval(tokenizer=tokenizer, seed=590)
corpus_audit = audit_controlled_retrieval_corpus(all_controlled_rows)
assert corpus_audit['passed'], corpus_audit
# 25 deterministic train/validation rows per length = 100 baseline-development records; held-out test remains unused.
rows = load_controlled_retrieval(tokenizer=tokenizer, seed=590, max_examples=25, partitions=('train', 'validation'))
assert len(rows) == 100 and all(row['partition'] in {'train', 'validation'} for row in rows)
build_benchmark_manifest(rows, run_root / 'manifests' / 't4_controlled_baseline_rows.json', benchmark_name='t4_controlled_baseline', seed=590)
print(pd.DataFrame(rows).groupby(['planned_context_length','partition']).size())
'''),
    md('''## Step 2: Two-Pass Attention Protocol

Extracting attention matrices from every layer for every token is incredibly memory-intensive. To survive on a T4, we use a two-pass discipline: a short, eager pass collects attention statistics for the Top-K policies, and the main evaluation pass uses memory-efficient SDPA without returning attention weights.'''),
    code(r'''
profile_row = rows[0]
profile_ids = tokenise_prompt(tokenizer, profile_row['prompt'], min(profile.attention_profile_context_cap, profile.max_context_tokens), next(model.parameters()).device)
profile_meta = profile_short_context(model, profile_ids, max_tokens=min(profile_ids.shape[-1], profile.attention_profile_context_cap), run_root=run_root, tag='t4_short_profile', prefill_chunk_tokens=profile.prefill_chunk_tokens)
attention_profile = load_profile(profile_meta['profile_path'], device=next(model.parameters()).device) if profile_meta['attention_available'] else None
write_json(run_root / 'attention_profiles' / 't4_profile_protocol.json', profile_meta)
print(profile_meta)
'''),
    md('''## Step 3: Matched-Budget Baseline Sweep

This is the main event. We will evaluate every policy using the exact same budget values to ensure a fair comparison. 

**Before You Run:**
Set `RUN_T4_SWEEP = True` only if you have confirmed your GPU has sufficient memory (at least 15GB free). Otherwise, leave it `False` to record a safe "not run" status.'''),
    code(r'''
if RUN_T4_SWEEP:
    records = evaluate_rows(
        model, tokenizer, rows, POLICY_SPECS, profile.budgets, profile.max_context_tokens,
        profile.decode_tokens, profile.prefill_chunk_tokens, run_root, profile=attention_profile,
        artifact_stem='t4_controlled_retrieval_baselines',
    )
else:
    records = [requires_gpu_record('T4 or larger', 'Set RUN_T4_SWEEP=True', 'Evaluation deliberately disabled until user confirms runtime allocation.')]
pd.DataFrame(records).to_csv(run_root / 'results' / 't4_controlled_retrieval_baselines.csv', index=False)
display(pd.DataFrame(records).head())
'''),
    md('''## Step 4: Results-Ready Visualization

Let us visualize the quality-versus-budget trade-off for the heuristics.

### What to Look For
The plot is generated *only* from successfully observed rows. If your run recorded hardware limits, those points are safely excluded from the curves but preserved in the CSV artifact.

**Next step:** Save the `run_root` directory. We will aggregate these baseline results in Notebook 05.'''),
    code(r'''
df = pd.read_csv(run_root / 'results' / 't4_controlled_retrieval_baselines.csv')
observed = df[df.get('status', pd.Series(dtype=str)).eq('observed')] if 'status' in df else pd.DataFrame()
if not observed.empty:
    plt.figure(figsize=(9,4)); sns.lineplot(data=observed, x='budget', y='substring_match', hue='policy', marker='o'); plt.title('Observed controlled-retrieval quality proxy versus cache budget — T4'); plt.tight_layout(); plt.savefig(run_root / 'figures' / 't4_quality_budget.png', dpi=180); plt.show()
else:
    print('No observed sweep rows yet; see the explicit status table instead of a fabricated curve.')
'''),
]
write_notebook('02_t4_ruler_baseline_sweeps.ipynb', 'KV Eviction — 02 T4 RULER Baselines', cells02)

# 03 RTX learned
cells03 = [
    md('''# 🚀 03: Controlled Retrieval, Oracle Labels, and Learned Eviction

*Part of the KV-Cache Eviction Capstone Series*
*Estimated time: 30–60 minutes*

---

This notebook is the core development workflow for our learned eviction policy. It uses the canonical controlled retrieval corpus, generates oracle importance labels using attention and ablation, trains a 28-input MLP scorer, and selects the best checkpoint using strictly held-out validation data.

LongBench is reserved for a future external-validity extension and is not used for the current training, validation, or locked-test claims.

### Experiment Roadmap
```text
[Controlled corpus] → [Oracle labels] → [Train] → [Validation] → [Frozen checkpoint]
```'''),
    md('''## Step 1: Why Does This Matter?

Machine learning is notoriously vulnerable to data leakage. If our learned policy sees the test data during training, or if we select our checkpoint based on test-set performance, our results will be an illusion.

**Learning and Leakage Controls:**
- We assign examples to deterministic 60/20/20 train/validation/test partitions.
- We fit feature normalization **on train only**.
- We use validation NDCG@3 for checkpoint selection.
- We **do not run** benchmark test rows here. They are locked away until Notebook 04.'''),
    code("NOTEBOOK_ID = '03_rtx_pro6000_controlled_learned_eviction'\nREQUESTED_PROFILE = 'rtx_pro_6000'\n" + BOOTSTRAP),
    code(RUNTIME),
    code(MODEL_LOAD),
    code(r'''
RUN_LABEL_GENERATION = False
RUN_TRAINING = False
RUN_VALIDATION_EVAL = False

all_rows = load_controlled_retrieval(tokenizer=tokenizer, seed=590)
corpus_audit = audit_controlled_retrieval_corpus(all_rows)
assert corpus_audit['passed'], corpus_audit
partitions = pd.DataFrame(all_rows)[['benchmark','task','row_id','partition','planned_context_length','answer_location_band']]
partitions.to_csv(run_root / 'manifests' / 'controlled_retrieval_partitions.csv', index=False)
print(partitions.groupby(['planned_context_length','partition']).size().unstack(fill_value=0))
train_rows = [row for row in all_rows if row['partition'] == 'train']
validation_rows = [row for row in all_rows if row['partition'] == 'validation']
test_rows = [row for row in all_rows if row['partition'] == 'test']
assert len(train_rows) == 120 and len(validation_rows) == 40 and len(test_rows) == 40
'''),
    md('''## Step 2: Oracle Labels from Model Evidence

How do we know which tokens are actually important? We measure it. We generate labels by combining attention evidence with the actual increase in loss (Negative Log-Likelihood) caused by ablating (removing) that block from the cache. These are self-generated model labels.'''),
    code(r'''
import gc

from kvcore.labels import label_blocks, save_labels
from kvcore.evaluation import answer_ids, tokenise_prompt

label_rows = []
if RUN_LABEL_GENERATION:
    # Create labels from both development partitions. The locked test partition
    # is deliberately excluded so validation-only checkpoint selection remains valid.
    LABEL_TRAIN_PER_LENGTH = 3
    LABEL_VALIDATION_PER_LENGTH = 2
    # Eager attention is used only to generate short-context oracle features.
    # A fixed cap bounds the peak quadratic attention allocation per document.
    LABEL_PROFILE_CONTEXT_CAP = 1024
    LABEL_MAX_COUNTERFACTUAL_BLOCKS = 4
    label_inputs = []
    label_memory_audit = []
    for source_rows, partition_name, per_length in (
        (train_rows, 'train', LABEL_TRAIN_PER_LENGTH),
        (validation_rows, 'validation', LABEL_VALIDATION_PER_LENGTH),
    ):
        for planned_length in CONTROLLED_RETRIEVAL['context_lengths']:
            candidates = [row for row in source_rows if row['planned_context_length'] == planned_length]
            chosen = candidates[:per_length]
            if len(chosen) != per_length:
                raise RuntimeError(f'Expected {per_length} {partition_name} records at length {planned_length}, found {len(chosen)}.')
            label_inputs.extend(chosen)
    label_input_summary = pd.DataFrame(label_inputs).groupby(['planned_context_length', 'partition']).size()
    print('Label-document selection by length and partition:\n', label_input_summary)
    assert sum(row['partition'] == 'train' for row in label_inputs) == 12
    assert sum(row['partition'] == 'validation' for row in label_inputs) == 8
    assert not any(row['partition'] == 'test' for row in label_inputs)

    for document_index, row in enumerate(label_inputs, start=1):
        prefix_ids = answer = profile_meta = blocks = None
        try:
            prefix_ids = tokenise_prompt(tokenizer, row['prompt'], LABEL_PROFILE_CONTEXT_CAP, next(model.parameters()).device)
            answer = answer_ids(tokenizer, row.get('answers', []), next(model.parameters()).device)
            profile_meta = profile_short_context(
                model, prefix_ids, min(prefix_ids.shape[-1], LABEL_PROFILE_CONTEXT_CAP), run_root,
                tag=f"label_{row['task']}_{row['row_id']}",
                prefill_chunk_tokens=min(profile.prefill_chunk_tokens, LABEL_PROFILE_CONTEXT_CAP),
            )
            if not profile_meta['attention_available']:
                raise RuntimeError('Eager attention was unavailable; do not substitute fabricated attention labels.')
            blocks = label_blocks(
                model, prefix_ids, answer, profile_meta['profile_path'], block_size=16,
                max_counterfactual_blocks=LABEL_MAX_COUNTERFACTUAL_BLOCKS,
                attention_weight=0.5, prefill_chunk_tokens=min(profile.prefill_chunk_tokens, LABEL_PROFILE_CONTEXT_CAP),
            )
            for item in blocks:
                item.update({
                    'example_id': f"{row['task']}::{row['row_id']}", 'task': row['task'],
                    'source_partition': row['partition'], 'planned_context_length': row['planned_context_length'],
                    'label_profile_context_tokens': int(prefix_ids.shape[-1]),
                })
            label_rows.extend(blocks)
        finally:
            # The label rows hold CPU features only. Release all per-document
            # model tensors before moving to the next eager-attention profile.
            prefix_ids = answer = profile_meta = blocks = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                free_bytes, total_bytes = torch.cuda.mem_get_info()
                audit_row = {
                    'document_index': document_index, 'row_id': row['row_id'],
                    'partition': row['partition'], 'planned_context_length': row['planned_context_length'],
                    'free_gib_after_cleanup': round(free_bytes / 2**30, 3),
                    'total_gib': round(total_bytes / 2**30, 3),
                    'allocated_gib': round(torch.cuda.memory_allocated() / 2**30, 3),
                    'reserved_gib': round(torch.cuda.memory_reserved() / 2**30, 3),
                }
                label_memory_audit.append(audit_row)
                print(f"Label document {document_index}/{len(label_inputs)} complete; free VRAM after cleanup: {audit_row['free_gib_after_cleanup']:.2f} GiB")
    label_path = save_labels(label_rows, run_root / 'labels' / 'oracle_block_labels.pt', {
        'label_definition': '0.5 attention + 0.5 NLL impact', 'block_size': 16,
        'source': 'controlled_train_validation_short_context_oracle',
        'label_profile_context_cap': LABEL_PROFILE_CONTEXT_CAP,
        'max_counterfactual_blocks': LABEL_MAX_COUNTERFACTUAL_BLOCKS,
        'label_documents': {'train': 12, 'validation': 8, 'test': 0},
    })
    write_json(run_root / 'manifests' / 'label_document_selection.json', {
        'train_per_length': LABEL_TRAIN_PER_LENGTH, 'validation_per_length': LABEL_VALIDATION_PER_LENGTH,
        'label_profile_context_cap': LABEL_PROFILE_CONTEXT_CAP,
        'max_counterfactual_blocks': LABEL_MAX_COUNTERFACTUAL_BLOCKS,
        'rows': [{'row_id': row['row_id'], 'partition': row['partition'], 'planned_context_length': row['planned_context_length']} for row in label_inputs],
    })
    write_json(run_root / 'runtime' / 'label_generation_memory_audit.json', label_memory_audit)
    print('Saved', len(label_rows), 'labels to', label_path)
else:
    print('Labels are intentionally not generated until RUN_LABEL_GENERATION=True; no pseudo-labels are fabricated.')
'''),
    md('''## Step 3: Learned 28-Input Block Ranker

We train a lightweight MLP to predict the oracle importance score from 28 easily computed attention and positional features. The training loop uses pairwise logistic ranking loss to learn which block to keep when forced to choose between two.'''),
    code(r'''
from kvcore.scorer import ScorerConfig, train_scorer, load_scorer, split_documents, ranking_metrics

TRAIN_CONFIG_GRID = [
    {'learning_rate': 1e-3, 'dropout': 0.10, 'weight_decay': 1e-4, 'block_size': 16},
    {'learning_rate': 3e-4, 'dropout': 0.10, 'weight_decay': 1e-4, 'block_size': 16},
    {'learning_rate': 1e-3, 'dropout': 0.20, 'weight_decay': 1e-4, 'block_size': 16},
]
# Validation-only model selection. Do not inspect held-out test ranking metrics here.
if RUN_TRAINING:
    if not label_rows:
        import torch
        label_rows = torch.load(run_root / 'labels' / 'oracle_block_labels.pt', map_location='cpu')['labels']
    trials = []
    best = None
    for trial_index, params in enumerate(TRAIN_CONFIG_GRID):
        trial_root = run_root / 'trials' / f'trial_{trial_index:02d}'
        config = ScorerConfig(learning_rate=params['learning_rate'], dropout=params['dropout'], weight_decay=params['weight_decay'])
        scorer, normaliser, summary = train_scorer(label_rows, config=config, run_root=trial_root)
        record = {'trial': trial_index, **params, 'validation_ndcg_at_3': summary['best_validation_ndcg_at_3'], 'checkpoint': str(trial_root / 'checkpoints' / 'learned_kv_scorer.pt')}
        trials.append(record)
        if best is None or record['validation_ndcg_at_3'] > best['validation_ndcg_at_3']:
            best = record
    pd.DataFrame(trials).to_csv(run_root / 'results' / 'validation_model_selection.csv', index=False)
    write_json(run_root / 'checkpoints' / 'selected_by_validation_only.json', best)
    print('Selected checkpoint from validation only:', best)
else:
    print('Training disabled until RUN_TRAINING=True. This preserves a no-fabrication execution path.')
'''),
    md('''## Step 4: Validation-Only Policy Evaluation

Let us see how our newly trained policy compares to the baselines.

### What to Look For
Crucially, this evaluation runs **only on the validation partition**. If the learned policy outperforms the heuristics here, we have a strong candidate for the locked test.

**Next step:** Save the `run_root` directory, especially the `selected_by_validation_only.json` checkpoint. Notebook 04 requires it.'''),
    code(r'''
if RUN_VALIDATION_EVAL:
    selected = json.loads((run_root / 'checkpoints' / 'selected_by_validation_only.json').read_text())
    learned, normaliser, payload = load_scorer(selected['checkpoint'])
    assert_checkpoint_reload(learned, normaliser, selected['checkpoint'], load_scorer)
    specs = [
        {'type': 'full'}, {'type': 'fifo'}, {'type': 'sink_recent'}, {'type': 'h2o'},
        {'type': 'learned_block', 'block_size': 16},
    ]
    records = evaluate_rows(model, tokenizer, validation_rows, specs, profile.budgets, profile.max_context_tokens, profile.decode_tokens, profile.prefill_chunk_tokens, run_root, learned_model=learned, normaliser=normaliser, artifact_stem='rtx_validation_only')
else:
    records = [requires_gpu_record('RTX PRO 6000 or equivalent', 'Set RUN_VALIDATION_EVAL=True', 'Validation policy evaluation is user-triggered.')]
pd.DataFrame(records).to_csv(run_root / 'results' / 'rtx_validation_only.csv', index=False)
'''),
]
write_notebook('03_rtx_pro6000_longbench_learned_eviction.ipynb', 'KV Eviction — 03 RTX PRO 6000 Controlled Learned Eviction', cells03)

# 04 H100
cells04 = [
    md('''# 🚀 04: H100 Locked Testing and Serving Validation

*Part of the KV-Cache Eviction Capstone Series*
*Estimated time: 60–120 minutes (if executed at scale)*

---

Welcome to the final exam. This notebook is intentionally separated from development. We will load the frozen checkpoint selected in Notebook 03 and evaluate it on the 40 held-out controlled retrieval records spanning planned contexts of 1,024 through 8,192 tokens. An H100 may be used as the locked-test hardware, but this corpus does not justify a claim of 64k/128k evaluation.

### Experiment Roadmap
```text
[Frozen checkpoint] → [H100 preflight] → [Locked test] → [Observed artifacts]
```'''),
    md('''## Step 1: Why Does This Matter?

A scientific result must be tested on unseen data using a frozen model. If we tweak the policy now, we invalidate the experiment.

**Locked-Test Contract:**
The checkpoint and hyperparameters are fixed. No test quality metric may drive model selection. If the runtime cannot support H100-scale contexts, the run is explicitly skipped with a structured status record.'''),
    code("NOTEBOOK_ID = '04_h100_locked_test_serving'\nREQUESTED_PROFILE = 'h100'\n" + BOOTSTRAP),
    code(RUNTIME),
    code(MODEL_LOAD),
    code(r'''
RUN_H100_TEST = False
RUN_FP8_ANALYTIC_ACCOUNTING = True
RUN_SERVING_MEASUREMENT = False
CHECKPOINT_PATH = None  # Set to the validation-selected checkpoint exported from notebook 03.

print('Locked-test corpus:', CONTROLLED_RETRIEVAL['name'])
print('Planned context lengths:', CONTROLLED_RETRIEVAL['context_lengths'])
print('Held-out test records:', len(CONTROLLED_RETRIEVAL['context_lengths']) * CONTROLLED_RETRIEVAL['partition_counts']['test'])
'''),
    md('''## Step 2: Preflight Scale and Cache Accounting

Before we allocate massive caches, we verify the actual runtime. This cell records an honest condition if the hardware cannot support the requested scale.'''),
    code(r'''
from kvcore.evaluation import fp8_payload_accounting
preflight = {'requested_profile': 'h100', 'active_profile': profile.name, 'runtime_status': runtime_status, 'max_context_tokens': profile.max_context_tokens, 'dataset': CONTROLLED_RETRIEVAL['name'], 'planned_context_lengths': list(CONTROLLED_RETRIEVAL['context_lengths']), 'locked_test_records': 40}
if profile.name != 'h100':
    preflight['status'] = 'requires H100 — not run here'
    preflight['reason'] = 'Detected runtime does not meet the H100-scale profile.'
else:
    preflight['status'] = 'eligible_for_h100_run'
write_json(run_root / 'manifests' / 'h100_preflight.json', preflight)
print(preflight)
'''),
    md('''## Step 3: Locked-Test RULER Sweep

We now evaluate the full cache, the heuristics, and our frozen learned policy on the held-out test rows.'''),
    code(r'''
if RUN_H100_TEST:
    if CHECKPOINT_PATH is None:
        raise ValueError('Set CHECKPOINT_PATH to the validation-selected scorer checkpoint from notebook 03.')
    learned, normaliser, payload = load_scorer(CHECKPOINT_PATH)
    assert_checkpoint_reload(learned, normaliser, CHECKPOINT_PATH, load_scorer)
    all_rows = load_controlled_retrieval(tokenizer=tokenizer, seed=590)
    corpus_audit = audit_controlled_retrieval_corpus(all_rows)
    assert corpus_audit['passed'], corpus_audit
    test_rows = [row for row in all_rows if row['partition'] == 'test']
    assert len(test_rows) == 40, 'Locked test must retain all 40 preassigned test records.'
    build_benchmark_manifest(test_rows, run_root / 'manifests' / 'h100_locked_test_rows.json', benchmark_name='controlled_locked_test', seed=590)
    profile_ids = tokenise_prompt(tokenizer, test_rows[0]['prompt'], profile.attention_profile_context_cap, next(model.parameters()).device)
    meta = profile_short_context(model, profile_ids, min(profile_ids.shape[-1], profile.attention_profile_context_cap), run_root, 'h100_short_profile', profile.prefill_chunk_tokens)
    profile_data = load_profile(meta['profile_path'], device=next(model.parameters()).device) if meta['attention_available'] else None
    specs = [{'type': 'full'}, {'type': 'fifo'}, {'type': 'uniform'}, {'type': 'sink_recent'}, {'type': 'attention_topk'}, {'type': 'h2o'}, {'type': 'learned_block', 'block_size': 16}]
    test_records = evaluate_rows(model, tokenizer, test_rows, specs, profile.budgets, profile.max_context_tokens, profile.decode_tokens, profile.prefill_chunk_tokens, run_root, profile=profile_data, learned_model=learned, normaliser=normaliser, artifact_stem='h100_controlled_locked_test')
else:
    test_records = [requires_gpu_record('H100', 'Set RUN_H100_TEST=True and CHECKPOINT_PATH', 'Locked test execution is deliberately gated.')]
pd.DataFrame(test_records).to_csv(run_root / 'results' / 'h100_locked_test.csv', index=False)
'''),
    md('''## Step 4: FP8 and Serving Measurements

Finally, we record serving throughput and FP8 cache payload estimates. 

### What to Look For
Notice that FP8 estimates are explicitly labeled as **analytic** unless an actual FP8 cache backend is detected. We do not conflate synthetic formulas with observed throughput.

**Next step:** Save the `run_root` directory. We are ready for final aggregation in Notebook 05.'''),
    code(r'''
if RUN_FP8_ANALYTIC_ACCOUNTING:
    probe_ids = tokenizer('FP8 accounting probe. ' * 128, return_tensors='pt').input_ids.to(next(model.parameters()).device)
    probe_cache, _ = prefill(model, probe_ids, chunk_size=min(profile.prefill_chunk_tokens, probe_ids.shape[-1]), output_attentions=False)
    fp8 = fp8_payload_accounting(probe_cache)
    assert_fp8_byte_ratio(fp8)
    write_json(run_root / 'results' / 'fp8_analytic_accounting.json', fp8)
    print(fp8)

if RUN_SERVING_MEASUREMENT:
    print('Run the separate vLLM/SGLang service harness with the same model revision and workload manifest; export observed throughput/latency to results/serving_observed.csv.')
else:
    write_json(run_root / 'results' / 'serving_status.json', requires_gpu_record('H100 + serving backend', 'Set RUN_SERVING_MEASUREMENT=True', 'Serving is intentionally not inferred from analytic cache accounting.'))
'''),
    md('''## Step 5: Pareto-Frontier Input Contract

This notebook prepares the artifacts for Pareto analysis. It does not label the frontier itself; that strict aggregation is reserved for Notebook 05.'''),
    code("print('H100 notebook prepared. Run root:', run_root)\n" + ARTIFACT_NOTE),
]
write_notebook('04_h100_locked_test_serving.ipynb', 'KV Eviction — 04 H100 Locked Test and Serving', cells04)

# 05 results / frontier
cells05 = [
    md('''# 🚀 05: Results Aggregation and Pareto Analysis

*Part of the KV-Cache Eviction Capstone Series*
*Estimated time: 5 minutes*

---

We made it. This final notebook aggregates the artifacts produced by the preceding notebooks. It strictly processes **only observed** CSV artifacts, marks hardware-required runs explicitly, and exports publication-ready tables and plots.

### Experiment Roadmap
```text
[Artifacts] → [Validation] → [Observed-only tables] → [Pareto analysis] → [Report]
```'''),
    code("NOTEBOOK_ID = '05_results_pareto_and_reporting'\nREQUESTED_PROFILE = 't4'\n" + BOOTSTRAP),
    code(r'''
from pathlib import Path
import pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns, json

# Upload a ZIP of a completed run_root or mount Google Drive; no fabricated fallback data exists.
ARTIFACT_ROOT = Path('artifacts')
if not ARTIFACT_ROOT.exists():
    print('Place completed run artifacts beneath ./artifacts before aggregating. This notebook will not invent results.')
'''),
    md('''## Step 1: Why Does This Matter?

Data aggregation must be transparent. By strictly separating the execution notebooks (01–04) from the reporting notebook (05), we ensure that our final charts are drawn exclusively from traceable, observed artifacts.

Let us discover and validate the artifacts uploaded to the `artifacts/` directory.'''),
    code(r'''
def find_csvs(root): return sorted(root.rglob('*.csv')) if root.exists() else []
files = find_csvs(ARTIFACT_ROOT)
print('Found CSV artifacts:', [str(f) for f in files])
frames = []
validated_sources = []
for path in files:
    try:
        # Validate the on-disk artifact before parsing. assert_artifact accepts a
        # file path, not an in-memory DataFrame or list of records.
        assert_artifact(path)
        frame = pd.read_csv(path)
        if {'policy', 'budget'}.issubset(frame.columns):
            frame['artifact_source'] = str(path)
            frames.append(frame)
            validated_sources.append(str(path))
    except Exception as error:
        print('Unreadable or invalid artifact:', path, repr(error))
if not frames:
    print('No compatible observed policy artifacts found. Run notebooks 02–04 and copy their run roots into artifacts/.')
results = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
if not results.empty:
    # Notebook 04 intentionally writes a resumable-core CSV and a final-copy CSV.
    # Collapse only byte-identical measurements, retaining all materially distinct
    # artifacts from other notebooks/hardware tiers.
    identity = [name for name in ('benchmark','task','row_id','partition','policy','budget') if name in results.columns]
    value_columns = [name for name in results.columns if name != 'artifact_source']
    if identity:
        results = results.sort_values('artifact_source').copy()
        duplicate_mask = results.duplicated(subset=value_columns, keep='first')
        if duplicate_mask.any():
            print(f'Deduplicated {int(duplicate_mask.sum())} byte-equivalent policy records from final-copy exports.')
            results = results.loc[~duplicate_mask].copy()
    results.to_csv('aggregated_results.csv', index=False)
    write_json('aggregated_artifact_provenance.json', {
        'validated_sources': validated_sources,
        'raw_rows': int(sum(len(frame) for frame in frames)),
        'aggregated_rows': int(len(results)),
    })
    display(results.head())
'''),
    md('''## Step 2: Quality, Memory, and Latency Views

We generate plots only for metrics that were successfully observed. Status rows (like OOM fallbacks) remain in the tables for auditing but do not corrupt the data points.'''),
    code(r'''
if not results.empty and 'status' in results:
    observed = results[results['status'].eq('observed')].copy()
else:
    observed = pd.DataFrame()

if not observed.empty:
    if {'budget','substring_match','policy'}.issubset(observed.columns):
        plt.figure(figsize=(9,4)); sns.lineplot(data=observed, x='budget', y='substring_match', hue='policy', marker='o'); plt.title('Observed quality proxy versus cache budget'); plt.tight_layout(); plt.savefig('quality_vs_budget.png', dpi=180); plt.show()
    if {'physical_kv_bytes_post_evict','substring_match','policy'}.issubset(observed.columns):
        plt.figure(figsize=(7,5)); sns.scatterplot(data=observed, x='physical_kv_bytes_post_evict', y='substring_match', hue='policy'); plt.title('Observed quality versus physical KV payload'); plt.tight_layout(); plt.savefig('quality_vs_memory.png', dpi=180); plt.show()
else:
    print('No observed points: dashboards intentionally remain empty.')
'''),
    md('''## Step 3: Pareto-Frontier Analysis

A policy/budget point is "nondominated" (on the Pareto frontier) if no other configuration can achieve equal or better quality using less memory and less time. Let us calculate the frontier for our test results.'''),
    code(r'''
def pareto_flags(frame):
    required = {'physical_kv_bytes_post_evict','elapsed_seconds','substring_match'}
    if frame.empty or not required.issubset(frame.columns):
        return pd.Series(dtype=bool)
    flags = []
    values = frame.reset_index(drop=True)
    for i, row in values.iterrows():
        other = values.drop(i)
        dominates = (
            (other['physical_kv_bytes_post_evict'] <= row['physical_kv_bytes_post_evict']) &
            (other['elapsed_seconds'] <= row['elapsed_seconds']) &
            (other['substring_match'] >= row['substring_match']) &
            ((other['physical_kv_bytes_post_evict'] < row['physical_kv_bytes_post_evict']) |
             (other['elapsed_seconds'] < row['elapsed_seconds']) |
             (other['substring_match'] > row['substring_match']))
        ).any()
        flags.append(not dominates)
    return pd.Series(flags, index=frame.index)

if not observed.empty:
    observed['pareto_nondominated'] = pareto_flags(observed)
    observed.to_csv('aggregated_results_with_pareto.csv', index=False)
    print(observed['pareto_nondominated'].value_counts(dropna=False))
else:
    print('Pareto analysis requires observed results, not planned configurations.')
'''),
    md('''## Step 4: Final Reproducibility Checklist

### What to Look For
Before using these tables in the final report, verify that the output CSVs contain complete provenance: model revisions, GPU metadata, seeds, budget, and measurement class. 

Congratulations! The end-to-end evaluation is complete.'''),
]
write_notebook('05_results_pareto_and_reporting.ipynb', 'KV Eviction — 05 Results and Pareto Analysis', cells05)

# Bundle shared core.
bundle = ROOT / 'kvcore_bundle.zip'
with zipfile.ZipFile(bundle, 'w', zipfile.ZIP_DEFLATED) as zf:
    for path in sorted(CORE_DIR.rglob('*')):
        if path.is_file() and path.suffix == '.py':
            zf.write(path, path.relative_to(ROOT))

requirements = '\n'.join([
    'transformers==4.56.2', 'accelerate==1.10.1', 'datasets==4.0.0', 'huggingface_hub==0.34.4',
    'bitsandbytes==0.47.0', 'safetensors==0.6.2', 'sentencepiece==0.2.1', 'scipy==1.16.1',
    'matplotlib==3.10.6', 'seaborn==0.13.2', 'pandas==2.3.2',
]) + '\n'
(ROOT / 'requirements.txt').write_text(requirements)

README = clean('''
# KV-Cache Eviction — GPU-Profiled Google Colab Suite

## Delivered notebooks

| Order | Notebook | Primary GPU profile | Purpose |
|---:|---|---|---|
| 00 | `00_foundations_environment_eda.ipynb` | Any GPU | Runtime manifest, GQA-aware formula gate, and canonical 200-record controlled-corpus EDA. |
| 01 | `01_evicting_cache_core_tests.ipynb` | T4 or larger | Cache lifecycle, compaction, budget, logical-position, and streaming-decode correctness. |
| 02 | `02_t4_ruler_baseline_sweeps.ipynb` | NVIDIA T4 | Controlled retrieval baselines under matched cache budgets; 100 deterministic non-test records. |
| 03 | `03_rtx_pro6000_longbench_learned_eviction.ipynb` | RTX PRO 6000 / equivalent | Controlled-corpus labels, learned scorer, and validation-only model selection. |
| 04 | `04_h100_locked_test_serving.ipynb` | NVIDIA H100 | Locked test on 40 controlled held-out records, analytic FP8 accounting, and serving boundary. |
| 05 | `05_results_pareto_and_reporting.ipynb` | Any GPU | Aggregate observed artifacts only; quality/memory/latency/Pareto results. |

## Colab execution

1. Upload **one notebook** and `kvcore_bundle.zip` to a Colab GPU runtime.
2. Run the bootstrap cell. It installs immutable package versions and prompts for the shared bundle if necessary.
3. Run notebooks in numeric order. Export a completed `run_root` after every notebook.
4. Enable only the explicitly gated execution flags after reviewing the GPU, model license, and cost implications.
5. Copy completed `run_root` directories into the `artifacts/` directory for notebook 05.

## GPU-profile behavior

The notebooks never claim an H100, RTX PRO 6000, or T4 measurement unless that execution actually occurs. The runtime selector records the detected hardware and emits a structured **requires <GPU> — not run here** status when a requested scale cannot execute. Do not relabel fallback results as target-tier results.

| Profile | Pinned model | Expected scale | Cache load mode |
|---|---|---:|---|
| T4 | Qwen2.5-1.5B-Instruct | up to 8k | 4-bit + FP16 compute |
| RTX PRO 6000 | Qwen2.5-7B-Instruct | up to 32k | BF16 or 4-bit fallback |
| H100 | Qwen2.5-7B-Instruct-1M | up to 128k | BF16 |

## Benchmark and label protocol

- The active corpus is a deterministic 200-record RULER-like single-needle retrieval dataset with planned lengths 1,024/2,048/4,096/8,192 and an exact 30/10/10 split in every length stratum.
- **LongBench** is retained in the shared package only for a future, separately reported external-validity extension; it is not an active data source in this snapshot.
- Learned-importance labels are generated only from model attention evidence and controlled gold-answer NLL changes under cache ablation. No external label set is required.
- Long-context evaluation requests no attention outputs. Attention profiles are generated only in a capped eager short-context pass.

## Results integrity

The suite writes only observed values or explicit status/error records. It does not generate simulated quality, latency, memory, FP8, or serving outcomes. FP8 payload calculations are labelled analytic until a runtime exposes an observed FP8 cache backend. Serving throughput and latency require a separately measured vLLM/SGLang-style runtime and are never inferred from cache payload formulas.
''')
(ROOT / 'README.md').write_text(README)

# Build a delivery archive.
DELIVERY = Path('/home/ubuntu/work/kv_colab_v2/deliverables')
DELIVERY.mkdir(parents=True, exist_ok=True)
for path in list(NOTEBOOK_DIR.glob('*.ipynb')) + [bundle, ROOT / 'README.md', ROOT / 'requirements.txt', Path('/home/ubuntu/work/kv_colab_v2/official_compatibility_notes.md'), Path('/home/ubuntu/work/kv_colab_v2/suite_architecture.md')]:
    if path.exists():
        shutil.copy2(path, DELIVERY / path.name)
# Only explicit suite members enter the suite archive. This avoids recursively
# embedding prior consolidated archives or unrelated report files on rebuild.
suite_members = [
    DELIVERY / path.name
    for path in list(NOTEBOOK_DIR.glob('*.ipynb')) + [
        bundle, ROOT / 'README.md', ROOT / 'requirements.txt',
        Path('/home/ubuntu/work/kv_colab_v2/official_compatibility_notes.md'),
        Path('/home/ubuntu/work/kv_colab_v2/suite_architecture.md'),
    ]
]
for optional_name in ('validation_report_v2.json', 'KV_Eviction_Colab_Suite_v2_Step_by_Step_Guide.md'):
    optional_path = DELIVERY / optional_name
    if optional_path.exists():
        suite_members.append(optional_path)
with zipfile.ZipFile(DELIVERY / 'KV_Eviction_Colab_Suite_v2.zip', 'w', zipfile.ZIP_DEFLATED) as zf:
    for path in sorted(set(suite_members)):
        if path.exists():
            zf.write(path, path.name)
print('Generated:', sorted(p.name for p in DELIVERY.iterdir()))
