from __future__ import annotations

import ast
import importlib
import inspect
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import nbformat

ROOT = Path('/home/ubuntu/work/kv_colab_v2/suite')
NOTEBOOK_DIR = ROOT / 'notebooks'
REPORT = Path('/home/ubuntu/work/kv_colab_v2/deliverables/validation_report_v2.json')

results: dict[str, object] = {
    'suite': 'KV Eviction Colab Suite v2',
    'python_module_syntax': {},
    'notebook_code_syntax': {},
    'requirements': {},
    'bundle': {},
    'shared_import': {},
    'runtime_manifest_regression': {},
    'legacy_notebook00_api_regression': {},
    'evaluation_oom_handler_regression': {},
    'controlled_corpus_regression': {},
    'source_partition_regression': {},
    'label_cache_clone_regression': {},
    'notebook03_label_split_regression': {},
    'notebook03_memory_safety_regression': {},
    'scorer_tensorize_regression': {},
    'scorer_checkpoint_dir_regression': {},
    'pass': False,
}

for path in sorted((ROOT / 'kvcore').glob('*.py')):
    try:
        ast.parse(path.read_text())
        results['python_module_syntax'][path.name] = 'passed'
    except SyntaxError as err:
        results['python_module_syntax'][path.name] = f'failed: {err}'

required_notebooks = [
    '00_foundations_environment_eda.ipynb',
    '01_evicting_cache_core_tests.ipynb',
    '02_t4_ruler_baseline_sweeps.ipynb',
    '03_rtx_pro6000_longbench_learned_eviction.ipynb',
    '04_h100_locked_test_serving.ipynb',
    '05_results_pareto_and_reporting.ipynb',
]
for name in required_notebooks:
    path = NOTEBOOK_DIR / name
    try:
        nb = nbformat.read(path, as_version=4)
        for index, cell in enumerate(nb.cells):
            if cell.cell_type == 'code':
                compile(cell.source, f'{name}:cell_{index}', 'exec')
        text = '\n'.join(c.source for c in nb.cells)
        requirements = {
            'bootstrap': 'kvcore_bundle.zip' in text,
            'pinned_versions': 'transformers==4.56.2' in text,
            'profile_gate': 'REQUESTED_PROFILE' in text,
        }
        results['notebook_code_syntax'][name] = {'status': 'passed', 'cells': len(nb.cells), 'requirements': requirements}
    except Exception as err:
        results['notebook_code_syntax'][name] = {'status': f'failed: {err}'}

bundle = ROOT / 'kvcore_bundle.zip'
results['bundle'] = {'exists': bundle.exists(), 'bytes': bundle.stat().st_size if bundle.exists() else 0}

sys.path.insert(0, str(ROOT))
try:
    kvcore = importlib.import_module('kvcore')
    expected = ['EvictingCache', 'formula_gate', 'load_controlled_retrieval', 'audit_controlled_retrieval_corpus', 'load_ruler', 'make_policy', 'ImportanceScorer', 'evaluate_rows', 'requires_gpu_record', 'tokenise_prompt', 'write_json']
    missing = [name for name in expected if not hasattr(kvcore, name)]
    results['shared_import'] = {'status': 'passed' if not missing else 'failed', 'missing_exports': missing}
except Exception as err:
    results['shared_import'] = {'status': f'failed: {err}', 'missing_exports': []}

# Regression test for the former duplicate requested_profile error. It calls
# the manifest helper in the previous positional-plus-keyword style.
try:
    from kvcore.runtime import run_manifest, select_profile
    with tempfile.TemporaryDirectory() as directory:
        profile, status = select_profile('t4')
        manifest_path = run_manifest(
            directory,
            '00_foundations',
            profile.name,
            requested_profile='t4',
            active_profile=profile.name,
            runtime_status=status,
        )
        manifest_payload = json.loads(Path(manifest_path).read_text())
        assert manifest_payload['notebook'] == '00_foundations'
        assert manifest_payload['requested_profile'] == 't4'
        assert manifest_payload['active_profile'] == profile.name
        assert 'runtime_status' in manifest_payload['extra']
    results['runtime_manifest_regression'] = {'status': 'passed'}
except Exception as err:
    results['runtime_manifest_regression'] = {'status': f'failed: {err}'}

# Regression test for notebook 00 calls from the originally delivered build.
try:
    from kvcore import assert_formula_gate, build_benchmark_manifest, formula_for_model, formula_gate, write_json

    config = SimpleNamespace(
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        hidden_size=16,
        head_dim=4,
    )
    expected = formula_for_model(config, retained_tokens=3, batch_size=1, dtype='float16')

    class FakeCache:
        physical_length = 3

        def __init__(self, payload_bytes):
            self._payload_bytes = payload_bytes

        def payload_bytes(self):
            return self._payload_bytes

    cache = FakeCache(expected['formula_bytes'])
    gate = formula_gate(cache, expected['formula_bytes'], tolerance=0.02)
    assert_formula_gate(gate)
    with tempfile.TemporaryDirectory() as directory:
        directory_path = Path(directory)
        json_path = write_json(directory_path / 'checks' / 'gate.json', {'expected': expected, 'gate': gate})
        assert json_path.exists()
        rows = [{
            'benchmark': 'longbench_v1', 'task': 'qasper', 'row_id': 'compat-1',
            'partition': 'train', 'source_revision': 'pinned',
        }]
        manifest_path = build_benchmark_manifest(rows, directory_path / 'manifests' / 'eda_rows.json')
        assert manifest_path.exists()
    results['legacy_notebook00_api_regression'] = {'status': 'passed'}
except Exception as err:
    results['legacy_notebook00_api_regression'] = {'status': f'failed: {err}'}

# Regression test for the controlled RULER-like corpus described in the Methods document.
try:
    from kvcore import audit_controlled_retrieval_corpus, load_controlled_retrieval

    class WhitespaceTokenizer:
        def __call__(self, text, add_special_tokens=False):
            return SimpleNamespace(input_ids=str(text).split())
        def encode(self, text, add_special_tokens=False):
            return [1 for _ in str(text).split()]
        def decode(self, ids, skip_special_tokens=True, clean_up_tokenization_spaces=False):
            return " ".join(["context"] * len(ids))

    controlled_rows = load_controlled_retrieval(tokenizer=WhitespaceTokenizer(), seed=590)
    audit = audit_controlled_retrieval_corpus(controlled_rows)
    assert audit['passed'], audit
    assert len(controlled_rows) == 200
    for length in (1024, 2048, 4096, 8192):
        subset = [row for row in controlled_rows if row['planned_context_length'] == length]
        assert len(subset) == 50
        assert sum(row['partition'] == 'train' for row in subset) == 30
        assert sum(row['partition'] == 'validation' for row in subset) == 10
        assert sum(row['partition'] == 'test' for row in subset) == 10
    results['controlled_corpus_regression'] = {'status': 'passed', 'rows': len(controlled_rows)}
except Exception as err:
    results['controlled_corpus_regression'] = {'status': f'failed: {err}'}

# Regression test: scorer must honor a supplied controlled-corpus partition.
try:
    from kvcore.scorer import split_documents
    partitioned = split_documents([
        {'example_id': 'controlled-1', 'source_partition': 'train'},
        {'example_id': 'controlled-2', 'source_partition': 'validation'},
        {'example_id': 'controlled-3', 'source_partition': 'test'},
    ])
    assert len(partitioned['train']) == len(partitioned['validation']) == len(partitioned['test']) == 1
    results['source_partition_regression'] = {'status': 'passed'}
except Exception as err:
    results['source_partition_regression'] = {'status': f'failed: {err}'}

# Regression test: Stage 2 must stack saved tensor-valued oracle features.
try:
    import torch
    from kvcore.scorer import _tensorize
    tensor_rows = [
        {'feature': torch.arange(28, dtype=torch.float32), 'target': 0.25, 'example_id': 'train::a'},
        {'feature': torch.arange(28, dtype=torch.float32) + 1, 'target': 0.75, 'example_id': 'validation::b'},
    ]
    features, targets, documents = _tensorize(tensor_rows, device='cpu')
    assert tuple(features.shape) == (2, 28)
    assert tuple(targets.shape) == (2,)
    assert documents == ['train::a', 'validation::b']
    results['scorer_tensorize_regression'] = {'status': 'passed', 'shape': list(features.shape)}
except Exception as err:
    results['scorer_tensorize_regression'] = {'status': f'failed: {err}'}

# Regression test: scorer trials must create nested checkpoint directories.
try:
    import torch
    from kvcore.scorer import ScorerConfig, load_scorer, train_scorer
    from kvcore.tests import assert_checkpoint_reload
    tiny_rows = []
    for partition, prefix in [('train', 'train'), ('validation', 'validation')]:
        for block_index, target in enumerate((0.2, 0.8)):
            tiny_rows.append({
                'feature': torch.arange(28, dtype=torch.float32) + block_index,
                'target': target,
                'example_id': f'{prefix}::doc_0',
                'source_partition': partition,
            })
    with tempfile.TemporaryDirectory() as temp_dir:
        trial_root = Path(temp_dir) / 'trials' / 'trial_00'
        config = ScorerConfig(max_epochs=1, patience=1, batch_size=4)
        scorer, normaliser, _ = train_scorer(tiny_rows, config=config, device='cpu', run_root=trial_root)
        checkpoint_file = trial_root / 'checkpoints' / 'learned_kv_scorer.pt'
        manifest_file = trial_root / 'checkpoints' / 'learned_kv_scorer_manifest.json'
        assert checkpoint_file.exists() and manifest_file.exists()
        assert_checkpoint_reload(scorer, normaliser, checkpoint_file, load_scorer)
    results['scorer_checkpoint_dir_regression'] = {'status': 'passed'}
except Exception as err:
    results['scorer_checkpoint_dir_regression'] = {'status': f'failed: {err}'}

# Regression test: Oracle labels must use no-grad inference and bounded-memory cleanup.
try:
    from kvcore import cache as cache_module
    from kvcore import labels as labels_module
    from kvcore import profiling as profiling_module
    cache_source = inspect.getsource(cache_module.prefill)
    labels_source = inspect.getsource(labels_module.label_blocks) + inspect.getsource(labels_module.gold_continuation_nll)
    profiling_source = inspect.getsource(profiling_module.profile_short_context)
    generator_source = (ROOT.parent / 'build_colab_v2.py').read_text()
    notebook03_source = (ROOT / 'notebooks' / '03_rtx_pro6000_longbench_learned_eviction.ipynb').read_text()
    assert 'with torch.no_grad()' in cache_source
    assert 'with torch.no_grad()' in labels_source
    assert 'torch.cuda.empty_cache()' in labels_source
    assert 'disable_attention_outputs(model)' in profiling_source
    assert 'LABEL_PROFILE_CONTEXT_CAP = 1024' in generator_source
    assert "assert_checkpoint_reload(learned, normaliser, selected['checkpoint'], load_scorer)" in generator_source
    assert 'assert_checkpoint_reload(learned, normaliser, CHECKPOINT_PATH, load_scorer)' in generator_source
    assert 'label_generation_memory_audit.json' in notebook03_source
    results['notebook03_memory_safety_regression'] = {'status': 'passed', 'label_profile_context_cap': 1024}
except Exception as err:
    results['notebook03_memory_safety_regression'] = {'status': f'failed: {err}'}

# Regression test: Notebook 03 label documents must span train and validation only.
try:
    from kvcore import load_controlled_retrieval
    selected_rows = []
    all_rows = load_controlled_retrieval(seed=590)
    for partition, per_length in (("train", 3), ("validation", 2)):
        for length in (1024, 2048, 4096, 8192):
            candidates = [row for row in all_rows if row['partition'] == partition and row['planned_context_length'] == length]
            selected_rows.extend(candidates[:per_length])
    assert len(selected_rows) == 20
    assert sum(row['partition'] == 'train' for row in selected_rows) == 12
    assert sum(row['partition'] == 'validation' for row in selected_rows) == 8
    assert not any(row['partition'] == 'test' for row in selected_rows)
    results['notebook03_label_split_regression'] = {'status': 'passed', 'train_documents': 12, 'validation_documents': 8, 'test_documents': 0}
except Exception as err:
    results['notebook03_label_split_regression'] = {'status': f'failed: {err}'}

# Regression test: oracle-label cache cloning must preserve a multi-layer modern DynamicCache layout.
try:
    import torch
    from kvcore.cache import EvictingCache
    from kvcore.labels import _clone_cache

    source_cache = EvictingCache()
    positions = torch.arange(5, dtype=torch.long)
    for layer_index in range(4):
        key = torch.zeros((1, 2, 5, 4), dtype=torch.float16)
        value = torch.ones((1, 2, 5, 4), dtype=torch.float16)
        source_cache.update(key, value, layer_index, {'cache_position': positions})
    clone_cache = _clone_cache(source_cache)
    assert len(clone_cache.layers) == 4
    assert clone_cache.physical_length == 5 and clone_cache.logical_length == 5
    clone_cache.compact(torch.tensor([0, 2, 4], dtype=torch.long), reason='validator_clone_regression')
    assert clone_cache.physical_length == 3
    assert all(layer.keys.shape[-2] == 3 and layer.values.shape[-2] == 3 for layer in clone_cache.layers)
    results['label_cache_clone_regression'] = {'status': 'passed'}
except Exception as err:
    results['label_cache_clone_regression'] = {'status': f'failed: {err}'}

# Regression test: evaluation must bind torch before catching CUDA OOM errors.
try:
    from kvcore import evaluation as evaluation_module
    evaluation_source = inspect.getsource(evaluation_module.evaluate_rows)
    assert 'torch = _torch()' in evaluation_source
    assert 'except torch.cuda.OutOfMemoryError:' in evaluation_source
    results['evaluation_oom_handler_regression'] = {'status': 'passed'}
except Exception as err:
    results['evaluation_oom_handler_regression'] = {'status': f'failed: {err}'}

all_python = all(v == 'passed' for v in results['python_module_syntax'].values())
all_notebooks = all(item.get('status') == 'passed' for item in results['notebook_code_syntax'].values())
results['pass'] = bool(all_python and all_notebooks and results['bundle']['exists'] and results['shared_import'].get('status') == 'passed' and results['runtime_manifest_regression'].get('status') == 'passed' and results['legacy_notebook00_api_regression'].get('status') == 'passed' and results['controlled_corpus_regression'].get('status') == 'passed' and results['source_partition_regression'].get('status') == 'passed' and results['notebook03_memory_safety_regression'].get('status') == 'passed' and results['notebook03_label_split_regression'].get('status') == 'passed' and results['scorer_tensorize_regression'].get('status') == 'passed' and results['scorer_checkpoint_dir_regression'].get('status') == 'passed' and results['label_cache_clone_regression'].get('status') == 'passed' and results['evaluation_oom_handler_regression'].get('status') == 'passed')
REPORT.write_text(json.dumps(results, indent=2))
print(json.dumps(results, indent=2))
if not results['pass']:
    raise SystemExit(1)
