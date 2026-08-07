# A100 Locked-Test Evidence Audit and Notebook 05 Input

## Evidence status

The uploaded A100 locked-test artifacts meet the **execution-completeness and export-consistency requirements**. The frozen selection record identified Trial 0, with a validation-only NDCG@3 of 0.9780. The uploaded Drive resume manifest reports `status: complete`, `expected_records: 1000`, and `completed_records: 1000`. It identifies the frozen Qwen/Qwen2.5-7B-Instruct-1M revision `e28526f7bb80e2a9c8af03b831a9af3812f18fba`, the A100 hardware label, the original scorer checkpoint’s SHA-256, and the 40-row locked test.

The durable journal, the `a100_controlled_locked_test.csv` export, and the `a100_locked_test.csv` final export each contain **1,000 records**. Their row identities and measurement values agree exactly after harmless column-order normalization. There are no duplicate `(row, policy, budget)` identities and every measurement is explicitly labeled `observed`.

| Audit item | Observed value |
|---|---:|
| Locked-test rows | 40 |
| Planned context strata | 1,024; 2,048; 4,096; 8,192 tokens |
| Test rows per stratum | 10 |
| Policies | 7 |
| Measurements | 1,000 |
| Status `observed` | 1,000 / 1,000 |
| Resume-manifest status | complete |
| Core export equals final export | True |
| Core export equals progress journal | True |

## Observed outcomes and reporting boundary

All figures and tables below are **A100 observed locked-test evidence**. They are neither H100 evidence nor estimated/analytic serving data. The suite recorded 32 decoded tokens per measurement using the transparent `exact_match` and `substring_match` quality proxies.

> The quality evidence does **not** support a retrieval-effectiveness claim for this execution. Exact match is 0.000 across all 1,000 measurements, and overall substring match is 0.021. The forced-eviction subset is also 0.000 on both proxies. Representative saved traces show repetitive filler-token output even under the full-cache reference configuration. Therefore, the failure is present before eviction-policy comparison and cannot be attributed to learned eviction. This run supports cache/latency instrumentation evidence, but not a claim that the learned policy preserves retrieval quality.

Across all measurements, the mean physical KV-cache reduction relative to the full-cache reference was 11.034%. In the **360 forced-eviction measurements**—where the policy budget is below the prompt length—the mean reduction was 29.455%. These are observed payload reductions. They should not be restated as end-to-end serving-memory savings without measuring the full model-serving stack.

## Forced-eviction comparison

The table reports only configurations with budget below prompt length, grouped over their locked-test context/budget points. Values are descriptive A100 observations, not claims of statistical significance.

| policy         |   context_budget_points |   measurements |   exact_match_rate |   substring_match_rate |   mean_elapsed_seconds |   median_of_context_medians_seconds |   mean_policy_overhead_seconds |   mean_cache_reduction_pct_vs_full |   mean_elapsed_delta_seconds_vs_full |
|:---------------|------------------------:|---------------:|-------------------:|-----------------------:|-----------------------:|------------------------------------:|-------------------------------:|-----------------------------------:|-------------------------------------:|
| attention_topk |                       6 |             60 |                  0 |                      0 |                  1.929 |                               1.947 |                          0.154 |                             29.99  |                                0.141 |
| fifo           |                       6 |             60 |                  0 |                      0 |                  1.917 |                               1.932 |                          0.151 |                             29.184 |                                0.129 |
| h2o            |                       6 |             60 |                  0 |                      0 |                 41.047 |                              24.814 |                         37.679 |                             29.184 |                               39.259 |
| learned_block  |                       6 |             60 |                  0 |                      0 |                  2.534 |                               2.385 |                          0.72  |                             29.99  |                                0.746 |
| sink_recent    |                       6 |             60 |                  0 |                      0 |                  1.913 |                               1.933 |                          0.144 |                             29.191 |                                0.125 |
| uniform        |                       6 |             60 |                  0 |                      0 |                  1.911 |                               1.93  |                          0.148 |                             29.189 |                                0.123 |

The simple heuristics clustered at roughly 1.91–1.93 seconds per forced-eviction measurement, with a mean cache reduction near 29%. The learned block policy produced a similar cache reduction (29.990%) but its observed mean elapsed time was 2.534 seconds, including 0.720 seconds of policy overhead. H2O produced similar cache reduction but substantially higher elapsed time (41.047 seconds) and overhead (37.679 seconds) in this implementation. Because the full-cache configuration and all policies share the same near-zero quality outcome under this prompt/decode protocol, **no retrieval-preservation or learned-policy superiority conclusion is warranted**.

## Context-stratified observations

|   planned_context_length |   measurements |   test_rows |   exact_match_rate |   substring_match_rate |   mean_elapsed_seconds |   median_elapsed_seconds |   mean_cache_reduction_pct_vs_full |
|-------------------------:|---------------:|------------:|-------------------:|-----------------------:|-----------------------:|-------------------------:|-----------------------------------:|
|                     1024 |            250 |          10 |                  0 |                  0.084 |                  1.524 |                    1.294 |                              1.124 |
|                     2048 |            250 |          10 |                  0 |                  0     |                  3.033 |                    1.469 |                              0.574 |
|                     4096 |            250 |          10 |                  0 |                  0     |                  6.423 |                    1.7   |                             12.292 |
|                     8192 |            250 |          10 |                  0 |                  0     |                 14.926 |                    2.263 |                             30.146 |

The average elapsed time rises materially with planned context length. The low substring-match rate appears only in the 1,024-token stratum; it is zero in the 2,048-, 4,096-, and 8,192-token strata. This is further evidence that this locked run requires a prompt-formatting and task-scoring correction before using the quality column for a policy Pareto frontier.

## Required language for Notebook 05 and the capstone report

Use language equivalent to the following:

> “On the completed A100 80 GB locked test, all 1,000 predeclared policy–budget measurements completed and were journaled with a complete immutable resume manifest. The run observed physical KV-cache payload reductions, including a mean 29.45% reduction in the forced-eviction subset. However, the full-cache reference and eviction policies produced near-zero retrieval quality under the present plain-prompt, greedy-decoding protocol (0.0 exact match overall; 2.1% substring match overall). Consequently, these measurements are reported as A100 cache and latency instrumentation evidence only; they do not establish retrieval preservation or learned-policy superiority. A corrected chat-template/prompt-and-scoring protocol is required before making an effectiveness claim.”

## Next experimental action

Do **not** overwrite or modify this completed locked-test evidence. Treat the run as a reproducible negative/control result and retain it in reporting. Before any future effectiveness evaluation, conduct a separate, versioned protocol correction on validation/development data: apply the model’s chat template or an equivalent answer-compatible prompt format; verify full-cache retrieval on a small validation sanity set; then lock a new test protocol and checkpoint fingerprint before executing another held-out test. Do not use the current locked test to retune the scorer or select a policy.

## Included files

| File | Use |
|---|---|
| `a100_observed_policy_budget_by_context.csv` | Complete observed policy/budget/context aggregation. |
| `a100_forced_eviction_policy_budget_by_context.csv` | Only budget-below-context results. |
| `a100_forced_eviction_policy_summary.csv` | Descriptive forced-eviction policy comparison. |
| `a100_observed_context_summary.csv` | Context-stratified observed summary. |
| `a100_forced_eviction_cache_reduction.png` | Cache reduction visualization. |
| `a100_forced_eviction_elapsed_time.png` | Elapsed-time visualization. |
| `a100_quality_proxy_by_context.png` | Quality-proxy visualization showing the current protocol limitation. |
