# Merged-L Coverage Experiment v1

Registered on 2026-09-06 (Asia/Shanghai). Current run: `coverage-01`.

Status: complete and independently audited on 2026-09-06. All 18 preflight passes, 10,288 full-dev blocks and 174,896 candidates are sealed. The independent audit recomputed all 240 registered CI targets in all six strata. See [concise findings](RESULTS.md), [audited Chinese report](results/coverage-01/REPORT.md) and [audit receipt](audits/coverage-01-full-dev/audit.json).

The original execution process exited normally and its independent deadline guard released without sending any stop signal. [The earlier handoff](HANDOFF.md) is historical. The separately registered [14B/27B queue](../general_model_ld_replication_queue_v1/README.md) started 14B preflight only after this run completed and released its processes. Completion of 8B does not certify either new model.

## Registration

- Protocol: [merged-L registration](../../../docs/research/experiment-plans/general-model-ld-coverage-numerical-measurement-v1.md).
- Plan: `gmlcoverage-64e2986d5de1d9c9ff0ebc9c2d72944208b833ca9aacf115ec71a14403e0c6f4`.
- Plan file SHA256: `31eb78cb942a6d929a04cd74c32b302d848ae4c7f73c782a37ae48033af3359d`.
- Resource: frozen 833-entry lexicon; one global ID-sorted deduplicated `Lnew = Lq union Ld` block. The original ten selected demos and their order are unchanged.
- Core conditions: C0, CLnew, CD, CLDnew, PLnew, PD. Auxiliary references: CLq, CLqD. All eight are newly scored, without importing prior candidate scores.
- Scope: 643 queries, hate/group, 10,288 blocks, 174,896 candidates; main answer-token logprob excludes EOS.
- Analysis: ten contrasts, 240 CI targets, six original strata, 10,000 query bootstrap draws, descriptive pointwise 95% intervals. Hit strata still refer to Lq.

## Input Acceptance

All eight conditions passed CPU token boundaries, neutral controls, and complete-answer/EOS plus 64-padding capacity checks. New L covers 640/643 queries; maximum 17 entries, 773 dictionary tokens, 2,151 complete sequence tokens (8,192 limit).

Original regression 8 and validation 24 query IDs remain unchanged. Metadata-only boundary additions are `2312`, `612`, `4239`, `5975`. The four representatives cover empty Lnew, largest union, longest dictionary, and longest complete input. The plan records selection before any new GPU scoring.

53 new tests passed, plus inherited statistics/kernel-execution regression checks. The earlier CPU-only plan `gmlcoverage-c07e103d0a12cabb48a61a49f5d77625df665037795890e579e5c56e197f4730` is retained as an unexecuted build revision; the current plan adds loader identity checks. No old experiment is overwritten.

## Execution

Four identical FP32 L20 replicas; true batch 1, eager attention, TF32 off, no KV cache. Each of the three preflight cohorts runs baseline/CPU-FP64 arithmetic reference, repeat, padding, uncached prefix, candidate order, and physical replica checks. E8 remains `0.00067138671875`, epsilon remains `0.0013427734375`; repeat/arithmetic tolerance remains `1e-4`. No automatic tolerance or profile search.

```bash
.conda/stage1-p0/bin/python scripts/stage1/general_model_coverage.py validate --plan exps/causal_context/general_model_ld_coverage_v1/plan_ref.json
.conda/stage1-p0/bin/python scripts/stage1/general_model_coverage.py run --plan exps/causal_context/general_model_ld_coverage_v1/plan_ref.json --output exps/causal_context/general_model_ld_coverage_v1/runs/coverage-01
```

Terminal runs validate without restarting GPUs. Failed runs remain sealed. Query gold is not deserialized until complete raw scores and all preflight/shard evidence have been verified. Test data is never opened.

## Audit Notes

[Input review](audits/input_review.json) independently reconstructed all prompts and their resource identities. [CPU verification](audits/cpu_verification.json) records 292 main-suite tests and 39 current standalone audit/report tests. [Regression-stage review](audits/regression-stage-review.json) independently verified all six sealed regression passes; it does not claim that the remaining preflight cohorts passed.

The unchanged v2 scorer recognizes only regression/validation in its legacy row-level `cohort` field. Rows from `boundary-b1-*` therefore retain `cohort=dev`. Boundary membership is instead verified against the exact registered pass name, query IDs and full ordered record matrix. The independent coverage auditor explicitly checks this legacy mapping without rewriting raw rows or loosening any numeric gate. Do not use the legacy field alone to distinguish boundary preflight from the full `dev-b1` pass.

[GPU monitoring note](audits/gpu-monitoring-note.md) records the HAMI per-device memory accounting anomaly; the reported memory.used values are not used to infer per-device residency or waive numerical checks.
