# Merged-L Replication: Qwen3-14B

Current scientific run: `runs/replication-01`, stopped at the user's request on
2026-09-06 at 22:36:06 Asia/Shanghai. All owned processes and the writer lock are
released. The user will choose an acceleration approach before a new run; do not
automatically resume, benchmark or switch execution profiles.

Nine of 18 preflight checks have sealed evidence. Validation candidate order
retains 289/384 blocks; all preflight passes retain 2,465 blocks and 40,480
candidate scores. All 10 checkpoint databases passed integrity, metadata and
payload-hash checks. Full dev has not started. The queue's recorded failed status
is a stop-time lifecycle error with stale job fields, not a live workload or a
scientific numerical failure. See the [user-stop record](../general_model_ld_replication_queue_v1/USER_STOP_20260906_2236.md).

## Frozen Scope

- Plan: `gmlrep-b64f97a77578d0c5e1f91adc1b4c509df409d3f56c04c4329b2ca6ed8718f2d5`.
- Plan SHA256: `efcf9e4079fc790db0fd92d60b78b3f00091954958780429fb16835ed51f0b5c`.
- All 643 dev queries, hate/group, eight conditions, 10,288 blocks and 174,896 candidates.
- Frozen 833-entry lexicon, global ID-sorted Lq union Ld, same ten demos per query.
- FP32, true candidate batch 1, eager uncached scoring; one layer-sharded model
  on GPUs 0/1, with physical-placement checks on GPUs 2/3.
- Original regression 8 and validation 24 queries plus four metadata boundaries.
- Frozen primary answer-only logprob sums, auxiliaries and numerical tolerances.

## Current Window

See the [24:00 renewal authorization and schedule](../general_model_ld_replication_queue_v1/WINDOW_20260906_2400.md).
The 14B-only queue resumed the same plan at 20:12 and was handed back after stable
startup at 20:15. Its 23:55/24:00 deadline was superseded by the user's early-stop
request at 22:34. No 27B job or dual-pair cutover occurred. The queue and watchdog
have exited; the unused remainder of the reservation does not authorize restart.

Authoritative state files:

- [Scientific run state](runs/replication-01/run_manifest.json).
- [Queue state](../general_model_ld_replication_queue_v1/runs/window-20260906-03-14b/state.json).
- [Watchdog state](../general_model_ld_replication_queue_v1/runs/window-20260906-03-14b/watchdog.json).
- [Preflight log](../general_model_ld_replication_queue_v1/runs/window-20260906-03-14b/logs/14b-preflight.log).

Before continuation, the plan, inputs and 33 frozen/current source files were
verified. All four checkpoint databases passed integrity, payload hash,
finite-readout and geometry checks; the saved runtime passed CPU validation.
The old processes had exited and the writer lock was free. GPU/runtime identity
was checked again during actual continuation; mismatches stop rather than mix runs.

At the earlier 20:00 closeout, all nine databases passed integrity and content checks, including
all eight sealed checks and the partial prefix. A total of 2,087 blocks and
34,144 candidate scores were preserved across preflight passes. The frozen inputs,
runtime and 33 source files were revalidated; no gold or scientific analysis was
read. The [checkpoint audit](../general_model_ld_replication_queue_v1/checkpoint_audit_20260906_2000.json)
and all 60 evidence hashes were rechecked before renewal. Its counts and hashes
describe the stopped state; normal continuation advances the mutable checkpoints.

## Proposed Parallel Execution

The [dual-pair execution proposal](../../../docs/research/experiment-plans/general-model-ld-14b-dual-pair-proposal-v1.md)
specifies GPUs 0/1 and 2/3 as two replicas, fixed query-level assignment, separate
checkpoints, a new plan/run identity and complete numerical revalidation. At 14:17
Asia/Shanghai on 2026-09-06, the user chose to keep the current 14B run on its
existing plan and reserve dual-pair execution for subsequent experiments. This
proposal is not implemented and does not change or restart the current run.
For future adoption, old scores would remain evidence and would not be imported
into new formal raw.

## Independent Audit Tools

`audits/independent_replication_audit.py` and `audits/render_replication_report.py`
are CPU-only postprocessing tools for this single-pair 14B execution schema.
They are not a passed audit of the still-incomplete scientific run, and cannot
accept a future dual-pair schema without a separately validated adapter. The
auditor requires a terminal complete run and independently checks raw, checkpoint,
runtime and preflight evidence before any explicitly authorized gold access.
Scientific tables and the report require the complete gold/CI verification path;
a raw-only receipt is not a scientific result.

## Earlier Evidence

The [morning handoff](../general_model_ld_replication_queue_v1/HANDOFF_20260906.md)
and [window acceptance](../general_model_ld_replication_queue_v1/window_acceptance_20260906.json)
remain historical. They record three sealed 128-block passes and 82 committed
prefix blocks at 09:55. Those committed blocks are reused; the one unfinished
attempt is recomputed. Final scientific results require complete raw, analysis
and independent audit, not merely progress counts.
