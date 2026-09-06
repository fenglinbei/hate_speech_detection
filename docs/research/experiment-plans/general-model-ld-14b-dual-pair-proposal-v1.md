# Qwen3-14B Dual-Pair Execution Proposal

Status: retained for future experiments, by the user's 2026-09-06 14:17
Asia/Shanghai decision. This is not a frozen plan or GPU cutover authorization.
The user chose to finish the current 14B run under its existing single-pair plan,
preserving compatible checkpoints and avoiding a restart for this proposal.
Do not switch the current run to dual-pair execution. Subsequent experiments
should implement and validate this proposal before adopting it. The current GPU
reservation still ends at 20:00 Asia/Shanghai, with graceful stop at 19:55 and an
independent watchdog enforcing the deadline; this decision does not extend it.

## Scope and Invariants

Use two independently layer-sharded Qwen3-14B replicas: worker A on physical GPUs
0/1 and worker B on physical GPUs 2/3. Each candidate remains true batch 1, FP32,
eager, uncached teacher forcing. Preserve the existing layer partition and every
scientific input, candidate, score definition, numerical tolerance and analysis
boundary. The 643 dev queries still produce 10,288 blocks and 174,896 candidates.
This is an execution amendment, not a larger statistical sample.

The parent is the single-pair plan recorded by
`exps/causal_context/general_model_ld_coverage_14b_v1/plan_ref.json`:
`gmlrep-b64f97a77578d0c5e1f91adc1b4c509df409d3f56c04c4329b2ca6ed8718f2d5`.
Do not modify its frozen source, plan, contexts, runtime or checkpoint metadata.
No quantization, lower precision, offload, caching, batching, automatic profile
search, input shortening, gold inspection or scientific-effect-based selection
is introduced. Qwen3.8-27B is outside this window's proposed execution scope.

## Deterministic Assignment

The ownership unit is one complete query: both tasks and all eight conditions
belong to the same worker. The commit unit remains a query/task/condition block
with all 2 hate or all 32 group candidates; incomplete blocks are recomputed.

Build separate assignment maps for dev, regression, validation and boundary
from the already frozen token metadata. For each query, sum
`prompt_tokens + answer_tokens + 1` over its 16 contexts and every candidate;
the extra token accounts for the auxiliary EOS position. Sort queries by
descending summed cost, breaking ties by original frozen frame position. Assign
each query to the worker minimizing `(assigned_cost, assigned_query_count,
worker_id)`. Workers retain original context order within their shard. Freeze
the algorithm version, cost inputs, maps and hashes before any new GPU scoring.

No dynamic stealing, score-dependent scheduling or reassignment after a crash.
All ordinary passes reuse their cohort's assignment. The physical remapping
challenge flips the complete query's owner, so every layer processing that query
executes on a different physical GPU. Placement identity and challenge assignment
are separate fields; the existing `replica_shift` parameter must not ambiguously
serve as both. Native per-worker runtime identities must remain intact.

CPU-only metadata estimates made on the parent inputs:

| Cohort | A queries | B queries | A token-cost proxy | B token-cost proxy |
| --- | ---: | ---: | ---: | ---: |
| dev | 321 | 322 | 48,201,842 | 48,290,304 |
| regression | 4 | 4 | 638,034 | 630,516 |
| validation | 12 | 12 | 2,028,646 | 2,029,914 |
| boundary | 2 | 2 | 447,450 | 389,156 |

The dev difference is about 0.18% of the larger proxy and validation about 0.06%.
These estimates describe balance, not measured throughput or a promised speedup.

## Checkpoint and Merge Contract

Use a new plan ID, run directory and schema. Retain old checkpoints unchanged as
historical evidence and optional numerical comparison material; do not import
their scores into new formal raw. The old runtime and shard-local batch ordinals
make splitting its SQLite database or rewriting identities invalid. Any future
migration would need a separately specified and validated protocol.

Give each worker one exclusive SQLite writer and its own checkpoint directory.
Bind resume to the parent input hashes, new source and plan hashes, assignment
hash, worker ID, native runtime identity, physical UUIDs, pass, scoring profile,
record order and prompt/candidate identities. A mismatch fails closed. Preserve
committed transactions after interruption; recompute an uncommitted block.

The coordinator publishes a complete pass only after both workers have sealed
their results and independent checks establish exact coverage, unique ownership,
no duplicates, finite scores, expected runtime attribution and canonical order.
The top-level manifest binds both worker manifests and their native identities;
do not overwrite runner identities with a synthetic pool-wide identity. Partial
worker success never implies complete cohort or dev success. Merge atomically
into a separate output, leaving worker evidence intact and avoiding concurrent
writers. Repeated merge after a crash must reproduce the same canonical payload.

## Acceptance Gates

1. CPU tests prove deterministic ownership, exact candidate coverage, unchanged
   scientific inputs, rejection of wrong plans/runtimes/assignments, atomic block
   recovery, missing/duplicate shard rejection and deterministic merge. Test one
   worker failing, coordinator failure and independent deadline cleanup of both
   workers without touching unrelated processes.
2. Keep both models resident. On the original eight regression queries, execute
   the frozen A and B assignments serially (A then B), then concurrently with the
   same inputs and options. Compare each query on the same physical pair across
   the serial and concurrent passes. The concurrent pass may also supply the
   new regression baseline; the serial evidence is additional, not a substitute
   for an original check. The per-readout absolute tolerance remains `0.0001`.
3. Re-run all six original checks for the original 8 regression, 24 validation
   and 4 boundary queries under the new execution: identical-logits CPU-FP64
   reference, repeat, +64 masked padding, uncached independent prefix, candidate
   order, and physical remapping. These are 18 complete cohort/check results,
   not two independently duplicated cohorts counted as extra evidence.
4. Reference and repeat tolerance remain `0.0001`; padding, prefix, candidate
   order and remapping remain `0.0013427734375`, on all registered token, score
   and margin readouts. Restore canonical candidate order before comparison.
   Failed checks are sealed and stop expansion; no tolerance widening or
   automatic retry with a different numerical profile is allowed.
5. Verify actual different physical UUIDs for every layer in the remapping
   challenge, native per-candidate runtime attribution, FP32 operator checks,
   true batch-one causal geometry, EOS and full candidate coverage. Preserve
   per-device memory peaks. Exercise longest registered boundary inputs with
   both models resident, not merely a one-replica loading test.
6. Record synchronized forward intervals from both workers and demonstrate
   actual overlapping scoring, rather than inferring concurrency from loaded
   weights or utilization. Compare serial and concurrent wall-clock time for
   identical regression work and options, separating load/validation/merge
   overhead from useful candidate throughput. Report measured speedup and
   uncertainty; do not assume two replicas deliver twice the throughput.
7. Full dev starts only after every gate passes. Gold remains unread until
   complete raw, all new preflight and concurrency evidence, worker identities
   and seals are verified. The current single-pair independent auditor must
   reject this new schema until a separate dual-pair adapter has been validated.

## Cutover and Recovery

The following is a future adoption procedure, not an instruction to cut over the
current 14B run. The user's later decision above takes precedence for this run.

Prepare and CPU-test new code and the amended package while the old run advances.
After explicit approval, request a queue-level stop that prevents the old dev job
from starting, let the owned job commit its current block, and verify process,
writer-lock and GPU release before loading either new replica. Preserve the old
queue's lifecycle receipts. Never run the two queues against the same output.

The new queue must own the coordinator and both worker process identities,
including start times and descendants. Its separate watchdog survives supervisor
failure, stops new work at 19:55 and escalates only owned processes before 20:00.
One worker failure stops its peer and leaves both committed checkpoints available.
Recovery in another authorized window requires the exact same frozen identity;
the assistant's 30-60 minute status waits are not the deadline mechanism.

Changing the execution strategy incurs implementation and revalidation costs.
Do not promise that all preflight and full dev fit in the remaining reservation;
use measured end-to-end progress to update the estimate after the new benchmark.
