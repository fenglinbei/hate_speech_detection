# 14B / 27B GPU Window: 2026-09-06

Current state: **stopped at the user's request at 22:36:06 Asia/Shanghai**.
All owned processes and writer locks are released. The user will select an
acceleration approach before a new run; no automatic restart or benchmark is
authorized. Nine of 18 preflight checks have sealed evidence and validation
candidate order retains 289/384 blocks. Full dev has not started. Queue03 records
a stop-time lifecycle failure with stale job fields despite verified shutdown;
see the [stop record and checkpoint acceptance](USER_STOP_20260906_2236.md).

The [24:00 renewal](WINDOW_20260906_2400.md) and [20:15 running handoff](RUNNING_HANDOFF_20260906_2400.md)
are historical. No dual-pair cutover or 27B job occurred.

Prior window closed: the [14B-only continuation window](WINDOW_20260906_2000.md)
released all owned processes at 19:55:02 Asia/Shanghai, before its 20:00 deadline.
Queue `runs/window-20260906-02-14b` is stopped. Eight of 18 preflight checks passed;
validation prefix retains 295/384 blocks, and full dev has not started. The
original single-pair plan and checkpoints are preserved. See the [latest closeout](HANDOFF_20260906_2000.md).

Earlier morning window: the independent watchdog safely released all owned processes at
09:55:01, before the 10:00 deadline. 8B full dev and independent audit completed;
14B preflight is checkpointed and interrupted; 27B GPU work and both new-model
full-dev jobs remain unstarted. See the [final handoff](HANDOFF_20260906.md).

Operational namespace only. Scientific plans and outputs remain in the separate
`general_model_ld_coverage_14b_v1` and `general_model_ld_coverage_27b_v1` directories.
The frozen 8B plan and outputs are not modified by this queue.

## Morning Order and Deadline

1. Wait for 8B `coverage-01` to finish, release its writer lock, and exit.
2. Run 14B numerical preflight, then 27B numerical preflight.
3. Only if BOTH preflights pass, run 14B full dev, then 27B full dev.

Every full dev covers 643 queries, two tasks and eight conditions:
10,288 blocks and 174,896 candidate scores. Preflight cohorts are engineering
checks, not a reduced scientific population. Queued work is not completed work.

The independent detached watchdog stops new jobs and requests interruption at
09:55 Asia/Shanghai. The hard GPU cutoff is 10:00 on 2026-09-06. It escalates
signals only for identity-verified processes owned by this queue. It does not
depend on the assistant, terminal polling or a future wakeup.

## Inspection

From `/data/liaozijie/hate_speech_detection`:

```bash
env PYTHONPATH=src .conda/stage1-p0/bin/python scripts/stage1/general_model_gpu_queue.py status --output exps/causal_context/general_model_ld_replication_queue_v1/runs/window-20260906-03-14b
```

The run directory contains immutable launch configuration and source snapshot,
`state.json`, `launch.json`, `ownership.json`, `watchdog.json`, and individual job
logs under `logs/`. `watchdog.json` must report release with no remaining owned
processes before declaring this queue's GPUs released. In the earlier morning
window, a separate identity-bound deadline guard for the then-running 8B process was recorded in
`existing-8b-watchdog-01` and is now released. It protected the exact registered
8B process and its observed descendants independently of that queue. Its `state.json` must also
show `released` with no remaining processes. Nine host CPU lifecycle tests passed
before launch. This kernel lacks pidfd support, so signaling uses explicitly
recorded PID/start-time/session rechecks; the small check-to-signal race is a
documented platform limitation.

## Interrupted Work

Retain partial artifacts and committed SQLite blocks. Do not delete a failed
preflight, change precision or tolerances, or import scores from another model.
Do not rerun this expired window configuration. Resumption requires a new GPU
window and launch configuration, the same validated plan/runtime identity, and
explicit authorization. An interrupted run is distinct from a numerical failure.

While only computation remains, the assistant waits for 30-60 minutes at a time,
checks state once on waking, and stays quiet unless work completes, fails, or
needs a decision. Scientific findings require sealed results and independent audit.

## CPU Acceptance

The joint replication loader, package, execution and sharded-runtime suite passed
58 tests. The detached queue passed 12 process-lifecycle tests. These are CPU
acceptance results, not evidence that either new model passed GPU preflight.

Before any new-model GPU score, one 27B placebo construction failure was fixed
using a registered fixed complete-sentence fallback. All 2,572 27B neutral
controls passed a CPU sweep; only `3072:group:PD` used the fallback (source 188,
control 194 tokens, difference 6 within the unchanged allowance of 8).
The original failed-build log and the first unexecuted 14B CPU plan are retained.
Both execution plans are rebuilt with the same final source snapshot.
