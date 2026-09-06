# Running Qwen3-14B Handoff: 2026-09-06, 24:00

Historical handoff. The user subsequently requested an early stop; the job,
supervisor and watchdog exited at 22:36:06 Asia/Shanghai. The commands and live
status description below document the earlier handoff, not an active run.
See the [current stop record](USER_STOP_20260906_2236.md).

The user requested execution control after stable startup. At 20:15
Asia/Shanghai, actual validation-prefix progress had advanced from 295 to
298/384 committed blocks. All eight previously sealed preflight checks were
revalidated under the unchanged runtime. The supervisor, job and independent
watchdog were alive. This confirms continuation, not complete preflight.

**Execution is handed back to the user.** The assistant stops active polling;
the detached queue and independent watchdog continue. There is no promise of a
later assistant notification or automatic closeout in this task. The queue will
run full dev only if all registered preflight checks pass. A numerical failure
blocks dev rather than changing the profile or loosening tolerances.

## Live Execution

- Same scientific run: `general_model_ld_coverage_14b_v1/runs/replication-01`.
- Same plan: `gmlrep-b64f97a77578d0c5e1f91adc1b4c509df409d3f56c04c4329b2ca6ed8718f2d5`.
- Queue: `runs/window-20260906-03-14b` in this directory.
- Preflight PID: 2539956, start ticks 340504898.
- Supervisor PID: 2539952, start ticks 340504888.
- Watchdog PID: 2539951, start ticks 340504888.
- Graceful stop: **23:55 Asia/Shanghai** (`2026-09-06T15:55:00Z`).
- Hard deadline: **24:00 on September 6**, equivalently **00:00 on September 7**
  (`2026-09-06T16:00:00Z`).

The watchdog is independent of this conversation and was ready before the GPU
job started. It stops new work, requests SIGINT, and escalates only against
registered owned process identities if needed. Do not kill only the supervisor,
reuse expired launch configurations, or run another writer against this output.
No dual-pair cutover or 27B execution was introduced.

## Inspect

Run in the host terminal so process identity checks use the correct namespace:

```bash
env PYTHONPATH=/data/liaozijie/hate_speech_detection/src /data/liaozijie/hate_speech_detection/.conda/stage1-p0/bin/python /data/liaozijie/hate_speech_detection/scripts/stage1/general_model_gpu_queue.py status --output /data/liaozijie/hate_speech_detection/exps/causal_context/general_model_ld_replication_queue_v1/runs/window-20260906-03-14b
```

- [Queue state](runs/window-20260906-03-14b/state.json).
- [Watchdog state](runs/window-20260906-03-14b/watchdog.json).
- [Preflight log](runs/window-20260906-03-14b/logs/14b-preflight.log).
- [Scientific run state](../general_model_ld_coverage_14b_v1/runs/replication-01/run_manifest.json).
- [Stable-launch acceptance](launch_acceptance_20260906_2400.json).

The 298/384 count is a handoff observation, not a live counter. At the deadline,
look for a terminal queue, watchdog `released` and an empty owned-process list.
Committed blocks remain recoverable; an uncommitted block is recomputed in a
newly authorized window. Do not interpret interruption as numerical failure,
or partial preflight as completed scientific results.

## Optional Early Stop

Only run this when intentionally requesting an early, checkpoint-preserving
queue stop. It asks the existing watchdog to stop the owned job and prevents
the dependent dev job from starting; it does not launch another process manager.

```bash
env PYTHONPATH=/data/liaozijie/hate_speech_detection/src /data/liaozijie/hate_speech_detection/.conda/stage1-p0/bin/python -c 'from pathlib import Path; from diagnostics.general_model_gpu_queue import request_stop; request_stop(Path("/data/liaozijie/hate_speech_detection/exps/causal_context/general_model_ld_replication_queue_v1/runs/window-20260906-03-14b"), "user-requested-early-stop")'
```

Then inspect queue/watchdog release before treating GPU ownership as free. No
early-stop request was executed as part of this handoff.
