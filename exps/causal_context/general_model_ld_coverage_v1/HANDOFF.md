# Coverage-01 Execution Handoff

Superseded operational status, 2026-09-06: `coverage-01` completed, exited normally,
and passed the independent full-dev/all-CI audit. See
[the audited results](results/coverage-01/REPORT.md). The process/PID information
below is historical; do not use it to restart or signal a current process.

Handoff requested by the user on 2026-09-06 (Asia/Shanghai): leave the long-running experiment active after stability checks; the user will notify the assistant when it finishes. Active assistant monitoring stops after handoff. Final independent audit and scientific reporting remain pending.

## Existing Process

- Run: `runs/coverage-01`, under this experiment directory.
- Plan: `gmlcoverage-64e2986d5de1d9c9ff0ebc9c2d72944208b833ca9aacf115ec71a14403e0c6f4`.
- Main PID at handoff: `1056529`; tool session: `79780`.
- The process is in its own OS session, managed by the Codex app-server. It has NOT been migrated into tmux/nohup and has NOT been interrupted for handoff.
- Keep the machine and the Codex background service alive. Survival after shutting down that service is not guaranteed.
- Do not start another copy of the run while this process is active. The writer lock prevents concurrent writers, but a duplicate launch is unnecessary.

The frozen plan, input data, numerical profile, thresholds, and committed scores are unchanged. This document and the read-only status helper are operational aids, not amendments to the frozen scientific registration.

## View Progress

From any working directory:

```bash
/data/liaozijie/hate_speech_detection/.conda/stage1-p0/bin/python /data/liaozijie/hate_speech_detection/exps/causal_context/general_model_ld_coverage_v1/audits/coverage_status.py
```

For a refreshing terminal view:

```bash
watch -n 60 /data/liaozijie/hate_speech_detection/.conda/stage1-p0/bin/python /data/liaozijie/hate_speech_detection/exps/causal_context/general_model_ld_coverage_v1/audits/coverage_status.py
```

Stopping `watch` with Ctrl-C stops only the status viewer. It does not stop the experiment. The helper reads metadata and committed block counts, not score payloads or query gold. A progress snapshot is not an independent audit, a heartbeat, or proof that a process is alive.

The authoritative run manifest is [run_manifest.json](runs/coverage-01/run_manifest.json). GPU memory accounting on this host is unreliable; see [the monitoring note](audits/gpu-monitoring-note.md).

## What Happens Automatically

The existing process completes the registered 18 preflight passes. Only after every numerical gate passes does it expand to all 643 dev queries, 10,288 blocks and 174,896 candidates. It then seals raw results, releases its GPU workers and runs the registered CPU analysis. Any gate failure prevents full dev. There is no automatic threshold change or numerical-profile search.

Completion requires `run_manifest.json` to contain all of:

- `status: complete`
- `full_dev_started: true`
- `analysis_published: true`
- `preflight_report_sha256`, `raw_manifest_sha256`, and `analysis_manifest_sha256`

`raw_complete` means raw scoring finished but analysis has not yet been published. `running` is a persisted lifecycle status, not a live-process check. `failed` or `preflight_failed` means stop and retain the evidence; do not alter thresholds, delete outputs, or automatically retry. Notify the assistant on completion or failure.

## Interruption and Follow-Up

If the process disappears while the manifest still says `running` or `interrupted`, preserve all files and notify the assistant. The runner supports same-plan, same-runtime checkpoint recovery, but verify process exit and lock release before any restart. No recovery is being initiated as part of this handoff.

After the user's completion notification, the assistant will verify terminal manifests and all preflight/raw identities, run the prepared independent full-dev audit including the registered confidence intervals, and produce the Chinese results summary. The pipeline's `complete` status alone does not claim that this independent audit has passed.
