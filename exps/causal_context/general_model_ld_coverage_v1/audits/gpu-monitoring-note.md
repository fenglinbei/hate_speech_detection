# GPU Monitoring Note

Date: 2026-09-06, during coverage-01 regression preflight. This is an engineering observation, not a numerical-gate exception.

The launcher reported HAMI `SET_TASK_PID FAILED` / `host pid is error` warnings, then initialized four workers and continued scoring. The runtime identity SHA is `28a3e693a90f6af32830df1c28753eeda80a05fe7e153725277389b6ae8102bc`.

A read-only nvidia-smi observation reported memory.used as 161,615 MiB for GPU 0 and 0 MiB for GPUs 1-3, while each physical total was 46,068 MiB and all four devices had 93-100% utilization. These values cannot represent ordinary per-device residency. Unsetting the environment variable LD_PRELOAD for nvidia-smi did not remove HAMI, because `/etc/ld.so.preload` globally loads `/usr/local/vgpu/libvgpu.so`. No preload, driver, allocation, or running experiment setting was changed.

Independent inspection of this run's runtime receipt found four distinct physical UUIDs, each matching its PyTorch logical UUID and its counterpart in numeric-04. Both old and new runtime receipts report torch total_memory_bytes=0 while nvidia-smi total capacity is 46,068 MiB. The reporting anomaly therefore predates this experiment, although that does not prove it has no numerical effect.

The sealed regression-r0 GPU 2 shard had 32 blocks / 544 candidates, attributed only to physical index 2 and UUID `GPU-278f5973-425f-514a-8a2d-db77f55d481b`. Candidate metadata recorded true batch 1, finite/boundary/reference checks, and a PyTorch allocated-memory peak of 33,208,580,096 bytes (approximately 30.93 GiB). PyTorch allocated memory is not a measurement of total device residency or available OOM margin.

Interpretation: the observed anomaly is consistent with virtualized monitoring/accounting. It is not evidence that all model replicas occupy GPU 0. Do not use these NVML memory.used values for per-device memory comparisons. Retain physical UUID and producer evidence, actual forward geometry, and all new repeat/arithmetic/padding/prefix/order/replica gates. No tolerance is loosened and no failure may be waived because the old run passed.
