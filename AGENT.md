# GPU Experiment Execution

## Performance Under Scientific Constraints

When running GPU tasks, proactively seek the highest validated useful throughput
available within the actual hardware, memory, software and reservation limits.
Do not default to the simplest or previously implemented execution strategy
without considering whether available GPUs could safely accelerate the work.
Scientific validity, reproducibility and data-use boundaries remain mandatory.

Before a substantial GPU run:

1. Inventory available devices, memory capacity, topology and conflicting jobs.
   Distinguish model placement used to fit weights from concurrent computation
   that improves throughput. Do not treat an unreliable utilization or memory
   counter as proof of correct placement or free capacity.
2. Evaluate suitable data, model and hybrid parallel strategies, including
   multiple independently sharded model replicas when a model cannot fit on one
   GPU. Explicitly account for otherwise idle allocated GPUs. For example, a
   model requiring two GPUs on a four-GPU host merits evaluating two two-GPU
   replicas, not only one two-GPU replica or one four-GPU model.
3. Prefer proven local implementations where applicable. Compare useful work per
   wall-clock time, including loading, synchronization, validation, checkpointing
   and recovery costs; GPU utilization alone is not the objective. Use bounded,
   authorized engineering benchmarks without inspecting gold or selecting on
   favorable scientific effects. Do not claim optimality without measurements.
4. Preserve registered inputs, candidates, score definitions, precision, numeric
   thresholds and analysis boundaries. Do not silently enable quantization,
   reduced precision, different batching, caching, approximate algorithms or
   automatic profile search to obtain speed. An execution change that conflicts
   with a frozen registration requires an explicit amendment and validation.
5. Freeze deterministic task assignment and execution identities before scoring.
   Validate numerical equivalence, concurrent-versus-serial behavior, physical
   placement and exact result coverage before expanding a new parallel strategy.
   Failed checks do not authorize widening tolerances or switching profiles.
6. Keep one writer per checkpoint store, atomic complete scoring blocks and
   source/plan/runtime/assignment-bound resume. Never mix incompatible old and
   new execution results by editing identity metadata. Preserve prior evidence;
   any proposed migration needs separately specified provenance and validation.
7. Retain independent reservation watchdogs, identity-scoped process cleanup and
   a graceful-stop margin. More parallelism must not create unowned GPU workers,
   kill unrelated jobs or extend beyond an authorized reservation.

Explain the selected strategy, expected throughput and remaining limitations to
the user. If a potentially faster safe strategy is not used, state the concrete
reason and the validation or implementation needed, rather than leaving the
hardware tradeoff implicit. Do not promise completion inside a window when
measured throughput does not support that estimate.
