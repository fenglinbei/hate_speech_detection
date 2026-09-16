# Merged-L 14B / 27B Replication and GPU Window

Registered: 2026-09-06, Asia/Shanghai. The user's new authorization adds Qwen3-14B and Qwen3.8-27B to the merged-L study, after the active Qwen3-8B full-dev run finishes. This amendment does not modify the frozen 8B experiment, its dependencies, data, scores or numerical thresholds.

## Scientific Scope

Each new model uses all 643 frozen dev queries, hate and group classification, and the same eight conditions: C0, CLnew, CD, CLDnew, PLnew, PD, CLq, CLqD. There are 10,288 blocks and 174,896 candidates per model. The six core conditions and two newly scored Lq references are retained. No extraction, test access, fine-tuning, activation intervention or model substitution is authorized.

Lnew remains the globally deduplicated, frozen-ID-sorted union of query and ten fixed-demo lexical hits. The 833 entries, senses, selected hit IDs, demonstration IDs, order and task answers are inherited from the frozen 8B inputs. Matching and retrieval are NOT rerun under the new dependency environment. All substantive messages are byte-identical to the 8B messages; only model-native chat templates, tokenization and tokenizer-specific neutral control length matching are re-materialized. Neutral controls keep the original absolute 8-token / relative 2% tolerance. Inputs are never shortened to fit.

CPU construction amendment, before any new-model GPU score: the inherited additive placebo constructor cannot shorten its initial neutral template. For Qwen3.8-27B, query 3072's group PD block initially had 206 tokens against a 188-token source, outside the unchanged 8-token allowance. New replication inputs therefore register a compact complete-sentence pool for failed neutral controls only: `\u706f\u4eae\u3002`, `\u706f\u706d\u3002`, `\u95e8\u5173\u7740\u3002`, `\u706f\u5f00\u7740\u3002` (lamps on/off, door closed, light on). All ten demonstration slots, task output placeholders, dictionary entry/sense layout and whole-sentence requirement remain intact. Candidate compact templates use the original additive whole-sentence length matcher, selecting minimum complete-block token error with fixed pool-order tie-break. Valid original controls are not changed. Any remaining failure still blocks construction; no token tolerance is widened. The compact policy/material hash, original failed construction and applied record IDs are recorded. The first 14B CPU-only build is retained as an unexecuted source revision and both model plans are rebuilt with this registered adapter. No original 8B file or output is changed.

Canonical answer strings and group ordering are unchanged; token IDs and lengths are frozen separately for each tokenizer. Primary scores exclude EOS. The auxiliary EOS target is the tokenizer's assistant end token (`im_end`), not an arbitrary member of the model's generation stop-token list. All generation stop IDs, the selected EOS and pad are recorded. This distinction matters for Qwen3.8-27B, whose generation config permits both `im_end` and `endoftext`.

The original ten contrasts, six Lq/gold-size strata, 240 CI targets, 10,000 seed-42 paired query bootstrap and gold/EOS/cardinality auxiliaries are unchanged. Query gold is deserialized only after complete raw results and preflight evidence are sealed and verified. Candidate score payloads remain model-specific; no scores are imported from another model or protocol.

Qwen3.8-27B is the specified native vision-language checkpoint with the `qwen3_5` hybrid architecture, used with text inputs only. Comparing it with Qwen3-8B/14B is not a controlled parameter-count-only comparison. Changes in architecture, training, tokenizer and execution environment must be reported. No image/video data or vision task is introduced.

## Sources and Runtime

Read-only model sources are explicitly allowed outside the workspace at `/data/models/Qwen3-14B` and `/data/models/Qwen/Qwen3.8-27B`. Exact config, tokenizer, weight index and every referenced safetensors shard are SHA256-bound before scoring. No symlink is used to bypass the original 8B package's source-root rules.

The new environment is `/data/liaozijie/conda/accelerate-fc-mrec-clean/bin/python`. Package versions and new/borrowed source identities are frozen in each plan. The old `.conda/stage1-p0` environment is untouched.

Use FP32 parameters, eager attention, TF32 disabled, true candidate batch 1, full uncached sequence teacher forcing and full-vocabulary projection only at answer/EOS prediction positions. Normalize logprobs in FP32 and aggregate in FP64. No quantization, CPU/disk offload, KV cache, mixed-model scores, automatic precision changes or automatic profile search.

For Qwen3.8-27B, use the installed Transformers native torch DeltaNet implementation, with its optional fused kernels absent. Audit FP32 model tensors and floating CUDA operator inputs/outputs during scoring. The ordinary next-token language head is used; the checkpoint's auxiliary `mtp.*` tensors are not executed by the Hugging Face model class. Their exclusion is explicitly recorded, while every non-MTP parameter, including the unused-in-forward vision encoder, must be accounted for. All checkpoint shards, including those containing MTP, remain source-hashed. This is not a transformed or replacement text-only checkpoint.

14B uses one layer-sharded model on physical GPUs 0/1; the physical remapping challenge uses GPUs 2/3. 27B uses one layer-sharded model on GPUs 0/1/2/3, with a cyclically shifted placement for the remapping challenge. Device maps are explicit and frozen, not chosen by automatic free-memory heuristics. All devices are synchronized and their memory peaks recorded. This is model parallelism, not the four independent single-GPU replicas used for 8B.

## Preflight and Gates

Retain the original regression 8 and validation 24 query IDs. Select at most four additional boundary representatives from frozen metadata after each tokenizer's materialization, using the inherited empty-union / maximum union / dictionary length / complete input rules and ID tie-break. The main population remains 643 regardless of engineering cohort size. Check all 174,896 prompt/candidate boundaries and complete-answer/EOS plus 64-padding lengths on CPU before scoring, with an 8,192-token limit.

For every cohort, run baseline with identical-logits CPU-FP64 reference, repeat, +64 masked padding, uncached independent-prefix reference, candidate-order challenge, and actual physical layer remapping. The first five passes for all cohorts can use one baseline model load; remapping passes follow with the alternative frozen map. All six checks are still required for every nonempty cohort. Completed atomic blocks may be reused only under exactly matching plan, runtime, profile and prompt identities.

Repeat and same-logits reference tolerances are 0.0001. Padding, prefix, candidate order and remapping tolerances remain 0.0013427734375 across all registered token/score/margin readouts. The inherited E8 value 0.00067138671875 is provenance, not a newly measured E14/E27 calibration. Applying this fixed engineering threshold to a new model does not assert that 8B validated that model. Every new model must independently pass it; failures are sealed without widening tolerances or trying another profile.

Passing preflight requires complete candidate coverage, finite values, correct causal shift, canonical order restoration, explicit EOS, uncached true-batch-one geometry, source/runtime identities and actual different physical placement. A merely successful load or a few generated examples does not count as passing this numerical preflight.

## Queue and Deadline

Wait for the active 8B run to reach `complete` and release its writer/GPU processes. Then run 14B preflight and 27B preflight sequentially. A failed model does not prevent recording the other model's preflight, but neither full-dev job starts until both preflights pass. Full-dev order is 14B, then 27B. Jobs not reached within the GPU window remain unstarted; interrupted jobs retain committed checkpoints. No failed job is automatically retried.

The GPU reservation ends on **2026-09-06 at 10:00 Asia/Shanghai**, equivalent to **2026-09-06T02:00:00Z**. Stop starting jobs and request graceful interruption at **09:55** (`01:55:00Z`). A separate detached watchdog retains ownership records and escalates to terminate/kill only this queue's processes if they fail to exit before the deadline. It never kills unrelated GPU users or resets GPUs. Supervisor failure does not cancel the watchdog.

The queue, watchdog, immutable launch config, job logs and lifecycle receipts are separate from scientific raw artifacts. Graceful interruption is operational, not a numerical failure, and does not authorize changing a model's frozen plan. Hard termination preserves committed SQLite transactions but may lose the currently uncommitted block. Keep partial artifacts for recovery in the next authorized GPU window.

After both preflights pass and a full-dev job is stably running, the assistant may hand execution back to the user with progress, log, deadline and recovery instructions. Full scientific results require completed raw and analysis plus subsequent independent audit; queued or interrupted work is not reported as completed.
