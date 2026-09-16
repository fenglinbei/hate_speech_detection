# Human Review Workbenches

## Q01 six-module run and result verification complete (2026-09-16)

NEW `q01-module-refinement-results-v1/current.json` selects frozen-01/run-01/
results-01 and the hashed assistant interpretation under
`reviews/q01-module-refinement-v1/interpretation-01`. The same run completed all
12 passes at 12:38:10 Asia/Shanghai, all four resumed workers exited normally
at 12:38:11, and the supervisor completed CPU analysis at 12:38:33. At 14:21
all owned PIDs were absent and all four GPUs had zero memory/utilization.
The run and window-20260916-poll-02 are TERMINAL COMPLETE; do not restart them.
Earlier preparation, execution and partial-progress selectors remain historical.

All 67,200 candidate evaluations / 100,640 scoring forwards completed. Actual
prompt-only captures are 4,976, including 368 extra captures due to resuming.
Both invocations and all 1,711 reused checkpoint requests are preserved; all
eight old reference/repeat shard byte prefixes match the closeout hashes.
Repeat, member order and physical replica differences are zero. Science padding
max is 0.0003204345703125; prefix is 0.000057220458984375. All raw gates and six
derived gates (70,848 readouts each) pass unchanged tolerances. Full CPU check
and exact analysis-byte reconstruction pass. The independent 60-digit Decimal
audit checks 7,820 atoms, 70,848 expression values AND bounds, all 12 module/metric
rankings, 36 secondary exceptions and 10,880 raw predictions; max expression
error is about 2.37e-15 and bounds match exactly. Audits/final-closeout-01.json
and resource-release-final-01.json bind the evidence. No new GPU forward was
performed during this result review. Preserve original and new pinned sources.

Formal original-sum nomination selects attention35/pre_answer for C (three
eligible units; worst gain 4.0501%) and MLP34/pre_answer for I (only eligible
unit; worst gain 58.6958%). C's original-sum gains are only 4.05-7.67%; A/B
forward R reverses in all four groups and reverse R also reverses at F2:O-N2.
There are 20 C secondary exception rows. MLP34 I is aligned and numerically
resolved in all 192 view/group/direction rows, but only 176 improve residuals
beyond bounds. Original-mean F1:O-N2/K and recalibrated-NCC F2:O-N2/K overshoot
to 2.061x/2.340x targets; all 16 I secondary exceptions remain. A/B forward and
reverse I gains are smaller, 17.53-32.91% and 14.51-26.01%. Fixed-prior NCC
equals original mean algebraically; it is not independent replication.

Whole-block34 historical C/I recovery remains 99.45-99.82% / 88.93-98.05%; block35
output diagnostic is exact in original sum. Module shares are not additive.
MLP34 demo-9 controls are unresolved; attention35 demo-9 is structural zero.
Neutral I fails to recover under MLP34 in original/NCC, but A/B-forward neutral
I also transfers partially; do not claim exclusive lexicon/glyph specificity.
For both selected units, R patched non-hate is 0/8 in recalibrated NCC and A/B
forward. Directional I transfer is not functional label repair. Original-sum
R non-hate is 6/8 attention35 and 8/8 MLP34, versus 4/8 recipient baseline;
these are correlated arms of one exposed query, not independent rescued cases.

Read INTERPRETATION.md and all-view/exception/arm tables together. Future fixed
site transfer to exposed Q05/Q03/Q04/old3169 needs a separate protocol/freeze;
no transfer, head/neuron scan or serial-path intervention has started. All old
reference/use/material/result pointers and human fields stay unchanged. Q01
has bulk-adopted non-hate and no original Gold; mechanism_ready remains false.

## Q01 module resume running; no fixed deadline (2026-09-16 09:05)

The user newly authorized 15-minute idle checks and explicitly confirmed
"空闲后运行至完成，无固定截止时间". This supersedes the expired 01:00 cutoff
for this continuation. NEW `q01-module-refinement-v1/execution-02.json` selects
`reviews/q01-module-refinement-v1/window-20260916-poll-02/config.json`, SHA256
`e12b0d29628dbb40135b5f4e0180cb5f1d510618020e9d88c2e6f07e3e9254bd`.
The server supervisor PID 978785 found all four original GPUs idle at 09:00:09
Asia/Shanghai and launched parent 978794. It is RUNNING, not waiting for another
window. Do not duplicate this executor or restart historical supervisors.

The original run-01 entered invocation a91d88ef-67f6-4643-af4e-65a74e6ffe03 at
09:00:12, with no stop epoch or pass cutoff. Workers 978899/978900/978901/978902
use the original GPUs 0/1/2/3, UUIDs and numerical runtime. At 09:05:31 all six
owned supervisor/parent/worker PIDs were alive and all four shards had new scores.
The repeat pass had 2,618 saved requests (679/639/675/625), including all 1,711
old checkpoints and 907 new requests. All eight previous reference/repeat shard
byte prefixes match their closeout hashes exactly, and the sealed reference
shards are unchanged. The receipt is the new window's `confirmed-scoring.json`.
These are running snapshots; read its `state.json`, `gpu-run.log` and
`run-01/run_manifest.json` for current progress.

Separate source-pinned `idle-resume-02/manifest.json`, SHA256
`3fe4d29f787f203e4a32b121a174c4c929ed46ca3caf53e512c7010bb49d53be`, adds
`q01_module_poll_v2.py`, `schedule_q01_module_poll_v2.py` and
`run_q01_module_polled_v2.py`. Nine CPU orchestration tests and the original
CPU resume-check pass. The wrapper exposes the unchanged executor's existing
no-deadline mode; all scientific code, freeze, tolerances and earlier artifacts
remain unchanged. Do not edit these new pinned files either. The fixed polling
interval is 900 seconds; its first check was already idle. This is a server
process, not an app chat wakeup.

The run will complete the remaining engineering gates before science, then
release owned workers and perform the original CPU verification/analysis.
Failures stop automatic retries. Creating CANCEL in this NEW window directory
cancels only its owned controller/run; a forced cancellation is not a guaranteed
resumable checkpoint. Keep the same plan/allocation and all saved requests.
At the snapshot only engineering-reference was sealed; full GPU qualification
and module nomination were still pending. All reference/use/human fields and
mechanism_ready=false remain unchanged. Earlier closeout and preparation
selectors below are historical and do not describe the current running state.

## Q01 module window closed; CPU delivery complete, GPU qualification partial (2026-09-16)

NEW authoritative progress: `q01-module-refinement-v1/engineering-progress-01.json`
selects the closeout of `reviews/q01-module-refinement-v1/run-01`. The run paused
normally at 00:52:01 Asia/Shanghai; all four workers exited with code 0. At00:53:38
all six owned supervisor/parent/worker PIDs were absent and all four GPUs showed
zero memory/utilization. The supervisor is terminal paused_at_window_end; do not
restart it or launch GPU work without a new window authorization.

Only engineering-reference is sealed: 2,880 requests /5,760 candidates. All96
historical replays and all2,784 engineering expectations have zero maximum error;
same-logits FP64 error is3.3418916984828684e-06 under the unchanged0.0001 tolerance.
Independent60-digit Decimal reconstruction covers86,400 candidate-score scalars
and all engineering expectations, max arithmetic error2.96e-16. The repeat pass
has1,711 saved requests (per GPU444/417/443/407), all zero difference from the
reference on the checked prefix. CPU checkpoint/capture/producer audit passes.
Total candidate evaluations9,182; additional prompt-only captures768. Remaining
engineering25,378 plus science32,640 =58,018 candidates. Engineering is1/6 passes
complete, NOT fully GPU accepted; science has no new scores or module nomination.

The user conditionally authorized full science if all scoring/acceptance/release
could finish by01:00. The measured ordinary-pass estimate alone was160 minutes
for the full matrix (prefix adds time), so science was not entered. Resume this
same paused run in a new authorized window, with the same four devices, UUIDs,
runtime and source-pinned plan. Never repeat sealed or checkpointed requests.
The delivery README and audits/window-closeout-01.json contain the exact state.
Original/current preparation selectors remain historical; module mechanism_ready
stays false, all human/reference/use fields are unchanged. Below is the launch
record, not the current running state.

## Q01 six-module preparation and engineering window (2026-09-16)

The user authorized the six-module delivery and pre-full-run preparation, with
GPU engineering/tests until 01:00 Asia/Shanghai; full science is allowed in this
window ONLY if complete scoring, acceptance and release can finish before 01:00.
`q01-module-refinement-v1/current.json` selects a NEW independent freeze under
`reviews/q01-module-refinement-v1/frozen-01`, manifest
`63683f4ebedefbf710f0cb3bfd8aab2ef2b865fd6a0abf779eb84c3a212c05c4`.
Do not edit this freeze or its new source-pinned q01_module_* implementation.
All previous freezes, code, references, human fields and selectors are unchanged.

Six units are layers33/34/35 x attention/MLP at pre_answer, with all four contrasts,
both surfaces/directions, original/NCC/A-B and probes retained. 96 source prompts,
192 boundaries and 672 position records are exact parent bytes. The matrix has
5,504 requests including 768 primary and 320 whole-block historical bridges.
Twelve passes cost 67,200 candidates / 100,640 scoring forwards; engineering costs
34,560 and science32,640 candidates. Capture calls are additional. Sixteen CPU
tests, byte reconstruction and the independent353-source input audit pass.
These are not GPU numerical acceptance. Module outputs are patched before their
residual addition; final-layer earlier-position nonpropagation controls apply to
attention35/MLP35 only. Nomination keys include module, and an already completed
engineering-only stage cannot accidentally continue into science. Keep fixed and
fully recalibrated NCC separate, all exceptions, and mechanism_ready=false.

The new supervisor PID80715 launched parent80719 at00:29:17, using
`reviews/q01-module-refinement-v1/window-20260916-01/config.json`; config SHA
`0c85e56dbe7a24a6d9b0210b28b9d7b582f2261855a7b88bb02c11fe99ea37ed`.
Read its state and NEW run-01 manifest before acting; do not duplicate executors.
It defaults to engineering only, checkpoints00:52 and falls back at00:56 to
release owned processes before01:00. CANCEL in that window directory cancels
only its owned run. This is a server process, not an app chat wakeup. Historical
windows and the first-round completed run must not be restarted.
Use scripts/review/run_q01_module_refinement.py validate/resume-check/
engineering-check/check/analyze (CPU) or explicit run --phase engineering|full.
Only run loads weights. A paused run requires the same allocation/UUIDs/runtime.
Consult the new work README and state/logs for current, authoritative progress.

## Q01 full local-mechanism run and interpretation complete (2026-09-15)

`q01-local-mechanism-results-v1/current.json` is the NEW completed-result selector
for frozen-01/run-01/results-01 and the hashed assistant interpretation under
`reviews/q01-local-mechanism-v1/interpretation-01`. Original preparation, interim,
reference/use/material/result pointers and all human fields remain unchanged.
The 12-pass run completed at 23:26:28 Asia/Shanghai; all night workers exited
normally at 23:26:29, automatic analysis ended 23:27:16. Post-exit inspection found
all owned processes absent and all four GPUs at zero memory/utilization.
Both invocations and all 3,879 reused checkpoint requests are retained. Do not
restart this terminal run or any completed/cancelled window supervisor.

All 144,192 candidate evaluations / 214,768 scoring forwards completed. Actual
prompt-only captures total 4,647, including 39 extra captures due to resuming.
All engineering/scientific numerical gates pass unchanged epsilon. Repeats,
member order and physical replica error are zero. Science padding maximum is
0.00038909912109375 and prefix is 0.000057220458984375. Each of six scientific
derived gates covers 191,808 readouts. Exact final analysis reconstruction passes.
The independently implemented 60-digit Decimal audit verifies 20,612 atoms,
191,808 expression values AND bounds, all selection rankings and 27,776 raw
predictions; maximum expression error is about 2.55e-15 and bounds match exactly.
All 21,312 interim C/I metric rows match the final reference-pass analysis.
The original receipt-format defect is fixed only by the source-pinned v2
amendment; keep original and new frozen code immutable. Use the v2 entry point.

Formal original-sum nomination selects block 34 / pre_answer for BOTH C and I,
deduplicated to one unit. C has 13 eligible units; I only block 30 and 34 at
pre_answer. Selected-unit worst-group/direction gains are 99.45% C and 88.93% I.
All five main views, EOS auxiliaries and all single/LOO probes preserve aligned,
numerically resolved improvement at this selected unit. Fixed-prior NCC remains
algebraically identical to original mean, not independent replication. All
secondary/control rows remain. Neutral N1/N2 differences also transfer almost
fully at this site, but their absolute targets are much smaller. The layer-35
output diagnostic is nearly exact; same-layer demo-9 control is unresolved.
Interpret this as local downstream decision-state transfer, not a specialized
sense-selection/glyph circuit, uniqueness, necessity, or a tested serial path.
Block-15 query_hehe has asymmetric C and four N-to-O I overshoots; its fully
recalibrated NCC is unavailable. Do not replace the registered winner with it.

Label flips are separate. For selected-unit R/K, patched non-hate counts out of
8 correlated arms each are sum 8/4, mean 8/6, recalibrated NCC 8/0, AB forward
8/0, AB reverse 8/8. Original mean K has two donor-label mismatches at F2:O-N1/A
and F2:O-N2/A sharing donor margin -0.00220642; patched margins are +0.00902939
and +0.00810649. Preserve these although the nomination's C/I direction-exception
list is empty. These are conditions on one exposed Q01, not rescued independent
cases. Its bulk-adopted non-hate reference has no original Gold.

Next registered refinement, if separately authorized/frozen, is layers 33/34/35
at pre_answer crossed with attention/MLP: SIX distinct units after C/I dedup.
No refinement input freeze or new GPU task has been started. Functional transfer
to Q05/Q03/Q04/old3169 remains future, exposed-case work. Mechanism_ready stays
false. Earlier running and pending notes below record their preparation times.

## Q01 immediate resume authorized and launched (2026-09-15 21:23)

The user confirmed GPUs were idle and explicitly requested immediate execution.
All four original GPUs were verified at zero memory/utilization, with no compute
processes. The previous waiting window `window-20260915-night-01` was cancelled
through its CANCEL file and exited normally at 21:23:00 before any replacement
executor was launched. It is historical and must not be restarted.

The active controller is now `reviews/q01-local-mechanism-v1/window-20260915-night-now-01`,
config SHA256 730343150411dc076abae45370571d4d847c1537199b7d5b04842c75c0f5d413.
Supervisor PID 3880634 launched run parent PID 3880645 at 21:23:01. The original
run-01 passed its nine sealed-pass reconstruction checks and entered a second
invocation c1b31d66-b8f9-4ad5-b51e-003d49180f68 at 21:23:27. All four workers
entered science-prefix at 21:24:20. At 21:24:59, all four shards had new complete
scores (35 new requests total), and the original 3,879-request checkpoint byte
prefixes matched their CPU receipt hashes exactly. Worker PIDs are
3882027/3882034/3882057/3882060 on GPUs 0/1/2/3; GPU utilization was 99–100%.
The verification is in the new window's confirmed-scoring.json. Check this NEW
window's state/logs and run-01/run_manifest.json for actual live progress;
do not duplicate the executor.

The same pinned receipt-json-fix-01/v2 implementation and original four UUIDs,
scientific input, checkpoints and numeric runtime apply. The deadline is still
2026-09-16 00:45 checkpoint / 00:55 owned-process fallback / before 01:00 GPU
exit, Asia/Shanghai. Creating CANCEL in the NEW window directory cancels this
controller/run. No reference or scientific result pointer was modified. Formal
analysis and nomination still require all twelve sealed passes and all gates.

## Q01 receipt fix complete; 21:00–01:00 resume scheduled (2026-09-15)

The user explicitly authorized the serialization fix and resuming at 21:00,
stopping before 01:00. Separate `reviews/q01-local-mechanism-v1/receipt-json-fix-01`
holds the source-pinned amendment (manifest SHA256
66bf98cd1df2a53e534e2f86400ebe7b5de0969f1affbdf10355ee3fcfee7ffd).
Do not edit these new pinned files or the original freeze/implementation.
`q01_mechanism_execution_v2.py` differs from the old executor only in the exact
JSON normalization of reconstructed acceptance before equality comparison.
Seven regression tests and actual CPU revalidation of all nine sealed passes
and all 3,879 partial-prefix requests passed. Partial max error is
0.000057220458984375 under unchanged epsilon 0.0013427734375. The remaining
budget is 33,906 candidate scores. No new GPU forward occurred in preparation.
The earlier note requiring a fix is now satisfied by this separate amendment.

Use `scripts/review/run_q01_local_mechanism_v2.py` for validate/resume-check/run/
check/analyze. It checks the amendment including in spawned worker bootstrap;
the scientific plan, full source lease, original worker/scorer, same-GPU/runtime
requirements and all-twelve-pass analysis gate remain. The old entry point
retains the historical receipt-comparison defect and must not resume this run.

Night supervisor PID 3826856 was launched and verified sleeping at 20:55,
using `reviews/q01-local-mechanism-v1/window-20260915-night-01/config.json`
(SHA256 7527700f57fec8d9f31090e71d45b9e09cbf68b48e065f15b59f45631a2dd3ae).
It starts checks at 2026-09-15 21:00 Asia/Shanghai, polls every 30 minutes if
busy, requires all original [0,1,2,3], checkpoints at 2026-09-16 00:45 and
enforces an owned-process fallback at 00:55 before the 01:00 user deadline.
The previous supervisor and all its worker PIDs were verified absent.
Read this new window's state.json and run-01/run_manifest.json before acting;
do not create a duplicate. Creating CANCEL in the new window directory cancels
only this supervisor/owned run. This is a server process, not an app automation
or automatic chat wakeup. If complete, it releases workers then performs CPU
final verification/analysis. Failure stops automatic retries. New source fixes
or module refinement require a new version; old references, pointers and human
fields remain unchanged and mechanism_ready stays false.

## Q01 paused window and provisional readout (2026-09-15)

The scheduled run started at 10:30 after earlier availability checks were busy.
It paused normally at 16:45:03 Asia/Shanghai, with all four owned workers exiting
with code 0. Nine passes are sealed: all six engineering passes and science
reference/repeat/padding. Science-prefix has 3,879 / 6,944 checkpointed requests;
member_order and replica remain. Supervisor state is paused_at_window_end.
Do not relaunch without another authorized GPU window; preserve the same run,
plan, allocation and UUIDs when an eligible resume becomes possible.

The user asked for preliminary results. Separate
`reviews/q01-local-mechanism-v1/interim-01/INTERIM.md` and `snapshot.json` describe
Gold-free reads of the complete science-reference frame, excluding incomplete
prefix scores. All nine sealed artifacts, captures, producers and numerical gates
were rechecked on CPU. No query reference was parsed, no GPU forward or formal
nomination was executed, and no historical/final-result pointer was changed.
All 10,656 comparisons, 21,312 C/I rows and all secondary/control views are saved.

This audit found an existing frozen-checker serialization defect: the derived
largest-error key is a tuple in memory but a list after JSON loading. Direct
object equality in verify_completed_passes raises gate reconstruction differs
for science-reference and science-padding, despite exact equality of every
numeric/text value and gate after JSON round-trip. The interim verifier checks
the exact JSON representation and records those two sole type differences.
The original failing log is retained. **Before resuming or final check/analyze,
prepare a separately versioned, source-pinned serialization fix; do not edit
frozen code, change numeric tolerances, or rerun sealed scores.** This is a
receipt-comparison defect, not a failed model numerical gate.

Descriptively, block 34 / pre_answer transfers C/I in all four groups/both
directions across all views/probes; original-sum closeness gains are
99.45–99.82% (C) and 88.93–98.05% (I). This is not yet a formal nomination.
Neutral N1/N2 differences also transfer almost fully there; their absolute
targets are much smaller, and the block-35 output diagnostic is nearly exact.
Thus preserve general decision-state transfer as an explanation. Block 15
query_hehe C is direction-asymmetric and its N-to-O I overshoots in all four
groups; query_hehe has no recalibrated NCC. Do not call these a sense-selection
circuit, proof of exclusive glyph causation, correction, or a tested serial
path. All prior source/human fields and mechanism_ready=false remain.

## Q01 local mechanism freeze and scheduled window (2026-09-15)

The user authorized the formal protocol, per-prompt positions, bidirectional
interventions/controls, scoring, hooks and acceptance schedule. The separate
`q01-local-mechanism-v1/current.json` selects `reviews/q01-local-mechanism-v1/frozen-01`.
Its manifest is `e9391ef24ff6cf136840040d21430890098dcf266a98c66e899cc111bb9fc3c0`.
Do not edit this freeze or any newly source-pinned implementation in place.
All earlier reference/use/material/result pointers and human fields are unchanged.

This is exploratory localization on the single exposed FD-3169-Q01, with no
original Gold. Eight layers [0,5,10,15,20,25,30,34] x four main roles preserve
the four F1/F2 O/N1/N2 contrasts, both demo surfaces, both intervention directions
and all three encodings. 96 source prompts, 192 candidate boundaries, 672 role
records, 96 pair proofs and 11,920 request records passed CPU reconstruction and
an independent 315-source audit. All 24 CPU tests pass, including synthetic
12-pass gates, causal toy hooks, a tiny random CPU Qwen3 structural check, actual
prompt-only caches, checkpoint reuse, capture provenance and independent Decimal
reproduction of the four historical C/I targets. No 8B weights or GPU forward
were used in preparation; GPU numerical acceptance is still pending.

The primary grid costs 6,912 candidate scores per pass; complete engineering,
site/neutral controls and 12 passes total 144,192 candidate evaluations and
214,768 scoring forwards, with donor capture counted separately. Engineering
must pass before scientific scans. Keep fixed-prior NCC separate from fully
recalibrated NCC; its effects AND targets equal original answer-mean. No probe
has query_hehe, so that role has no recalibrated NCC. Shared physical terms
cancel before bounds. C/I each nominate at most one unit by the registered
original-sum worst-group/direction rule. Secondary exceptions cannot replace
the winner; module refinement needs a separate freeze. Mechanism readiness stays
false; this is not exclusive glyph causation, an abstract sense circuit, or a
held-out mechanism result.

The latest user authorized four cards from 08:30 onward on 2026-09-15, checking
every 30 minutes if busy, with the window ending at 17:00. A separate server
supervisor was launched (PID 1983692, verified sleeping) using
`reviews/q01-local-mechanism-v1/window-20260915-01/config.json`. Read its
`state.json` and any `run-01/run_manifest.json` before acting; do not start a
duplicate executor. The app scheduling tool was unavailable, so this is a server
process, not an app task/automatic chat wakeup. It waits until 08:30, requires all
four [0,1,2,3] idle, launches the frozen run, checkpoints from 16:45 and enforces
an owned-process exit fallback at 16:55. Completed execution is independently
checked and analyzed without another GPU forward. State/logs are authoritative
over this recorded preparation-time note. Failure stops automatic retries.

Creating `CANCEL` in that window directory cancels this supervisor/owned run;
use it promptly if the user withdraws or changes this authorization. It never
kills another user's process. A paused run can resume only with the same device
list, UUIDs, runtime and plan; failed/completed runs reject new forwards. Use
`scripts/review/run_q01_local_mechanism.py validate|run|check|analyze` and
`scripts/review/prepare_q01_local_mechanism.py check`. Only `run` loads models.

## Functional query experiment complete (2026-09-15)

`functional-query-results-v1/current.json` selects both completed four-GPU
stages and the hashed assistant interpretation. The eight texts, hate labels
and applicability suggestions were explicitly accepted in bulk; preserve the
separate feedback, AI authorship, original draft null fields and the Q07/Q08
implicit-reference caveats. New IDs have no original Gold or original_correct.
The original draft/current and execution selector remain historical and unchanged.

Stage 1 completed 128 new conditions, 476 unique prompts/952 candidates and
5,840 candidate evaluations. Stage 2 completed 64 F1 scope conditions plus the
16 F1/O new-query replays and four old O anchors: 312 prompts/624 candidates and
3,872 evaluations. Total is 192 new scientific conditions on eight synthetic
queries, 708 distinct underlying prompts and 9,712 candidate evaluations.
Both eight-pass runs passed ten raw and six derived gates (5,904 and 4,128
readouts per derived gate). History, repeats, member order, physical producer
changes and the 80-prompt stage bridge have zero error. Padding maxima are
0.000244140625 / 0.000453948974609375; prefix maxima are
0.000118255615234375 / 0.0001678466796875. Epsilon is unchanged.
Seven CPU tests, both byte reconstructions, 287-source input audits, exact
analysis reconstruction and independent 50-digit Decimal checks of 35,528
scalars / 16,120 directions / 5,184 predictions pass. All eight owned workers
exited normally; the final inventory shows all four GPUs at zero memory/utilization.
Do not restart terminal runs or edit their pinned scientific implementation.

O/N1/N2/D correct-condition counts in original sum/mean/NCC/A-B forward/reverse
are 22/21/16/20/26, 22/22/24/22/20, 21/19/25/22/20, 24/23/22/24/20 out of 32
correlated conditions per arm. Both ordinary-laughter queries are non-hate
under every entry-absent condition in all five main views, but other queries
do not improve together. Q03 remains wrong in original/NCC/forward throughout;
Q02 and Q08 have losses under some removals/mappings. All 32 stage-1 K comparisons
are positive, which is ordering, not correct thresholds. A-family J0<0 and T0>0
hold across both fillers, both demo families and five views, but B-family
dependence/Delta-K and some A-family effects reverse with mapping or wording.
NCC backgrounds cancel in cross-query contrasts; agreement with original mean
is algebraic, not independent replication. Keep every probe and unresolved row.

Scope notes do not provide stable functional repair. Stage-2 O/P1/X1/P2/X2
correct counts are 11/10/8/10/13, 9/10/8/10/13, 10/9/8/10/13,
10/10/8/10/13, 10/11/8/10/12 out of 16 per arm. All 64 new scope/restatement
conditions predict hate in NCC. Q01 scope C is positive and I negative across
five views/all single/LOO probes, but original sum still misclassifies all P/X
conditions. Q05 I remains mapping-sensitive; Q03/Q04 small effects and Q08 C
include probe, reverse and unresolved exceptions.

Q01's four O-to-N1/N2 matched contrasts across F1/F2 provide a local C>0/I<0
input result stable in all five main views and all NCC probes/LOO. Original
Q01 O delta is positive, unlike old 3169, so negative I is not a universal
increase in a negative word bias. The proposed four-contrast mechanism study
is post-outcome prioritization only: no hook/self-patch/internal protocol or
forward has been prepared/executed, mechanism_ready stays false, and reserve
remains inaccessible. Online records, old reference/use/result pointers and
human mechanism fields remain unchanged. Earlier running/draft notes below
describe their historical time.

## Functional queries: bulk adoption and four-GPU execution (2026-09-14)

The user explicitly accepted the AI labels and suggestions and authorized this
experiment with four available GPUs. `functional-query-diagnostics-v1/feedback-01.json`
binds bulk adoption of all eight exact query texts, hate labels and applicability
proposals to draft manifest 90ec91cbb8e18cfc8f19383fd5a719c4cb8d2c348e35a622cdd7148f532b86ad.
Preserve AI authorship, the implicit-reference caveats for Q07/Q08, and the
distinction between bulk adoption and individual question/answer adjudication.
The old draft's null human fields and current pointer remain historical.

`functional-query-diagnostics-v1/execution-01.json` selects two separate frozen
plans under `reviews/functional-query-diagnostics-v1/execution-01`. Stage 1 has
128 new conditions, four old O anchors, 476 unique prompts/952 candidates and
5,840 candidate evaluations. Stage 2 has 64 fixed F1 scope-note conditions plus
16 F1/O new-query replays and the four historical anchors: 312 unique prompts/
624 candidates and 3,872 evaluations. The additional O evaluations are stage
bridges, not new scientific conditions. Both stages share 80 identical inputs.
All 608 draft comparisons are covered (360 in stage 1, 260 in stage 2, 12 overlap).

Seven CPU tests pass, including both full synthetic run/check/analysis lifecycles,
reference isolation, alias deduplication, background cancellation, physical GPU
assignment and independent Decimal audits. Both freezes reconstruct byte-for-byte;
independent tokenizer audits verify 287 sources and 788 stage prompt reconstructions.
These CPU receipts are not GPU numerical acceptance. The four-GPU primary run
has started; consult run-stage-1-01/run_manifest.json and audits/gpu-stage-1-01.log.
Run stage 2 only after stage 1 is sealed, preserving all prior outputs.

Use the new freeze/run/analyze/audit_evidence_functional_queries.py scripts with
explicit stage paths. Do not edit pinned code or use the old draft preparer as a
GPU executor. The original FP32 kernels, model lease/hash checks and epsilon are
unchanged. Every candidate changes physical producer during the replica pass;
this is not four independent evaluations of every candidate. Analysis reads
adopted references only after all gates and the raw seal; new queries' original
reference/correctness remain null. Keep stage-2 O scores distinct and verify the
80-input bridge. Online sessions, old reference/use/results pointers and human
mechanism judgments remain unchanged; mechanism_ready stays false.

## Functional-query diagnostic draft prepared (2026-09-14)

The user requested eight draft queries, separate applicability/label review tables,
a compact input matrix, and pre-outcome interpretation criteria. The separate
`functional-query-diagnostics-v1/current.json` selects draft-01, not a formal input
freeze. All new query text adoption, applicability and hate human fields remain
null. New IDs FD-3169-Q01 through Q08 have no original dataset Gold and must not
inherit 3169's non-hate reference. AI authorship and prior discovery exposure are
explicit. Do not merge these queries into the old discovery/reserve review session.

First-stage proposal: 8 queries x 4 demo versions x O/N1/N2 = 96 primary conditions,
plus 32 natural-deletion conditions and four separate original-query O anchors.
The fixed second-stage P1/X1/P2/X2 supplement uses F1 only (64 conditions); it is
not silently part of the primary matrix. Q07/Q08 use an implicit group designation:
its referent needs separate review even if the hate label is resolved. Query A/B
families differ in reference explicitness and some attack constructions, not only
wording. Both are crossed with both demo families in stage 1. The paired word gap
always changes demo wording; query 嘿嘿 itself is not replaced.

Preparation uses only the local CPU tokenizer. There is no run subcommand or new
GPU task. Use `prepare_evidence_functional_queries.py --check` to reconstruct draft
bytes. The registered C/I and cross-query comparisons distinguish global shifts,
reference agreement and context dependence. Shared NCC background terms cancel
in cross-query contrasts; equality to original answer-mean is an algebraic check,
not independent replication. Preserve old pointers and new draft provenance;
human feedback, text revisions and the eventual executable freeze require separate
versions. Online records and mechanism readiness remain unchanged.

## 3169 lexicon dependency and scope-note experiment complete (2026-09-14)

The user explicitly requested deletion of the 嘿嘿 entry, the same two frozen
嘿嘿/哈哈 laughter pairs, length/position controls, and a literal-laughter scope
note. `lexicon-scope-results-v1/current.json` selects completed frozen-02/run-02/
results-02 and the assistant interpretation. The original lexicon-scope-v1
current pointer and frozen-01 remain historical; use
`lexicon-scope-v1/execution-model-path-02.json` for the actual execution paths.
The existing `run_evidence_lexicon_scope.py` takes explicit --plan/--run/--output
paths for version 02. Never restart the terminal runs or edit pinned code.

40 real conditions / 320 prompts / 640 candidates completed all eight passes
(3,968 candidate evaluations), 10 raw and 6 derived numerical gates (1,968
readouts each). History, repeats, member order and GPU replica error are zero;
padding max 0.000370025634765625, prefix 0.000118255615234375, unchanged epsilon.
Seven preparation and three model-path CPU tests pass; independent input audit
checks 230 sources and 352 geometry proofs. Decimal audit verifies 21,062 scalars,
6,604 directions and 520 predictions; analysis bytes reconstruct exactly. GPU 1
and 2 workers exited normally and post-exit memory/utilization were zero.

The first run failed before scoring because the existing model shards had become
symlinks to /data/models/Qwen3-8B. Preserve that failed run and shared links. A
private regular-file copy under this experiment's ignored work tree matches all
11 original file hashes (16,397,431,454 bytes); model-load-package-01 changes only
source paths. Frozen-02 preserves every scientific input byte and retains the
unchanged registry's fresh full-hash and regular-file lease checks. Do not bypass
the source lease or replace shared model paths to rerun an old preparation.

Removing lex-0419 or replacing it at its slot with either matched neutral entry
reverses the pair gap under original scores/NCC and reduces its negative A/B
magnitude. All 6 deletion/replacement interactions are positive in 12 raw modes
plus NCC; all their single/LOO NCC interactions are positive. Natural deletion's
A/B residual remains negative (magnitude reduced about 76–87%). Moving the same
filler to dictionary end preserves the pattern but changes magnitudes; position
effects are not zero and the relocation moves multiple dictionary entries.

Scope notes X1/X2 versus matched restatements P1/P2 make the negative gap larger
in original sum/mean, NCC and A/B forward across all four comparisons; reverse
mapping has one negative and three positive interactions. Both margins can move
toward non-hate while the 嘿嘿-minus-哈哈 gap becomes more negative. Retain F1/P2's
unresolved NCC pair and all probe exceptions. Entry presentation contributes to
the local effect; this does not prove an independent glyph bias or that adding
an applicability caveat repairs overgeneralization. All 20 entry-absent conditions
predict non-hate in five main views; these remain one exposed query, not 20 rescued
cases. Preserve all old reference/use/material/run pointers and human fields;
new experimental wording is assistant-authored and mechanism readiness stays false.

## Content decomposition GPU results complete (2026-09-14)

`content-decomposition-results-v1/current.json` now selects the completed
`frozen-02` / `run-02` / `results-02` two-GPU execution and hashed assistant
interpretation. The original four-card preparation/selector and all prior
materials, references, case-use records, freezes and code remain unchanged.
GPU 1 and 2 workers exited; ownership and release receipts are in the private
work tree's audits. Do not restart a terminal run or treat earlier running notes
below as current state.

All 8 passes / 2,816 candidate evaluations completed: 208 prompts, 416 candidates,
80 historical replays, 14 primary + 12 total-input bridge + 4 historical contrasts.
Historical, repeat, member-order and physical-replica differences are exactly zero.
Padding maximum is 0.000141143798828125; prefix maximum is 0.00009918212890625.
All 10 raw and 6 derived gates (672 readouts each) pass without changing epsilon.
The 15 preparation tests, 4 allocation tests, byte reconstruction, 175-source input
audit and independent 50-digit Decimal checks of 2,210 scalars / 780 directions
pass. Two physical replicas are proved for this run, not four. Explicit paths
passed to the frozen analysis CLI must be absolute; its defaults are absolute.

3169's four designation/laughter pairs remain negative in original sum/mean,
NCC, both A/B mappings and EOS auxiliary modes. The first laughter pair has
NCC -0.042335 and all LOO directions negative, but the single-space probe gives
+0.010276; retain that exception. Shared written-form association in these two
attacking contexts is supported, not a context-free glyph cause or attack-presence
factorial. Pluralization and explicit-reference clarity remain scoped limitations.

541's four conditional rule effects are positive in NCC with stable probe/LOO
signs, but A/B forward flips by wording and reverse is negative for all four;
original scores also cross over. The two NCC interactions are negative, and the
first A/B forward interaction is unresolved under answer-mean and mean-with-EOS.
Do not promote this into stable abstract-rule utilization. All 26 NCC conditions
and A/B-forward conditions predict hate; both query references remain non-hate.
For the 8 new conditions per query, correct counts in original sum/mean, NCC,
A/B forward/reverse are 8/8/0/0/8 (541) and 1/2/0/0/6 (3169). These are correlated
prompt conditions on two discovery queries, not independent test accuracy.

Human source fields, individual material adoption and mechanism readiness remain
unchanged. Functional query controls remain a separate future phase with new IDs
and references. Use the linked interpretation and all reverse/unresolved/probe
exceptions rather than selecting a favorable score or mapping.

## Content GPU window: two-card execution (2026-09-14)

The user has now explicitly authorized “可以启动GPU运行，目前只有两卡空闲，剩下的两卡晚点才能空出来”.
This supersedes the earlier pre-GPU hold below for this experiment. Only physical
GPUs 1 and 2 were verified available; GPUs 0 and 3 are excluded. The separate
`content-decomposition-v1/execution-two-gpu-01.json` selects `frozen-02`, preserving
all original frozen scientific artifacts byte-for-byte. Allocation [1,2] and
its new runner are independently source-pinned; eight passes still contain
2,816 candidate evaluations, and shift=1 moves every candidate to the other GPU.
This run does not claim a new four-GPU replication. Do not grow the pool midway
or edit `frozen-01`, its selector, or any frozen implementation.

Use `scripts/review/run_evidence_content_execution_v2.py run|check|analyze` for
`run-02`/`results-02`; the old default runner still targets the historical four-card
preparation. The two-card amendment has four passing CPU tests and byte checks;
its independent input audit verifies 175 sources and all 208 prompts. GPU work
has started, with stdout in the private work tree's
`audits/gpu-run-two-card-01.log`. Consult `run-02/run_manifest.json` for current
progress. Query references remain inaccessible to analysis until raw scores
are sealed and every numerical gate passes. Human fields and mechanism readiness
are unchanged. The older pre-GPU closeout below describes its recorded time.

## Content decomposition: prepared, GPU pending (2026-09-14)

The latest user authorized continuous wording preparation, AI checking, input
freezing and implementation, with interim wording reports, and explicitly said
“目前没有到我的GPU窗口，所以先不运行”. Do not start a model forward until the
user announces the GPU window. Earlier GPU authorizations do not override this
latest constraint. `content-decomposition-v1/current.json` selects the immutable
GPU-ready input freeze within the evidence experiment. Its README contains the
full wording, numerical plan, CPU receipts and future run commands.

There are 16 new assistant-authored texts: 541/hate has two four-cell topic
(basketball fan / heterosexual identity) by countergeneralization wordings, all
presented non-hate; 3169/hate has two group-designation pairs (嘿嘿 / 黑人) and
two literal-laughter pairs (嘿嘿 / 哈哈), all presented hate. Group pairs add 们
in both arms for explicit plural reference and matching token lengths; this
change is separately bridged to the prior input. Both 3169 contexts retain an
attack, so do not describe a pure glyph-by-attack factorial or attack-presence
effect. Explicit group designation also changes referential clarity.

The original R query/dictionary/other nine demos/answers/order stay fixed, at
541 slot 5 and 3169 slot 3. Original-label real prompts are 918 tokens for all
new 541 cells and 743 for all new 3169 cells (one more than old 742). All primary
contrasts match complete external token geometry within encoding/probe; internal
semantic token alignment is not claimed. Ten exact historical R-A/C1/C2/D1/D2
anchors bring the frame to 26 content conditions: 208 prompts, 416 candidates,
80 historical replay prompts across original/probes/both A/B mappings, and
14 primary + 12 total-input bridge + 4 historical comparisons. Retain all three
score groups and every per-condition probe, single/LOO diagnostic, reverse and
unresolved result. Eight planned passes total 2,816 candidate evaluations;
10 raw and 6 derived gates (672 readouts each) retain the prior epsilon.

15 CPU tests pass, including a full synthetic eight-pass/checkpoint/analysis
lifecycle. Hardware/runtime validators were mocked only in the isolated CPU
test; this is not a claim of GPU numerical acceptance. Byte reconstruction and
an independent check of 154 sources, 208 prompt replays, 416 candidate boundaries,
112 primary comparison/variant proofs and 80 unchanged historical candidate
payloads pass. No model weights/forward, real run/results directory, online
writeback, individual human adoption or mechanism readiness was created.
The user authorized preparation without pauses; preserve AI judgments and null
individual human fields rather than treating that as per-text human adoption.

Use `scripts/review/freeze_evidence_content_decomposition.py --check` and
`scripts/review/run_evidence_content_decomposition.py validate` for CPU checks.
Only the explicit `run` command loads models; `check` and `analyze` verify sealed
scores, with query references parsed only after all gates. The separate
`audit_evidence_content_decomposition.py` checks input reconstruction, and
`analyze_evidence_content_decomposition.py --check` rebuilds analysis bytes.
Preserve the new freeze and all old source-pinned implementations and pointers;
new code/material changes require a new version. Functional query controls are
still deferred to a separate phase with new IDs and references.

## Original labels, NCC and A/B mappings (2026-09-14)

The user authorized “请帮我按这个顺序执行，弄一组新的结果，其中若有任何需要冻结项可以询问我”.
`label-calibration-v1/current.json` selects the separate immutable input/protocol
freeze; `label-calibration-results-v1/current.json` selects its completed run,
analysis and hashed assistant interpretation. Both are within
`exps/causal_context/general_model_evidence_applicability_v1`. Preserve all prior
material, reference, case-use, matched-input and run pointers.

The same 36 conditions and 64 comparisons now have original full-label scoring,
180 original-label background prompts (five per condition), and 72 A/B prompts
(forward and reverse). NCC uses the full JSON label including quotes, excludes
EOS, length-normalizes token logprobs, normalizes across the two classes, averages
the five probe probabilities, then calibrates. Probes are the exact empty string,
single space, N/A, [MASK], and Lorem ipsum, substituted only for the query JSON.
This explicit local aggregation specification is frozen, not a claim of a
bit-identical reproduction of unavailable author code. All single-probe and
leave-one-out diagnostics are retained. A/B mappings change the system output
instruction, every demo answer and output candidates together; other content is
fixed. Their forward/reverse token layouts match; the bridge from original labels
changes instruction/answer lengths and is recorded separately.

Eight passes scored 288 prompts/576 candidates (3,600 candidate evaluations).
All 36 historical inputs replay exactly; repeat, member-order and physical GPU
replica errors are zero. Padding max error is 0.00023651123046875 and prefix max
error is 0.00009918212890625, below the unchanged epsilon. Ten raw numeric gates
and six derived gates (1,200 readouts each) pass. NCC uses propagated bounds:
2 epsilon per calibrated margin, 4 epsilon per pair and 8 epsilon per four-term
interaction. Eight CPU tests, byte reconstruction, 576 token boundaries, and an
independent 50-digit Decimal check of 4,228 scalars pass. All 126 source files
remain unchanged and all four GPU replicas exited; GPU release is recorded.

For 3169, all six D-minus-C comparisons across R/P0/P1 and two wordings remain
negative in original scores, NCC and both A/B mappings. For 541, those six are
positive in NCC and both A/B mappings, whereas original mean has two positive,
three negative and one unresolved result. All twelve have consistent single-probe
and leave-one-out directions. This concerns relative input effects, not improved
classification: both queries' original/reviewed references remain non-hate, but
NCC predicts hate for all 36 conditions. Correct-condition counts (original sum,
original mean, NCC, A/B forward, A/B reverse) are 6/7/0/0/16 of 18 for 541 and
4/5/0/0/7 of 18 for 3169. Do not adopt NCC or one favorable mapping as a replacement
primary score, alter references, or select probes after seeing these results.

Retain unstable effects: 541 deletion and L-minus-S reverse under A/B relative
to original/NCC; 3169 original-content position has a small NCC residual that
reverses when N/A is left out, and A/B directions disagree. 3169's second
position-by-content interaction also depends on the score/mapping. These are
two exposed discovery queries, not 36 or 64 independent samples. Future semantic
factor materials or new label/probe controls require a separate frozen version;
human fields, prior use decisions and activation-patching readiness are unchanged.

Use `scripts/review/freeze_evidence_label_calibration.py --check`,
`scripts/review/run_evidence_label_calibration.py validate|check|analyze`, and
`scripts/review/analyze_evidence_label_calibration.py --check`. Completed `run`
outputs reject new forwards. The new binary catalog/execution adapter and its
tests are source-pinned along with unchanged forward kernels; do not edit them
in place after this freeze.

## Case adjudication and dual-reference closeout (2026-09-12)

After material finalization, the user completed all 32/32 case reviews in the
authoritative evidence session. The read-only snapshot has revision
`47c6ab2652111fc7b20e6f6824a1e1ea5879380a830fddfccbcbbeb3ba0d5419`.
Within `exps/causal_context/general_model_evidence_applicability_v1`,
`analysis-reference-v1/current.json` selects 64 frozen, eligible task references.
The user explicitly resolved four material/case differences in favor of final
case labels: #3683 hate non-hate; #3919 group []; #3950 group [Sexism]; #4615 group
[Racism, Sexism]. Preserve both source layers and the scoped confirmation.

`dual-reference-v1/current.json` selects the completed CPU evaluation: all 384
historical predictions reproduced and 6,528 candidate scores verified on the
same 32 discovery cases. Run `scripts/review/evaluate_evidence_references.py`
for validate/evaluate/check; existing outputs and reference freezes must not be
overwritten. The reference pointer's evaluation-pending field is historical;
read the separate evaluation pointer for the completed state. New human or
implementation changes require a new version, preserving the recorded source.

The user has authorized the available GPU window for tests and evaluation.
This dual-reference evaluation needs no new model forward. Case explanations
remain deferred, and all 64 input-control eligibility fields remain false until
specific interventions, hypotheses, and alternatives are separately frozen.
Do not promote derived behavior masks into human mechanism judgments.

`case-uses-v1/current.json` now selects a 64-task case-use proposal derived from
that freeze/evaluation and all 1,072 material records. Roles and preparation
order are AI proposals, separate from unchanged human use decisions. Its strict
U screen requires an available demo reference matching the original answer and
both adopted topic/rule fields equal to none. Partial or unadopted adapter fields
do not pass; answer differences remain separate. 5086 and 541 have no strict U
candidates in either task, so retain their research value through separately
defined interventions rather than relaxing this screen to force a cohort.

The user subsequently authorized “好的，可以按这四步开始实施”.
`case-uses-implementation-v1/current.json` selects the implemented 64-task work
table and 12 explicit protocols/26 prepared inputs for six priority cases.
This authorizes workflow implementation, not retrospective individual human
adoption of AI roles or mechanism hypotheses. Original reference/use pointers
remain historical and immutable. 12 source text/token replays and 412 candidate
boundaries pass; current model numeric replay and scoring have not run.
3169's lex-0419 remains reasonable with no adopted rewrite, so its prepared
contrast removes demo 3660 while retaining the complete dictionary. 5086/hate
uses a separately named related-demo source factor for 7248/lex-0073, not the
strict U branch. Answer revisions and 6037's specific edge removal are separate.
Natural edits and partial operation controls do not establish position control
or readiness for activation patching. Preserve fixed inputs and no-effect or
reverse results in subsequent execution.

The user then authorized the available GPU for the next stage on 2026-09-12.
`input-interventions-v1/current.json` selects completed execution of these 26
conditions (412 candidates), eight score passes, and a new 64-task use table.
12 historical baselines replay exactly; repeated runs, member order and physical
GPU replica changes have zero maximum error. Padding/prefix checks pass the
unchanged epsilon. All four model replicas exited and GPUs were released.
The 12 scored tasks have eligibility limited to their registered natural edits;
the other 52 tasks and all human fields are unchanged. This is not activation
patching readiness. 5086/hate has three wrong-to-right arm comparisons only in
the answer-sum primary score (also sum with EOS); both length-normalized modes
remain wrong in all four arms, and its answer-mean interaction is numerically
unresolved. Do not present these as three independent rescued cases or robust
cross-score improvement. 5086/group improves after its single answer revision;
541/hate worsens after deletion; 3169 and 6037 retain their predictions with
nonzero score changes. Preserve all original freezes and prepared pointers;
new analysis is assistant-authored, not new human mechanism adjudication.

## Matched input materials (2026-09-12)

`matched-materials-v1/current.json` selects the first immutable preparation:
8 replacement/paraphrase texts for 541/hate and 3169/hate, 12 prompts including
the original/deletion references. The user later said “第一批没什么问题，还有下一批吗？”.
`matched-materials-review-v1/current.json` records this as overall no-objection
bound to those 8 exact text hashes, separately from the original draft pointer.
Do not fabricate individual field decisions or rewrite the first preparation.

The user requested continued execution of material preparation.
`position-length-materials-v1/current.json` selects the second preparation:
6 new texts (two matched partners and four short/long variants), 20 position
contexts, four length contexts and 12 first-batch replays. These new texts remain
AI drafts pending review; the first-batch feedback does not accept them.
The position pairs swap 541 slots 5/9 and 3169 slots 3/2 in a common rewritten
partner background; they move two examples, not an isolated absolute position.
Record the partner-rewrite bridge separately. Length edits also change wording,
density and downstream positions. 10 swap checks, 10 bridges, four length shifts,
72 candidate boundaries and byte reconstruction pass without model forward.
All earlier reference/use/run pointers and human source fields remain unchanged.
Neither material batch is activation-patching ready or deployed to the online UI.

The user later replied “第二批也没有问题”. The separate
`matched-materials-review-v1/batch-02-feedback-01.json` binds this overall
acceptance to the six second-batch text hashes. The first feedback pointer is
already pinned by the second preparation, so leave it unchanged. Both accepted
batches now feed `matched-input-freeze-v1/current.json`: 14 texts, 36 contexts,
72 candidates, four historical numeric replays and 64 frozen comparisons for
541/hate and 3169/hate. Batch acceptance permits their described input uses;
AI field authorship and original blank individual human fields remain intact.

Use `scripts/review/freeze_evidence_matched_inputs.py --check` and
`scripts/review/run_evidence_matched_inputs.py validate|run|check|analyze` for this
separate frozen plan; the prior executor is specific to the old 26-condition run.
Five tests, all 36 token replays, 72 answer boundaries, byte reconstruction and
independent verification of 104 sources pass. Current-model replay/scoring of
these inputs has not started. The eight-pass plan retains the prior FP32 scorer,
epsilon and all four score modes. Reference labels are loaded only for analysis
after raw scores are sealed and all numerical gates pass. Retain all no-effect,
reverse, unresolved and score-sensitive results. Preserve every earlier freeze,
draft pointer, completed run and human source record; mechanism readiness remains
false. New code or material changes require a new version.

On 2026-09-13 the user authorized “当前GPU已空闲，可以启动任务”.
`matched-input-results-v1/current.json` now selects the completed 36-condition,
72-candidate execution and four-mode analysis (144 condition / 256 contrast rows).
All eight passes and ten numerical gates pass at unchanged thresholds; historical
replays, repeats, member order and GPU replica differences are exactly zero.
The four replicas exited and GPUs were released. Independent recomputation of
2,608 scalars and all 104 source hashes passes. The preparation/freeze pointers
remain historical; read the separate results pointer for completed execution.

3169's two fixed-slot content replacements predict non-hate in all four modes,
while its semantic-preserving versions remain hate. All six D-minus-C comparisons
across original/common-partner/swapped settings are negative in all four modes;
this concerns the whole semantic/cue combination, not 嘿嘿 alone. Common-partner
rewriting and swaps still affect predictions, and the two position-by-semantics
interactions have opposite signs. 541's neutral content replacements also retain
correct predictions, so deletion harm is not uniquely attributable to the removed
countergeneralization semantics. Its D-minus-C advantages depend on wording and
score definition; P0 partner rewriting reverses all five primary predictions.
Preserve these caveats and short/long wording confounds. New interpretations are
assistant-authored, no human fields were changed, and mechanism readiness remains
false. Do not count conditions as independent cases or start new interventions
from these interpretations without their own input/protocol freeze.

## Final evidence annotation result (2026-09-12)

The user explicitly authorized “可以写回了，并作为本次标注的最终结果”.
The 954-object local result is now accepted and written to the authoritative
evidence session, in addition to its 118 prior confirmations: 1,072/1,072 material
objects are confirmed. The final pointer and receipt are in this run's
`final-results-v1/current.json` and `writeback-receipt.json`. Earlier local exports
retain their original pre-writeback status and provenance; do not overwrite them.

This is explicit bulk acceptance, not a claim that every AI field was previously
adjudicated individually. The final artifact retains those field sources and
histories. Native `final_annotation` receipts preserve severity, adopted wording,
and task-policy references; `session.finalizations` preserves authorization,
policy documents and the separate 女圈 lexicon addition. Frozen definitions and
hit judgments against their original senses remain unchanged.

At material finalization the separate case-level stage was 3/32; that batch
did not invent those judgments. The later 32/32 closeout is recorded above.
Do not confuse case-stage progress with the
completed material annotation scope. Future code must support the finalization
reader; old readers must not be restarted on the new authoritative records.

## Local adjudication preferences (2026-09-11)

For the evidence reannotation run, read
`exps/causal_context/general_model_evidence_applicability_v1/ai_reviews/v2-reannotation-20260911/preferences/preferences.md`
before drafting the next batch. It links the machine-readable case index, explicit
rule confirmations, and immutable snapshots. Bind the snapshot used by each batch;
refresh the current table after recording new explicit decisions with
`python scripts/review/export_adjudication_preferences.py`, then run `--check`.

Keep user decisions scoped to the fields actually answered. Compare similar cases
by target, proposition, author stance, available context, and applicable policy.
When a decision may conflict with an earlier one, show both cases and the specific
field, and ask whether this is a case distinction or a rule revision. Preserve the
old decision; do not silently overwrite it or turn an AI interpretation into a
human rule. The table's AI summaries are review aids, not additional confirmations.

The latest local group supplement is recorded in this run's
`group_priority_same_target_context.json`: the user clarified that personal insult
yields to Racism only when the same attacked person is identified as Black.
Co-occurrence in one sentence is insufficient; separate personal targets and
independent political or other identities retain their own others contribution.
Do not infer the addressee's race from supporting Black people. The explicit
recheck changed #5423 to Racism + others; #5615 remains Racism. Old decisions and
the earlier, overly broad AI applications remain available as history.

The current local 270-sentence result is selected by `sentence_context.json` and
`sentence-completion-v1/current.json`, with JSON/CSV exports and field provenance.
Earlier batch exports and the 210-row severity mapping are historical layers;
do not use them as the latest result or overwrite them to erase amendments.
Rebuild the closeout with `python scripts/review/export_sentence_completion.py`
and verify with `--check`. Local sentence completion is not online confirmation;
resource applicability review remains a separate queue.

Local definition review now uses `definition_context.json` in the same run,
pointing to `resource-reviews-v1/definitions/current.json`. Before continuing a
resource batch, also read `resource-reviews-v1/preferences.md` and its README.
Export with `python scripts/review/export_definition_reviews.py`, then `--check`.
Only explicit definition replies count as new human fields. Keep original
definitions, AI rewrite proposals, and explicitly adopted rewrites separate;
a definition verdict does not automatically accept the proposed rewrite.
Previously confirmed definitions are calibration references, not new completions.

Local occurrence review uses `hit_context.json` and
`resource-reviews-v1/hits/current.json`. Read the linked hit preferences and README
before continuing; export with `python scripts/review/export_hit_reviews.py` and
verify with `--check`. Keep source-position fit and query fit independent, and
retain the original definition alongside any adopted-rewrite comparison. A shared
rule reply does not confirm every similar hit; multi-position questions stay
pending until all explicitly requested fields are answered.

When a task needs human review, first inspect and reuse this repository's existing
three-column review workbenches. The user prefers a consistent review experience
over introducing a separate UI framework or interaction model for each experiment.

## Layout and interaction

- Left: searchable case queue, status filters, progress, and previous/next navigation.
- Center: the current query and supporting material, with readable full text and
  expandable secondary details. Keep the evidence visible while editing a decision.
- Right: task-specific review fields, clear save state, save draft, and confirm-and-next.
- Reuse the existing visual tokens, responsive sidebar, keyboard conventions,
  delayed autosave, and resumable server-side session behavior where applicable.
- Protect unsaved edits during navigation and failed requests. On a revision conflict,
  retain the local draft and make recovery explicit; never silently overwrite it.
- A confirmed record can be reopened with a brief reason while retaining its prior
  decision. Export review records in a usable JSON/CSV format when the task needs it.

## Existing implementations

- Shared UI styles and helpers: `tools/wp3_candidate_review_ui/styles.css` and `core.js`.
- Input/resource review: `tools/exploratory_qwen3_ld_review_ui/`.
- Autosave, conflict recovery, and amendment interactions:
  `tools/annotated_lexicon_operation_review_ui/`.
- Atomic JSON persistence and file locking:
  `src/build_lex/annotated_lexicon_repair.py`.
- Paired-case resource/trajectory review: `tools/general_model_paired_review_ui/`.

Adapt the review fields and phases to the actual task; do not copy irrelevant
approval gates or require every workflow to use two phases. When a task requires
resource-first review, persist those notes before revealing predictions or AI
interpretations, and enforce that sequence in the server API as well as the UI.

Keep human records separate from AI-assisted notes and frozen experiment inputs.
Do not mark human review complete based on model output or automated test actions.
Show only the authorized review queue; respect any reserved cases. Test meaningful
save/resume, phase-gating, navigation, conflict, and responsive behavior using an
isolated test session, without populating the user's real review records.

## Paired-case deployment and authoritative records

The authoritative paired-case session is on `digitalocean-sgp` (`165.22.48.237`):
`/var/lib/hsd-general-model-paired-review/session.json`, available through
`https://hsd.fenglin.pro`. The verified 2026-09-08 cutover preserved the existing
3/12 confirmed reviews. The unit `hsd-general-model-paired-review.service` is
enabled and active, runs as `hsd-review`, and binds to `127.0.0.1:8772`.
It uses `/opt/hsd-general-model-paired-review/current`, a symlink to
`releases/<full-archive-sha256>`. A nonempty existing session is required before
startup so a missing state file cannot silently become an empty production session.

The old local writer and aliyun reverse tunnel are stopped. The local
`exps/causal_context/general_model_ld_nolabel_paired_cases_v1/reviews/paired-cases-02/session.json`
is a retained backup, protected by the adjacent
`session.json.remote-authority.json`. Direct private SSH access now forwards this
development machine's `127.0.0.1:8772` to the authoritative DO service.

- Build with `deploy/general_model_paired_review/build_release.py`. Verify archive
  and per-file hashes; preserve the original frozen manifest bytes. Package only
  the code/static dependency closure and indexed discovery artifacts, excluding
  human sessions, credentials, logs, models, reserve cases, and unused inputs.
- Maintain one writable authoritative session. For an authorized migration, stop
  the old writer, back up and verify exact session bytes, preserve reviewer/source
  identity, and check that restarts retain progress.
- Keep the local backup and its `session.json.remote-authority.json` marker.
  The CLI and old forwarding script refuse to start that stale writer. Do not
  copy this marker beside the remote production session.
- Roll back code independently of records. Preserve the latest authoritative
  session; never remove a migration marker merely to start stale local records,
  replace new decisions with an old backup, or run both writers.
- Keep local runtime files in the experiment's ignored `reviews/` tree. The
  private HTTPS login file is `reviews/paired-cases-02/runtime/digitalocean-login.json`
  within that experiment. Keep credentials out of operational output, logs,
  commits, and packages; provide them directly to the user when explicitly requested.
- Automated saves and confirmations belong in isolated sessions. The HTTPS smoke
  script requires reviewer `automated-deployment-test`; check the real service
  using read-only health/bootstrap requests.

See `deploy/general_model_paired_review/README.md` and
`deploy/general_model_paired_review/digitalocean-sgp/README.md` for operations.

## Evidence applicability review (AI-assisted, 2026-09-08)

The approved evidence mode is live at `https://hsd.fenglin.pro/evidence/`, using
the same hsd service and loopback port. Its separate authoritative session is
`/var/lib/hsd-general-model-paired-review/evidence-applicability-v1/session.json`.
The original `/` mode keeps its original session and 3 prior confirmations;
the new mode launched at 0/32 confirmed with 1,072 AI-drafted material objects.
Maintain one writer per record layer. Never substitute either layer for the other.

- Active frozen policy: `docs/research/annotation-guidelines/evidence-applicability-annotation-policy-v2.md`.
  The approved group-scope amendment went live on 2026-09-09 (Asia/Shanghai).
  Its separate file is `exps/causal_context/general_model_evidence_applicability_v1/policies/group-scope-v2/policy_amendment.json`;
  production uses `current/evidence/evidence_policy.json` via `--evidence-policy`.
  Preserve the v1 policy and immutable v1 AI bundle as provenance.
  The user approved AI-first choice fields; do not reintroduce an independent
  human-before-AI gate for this mode. Human confirmation remains explicit.
- Immutable AI bundle: `exps/causal_context/general_model_evidence_applicability_v1/bundle/evidence_bundle.json`.
  Query/demo labels were drafted without original answers or predictions;
  demo-answer comparisons were produced separately afterward. The bundle binds
  a read-only parent session snapshot and prior exposure, including the case 3169
  walkthrough. Prior confirmed records do not count as new adjudications.
- Build with the release builder's `--evidence-bundle` and `--evidence-policy` options. Both session files
  must already be nonempty for production startup. Preserve both latest sessions
  across code changes; the first-activation script refuses existing evidence state.
- The explicit v1-to-v2 migration preserved all prior decisions (1 case and 39
  material confirmations at cutover), drafts, exposures, snapshots and events.
  It added task-specific policy/recheck state; 30 previously confirmed group
  materials need rechecking, while 9 unrelated lexicon confirmations remain valid.
  A lower current-policy progress count is not permission to restore an old backup.
  Keep the complete parent-session archive and the latest authoritative session.
- Group-only rechecking retains confirmed hate fields and shared evidence;
  changing hate or shared facts requires explicit reopening. Real shared-fact
  changes must invalidate affected hate decisions, even after a new material snapshot.
  Use `policy_changed` only for resolved group decisions under the new rule;
  it is not a source-label error. Exports retain per-task policy and eligibility.
- Startup never silently migrates policy. `update_evidence_policy_release.py`
  performed the one-time transition; do not rerun it against an already migrated
  session. Code rollback must keep the active policy and latest records compatible.
  After a committed migration failure, repair forward with the new policy-aware
  code; never reopen that session using a v1 writer or restore old decisions.
- Use `smoke_evidence_readonly.cjs` for production checks. Automated mutations
  belong only in isolated sessions. Shared demo/definition amendments invalidate
  dependent references while retaining snapshots and prior decisions.
- Deployment receipts and latest release identity are in
  `exps/causal_context/general_model_evidence_applicability_v1/execution-status.md`.

## Private SSH access and shared-server boundaries

- Manage direct private access with
  `bash deploy/general_model_paired_review/digitalocean-sgp/private-access.sh`
  and `start`, `status`, or `stop`. Its independent tmux session forwards local
  `127.0.0.1:8772` to `digitalocean-sgp 127.0.0.1:8772` without a local writer.
  `run` is foreground mode, stopped with Ctrl-C. The old aliyun writer setup is
  archived; its `start` and `run-web` are rejected by the migration marker.
- Bind services and SSH listeners to loopback. Reuse existing aliases and keys,
  require known-host verification and batch authentication, and use
  `ExitOnForwardFailure`, 30-second keepalives and a three-failure limit.
  The supervisor retries after three seconds and refuses occupied ports.
- Use systemd remotely and the dedicated tmux session in this container. Stop
  only this task's processes. tmux survives terminal closure; rerun private-access
  `start` after a container restart. Keep browser-side port 8772 aligned with
  application Host/Origin checks and verify backend/listener/browser separately.
- Protect every public application path with HTTPS authentication; bootstrap
  tokens do not authenticate reviewers.
- For hsd deployment work, the user's explicit boundary is to change only the
  hsd site/service. Preserve `pdf.fenglin.pro`, its Nginx configuration, upstream
  `127.0.0.1:8787`, and `pdf-translate-reader.service`; do not restart that service.
  Run `nginx -t` before a graceful reload. Compare PDF configuration/static hashes,
  backend PID/start time/restart count, and HTTPS response before and after the
  reload to establish that the shared server's PDF application is unaffected.
