# Human Review Workbenches

## Three-model cross-term experiment COMPLETE, audited and released (2026-09-18)

NEW docs/research/experiment-plans/cross-model-applicability-results-v1/current.json
is the authoritative completed-result selector. It selects the three completed
Qwen8B/14B run-01 and GLM run-03 results, comparison-01 and the assistant-authored
interpretation-01/INTERPRETATION.md. Final closeout manifest SHA256
c53f0ef6882d0d84f48cf97e3ddc506ab0385d8279c73bb4eb3eaf9871ad33f6.
GLM full finished at12:20:28 Asia/Shanghai. All16 owned controller/worker PIDs
are absent; final NVML shows four idle cards, zero memory/utilization and no
compute processes. Never restart completed runs or old GLM01/02 diagnostics.
The paused/failed GLM history and all original code/freezes/selectors remain.

Final three-model numerical and16/16-format qualification passes unchanged
standards. Repeat/order and logprob-identity errors are0; padding maxima are
0.000225067138671875 (8B),0.000148773193359375 (14B),0.0000362396240234375 (GLM).
Separate margin bounds are twice those values. All production replays pass.
Independent60-digit Decimal audits cover9564 primary atoms and3792 newly
computed expression values AND bounds, directions and classification transitions;
primary/expression arithmetic errors are0. Complete analysis reconstruction and
the3960-model-expression/1620-prediction cross-model table audit pass. The extra
168 old8B expressions preserve historical results. All156 old8B replays are
exactly equal; retain their original heterogeneous bounds and physical IDs.
Actual forwards10970 =8100 valid engineering +1464 science +1358 superseded
header diagnostics +48 format. GLM03 reused the2700 immutable GLM02 engineering
records with original producers under the pinned UTF8 amendment; no re-forwards
were needed. Both the header and I/O amendments and new audit/report scripts are
now source-pinned. Do not edit them in place. All final analysis was CPU-only.

Science scope remains TWO naturalistic primary families and TWO explanatory
supplementary families, not four homogeneous primary families; one conservative
dependency cluster and no independent confirmation. Each model classifies all
192 naturalistic conditions consistently with the bulk-adopted references. The
208 primary comparisons have93 directions shared by all three and115 with an
opposing model; Qwen8B-vs14B alone has127 same/81 opposite. I has only13/64 shared
directions. Most dictionary E effects reduce reference-aligned margins, but N
often does too and L-minus-N is mixed. Preserve general-context explanations;
do not call margin changes demonstrated classification damage/repair or a shared
semantic/glyph gate. Supplementary reference-consistent counts are179/185/188
of192 (8B/14B/GLM); all failures and comparisons remain separate from primary.

All12 legacy queries and36 external N rows are retained. J2 core consistency is
8/10,10/10,10/10; B4 is10/10,7/10,6/10; B3/J3 remain2/10 for each model. Do not
select only J2 or claim larger models uniformly improve. The report recommends
CPU material design that separates word sense, author attack stance and rule
fit, and addresses the observed classification ceiling before mechanism targets.
No new materials, confirmation set or internal intervention has been started.
Human reference fields and AI/bulk-adoption provenance remain unchanged. Run
state reference-join=false describes the scoring worker; analysis joined only
after each raw seal and normal final worker release, as recorded by result audits.

## UTF-8 I/O amendment and new GLM continuation; Qwen production running (2026-09-18)

NEW cross-model-applicability-execution-v1/execution-03.json is authoritative.
Both original Qwen runs passed2700 numerical records,16/16 full format probes,
normal exit and independent CPU/Decimal checks, and now run phase full (workers
1881563/1881714). All156 historical8B replays are exactly equal to original
science; original endpoints and individual bounds remain authoritative.

GLM run-02 completed all2700 engineering forwards, then failed at JSON read
with UnicodeDecodeError because the process default text encoding was ASCII.
It is TERMINAL FAILED and released; do not restart it. CPU reconstruction of
every record/vector/producer passes the unchanged numerical gates: repeat/order0,
padding max0.0000362396240234375, bound0.000072479248046875. Format was still pending.

Separate utf8-io-amendment-01 manifest SHA256
abca0687d2a3da2b04ba1486ba565f66db5f75e14e1296fd1ebc09a068d98fb5 pins
run_cross_model_applicability_utf8_v1.py and its preparation script. They must not
be edited. PYTHONUTF8=1 is mandatory for the amended entry point, including CPU
check/analyze. Forced C locale reproduces the old read failure; UTF8 mode fixes
it on identical bytes. Numerical/model code, science input and tolerances are
unchanged. The2700-record import preserves old physical score IDs, original
absolute vectors and producer files, validates their hashes and same allocation,
and only permits engineering imports from this documented released I/O failure.
All2700 imported records reconstruct; wrong prefixes are rejected; production
paths point to the NEW run. The CPU fixture is explicitly synthetic, no GPU work.

GLM now selects bound-glm4-9b-chat-03.json and run-glm4-9b-chat-03, initial
engineering worker1893699 on original GPU3. This is a NEW run, not a restart of
failed run-02. It reuses the2700 audited records and only adds format forwards
before qualification; science still requires full format, normal exit and CPU
check. Use --prepared prepared-glm-content-boundary-01 with the UTF8 entry point.
The previous GLM01/02, all original code and freezes remain unchanged. GLM
score auditing uses the separately added audit_cross_model_applicability_scores_v2.py;
preserve v1 because earlier Qwen audit receipts already pin it. Read selected
state.json/gpu.log files for progress; do not duplicate controllers.

## GLM native content boundary corrected in separate freeze; Qwen unchanged (2026-09-18)

NEW cross-model-applicability-execution-v1/execution-02.json supersedes the
launch-time execution-01 selection for GLM only. Qwen3-8B/14B continue their
original run-01 directories and prepared-01. GLM now uses
prepared-glm-content-boundary-01 (manifest SHA256
5b632c68a0bebb1510e1f7e85ea42025e1f415ea0f7d61f94e8f26f8cedf2f9a),
bound-glm4-9b-chat-02.json and run-glm4-9b-chat-02 on the same GPU3. New GLM
engineering worker1841160 has started. Read the selected run states; do not
duplicate executors. The unchanged v1 runner accepts --prepared for this new
freeze; always provide the amended prepared path for GLM02 run/check/analyze.

The old GLM native generation prefix ended at assistant token151337. Native
build_single_message and complete chat-template rendering require newline198
between the empty assistant metadata and answer content. All16 predetermined
old first-token probes were that newline. The old input therefore scored at a
header boundary. This is not evidence about task-label accuracy. A separate
source-pinned prepare_glm_content_boundary_v1.py appends exactly that one token
to all540 GLM inputs;1080 full native completed-answer reconstructions pass.
Task messages, materials, IDs, labels, relations, candidates, FP32 adapters,
arithmetic, tolerances and probe selection are unchanged. Preserve original
scientific token files as historical native generation prefixes. The correction
is explicitly post-engineering-format exposure and needs fresh GPU qualification.
Never edit the new manifest, proofs, amendment or pinned preparation script.

Old run-glm4-9b-chat-01 stopped normally through STOP at03:35:11 UTC with1358
committed diagnostic forwards:540reference,540repeat,278left-padding. Its worker
exited0 and was released. It is PAUSED BUT SUPERSEDED: do not resume or repurpose
it. No production or query-reference join occurred. Retain its first-token
diagnostics and audits/glm-original-boundary-closeout-01.json. All prior freezes
and code remain immutable. The score-audit script is a separate CPU-only reader;
it does not replace or modify the frozen scorer.

NVML aggregate used-memory is misattributed across logical GPU indices while
models coexist; torch properties report total_memory=0. Per-process NVML UUIDs
and memory match actual bindings and CUDA identities; evidence is in
audits/gpu-placement-01.json. Preserve the anomalous readings. Do not bypass
the unchanged idle checks or alter the container/driver environment. If those
checks require all workers to exit before a next phase can bind, wait for release.

## Cross-model GPU window authorized; three engineering runs started (2026-09-18)

The user now explicitly said "GPU已空闲，可以开始下一步". This supersedes the
CPU-only stopping point below. Authorization is recorded in the NEW
reviews/cross-model-applicability-execution-v1/launch-01/authorization.json
(SHA256 cd515bb7939840974ebaf30fd190f741061542da2f5362493b5294c18a0dce8a).
All four NVIDIA L20 cards were verified idle through host NVML, with no compute
processes. Allocation is Qwen3-8B on GPU0, Qwen3-14B on GPU1/2, GLM-4-9B-Chat on
GPU3. No new deadline was specified. Actual bound files pin UUIDs and runtime;
all earlier scientific/preparation freezes and selectors remain unchanged.

NEW cross-model-applicability-execution-v1/execution-01.json selects launch-01,
manifest SHA256 785f3187d53d223223658b7509f321946d4b67592f235f36135d95bd1c3d72a2.
All three engineering controllers have started. Run directories are
run-qwen3-8b-01, run-qwen3-14b-01 and run-glm4-9b-chat-01; initial workers are
1809052, 1811622 and 1814266, respectively. Qwen8B has committed scores and
Qwen14B has loaded its weights. Read each NEW run's state.json and gpu.log for
live status; do not duplicate any executor. Engineering is not yet qualified.
Only a qualified, normally released
run may advance to full using the same binding, after CPU check. Failed/complete
runs are terminal, and failures must not trigger automatic retries or relaxed
criteria. Scientific reference joins still require complete raw seals and normal
worker exit. Preserve other users' processes and all historical runs.

## Cross-model second batch adopted; science freeze and CPU runtime ready (2026-09-18)

The user explicitly accepted the additional JSC/MT dataset after the explanation
of its role: "好的，理解了，审核结果是这组数据都可以通过，可以继续下一步".
NEW cross-model-applicability-v1/feedback-02.json records24 additional materials,
120 relations and S7-S8 under the clarified supplementary interpretation. All48
original materials and240 relations are now bulk adopted, retaining AI authorship,
original Gold=null and disputed displayed answers for label swaps. No individual
adjudication is invented. draft-01,draft-02,adopted-01 and previous pinned code
remain immutable; newly adopted relation IDs end in -adopt02 with supersedes.

NEW scientific frozen-01 manifest SHA256
90714b751b5467936c0b9de4028b831353c33d2de904b21fabf1b8febd81868c
preserves all384 new inputs,156 legacy inputs and all native token records byte
for byte. Scope is TWO naturalistic primary development families HP/XC plus TWO
explanatory supplementary families JSC/MT, not four homogeneous primary families.
The original four-naturalistic-family budget is still unmet; do not silently add
materials or count the supplement toward it. All1152 expressions remain; only208
naturalistic readouts have primary priority. Both strata and all exceptions must
be reported separately; no pooled A/B primary mean, no confirmation claim, still
one conservative dependency cluster. This is a scientific-input freeze, not GPU
numerical acceptance or an allocation/runtime-layout freeze.

NEW cross-model-applicability-execution-v1/current.json selects
reviews/cross-model-applicability-execution-v1/prepared-01, manifest SHA256
fb51dd21999638155066ba77f4dfcaa47a6e941e29c3463f8784371390e57293.
Start at its README.md. New pinned code: cross_model_applicability_{models,
execution}_v1.py and run/test/audit_cross_model_applicability_*_v1.py; the separate
freeze_cross_model_applicability_v1.py is also pinned. Never edit these in place.
CPU check/audit are read-only; code changes require another version/amendment.

Native Qwen/GLM adapters select the last VALID hidden state before the LM head,
including right padding; all floating model tensors/logits must be FP32, eager
attention, batch1, no KV cache, no CPU/disk offload or quantization. GLM's local
GenerationMixin incompatibility is avoided with direct native forward and manual
greedy format diagnostics, not by modifying its source or upgrading weights.
Nine CPU tests pass: tiny random native models, tiny safetensor reloads, pad/answer
positions,60-digit Decimal arithmetic, physical-alias cancellation, strict gates,
paused checkpoint reuse and full synthetic lifecycle, stratified new analysis,
all12 legacy queries/168 comparisons and36 external N rows. Native-vs-selected
CPU max errors are5.96e-8 Qwen and1.12e-8 GLM. No research pretrained weights or
GPU forward were loaded; CPU fixture releases are explicitly synthetic.

1620 scoring input records are reference-free and match frozen tokens exactly.
Per model:540 engineering inputs x5passes; new production384 for8B and540 each
for14B/GLM, with legacy production first on the two new models. Total8100
engineering +1464 scientific =9564 prompt-only forwards;16 fixed format probes
per model add at most336 forwards. The156 old8B engineering replays cannot
replace old scientific endpoints/bounds. Each model needs its own actual-device
qualification; no old measured bound is inherited. Repeat/order cap0, padding
cap0.001, bound=max(1e-6,2*observed max),16/16 exact-label-then-EOS format gate.
Science comparisons keep raw/ref-aligned effects, bounds, ties, repairs/damage.

No bound file, GPU run, queue or idle polling exists. GPU allocation is null and
actual CUDA/VRAM/throughput checks remain pending; an unprivileged nvidia-smi
probe could not access the driver, which does not establish host GPU occupancy.
In a newly authorized GPU window, bind the actual idle UUID(s), run engineering,
then check. Default run is engineering-only; full is explicit. Qualified runs
may advance to full, explicit paused runs may resume the same binding/runtime;
failed/complete runs must not restart. STOP pauses at committed-request boundaries.
Reference joins require raw seals plus normal owned-worker exit. Preserve all
historical selectors, GPU runs and other users' processes. No confirmation set
or internal-mechanism experiment was started.

## Cross-model first batch adopted; four-term CPU expansion delivered (2026-09-18)

The user explicitly said "待审核项我都过了一遍，没什么问题，可以开始下一步".
NEW cross-model-applicability-v1/feedback-01.json records bulk adoption of all
draft-01 pending S1-S6,24 original materials and120 relation proposals. The
separate adopted-01 material-review freeze preserves AI authorship, exact text,
labels, severity, scopes and limitations; no individual question/answer review
or original Gold is invented. Swapped displayed answers remain disputed; their
human reference is the correct answer to the original text. Original draft-01,
its null human fields and four pinned scripts remain immutable. The previous
current selector is preserved under adopted-01/previous-selector.json.

NEW current.json selects draft-02, manifest SHA256
eda2b3e724fe6b090cd3a89093a81ef2d93cfdc6587add1426e850d2210f0ae6,
and separately binds adopted-01/manifest.json SHA256
1d8a177c80bf84bca60885ffd1ba5110769e1731a1d984ef969e6a5f2b09c686.
Read draft-02/REVIEW.md; ADDITIONAL-MATERIALS.md and ADDITIONAL-RELATIONS.md show
only new items. 寄生虫/木头 add8 queries,24 original materials,120 relations and
192 inputs; ALL additions and S7-S8 are pending review, not covered by the prior
user message. Combined scope:4 lexical families,16 queries,384 inputs,240
relations,1152 linear readouts. The first192 messages and all156 legacy inputs
are exact original bytes. First-batch adopted relation IDs carry -adopt01 and
supersedes; source-quality prose preserves pre-adoption AI wording and current
adoption is given by provenance. No historical experiment selector was changed.

New construction B uses neutral word explanations as foreground demos (无) and
a fixed attack anchor (有). It changes foreground labels, wording and ordered
labels relative to construction A; report each stratum and all queries, not a
pure causal construction/order effect. Semantic information can be direct while
the specific word-explanation rule is none for these targets; never infer rule
fit from answer equality. Two construction strata still conservatively share
CMAD-DEV-C01; four terms are not four independent confirmation clusters. All
queries still align ordinary sense with nonattack and derogatory sense with
attack. No new confirmation sample, model prediction or internal intervention.

CPU checks pass:1620 native prompts,3240 candidate continuation boundaries,
5120 relation bindings,1152 exact nonadditive-field expression checks and320
additive-null interaction checks. All1044 first-batch/legacy tokenized records
are unchanged. Original full23-shard hashes are reused; metadata is rehashed and
shard size/mtime checked, not another full weight hash audit. CUDA remained
uninitialized and no weights or forwards were loaded. Accepted models remain
Qwen3-8B,Qwen3-14B,local GLM-4-9B-Chat with the original GLM implementation caveat.

Preserve both new manifests and their pinned preparation/checker scripts.
Use check_cross_model_applicability_expansion_v1.py for read-only verification.
Full scientific/runtime freeze, model forward adapters, output compliance,
numerical GPU qualification and allocation are still pending. No GPU executor,
queue or idle poll was started. Next: review only the new materials/relations and
S7-S8, then freeze full inputs and prepare versioned runtime adapters. Do not
restart terminal experiments or another user's training under this CPU request.

## Cross-model A-route CPU draft delivered; human review pending (2026-09-18)

NEW `docs/research/experiment-plans/cross-model-applicability-v1/current.json`
selects draft-01 and its hashed review manifest. Start at REVIEW.md. The user
authorized CPU design and the first sample package only. This is NOT a scientific
or execution freeze: all new human fields are null and eligible_for_GPU=false.
Do not launch, queue or resume GPU work under this request. All old selectors,
freezes, result references and human decisions remain unchanged.

The AI draft contains 花瓶/小丑,8queries,4senses,10original demos,2N controls,
192new inputs (120core/72auxiliary),120relation proposals and576expressions.
There are24original material-review rows; label-swapped variants keep original
references and have disputed displayed-answer quality. D four cells manipulate
the narrowly named semantic_reference_fit, not a combined applicability score.
Low-overlap demos are descriptive paraphrases, not semantically unrelated terms.
Fixed independent insults preserve demo label composition but introduce a scope
limitation; S2/S3/S4 require review. Two lexical families conservatively share ONE
construction cluster, all designated development; no target outcomes have been
seen and no confirmation set is established.

Qwen3-8B, Qwen3-14B and local GLM-4-9B-Chat are proposed, not yet human-frozen.
All23local weight shards were hashed and match public official commits;8B also
matches its old freeze. GLM's generation_config/tokenizer_config/tokenization
code/modeling code differ from the checked upstream commit; local files are
preserved and pinned. Do not silently replace them.1044CPU prompt reconstructions
and2088single-token boundaries pass, including all156old8B bytes/token sequences.
An independently implemented CPU checker validates2560relation bindings,
576expressions and160exact additive-null interactions. No model tensors were
loaded, CUDA was not initialized, and no model forward or GPU qualification ran.

Preserve delivered draft bytes and four manifest-pinned preparation/audit scripts.
Writers refuse an existing manifest. Use check_cross_model_applicability_draft_v1.py
for read-only verification; revisions/adoption require separate versions. Review
does not imply all models behave alike or that fields are already human-adopted.

## Cross-term second behavioral run, audits and interpretation complete (2026-09-18)

NEW `docs/research/experiment-plans/cross-term-behavior-results-v1/current.json`
selects prepared-01/bound-01/run-01/results-01 and the separately hashed assistant
interpretation under `reviews/cross-term-behavior-execution-v1/interpretation-01`.
TERMINAL COMPLETE: never restart this run. Earlier preparation/execution selectors
are historical; all scientific/material/reference/human fields remain unchanged.

Original GPU0 alone completed300 prompt-only forwards/600candidate values:
84historical bridge +180new engineering +36new science. Controller launch was
00:03:53 and completion00:07:21 Asia/Shanghai,207.93seconds. Worker675587 exited
normally; controller675235 is absent too. Final inventory found all four GPUs
at zero memory/utilization and no compute process. Bridge, repeat, both padding
variants, reverse order and science replay all have exactly0 margin difference.
The new36-input bound takes the preset1e-6 floor; original84 endpoints retain
their original physical IDs,qualification references and0.00045013427734375 bound.
This is not a changed tolerance or precision. All120 classifications and168
expressions are resolved, with no ties/unresolved rows. Full CPU reconstruction
and exact analysis bytes pass. Independent60-digit Decimal audit verifies384raw
vector margins (300new evaluations +84old endpoints),120predictions and168
expression values AND bounds, with zero expression error. No new generation.

For the SAME12 queries, absent/L correct counts are none9/8, same_A9/10,
same_B11/11, other_A9/8, other_B9/9. J2 L+none m=-10.785946 becomes-1.642616
with other_A (still wrong) or+10.232662 with other_B (repair). Both moderate
the negative L increment: E0=-33.251907,E_A=-26.161196,E_B=-16.422789;
I_A=+7.090712,I_B=+16.829119. L remains adverse under either package, and
other_A+L damages its otherwise correct no-L counterpart. B's D_empty=+4.189489
and D_L=+21.018608 must both be retained. K_A=-21.188265,K_B=-14.718592:
correct labels do not make same/other effects equivalent. The two J2 repair
comparison rows share one other_B+L endpoint, not two independent repaired cases.
Demo occurrence of 京巴 is not necessary for this J2 classification outcome;
do not infer abstract sense gating or complete removal of the L effect.

G3/J3/B3 remain wrong under all other_A/B x absent/L conditions. Same_B's J3/B3
repairs do not transfer to these other_B materials; G3 remains correct only in
the historical same_A+L condition. Under L, both other packages worsen all three
anti-insult margins relative to no demos. G1/J1/B1,G4/J4/B4 retain attack labels
and G2/B2 stay correct, but their continuous margins change. Reference-aligned
E_A/E_B directions are4/8 and3/9 positive/negative; I_A/I_B9/3 and5/7;
K_A/K_B3/9 and4/8. Preserve all adverse and secondary rows. These are3exposed
terms,138comparisons containing new endpoints and30all-old comparisons, with
old36N conditions kept externally; no new confirmation family/internal intervention.

The specifically authorized lj G-LLaVA training was stopped via its verified
launcher3844994 SIGTERM, not broad process matching. All associated training
PIDs exited; its original interactive shell remained. checkpoint-5500 retained
matching files,sizes,mtimes,small-file hashes and archive structure. No extra
checkpoint or restore test was performed; progress after5500 was unsaved.
Do not auto-resume that training. Stop and release receipts are under launch-01;
the earlier running entry below is historical. Read INTERPRETATION.md together
with all scores/comparisons and the immutable CPU/result audit receipts.

## Cross-term second behavioral GPU execution authorized and started (2026-09-18)

The user authorized GPU execution and explicitly confirmed stopping lj's G-LLaVA
training PIDs3845199–3845202. Only its verified DeepSpeed launcher3844994 received
SIGTERM as UID1003. Launcher/workers exited, checkpoint-5500's file sizes/mtimes,
small-file hashes and archive inventories remained unchanged. No new checkpoint
was forced; training after step5500 was not saved. The original training stays
stopped; do not restart it without authorization. Evidence is under
`reviews/cross-term-behavior-execution-v1/launch-01/stop-{before,signal,after}.json`.

NEW `cross-term-behavior-execution-v1/execution-01.json` selects the unchanged
prepared-01, bound-01.json (SHA256
4caf5af9a15dcaf959109a496f903968636d88cc9587b0ed2bfe811a24e7a90e), and NEW run-01.
All four GPUs were idle at binding. Controller675235 launched worker675587 on
original GPU0 at2026-09-18 00:03:53 Asia/Shanghai, invocation
59b21399-bb2d-4055-a9aa-ebdf956b57d0. Initial state was loading_model. Read the
NEW run_manifest.json/gpu.log for current progress; do not duplicate executors.
No fixed deadline was imposed. Source/weights revalidation passed before launch.
The earlier CPU-only instruction below is superseded by the new explicit launch
authorization; its selector and delivery remain historical and unchanged.

## Cross-term second behavioral CPU execution preparation complete; stop before GPU (2026-09-17)

The latest user instructed "请开始下一步，直至GPU任务需求前停止，目前GPU暂未空闲".
NEW `docs/research/experiment-plans/cross-term-behavior-execution-v1/current.json`
selects `reviews/cross-term-behavior-execution-v1/prepared-01`, manifest SHA256
`9acc64f352086603b56a42aac4dd39d370d84f7acb27819774235967265075dd`.
Read the adjacent DELIVERY.md. CPU preparation, implementation and independent
audits are complete; actual allocation is null and GPU numerical qualification
is PENDING. No bound file, real run, executor, queue or idle polling was created.
Do not launch or schedule GPU work under this CPU-only instruction. A new usable
window authorization is needed. All older freezes, selectors and human fields
remain unchanged; never restart any terminal run.

The new source-pinned cross_term_behavior_* implementation reuses the unchanged
first-round native FP32 scorer and FP64 math. Do not edit its five pinned files
or prepared-01. Future binding requires the SAME GPU 0/L20 UUID
GPU-09b29c25-c372-62f4-3098-9734013e93c0 and original driver/runtime/weights;
only that one card is required. All 84 historical core inputs are replayed first
and their primary margins must reproduce exactly (maximum difference 0).
Bridge scores never replace old physical endpoints or become new scientific cases.
Then 36 new inputs receive five engineering passes (180 forwards) and one science
pass (36). Total 300 prompt-only forwards /600 candidate values; no new generation.
Repeat/order difference must be 0, padding cap stays 0.001, and new bound is
max(0.000001,2*new engineering maximum). Old bounds/qualification identities remain
original. Failure stops; no tolerance relaxation, substitution or automatic retry.

Ten CPU tests pass, including tiny random CPU Qwen3 and the actual worker driven
by synthetic logits: pause at 88, reuse all 88 saved requests, finish exactly 300
forwards, preserve bytes, detect corruption/producer changes, and reconstruct the
120-condition/168-expression analysis. Synthetic records only existed in /tmp.
Full first-round CPU reconstruction passed; 16,381,516,776 weight bytes were hashed
without loading 8B tensors. Independent audit verifies 234 artifact/source hashes,
120 tokenizer reconstructions,240 bare-answer boundaries,84 historical margins,
and60-digit Decimal checks of1,176 synthetic expressions AND bounds plus840
predictions (max expression error0). These are not GPU numerical acceptance.

Use scripts/review/run_cross_term_behavior_v1.py prepare|validate|bind|run|
resume-check|check|analyze; only future bind inventories hardware and run loads
weights. The worker never parses reference-bearing design/analysis files.
Reference joining is allowed only after bridge/engineering/science seals and
owned-worker release. Preserve all168 registered comparisons,138 with new endpoints
and30 all-old, and external36 N conditions. This remains three exposed development
terms, not pure semantics/rule-fit causation, independent confirmation or mechanism.
The scientific-review selector below stays historical with respect to CPU runtime
preparation; its material and adoption status are still authoritative.

## Cross-term second behavioral review adopted and input design frozen (2026-09-17)

The user explicitly accepted "S1–S4及36条关系按建议接受". The selector
`docs/research/experiment-plans/cross-term-behavior-discrimination-v1/current.json`
now selects frozen-01 and feedback-01.json. Manifest SHA256 is
`9e0e03ac66b4815c95d6466041cf4d886e3a7bf2bf4f1a814f862df88fe0b722`.
All 36 new relations (96 dimension fields) and S1–S4 are bulk adopted, retaining
AI authorship and exposure. There are zero pending new relations. Garbage/废物's
four D semantic partials and two L-to-demo partials are adopted within their
exact scopes. D rule counts remain 6 direct / 10 partial / 8 none.

The frozen core is 36 new unscored inputs + 84 historical inputs, with all
36 old N conditions retained externally. All new prompt bytes, prior materials
and 168 comparison formulas match the review draft. Eight supporting files are
byte-identical copies. CPU checks cover 120 prompts / 240 answer boundaries,
36 schema records, 504 appearances and 192 context bindings. An independent
adoption audit checks every accepted field, old/new identity and source hash.
The 138 comparisons with new endpoints and 30 already-exposed comparisons retain
their different exposure; none are new independent confirmation families.

Preserve draft-01, its blank human template, selector-history/draft-01.json,
the new frozen package and source-pinned freezer, all old relations and the
11 old AI scope observations. This reply did not re-adjudicate historical labels
or create a binary rule-fit causal grouping. New runtime binding, historical
numerical bridging and new-input GPU qualification are still pending. No new
model forward occurred. Prior session authorization remains separately recorded;
do not restart terminal first-round/Q01 runs. The draft-delivery note below is
historical and no longer describes the active adoption state.

## Cross-term second behavioral draft delivered for review (2026-09-17)

`docs/research/experiment-plans/cross-term-behavior-discrimination-v1/current.json`
selects draft-01, a review draft, NOT an adopted scientific or execution freeze.
The user requested the next deliverable and its review queue. Start at REVIEW.md.
All 12 query texts, 3 definitions and 16 demo texts/answers are exact prior sources.
The core has 120 conditions: 84 historical inputs and 36 new, unscored inputs.
The old 36 N conditions remain external, for 156 unique inputs across both stages.
No weights or GPU forward were used; do not restart the completed first run.

There are 24 new D-to-query and 12 new L-to-X-demo AI relation proposals, with
all human fields null. Garbage/废物 has four proposed partial D semantic links
and two partial L-to-demo links; these need review and are not adopted facts.
New D rule proposals count 6 direct/10 partial/8 none. Specific scopes, opposite
stance limitations and old-scope differences are explicit. Preserve the old84
relations, old12 L-to-query edges and11 AI observations; no binary rule-fit
causal grouping or pure same/other-word manipulation is claimed.

CPU checks reconstruct all120 inputs/240 bare-answer boundaries,36 new schema
records,504 presence rows and192 context bindings. Independent checks cover54
artifact hashes, the exact input/relation sets,84 historical score identities
and1,176 exact Fraction formula evaluations. Of168 proposed comparisons,138
contain new endpoints and30 use only already-exposed endpoints. These are
dependent development readouts, not independent confirmation or new results.
The draft manifest is51d5028484f3f5d2efb259157b9fb35c3fe9636deaa752de78fa109a23a6ade9.
Preserve this review snapshot and its pinned preparer; put feedback, changes,
adoption and eventual runtime binding in separate versions. New GPU qualification
and historical numerical bridging remain future work after review.

## Cross-term next-token GPU run and independent result audit complete (2026-09-17)

NEW authoritative selector:
`docs/research/experiment-plans/cross-term-next-token-v1/current.json` selects
`reviews/cross-term-next-token-v1/frozen-01/run-01/results-01` through separate
paths and the hashed assistant interpretation under `interpretation-01`.
The run completed at 12:30:37 Asia/Shanghai. Controller 3506906 and worker 3507209
are absent; the worker exited normally. Exit and final host checks show all four
GPUs at zero memory/utilization. TERMINAL COMPLETE: do not restart this run.
Only GPU0 was used; no parallel replica/other-GPU numerical claim was made.

All 600 engineering +120 science prompt-only forwards completed (1,440 main
candidate values), plus 24 forwards for 12 free-generation baseline diagnostics.
All 12 generated answers strictly meet the one-character format. Both candidates
share one stored full-vocabulary FP32 vector per main request. Repeat, reversed
request order and science-to-reference maximum margin differences are zero.
Left/right padding maxima are both 0.000225067138671875 under the unchanged0.001
cap; the sealed global margin bound is0.00045013427734375. All120 classifications
and252 expression directions are resolved; no ties or numerical-unresolved rows.

Eight CPU implementation tests, the22-source/120-input audit, final raw-vector
reconstruction and analysis reconstruction pass. Independent60-digit Decimal
checks cover720 raw vectors/1,440 candidate logits, all252 expression values AND
bounds and120 predictions/correctness states; margin/expression/bound differences
are zero. The3,600 long-double auxiliary checks have max error about6.01e-15.
All24 generation step vectors/prefixes/greedy choices were independently checked.
Preserve original preparation, all new source-pinned code/freezes and raw records.

For the SAME12 queries, absent/L/N correct-condition counts are9/8/9 with no D,
9/10/9 with same_A and11/11/10 with same_B. All errors occur on G3/J3/B3 or J2.
The three explicit anti-insult queries G3/J3/B3 are wrong in the empty baseline.
Same_B repairs J3/B3, but G3 only repairs under same_A+L in this frame. On J2,
L alone changes m from+22.4659614563 to−10.7859458923 (damage); it stays correct
when L is added under same_A or same_B. These are correlated development arms,
not independent rescued cases or a pure rule-applicability causal effect.

N is demonstrably not inert: under same_B it reduces reference-aligned margins
for all12 queries and damages J3 (+5.8747406006 to−10.3005523682). J3's same_B+L
margin remains correct at+1.0763130188, so L−N is a repair relative to N while L
still worsens its margin relative to the empty lexicon slot. Do not count this
as repair of the no-lexicon case or infer benefit from L−N alone. Other_A/B each
score6/6 on the SIX ordinary/separate-insult queries whose baseline was already
6/6; never rank those against the full12-query arms.

Read INTERPRETATION.md plus all scores/comparisons/query-overview TSVs. Original
Gold stays null, earlier human decisions and11 AI relation-scope observations
are unchanged. All materials are now outcome-exposed development material;
no new query, confirmation set, internal intervention or mechanism claim was
created. run_manifest's analysis_reference_join_performed=false describes the
GPU worker; the separate sealed CPU results perform the subsequent reference
join. The earlier running snapshot below is historical.

## Cross-term GPU scorer bound and running (2026-09-17)

The user authorized implementation/binding and free use of four idle GPUs.
The new execution preserves the prepared SINGLE-GPU protocol on physical GPU 0,
UUID GPU-09b29c25-c372-62f4-3098-9734013e93c0 (NVIDIA L20); GPUs 1/2/3 are unused.
`docs/research/experiment-plans/cross-term-next-token-v1/current.json` selects
`reviews/cross-term-next-token-v1/frozen-01`, manifest SHA256
`3fcb4793f42a62f15d66bef4294907c61c3919c8d0c543606682377292e5cbfb`, and run-01.
All new source-pinned cross_term_next_token_*_v1 files, math preparation and
earlier freezes are immutable. Five weight shards were freshly fully hashed;
runtime versions, critical model code, tokenizer and physical GPU are bound.

Eight CPU tests pass, including a random tiny CPU Qwen3 position/padding test,
checkpoint reconstruction/corruption, non-overwrite and engineering gate failure.
The actual worker loaded the full checkpoint without missing/unexpected keys,
verified all floating tensors FP32 on its bound device and eager attention.
At 12:25 Asia/Shanghai engineering-reference had 45/120 records. Controller PID
3506906 and worker PID3507209 belong to this run; invocation is
30a81074-4557-4e90-9c39-c4af72eeea30. Read run_manifest.json and gpu.log for current
state; do not duplicate or restart another executor. This is a server process.

Five engineering passes cover all 120 prompts before a bound is sealed; then
120 production forwards and 12 free-generation diagnostic prompts execute.
Every main request preserves a full FP32 vocabulary vector, input/mask/positions,
the shared candidate forward, producer identity and an atomic receipt. Query
references are not parsed in the worker. Failed/completed runs reject restarts;
explicit paused resumes require the same binding and reuse every valid receipt.
Creating STOP under the owned run directory pauses at a record boundary. Old Q01
controllers and their completed runs must not be restarted. GPU access requires
the approved host execution context; sandbox NVML cannot access the driver.

## Cross-term relation adoption frozen; next-token execution preparation complete (2026-09-17)

The user accepted the four discussed rule-fit corrections, confirmed no further
questions about the remaining reviewed relations, and authorized freeze/next step.
NEW `docs/research/experiment-plans/cross-term-joint-v1/current.json` selects
`frozen-01`, manifest SHA256
`a493e250fe071bd45dd5fa2e4174c58b3ffdfba41570074c4fafcc5aef3cf8ad`.
Its feedback-01.json binds the exact reply and all 84 field-scoped bulk decisions.
G4 <- CTDD-G-B-01 is now rule partial; J2 <- CTDD-X-A-01, J2 <- CTDD-X-A-02,
and B2 <- CTDD-X-A-01 are rule none. All four semantic/reference values remain
none. The other 80 values are preserved, with AI authorship, exposure and original
source-quality provenance. D-to-query rule counts are 17 direct/17 partial/38 none;
12 L-to-demo edges are also adopted, and 12 old adopted L-to-query edges are reused.
Do not modify this new frozen package or any preceding draft/source freeze.

Read rule-calibration.md, relation-review.md and rule-review-audit.json together.
The accepted calibration requires specific matching judgment structures; common
task requirements, ordinary nouns, label agreement/disagreement or grade differences
alone do not establish rule fit. Post-adoption full-table AI review flags 11 remaining
scope-comparability observations. These are AI comments, NOT new human decisions or
replacement values. Do not silently change the adopted records. No rule-fit binary
causal grouping is registered; all conditions remain in fixed-D input comparisons.

The 120 full prompts and token IDs are byte-identical to the reviewed draft; 252
comparison formulas are unchanged (108 fixed-D, 72 interactions, 72 D diagnostics).
New relation IDs, linked L-to-demo IDs and 432 presence rows are consistent. CPU
adoption/schema/source/input/contrast checks pass; the local tokenizer rechecked all
240 bare answer boundaries. No weights or GPU forward were used. Scientific input
design is fixed, but numerical runtime/execution is not yet frozen or qualified.

NEW `docs/research/experiment-plans/cross-term-execution-prep-20260917-v1` is the
next-step CPU preparation, manifest SHA256
`4633d723939534c7e189b774fabed0a05641f1cf49b8e00b1ed5d7c2a46a011b`.
Its model-inputs.jsonl excludes query references and review metadata; analysis-plan
is separate and forbidden to the scoring phase. The pure CPU next_token_math.py
is NOT a model forward executor. Independent 60-digit Decimal checks of 120
synthetic readouts/720 scalars and all 252 expression values AND bounds pass;
max scalar error is about 5.01e-16, expression and bound differences are zero.
Tie/unqualified/repair/alias-cancellation and four invalid-input checks pass.

The proposed single-GPU FP32 eager batch-1/no-cache qualification covers every
input with reference, repeat, left/right padding and reverse request order: 600
engineering plus 120 production prompt forwards, 1,440 candidate values. Twelve
strict free-generation diagnostic prompts are extra. This is a prospective
engineering plan, not GPU acceptance: the bound is still null and old Q01
tolerances are not inherited. Before any model run, implement/source-pin the new
prompt-only forward wrapper and receipt/checkpoint checks, bind verified weights,
runtime and physical allocation, then execute the engineering gate. No GPU job or
new window was started. Current preparation does not require or imply a new human
review of already adopted material. Earlier selectors below remain historical.

## Sixteen D sources adopted; joint design and relation review prepared (2026-09-17)

The user approved all sixteen exact demos and asked to proceed. NEW
`docs/research/experiment-plans/cross-term-demos-v1/current.json` selects
`frozen-01`, manifest SHA256
`39b7466f61cf28e7ff270220a5f63cce93525c41cefe60e5d22ee6ff487feeaf`.
`feedback-01.json` preserves the exact reply and per-demo text/grade/answer scope.
Active references are `demos[].human_review`: eight grade-0, four grade-1 and
four grade-2 cases, with eight 有/eight 无. Keep AI authorship, original proposals
and null original Gold. This reply adopts source quality, NOT the sixty draft
applicability ratings or an execution matrix. The source freeze/validator is
immutable; source-quality adoption and relation provenance stay separate.

`docs/research/experiment-plans/cross-term-joint-design-20260917-v1/README.md`
and `relation-review.md` are the next reviewable design. It reuses all twelve
accepted queries and all sixteen demos without rewriting text. Eighty-four AI
edges comprise 72 D→query and 12 L→demo; source quality is adopted, applicability
is not. D rule proposals count 18 direct/18 partial/36 none, with exact scopes.
Twelve already adopted L→query edges remain external immutable references.
L is assigned independently of D. Per-condition presence records distinguish
actual candidate L from absent/N slots; linked L→demo IDs do not imply exposure
in every condition or a D-triggered retrieval.

The design has 120 unique candidate inputs (108 full same-term contexts plus 12
retained other-term diagnostics). Sixty are new versus the earlier unique frame;
66 old condition aliases point to 60 inputs. The primary proposed contrasts hold
D text/answers/order fixed while changing L/orthographic/absent content. A/B is
retained as a whole-bundle diagnostic, not pure rule-applicability causation.
There are 108 fixed-D contrasts, 72 lexicon interactions and 72 D diagnostics.
Using `.conda/stage1-p0/bin/python`, CPU tokenization checks all 120 prompts and
240 bare-answer boundaries, with 30 exact old-D message bridges. Length is
572–661 tokens; L-minus-N is -1, 5 or 6, so length matching is NOT established.
An independent visible-input/adoption/comparison audit passes. No weights were
loaded, no GPU forward was run, and no numerical acceptance or practical effect
threshold is inherited. Relationship/design review and a separate input/runtime
freeze remain before execution. Preserve all preceding drafts and selectors.

## Cross-term material review frozen; D construction draft started (2026-09-17)

`docs/research/experiment-plans/cross-term-materials-v1/current.json` selects
`frozen-01`, manifest SHA256
`0f3b4222baf0ea6488e39ea32c33dbd551332f405acad26a42622679b2639df2`.
The user explicitly confirmed G1=2 and G4/J4/B4=1, identified their shared oral
insult mode, accepted the remaining v2 material, and requested freeze/next step.
`feedback-01.json` preserves the exact reply and per-object scope. Twelve query
references, three definitions, three candidate form controls and twelve L→query
relations are adopted. Active grades are 2 for G1/J1/B1, 1 for G4/J4/B4, and 0
for the six other queries. Named grade decisions and bulk adoption are distinct.
Original AI proposals, text authorship, null original Gold and both drafts remain.

The three oral-insult queries share a new dependency/calibration group. This is
scoped to those exact sentences; historical #3585 remains 2 and the frozen 0–4
task protocol is unchanged. Preserve the new freeze and its validator in place;
use a new version for changes. CPU material/hash/adoption/schema checks pass.
N inertness/length and a complete input/execution freeze remain unqualified.
No online review, old reannotation exports or historical selectors were changed.

`docs/research/experiment-plans/cross-term-demos-draft-20260917-v1/README.md`
starts the next stage with G2/G4/J2/J4/B2/B4, sixteen new AI demos, eight balanced
有→无 packs and thirty candidate D-only conditions (L absent). Forty-eight D→query
edges include 12 direct, 12 partial and 24 none rule proposals; twelve candidate
L→demo edges are separate and not presented. Source quality of the exact frozen
definitions is reused, but all new demo human fields and new relation adoption
remain empty. Sixty schema/span checks and thirty visible-preview reconstructions
pass. Fifty-four comparisons are proposals, not registered tests. A/B bundles
also change positive-example severity (1/2), stance and construction; shared
有病吧 and reused X packs remain explicit confounds. Do not call this a clean
causal applicability design or infer relationships from answer agreement. No
model forward or GPU task was started. Review this draft before further freezing.

## Cross-term natural-expression revision prepared (2026-09-17)

`docs/research/experiment-plans/cross-term-materials-draft-20260917-v2/README.md`
is the latest review draft. At the user's request for natural whole utterances,
all 12 queries now use conversational scenes, with new `-v2` material/relation
IDs and explicit links to v1. Six existing reviewed cases provide exact wording
references with source hashes and separate historical grade provenance. These
rewrites remain synthetic AI drafts, not original corpus cases or human adoption.

G4/J4/B4 use littering, dog-breed disagreement and a blocked bus door, with
没脑子/眼瞎/有病吧 respectively. B4's AI grade proposal changes from 2 to 1;
its proposed binary label stays 有. J1 and B4 severity boundaries still need
review. Added context and wording are not across-query minimal pairs, and the
same shared construction dependency remains. No new independent family exists.

All 12 revised relations pass the frozen schema, exact span and text-hash checks;
the 36 proposed conditions and 36 comparisons point to the revised queries.
All human fields and original Gold remain null. Lexicon/control texts, v1 draft,
frozen protocol, online review and experiment selectors are unchanged. No model
forward, prompt-frame freeze or execution was performed. Preserve both drafts.

## First cross-term material draft prepared (2026-09-17)

`docs/research/experiment-plans/cross-term-materials-draft-20260917-v1/README.md`
contains the user-requested first draft: 垃圾, 京巴 and 公交车; 12 new synthetic
queries, 3 new single-sense definition wordings, 3 form-control proposals and
36 proposed conditions with demos absent. All labels, grades, definitions and
12 schema-valid lexicon-to-query relations are AI drafts. Every new human field
and original Gold remains null; there has been no material adoption or GPU run.

These terms were exposed in historical review and share construction templates;
they are development/calibration material, not independent confirmation families.
京巴's old definition mixed dog and insult senses; the new wording selects the
regional insult sense and needs its own review. N controls describe spelling;
they are not meaning-free or length-matched, and no full prompt frame is frozen.
The draft manifest binds the completed protocol and historical preference/source
snapshots. Old source adoption states are not copied onto new wordings. Preserve
the frozen protocol, previous drafts and all real human records when revising.

## New task, applicability and primary-scoring protocol frozen (2026-09-16)

The user explicitly authorized freezing the scoring specification and the
previously proposed applicability fields, and closing this protocol stage.
`docs/research/experiment-plans/task-applicability-scoring-v1/current.json`
selects `frozen-01`, manifest SHA256
`d8b8f3a53f0b0be5d3e8726e2b53b93f5a4c16e88e59c4a5d826235c34750fa4`.
This is a COMPLETED normative protocol freeze, not a scientific material or
execution freeze. Preserve the new frozen files and all historical sources.

The model task is byte-identical to the approved v2 prompt: full severity 0–4
anchors, 0→无 and 1–4→有, with one bare character as the final answer. The
relation schema separates source quality, lexicon-to-query/demo sense fit,
demo semantic/reference and rule fit, literal overlap, task relevance and
review/exposure provenance. Partial, unclear, unreviewed and structurally
inapplicable values remain distinct. No material-level human decisions arose.

Primary m is next-token z无−z有, IDs 42192/18830, using one shared prompt-only
forward; no quotes, EOS, length normalization or NCC. Exact ties, numerical
unresolved and unqualified scores are separate. Run-specific error bounds
require a pre-science qualification receipt; no old Q01 acceptance is inherited.
The CPU receipt verifies 10 artifacts, 13 prior sources, 5 tokenizer files,
6 relation fixtures, 6 invalid-record rejections, 8 prompts/16 answer boundaries
and synthetic arithmetic/bound propagation. No weights or GPU forward were used.
Fixtures are not scientific materials. Next is material construction/review,
then a separate input/execution freeze. Earlier discussion/review drafts keep
their historical pending status; use this new entry for current protocol state.
All old experiment selectors and online human fields remain unchanged.

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
