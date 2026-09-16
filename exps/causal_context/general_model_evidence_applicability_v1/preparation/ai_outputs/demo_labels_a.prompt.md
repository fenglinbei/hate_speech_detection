# AI demo label draft generation record

Generator: demo_ai_drafts_a
Review kind: ai_note
Input scope: demo_text_and_policy_only
Model revision: not_exposed
Batch context: 140 demo texts; no query Gold, demo original answers, or predictions

Assigned task / substantive generation instructions:
Generate initial hate and group label drafts independently for every demo in the two assigned 70-record JSONL text inputs, using only each demo text and the frozen evidence-applicability-annotation-policy/v1. Do not read queries, query Gold, original demo answers, original status labels, predictions, other experiment data or other drafts. Do not infer an omitted target or stance from adjacent demos. Separate actual evaluation targets from background identity mentions and separate author stance from quoted speech. Hate and group are independent. Keep pure personal or institutional offense unresolved where the frozen task boundary is unspecified; ordinary behavior criticism may be non-hate. A clearly defended group can still have a group category. Keep complete group sets null where nationality/ethnicity, individual/institution, historical coverage or another material category boundary cannot be resolved. Identify the specific ambiguity; missing context by itself is not sufficient. Do not convert AI output to human review.

Exact output contract:
One JSON object per assigned id with values hate, group, hate_reason, group_reason, stance, target_types, expression_types, evidence and brief Chinese note; provenance is retained on every object. No original-answer comparison statuses are included.

Generation method:
Each semantic decision, reason, stance, target/expression choice, evidence quotation and Chinese note was manually authored by the assistant. Python only expanded those authored rows into the requested schema, located literal quote substrings, calculated Unicode code-point half-open offsets and validated record coverage/enumerations/offsets. No automatic keyword-to-label heuristics were used. Both input batches appeared in the same assistant context; no labels were derived from neighboring records or any original answers. Shared repository/workspace instructions and this bounded assignment were also available as operational context.

Read source identities (SHA-256):
- docs/research/annotation-guidelines/evidence-applicability-annotation-policy-v1.md: 660d973c6248b17068ce44f68ebc4680515e54fd17354bd51650ace7c436557a
- exps/causal_context/general_model_evidence_applicability_v1/preparation/ai_inputs/demos_part1.jsonl: 9de7a82939403f7df85763e743b87bcb550ae77e17ec7150966baf54cb81c99f
- exps/causal_context/general_model_evidence_applicability_v1/preparation/ai_inputs/demos_part2.jsonl: 6b95914aa4fec21881b0261383ecb8944bf615d5107a6bbad749f550b1384690

Validation: 140 unique ids in exact input order, 225 exact evidence spans, permitted labels and provenance checked.
