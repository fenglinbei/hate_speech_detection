# Human Review Workbenches

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
