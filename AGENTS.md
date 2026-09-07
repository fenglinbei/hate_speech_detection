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

## Private SSH access through aliyun

For review workbenches, reuse private SSH forwarding through the existing
`aliyun` host alias. The paired-case setup and commands are documented in
`deploy/general_model_paired_review/README.md`.

The access path is the browser computer's `127.0.0.1:8772`, through a local SSH
forward to `aliyun 127.0.0.1:18772`, then through the development machine's reverse
SSH tunnel to its review service on `127.0.0.1:8772`.

- Bind the review service and both SSH listeners to loopback. Reuse the existing
  SSH alias and keys; keep credentials out of repository files and logs.
- Use `ExitOnForwardFailure`, SSH keepalives, and a persistent process manager
  with retries. Check that systemd actually works before choosing it. This
  development container uses a dedicated tmux socket/session with separate web
  and tunnel retry loops.
- Manage the development side with
  `bash deploy/general_model_paired_review/review-forward.sh start`, `status`,
  or `stop`. It resumes the existing session, protects against duplicate starts,
  and refuses to compete with an occupied local port. Stop only this task's
  processes. tmux survives terminal closure; run `start` again after a container
  restart, or use a working service manager when automatic boot is needed.
- On the browser computer, use
  `ssh -NT -o ExitOnForwardFailure=yes -L 127.0.0.1:8772:127.0.0.1:18772 aliyun`,
  then open `http://127.0.0.1:8772/`. Use the documented keepalive options for
  longer sessions. This client-side command belongs on the browser computer,
  not on the development machine where port 8772 is already occupied.
- Keep the browser-side port aligned with the application's allowed Host/Origin.
  When adapting this setup to another task, check both ports and the application
  allowlist rather than changing a port in only one command.
- Keep human records and runtime logs in the experiment's ignored `reviews/`
  directory. Back up the session before migration and verify that process
  restarts preserve progress.
- Check the local service, remote reverse listener, and client-side forward
  separately using health endpoints and static assets. Any save/confirm test
  belongs in an isolated test session, without populating real human records.
