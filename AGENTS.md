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

## Paired-case deployment and authoritative records

The authoritative paired-case session currently remains on the development machine:
`exps/causal_context/general_model_ld_nolabel_paired_cases_v1/reviews/paired-cases-02/session.json`.
Its local web service and aliyun reverse tunnel remain the active review path.
DigitalOcean deployment is prepared, but the actual human session has not been
migrated. Keep its production unit stopped and HTTPS in maintenance mode until
an explicitly authorized, verified session cutover is completed.

The prepared target is `digitalocean-sgp` (`165.22.48.237`), serving
`https://hsd.fenglin.pro`. The unit `hsd-general-model-paired-review.service` runs
as `hsd-review`, binds to `127.0.0.1:8772`, and uses
`/opt/hsd-general-model-paired-review/current`, a symlink to
`releases/<full-archive-sha256>`. Its intended authoritative session is
`/var/lib/hsd-general-model-paired-review/session.json`. The unit requires a
nonempty existing session before startup, preventing accidental blank production
sessions while migration is pending.

- Build with `deploy/general_model_paired_review/build_release.py`. Verify archive
  and per-file hashes; preserve the original frozen manifest bytes. Package only
  the code/static dependency closure and indexed discovery artifacts, excluding
  human sessions, credentials, logs, models, reserve cases, and unused inputs.
- Maintain one writable authoritative session. For an authorized migration, stop
  the old writer, back up and verify exact session bytes, preserve reviewer/source
  identity, and check that restarts retain progress.
- After successful cutover, keep the local session as a backup and place
  `session.json.remote-authority.json` beside it. The CLI and old forwarding
  script then refuse to start its stale writer. Do not copy this marker beside
  the remote production session. No marker is needed before cutover.
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

## Private SSH access

- Before cutover, manage the local web service and reverse tunnel with
  `bash deploy/general_model_paired_review/review-forward.sh start`, `status`,
  or `stop`. The browser computer forwards its `127.0.0.1:8772` to
  `aliyun 127.0.0.1:18772`, which reaches the development service.
- After cutover, use
  `bash deploy/general_model_paired_review/digitalocean-sgp/private-access.sh`
  with `start`, `status`, or `stop`. Its independent tmux session forwards local
  `127.0.0.1:8772` directly to `digitalocean-sgp 127.0.0.1:8772` without starting
  a local writer. `run` is foreground mode, stopped with Ctrl-C.
- Bind services and SSH listeners to loopback. Reuse existing aliases and keys,
  require known-host verification and batch authentication, and use
  `ExitOnForwardFailure`, 30-second keepalives and a three-failure limit.
  Supervisors retry after three seconds and refuse occupied ports.
- Use working process managers: systemd remotely and dedicated tmux sessions in
  this container. Stop only this task's processes. tmux survives terminal closure;
  rerun the appropriate `start` after a container restart.
- Keep browser-side port 8772 aligned with application Host/Origin checks. Check
  backend, SSH listeners, and browser access separately. Protect every public
  application path with HTTPS authentication; bootstrap tokens do not authenticate
  reviewers.
