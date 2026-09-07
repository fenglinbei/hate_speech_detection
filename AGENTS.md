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

## Public access through aliyun

For this user's review pages, the established forwarding destination is SSH host
`aliyun` and HTTPS origin `https://hsd.fenglin.pro`. When forwarding or publication
is requested or already authorized, reuse that route and the existing login.
Inspect the active domain configuration and upstream before switching tasks.

The paired-case implementation and operational record are in
`deploy/general_model_paired_review/README.md`. Its current route is:

`hsd.fenglin.pro:443 -> aliyun Nginx -> aliyun 127.0.0.1:18772 -> SSH reverse tunnel -> local 127.0.0.1:8772`.

- Run the review server with `--public-origin https://hsd.fenglin.pro` so the
  intended Host and Origin are accepted by both page and save endpoints.
- Bind both the review server and reverse listener to loopback. Reuse the SSH
  alias and existing keys; do not copy credentials into the repository or logs.
- Preserve the site's TLS certificate, Basic Auth, forwarded Host and Origin,
  request headers, and body-size limit. Back up the current Nginx site before
  changing its upstream, run `nginx -t`, and reload only after validation succeeds.
- Use `ExitOnForwardFailure`, SSH keepalives, and a persistent process manager
  with retries. Check that systemd actually works before choosing it. This
  development container has no usable systemd bus, so the current implementation
  uses a dedicated tmux socket/session with separate web and tunnel retry loops.
- Manage this instance with `bash deploy/general_model_paired_review/review-forward.sh
  start`, `status`, or `stop`. It resumes the existing session and refuses to
  compete with an occupied local port. Stop only processes owned by this task.
  tmux survives terminal closure; after the development container restarts, run
  `start` again. Prefer a working service manager when automatic boot is needed.
- Keep human records and runtime logs under the experiment's ignored `reviews/`
  directory. Preserve a session backup before migration, and verify that process
  restarts retain review progress. Do not reopen retired review services merely
  to reuse their public route.
- Check the local service, the remote loopback tunnel, Nginx/TLS/login protection,
  and external domain access separately. Healthy loopback requests do not prove
  that the public URL works. Use static assets and health endpoints for deployment
  checks; any save/confirm test belongs in an isolated test session.
- On 2026-09-07, this route was configured and the tunnel worked, but external
  access returned Alibaba Cloud's `Non-compliance ICP Filing` block. The user
  confirmed that this domain has no filing and that their other filed domain's
  business scope does not include this review. Login credentials do not resolve
  the provider-side block. Use the documented private SSH access for current
  review; do not substitute the other domain without resolving that constraint.
  Revalidate externally after the hosting/filing situation changes before
  claiming the public page is usable.
