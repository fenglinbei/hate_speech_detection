# Annotated lexicon repair v1 review deployment

Current status (2026-09-04): repair experiments and further annotation are
paused by user decision. The repair review unit is stopped and disabled;
the partial stage-two session is backed up, not finalized. See the
[pause and resource-freeze decision](../../docs/research/experiment-plans/lexicon-resource-freeze-decision-20260904.md).
The deployment instructions and initial-release status below are historical.

This deployment adapts the established WP3 2.1 / G3 workbench flow for the
three independent repair review stages.  The first release serves only the
39-item `span-gold` frame.  It contains development query text, legacy hits,
known omission candidates, and prior input-audit notes; it contains no task
gold labels, demonstrations, model outputs, model weights, or sealed-test
data.

As of 2026-09-04 the public route serves the second, independent
`repair-operation` stage (56 cards). The first-stage session is frozen and
preserved. See the [stage-two deployment and review handoff](../../docs/research/experiment-plans/annotated-lexicon-repair-v1-operation-deployment.md)
for the exact frame, runtime, active release, verification and rollback.
The stage-two unit template is `hsd-annotated-lexicon-operation-review.service`,
installed under the existing service name; its session is
`/var/lib/hsd-annotated-lexicon-repair-review/repair-operation-session.json`.
The first-release instructions below are retained as historical baseline.

Production shape:

- release root: `/opt/hsd-annotated-lexicon-repair-review/releases/<release-id>`;
- active release link: `/opt/hsd-annotated-lexicon-repair-review/current`;
- systemd unit: `hsd-annotated-lexicon-repair-review.service`;
- stage-specific persistent session:
  `/var/lib/hsd-annotated-lexicon-repair-review/span-gold-session.json`;
- loopback upstream: `127.0.0.1:8769`;
- exact public origin: `https://hsd.fenglin.pro`;
- existing Basic Auth file: `/etc/nginx/.htpasswd-hsd-wp3-review`.

## Safe rollout

1. Read and record the current `current` symlinks, service states, enabled
   Nginx-site hash, and formal G3 session hash.  Do not treat prior deployment
   notes as current authority.
2. Build a content-addressed release archive containing only the repair CLI,
   repair storage module, review UI, shared review assets, and frozen stage
   frame.  Verify its SHA-256 before and after upload.
3. Extract below the release root, make the release immutable to the service
   user, and atomically update `current`.
4. Install the staged unit and run `serve-span-gold --check` against the exact
   remote release/frame/session/public-origin arguments.  It must report the
   expected frame ID and `0 / 39` for a new session.
5. Start the service and verify it listens only on `127.0.0.1:8769`, has zero
   restarts, and writes only its owner-only StateDirectory.
6. Back up the active Nginx site, install the repair template, run `nginx -t`,
   then reload.  Verify HTTPS first returns Basic Auth 401, and an
   authenticated bootstrap returns the exact frame and expected revision.
   Do not submit a production smoke decision.

## Stage transitions and rollback

Each later review stage receives a new content-addressed frame and a separate
session file.  Decisions are never copied between stages.  Before every
transition, stop writes, download and verify the browser/server snapshot, and
preserve the completed session.

To roll back public access, restore the backed-up formal G3 Nginx site pointing
to `127.0.0.1:8767`, run `nginx -t`, reload Nginx, and stop this service.  Do
not delete releases or review sessions, and never copy repair decisions into
the formal G3 session.
