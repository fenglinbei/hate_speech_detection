# WP3 S2.1 remote deployment

The production shape is Nginx TLS + Basic Auth in front of the annotation
service bound to `127.0.0.1:8766`. The application additionally checks the
exact public HTTPS Origin and retains its random per-process session token and
revision/CAS checks.

Persistent human work lives only in `/var/lib/hsd-wp3-review/session.json`.
Application releases under `/opt/hsd-wp3-review/releases/` are immutable and
may be replaced without replacing the annotation session.

## UI upgrade rollout

This UI release keeps the v1 review session, raw lock, CAS revision, and export
contracts unchanged. For the approved clean-session rollout:

1. Stop the review systemd service.
2. Copy the active session to a timestamped, mode-0600 backup owned by the
   review service account.
3. Move the active session out of the canonical path; never delete the backup.
4. Install the new immutable release and atomically update the current symlink.
5. Start the service. It creates a fresh v1 session for the configured reviewer.
6. Verify Basic Auth and complete a browser smoke test: load stage A, select a
   span, save, refresh, confirm, and export. Remote curl behavior is not part of
   this rollout.

Rollback stops the service, restores the prior release symlink, and either
keeps the new clean session or restores the timestamped pre-upgrade backup.
Do not combine annotations from the two sessions.

The completed 2026-08-29 full rollout and its exact recovery anchors are
recorded in `DEPLOYMENT-20260829T151622+0800.md`. The later static-only ABC
criteria guide release, which preserved the active review session byte for
byte, is recorded in `DEPLOYMENT-20260829T165132+0800.md`.

The bootstrap Nginx file exposes only the ACME HTTP-01 path and returns 503 for
all application requests. Install the TLS Nginx file only after public DNS
resolves `hsd.fenglin.pro` to the server and a trusted certificate exists.
