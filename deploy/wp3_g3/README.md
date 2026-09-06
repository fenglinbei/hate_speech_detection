# WP3 G3 form-relation review deployment

This directory prepares an independent G3 human-review service. It does not
replace, migrate, or reuse the existing S2.1 annotation session.

The intended production shape is the existing Nginx TLS and Basic Auth layer
in front of a new application process bound only to `127.0.0.1:8767`:

- release root: `/opt/hsd-wp3-g3-review/releases/<release-id>`;
- active release link: `/opt/hsd-wp3-g3-review/current`;
- systemd unit: `hsd-wp3-g3-review.service`;
- persistent session: `/var/lib/hsd-wp3-g3-review/session.json`;
- exact public origin: `https://hsd.fenglin.pro`;
- existing Basic Auth file: `/etc/nginx/.htpasswd-hsd-wp3-review`.

All G3 artifacts and the review itself remain
`development-only / non-sealed / non-scientific`.

## Frozen artifact binding

The service unit is bound to the independently validated artifacts below:

- source bundle: `wp3g3sources-d190a80c1e4fb970ee7495a3f3ce62f569cd2e0e8187e2d265868222b3fb1c5b`;
- relation-review frame: `wp3g3formframe-3ddee0733a9bf9bee2cf2f3cc1448fd2ec747e3ddb6b35100bbc3a79b26d1e00`.

Keep these explicit content-addressed directories in `ExecStart`; do not
substitute mutable locator refs. Reject any staged unit containing a
`__WP3_G3_` placeholder.

## Safe rollout order

1. Record the active S2.1 release symlink, the SHA-256 and metadata of
   `/var/lib/hsd-wp3-review/session.json`, and the current Nginx site SHA-256.
2. Install the tested repository snapshot as a new, immutable directory below
   `/opt/hsd-wp3-g3-review/releases/`, owned by `root:hsd-review` and not
   writable by the service account. Atomically point `current` to that release.
3. Stage the substituted systemd unit and run the CLI's
   `serve-g3-reference --check` with the exact source bundle, frame, reviewer,
   session, and public-origin arguments. The check must report the expected
   frame ID and item count.
4. Install and start `hsd-wp3-g3-review.service`. Verify that it listens only
   on `127.0.0.1:8767`, has zero restarts, and created the G3 state directory
   with mode `0700` and session file with mode `0600`.
5. Make a recoverable backup of the enabled Nginx site. Stage
   `hsd.fenglin.pro.nginx`, run `nginx -t`, and reload Nginx only after the new
   loopback service passes its read-only bootstrap check.
6. Through HTTPS, verify the existing Basic Auth credential, the G3 frame ID,
   the initial `0 / N` confirmation state, CSP/security headers, and the absence
   of browser network requests to third-party origins. Do not make a trial
   annotation against the production session.

The Nginx template retains the existing certificate paths, TLS policy, HSTS,
security headers, log paths, request limits, and htpasswd file. Only the
browser-visible Basic Auth realm and upstream port differ from the current
S2.1 site.

## Rollback

Restore the backed-up Nginx site pointing to `127.0.0.1:8766`, run
`nginx -t`, and reload Nginx. Stop the G3 service if it is no longer exposed.
Do not delete either review session and never copy annotations between
`/var/lib/hsd-wp3-review/session.json` and
`/var/lib/hsd-wp3-g3-review/session.json`.

## Read-only baseline observed on 2026-08-30

The remote audit made no changes. At `2026-08-30T22:36:14+08:00`:

- Nginx and `hsd-wp3-review` were enabled and active; both reported zero
  restarts.
- `hsd-wp3-review` listened only on `127.0.0.1:8766` and used release
  `eb4f9215074d1776f534c1f389101c6946cdfd21f4e430031eb22a391aa3bd9c`.
- the existing S2.1 session was mode `0600`, owned by
  `hsd-review:hsd-review`, in `diagnostic` phase, and had SHA-256
  `13895c8f4076b48fd2fe49a49da7247bcaa54809168cb37518c78f2c7935f8c9`;
- the existing Let's Encrypt certificate covered `hsd.fenglin.pro` and was
  valid through `2026-11-27T04:50:15Z`;
- Nginx used `/etc/nginx/.htpasswd-hsd-wp3-review` and proxied HTTPS requests
  to port `8766`.

Re-audit these mutable anchors immediately before any rollout; this section is
not deployment authority.
