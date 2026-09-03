# Annotated lexicon repair v1: span-gold deployment record

Deployment date: 2026-09-04 (Asia/Shanghai)

This record covers the first, independent review stage of the bounded
annotated-lexicon repair.  The public page contains development queries,
legacy candidate spans, known omission candidates, and prior audit notes.  It
contains no sealed labels, demonstrations, model outputs, model weights, or
test data.

## Frozen inputs and release

- Git implementation commit: `fcfd809f2`
- Public origin: `https://hsd.fenglin.pro`
- Release archive SHA-256 / release ID:
  `42625b2c7d5a104755fabadf598fa9d3bfaa1314c31c63200e86037447fb80be`
- Frozen frame ID:
  `span-gold-frame-58982b9b0867881940e3ba344bd94e552cc583f0c548107c39d9a1fbf1f49291`
- Frozen frame file SHA-256:
  `76d017a74d398dcd1aecee998f76749c389ee03b657a9c2ddcd612bef78afef6`
- Review size: 39 items (34 rejected pilot items, 3 accepted controls, 2
  prior adjudications)

The release is active through:

`/opt/hsd-annotated-lexicon-repair-review/current`

which points to the content-addressed directory below
`/opt/hsd-annotated-lexicon-repair-review/releases/`.

## Pre-cutover rollback baseline

- Formal G3 service: `hsd-wp3-g3-review.service`, active with zero restarts,
  loopback port `8767`
- Formal G3 release ID:
  `ea64226300b565012cf55b2d5bc25ef373f0c5882be62af47619b93ccdcfd15d`
- Formal G3 session SHA-256:
  `2065a3685454d764ab6be6e2a33489906967c35e062f6ac80f5198ab08c69207`
- Previous Nginx site SHA-256:
  `4e48bd7b24447826912a72255e26b90132f3066c3e81e3dbc0890e480cf9f42b`
- Previous Nginx configuration backup:
  `/etc/nginx/sites-available/hsd.fenglin.pro.pre-annotated-lexicon-span-gold-20260904-4e48bd7b`

The formal G3 process remained running throughout the cutover.  Its session
hash was unchanged after the new public route was enabled.

## Post-cutover verification

- `serve-span-gold --check`: expected frame ID, `0 / 39`, development-only
- Repair service: active/running, zero restarts
- Repair listener: only `127.0.0.1:8769`
- Initial repair session SHA-256:
  `6d0b0a275a1490589484471a2b7d6fee1f85b7837c9e19cc5d8a928cf7471d00`
- Initial repair session revision:
  `345f51ca285be345900a9c9f4a41c6c0164f799c4966ca5534344f372cd556ad`
- Initial repair status: 0 confirmed, 39 open, 0 amendments
- Nginx configuration validation: successful
- Active Nginx site SHA-256:
  `f229557832b65fa18926891c036b7caab974d94a167b909755e50481c28f9e06`
- TLS/SNI request through Nginx: HTTP 401 with the expected
  `Annotated lexicon span review` Basic Auth realm and security headers
- Direct upstream health: `{"stage":"span-gold","status":"ok"}`
- Direct upstream bootstrap: exact frame ID and 39 items; the session token
  was deliberately excluded from deployment logs

No production smoke decision was submitted.

## Stage completion rule

All 39 items must be confirmed in this one pass.  Finalization is allowed only
when there are zero open items and no deferred decisions.  The exported
browser/server snapshot and checksums must be retained before the next
content-addressed repair-operation stage replaces this public route.

