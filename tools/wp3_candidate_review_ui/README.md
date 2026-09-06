# WP3 S2.1 local development reviewer

Use the fixed Stage-1 CLI; do not serve these files with a generic web server:

```bash
/usr/bin/python3 scripts/stage1/wp3_candidate_review.py serve-dev-review \
  --reviewer-id REVIEWER_ID
```

The service binds only to loopback. It validates Host/Origin, applies a strict
CSP, uses a random session token and a request-size limit, and writes every
draft through a revision/CAS check. The browser submits only surfaces and
occurrence ordinals; the server resolves and persists offsets.

The reviewer UI is a three-pane workbench on desktop and a full single-column
workflow on phones. Phone-only reason and proposal selectors open as full-width
bottom panels, while the case queue uses a full-screen drawer. It loads the
bootstrap endpoint once, fetches only the active case through the per-case
endpoint, and keeps at most five cases in
memory. Draft mutations return compact deltas. The legacy full-state endpoint
remains available for one compatibility cycle, but the browser no longer uses
it.

Draft writes are serialized and bound to a captured case/proposal payload.
Navigation flushes valid pending work and stops on validation or network
failure. Confirmed items remain immutable until an amendment reason is
recorded through the reopen flow.

Phase B is not included in the browser state until all 424 Phase-A cases are
confirmed and the raw lock is committed. Exports contain annotations, a draft
reviewer declaration, a manifest, and `SHA256SUMS`. The immutable declaration
is emitted only by `finalize-dev-gold`.

## UI tests

The zero-dependency controller tests run with:

    node tools/wp3_candidate_review_ui/test_core.cjs

Browser tests require Node 20 or newer and the development-only Playwright
dependency:

    npm install
    npx playwright install chromium webkit
    npm run test:ui-e2e

The Playwright suite intercepts review APIs with deterministic raw and
diagnostic fixtures. It covers desktop, compact desktop, tablet, and two phone
viewports; no Playwright runtime is deployed with the review service.

## Project-local change records

UI requirements, evidence, implementation results, and deployment anchors are
kept inside this annotation project so the directory can be extracted from the
larger HSD repository without losing its operational history.

- [2026-08-29 usability round 2: quick A/B/C annotation, frozen-content search,
  and visible-queue navigation](docs/usability-round-2-20260829/README.md)
