# WP3 S2.1 development review handoff

> DEVELOPMENT ONLY / NON-SEALED / NON-SCIENTIFIC

> **2026-08-30 protocol amendment:** this legacy Phase-B queue was superseded
> after the 424 raw annotations were locked. Do not complete the 6,306 legacy
> decisions and do not run `finalize-dev-gold` for S2.1b. Preserve this frame,
> session, export, and service byte-for-byte. The accepted current-generator
> lifecycle is defined in
> [`wp3-s21b-current-generator-revision.md`](wp3-s21b-current-generator-revision.md).

This document now describes the preserved legacy S2.1 lifecycle. It does not
call a model, access the network, create an S2.2 sample, or write
`lexicon_ref.json`. Its locked raw annotations remain an input to S2.1b, but
its historical proposal queue is no longer a completion gate.

## Frozen frame

The active locator is:

`exps/causal_context/stage1_p0/wp3_candidate_generators_v2/refs/development_frame_ref.json`

The current frame contains 424 unique formal-fit records: 200 historical A1
records plus 240 historical dual-model records with a 16-record intersection.
The dual-model package's 48 hidden-repeat pages are excluded. The frame is
reconstructed from the formal fit artifact; old package text is never trusted.

Run the independent validator before review:

```bash
/usr/bin/python3 scripts/stage1/wp3_candidate_review.py validate-dev-frame
```

## Start or resume review

Choose a stable reviewer ID and keep using exactly the same value:

```bash
/usr/bin/python3 scripts/stage1/wp3_candidate_review.py serve-dev-review \
  --reviewer-id REVIEWER_ID
```

Open the printed `127.0.0.1` URL. The service writes drafts after a short
debounce and uses revision/CAS checks. Confirmed items require an explicit
reopen and amendment reason before they can change.

Phase A exposes only `case_id`, `blind_alias`, and the formal-fit `content`.
Each mention is entered as an exact surface plus a 1-based occurrence ordinal;
the server resolves offsets. A/B/C routes are provisional development
hypotheses, not formal evidence tiers. Do not create R mentions.

Phase B is not sent to the browser until all 424 Phase-A cases are confirmed
and the raw lock is committed. It contains 6,306 de-duplicated anonymous
proposals in the current frame. Source/model identities, descriptions,
confidence, source counts, task labels, and historical category strata are not
shown. `defer` is allowed while reviewing but must be reduced to zero before
finalization.

The export button creates a ZIP containing annotations, a draft reviewer
declaration, `manifest.json`, and `SHA256SUMS`. It is a working export, not the
immutable gold artifact.

## Legacy finalize and report

The following commands are retained only to validate the historical lifecycle.
Under the accepted S2.1b amendment they must not be used to force completion of
the 6,306 legacy proposals.

After both phases are complete:

```bash
/usr/bin/python3 scripts/stage1/wp3_candidate_review.py finalize-dev-gold \
  --reviewer-id REVIEWER_ID

/usr/bin/python3 scripts/stage1/wp3_candidate_review.py report-dev
```

Finalization revalidates the formal data and train-partition dependencies,
handbook, generator config and implementation, annotation schemas, 424 raw
annotations, every offset, all diagnostic decisions, amendments, reviewer ID,
and all artifact hashes. Finalized gold is immutable; corrections require a
new review revision and a new content-addressed artifact.

The report is also content-addressed and independently validated. Historical
A1 and dual-model metrics remain diagnostic only. Current G3 metrics are
development diagnostics and do not authorize candidate publication.

## S2.2 boundary

No S2.2 sampler, frame, hidden repeats, gold, or gate exists in this
implementation. Start S2.2 only under a separate plan after S2.1 gold/report,
the G1/G2 pilot, mechanism/normalizer/provider freezes, and explicit model-call
authorization.
