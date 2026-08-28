# Stage 1 data adjudication rubric (v1)

This rubric applies unchanged to train, dev, and test. Reviewers must not inspect
experimental-condition outputs, model predictions, scores, or downstream metrics.

## General rules

1. Complete exactly one adjudication row for every blocking `issue_id` in the
   exported template. Do not add, remove, merge, or rename issues.
2. `accepted` preserves the source annotation and therefore requires an empty
   `edits` list. It is allowed only when the issue explicitly has
   `accept_allowed=true`.
3. `corrected` requires at least one `set` edit on an issue-authorized JSON
   pointer. Do not change text or labels outside the issue's allowlist.
4. Every row requires a non-empty reviewer ID, review time, reason code, and
   free-text reason. The reviewer ID must match the signed declaration.
5. Target and argument corrections must be an explicit JSON string or JSON
   `null`; never use the string `"NULL"` in corrected data.

## Issue-specific decisions

### `group-hate-atypical`

A concrete protected-group annotation with `hateful="non-hate"` may be a valid
independent field combination. Preserve it with `accepted` and reason code
`valid-independent-label-combination` when the annotation is semantically
supported. Otherwise use `corrected` with `correct-source-label` or
`resolve-group-hate-conflict`.

### `hateful-null-sentinel`

The source value `hateful="NULL"` is unknown, not non-hate. It cannot be
accepted. Set `hateful` explicitly to `hate` or `non-hate` using
`replace-legacy-null` or `correct-source-label`.

### `group-hate-conflict`

`targeted_group="non-hate"` together with `hateful="hate"` requires an explicit
correction of the group, the hateful label, or both. Use
`resolve-group-hate-conflict` or `correct-source-label`; it cannot be accepted as
written.

### `non-string-quad-field`

Numeric target/argument cells cannot enter the canonical quad protocol. Inspect
the source content and set the field to the intended text with
`coerce-numeric-annotation`, or to JSON `null` with `set-explicit-null`. The
numeric value cannot be accepted unchanged.

### `duplicate-record-id`

Duplicate record IDs cannot be accepted. Correct the erroneous ID with
`correct-source-record-id`; the finalized train/dev/test ID sets must be globally
unique.

## Non-blocking warnings

Target or argument text that is not an exact substring of `content` is reported
for provenance only. It is not exported into the blocking adjudication template
and must not be auto-rewritten from substring heuristics.
