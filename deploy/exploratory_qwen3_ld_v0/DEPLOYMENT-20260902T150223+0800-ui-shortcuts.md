# Exploratory Qwen3 L/D audit UI clarity and shortcut update

- Activated: `2026-09-02T15:02:23+08:00`
- Public origin: `https://hsd.fenglin.pro`
- Release/archive ID: `30eaacdbf56d90a553bdf55634d066b4dfd1b01be03e020e3e220a3710299f90`
- Candidate frame: `pilot-frame-20a595e31db65048612e0570e78fb4d9bd4ffd43c4d6b9b672451fd323de45bf`

This UI-only release makes the DefinitionSwap judgment direction explicit,
replaces wrapped pill-shaped pragmatic tags with a stable two-column card
layout, and exposes direct keyboard shortcuts on every affected control.

The underlying decision schema, backend, candidate frame and session path were
not changed. The session SHA-256 was identical before and after deployment:
`03b10770ae3104efc81077e533ce8eea368ca1cbf59a6dad2071a8d0c184a41e`.
The state remained `0 / 128` confirmed with zero deferred decisions and zero
amendments.

## Keyboard mapping

- `Q / W`: relevance pass/fail; reused as No-hit confirmed/missed because the
  two panels are mutually exclusive;
- `E / R`: span boundary pass/fail;
- `A / S / D`: definition good/usable/poor;
- `F / G`: current-sense pass/fail;
- `Z / X`: swap effective/ineffective;
- `Shift+1` through `Shift+6`: toggle the six pragmatic tags;
- existing disposition, navigation, save and confirmation shortcuts remain
  unchanged.

## Verification

- core shortcut and validation tests passed;
- JavaScript syntax checks passed on both UI modules;
- pilot Python tests passed (`4 passed`);
- the desktop page was rendered against the real 128-item local frame before
  deployment;
- deployed frontend hashes matched the local release:
  - `index.html`: `c889a00ab7361260fa4ea3e1d1c446e977de903dec128793697ee7d921e529aa`
  - `app.js`: `399876900a235fc0582b031e53978cb0b54f3ca5942aeffadabba53a25518c54`
  - `core.js`: `d5018b2cc8b4196f80a13e4331588b6fb7757605d6e6a1016da0d9a4b9543434`
  - `styles.css`: `ffb0e605ece6e80d22f7da67230ca993882d43b8c94bbbfcebc5b09752454934`
- the service is active/running with `NRestarts=0` and
  `ExecMainStatus=0`;
- the public HTTPS endpoint continues to require the existing Basic Auth realm.

The previous content-addressed release remains available at
`/opt/hsd-qwen3-ld-pilot-review/releases/3ef9d7fd10ae6c800e54f1c42b0b170d3fbde8fe916a4ecdb18ea92de77e47cc`
for rollback.
