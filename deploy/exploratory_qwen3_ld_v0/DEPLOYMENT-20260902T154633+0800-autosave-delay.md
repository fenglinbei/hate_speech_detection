# Audit autosave debounce update

- Activated: `2026-09-02T15:46:33+08:00`
- Public origin: `https://hsd.fenglin.pro`
- Release/archive ID: `2d7415fe745ca6dfbc282b0a22232b5d5eb1320e1b7e3a5e70222cf4aee80a3c`
- Deployed `app.js` SHA-256:
  `2eef20b596fd438160098760c4c584176aab2b465937d8fd89eb205059abd10e`

The automatic draft-save debounce was increased from 700 ms to 5,000 ms so a
short sequence of audit choices is coalesced into one save. The status line now
shows `有未保存修改 · 5 秒后自动保存` while the timer is pending. Explicit save,
confirmation and navigation flush behavior remain immediate.

This was an `app.js`-only behavior change. The frame, backend, schema and audit
session were not changed. Immediately before and after deployment, the session
had 12 confirmed decisions, zero deferred decisions and zero amendments. Its
SHA-256 remained:
`bf31fe1dc89021a28bd3d081f8eb809ed912c5ed042208d37ad12e5d9614f722`.

JavaScript syntax and UI core tests passed. The deployed service was
active/running with `NRestarts=0` and `ExecMainStatus=0`.
