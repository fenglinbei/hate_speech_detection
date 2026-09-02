# Pragmatic-tag audit guidance update

- Activated: `2026-09-02T15:20:57+08:00`
- Public origin: `https://hsd.fenglin.pro`
- Release/archive ID: `3ab14f1af8ae6c6a0800758f6f70b01ae893e382fa171b1ba0a9d822a878690b`
- Candidate frame: `pilot-frame-20a595e31db65048612e0570e78fb4d9bd4ffd43c4d6b9b672451fd323de45bf`

The audit-help dialog now explains that pragmatic tags describe how the query
uses an expression, may be combined, and should all remain empty when there is
no explicit textual evidence. Each of the six tags has separate applicable and
non-applicable criteria. The direct-stance example
`反女拳就要从你这种男人开始！` is included to distinguish lexical opposition
from propositional negation and counterspeech.

This was a static HTML/CSS release only. The backend, decision schema, frame
and session path were unchanged. The audit session SHA-256 was identical before
and after deployment:
`d74b557634beb48a52f67d57fc944c61b5bc49b8cff958e3f720e493d4ada22e`.

Verification:

- guidance HTML parsed successfully and contained every required rule;
- JavaScript syntax and shortcut/core tests passed;
- deployed `index.html` SHA-256:
  `32cef1d037a541d0caf834e3097cb374ddf29ebcbae936801e54d80b8b44fd51`;
- deployed `styles.css` SHA-256:
  `299a61ccb12829d667c3a7eebddfc5e883cf8cf1d77a11f7eec0da2d7553f916`;
- service state was active/running with `NRestarts=0` and
  `ExecMainStatus=0`;
- the public HTTPS endpoint retained its existing Basic Auth protection.
