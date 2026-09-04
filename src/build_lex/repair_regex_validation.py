"""Fail-closed validation for reviewer-authored controlled match policies.

Only JSON structure is inspected in the HTTP process.  The pinned ``regex``
engine is imported and compiles expressions in a short-lived, resource-limited
child process; arbitrary reviewer expressions are never compiled here.
"""

from __future__ import annotations

import json
import subprocess
import sys
from typing import Any

from build_lex.annotated_lexicon_repair import LexiconRepairError


REGEX_VERSION = "2026.4.4"
MAX_PATTERN_CHARS = 256
MAX_RULES_PER_ENTRY = 8
MAX_RULE_ID_CHARS = 128
MAX_WORKER_INPUT_BYTES = 32 * 1024
WORKER_WALL_TIMEOUT_SECONDS = 3.0
WORKER_MEMORY_BYTES = 256 * 1024 * 1024
_RULE_TARGETS = frozenset({"left", "right", "surface", "context"})

# This program is constant; policy text is passed exclusively as JSON on stdin.
# -I prevents user site/PYTHONPATH imports, and resource limits apply before the
# regex extension is imported.  A failed import/compile/limit is never accepted.
_WORKER_PROGRAM = r'''
import json
import resource
import sys

def main():
    resource.setrlimit(resource.RLIMIT_CPU, (1, 2))
    resource.setrlimit(resource.RLIMIT_AS, (268435456, 268435456))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))
    resource.setrlimit(resource.RLIMIT_NOFILE, (16, 16))
    payload = sys.stdin.buffer.read(32769)
    if len(payload) > 32768:
        raise ValueError("worker input exceeds limit")
    request = json.loads(payload)
    if not isinstance(request, dict) or set(request) != {"operation", "patterns"}:
        raise ValueError("worker request fields differ")
    if request["operation"] not in {"runtime", "compile"}:
        raise ValueError("worker operation is invalid")
    patterns = request["patterns"]
    if not isinstance(patterns, list) or len(patterns) > 8:
        raise ValueError("worker patterns are invalid")
    if any(not isinstance(p, str) or not p or len(p) > 256 for p in patterns):
        raise ValueError("worker pattern length is invalid")
    import regex
    if regex.__version__ != "2026.4.4":
        raise ValueError("repair runtime requires regex==2026.4.4")
    for index, pattern in enumerate(patterns):
        try:
            regex.compile(pattern)
        except Exception:
            raise ValueError("regex compilation failed for rule index " + str(index))
    print(json.dumps({"ok": True, "regex_version": regex.__version__, "compiled_count": len(patterns)}))

try:
    main()
except BaseException as exc:
    message = str(exc) if isinstance(exc, ValueError) else "isolated regex validation failed"
    print(json.dumps({"ok": False, "error": message}))
    sys.exit(2)
'''


def _run_worker(patterns: list[str], *, runtime_only: bool = False) -> dict[str, Any]:
    try:
        payload = json.dumps(
            {"operation": "runtime" if runtime_only else "compile", "patterns": patterns},
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise LexiconRepairError("regex validation input is not valid UTF-8 JSON") from exc
    if len(payload) > MAX_WORKER_INPUT_BYTES:
        raise LexiconRepairError("regex validation input exceeds byte limit")
    try:
        completed = subprocess.run(
            [sys.executable, "-I", "-c", _WORKER_PROGRAM],
            input=payload,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=WORKER_WALL_TIMEOUT_SECONDS,
            check=False,
            close_fds=True,
            cwd="/",
        )
    except subprocess.TimeoutExpired as exc:
        raise LexiconRepairError("isolated regex validation exceeded wall-clock limit") from exc
    except (OSError, ValueError) as exc:
        raise LexiconRepairError("could not start isolated regex validation") from exc
    if len(completed.stdout) > 4096:
        raise LexiconRepairError("isolated regex validator returned excessive output")
    try:
        result = json.loads(completed.stdout)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise LexiconRepairError("isolated regex validator failed or was terminated") from exc
    if not isinstance(result, dict):
        raise LexiconRepairError("isolated regex validator returned an invalid result")
    if completed.returncode != 0 or result.get("ok") is not True:
        message = str(result.get("error", "isolated regex validation failed"))[:256]
        raise LexiconRepairError(message)
    if (
        set(result) != {"ok", "regex_version", "compiled_count"}
        or result["regex_version"] != REGEX_VERSION
        or type(result["compiled_count"]) is not int
        or result["compiled_count"] != len(patterns)
    ):
        raise LexiconRepairError("isolated regex validator runtime or result differs")
    return result


def check_regex_runtime() -> dict[str, Any]:
    """Verify the pinned dependency and the Linux resource sandbox at startup."""

    result = _run_worker([], runtime_only=True)
    return {
        "regex_version": result["regex_version"],
        "validation_isolation": "resource-limited-subprocess/v1",
        "wall_timeout_seconds": WORKER_WALL_TIMEOUT_SECONDS,
        "memory_limit_bytes": WORKER_MEMORY_BYTES,
    }


def validate_match_policy(policy: Any) -> dict[str, list[dict[str, str]]]:
    """Return canonical policy structure after bounded, isolated compilation.

    Empty groups need no engine invocation.  Rule IDs are unique across both
    groups; targets and limits match ``ControlledLexiconMatcher``.  String-only
    fields are enforced here to avoid silently coercing reviewer JSON.
    """

    if not isinstance(policy, dict) or set(policy) - {"require_any", "exclude_any"}:
        raise LexiconRepairError("match_policy fields differ")
    normalized: dict[str, list[dict[str, str]]] = {"require_any": [], "exclude_any": []}
    seen_ids: set[str] = set()
    patterns: list[str] = []
    for group in normalized:
        rows = policy.get(group, [])
        if rows is None:
            rows = []
        if not isinstance(rows, list):
            raise LexiconRepairError("match_policy groups must be lists")
        if len(patterns) + len(rows) > MAX_RULES_PER_ENTRY:
            raise LexiconRepairError(f"match_policy exceeds {MAX_RULES_PER_ENTRY} rules")
        for row in rows:
            if not isinstance(row, dict) or set(row) != {"rule_id", "target", "pattern"}:
                raise LexiconRepairError("match_policy rule fields differ")
            if any(not isinstance(row[key], str) for key in row):
                raise LexiconRepairError("match_policy rule values must be strings")
            try:
                for value in row.values():
                    value.encode("utf-8")
            except UnicodeError as exc:
                raise LexiconRepairError("match_policy rule contains invalid Unicode") from exc
            rule_id = row["rule_id"].strip()
            if not rule_id or len(rule_id) > MAX_RULE_ID_CHARS or rule_id in seen_ids:
                raise LexiconRepairError("match_policy rule_id is missing, too long or duplicated")
            if row["target"] not in _RULE_TARGETS:
                raise LexiconRepairError("match_policy rule target is invalid")
            pattern = row["pattern"]
            if not pattern or len(pattern) > MAX_PATTERN_CHARS:
                raise LexiconRepairError("match_policy pattern length is invalid")
            normalized[group].append(
                {"rule_id": rule_id, "target": row["target"], "pattern": pattern}
            )
            seen_ids.add(rule_id)
            patterns.append(pattern)
    if patterns:
        _run_worker(patterns)
    return normalized


__all__ = ["REGEX_VERSION", "check_regex_runtime", "validate_match_policy"]
