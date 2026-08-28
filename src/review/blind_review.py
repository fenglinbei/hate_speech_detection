"""Independent dual-model blind review for Stage 1 adjudication.

The module deliberately keeps credentials out of every returned object.  API
keys are resolved only when constructing an HTTP request and are never written
to request, response, provenance, or error artifacts.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import requests

from data.stage1_data import Stage1DataError, _validate_completed_rows
from data.training_artifacts import TrainingArtifactError, resolve_locator_ref
from review.d14_contract import (
    D14_POLICY_SCHEMA,
    normalize_provider_api_base,
    validate_d14_policy_document,
    validate_requested_returned_model,
    validate_reviewer_resolution,
)


POLICY_SCHEMA = D14_POLICY_SCHEMA
MODEL_REVIEW_SCHEMA = "stage1-model-review/v1"
CONSENSUS_SCHEMA = "stage1-review-consensus/v1"
RUN_SCHEMA = "stage1-dual-blind-review-run/v1"
QUEUE_SCHEMA = "stage1-human-review-queue/v1"
FINAL_ROW_SCHEMA = "stage1-data-adjudication-row/v1"

SENSITIVE_KEYS = {
    "api_key",
    "apikey",
    "authorization",
    "credential",
    "credentials",
    "headers",
    "password",
    "proxy_authorization",
    "request_headers",
}
SENSITIVE_ERROR_CODE = "raw-response-contained-credential"
_AUTHORIZATION_FRAGMENT_RE = re.compile(
    r"(?i)(?:[\"']?(?:proxy[-_ ]?)?authorization[\"']?)\s*[:=]\s*"
    r"[\"']?(?:bearer|basic|token)\s+([^\s\"',;}\]]+)"
)
_AUTHORIZATION_PLACEHOLDERS = {
    "authentication",
    "credential",
    "credentials",
    "example",
    "header",
    "omitted",
    "placeholder",
    "redacted",
    "scheme",
    "token",
    "value",
}


class BlindReviewError(RuntimeError):
    """Raised when a review transport or contract is invalid."""


def _normalized_key(value: Any) -> str:
    return unicodedata.normalize("NFKC", str(value)).strip().casefold().replace("-", "_")


def _text_contains_authorization_fragment(value: str) -> bool:
    """Recognize concrete header fragments, not ordinary prose about auth."""

    for match in _AUTHORIZATION_FRAGMENT_RE.finditer(value):
        credential = match.group(1).strip("<>[]()")
        if not credential:
            continue
        lowered = credential.casefold()
        if lowered in _AUTHORIZATION_PLACEHOLDERS:
            continue
        if len(credential) >= 8:
            return True
    return False


def _contains_sensitive_evidence(value: Any, secrets: Sequence[str]) -> bool:
    """Recursively inspect an in-memory value without rendering it publicly."""

    secret_values = tuple(secret for secret in secrets if isinstance(secret, str) and secret)
    seen: set[int] = set()

    def inspect(current: Any) -> bool:
        if isinstance(current, BaseException):
            return inspect(str(current))
        if isinstance(current, str):
            return any(secret in current for secret in secret_values) or _text_contains_authorization_fragment(current)
        if isinstance(current, bytes):
            rendered = current.decode("utf-8", errors="replace")
            return any(secret in rendered for secret in secret_values) or _text_contains_authorization_fragment(rendered)
        if isinstance(current, Mapping):
            marker = id(current)
            if marker in seen:
                return False
            seen.add(marker)
            for key, nested in current.items():
                if _normalized_key(key) in SENSITIVE_KEYS:
                    return True
                if inspect(str(key)) or inspect(nested):
                    return True
            return False
        if isinstance(current, (list, tuple, set, frozenset)):
            marker = id(current)
            if marker in seen:
                return False
            seen.add(marker)
            return any(inspect(nested) for nested in current)
        return False

    return inspect(value)


def _assert_safe_persisted_value(value: Any, secrets: Sequence[str]) -> None:
    if _contains_sensitive_evidence(value, secrets):
        raise BlindReviewError("blind-review model evidence contains credential material")
    try:
        canonical_json_bytes(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise BlindReviewError("blind-review model evidence is not canonical JSON") from exc


def _secret_values(policy: Mapping[str, Any], env_values: Mapping[str, str]) -> tuple[str, ...]:
    """Return only resolved API-key values; never names or other configuration."""

    values: list[str] = []
    for reviewer in policy["reviewers"]:
        env_name = reviewer.get("api_key_env")
        if not isinstance(env_name, str) or not env_name:
            continue
        value = os.environ.get(env_name) or env_values.get(env_name)
        if isinstance(value, str) and value:
            values.append(value)
    return tuple(sorted(set(values)))


def _safe_api_base(value: str) -> str:
    try:
        return normalize_provider_api_base(value)
    except ValueError as exc:
        raise BlindReviewError(str(exc)) from exc


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise BlindReviewError(f"blank JSONL line at {path}:{line_number}")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise BlindReviewError(f"JSONL row must be an object at {path}:{line_number}")
            rows.append(value)
    return rows


def _atomic_write(path: str | Path, payload: bytes) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _write_json(path: str | Path, value: Any) -> None:
    _atomic_write(path, canonical_json_bytes(value) + b"\n")


def _write_jsonl(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    ordered = sorted((dict(row) for row in rows), key=lambda row: str(row.get("issue_id", row.get("blind_item_id", ""))))
    payload = b"".join(canonical_json_bytes(row) + b"\n" for row in ordered)
    _atomic_write(path, payload)


def load_env_file(path: str | Path, *, allowed_names: set[str]) -> dict[str, str]:
    """Read only explicitly allowed simple KEY=VALUE entries.

    Values are returned in memory only.  Shell expansion, command substitution,
    and interpolation are intentionally unsupported.
    """

    values: dict[str, str] = {}
    source = Path(path)
    if not source.is_file():
        return values
    for line_number, raw_line in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        if "=" not in line:
            continue
        name, value = line.split("=", 1)
        name = name.strip()
        if name not in allowed_names:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        if not value:
            raise BlindReviewError(f"{name} is empty in {source}:{line_number}")
        values[name] = value
    return values


def load_review_policy(path: str | Path) -> dict[str, Any]:
    policy = _read_json(path)
    if not isinstance(policy, dict) or policy.get("schema_version") != POLICY_SCHEMA:
        raise BlindReviewError("unsupported blind-review policy")
    # The resolved policy is itself persisted.  It may name an environment
    # variable (``api_key_env``), but it must never carry a key/header value.
    _assert_safe_persisted_value(policy, ())
    try:
        return validate_d14_policy_document(policy)
    except ValueError as exc:
        raise BlindReviewError(str(exc)) from exc


def _resolve_reviewer(reviewer: Mapping[str, Any], env_values: Mapping[str, str]) -> dict[str, str]:
    def resolve(name_key: str, default_key: str | None = None) -> str:
        env_name = reviewer.get(name_key)
        value = None
        if isinstance(env_name, str) and env_name:
            value = os.environ.get(env_name) or env_values.get(env_name)
        if value is None and default_key is not None:
            default = reviewer.get(default_key)
            value = default if isinstance(default, str) else None
        if not value:
            raise BlindReviewError(f"reviewer {reviewer.get('reviewer_id')} cannot resolve {name_key}")
        return value

    return {
        "reviewer_id": str(reviewer["reviewer_id"]),
        "api_base": _safe_api_base(resolve("api_base_env", "default_api_base")),
        "api_key": resolve("api_key_env"),
        "model": resolve("model_env", "default_model"),
    }


def _resolve_live_d14_reviewers(
    policy: Mapping[str, Any], env_values: Mapping[str, str]
) -> list[dict[str, str]]:
    resolved = [_resolve_reviewer(reviewer, env_values) for reviewer in policy["reviewers"]]
    projection = [
        {
            "reviewer_id": reviewer["reviewer_id"],
            "api_base": reviewer["api_base"],
            "model": reviewer["model"],
        }
        for reviewer in resolved
    ]
    try:
        validate_reviewer_resolution(projection, require_live_d14=True)
    except ValueError as exc:
        raise BlindReviewError(str(exc)) from exc
    return resolved


def _chat_url(api_base: str) -> str:
    return api_base if api_base.endswith("/chat/completions") else api_base + "/chat/completions"


def _reason_codes(rubric_meta: Mapping[str, Any], issue: Mapping[str, Any]) -> dict[str, list[str]]:
    value = rubric_meta.get("reason_codes", {}).get(issue.get("issue_code"), {})
    return {
        "accepted": list(value.get("accepted", [])),
        "corrected": list(value.get("corrected", [])),
    }


def _blind_order(issues: Sequence[Mapping[str, Any]], seed: int) -> list[tuple[str, dict[str, Any]]]:
    ordered = sorted(
        (dict(issue) for issue in issues),
        key=lambda issue: hashlib.sha256(f"{seed}:{issue['issue_id']}".encode()).hexdigest(),
    )
    return [(f"B{index:04d}", issue) for index, issue in enumerate(ordered, 1)]


SYSTEM_PROMPT = """你是仇恨言论数据质量审查员。你必须独立审查，不得猜测其他审查员意见。
你看不到实验 condition、下游模型预测或分数。只依据给出的原句、当前四元组、issue 规则和允许编辑路径。
返回且只返回一个 JSON 对象，必须恰含 decision、edits、reason_code、reason、confidence 五个键。
decision 只能是 accepted、corrected 或 abstain。accepted 时 edits=[]；corrected 时每个 edit 必须恰含 location_index、op、json_pointer、value，op 只能是 set，且路径必须来自允许列表。
confidence 是 0 到 1 的数。无法可靠裁决时使用 abstain；不要从 targeted_group 自动推导 hateful。"""

SYSTEM_PROMPT += """
corrected 示例（键名必须完全一致）：
{"decision":"corrected","edits":[{"location_index":0,"op":"set","json_pointer":"/quadruples/0/hateful","value":"hate"}],"reason_code":"correct-source-label","reason":"结合原句语义修正标签。","confidence":0.95}
accepted 示例：
{"decision":"accepted","edits":[],"reason_code":"valid-independent-label-combination","reason":"两个字段可独立成立。","confidence":0.95}"""


def _item_prompt(blind_item_id: str, issue: Mapping[str, Any], rubric_meta: Mapping[str, Any]) -> str:
    payload = {
        "blind_item_id": blind_item_id,
        "issue_code": issue["issue_code"],
        "issue_subtype": issue["issue_subtype"],
        "accept_allowed": issue["accept_allowed"],
        "allowed_edit_paths": issue["allowed_edit_paths"],
        "allowed_reason_codes": _reason_codes(rubric_meta, issue),
        "review_context": issue["review_context"],
        "required_edit_shape": {
            "location_index": 0,
            "op": "set",
            "json_pointer": "one allowed path",
            "value": "corrected JSON value",
        },
    }
    return "请审查以下盲化 item，并输出 JSON：\n" + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def _extract_json_object(text: str) -> dict[str, Any]:
    candidate = text.strip()
    if candidate.startswith("```"):
        lines = candidate.splitlines()
        if len(lines) >= 3 and lines[-1].strip() == "```":
            candidate = "\n".join(lines[1:-1]).strip()
    value = json.loads(candidate)
    if not isinstance(value, dict):
        raise BlindReviewError("review response must be a JSON object")
    return value


def _normalize_judgement_shape(value: Mapping[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Normalize one documented provider key quirk without changing values."""

    normalized = dict(value)
    edits = normalized.get("edits")
    if not isinstance(edits, list):
        return normalized, []
    normalized_edits: list[Any] = []
    codes: list[str] = []
    for edit in edits:
        if isinstance(edit, dict) and set(edit) == {"location_index", "op", "_pointer", "value"}:
            edit = {
                "location_index": edit["location_index"],
                "op": edit["op"],
                "json_pointer": edit["_pointer"],
                "value": edit["value"],
            }
            codes.append("edit-key:_pointer-to-json_pointer/v1")
        normalized_edits.append(edit)
    normalized["edits"] = normalized_edits
    return normalized, codes


def _validate_judgement(value: Mapping[str, Any], issue: Mapping[str, Any], rubric_meta: Mapping[str, Any]) -> dict[str, Any]:
    required = {"decision", "edits", "reason_code", "reason", "confidence"}
    if set(value) != required:
        raise BlindReviewError(f"review response keys must equal {sorted(required)}")
    decision = value["decision"]
    if decision not in {"accepted", "corrected", "abstain"}:
        raise BlindReviewError("invalid review decision")
    confidence = value["confidence"]
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not 0 <= float(confidence) <= 1:
        raise BlindReviewError("confidence must be in [0,1]")
    if not isinstance(value["reason"], str) or not value["reason"].strip():
        raise BlindReviewError("reason must be non-empty")
    if not isinstance(value["reason_code"], str):
        raise BlindReviewError("reason_code must be a string")
    edits = value["edits"]
    if not isinstance(edits, list):
        raise BlindReviewError("edits must be an array")
    if decision == "abstain":
        if edits:
            raise BlindReviewError("abstain cannot contain edits")
    elif decision == "accepted":
        if not issue["accept_allowed"] or edits:
            raise BlindReviewError("accepted is not permitted for this issue")
        if value["reason_code"] not in _reason_codes(rubric_meta, issue)["accepted"]:
            raise BlindReviewError("invalid accepted reason_code")
    else:
        if not edits:
            raise BlindReviewError("corrected requires edits")
        if value["reason_code"] not in _reason_codes(rubric_meta, issue)["corrected"]:
            raise BlindReviewError("invalid corrected reason_code")
    canonical_edits: list[dict[str, Any]] = []
    for edit in edits:
        if not isinstance(edit, dict) or set(edit) != {"location_index", "op", "json_pointer", "value"}:
            raise BlindReviewError("malformed edit")
        if edit["location_index"] != 0 or edit["op"] != "set":
            raise BlindReviewError("only location_index=0 and op=set are allowed")
        if edit["json_pointer"] not in issue["allowed_edit_paths"]:
            raise BlindReviewError("unauthorized edit path")
        pointer = edit["json_pointer"]
        corrected_value = edit["value"]
        if issue["issue_kind"] == "field-type" and not (corrected_value is None or isinstance(corrected_value, str)):
            raise BlindReviewError("field-type correction must be string or null")
        if pointer.endswith("/hateful") and corrected_value not in {"hate", "non-hate"}:
            raise BlindReviewError("hateful correction must be hate or non-hate")
        canonical_edits.append(dict(edit))
    canonical_edits.sort(key=lambda edit: (edit["location_index"], edit["json_pointer"]))
    return {
        "decision": decision,
        "edits": canonical_edits,
        "reason_code": value["reason_code"],
        "reason": value["reason"].strip(),
        "confidence": float(confidence),
    }


Transport = Callable[[str, dict[str, Any], dict[str, str], int], dict[str, Any]]


def http_transport(url: str, request_payload: dict[str, Any], headers: dict[str, str], timeout: int) -> dict[str, Any]:
    response = requests.post(url, json=request_payload, headers=headers, timeout=timeout)
    if not response.ok:
        request_id = response.headers.get("x-request-id") or response.headers.get("request-id")
        raise BlindReviewError(f"review API HTTP {response.status_code}; request_id={request_id or 'unavailable'}")
    value = response.json()
    if not isinstance(value, dict):
        raise BlindReviewError("review API response must be an object")
    return value


def _error_code(exc: Exception, secrets: Sequence[str]) -> str:
    # Inspect transient exception text, but never return or persist that text.
    if _contains_sensitive_evidence(str(exc), secrets):
        return SENSITIVE_ERROR_CODE
    if isinstance(exc, BlindReviewError):
        message = str(exc)
        if "credential" in message:
            return SENSITIVE_ERROR_CODE
        if "finish_reason" in message:
            return "finish-reason-not-stop"
        if "empty" in message:
            return "empty-response-content"
        if "JSON object" in message or "choices" in message or "canonical JSON" in message:
            return "provider-response-invalid"
        return "judgement-schema-invalid"
    if isinstance(exc, json.JSONDecodeError):
        return "response-json-invalid"
    if isinstance(exc, requests.RequestException):
        return "transport-failure"
    return "transport-failure"


def _safe_parent_error(value: Any) -> dict[str, str]:
    """Keep retry lineage useful without copying legacy exception text."""

    if isinstance(value, Mapping):
        code = value.get("code")
        if isinstance(code, str) and re.fullmatch(r"[a-z0-9][a-z0-9-]{0,63}", code):
            return {"code": code}
    return {"code": "parent-review-failure"}


def _call_reviewer(
    resolved: Mapping[str, str],
    *,
    policy: Mapping[str, Any],
    blind_item_id: str,
    issue: Mapping[str, Any],
    rubric_meta: Mapping[str, Any],
    secrets: Sequence[str],
    transport: Transport,
) -> dict[str, Any]:
    prompt = _item_prompt(blind_item_id, issue, rubric_meta)
    request_payload: dict[str, Any] = {
        "model": resolved["model"],
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "temperature": policy["temperature"],
        "top_p": policy.get("top_p", 1),
        "stream": False,
        "max_tokens": int(policy.get("max_tokens", 4096)),
        "response_format": {"type": "json_object"},
    }
    reviewer_options = policy.get("reviewer_request_options", {}).get(
        resolved["reviewer_id"], {}
    )
    if reviewer_options.get("thinking") is not None:
        request_payload["thinking"] = reviewer_options["thinking"]
    if reviewer_options.get("reasoning_effort") is not None:
        request_payload["reasoning_effort"] = reviewer_options["reasoning_effort"]
    request_record = {
        "url": _chat_url(resolved["api_base"]),
        "payload": request_payload,
        "prompt_sha256": hashlib.sha256((SYSTEM_PROMPT + "\n" + prompt).encode("utf-8")).hexdigest(),
    }
    # This also catches an accidental credential in model/base overrides or in
    # the audit payload before either a network request or an artifact write.
    _assert_safe_persisted_value(request_record, secrets)
    raw: dict[str, Any] | None = None
    try:
        raw_value = transport(
            request_record["url"],
            request_payload,
            {"Authorization": f"Bearer {resolved['api_key']}", "Content-Type": "application/json"},
            int(policy.get("request_timeout_seconds", 120)),
        )
        if _contains_sensitive_evidence(raw_value, secrets):
            raise BlindReviewError("provider raw response contained a credential")
        if not isinstance(raw_value, Mapping):
            raise BlindReviewError("review API response must be a JSON object")
        try:
            canonical_json_bytes(raw_value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise BlindReviewError("review API response is not canonical JSON") from exc
        raw = dict(raw_value)
        try:
            validate_requested_returned_model(resolved["model"], raw.get("model"))
        except ValueError as exc:
            raise BlindReviewError(str(exc)) from exc
        choices = raw.get("choices")
        if not isinstance(choices, list) or not choices:
            raise BlindReviewError("review API response has no choices")
        choice = choices[0]
        finish_reason = choice.get("finish_reason") if isinstance(choice, dict) else None
        if finish_reason != "stop":
            raise BlindReviewError(f"review finish_reason={finish_reason!r}")
        message = choice.get("message") if isinstance(choice, dict) else None
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str) or not content.strip():
            raise BlindReviewError("review response content is empty")
        parsed, normalizations = _normalize_judgement_shape(_extract_json_object(content))
        judgement = _validate_judgement(parsed, issue, rubric_meta)
        error = None
    except Exception as exc:
        judgement = None
        normalizations = []
        code = _error_code(exc, secrets)
        if code == SENSITIVE_ERROR_CODE:
            raw = None
        error = {"code": code}
    row = {
        "schema_version": MODEL_REVIEW_SCHEMA,
        "issue_id": issue["issue_id"],
        "blind_item_id": blind_item_id,
        "reviewer_id": resolved["reviewer_id"],
        "provider_api_base": resolved["api_base"],
        "requested_model": resolved["model"],
        "returned_model": raw.get("model") if isinstance(raw, dict) else None,
        "provider_request_id": raw.get("id") if isinstance(raw, dict) else None,
        "prompt_sha256": request_record["prompt_sha256"],
        "request_without_secrets": request_record,
        "raw_response": raw,
        "judgement": judgement,
        "normalizations": normalizations,
        "error": error,
    }
    _assert_safe_persisted_value(row, secrets)
    return row


def _decision_projection(review: Mapping[str, Any]) -> Any:
    judgement = review.get("judgement")
    if not isinstance(judgement, dict) or judgement.get("decision") == "abstain":
        return None
    return {
        "decision": judgement["decision"],
        "edits": judgement["edits"],
        "reason_code": judgement["reason_code"],
    }


def _qc_selected(issue_id: str, fraction: float) -> bool:
    if fraction <= 0:
        return False
    threshold = int(fraction * (1 << 256))
    value = int(hashlib.sha256(f"agreement-qc-v1:{issue_id}".encode()).hexdigest(), 16)
    return value < threshold


def build_consensus(
    issues: Sequence[Mapping[str, Any]],
    reviews_by_reviewer: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    low_confidence_threshold: float,
    agreement_qc_fraction: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Return consensus rows, human queue packets, and auto-adjudication rows."""

    reviewer_ids = sorted(reviews_by_reviewer)
    if len(reviewer_ids) != 2:
        raise BlindReviewError("consensus requires exactly two reviewer streams")
    maps = {
        reviewer_id: {row["issue_id"]: dict(row) for row in rows}
        for reviewer_id, rows in reviews_by_reviewer.items()
    }
    consensus_rows: list[dict[str, Any]] = []
    queue: list[dict[str, Any]] = []
    auto_rows: list[dict[str, Any]] = []
    reviewed_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    for issue in sorted(issues, key=lambda row: row["issue_id"]):
        issue_id = issue["issue_id"]
        first, second = (maps[reviewer_ids[0]].get(issue_id), maps[reviewer_ids[1]].get(issue_id))
        if first is None or second is None:
            raise BlindReviewError(f"missing review for {issue_id}")
        reasons: list[str] = []
        if first.get("error") or second.get("error"):
            reasons.append("parse_failure")
        projections = (_decision_projection(first), _decision_projection(second))
        if None in projections:
            if "parse_failure" not in reasons:
                reasons.append("low_confidence")
        elif projections[0] != projections[1]:
            reasons.append("model_disagreement")
        confidences = [
            row.get("judgement", {}).get("confidence")
            for row in (first, second)
            if isinstance(row.get("judgement"), dict)
        ]
        if confidences and min(confidences) < low_confidence_threshold and "low_confidence" not in reasons:
            reasons.append("low_confidence")
        if not reasons and _qc_selected(issue_id, agreement_qc_fraction):
            reasons.append("agreement_qc_sample")
        agreed = projections[0] is not None and projections[0] == projections[1]
        consensus_rows.append(
            {
                "schema_version": CONSENSUS_SCHEMA,
                "issue_id": issue_id,
                "agreed": agreed,
                "decision_projection": projections[0] if agreed else None,
                "human_queue_reasons": reasons,
            }
        )
        if reasons:
            queue.append(
                {
                    "schema_version": QUEUE_SCHEMA,
                    "issue_id": issue_id,
                    "issue_code": issue["issue_code"],
                    "issue_kind": issue["issue_kind"],
                    "accept_allowed": issue["accept_allowed"],
                    "allowed_edit_paths": issue["allowed_edit_paths"],
                    "review_context": issue["review_context"],
                    "queue_reasons": reasons,
                    "model_votes_hidden": True,
                }
            )
            continue
        judgement_a = first["judgement"]
        judgement_b = second["judgement"]
        auto_rows.append(
            {
                "schema_version": FINAL_ROW_SCHEMA,
                "data_audit_id": issue["data_audit_id"],
                "issue_id": issue_id,
                "issue_kind": issue["issue_kind"],
                "decision": projections[0]["decision"],
                "edits": projections[0]["edits"],
                "reason_code": projections[0]["reason_code"],
                "reason": f"Two independent blind reviewers agreed. A: {judgement_a['reason']} B: {judgement_b['reason']}",
                "reviewer_id": "dual-blind-panel-v1",
                "reviewed_at": reviewed_at,
            }
        )
    return consensus_rows, queue, auto_rows


def _resolve_audit_target(audit_ref: str | Path) -> tuple[dict[str, Any], Path]:
    try:
        return resolve_locator_ref(audit_ref, expected_kind="data-audit")
    except TrainingArtifactError as exc:
        raise BlindReviewError(f"data-audit target cannot be resolved: {exc}") from exc


def run_data_review(
    *,
    audit_ref: str | Path,
    policy_path: str | Path,
    env_file: str | Path,
    output_dir: str | Path,
    issue_kinds: Sequence[str] | None = None,
    transport: Transport = http_transport,
) -> dict[str, Any]:
    policy = load_review_policy(policy_path)
    locator, audit_target = _resolve_audit_target(audit_ref)
    issues = _read_jsonl(audit_target / "issues.jsonl")
    if issue_kinds is not None:
        requested_kinds = tuple(dict.fromkeys(str(value) for value in issue_kinds))
        if not requested_kinds or any(not value for value in requested_kinds):
            raise BlindReviewError("issue_kinds must contain non-empty values")
        known_kinds = {str(issue.get("issue_kind")) for issue in issues}
        unknown = sorted(set(requested_kinds) - known_kinds)
        if unknown:
            raise BlindReviewError(f"unknown issue kinds: {unknown}")
        issues = [issue for issue in issues if issue.get("issue_kind") in requested_kinds]
    else:
        requested_kinds = tuple(sorted({str(issue.get("issue_kind")) for issue in issues}))
    if not issues:
        raise BlindReviewError("selected data review scope is empty")
    rubric_meta = _read_json(audit_target / "adjudication_rubric.meta.json")
    key_names = {str(reviewer["api_key_env"]) for reviewer in policy["reviewers"]}
    key_names |= {str(reviewer.get("api_base_env")) for reviewer in policy["reviewers"] if reviewer.get("api_base_env")}
    key_names |= {str(reviewer.get("model_env")) for reviewer in policy["reviewers"] if reviewer.get("model_env")}
    env_values = load_env_file(env_file, allowed_names=key_names)
    secrets = _secret_values(policy, env_values)
    _assert_safe_persisted_value(policy, secrets)
    resolved_reviewers = _resolve_live_d14_reviewers(policy, env_values)
    blind_items = _blind_order(issues, int(policy["blindness"]["stable_shuffle_seed"]))
    reviews_by_reviewer: dict[str, list[dict[str, Any]]] = {}
    for resolved in resolved_reviewers:
        def review_one(item: tuple[str, dict[str, Any]]) -> dict[str, Any]:
            blind_item_id, issue = item
            return _call_reviewer(
                    resolved,
                    policy=policy,
                    blind_item_id=blind_item_id,
                    issue=issue,
                    rubric_meta=rubric_meta,
                    secrets=secrets,
                    transport=transport,
                )
        concurrency = int(policy.get("max_concurrency_per_reviewer", 1))
        if concurrency < 1 or concurrency > 8:
            raise BlindReviewError("max_concurrency_per_reviewer must be in [1,8]")
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            stream = list(executor.map(review_one, blind_items))
        reviews_by_reviewer[resolved["reviewer_id"]] = stream
    consensus, queue, auto_rows = build_consensus(
        issues,
        reviews_by_reviewer,
        low_confidence_threshold=float(policy["consensus"]["low_confidence_threshold"]),
        agreement_qc_fraction=float(policy["consensus"]["agreement_qc_fraction"]),
    )
    model_projection = [
        {
            "reviewer_id": reviewer["reviewer_id"],
            "api_base": reviewer["api_base"],
            "model": reviewer["model"],
        }
        for reviewer in resolved_reviewers
    ]
    run_inputs = {
        "schema_version": RUN_SCHEMA,
        "audit_artifact_id": locator["artifact_id"],
        "audit_payload_manifest_sha256": locator["payload_manifest_sha256"],
        "policy_sha256": hashlib.sha256(Path(policy_path).read_bytes()).hexdigest(),
        "system_prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest(),
        "reviewers": model_projection,
        "issue_scope": {
            "issue_kinds": list(requested_kinds),
            "issue_count": len(issues),
            "scope_policy": "explicit-audit-subset/v1" if issue_kinds is not None else "complete-audit/v1",
        },
        "ordered_issue_ids_sha256": canonical_sha256([issue["issue_id"] for _, issue in blind_items]),
    }
    run_id = "dblind-" + canonical_sha256(run_inputs)
    destination = Path(output_dir) / run_id
    if destination.exists():
        raise BlindReviewError(f"review target already exists: {destination}")
    human_templates = [
        {
            "schema_version": FINAL_ROW_SCHEMA,
            "data_audit_id": next(issue["data_audit_id"] for issue in issues if issue["issue_id"] == packet["issue_id"]),
            "issue_id": packet["issue_id"],
            "issue_kind": packet["issue_kind"],
            "decision": "",
            "edits": [],
            "reason_code": "",
            "reason": "",
            "reviewer_id": "dual-blind-panel-v1",
            "reviewed_at": "",
        }
        for packet in queue
    ]
    summary = {
        "schema_version": "stage1-dual-blind-review-summary/v1",
        "review_run_id": run_id,
        "issue_count": len(issues),
        "auto_adjudication_count": len(auto_rows),
        "human_queue_count": len(queue),
        "human_queue_reasons": {
            reason: sum(reason in packet["queue_reasons"] for packet in queue)
            for reason in policy["consensus"]["human_queue_reasons"]
        },
        "target_path": str(destination.resolve()),
    }
    persisted_documents = {
        "run_meta": {**run_inputs, "review_run_id": run_id},
        "policy": policy,
        "model_resolution": {"reviewers": model_projection},
        "model_reviews": reviews_by_reviewer,
        "consensus": consensus,
        "human_queue": queue,
        "auto_adjudication": auto_rows,
        "human_adjudication_template": human_templates,
        "summary": summary,
    }
    _assert_safe_persisted_value(persisted_documents, secrets)

    destination.mkdir(parents=True)
    _write_json(destination / "run.meta.json", persisted_documents["run_meta"])
    _write_json(destination / "policy.resolved.json", policy)
    _write_json(destination / "model_resolution.json", persisted_documents["model_resolution"])
    for reviewer_id, rows in reviews_by_reviewer.items():
        _write_jsonl(destination / "model_reviews" / f"{reviewer_id}.jsonl", rows)
    _write_jsonl(destination / "consensus.jsonl", consensus)
    _write_jsonl(destination / "human_queue.jsonl", queue)
    _write_jsonl(destination / "auto_adjudication.jsonl", auto_rows)
    _write_jsonl(destination / "human_adjudication_template.jsonl", human_templates)
    _write_json(destination / "summary.json", summary)
    return summary


def preflight_data_review(
    *,
    audit_ref: str | Path,
    policy_path: str | Path,
    env_file: str | Path,
    issue_kinds: Sequence[str] | None = None,
    reviewer_ids: Sequence[str] | None = None,
    transport: Transport = http_transport,
) -> dict[str, Any]:
    """Send one blind item to each reviewer and return a secret-free status."""

    policy = load_review_policy(policy_path)
    _, audit_target = _resolve_audit_target(audit_ref)
    issues = _read_jsonl(audit_target / "issues.jsonl")
    if issue_kinds is not None:
        requested = {str(value) for value in issue_kinds}
        issues = [issue for issue in issues if str(issue.get("issue_kind")) in requested]
    rubric_meta = _read_json(audit_target / "adjudication_rubric.meta.json")
    if not issues:
        raise BlindReviewError("data audit contains no review items")
    key_names = {str(reviewer["api_key_env"]) for reviewer in policy["reviewers"]}
    key_names |= {
        str(reviewer.get("api_base_env"))
        for reviewer in policy["reviewers"]
        if reviewer.get("api_base_env")
    }
    key_names |= {
        str(reviewer.get("model_env"))
        for reviewer in policy["reviewers"]
        if reviewer.get("model_env")
    }
    env_values = load_env_file(env_file, allowed_names=key_names)
    secrets = _secret_values(policy, env_values)
    _assert_safe_persisted_value(policy, secrets)
    blind_item_id, issue = _blind_order(
        issues, int(policy["blindness"]["stable_shuffle_seed"])
    )[0]
    results = []
    reviewers = _resolve_live_d14_reviewers(policy, env_values)
    if reviewer_ids is not None:
        selected_ids = {str(value) for value in reviewer_ids}
        reviewers = [reviewer for reviewer in reviewers if reviewer["reviewer_id"] in selected_ids]
        if not reviewers or {reviewer["reviewer_id"] for reviewer in reviewers} != selected_ids:
            raise BlindReviewError("preflight reviewer scope contains an unknown reviewer")
    for resolved in reviewers:
        row = _call_reviewer(
            resolved,
            policy=policy,
            blind_item_id=blind_item_id,
            issue=issue,
            rubric_meta=rubric_meta,
            secrets=secrets,
            transport=transport,
        )
        response_shape = None
        if isinstance(row.get("raw_response"), dict):
            try:
                content = row["raw_response"]["choices"][0]["message"]["content"]
                parsed = _extract_json_object(content)
                response_shape = {
                    "top_level_keys": sorted(parsed),
                    "edit_key_sets": [sorted(edit) for edit in parsed.get("edits", []) if isinstance(edit, dict)],
                }
            except Exception:
                response_shape = {"unavailable": True}
        results.append(
            {
                "reviewer_id": resolved["reviewer_id"],
                "api_base": resolved["api_base"],
                "requested_model": resolved["model"],
                "returned_model": row["returned_model"],
                "provider_request_id": row["provider_request_id"],
                "response_shape": response_shape,
                "ok": row["error"] is None,
                "error": row["error"],
            }
        )
    report = {
        "schema_version": "stage1-dual-blind-review-preflight/v1",
        "blind_item_id": blind_item_id,
        "issue_kind": issue["issue_kind"],
        "all_ok": bool(results) and all(result["ok"] for result in results),
        "reviewers": results,
    }
    _assert_safe_persisted_value(report, secrets)
    return report


def retry_failed_data_review(
    *,
    audit_ref: str | Path,
    policy_path: str | Path,
    env_file: str | Path,
    parent_review_target: str | Path,
    output_dir: str | Path,
    transport: Transport = http_transport,
) -> dict[str, Any]:
    """Retry only transport/parse failures while reusing successful blind votes.

    A retry remains a single request per failed reviewer/item under the supplied
    policy.  The failed parent target is immutable and is recorded in the new
    run lineage; successful votes are copied byte-for-byte into the consolidated
    streams rather than requested again.
    """

    policy = load_review_policy(policy_path)
    locator, audit_target = _resolve_audit_target(audit_ref)
    all_issues = _read_jsonl(audit_target / "issues.jsonl")
    issues_by_id = {str(issue["issue_id"]): issue for issue in all_issues}
    rubric_meta = _read_json(audit_target / "adjudication_rubric.meta.json")
    parent = Path(parent_review_target)
    parent_meta = _read_json(parent / "run.meta.json")
    parent_run_id = parent_meta.get("review_run_id")
    if not isinstance(parent_run_id, str) or parent.name != parent_run_id:
        raise BlindReviewError("parent review target identity mismatch")
    if parent_meta.get("audit_artifact_id") != locator.get("artifact_id"):
        raise BlindReviewError("parent review target uses a different audit")

    key_names = {str(reviewer["api_key_env"]) for reviewer in policy["reviewers"]}
    key_names |= {
        str(reviewer.get("api_base_env"))
        for reviewer in policy["reviewers"]
        if reviewer.get("api_base_env")
    }
    key_names |= {
        str(reviewer.get("model_env"))
        for reviewer in policy["reviewers"]
        if reviewer.get("model_env")
    }
    env_values = load_env_file(env_file, allowed_names=key_names)
    secrets = _secret_values(policy, env_values)
    _assert_safe_persisted_value(policy, secrets)
    resolved_reviewers = _resolve_live_d14_reviewers(policy, env_values)
    reviews_by_reviewer: dict[str, list[dict[str, Any]]] = {}
    ordered_issue_ids: list[str] | None = None
    retry_counts: dict[str, int] = {}
    retry_issue_ids: dict[str, list[str]] = {}
    for resolved in resolved_reviewers:
        reviewer_id = resolved["reviewer_id"]
        source_path = parent / "model_reviews" / f"{reviewer_id}.jsonl"
        prior_rows = _read_jsonl(source_path)
        # Legacy v1 rows remain readable, but unsafe evidence is never copied
        # into a new retry artifact.
        _assert_safe_persisted_value(prior_rows, secrets)
        prior_map = {str(row["issue_id"]): dict(row) for row in prior_rows}
        if len(prior_map) != len(prior_rows):
            raise BlindReviewError(f"parent stream {reviewer_id} has duplicate issue IDs")
        current_ids = sorted(prior_map)
        if ordered_issue_ids is None:
            ordered_issue_ids = current_ids
        elif current_ids != ordered_issue_ids:
            raise BlindReviewError("parent reviewer streams have different issue sets")
        failed_ids = sorted(issue_id for issue_id, row in prior_map.items() if row.get("error") is not None)
        retry_counts[reviewer_id] = len(failed_ids)
        retry_issue_ids[reviewer_id] = failed_ids

        def retry_one(issue_id: str) -> tuple[str, dict[str, Any]]:
            if issue_id not in issues_by_id:
                raise BlindReviewError(f"parent review references unknown issue {issue_id}")
            prior = prior_map[issue_id]
            blind_item_id = prior.get("blind_item_id")
            if not isinstance(blind_item_id, str):
                raise BlindReviewError("parent review lacks blind_item_id")
            retried = _call_reviewer(
                resolved,
                policy=policy,
                blind_item_id=blind_item_id,
                issue=issues_by_id[issue_id],
                rubric_meta=rubric_meta,
                secrets=secrets,
                transport=transport,
            )
            retried["retry_lineage"] = {
                "parent_review_run_id": parent_run_id,
                "previous_error": _safe_parent_error(prior.get("error")),
                "attempt": 2,
            }
            _assert_safe_persisted_value(retried, secrets)
            return issue_id, retried

        concurrency = int(policy.get("max_concurrency_per_reviewer", 1))
        if concurrency < 1 or concurrency > 8:
            raise BlindReviewError("max_concurrency_per_reviewer must be in [1,8]")
        if failed_ids:
            with ThreadPoolExecutor(max_workers=concurrency) as executor:
                replacements = dict(executor.map(retry_one, failed_ids))
            prior_map.update(replacements)
        reviews_by_reviewer[reviewer_id] = [prior_map[issue_id] for issue_id in current_ids]

    if ordered_issue_ids is None or not ordered_issue_ids:
        raise BlindReviewError("parent review has an empty issue frame")
    issues = [issues_by_id[issue_id] for issue_id in ordered_issue_ids]
    consensus, queue, auto_rows = build_consensus(
        issues,
        reviews_by_reviewer,
        low_confidence_threshold=float(policy["consensus"]["low_confidence_threshold"]),
        agreement_qc_fraction=float(policy["consensus"]["agreement_qc_fraction"]),
    )
    model_projection = [
        {
            "reviewer_id": reviewer["reviewer_id"],
            "api_base": reviewer["api_base"],
            "model": reviewer["model"],
        }
        for reviewer in resolved_reviewers
    ]
    run_inputs = {
        "schema_version": RUN_SCHEMA,
        "audit_artifact_id": locator["artifact_id"],
        "audit_payload_manifest_sha256": locator["payload_manifest_sha256"],
        "parent_review_run_id": parent_run_id,
        "retry_policy": "failed-reviewer-items-only/v1",
        "retry_issue_ids_sha256_by_reviewer": {
            reviewer_id: canonical_sha256(issue_ids)
            for reviewer_id, issue_ids in sorted(retry_issue_ids.items())
        },
        "policy_sha256": hashlib.sha256(Path(policy_path).read_bytes()).hexdigest(),
        "system_prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest(),
        "reviewers": model_projection,
        "ordered_issue_ids_sha256": canonical_sha256(ordered_issue_ids),
    }
    run_id = "dblind-" + canonical_sha256(run_inputs)
    destination = Path(output_dir) / run_id
    if destination.exists():
        raise BlindReviewError(f"review target already exists: {destination}")
    human_templates = [
        {
            "schema_version": FINAL_ROW_SCHEMA,
            "data_audit_id": issues_by_id[packet["issue_id"]]["data_audit_id"],
            "issue_id": packet["issue_id"],
            "issue_kind": packet["issue_kind"],
            "decision": "",
            "edits": [],
            "reason_code": "",
            "reason": "",
            "reviewer_id": "dual-blind-panel-v1",
            "reviewed_at": "",
        }
        for packet in queue
    ]
    summary = {
        "schema_version": "stage1-dual-blind-review-summary/v1",
        "review_run_id": run_id,
        "parent_review_run_id": parent_run_id,
        "issue_count": len(issues),
        "retry_counts_by_reviewer": retry_counts,
        "remaining_errors_by_reviewer": {
            reviewer_id: sum(row.get("error") is not None for row in rows)
            for reviewer_id, rows in reviews_by_reviewer.items()
        },
        "auto_adjudication_count": len(auto_rows),
        "human_queue_count": len(queue),
        "human_queue_reasons": {
            reason: sum(reason in packet["queue_reasons"] for packet in queue)
            for reason in policy["consensus"]["human_queue_reasons"]
        },
        "target_path": str(destination.resolve()),
    }
    persisted_documents = {
        "run_meta": {**run_inputs, "review_run_id": run_id},
        "policy": policy,
        "model_resolution": {"reviewers": model_projection},
        "model_reviews": reviews_by_reviewer,
        "consensus": consensus,
        "human_queue": queue,
        "auto_adjudication": auto_rows,
        "human_adjudication_template": human_templates,
        "summary": summary,
    }
    _assert_safe_persisted_value(persisted_documents, secrets)

    destination.mkdir(parents=True)
    _write_json(destination / "run.meta.json", persisted_documents["run_meta"])
    _write_json(destination / "policy.resolved.json", policy)
    _write_json(destination / "model_resolution.json", persisted_documents["model_resolution"])
    for reviewer_id, rows in reviews_by_reviewer.items():
        _write_jsonl(destination / "model_reviews" / f"{reviewer_id}.jsonl", rows)
    _write_jsonl(destination / "consensus.jsonl", consensus)
    _write_jsonl(destination / "human_queue.jsonl", queue)
    _write_jsonl(destination / "auto_adjudication.jsonl", auto_rows)
    _write_jsonl(destination / "human_adjudication_template.jsonl", human_templates)
    _write_json(destination / "summary.json", summary)
    return summary


def merge_human_adjudication(
    *,
    audit_ref: str | Path,
    review_target: str | Path | None = None,
    review_ref: str | Path | None = None,
    workspace_root: str | Path | None = None,
    human_completed: str | Path,
    output: str | Path,
) -> dict[str, Any]:
    locator, audit_target = _resolve_audit_target(audit_ref)
    if (review_target is None) == (review_ref is None):
        raise BlindReviewError(
            "provide exactly one sealed review_ref or legacy review_target"
        )
    expected_review_run_id: str
    if review_ref is not None:
        if workspace_root is None:
            raise BlindReviewError("sealed review_ref requires workspace_root")
        try:
            from review.data_review_artifact import resolve_data_review_ref

            _, target, review_report = resolve_data_review_ref(
                review_ref, workspace_root=workspace_root
            )
        except (OSError, RuntimeError, ValueError) as exc:
            raise BlindReviewError(str(exc)) from exc
        if review_report.get("data_audit_id") != locator.get("artifact_id"):
            raise BlindReviewError(
                "sealed review ref does not bind the supplied data audit"
            )
        expected_review_run_id = str(review_report["source_review_run_id"])
    else:
        target = Path(review_target)
        expected_review_run_id = target.name
    run_meta = _read_json(target / "run.meta.json")
    if (
        not isinstance(run_meta, dict)
        or run_meta.get("review_run_id") != expected_review_run_id
        or run_meta.get("audit_artifact_id") != locator["artifact_id"]
        or run_meta.get("audit_payload_manifest_sha256")
        != locator["payload_manifest_sha256"]
    ):
        raise BlindReviewError("review target does not bind the supplied data audit")
    audit_issues = _read_jsonl(audit_target / "issues.jsonl")
    rubric_meta = _read_json(audit_target / "adjudication_rubric.meta.json")
    issues_by_id = {str(issue["issue_id"]): issue for issue in audit_issues}
    if len(issues_by_id) != len(audit_issues):
        raise BlindReviewError("data audit contains duplicate issue IDs")
    auto_rows = _read_jsonl(target / "auto_adjudication.jsonl")
    templates = _read_jsonl(target / "human_adjudication_template.jsonl")
    queue_rows = _read_jsonl(target / "human_queue.jsonl")
    consensus_rows = _read_jsonl(target / "consensus.jsonl")
    human_rows = _read_jsonl(human_completed)
    expected = {row["issue_id"] for row in templates}
    queue_ids = {row.get("issue_id") for row in queue_rows}
    auto_ids = {row.get("issue_id") for row in auto_rows}
    consensus_ids = {row.get("issue_id") for row in consensus_rows}
    if (
        len(expected) != len(templates)
        or len(queue_ids) != len(queue_rows)
        or len(auto_ids) != len(auto_rows)
        or len(consensus_ids) != len(consensus_rows)
        or expected != queue_ids
        or expected.intersection(auto_ids)
        or expected.union(auto_ids) != consensus_ids
    ):
        raise BlindReviewError("review target human/auto/consensus frames are inconsistent")
    if not consensus_ids or not all(isinstance(issue_id, str) for issue_id in consensus_ids):
        raise BlindReviewError("review target consensus frame is empty or malformed")
    unknown_consensus = sorted(consensus_ids - set(issues_by_id))
    if unknown_consensus:
        raise BlindReviewError(
            f"review target contains unknown audit issues: {unknown_consensus}"
        )
    scope = run_meta.get("issue_scope")
    if isinstance(scope, dict):
        scope_kinds = scope.get("issue_kinds")
        if (
            not isinstance(scope_kinds, list)
            or not scope_kinds
            or len(scope_kinds) != len(set(scope_kinds))
            or any(not isinstance(kind, str) or not kind for kind in scope_kinds)
            or scope.get("issue_count") != len(consensus_ids)
            or scope.get("scope_policy")
            not in {"explicit-audit-subset/v1", "complete-audit/v1"}
        ):
            raise BlindReviewError("review target has malformed issue_scope")
    elif isinstance(run_meta.get("parent_review_run_id"), str):
        scope_kinds = sorted(
            {str(issues_by_id[str(issue_id)]["issue_kind"]) for issue_id in consensus_ids}
        )
    else:
        raise BlindReviewError("review target does not freeze its audit issue scope")
    expected_scope_ids = {
        issue_id
        for issue_id, issue in issues_by_id.items()
        if issue.get("issue_kind") in set(scope_kinds)
    }
    if consensus_ids != expected_scope_ids:
        raise BlindReviewError("review target omits or adds issues within its frozen scope")
    if isinstance(run_meta.get("parent_review_run_id"), str):
        expected_order_hash = canonical_sha256(sorted(consensus_ids))
    else:
        policy = load_review_policy(target / "policy.resolved.json")
        scoped_issues = [issues_by_id[issue_id] for issue_id in sorted(consensus_ids)]
        blind_order = _blind_order(
            scoped_issues, int(policy["blindness"]["stable_shuffle_seed"])
        )
        expected_order_hash = canonical_sha256(
            [issue["issue_id"] for _, issue in blind_order]
        )
    if run_meta.get("ordered_issue_ids_sha256") != expected_order_hash:
        raise BlindReviewError("review target ordered issue frame does not match the audit")
    actual = {row.get("issue_id") for row in human_rows}
    if actual != expected or len(actual) != len(human_rows):
        raise BlindReviewError("human adjudication issue set does not equal the frozen queue")
    merged = list(auto_rows)
    expected_keys = set(templates[0]) if templates else set()
    templates_by_id = {str(row["issue_id"]): row for row in templates}
    for row in human_rows:
        if set(row) != expected_keys:
            raise BlindReviewError("human adjudication row shape changed")
        normalized = dict(row)
        issue_id = str(normalized["issue_id"])
        template = templates_by_id[issue_id]
        for field in (
            "schema_version",
            "data_audit_id",
            "issue_id",
            "issue_kind",
            "reviewer_id",
        ):
            if normalized.get(field) != template.get(field):
                raise BlindReviewError(
                    f"human adjudication row {issue_id} changed frozen field {field}"
                )
        if normalized.get("reviewer_id") != "dual-blind-panel-v1":
            raise BlindReviewError(
                f"human adjudication row {issue_id} has the wrong reviewer identity"
            )
        decision = normalized.get("decision")
        edits = normalized.get("edits")
        if decision not in {"accepted", "corrected"}:
            raise BlindReviewError(f"human adjudication row {normalized.get('issue_id')} is incomplete")
        if not isinstance(edits, list) or (decision == "accepted" and edits) or (
            decision == "corrected" and not edits
        ):
            raise BlindReviewError(
                f"human adjudication row {issue_id} has edits inconsistent with its decision"
            )
        if not isinstance(normalized.get("reason_code"), str) or not normalized[
            "reason_code"
        ]:
            raise BlindReviewError(f"human adjudication row {issue_id} lacks reason_code")
        if not isinstance(normalized.get("reason"), str) or not normalized["reason"].strip():
            raise BlindReviewError(f"human adjudication row {issue_id} lacks reason")
        if not isinstance(normalized.get("reviewed_at"), str) or not normalized[
            "reviewed_at"
        ]:
            raise BlindReviewError(f"human adjudication row {issue_id} lacks reviewed_at")
        merged.append(normalized)
    if len({row["issue_id"] for row in merged}) != len(merged):
        raise BlindReviewError("merged adjudication contains duplicate issue IDs")
    merged_ids = {str(row["issue_id"]) for row in merged}
    unknown = sorted(merged_ids - set(issues_by_id))
    if unknown:
        raise BlindReviewError(f"review target contains unknown audit issues: {unknown}")
    selected_issues = [issues_by_id[issue_id] for issue_id in sorted(merged_ids)]
    try:
        validated = _validate_completed_rows(
            merged,
            selected_issues,
            rubric_meta,
            reviewer_id="dual-blind-panel-v1",
        )
    except Stage1DataError as exc:
        raise BlindReviewError(f"merged human adjudication is invalid: {exc}") from exc
    _write_jsonl(output, validated)
    return {"output": str(Path(output).resolve()), "row_count": len(validated)}


def export_human_adjudication_workspace(
    *,
    review_target: str | Path | None = None,
    review_ref: str | Path | None = None,
    workspace_root: str | Path | None = None,
    output: str | Path,
    packet_output: str | Path,
) -> dict[str, Any]:
    """Export the frozen human queue without exposing either model vote."""

    if (review_target is None) == (review_ref is None):
        raise BlindReviewError(
            "provide exactly one sealed review_ref or legacy review_target"
        )
    if review_ref is not None:
        if workspace_root is None:
            raise BlindReviewError("sealed review_ref requires workspace_root")
        try:
            from review.data_review_artifact import resolve_data_review_ref

            _, target, _ = resolve_data_review_ref(
                review_ref, workspace_root=workspace_root
            )
        except (OSError, RuntimeError, ValueError) as exc:
            raise BlindReviewError(str(exc)) from exc
    else:
        target = Path(review_target)
    templates = _read_jsonl(target / "human_adjudication_template.jsonl")
    packets = _read_jsonl(target / "human_queue.jsonl")
    auto_rows = _read_jsonl(target / "auto_adjudication.jsonl")
    consensus = _read_jsonl(target / "consensus.jsonl")
    template_ids = {row.get("issue_id") for row in templates}
    packet_ids = {row.get("issue_id") for row in packets}
    auto_ids = {row.get("issue_id") for row in auto_rows}
    consensus_ids = {row.get("issue_id") for row in consensus}
    if (
        len(template_ids) != len(templates)
        or len(packet_ids) != len(packets)
        or template_ids != packet_ids
        or template_ids.intersection(auto_ids)
        or template_ids.union(auto_ids) != consensus_ids
    ):
        raise BlindReviewError("review target human/auto/consensus issue frames are inconsistent")
    if any(packet.get("model_votes_hidden") is not True for packet in packets):
        raise BlindReviewError("human review packet does not hide model votes")
    _write_jsonl(output, templates)
    _write_jsonl(packet_output, packets)
    return {
        "output": str(Path(output).resolve()),
        "packet_output": str(Path(packet_output).resolve()),
        "row_count": len(templates),
        "auto_adjudication_count": len(auto_rows),
    }


__all__ = [
    "BlindReviewError",
    "SYSTEM_PROMPT",
    "build_consensus",
    "export_human_adjudication_workspace",
    "http_transport",
    "load_env_file",
    "load_review_policy",
    "merge_human_adjudication",
    "preflight_data_review",
    "retry_failed_data_review",
    "run_data_review",
]
