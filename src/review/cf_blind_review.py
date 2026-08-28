"""Immutable dual-model blind review over a frozen counterfactual proposal.

Only ``target`` and ``argument`` candidates marked ``review_required`` are sent
to the two frozen D14 reviewers.  Deterministic ``targeted_group``/``hateful``
foils are emitted directly as ``not_required``.  API credentials exist only in
the transport call and are rejected if a provider echoes them into a response.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from data.counterfactual_lifecycle import (
    CounterfactualLifecycleError,
    prepare_reviewer_declaration,
    validate_proposal_ref,
    validate_proposal_target,
)
from data.training_artifacts import (
    TrainingArtifactError,
    canonical_json_bytes,
    canonical_jsonl_bytes,
    canonical_sha256,
    ensure_exact_file_set,
    finalize_target_atomic,
    load_json,
    load_jsonl,
    new_staging_directory,
    portable_dependency,
    resolve_dependency_target,
    resolve_locator_ref,
    sha256_file,
    validate_dependency_ref,
    validate_json_schema,
    validate_payload_manifest,
    write_canonical_json,
    write_canonical_jsonl,
    write_locator_ref,
)
from review.blind_review import (
    BlindReviewError,
    Transport,
    _chat_url,
    _extract_json_object,
    _resolve_reviewer,
    http_transport,
    load_env_file,
    load_review_policy,
)
from review.d14_contract import (
    D14_LIVE_EXECUTION_MODE,
    D14_REVIEWER_IDS,
    D14_SYNTHETIC_EXECUTION_MODE,
    normalize_provider_api_base,
    validate_d14_policy_document,
    validate_requested_returned_model,
    validate_reviewer_resolution,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_ROOT = REPOSITORY_ROOT / "schemas"
ARTIFACT_KIND = "cf-blind-review"
RUN_SCHEMA = "stage1-cf-dual-blind-review-run/v1"
MODEL_REVIEW_SCHEMA = "stage1-cf-model-review/v1"
CONSENSUS_SCHEMA = "stage1-cf-review-consensus/v1"
QUEUE_SCHEMA = "stage1-cf-human-review-queue/v1"
SUMMARY_SCHEMA = "stage1-cf-dual-blind-review-summary/v1"
PROVENANCE_SCHEMA = "stage1-cf-dual-blind-review-provenance/v1"
FINAL_REVIEW_SCHEMA = "stage1-cf-review-row/v1"
PANEL_REVIEWER_ID = "dual-blind-panel-v1"
REVIEWER_IDS = D14_REVIEWER_IDS
QUEUE_REASONS = (
    "model_disagreement",
    "parse_failure",
    "low_confidence",
    "agreement_qc_sample",
)
ERROR_CODES = {
    "transport-failure",
    "provider-response-invalid",
    "finish-reason-not-stop",
    "empty-response-content",
    "response-json-invalid",
    "judgement-schema-invalid",
    "raw-response-contained-credential",
}
SENSITIVE_KEYS = {
    "api_key",
    "apikey",
    "authorization",
    "credential",
    "credentials",
    "password",
    "request_headers",
    "headers",
}
CF_BLIND_ID_RE = re.compile(r"^cfblind-[0-9a-f]{64}$")


SYSTEM_PROMPT = """你是 Stage 1 反事实候选的独立质量审查员。你看不到实验 condition、模型预测、log-probability、margin、数据 split 或另一位审查员的意见。
你只判断给定 candidate 能否作为同一 query、同一 gold tuple、同一字段的局部反事实。
返回且只返回一个 JSON 对象，必须恰含 decision、reason_code、note、confidence 四个键。
decision 只能是 pass、reject 或 abstain。pass 的 reason_code 只能是 valid-local-foil；reject 必须使用给定 reject reason；无法可靠判断时用 abstain。
confidence 必须是 0 到 1 的数。不要改写候选，不要推断其他字段，也不要输出额外说明。"""


def _secret_values(policy: Mapping[str, Any], env_values: Mapping[str, str]) -> tuple[str, ...]:
    values: list[str] = []
    for reviewer in policy["reviewers"]:
        env_name = reviewer.get("api_key_env")
        if not isinstance(env_name, str):
            continue
        value = os.environ.get(env_name) or env_values.get(env_name)
        if isinstance(value, str) and len(value) >= 8:
            values.append(value)
    return tuple(sorted(set(values)))


def _contains_secret(value: Any, secrets: Sequence[str]) -> bool:
    try:
        rendered = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError):
        return False
    return any(secret and secret in rendered for secret in secrets)


def _assert_no_sensitive_keys(value: Any) -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if str(key).lower() in SENSITIVE_KEYS:
                raise BlindReviewError("CF blind-review artifact contains a credential-like key")
            _assert_no_sensitive_keys(nested)
    elif isinstance(value, list):
        for nested in value:
            _assert_no_sensitive_keys(nested)


def _safe_api_base(value: str) -> str:
    try:
        return normalize_provider_api_base(value)
    except ValueError as exc:
        raise BlindReviewError(str(exc)) from exc


def _validate_policy_document(document: Mapping[str, Any]) -> dict[str, Any]:
    try:
        return validate_d14_policy_document(document)
    except ValueError as exc:
        raise BlindReviewError(str(exc)) from exc


def _load_policy(path: str | Path) -> dict[str, Any]:
    return _validate_policy_document(load_review_policy(path))


def _credential_env_names(policy: Mapping[str, Any]) -> set[str]:
    names: set[str] = set()
    for reviewer in policy["reviewers"]:
        for key in ("api_key_env", "api_base_env", "model_env"):
            value = reviewer.get(key)
            if isinstance(value, str) and value:
                names.add(value)
    return names


def _resolved_reviewers(
    policy: Mapping[str, Any],
    env_values: Mapping[str, str],
    *,
    require_live_d14: bool,
) -> list[dict[str, str]]:
    resolved = [_resolve_reviewer(row, env_values) for row in policy["reviewers"]]
    for row in resolved:
        row["api_base"] = _safe_api_base(row["api_base"])
    try:
        validate_reviewer_resolution(
            [
                {
                    "reviewer_id": row["reviewer_id"],
                    "api_base": row["api_base"],
                    "model": row["model"],
                }
                for row in resolved
            ],
            require_live_d14=require_live_d14,
        )
    except ValueError as exc:
        raise BlindReviewError(str(exc)) from exc
    return sorted(resolved, key=lambda row: row["reviewer_id"])


def _blind_order(
    candidates: Sequence[Mapping[str, Any]], seed: int
) -> list[tuple[str, dict[str, Any]]]:
    ordered = sorted(
        (dict(row) for row in candidates),
        key=lambda row: hashlib.sha256(
            f"cf-blind-order/v1:{seed}:{row['candidate_id']}".encode("utf-8")
        ).hexdigest(),
    )
    return [(f"CFB{index:05d}", row) for index, row in enumerate(ordered, 1)]


def _candidate_prompt(
    blind_item_id: str,
    candidate: Mapping[str, Any],
    rubric_meta: Mapping[str, Any],
    rubric_body: str,
) -> str:
    payload = {
        "blind_item_id": blind_item_id,
        "cf_proposal_id": candidate["cf_proposal_id"],
        "candidate_id": candidate["candidate_id"],
        "query_content": candidate["query_content"],
        "gold_quadruples": candidate["gold_quadruples"],
        "tuple_index": candidate["tuple_index"],
        "field": candidate["field"],
        "gold_value": candidate["gold_value"],
        "candidate_value": candidate["candidate_value"],
        "family": candidate["family"],
        "source": candidate["source"],
        "pass_reason_codes": rubric_meta["pass_reason_codes"],
        "reject_reason_codes": rubric_meta["reject_reason_codes"],
        "frozen_review_rubric": rubric_body,
    }
    return "请独立审查以下盲化反事实候选，并输出 JSON：\n" + json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _validate_judgement(
    value: Mapping[str, Any],
    *,
    rubric_meta: Mapping[str, Any],
) -> dict[str, Any]:
    if set(value) != {"decision", "reason_code", "note", "confidence"}:
        raise BlindReviewError("CF review response has non-canonical keys")
    decision = value.get("decision")
    if decision not in {"pass", "reject", "abstain"}:
        raise BlindReviewError("CF review decision is invalid")
    reason_code = value.get("reason_code")
    if not isinstance(reason_code, str):
        raise BlindReviewError("CF review reason_code must be a string")
    if decision == "pass" and reason_code not in rubric_meta["pass_reason_codes"]:
        raise BlindReviewError("CF pass reason_code is invalid")
    if decision == "reject" and reason_code not in rubric_meta["reject_reason_codes"]:
        raise BlindReviewError("CF reject reason_code is invalid")
    if decision == "abstain" and reason_code not in {
        "",
        *rubric_meta["reject_reason_codes"],
    }:
        raise BlindReviewError("CF abstain reason_code is invalid")
    note = value.get("note")
    if not isinstance(note, str):
        raise BlindReviewError("CF review note must be a string")
    confidence = value.get("confidence")
    if (
        isinstance(confidence, bool)
        or not isinstance(confidence, (int, float))
        or not 0 <= float(confidence) <= 1
    ):
        raise BlindReviewError("CF review confidence must be in [0,1]")
    return {
        "decision": decision,
        "reason_code": reason_code,
        "note": note.strip(),
        "confidence": float(confidence),
    }


def _error_code(exc: Exception) -> str:
    if isinstance(exc, json.JSONDecodeError):
        return "response-json-invalid"
    if isinstance(exc, BlindReviewError):
        text = str(exc)
        if "finish_reason" in text:
            return "finish-reason-not-stop"
        if "empty" in text:
            return "empty-response-content"
        if "credential" in text:
            return "raw-response-contained-credential"
        if "response" in text or "choices" in text:
            return "provider-response-invalid"
        return "judgement-schema-invalid"
    return "transport-failure"


def _call_reviewer(
    resolved: Mapping[str, str],
    *,
    policy: Mapping[str, Any],
    blind_item_id: str,
    candidate: Mapping[str, Any],
    rubric_meta: Mapping[str, Any],
    rubric_body: str,
    secrets: Sequence[str],
    transport: Transport,
) -> dict[str, Any]:
    prompt = _candidate_prompt(blind_item_id, candidate, rubric_meta, rubric_body)
    request_payload: dict[str, Any] = {
        "model": resolved["model"],
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0,
        "top_p": policy.get("top_p", 1),
        "stream": False,
        "max_tokens": int(policy.get("max_tokens", 4096)),
        "response_format": {"type": "json_object"},
    }
    reviewer_options = policy.get("reviewer_request_options", {}).get(
        resolved["reviewer_id"], {}
    )
    for key in ("thinking", "reasoning_effort"):
        if reviewer_options.get(key) is not None:
            request_payload[key] = reviewer_options[key]
    url = _chat_url(resolved["api_base"])
    prompt_hash = hashlib.sha256((SYSTEM_PROMPT + "\n" + prompt).encode("utf-8")).hexdigest()
    request_record = {
        "url": url,
        "payload": request_payload,
        "prompt_sha256": prompt_hash,
    }
    raw: dict[str, Any] | None = None
    judgement: dict[str, Any] | None = None
    error: dict[str, str] | None = None
    try:
        raw_value = transport(
            url,
            request_payload,
            {
                "Authorization": f"Bearer {resolved['api_key']}",
                "Content-Type": "application/json",
            },
            int(policy.get("request_timeout_seconds", 120)),
        )
        if not isinstance(raw_value, dict):
            raise BlindReviewError("provider response must be an object")
        if _contains_secret(raw_value, secrets):
            raise BlindReviewError("provider raw response contained a credential")
        _assert_no_sensitive_keys(raw_value)
        raw = raw_value
        try:
            validate_requested_returned_model(resolved["model"], raw.get("model"))
        except ValueError as exc:
            raise BlindReviewError(str(exc)) from exc
        choices = raw.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], Mapping):
            raise BlindReviewError("provider response has no choices")
        if choices[0].get("finish_reason") != "stop":
            raise BlindReviewError("provider finish_reason is not stop")
        message = choices[0].get("message")
        content = message.get("content") if isinstance(message, Mapping) else None
        if not isinstance(content, str) or not content.strip():
            raise BlindReviewError("provider response content is empty")
        parsed = _extract_json_object(content)
        judgement = _validate_judgement(parsed, rubric_meta=rubric_meta)
    except Exception as exc:
        if _error_code(exc) == "raw-response-contained-credential":
            raw = None
        error = {"code": _error_code(exc)}
        judgement = None
    row = {
        "schema_version": MODEL_REVIEW_SCHEMA,
        "cf_proposal_id": candidate["cf_proposal_id"],
        "candidate_id": candidate["candidate_id"],
        "blind_item_id": blind_item_id,
        "reviewer_id": resolved["reviewer_id"],
        "provider_api_base": resolved["api_base"],
        "requested_model": resolved["model"],
        "returned_model": raw.get("model") if isinstance(raw, dict) else None,
        "provider_request_id": raw.get("id") if isinstance(raw, dict) else None,
        "prompt_sha256": prompt_hash,
        "request_without_secrets": request_record,
        "raw_response": raw,
        "judgement": judgement,
        "error": error,
    }
    _assert_no_sensitive_keys(row)
    if _contains_secret(row, secrets):
        raise BlindReviewError("credential escaped CF reviewer transport boundary")
    return row


def _decision_projection(review: Mapping[str, Any]) -> dict[str, str] | None:
    judgement = review.get("judgement")
    if not isinstance(judgement, Mapping) or judgement.get("decision") == "abstain":
        return None
    decision = judgement.get("decision")
    reason_code = judgement.get("reason_code")
    if decision not in {"pass", "reject"} or not isinstance(reason_code, str):
        return None
    return {"decision": decision, "reason_code": reason_code}


def _qc_selected(candidate_id: str, fraction: float) -> bool:
    if not 0 <= fraction <= 1:
        raise BlindReviewError("agreement QC fraction must be in [0,1]")
    if fraction == 0:
        return False
    threshold = int(fraction * (1 << 256))
    value = int(
        hashlib.sha256(f"agreement-qc-v1:{candidate_id}".encode("utf-8")).hexdigest(),
        16,
    )
    return value < threshold


def _queue_packet(candidate: Mapping[str, Any], reasons: Sequence[str]) -> dict[str, Any]:
    return {
        "schema_version": QUEUE_SCHEMA,
        "cf_proposal_id": candidate["cf_proposal_id"],
        "candidate_id": candidate["candidate_id"],
        "query_id": candidate["query_id"],
        "query_content": candidate["query_content"],
        "gold_quadruples": candidate["gold_quadruples"],
        "tuple_index": candidate["tuple_index"],
        "field": candidate["field"],
        "gold_value": candidate["gold_value"],
        "candidate_value": candidate["candidate_value"],
        "family": candidate["family"],
        "source": candidate["source"],
        "queue_reasons": list(reasons),
        "model_votes_hidden": True,
    }


def _human_template(candidate: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": FINAL_REVIEW_SCHEMA,
        "cf_proposal_id": candidate["cf_proposal_id"],
        "candidate_id": candidate["candidate_id"],
        "decision": "",
        "reason_code": "",
        "note": "",
        "reviewer_id": PANEL_REVIEWER_ID,
    }


def build_cf_consensus(
    proposal_rows: Sequence[Mapping[str, Any]],
    reviews_by_reviewer: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    low_confidence_threshold: float,
    agreement_qc_fraction: float,
    panel_reviewer_id: str = PANEL_REVIEWER_ID,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Build exact-agreement auto decisions and a model-blind human queue."""

    if not isinstance(panel_reviewer_id, str) or not panel_reviewer_id:
        raise BlindReviewError("panel reviewer ID must be non-empty")
    if not 0 <= low_confidence_threshold <= 1:
        raise BlindReviewError("low-confidence threshold must be in [0,1]")
    if set(reviews_by_reviewer) != set(REVIEWER_IDS):
        raise BlindReviewError("CF consensus requires the frozen GLM and DeepSeek streams")
    candidates: dict[str, dict[str, Any]] = {}
    for raw in proposal_rows:
        row = dict(raw)
        candidate_id = row.get("candidate_id")
        if not isinstance(candidate_id, str) or candidate_id in candidates:
            raise BlindReviewError("proposal review frame contains an invalid/duplicate candidate ID")
        if row.get("review_required") is True:
            candidates[candidate_id] = row
    if not candidates:
        raise BlindReviewError("CF proposal has no review-required candidates")
    streams: dict[str, dict[str, dict[str, Any]]] = {}
    for reviewer_id in REVIEWER_IDS:
        stream: dict[str, dict[str, Any]] = {}
        for raw in reviews_by_reviewer[reviewer_id]:
            row = dict(raw)
            candidate_id = row.get("candidate_id")
            if not isinstance(candidate_id, str) or candidate_id in stream:
                raise BlindReviewError(f"reviewer {reviewer_id} has duplicate/invalid candidate IDs")
            if row.get("reviewer_id") not in {None, reviewer_id}:
                raise BlindReviewError("review row reviewer_id differs from its stream")
            stream[candidate_id] = row
        if set(stream) != set(candidates):
            raise BlindReviewError(f"reviewer {reviewer_id} candidate frame is not exact")
        streams[reviewer_id] = stream

    consensus_rows: list[dict[str, Any]] = []
    queue: list[dict[str, Any]] = []
    auto_rows: list[dict[str, Any]] = []
    for candidate_id in sorted(candidates):
        candidate = candidates[candidate_id]
        first = streams[REVIEWER_IDS[0]][candidate_id]
        second = streams[REVIEWER_IDS[1]][candidate_id]
        flags: set[str] = set()
        if first.get("error") or second.get("error"):
            flags.add("parse_failure")
        projections = (_decision_projection(first), _decision_projection(second))
        if not flags:
            if projections[0] is None or projections[1] is None:
                flags.add("low_confidence")
            elif projections[0] != projections[1]:
                flags.add("model_disagreement")
        confidences = []
        for review in (first, second):
            judgement = review.get("judgement")
            if isinstance(judgement, Mapping):
                value = judgement.get("confidence")
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    confidences.append(float(value))
        if (
            not first.get("error")
            and not second.get("error")
            and (len(confidences) != 2 or min(confidences) < low_confidence_threshold)
        ):
            flags.add("low_confidence")
        if not flags and _qc_selected(candidate_id, agreement_qc_fraction):
            flags.add("agreement_qc_sample")
        reasons = [reason for reason in QUEUE_REASONS if reason in flags]
        agreed = projections[0] is not None and projections[0] == projections[1]
        consensus_rows.append(
            {
                "schema_version": CONSENSUS_SCHEMA,
                "cf_proposal_id": candidate["cf_proposal_id"],
                "candidate_id": candidate_id,
                "agreed": agreed,
                "decision_projection": projections[0] if agreed else None,
                "human_queue_reasons": reasons,
            }
        )
        if reasons:
            queue.append(_queue_packet(candidate, reasons))
            continue
        if not agreed or projections[0] is None:
            raise BlindReviewError("non-agreement escaped the CF human queue")
        auto_rows.append(
            {
                "schema_version": FINAL_REVIEW_SCHEMA,
                "cf_proposal_id": candidate["cf_proposal_id"],
                "candidate_id": candidate_id,
                "decision": projections[0]["decision"],
                "reason_code": projections[0]["reason_code"],
                "note": "Two independent blind reviewers reached exact structured agreement.",
                "reviewer_id": panel_reviewer_id,
            }
        )
    return consensus_rows, queue, auto_rows


def _deterministic_review_rows(
    proposal_rows: Sequence[Mapping[str, Any]],
    *,
    panel_reviewer_id: str = PANEL_REVIEWER_ID,
) -> list[dict[str, Any]]:
    rows = []
    for candidate in sorted(proposal_rows, key=lambda row: str(row["candidate_id"])):
        if candidate.get("review_required") is True:
            continue
        if candidate.get("field") not in {"targeted_group", "hateful"}:
            raise BlindReviewError("only group/hate candidates may bypass subjective CF review")
        rows.append(
            {
                "schema_version": FINAL_REVIEW_SCHEMA,
                "cf_proposal_id": candidate["cf_proposal_id"],
                "candidate_id": candidate["candidate_id"],
                "decision": "not_required",
                "reason_code": "deterministic-label-foil",
                "note": "Deterministic canonical label foil; subjective review is not required.",
                "reviewer_id": panel_reviewer_id,
            }
        )
    return rows


def _proposal_snapshot(
    proposal_ref: str | Path,
    *,
    workspace_root: str | Path,
) -> tuple[
    dict[str, Any],
    Path,
    dict[str, Any],
    list[dict[str, Any]],
    dict[str, Any],
    str,
    bool,
]:
    try:
        validation = validate_proposal_ref(proposal_ref, workspace_root=workspace_root)
        locator, target = resolve_locator_ref(proposal_ref, "cf-proposal")
        dependency = portable_dependency(locator, target, workspace_root)
        config = load_json(target / "config.resolved.json")
        split = config["split"]
        rows = load_jsonl(target / f"candidates.{split}.jsonl")
        rubric_meta = load_json(target / "review_rubric.meta.json")
        rubric_body = (target / "review_rubric.md").read_text(encoding="utf-8")
    except (TrainingArtifactError, OSError, KeyError) as exc:
        raise BlindReviewError(f"cannot load immutable CF proposal: {exc}") from exc
    if validation.get("cf_proposal_id") != locator.get("artifact_id"):
        raise BlindReviewError("CF proposal validation/locator ID mismatch")
    return (
        locator,
        target,
        dependency,
        rows,
        rubric_meta,
        rubric_body,
        validation.get("scientific_eligible") is True,
    )


def _validate_transport_contract(
    *,
    scientific_eligible: bool,
    execution_mode: str,
    transport: Transport,
) -> None:
    """Keep injected transports outside every scientific/formal CF run."""

    if execution_mode == D14_LIVE_EXECUTION_MODE:
        if transport is not http_transport:
            raise BlindReviewError(
                "live D14 CF review requires the built-in HTTP transport"
            )
        return
    if execution_mode == D14_SYNTHETIC_EXECUTION_MODE:
        if scientific_eligible:
            raise BlindReviewError(
                "scientific/formal CF review forbids an injected synthetic transport"
            )
        if transport is http_transport:
            raise BlindReviewError(
                "engineering-synthetic mode requires an explicitly injected transport"
            )
        return
    raise BlindReviewError("CF review execution mode is not recognized")


def _model_projection(reviewers: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
    return [
        {
            "reviewer_id": row["reviewer_id"],
            "api_base": row["api_base"],
            "model": row["model"],
        }
        for row in sorted(reviewers, key=lambda item: item["reviewer_id"])
    ]


def _run_inputs(
    *,
    proposal_dependency: Mapping[str, Any],
    policy: Mapping[str, Any],
    rubric_meta: Mapping[str, Any],
    reviewers: Sequence[Mapping[str, str]],
    blind_items: Sequence[tuple[str, Mapping[str, Any]]],
    automatic_rows: Sequence[Mapping[str, Any]],
    scientific_eligible: bool,
    execution_mode: str,
) -> dict[str, Any]:
    return {
        "schema_version": RUN_SCHEMA,
        "proposal_dependency": dict(proposal_dependency),
        "policy_sha256": canonical_sha256(policy),
        "system_prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest(),
        "rubric_body_sha256": rubric_meta["rubric_body_sha256"],
        "rubric_meta_sha256": canonical_sha256(rubric_meta),
        "review_code_sha256": sha256_file(__file__),
        "scientific_eligible": scientific_eligible,
        "review_execution_mode": execution_mode,
        "reviewers": _model_projection(reviewers),
        "ordered_review_candidate_ids_sha256": canonical_sha256(
            [row["candidate_id"] for _, row in blind_items]
        ),
        "review_candidate_count": len(blind_items),
        "automatic_candidate_ids_sha256": canonical_sha256(
            [row["candidate_id"] for row in automatic_rows]
        ),
        "automatic_candidate_count": len(automatic_rows),
        "panel_reviewer_id": PANEL_REVIEWER_ID,
    }


def _summary(
    *,
    run_id: str,
    proposal_rows: Sequence[Mapping[str, Any]],
    reviews_by_reviewer: Mapping[str, Sequence[Mapping[str, Any]]],
    auto_rows: Sequence[Mapping[str, Any]],
    queue: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    required_count = sum(row.get("review_required") is True for row in proposal_rows)
    return {
        "schema_version": SUMMARY_SCHEMA,
        "cf_blind_review_id": run_id,
        "candidate_count": len(proposal_rows),
        "review_required_count": required_count,
        "deterministic_not_required_count": len(proposal_rows) - required_count,
        "model_call_count": sum(len(rows) for rows in reviews_by_reviewer.values()),
        "auto_review_count": len(auto_rows),
        "human_queue_count": len(queue),
        "human_queue_reasons": {
            reason: sum(reason in packet["queue_reasons"] for packet in queue)
            for reason in QUEUE_REASONS
        },
    }


def _provenance(
    *,
    run_id: str,
    proposal_dependency: Mapping[str, Any],
    model_rows: Mapping[str, Sequence[Mapping[str, Any]]],
    consensus: Sequence[Mapping[str, Any]],
    auto_rows: Sequence[Mapping[str, Any]],
    queue: Sequence[Mapping[str, Any]],
    scientific_eligible: bool,
    execution_mode: str,
) -> dict[str, Any]:
    return {
        "schema_version": PROVENANCE_SCHEMA,
        "cf_blind_review_id": run_id,
        "proposal_dependency": dict(proposal_dependency),
        "scientific_eligible": scientific_eligible,
        "review_execution_mode": execution_mode,
        "synthetic_transport_injected": (
            execution_mode == D14_SYNTHETIC_EXECUTION_MODE
        ),
        "model_review_rows_sha256": {
            reviewer_id: hashlib.sha256(
                canonical_jsonl_bytes(rows, key="candidate_id")
            ).hexdigest()
            for reviewer_id, rows in sorted(model_rows.items())
        },
        "consensus_rows_sha256": hashlib.sha256(
            canonical_jsonl_bytes(consensus, key="candidate_id")
        ).hexdigest(),
        "auto_review_rows_sha256": hashlib.sha256(
            canonical_jsonl_bytes(auto_rows, key="candidate_id")
        ).hexdigest(),
        "human_queue_rows_sha256": hashlib.sha256(
            canonical_jsonl_bytes(queue, key="candidate_id")
        ).hexdigest(),
        "raw_request_without_secrets_retained": True,
        "raw_response_retained_unless_credential_echoed": True,
        "credential_echo_drop_count": sum(
            row.get("error", {}).get("code") == "raw-response-contained-credential"
            for rows in model_rows.values()
            for row in rows
            if isinstance(row.get("error"), Mapping)
        ),
        "temperature": 0,
        "attempts_per_candidate_reviewer": 1,
        "saw_condition_outputs": False,
        "saw_model_scores": False,
    }


def run_cf_blind_review(
    *,
    proposal_ref: str | Path,
    policy_path: str | Path,
    env_file: str | Path,
    output_dir: str | Path,
    write_ref: str | Path,
    workspace_root: str | Path = REPOSITORY_ROOT,
    transport: Transport = http_transport,
    execution_mode: str = D14_LIVE_EXECUTION_MODE,
) -> dict[str, Any]:
    """Run each required CF candidate once through each independent reviewer."""

    policy = _load_policy(policy_path)
    (
        locator,
        proposal_target,
        dependency,
        rows,
        rubric_meta,
        rubric_body,
        scientific_eligible,
    ) = _proposal_snapshot(proposal_ref, workspace_root=workspace_root)
    _validate_transport_contract(
        scientific_eligible=scientific_eligible,
        execution_mode=execution_mode,
        transport=transport,
    )
    review_rows = [row for row in rows if row["review_required"] is True]
    deterministic_rows = _deterministic_review_rows(rows)
    if not review_rows:
        raise BlindReviewError("CF proposal has no target/argument candidates to review")
    env_values = load_env_file(env_file, allowed_names=_credential_env_names(policy))
    reviewers = _resolved_reviewers(
        policy,
        env_values,
        require_live_d14=execution_mode == D14_LIVE_EXECUTION_MODE,
    )
    secrets = _secret_values(policy, env_values)
    if len(secrets) != 2:
        raise BlindReviewError("CF review requires two independently resolved API credentials")
    blind_items = _blind_order(
        review_rows, int(policy["blindness"]["stable_shuffle_seed"])
    )
    reviews_by_reviewer: dict[str, list[dict[str, Any]]] = {}
    concurrency = int(policy.get("max_concurrency_per_reviewer", 1))
    if not 1 <= concurrency <= 8:
        raise BlindReviewError("max_concurrency_per_reviewer must be in [1,8]")
    for resolved in reviewers:
        def review_one(item: tuple[str, dict[str, Any]]) -> dict[str, Any]:
            blind_item_id, candidate = item
            return _call_reviewer(
                resolved,
                policy=policy,
                blind_item_id=blind_item_id,
                candidate=candidate,
                rubric_meta=rubric_meta,
                rubric_body=rubric_body,
                secrets=secrets,
                transport=transport,
            )

        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            stream = list(executor.map(review_one, blind_items))
        reviews_by_reviewer[resolved["reviewer_id"]] = stream

    # Revalidate after all external calls so proposal replacement/tampering cannot
    # silently change the reviewed frame.
    post_validation = validate_proposal_ref(proposal_ref, workspace_root=workspace_root)
    if post_validation["payload_manifest_sha256"] != locator["payload_manifest_sha256"]:
        raise BlindReviewError("CF proposal changed while blind review was running")
    consensus, queue, agreed_rows = build_cf_consensus(
        rows,
        reviews_by_reviewer,
        low_confidence_threshold=float(policy["consensus"]["low_confidence_threshold"]),
        agreement_qc_fraction=float(policy["consensus"]["agreement_qc_fraction"]),
    )
    all_auto_rows = sorted(
        [*agreed_rows, *deterministic_rows], key=lambda row: row["candidate_id"]
    )
    candidate_map = {row["candidate_id"]: row for row in rows}
    templates = [_human_template(candidate_map[row["candidate_id"]]) for row in queue]
    run_inputs = _run_inputs(
        proposal_dependency=dependency,
        policy=policy,
        rubric_meta=rubric_meta,
        reviewers=reviewers,
        blind_items=blind_items,
        automatic_rows=deterministic_rows,
        scientific_eligible=scientific_eligible,
        execution_mode=execution_mode,
    )
    run_id = "cfblind-" + canonical_sha256(run_inputs)
    run_meta = {**run_inputs, "cf_blind_review_id": run_id}
    model_resolution = {"reviewers": _model_projection(reviewers)}
    summary = _summary(
        run_id=run_id,
        proposal_rows=rows,
        reviews_by_reviewer=reviews_by_reviewer,
        auto_rows=all_auto_rows,
        queue=queue,
    )
    provenance = _provenance(
        run_id=run_id,
        proposal_dependency=dependency,
        model_rows=reviews_by_reviewer,
        consensus=consensus,
        auto_rows=all_auto_rows,
        queue=queue,
        scientific_eligible=scientific_eligible,
        execution_mode=execution_mode,
    )
    for value in (run_meta, policy, model_resolution, summary, provenance):
        _assert_no_sensitive_keys(value)
        if _contains_secret(value, secrets):
            raise BlindReviewError("credential escaped into CF run metadata")
    parent = Path(output_dir)
    target = parent / run_id
    staging = new_staging_directory(parent, run_id)
    try:
        write_canonical_json(staging / "proposal_ref.json", dependency)
        write_canonical_json(staging / "run.meta.json", run_meta)
        write_canonical_json(staging / "policy.resolved.json", policy)
        write_canonical_json(staging / "model_resolution.json", model_resolution)
        for reviewer_id, model_reviews in reviews_by_reviewer.items():
            write_canonical_jsonl(
                staging / "model_reviews" / f"{reviewer_id}.jsonl",
                model_reviews,
                key="candidate_id",
            )
        write_canonical_jsonl(staging / "consensus.jsonl", consensus, key="candidate_id")
        write_canonical_jsonl(staging / "human_queue.jsonl", queue, key="candidate_id")
        write_canonical_jsonl(staging / "auto_review.jsonl", all_auto_rows, key="candidate_id")
        write_canonical_jsonl(
            staging / "human_review_template.jsonl", templates, key="candidate_id"
        )
        write_canonical_json(staging / "summary.json", summary)
        write_canonical_json(staging / "provenance.json", provenance)
        payload_hash = finalize_target_atomic(
            staging,
            target,
            validate_staging=lambda candidate: validate_cf_blind_review_target(
                candidate,
                workspace_root=workspace_root,
                proposal_target=proposal_target,
                require_directory_name=False,
            ),
        )
    except (TrainingArtifactError, OSError) as exc:
        if staging.exists():
            shutil.rmtree(staging)
        raise BlindReviewError(f"cannot publish immutable CF blind review: {exc}") from exc
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    try:
        return write_locator_ref(
            write_ref,
            artifact_kind=ARTIFACT_KIND,
            artifact_id=run_id,
            target=target,
            payload_manifest_sha256=payload_hash,
        )
    except TrainingArtifactError as exc:
        raise BlindReviewError(f"cannot write CF blind-review locator: {exc}") from exc


def _expected_file_set() -> set[str]:
    return {
        "proposal_ref.json",
        "run.meta.json",
        "policy.resolved.json",
        "model_resolution.json",
        "model_reviews/deepseek_flash.jsonl",
        "model_reviews/glm_high_parameter.jsonl",
        "consensus.jsonl",
        "human_queue.jsonl",
        "auto_review.jsonl",
        "human_review_template.jsonl",
        "summary.json",
        "provenance.json",
        "payload_manifest.json",
    }


def _assert_canonical_json(path: Path, document: Any) -> None:
    if path.read_bytes() != canonical_json_bytes(document) + b"\n":
        raise BlindReviewError(f"CF blind-review JSON is not canonical: {path.name}")


def _assert_canonical_jsonl(
    path: Path, rows: Sequence[Mapping[str, Any]], *, key: str = "candidate_id"
) -> None:
    try:
        expected = canonical_jsonl_bytes(rows, key=key)
    except TrainingArtifactError as exc:
        raise BlindReviewError(f"invalid CF blind-review JSONL: {path.name}: {exc}") from exc
    if path.read_bytes() != expected:
        raise BlindReviewError(f"CF blind-review JSONL is not canonical: {path.name}")


def _validate_model_review_row(
    row: Mapping[str, Any],
    *,
    candidate: Mapping[str, Any],
    blind_item_id: str,
    reviewer: Mapping[str, str],
    policy: Mapping[str, Any],
    rubric_meta: Mapping[str, Any],
    rubric_body: str,
) -> None:
    expected_keys = {
        "schema_version",
        "cf_proposal_id",
        "candidate_id",
        "blind_item_id",
        "reviewer_id",
        "provider_api_base",
        "requested_model",
        "returned_model",
        "provider_request_id",
        "prompt_sha256",
        "request_without_secrets",
        "raw_response",
        "judgement",
        "error",
    }
    if set(row) != expected_keys or row.get("schema_version") != MODEL_REVIEW_SCHEMA:
        raise BlindReviewError("CF model-review row has non-canonical fields/schema")
    if (
        row.get("cf_proposal_id") != candidate["cf_proposal_id"]
        or row.get("candidate_id") != candidate["candidate_id"]
        or row.get("blind_item_id") != blind_item_id
        or row.get("reviewer_id") != reviewer["reviewer_id"]
        or row.get("provider_api_base") != reviewer["api_base"]
        or row.get("requested_model") != reviewer["model"]
    ):
        raise BlindReviewError("CF model-review identity/frame mismatch")
    _assert_no_sensitive_keys(row)
    prompt = _candidate_prompt(blind_item_id, candidate, rubric_meta, rubric_body)
    prompt_hash = hashlib.sha256((SYSTEM_PROMPT + "\n" + prompt).encode("utf-8")).hexdigest()
    expected_payload: dict[str, Any] = {
        "model": reviewer["model"],
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0,
        "top_p": policy["top_p"],
        "stream": False,
        "max_tokens": int(policy["max_tokens"]),
        "response_format": {"type": "json_object"},
    }
    options = policy["reviewer_request_options"].get(reviewer["reviewer_id"], {})
    for key in ("thinking", "reasoning_effort"):
        if options.get(key) is not None:
            expected_payload[key] = options[key]
    expected_request = {
        "url": _chat_url(reviewer["api_base"]),
        "payload": expected_payload,
        "prompt_sha256": prompt_hash,
    }
    if row.get("prompt_sha256") != prompt_hash or row.get("request_without_secrets") != expected_request:
        raise BlindReviewError("CF model-review prompt/request provenance mismatch")
    raw = row.get("raw_response")
    error = row.get("error")
    judgement = row.get("judgement")
    try:
        validate_requested_returned_model(
            row.get("requested_model"), row.get("returned_model")
        )
    except ValueError as exc:
        raise BlindReviewError(str(exc)) from exc
    if not isinstance(raw, Mapping):
        raise BlindReviewError(
            "D14 CF review must retain a raw response with provider model identity"
        )
    if raw.get("model") != row.get("returned_model"):
        raise BlindReviewError("CF raw/returned model identity mismatch")
    if error is None:
        if not isinstance(judgement, Mapping):
            raise BlindReviewError("successful CF model review must retain raw response/judgement")
        choices = raw.get("choices")
        if (
            not isinstance(choices, list)
            or not choices
            or not isinstance(choices[0], Mapping)
            or choices[0].get("finish_reason") != "stop"
            or not isinstance(choices[0].get("message"), Mapping)
            or not isinstance(choices[0]["message"].get("content"), str)
        ):
            raise BlindReviewError("successful CF raw response has a non-canonical choice")
        try:
            parsed = _extract_json_object(choices[0]["message"]["content"])
        except (BlindReviewError, ValueError, KeyError, TypeError) as exc:
            raise BlindReviewError("successful CF raw response cannot be parsed") from exc
        if (
            _validate_judgement(parsed, rubric_meta=rubric_meta) != judgement
            or _validate_judgement(judgement, rubric_meta=rubric_meta) != judgement
        ):
            raise BlindReviewError("CF model-review judgement is not normalized")
    else:
        if (
            not isinstance(error, Mapping)
            or set(error) != {"code"}
            or error.get("code") not in ERROR_CODES
            or judgement is not None
        ):
            raise BlindReviewError("failed CF model review has invalid structured error")
        if error["code"] == "raw-response-contained-credential" and raw is not None:
            raise BlindReviewError("credential-echo response must be discarded")
    expected_returned_model = raw.get("model")
    expected_request_id = raw.get("id")
    if row.get("returned_model") != expected_returned_model or row.get(
        "provider_request_id"
    ) != expected_request_id:
        raise BlindReviewError("CF provider response provenance mismatch")


def _validate_final_row(
    row: Mapping[str, Any],
    *,
    candidate: Mapping[str, Any],
    rubric_meta: Mapping[str, Any],
    queued: bool = False,
) -> None:
    try:
        validate_json_schema(row, SCHEMA_ROOT / "stage1_cf_review_v1.schema.json")
    except TrainingArtifactError as exc:
        raise BlindReviewError(f"invalid completed CF review row: {exc}") from exc
    if (
        row.get("cf_proposal_id") != candidate["cf_proposal_id"]
        or row.get("candidate_id") != candidate["candidate_id"]
        or row.get("reviewer_id") != PANEL_REVIEWER_ID
    ):
        raise BlindReviewError("completed CF review identity/reviewer mismatch")
    decision = row["decision"]
    reason = row["reason_code"]
    if candidate["review_required"] is True:
        if decision == "pass" and reason not in rubric_meta["pass_reason_codes"]:
            raise BlindReviewError("completed CF pass reason is invalid")
        if decision == "reject" and reason not in rubric_meta["reject_reason_codes"]:
            raise BlindReviewError("completed CF reject reason is invalid")
        if decision not in {"pass", "reject"}:
            raise BlindReviewError("subjective CF candidate must end in pass/reject")
    elif decision != "not_required" or reason != "deterministic-label-foil":
        raise BlindReviewError("deterministic group/hate CF review is invalid")
    if queued and candidate["review_required"] is not True:
        raise BlindReviewError("deterministic CF candidate cannot enter the human queue")


def validate_cf_blind_review_target(
    target_dir: str | Path,
    *,
    workspace_root: str | Path = REPOSITORY_ROOT,
    proposal_target: str | Path | None = None,
    require_directory_name: bool = True,
) -> dict[str, Any]:
    """Replay a CF blind-review artifact without resolving credentials or calling APIs."""

    target = Path(target_dir)
    try:
        payload_hash = validate_payload_manifest(target)
        ensure_exact_file_set(target, _expected_file_set())
        dependency = validate_dependency_ref(
            load_json(target / "proposal_ref.json"), expected_kind="cf-proposal"
        )
        resolved_proposal = (
            Path(proposal_target).resolve()
            if proposal_target is not None
            else resolve_dependency_target(dependency, workspace_root)
        )
        if (
            resolved_proposal.name != dependency["artifact_id"]
            or validate_payload_manifest(resolved_proposal)
            != dependency["payload_manifest_sha256"]
        ):
            raise BlindReviewError("CF blind-review proposal dependency does not match target")
        proposal_validation = validate_proposal_target(
            resolved_proposal, workspace_root=workspace_root
        )
        config = load_json(resolved_proposal / "config.resolved.json")
        proposal_rows = load_jsonl(
            resolved_proposal / f"candidates.{config['split']}.jsonl"
        )
        rubric_meta = load_json(resolved_proposal / "review_rubric.meta.json")
        rubric_body = (resolved_proposal / "review_rubric.md").read_text(encoding="utf-8")
        policy = _validate_policy_document(load_json(target / "policy.resolved.json"))
        run_meta = load_json(target / "run.meta.json")
        model_resolution = load_json(target / "model_resolution.json")
        consensus = load_jsonl(target / "consensus.jsonl")
        queue = load_jsonl(target / "human_queue.jsonl")
        auto_rows = load_jsonl(target / "auto_review.jsonl")
        templates = load_jsonl(target / "human_review_template.jsonl")
        summary = load_json(target / "summary.json")
        provenance = load_json(target / "provenance.json")
        model_rows = {
            reviewer_id: load_jsonl(target / "model_reviews" / f"{reviewer_id}.jsonl")
            for reviewer_id in REVIEWER_IDS
        }
    except (TrainingArtifactError, CounterfactualLifecycleError, OSError, KeyError) as exc:
        raise BlindReviewError(f"cannot validate CF blind-review target: {exc}") from exc
    for path, document in (
        (target / "proposal_ref.json", dependency),
        (target / "run.meta.json", run_meta),
        (target / "policy.resolved.json", policy),
        (target / "model_resolution.json", model_resolution),
        (target / "summary.json", summary),
        (target / "provenance.json", provenance),
    ):
        _assert_canonical_json(path, document)
    for reviewer_id in REVIEWER_IDS:
        _assert_canonical_jsonl(
            target / "model_reviews" / f"{reviewer_id}.jsonl",
            model_rows[reviewer_id],
        )
    for filename, rows in (
        ("consensus.jsonl", consensus),
        ("human_queue.jsonl", queue),
        ("auto_review.jsonl", auto_rows),
        ("human_review_template.jsonl", templates),
    ):
        _assert_canonical_jsonl(target / filename, rows)
    for value in (
        dependency,
        run_meta,
        policy,
        model_resolution,
        consensus,
        queue,
        auto_rows,
        templates,
        summary,
        provenance,
        *model_rows.values(),
    ):
        _assert_no_sensitive_keys(value)

    scientific_eligible = config.get("scientific_eligible")
    if not isinstance(scientific_eligible, bool):
        raise BlindReviewError("CF proposal scientific eligibility is malformed")
    execution_mode = run_meta.get("review_execution_mode")
    if execution_mode not in {
        D14_LIVE_EXECUTION_MODE,
        D14_SYNTHETIC_EXECUTION_MODE,
    }:
        raise BlindReviewError("CF blind-review execution mode is missing/non-canonical")
    if scientific_eligible and execution_mode != D14_LIVE_EXECUTION_MODE:
        raise BlindReviewError(
            "scientific/formal CF review cannot claim a synthetic transport"
        )

    reviewers = model_resolution.get("reviewers") if isinstance(model_resolution, Mapping) else None
    if (
        not isinstance(reviewers, list)
        or len(reviewers) != 2
        or {row.get("reviewer_id") for row in reviewers if isinstance(row, Mapping)}
        != set(REVIEWER_IDS)
    ):
        raise BlindReviewError("CF model resolution is not the exact frozen two-model frame")
    normalized_reviewers: list[dict[str, str]] = []
    for raw in reviewers:
        if not isinstance(raw, Mapping) or set(raw) != {"reviewer_id", "api_base", "model"}:
            raise BlindReviewError("CF model-resolution row is non-canonical")
        if not all(isinstance(raw.get(key), str) and raw[key] for key in raw):
            raise BlindReviewError("CF model-resolution fields must be non-empty strings")
        normalized_reviewers.append(
            {
                "reviewer_id": str(raw["reviewer_id"]),
                "api_base": _safe_api_base(str(raw["api_base"])),
                "model": str(raw["model"]),
            }
        )
    try:
        validate_reviewer_resolution(
            normalized_reviewers,
            require_live_d14=execution_mode == D14_LIVE_EXECUTION_MODE,
        )
    except ValueError as exc:
        raise BlindReviewError(str(exc)) from exc
    review_candidates = [row for row in proposal_rows if row["review_required"] is True]
    deterministic_rows = _deterministic_review_rows(proposal_rows)
    blind_items = _blind_order(
        review_candidates, int(policy["blindness"]["stable_shuffle_seed"])
    )
    blind_map = {row["candidate_id"]: blind_id for blind_id, row in blind_items}
    candidate_map = {row["candidate_id"]: row for row in proposal_rows}
    reviewer_map = {row["reviewer_id"]: row for row in normalized_reviewers}
    for reviewer_id in REVIEWER_IDS:
        rows = model_rows[reviewer_id]
        if len(rows) != len(review_candidates) or {
            row.get("candidate_id") for row in rows
        } != set(blind_map):
            raise BlindReviewError("CF model-review candidate frame is incomplete/non-unique")
        for row in rows:
            candidate_id = row["candidate_id"]
            _validate_model_review_row(
                row,
                candidate=candidate_map[candidate_id],
                blind_item_id=blind_map[candidate_id],
                reviewer=reviewer_map[reviewer_id],
                policy=policy,
                rubric_meta=rubric_meta,
                rubric_body=rubric_body,
            )
    expected_consensus, expected_queue, expected_agreed = build_cf_consensus(
        proposal_rows,
        model_rows,
        low_confidence_threshold=float(policy["consensus"]["low_confidence_threshold"]),
        agreement_qc_fraction=float(policy["consensus"]["agreement_qc_fraction"]),
    )
    if consensus != sorted(expected_consensus, key=lambda row: row["candidate_id"]):
        raise BlindReviewError("stored CF consensus cannot be replayed")
    if queue != sorted(expected_queue, key=lambda row: row["candidate_id"]):
        raise BlindReviewError("stored CF human queue cannot be replayed")
    expected_auto = sorted(
        [*expected_agreed, *deterministic_rows], key=lambda row: row["candidate_id"]
    )
    if auto_rows != expected_auto:
        raise BlindReviewError("stored CF automatic review rows cannot be replayed")
    expected_templates = sorted(
        [_human_template(candidate_map[row["candidate_id"]]) for row in expected_queue],
        key=lambda row: row["candidate_id"],
    )
    if templates != expected_templates:
        raise BlindReviewError("stored CF human templates cannot be replayed")
    for row in auto_rows:
        _validate_final_row(
            row, candidate=candidate_map[row["candidate_id"]], rubric_meta=rubric_meta
        )
    run_inputs = _run_inputs(
        proposal_dependency=dependency,
        policy=policy,
        rubric_meta=rubric_meta,
        reviewers=normalized_reviewers,
        blind_items=blind_items,
        automatic_rows=deterministic_rows,
        scientific_eligible=scientific_eligible,
        execution_mode=str(execution_mode),
    )
    run_id = "cfblind-" + canonical_sha256(run_inputs)
    if run_meta != {**run_inputs, "cf_blind_review_id": run_id}:
        raise BlindReviewError("CF blind-review run ID/inputs cannot be reproduced")
    if require_directory_name and target.name != run_id:
        raise BlindReviewError("CF blind-review target directory differs from run ID")
    expected_summary = _summary(
        run_id=run_id,
        proposal_rows=proposal_rows,
        reviews_by_reviewer=model_rows,
        auto_rows=auto_rows,
        queue=queue,
    )
    if summary != expected_summary:
        raise BlindReviewError("CF blind-review summary cannot be reproduced")
    expected_provenance = _provenance(
        run_id=run_id,
        proposal_dependency=dependency,
        model_rows=model_rows,
        consensus=consensus,
        auto_rows=auto_rows,
        queue=queue,
        scientific_eligible=scientific_eligible,
        execution_mode=str(execution_mode),
    )
    if provenance != expected_provenance:
        raise BlindReviewError("CF blind-review provenance cannot be reproduced")
    if proposal_validation["cf_proposal_id"] != dependency["artifact_id"]:
        raise BlindReviewError("CF proposal report/dependency mismatch")
    return {
        "schema_version": "stage1-cf-dual-blind-review-validation/v1",
        "valid": True,
        "cf_blind_review_id": run_id,
        "cf_proposal_id": dependency["artifact_id"],
        "scientific_eligible": scientific_eligible,
        "review_execution_mode": execution_mode,
        "candidate_count": len(proposal_rows),
        "review_required_count": len(review_candidates),
        "auto_review_count": len(auto_rows),
        "human_queue_count": len(queue),
        "payload_manifest_sha256": payload_hash,
    }


def validate_cf_blind_review_ref(
    review_ref: str | Path,
    *,
    workspace_root: str | Path = REPOSITORY_ROOT,
) -> dict[str, Any]:
    try:
        locator, target = resolve_locator_ref(review_ref, ARTIFACT_KIND)
    except TrainingArtifactError as exc:
        raise BlindReviewError(f"invalid CF blind-review locator: {exc}") from exc
    report = validate_cf_blind_review_target(target, workspace_root=workspace_root)
    if (
        report["cf_blind_review_id"] != locator["artifact_id"]
        or report["payload_manifest_sha256"] != locator["payload_manifest_sha256"]
    ):
        raise BlindReviewError("CF blind-review locator differs from immutable target")
    return report


def _review_snapshot(
    review_ref: str | Path,
    *,
    workspace_root: str | Path,
) -> tuple[Path, Path, dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    validate_cf_blind_review_ref(review_ref, workspace_root=workspace_root)
    try:
        _, review_target = resolve_locator_ref(review_ref, ARTIFACT_KIND)
        dependency = validate_dependency_ref(
            load_json(review_target / "proposal_ref.json"), expected_kind="cf-proposal"
        )
        proposal_target = resolve_dependency_target(dependency, workspace_root)
        config = load_json(proposal_target / "config.resolved.json")
        proposal_rows = load_jsonl(
            proposal_target / f"candidates.{config['split']}.jsonl"
        )
        rubric_meta = load_json(proposal_target / "review_rubric.meta.json")
    except (TrainingArtifactError, OSError, KeyError) as exc:
        raise BlindReviewError(f"cannot resolve CF blind-review workspace: {exc}") from exc
    return review_target, proposal_target, dependency, proposal_rows, rubric_meta


def export_cf_human_review(
    *,
    blind_review_ref: str | Path,
    output: str | Path,
    packet_output: str | Path,
    workspace_root: str | Path = REPOSITORY_ROOT,
) -> dict[str, Any]:
    """Export only the frozen human templates and model-blind queue packets."""

    review_target, _, dependency, proposal_rows, _ = _review_snapshot(
        blind_review_ref, workspace_root=workspace_root
    )
    templates = load_jsonl(review_target / "human_review_template.jsonl")
    packets = load_jsonl(review_target / "human_queue.jsonl")
    auto_rows = load_jsonl(review_target / "auto_review.jsonl")
    candidate_ids = {row["candidate_id"] for row in proposal_rows}
    template_ids = [row.get("candidate_id") for row in templates]
    packet_ids = [row.get("candidate_id") for row in packets]
    auto_ids = [row.get("candidate_id") for row in auto_rows]
    if (
        len(template_ids) != len(set(template_ids))
        or len(packet_ids) != len(set(packet_ids))
        or len(auto_ids) != len(set(auto_ids))
        or set(template_ids) != set(packet_ids)
        or set(template_ids).intersection(auto_ids)
        or set(template_ids).union(auto_ids) != candidate_ids
    ):
        raise BlindReviewError("CF human/auto/proposal candidate frames are inconsistent")
    allowed_packet_keys = {
        "schema_version",
        "cf_proposal_id",
        "candidate_id",
        "query_id",
        "query_content",
        "gold_quadruples",
        "tuple_index",
        "field",
        "gold_value",
        "candidate_value",
        "family",
        "source",
        "queue_reasons",
        "model_votes_hidden",
    }
    if any(
        set(packet) != allowed_packet_keys
        or packet.get("schema_version") != QUEUE_SCHEMA
        or packet.get("model_votes_hidden") is not True
        for packet in packets
    ):
        raise BlindReviewError("CF human packet shape/blindness contract is invalid")
    forbidden_terms = {
        "judgement",
        "raw_response",
        "reviewer_id",
        "requested_model",
        "provider_api_base",
        "decision_projection",
    }
    if any(set(packet).intersection(forbidden_terms) for packet in packets):
        raise BlindReviewError("CF human packets expose model provenance or votes")
    write_canonical_jsonl(output, templates, key="candidate_id")
    write_canonical_jsonl(packet_output, packets, key="candidate_id")
    return {
        "schema_version": "stage1-cf-human-review-export/v1",
        "cf_proposal_id": dependency["artifact_id"],
        "row_count": len(templates),
        "auto_review_count": len(auto_rows),
        "output": str(Path(output).resolve()),
        "packet_output": str(Path(packet_output).resolve()),
        "model_votes_hidden": True,
    }


def merge_cf_human_review(
    *,
    blind_review_ref: str | Path,
    human_completed: str | Path,
    output: str | Path,
    declaration_output: str | Path | None = None,
    workspace_root: str | Path = REPOSITORY_ROOT,
    reviewer_id: str = PANEL_REVIEWER_ID,
) -> dict[str, Any]:
    """Merge the exact frozen queue with auto rows into declaration-ready JSONL."""

    if reviewer_id != PANEL_REVIEWER_ID:
        raise BlindReviewError("CF dual-review merge uses the frozen panel reviewer ID")
    review_target, proposal_target, dependency, proposal_rows, rubric_meta = _review_snapshot(
        blind_review_ref, workspace_root=workspace_root
    )
    templates = load_jsonl(review_target / "human_review_template.jsonl")
    auto_rows = load_jsonl(review_target / "auto_review.jsonl")
    try:
        human_rows = load_jsonl(human_completed)
    except TrainingArtifactError as exc:
        raise BlindReviewError(f"cannot load completed CF human review: {exc}") from exc
    expected_ids = [row["candidate_id"] for row in templates]
    actual_ids = [row.get("candidate_id") for row in human_rows]
    if len(actual_ids) != len(set(actual_ids)) or set(actual_ids) != set(expected_ids):
        raise BlindReviewError("completed CF human candidate set differs from frozen queue")
    expected_keys = {
        "schema_version",
        "cf_proposal_id",
        "candidate_id",
        "decision",
        "reason_code",
        "note",
        "reviewer_id",
    }
    candidate_map = {row["candidate_id"]: row for row in proposal_rows}
    normalized_human: list[dict[str, Any]] = []
    for raw in human_rows:
        if set(raw) != expected_keys:
            raise BlindReviewError("completed CF human row shape changed")
        normalized = dict(raw)
        normalized["reviewer_id"] = PANEL_REVIEWER_ID
        candidate_id = normalized.get("candidate_id")
        if candidate_id not in candidate_map:
            raise BlindReviewError("completed CF human row references unknown candidate")
        _validate_final_row(
            normalized,
            candidate=candidate_map[candidate_id],
            rubric_meta=rubric_meta,
            queued=True,
        )
        normalized_human.append(normalized)
    merged = sorted([*auto_rows, *normalized_human], key=lambda row: row["candidate_id"])
    merged_ids = [row["candidate_id"] for row in merged]
    if len(merged_ids) != len(set(merged_ids)) or set(merged_ids) != set(candidate_map):
        raise BlindReviewError("merged CF review candidate set is not exactly the proposal")
    for row in merged:
        _validate_final_row(
            row, candidate=candidate_map[row["candidate_id"]], rubric_meta=rubric_meta
        )
    write_canonical_jsonl(output, merged, key="candidate_id")
    try:
        declaration = prepare_reviewer_declaration(
            blind_review_ref=blind_review_ref,
            review_file=output,
            reviewer_id=PANEL_REVIEWER_ID,
            workspace_root=workspace_root,
        )
    except CounterfactualLifecycleError as exc:
        raise BlindReviewError(
            f"merged CF review failed blind-lineage validation: {exc}"
        ) from exc
    if declaration_output is not None:
        write_canonical_json(declaration_output, declaration)
    # Keep the resolved target live through the merge and fail if it changed.
    if validate_payload_manifest(proposal_target) != dependency["payload_manifest_sha256"]:
        raise BlindReviewError("CF proposal changed during human-review merge")
    result = {
        "schema_version": "stage1-cf-human-review-merge/v1",
        "cf_proposal_id": dependency["artifact_id"],
        "row_count": len(merged),
        "human_review_count": len(normalized_human),
        "auto_review_count": len(auto_rows),
        "reviewer_id": PANEL_REVIEWER_ID,
        "cf_blind_review_id": declaration["cf_blind_review_dependency"][
            "artifact_id"
        ],
        "human_queue_rows_sha256": declaration["human_queue_rows_sha256"],
        "completed_review_rows_sha256": declaration["completed_review_rows_sha256"],
        "output": str(Path(output).resolve()),
    }
    if declaration_output is not None:
        result["declaration_output"] = str(Path(declaration_output).resolve())
    return result


def preflight_cf_blind_review(
    *,
    proposal_ref: str | Path,
    policy_path: str | Path,
    env_file: str | Path,
    workspace_root: str | Path = REPOSITORY_ROOT,
    transport: Transport = http_transport,
    execution_mode: str = D14_LIVE_EXECUTION_MODE,
) -> dict[str, Any]:
    """Call each provider once on one required candidate without writing an artifact."""

    policy = _load_policy(policy_path)
    (
        _,
        _,
        dependency,
        rows,
        rubric_meta,
        rubric_body,
        scientific_eligible,
    ) = _proposal_snapshot(proposal_ref, workspace_root=workspace_root)
    _validate_transport_contract(
        scientific_eligible=scientific_eligible,
        execution_mode=execution_mode,
        transport=transport,
    )
    review_rows = [row for row in rows if row["review_required"] is True]
    if not review_rows:
        raise BlindReviewError("CF proposal has no review-required preflight candidate")
    env_values = load_env_file(env_file, allowed_names=_credential_env_names(policy))
    reviewers = _resolved_reviewers(
        policy,
        env_values,
        require_live_d14=execution_mode == D14_LIVE_EXECUTION_MODE,
    )
    secrets = _secret_values(policy, env_values)
    if len(secrets) != 2:
        raise BlindReviewError("CF preflight requires two independent API credentials")
    blind_id, candidate = _blind_order(
        review_rows, int(policy["blindness"]["stable_shuffle_seed"])
    )[0]
    results = []
    for reviewer in reviewers:
        row = _call_reviewer(
            reviewer,
            policy=policy,
            blind_item_id=blind_id,
            candidate=candidate,
            rubric_meta=rubric_meta,
            rubric_body=rubric_body,
            secrets=secrets,
            transport=transport,
        )
        results.append(
            {
                "reviewer_id": reviewer["reviewer_id"],
                "model": reviewer["model"],
                "ok": row["error"] is None,
                "error": row["error"],
            }
        )
    return {
        "schema_version": "stage1-cf-dual-blind-review-preflight/v1",
        "cf_proposal_id": dependency["artifact_id"],
        "scientific_eligible": scientific_eligible,
        "review_execution_mode": execution_mode,
        "candidate_id": candidate["candidate_id"],
        "results": results,
        "artifact_written": False,
    }


__all__ = [
    "ARTIFACT_KIND",
    "BlindReviewError",
    "PANEL_REVIEWER_ID",
    "SYSTEM_PROMPT",
    "build_cf_consensus",
    "export_cf_human_review",
    "merge_cf_human_review",
    "preflight_cf_blind_review",
    "run_cf_blind_review",
    "validate_cf_blind_review_ref",
    "validate_cf_blind_review_target",
]
