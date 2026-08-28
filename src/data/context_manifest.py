"""Pure Stage 1 context rendering, one-pass budgeting, and validation.

Retrieval and selection are intentionally outside this module.  It accepts a
frozen selected evidence set plus content-addressed catalogs, freezes budget on
CLD exactly once, then derives C0/CL/CD/CLD only by subset deletion.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any, Protocol


CONDITIONS = ("C0", "CL", "CD", "CLD")
CONTEXT_RECORD_SCHEMA = "stage1-context-record/v1"
TRIM_POLICY = "drop-demo-tail-then-fail/v1"


class ContextManifestError(ValueError):
    """Raised when a frozen context invariant is violated."""


class ContextOverflow(ContextManifestError):
    """Raised when CLD cannot fit without deleting lexicon/query/schema text."""


class ChatTokenizer(Protocol):
    def apply_chat_template(
        self,
        conversation: Sequence[Mapping[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        enable_thinking: bool,
    ) -> Any: ...


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


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.replace("\r\n", "\n").encode("utf-8")).hexdigest()


def stable_demo_id(*, source_record_id: str, content_sha256: str, gold_sha256: str) -> str:
    payload = {
        "source_record_id": source_record_id,
        "content_sha256": content_sha256,
        "gold_sha256": gold_sha256,
    }
    return "demo:v1:" + canonical_sha256(payload)


def stable_lexicon_id(
    *, term: str, category: str, definition: str, variants: Sequence[str]
) -> str:
    payload = {
        "term": term,
        "category": category,
        "definition": definition,
        "variants": list(variants),
    }
    return "lex:v1:" + canonical_sha256(payload)


def stable_term_evidence_id(
    *,
    term: str,
    definition: str,
    variants: Sequence[str],
    usage_notes: str = "",
    ambiguity_notes: str = "",
) -> str:
    payload = {
        "term": term,
        "definition": definition,
        "variants": list(variants),
        "usage_notes": usage_notes,
        "ambiguity_notes": ambiguity_notes,
    }
    return "lex:v2:" + canonical_sha256(payload)


def _catalog_item(catalog: Mapping[str, Mapping[str, Any]], item_id: str, kind: str) -> Mapping[str, Any]:
    try:
        item = catalog[item_id]
    except KeyError as exc:
        raise ContextManifestError(f"unknown {kind} catalog ID: {item_id}") from exc
    if item.get(f"{kind}_id") not in {None, item_id}:
        raise ContextManifestError(f"{kind} catalog identity mismatch: {item_id}")
    block = item.get("rendered_block")
    if not isinstance(block, str):
        raise ContextManifestError(f"{kind} {item_id} lacks rendered_block")
    expected_hash = item.get("rendered_block_sha256")
    if expected_hash is not None and expected_hash != text_sha256(block):
        raise ContextManifestError(f"{kind} block hash mismatch: {item_id}")
    return item


def _evidence_text(
    ids: Sequence[str], catalog: Mapping[str, Mapping[str, Any]], kind: str
) -> str:
    blocks = [_catalog_item(catalog, item_id, kind)["rendered_block"] for item_id in ids]
    return "\n\n".join(blocks)


def render_messages(
    *,
    query_content: str,
    lexicon_ids: Sequence[str],
    demo_ids: Sequence[str],
    lexicon_catalog: Mapping[str, Mapping[str, Any]],
    demo_catalog: Mapping[str, Mapping[str, Any]],
    system_prompt: str,
    user_prompt_template: str,
) -> list[dict[str, str]]:
    """Render messages without I/O, retrieval, selection, or truncation."""

    if not isinstance(query_content, str) or not query_content:
        raise ContextManifestError("query content must be non-empty text")
    required_placeholders = ("{lexicons}", "{examples}", "{text}")
    if not all(placeholder in user_prompt_template for placeholder in required_placeholders):
        raise ContextManifestError("user prompt template lacks a Stage 1 placeholder")
    lexicons = _evidence_text(lexicon_ids, lexicon_catalog, "lexicon")
    examples = _evidence_text(demo_ids, demo_catalog, "demo")
    user = user_prompt_template.format(
        lexicons=lexicons or "（无）",
        examples=examples or "（无）",
        text=query_content,
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user},
    ]


def chat_prompt_text(messages: Sequence[Mapping[str, str]], tokenizer: ChatTokenizer) -> str:
    try:
        rendered = tokenizer.apply_chat_template(
            list(messages),
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        rendered = tokenizer.apply_chat_template(
            list(messages),
            tokenize=False,
            add_generation_prompt=True,
        )
    if not isinstance(rendered, str):
        raise ContextManifestError("tokenizer chat template did not return text")
    return rendered


def token_count(text: str, tokenizer: Any) -> int:
    if hasattr(tokenizer, "encode"):
        encoded = tokenizer.encode(text, add_special_tokens=False)
    else:
        encoded = tokenizer(text, add_special_tokens=False)["input_ids"]
    if encoded and isinstance(encoded[0], list):
        encoded = encoded[0]
    return len(encoded)


def _render_condition(
    *,
    query_content: str,
    condition: str,
    final_lexicon_ids: Sequence[str],
    final_demo_ids: Sequence[str],
    lexicon_catalog: Mapping[str, Mapping[str, Any]],
    demo_catalog: Mapping[str, Mapping[str, Any]],
    system_prompt: str,
    user_prompt_template: str,
    tokenizer: ChatTokenizer,
) -> dict[str, Any]:
    if condition not in CONDITIONS:
        raise ContextManifestError(f"unknown condition: {condition}")
    lexicon_ids = list(final_lexicon_ids) if condition in {"CL", "CLD"} else []
    demo_ids = list(final_demo_ids) if condition in {"CD", "CLD"} else []
    messages = render_messages(
        query_content=query_content,
        lexicon_ids=lexicon_ids,
        demo_ids=demo_ids,
        lexicon_catalog=lexicon_catalog,
        demo_catalog=demo_catalog,
        system_prompt=system_prompt,
        user_prompt_template=user_prompt_template,
    )
    chat_text = chat_prompt_text(messages, tokenizer)
    return {
        "lexicon_ids": lexicon_ids,
        "demo_ids": demo_ids,
        "chat_prompt_tokens": token_count(chat_text, tokenizer),
        "chat_prompt_sha256": text_sha256(chat_text),
        "messages": messages,
    }


def finalize_context_budget(
    record: Mapping[str, Any],
    *,
    lexicon_catalog: Mapping[str, Mapping[str, Any]],
    demo_catalog: Mapping[str, Mapping[str, Any]],
    system_prompt: str,
    user_prompt_template: str,
    tokenizer: ChatTokenizer,
    max_sequence_tokens: int,
    completion_reserve_tokens: int,
    trim_policy: str = TRIM_POLICY,
) -> dict[str, Any]:
    """Freeze CLD once, then derive the other three conditions by deletion."""

    if trim_policy != TRIM_POLICY:
        raise ContextManifestError(f"unsupported trim policy: {trim_policy}")
    if max_sequence_tokens <= 0 or completion_reserve_tokens <= 0:
        raise ContextManifestError("token budgets must be positive")
    if completion_reserve_tokens >= max_sequence_tokens:
        raise ContextManifestError("completion reserve must be smaller than max sequence")
    result = copy.deepcopy(dict(record))
    query = result.get("query")
    if not isinstance(query, dict) or not isinstance(query.get("content"), str):
        raise ContextManifestError("record.query.content is required for budget rendering")
    selection = result.get("selection")
    if not isinstance(selection, dict):
        raise ContextManifestError("record.selection is required")
    demos = selection.get("demos")
    lexicons = selection.get("lexicons")
    if not isinstance(demos, dict) or not isinstance(lexicons, dict):
        raise ContextManifestError("record selection must contain demos and lexicons")
    demo_before = list(demos.get("prompt_order_before_budget", []))
    lexicon_before = list(lexicons.get("prompt_order_before_budget", []))
    if len(demo_before) != len(set(demo_before)) or len(lexicon_before) != len(set(lexicon_before)):
        raise ContextManifestError("pre-budget evidence IDs must be unique")

    def cld_count(current_demo_ids: Sequence[str]) -> int:
        condition = _render_condition(
            query_content=query["content"],
            condition="CLD",
            final_lexicon_ids=lexicon_before,
            final_demo_ids=current_demo_ids,
            lexicon_catalog=lexicon_catalog,
            demo_catalog=demo_catalog,
            system_prompt=system_prompt,
            user_prompt_template=user_prompt_template,
            tokenizer=tokenizer,
        )
        return int(condition["chat_prompt_tokens"])

    final_demo_ids = list(demo_before)
    before_tokens = cld_count(final_demo_ids)
    while final_demo_ids and cld_count(final_demo_ids) + completion_reserve_tokens > max_sequence_tokens:
        final_demo_ids.pop()
    after_tokens = cld_count(final_demo_ids)
    if after_tokens + completion_reserve_tokens > max_sequence_tokens:
        raise ContextOverflow(
            "base prompt plus frozen lexicon exceeds the sequence budget; "
            "query/schema/lexicon truncation is forbidden"
        )
    demos["prompt_order_final"] = final_demo_ids
    demos["budget_dropped_ids"] = demo_before[len(final_demo_ids) :]
    lexicons["prompt_order_final"] = lexicon_before
    lexicons["budget_dropped_ids"] = []
    conditions = {
        condition: _render_condition(
            query_content=query["content"],
            condition=condition,
            final_lexicon_ids=lexicon_before,
            final_demo_ids=final_demo_ids,
            lexicon_catalog=lexicon_catalog,
            demo_catalog=demo_catalog,
            system_prompt=system_prompt,
            user_prompt_template=user_prompt_template,
            tokenizer=tokenizer,
        )
        for condition in CONDITIONS
    }
    result["budget"] = {
        "max_sequence_tokens": max_sequence_tokens,
        "completion_reserve_tokens": completion_reserve_tokens,
        "trim_policy": trim_policy,
        "cld_chat_tokens_before": before_tokens,
        "cld_chat_tokens_after": after_tokens,
        "tail_truncated": False,
        "status": "ok",
    }
    result["conditions"] = conditions
    result["schema_version"] = CONTEXT_RECORD_SCHEMA
    result["record_sha256"] = canonical_sha256(
        {key: value for key, value in result.items() if key != "record_sha256"}
    )
    validate_context_record(result)
    return result


def validate_context_record(record: Mapping[str, Any]) -> None:
    if record.get("schema_version") != CONTEXT_RECORD_SCHEMA:
        raise ContextManifestError("wrong context record schema")
    selection = record.get("selection", {})
    demos = selection.get("demos", {})
    lexicons = selection.get("lexicons", {})
    demo_before = demos.get("prompt_order_before_budget")
    demo_final = demos.get("prompt_order_final")
    demo_dropped = demos.get("budget_dropped_ids")
    lex_before = lexicons.get("prompt_order_before_budget")
    lex_final = lexicons.get("prompt_order_final")
    if not all(isinstance(value, list) for value in (demo_before, demo_final, demo_dropped, lex_before, lex_final)):
        raise ContextManifestError("context selection orders must be arrays")
    if demo_final != demo_before[: len(demo_final)]:
        raise ContextManifestError("final demo order must be an exact pre-budget prefix")
    if demo_dropped != demo_before[len(demo_final) :]:
        raise ContextManifestError("dropped demo IDs must be the exact suffix")
    if lex_final != lex_before or lexicons.get("budget_dropped_ids") != []:
        raise ContextManifestError("drop-demo-tail policy cannot remove lexicon evidence")
    conditions = record.get("conditions")
    if not isinstance(conditions, dict) or set(conditions) != set(CONDITIONS):
        raise ContextManifestError("context record must have exactly C0/CL/CD/CLD")
    expected = {
        "C0": ([], []),
        "CL": (lex_final, []),
        "CD": ([], demo_final),
        "CLD": (lex_final, demo_final),
    }
    for condition, (expected_lex, expected_demo) in expected.items():
        payload = conditions[condition]
        if payload.get("lexicon_ids") != expected_lex or payload.get("demo_ids") != expected_demo:
            raise ContextManifestError(f"condition {condition} is not a pure evidence subset")
        if payload.get("chat_prompt_tokens", 0) <= 0 or not isinstance(payload.get("chat_prompt_sha256"), str):
            raise ContextManifestError(f"condition {condition} lacks prompt audit fields")
    budget = record.get("budget", {})
    if budget.get("trim_policy") != TRIM_POLICY or budget.get("tail_truncated") is not False:
        raise ContextManifestError("context budget policy/tail flag is invalid")
    if conditions["CLD"]["chat_prompt_tokens"] != budget.get("cld_chat_tokens_after"):
        raise ContextManifestError("CLD token count disagrees with budget record")
    if conditions["CLD"]["chat_prompt_tokens"] + budget.get("completion_reserve_tokens", 0) > budget.get("max_sequence_tokens", 0):
        raise ContextManifestError("final CLD exceeds the frozen sequence budget")
    recorded_hash = record.get("record_sha256")
    if recorded_hash is not None:
        actual = canonical_sha256({key: value for key, value in record.items() if key != "record_sha256"})
        if recorded_hash != actual:
            raise ContextManifestError("context record hash mismatch")


def render_condition_item(record: Mapping[str, Any], condition: str) -> dict[str, Any]:
    """Return an adapter-neutral item from already frozen condition messages."""

    validate_context_record(record)
    if condition not in CONDITIONS:
        raise ContextManifestError(f"unknown condition: {condition}")
    payload = record["conditions"][condition]
    query = record["query"]
    return {
        "id": str(query["id"]),
        "content": query["content"],
        "gt_quadruples": query["gold"],
        "messages_list": [payload["messages"]],
        "context_manifest": {
            "context_build_id": record.get("context_build_id"),
            "record_sha256": record["record_sha256"],
            "condition": condition,
            "lexicon_ids": payload["lexicon_ids"],
            "demo_ids": payload["demo_ids"],
            "chat_prompt_tokens": payload["chat_prompt_tokens"],
            "chat_prompt_sha256": payload["chat_prompt_sha256"],
        },
    }


def render_sft_item(record: Mapping[str, Any], condition: str) -> dict[str, Any]:
    """Render the same frozen condition into the existing SFT loader shape."""

    from utils.quadruple import serialize_quadruples

    runner_item = render_condition_item(record, condition)
    messages = runner_item["messages_list"][0]
    if len(messages) != 2 or messages[0].get("role") != "system" or messages[1].get("role") != "user":
        raise ContextManifestError("Stage 1 SFT requires exactly system+user messages")
    return {
        "id": runner_item["id"],
        "instruction": messages[0]["content"],
        "input": messages[1]["content"],
        "output": serialize_quadruples(runner_item["gt_quadruples"]),
        "content": runner_item["content"],
        "metadata": {"condition": condition, "output_protocol": "quad-json-v1"},
        "context_manifest": runner_item["context_manifest"],
    }


__all__ = [
    "CONDITIONS",
    "CONTEXT_RECORD_SCHEMA",
    "TRIM_POLICY",
    "ContextManifestError",
    "ContextOverflow",
    "canonical_sha256",
    "chat_prompt_text",
    "finalize_context_budget",
    "render_condition_item",
    "render_sft_item",
    "render_messages",
    "stable_demo_id",
    "stable_lexicon_id",
    "stable_term_evidence_id",
    "text_sha256",
    "token_count",
    "validate_context_record",
]
