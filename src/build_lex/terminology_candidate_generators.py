"""Offline contracts for the WP3 v2 terminology candidate generators.

This module intentionally stops before provider execution and artifact
publication.  It provides deterministic request rendering, strict response
normalization, form-rule proposals, and exact-occurrence unioning for:

* G1 minimal rewrite and alignment;
* G2 direct terminology mention extraction; and
* G3 phonetic/orthographic/mixed-form rules.

Generator output is only a proposal.  It never assigns A/B/C/R, writes a
lexicon entry, or consumes task annotations.  Numeric offsets supplied by a
model are rejected by the canonical response shapes; offsets are resolved
from the copied source surface and its 1-based occurrence ordinal.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any


CONFIG_SCHEMA_VERSION = "wp3-candidate-generators-config/v1"
OBSERVATION_SCHEMA_VERSION = "wp3-candidate-observation/v1"
CANDIDATE_SCHEMA_VERSION = "wp3-candidate-mention/v1"
RESOURCE_ROLE = "wp3-candidate-generators/v2"
SOURCE_POLICY = "fit-content-only-no-task-fields/v1"
HANDBOOK_VERSION = "wp3-terminology-evidence-handbook/v1.0"

G1_PROMPT_VERSIONS = {
    "surface_decode": "wp3-g1-surface-decode/v1",
    "lexical_pragmatic": "wp3-g1-lexical-pragmatic/v1",
}
G2_PROMPT_VERSION = "wp3-g2-direct-mention/v1"
G3_RULE_VERSION = "wp3-form-rules/v1"

TASK_FIELD_KEYS = frozenset(
    {
        "annotation_count",
        "argument",
        "categories",
        "category",
        "category_counts",
        "category_purity",
        "hate_count",
        "hate_precision",
        "hateful",
        "label",
        "labels",
        "log_odds",
        "non_hate_count",
        "nonhate_penalty",
        "primary_category",
        "quadruples",
        "target",
        "targeted_group",
        "task_prediction",
    }
)
PUBLIC_TASK_KEYS = frozenset({"task_id", "blind_alias", "content"})
GENERATORS = frozenset({"g1_rewrite", "g2_direct", "g3_form_rule"})
MECHANISMS = frozenset(
    {
        "abbreviation",
        "emoji",
        "fixed_expression",
        "known_variant",
        "metaphor",
        "mixed_script",
        "orthographic_variant",
        "other_nontransparent",
        "phonetic_variant",
        "pragmatic_usage",
        "separator_insertion",
        "slang",
        "unicode_nfkc",
    }
)
G1_PASS_MECHANISMS = {
    "surface_decode": frozenset(
        {
            "abbreviation",
            "emoji",
            "known_variant",
            "mixed_script",
            "orthographic_variant",
            "phonetic_variant",
            "separator_insertion",
            "unicode_nfkc",
        }
    ),
    "lexical_pragmatic": frozenset(
        {
            "fixed_expression",
            "metaphor",
            "other_nontransparent",
            "pragmatic_usage",
            "slang",
        }
    ),
}

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_CJK_RUN_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]+")
_LATIN_CJK_RE = re.compile(r"[A-Za-z0-9]+[\u3400-\u4dbf\u4e00-\u9fff]{1,2}")
_CJK_LATIN_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]{1,2}[A-Za-z0-9]+")
_LATIN_DIGIT_RE = re.compile(
    r"(?<![A-Za-z0-9])(?=[A-Za-z0-9]*[A-Za-z])(?=[A-Za-z0-9]*[0-9])[A-Za-z0-9]+(?![A-Za-z0-9])"
)
_EMOJI_RE = re.compile(
    "["
    "\U0001F1E6-\U0001F1FF"
    "\U0001F300-\U0001FAFF"
    "\u2600-\u27BF"
    "](?:[\uFE0E\uFE0F\u200D]|[\U0001F300-\U0001FAFF\u2600-\u27BF])*"
)
_SEPARATOR_RE = r"[\s._*·~\-—|/\\]+"


G1_SURFACE_SYSTEM_PROMPT = """你是 WP3 的局部形式解码提议器。
你只会收到一条原始文本。

把文本改写为语义等价、标准、直白的中文，但只修改依赖缩写、音形替代、混写、数字、符号、
emoji、全半角或分隔编码才能理解的最小连续表达。不要改写普通负面词、普通身份称谓、完整句子或
仅因下游任务相关而显眼的词。

每个 source_surface 必须逐字复制自原文；用 occurrence_ordinal 表示它在原文中的第几次出现。
不要输出 start/end。所有 edits 必须互不重叠，且 edits 应能逐字重建 rewritten_text。
可以返回空 edits。

mechanism 只能是 abbreviation、phonetic_variant、orthographic_variant、mixed_script、
unicode_nfkc、emoji、separator_insertion 或 known_variant。

只输出 JSON object：
{
  "rewritten_text":"...",
  "edits":[{
    "source_surface":"...",
    "occurrence_ordinal":1,
    "replacement":"...",
    "mechanism":"mixed_script",
    "requires_context":false,
    "reason":"..."
  }],
  "record_reason":"..."
}"""

G1_LEXICAL_SYSTEM_PROMPT = """你是 WP3 的局部词汇与语用改写提议器。
你只会收到一条原始文本。

把文本改写为语义等价、标准、直白的中文，但只修改不了解网络俚语、固定搭配、非组合隐喻或
特殊语用就难以理解的最小连续表达。不要处理仅有形式编码的问题，不要选择普通群体名、通用辱骂、
语法碎片、完整句子或透明组合。

每个 source_surface 必须逐字复制自原文；用 occurrence_ordinal 表示它在原文中的第几次出现。
不要输出 start/end。所有 edits 必须互不重叠，且 edits 应能逐字重建 rewritten_text。
可以返回空 edits。

mechanism 只能是 slang、fixed_expression、metaphor、pragmatic_usage 或
other_nontransparent。

只输出 JSON object：
{
  "rewritten_text":"...",
  "edits":[{
    "source_surface":"...",
    "occurrence_ordinal":1,
    "replacement":"...",
    "mechanism":"slang",
    "requires_context":true,
    "reason":"..."
  }],
  "record_reason":"..."
}"""

G2_SYSTEM_PROMPT = """你是 WP3 的术语 mention 提议器。你只会收到一条原始文本。

找出最小连续表达：如果不了解网络语、缩写、编码形式、固定搭配或特殊语用，就可能无法正确理解它。
候选可以是中性表达，也可以有多义；是否被提议不代表当前句子应得出任何下游结论。

硬规则：
1. surface 必须逐字复制自原文，用 occurrence_ordinal 标记第几次出现，不要输出 start/end。
2. 排除普通人名地名身份词、一般情绪词、通用辱骂、透明组合、语法碎片、完整句和一次性修辞。
3. 选最小但语义完整的 span。easy girl 的短语义不能投给 easy；全国女性中的连续字符不等于
独立术语；女拳不、女拳都、些女拳是碎片，完整的女拳需另提。
4. 同一表达出现多次时分别提议。最多 8 个，可以返回空数组。

mechanism 只能是 abbreviation、phonetic_variant、orthographic_variant、mixed_script、
unicode_nfkc、emoji、separator_insertion、slang、fixed_expression、metaphor、
pragmatic_usage、known_variant 或 other_nontransparent。不确定具体机制时使用
other_nontransparent。

只输出 JSON object：
{
  "mentions":[{
    "surface":"...",
    "occurrence_ordinal":1,
    "mechanism":"abbreviation",
    "requires_context":true,
    "reason":"..."
  }],
  "record_reason":"..."
}"""


class CandidateGeneratorError(RuntimeError):
    """Raised when a v2 generator contract or output is invalid."""


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        rendered = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise CandidateGeneratorError(f"value is not canonical JSON: {exc}") from exc
    return rendered.encode("utf-8")


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    try:
        with Path(path).open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise CandidateGeneratorError(f"cannot hash {path}: {exc}") from exc
    return digest.hexdigest()


def _forbidden_key_paths(value: Any, path: tuple[str, ...] = ()) -> list[str]:
    found: list[str] = []
    if isinstance(value, Mapping):
        for key, inner in value.items():
            child = (*path, str(key))
            if str(key).casefold() in TASK_FIELD_KEYS:
                found.append(".".join(child))
            found.extend(_forbidden_key_paths(inner, child))
    elif isinstance(value, list):
        for index, inner in enumerate(value):
            found.extend(_forbidden_key_paths(inner, (*path, str(index))))
    return found


def _require_text(value: Any, name: str, *, maximum: int) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or len(value) > maximum
    ):
        raise CandidateGeneratorError(f"{name} must be non-empty trimmed text")
    return value


def _require_int(value: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise CandidateGeneratorError(f"{name} must be an integer >= {minimum}")
    return value


def _content_sha256(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def exact_occurrences(content: str, surface: str) -> list[tuple[int, int]]:
    """Return overlapping Python-code-point offsets for one exact surface."""

    if not surface:
        return []
    result: list[tuple[int, int]] = []
    cursor = 0
    while cursor <= len(content) - len(surface):
        start = content.find(surface, cursor)
        if start < 0:
            break
        result.append((start, start + len(surface)))
        cursor = start + 1
    return result


def resolve_exact_span(
    content: str, surface: str, occurrence_ordinal: int
) -> tuple[int, int]:
    ordinal = _require_int(occurrence_ordinal, "occurrence_ordinal", minimum=1)
    positions = exact_occurrences(content, surface)
    if ordinal > len(positions):
        raise CandidateGeneratorError("surface occurrence cannot be resolved")
    return positions[ordinal - 1]


def occurrence_ordinal(content: str, surface: str, start: int, end: int) -> int:
    try:
        return exact_occurrences(content, surface).index((start, end)) + 1
    except ValueError as exc:
        raise CandidateGeneratorError("offsets do not identify the stated surface") from exc


def build_public_task(
    record: Mapping[str, Any], *, blind_alias: str | None = None
) -> dict[str, str]:
    """Project an arbitrary fit row to the only object a model runner may use."""

    record_id = _require_text(record.get("id"), "record.id", maximum=256)
    content = record.get("content")
    if not isinstance(content, str) or not content:
        raise CandidateGeneratorError("record.content must be non-empty text")
    content_sha = _content_sha256(content)
    task_id = "wp3task-" + hashlib.sha256(
        f"{record_id}\x1f{content_sha}".encode("utf-8")
    ).hexdigest()[:32]
    alias = blind_alias or ("术语-" + task_id[-8:])
    task = {"task_id": task_id, "blind_alias": alias, "content": content}
    if _forbidden_key_paths(task):
        raise CandidateGeneratorError("public task contains a forbidden field")
    return task


def _validate_public_task(task: Mapping[str, Any]) -> None:
    if set(task) != PUBLIC_TASK_KEYS or _forbidden_key_paths(task):
        raise CandidateGeneratorError("model task fields are not canonical")
    _require_text(task.get("task_id"), "task_id", maximum=80)
    _require_text(task.get("blind_alias"), "blind_alias", maximum=80)
    if not isinstance(task.get("content"), str) or not task["content"]:
        raise CandidateGeneratorError("task content must be non-empty text")


def _request(model: str, system_prompt: str, content: str) -> dict[str, Any]:
    _require_text(model, "model", maximum=160)
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": json.dumps(
                    {"original_record": content},
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            },
        ],
        "temperature": 0,
        "top_p": 1,
        "max_tokens": 1024,
        "stream": False,
        "response_format": {"type": "json_object"},
    }


def build_g1_request(
    task: Mapping[str, Any], *, pass_name: str, model: str
) -> dict[str, Any]:
    _validate_public_task(task)
    if pass_name == "surface_decode":
        prompt = G1_SURFACE_SYSTEM_PROMPT
    elif pass_name == "lexical_pragmatic":
        prompt = G1_LEXICAL_SYSTEM_PROMPT
    else:
        raise CandidateGeneratorError("unsupported G1 pass")
    request = _request(model, prompt, str(task["content"]))
    if _forbidden_key_paths(request):
        raise CandidateGeneratorError("G1 request contains a forbidden field")
    return request


def build_g2_request(task: Mapping[str, Any], *, model: str) -> dict[str, Any]:
    _validate_public_task(task)
    request = _request(model, G2_SYSTEM_PROMPT, str(task["content"]))
    if _forbidden_key_paths(request):
        raise CandidateGeneratorError("G2 request contains a forbidden field")
    return request


def build_model_source(
    *,
    provider: str,
    model: str,
    prompt_version: str,
    response: str | Mapping[str, Any],
) -> dict[str, Any]:
    provider_value = _require_text(provider, "provider", maximum=80)
    model_value = _require_text(model, "model", maximum=160)
    prompt_value = _require_text(prompt_version, "prompt_version", maximum=160)
    response_bytes = (
        response.encode("utf-8")
        if isinstance(response, str)
        else _canonical_json_bytes(response)
    )
    payload = {
        "kind": "model",
        "provider": provider_value,
        "model": model_value,
        "prompt_version": prompt_value,
        "response_sha256": hashlib.sha256(response_bytes).hexdigest(),
    }
    return {
        **payload,
        "source_id": "modelsrc-" + _canonical_sha256(payload)[:32],
    }


def build_rule_source(
    *, rule_version: str, reference_sha256: str | None
) -> dict[str, Any]:
    rule_value = _require_text(rule_version, "rule_version", maximum=160)
    if reference_sha256 is not None and not _SHA256_RE.fullmatch(reference_sha256):
        raise CandidateGeneratorError("reference_sha256 is invalid")
    payload = {
        "kind": "rule",
        "rule_version": rule_value,
        "reference_sha256": reference_sha256,
    }
    return {
        **payload,
        "source_id": "rulesrc-" + _canonical_sha256(payload)[:32],
    }


def _validate_source(source: Any) -> dict[str, Any]:
    if not isinstance(source, Mapping):
        raise CandidateGeneratorError("observation source must be an object")
    kind = source.get("kind")
    if kind == "model":
        if set(source) != {
            "kind",
            "source_id",
            "provider",
            "model",
            "prompt_version",
            "response_sha256",
        }:
            raise CandidateGeneratorError("model source fields are not canonical")
        if not _SHA256_RE.fullmatch(str(source.get("response_sha256", ""))):
            raise CandidateGeneratorError("model response hash is invalid")
        payload = {
            key: source[key]
            for key in ("kind", "provider", "model", "prompt_version", "response_sha256")
        }
        expected_id = "modelsrc-" + _canonical_sha256(payload)[:32]
        if source.get("source_id") != expected_id:
            raise CandidateGeneratorError("model source id is not reproducible")
    elif kind == "rule":
        if set(source) != {
            "kind",
            "source_id",
            "rule_version",
            "reference_sha256",
        }:
            raise CandidateGeneratorError("rule source fields are not canonical")
        expected = build_rule_source(
            rule_version=str(source["rule_version"]),
            reference_sha256=source.get("reference_sha256"),
        )
        if dict(source) != expected:
            raise CandidateGeneratorError("rule source id is not reproducible")
    else:
        raise CandidateGeneratorError("unknown observation source kind")
    return dict(source)


def _make_observation(
    *,
    record_id: str,
    content: str,
    surface: str,
    start: int,
    end: int,
    generator: str,
    generator_variant: str,
    mechanism: str,
    replacement: str | None,
    requires_context: bool | None,
    rationale: str,
    source: Mapping[str, Any],
) -> dict[str, Any]:
    if generator not in GENERATORS:
        raise CandidateGeneratorError("unknown generator")
    if mechanism not in MECHANISMS:
        raise CandidateGeneratorError("unknown mechanism")
    surface_value = _require_text(surface, "surface", maximum=80)
    if content[start:end] != surface_value:
        raise CandidateGeneratorError("observation offsets differ from raw content")
    replacement_value = None
    if replacement is not None:
        replacement_value = _require_text(replacement, "replacement", maximum=160)
    rationale_value = _require_text(rationale, "rationale", maximum=500)
    source_value = _validate_source(source)
    base = {
        "record_id": _require_text(record_id, "record_id", maximum=256),
        "content_sha256": _content_sha256(content),
        "surface": surface_value,
        "start": _require_int(start, "start"),
        "end": _require_int(end, "end", minimum=1),
        "occurrence_ordinal": occurrence_ordinal(content, surface_value, start, end),
        "generator": generator,
        "generator_variant": _require_text(
            generator_variant, "generator_variant", maximum=80
        ),
        "mechanism": mechanism,
        "replacement": replacement_value,
        "requires_context": requires_context,
        "rationale": rationale_value,
        "source": source_value,
    }
    if requires_context is not None and not isinstance(requires_context, bool):
        raise CandidateGeneratorError("requires_context must be boolean or null")
    observation = {
        "schema_version": OBSERVATION_SCHEMA_VERSION,
        "observation_id": "wp3obs-" + _canonical_sha256(base)[:32],
        **base,
    }
    validate_observation(observation, content=content)
    return observation


def _observation_base(observation: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: observation[key]
        for key in (
            "record_id",
            "content_sha256",
            "surface",
            "start",
            "end",
            "occurrence_ordinal",
            "generator",
            "generator_variant",
            "mechanism",
            "replacement",
            "requires_context",
            "rationale",
            "source",
        )
    }


def validate_observation(
    observation: Mapping[str, Any], *, content: str | None = None
) -> dict[str, Any]:
    expected_fields = {
        "schema_version",
        "observation_id",
        "record_id",
        "content_sha256",
        "surface",
        "start",
        "end",
        "occurrence_ordinal",
        "generator",
        "generator_variant",
        "mechanism",
        "replacement",
        "requires_context",
        "rationale",
        "source",
    }
    if not isinstance(observation, Mapping) or set(observation) != expected_fields:
        raise CandidateGeneratorError("observation fields are not canonical")
    if observation.get("schema_version") != OBSERVATION_SCHEMA_VERSION:
        raise CandidateGeneratorError("unsupported observation schema")
    if _forbidden_key_paths(observation):
        raise CandidateGeneratorError("observation contains a forbidden task field")
    _require_text(observation.get("record_id"), "record_id", maximum=256)
    surface = _require_text(observation.get("surface"), "surface", maximum=80)
    start = _require_int(observation.get("start"), "start")
    end = _require_int(observation.get("end"), "end", minimum=1)
    ordinal = _require_int(
        observation.get("occurrence_ordinal"), "occurrence_ordinal", minimum=1
    )
    if end <= start:
        raise CandidateGeneratorError("observation end must exceed start")
    if observation.get("generator") not in GENERATORS:
        raise CandidateGeneratorError("observation generator is invalid")
    _require_text(
        observation.get("generator_variant"), "generator_variant", maximum=80
    )
    if observation.get("mechanism") not in MECHANISMS:
        raise CandidateGeneratorError("observation mechanism is invalid")
    replacement = observation.get("replacement")
    if replacement is not None:
        _require_text(replacement, "replacement", maximum=160)
    context_value = observation.get("requires_context")
    if context_value is not None and not isinstance(context_value, bool):
        raise CandidateGeneratorError("requires_context is invalid")
    _require_text(observation.get("rationale"), "rationale", maximum=500)
    _validate_source(observation.get("source"))
    content_sha = str(observation.get("content_sha256", ""))
    if not _SHA256_RE.fullmatch(content_sha):
        raise CandidateGeneratorError("content_sha256 is invalid")
    expected_id = "wp3obs-" + _canonical_sha256(_observation_base(observation))[:32]
    if observation.get("observation_id") != expected_id:
        raise CandidateGeneratorError("observation id is not reproducible")
    if content is not None:
        if _content_sha256(content) != content_sha:
            raise CandidateGeneratorError("observation content hash differs")
        if content[start:end] != surface:
            raise CandidateGeneratorError("observation surface differs from content")
        if occurrence_ordinal(content, surface, start, end) != ordinal:
            raise CandidateGeneratorError("observation occurrence ordinal differs")
    return dict(observation)


def normalize_g1_response(
    parsed: Mapping[str, Any],
    *,
    record_id: str,
    content: str,
    pass_name: str,
    source: Mapping[str, Any],
    max_edits: int = 8,
) -> list[dict[str, Any]]:
    """Validate G1 edits and prove they exactly reconstruct rewritten_text."""

    if pass_name not in G1_PROMPT_VERSIONS:
        raise CandidateGeneratorError("unsupported G1 pass")
    if not isinstance(parsed, Mapping) or set(parsed) != {
        "rewritten_text",
        "edits",
        "record_reason",
    }:
        raise CandidateGeneratorError("G1 response fields are not canonical")
    if _forbidden_key_paths(parsed):
        raise CandidateGeneratorError("G1 response contains a forbidden task field")
    rewritten = parsed.get("rewritten_text")
    edits = parsed.get("edits")
    record_reason = parsed.get("record_reason")
    if not isinstance(rewritten, str) or not isinstance(edits, list):
        raise CandidateGeneratorError("G1 response types are invalid")
    if not isinstance(record_reason, str):
        raise CandidateGeneratorError("G1 record_reason must be text")
    if len(edits) > max_edits:
        raise CandidateGeneratorError("G1 returned too many edits")

    normalized: list[tuple[int, int, dict[str, Any]]] = []
    seen: set[tuple[int, int, str]] = set()
    for index, row in enumerate(edits):
        if not isinstance(row, Mapping) or set(row) != {
            "source_surface",
            "occurrence_ordinal",
            "replacement",
            "mechanism",
            "requires_context",
            "reason",
        }:
            raise CandidateGeneratorError(f"G1 edit {index} fields are invalid")
        surface = _require_text(
            row.get("source_surface"), f"G1 edit {index} surface", maximum=80
        )
        replacement = _require_text(
            row.get("replacement"), f"G1 edit {index} replacement", maximum=160
        )
        if replacement == surface:
            raise CandidateGeneratorError(f"G1 edit {index} does not change the text")
        mechanism = str(row.get("mechanism", ""))
        if mechanism not in G1_PASS_MECHANISMS[pass_name]:
            raise CandidateGeneratorError(f"G1 edit {index} mechanism is invalid")
        context_value = row.get("requires_context")
        if not isinstance(context_value, bool):
            raise CandidateGeneratorError(
                f"G1 edit {index} requires_context must be boolean"
            )
        reason = _require_text(
            row.get("reason"), f"G1 edit {index} reason", maximum=500
        )
        start, end = resolve_exact_span(
            content,
            surface,
            _require_int(
                row.get("occurrence_ordinal"),
                f"G1 edit {index} occurrence_ordinal",
                minimum=1,
            ),
        )
        identity = (start, end, surface)
        if identity in seen:
            raise CandidateGeneratorError(f"G1 edit {index} is duplicated")
        seen.add(identity)
        normalized.append(
            (
                start,
                end,
                {
                    "surface": surface,
                    "replacement": replacement,
                    "mechanism": mechanism,
                    "requires_context": context_value,
                    "reason": reason,
                },
            )
        )

    normalized.sort(key=lambda item: (item[0], item[1], item[2]["surface"]))
    cursor = 0
    pieces: list[str] = []
    for start, end, edit in normalized:
        if start < cursor:
            raise CandidateGeneratorError("G1 edits overlap")
        pieces.append(content[cursor:start])
        pieces.append(str(edit["replacement"]))
        cursor = end
    pieces.append(content[cursor:])
    if "".join(pieces) != rewritten:
        raise CandidateGeneratorError("G1 edits do not reconstruct rewritten_text")

    return [
        _make_observation(
            record_id=record_id,
            content=content,
            surface=str(edit["surface"]),
            start=start,
            end=end,
            generator="g1_rewrite",
            generator_variant=pass_name,
            mechanism=str(edit["mechanism"]),
            replacement=str(edit["replacement"]),
            requires_context=bool(edit["requires_context"]),
            rationale=str(edit["reason"]),
            source=source,
        )
        for start, end, edit in normalized
    ]


def normalize_g2_response(
    parsed: Mapping[str, Any],
    *,
    record_id: str,
    content: str,
    source: Mapping[str, Any],
    max_mentions: int = 8,
) -> list[dict[str, Any]]:
    """Normalize direct mention output using copied surfaces, never offsets."""

    if not isinstance(parsed, Mapping) or set(parsed) != {
        "mentions",
        "record_reason",
    }:
        raise CandidateGeneratorError("G2 response fields are not canonical")
    if _forbidden_key_paths(parsed):
        raise CandidateGeneratorError("G2 response contains a forbidden task field")
    mentions = parsed.get("mentions")
    if not isinstance(mentions, list) or not isinstance(parsed.get("record_reason"), str):
        raise CandidateGeneratorError("G2 response types are invalid")
    if len(mentions) > max_mentions:
        raise CandidateGeneratorError("G2 returned too many mentions")
    result: list[dict[str, Any]] = []
    seen: set[tuple[int, int, str]] = set()
    for index, row in enumerate(mentions):
        if not isinstance(row, Mapping) or set(row) != {
            "surface",
            "occurrence_ordinal",
            "mechanism",
            "requires_context",
            "reason",
        }:
            raise CandidateGeneratorError(f"G2 mention {index} fields are invalid")
        surface = _require_text(
            row.get("surface"), f"G2 mention {index} surface", maximum=80
        )
        mechanism = str(row.get("mechanism", ""))
        if mechanism not in MECHANISMS:
            raise CandidateGeneratorError(f"G2 mention {index} mechanism is invalid")
        context_value = row.get("requires_context")
        if not isinstance(context_value, bool):
            raise CandidateGeneratorError(
                f"G2 mention {index} requires_context must be boolean"
            )
        reason = _require_text(
            row.get("reason"), f"G2 mention {index} reason", maximum=500
        )
        start, end = resolve_exact_span(
            content,
            surface,
            _require_int(
                row.get("occurrence_ordinal"),
                f"G2 mention {index} occurrence_ordinal",
                minimum=1,
            ),
        )
        identity = (start, end, surface)
        if identity in seen:
            raise CandidateGeneratorError(f"G2 mention {index} is duplicated")
        seen.add(identity)
        result.append(
            _make_observation(
                record_id=record_id,
                content=content,
                surface=surface,
                start=start,
                end=end,
                generator="g2_direct",
                generator_variant="direct_mention",
                mechanism=mechanism,
                replacement=None,
                requires_context=context_value,
                rationale=reason,
                source=source,
            )
        )
    result.sort(key=lambda row: (row["start"], row["end"], row["surface"]))
    return result


def normalize_form_reference(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Validate the label-free reference consumed by G3.

    The reference stores form knowledge only.  It is deliberately too small to
    act as a released lexicon and cannot contain task fields.
    """

    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping) or set(row) != {
            "canonical",
            "pinyin",
            "initials",
            "variants",
        }:
            raise CandidateGeneratorError(
                f"form reference row {index} fields are not canonical"
            )
        if _forbidden_key_paths(row):
            raise CandidateGeneratorError("form reference contains a forbidden field")
        canonical = _require_text(
            row.get("canonical"), f"form reference {index} canonical", maximum=80
        )
        lookup = unicodedata.normalize("NFKC", canonical).casefold()
        if lookup in seen:
            raise CandidateGeneratorError("form reference canonical is duplicated")
        seen.add(lookup)
        pinyin = row.get("pinyin")
        if not isinstance(pinyin, list) or any(
            not isinstance(value, str) or re.fullmatch(r"[a-z]+", value) is None
            for value in pinyin
        ):
            raise CandidateGeneratorError("form reference pinyin must be lowercase syllables")
        initials = row.get("initials")
        if initials is not None and (
            not isinstance(initials, str)
            or re.fullmatch(r"[a-z0-9]+", initials) is None
        ):
            raise CandidateGeneratorError("form reference initials are invalid")
        variants = row.get("variants")
        if not isinstance(variants, list):
            raise CandidateGeneratorError("form reference variants must be an array")
        normalized_variants = sorted(
            {
                _require_text(
                    value, f"form reference {index} variant", maximum=80
                )
                for value in variants
            }
        )
        if canonical in normalized_variants:
            raise CandidateGeneratorError("canonical cannot repeat as its own variant")
        normalized.append(
            {
                "canonical": canonical,
                "pinyin": list(pinyin),
                "initials": initials,
                "variants": normalized_variants,
            }
        )
    normalized.sort(key=lambda row: unicodedata.normalize("NFKC", row["canonical"]).casefold())
    return normalized


def form_reference_sha256(rows: Sequence[Mapping[str, Any]]) -> str:
    return _canonical_sha256(normalize_form_reference(rows))


def _nfkc_changed_spans(content: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start: int | None = None
    for index, char in enumerate(content):
        changed = unicodedata.normalize("NFKC", char) != char and not char.isspace()
        if changed and start is None:
            start = index
        elif not changed and start is not None:
            spans.append((start, index))
            start = None
    if start is not None:
        spans.append((start, len(content)))
    return spans


def _syllable_distance(left: Sequence[str], right: Sequence[str]) -> int:
    previous = list(range(len(right) + 1))
    for left_index, left_value in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_value in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1] + (left_value != right_value),
                )
            )
        previous = current
    return previous[-1]


def generate_g3_observations(
    *,
    record_id: str,
    content: str,
    reference_rows: Sequence[Mapping[str, Any]] = (),
    romanizer: Callable[[str], Sequence[str]] | None = None,
    rule_version: str = G3_RULE_VERSION,
    max_surface_chars: int = 80,
    mixed_script_max_chars: int = 24,
    phonetic_max_distance: int = 0,
) -> list[dict[str, Any]]:
    """Generate deterministic G3 observations without enumerating all n-grams."""

    record_value = _require_text(record_id, "record_id", maximum=256)
    if not isinstance(content, str) or not content:
        raise CandidateGeneratorError("content must be non-empty text")
    _require_int(max_surface_chars, "max_surface_chars", minimum=1)
    _require_int(mixed_script_max_chars, "mixed_script_max_chars", minimum=2)
    _require_int(phonetic_max_distance, "phonetic_max_distance", minimum=0)
    references = normalize_form_reference(reference_rows)
    reference_hash = _canonical_sha256(references) if references else None
    source = build_rule_source(
        rule_version=rule_version, reference_sha256=reference_hash
    )
    observations: dict[tuple[Any, ...], dict[str, Any]] = {}

    def add(
        *,
        start: int,
        end: int,
        variant: str,
        mechanism: str,
        replacement: str | None,
        rationale: str,
    ) -> None:
        surface = content[start:end]
        if not surface.strip() or len(surface) > max_surface_chars:
            return
        key = (start, end, surface, variant, mechanism, replacement)
        observations[key] = _make_observation(
            record_id=record_value,
            content=content,
            surface=surface,
            start=start,
            end=end,
            generator="g3_form_rule",
            generator_variant=variant,
            mechanism=mechanism,
            replacement=replacement,
            requires_context=None,
            rationale=rationale,
            source=source,
        )

    mixed_positions: set[tuple[int, int]] = set()
    for pattern in (_LATIN_CJK_RE, _CJK_LATIN_RE, _LATIN_DIGIT_RE):
        for match in pattern.finditer(content):
            mixed_positions.add((match.start(), match.end()))
    for start, end in sorted(mixed_positions):
        surface = content[start:end]
        if len(surface) <= mixed_script_max_chars:
            add(
                start=start,
                end=end,
                variant="mixed_script",
                mechanism="mixed_script",
                replacement=None,
                rationale="Latin、数字或汉字混写触发形式解码候选。",
            )

    for start, end in _nfkc_changed_spans(content):
        replacement = unicodedata.normalize("NFKC", content[start:end])
        if replacement != content[start:end]:
            add(
                start=start,
                end=end,
                variant="unicode_nfkc",
                mechanism="unicode_nfkc",
                replacement=replacement,
                rationale="兼容字符规范化后发生确定性变化。",
            )

    for match in _EMOJI_RE.finditer(content):
        add(
            start=match.start(),
            end=match.end(),
            variant="emoji",
            mechanism="emoji",
            replacement=None,
            rationale="emoji 序列可能承载需要结合文本解码的表达。",
        )

    for reference in references:
        canonical = str(reference["canonical"])
        variants = list(reference["variants"])
        initials = reference.get("initials")
        if initials and initials not in variants:
            variants.append(str(initials))
        for variant_surface in sorted(set(variants)):
            mechanism = (
                "abbreviation"
                if initials and variant_surface.casefold() == str(initials).casefold()
                else "known_variant"
            )
            if (
                variant_surface.isascii()
                and variant_surface.replace("-", "").replace("_", "").isalnum()
            ):
                pattern = re.compile(
                    rf"(?<![A-Za-z0-9]){re.escape(variant_surface)}(?![A-Za-z0-9])",
                    re.IGNORECASE,
                )
                positions = [
                    (match.start(), match.end()) for match in pattern.finditer(content)
                ]
            else:
                positions = exact_occurrences(content, variant_surface)
            for start, end in positions:
                add(
                    start=start,
                    end=end,
                    variant=(
                        "pinyin_initials"
                        if mechanism == "abbreviation"
                        else "known_variant"
                    ),
                    mechanism=mechanism,
                    replacement=canonical,
                    rationale="冻结形式 reference 命中候选变体。",
                )

        bases = {canonical, *variants}
        for base in sorted(bases):
            if len(base) < 2 or len(base) > 12:
                continue
            pattern = re.compile(
                _SEPARATOR_RE.join(re.escape(char) for char in base),
                re.IGNORECASE,
            )
            for match in pattern.finditer(content):
                if match.group(0).casefold() == base.casefold():
                    continue
                add(
                    start=match.start(),
                    end=match.end(),
                    variant="separator_insertion",
                    mechanism="separator_insertion",
                    replacement=canonical,
                    rationale="移除插入分隔符后与冻结 reference form 一致。",
                )

    if romanizer is not None:
        by_length: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for reference in references:
            if reference["pinyin"]:
                by_length[len(reference["pinyin"])].append(reference)
        for run in _CJK_RUN_RE.finditer(content):
            sequence = run.group(0)
            for width, same_length_references in sorted(by_length.items()):
                if width > len(sequence):
                    continue
                for offset in range(0, len(sequence) - width + 1):
                    surface = sequence[offset : offset + width]
                    raw_pinyin = romanizer(surface)
                    if isinstance(raw_pinyin, str) or not isinstance(raw_pinyin, Sequence):
                        raise CandidateGeneratorError(
                            "romanizer must return a syllable sequence"
                        )
                    pinyin = [str(value).casefold() for value in raw_pinyin]
                    if len(pinyin) != width or any(
                        re.fullmatch(r"[a-z]+", value) is None for value in pinyin
                    ):
                        raise CandidateGeneratorError(
                            "romanizer returned invalid normalized syllables"
                        )
                    for reference in same_length_references:
                        canonical = str(reference["canonical"])
                        known_forms = {canonical, *reference["variants"]}
                        if surface in known_forms:
                            continue
                        if (
                            _syllable_distance(pinyin, reference["pinyin"])
                            <= phonetic_max_distance
                        ):
                            start = run.start() + offset
                            add(
                                start=start,
                                end=start + width,
                                variant="phonetic_reference",
                                mechanism="phonetic_variant",
                                replacement=canonical,
                                rationale="无声调拼音序列与冻结 reference 匹配。",
                            )

    return sorted(
        observations.values(),
        key=lambda row: (
            row["start"],
            row["end"],
            row["surface"],
            row["generator_variant"],
            row["mechanism"],
            row["replacement"] or "",
        ),
    )


def _candidate_identity(candidate: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: candidate[key]
        for key in (
            "record_id",
            "content_sha256",
            "surface",
            "start",
            "end",
            "occurrence_ordinal",
        )
    }


def merge_observations(
    observations: Sequence[Mapping[str, Any]],
    *,
    contents_by_record_id: Mapping[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Union observations by exact occurrence while preserving all provenance."""

    normalized: list[dict[str, Any]] = []
    observation_ids: set[str] = set()
    content_hashes_by_record: dict[str, str] = {}
    for row in observations:
        record_id = str(row.get("record_id", ""))
        content = (
            contents_by_record_id.get(record_id)
            if contents_by_record_id is not None
            else None
        )
        value = validate_observation(row, content=content)
        observation_id = str(value["observation_id"])
        if observation_id in observation_ids:
            raise CandidateGeneratorError("observation id is duplicated")
        observation_ids.add(observation_id)
        prior_hash = content_hashes_by_record.setdefault(
            record_id, str(value["content_sha256"])
        )
        if prior_hash != value["content_sha256"]:
            raise CandidateGeneratorError("one record_id has multiple content hashes")
        normalized.append(value)

    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in normalized:
        key = (
            row["record_id"],
            row["content_sha256"],
            row["start"],
            row["end"],
            row["surface"],
        )
        grouped[key].append(row)

    candidates: list[dict[str, Any]] = []
    for rows in grouped.values():
        first = rows[0]
        ordinals = {int(row["occurrence_ordinal"]) for row in rows}
        if len(ordinals) != 1:
            raise CandidateGeneratorError("merged occurrence ordinals disagree")
        votes = {"true": 0, "false": 0, "unknown": 0}
        for row in rows:
            value = row["requires_context"]
            votes["unknown" if value is None else str(value).lower()] += 1
        candidate = {
            "schema_version": CANDIDATE_SCHEMA_VERSION,
            "candidate_id": "",
            "record_id": first["record_id"],
            "content_sha256": first["content_sha256"],
            "surface": first["surface"],
            "start": first["start"],
            "end": first["end"],
            "occurrence_ordinal": first["occurrence_ordinal"],
            "observation_ids": sorted(row["observation_id"] for row in rows),
            "generators": sorted({row["generator"] for row in rows}),
            "generator_variants": sorted(
                {row["generator_variant"] for row in rows}
            ),
            "mechanisms": sorted({row["mechanism"] for row in rows}),
            "replacement_hypotheses": sorted(
                {str(row["replacement"]) for row in rows if row["replacement"]}
            ),
            "requires_context_votes": votes,
            "review_status": "unreviewed",
        }
        candidate["candidate_id"] = (
            "wp3cand-" + _canonical_sha256(_candidate_identity(candidate))[:32]
        )
        validate_candidate(
            candidate,
            content=(
                contents_by_record_id.get(str(candidate["record_id"]))
                if contents_by_record_id is not None
                else None
            ),
        )
        candidates.append(candidate)
    candidates.sort(
        key=lambda row: (
            row["record_id"],
            row["start"],
            row["end"],
            row["surface"],
        )
    )
    return candidates


def validate_candidate(
    candidate: Mapping[str, Any], *, content: str | None = None
) -> dict[str, Any]:
    expected_fields = {
        "schema_version",
        "candidate_id",
        "record_id",
        "content_sha256",
        "surface",
        "start",
        "end",
        "occurrence_ordinal",
        "observation_ids",
        "generators",
        "generator_variants",
        "mechanisms",
        "replacement_hypotheses",
        "requires_context_votes",
        "review_status",
    }
    if not isinstance(candidate, Mapping) or set(candidate) != expected_fields:
        raise CandidateGeneratorError("candidate fields are not canonical")
    if candidate.get("schema_version") != CANDIDATE_SCHEMA_VERSION:
        raise CandidateGeneratorError("unsupported candidate schema")
    if _forbidden_key_paths(candidate):
        raise CandidateGeneratorError("candidate contains a forbidden task field")
    _require_text(candidate.get("record_id"), "record_id", maximum=256)
    surface = _require_text(candidate.get("surface"), "surface", maximum=80)
    start = _require_int(candidate.get("start"), "start")
    end = _require_int(candidate.get("end"), "end", minimum=1)
    ordinal = _require_int(
        candidate.get("occurrence_ordinal"), "occurrence_ordinal", minimum=1
    )
    if end <= start:
        raise CandidateGeneratorError("candidate end must exceed start")
    content_sha = str(candidate.get("content_sha256", ""))
    if not _SHA256_RE.fullmatch(content_sha):
        raise CandidateGeneratorError("candidate content hash is invalid")
    for field in (
        "observation_ids",
        "generators",
        "generator_variants",
        "mechanisms",
        "replacement_hypotheses",
    ):
        value = candidate.get(field)
        if not isinstance(value, list) or len(value) != len(set(value)):
            raise CandidateGeneratorError(f"candidate {field} must be a unique array")
    if not candidate["observation_ids"] or not candidate["generators"]:
        raise CandidateGeneratorError("candidate must retain observations and generators")
    if any(generator not in GENERATORS for generator in candidate["generators"]):
        raise CandidateGeneratorError("candidate generator is invalid")
    if any(mechanism not in MECHANISMS for mechanism in candidate["mechanisms"]):
        raise CandidateGeneratorError("candidate mechanism is invalid")
    if not candidate["generator_variants"] or not candidate["mechanisms"]:
        raise CandidateGeneratorError("candidate variants/mechanisms cannot be empty")
    for replacement in candidate["replacement_hypotheses"]:
        _require_text(replacement, "replacement_hypothesis", maximum=160)
    votes = candidate.get("requires_context_votes")
    if not isinstance(votes, Mapping) or set(votes) != {"true", "false", "unknown"}:
        raise CandidateGeneratorError("candidate context votes are invalid")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in votes.values()
    ):
        raise CandidateGeneratorError("candidate context vote counts are invalid")
    if sum(votes.values()) != len(candidate["observation_ids"]):
        raise CandidateGeneratorError("candidate context votes do not cover observations")
    if candidate.get("review_status") != "unreviewed":
        raise CandidateGeneratorError("generator candidates must remain unreviewed")
    expected_id = "wp3cand-" + _canonical_sha256(_candidate_identity(candidate))[:32]
    if candidate.get("candidate_id") != expected_id:
        raise CandidateGeneratorError("candidate id is not reproducible")
    if content is not None:
        if _content_sha256(content) != content_sha:
            raise CandidateGeneratorError("candidate content hash differs")
        if content[start:end] != surface:
            raise CandidateGeneratorError("candidate surface differs from content")
        if occurrence_ordinal(content, surface, start, end) != ordinal:
            raise CandidateGeneratorError("candidate occurrence ordinal differs")
    return dict(candidate)


def load_generator_config(
    path: str | Path, *, workspace_root: str | Path | None = None
) -> dict[str, Any]:
    """Load the offline Step 2 config and verify its frozen handbook binding."""

    try:
        with Path(path).open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CandidateGeneratorError(f"cannot load generator config: {exc}") from exc
    expected_fields = {
        "schema_version",
        "resource_role",
        "source_policy",
        "handbook",
        "expected_fit_count",
        "artifact_root",
        "g1",
        "g2",
        "g3",
        "aggregation",
        "pilot",
        "execution",
    }
    if not isinstance(value, dict) or set(value) != expected_fields:
        raise CandidateGeneratorError("generator config fields are not canonical")
    if value.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise CandidateGeneratorError("unsupported generator config schema")
    if value.get("resource_role") != RESOURCE_ROLE:
        raise CandidateGeneratorError("generator resource role differs")
    if value.get("source_policy") != SOURCE_POLICY:
        raise CandidateGeneratorError("generator source policy differs")
    if _forbidden_key_paths(value):
        raise CandidateGeneratorError("generator config contains a forbidden task field")
    if value.get("expected_fit_count") != 5165:
        raise CandidateGeneratorError("generator config must bind the 5,165 fit records")
    artifact_root = value.get("artifact_root")
    if not isinstance(artifact_root, str) or not artifact_root:
        raise CandidateGeneratorError("artifact_root must be non-empty text")
    artifact_path = Path(artifact_root)
    if artifact_path.is_absolute() or ".." in artifact_path.parts:
        raise CandidateGeneratorError("artifact_root must be workspace-relative")

    handbook = value.get("handbook")
    if not isinstance(handbook, Mapping) or set(handbook) != {
        "version",
        "path",
        "sha256",
    }:
        raise CandidateGeneratorError("handbook binding fields are invalid")
    if handbook.get("version") != HANDBOOK_VERSION:
        raise CandidateGeneratorError("handbook version differs")
    if not _SHA256_RE.fullmatch(str(handbook.get("sha256", ""))):
        raise CandidateGeneratorError("handbook hash is invalid")
    handbook_path = Path(str(handbook.get("path", "")))
    if handbook_path.is_absolute() or ".." in handbook_path.parts:
        raise CandidateGeneratorError("handbook path must be workspace-relative")
    if workspace_root is not None:
        resolved = Path(workspace_root) / handbook_path
        if _sha256_file(resolved) != handbook["sha256"]:
            raise CandidateGeneratorError("frozen handbook hash differs")

    if value.get("g1", {}).get("passes") != [
        "surface_decode",
        "lexical_pragmatic",
    ]:
        raise CandidateGeneratorError("G1 passes are not frozen")
    if value.get("g2", {}).get("aggregation") != "provider-union-no-vote/v1":
        raise CandidateGeneratorError("G2 aggregation is not a provider union")
    aggregation = value.get("aggregation")
    if not isinstance(aggregation, Mapping) or any(
        aggregation.get(field) is not expected
        for field, expected in {
            "preserve_nested": True,
            "preserve_overlaps": True,
            "majority_vote": False,
            "automatic_tier_assignment": False,
            "automatic_publication": False,
        }.items()
    ):
        raise CandidateGeneratorError("candidate aggregation safety flags differ")
    execution = value.get("execution")
    if not isinstance(execution, Mapping) or any(
        execution.get(field) is not False
        for field in (
            "model_calls_authorized",
            "network_calls_authorized",
            "paid_calls_authorized",
            "formal_artifact_publication_authorized",
        )
    ):
        raise CandidateGeneratorError("this startup config must remain offline")
    return value
