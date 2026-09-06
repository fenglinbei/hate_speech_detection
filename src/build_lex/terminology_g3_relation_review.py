"""Offline v2 public-source to G3 form-relation review-frame bridge.

This module never fetches a URL and never makes a form decision.  It consumes
only a separately validated ``wp3-g3-public-source-bundle/v2`` artifact.  Pair
seeds are location hints: their old quotes/evidence identifiers are discarded
and every emitted proposal is relocated in the frozen v2 text.  CHIME rows are
read only from the bundle's filtered 185-row candidate projection and require
one explicit, contiguous form-relation clause before entering human review.
"""

from __future__ import annotations

import copy
import hashlib
import re
import shutil
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from build_lex.terminology_g3_form_reference import (
    ACTIONS,
    FAMILIES,
    FRAME_ARTIFACT_KIND,
    FRAME_ID_PREFIX,
    G3FormReferenceError,
    REFERENCE_ROLE,
    SCOPE,
    SOURCE_BUNDLE_ARTIFACT_KIND,
    _array_of_objects,
    _dependency,
    _forbidden_key_paths,
    _load_object,
    _object,
    _occurrence_span,
    _optional_text,
    _target_name_matches,
    _trimmed_text,
    _validate_schema,
)
from data.training_artifacts import (
    TrainingArtifactError,
    canonical_sha256,
    ensure_exact_file_set,
    finalize_target_atomic,
    load_json,
    new_staging_directory,
    sha256_file,
    validate_payload_manifest,
    write_canonical_json,
    write_locator_ref,
)


CLAIMS_SCHEMA_VERSION = "wp3-g3-form-relation-claims/v2"
SEEDS_SCHEMA_VERSION = "wp3-g3-form-relation-seeds/v2"
FRAME_SCHEMA_VERSION = "wp3-g3-form-review-frame/v2"
REPORT_SCHEMA_VERSION = "wp3-g3-form-relation-extraction-report/v2"
EXTRACTOR_ID = "offline-explicit-form-relations/v2"
SEED_ROLE = "pair-location-hints-only-not-evidence"
FRAME_REVIEW_POLICY = "explicit-human-confirmation-no-auto-accept/v2"
CLAIMS_SCHEMA_PATH = "schemas/wp3_g3_form_relation_claims_v2.schema.json"
SEEDS_SCHEMA_PATH = "schemas/wp3_g3_form_relation_seeds_v2.schema.json"

ALLOWED_SOURCE_ROLES = frozenset({"direct_evidence", "candidate_pool"})
NON_RELATION_SOURCE_ROLES = frozenset(
    {
        "prevalence_or_taxonomy",
        "method_or_prevalence",
        "method_only",
        "mechanism_background",
    }
)
CHIME_PROJECTION_SCHEMA_VERSION = "wp3-g3-chime-form-candidate-projection/v1"
CHIME_ALLOWED_TYPES = ("abbreviation", "homophonic pun")
CHIME_EXPECTED_COUNTS = {"abbreviation": 52, "homophonic pun": 133}
CHIME_EXPECTED_CANDIDATE_COUNT = 185
MAX_RELATION_QUOTE_CHARS = 1000
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

# Keys prohibited from crossing the relation-frame boundary.  ``meaning`` and
# ``origin`` are intentionally absent from the frame even though the frozen
# CHIME projection is consulted internally to locate a minimal relation clause.
FRAME_FORBIDDEN_KEYS = frozenset(
    {
        "definition",
        "definitions",
        "meaning",
        "origin",
        "examples",
        "profanity",
        "offense",
        "label",
        "labels",
        "abc",
        "a_candidate",
        "b_candidate",
        "c_candidate",
        "r",
        "verdict",
        "record_id",
        "case_id",
        "task",
        "task_label",
        "gold",
        "fit",
        "legacy_lexicon",
    }
)

_CLAUSE_BOUNDARY_RE = re.compile(r"[。！？!?；;\n\r]")
_RELATION_MARKER_RE = re.compile(
    r"拼音|首字母|缩写|简称|简写|谐音|近音|同音|音译|变体|异体|"
    r"书写|写作|替代|拆字|听起来|正确的词|代表|表示|意思是|\bor\b",
    re.IGNORECASE,
)

_CHIME_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"[“\"](?P<surface>[^”\"\n]{1,80})[”\"]"
        r"(?:是|为|即|就是)"
        r"[“\"](?P<canonical>[^”\"\n]{1,80})[”\"]"
        r"(?:（[^）\n]{0,80}）)?的"
        r"(?P<marker>拼音(?:首字母)?缩写|首字母缩写|缩写|简称|简写|"
        r"谐音(?:网络用语|梗)?|近音|同音|音译|书写变体|异体)",
        re.IGNORECASE,
    ),
    re.compile(
        r"[“\"](?P<surface>[^”\"\n]{1,80})[”\"]"
        r"(?:是|为|即|就是)"
        r"[“\"](?P<canonical>[^”\"\n]{1,80})[”\"]"
        r"(?:的)?(?P<marker>谐音网络用语|谐音梗|谐音|近音|同音|音译)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?P<surface>[A-Za-z0-9\u3400-\u9fff]{1,40})"
        r"(?:是|为|即|就是)"
        r"[“\"](?P<canonical>[^”\"\n]{1,80})[”\"]"
        r"的(?P<marker>拼音(?:首字母)?缩写|首字母缩写|缩写|简称|简写|"
        r"谐音(?:网络用语|梗)?|近音|同音|音译)",
        re.IGNORECASE,
    ),
    re.compile(
        r"[“\"](?P<surface>[^”\"\n]{1,80})[”\"]"
        r"(?:是|为|即|就是)"
        r"(?P<canonical>[A-Za-z0-9\u3400-\u9fff ]{1,80})"
        r"的(?P<marker>拼音(?:首字母)?缩写|首字母缩写|缩写|简称|简写|"
        r"谐音(?:网络用语|梗)?|近音|同音|音译)",
        re.IGNORECASE,
    ),
)


def _validate_source_bundle_v2(
    source_bundle_dir: str | Path,
    *,
    workspace_root: str | Path,
    require_current_implementation: bool,
) -> dict[str, Any]:
    try:
        from build_lex.terminology_g3_source_catalog_v2 import (
            G3SourceCatalogV2Error,
            validate_public_source_bundle_v2,
        )
    except ImportError as exc:
        raise G3FormReferenceError(
            "the v2 public-source bundle validator is unavailable"
        ) from exc
    try:
        result = validate_public_source_bundle_v2(
            source_bundle_dir,
            workspace_root=workspace_root,
            require_current_implementation=require_current_implementation,
        )
    except G3SourceCatalogV2Error as exc:
        raise G3FormReferenceError(str(exc)) from exc
    if not isinstance(result, Mapping):
        raise G3FormReferenceError("v2 public-source validator returned no binding")
    return dict(result)


def _forbidden_frame_key_paths(value: Any, path: tuple[str, ...] = ()) -> list[str]:
    found: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            child_path = (*path, str(key))
            if str(key).casefold() in FRAME_FORBIDDEN_KEYS:
                found.append(".".join(child_path))
            found.extend(_forbidden_frame_key_paths(child, child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(_forbidden_frame_key_paths(child, (*path, str(index))))
    return found


def _component_id(row: Mapping[str, Any]) -> str:
    value = _trimmed_text(row.get("component_id"), "component_id", maximum=160)
    if not re.fullmatch(r"[a-z0-9][a-z0-9._-]{0,159}", value):
        raise G3FormReferenceError("component_id is not canonical")
    return value


def _component_role(row: Mapping[str, Any]) -> str:
    role = _trimmed_text(row.get("source_role"), "source_role", maximum=80)
    if role not in ALLOWED_SOURCE_ROLES | NON_RELATION_SOURCE_ROLES:
        raise G3FormReferenceError(f"unknown v2 source_role: {role}")
    return role


def _safe_bundle_file(bundle: Path, logical_path: str, label: str) -> Path:
    relative = Path(logical_path)
    if (
        relative.is_absolute()
        or ".." in relative.parts
        or logical_path != relative.as_posix()
    ):
        raise G3FormReferenceError(f"{label} path is not portable")
    target = (bundle / relative).resolve()
    try:
        target.relative_to(bundle.resolve())
    except ValueError as exc:
        raise G3FormReferenceError(f"{label} escapes the source bundle") from exc
    if not target.is_file() or target.is_symlink():
        raise G3FormReferenceError(f"{label} is not a regular file")
    return target


def _normalized_component_text(
    source: Mapping[str, Any], component: Mapping[str, Any]
) -> tuple[str, str, str] | None:
    logical = component.get("normalized_file")
    digest = component.get("normalized_sha256")
    if logical is None and digest is None:
        return None
    if not isinstance(logical, str) or not SHA256_RE.fullmatch(str(digest or "")):
        raise G3FormReferenceError("normalized component binding is incomplete")
    target = _safe_bundle_file(Path(str(source["target"])), logical, "normalized_file")
    if sha256_file(target) != digest:
        raise G3FormReferenceError("normalized component hash differs")
    try:
        text = target.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise G3FormReferenceError("normalized component text is not UTF-8") from exc
    if not text:
        raise G3FormReferenceError("normalized component text is empty")
    return text, logical, str(digest)


def _candidate_projection(
    source: Mapping[str, Any], component: Mapping[str, Any]
) -> tuple[dict[str, Any], str, str] | None:
    metadata = component.get("candidate_projection")
    if metadata is None:
        return None
    projection = _object(metadata, "candidate_projection")
    if set(projection) != {
        "schema_version",
        "file",
        "size_bytes",
        "sha256",
        "count",
        "allowed_types",
    }:
        raise G3FormReferenceError("candidate_projection fields are not canonical")
    logical = str(projection.get("file", ""))
    digest = str(projection.get("sha256", ""))
    count = projection.get("count")
    if (
        not SHA256_RE.fullmatch(digest)
        or isinstance(count, bool)
        or not isinstance(count, int)
        or count < 1
        or projection.get("schema_version")
        != CHIME_PROJECTION_SCHEMA_VERSION
        or isinstance(projection.get("size_bytes"), bool)
        or not isinstance(projection.get("size_bytes"), int)
        or projection.get("size_bytes") <= 0
        or projection.get("allowed_types") != list(CHIME_ALLOWED_TYPES)
    ):
        raise G3FormReferenceError("candidate_projection binding is invalid")
    target = _safe_bundle_file(
        Path(str(source["target"])), logical, "candidate_projection"
    )
    if target.stat().st_size != projection["size_bytes"] or sha256_file(target) != digest:
        raise G3FormReferenceError("candidate_projection hash differs")
    value = _load_object(target, "candidate_projection")
    if value.get("candidate_count") != count:
        raise G3FormReferenceError("candidate_projection count differs")
    return value, logical, digest


def _component_provenance(component: Mapping[str, Any]) -> dict[str, Any]:
    source_id = _trimmed_text(component.get("source_id"), "source_id", maximum=160)
    acquisition_mode = _trimmed_text(
        component.get("acquisition_mode"), "acquisition_mode", maximum=80
    )
    publisher = component.get("publisher")
    if not isinstance(publisher, str) or not publisher.strip():
        publisher = source_id
    requested_url = component.get("requested_url")
    final_url = component.get("final_url")
    snapshot_url = component.get("snapshot_url")
    for label, value in (
        ("requested_url", requested_url),
        ("final_url", final_url),
        ("snapshot_url", snapshot_url),
    ):
        if value is not None and not isinstance(value, str):
            raise G3FormReferenceError(f"component {label} must be text or null")
    return {
        "component_id": _component_id(component),
        "source_id": source_id,
        "source_role": _component_role(component),
        "acquisition_mode": acquisition_mode,
        "publisher": publisher.strip(),
        "requested_url": requested_url,
        "final_url": final_url,
        "snapshot_url": snapshot_url,
    }


def _component_relation_eligible(component: Mapping[str, Any]) -> bool:
    role = _component_role(component)
    eligible = component.get("evidence_eligible")
    review_required = component.get("human_review_required")
    if not isinstance(eligible, bool) or not isinstance(review_required, bool):
        raise G3FormReferenceError("component relation-review gates are missing")
    expected = role in ALLOWED_SOURCE_ROLES
    if eligible is not expected or review_required is not expected:
        raise G3FormReferenceError("component relation-review gates differ from source_role")
    return expected


def _occurrence_ordinal(content: str, quote: str, start: int) -> int:
    cursor = 0
    ordinal = 0
    while True:
        found = content.find(quote, cursor)
        if found < 0:
            raise G3FormReferenceError("relation quote no longer replays")
        ordinal += 1
        if found == start:
            return ordinal
        if found > start:
            raise G3FormReferenceError("relation quote start is not an occurrence")
        cursor = found + 1


def _family_from_marker(marker: str, source_type: str | None = None) -> str:
    folded = marker.casefold()
    if any(token in folded for token in ("书写", "异体", "变体", "替代", "拆字")):
        return "orthographic_variant"
    if any(
        token in folded
        for token in (
            "拼音",
            "首字母",
            "缩写",
            "简称",
            "简写",
            "谐音",
            "近音",
            "同音",
            "音译",
        )
    ):
        return "phonetic_variant"
    if source_type in CHIME_ALLOWED_TYPES:
        return "phonetic_variant"
    return "known_variant"


def _relation_clause_for_pair(
    content: str, surface: str, canonical: str
) -> tuple[str, int, int, str, str] | None:
    """Return the shortest exact frozen clause supporting one pair seed."""

    # Prefer byte-for-byte spelling from the pair seed.  Falling back to a
    # case-insensitive lookup is useful for ordinary alphabetic evidence, but
    # it must never collapse a case-only relation (for example ``mp3`` ->
    # ``MP3``) onto the same occurrence.
    surface_matches = list(re.finditer(re.escape(surface), content)) or list(
        re.finditer(re.escape(surface), content, re.IGNORECASE)
    )
    canonical_matches = list(re.finditer(re.escape(canonical), content)) or list(
        re.finditer(re.escape(canonical), content, re.IGNORECASE)
    )
    candidates: list[tuple[int, int, str, int, int, str, str]] = []
    for surface_match in surface_matches:
        for canonical_match in canonical_matches:
            if surface_match.span() == canonical_match.span():
                continue
            low = min(surface_match.start(), canonical_match.start())
            high = max(surface_match.end(), canonical_match.end())
            if high - low > MAX_RELATION_QUOTE_CHARS:
                continue
            left_matches = list(_CLAUSE_BOUNDARY_RE.finditer(content, 0, low))
            left = left_matches[-1].end() if left_matches else 0
            right_match = _CLAUSE_BOUNDARY_RE.search(content, high)
            right = right_match.start() if right_match else len(content)
            while left < right and content[left].isspace():
                left += 1
            while right > left and content[right - 1].isspace():
                right -= 1
            quote = content[left:right]
            if not quote or len(quote) > MAX_RELATION_QUOTE_CHARS:
                window_left = max(0, low - 160)
                window_right = min(len(content), high + 160)
                while window_left < low and content[window_left].isspace():
                    window_left += 1
                while window_right > high and content[window_right - 1].isspace():
                    window_right -= 1
                left, right = window_left, window_right
                quote = content[left:right]
            actual_surface = content[surface_match.start() : surface_match.end()]
            actual_canonical = content[canonical_match.start() : canonical_match.end()]
            if actual_surface not in quote or actual_canonical not in quote:
                continue
            compact_parenthetical = bool(
                re.search(
                    re.escape(actual_surface)
                    + r"\s*[（(]\s*"
                    + re.escape(actual_canonical)
                    + r"\s*[）)]",
                    quote,
                    re.IGNORECASE,
                )
                or re.search(
                    re.escape(actual_canonical)
                    + r"\s*[（(]\s*"
                    + re.escape(actual_surface)
                    + r"\s*[）)]",
                    quote,
                    re.IGNORECASE,
                )
            )
            if not _RELATION_MARKER_RE.search(quote) and not compact_parenthetical:
                continue
            candidates.append(
                (
                    len(quote),
                    left,
                    quote,
                    surface_match.start(),
                    canonical_match.start(),
                    actual_surface,
                    actual_canonical,
                )
            )
    if not candidates:
        return None
    _, start, quote, _, _, actual_surface, actual_canonical = min(
        candidates, key=lambda row: (row[0], row[1], row[2], row[3], row[4])
    )
    return quote, start, start + len(quote), actual_surface, actual_canonical


def pair_seeds_from_v1_extraction(path: str | Path) -> list[dict[str, Any]]:
    """Project v1 items to pair-only seeds, intentionally discarding old evidence."""

    value = _load_object(path, "v1 form extraction")
    if value.get("schema_version") != "wp3-g3-form-extraction/v1":
        raise G3FormReferenceError("pair seed source is not a v1 form extraction")
    evidence = _array_of_objects(value.get("evidence"), "v1 extraction evidence")
    evidence_source: dict[str, str] = {}
    for row in evidence:
        evidence_id = str(row.get("evidence_id", ""))
        source_id = str(row.get("source_id", ""))
        if not evidence_id or not source_id or evidence_id in evidence_source:
            raise G3FormReferenceError("v1 evidence identity is invalid")
        evidence_source[evidence_id] = source_id
    seeds: list[dict[str, Any]] = []
    for row in _array_of_objects(value.get("items"), "v1 extraction items"):
        evidence_ids = row.get("evidence_ids")
        if not isinstance(evidence_ids, list) or not evidence_ids:
            raise G3FormReferenceError("v1 pair seed has no source evidence")
        source_ids = {evidence_source.get(str(identifier)) for identifier in evidence_ids}
        if None in source_ids or len(source_ids) != 1:
            raise G3FormReferenceError("v1 pair seed source is ambiguous")
        item_id = _trimmed_text(row.get("item_id"), "v1 item_id", maximum=160)
        suffix = re.sub(r"[^a-z0-9-]+", "-", item_id.casefold()).strip("-")
        seeds.append(
            {
                "seed_id": f"g3seed-v1-{suffix}",
                "source_id": next(iter(source_ids)),
                "surface": _trimmed_text(row.get("surface"), "seed surface", maximum=160),
                "canonical": _trimmed_text(
                    row.get("canonical"), "seed canonical", maximum=160
                ),
                "proposed_family": row.get("proposed_family"),
                "phonetic_scan_enabled": row.get("phonetic_scan_enabled"),
            }
        )
    return seeds


def load_pair_seeds(
    seed_paths: Sequence[str | Path], *, workspace_root: str | Path
) -> list[dict[str, Any]]:
    root = Path(workspace_root).resolve()
    seeds: list[dict[str, Any]] = []
    for seed_path in seed_paths:
        path = Path(seed_path).resolve()
        value = _load_object(path, "form relation seed document")
        if value.get("schema_version") == "wp3-g3-form-extraction/v1":
            seeds.extend(pair_seeds_from_v1_extraction(path))
            continue
        _validate_schema(value, root / SEEDS_SCHEMA_PATH)
        if (
            value.get("schema_version") != SEEDS_SCHEMA_VERSION
            or value.get("seed_role") != SEED_ROLE
        ):
            raise G3FormReferenceError("form relation seed contract differs")
        seeds.extend(_array_of_objects(value.get("seeds"), "form relation seeds"))
    ids: set[str] = set()
    normalized: list[dict[str, Any]] = []
    for row in seeds:
        if set(row) != {
            "seed_id",
            "source_id",
            "surface",
            "canonical",
            "proposed_family",
            "phonetic_scan_enabled",
        }:
            raise G3FormReferenceError("form relation seed fields are not canonical")
        seed_id = _trimmed_text(row.get("seed_id"), "seed_id", maximum=180)
        if not re.fullmatch(r"g3seed-[a-z0-9-]+", seed_id) or seed_id in ids:
            raise G3FormReferenceError("form relation seed ID is invalid or duplicated")
        ids.add(seed_id)
        surface = _trimmed_text(row.get("surface"), "seed surface", maximum=160)
        canonical = _trimmed_text(row.get("canonical"), "seed canonical", maximum=160)
        if surface == canonical or row.get("proposed_family") not in FAMILIES:
            raise G3FormReferenceError("form relation seed pair is invalid")
        if not isinstance(row.get("phonetic_scan_enabled"), bool):
            raise G3FormReferenceError("seed phonetic_scan_enabled must be boolean")
        normalized.append(copy.deepcopy(row))
    return sorted(normalized, key=lambda row: row["seed_id"])


def _chime_row_text(row: Mapping[str, Any]) -> str:
    parts = [str(row["meme"]), str(row["meaning"])]
    if isinstance(row.get("origin"), str):
        parts.append(str(row["origin"]))
    return "\n".join(parts)


def _validate_chime_projection(
    projection: Mapping[str, Any], component_id: str
) -> list[dict[str, Any]]:
    if (
        projection.get("schema_version") != CHIME_PROJECTION_SCHEMA_VERSION
        or projection.get("source_component_id") != component_id
        or projection.get("source_record_count") != 1458
        or projection.get("allowed_types") != list(CHIME_ALLOWED_TYPES)
        or projection.get("candidate_count") != CHIME_EXPECTED_CANDIDATE_COUNT
    ):
        raise G3FormReferenceError("CHIME candidate projection contract differs")
    rows = _array_of_objects(projection.get("rows"), "CHIME candidate rows")
    if len(rows) != CHIME_EXPECTED_CANDIDATE_COUNT:
        raise G3FormReferenceError("CHIME candidate projection is not 185 rows")
    ordinals: set[int] = set()
    counts: Counter[str] = Counter()
    expected_fields = {
        "source_row_ordinal",
        "meme",
        "meaning",
        "origin",
        "type_cn",
        "type_en",
    }
    for row in rows:
        if set(row) != expected_fields:
            raise G3FormReferenceError("CHIME candidate row fields differ")
        ordinal = row.get("source_row_ordinal")
        if (
            isinstance(ordinal, bool)
            or not isinstance(ordinal, int)
            or ordinal < 1
            or ordinal > 1458
            or ordinal in ordinals
        ):
            raise G3FormReferenceError("CHIME source row ordinal is invalid")
        ordinals.add(ordinal)
        source_type = row.get("type_en")
        if source_type not in CHIME_ALLOWED_TYPES:
            raise G3FormReferenceError("CHIME candidate type escaped the allowlist")
        counts[str(source_type)] += 1
        _trimmed_text(row.get("meme"), "CHIME meme", maximum=160)
        _trimmed_text(row.get("meaning"), "CHIME relation text", maximum=10000)
        if row.get("origin") is not None:
            _trimmed_text(row.get("origin"), "CHIME origin text", maximum=10000)
        _trimmed_text(row.get("type_cn"), "CHIME type_cn", maximum=80)
    if dict(counts) != CHIME_EXPECTED_COUNTS:
        raise G3FormReferenceError("CHIME 52/133 type counts differ")
    return sorted(rows, key=lambda row: int(row["source_row_ordinal"]))


def _extract_chime_claims(
    *, component: Mapping[str, Any], projection: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    component_id = _component_id(component)
    rows = _validate_chime_projection(projection, component_id)
    claims: list[dict[str, Any]] = []
    dispositions: list[dict[str, Any]] = []
    for row in rows:
        row_claims: list[dict[str, Any]] = []
        meme = str(row["meme"])
        row_text = _chime_row_text(row)
        for field_value in (row["meaning"], row.get("origin")):
            if not isinstance(field_value, str):
                continue
            for pattern in _CHIME_PATTERNS:
                for match in pattern.finditer(field_value):
                    surface = match.group("surface").strip()
                    canonical = match.group("canonical").strip()
                    marker = match.group("marker").strip()
                    if (
                        not surface
                        or not canonical
                        or surface == canonical
                        or (
                            surface != meme
                            and surface not in meme
                            and meme not in surface
                        )
                        or canonical.startswith(
                            ("一种", "一个", "用于", "指", "来自", "源自")
                        )
                    ):
                        continue
                    quote = match.group(0).strip()
                    if surface not in quote or canonical not in quote:
                        continue
                    start = row_text.find(quote)
                    if start < 0:
                        raise G3FormReferenceError(
                            "CHIME relation clause does not replay from its filtered row"
                        )
                    claim = {
                        "component_id": component_id,
                        "surface": surface,
                        "canonical": canonical,
                        "proposed_family": _family_from_marker(
                            marker, str(row["type_en"])
                        ),
                        "phonetic_scan_enabled": False,
                        "quote": quote,
                        "occurrence_ordinal": _occurrence_ordinal(
                            row_text, quote, start
                        ),
                        "relation_note": (
                            "CHIME 候选池冻结行逐字给出形式关系；仍须人工决定。"
                        ),
                        "source_row_ordinal": int(row["source_row_ordinal"]),
                        "_source_text": row_text,
                        "_source_text_kind": "candidate_projection_row",
                        "_source_text_sha256": hashlib.sha256(
                            row_text.encode("utf-8")
                        ).hexdigest(),
                        "_proposal_origin": "chime-explicit-relation-regex/v1",
                    }
                    row_claims.append(claim)
        unique: dict[tuple[str, str, str, str], dict[str, Any]] = {}
        for claim in row_claims:
            key = (
                claim["surface"],
                claim["canonical"],
                claim["proposed_family"],
                claim["quote"],
            )
            unique[key] = claim
        claims.extend(unique[key] for key in sorted(unique))
        dispositions.append(
            {
                "component_id": component_id,
                "source_row_ordinal": int(row["source_row_ordinal"]),
                "disposition": (
                    "relation_items_emitted"
                    if unique
                    else "no_explicit_form_relation"
                ),
                "emitted_count": len(unique),
            }
        )
    return claims, dispositions


def _extract_seed_claims(
    *,
    source: Mapping[str, Any],
    components: Sequence[Mapping[str, Any]],
    seeds: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    components_by_source: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for component in components:
        components_by_source[str(component.get("source_id", ""))].append(component)
    claims: list[dict[str, Any]] = []
    dispositions: list[dict[str, Any]] = []
    for seed in seeds:
        emitted = 0
        candidates = sorted(
            components_by_source.get(str(seed["source_id"]), []),
            key=_component_id,
        )
        reason = "seed_source_unavailable"
        for component in candidates:
            component_id = _component_id(component)
            role = _component_role(component)
            if not _component_relation_eligible(component):
                reason = "source_role_not_relation_evidence"
                continue
            normalized = _normalized_component_text(source, component)
            if normalized is None:
                reason = "no_replayable_text"
                continue
            content, _logical, digest = normalized
            located = _relation_clause_for_pair(
                content, str(seed["surface"]), str(seed["canonical"])
            )
            if located is None:
                reason = "no_explicit_form_relation"
                continue
            quote, start, _end, actual_surface, actual_canonical = located
            claims.append(
                {
                    "component_id": component_id,
                    "surface": actual_surface,
                    "canonical": actual_canonical,
                    "proposed_family": seed["proposed_family"],
                    "phonetic_scan_enabled": seed["phonetic_scan_enabled"],
                    "quote": quote,
                    "occurrence_ordinal": _occurrence_ordinal(content, quote, start),
                    "relation_note": (
                        "pair seed 仅用于定位；该证据已从冻结 v2 正文重新提取。"
                    ),
                    "source_row_ordinal": None,
                    "_source_text": content,
                    "_source_text_kind": "normalized_component_text",
                    "_source_text_sha256": digest,
                    "_proposal_origin": "pair-seed-relocation/v2",
                }
            )
            emitted += 1
        dispositions.append(
            {
                "seed_id": seed["seed_id"],
                "source_id": seed["source_id"],
                "disposition": "relation_items_emitted" if emitted else reason,
                "emitted_count": emitted,
            }
        )
    return claims, dispositions


def _claim_public_projection(claim: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: copy.deepcopy(claim[key])
        for key in (
            "component_id",
            "surface",
            "canonical",
            "proposed_family",
            "phonetic_scan_enabled",
            "quote",
            "occurrence_ordinal",
            "relation_note",
            "source_row_ordinal",
        )
    }


def _build_evidence_and_items(
    claims: Sequence[Mapping[str, Any]],
    components_by_id: Mapping[str, Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    evidence: list[dict[str, Any]] = []
    grouped: dict[tuple[str, str, str, bool, str], list[str]] = defaultdict(list)
    seen_claims: set[str] = set()
    for claim in claims:
        public_claim = _claim_public_projection(claim)
        claim_sha = canonical_sha256(public_claim)
        if claim_sha in seen_claims:
            continue
        seen_claims.add(claim_sha)
        component_id = str(claim["component_id"])
        component = components_by_id[component_id]
        provenance = _component_provenance(component)
        quote = str(claim["quote"])
        source_text = str(claim["_source_text"])
        start, end = _occurrence_span(
            source_text, quote, int(claim["occurrence_ordinal"])
        )
        if (
            str(claim["surface"]) not in quote
            or str(claim["canonical"]) not in quote
        ):
            raise G3FormReferenceError(
                "one relation quote must contain both surface and canonical"
            )
        evidence_id = "g3ev-v2-" + claim_sha[:40]
        evidence.append(
            {
                "evidence_id": evidence_id,
                **provenance,
                "quote": quote,
                "occurrence_ordinal": int(claim["occurrence_ordinal"]),
                "snapshot_start": start,
                "snapshot_end": end,
                "evidence_text_sha256": hashlib.sha256(
                    quote.encode("utf-8")
                ).hexdigest(),
                "source_text_kind": claim["_source_text_kind"],
                "source_text_sha256": claim["_source_text_sha256"],
                "source_row_ordinal": claim["source_row_ordinal"],
                "relation_note": claim["relation_note"],
                "proposal_origin": claim["_proposal_origin"],
                "relation_contract": "single-quote-surface-and-canonical/v2",
            }
        )
        key = (
            str(claim["surface"]),
            str(claim["canonical"]),
            str(claim["proposed_family"]),
            bool(claim["phonetic_scan_enabled"]),
            component_id,
        )
        grouped[key].append(evidence_id)
    items: list[dict[str, Any]] = []
    for (surface, canonical, family, scan, component_id), evidence_ids in sorted(
        grouped.items()
    ):
        identity = {
            "surface": surface,
            "canonical": canonical,
            "proposed_family": family,
            "phonetic_scan_enabled": scan,
            "component_id": component_id,
            "evidence_ids": sorted(set(evidence_ids)),
        }
        items.append(
            {
                "item_id": "g3form-v2-" + canonical_sha256(identity)[:40],
                "surface": surface,
                "canonical": canonical,
                "proposed_family": family,
                "phonetic_scan_enabled": scan,
                "evidence_ids": identity["evidence_ids"],
            }
        )
    evidence.sort(key=lambda row: row["evidence_id"])
    items.sort(key=lambda row: row["item_id"])
    return evidence, items


def extract_form_relation_claims_v2(
    *,
    source_bundle_dir: str | Path,
    workspace_root: str | Path,
    seed_paths: Sequence[str | Path] = (),
    require_current_implementation: bool = True,
) -> dict[str, Any]:
    """Produce review proposals and dispositions without making decisions."""

    source = _validate_source_bundle_v2(
        source_bundle_dir,
        workspace_root=workspace_root,
        require_current_implementation=require_current_implementation,
    )
    components = _array_of_objects(source.get("components"), "v2 source components")
    components_by_id = {_component_id(row): row for row in components}
    if len(components_by_id) != len(components):
        raise G3FormReferenceError("v2 source component IDs are duplicated")
    for row in components:
        _component_provenance(row)
    seeds = load_pair_seeds(seed_paths, workspace_root=workspace_root)
    claims, seed_dispositions = _extract_seed_claims(
        source=source, components=components, seeds=seeds
    )
    chime_dispositions: list[dict[str, Any]] = []
    chime_component_count = 0
    for component in sorted(components, key=_component_id):
        projection_result = _candidate_projection(source, component)
        if projection_result is None:
            continue
        projection, _logical, _digest = projection_result
        if projection.get("schema_version") != CHIME_PROJECTION_SCHEMA_VERSION:
            raise G3FormReferenceError("unsupported candidate projection schema")
        chime_component_count += 1
        if (
            _component_role(component) != "candidate_pool"
            or not _component_relation_eligible(component)
        ):
            raise G3FormReferenceError("CHIME projection must remain candidate_pool")
        extracted, dispositions = _extract_chime_claims(
            component=component, projection=projection
        )
        claims.extend(extracted)
        chime_dispositions.extend(dispositions)
    evidence, items = _build_evidence_and_items(claims, components_by_id)
    role_counts = Counter(_component_role(row) for row in components)
    disposition_counts = Counter(
        row["disposition"] for row in [*seed_dispositions, *chime_dispositions]
    )
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "extractor": EXTRACTOR_ID,
        "source_bundle_id": source["source_bundle_id"],
        "seed_count": len(seeds),
        "chime_component_count": chime_component_count,
        "chime_candidate_row_count": len(chime_dispositions),
        "claim_count_before_deduplication": len(claims),
        "evidence_count": len(evidence),
        "item_count": len(items),
        "source_role_counts": dict(sorted(role_counts.items())),
        "disposition_counts": dict(sorted(disposition_counts.items())),
        "seed_dispositions": seed_dispositions,
        "chime_row_dispositions": chime_dispositions,
        "guarantees": {
            "offline_only": True,
            "no_human_decisions_made": True,
            "old_evidence_not_reused": True,
            "surface_and_canonical_share_each_quote": True,
            "prevalence_or_method_cannot_emit_alone": True,
            "chime_only_52_abbreviation_plus_133_homophonic_rows": True,
            "forbidden_task_gold_definition_fields_not_carried": True,
        },
    }
    forbidden = _forbidden_frame_key_paths(
        {"items": items, "evidence": evidence, "report": report}
    )
    if forbidden:
        raise G3FormReferenceError(
            "v2 relation frame projection contains forbidden keys: "
            + ", ".join(forbidden)
        )
    claims_document = {
        "schema_version": CLAIMS_SCHEMA_VERSION,
        "extractor": EXTRACTOR_ID,
        "claims": [_claim_public_projection(row) for row in claims],
    }
    _validate_schema(
        claims_document, Path(workspace_root).resolve() / CLAIMS_SCHEMA_PATH
    )
    return {
        "source": source,
        "claims": claims_document,
        "items": items,
        "evidence": evidence,
        "report": report,
    }


def build_form_relation_review_frame_v2(
    *,
    source_bundle_dir: str | Path,
    workspace_root: str | Path,
    output_root: str | Path,
    seed_paths: Sequence[str | Path] = (),
    write_ref: str | Path | None = None,
) -> dict[str, Any]:
    extracted = extract_form_relation_claims_v2(
        source_bundle_dir=source_bundle_dir,
        workspace_root=workspace_root,
        seed_paths=seed_paths,
        require_current_implementation=True,
    )
    source = extracted["source"]
    items = extracted["items"]
    evidence = extracted["evidence"]
    report = extracted["report"]
    if not items:
        raise G3FormReferenceError(
            "v2 form extraction produced no replayable relation review items"
        )
    identity = {
        "schema_version": FRAME_SCHEMA_VERSION,
        "artifact_kind": FRAME_ARTIFACT_KIND,
        "source_bundle_dependency": _dependency(
            source, SOURCE_BUNDLE_ARTIFACT_KIND, "source_bundle_id"
        ),
        "claims_schema_sha256": sha256_file(
            Path(workspace_root).resolve() / CLAIMS_SCHEMA_PATH
        ),
        "claims_sha256": canonical_sha256(extracted["claims"]),
        "items_sha256": canonical_sha256(items),
        "evidence_sha256": canonical_sha256(evidence),
        "extraction_report_sha256": canonical_sha256(report),
        "item_count": len(items),
        "evidence_count": len(evidence),
        "decision_actions": sorted(ACTIONS),
        "variant_families": sorted(FAMILIES),
        "reference_role": REFERENCE_ROLE,
        "scope": SCOPE,
        "scientific_eligible": False,
        "sealed": False,
        "review_policy": FRAME_REVIEW_POLICY,
        "builder_implementation_sha256": sha256_file(Path(__file__)),
    }
    frame_id = FRAME_ID_PREFIX + canonical_sha256(identity)
    manifest = {**identity, "frame_id": frame_id}
    output_parent = Path(output_root).resolve()
    target = output_parent / frame_id
    if not target.exists():
        staging = new_staging_directory(output_parent, frame_id)
        try:
            write_canonical_json(staging / "manifest.json", manifest)
            write_canonical_json(staging / "claims.json", extracted["claims"])
            write_canonical_json(staging / "items.json", items)
            write_canonical_json(staging / "evidence.json", evidence)
            write_canonical_json(staging / "extraction_report.json", report)
            payload_hash = finalize_target_atomic(
                staging,
                target,
                validate_staging=lambda staged: validate_form_relation_review_frame_v2(
                    staged,
                    source_bundle_dir=source_bundle_dir,
                    workspace_root=workspace_root,
                    require_current_implementation=True,
                ),
            )
        finally:
            if staging.exists():
                shutil.rmtree(staging)
    else:
        validated = validate_form_relation_review_frame_v2(
            target,
            source_bundle_dir=source_bundle_dir,
            workspace_root=workspace_root,
            require_current_implementation=True,
        )
        payload_hash = validated["payload_manifest_sha256"]
    if write_ref is not None:
        write_locator_ref(
            write_ref,
            artifact_kind=FRAME_ARTIFACT_KIND,
            artifact_id=frame_id,
            target=target,
            payload_manifest_sha256=payload_hash,
        )
    return {
        "frame_id": frame_id,
        "target": str(target),
        "payload_manifest_sha256": payload_hash,
        "manifest": manifest,
        "items": items,
        "evidence": evidence,
        "report": report,
        "source": source,
    }


def _replay_frame_evidence(
    *,
    source: Mapping[str, Any],
    component: Mapping[str, Any],
    row: Mapping[str, Any],
) -> None:
    source_row_ordinal = row.get("source_row_ordinal")
    if row.get("source_text_kind") == "normalized_component_text":
        if source_row_ordinal is not None:
            raise G3FormReferenceError("normalized evidence cannot name a source row")
        normalized = _normalized_component_text(source, component)
        if normalized is None:
            raise G3FormReferenceError("frame cites a missing normalized component")
        source_text, _logical, digest = normalized
    elif row.get("source_text_kind") == "candidate_projection_row":
        if isinstance(source_row_ordinal, bool) or not isinstance(
            source_row_ordinal, int
        ):
            raise G3FormReferenceError("candidate evidence lacks source row ordinal")
        result = _candidate_projection(source, component)
        if result is None:
            raise G3FormReferenceError("frame cites a missing candidate projection")
        projection, _logical, _digest = result
        projection_rows = _validate_chime_projection(
            projection, _component_id(component)
        )
        rows_by_ordinal = {
            int(candidate["source_row_ordinal"]): candidate
            for candidate in projection_rows
        }
        if source_row_ordinal not in rows_by_ordinal:
            raise G3FormReferenceError("candidate evidence row is outside CHIME-185")
        source_text = _chime_row_text(rows_by_ordinal[source_row_ordinal])
        digest = hashlib.sha256(source_text.encode("utf-8")).hexdigest()
    else:
        raise G3FormReferenceError("frame evidence source_text_kind is invalid")
    if row.get("source_text_sha256") != digest:
        raise G3FormReferenceError("frame evidence source text hash differs")
    quote = _trimmed_text(row.get("quote"), "frame evidence quote", maximum=4000)
    ordinal = row.get("occurrence_ordinal")
    if isinstance(ordinal, bool) or not isinstance(ordinal, int) or ordinal < 1:
        raise G3FormReferenceError("frame evidence ordinal is invalid")
    start, end = _occurrence_span(source_text, quote, ordinal)
    if row.get("snapshot_start") != start or row.get("snapshot_end") != end:
        raise G3FormReferenceError("frame evidence offsets do not replay")
    if row.get("evidence_text_sha256") != hashlib.sha256(
        quote.encode("utf-8")
    ).hexdigest():
        raise G3FormReferenceError("frame evidence quote hash differs")


def _hydrate_frozen_claims(
    *,
    source: Mapping[str, Any],
    claims: Mapping[str, Any],
    components_by_id: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    hydrated: list[dict[str, Any]] = []
    chime_allowed_claims: dict[str, set[str]] = {}
    for public_claim in _array_of_objects(claims.get("claims"), "v2 relation claims"):
        component_id = str(public_claim.get("component_id", ""))
        if component_id not in components_by_id:
            raise G3FormReferenceError("v2 relation claim component is unknown")
        component = components_by_id[component_id]
        if not _component_relation_eligible(component):
            raise G3FormReferenceError("non-relation source emitted a v2 claim")
        source_row_ordinal = public_claim.get("source_row_ordinal")
        if source_row_ordinal is None:
            normalized = _normalized_component_text(source, component)
            if normalized is None:
                raise G3FormReferenceError("v2 relation claim lacks frozen text")
            source_text, _logical, digest = normalized
            source_text_kind = "normalized_component_text"
            proposal_origin = "pair-seed-relocation/v2"
        else:
            if isinstance(source_row_ordinal, bool) or not isinstance(
                source_row_ordinal, int
            ):
                raise G3FormReferenceError("v2 claim source row ordinal is invalid")
            result = _candidate_projection(source, component)
            if result is None:
                raise G3FormReferenceError("v2 relation claim lacks candidate projection")
            projection, _logical, _projection_digest = result
            rows = _validate_chime_projection(projection, component_id)
            rows_by_ordinal = {
                int(row["source_row_ordinal"]): row for row in rows
            }
            if source_row_ordinal not in rows_by_ordinal:
                raise G3FormReferenceError("v2 relation claim escaped CHIME-185")
            source_text = _chime_row_text(rows_by_ordinal[source_row_ordinal])
            digest = hashlib.sha256(source_text.encode("utf-8")).hexdigest()
            source_text_kind = "candidate_projection_row"
            proposal_origin = "chime-explicit-relation-regex/v1"
        quote = _trimmed_text(
            public_claim.get("quote"), "v2 relation claim quote", maximum=4000
        )
        surface = _trimmed_text(
            public_claim.get("surface"), "v2 relation claim surface", maximum=160
        )
        canonical = _trimmed_text(
            public_claim.get("canonical"), "v2 relation claim canonical", maximum=160
        )
        if surface not in quote or canonical not in quote:
            raise G3FormReferenceError(
                "v2 relation claim quote lacks surface or canonical"
            )
        ordinal = public_claim.get("occurrence_ordinal")
        if isinstance(ordinal, bool) or not isinstance(ordinal, int) or ordinal < 1:
            raise G3FormReferenceError("v2 relation claim ordinal is invalid")
        _occurrence_span(source_text, quote, ordinal)
        if source_text_kind == "normalized_component_text":
            relocated = _relation_clause_for_pair(source_text, surface, canonical)
            if relocated is None or relocated[0] != quote:
                raise G3FormReferenceError(
                    "v2 pair claim is not the deterministic explicit relation clause"
                )
        else:
            if component_id not in chime_allowed_claims:
                projection_result = _candidate_projection(source, component)
                if projection_result is None:  # already checked above
                    raise G3FormReferenceError("CHIME projection disappeared")
                projection_value, _logical, _digest = projection_result
                generated, _dispositions = _extract_chime_claims(
                    component=component, projection=projection_value
                )
                chime_allowed_claims[component_id] = {
                    canonical_sha256(_claim_public_projection(row))
                    for row in generated
                }
            if canonical_sha256(public_claim) not in chime_allowed_claims[component_id]:
                raise G3FormReferenceError(
                    "v2 CHIME claim was not emitted by the frozen explicit regex"
                )
        hydrated.append(
            {
                **copy.deepcopy(public_claim),
                "_source_text": source_text,
                "_source_text_kind": source_text_kind,
                "_source_text_sha256": digest,
                "_proposal_origin": proposal_origin,
            }
        )
    return hydrated


def validate_form_relation_review_frame_v2(
    frame_dir: str | Path,
    *,
    source_bundle_dir: str | Path,
    workspace_root: str | Path,
    require_current_implementation: bool = True,
) -> dict[str, Any]:
    target = Path(frame_dir).resolve()
    try:
        payload_hash = validate_payload_manifest(target)
        ensure_exact_file_set(
            target,
            {
                "manifest.json",
                "claims.json",
                "items.json",
                "evidence.json",
                "extraction_report.json",
                "payload_manifest.json",
            },
        )
    except TrainingArtifactError as exc:
        raise G3FormReferenceError(str(exc)) from exc
    source = _validate_source_bundle_v2(
        source_bundle_dir,
        workspace_root=workspace_root,
        require_current_implementation=require_current_implementation,
    )
    manifest = _load_object(target / "manifest.json", "v2 form-review manifest")
    claims = _load_object(target / "claims.json", "v2 relation claims")
    items = _array_of_objects(load_json(target / "items.json"), "v2 form-review items")
    evidence = _array_of_objects(
        load_json(target / "evidence.json"), "v2 form-review evidence"
    )
    report = _load_object(
        target / "extraction_report.json", "v2 extraction report"
    )
    _validate_schema(
        claims, Path(workspace_root).resolve() / CLAIMS_SCHEMA_PATH
    )
    forbidden = _forbidden_frame_key_paths(
        {"claims": claims, "items": items, "evidence": evidence, "report": report}
    )
    if forbidden:
        raise G3FormReferenceError(
            "v2 relation frame contains forbidden keys: " + ", ".join(forbidden)
        )
    identity = {
        key: copy.deepcopy(value) for key, value in manifest.items() if key != "frame_id"
    }
    frame_id = FRAME_ID_PREFIX + canonical_sha256(identity)
    if manifest.get("frame_id") != frame_id or not _target_name_matches(
        target, frame_id
    ):
        raise G3FormReferenceError("v2 form-review content identity differs")
    if (
        manifest.get("schema_version") != FRAME_SCHEMA_VERSION
        or manifest.get("artifact_kind") != FRAME_ARTIFACT_KIND
        or manifest.get("source_bundle_dependency")
        != _dependency(source, SOURCE_BUNDLE_ARTIFACT_KIND, "source_bundle_id")
        or manifest.get("claims_schema_sha256")
        != sha256_file(Path(workspace_root).resolve() / CLAIMS_SCHEMA_PATH)
        or manifest.get("claims_sha256") != canonical_sha256(claims)
        or manifest.get("items_sha256") != canonical_sha256(items)
        or manifest.get("evidence_sha256") != canonical_sha256(evidence)
        or manifest.get("extraction_report_sha256") != canonical_sha256(report)
        or manifest.get("item_count") != len(items)
        or manifest.get("evidence_count") != len(evidence)
        or manifest.get("decision_actions") != sorted(ACTIONS)
        or manifest.get("variant_families") != sorted(FAMILIES)
        or manifest.get("review_policy") != FRAME_REVIEW_POLICY
        or manifest.get("reference_role") != REFERENCE_ROLE
        or manifest.get("scope") != SCOPE
        or manifest.get("scientific_eligible") is not False
        or manifest.get("sealed") is not False
    ):
        raise G3FormReferenceError("v2 form-review frame bindings differ")
    if require_current_implementation and manifest.get(
        "builder_implementation_sha256"
    ) != sha256_file(Path(__file__)):
        raise G3FormReferenceError("v2 form-review implementation drifted")
    if (
        report.get("schema_version") != REPORT_SCHEMA_VERSION
        or report.get("extractor") != EXTRACTOR_ID
        or report.get("source_bundle_id") != source["source_bundle_id"]
        or report.get("item_count") != len(items)
        or report.get("evidence_count") != len(evidence)
    ):
        raise G3FormReferenceError("v2 extraction report bindings differ")
    components = _array_of_objects(source.get("components"), "v2 source components")
    components_by_id = {_component_id(row): row for row in components}
    if len(components_by_id) != len(components):
        raise G3FormReferenceError("v2 source component IDs are duplicated")
    hydrated_claims = _hydrate_frozen_claims(
        source=source,
        claims=claims,
        components_by_id=components_by_id,
    )
    expected_evidence, expected_items = _build_evidence_and_items(
        hydrated_claims, components_by_id
    )
    if evidence != expected_evidence or items != expected_items:
        raise G3FormReferenceError(
            "v2 items/evidence do not replay from frozen relation claims"
        )
    seed_dispositions = _array_of_objects(
        report.get("seed_dispositions"), "seed dispositions"
    )
    chime_dispositions = _array_of_objects(
        report.get("chime_row_dispositions"), "CHIME row dispositions"
    )
    expected_guarantees = {
        "offline_only": True,
        "no_human_decisions_made": True,
        "old_evidence_not_reused": True,
        "surface_and_canonical_share_each_quote": True,
        "prevalence_or_method_cannot_emit_alone": True,
        "chime_only_52_abbreviation_plus_133_homophonic_rows": True,
        "forbidden_task_gold_definition_fields_not_carried": True,
    }
    disposition_counts = Counter(
        row.get("disposition")
        for row in [*seed_dispositions, *chime_dispositions]
    )
    role_counts = Counter(_component_role(row) for row in components)
    if (
        report.get("seed_count") != len(seed_dispositions)
        or report.get("chime_candidate_row_count") != len(chime_dispositions)
        or report.get("claim_count_before_deduplication")
        != len(claims["claims"])
        or report.get("source_role_counts") != dict(sorted(role_counts.items()))
        or report.get("disposition_counts")
        != dict(sorted(disposition_counts.items()))
        or report.get("guarantees") != expected_guarantees
    ):
        raise G3FormReferenceError("v2 extraction report does not replay")
    if report.get("chime_component_count") == 0:
        if chime_dispositions:
            raise G3FormReferenceError("CHIME dispositions exist without a projection")
    elif (
        report.get("chime_component_count") != 1
        or len(chime_dispositions) != CHIME_EXPECTED_CANDIDATE_COUNT
    ):
        raise G3FormReferenceError("CHIME extraction coverage differs from 185 rows")
    expected_evidence_fields = {
        "evidence_id",
        "component_id",
        "source_id",
        "source_role",
        "acquisition_mode",
        "publisher",
        "requested_url",
        "final_url",
        "snapshot_url",
        "quote",
        "occurrence_ordinal",
        "snapshot_start",
        "snapshot_end",
        "evidence_text_sha256",
        "source_text_kind",
        "source_text_sha256",
        "source_row_ordinal",
        "relation_note",
        "proposal_origin",
        "relation_contract",
    }
    evidence_by_id: dict[str, dict[str, Any]] = {}
    for row in evidence:
        if set(row) != expected_evidence_fields:
            raise G3FormReferenceError("v2 evidence fields are not canonical")
        evidence_id = str(row.get("evidence_id", ""))
        if not re.fullmatch(r"g3ev-v2-[0-9a-f]{40}", evidence_id):
            raise G3FormReferenceError("v2 evidence ID is invalid")
        if evidence_id in evidence_by_id:
            raise G3FormReferenceError("v2 evidence IDs are duplicated")
        component_id = str(row.get("component_id", ""))
        if component_id not in components_by_id:
            raise G3FormReferenceError("v2 evidence component is unknown")
        component = components_by_id[component_id]
        provenance = _component_provenance(component)
        if any(row.get(key) != value for key, value in provenance.items()):
            raise G3FormReferenceError("v2 evidence provenance differs from source bundle")
        if not _component_relation_eligible(component):
            raise G3FormReferenceError("non-relation source emitted v2 evidence")
        if row.get("relation_contract") != "single-quote-surface-and-canonical/v2":
            raise G3FormReferenceError("v2 relation evidence contract differs")
        _optional_text(row.get("relation_note"), "relation note", maximum=1000)
        _replay_frame_evidence(source=source, component=component, row=row)
        evidence_by_id[evidence_id] = row
    item_ids: set[str] = set()
    for row in items:
        if set(row) != {
            "item_id",
            "surface",
            "canonical",
            "proposed_family",
            "phonetic_scan_enabled",
            "evidence_ids",
        }:
            raise G3FormReferenceError("v2 form item fields are not canonical")
        item_id = str(row.get("item_id", ""))
        if not re.fullmatch(r"g3form-v2-[0-9a-f]{40}", item_id) or item_id in item_ids:
            raise G3FormReferenceError("v2 form item ID is invalid or duplicated")
        item_ids.add(item_id)
        surface = _trimmed_text(row.get("surface"), "form surface", maximum=160)
        canonical = _trimmed_text(row.get("canonical"), "form canonical", maximum=160)
        if surface == canonical or row.get("proposed_family") not in FAMILIES:
            raise G3FormReferenceError("v2 form item pair is invalid")
        if not isinstance(row.get("phonetic_scan_enabled"), bool):
            raise G3FormReferenceError("v2 form item scan flag is invalid")
        evidence_ids = row.get("evidence_ids")
        if (
            not isinstance(evidence_ids, list)
            or not evidence_ids
            or len(evidence_ids) != len(set(evidence_ids))
            or any(identifier not in evidence_by_id for identifier in evidence_ids)
        ):
            raise G3FormReferenceError("v2 form item evidence IDs are invalid")
        selected = [evidence_by_id[identifier] for identifier in evidence_ids]
        if len({evidence_row["component_id"] for evidence_row in selected}) != 1:
            raise G3FormReferenceError("one v2 form item must use one source component")
        if any(
            surface not in evidence_row["quote"]
            or canonical not in evidence_row["quote"]
            for evidence_row in selected
        ):
            raise G3FormReferenceError(
                "every v2 relation quote must contain surface and canonical"
            )
    if not items:
        raise G3FormReferenceError("v2 form-review frame cannot be empty")
    return {
        "frame_id": frame_id,
        "target": str(target),
        "payload_manifest_sha256": payload_hash,
        "manifest": manifest,
        "items": items,
        "evidence": evidence,
        "report": report,
        "claims": claims,
        "source": source,
    }


__all__ = [
    "FRAME_SCHEMA_VERSION",
    "build_form_relation_review_frame_v2",
    "extract_form_relation_claims_v2",
    "load_pair_seeds",
    "pair_seeds_from_v1_extraction",
    "validate_form_relation_review_frame_v2",
]
