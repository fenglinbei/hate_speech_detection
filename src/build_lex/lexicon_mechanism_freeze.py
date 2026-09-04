"""Deterministic resource snapshot under the user's batch-accept directive.

This is not the repair-review finalizer. It preserves human edits and review
history, accepts untouched proposals under separate authority, and records
missing definitions without inventing them or claiming completed blind review.
"""

from __future__ import annotations

import copy
import platform
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from build_lex.annotated_lexicon_operation_review import (
    CATEGORIES, FRAME_SCHEMA, entry_id, read_operation_session, validate_entry,
)
from build_lex.annotated_lexicon_repair import (
    REPAIR_SOURCE_SHA256, LexiconRepairError, canonical_sha256, file_sha256,
    read_json, write_json,
)
from build_lex.repair_regex_validation import check_regex_runtime, validate_match_policy
from rag.controlled_lexicon_matcher import (
    MATCHER_POLICY_VERSION, NORMALIZATION_VERSION, ControlledLexiconMatcher,
    normalize_surface,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
POLICY_VERSION = "preserve-saved-edits-default-accept-untouched-proposals/v1"
RENDERER_VERSION = "numbered-senses-explicit-missing-no-category-in-definition/v1"
MISSING_DEFINITION = "未提供释义（冻结时为空；未补写）"
DIRECTIVE = "https://hsd.fenglin.pro的内容除了我改动了的，剩下的先默认接受提案然后加入此次冻结边界，然后给我最终产物"
EXPECTED_SESSION_REVISION = "fa3a0ccc050f9c3be6276183ecf4d19966d439a09d047a4e5ee0e724a02c1745"
EXPECTED_SESSION_SHA256 = "867d9b59e89e4452efcf589d1eab31f184542ba0cc312df4aba4097107c9da9d"
SCHEMA_VERSION = "annotated-lexicon-mechanism-frozen/v1"
FILE_NAMES = {
    "decisions": "decisions.json", "manifest": "manifest.json",
    "validation": "validation.json", "review_session": "review_session_snapshot.json",
}


def _source_groups(payload: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for index, raw in enumerate(payload["terms"]):
        if set(raw) != {"term", "category", "definition"}:
            raise LexiconRepairError("source fields differ")
        if any(not isinstance(raw[key], str) or not raw[key].strip() for key in raw):
            raise LexiconRepairError("source contains an empty/non-string field")
        grouped[raw["term"]].append({"source_row_index": index, "lexicon_id": f"lex-{index:04d}", **raw})
    return dict(grouped)


def _original_entry(term: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    senses = []
    for row in rows:
        categories = [value.strip() for value in row["category"].split(",")]
        if any(value not in CATEGORIES for value in categories):
            raise LexiconRepairError("unrecognized original category")
        senses.append({
            "sense_id": row["lexicon_id"] + ":sense:01",
            "definition": row["definition"],
            "categories": list(dict.fromkeys(categories)),
        })
    return {
        "lexicon_id": entry_id(term, rows), "term": term, "variants": [],
        "senses": senses, "match_policy": {"require_any": [], "exclude_any": []},
    }


def render_entry(entry: dict[str, Any]) -> dict[str, str]:
    """Never put categories, review notes or proposal rationales in definitions."""
    labels = {label for sense in entry["senses"] for label in sense["categories"]}
    return {
        "category": ", ".join(label for label in CATEGORIES if label in labels),
        "definition": "\n".join(
            f"[{index}] {sense['definition'] if sense['definition'].strip() else MISSING_DEFINITION}"
            for index, sense in enumerate(entry["senses"], 1)
        ),
    }


def _validate_snapshot_inputs(
    lexicon_path: Path, frame_path: Path, session_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if file_sha256(lexicon_path) != REPAIR_SOURCE_SHA256:
        raise LexiconRepairError("source lexicon differs from frozen authority")
    if file_sha256(session_path) != EXPECTED_SESSION_SHA256:
        raise LexiconRepairError("review snapshot differs from verified paused session")
    source, frame = read_json(lexicon_path), read_json(frame_path)
    session = read_operation_session(session_path)
    manifest = frame["manifest"]
    if manifest.get("schema_version") != FRAME_SCHEMA:
        raise LexiconRepairError("operation frame schema differs")
    if manifest.get("items_sha256") != canonical_sha256(frame["items"]):
        raise LexiconRepairError("operation frame items differ")
    identity = {key: value for key, value in manifest.items() if key != "frame_id"}
    if manifest.get("frame_id") != "operation-frame-" + canonical_sha256(identity):
        raise LexiconRepairError("operation frame identity differs")
    if manifest.get("source_lexicon_sha256") != REPAIR_SOURCE_SHA256:
        raise LexiconRepairError("operation frame source differs")
    if (session["revision"] != EXPECTED_SESSION_REVISION
            or session["frame_id"] != manifest["frame_id"]
            or session["frame_sha256"] != canonical_sha256(frame)):
        raise LexiconRepairError("review snapshot/frame binding differs")
    item_ids = [item["item_id"] for item in frame["items"]]
    if len(set(item_ids)) != len(item_ids) or set(item_ids) != set(session["decisions"]):
        raise LexiconRepairError("review inventory differs")
    return source, frame, session


def select_review_entry(item: dict[str, Any], decision: dict[str, Any]) -> tuple[dict[str, Any], str]:
    """Selection authority is separate from the immutable historical status."""
    status, resolution = decision["status"], decision["resolution"]
    if status == "confirmed" and resolution in {"approve", "revise"}:
        return copy.deepcopy(decision["entry"]), "human_confirmed_preserved"
    if status == "deferred" and resolution == "defer":
        # This one exception is tied to the actually inspected user edit, not
        # a generic conversion of arbitrary deferrals into approvals.
        if (item["term"] != "它们" or "可以先保留" not in decision["notes"]
                or decision["entry"] == item["proposed_entry"]):
            raise LexiconRepairError("deferred item is outside this freeze authority")
        return copy.deepcopy(decision["entry"]), "user_edited_deferred_preserved"
    if status == "draft" and resolution not in {"reject", "defer"}:
        if decision["entry"] != item["proposed_entry"]:
            return copy.deepcopy(decision["entry"]), "user_saved_draft_preserved"
        return copy.deepcopy(item["proposed_entry"]), "user_batch_default_accept"
    raise LexiconRepairError("review status/resolution is outside this freeze authority")


def _validate_effective_entry(item: dict[str, Any], entry: dict[str, Any], allowed: set[str]) -> dict[str, Any]:
    entry = validate_entry(item, entry, strict=False, allowed_surfaces=allowed, compile_regex=False)
    if not entry["senses"]:
        raise LexiconRepairError("a frozen entry must declare a sense, even if its definition is missing")
    seen = {normalize_surface(entry["term"])}
    for variant in entry["variants"]:
        normalized = normalize_surface(variant)
        if not variant.strip() or variant not in allowed or normalized in seen:
            raise LexiconRepairError("variant is empty, undeclared or duplicated")
        seen.add(normalized)
    if any(not sense["categories"] for sense in entry["senses"]):
        raise LexiconRepairError("missing categories cannot be invented during freeze")
    normalized_policy = validate_match_policy(entry["match_policy"])
    if normalized_policy != entry["match_policy"]:
        raise LexiconRepairError("saved match policy requires an unapproved rewrite")
    return entry


def _span_diagnostic(matcher: ControlledLexiconMatcher, reference: dict[str, Any]) -> dict[str, Any]:
    rows = []
    for row in reference["rows"]:
        observed = matcher.match(row["query_content"])
        expected = {(hit["span"][0], hit["span"][1], hit["surface"]) for hit in row["expected_spans"]}
        actual = {(hit["span"][0], hit["span"][1], hit["raw_surface"]) for hit in observed["selected_spans"]}
        rows.append({
            "source_item_id": row["source_item_id"], "exact": expected == actual,
            "missing": [list(value) for value in sorted(expected - actual)],
            "extra": [list(value) for value in sorted(actual - expected)],
        })
    return {
        "purpose": "non-gating-static-compatibility-diagnostic-not-a-new-experiment",
        "reference_id": reference["reference_id"], "item_count": len(rows),
        "exact_count": sum(row["exact"] for row in rows), "rows": rows,
        "human_blind_review_completed": False, "repair_acceptance_claimed": False,
    }


def build_mechanism_freeze(
    *, lexicon_path: Path, frame_path: Path, session_path: Path, span_reference_path: Path,
) -> dict[str, dict[str, Any]]:
    source, frame, session = _validate_snapshot_inputs(lexicon_path, frame_path, session_path)
    runtime_check = check_regex_runtime()
    reference = read_json(span_reference_path)
    if canonical_sha256(reference) != frame["manifest"]["span_reference_sha256"]:
        raise LexiconRepairError("span reference binding differs")
    groups = _source_groups(source)
    if len(source["terms"]) != 829 or len(groups) != 818:
        raise LexiconRepairError("source inventory differs")
    entries = {term: _original_entry(term, rows) for term, rows in groups.items()}
    operation_records = []
    allowed = set(frame["manifest"]["allowed_candidate_surfaces"])
    for item in frame["items"]:
        term = item["term"]
        if item["source_entries"] != groups.get(term, []):
            raise LexiconRepairError("operation provenance differs from original rows")
        selected, selection = select_review_entry(item, session["decisions"][item["item_id"]])
        selected = _validate_effective_entry(item, selected, allowed)
        if selected["lexicon_id"] != entry_id(term, groups.get(term, [])):
            raise LexiconRepairError("stable entry identity differs")
        entries[term] = selected
        decision = session["decisions"][item["item_id"]]
        operation_records.append({
            "item_id": item["item_id"], "term": term, "selection_source": selection,
            "historical_status": decision["status"], "historical_resolution": decision["resolution"],
            "historical_notes": decision["notes"],
            "entry_changed_from_proposal": decision["entry"] != item["proposed_entry"],
            "proposal_entry_sha256": canonical_sha256(item["proposed_entry"]),
            "effective_entry_sha256": canonical_sha256(selected), "effective_entry": selected,
            "source_record_indices": [row["source_row_index"] for row in groups.get(term, [])],
            "inherited_proposal_flags_not_current_verdicts": item["flags"],
        })
    by_term = {row["term"]: row for row in operation_records}
    terms = []
    for term, entry in sorted(entries.items(), key=lambda pair: (
        not bool(groups.get(pair[0])),
        groups[pair[0]][0]["source_row_index"] if pair[0] in groups else 0,
        pair[0],
    )):
        terms.append({
            **entry, **render_entry(entry),
            "source_record_indices": [row["source_row_index"] for row in groups.get(term, [])],
            "freeze_selection_source": by_term[term]["selection_source"] if term in by_term else "original_preserved",
        })
    if len({entry["lexicon_id"] for entry in terms}) != len(terms):
        raise LexiconRepairError("duplicate stable entry ID")
    sense_ids = [sense["sense_id"] for entry in terms for sense in entry["senses"]]
    if len(set(sense_ids)) != len(sense_ids):
        raise LexiconRepairError("duplicate stable sense ID")
    if sorted(index for entry in terms for index in entry["source_record_indices"]) != list(range(829)):
        raise LexiconRepairError("source record provenance was lost or duplicated")
    missing = [{"term": entry["term"], "sense_id": sense["sense_id"]}
               for entry in terms for sense in entry["senses"] if not sense["definition"].strip()]
    counts = {
        "source_records": 829, "source_unique_terms": 818, "duplicate_source_groups": 11,
        "runtime_entries": len(terms), "new_entries": sum(not entry["source_record_indices"] for entry in terms),
        "senses": len(sense_ids), "missing_definitions": len(missing),
        "entries_with_rules": sum(any(entry["match_policy"].values()) for entry in terms),
        "rule_count": sum(len(rules) for entry in terms for rules in entry["match_policy"].values()),
        "operation_selections": dict(Counter(row["selection_source"] for row in operation_records)),
        "historical_review_status": dict(Counter(row["status"] for row in session["decisions"].values())),
    }
    decisions = {
        "schema_version": "annotated-lexicon-mechanism-freeze-decisions/v1",
        "policy_version": POLICY_VERSION, "user_directive": DIRECTIVE,
        "review_session_revision": session["revision"], "historical_session_modified": False,
        "operations": operation_records,
    }
    sources = {
        "source_lexicon_sha256": file_sha256(lexicon_path), "review_frame_sha256": file_sha256(frame_path),
        "review_session_sha256": file_sha256(session_path), "span_reference_sha256": file_sha256(span_reference_path),
    }
    code_hashes = {name: file_sha256(REPOSITORY_ROOT / name) for name in (
        "src/build_lex/lexicon_mechanism_freeze.py", "src/build_lex/annotated_lexicon_operation_review.py",
        "src/build_lex/annotated_lexicon_repair.py", "src/build_lex/repair_regex_validation.py",
        "src/rag/controlled_lexicon_matcher.py",
        "scripts/stage1/freeze_lexicon_mechanism.py", "schemas/annotated_lexicon_mechanism_frozen_v1.schema.json",
    )}
    matcher = ControlledLexiconMatcher(terms, lexicon_sha256=REPAIR_SOURCE_SHA256)
    identity = {
        "schema_version": SCHEMA_VERSION, "policy_version": POLICY_VERSION, **sources,
        "terms_sha256": canonical_sha256(terms), "decisions_sha256": canonical_sha256(decisions),
        "matcher_policy_sha256": matcher.policy_sha256, "matcher_policy_version": MATCHER_POLICY_VERSION,
        "normalization_version": NORMALIZATION_VERSION, "renderer_version": RENDERER_VERSION,
        "unicode_database_version": unicodedata.unidata_version,
        "regex_version": runtime_check["regex_version"], "builder_code_sha256": code_hashes,
    }
    build_id = "lexicon-mechanism-frozen-" + canonical_sha256(identity)
    lexicon = {
        "schema_version": SCHEMA_VERSION, "lexicon_build_id": build_id,
        "description": "用于 LLM 机制研究的固定资源；保留用户编辑，其余提案批量接受；不宣称完整人审或无语义噪声。",
        "matcher_policy_version": MATCHER_POLICY_VERSION, "matcher_policy_sha256": matcher.policy_sha256,
        "normalization_version": NORMALIZATION_VERSION, "renderer_version": RENDERER_VERSION,
        "terms_sha256": identity["terms_sha256"], "freeze_identity": identity,
        "quality": {"development_only": True, "fully_human_reviewed": False,
                    "missing_definition_count": len(missing), "sense_disambiguation_implemented": False},
        "terms": terms,
    }
    artifact_sha = canonical_sha256(lexicon)
    final_matcher = ControlledLexiconMatcher(terms, lexicon_sha256=artifact_sha, policy_sha256=matcher.policy_sha256)
    diagnostic = _span_diagnostic(final_matcher, reference)
    validation = {
        "schema_version": "annotated-lexicon-mechanism-freeze-validation/v1", "lexicon_build_id": build_id,
        "lexicon_sha256": artifact_sha, "structural_checks_passed": True,
        "normalized_surface_collisions": 0, "all_source_records_accounted_for": True,
        "all_saved_effective_entries_preserved": True, "isolated_regex_check": runtime_check,
        "missing_definitions": missing, "span_diagnostic": diagnostic,
        "limits": [
            "7 个空释义词条按授权保留，空值未补写；其较长词面仍参与重叠选择，可能压制更短候选。",
            "它们保留已编辑内容和历史暂缓标记，未宣称原候选资格问题已解决。",
            "它保留实际保存的掀它东西规则；备注与狭窄规则的张力未自动修正。",
            "多义同时渲染不等于上下文义项消歧；未实现备注中提及的后续语义筛选。",
            "静态 span 诊断仅报告差异，不作为继续人工修复的门槛。",
        ],
    }
    manifest = {
        "schema_version": "annotated-lexicon-mechanism-freeze-manifest/v1", "lexicon_build_id": build_id,
        "identity": identity, "counts": counts, "source_review_frame_id": frame["manifest"]["frame_id"],
        "source_review_session_revision": session["revision"],
        "artifact_sha256": {"lexicon": artifact_sha, "decisions": canonical_sha256(decisions),
                            "review_session_snapshot": canonical_sha256(session), "validation": canonical_sha256(validation)},
        "runtime_contract": {
            "runtime_resource_count": 1, "lexicon_schema": "cold",
            "expected_lexicon_count": len(terms), "exact_top_k": -1,
            "controlled_matcher_required": True, "legacy_v0_config_must_not_be_reused": True,
            "max_sequence_tokens": 8192, "overflow_policy": "explicit-error-no-silent-trimming",
            "regex_version": runtime_check["regex_version"],
            "python_verified": platform.python_version(), "unicode_database_version": unicodedata.unidata_version,
            "review_metadata_in_prompt": False, "gpu_or_new_human_review_required": False,
        },
    }
    return {"lexicon": lexicon, "manifest": manifest, "decisions": decisions,
            "validation": validation, "review_session": session}


def write_freeze(bundle: dict[str, dict[str, Any]], *, output_path: Path, record_root: Path) -> dict[str, str]:
    targets = {"lexicon": output_path, **{key: record_root / value for key, value in FILE_NAMES.items()}}
    # Preflight the entire bundle before writing anything; never overwrite a
    # different freeze, source lexicon, symlink, or historical review session.
    for key, target in targets.items():
        if target.is_symlink():
            raise LexiconRepairError("freeze output may not be a symlink")
        if target.exists() and (not target.is_file() or read_json(target) != bundle[key]):
            raise LexiconRepairError(f"refusing to overwrite differing freeze output: {target}")
    for key, target in targets.items():
        if not target.exists():
            write_json(target, bundle[key], mode=0o444)
        if file_sha256(target) != canonical_sha256(bundle[key]):
            raise LexiconRepairError("freeze output bytes differ from manifest")
    return {key: str(target.resolve()) for key, target in targets.items()}


def verify_freeze(bundle: dict[str, dict[str, Any]], *, output_path: Path, record_root: Path) -> dict[str, str]:
    """Compare rebuilt bytes with every frozen output, without writing anything."""
    targets = {"lexicon": output_path, **{key: record_root / value for key, value in FILE_NAMES.items()}}
    for key, target in targets.items():
        if target.is_symlink() or not target.is_file():
            raise LexiconRepairError(f"missing or unsafe frozen output: {target}")
        if file_sha256(target) != canonical_sha256(bundle[key]):
            raise LexiconRepairError(f"frozen output checksum differs: {target}")
    return {key: str(target.resolve()) for key, target in targets.items()}
