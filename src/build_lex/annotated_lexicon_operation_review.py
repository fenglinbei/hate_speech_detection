"""Independent, evidence-bound entry-operation review for lexicon repair v1.

This stage records human decisions, not a runtime lexicon.  Rejection or a
confirmed review does not bypass the later whole-lexicon regression gate.
"""

from __future__ import annotations

import copy
import hashlib
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlsplit

from build_lex.annotated_lexicon_repair import (
    LexiconRepairConflict, LexiconRepairError, REPAIR_SOURCE_SHA256,
    _now_iso, _session_lock, _with_revision, canonical_sha256,
    file_sha256, read_json, write_json,
)
from build_lex.repair_regex_validation import validate_match_policy
from rag.controlled_lexicon_matcher import normalize_surface


FRAME_SCHEMA = "annotated-lexicon-operation-frame/v1"
SESSION_SCHEMA = "annotated-lexicon-operation-session/v1"
CATEGORIES = ("Racism", "Sexism", "LGBTQ", "Region", "others")
RESOLUTIONS = {None, "approve", "revise", "reject", "defer"}
ENTRY_FIELDS = {"lexicon_id", "term", "variants", "senses", "match_policy"}


def entry_id(term: str, source_entries: list[dict[str, Any]]) -> str:
    if source_entries:
        return min(source_entries, key=lambda row: row["source_row_index"])["lexicon_id"]
    return "lex-new-" + hashlib.sha256(term.encode("utf-8")).hexdigest()[:16]


def _text(value: Any, name: str, limit: int, *, required: bool = False) -> str:
    if not isinstance(value, str) or len(value) > limit:
        raise ValueError(f"{name} 必须是最多 {limit} 字符的文本")
    if required and not value.strip():
        raise ValueError(f"{name} 不得为空")
    return value


def _draft_policy(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) - {"require_any", "exclude_any"}:
        raise ValueError("match_policy 只能包含 require_any / exclude_any")
    result: dict[str, Any] = {}
    count = 0
    for group in ("require_any", "exclude_any"):
        rules = value.get(group, [])
        if not isinstance(rules, list):
            raise ValueError("规则组必须为列表")
        count += len(rules)
        result[group] = []
        for rule in rules:
            if not isinstance(rule, dict) or set(rule) != {"rule_id", "target", "pattern"}:
                raise ValueError("规则字段须为 rule_id / target / pattern")
            _text(rule["rule_id"], "rule_id", 100)
            _text(rule["pattern"], "pattern", 256)
            if rule["target"] not in {"left", "right", "surface", "context"}:
                raise ValueError("规则 target 无效")
            result[group].append(dict(rule))
    if count > 8:
        raise ValueError("每条词条最多 8 条上下文规则")
    return result


def validate_entry(
    item: Mapping[str, Any], entry: Any, *, strict: bool,
    allowed_surfaces: set[str], compile_regex: bool = True,
) -> dict[str, Any]:
    if not isinstance(entry, dict) or set(entry) != ENTRY_FIELDS:
        raise ValueError("词条字段须为 lexicon_id、term、variants、senses、match_policy")
    if entry["term"] != item["term"] or entry["lexicon_id"] != item["proposed_entry"]["lexicon_id"]:
        raise ValueError("本轮 term 和稳定 lexicon_id 不可更改；新词面需遵守封闭审核范围")
    result = copy.deepcopy(entry)
    variants = result["variants"]
    if not isinstance(variants, list) or len(variants) > 32:
        raise ValueError("variants 必须为最多 32 项的列表")
    normalized = {normalize_surface(result["term"])}
    for variant in variants:
        _text(variant, "variant", 100, required=strict)
        if strict:
            if variant not in allowed_surfaces:
                raise ValueError(f"variant {variant!r} 不在本轮预声明词面内")
            surface = normalize_surface(variant)
            if surface in normalized:
                raise ValueError("variant 与 term 或其他 variant 规范化后重复；大小写等价不必重复声明")
            normalized.add(surface)
    senses = result["senses"]
    if not isinstance(senses, list) or len(senses) > 16 or (strict and not senses):
        raise ValueError("采用的词条须有 1–16 个义项")
    ids: set[str] = set()
    for sense in senses:
        if not isinstance(sense, dict) or set(sense) != {"sense_id", "definition", "categories"}:
            raise ValueError("义项字段须为 sense_id / definition / categories")
        _text(sense["sense_id"], "sense_id", 100, required=True)
        if sense["sense_id"] in ids:
            raise ValueError("sense_id 不可重复")
        ids.add(sense["sense_id"])
        _text(sense["definition"], "义项解释", 3000, required=strict)
        categories = sense["categories"]
        if not isinstance(categories, list) or any(value not in CATEGORIES for value in categories):
            raise ValueError("义项 categories 必须从既定五类中选择")
        if len(set(categories)) != len(categories) or (strict and not categories):
            raise ValueError("采用的义项至少选择一个类别，且不得重复")
    result["match_policy"] = _draft_policy(result["match_policy"])
    if strict:
        rule_ids: set[str] = set()
        for rules in result["match_policy"].values():
            for rule in rules:
                if not rule["rule_id"].strip() or not rule["pattern"] or rule["rule_id"] in rule_ids:
                    raise ValueError("采用的规则必须有非空、唯一 ID 和非空表达式")
                rule_ids.add(rule["rule_id"])
        if compile_regex:
            result["match_policy"] = validate_match_policy(result["match_policy"])
    return result


def render_legacy_entry(entry: Mapping[str, Any]) -> dict[str, str]:
    categories = {category for sense in entry["senses"] for category in sense["categories"]}
    definitions = [sense["definition"].strip() for sense in entry["senses"]]
    return {
        "term": str(entry["term"]),
        "category": ", ".join(category for category in CATEGORIES if category in categories),
        "definition": "\n".join(f"[{index}] {definition}" for index, definition in enumerate(definitions, 1)),
    }


def build_operation_frame(
    *, lexicon_path: Path, span_frame_path: Path, span_reference_path: Path,
    proposals_path: Path, output_path: Path,
) -> dict[str, Any]:
    if file_sha256(lexicon_path) != REPAIR_SOURCE_SHA256:
        raise LexiconRepairError("source lexicon differs from frozen repair input")
    lexicon = read_json(lexicon_path)
    span_frame = read_json(span_frame_path)
    reference = read_json(span_reference_path)
    proposals = read_json(proposals_path)
    if reference.get("schema_version") != "annotated-lexicon-span-gold-reference/v1":
        raise LexiconRepairError("a finalized span-gold reference is required")
    if reference.get("frame_sha256") != canonical_sha256(span_frame):
        raise LexiconRepairError("span reference and frame differ")
    if reference.get("source_lexicon_sha256") != REPAIR_SOURCE_SHA256:
        raise LexiconRepairError("reference lexicon differs")
    if reference.get("rows_sha256") != canonical_sha256(reference.get("rows", [])):
        raise LexiconRepairError("reference row hash differs")
    identity = {key: value for key, value in reference.items() if key not in {"rows", "reference_id"}}
    if reference.get("reference_id") != "span-gold-ref-" + canonical_sha256(identity):
        raise LexiconRepairError("reference identity differs")
    if span_frame["manifest"]["items_sha256"] != canonical_sha256(span_frame["items"]):
        raise LexiconRepairError("span frame items differ")
    references = {row["source_item_id"]: row for row in reference["rows"]}
    if len(references) != len(reference["rows"]) or set(references) != {row["source_item_id"] for row in span_frame["items"]}:
        raise LexiconRepairError("span reference inventory differs")

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    normalized_terms: dict[str, str] = {}
    for index, row in enumerate(lexicon["terms"]):
        term = row["term"]
        normalized = normalize_surface(term)
        if normalized in normalized_terms and normalized_terms[normalized] != term:
            raise LexiconRepairError("source contains an unresolved normalized term collision")
        normalized_terms[normalized] = term
        grouped[term].append({"source_row_index": index, "lexicon_id": f"lex-{index:04d}", **row})
    duplicates = {term for term, rows in grouped.items() if len(rows) > 1}
    if len(duplicates) != 11:
        raise LexiconRepairError("expected all 11 duplicate term groups")
    required = set(span_frame["manifest"]["affected_terms"]) | duplicates
    final_surfaces = set()
    for row in reference["rows"]:
        for span in row["expected_spans"]:
            final_surfaces.add(span["surface"])
            required.add(normalized_terms.get(normalize_surface(span["surface"]), span["surface"]))
    if proposals.get("schema_version") != "annotated-lexicon-operation-proposals/v1":
        raise LexiconRepairError("proposal schema differs")
    proposal_items = proposals.get("items", [])
    by_term = {row["term"]: row for row in proposal_items}
    if len(by_term) != len(proposal_items) or set(by_term) != required:
        raise LexiconRepairError(f"proposal inventory differs: missing={sorted(required-set(by_term))}, extra={sorted(set(by_term)-required)}")
    allowed_surfaces = required | final_surfaces
    items = []
    for term in sorted(required, key=lambda value: (not bool(grouped.get(value)), value)):
        proposal = by_term[term]
        source_entries = grouped.get(term, [])
        proposed = copy.deepcopy(proposal["proposed_entry"])
        if proposed.get("term") != term or proposed.get("lexicon_id") != entry_id(term, source_entries):
            raise LexiconRepairError(f"{term}: proposal stable identity differs")
        evidence = []
        for source in span_frame["items"]:
            ref = references[source["source_item_id"]]
            if ref["query_content"] != source["query_content"] or ref["content_sha256"] != source["content_sha256"]:
                raise LexiconRepairError("reference source text differs")
            mentions = any(hit["term"] == term for candidate in source["candidates"] for hit in candidate["source_hits"])
            mentions = mentions or any(normalize_surface(span["surface"]) == normalize_surface(term) for span in ref["expected_spans"])
            if not mentions:
                continue
            wanted = {(span["span"][0], span["span"][1]) for span in ref["expected_spans"]}
            evidence.append({
                "source_item_id": source["source_item_id"], "query_content": source["query_content"],
                "expected_spans": copy.deepcopy(ref["expected_spans"]),
                "dropped_candidates": [{"surface": row["surface"], "span": row["span"]} for row in source["candidates"] if tuple(row["span"]) not in wanted],
                "review_notes": ref["review_notes"], "prior_audit": copy.deepcopy(source.get("prior_audit", {})),
            })
        item = {
            "item_id": "operation-" + canonical_sha256([REPAIR_SOURCE_SHA256, term])[:20],
            "term": term,
            "operation_kind": "duplicate_merge" if len(source_entries) > 1 else "existing" if source_entries else "new_entry",
            "source_entries": source_entries, "proposed_entry": proposed,
            "rationale": _text(proposal.get("rationale", ""), "rationale", 6000, required=True),
            "open_questions": copy.deepcopy(proposal.get("open_questions", [])),
            "flags": copy.deepcopy(proposal.get("flags", [])),
            "external_evidence": copy.deepcopy(proposal.get("external_evidence", [])),
            "evidence": evidence, "legacy_preview": render_legacy_entry(proposed),
        }
        for field in ("open_questions", "flags"):
            if not isinstance(item[field], list) or any(not isinstance(v, str) or len(v) > 3000 for v in item[field]):
                raise LexiconRepairError(f"{term}: {field} invalid")
        for external in item["external_evidence"]:
            if set(external) != {"title", "url", "note"} or urlsplit(external["url"]).scheme != "https":
                raise LexiconRepairError("external evidence must have an HTTPS source")
        item["proposed_entry"] = validate_entry(item, proposed, strict=False, allowed_surfaces=allowed_surfaces)
        item["proposed_entry"]["match_policy"] = validate_match_policy(item["proposed_entry"]["match_policy"])
        item["legacy_preview"] = render_legacy_entry(item["proposed_entry"])
        items.append(item)
    manifest = {
        "schema_version": FRAME_SCHEMA, "development_only": True, "single_pass_review": True,
        "source_lexicon_sha256": REPAIR_SOURCE_SHA256,
        "span_frame_sha256": canonical_sha256(span_frame),
        "span_reference_id": reference["reference_id"], "span_reference_sha256": canonical_sha256(reference),
        "proposal_sha256": canonical_sha256(proposals), "item_count": len(items),
        "kind_counts": dict(Counter(item["operation_kind"] for item in items)),
        "duplicate_term_group_count": len(duplicates), "items_sha256": canonical_sha256(items),
        "allowed_candidate_surfaces": sorted(allowed_surfaces),
        "matcher_contract": {"candidate_sources": ["term", "explicit_variant", "controlled_normalization"], "context_window_per_side": 16, "regex_role": "eligibility-only", "sense_selection": "not-implemented-entry-eligibility-only"},
    }
    manifest["frame_id"] = "operation-frame-" + canonical_sha256(manifest)
    frame = {"manifest": manifest, "items": items}
    if output_path.exists():
        if read_json(output_path) != frame:
            raise LexiconRepairError("refusing to overwrite a different operation frame")
    else:
        write_json(output_path, frame)
    return frame


def read_operation_session(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise LexiconRepairError("operation session is unavailable or unsafe")
    session = read_json(path)
    if session.get("schema_version") != SESSION_SCHEMA or session != _with_revision(session):
        raise LexiconRepairError("operation session schema/revision differs")
    return session


def validate_operation_decision(
    item: Mapping[str, Any], decision: Any, *, confirm: bool, allowed_surfaces: set[str], compile_regex: bool = True,
) -> dict[str, Any]:
    if not isinstance(decision, dict) or set(decision) != {"resolution", "entry", "notes"}:
        raise ValueError("决定字段须为 resolution、entry、notes")
    resolution = decision["resolution"]
    if not isinstance(resolution, (str, type(None))) or resolution not in RESOLUTIONS:
        raise ValueError("请选择采用提案、修订采用、不采用或暂缓")
    notes = _text(decision["notes"], "审核备注", 2000).strip()
    if confirm and resolution is None:
        raise ValueError("确认前须选择处理决定")
    adopted = resolution in {"approve", "revise"}
    if confirm and (resolution in {"reject", "defer"} or item["open_questions"]) and not notes:
        raise ValueError("不采用、暂缓或存在待核实问题的条目，须填写判断理由/核实依据")
    entry = decision["entry"]
    if entry is None:
        if adopted or (not confirm and resolution not in {"reject", "defer"}):
            raise ValueError("采用时须提供完整词条")
    else:
        entry = validate_entry(item, entry, strict=confirm and adopted, allowed_surfaces=allowed_surfaces, compile_regex=compile_regex)
    if confirm and resolution == "approve" and entry != item["proposed_entry"]:
        raise ValueError("词条已修改，请选择修订采用，而非原样采用提案")
    if confirm and adopted and "policy_unresolved" in item.get("flags", []) and entry["match_policy"] == item["proposed_entry"]["match_policy"]:
        raise ValueError("该提案的候选资格策略尚未解决；须修订规则或暂缓，不能原样采用")
    return {"resolution": resolution, "entry": entry, "notes": notes}


class OperationReviewStore:
    def __init__(self, *, frame_path: Path, session_path: Path, reviewer_id: str) -> None:
        if session_path.is_symlink():
            raise LexiconRepairError("session symlinks are not allowed")
        self.frame_path = frame_path.resolve()
        self.session_path = session_path.resolve()
        self.frame = read_json(self.frame_path)
        manifest = self.frame.get("manifest", {})
        if manifest.get("schema_version") != FRAME_SCHEMA or manifest.get("items_sha256") != canonical_sha256(self.frame.get("items", [])):
            raise LexiconRepairError("operation frame schema/items hash differs")
        identity = {key: value for key, value in manifest.items() if key != "frame_id"}
        if manifest.get("frame_id") != "operation-frame-" + canonical_sha256(identity):
            raise LexiconRepairError("operation frame identity differs")
        self.frame_id = manifest["frame_id"]
        self.item_by_id = {item["item_id"]: item for item in self.frame["items"]}
        if len(self.item_by_id) != len(self.frame["items"]) or manifest["item_count"] != len(self.item_by_id):
            raise LexiconRepairError("operation item inventory differs")
        self.allowed_surfaces = set(manifest["allowed_candidate_surfaces"])
        reviewer_id = _text(reviewer_id, "reviewer_id", 100, required=True).strip()
        with _session_lock(self.session_path):
            if self.session_path.exists():
                session = self._read()
                if session["reviewer_id"] != reviewer_id:
                    raise LexiconRepairError("operation session belongs to another reviewer")
            else:
                session = _with_revision({
                    "schema_version": SESSION_SCHEMA, "frame_id": self.frame_id,
                    "frame_sha256": canonical_sha256(self.frame), "reviewer_id": reviewer_id,
                    "single_pass_review": True, "finalized_bundle_id": None,
                    "created_at": _now_iso(), "updated_at": _now_iso(), "amendments": [],
                    "decisions": {key: {"status": "draft", "resolution": None, "entry": copy.deepcopy(item["proposed_entry"]), "notes": ""} for key, item in self.item_by_id.items()},
                })
                write_json(self.session_path, session)

    def _read(self) -> dict[str, Any]:
        session = read_operation_session(self.session_path)
        if session.get("frame_id") != self.frame_id or session.get("frame_sha256") != canonical_sha256(self.frame):
            raise LexiconRepairError("operation session belongs to another frame")
        if set(session.get("decisions", {})) != set(self.item_by_id):
            raise LexiconRepairError("operation session inventory differs")
        for item_id, decision in session["decisions"].items():
            if decision.get("status") not in {"draft", "confirmed", "deferred"}:
                raise LexiconRepairError("operation decision status differs")
            if decision["status"] == "deferred" and decision.get("resolution") != "defer":
                raise LexiconRepairError("deferred decision resolution differs")
            if decision["status"] == "confirmed" and decision.get("resolution") not in {"approve", "revise", "reject"}:
                raise LexiconRepairError("confirmed decision resolution differs")
            payload = {key: value for key, value in decision.items() if key != "status"}
            try:
                normalized = validate_operation_decision(self.item_by_id[item_id], payload, confirm=decision["status"] != "draft", allowed_surfaces=self.allowed_surfaces, compile_regex=False)
            except (TypeError, ValueError) as exc:
                raise LexiconRepairError("persisted operation decision is invalid") from exc
            if normalized != payload:
                raise LexiconRepairError("persisted operation decision is not canonical")
        return session

    def _status(self, session: Mapping[str, Any]) -> dict[str, Any]:
        counts = Counter(row["status"] for row in session["decisions"].values())
        return {"item_count": len(self.item_by_id), "confirmed_count": counts["confirmed"],
                "deferred_count": counts["deferred"], "open_count": len(self.item_by_id)-counts["confirmed"],
                "draft_count": counts["draft"], "amendment_count": len(session["amendments"]),
                "finalized_bundle_id": session["finalized_bundle_id"], "revision": session["revision"]}

    def _summary(self, session: Mapping[str, Any], item_id: str) -> dict[str, Any]:
        item = self.item_by_id[item_id]
        decision = session["decisions"][item_id]
        return {"item_id": item_id, "term": item["term"], "operation_kind": item["operation_kind"],
                "flags": item["flags"], "status": decision["status"], "resolution": decision["resolution"],
                "query_preview": item["evidence"][0]["query_content"][:100] if item["evidence"] else "重复组：本轮无句级证据"}

    def bootstrap(self) -> dict[str, Any]:
        with _session_lock(self.session_path):
            session = self._read()
            return {"schema_version": "annotated-lexicon-operation-bootstrap/v1", "frame_id": self.frame_id,
                    "revision": session["revision"], "status": self._status(session),
                    "items": [self._summary(session, key) for key in self.item_by_id],
                    "warnings": ["DEVELOPMENT ONLY / NON-SEALED", "独立单次操作审核；无模型输出或任务标签。", "确认并不跳过整体词典冲突、覆盖与盲审门槛。"]}

    def item_state(self, item_id: str) -> dict[str, Any]:
        if item_id not in self.item_by_id:
            raise ValueError("unknown operation item")
        with _session_lock(self.session_path):
            session = self._read()
            return {"revision": session["revision"], "item": copy.deepcopy(self.item_by_id[item_id]),
                    "decision": copy.deepcopy(session["decisions"][item_id]), "item_summary": self._summary(session, item_id)}

    def save(self, *, expected_revision: str, item_id: str, decision: Any, confirm: bool) -> dict[str, Any]:
        if item_id not in self.item_by_id or not isinstance(confirm, bool):
            raise ValueError("invalid operation save request")
        normalized = validate_operation_decision(self.item_by_id[item_id], decision, confirm=confirm, allowed_surfaces=self.allowed_surfaces)
        with _session_lock(self.session_path):
            session = self._read()
            if session["revision"] != expected_revision:
                raise LexiconRepairConflict("审核会话已变化，请重新载入并核对本地草稿")
            if session["finalized_bundle_id"] is not None:
                raise ValueError("已冻结的操作会话不可修改")
            if session["decisions"][item_id]["status"] != "draft":
                raise ValueError("已确认/暂缓的决定须先重新打开并留理由")
            updated = copy.deepcopy(session)
            status = "deferred" if confirm and normalized["resolution"] == "defer" else "confirmed" if confirm else "draft"
            updated["decisions"][item_id] = {"status": status, **normalized}
            updated["updated_at"] = _now_iso()
            updated = _with_revision(updated)
            write_json(self.session_path, updated)
            return {"revision": updated["revision"], "status": self._status(updated),
                    "decision": copy.deepcopy(updated["decisions"][item_id]), "item_summary": self._summary(updated, item_id)}

    def reopen(self, *, expected_revision: str, item_id: str, reason: str) -> dict[str, Any]:
        reason = _text(reason, "重新打开理由", 1000, required=True).strip()
        if item_id not in self.item_by_id:
            raise ValueError("unknown operation item")
        with _session_lock(self.session_path):
            session = self._read()
            if session["revision"] != expected_revision:
                raise LexiconRepairConflict("审核会话已变化，请重新载入")
            if session["finalized_bundle_id"] is not None or session["decisions"][item_id]["status"] == "draft":
                raise ValueError("仅未冻结的已确认/暂缓项可以重新打开")
            updated = copy.deepcopy(session)
            updated["decisions"][item_id]["status"] = "draft"
            updated["amendments"].append({"item_id": item_id, "reason": reason, "prior_revision": session["revision"], "reopened_at": _now_iso()})
            updated["updated_at"] = _now_iso()
            updated = _with_revision(updated)
            write_json(self.session_path, updated)
            return {"revision": updated["revision"], "status": self._status(updated),
                    "decision": copy.deepcopy(updated["decisions"][item_id]), "item_summary": self._summary(updated, item_id)}

    def snapshot(self) -> dict[str, Any]:
        with _session_lock(self.session_path):
            session = self._read()
            return {"schema_version": "annotated-lexicon-operation-review-snapshot/v1", "exported_at": _now_iso(),
                    "frame": copy.deepcopy(self.frame), "session": session,
                    "checksums": {"frame": canonical_sha256(self.frame), "session": canonical_sha256(session)}}
