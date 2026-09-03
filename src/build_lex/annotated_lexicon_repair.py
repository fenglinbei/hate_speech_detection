"""Review and materialization primitives for annotated lexicon repair v1."""

from __future__ import annotations

import copy
import fcntl
import hashlib
import json
import os
import tempfile
from collections import Counter
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence


SPAN_GOLD_FRAME_SCHEMA = "annotated-lexicon-span-gold-frame/v1"
SPAN_GOLD_SESSION_SCHEMA = "annotated-lexicon-span-gold-session/v1"
SPAN_GOLD_REFERENCE_SCHEMA = "annotated-lexicon-span-gold-reference/v1"
REPAIR_SOURCE_SHA256 = "a4a2d1e7826419a365962ded70806a610345d21a998454adeda8b1f999738565"
APPROVED_ADJUDICATION_ITEM_IDS = (
    "blind-23c8568793c27af8",  # nested 妈宝女 / 妈宝; later policy chooses longest
    "blind-da05e9386b49d061",  # ordinary Cantonese composition; expected no span
)
KNOWN_OMISSIONS: dict[str, tuple[str, ...]] = {
    "blind-564b9e2044612908": ("畒勾", "蝲蝲蛄"),
    "blind-1dfbe73ebdd4a7f7": ("腐女", "瞎bb"),
    "blind-3dae416bb0e6ded1": ("丰县",),
    "blind-4cb44b3e743559bc": ("盖盖",),
    "blind-9eb5c535b4795279": ("褐兰州",),
}


class LexiconRepairError(RuntimeError):
    """Fail-closed repair lifecycle error."""


class LexiconRepairConflict(LexiconRepairError):
    """Review session changed concurrently."""


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8") + b"\n"


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, value: Any, *, mode: int = 0o600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(canonical_bytes(value))
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, mode)
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _find_spans(content: str, surface: str) -> list[list[int]]:
    spans: list[list[int]] = []
    cursor = 0
    while True:
        start = content.find(surface, cursor)
        if start < 0:
            return spans
        spans.append([start, start + len(surface)])
        cursor = start + max(1, len(surface))


def _candidate_id(source_item_id: str, start: int, end: int, surface: str) -> str:
    identity = canonical_sha256([source_item_id, start, end, surface])[:20]
    return f"span-{identity}"


def _span_candidates(item: Mapping[str, Any]) -> list[dict[str, Any]]:
    source_item_id = str(item["item_id"])
    content = str(item["query_content"])
    grouped: dict[tuple[int, int, str], dict[str, Any]] = {}
    for hit in item.get("lexicon_hits", []):
        for span in hit.get("match_spans", []):
            start, end = int(span[0]), int(span[1])
            surface = content[start:end]
            key = (start, end, surface)
            row = grouped.setdefault(
                key,
                {
                    "candidate_id": _candidate_id(source_item_id, start, end, surface),
                    "surface": surface,
                    "span": [start, end],
                    "source_types": [],
                    "source_hits": [],
                },
            )
            if "legacy_hit" not in row["source_types"]:
                row["source_types"].append("legacy_hit")
            provenance = {
                "lexicon_id": str(hit["lexicon_id"]),
                "term": str(hit["term"]),
                "category": str(hit["category"]),
                "definition": str(hit["definition"]),
            }
            if provenance not in row["source_hits"]:
                row["source_hits"].append(provenance)
    for surface in KNOWN_OMISSIONS.get(source_item_id, ()):
        spans = _find_spans(content, surface)
        if not spans:
            raise LexiconRepairError(
                f"known omission {surface!r} is absent from {source_item_id}"
            )
        for start, end in spans:
            key = (start, end, surface)
            row = grouped.setdefault(
                key,
                {
                    "candidate_id": _candidate_id(source_item_id, start, end, surface),
                    "surface": surface,
                    "span": [start, end],
                    "source_types": [],
                    "source_hits": [],
                },
            )
            if "known_omission" not in row["source_types"]:
                row["source_types"].append("known_omission")
    return sorted(
        grouped.values(),
        key=lambda row: (
            int(row["span"][0]),
            -(int(row["span"][1]) - int(row["span"][0])),
            str(row["candidate_id"]),
        ),
    )


def build_span_gold_frame(
    *,
    candidate_frame_path: Path,
    audit_session_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Build the closed 34-reject + related-control span review frame."""

    candidate_frame_path = candidate_frame_path.resolve()
    audit_session_path = audit_session_path.resolve()
    frame = read_json(candidate_frame_path)
    session = read_json(audit_session_path)
    source_hashes = frame.get("manifest", {}).get("source_hashes", {})
    if source_hashes.get("lexicon") != REPAIR_SOURCE_SHA256:
        raise LexiconRepairError("candidate frame is not bound to the frozen source lexicon")
    if session.get("frame_id") != frame.get("manifest", {}).get("frame_id"):
        raise LexiconRepairError("audit session and candidate frame differ")
    if set(session.get("decisions", {})) != {
        str(item["item_id"]) for item in frame.get("items", [])
    }:
        raise LexiconRepairError("audit decision inventory differs")
    if any(
        decision.get("status") != "confirmed"
        for decision in session["decisions"].values()
    ):
        raise LexiconRepairError("source input audit must be fully confirmed")

    rejected = {
        item_id
        for item_id, decision in session["decisions"].items()
        if decision.get("disposition") == "reject"
    }
    if len(rejected) != 34:
        raise LexiconRepairError("source audit must contain exactly 34 rejects")
    item_by_id = {str(item["item_id"]): item for item in frame["items"]}
    affected_terms = {
        str(hit["term"])
        for item_id in rejected
        for hit in item_by_id[item_id].get("lexicon_hits", [])
    }
    related_controls = {
        str(item["item_id"])
        for item in frame["items"]
        if session["decisions"][str(item["item_id"])].get("disposition") == "accept"
        and any(str(hit["term"]) in affected_terms for hit in item.get("lexicon_hits", []))
    }
    selected_ids = rejected | related_controls | set(APPROVED_ADJUDICATION_ITEM_IDS)
    items: list[dict[str, Any]] = []
    for source_item in frame["items"]:
        source_item_id = str(source_item["item_id"])
        if source_item_id not in selected_ids:
            continue
        if source_item_id in rejected:
            cohort = "repair_target"
        elif source_item_id in APPROVED_ADJUDICATION_ITEM_IDS:
            cohort = "approved_adjudication"
        else:
            cohort = "accepted_control"
        decision = session["decisions"][source_item_id]
        candidates = _span_candidates(source_item)
        public_identity = {
            "source_item_id": source_item_id,
            "content_sha256": str(source_item["content_sha256"]),
            "cohort": cohort,
            "candidate_ids": [row["candidate_id"] for row in candidates],
        }
        items.append(
            {
                "item_id": "gold-" + canonical_sha256(public_identity)[:20],
                "source_item_id": source_item_id,
                "cohort": cohort,
                "query_content": str(source_item["query_content"]),
                "content_sha256": str(source_item["content_sha256"]),
                "prior_audit": {
                    "disposition": str(decision["disposition"]),
                    "relevance": decision.get("relevance"),
                    "boundary": decision.get("boundary"),
                    "definition_quality": decision.get("definition_quality"),
                    "sense_fit": decision.get("sense_fit"),
                    "no_hit_verified": decision.get("no_hit_verified"),
                    "notes": str(decision.get("notes", "")),
                },
                "candidates": candidates,
            }
        )
    cohort_order = {
        "repair_target": 0,
        "accepted_control": 1,
        "approved_adjudication": 2,
    }
    items.sort(
        key=lambda row: (
            cohort_order[str(row["cohort"])],
            str(row["source_item_id"]),
        )
    )
    if len(items) != 39:
        raise LexiconRepairError(
            f"span-gold frame expected 39 items, observed {len(items)}"
        )
    identity = {
        "schema_version": SPAN_GOLD_FRAME_SCHEMA,
        "development_only": True,
        "source_candidate_frame_sha256": file_sha256(candidate_frame_path),
        "source_audit_session_sha256": file_sha256(audit_session_path),
        "source_audit_revision": str(session["revision"]),
        "source_lexicon_sha256": REPAIR_SOURCE_SHA256,
        "affected_terms": sorted(affected_terms),
        "cohort_counts": dict(sorted(Counter(row["cohort"] for row in items).items())),
        "item_count": len(items),
        "items_sha256": canonical_sha256(items),
        "single_pass_review": True,
    }
    payload = {
        "manifest": {
            **identity,
            "frame_id": "span-gold-frame-" + canonical_sha256(identity),
        },
        "items": items,
    }
    write_json(output_path, payload)
    return payload


def _session_payload(session: Mapping[str, Any]) -> dict[str, Any]:
    return {key: copy.deepcopy(value) for key, value in session.items() if key != "revision"}


def _with_revision(session: Mapping[str, Any]) -> dict[str, Any]:
    payload = _session_payload(session)
    payload["revision"] = canonical_sha256(payload)
    return payload


@contextmanager
def _session_lock(path: Path) -> Iterator[None]:
    lock_path = path.with_name(path.name + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        with os.fdopen(descriptor, "r+b", closefd=True) as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            yield
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    finally:
        pass


def read_span_gold_session(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise LexiconRepairError("span-gold session is unavailable or unsafe")
    session = read_json(path)
    if session.get("schema_version") != SPAN_GOLD_SESSION_SCHEMA:
        raise LexiconRepairError("span-gold session schema differs")
    if session.get("revision") != canonical_sha256(_session_payload(session)):
        raise LexiconRepairError("span-gold session revision differs")
    return session


def _default_span_decision(item: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "status": "draft",
        "candidate_actions": {
            str(candidate["candidate_id"]): None for candidate in item["candidates"]
        },
        "additional_spans": [],
        "notes": "",
    }


def create_span_gold_session(
    *, frame: Mapping[str, Any], session_path: Path, reviewer_id: str
) -> dict[str, Any]:
    reviewer = reviewer_id.strip()
    if not reviewer or len(reviewer) > 100:
        raise LexiconRepairError("reviewer_id is invalid")
    frame_id = str(frame.get("manifest", {}).get("frame_id", ""))
    if frame.get("manifest", {}).get("schema_version") != SPAN_GOLD_FRAME_SCHEMA:
        raise LexiconRepairError("span-gold frame schema differs")
    with _session_lock(session_path):
        if session_path.exists():
            session = read_span_gold_session(session_path)
            if session.get("frame_id") != frame_id or session.get("reviewer_id") != reviewer:
                raise LexiconRepairError("existing session belongs to another frame/reviewer")
            return session
        session = _with_revision(
            {
                "schema_version": SPAN_GOLD_SESSION_SCHEMA,
                "frame_id": frame_id,
                "frame_sha256": canonical_sha256(frame),
                "reviewer_id": reviewer,
                "single_pass_review": True,
                "decisions": {
                    str(item["item_id"]): _default_span_decision(item)
                    for item in frame["items"]
                },
                "amendments": [],
                "finalized_reference_id": None,
                "created_at": _now_iso(),
                "updated_at": _now_iso(),
            }
        )
        write_json(session_path, session)
        return session


def _normalize_additional_spans(
    query: str, value: Any
) -> list[dict[str, Any]]:
    if not isinstance(value, list) or len(value) > 32:
        raise ValueError("additional_spans must be a list of at most 32 rows")
    result: list[dict[str, Any]] = []
    seen: set[tuple[int, int]] = set()
    for row in value:
        if not isinstance(row, dict) or set(row) != {"start", "end", "surface", "reason"}:
            raise ValueError("additional span fields differ")
        start, end = row.get("start"), row.get("end")
        if not isinstance(start, int) or not isinstance(end, int) or not 0 <= start < end <= len(query):
            raise ValueError("additional span boundaries are invalid")
        surface = str(row.get("surface", ""))
        reason = str(row.get("reason", "")).strip()
        if query[start:end] != surface:
            raise ValueError("additional span surface does not match query boundaries")
        if not reason or len(reason) > 500:
            raise ValueError("additional span requires a concise reason")
        if (start, end) in seen:
            raise ValueError("additional span is duplicated")
        seen.add((start, end))
        result.append({"start": start, "end": end, "surface": surface, "reason": reason})
    return sorted(result, key=lambda row: (row["start"], row["end"], row["surface"]))


def validate_span_decision(
    item: Mapping[str, Any], decision: Any, *, confirm: bool
) -> dict[str, Any]:
    fields = {"candidate_actions", "additional_spans", "notes"}
    if not isinstance(decision, dict) or set(decision) != fields:
        raise ValueError("span-gold decision fields differ")
    expected_ids = {str(row["candidate_id"]) for row in item["candidates"]}
    actions = decision.get("candidate_actions")
    if not isinstance(actions, dict) or set(actions) != expected_ids:
        raise ValueError("candidate action inventory differs")
    if any(value not in {None, "keep", "drop"} for value in actions.values()):
        raise ValueError("candidate action must be keep or drop")
    if confirm and any(value is None for value in actions.values()):
        raise ValueError("every candidate occurrence must be decided")
    notes = str(decision.get("notes", "")).strip()
    if len(notes) > 2000:
        raise ValueError("notes exceed 2000 characters")
    additional = _normalize_additional_spans(
        str(item["query_content"]), decision.get("additional_spans")
    )
    expected: list[tuple[int, int, str]] = []
    for candidate in item["candidates"]:
        if actions[str(candidate["candidate_id"])] == "keep":
            expected.append(
                (
                    int(candidate["span"][0]),
                    int(candidate["span"][1]),
                    str(candidate["surface"]),
                )
            )
    expected.extend(
        (row["start"], row["end"], row["surface"]) for row in additional
    )
    if len({(start, end) for start, end, _surface in expected}) != len(expected):
        raise ValueError("expected spans contain duplicate boundaries")
    ordered = sorted(expected)
    for left, right in zip(ordered, ordered[1:]):
        if left[1] > right[0]:
            raise ValueError("expected spans must not overlap")
    return {
        "candidate_actions": {key: actions[key] for key in sorted(actions)},
        "additional_spans": additional,
        "notes": notes,
    }


class SpanGoldReviewStore:
    """CAS-backed storage for one immutable span-gold frame."""

    def __init__(self, *, frame_path: Path, session_path: Path, reviewer_id: str) -> None:
        self.frame_path = frame_path.resolve()
        self.session_path = session_path.resolve()
        self.frame = read_json(self.frame_path)
        if self.frame.get("manifest", {}).get("schema_version") != SPAN_GOLD_FRAME_SCHEMA:
            raise LexiconRepairError("span-gold frame schema differs")
        if self.frame.get("manifest", {}).get("items_sha256") != canonical_sha256(
            self.frame.get("items", [])
        ):
            raise LexiconRepairError("span-gold frame items hash differs")
        self.item_by_id = {str(item["item_id"]): item for item in self.frame["items"]}
        if len(self.item_by_id) != len(self.frame["items"]):
            raise LexiconRepairError("span-gold item IDs are duplicated")
        create_span_gold_session(
            frame=self.frame,
            session_path=self.session_path,
            reviewer_id=reviewer_id,
        )

    def _status(self, session: Mapping[str, Any]) -> dict[str, Any]:
        confirmed = sum(
            decision["status"] == "confirmed"
            for decision in session["decisions"].values()
        )
        return {
            "frame_id": str(self.frame["manifest"]["frame_id"]),
            "revision": str(session["revision"]),
            "reviewer_id": str(session["reviewer_id"]),
            "item_count": len(self.item_by_id),
            "confirmed_count": confirmed,
            "open_count": len(self.item_by_id) - confirmed,
            "amendment_count": len(session["amendments"]),
            "finalized_reference_id": session["finalized_reference_id"],
        }

    def _summary(self, session: Mapping[str, Any], item_id: str) -> dict[str, Any]:
        item = self.item_by_id[item_id]
        decision = session["decisions"][item_id]
        return {
            "item_id": item_id,
            "source_item_id": str(item["source_item_id"]),
            "cohort": str(item["cohort"]),
            "query_preview": str(item["query_content"])[:80],
            "candidate_count": len(item["candidates"]),
            "surfaces": [str(row["surface"]) for row in item["candidates"]],
            "status": str(decision["status"]),
        }

    def bootstrap(self) -> dict[str, Any]:
        with _session_lock(self.session_path):
            session = read_span_gold_session(self.session_path)
            return {
                "schema_version": "annotated-lexicon-span-gold-bootstrap/v1",
                "frame_id": str(self.frame["manifest"]["frame_id"]),
                "revision": str(session["revision"]),
                "status": self._status(session),
                "items": [
                    self._summary(session, str(item["item_id"]))
                    for item in self.frame["items"]
                ],
                "warnings": [
                    "DEVELOPMENT ONLY / NON-SEALED / NON-SCIENTIFIC",
                    "Single-pass exact span review; model outputs and gold labels are absent.",
                ],
            }

    def item_state(self, item_id: str) -> dict[str, Any]:
        if item_id not in self.item_by_id:
            raise ValueError("unknown span-gold item")
        with _session_lock(self.session_path):
            session = read_span_gold_session(self.session_path)
            return {
                "schema_version": "annotated-lexicon-span-gold-item/v1",
                "revision": str(session["revision"]),
                "item": copy.deepcopy(self.item_by_id[item_id]),
                "decision": copy.deepcopy(session["decisions"][item_id]),
                "item_summary": self._summary(session, item_id),
            }

    def save(
        self,
        *,
        expected_revision: str,
        item_id: str,
        decision: Mapping[str, Any],
        confirm: bool,
    ) -> dict[str, Any]:
        if item_id not in self.item_by_id:
            raise ValueError("unknown span-gold item")
        normalized = validate_span_decision(
            self.item_by_id[item_id], decision, confirm=confirm
        )
        with _session_lock(self.session_path):
            session = read_span_gold_session(self.session_path)
            if session["revision"] != expected_revision:
                raise LexiconRepairConflict("span-gold session changed concurrently")
            if session["finalized_reference_id"] is not None:
                raise ValueError("finalized span-gold session is immutable")
            if session["decisions"][item_id]["status"] == "confirmed":
                raise ValueError("confirmed decision must be reopened")
            updated = copy.deepcopy(session)
            updated["decisions"][item_id] = {
                "status": "confirmed" if confirm else "draft",
                **normalized,
            }
            updated["updated_at"] = _now_iso()
            updated = _with_revision(updated)
            write_json(self.session_path, updated)
            return {
                "schema_version": "annotated-lexicon-span-gold-mutation/v1",
                "revision": str(updated["revision"]),
                "status": self._status(updated),
                "decision": copy.deepcopy(updated["decisions"][item_id]),
                "item_summary": self._summary(updated, item_id),
            }

    def reopen(
        self, *, expected_revision: str, item_id: str, reason: str
    ) -> dict[str, Any]:
        reason = reason.strip()
        if item_id not in self.item_by_id or not reason or len(reason) > 1000:
            raise ValueError("reopen request is invalid")
        with _session_lock(self.session_path):
            session = read_span_gold_session(self.session_path)
            if session["revision"] != expected_revision:
                raise LexiconRepairConflict("span-gold session changed concurrently")
            if session["finalized_reference_id"] is not None:
                raise ValueError("finalized span-gold session is immutable")
            if session["decisions"][item_id]["status"] != "confirmed":
                raise ValueError("only confirmed decisions can be reopened")
            updated = copy.deepcopy(session)
            updated["decisions"][item_id]["status"] = "draft"
            updated["amendments"].append(
                {
                    "item_id": item_id,
                    "reason": reason,
                    "prior_revision": str(session["revision"]),
                    "reopened_at": _now_iso(),
                }
            )
            updated["updated_at"] = _now_iso()
            updated = _with_revision(updated)
            write_json(self.session_path, updated)
            return {
                "schema_version": "annotated-lexicon-span-gold-mutation/v1",
                "revision": str(updated["revision"]),
                "status": self._status(updated),
                "decision": copy.deepcopy(updated["decisions"][item_id]),
                "item_summary": self._summary(updated, item_id),
            }

    def snapshot(self) -> dict[str, Any]:
        with _session_lock(self.session_path):
            session = read_span_gold_session(self.session_path)
            payload = {
                "schema_version": "annotated-lexicon-span-gold-review-snapshot/v1",
                "exported_at": _now_iso(),
                "frame": self.frame,
                "session": session,
            }
            payload["checksums"] = {
                "frame": canonical_sha256(self.frame),
                "session": canonical_sha256(session),
            }
            return payload


def _expected_spans(item: Mapping[str, Any], decision: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    actions = decision["candidate_actions"]
    for candidate in item["candidates"]:
        if actions[str(candidate["candidate_id"])] == "keep":
            rows.append(
                {
                    "surface": str(candidate["surface"]),
                    "span": [int(candidate["span"][0]), int(candidate["span"][1])],
                    "source_candidate_id": str(candidate["candidate_id"]),
                }
            )
    rows.extend(
        {
            "surface": str(row["surface"]),
            "span": [int(row["start"]), int(row["end"])],
            "source_candidate_id": None,
            "reason": str(row["reason"]),
        }
        for row in decision["additional_spans"]
    )
    return sorted(rows, key=lambda row: (row["span"][0], row["span"][1], row["surface"]))


def finalize_span_gold(
    *, frame_path: Path, session_path: Path, output_path: Path
) -> dict[str, Any]:
    store = SpanGoldReviewStore(
        frame_path=frame_path,
        session_path=session_path,
        reviewer_id=str(read_span_gold_session(session_path)["reviewer_id"]),
    )
    with _session_lock(store.session_path):
        session = read_span_gold_session(store.session_path)
        if session["finalized_reference_id"] is not None:
            if output_path.is_file():
                reference = read_json(output_path)
                if reference.get("reference_id") == session["finalized_reference_id"]:
                    return reference
            raise LexiconRepairError("session is already finalized to another reference")
        if any(row["status"] != "confirmed" for row in session["decisions"].values()):
            raise LexiconRepairError("every span-gold item must be confirmed")
        rows = []
        for item in store.frame["items"]:
            decision = session["decisions"][str(item["item_id"])]
            rows.append(
                {
                    "item_id": str(item["item_id"]),
                    "source_item_id": str(item["source_item_id"]),
                    "cohort": str(item["cohort"]),
                    "content_sha256": str(item["content_sha256"]),
                    "query_content": str(item["query_content"]),
                    "expected_spans": _expected_spans(item, decision),
                    "review_notes": str(decision["notes"]),
                }
            )
        identity = {
            "schema_version": SPAN_GOLD_REFERENCE_SCHEMA,
            "frame_id": str(store.frame["manifest"]["frame_id"]),
            "frame_sha256": canonical_sha256(store.frame),
            "review_session_revision": str(session["revision"]),
            "reviewer_id": str(session["reviewer_id"]),
            "source_lexicon_sha256": REPAIR_SOURCE_SHA256,
            "item_count": len(rows),
            "rows_sha256": canonical_sha256(rows),
            "development_only": True,
        }
        reference_id = "span-gold-ref-" + canonical_sha256(identity)
        reference = {**identity, "reference_id": reference_id, "rows": rows}
        write_json(output_path, reference)
        updated = copy.deepcopy(session)
        updated["finalized_reference_id"] = reference_id
        updated["updated_at"] = _now_iso()
        updated = _with_revision(updated)
        write_json(store.session_path, updated)
        return reference


__all__ = [
    "APPROVED_ADJUDICATION_ITEM_IDS",
    "KNOWN_OMISSIONS",
    "LexiconRepairConflict",
    "LexiconRepairError",
    "REPAIR_SOURCE_SHA256",
    "SPAN_GOLD_FRAME_SCHEMA",
    "SPAN_GOLD_REFERENCE_SCHEMA",
    "SPAN_GOLD_SESSION_SCHEMA",
    "SpanGoldReviewStore",
    "build_span_gold_frame",
    "canonical_bytes",
    "canonical_sha256",
    "create_span_gold_session",
    "file_sha256",
    "finalize_span_gold",
    "read_json",
    "read_span_gold_session",
    "validate_span_decision",
    "write_json",
]
