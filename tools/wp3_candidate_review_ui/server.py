#!/usr/bin/env python3
"""Serve the local-only, two-phase WP3 S2.1 development reviewer."""

from __future__ import annotations

import hashlib
import io
import ipaddress
import json
import secrets
import sys
import threading
import zipfile
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import parse_qs, urlsplit


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from build_lex.terminology_candidate_review import (  # noqa: E402
    CandidateReviewConflict,
    CandidateReviewError,
    _read_session,
    create_review_session,
    lock_raw_phase,
    reopen_review_item,
    review_session_status,
    save_diagnostic_decision,
    save_raw_annotation,
    validate_development_frame,
)
from data.training_artifacts import canonical_json_bytes, load_json  # noqa: E402


STATE_SCHEMA_VERSION = "wp3-s21-review-web-state/v1"
BOOTSTRAP_SCHEMA_VERSION = "wp3-s21-review-bootstrap/v1"
CASE_SCHEMA_VERSION = "wp3-s21-review-case/v1"
MUTATION_SCHEMA_VERSION = "wp3-s21-review-mutation/v1"
EXPORT_SCHEMA_VERSION = "wp3-s21-review-export/v1"
SEARCH_SCHEMA_VERSION = "wp3-s21-review-search/v1"
MAX_REQUEST_BYTES = 128 * 1024
MAX_SEARCH_QUERY_CHARS = 80
MAX_SEARCH_RESULTS = 424
SEARCH_SNIPPET_CHARS = 96
SEARCH_MODES = frozenset({"literal", "all_terms", "fuzzy"})
ASSET_NAMES = frozenset({"index.html", "core.js", "app.js", "styles.css"})


class WebReviewError(RuntimeError):
    """Safe request-edge error."""


def _canonical_wire(value: Any) -> bytes:
    return canonical_json_bytes(value) + b"\n"


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _casefold_with_offsets(value: str) -> tuple[str, list[int]]:
    """Return a casefolded string and map each folded code point to its source."""

    chunks: list[str] = []
    offsets: list[int] = []
    for offset, character in enumerate(value):
        folded = character.casefold()
        chunks.append(folded)
        offsets.extend([offset] * len(folded))
    return "".join(chunks), offsets


def _literal_span(value: str, query: str) -> tuple[int, int] | None:
    folded_value, offsets = _casefold_with_offsets(value)
    folded_query = query.casefold()
    if not folded_query:
        return None
    folded_start = folded_value.find(folded_query)
    if folded_start < 0:
        return None
    folded_end = folded_start + len(folded_query)
    return offsets[folded_start], offsets[folded_end - 1] + 1


def _best_fuzzy_span(value: str, query: str) -> tuple[int, int, int] | None:
    """Find the minimum-edit-distance occurrence of query inside value.

    The dynamic program uses a zero-cost text prefix, which computes a local
    (substring) Levenshtein match in O(len(value) * len(query)).  Source offsets
    are retained so Unicode case folding cannot corrupt the returned span.
    """

    folded_value, offsets = _casefold_with_offsets(value)
    folded_query = query.casefold()
    if not folded_value or not folded_query:
        return None

    width = len(folded_value)
    previous_cost = [0] * (width + 1)
    previous_start = list(range(width + 1))
    for query_offset, query_character in enumerate(folded_query, start=1):
        current_cost = [query_offset] + [0] * width
        current_start = [0] * (width + 1)
        for text_offset, text_character in enumerate(folded_value, start=1):
            cost = previous_cost[text_offset - 1] + (
                query_character != text_character
            )
            start = previous_start[text_offset - 1]
            deletion_cost = previous_cost[text_offset] + 1
            deletion_start = previous_start[text_offset]
            if deletion_cost < cost or (
                deletion_cost == cost and deletion_start < start
            ):
                cost = deletion_cost
                start = deletion_start
            insertion_cost = current_cost[text_offset - 1] + 1
            insertion_start = current_start[text_offset - 1]
            if insertion_cost < cost or (
                insertion_cost == cost and insertion_start < start
            ):
                cost = insertion_cost
                start = insertion_start
            current_cost[text_offset] = cost
            current_start[text_offset] = start
        previous_cost = current_cost
        previous_start = current_start

    best: tuple[int, int, int, int] | None = None
    query_length = len(folded_query)
    for folded_end in range(1, width + 1):
        folded_start = previous_start[folded_end]
        if folded_start >= folded_end:
            continue
        distance = previous_cost[folded_end]
        length_delta = abs((folded_end - folded_start) - query_length)
        candidate = (distance, length_delta, folded_start, folded_end)
        if best is None or candidate < best:
            best = candidate
    if best is None:
        return None
    distance, _, folded_start, folded_end = best
    return offsets[folded_start], offsets[folded_end - 1] + 1, distance


def _search_snippet(
    value: str,
    start: int,
    end: int,
    *,
    force_elision: bool = False,
) -> tuple[str, int, int]:
    """Build a bounded snippet and return its snippet-relative match span."""

    span_length = end - start
    context_budget = max(0, SEARCH_SNIPPET_CHARS - span_length)
    left_context = min(start, context_budget // 2)
    right_context = min(len(value) - end, context_budget - left_context)
    left_context = min(start, context_budget - right_context)
    window_start = start - left_context
    window_end = end + right_context
    if force_elision and window_start == 0 and window_end == len(value):
        # Frozen content must remain a short locator, never a replacement for
        # the per-Case projection.  Drop one non-match edge where possible;
        # an exact whole-content query is clipped by one code point as well.
        if start > 0:
            window_start = 1
        elif end < len(value):
            window_end = max(end, len(value) - 1)
        elif len(value) > 1:
            window_end = len(value) - 1
        else:
            return "…", 0, 0
    prefix = "…" if window_start > 0 else ""
    suffix = "…" if window_end < len(value) else ""
    snippet = prefix + value[window_start:window_end] + suffix
    visible_match_start = max(start, window_start)
    visible_match_end = min(end, window_end)
    relative_start = len(prefix) + visible_match_start - window_start
    relative_end = len(prefix) + visible_match_end - window_start
    return snippet, relative_start, relative_end


def _loopback_authority(value: str) -> bool:
    if not value or len(value) > 255:
        return False
    try:
        parsed = urlsplit("//" + value)
        hostname = parsed.hostname
        _ = parsed.port
    except ValueError:
        return False
    if (
        hostname is None
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path
        or parsed.query
        or parsed.fragment
    ):
        return False
    if hostname.casefold() == "localhost":
        return True
    try:
        return ipaddress.ip_address(hostname).is_loopback
    except ValueError:
        return False


def _loopback_origin(value: str) -> bool:
    try:
        parsed = urlsplit(value)
        _ = parsed.port
    except ValueError:
        return False
    return (
        parsed.scheme == "http"
        and not parsed.path
        and not parsed.query
        and not parsed.fragment
        and _loopback_authority(parsed.netloc)
    )


def _loopback_host(value: str) -> str:
    if value == "localhost":
        return value
    try:
        address = ipaddress.ip_address(value)
    except ValueError as exc:
        raise WebReviewError("host must be localhost or a loopback IP") from exc
    if not address.is_loopback:
        raise WebReviewError("review service may bind only to loopback")
    return value


class ReviewService:
    """Own one fixed frame and one CAS-protected working session."""

    def __init__(
        self,
        *,
        workspace_root: Path,
        frame_dir: Path,
        generator_config_path: Path,
        session_path: Path,
        reviewer_id: str,
    ) -> None:
        self.workspace_root = workspace_root.resolve()
        self.frame_dir = frame_dir.resolve()
        self.generator_config_path = generator_config_path.resolve()
        self.session_path = session_path.resolve()
        self.asset_root = Path(__file__).resolve().parent
        self.session_path.parent.mkdir(parents=True, exist_ok=True)
        self.frame = validate_development_frame(
            self.frame_dir,
            workspace_root=self.workspace_root,
            generator_config_path=self.generator_config_path,
        )
        self.cases = self.frame["cases"]
        self.case_by_id = {str(row["case_id"]): row for row in self.cases}
        self.proposal_rows = self.frame["proposal_rows"]
        self.proposals_by_case = {
            str(row["case_id"]): row["proposals"] for row in self.proposal_rows
        }
        self.session_token = secrets.token_urlsafe(32)
        self.allowed_hosts: set[str] = set()
        self.allowed_origins: set[str] = set()
        self._lock = threading.RLock()
        create_review_session(
            frame_dir=self.frame_dir,
            session_path=self.session_path,
            reviewer_id=reviewer_id,
            workspace_root=self.workspace_root,
            generator_config_path=self.generator_config_path,
        )

    def configure_network(self, port: int, public_origin: str | None = None) -> None:
        self.allowed_hosts = {
            f"127.0.0.1:{port}",
            f"localhost:{port}",
            f"[::1]:{port}",
        }
        self.allowed_origins = {
            f"http://127.0.0.1:{port}",
            f"http://localhost:{port}",
            f"http://[::1]:{port}",
        }
        if public_origin is not None:
            try:
                parsed = urlsplit(public_origin)
                _ = parsed.port
            except ValueError as exc:
                raise WebReviewError("public origin is invalid") from exc
            if (
                parsed.scheme != "https"
                or not parsed.hostname
                or parsed.username is not None
                or parsed.password is not None
                or parsed.path
                or parsed.query
                or parsed.fragment
            ):
                raise WebReviewError(
                    "public origin must be an https origin without a path"
                )
            self.allowed_hosts.add(parsed.netloc)
            self.allowed_origins.add(public_origin)

    @staticmethod
    def _codes() -> dict[str, Any]:
        return {
            "reason_codes": [
                "stable_core_candidate",
                "context_required",
                "evidence_required",
                "transparent",
                "ordinary_identity_or_name",
                "generic_insult",
                "fragment",
                "function_word_attached",
                "sentence_level",
                "wrong_boundary",
                "substring_projection",
                "unsupported_sense",
                "one_off_creation",
                "ambiguous_surface",
                "context_polysemy",
                "quoted_or_reclaimed",
                "evidence_conflict",
                "variant_unresolved",
                "non_contiguous_unresolved",
                "label_derived",
                "not_fit_attested",
                "duplicate",
                "no_neutral_gloss",
                "other",
            ],
            "provisional_routes": [
                "A_candidate",
                "B_candidate",
                "C_candidate",
            ],
            "proposal_actions": [
                "accept",
                "trim",
                "expand",
                "split",
                "reject",
                "defer",
            ],
            "warnings": [
                "DEVELOPMENT ONLY / NON-SEALED / NON-SCIENTIFIC",
                "Provisional A/B/C routes are development hypotheses, not evidence tiers.",
            ],
        }

    def _status(
        self,
        session: Mapping[str, Any],
        *,
        reveal_locked_diagnostic: bool = False,
    ) -> dict[str, Any]:
        raw_confirmed = sum(
            isinstance(row, Mapping) and row.get("status") == "confirmed"
            for row in session["raw_annotations"].values()
        )
        diagnostic_confirmed = sum(
            isinstance(row, Mapping) and row.get("status") == "confirmed"
            for row in session["diagnostic_decisions"].values()
        )
        proposal_count = sum(len(row["proposals"]) for row in self.proposal_rows)
        diagnostic_total = (
            proposal_count
            if session["phase"] == "diagnostic" or reveal_locked_diagnostic
            else 0
        )
        return {
            "frame_id": session["frame_id"],
            "reviewer_id": session["reviewer_id"],
            "phase": session["phase"],
            "revision": session["revision"],
            "raw": {"confirmed": raw_confirmed, "total": len(self.cases)},
            "diagnostic": {
                "confirmed": diagnostic_confirmed,
                "total": diagnostic_total,
            },
            "amendment_count": len(session["amendments"]),
            "finalized_gold_id": session["finalized_gold_id"],
        }

    def _case_summary(
        self, session: Mapping[str, Any], case_id: str
    ) -> dict[str, Any]:
        case = self.case_by_id[case_id]
        raw = session["raw_annotations"][case_id]
        proposals = self.proposals_by_case[case_id]
        decisions = session["diagnostic_decisions"]
        diagnostic_confirmed = 0
        diagnostic_draft = 0
        if session["phase"] == "diagnostic":
            for proposal in proposals:
                decision = decisions.get(f"{case_id}:{proposal['proposal_id']}")
                if isinstance(decision, Mapping):
                    if decision.get("status") == "confirmed":
                        diagnostic_confirmed += 1
                    else:
                        diagnostic_draft += 1
        total = len(proposals)
        complete = (
            raw.get("status") == "confirmed"
            if session["phase"] == "raw"
            else diagnostic_confirmed == total
        )
        return {
            "case_id": case_id,
            "blind_alias": case["blind_alias"],
            "raw_status": raw.get("status"),
            "diagnostic": {
                "confirmed": diagnostic_confirmed,
                "draft": diagnostic_draft,
                "total": total if session["phase"] == "diagnostic" else 0,
            },
            "complete": complete,
        }

    def _project_case(
        self, session: Mapping[str, Any], case_id: str
    ) -> dict[str, Any]:
        if case_id not in self.case_by_id:
            raise WebReviewError("unknown development case")
        case = self.case_by_id[case_id]
        projected = {
            "case_id": case_id,
            "blind_alias": case["blind_alias"],
            "content": case["content"],
            "raw_annotation": session["raw_annotations"][case_id],
        }
        if session["phase"] == "diagnostic":
            projected["proposals"] = self.proposals_by_case[case_id]
            projected["diagnostic_decisions"] = {
                proposal["proposal_id"]: session["diagnostic_decisions"].get(
                    f"{case_id}:{proposal['proposal_id']}"
                )
                for proposal in self.proposals_by_case[case_id]
            }
        return projected

    def _bootstrap(self, session: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "schema_version": BOOTSTRAP_SCHEMA_VERSION,
            "session_token": self.session_token,
            "frame_id": self.frame["frame_id"],
            "reviewer_id": session["reviewer_id"],
            "phase": session["phase"],
            "revision": session["revision"],
            "status": self._status(session),
            "case_summaries": [
                self._case_summary(session, str(case["case_id"]))
                for case in self.cases
            ],
            **self._codes(),
        }

    def bootstrap(self) -> dict[str, Any]:
        with self._lock:
            return self._bootstrap(_read_session(self.session_path))

    @staticmethod
    def _search_match(
        case: Mapping[str, Any],
        field: str,
        span: tuple[int, int],
        *,
        distance: int = 0,
    ) -> dict[str, Any]:
        value = str(case[field])
        snippet, match_start, match_end = _search_snippet(
            value,
            span[0],
            span[1],
            force_elision=field == "content",
        )
        return {
            "case_id": str(case["case_id"]),
            "matched_field": field,
            "snippet": snippet,
            "match_start": match_start,
            "match_end": match_end,
            "distance": distance,
        }

    def search_cases(self, query: str, mode: str) -> dict[str, Any]:
        if not isinstance(query, str):
            raise WebReviewError("search query is invalid")
        if len(query) > MAX_SEARCH_QUERY_CHARS:
            raise WebReviewError(
                f"search query must be at most {MAX_SEARCH_QUERY_CHARS} characters"
            )
        query = query.strip()
        if not query:
            raise WebReviewError("search query must not be empty")
        if mode not in SEARCH_MODES:
            raise WebReviewError("search mode is invalid")

        fields = ("case_id", "blind_alias", "content")
        matches: list[dict[str, Any]] = []
        if mode == "literal":
            for case in self.cases:
                for field in fields:
                    span = _literal_span(str(case[field]), query)
                    if span is not None:
                        matches.append(self._search_match(case, field, span))
                        break
        elif mode == "all_terms":
            terms: list[str] = []
            seen_terms: set[str] = set()
            for term in query.split():
                folded = term.casefold()
                if folded not in seen_terms:
                    terms.append(term)
                    seen_terms.add(folded)
            if not terms:
                raise WebReviewError("search query must contain a term")
            for case in self.cases:
                term_matches: list[tuple[str, tuple[int, int]]] = []
                for term in terms:
                    term_match = next(
                        (
                            (field, span)
                            for field in fields
                            if (
                                span := _literal_span(str(case[field]), term)
                            )
                            is not None
                        ),
                        None,
                    )
                    if term_match is None:
                        break
                    term_matches.append(term_match)
                else:
                    representative = next(
                        (
                            term_match
                            for term_match in term_matches
                            if term_match[0] == "content"
                        ),
                        term_matches[0],
                    )
                    matches.append(
                        self._search_match(
                            case, representative[0], representative[1]
                        )
                    )
        else:
            if len(query) < 3:
                return self.search_cases(query, "literal") | {"mode": mode}
            threshold = min(4, max(1, int(len(query) * 0.2)))
            ranked: list[tuple[int, float, int, dict[str, Any]]] = []
            folded_query_length = max(1, len(query.casefold()))
            for frame_offset, case in enumerate(self.cases):
                identifier_match = next(
                    (
                        (field, span)
                        for field in ("case_id", "blind_alias")
                        if (
                            span := _literal_span(str(case[field]), query)
                        )
                        is not None
                    ),
                    None,
                )
                if identifier_match is not None:
                    match = self._search_match(
                        case, identifier_match[0], identifier_match[1]
                    )
                    ranked.append((0, 0.0, frame_offset, match))
                    continue
                fuzzy = _best_fuzzy_span(str(case["content"]), query)
                if fuzzy is None or fuzzy[2] > threshold:
                    continue
                match = self._search_match(
                    case,
                    "content",
                    (fuzzy[0], fuzzy[1]),
                    distance=fuzzy[2],
                )
                ranked.append(
                    (
                        1,
                        fuzzy[2] / folded_query_length,
                        frame_offset,
                        match,
                    )
                )
            ranked.sort(key=lambda item: item[:3])
            matches = [item[3] for item in ranked]

        return {
            "schema_version": SEARCH_SCHEMA_VERSION,
            "frame_id": self.frame["frame_id"],
            "query": query,
            "mode": mode,
            "matches": matches[:MAX_SEARCH_RESULTS],
        }

    def case_state(self, case_id: str) -> dict[str, Any]:
        with self._lock:
            session = _read_session(self.session_path)
            return {
                "schema_version": CASE_SCHEMA_VERSION,
                "revision": session["revision"],
                "phase": session["phase"],
                "case": self._project_case(session, case_id),
                "case_summary": self._case_summary(session, case_id),
            }

    def _mutation(
        self,
        session: Mapping[str, Any],
        case_id: str,
        *,
        raw_annotation: Mapping[str, Any] | None = None,
        proposal_id: str | None = None,
        diagnostic_decision: Mapping[str, Any] | None = None,
        include_case: bool = False,
    ) -> dict[str, Any]:
        result: dict[str, Any] = {
            "schema_version": MUTATION_SCHEMA_VERSION,
            "revision": session["revision"],
            "phase": session["phase"],
            "status": self._status(session),
            "case_summary": self._case_summary(session, case_id),
            "case_id": case_id,
        }
        if raw_annotation is not None:
            result["raw_annotation"] = raw_annotation
        if proposal_id is not None:
            result["proposal_id"] = proposal_id
            result["diagnostic_decision"] = diagnostic_decision
        if include_case:
            result["case"] = self._project_case(session, case_id)
        return result

    def state(self) -> dict[str, Any]:
        with self._lock:
            session = _read_session(self.session_path)
            status = self._status(session, reveal_locked_diagnostic=True)
            cases = [
                self._project_case(session, str(case["case_id"]))
                for case in self.cases
            ]
            return {
                "schema_version": STATE_SCHEMA_VERSION,
                "session_token": self.session_token,
                "frame_id": self.frame["frame_id"],
                "reviewer_id": session["reviewer_id"],
                "phase": session["phase"],
                "revision": session["revision"],
                "status": status,
                "cases": cases,
                **self._codes(),
            }

    def save_raw(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        required = {
            "session_token",
            "expected_revision",
            "case_id",
            "annotation",
            "confirm",
        }
        if set(payload) != required or not isinstance(payload["confirm"], bool):
            raise WebReviewError("raw save fields are invalid")
        with self._lock:
            session = save_raw_annotation(
                frame_dir=self.frame_dir,
                session_path=self.session_path,
                case_id=str(payload["case_id"]),
                annotation=payload["annotation"],
                confirm=payload["confirm"],
                expected_revision=str(payload["expected_revision"]),
            )
            case_id = str(payload["case_id"])
            return self._mutation(
                session,
                case_id,
                raw_annotation=session["raw_annotations"][case_id],
            )

    def lock_raw(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        if set(payload) != {"session_token", "expected_revision"}:
            raise WebReviewError("raw lock fields are invalid")
        with self._lock:
            session = lock_raw_phase(
                session_path=self.session_path,
                expected_revision=str(payload["expected_revision"]),
            )
            return self._bootstrap(session)

    def save_diagnostic(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        required = {
            "session_token",
            "expected_revision",
            "case_id",
            "proposal_id",
            "decision",
            "confirm",
        }
        if set(payload) != required or not isinstance(payload["confirm"], bool):
            raise WebReviewError("diagnostic save fields are invalid")
        with self._lock:
            session = save_diagnostic_decision(
                frame_dir=self.frame_dir,
                session_path=self.session_path,
                case_id=str(payload["case_id"]),
                proposal_id=str(payload["proposal_id"]),
                decision=payload["decision"],
                confirm=payload["confirm"],
                expected_revision=str(payload["expected_revision"]),
            )
            case_id = str(payload["case_id"])
            proposal_id = str(payload["proposal_id"])
            return self._mutation(
                session,
                case_id,
                proposal_id=proposal_id,
                diagnostic_decision=session["diagnostic_decisions"][
                    f"{case_id}:{proposal_id}"
                ],
            )

    def reopen(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        required = {
            "session_token",
            "expected_revision",
            "scope",
            "case_id",
            "proposal_id",
            "reason",
        }
        if set(payload) != required:
            raise WebReviewError("reopen fields are invalid")
        proposal_id = payload["proposal_id"]
        if proposal_id is not None and not isinstance(proposal_id, str):
            raise WebReviewError("proposal_id is invalid")
        with self._lock:
            session = reopen_review_item(
                session_path=self.session_path,
                scope=str(payload["scope"]),
                case_id=str(payload["case_id"]),
                proposal_id=proposal_id,
                reason=str(payload["reason"]),
                expected_revision=str(payload["expected_revision"]),
            )
            return self._mutation(
                session,
                str(payload["case_id"]),
                include_case=True,
            )

    def export_zip(self, payload: Mapping[str, Any]) -> bytes:
        if set(payload) != {"session_token", "expected_revision"}:
            raise WebReviewError("export fields are invalid")
        with self._lock:
            session = _read_session(self.session_path)
            if session["revision"] != payload["expected_revision"]:
                raise CandidateReviewConflict("review session changed concurrently")
            status = review_session_status(
                frame_dir=self.frame_dir, session_path=self.session_path
            )
            annotations = {
                "schema_version": "wp3-s21-review-session-export/v1",
                "frame_id": session["frame_id"],
                "reviewer_id": session["reviewer_id"],
                "phase": session["phase"],
                "revision": session["revision"],
                "raw_annotations": session["raw_annotations"],
                "raw_lock": session["raw_lock"],
                "diagnostic_decisions": session["diagnostic_decisions"],
                "amendments": session["amendments"],
            }
            declaration = {
                "schema_version": "wp3-s21-reviewer-declaration-draft/v1",
                "frame_id": session["frame_id"],
                "reviewer_id": session["reviewer_id"],
                "development_only_nonsealed_nonscientific": True,
                "raw_complete": status["raw"]["confirmed"] == status["raw"]["total"],
                "diagnostic_complete": (
                    status["diagnostic"]["confirmed"]
                    == status["diagnostic"]["total"]
                ),
                "signed": False,
                "note": "finalize-dev-gold validates and emits the immutable declaration",
            }
            files = {
                "annotations.json": _canonical_wire(annotations),
                "reviewer_declaration.json": _canonical_wire(declaration),
            }
            manifest = {
                "schema_version": EXPORT_SCHEMA_VERSION,
                "frame_id": session["frame_id"],
                "session_revision": session["revision"],
                "files": {
                    name: {"sha256": _sha256(wire), "size": len(wire)}
                    for name, wire in sorted(files.items())
                },
            }
            files["manifest.json"] = _canonical_wire(manifest)
            sums = "".join(
                f"{_sha256(wire)}  {name}\n" for name, wire in sorted(files.items())
            ).encode("utf-8")
            files["SHA256SUMS"] = sums
            buffer = io.BytesIO()
            with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
                for name, wire in sorted(files.items()):
                    archive.writestr(name, wire)
            return buffer.getvalue()


class ReviewRequestHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    service: ReviewService

    def log_message(self, format_string: str, *args: Any) -> None:
        sys.stderr.write("[wp3-s21-review] " + (format_string % args) + "\n")

    def _host_allowed(self) -> bool:
        host = self.headers.get("Host", "")
        return host in self.service.allowed_hosts or _loopback_authority(host)

    def _origin_allowed(self) -> bool:
        origin = self.headers.get("Origin")
        return (
            origin is None
            or origin in self.service.allowed_origins
            or _loopback_origin(origin)
        )

    def _headers(self, content_type: str, length: int) -> None:
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(length))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Resource-Policy", "same-origin")
        self.send_header(
            "Permissions-Policy",
            "camera=(), microphone=(), geolocation=(), payment=(), usb=()",
        )
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; script-src 'self'; style-src 'self'; "
            "img-src 'self' data:; connect-src 'self'; object-src 'none'; "
            "base-uri 'none'; frame-ancestors 'none'; form-action 'none'",
        )

    def _send(
        self,
        status: HTTPStatus,
        wire: bytes,
        content_type: str,
        *,
        headers: Mapping[str, str] | None = None,
        head_only: bool = False,
    ) -> None:
        self.send_response(status.value)
        self._headers(content_type, len(wire))
        for key, value in (headers or {}).items():
            self.send_header(key, value)
        self.end_headers()
        if not head_only:
            self.wfile.write(wire)

    def _json(self, status: HTTPStatus, value: Mapping[str, Any]) -> None:
        self._send(status, canonical_json_bytes(value), "application/json; charset=utf-8")

    def _error(self, status: HTTPStatus, message: str) -> None:
        self._json(status, {"status": status.value, "error": message})

    def _asset(self, name: str, *, head_only: bool = False) -> None:
        if name not in ASSET_NAMES:
            self._error(HTTPStatus.NOT_FOUND, "not found")
            return
        path = self.service.asset_root / name
        if path.is_symlink() or not path.is_file():
            self._error(HTTPStatus.INTERNAL_SERVER_ERROR, "UI asset missing")
            return
        content_type = {
            "index.html": "text/html; charset=utf-8",
            "core.js": "text/javascript; charset=utf-8",
            "app.js": "text/javascript; charset=utf-8",
            "styles.css": "text/css; charset=utf-8",
        }[name]
        self._send(
            HTTPStatus.OK,
            path.read_bytes(),
            content_type,
            head_only=head_only,
        )

    def _read_payload(self) -> dict[str, Any]:
        raw_length = self.headers.get("Content-Length")
        try:
            length = int(raw_length or "")
        except ValueError as exc:
            raise WebReviewError("invalid Content-Length") from exc
        if length <= 0 or length > MAX_REQUEST_BYTES:
            raise WebReviewError("request body size is invalid")
        if self.headers.get("Content-Type", "").split(";", 1)[0] != "application/json":
            raise WebReviewError("request must use application/json")
        try:
            value = json.loads(self.rfile.read(length))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise WebReviewError("request JSON is invalid") from exc
        if not isinstance(value, dict):
            raise WebReviewError("request JSON must be an object")
        return value

    def _authorized(self, payload: Mapping[str, Any]) -> bool:
        token = payload.get("session_token")
        return isinstance(token, str) and secrets.compare_digest(
            token, self.service.session_token
        )

    def do_HEAD(self) -> None:  # noqa: N802
        if not self._host_allowed():
            self._error(HTTPStatus.FORBIDDEN, "Host not allowed")
            return
        path = urlsplit(self.path).path
        if path in {"/", "/index.html"}:
            self._asset("index.html", head_only=True)
        elif path in {"/core.js", "/app.js", "/styles.css"}:
            self._asset(path[1:], head_only=True)
        else:
            self._error(HTTPStatus.NOT_FOUND, "not found")

    def do_GET(self) -> None:  # noqa: N802
        if not self._host_allowed():
            self._error(HTTPStatus.FORBIDDEN, "Host not allowed")
            return
        parsed = urlsplit(self.path)
        path = parsed.path
        if path in {"/", "/index.html"}:
            self._asset("index.html")
        elif path in {"/core.js", "/app.js", "/styles.css"}:
            self._asset(path[1:])
        elif path == "/api/health":
            self._json(HTTPStatus.OK, {"status": "ok"})
        elif path == "/api/state":
            try:
                self._json(HTTPStatus.OK, self.service.state())
            except (CandidateReviewError, OSError):
                self._error(HTTPStatus.CONFLICT, "frozen review state is invalid")
        elif path == "/api/bootstrap":
            try:
                self._json(HTTPStatus.OK, self.service.bootstrap())
            except (CandidateReviewError, OSError):
                self._error(HTTPStatus.CONFLICT, "frozen review state is invalid")
        elif path == "/api/cases/search":
            try:
                parameters = parse_qs(parsed.query, keep_blank_values=True)
                if (
                    set(parameters) - {"q", "mode"}
                    or len(parameters.get("q", [])) != 1
                    or len(parameters.get("mode", ["literal"])) != 1
                ):
                    raise WebReviewError("search parameters are invalid")
                query = parameters["q"][0]
                mode = parameters.get("mode", ["literal"])[0]
                self._json(
                    HTTPStatus.OK, self.service.search_cases(query, mode)
                )
            except WebReviewError as exc:
                self._error(HTTPStatus.UNPROCESSABLE_ENTITY, str(exc))
            except (CandidateReviewError, OSError):
                self._error(HTTPStatus.CONFLICT, "frozen review state is invalid")
        elif path.startswith("/api/cases/"):
            case_id = path.removeprefix("/api/cases/")
            if not case_id or "/" in case_id:
                self._error(HTTPStatus.NOT_FOUND, "case not found")
                return
            try:
                self._json(HTTPStatus.OK, self.service.case_state(case_id))
            except WebReviewError:
                self._error(HTTPStatus.NOT_FOUND, "case not found")
            except (CandidateReviewError, OSError):
                self._error(HTTPStatus.CONFLICT, "frozen review state is invalid")
        elif path == "/favicon.ico":
            self._send(HTTPStatus.NO_CONTENT, b"", "image/x-icon")
        else:
            self._error(HTTPStatus.NOT_FOUND, "not found")

    def do_POST(self) -> None:  # noqa: N802
        if not self._host_allowed() or not self._origin_allowed():
            # A rejected HTTP/1.1 request body must never be interpreted as a
            # second request on the same connection.
            self.close_connection = True
            self._error(HTTPStatus.FORBIDDEN, "request origin not allowed")
            return
        try:
            payload = self._read_payload()
            if not self._authorized(payload):
                self.close_connection = True
                self._error(HTTPStatus.FORBIDDEN, "invalid session token")
                return
            path = urlsplit(self.path).path
            if path == "/api/raw":
                self._json(HTTPStatus.OK, self.service.save_raw(payload))
            elif path == "/api/lock-raw":
                self._json(HTTPStatus.OK, self.service.lock_raw(payload))
            elif path == "/api/diagnostic":
                self._json(HTTPStatus.OK, self.service.save_diagnostic(payload))
            elif path == "/api/reopen":
                self._json(HTTPStatus.OK, self.service.reopen(payload))
            elif path == "/api/export":
                archive = self.service.export_zip(payload)
                self._send(
                    HTTPStatus.OK,
                    archive,
                    "application/zip",
                    headers={
                        "Content-Disposition": (
                            'attachment; filename="wp3-s21-development-review.zip"'
                        )
                    },
                )
            else:
                self._error(HTTPStatus.NOT_FOUND, "API route not found")
        except CandidateReviewConflict as exc:
            self._error(HTTPStatus.CONFLICT, str(exc))
        except (CandidateReviewError, WebReviewError, TypeError, ValueError) as exc:
            self._error(HTTPStatus.UNPROCESSABLE_ENTITY, str(exc))
        except OSError:
            self._error(HTTPStatus.CONFLICT, "review state could not be written")
        except Exception as exc:  # pragma: no cover - fail closed at HTTP edge.
            print(
                f"[wp3-s21-review] unexpected failure: {type(exc).__name__}",
                file=sys.stderr,
            )
            self._error(HTTPStatus.INTERNAL_SERVER_ERROR, "unexpected local error")


def run_server(
    *,
    workspace_root: Path,
    frame_dir: Path,
    generator_config_path: Path,
    session_path: Path,
    reviewer_id: str,
    host: str,
    port: int,
    check: bool = False,
    public_origin: str | None = None,
) -> int:
    host = _loopback_host(host)
    if not 0 <= port <= 65535:
        raise WebReviewError("port must be in 0..65535")
    service = ReviewService(
        workspace_root=workspace_root,
        frame_dir=frame_dir,
        generator_config_path=generator_config_path,
        session_path=session_path,
        reviewer_id=reviewer_id,
    )
    if check:
        state = service.state()
        print(
            json.dumps(
                {
                    "frame_id": state["frame_id"],
                    "phase": state["phase"],
                    "raw": state["status"]["raw"],
                    "diagnostic": state["status"]["diagnostic"],
                    "development_only": True,
                    "sealed": False,
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 0
    handler = type(
        "BoundReviewRequestHandler", (ReviewRequestHandler,), {"service": service}
    )
    server = ThreadingHTTPServer((host, port), handler)
    server.daemon_threads = True
    actual_port = int(server.server_address[1])
    service.configure_network(actual_port, public_origin=public_origin)
    print(
        json.dumps(
            {
                "url": f"http://127.0.0.1:{actual_port}/",
                "frame_id": service.frame["frame_id"],
                "session_file": str(service.session_path),
                "development_only": True,
                "sealed": False,
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


__all__ = [
    "MAX_REQUEST_BYTES",
    "MAX_SEARCH_QUERY_CHARS",
    "ReviewRequestHandler",
    "ReviewService",
    "SEARCH_MODES",
    "SEARCH_SCHEMA_VERSION",
    "SEARCH_SNIPPET_CHARS",
    "WebReviewError",
    "_loopback_authority",
    "_loopback_origin",
    "run_server",
]
