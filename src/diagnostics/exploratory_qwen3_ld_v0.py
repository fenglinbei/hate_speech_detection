"""Frozen exploratory Qwen3 L/D context-sensitivity pilot.

This module deliberately lives outside the formal Stage-1 model/generation
registry.  It evaluates an untrained base/instruct model and therefore must not
be registered as M_LD, M_drop, or a formal Stage-1 result.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import random
import re
import secrets
import tempfile
import threading
from collections import Counter, defaultdict
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import urlsplit

from rag.controlled_lexicon_matcher import ControlledLexiconMatcher


SCHEMA_VERSION = "exploratory-qwen3-ld-mechanism-pilot/v0"
FRAME_SCHEMA_VERSION = "exploratory-qwen3-ld-candidate-frame/v0"
FROZEN_FRAME_SCHEMA_VERSION = "exploratory-qwen3-ld-frozen-frame/v0"
CONTEXT_SCHEMA_VERSION = "exploratory-qwen3-ld-context-grid/v0"
SESSION_SCHEMA_VERSION = "exploratory-qwen3-ld-input-audit-session/v1"
LEDGER_SCHEMA_VERSION = "exploratory-qwen3-ld-generation/v0"
MARGIN_SCHEMA_VERSION = "exploratory-qwen3-ld-margin/v0"

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPOSITORY_ROOT / "config/stage1/exploratory_qwen3_ld_mechanism_pilot_v0.json"
DEFAULT_OUTPUT = REPOSITORY_ROOT / "exps/causal_context/stage1_exploratory_qwen3_ld_v0"
DEFAULT_AUDIT_SESSION_NAME = "input_quality_session.json"

PRAGMATIC_PATTERNS = {
    "negation": re.compile(r"不是|并非|不能|不要|别再|不等于|无关"),
    "quote": re.compile(r"[“”「」『』]|所谓|有人说|声称|称作|叫做"),
    "counterspeech": re.compile(r"反对|抵制|举报|歧视|洗不白|不能一棒子打死"),
    "reclaimed": re.compile(r"自称|我们|我也是|自嘲|圈内"),
    "irony": re.compile(r"反讽|讽刺|开玩笑|呵呵|建议|好家伙"),
    "discussion": re.compile(r"讨论|问题|统计|媒体|为什么|什么意思|指的是"),
}
AMBIGUOUS_DEFINITION_RE = re.compile(r"也可|也指|有时|可能|泛指|根据语境|既可|既指|一类|某些|通常")
GROUP_ORDER = ("Racism", "Region", "LGBTQ", "Sexism", "others", "non-hate")


class PilotError(RuntimeError):
    """Fail-closed pilot lifecycle error."""


class PilotReviewConflict(PilotError):
    """Audit session changed concurrently."""


class PilotWebReviewError(PilotError):
    """Safe error at the local HTTP boundary."""


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8") + b"\n"


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_sha256(value: Any) -> str:
    return sha256_bytes(canonical_bytes(value))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any, *, mode: int | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_bytes(value)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        if mode is not None:
            os.chmod(path, mode)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise PilotError(f"{path}:{line_number} is not a JSON object")
            rows.append(value)
    return rows


def append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab", buffering=0) as handle:
        handle.write(canonical_bytes(dict(value)))
        os.fsync(handle.fileno())


def resolve_config(config_path: Path = DEFAULT_CONFIG) -> tuple[dict[str, Any], Path]:
    path = config_path.resolve()
    config = load_json(path)
    if config.get("schema_version") != SCHEMA_VERSION:
        raise PilotError("pilot config schema differs")
    if config.get("scope", {}).get("development_only") is not True:
        raise PilotError("pilot must remain development-only")
    if config.get("scope", {}).get("sealed_test_allowed") is not False:
        raise PilotError("sealed test access must remain forbidden")
    return config, path


def source_path(config: Mapping[str, Any], key: str) -> Path:
    path = (REPOSITORY_ROOT / str(config["sources"][key])).resolve()
    if not path.is_file() and key not in {"model", "tokenizer", "embedding_model"}:
        raise PilotError(f"source {key} is unavailable: {path}")
    return path


def _source_hashes(config: Mapping[str, Any], config_path: Path) -> dict[str, str]:
    return {
        "config": file_sha256(config_path),
        "lexicon": file_sha256(source_path(config, "lexicon")),
        "train": file_sha256(source_path(config, "train")),
        "dev": file_sha256(source_path(config, "dev")),
        "fit_partition": file_sha256(source_path(config, "fit_partition")),
    }


def load_lexicon(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    payload = load_json(source_path(config, "lexicon"))
    rows = payload.get("terms") if isinstance(payload, dict) else None
    if not isinstance(rows, list) or len(rows) != int(config["sources"]["expected_lexicon_count"]):
        raise PilotError("annotated lexicon count differs from frozen config")
    seen: set[tuple[str, str, str]] = set()
    result: list[dict[str, Any]] = []
    for ordinal, raw in enumerate(rows):
        if not isinstance(raw, dict):
            raise PilotError("lexicon term is not an object")
        term = str(raw.get("term", ""))
        category = str(raw.get("category", ""))
        definition = str(raw.get("definition", ""))
        if not term or not category or not definition:
            raise PilotError("lexicon term/category/definition must be non-empty")
        key = (term, category, definition)
        if key in seen:
            continue
        seen.add(key)
        result.append(
            {
                "lexicon_id": str(raw.get("lexicon_id") or f"lex-{ordinal:04d}"),
                "stable_ordinal": ordinal,
                "term": term,
                "category": category,
                "definition": definition,
                **{
                    key: copy.deepcopy(raw[key])
                    for key in ("senses", "variants", "match_policy")
                    if key in raw
                },
            }
        )
    return result


def exact_hits(
    content: str,
    lexicon: Sequence[Mapping[str, Any]],
    top_k: int = 5,
    *,
    lexicon_sha256: str | None = None,
) -> list[dict[str, Any]]:
    controlled = any(
        any(key in entry for key in ("senses", "variants", "match_policy"))
        for entry in lexicon
    )
    if controlled:
        if top_k >= 0:
            raise PilotError("controlled repaired lexicon forbids top-k truncation")
        matcher = ControlledLexiconMatcher(
            lexicon,
            lexicon_sha256=lexicon_sha256 or canonical_sha256(list(lexicon)),
        )
        return [dict(row) for row in matcher.match(content)["selected_hits"]]
    hits: list[dict[str, Any]] = []
    for entry in lexicon:
        term = str(entry["term"])
        start = content.find(term)
        if start < 0:
            continue
        spans: list[list[int]] = []
        cursor = 0
        while True:
            position = content.find(term, cursor)
            if position < 0:
                break
            spans.append([position, position + len(term)])
            cursor = position + max(1, len(term))
        hits.append({**dict(entry), "first_start": start, "match_spans": spans})
    hits.sort(
        key=lambda row: (
            int(row["first_start"]),
            -len(str(row["term"])),
            int(row["stable_ordinal"]),
        )
    )
    return hits[:top_k]


def _gold_groups(row: Mapping[str, Any]) -> set[str]:
    return {
        str(label)
        for quad in row.get("quadruples", [])
        for label in quad.get("targeted_group", [])
    }


def _pragmatic_tags(content: str) -> list[str]:
    return [name for name, pattern in PRAGMATIC_PATTERNS.items() if pattern.search(content)]


def _classify_candidate(row: Mapping[str, Any], hits: Sequence[Mapping[str, Any]]) -> tuple[str, list[str]]:
    content = str(row["content"])
    tags = _pragmatic_tags(content)
    if not hits:
        return "no_hit", tags
    if len(hits) >= 2:
        return "multi_hit", tags
    if tags:
        return "pragmatic", tags
    gold_groups = _gold_groups(row)
    mismatch = any(str(hit["category"]) not in gold_groups for hit in hits)
    ambiguous = any(AMBIGUOUS_DEFINITION_RE.search(str(hit["definition"])) for hit in hits)
    return ("suspicious" if mismatch or ambiguous else "nontransparent"), tags


def _stable_rank(namespace: str, query_id: str, seed: int) -> str:
    return hashlib.sha256(f"{namespace}\0{seed}\0{query_id}".encode()).hexdigest()


def _definition_donor(
    hit: Mapping[str, Any],
    content: str,
    lexicon: Sequence[Mapping[str, Any]],
    token_length: Any | None = None,
) -> dict[str, Any]:
    eligible = [
        row
        for row in lexicon
        if row["term"] != hit["term"]
        and row["category"] != hit["category"]
        and str(row["term"]) not in content
    ]
    if not eligible:
        raise PilotError(f"no DefinitionSwap donor for {hit['lexicon_id']}")
    length = token_length or (lambda value: len(str(value)))
    eligible.sort(
        key=lambda row: (
            abs(length(str(row["definition"])) - length(str(hit["definition"]))),
            int(row["stable_ordinal"]),
        )
    )
    return copy.deepcopy(eligible[0])


def _blind_id(query_id: str, role: str, namespace: str) -> str:
    digest = hashlib.sha256(f"{namespace}\0{role}\0{query_id}".encode()).hexdigest()[:16]
    return f"blind-{digest}"


def _audit_projection(candidate: Mapping[str, Any], namespace: str) -> tuple[dict[str, Any], dict[str, Any]]:
    blind_id = _blind_id(str(candidate["query_id"]), str(candidate["selection_role"]), namespace)
    hits = candidate["lexicon_hits"]
    if hits:
        surface = " / ".join(str(row["term"]) for row in hits)
        detail_parts = []
        for ordinal, row in enumerate(hits, start=1):
            donor = row["definition_swap_donor"]
            detail_parts.append(
                f"命中 {ordinal}: term={row['term']} | span={row['match_spans']} | "
                f"category={row['category']} | definition={row['definition']}\n"
                f"预注册 DefinitionSwap donor: term={donor['term']} | "
                f"category={donor['category']} | definition={donor['definition']}"
            )
    else:
        surface = "无精确词典命中"
        detail_parts = ["精确 substring 检索结果为空；只审核查询内容与 no-hit 判断。"]
    canonical = f"盲化查询 {blind_id}"
    quote = f"{canonical}\n待审核查询：{candidate['content']}\n精确命中：{surface}"
    evidence_id = f"audit-evidence-{blind_id.removeprefix('blind-')}"
    item = {
        "item_id": blind_id,
        "audit_kind": "lex_hit" if hits else "no_hit",
        "query_content": candidate["content"],
        "content_sha256": sha256_bytes(str(candidate["content"]).encode()),
        "hit_count": len(hits),
        "lexicon_hits": [
            {
                "lexicon_id": row["lexicon_id"],
                "term": row["term"],
                "category": row["category"],
                "definition": row["definition"],
                "match_spans": row["match_spans"],
                "definition_swap_donor": {
                    "lexicon_id": row["definition_swap_donor"]["lexicon_id"],
                    "term": row["definition_swap_donor"]["term"],
                    "category": row["definition_swap_donor"]["category"],
                    "definition": row["definition_swap_donor"]["definition"],
                },
            }
            for row in hits
        ],
        "evidence_ids": [evidence_id],
    }
    evidence = {
        "evidence_id": evidence_id,
        "source_id": "exploratory-qwen3-ld-v0-input-audit",
        "publisher": "本地冻结开发输入",
        "source_role": "development_input_audit",
        "acquisition_mode": "local_frozen_artifact",
        "component_id": blind_id,
        "relation_contract": "single-quote-surface-and-canonical/v2",
        "quote": quote,
        "relation_note": (
            "审核命中相关性、边界、定义质量、当前 sense，以及 swap 的自然性/不相容性。"
            "接受=全部满足；驳回备注 fail=...；可追加 tags=...。\n" + "\n\n".join(detail_parts)
        ),
    }
    return item, evidence


def build_candidate_frame(
    *,
    config_path: Path = DEFAULT_CONFIG,
    output_root: Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    config, resolved_config_path = resolve_config(config_path)
    output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    lexicon = load_lexicon(config)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        source_path(config, "tokenizer"), use_fast=True, trust_remote_code=False
    )
    token_length_cache: dict[str, int] = {}

    def token_length(value: str) -> int:
        if value not in token_length_cache:
            token_length_cache[value] = len(tokenizer.encode(value, add_special_tokens=False))
        return token_length_cache[value]

    dev = load_json(source_path(config, "dev"))
    if not isinstance(dev, list) or len(dev) != int(config["sources"]["expected_dev_count"]):
        raise PilotError("dev count differs from frozen config")
    top_k = int(config["frame"]["exact_top_k"])
    lexicon_source_sha256 = file_sha256(source_path(config, "lexicon"))
    seed = int(config["frame"]["seed"])
    namespace = str(config["interventions"]["seed_namespace"])
    by_stratum: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in dev:
        content = str(row["content"])
        hits = exact_hits(
            content,
            lexicon,
            top_k=top_k,
            lexicon_sha256=lexicon_source_sha256,
        )
        enriched_hits = []
        for hit in hits:
            enriched_hits.append(
                {
                    **hit,
                    "definition_swap_donor": _definition_donor(
                        hit, content, lexicon, token_length=token_length
                    ),
                }
            )
        stratum, tags = _classify_candidate(row, enriched_hits)
        by_stratum[stratum].append(
            {
                "query_id": str(row["id"]),
                "content": content,
                "gold": copy.deepcopy(row["quadruples"]),
                "stratum": stratum,
                "automatic_pragmatic_tags": tags,
                "lexicon_hits": enriched_hits,
            }
        )
    selected: list[dict[str, Any]] = []
    inventory: dict[str, Any] = {}
    reserve_multiplier = int(config["frame"]["reserve_multiplier"])
    for stratum, quota_value in config["frame"]["strata"].items():
        quota = int(quota_value)
        pool = sorted(
            by_stratum[stratum],
            key=lambda row: (_stable_rank(stratum, row["query_id"], seed), row["query_id"]),
        )
        required = quota * reserve_multiplier
        if len(pool) < required:
            raise PilotError(f"stratum {stratum} has {len(pool)} rows; needs {required}")
        for ordinal, candidate in enumerate(pool[:required]):
            selected.append(
                {
                    **candidate,
                    "selection_role": "primary" if ordinal < quota else "reserve",
                    "stratum_ordinal": ordinal,
                }
            )
        inventory[stratum] = {"available": len(pool), "quota": quota, "frozen_candidates": required}
    selected.sort(key=lambda row: (list(config["frame"]["strata"]).index(row["stratum"]), row["stratum_ordinal"]))
    audit_items: list[dict[str, Any]] = []
    audit_evidence: list[dict[str, Any]] = []
    private_rows: list[dict[str, Any]] = []
    for candidate in selected:
        item, evidence = _audit_projection(candidate, namespace)
        audit_items.append(item)
        audit_evidence.append(evidence)
        private_rows.append({**candidate, "blind_id": item["item_id"]})
    public_identity = {
        "schema_version": FRAME_SCHEMA_VERSION,
        "experiment_id": config["experiment_id"],
        "source_hashes": _source_hashes(config, resolved_config_path),
        "sampling_policy": copy.deepcopy(config["frame"]),
        "inventory": inventory,
        "audit_items_sha256": canonical_sha256(audit_items),
        "audit_evidence_sha256": canonical_sha256(audit_evidence),
        "candidate_count": len(private_rows),
        "development_only": True,
        "scientific_eligible": False,
    }
    frame_id = f"pilot-frame-{canonical_sha256(public_identity)}"
    manifest = {**public_identity, "frame_id": frame_id}
    frame = {"manifest": manifest, "items": audit_items, "evidence": audit_evidence}
    private = {
        "schema_version": FRAME_SCHEMA_VERSION,
        "frame_id": frame_id,
        "rows": private_rows,
        "private_contains_gold": True,
        "ui_must_not_serve": True,
    }
    write_json(output_root / "candidate_frame.json", frame)
    write_json(output_root / "candidate_private.json", private, mode=0o600)
    result = {
        "frame_id": frame_id,
        "candidate_count": len(private_rows),
        "audit_item_count": len(audit_items),
        "strata": inventory,
        "candidate_frame": str(output_root / "candidate_frame.json"),
        "private_frame": str(output_root / "candidate_private.json"),
    }
    write_json(output_root / "build_frame_receipt.json", result)
    return result


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _session_payload(session: Mapping[str, Any]) -> dict[str, Any]:
    return {key: copy.deepcopy(value) for key, value in session.items() if key != "revision"}


def _with_revision(session: Mapping[str, Any]) -> dict[str, Any]:
    result = _session_payload(session)
    result["revision"] = canonical_sha256(result)
    return result


def _read_audit_session(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise PilotError("audit session is unavailable or unsafe")
    session = load_json(path)
    if session.get("revision") != canonical_sha256(_session_payload(session)):
        raise PilotError("audit session revision differs")
    return session


class PilotInputAuditService:
    """Task-specific input audit using the established WP3/G3 workbench flow."""

    def __init__(self, *, frame_path: Path, session_path: Path, reviewer_id: str) -> None:
        frame = load_json(frame_path)
        if frame.get("manifest", {}).get("schema_version") != FRAME_SCHEMA_VERSION:
            raise PilotError("candidate audit frame schema differs")
        self.frame = {
            "frame_id": frame["manifest"]["frame_id"],
            "payload_manifest_sha256": canonical_sha256(frame),
            "items": frame["items"],
            "evidence": frame["evidence"],
        }
        self.item_by_id = {row["item_id"]: row for row in self.frame["items"]}
        self.evidence_by_id = {row["evidence_id"]: row for row in self.frame["evidence"]}
        self.session_path = session_path.resolve()
        self.asset_root = REPOSITORY_ROOT / "tools/exploratory_qwen3_ld_review_ui"
        self.session_token = secrets.token_urlsafe(32)
        self.allowed_hosts: set[str] = set()
        self.allowed_origins: set[str] = set()
        self._lock = threading.RLock()
        self._Conflict = PilotReviewConflict
        self._WebError = PilotWebReviewError
        self._create_session(reviewer_id)

    def _create_session(self, reviewer_id: str) -> None:
        reviewer = reviewer_id.strip()
        if not reviewer or len(reviewer) > 100:
            raise PilotError("reviewer_id is invalid")
        self.session_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(self.session_path.parent, 0o700)
        if self.session_path.exists():
            session = _read_audit_session(self.session_path)
            if session.get("frame_id") != self.frame["frame_id"] or session.get("reviewer_id") != reviewer:
                raise PilotError("existing audit session belongs to another frame/reviewer")
            return
        decisions = {item["item_id"]: self._default_decision(item) for item in self.frame["items"]}
        session = _with_revision(
            {
                "schema_version": SESSION_SCHEMA_VERSION,
                "frame_id": self.frame["frame_id"],
                "frame_payload_manifest_sha256": self.frame["payload_manifest_sha256"],
                "reviewer_id": reviewer,
                "decisions": decisions,
                "amendments": [],
                "finalized_reference_id": None,
                "created_at": _now_iso(),
                "updated_at": _now_iso(),
            }
        )
        write_json(self.session_path, session, mode=0o600)

    @staticmethod
    def _default_decision(item: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "status": "draft",
            "disposition": "defer",
            "relevance": None,
            "boundary": None,
            "definition_quality": None,
            "sense_fit": None,
            "swap_incompatibility": None,
            "no_hit_verified": None,
            "pragmatic_tags": [],
            "notes": "",
        }

    def configure_network(self, port: int, public_origin: str | None = None) -> None:
        self.allowed_hosts = {f"127.0.0.1:{port}", f"localhost:{port}", f"[::1]:{port}"}
        self.allowed_origins = {
            f"http://127.0.0.1:{port}", f"http://localhost:{port}", f"http://[::1]:{port}"
        }
        if public_origin is not None:
            parsed = urlsplit(public_origin)
            if (
                parsed.scheme != "https"
                or not parsed.hostname
                or parsed.username is not None
                or parsed.password is not None
                or parsed.path
                or parsed.query
                or parsed.fragment
            ):
                raise PilotWebReviewError("public origin must be an exact HTTPS origin without a path")
            self.allowed_hosts.add(parsed.netloc)
            self.allowed_origins.add(public_origin)

    def _status(self, session: Mapping[str, Any]) -> dict[str, Any]:
        confirmed = [row for row in session["decisions"].values() if row["status"] == "confirmed"]
        actions = Counter(str(row["disposition"]) for row in confirmed)
        return {
            "frame_id": self.frame["frame_id"],
            "reviewer_id": session["reviewer_id"],
            "revision": session["revision"],
            "finalized_reference_id": None,
            "item_count": len(self.item_by_id),
            "confirmed_count": len(confirmed),
            "deferred_count": sum(row["disposition"] == "defer" for row in confirmed),
            "amendment_count": len(session["amendments"]),
            "action_counts": dict(sorted(actions.items())),
        }

    def _summary(self, session: Mapping[str, Any], item_id: str) -> dict[str, Any]:
        item = self.item_by_id[item_id]
        decision = session["decisions"][item_id]
        return {
            "item_id": item_id,
            "audit_kind": item["audit_kind"],
            "query_preview": str(item["query_content"])[:80],
            "hit_count": item["hit_count"],
            "terms": [row["term"] for row in item["lexicon_hits"]],
            "status": decision["status"],
            "disposition": decision["disposition"],
        }

    def bootstrap(self) -> dict[str, Any]:
        with self._lock:
            session = _read_audit_session(self.session_path)
            return {
                "schema_version": "exploratory-qwen3-ld-input-audit-bootstrap/v1",
                "session_token": self.session_token,
                "frame_id": self.frame["frame_id"],
                "reviewer_id": session["reviewer_id"],
                "revision": session["revision"],
                "status": self._status(session),
                "dispositions": ["accept", "reject", "defer"],
                "pragmatic_tags": ["quote", "negation", "counterspeech", "reclaimed", "irony", "discussion"],
                "items": [self._summary(session, item["item_id"]) for item in self.frame["items"]],
                "warnings": [
                    "DEVELOPMENT ONLY / NON-SEALED / NON-SCIENTIFIC",
                    "WP3 2.1/G3 workbench behavior adapted to the pilot input-audit fields.",
                    "Gold, demonstrations, conditions, and model outputs are absent.",
                ],
            }

    def item_state(self, item_id: str) -> dict[str, Any]:
        with self._lock:
            if item_id not in self.item_by_id:
                raise self._WebError("unknown audit item")
            session = _read_audit_session(self.session_path)
            item = self.item_by_id[item_id]
            return {
                "schema_version": "exploratory-qwen3-ld-input-audit-item/v1",
                "revision": session["revision"],
                "item": item,
                "evidence": [self.evidence_by_id[value] for value in item["evidence_ids"]],
                "decision": session["decisions"][item_id],
                "item_summary": self._summary(session, item_id),
            }

    def _validate_decision(self, item_id: str, decision: Any, *, confirm: bool) -> dict[str, Any]:
        fields = {
            "disposition", "relevance", "boundary", "definition_quality", "sense_fit",
            "swap_incompatibility", "no_hit_verified", "pragmatic_tags", "notes"
        }
        if not isinstance(decision, dict) or set(decision) != fields:
            raise ValueError("audit decision fields differ")
        item = self.item_by_id[item_id]
        disposition = decision.get("disposition")
        if disposition not in {"accept", "reject", "defer"}:
            raise ValueError("audit disposition is invalid")
        notes = str(decision.get("notes", "")).strip()
        if len(notes) > 2000:
            raise ValueError("audit notes exceed 2000 characters")
        binary = {None, "pass", "fail"}
        for field in ("relevance", "boundary", "sense_fit", "swap_incompatibility", "no_hit_verified"):
            if decision.get(field) not in binary:
                raise ValueError(f"{field} is invalid")
        if decision.get("definition_quality") not in {None, "good", "usable", "poor"}:
            raise ValueError("definition_quality is invalid")
        tags = decision.get("pragmatic_tags")
        allowed_tags = {"quote", "negation", "counterspeech", "reclaimed", "irony", "discussion"}
        if not isinstance(tags, list) or len(tags) != len(set(tags)) or any(tag not in allowed_tags for tag in tags):
            raise ValueError("pragmatic_tags are invalid")
        if item["audit_kind"] == "lex_hit":
            if decision.get("no_hit_verified") is not None:
                raise ValueError("lex-hit item cannot set no_hit_verified")
            core = [decision.get(field) for field in ("relevance", "boundary", "sense_fit", "swap_incompatibility")]
            definition = decision.get("definition_quality")
            if confirm and disposition != "defer" and (None in core or definition is None):
                raise ValueError("all five lex-hit audit dimensions are required")
            failed = "fail" in core or definition == "poor"
        else:
            if any(decision.get(field) is not None for field in ("relevance", "boundary", "definition_quality", "sense_fit", "swap_incompatibility")):
                raise ValueError("no-hit item cannot set lex-hit dimensions")
            if confirm and disposition != "defer" and decision.get("no_hit_verified") is None:
                raise ValueError("no_hit_verified is required")
            failed = decision.get("no_hit_verified") == "fail"
        if confirm and disposition == "accept" and failed:
            raise ValueError("accept requires every applicable audit dimension to pass")
        if confirm and disposition == "reject" and not failed:
            raise ValueError("reject requires at least one failed audit dimension")
        if confirm and disposition == "reject" and not notes:
            raise ValueError("reject requires a concise note")
        return {key: (notes if key == "notes" else copy.deepcopy(decision[key])) for key in fields}

    def save(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        required = {"session_token", "expected_revision", "item_id", "decision", "confirm"}
        if set(payload) != required or not isinstance(payload.get("confirm"), bool):
            raise ValueError("audit save fields differ")
        item_id = str(payload["item_id"])
        if item_id not in self.item_by_id:
            raise ValueError("unknown audit item")
        normalized = self._validate_decision(item_id, payload["decision"], confirm=bool(payload["confirm"]))
        with self._lock:
            session = _read_audit_session(self.session_path)
            if session["revision"] != str(payload["expected_revision"]):
                raise self._Conflict("audit session changed concurrently")
            if session["decisions"][item_id]["status"] == "confirmed":
                raise ValueError("confirmed audit decision must be reopened")
            updated = copy.deepcopy(session)
            updated["decisions"][item_id] = {
                "status": "confirmed" if payload["confirm"] else "draft", **normalized
            }
            updated["updated_at"] = _now_iso()
            updated = _with_revision(updated)
            write_json(self.session_path, updated, mode=0o600)
            return {
                "schema_version": "exploratory-qwen3-ld-input-audit-mutation/v1",
                "revision": updated["revision"],
                "status": self._status(updated),
                "decision": updated["decisions"][item_id],
                "item_summary": self._summary(updated, item_id),
            }

    def reopen(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        required = {"session_token", "expected_revision", "item_id", "reason"}
        if set(payload) != required:
            raise ValueError("audit reopen fields differ")
        item_id = str(payload["item_id"])
        reason = str(payload["reason"]).strip()
        if not reason or len(reason) > 1000:
            raise ValueError("amendment reason is invalid")
        with self._lock:
            session = _read_audit_session(self.session_path)
            if session["revision"] != str(payload["expected_revision"]):
                raise self._Conflict("audit session changed concurrently")
            if item_id not in session["decisions"] or session["decisions"][item_id]["status"] != "confirmed":
                raise ValueError("only a confirmed audit decision can be reopened")
            updated = copy.deepcopy(session)
            updated["decisions"][item_id]["status"] = "draft"
            updated["amendments"].append(
                {"item_id": item_id, "reason": reason, "prior_revision": session["revision"], "reopened_at": _now_iso()}
            )
            updated["updated_at"] = _now_iso()
            updated = _with_revision(updated)
            write_json(self.session_path, updated, mode=0o600)
            return {
                "schema_version": "exploratory-qwen3-ld-input-audit-mutation/v1",
                "revision": updated["revision"],
                "status": self._status(updated),
                "decision": updated["decisions"][item_id],
                "item_summary": self._summary(updated, item_id),
            }


class PilotInputAuditRequestHandler(BaseHTTPRequestHandler):
    """HTTP boundary for the task-adapted input-audit workbench."""

    service: PilotInputAuditService
    asset_names = {"index.html", "app.js", "core.js", "styles.css", "review-base.css", "review-core.js"}

    def log_message(self, format_string: str, *args: Any) -> None:
        return

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
            "Content-Security-Policy",
            "default-src 'self'; script-src 'self'; style-src 'self'; "
            "img-src 'self' data:; connect-src 'self'; object-src 'none'; "
            "base-uri 'none'; frame-ancestors 'none'; form-action 'none'",
        )

    def _send(self, status: HTTPStatus, body: bytes, content_type: str) -> None:
        self.send_response(status.value)
        self._headers(content_type, len(body))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status: HTTPStatus, value: Mapping[str, Any]) -> None:
        self._send(status, canonical_bytes(dict(value)), "application/json; charset=utf-8")

    def _error(self, status: HTTPStatus, message: str) -> None:
        self._json(status, {"status": status.value, "error": message})

    def _host_allowed(self) -> bool:
        return self.headers.get("Host", "") in self.service.allowed_hosts

    def _origin_allowed(self) -> bool:
        origin = self.headers.get("Origin")
        return origin is None or origin in self.service.allowed_origins

    def _asset(self, name: str) -> None:
        if name not in self.asset_names:
            self._error(HTTPStatus.NOT_FOUND, "not found")
            return
        shared = {
            "review-base.css": REPOSITORY_ROOT / "tools/wp3_candidate_review_ui/styles.css",
            "review-core.js": REPOSITORY_ROOT / "tools/wp3_candidate_review_ui/core.js",
        }
        path = shared.get(name, self.service.asset_root / name)
        if not path.is_file() or path.is_symlink():
            self._error(HTTPStatus.INTERNAL_SERVER_ERROR, "input-audit UI asset missing")
            return
        content_type = {
            "index.html": "text/html; charset=utf-8",
            "app.js": "text/javascript; charset=utf-8",
            "core.js": "text/javascript; charset=utf-8",
            "review-core.js": "text/javascript; charset=utf-8",
            "styles.css": "text/css; charset=utf-8",
            "review-base.css": "text/css; charset=utf-8",
        }[name]
        self._send(HTTPStatus.OK, path.read_bytes(), content_type)

    def _read_payload(self) -> dict[str, Any]:
        try:
            length = int(self.headers.get("Content-Length", ""))
        except ValueError as exc:
            raise PilotWebReviewError("invalid Content-Length") from exc
        if length <= 0 or length > 64 * 1024:
            raise PilotWebReviewError("request body size is invalid")
        if self.headers.get("Content-Type", "").split(";", 1)[0] != "application/json":
            raise PilotWebReviewError("request must use application/json")
        value = json.loads(self.rfile.read(length))
        if not isinstance(value, dict):
            raise PilotWebReviewError("request JSON must be an object")
        return value

    def do_GET(self) -> None:  # noqa: N802
        if not self._host_allowed():
            self._error(HTTPStatus.FORBIDDEN, "Host not allowed")
            return
        path = urlsplit(self.path).path
        if path in {"/", "/index.html"}:
            self._asset("index.html")
        elif path in {"/app.js", "/core.js", "/review-core.js", "/styles.css", "/review-base.css"}:
            self._asset(path[1:])
        elif path == "/api/health":
            self._json(HTTPStatus.OK, {"status": "ok"})
        elif path == "/api/bootstrap":
            self._json(HTTPStatus.OK, self.service.bootstrap())
        elif path.startswith("/api/items/"):
            try:
                self._json(HTTPStatus.OK, self.service.item_state(path.removeprefix("/api/items/")))
            except PilotWebReviewError as exc:
                self._error(HTTPStatus.NOT_FOUND, str(exc))
        elif path == "/favicon.ico":
            self._send(HTTPStatus.NO_CONTENT, b"", "image/x-icon")
        else:
            self._error(HTTPStatus.NOT_FOUND, "not found")

    def do_POST(self) -> None:  # noqa: N802
        if not self._host_allowed() or not self._origin_allowed():
            self._error(HTTPStatus.FORBIDDEN, "request origin not allowed")
            return
        try:
            payload = self._read_payload()
            token = payload.get("session_token")
            if not isinstance(token, str) or not secrets.compare_digest(token, self.service.session_token):
                self._error(HTTPStatus.FORBIDDEN, "invalid session token")
                return
            path = urlsplit(self.path).path
            if path == "/api/save":
                self._json(HTTPStatus.OK, self.service.save(payload))
            elif path == "/api/reopen":
                self._json(HTTPStatus.OK, self.service.reopen(payload))
            else:
                self._error(HTTPStatus.NOT_FOUND, "API route not found")
        except PilotReviewConflict as exc:
            self._error(HTTPStatus.CONFLICT, str(exc))
        except (PilotError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            self._error(HTTPStatus.UNPROCESSABLE_ENTITY, str(exc))


def serve_audit(
    *,
    output_root: Path = DEFAULT_OUTPUT,
    frame_path: Path | None = None,
    session_path: Path | None = None,
    reviewer_id: str = "liaozijie",
    host: str = "127.0.0.1",
    port: int = 8766,
    check: bool = False,
    public_origin: str | None = None,
) -> dict[str, Any]:
    output_root = output_root.resolve()
    service = PilotInputAuditService(
        frame_path=(frame_path or output_root / "candidate_frame.json").resolve(),
        session_path=(session_path or output_root / "audit" / DEFAULT_AUDIT_SESSION_NAME).resolve(),
        reviewer_id=reviewer_id,
    )
    if host not in {"127.0.0.1", "localhost", "::1"}:
        raise PilotWebReviewError("audit service may bind only to a loopback host")
    if check:
        service.configure_network(port, public_origin=public_origin)
        state = service.bootstrap()
        return {"frame_id": state["frame_id"], **state["status"], "check": True}
    handler = type("PilotInputAuditHandler", (PilotInputAuditRequestHandler,), {"service": service})
    server = ThreadingHTTPServer((host, port), handler)
    server.daemon_threads = True
    actual_port = int(server.server_address[1])
    service.configure_network(actual_port, public_origin=public_origin)
    state = {
        "url": (public_origin + "/") if public_origin else f"http://127.0.0.1:{actual_port}/",
        "frame_id": service.frame["frame_id"],
        "session_file": str(service.session_path),
        "public_origin": public_origin,
        "development_only": True,
        "frontend": "tools/exploratory_qwen3_ld_review_ui",
        "interaction_pattern_reused_from": "WP3 2.1 / G3 form-relation review",
    }
    print(json.dumps(state, ensure_ascii=False, sort_keys=True), flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return state


def _parse_audit_notes(notes: str) -> dict[str, Any]:
    failures: list[str] = []
    tags: list[str] = []
    for part in re.split(r"[;；\n]", notes):
        key, separator, value = part.strip().partition("=")
        if not separator:
            continue
        values = [token.strip() for token in re.split(r"[,，|]", value) if token.strip()]
        if key.strip().lower() == "fail":
            failures.extend(values)
        if key.strip().lower() == "tags":
            tags.extend(values)
    return {"failures": sorted(set(failures)), "pragmatic_tags": sorted(set(tags))}


def freeze_frame(
    *,
    config_path: Path = DEFAULT_CONFIG,
    output_root: Path = DEFAULT_OUTPUT,
    session_path: Path | None = None,
) -> dict[str, Any]:
    config, resolved_config_path = resolve_config(config_path)
    output_root = output_root.resolve()
    public = load_json(output_root / "candidate_frame.json")
    private = load_json(output_root / "candidate_private.json")
    session = _read_audit_session(
        (session_path or output_root / "audit" / DEFAULT_AUDIT_SESSION_NAME).resolve()
    )
    if session.get("frame_id") != public.get("manifest", {}).get("frame_id"):
        raise PilotError("audit session/candidate frame binding differs")
    decisions = session["decisions"]
    if set(decisions) != {row["blind_id"] for row in private["rows"]}:
        raise PilotError("audit decision coverage differs")
    if any(row["status"] != "confirmed" for row in decisions.values()):
        remaining = sum(row["status"] != "confirmed" for row in decisions.values())
        raise PilotError(f"input audit is incomplete: {remaining} candidate(s) remain")
    if any(row["disposition"] == "defer" for row in decisions.values()):
        raise PilotError("defer must be zero before frame freeze")
    selected: list[dict[str, Any]] = []
    replacements: list[dict[str, Any]] = []
    by_stratum: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in private["rows"]:
        by_stratum[row["stratum"]].append(row)
    for stratum, quota_value in config["frame"]["strata"].items():
        quota = int(quota_value)
        ordered = sorted(by_stratum[stratum], key=lambda row: int(row["stratum_ordinal"]))
        eligible = [row for row in ordered if decisions[row["blind_id"]]["disposition"] == "accept"]
        if len(eligible) < quota:
            raise PilotError(f"stratum {stratum} has only {len(eligible)} accepted candidates; needs {quota}")
        chosen = eligible[:quota]
        primary_ids = {row["query_id"] for row in ordered[:quota]}
        for final_ordinal, row in enumerate(chosen):
            decision = decisions[row["blind_id"]]
            final = {
                **copy.deepcopy(row),
                "frame_ordinal": len(selected),
                "stratum_final_ordinal": final_ordinal,
                "audit": {
                    "reviewer_id": session["reviewer_id"],
                    "disposition": decision["disposition"],
                    "relevance": decision["relevance"],
                    "boundary": decision["boundary"],
                    "definition_quality": decision["definition_quality"],
                    "sense_fit": decision["sense_fit"],
                    "swap_incompatibility": decision["swap_incompatibility"],
                    "no_hit_verified": decision["no_hit_verified"],
                    "notes": decision["notes"],
                    "core_dimensions_pass": True,
                    "failures": [],
                    "pragmatic_tags": copy.deepcopy(decision["pragmatic_tags"]),
                },
            }
            selected.append(final)
            if row["query_id"] not in primary_ids:
                replacements.append(
                    {"stratum": stratum, "replacement_query_id": row["query_id"], "reason": "earlier-frozen-candidate-rejected"}
                )
    preflight_ids: list[str] = []
    for stratum, count_value in config["frame"]["preflight_strata"].items():
        count = int(count_value)
        rows = [row for row in selected if row["stratum"] == stratum]
        preflight_ids.extend(row["query_id"] for row in rows[:count])
    identity = {
        "schema_version": FROZEN_FRAME_SCHEMA_VERSION,
        "experiment_id": config["experiment_id"],
        "candidate_frame_id": public["manifest"]["frame_id"],
        "candidate_frame_sha256": file_sha256(output_root / "candidate_frame.json"),
        "candidate_private_sha256": file_sha256(output_root / "candidate_private.json"),
        "audit_session_revision": session["revision"],
        "config_sha256": file_sha256(resolved_config_path),
        "rows_sha256": canonical_sha256(selected),
        "query_count": len(selected),
        "preflight_query_ids": preflight_ids,
        "replacements": replacements,
        "development_only": True,
        "scientific_eligible": False,
    }
    frame_id = f"pilot-frozen-{canonical_sha256(identity)}"
    frozen = {"manifest": {**identity, "frame_id": frame_id}, "rows": selected}
    if len(selected) != 64 or len(preflight_ids) != 8:
        raise PilotError("frozen frame/preflight counts differ from 64/8")
    write_json(output_root / "frozen_frame.json", frozen, mode=0o600)
    receipt = {
        "frame_id": frame_id,
        "query_count": len(selected),
        "preflight_count": len(preflight_ids),
        "replacements": replacements,
        "strata": dict(Counter(row["stratum"] for row in selected)),
    }
    write_json(output_root / "freeze_frame_receipt.json", receipt)
    return receipt


def _fit_demo_rows(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    partition = read_jsonl(source_path(config, "fit_partition"))
    fit_ids = {str(row["query_id"]) for row in partition if row.get("partition") == "fit"}
    if len(fit_ids) != int(config["sources"]["expected_fit_count"]):
        raise PilotError("formal fit partition count differs")
    train = load_json(source_path(config, "train"))
    train_by_id = {str(row["id"]): row for row in train}
    if not fit_ids.issubset(train_by_id):
        raise PilotError("fit partition references unknown train IDs")
    rows = []
    for query_id in sorted(fit_ids, key=lambda value: (int(value) if value.isdigit() else math.inf, value)):
        row = train_by_id[query_id]
        if len(row.get("quadruples", [])) != 1:
            continue
        quad = row["quadruples"][0]
        groups = list(quad.get("targeted_group", []))
        source_class = "non-hate" if quad.get("hateful") == "non-hate" and groups == ["non-hate"] else groups[0]
        if source_class not in GROUP_ORDER:
            continue
        rows.append({**copy.deepcopy(row), "source_class": source_class})
    return rows


def _encode_bge(
    texts: Sequence[str],
    *,
    model_path: Path,
    device: str,
    batch_size: int = 64,
) -> Any:
    import torch
    import torch.nn.functional as torch_functional
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, trust_remote_code=False)
    model = AutoModel.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16 if device.startswith("cuda") else torch.float32,
        trust_remote_code=False,
    ).to(device)
    model.eval()
    blocks = []
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            encoded = tokenizer(
                list(texts[start : start + batch_size]),
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(device)
            output = model(**encoded)
            vectors = torch_functional.normalize(output.last_hidden_state[:, 0].float(), p=2, dim=1)
            blocks.append(vectors.cpu())
    result = torch.cat(blocks, dim=0)
    del model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return result


def _retrieve_demos(
    frame_rows: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    output_root: Path,
    *,
    device: str,
) -> dict[str, list[dict[str, Any]]]:
    import torch

    demos = _fit_demo_rows(config)
    embedding_path = output_root / "cache" / "bge_fit_single_tuple.pt"
    embedding_meta_path = output_root / "cache" / "bge_fit_single_tuple.meta.json"
    demo_identity = canonical_sha256(
        [{"id": str(row["id"]), "content": row["content"], "source_class": row["source_class"]} for row in demos]
    )
    model_path = source_path(config, "embedding_model")
    expected_meta = {
        "demo_identity": demo_identity,
        "embedding_model_path": str(model_path),
        "embedding_model_config_sha256": file_sha256(model_path / "config.json"),
    }
    if embedding_path.is_file() and embedding_meta_path.is_file() and load_json(embedding_meta_path) == expected_meta:
        demo_vectors = torch.load(embedding_path, map_location="cpu", weights_only=True)
    else:
        demo_vectors = _encode_bge([str(row["content"]) for row in demos], model_path=model_path, device=device)
        embedding_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(demo_vectors, embedding_path)
        write_json(embedding_meta_path, expected_meta)
    query_vectors = _encode_bge(
        ["为这个句子生成表示以用于检索相关文章：" + str(row["content"]) for row in frame_rows],
        model_path=model_path,
        device=device,
    )
    scores = query_vectors @ demo_vectors.T
    quotas = config["retrieval"]["allocated_class_top_k"]
    order = list(config["retrieval"]["source_class_order"])
    by_class: dict[str, list[int]] = {
        source_class: [index for index, row in enumerate(demos) if row["source_class"] == source_class]
        for source_class in order
    }
    selected_by_query: dict[str, list[dict[str, Any]]] = {}
    for query_index, frame_row in enumerate(frame_rows):
        assignments: list[dict[str, Any]] = []
        for source_class in order:
            ranked = sorted(
                by_class[source_class],
                key=lambda index: (-float(scores[query_index, index]), str(demos[index]["id"])),
            )
            needed = int(quotas[source_class])
            if len(ranked) < needed:
                raise PilotError(f"demo class {source_class} cannot fill quota {needed}")
            for rank, index in enumerate(ranked[:needed]):
                assignments.append(
                    {
                        **copy.deepcopy(demos[index]),
                        "retrieval_score": round(float(scores[query_index, index]), 8),
                        "quota_class": source_class,
                        "quota_rank": rank,
                    }
                )
        assignments.sort(key=lambda row: (-row["retrieval_score"], order.index(row["quota_class"]), str(row["id"])))
        if len(assignments) != int(config["retrieval"]["demo_top_k"]):
            raise PilotError("demo selection did not produce k=10")
        selected_by_query[str(frame_row["query_id"])] = assignments
    return selected_by_query


def _canonical_gold(quadruples: Sequence[Mapping[str, Any]]) -> str:
    from utils.quadruple import canonicalize_quadruples, serialize_quadruples

    return serialize_quadruples(canonicalize_quadruples(quadruples))


def _category_swap(category: str, tokenizer: Any, categories: Sequence[str]) -> str:
    source_length = len(tokenizer.encode(category, add_special_tokens=False))
    eligible = [value for value in categories if value != category]
    eligible.sort(key=lambda value: (abs(len(tokenizer.encode(value, add_special_tokens=False)) - source_length), categories.index(value)))
    if not eligible:
        raise PilotError("CategorySwap has no legal derangement")
    return eligible[0]


def _render_lexicon(
    hits: Sequence[Mapping[str, Any]],
    view: str,
    *,
    tokenizer: Any,
    categories: Sequence[str],
) -> tuple[str, list[dict[str, Any]]]:
    if not hits:
        return "（无精确词典命中）", []
    rendered: list[str] = []
    interventions: list[dict[str, Any]] = []
    for ordinal, hit in enumerate(hits, start=1):
        term = str(hit["term"])
        category = str(hit["category"])
        definition = str(hit["definition"])
        used_category = category
        used_definition = definition
        if view == "CategorySwap":
            used_category = _category_swap(category, tokenizer, categories)
        if view == "DefinitionSwap":
            used_definition = str(hit["definition_swap_donor"]["definition"])
        if view == "Definition":
            text = f"- 词条：{term}\n  定义：{used_definition}"
        elif view == "Category":
            text = f"- 词条：{term}\n  类别：{used_category}\n  中性说明：该类别仅是词义领域提示，不代表当前句子的任务答案。"
        else:
            text = f"- 词条：{term}\n  类别：{used_category}\n  定义：{used_definition}"
        rendered.append(text)
        interventions.append(
            {
                "ordinal": ordinal,
                "lexicon_id": hit["lexicon_id"],
                "term": term,
                "source_category": category,
                "rendered_category": used_category,
                "source_definition_sha256": sha256_bytes(definition.encode()),
                "rendered_definition_sha256": sha256_bytes(used_definition.encode()),
                "definition_donor_id": (
                    hit["definition_swap_donor"]["lexicon_id"] if view == "DefinitionSwap" else None
                ),
            }
        )
    return "\n".join(rendered), interventions


def _render_demos(demos: Sequence[Mapping[str, Any]], view: str) -> tuple[str, list[dict[str, Any]]]:
    outputs = [copy.deepcopy(row["quadruples"][0]) for row in demos]
    donor_indices = list(range(1, len(demos))) + [0]
    blocks = []
    trace = []
    for index, row in enumerate(demos):
        if view == "Full":
            output = _canonical_gold(row["quadruples"])
            block = f"示例 {index + 1}\n待分析句子：{row['content']}\ncanonical-quad-json/v1 JSON 数组：\n{output}"
        elif view == "Input":
            block = f"示例 {index + 1}\n待分析句子：{row['content']}\n标注输出：（隐藏）"
        elif view == "Schema":
            block = (
                f"示例 {index + 1}\n待分析句子：<示例输入已隐藏>\n"
                'canonical-quad-json/v1 结构：[{"target":"<span>","argument":"<span>",'
                '"targeted_group":["<class>"],"hateful":"<class>"}]'
            )
        elif view == "CrossLabelShuffle":
            donor = outputs[donor_indices[index]]
            changed = copy.deepcopy(row["quadruples"][0])
            changed["targeted_group"] = copy.deepcopy(donor["targeted_group"])
            changed["hateful"] = donor["hateful"]
            output = _canonical_gold([changed])
            block = f"示例 {index + 1}\n待分析句子：{row['content']}\ncanonical-quad-json/v1 JSON 数组：\n{output}"
        else:
            raise PilotError(f"unknown demo view {view}")
        blocks.append(block)
        trace.append(
            {
                "demo_id": str(row["id"]),
                "source_class": row["source_class"],
                "retrieval_score": row["retrieval_score"],
                "label_donor_demo_id": str(demos[donor_indices[index]]["id"]) if view == "CrossLabelShuffle" else None,
                "rendered_targeted_group": (
                    copy.deepcopy(outputs[donor_indices[index]]["targeted_group"])
                    if view == "CrossLabelShuffle"
                    else copy.deepcopy(outputs[index]["targeted_group"])
                ),
                "rendered_hateful": (
                    outputs[donor_indices[index]]["hateful"]
                    if view == "CrossLabelShuffle"
                    else outputs[index]["hateful"]
                ),
            }
        )
    return "\n\n".join(blocks), trace


def _neutral_blob(tokenizer: Any, target_tokens: int, *, kind: str, block_count: int | None = None) -> str:
    if target_tokens <= 0:
        return "（无）"
    if kind == "lexicon":
        unit = "背景资料：今天的公园开放时间照常，园艺记录描述了树木、步道与季节变化，不包含待分析句子的词义或任务标签。\n"
    else:
        unit = (
            '待分析句子：<无关中性占位文本>\ncanonical-quad-json/v1 结构：'
            '[{"target":"<span>","argument":"<span>","targeted_group":["<class>"],"hateful":"<class>"}]\n'
        )
    prefix = ""
    if block_count:
        prefix = "\n".join(f"中性占位示例 {index + 1}" for index in range(block_count)) + "\n"
    text = prefix + unit * max(2, target_tokens // max(1, len(tokenizer.encode(unit, add_special_tokens=False))) + 3)
    ids = tokenizer.encode(text, add_special_tokens=False)
    candidate = tokenizer.decode(ids[:target_tokens], skip_special_tokens=True)
    for _ in range(6):
        observed = len(tokenizer.encode(candidate, add_special_tokens=False))
        if observed == target_tokens:
            break
        if observed < target_tokens:
            candidate += tokenizer.decode(ids[: target_tokens - observed], skip_special_tokens=True)
        else:
            candidate = tokenizer.decode(tokenizer.encode(candidate, add_special_tokens=False)[:target_tokens], skip_special_tokens=True)
    return candidate


def _render_prompt_messages(content: str, lexicon_text: str, demo_text: str) -> list[dict[str, str]]:
    from prompt import STAGE1_QUAD_JSON_SYSTEM_PROMPT_V2

    user = f"""请从待分析句子抽取仇恨言论四元组。先按完整语境形成判断，再核验可用资源；资源可以为空、无关或相互冲突。

词典资源：
{lexicon_text}

示例资源：
{demo_text}

约束：词条命中、类别、定义或示例标签都不能直接推出当前句为 hate；须结合实际指向及引用、自称、否定、反讽、讨论、反仇恨语境独立判断。

待分析句子：
{content}

只输出 canonical-quad-json/v1 JSON 数组："""
    return [
        {"role": "system", "content": STAGE1_QUAD_JSON_SYSTEM_PROMPT_V2},
        {"role": "user", "content": user},
    ]


def _condition_views(condition: str) -> tuple[str | None, str | None, bool, bool]:
    mapping = {
        "C0": (None, None, False, False),
        "L-Full": ("Full", None, False, False),
        "D-Full": (None, "Full", False, False),
        "LD-Full": ("Full", "Full", False, False),
        "PL": ("Full", None, True, False),
        "PD": (None, "Full", False, True),
        "L-Definition": ("Definition", None, False, False),
        "L-Category": ("Category", None, False, False),
        "L-CategorySwap": ("CategorySwap", None, False, False),
        "L-DefinitionSwap": ("DefinitionSwap", None, False, False),
        "D-Input": (None, "Input", False, False),
        "D-Schema": (None, "Schema", False, False),
        "D-CrossLabelShuffle": (None, "CrossLabelShuffle", False, False),
        "LD-CategorySwap": ("CategorySwap", "Full", False, False),
        "LD-DefinitionSwap": ("DefinitionSwap", "Full", False, False),
    }
    if condition not in mapping:
        raise PilotError(f"unknown condition {condition}")
    return mapping[condition]


def build_context_grid(
    *,
    config_path: Path = DEFAULT_CONFIG,
    output_root: Path = DEFAULT_OUTPUT,
    device: str | None = None,
) -> dict[str, Any]:
    from transformers import AutoTokenizer

    config, resolved_config_path = resolve_config(config_path)
    output_root = output_root.resolve()
    frozen = load_json(output_root / "frozen_frame.json")
    if frozen.get("manifest", {}).get("schema_version") != FROZEN_FRAME_SCHEMA_VERSION:
        raise PilotError("frozen frame schema differs")
    runtime_device = device or str(config["runtime"]["device"])
    tokenizer = AutoTokenizer.from_pretrained(source_path(config, "tokenizer"), use_fast=True, trust_remote_code=False)
    demos_by_query = _retrieve_demos(frozen["rows"], config, output_root, device=runtime_device)
    categories = list(config["interventions"]["category_order"])
    contexts: list[dict[str, Any]] = []
    tolerance = config["interventions"]["placebo_tolerance"]
    for frame_row in frozen["rows"]:
        query_id = str(frame_row["query_id"])
        conditions = (
            config["matrix"]["no_hit_conditions"]
            if frame_row["stratum"] == "no_hit"
            else config["matrix"]["lex_hit_conditions"]
        )
        demos = demos_by_query[query_id]
        full_lex, _ = _render_lexicon(frame_row["lexicon_hits"], "Full", tokenizer=tokenizer, categories=categories)
        full_demo, _ = _render_demos(demos, "Full")
        placebo_lex = _neutral_blob(
            tokenizer,
            len(tokenizer.encode(full_lex, add_special_tokens=False)),
            kind="lexicon",
        )
        placebo_demo = _neutral_blob(
            tokenizer,
            len(tokenizer.encode(full_demo, add_special_tokens=False)),
            kind="demo",
            block_count=len(demos),
        )
        for condition in conditions:
            lex_view, demo_view, use_placebo_lex, use_placebo_demo = _condition_views(condition)
            if lex_view is None:
                lex_text, lex_trace = "（未提供）", []
            else:
                lex_text, lex_trace = _render_lexicon(
                    frame_row["lexicon_hits"], lex_view, tokenizer=tokenizer, categories=categories
                )
            if demo_view is None:
                demo_text, demo_trace = "（未提供）", []
            else:
                demo_text, demo_trace = _render_demos(demos, demo_view)
            if use_placebo_lex:
                lex_text = placebo_lex
            if use_placebo_demo:
                demo_text = placebo_demo
            if (
                use_placebo_lex and str(frame_row["content"]) in lex_text
            ) or (
                use_placebo_demo and str(frame_row["content"]) in demo_text
            ):
                raise PilotError(f"placebo leakage of full query text for {query_id}/{condition}")
            messages = _render_prompt_messages(str(frame_row["content"]), lex_text, demo_text)
            prompt_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
            if len(prompt_ids) > int(config["runtime"]["max_prompt_tokens"]):
                raise PilotError(f"frozen prompt overflow for {query_id}/{condition}: {len(prompt_ids)}")
            context = {
                "schema_version": "exploratory-qwen3-ld-context/v0",
                "query_id": query_id,
                "frame_ordinal": frame_row["frame_ordinal"],
                "stratum": frame_row["stratum"],
                "condition": condition,
                "content": frame_row["content"],
                "gold": frame_row["gold"],
                "messages": messages,
                "prompt_text": prompt_text,
                "prompt_tokens": len(prompt_ids),
                "prompt_sha256": sha256_bytes(prompt_text.encode()),
                "prompt_token_ids_sha256": canonical_sha256(prompt_ids),
                "lexicon_trace": lex_trace,
                "demo_trace": demo_trace,
                "placebo": {"lexicon": use_placebo_lex, "demo": use_placebo_demo},
            }
            context["context_sha256"] = canonical_sha256(context)
            contexts.append(context)
        lex_real_tokens = len(tokenizer.encode(full_lex, add_special_tokens=False))
        lex_placebo_tokens = len(tokenizer.encode(placebo_lex, add_special_tokens=False))
        demo_real_tokens = len(tokenizer.encode(full_demo, add_special_tokens=False))
        demo_placebo_tokens = len(tokenizer.encode(placebo_demo, add_special_tokens=False))
        for resource, real_count, placebo_count in (
            ("lexicon", lex_real_tokens, lex_placebo_tokens),
            ("demo", demo_real_tokens, demo_placebo_tokens),
        ):
            allowed = max(int(tolerance["absolute_tokens"]), math.ceil(float(tolerance["relative"]) * real_count))
            if abs(real_count - placebo_count) > allowed:
                raise PilotError(
                    f"{resource} placebo mismatch for {query_id}: real={real_count}, placebo={placebo_count}, allowed={allowed}"
                )
    expected = int(config["matrix"]["expected_generation_count"])
    if len(contexts) != expected:
        raise PilotError(f"context matrix count {len(contexts)} differs from {expected}")
    identity = {
        "schema_version": CONTEXT_SCHEMA_VERSION,
        "experiment_id": config["experiment_id"],
        "frozen_frame_id": frozen["manifest"]["frame_id"],
        "frozen_frame_sha256": file_sha256(output_root / "frozen_frame.json"),
        "config_sha256": file_sha256(resolved_config_path),
        "contexts_sha256": canonical_sha256(contexts),
        "context_count": len(contexts),
        "thinking": False,
        "development_only": True,
    }
    grid_id = f"pilot-contexts-{canonical_sha256(identity)}"
    grid = {"manifest": {**identity, "grid_id": grid_id}, "contexts": contexts}
    write_json(output_root / "context_grid.json", grid, mode=0o600)
    receipt = {
        "grid_id": grid_id,
        "context_count": len(contexts),
        "max_prompt_tokens": max(row["prompt_tokens"] for row in contexts),
        "condition_counts": dict(sorted(Counter(row["condition"] for row in contexts).items())),
    }
    write_json(output_root / "build_contexts_receipt.json", receipt)
    return receipt


def _load_generation_ledger(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    result: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        if row.get("schema_version") != LEDGER_SCHEMA_VERSION:
            raise PilotError("generation ledger schema differs")
        prompt_hash = str(row.get("prompt_sha256", ""))
        if prompt_hash in result and result[prompt_hash] != row:
            raise PilotError(f"generation ledger has conflicting duplicate {prompt_hash}")
        result[prompt_hash] = row
    return result


def _deadline_today(local_time: str) -> datetime:
    hour_text, minute_text = local_time.split(":", 1)
    now = datetime.now().astimezone()
    return now.replace(hour=int(hour_text), minute=int(minute_text), second=0, microsecond=0)


def _trim_generated_completion(
    output_ids: Sequence[int], *, eos_token_id: int | Sequence[int] | None
) -> list[int]:
    """Remove batch-only padding by retaining tokens through the first EOS."""

    result = [int(value) for value in output_ids]
    if eos_token_id is None:
        return result
    eos_ids = (
        {int(eos_token_id)}
        if isinstance(eos_token_id, int)
        else {int(value) for value in eos_token_id}
    )
    for index, token_id in enumerate(result):
        if token_id in eos_ids:
            return result[: index + 1]
    return result


class _HFRunner:
    def __init__(self, config: Mapping[str, Any], *, device: str | None = None) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.device = device or str(config["runtime"]["device"])
        if not self.device.startswith("cuda") or not torch.cuda.is_available():
            raise PilotError("the frozen pilot requires a CUDA single-GPU backend")
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        seed = int(config["frame"]["seed"])
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(True, warn_only=False)
        model_path = source_path(config, "model")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, trust_remote_code=False)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=False,
            low_cpu_mem_usage=True,
        ).to(self.device)
        self.model.eval()
        self.max_new_tokens = int(config["runtime"]["max_new_tokens"])
        self.max_sequence_tokens = int(config["runtime"]["max_sequence_tokens"])

    def close(self) -> None:
        del self.model
        self.torch.cuda.empty_cache()

    def generate_batch(self, contexts: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        torch = self.torch
        texts = [str(row["prompt_text"]) for row in contexts]
        encoded = self.tokenizer(
            texts,
            padding=True,
            add_special_tokens=False,
            return_tensors="pt",
        ).to(self.device)
        observed_lengths = encoded.attention_mask.sum(dim=1).tolist()
        expected_lengths = [int(row["prompt_tokens"]) for row in contexts]
        if observed_lengths != expected_lengths:
            raise PilotError(f"prompt token replay differs: {observed_lengths} != {expected_lengths}")
        with torch.inference_mode():
            generated = self.model.generate(
                **encoded,
                do_sample=False,
                max_new_tokens=self.max_new_tokens,
                use_cache=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        input_width = int(encoded.input_ids.shape[1])
        rows = []
        for index, context in enumerate(contexts):
            output_ids = _trim_generated_completion(
                generated[index, input_width:].tolist(),
                eos_token_id=self.tokenizer.eos_token_id,
            )
            raw = self.tokenizer.decode(output_ids, skip_special_tokens=True)
            rows.append(
                {
                    "schema_version": LEDGER_SCHEMA_VERSION,
                    "experiment_id": context.get("experiment_id", "stage1-exploratory-qwen3-ld-v0"),
                    "query_id": str(context["query_id"]),
                    "condition": context["condition"],
                    "context_sha256": context["context_sha256"],
                    "prompt_sha256": context["prompt_sha256"],
                    "prompt_token_ids_sha256": context["prompt_token_ids_sha256"],
                    "prompt_tokens": int(context["prompt_tokens"]),
                    "completion_token_ids": output_ids,
                    "completion_token_ids_sha256": canonical_sha256(output_ids),
                    "completion_tokens": len(output_ids),
                    "raw_output": raw,
                    "raw_output_sha256": sha256_bytes(raw.encode()),
                    "runner_status": "ok",
                    "thinking": False,
                    "generated_at": _now_iso(),
                }
            )
        return rows


def _generate_selected(
    *,
    contexts: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    ledger_path: Path,
    device: str | None,
    enforce_deadline: bool,
    stop_at_local_time: str | None = None,
    replay_contexts: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    existing = _load_generation_ledger(ledger_path)
    for context in contexts:
        prior = existing.get(str(context["prompt_sha256"]))
        if prior and (
            prior.get("context_sha256") != context.get("context_sha256")
            or prior.get("prompt_token_ids_sha256") != context.get("prompt_token_ids_sha256")
        ):
            raise PilotError("resume ledger hash binding differs")
    pending = [row for row in contexts if row["prompt_sha256"] not in existing]
    effective_stop_time = stop_at_local_time or str(
        config["runtime"]["stop_new_gpu_batches_local_time"]
    )
    stop_at = _deadline_today(effective_stop_time)
    batch_size = int(config["runtime"]["batch_size"])
    stopped_for_deadline = False
    replay_results: list[dict[str, Any]] = []
    runner: _HFRunner | None = None
    try:
        if pending or replay_contexts:
            runner = _HFRunner(config, device=device)
        for start in range(0, len(pending), batch_size):
            if enforce_deadline and datetime.now().astimezone() >= stop_at:
                stopped_for_deadline = True
                break
            batch = pending[start : start + batch_size]
            if runner is None:  # pragma: no cover - guarded by pending
                raise AssertionError("generation runner is unavailable")
            generated = runner.generate_batch(batch)
            for row in generated:
                append_jsonl(ledger_path, row)
                existing[row["prompt_sha256"]] = row
        replay_batch = (
            []
            if stopped_for_deadline or not replay_contexts
            else runner.generate_batch(replay_contexts)  # type: ignore[union-attr]
        )
        for context, observed in zip(replay_contexts, replay_batch, strict=True):
            expected = existing.get(str(context["prompt_sha256"]))
            replay_results.append(
                {
                    "query_id": context["query_id"],
                    "condition": context["condition"],
                    "prompt_sha256": context["prompt_sha256"],
                    "raw_output_equal": bool(expected and observed["raw_output"] == expected["raw_output"]),
                    "completion_token_ids_equal": bool(
                        expected and observed["completion_token_ids"] == expected["completion_token_ids"]
                    ),
                }
            )
    finally:
        if runner is not None:
            runner.close()
    complete_count = sum(row["prompt_sha256"] in existing for row in contexts)
    return {
        "requested": len(contexts),
        "complete": complete_count,
        "new": len(pending) - (len(contexts) - complete_count),
        "stopped_for_deadline": stopped_for_deadline,
        "effective_stop_new_gpu_batches_local_time": effective_stop_time,
        "replay": replay_results,
    }


def run_preflight(
    *,
    config_path: Path = DEFAULT_CONFIG,
    output_root: Path = DEFAULT_OUTPUT,
    device: str | None = None,
    stop_at_local_time: str | None = None,
) -> dict[str, Any]:
    from utils.quadruple import parse_quadruples

    config, _ = resolve_config(config_path)
    output_root = output_root.resolve()
    grid = load_json(output_root / "context_grid.json")
    frozen = load_json(output_root / "frozen_frame.json")
    preflight_ids = set(frozen["manifest"]["preflight_query_ids"])
    contexts = [row for row in grid["contexts"] if str(row["query_id"]) in preflight_ids]
    expected_count = sum(
        len(config["matrix"]["no_hit_conditions"] if row["stratum"] == "no_hit" else config["matrix"]["lex_hit_conditions"])
        for row in frozen["rows"]
        if str(row["query_id"]) in preflight_ids
    )
    if len(contexts) != expected_count:
        raise PilotError("preflight full-matrix selection count differs")
    core_names = set(config["preflight"]["core_conditions"])
    # The first four core contexts are also the first generation batch. Replay
    # that exact ordered batch so BF16 kernel geometry and padding are held fixed.
    replay_pool = [row for row in contexts if row["condition"] in core_names]
    replay_contexts = replay_pool[: int(config["preflight"]["determinism_replay_count"])]
    ledger_path = output_root / "generations" / "ledger.jsonl"
    generated = _generate_selected(
        contexts=contexts,
        config=config,
        ledger_path=ledger_path,
        device=device,
        enforce_deadline=True,
        stop_at_local_time=stop_at_local_time,
        replay_contexts=replay_contexts,
    )
    ledger = _load_generation_ledger(ledger_path)
    records = [ledger[row["prompt_sha256"]] for row in contexts if row["prompt_sha256"] in ledger]
    infrastructure_failures = sum(row["runner_status"] != "ok" for row in records)
    core_records = [row for row in records if row["condition"] in core_names]
    strict_count = sum(parse_quadruples(str(row["raw_output"]), mode="strict").strict_format_valid for row in core_records)
    strict_rate = strict_count / len(core_records) if core_records else 0.0
    passed = bool(
        generated["complete"] == len(contexts)
        and infrastructure_failures <= int(config["preflight"]["infrastructure_failure_max"])
        and strict_rate >= float(config["preflight"]["strict_json_rate_min"])
        and len(generated["replay"]) == int(config["preflight"]["determinism_replay_count"])
        and all(row["raw_output_equal"] and row["completion_token_ids_equal"] for row in generated["replay"])
    )
    receipt = {
        "schema_version": "exploratory-qwen3-ld-preflight/v0",
        "grid_id": grid["manifest"]["grid_id"],
        "grid_sha256": file_sha256(output_root / "context_grid.json"),
        "selected_query_count": len(preflight_ids),
        "selected_generation_count": len(contexts),
        "complete_generation_count": generated["complete"],
        "infrastructure_failures": infrastructure_failures,
        "core_strict_json_rate": strict_rate,
        "determinism_replay": generated["replay"],
        "effective_stop_new_gpu_batches_local_time": generated[
            "effective_stop_new_gpu_batches_local_time"
        ],
        "effect_metrics_inspected": False,
        "passed": passed,
        "created_at": _now_iso(),
    }
    write_json(output_root / "preflight_receipt.json", receipt)
    return receipt


def run_full_generation(
    *,
    config_path: Path = DEFAULT_CONFIG,
    output_root: Path = DEFAULT_OUTPUT,
    device: str | None = None,
    stop_at_local_time: str | None = None,
) -> dict[str, Any]:
    config, _ = resolve_config(config_path)
    output_root = output_root.resolve()
    grid = load_json(output_root / "context_grid.json")
    preflight = load_json(output_root / "preflight_receipt.json")
    if preflight.get("passed") is not True:
        raise PilotError("full generation is blocked because preflight did not pass")
    if (
        preflight.get("grid_id") != grid.get("manifest", {}).get("grid_id")
        or preflight.get("grid_sha256") != file_sha256(output_root / "context_grid.json")
    ):
        raise PilotError("preflight/context grid binding differs")
    contexts = grid["contexts"]
    if len(contexts) != int(config["matrix"]["expected_generation_count"]):
        raise PilotError("full context count differs")
    ledger_path = output_root / "generations" / "ledger.jsonl"
    generated = _generate_selected(
        contexts=contexts,
        config=config,
        ledger_path=ledger_path,
        device=device,
        enforce_deadline=True,
        stop_at_local_time=stop_at_local_time,
    )
    complete = generated["complete"] == len(contexts)
    receipt = {
        "schema_version": "exploratory-qwen3-ld-full-generation/v0",
        "grid_id": grid["manifest"]["grid_id"],
        "expected": len(contexts),
        "complete_count": generated["complete"],
        "complete": complete,
        "generation_ledger_sha256": file_sha256(ledger_path) if ledger_path.is_file() else None,
        "stopped_for_deadline": generated["stopped_for_deadline"],
        "effective_stop_new_gpu_batches_local_time": generated[
            "effective_stop_new_gpu_batches_local_time"
        ],
        "partial_effect_conclusion_allowed": False,
        "updated_at": _now_iso(),
    }
    write_json(output_root / "full_generation_receipt.json", receipt)
    return receipt


def _bootstrap_paired_ci(
    differences: Sequence[float], *, replicates: int, seed: int
) -> dict[str, float | int | None]:
    if not differences:
        return {"n": 0, "mean": None, "ci95_low": None, "ci95_high": None}
    rng = random.Random(seed)
    n = len(differences)
    estimates = []
    for _ in range(replicates):
        estimates.append(sum(differences[rng.randrange(n)] for _ in range(n)) / n)
    estimates.sort()
    low = estimates[max(0, math.floor(0.025 * replicates))]
    high = estimates[min(replicates - 1, math.ceil(0.975 * replicates) - 1)]
    return {"n": n, "mean": sum(differences) / n, "ci95_low": low, "ci95_high": high}


def _label_follow_record(context: Mapping[str, Any], raw_output: str) -> dict[str, Any] | None:
    from utils.quadruple import parse_quadruples

    parsed = parse_quadruples(raw_output, mode="strict")
    if not parsed.strict_format_valid:
        return None
    predicted_groups = {label for quad in parsed.quadruples for label in quad.targeted_group}
    predicted_hate = {quad.hateful for quad in parsed.quadruples}
    condition = str(context["condition"])
    if condition in {"L-CategorySwap", "LD-CategorySwap"}:
        labels = {str(row["rendered_category"]) for row in context["lexicon_trace"]}
        return {
            "kind": "lexicon_category_swap",
            "followed": bool(predicted_groups & labels),
            "candidate_labels": sorted(labels),
        }
    if condition == "D-CrossLabelShuffle":
        groups = {
            str(label)
            for row in context["demo_trace"]
            for label in row.get("rendered_targeted_group", [])
        }
        hate = {str(row["rendered_hateful"]) for row in context["demo_trace"]}
        return {
            "kind": "demo_cross_label_shuffle",
            "followed_group": bool(predicted_groups & groups),
            "followed_hate": bool(predicted_hate & hate),
            "candidate_groups": sorted(groups),
            "candidate_hateful": sorted(hate),
        }
    return None


def evaluate_generation_grid(
    *,
    config_path: Path = DEFAULT_CONFIG,
    output_root: Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    from metrics.stage1_metrics import aggregate_query_metrics, evaluate_query

    config, _ = resolve_config(config_path)
    output_root = output_root.resolve()
    full_receipt = load_json(output_root / "full_generation_receipt.json")
    if full_receipt.get("complete") is not True:
        raise PilotError("effect evaluation is forbidden for an incomplete generation grid")
    grid = load_json(output_root / "context_grid.json")
    ledger_path = output_root / "generations" / "ledger.jsonl"
    if full_receipt.get("generation_ledger_sha256") != file_sha256(ledger_path):
        raise PilotError("full-generation receipt/ledger binding differs")
    ledger = _load_generation_ledger(ledger_path)
    if len(grid["contexts"]) != int(config["matrix"]["expected_generation_count"]):
        raise PilotError("evaluation context denominator differs")
    metrics_rows: list[dict[str, Any]] = []
    context_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    label_follow_rows: list[dict[str, Any]] = []
    for context in grid["contexts"]:
        generated = ledger.get(str(context["prompt_sha256"]))
        if generated is None:
            raise PilotError("generation ledger is missing a frozen prompt")
        metric = evaluate_query(
            query_id=str(context["query_id"]),
            condition=str(context["condition"]),
            raw_output=str(generated["raw_output"]),
            gold=context["gold"],
            runner_status=str(generated["runner_status"]),
            content_sha256=sha256_bytes(str(context["content"]).encode()),
            gold_sha256=canonical_sha256(context["gold"]),
            prompt_sha256=str(context["prompt_sha256"]),
            context_record_sha256=str(context["context_sha256"]),
        )
        metric["stratum"] = context["stratum"]
        metrics_rows.append(metric)
        context_by_key[(str(context["query_id"]), str(context["condition"]))] = context
        following = _label_follow_record(context, str(generated["raw_output"]))
        if following is not None:
            label_follow_rows.append(
                {"query_id": str(context["query_id"]), "condition": context["condition"], **following}
            )
    metrics_path = output_root / "evaluation" / "query_metrics.jsonl"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    if metrics_path.exists():
        metrics_path.unlink()
    for row in metrics_rows:
        append_jsonl(metrics_path, row)
    by_condition: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in metrics_rows:
        by_condition[str(row["condition"])].append(row)
    condition_summaries = {
        condition: aggregate_query_metrics(sorted(rows, key=lambda row: int(str(row["id"]))))
        for condition, rows in sorted(by_condition.items())
    }
    by_key = {(str(row["id"]), str(row["condition"])): row for row in metrics_rows}
    endpoints = [
        "tuple/hard",
        "tuple/soft",
        "field_bound/targeted_group",
        "field_bound/hateful",
        "field_bound/group_hate_joint",
        "format/strict",
    ]
    replicates = int(config["analysis"]["bootstrap_replicates"])
    seed = int(config["analysis"]["bootstrap_seed"])
    paired_effects: dict[str, Any] = {}
    for condition in sorted(by_condition):
        if condition == "C0":
            continue
        rows = sorted(by_condition[condition], key=lambda row: int(str(row["id"])))
        effects: dict[str, Any] = {}
        for endpoint_index, endpoint in enumerate(endpoints):
            differences = []
            counts = {"n00": 0, "n01": 0, "n10": 0, "n11": 0}
            for row in rows:
                baseline = by_key.get((str(row["id"]), "C0"))
                if baseline is None:
                    raise PilotError(f"C0 pair missing for {row['id']}/{condition}")
                left = bool(baseline["correctness"][endpoint])
                right = bool(row["correctness"][endpoint])
                differences.append(float(right) - float(left))
                counts[f"n{int(left)}{int(right)}"] += 1
            effects[endpoint] = {
                **counts,
                **_bootstrap_paired_ci(
                    differences,
                    replicates=replicates,
                    seed=seed + endpoint_index + 1000 * list(sorted(by_condition)).index(condition),
                ),
            }
        paired_effects[condition] = effects
    direct_specs = {
        "D-Full_minus_D-CrossLabelShuffle": ("D-CrossLabelShuffle", "D-Full"),
        "D-Full_minus_D-Input": ("D-Input", "D-Full"),
        "D-Full_minus_D-Schema": ("D-Schema", "D-Full"),
        "D-Full_minus_PD": ("PD", "D-Full"),
        "L-Full_minus_PL": ("PL", "L-Full"),
        "L-Category_minus_L-CategorySwap": ("L-CategorySwap", "L-Category"),
        "L-Full_minus_L-Category": ("L-Category", "L-Full"),
        "L-Full_minus_L-Definition": ("L-Definition", "L-Full"),
        "L-Full_minus_L-DefinitionSwap": ("L-DefinitionSwap", "L-Full"),
        "LD-Full_minus_L-Full": ("L-Full", "LD-Full"),
        "LD-Full_minus_D-Full": ("D-Full", "LD-Full"),
        "LD-Full_minus_LD-CategorySwap": ("LD-CategorySwap", "LD-Full"),
        "LD-Full_minus_LD-DefinitionSwap": ("LD-DefinitionSwap", "LD-Full"),
    }
    direct_contrasts: dict[str, Any] = {}
    for contrast_index, (name, (left_condition, right_condition)) in enumerate(
        direct_specs.items()
    ):
        query_ids = sorted(
            {
                str(row["id"])
                for row in by_condition[right_condition]
                if (str(row["id"]), left_condition) in by_key
            },
            key=int,
        )
        contrast_effects = {}
        for endpoint_index, endpoint in enumerate(endpoints):
            differences = []
            counts = {"n00": 0, "n01": 0, "n10": 0, "n11": 0}
            for query_id in query_ids:
                left = bool(by_key[(query_id, left_condition)]["correctness"][endpoint])
                right = bool(by_key[(query_id, right_condition)]["correctness"][endpoint])
                differences.append(float(right) - float(left))
                counts[f"n{int(left)}{int(right)}"] += 1
            contrast_effects[endpoint] = {
                **counts,
                **_bootstrap_paired_ci(
                    differences,
                    replicates=replicates,
                    seed=seed + 70000 + 1000 * contrast_index + endpoint_index,
                ),
            }
        direct_contrasts[name] = {
            "left_condition": left_condition,
            "right_condition": right_condition,
            "effects": contrast_effects,
        }
    interaction: dict[str, Any] = {}
    hit_ids = sorted(
        {str(row["id"]) for row in metrics_rows if row["stratum"] != "no_hit"},
        key=int,
    )
    for endpoint_index, endpoint in enumerate(endpoints):
        values = []
        for query_id in hit_ids:
            required = {condition: by_key[(query_id, condition)] for condition in ("C0", "L-Full", "D-Full", "LD-Full")}
            values.append(
                float(required["LD-Full"]["correctness"][endpoint])
                - float(required["L-Full"]["correctness"][endpoint])
                - float(required["D-Full"]["correctness"][endpoint])
                + float(required["C0"]["correctness"][endpoint])
            )
        interaction[endpoint] = _bootstrap_paired_ci(
            values, replicates=replicates, seed=seed + 50000 + endpoint_index
        )
    strata_summaries: dict[str, Any] = {}
    for stratum in config["frame"]["strata"]:
        strata_summaries[stratum] = {}
        for condition, rows in sorted(by_condition.items()):
            subset = [row for row in rows if row["stratum"] == stratum]
            if subset:
                strata_summaries[stratum][condition] = aggregate_query_metrics(subset)
    label_follow_summary = {
        "lexicon_category_swap": {
            "n": sum(row["kind"] == "lexicon_category_swap" for row in label_follow_rows),
            "rate": (
                sum(bool(row["followed"]) for row in label_follow_rows if row["kind"] == "lexicon_category_swap")
                / max(1, sum(row["kind"] == "lexicon_category_swap" for row in label_follow_rows))
            ),
        },
        "demo_cross_label_shuffle": {
            "n": sum(row["kind"] == "demo_cross_label_shuffle" for row in label_follow_rows),
            "group_rate": (
                sum(bool(row["followed_group"]) for row in label_follow_rows if row["kind"] == "demo_cross_label_shuffle")
                / max(1, sum(row["kind"] == "demo_cross_label_shuffle" for row in label_follow_rows))
            ),
            "hate_rate": (
                sum(bool(row["followed_hate"]) for row in label_follow_rows if row["kind"] == "demo_cross_label_shuffle")
                / max(1, sum(row["kind"] == "demo_cross_label_shuffle" for row in label_follow_rows))
            ),
            "interpretation": (
                "descriptive rendered-label-set overlap only; class-quota demos cover broad "
                "group/hate support, so this is not an identified causal following rate"
            ),
        },
    }
    summary = {
        "schema_version": "exploratory-qwen3-ld-evaluation/v0",
        "experiment_id": config["experiment_id"],
        "grid_id": grid["manifest"]["grid_id"],
        "query_condition_count": len(metrics_rows),
        "condition_summaries": condition_summaries,
        "paired_vs_C0": paired_effects,
        "direct_contrasts": direct_contrasts,
        "LxD_interaction": interaction,
        "label_follow": label_follow_summary,
        "label_follow_rows": label_follow_rows,
        "strata_summaries": strata_summaries,
        "inference_scope": "fixed-enriched-development-frame-only",
        "population_p_values": False,
        "scientific_eligible": False,
        "generation_ledger_sha256": full_receipt["generation_ledger_sha256"],
    }
    write_json(output_root / "evaluation" / "summary.json", summary)
    report_lines = [
        "# Qwen3-8B 词典/示例机制探索试验 v0",
        "",
        "> DEVELOPMENT ONLY · NON-SEALED · NON-SCIENTIFIC",
        "",
        f"完整自由生成网格：{len(metrics_rows)} / {config['matrix']['expected_generation_count']}。以下差异只条件于固定的 64 条机制富集开发帧。",
        "",
        "## 条件摘要",
        "",
        "| 条件 | n | 严格 JSON | group F1 | hate F1 | hard tuple F1 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for condition, condition_summary in condition_summaries.items():
        report_lines.append(
            f"| {condition} | {condition_summary['query_count']} | "
            f"{condition_summary['format']['strict_format_rate']:.3f} | "
            f"{condition_summary['field_bound']['targeted_group']['f1']:.3f} | "
            f"{condition_summary['field_bound']['hateful']['f1']:.3f} | "
            f"{condition_summary['tuple']['hard']['f1']:.3f} |"
        )
    report_lines.extend(
        [
            "",
            "## 相对 C0 的组/仇恨联合 exact 变化",
            "",
            "| 条件 | n | 均值差 | 95% bootstrap CI | wrong→correct | correct→wrong |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    endpoint = "field_bound/group_hate_joint"
    for condition, effects in paired_effects.items():
        row = effects[endpoint]
        report_lines.append(
            f"| {condition} | {row['n']} | {row['mean']:.3f} | "
            f"[{row['ci95_low']:.3f}, {row['ci95_high']:.3f}] | {row['n01']} | {row['n10']} |"
        )
    report_lines.extend(
        [
            "",
            "## 由冻结条件派生的机制直接对比：组/仇恨联合 exact",
            "",
            "| 对比（右减左） | n | 均值差 | 95% bootstrap CI | wrong→correct | correct→wrong |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for name, contrast in direct_contrasts.items():
        row = contrast["effects"][endpoint]
        report_lines.append(
            f"| {name} | {row['n']} | {row['mean']:.3f} | "
            f"[{row['ci95_low']:.3f}, {row['ci95_high']:.3f}] | {row['n01']} | {row['n10']} |"
        )
    report_lines.extend(
        [
            "",
            "## 标签跟随诊断",
            "",
            f"- CategorySwap 可解析输出与 swap 类别候选集合的重合率：{label_follow_summary['lexicon_category_swap']['rate']:.3f}。",
            f"- Demo CrossLabelShuffle 可解析输出中的 group/hate 标签集合重合率（仅描述性）："
            f"{label_follow_summary['demo_cross_label_shuffle']['group_rate']:.3f} / "
            f"{label_follow_summary['demo_cross_label_shuffle']['hate_rate']:.3f}。",
            "- demo 使用类别配额，rendered label 集合覆盖很宽；上述集合重合率不能解释为已识别的因果标签跟随率。",
            "",
            "零效应、反向效应与格式失败均保留；本报告不提供总体 p 值或总体推广结论。",
            "",
        ]
    )
    report_path = output_root / "evaluation" / "report.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    return {
        "query_condition_count": len(metrics_rows),
        "summary": str(output_root / "evaluation" / "summary.json"),
        "report": str(report_path),
    }


def _foil_outputs(gold: Sequence[Mapping[str, Any]]) -> tuple[str, str, str, dict[str, Any]]:
    from utils.quadruple import canonicalize_quadruples, serialize_quadruples

    normalized = canonicalize_quadruples(gold)
    if not normalized:
        raise PilotError("margin scoring requires at least one gold quadruple")
    gold_text = serialize_quadruples(normalized)
    group_payload = [
        {
            "target": row.target,
            "argument": row.argument,
            "targeted_group": list(row.targeted_group),
            "hateful": row.hateful,
        }
        for row in normalized
    ]
    hate_payload = copy.deepcopy(group_payload)
    source_group = group_payload[0]["targeted_group"][0]
    source_index = GROUP_ORDER.index(source_group)
    group_foil = GROUP_ORDER[(source_index + 1) % len(GROUP_ORDER)]
    group_payload[0]["targeted_group"] = [group_foil]
    source_hate = hate_payload[0]["hateful"]
    hate_foil = "non-hate" if source_hate == "hate" else "hate"
    hate_payload[0]["hateful"] = hate_foil
    return (
        gold_text,
        serialize_quadruples(canonicalize_quadruples(group_payload)),
        serialize_quadruples(canonicalize_quadruples(hate_payload)),
        {
            "gold_group": list(normalized[0].targeted_group),
            "group_foil": [group_foil],
            "gold_hateful": normalized[0].hateful,
            "hate_foil": hate_foil,
            "modified_tuple_ordinal": 0,
        },
    )


def _score_completion_sequences(
    runner: _HFRunner,
    pairs: Sequence[tuple[str, str]],
) -> list[dict[str, float | int]]:
    import torch

    tokenizer = runner.tokenizer
    sequences: list[list[int]] = []
    prompt_lengths: list[int] = []
    completion_lengths: list[int] = []
    for prompt, completion in pairs:
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        completion_ids = tokenizer.encode(completion, add_special_tokens=False)
        if not completion_ids:
            raise PilotError("teacher-forcing completion tokenization is empty")
        sequences.append(prompt_ids + completion_ids)
        prompt_lengths.append(len(prompt_ids))
        completion_lengths.append(len(completion_ids))
    max_length = max(map(len, sequences))
    if max_length > runner.max_sequence_tokens:
        raise PilotError(
            f"teacher-forcing sequence overflow: {max_length} > {runner.max_sequence_tokens}"
        )
    pad_id = tokenizer.pad_token_id
    input_ids = torch.full((len(sequences), max_length), pad_id, dtype=torch.long, device=runner.device)
    attention = torch.zeros((len(sequences), max_length), dtype=torch.long, device=runner.device)
    for index, sequence in enumerate(sequences):
        input_ids[index, : len(sequence)] = torch.tensor(sequence, dtype=torch.long, device=runner.device)
        attention[index, : len(sequence)] = 1
    with torch.inference_mode():
        logits = runner.model(input_ids=input_ids, attention_mask=attention, use_cache=False).logits
    targets = input_ids[:, 1:]
    token_log_probs = _target_token_log_probs(logits[:, :-1, :], targets)
    del logits
    result = []
    for index, (prompt_length, completion_length) in enumerate(zip(prompt_lengths, completion_lengths, strict=True)):
        start = prompt_length - 1
        values = token_log_probs[index, start : start + completion_length]
        total = float(values.sum().item())
        result.append(
            {
                "log_probability_sum": total,
                "log_probability_mean": total / completion_length,
                "completion_tokens": completion_length,
            }
        )
    return result


def _target_token_log_probs(logits: Any, targets: Any, *, time_chunk: int = 32) -> Any:
    """Compute only requested token log-probabilities without a full FP32 softmax."""

    if time_chunk <= 0 or logits.ndim != 3 or targets.shape != logits.shape[:2]:
        raise ValueError("target log-probability shapes/chunk are invalid")
    target_logits = logits.gather(-1, targets.unsqueeze(-1)).squeeze(-1).float()
    result = target_logits.new_empty(target_logits.shape)
    for start in range(0, logits.shape[1], time_chunk):
        end = min(start + time_chunk, logits.shape[1])
        normalizer = logits[:, start:end, :].float().logsumexp(dim=-1)
        result[:, start:end] = target_logits[:, start:end] - normalizer
    return result


def run_margin_scoring(
    *,
    config_path: Path = DEFAULT_CONFIG,
    output_root: Path = DEFAULT_OUTPUT,
    device: str | None = None,
    stop_at_local_time: str | None = None,
    batch_size: int | None = None,
) -> dict[str, Any]:
    config, _ = resolve_config(config_path)
    output_root = output_root.resolve()
    grid = load_json(output_root / "context_grid.json")
    margin_conditions = set(config["matrix"]["margin_conditions"])
    contexts = [
        row
        for row in grid["contexts"]
        if row["stratum"] != "no_hit" and row["condition"] in margin_conditions
    ]
    expected = 56 * len(margin_conditions)
    if len(contexts) != expected:
        raise PilotError(f"margin context count {len(contexts)} differs from {expected}")
    ledger_path = output_root / "margins" / "ledger.jsonl"
    existing: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(ledger_path) if ledger_path.exists() else []:
        if row.get("schema_version") != MARGIN_SCHEMA_VERSION:
            raise PilotError("margin ledger schema differs")
        prompt_hash = str(row.get("prompt_sha256", ""))
        prior = existing.get(prompt_hash)
        if prior is not None and prior != row:
            raise PilotError(f"margin ledger has conflicting duplicate {prompt_hash}")
        existing[prompt_hash] = row
    context_by_prompt = {str(row["prompt_sha256"]): row for row in contexts}
    for prompt_hash, row in existing.items():
        context = context_by_prompt.get(prompt_hash)
        if context is None or row.get("context_sha256") != context.get("context_sha256"):
            raise PilotError("margin resume ledger hash binding differs")
    effective_stop_time = stop_at_local_time or str(
        config["runtime"]["stop_new_gpu_batches_local_time"]
    )
    stop_at = _deadline_today(effective_stop_time)
    effective_batch_size = batch_size or max(1, int(config["runtime"]["batch_size"]) // 2)
    if effective_batch_size <= 0:
        raise PilotError("margin batch size must be positive")
    pending = [row for row in contexts if row["prompt_sha256"] not in existing]
    stopped = False
    runner: _HFRunner | None = None
    try:
        if pending:
            runner = _HFRunner(config, device=device)
        for start in range(0, len(pending), effective_batch_size):
            if datetime.now().astimezone() >= stop_at:
                stopped = True
                break
            batch = pending[start : start + effective_batch_size]
            candidates = []
            metadata = []
            for context in batch:
                gold_text, group_foil_text, hate_foil_text, foil_meta = _foil_outputs(context["gold"])
                candidates.extend(
                    [
                        (str(context["prompt_text"]), gold_text),
                        (str(context["prompt_text"]), group_foil_text),
                        (str(context["prompt_text"]), hate_foil_text),
                    ]
                )
                metadata.append((context, gold_text, group_foil_text, hate_foil_text, foil_meta))
            if runner is None:  # pragma: no cover - guarded by pending
                raise AssertionError("margin runner is unavailable")
            scores = []
            for candidate in candidates:
                scores.extend(_score_completion_sequences(runner, [candidate]))
            for index, (context, gold_text, group_foil_text, hate_foil_text, foil_meta) in enumerate(metadata):
                gold_score, group_score, hate_score = scores[index * 3 : index * 3 + 3]
                row = {
                    "schema_version": MARGIN_SCHEMA_VERSION,
                    "query_id": context["query_id"],
                    "condition": context["condition"],
                    "stratum": context["stratum"],
                    "prompt_sha256": context["prompt_sha256"],
                    "context_sha256": context["context_sha256"],
                    "scoring_contract": "full-canonical-completion-sequence-logprob/v0",
                    "gold_completion_sha256": sha256_bytes(gold_text.encode()),
                    "group_foil_completion_sha256": sha256_bytes(group_foil_text.encode()),
                    "hate_foil_completion_sha256": sha256_bytes(hate_foil_text.encode()),
                    "foil": foil_meta,
                    "gold": gold_score,
                    "group_foil": group_score,
                    "hate_foil": hate_score,
                    "group_margin_sum": float(gold_score["log_probability_sum"]) - float(group_score["log_probability_sum"]),
                    "hate_margin_sum": float(gold_score["log_probability_sum"]) - float(hate_score["log_probability_sum"]),
                    "scored_at": _now_iso(),
                }
                append_jsonl(ledger_path, row)
                existing[str(context["prompt_sha256"])] = row
    finally:
        if runner is not None:
            runner.close()
    complete = len(existing) == len(contexts)
    by_condition: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for context in contexts:
        row = existing.get(str(context["prompt_sha256"]))
        if row:
            by_condition[str(context["condition"])].append(row)
    coverage_summary = {}
    for condition, rows in sorted(by_condition.items()):
        coverage_summary[condition] = {
            "n": len(rows),
            "group_margin_mean": sum(float(row["group_margin_sum"]) for row in rows) / len(rows),
            "hate_margin_mean": sum(float(row["hate_margin_sum"]) for row in rows) / len(rows),
            "group_gold_preferred_rate": sum(float(row["group_margin_sum"]) > 0 for row in rows) / len(rows),
            "hate_gold_preferred_rate": sum(float(row["hate_margin_sum"]) > 0 for row in rows) / len(rows),
        }
    paired_effects: dict[str, Any] = {}
    direct_contrasts: dict[str, Any] = {}
    if complete:
        by_key = {
            (str(row["query_id"]), str(row["condition"])): row
            for rows in by_condition.values()
            for row in rows
        }
        replicates = int(config["analysis"]["bootstrap_replicates"])
        seed = int(config["analysis"]["bootstrap_seed"])

        def continuous_contrast(
            left_condition: str,
            right_condition: str,
            *,
            contrast_seed: int,
        ) -> dict[str, Any]:
            query_ids = sorted(
                {
                    query_id
                    for query_id, condition in by_key
                    if condition == right_condition and (query_id, left_condition) in by_key
                },
                key=int,
            )
            result: dict[str, Any] = {"n": len(query_ids)}
            for offset, field in enumerate(("group_margin_sum", "hate_margin_sum")):
                differences = [
                    float(by_key[(query_id, right_condition)][field])
                    - float(by_key[(query_id, left_condition)][field])
                    for query_id in query_ids
                ]
                result[field] = _bootstrap_paired_ci(
                    differences,
                    replicates=replicates,
                    seed=contrast_seed + offset,
                )
            return result

        for condition_index, condition in enumerate(sorted(by_condition)):
            if condition != "C0":
                paired_effects[condition] = continuous_contrast(
                    "C0",
                    condition,
                    contrast_seed=seed + 90000 + 10 * condition_index,
                )
        margin_direct_specs = {
            "D-Full_minus_D-CrossLabelShuffle": ("D-CrossLabelShuffle", "D-Full"),
            "L-Category_minus_L-CategorySwap": ("L-CategorySwap", "L-Category"),
            "L-Full_minus_L-Category": ("L-Category", "L-Full"),
            "L-Full_minus_L-Definition": ("L-Definition", "L-Full"),
            "L-Full_minus_L-DefinitionSwap": ("L-DefinitionSwap", "L-Full"),
            "LD-Full_minus_L-Full": ("L-Full", "LD-Full"),
            "LD-Full_minus_D-Full": ("D-Full", "LD-Full"),
            "LD-Full_minus_LD-CategorySwap": ("LD-CategorySwap", "LD-Full"),
            "LD-Full_minus_LD-DefinitionSwap": ("LD-DefinitionSwap", "LD-Full"),
        }
        for contrast_index, (name, (left, right)) in enumerate(margin_direct_specs.items()):
            direct_contrasts[name] = {
                "left_condition": left,
                "right_condition": right,
                **continuous_contrast(
                    left,
                    right,
                    contrast_seed=seed + 100000 + 10 * contrast_index,
                ),
            }
    receipt = {
        "schema_version": "exploratory-qwen3-ld-margin-receipt/v0",
        "grid_id": grid["manifest"]["grid_id"],
        "expected": len(contexts),
        "complete_count": len(existing),
        "complete": complete,
        "stopped_for_deadline": stopped,
        "effective_stop_new_gpu_batches_local_time": effective_stop_time,
        "effective_batch_size": effective_batch_size,
        "candidate_forward_batch_size": 1,
        "margin_ledger_sha256": file_sha256(ledger_path) if ledger_path.is_file() else None,
        "partial_margin_conclusion_allowed": False,
        "coverage_summary": coverage_summary if complete else {},
        "coverage_only_if_incomplete": coverage_summary if not complete else {},
        "paired_vs_C0": paired_effects,
        "direct_contrasts": direct_contrasts,
        "updated_at": _now_iso(),
    }
    write_json(output_root / "margins" / "summary.json", receipt)
    if complete:
        report_lines = [
            "# Qwen3-8B teacher-forcing margin 覆盖层",
            "",
            "> DEVELOPMENT ONLY · NON-SEALED · NON-SCIENTIFIC",
            "",
            f"完整评分网格：{len(existing)} / {len(contexts)}。margin 为 gold 完整序列 log-probability sum 减对应单字段 foil。",
            "",
            "## 条件摘要",
            "",
            "| 条件 | n | group margin 均值 | group gold 优先率 | hate margin 均值 | hate gold 优先率 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for condition, values in coverage_summary.items():
            report_lines.append(
                f"| {condition} | {values['n']} | {values['group_margin_mean']:.3f} | "
                f"{values['group_gold_preferred_rate']:.3f} | {values['hate_margin_mean']:.3f} | "
                f"{values['hate_gold_preferred_rate']:.3f} |"
            )
        report_lines.extend(
            [
                "",
                "## 相对 C0 的成对 margin 变化",
                "",
                "| 条件 | n | group Δ [95% CI] | hate Δ [95% CI] |",
                "|---|---:|---:|---:|",
            ]
        )
        for condition, values in paired_effects.items():
            group = values["group_margin_sum"]
            hate = values["hate_margin_sum"]
            report_lines.append(
                f"| {condition} | {values['n']} | {group['mean']:.3f} "
                f"[{group['ci95_low']:.3f}, {group['ci95_high']:.3f}] | "
                f"{hate['mean']:.3f} [{hate['ci95_low']:.3f}, {hate['ci95_high']:.3f}] |"
            )
        report_lines.extend(
            [
                "",
                "## 直接对比 margin 变化",
                "",
                "| 对比（右减左） | n | group Δ [95% CI] | hate Δ [95% CI] |",
                "|---|---:|---:|---:|",
            ]
        )
        for name, values in direct_contrasts.items():
            group = values["group_margin_sum"]
            hate = values["hate_margin_sum"]
            report_lines.append(
                f"| {name} | {values['n']} | {group['mean']:.3f} "
                f"[{group['ci95_low']:.3f}, {group['ci95_high']:.3f}] | "
                f"{hate['mean']:.3f} [{hate['ci95_low']:.3f}, {hate['ci95_high']:.3f}] |"
            )
        report_lines.extend(
            [
                "",
                "零效应和反向效应均保留；本覆盖层不提供总体 p 值或总体推广结论。",
                "",
            ]
        )
        (output_root / "margins" / "report.md").write_text(
            "\n".join(report_lines), encoding="utf-8"
        )
    return receipt
