"""Explicit group-policy amendments; input bundles and historical decisions stay intact."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

if __name__ == "__main__":
    for _root in (Path(__file__).resolve().parents[2], Path(__file__).resolve().parents[2] / "src"):
        sys.path.insert(0, str(_root))

from build_lex.annotated_lexicon_repair import (
    LexiconRepairConflict as ReviewConflict, LexiconRepairError as ReviewError,
    _now_iso, _session_lock, _with_revision, canonical_sha256, file_sha256, read_json, write_json,
)


AMENDMENT_SCHEMA = "general-model-evidence-policy-amendment/v1"
GROUP_KINDS = {"query", "demo", "relation"}
GROUP_FIELDS = {
    "query": {"group", "group_reason"},
    "demo": {"group", "group_reason", "group_original_status"},
    "relation": {"topic_group", "rule_group"},
    "assessment": {"group", "group_original_status", "group_use", "explanation_choice"},
}


def policy_ref(policy: dict[str, Any]) -> dict[str, str]:
    return {key: policy[key] for key in ("version", "sha256")}


def read_policy(path: Path, bundle: dict[str, Any], bundle_sha256: str) -> dict[str, Any]:
    policy = read_json(path)
    if not isinstance(policy, dict) or policy.get("schema_version") != AMENDMENT_SCHEMA:
        raise ReviewError("规则修订文件版本不正确。")
    if not isinstance(policy.get("version"), str) or not policy["version"] or policy["version"] == bundle["policy"]["version"]:
        raise ReviewError("规则修订必须使用不同于初稿包的新版本。")
    text = policy.get("document_text")
    if not isinstance(text, str) or not text.strip() or hashlib.sha256(text.encode("utf-8")).hexdigest() != policy.get("sha256"):
        raise ReviewError("冻结规则正文与 SHA-256 不一致。")
    if policy.get("parent_policy") != policy_ref(bundle["policy"]) or policy.get("bundle_sha256") != bundle_sha256:
        raise ReviewError("规则修订的父规则或固定初稿包身份不一致。")
    approval = policy.get("approval")
    if not isinstance(approval, dict) or any(not isinstance(approval.get(key), str) or not approval[key].strip() for key in ("reviewer_id", "confirmed_at")):
        raise ReviewError("规则修订缺少明确的负责人确认记录。")
    impact = policy.get("impact")
    if not isinstance(impact, dict) or impact.get("tasks") != ["group"]:
        raise ReviewError("该迁移仅支持明确声明 group 的规则修订。")
    for name in ("object_ids", "case_ids"):
        ids = impact.get(name)
        if not isinstance(ids, list) or not ids or any(not isinstance(key, str) for key in ids) or len(ids) != len(set(ids)):
            raise ReviewError("规则影响清单必须是非空、无重复的授权 ID 清单。")
    affected = set(impact["object_ids"])
    objects, cases = bundle["objects"], bundle["cases"]
    if any(oid not in objects or objects[oid]["kind"] not in GROUP_KINDS for oid in affected):
        raise ReviewError("group 修订不能使未知、reserve、定义或命中对象失效。")
    expected_cases = {key for key, case in cases.items() if affected.intersection(case["object_ids"])}
    if set(impact["case_ids"]) != expected_cases:
        raise ReviewError("规则影响案例清单与共享材料引用不一致。")
    for oid, obj in objects.items():
        if obj["kind"] == "relation" and affected.intersection(obj.get("depends_on", [])) and oid not in affected:
            raise ReviewError("规则影响清单遗漏引用已修订查询或示例的关系。")
    return policy


def prior_task_reviews(row: dict[str, Any], policy: dict[str, Any]) -> dict[str, Any]:
    return {task: {"policy": policy_ref(policy), "status": row["status"],
                   "confirmed_at": row.get("confirmed_at"), "reviewer_id": row.get("reviewer_id"),
                   "version": row.get("version", 0)} for task in ("hate", "group")}


def mark_group_recheck(row: dict[str, Any], parent: dict[str, Any], policy: dict[str, Any]) -> None:
    row["task_reviews"] = prior_task_reviews(row, parent)
    old = copy.deepcopy(row["task_reviews"]["group"])
    row["task_reviews"]["group"].update(status="needs_recheck", target_policy=policy_ref(policy), supersedes=old)
    row["needs_group_recheck"] = True
    row["group_only_recheck"] = row["status"] == "confirmed"
    if "version" in row:
        row["non_group_version"] = row["version"]


def _backup_exact(path: Path, raw: bytes) -> None:
    if path.is_symlink():
        raise ReviewError("迁移备份不能使用符号链接。")
    if path.exists():
        if path.read_bytes() != raw:
            raise ReviewError("迁移备份已存在但字节不同。")
        return
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def migrate_policy_session(*, bundle_path: Path, policy_path: Path, session_path: Path,
                           reviewer_id: str, expected_revision: str, dry_run: bool = False,
                           actor: str = "explicit-policy-migration", expected_session_sha256: str | None = None,
                           backup_path: Path | None = None) -> dict[str, Any]:
    """Migrate one stopped writer's latest session; dry_run performs no session/backup write.

    There is intentionally no startup fallback. The caller must stop the writer,
    supply its current revision (and optionally exact bytes hash), and retain the
    exact backup. Existing v1 readers fail closed after this atomic transition.
    """
    from tools.general_model_paired_review_ui.evidence_store import EvidenceReviewStore

    session_path = Path(session_path)
    if not session_path.is_file() or session_path.is_symlink():
        raise ReviewError("显式迁移需要现有的真实会话文件。")
    if not isinstance(actor, str) or not actor.strip():
        raise ReviewError("迁移必须记录执行者。")
    store = EvidenceReviewStore(bundle_path=Path(bundle_path), session_path=session_path, reviewer_id=reviewer_id)
    policy = read_policy(Path(policy_path), store.bundle, store.bundle_sha256)
    with _session_lock(store.session_path):
        previous = store._read()
        raw = store.session_path.read_bytes()
        before_sha = hashlib.sha256(raw).hexdigest()
        if previous["revision"] != expected_revision or (expected_session_sha256 is not None and before_sha != expected_session_sha256):
            raise ReviewConflict("会话已变化；请停止写入者并重新获取最新快照，不能迁移旧记录。")
        if previous.get("policy_transition") or previous.get("policy_history"):
            raise ReviewError("该会话已有规则迁移，不能重复或倒退迁移。")
        stamp = _now_iso()
        session = copy.deepcopy(previous)
        session["policy"] = copy.deepcopy(policy)
        session["bundle_policy"] = copy.deepcopy(store.bundle["policy"])
        for oid in policy["impact"]["object_ids"]:
            mark_group_recheck(session["objects"][oid], previous["policy"], policy)
        for key in policy["impact"]["case_ids"]:
            mark_group_recheck(session["records"][key], previous["policy"], policy)
        transition = {"at": stamp, "actor": actor, "reviewer_id": reviewer_id,
                      "parent_policy": policy_ref(previous["policy"]), "active_policy": policy_ref(policy),
                      "source_revision": previous["revision"], "source_session_sha256": before_sha,
                      "impact": copy.deepcopy(policy["impact"]), "human_confirmations_added": 0}
        session["policy_transition"] = transition
        session["policy_history"] = [{"policy": copy.deepcopy(previous["policy"]),
                                      "session_sha256": before_sha, "snapshot_sha256": canonical_sha256(previous),
                                      "session": previous}]
        session["events"].append({"at": stamp, "action": "migrate_group_policy", **copy.deepcopy(transition)})
        session["updated_at"] = stamp
        session = _with_revision(session)
        backup = Path(backup_path) if backup_path else store.session_path.with_name(store.session_path.name + ".pre-policy-" + before_sha + ".json")
        if backup.resolve() in {store.session_path, store.bundle_path, Path(policy_path).resolve()}:
            raise ReviewError("迁移备份不能覆盖会话、规则或冻结初稿包。")
        result = {"dry_run": dry_run, "source_revision": previous["revision"], "source_session_sha256": before_sha,
                  "policy": policy_ref(policy), "affected_object_count": len(policy["impact"]["object_ids"]),
                  "affected_case_count": len(policy["impact"]["case_ids"]), "human_confirmations_added": 0,
                  "revision": session["revision"], "backup_path": str(backup)}
        if not dry_run:
            _backup_exact(backup, raw)
            write_json(store.session_path, session)
        return result


def main() -> int:
    parser = argparse.ArgumentParser(description="显式迁移独立证据审核的 group 规则；须先停止该会话写入者")
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--reviewer-id", required=True)
    parser.add_argument("--expected-revision", required=True)
    parser.add_argument("--expected-session-sha256")
    parser.add_argument("--backup", type=Path)
    parser.add_argument("--actor", default="explicit-policy-migration")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    result = migrate_policy_session(bundle_path=args.bundle, policy_path=args.policy, session_path=args.session,
                                    reviewer_id=args.reviewer_id, expected_revision=args.expected_revision,
                                    expected_session_sha256=args.expected_session_sha256, backup_path=args.backup,
                                    actor=args.actor, dry_run=args.dry_run)
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
