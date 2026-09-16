"""Versioned AI-assisted evidence review; immutable input and explicit human acts.

This store never opens the old paired-review writer. Bundle material is read-only;
human work lives in a separate, locked, atomically replaced session file.
"""

from __future__ import annotations

import copy
import csv
import io
from pathlib import Path
from typing import Any

from build_lex.annotated_lexicon_repair import (
    LexiconRepairConflict as ReviewConflict,
    LexiconRepairError as ReviewError,
    _now_iso, _session_lock, _with_revision, canonical_sha256,
    file_sha256, read_json, write_json,
)
from tools.general_model_paired_review_ui.evidence_schema import (
    CHOICES, KINDS, empty_assessment, normalize_assessment, normalize_values,
)
from tools.general_model_paired_review_ui.evidence_policy import (
    GROUP_FIELDS, GROUP_KINDS, policy_ref, prior_task_reviews, read_policy,
)
from tools.general_model_paired_review_ui.evidence_finalization import current_final, validate_finalizations

SCHEMA = "general-model-evidence-review/v1"
BUNDLE_SCHEMA = "general-model-evidence-bundle/v1"


def _empty_object() -> dict[str, Any]:
    return {"status": "unreviewed", "version": 0, "values": None,
            "review_kind": None, "adoption": None, "field_diff": {},
            "dependency_versions": {}, "source_version": None, "ai_version": None,
            "updated_at": None, "confirmed_at": None, "reviewer_id": None}


def _empty_case(prior_reference: dict[str, Any] | None = None) -> dict[str, Any]:
    exposure = dict.fromkeys(("gold", "resources", "results", "ai"), "unsure")
    for field, value in (prior_reference or {}).get("known_exposure", {}).items():
        if field in exposure and value == "seen":
            exposure[field] = "seen"
    return {"status": "unreviewed", "material_snapshots": [],
            "assessment": empty_assessment(), "confirmed_at": None, "updated_at": None,
            "prior_exposure": exposure}


def _forbidden_material(value: Any, *, query: bool = False) -> bool:
    """Catch accidental copying of full experiment cards into the material API."""
    banned = {"gold", "projection", "trajectories", "trajectory", "predictions", "prediction", "primary_bucket", "candidate_labels", "core_mask", "six_condition_mask", "profile"}
    if query:
        banned |= {"original_answer", "answer"}
    if isinstance(value, dict):
        return bool(banned.intersection(value)) or any(_forbidden_material(v, query=query) for v in value.values())
    if isinstance(value, list):
        return any(_forbidden_material(v, query=query) for v in value)
    return False


class EvidenceReviewStore:
    def __init__(self, *, bundle_path: Path, session_path: Path, reviewer_id: str,
                 policy_path: Path | None = None) -> None:
        self.bundle_path = Path(bundle_path).resolve()
        raw_session = Path(session_path)
        if raw_session.is_symlink():
            raise ReviewError("审核会话不能使用符号链接。")
        self.session_path = raw_session.resolve()
        if self.session_path == self.bundle_path or self.session_path.is_relative_to(self.bundle_path.parent):
            raise ReviewError("请将人工记录保存在独立 reviews 目录，不能覆盖初稿包。")
        if self.session_path.with_name(self.session_path.name + ".remote-authority.json").exists():
            raise ReviewError("该会话是远端权威记录的本地备份，不能启动旧写入者。")
        self.reviewer_id = reviewer_id.strip() if isinstance(reviewer_id, str) else ""
        if not self.reviewer_id or len(self.reviewer_id) > 100:
            raise ReviewError("复核人不能为空，且不得超过 100 字符。")
        self.bundle = read_json(self.bundle_path)
        self.bundle_sha256 = file_sha256(self.bundle_path)
        self._validate_bundle()
        self.order = self.bundle["order"]
        self.cases = self.bundle["cases"]
        self.objects = self.bundle["objects"]
        self.bundle_policy = self.bundle["policy"]
        self.policy_path = Path(policy_path).resolve() if policy_path is not None else None
        self.policy = read_policy(self.policy_path, self.bundle, self.bundle_sha256) if self.policy_path else self.bundle_policy
        self.amended = self.policy_path is not None
        with _session_lock(self.session_path):
            if self.session_path.exists():
                self._read()
            else:
                if self.amended:
                    raise ReviewError("新规则需要现有会话的显式迁移，启动服务不能创建或迁移替代记录。")
                stamp = _now_iso()
                write_json(self.session_path, _with_revision({
                    "schema_version": SCHEMA, "bundle_sha256": self.bundle_sha256,
                    "policy": copy.deepcopy(self.policy), "source_identity": copy.deepcopy(self.bundle["source_identity"]),
                    "reviewer_id": self.reviewer_id, "review_mode": "human_with_ai", "blind_review_claimed": False,
                    "created_at": stamp, "updated_at": stamp,
                    "objects": {key: _empty_object() for key in self.objects},
                    "records": {key: _empty_case(self.cases[key].get("prior_review_reference")) for key in self.order}, "events": [],
                }))

    def _validate_bundle(self) -> None:
        bundle = self.bundle
        if not isinstance(bundle, dict) or bundle.get("schema_version") != BUNDLE_SCHEMA:
            raise ReviewError("AI 初稿包版本不正确。")
        if not isinstance(bundle.get("policy"), dict) or not all(isinstance(bundle["policy"].get(k), str) and bundle["policy"][k] for k in ("version", "sha256")):
            raise ReviewError("初稿包缺少冻结规则版本或哈希。")
        cases, objects, order = bundle.get("cases"), bundle.get("objects"), bundle.get("order")
        if not isinstance(cases, dict) or not isinstance(objects, dict) or not isinstance(order, list) or not order or any(not isinstance(k, str) for k in order) or len(set(order)) != len(order) or set(order) != set(cases):
            raise ReviewError("授权 discovery 队列与案例不一致。")
        used: set[str] = set()
        for key, case in cases.items():
            ids = case.get("object_ids")
            if case.get("split", "discovery") != "discovery" or not isinstance(ids, list) or any(not isinstance(k, str) for k in ids) or len(set(ids)) != len(ids) or not set(ids) <= objects.keys():
                raise ReviewError("案例包含重复、未知或 reserve 对象。")
            query = case.get("query_object_id")
            if query not in ids or objects[query].get("kind") != "query" or not isinstance(case.get("comparison"), dict):
                raise ReviewError("案例缺少查询或独立结果对照。")
            prior = case.get("prior_review_reference", {})
            if not isinstance(prior, dict) or not isinstance(prior.get("known_exposure", {}), dict) or any(k not in {"gold", "resources", "results", "ai"} or v not in {"seen", "unsure"} for k, v in prior.get("known_exposure", {}).items()):
                raise ReviewError("旧会话只读来源或历史暴露记录不正确。")
            used.update(ids)
            for oid in ids:
                if not set(objects[oid].get("depends_on", [])) <= set(ids):
                    raise ReviewError("资源依赖必须在该案例的授权材料中。")
        if used != set(objects):
            raise ReviewError("初稿包包含授权队列未引用的对象。")
        for oid, obj in objects.items():
            if obj.get("id", oid) != oid or obj.get("kind") not in KINDS or not isinstance(obj.get("version"), str) or not obj["version"] or not isinstance(obj.get("source"), dict):
                raise ReviewError("初稿对象身份或版本不完整。")
            dependencies = obj.get("depends_on", [])
            if not isinstance(dependencies, list) or any(not isinstance(dep, str) or dep not in objects for dep in dependencies) or len(set(dependencies)) != len(dependencies) or oid in dependencies:
                raise ReviewError("对象依赖无效。")
            if _forbidden_material(obj["source"], query=obj["kind"] == "query"):
                raise ReviewError("材料层包含应在后续阶段开放的原 Gold 或结果。")
            draft = obj.get("ai_draft")
            if not isinstance(draft, dict) or not isinstance(draft.get("version"), str) or draft.get("review_kind", "ai_note") != "ai_note":
                raise ReviewError("初稿必须保存独立 AI 身份和版本。")
            normalize_values(obj["kind"], draft.get("values"), obj["source"])
        complete: set[str] = set()
        def visit(oid: str, active: set[str]) -> None:
            if oid in active:
                raise ReviewError("资源依赖中存在循环。")
            if oid in complete:
                return
            for dep in objects[oid].get("depends_on", []):
                visit(dep, active | {oid})
            complete.add(oid)
        for oid in objects:
            visit(oid, set())
        if "source_identity" not in bundle:
            raise ReviewError("初稿包缺少来源身份。")

    def _read(self) -> dict[str, Any]:
        return self._validate_session(read_json(self.session_path))

    def _validate_session(self, session: dict[str, Any]) -> dict[str, Any]:
        if session.get("schema_version") != SCHEMA or session.get("bundle_sha256") != self.bundle_sha256 or session.get("reviewer_id") != self.reviewer_id or session.get("policy") != self.policy or session.get("revision") != _with_revision(session)["revision"] or set(session.get("objects", {})) != set(self.objects) or set(session.get("records", {})) != set(self.cases):
            raise ReviewError("会话与初稿来源、规则、复核人或记录哈希不一致。")
        validate_finalizations(session, self.objects)
        for oid, row in session["objects"].items():
            if row["status"] not in {"unreviewed", "draft", "confirmed"} or type(row["version"]) is not int or row["version"] < 0:
                raise ReviewError("未知的对象审核状态或版本。")
            if row["values"] is not None:
                obj = self.objects[oid]
                normalize_values(obj["kind"], row["values"], obj["source"], required=row["status"] == "confirmed" and not row.get("needs_group_recheck"), allow_policy_changed=self.amended,
                                 final_label_only=bool(current_final(row)))
            if row["status"] == "confirmed" and (row["values"] is None or row["review_kind"] != "human_with_ai" or row["reviewer_id"] != self.reviewer_id or not row["confirmed_at"]):
                raise ReviewError("已确认对象缺少人工确认身份。")
        for key, row in session["records"].items():
            if row["status"] not in {"unreviewed", "draft", "confirmed"}:
                raise ReviewError("未知的案例审核状态。")
            known = self.cases[key].get("prior_review_reference", {}).get("known_exposure", {})
            if any(value == "seen" and row["prior_exposure"].get(field) != "seen" for field, value in known.items()):
                raise ReviewError("已知历史暴露不能被清空或改为未见过。")
            for snapshot in row["material_snapshots"]:
                if snapshot["sha256"] != canonical_sha256({k: v for k, v in snapshot.items() if k != "sha256"}) or set(snapshot["objects"]) != set(self.cases[key]["object_ids"]):
                    raise ReviewError("材料核对快照发生变化。")
            if not row["material_snapshots"] and row["assessment"] != empty_assessment():
                raise ReviewError("材料核对快照前出现了后续裁决。")
            if row["status"] == "confirmed":
                if not row["material_snapshots"] or not row["confirmed_at"]:
                    raise ReviewError("已确认案例缺少材料快照。")
                normalize_assessment(row["assessment"], self._gold(key), required=not row.get("needs_group_recheck"), allow_policy_changed=self.amended)
        if self.amended:
            transition = session.get("policy_transition", {})
            history = session.get("policy_history", [])
            if (transition.get("active_policy") != policy_ref(self.policy) or transition.get("impact") != self.policy["impact"]
                    or session.get("bundle_policy") != self.bundle_policy or len(history) != 1):
                raise ReviewError("新规则会话缺少显式迁移及父规则记录。")
            archive = history[0]
            prior = archive.get("session", {})
            if (archive.get("snapshot_sha256") != canonical_sha256(prior)
                    or prior.get("revision") != _with_revision(prior)["revision"]
                    or prior.get("revision") != transition.get("source_revision")
                    or prior.get("policy") != self.bundle_policy):
                raise ReviewError("迁移前的完整会话封存记录发生变化。")
            for name, ids in (("objects", self.policy["impact"]["object_ids"]), ("records", self.policy["impact"]["case_ids"])):
                for identity in ids:
                    row = session[name][identity]
                    tasks = row.get("task_reviews", {})
                    if set(tasks) != {"hate", "group"} or type(row.get("needs_group_recheck")) is not bool or type(row.get("group_only_recheck")) is not bool:
                        raise ReviewError("受影响记录缺少按任务的规则与重核状态。")
                    for task, state in tasks.items():
                        if state.get("status") not in {"unreviewed", "draft", "confirmed", "needs_recheck"}:
                            raise ReviewError("未知的按任务确认状态。")
                        allowed = [policy_ref(self.bundle_policy)] if task == "hate" else [policy_ref(self.bundle_policy), policy_ref(self.policy)]
                        final_policy = row.get("final_annotation", {}).get("task_policies", {}).get(task)
                        if final_policy:
                            allowed.append(final_policy)
                        if state.get("policy") not in allowed:
                            raise ReviewError("按任务记录引用了错误的规则版本。")
                    if not row["needs_group_recheck"] and tasks["group"]["policy"] not in [policy_ref(self.policy), row.get("final_annotation", {}).get("task_policies", {}).get("group")]:
                        raise ReviewError("旧规则 group 不能冒充已按新规则完成重核。")
        return session

    def _gold(self, key: str) -> dict[str, Any]:
        return self.cases[key]["comparison"].get("gold", {})

    @staticmethod
    def _task_status(row: dict[str, Any], task: str) -> str:
        return row.get("task_reviews", {}).get(task, {}).get("status", row["status"])

    @staticmethod
    def _semantic_version(row: dict[str, Any], task: str | None = None) -> int:
        return row.get("non_group_version", row["version"]) if task == "hate" else row["version"]

    def _object_ready(self, session: dict[str, Any], oid: str, task: str | None = None) -> bool:
        row = session["objects"][oid]
        status = self._task_status(row, "hate") if task == "hate" else row["status"]
        return status == "confirmed" and not self._object_stale(session, oid, task=task)

    def _object_stale(self, session: dict[str, Any], oid: str, seen: set[str] | None = None,
                      *, task: str | None = None) -> bool:
        row = session["objects"][oid]
        if task != "hate" and row.get("needs_group_recheck"):
            return True
        status = self._task_status(row, "hate") if task == "hate" else row["status"]
        if status != "confirmed":
            return False
        visited = set() if seen is None else set(seen)
        if oid in visited:
            raise ReviewError("资源依赖中存在循环。")
        visited.add(oid)
        # A word-sense hit does not depend on group-category decisions. A group's
        # new confirmation must not invalidate its already reviewed text/sense edge.
        dependency_task = "hate" if task == "hate" or (self.amended and self.objects[oid]["kind"] == "hit") else None
        refs = row.get("dependency_non_group_versions", row["dependency_versions"]) if dependency_task == "hate" else row["dependency_versions"]
        return any((self._task_status(session["objects"][dep], "hate") if dependency_task == "hate" else session["objects"][dep]["status"]) != "confirmed"
                   or refs.get(dep) != self._semantic_version(session["objects"][dep], dependency_task)
                   or self._object_stale(session, dep, visited, task=dependency_task)
                   for dep in self.objects[oid].get("depends_on", []))

    def _case_stale(self, session: dict[str, Any], key: str, *, task: str | None = None) -> bool:
        snapshots = session["records"][key]["material_snapshots"]
        if not snapshots:
            return False
        versions = snapshots[-1]["objects"]
        return any(not self._object_ready(session, oid, task) or self._semantic_version(session["objects"][oid], task) != self._semantic_version(ref, task) for oid, ref in versions.items())

    def _summary(self, session: dict[str, Any], key: str) -> dict[str, Any]:
        case, row = self.cases[key], session["records"][key]
        stale = self._case_stale(session, key)
        return {"item_id": key, "query_preview": self.objects[case["query_object_id"]]["source"].get("text", "")[:100],
                "status": "needs_attention" if stale or (row.get("needs_group_recheck") and row["status"] == "confirmed") else row["status"],
                "stage": "materials" if not row["material_snapshots"] or stale else "assessment",
                "object_count": len(case["object_ids"]),
                "confirmed_object_count": sum(session["objects"][oid]["status"] == "confirmed" and not self._object_stale(session, oid) for oid in case["object_ids"]),
                "stale": stale, "needs_group_recheck": row.get("needs_group_recheck", False),
                "requires_current_group_review": row.get("needs_group_recheck", False)}

    def _bootstrap(self, session: dict[str, Any]) -> dict[str, Any]:
        items = [self._summary(session, key) for key in self.order]
        choices = copy.deepcopy(CHOICES)
        if not self.amended:
            choices["original_status"].pop("policy_changed", None)
        return {"schema_version": SCHEMA, "revision": session["revision"], "reviewer_id": self.reviewer_id, "bundle_sha256": self.bundle_sha256,
                "policy": copy.deepcopy(self.policy), "source_identity": copy.deepcopy(self.bundle["source_identity"]),
                "bundle_policy": copy.deepcopy(self.bundle_policy), "policy_transition": copy.deepcopy(session.get("policy_transition")),
                "choices": choices, "items": items,
                "status": {"item_count": len(items), "confirmed_count": sum(r["status"] == "confirmed" for r in items),
                "object_count": len(self.objects), "confirmed_object_count": sum(r["status"] == "confirmed" and not self._object_stale(session, oid) for oid, r in session["objects"].items()),
                "needs_group_recheck_count": sum(bool(r.get("needs_group_recheck")) for r in session["objects"].values())}}

    def bootstrap(self) -> dict[str, Any]:
        with _session_lock(self.session_path):
            return self._bootstrap(self._read())

    def _item(self, session: dict[str, Any], key: str) -> dict[str, Any]:
        case, row = self.cases[key], session["records"][key]
        visible_review = copy.deepcopy(row)
        if row["material_snapshots"]:
            # This is a read-only presentation of demonstrable exposure. The
            # historical self-report remains byte-for-byte in the session/archive.
            visible_review["prior_exposure"] = dict.fromkeys(("gold", "resources", "results", "ai"), "seen")
        objects = []
        for oid in case["object_ids"]:
            obj, review = self.objects[oid], session["objects"][oid]
            visible_object_review = copy.deepcopy(review)
            group_only = bool(review.get("group_only_recheck")) and not self._object_stale(session, oid, task="hate")
            if "group_only_recheck" in visible_object_review:
                visible_object_review["group_only_recheck"] = group_only
            ai = obj["ai_draft"]
            effective = review["values"] if review["values"] is not None else normalize_values(obj["kind"], ai["values"], obj["source"])
            objects.append({"id": oid, "kind": obj["kind"], "version": obj["version"], "source": copy.deepcopy(obj["source"]),
                            "depends_on": copy.deepcopy(obj.get("depends_on", [])),
                            "ai_draft": {"version": ai["version"], "review_kind": "ai_note", "policy": copy.deepcopy(self.bundle_policy), "values": copy.deepcopy(ai["values"]), "provenance": copy.deepcopy(ai.get("provenance", {}))},
                            "review": visible_object_review, "effective_values": copy.deepcopy(effective),
                            "stale": self._object_stale(session, oid),
                            "needs_group_recheck": review.get("needs_group_recheck", False),
                            "group_only_recheck": group_only,
                            "task_stale": {"hate": self._object_stale(session, oid, task="hate"), "group": self._object_stale(session, oid)},
                            "task_reviews": copy.deepcopy(review.get("task_reviews", prior_task_reviews(review, self.bundle_policy)))})
        return {"revision": session["revision"], "item_id": key,
                "query": self.objects[case["query_object_id"]]["source"].get("text", ""),
                "query_object_id": case["query_object_id"], "objects": objects,
                "prior_review_reference": {field: copy.deepcopy(case["prior_review_reference"][field]) for field in ("source_session_sha256", "status", "record_sha256", "known_exposure") if field in case.get("prior_review_reference", {})},
                "review": visible_review, "stale": self._case_stale(session, key),
                "needs_group_recheck": row.get("needs_group_recheck", False),
                "group_only_recheck": row.get("group_only_recheck", False),
                "task_reviews": copy.deepcopy(row.get("task_reviews", prior_task_reviews(row, self.bundle_policy))),
                "comparison_previously_revealed": bool(row["material_snapshots"]),
                "comparison": copy.deepcopy(case["comparison"]) if row["material_snapshots"] else None}

    def item_state(self, key: str) -> dict[str, Any]:
        self._case_id(key)
        with _session_lock(self.session_path):
            session = self._read()
            return {**self._item(session, key), "bootstrap": self._bootstrap(session)}

    def _case_id(self, key: str) -> None:
        if not isinstance(key, str) or key not in self.cases:
            raise ReviewError("该案例不在授权 discovery 复核队列中。")

    def _object_id(self, key: str, oid: Any) -> None:
        if not isinstance(oid, str) or oid not in self.cases[key]["object_ids"]:
            raise ReviewError("该对象不在当前案例的可核对材料中。")

    def _event(self, session: dict[str, Any], key: str, action: str, **extra: Any) -> None:
        session["events"].append({"at": _now_iso(), "item_id": key, "action": action, "reviewer_id": self.reviewer_id, **copy.deepcopy(extra)})

    @staticmethod
    def _assert_group_only(kind: str, previous: dict[str, Any], values: dict[str, Any]) -> None:
        if any(previous.get(field) != value for field, value in values.items() if field not in GROUP_FIELDS[kind]):
            raise ReviewError("本次只重核 group。修改 hate 或共同依据前，请先明确重新打开原记录。")

    def _update_task_reviews(self, row: dict[str, Any], previous: dict[str, Any], *, confirm: bool,
                             group_only: bool) -> None:
        if not self.amended:
            return
        tasks = copy.deepcopy(previous.get("task_reviews", prior_task_reviews(previous, self.bundle_policy)))
        for task in (("group",) if group_only else ("hate", "group")):
            prior = tasks[task]
            state = {"policy": copy.deepcopy(previous.get("final_annotation", {}).get("task_policies", {}).get(task) or policy_ref(self.policy if task == "group" else self.bundle_policy)),
                     "status": "confirmed" if confirm else "draft", "version": row.get("version", 0),
                     "confirmed_at": _now_iso() if confirm else None,
                     "reviewer_id": self.reviewer_id if confirm else prior.get("reviewer_id"),
                     "supersedes": copy.deepcopy(prior.get("supersedes", prior))}
            if not confirm and previous.get("needs_group_recheck") and task == "group":
                state["target_policy"] = policy_ref(self.policy)
            tasks[task] = state
        row["task_reviews"] = tasks
        row["needs_group_recheck"] = bool(previous.get("needs_group_recheck")) and not confirm
        row["group_only_recheck"] = group_only and bool(previous.get("group_only_recheck")) and not confirm
        if "version" in row:
            row["non_group_version"] = self._semantic_version(previous, "hate") if group_only else row["version"]

    def _reopen_tasks(self, row: dict[str, Any]) -> None:
        if not self.amended:
            return
        tasks = row.get("task_reviews", prior_task_reviews(row, self.bundle_policy))
        for state in tasks.values():
            old = copy.deepcopy(state)
            state.update(status="draft", confirmed_at=None, supersedes=old)
        row["task_reviews"] = tasks
        row["group_only_recheck"] = False
        if "version" in row:
            row["non_group_version"] = row["version"]

    def _save_object(self, session: dict[str, Any], key: str, oid: str, values: Any, version: Any, *, confirm: bool) -> None:
        self._object_id(key, oid)
        obj, row = self.objects[oid], session["objects"][oid]
        if type(version) is not int or version != row["version"]:
            raise ReviewConflict("对象版本已变化。请保留当前草稿，再载入最新版本。")
        stale = self._object_stale(session, oid)
        group_only = bool(row.get("group_only_recheck")) and not self._object_stale(session, oid, task="hate")
        if row["status"] == "confirmed" and not group_only and not stale:
            raise ReviewConflict("该对象已确认，请先重新打开。")
        normalized = normalize_values(obj["kind"], values, obj["source"], required=confirm, allow_policy_changed=self.amended)
        if group_only:
            self._assert_group_only(obj["kind"], row["values"], normalized)
        dependency_task = "hate" if self.amended and obj["kind"] == "hit" else None
        if confirm and any(not self._object_ready(session, dep, dependency_task) for dep in obj.get("depends_on", [])):
            raise ReviewError("请先确认此项引用的查询、示例或义项，再核对当前关系。")
        previous = copy.deepcopy(row)
        if row["status"] == "confirmed" and stale and not group_only:
            self._event(session, key, "refresh_stale_object", object_id=oid, previous_record=previous)
        row.update(values=normalized, version=row["version"] + 1, status="confirmed" if confirm or group_only else "draft", updated_at=_now_iso())
        row["source_version"], row["ai_version"] = obj["version"], obj["ai_draft"]["version"]
        ai_values = normalize_values(obj["kind"], obj["ai_draft"]["values"], obj["source"])
        row["field_diff"] = {field: {"ai": ai_values[field], "human": value} for field, value in normalized.items() if value != ai_values[field]}
        if obj["kind"] in GROUP_KINDS:
            self._update_task_reviews(row, previous, confirm=confirm, group_only=group_only)
        if confirm:
            row.update(review_kind="human_with_ai", adoption="adopted_modified" if row["field_diff"] else "adopted_unchanged", reviewer_id=self.reviewer_id, confirmed_at=_now_iso())
            row["dependency_versions"] = {dep: session["objects"][dep]["version"] for dep in obj.get("depends_on", [])}
            if self.amended:
                row["dependency_non_group_versions"] = (copy.deepcopy(previous.get("dependency_non_group_versions", previous["dependency_versions"])) if group_only else
                    {dep: self._semantic_version(session["objects"][dep], "hate") for dep in obj.get("depends_on", [])})
            self._event(session, key, "confirm_object", object_id=oid, version=row["version"], previous_record=previous)

    def mutate(self, *, expected_revision: str, item_id: str, action: str,
               object_id: str | None = None, object_version: int | None = None,
               values: dict[str, Any] | None = None, objects: list[dict[str, Any]] | None = None,
               assessment: dict[str, Any] | None = None, reason: str = "",
               prior_exposure: dict[str, str] | None = None) -> dict[str, Any]:
        self._case_id(item_id)
        if action not in {"save_object", "confirm_object", "reopen_object", "confirm_batch", "reveal", "save_assessment", "confirm", "reopen", "save_exposure"}:
            raise ReviewError("未知的证据审核操作。")
        with _session_lock(self.session_path):
            session = self._read()
            if expected_revision != session["revision"]:
                raise ReviewConflict("记录已在另一页面更新。请保留当前草稿，再载入最新记录。")
            row = session["records"][item_id]
            stamp = _now_iso()
            if prior_exposure is not None:
                if not isinstance(prior_exposure, dict) or set(prior_exposure) != {"gold", "resources", "results", "ai"} or any(not isinstance(v, str) or v not in CHOICES["exposure"] for v in prior_exposure.values()):
                    raise ReviewError("历史暴露记录格式不正确。")
                known = dict(self.cases[item_id].get("prior_review_reference", {}).get("known_exposure", {}))
                known.update({field: "seen" for field, value in row["prior_exposure"].items() if value == "seen"})
                if row["material_snapshots"]:
                    known.update(gold="seen", resources="seen", results="seen", ai="seen")
                if any(value == "seen" and prior_exposure.get(field) != "seen" for field, value in known.items()):
                    raise ReviewError("旧会话已记录见过该材料，不能改成未见过或不确定。")
                if prior_exposure != row["prior_exposure"]:
                    self._event(session, item_id, "update_exposure", previous_exposure=row["prior_exposure"], exposure=prior_exposure)
                row["prior_exposure"] = copy.deepcopy(prior_exposure)
            if action in {"save_object", "confirm_object"}:
                self._save_object(session, item_id, object_id, values, object_version, confirm=action == "confirm_object")
            elif action == "confirm_batch":
                if not isinstance(objects, list) or not objects or len(objects) > 100:
                    raise ReviewError("请显式选择本次已核对的对象及版本。")
                ids = [o.get("object_id") for o in objects if isinstance(o, dict)]
                if len(ids) != len(objects) or any(not isinstance(oid, str) for oid in ids) or len(set(ids)) != len(ids):
                    raise ReviewError("批量清单包含重复或无效对象。")
                for selected in objects:
                    oid = selected["object_id"]
                    self._object_id(item_id, oid)
                    current = session["objects"][oid]
                    effective = current["values"] if current["values"] is not None else self.objects[oid]["ai_draft"]["values"]
                    self._save_object(session, item_id, oid, selected.get("values", effective), selected.get("object_version"), confirm=True)
                self._event(session, item_id, "confirm_batch", selected=[{"object_id": oid, "version": session["objects"][oid]["version"], "source_version": self.objects[oid]["version"], "ai_version": self.objects[oid]["ai_draft"]["version"]} for oid in ids])
            elif action == "reopen_object":
                self._object_id(item_id, object_id)
                target = session["objects"][object_id]
                if type(object_version) is not int or object_version != target["version"]:
                    raise ReviewConflict("对象版本已变化，请载入最新版本。")
                if target["status"] != "confirmed":
                    raise ReviewError("只有已确认对象需要重新打开。")
                self._require_reason(reason)
                self._event(session, item_id, action, object_id=object_id, reason=reason.strip(), previous_record=target)
                target.update(status="draft", version=target["version"] + 1, confirmed_at=None, updated_at=stamp)
                if self.objects[object_id]["kind"] in GROUP_KINDS:
                    self._reopen_tasks(target)
            elif action == "reveal":
                hate_changed = bool(self.amended and row["material_snapshots"] and self._case_stale(session, item_id, task="hate"))
                missing = [oid for oid in self.cases[item_id]["object_ids"] if session["objects"][oid]["status"] != "confirmed" or self._object_stale(session, oid)]
                if missing:
                    raise ReviewError("请逐项核对并确认全部必需材料；尚有 %d 项未确认或引用已更新。" % len(missing))
                if row["material_snapshots"] and not self._case_stale(session, item_id):
                    raise ReviewError("当前材料快照已保存，结果对照已经开放。")
                snapshot = {"at": stamp, "reviewer_id": self.reviewer_id, "policy": self.policy,
                            "objects": {oid: copy.deepcopy(session["objects"][oid]) for oid in self.cases[item_id]["object_ids"]}}
                if self.amended:
                    snapshot["task_policies"] = {"hate": policy_ref(self.bundle_policy), "group": policy_ref(self.policy)}
                snapshot["sha256"] = canonical_sha256(snapshot)
                if row["status"] == "confirmed":
                    self._event(session, item_id, "refresh_materials", previous_record=row)
                if hate_changed:
                    self._event(session, item_id, "invalidate_hate_after_material_change", previous_task_reviews=row.get("task_reviews"))
                    self._reopen_tasks(row)
                row["material_snapshots"].append(snapshot)
                if len(row["material_snapshots"]) == 1:
                    query_values = session["objects"][self.cases[item_id]["query_object_id"]]["values"]
                    row["assessment"].update({task: copy.deepcopy(query_values[task]) for task in ("hate", "group")})
                row.update(status="draft", confirmed_at=None)
                self._event(session, item_id, action, snapshot_sha256=snapshot["sha256"])
            elif action in {"save_assessment", "confirm"}:
                if not row["material_snapshots"] or self._case_stale(session, item_id):
                    raise ReviewError("请先保存当前版本的全部材料核对快照，再裁决原标注。")
                group_only = row.get("group_only_recheck", False)
                if row["status"] == "confirmed" and not group_only:
                    raise ReviewConflict("该案例已确认，请先重新打开。")
                previous = copy.deepcopy(row)
                normalized = normalize_assessment(assessment, self._gold(item_id), required=action == "confirm", allow_policy_changed=self.amended)
                if group_only:
                    self._assert_group_only("assessment", row["assessment"], normalized)
                row["assessment"] = normalized
                proposal = self.cases[item_id]["comparison"].get("ai_explanation", {})
                if isinstance(proposal, dict) and proposal.get("status") == "awaiting_material_review" and row["assessment"]["explanation_choice"] != "defer":
                    raise ReviewError("尚无基于材料裁决的解释提案，请选择暂不采纳；不影响完成本例。")
                row["status"] = "confirmed" if action == "confirm" else "draft"
                self._update_task_reviews(row, previous, confirm=action == "confirm", group_only=group_only)
                if action == "confirm":
                    row["confirmed_at"] = stamp
                    row["review_kind"] = "human_with_ai"
                    row["adjudication_mode"] = "single_review"
                    row["reviewer_id"] = self.reviewer_id
                    self._event(session, item_id, action, assessment=row["assessment"], material_sha256=row["material_snapshots"][-1]["sha256"])
            elif action == "reopen":
                if row["status"] != "confirmed" and not row.get("group_only_recheck"):
                    raise ReviewError("只有已确认案例需要重新打开。")
                self._require_reason(reason)
                self._event(session, item_id, action, reason=reason.strip(), previous_record=row)
                row.update(status="draft", confirmed_at=None)
                self._reopen_tasks(row)
            if row["status"] == "unreviewed":
                row["status"] = "draft"
            row["updated_at"], session["updated_at"] = stamp, stamp
            session = _with_revision(session)
            write_json(self.session_path, session)
            return {"bootstrap": self._bootstrap(session), "current": self._item(session, item_id)}

    @staticmethod
    def _require_reason(reason: Any) -> None:
        if not isinstance(reason, str) or not reason.strip() or len(reason) > 1000:
            raise ReviewError("重新打开时请选择或简短填写更正原因。")

    def _eligibility(self, session: dict[str, Any], row: dict[str, Any], values: dict[str, Any],
                     *, key: str | None = None, oid: str | None = None) -> dict[str, Any]:
        result = {}
        for task in ("hate", "group"):
            state = row.get("task_reviews", {}).get(task, {})
            status = self._task_status(row, task)
            stale = (self._case_stale(session, key, task="hate" if task == "hate" else None) if key is not None
                     else self._object_stale(session, oid, task="hate" if task == "hate" else None))
            pending = task == "group" and row.get("needs_group_recheck", False)
            policy = copy.deepcopy(state.get("policy", policy_ref(self.bundle_policy)))
            available = status == "confirmed" and not stale and not pending and task in values and values[task] is not None
            if key is not None and not row["material_snapshots"]:
                available = False
            availability = ("needs_recheck" if pending or stale else status if status != "confirmed"
                            else "resolved" if available else "unresolved")
            result[task] = {"policy": policy, "status": status, "availability": availability,
                            "reference_eligible": bool(available), "stale": bool(stale),
                            "confirmed_at": state.get("confirmed_at", row.get("confirmed_at")),
                            "reviewer_id": state.get("reviewer_id", row.get("reviewer_id"))}
        return result

    def snapshot(self, expected_revision: str) -> dict[str, Any]:
        with _session_lock(self.session_path):
            session = self._read()
            if expected_revision != session["revision"]:
                raise ReviewConflict("记录已更新，请载入最新记录后导出。")
            case_eligibility = {}
            for key in self.order:
                row = session["records"][key]
                tasks = self._eligibility(session, row, row["assessment"], key=key)
                case_eligibility[key] = {"stale": self._case_stale(session, key), "tasks": tasks,
                                         "reference_tasks": [task for task in ("hate", "group") if tasks[task]["reference_eligible"]]}
            return {"schema_version": SCHEMA + "-export", "exported_at": _now_iso(), "status": self._bootstrap(session)["status"],
                    "session": copy.deepcopy(session), "case_order": self.order,
                    "object_identity": {oid: {"kind": obj["kind"], "version": obj["version"], "source_sha256": canonical_sha256(obj["source"]), "ai_version": obj["ai_draft"]["version"], "ai_policy": policy_ref(self.bundle_policy)} for oid, obj in self.objects.items()},
                    "object_staleness": {oid: self._object_stale(session, oid) for oid in self.objects},
                    "object_eligibility": {oid: self._eligibility(session, row, row["values"] or {}, oid=oid) for oid, row in session["objects"].items() if self.objects[oid]["kind"] in GROUP_KINDS},
                    "case_eligibility": case_eligibility}

    def export_csv(self, snapshot: dict[str, Any]) -> bytes:
        task_columns = tuple(task + "_" + field for task in ("hate", "group") for field in ("policy_version", "policy_sha256", "availability", "reference_eligible"))
        columns = ("record_type", "case_id", "object_id", "kind", "status", "version", "reviewer_id", "review_kind", "adoption", "hate", "group", "hate_original_status", "group_original_status", "hate_use", "group_use", "stale", "needs_group_recheck", *task_columns, "attack_severity", "adopted_definition", "final_acceptance_id", "final_annotation_json", "values_json")
        handle = io.StringIO(newline="")
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        import json
        def output(row: dict[str, Any]) -> None:
            writer.writerow({k: "'" + v if isinstance(v, str) and v.startswith(("=", "+", "-", "@", "\t", "\r")) else v for k, v in row.items()})
        def task_fields(tasks: dict[str, Any]) -> dict[str, Any]:
            return {task + "_" + field: value for task, state in tasks.items() for field, value in (
                ("policy_version", state["policy"]["version"]), ("policy_sha256", state["policy"]["sha256"]),
                ("availability", state["availability"]), ("reference_eligible", state["reference_eligible"]))}
        for oid, row in snapshot["session"]["objects"].items():
            vals = row["values"] or {}
            final = current_final(row) or {}
            output({"record_type": "object", "object_id": oid, "kind": self.objects[oid]["kind"],
                    **{k: row.get(k, "") for k in ("status", "version", "reviewer_id", "review_kind", "adoption")},
                    **{k: json.dumps(vals[k], ensure_ascii=False) if k in vals else "" for k in ("hate", "group")},
                    **{k: vals.get(k, "") for k in ("hate_original_status", "group_original_status")},
                    "needs_group_recheck": row.get("needs_group_recheck", False),
                    "stale": snapshot["object_staleness"][oid],
                    **task_fields(snapshot["object_eligibility"].get(oid, {})),
                    "attack_severity": final.get("values", {}).get("attack_severity", ""),
                    "adopted_definition": final.get("adopted_definition", "") or "",
                    "final_acceptance_id": final.get("batch_id", ""),
                    "final_annotation_json": json.dumps(row.get("final_annotation"), ensure_ascii=False),
                    "values_json": json.dumps(vals, ensure_ascii=False)})
        for key in self.order:
            row = snapshot["session"]["records"][key]
            vals = row["assessment"]
            output({"record_type": "case", "case_id": key, "kind": "assessment", "status": row["status"],
                    "reviewer_id": row.get("reviewer_id", ""), "review_kind": row.get("review_kind", ""),
                    **{k: vals[k] for k in ("hate_original_status", "group_original_status", "hate_use", "group_use")},
                    **{k: json.dumps(vals[k], ensure_ascii=False) for k in ("hate", "group")},
                    "needs_group_recheck": row.get("needs_group_recheck", False),
                    **task_fields(snapshot["case_eligibility"][key]["tasks"]),
                    "stale": snapshot["case_eligibility"][key]["stale"], "values_json": json.dumps(vals, ensure_ascii=False)})
        return ("\ufeff" + handle.getvalue()).encode("utf-8")
