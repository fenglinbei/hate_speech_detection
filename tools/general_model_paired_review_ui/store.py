"""Human review over frozen paired-case artifacts; never writes experiment data."""

from __future__ import annotations

import copy
import csv
import io
from pathlib import Path
from typing import Any

from build_lex.annotated_lexicon_repair import (
    LexiconRepairConflict as ReviewConflict,
    LexiconRepairError as ReviewError,
    _now_iso,
    _session_lock,
    _with_revision,
    canonical_sha256,
    file_sha256,
    read_json,
    write_json,
)


SCHEMA = "general-model-paired-human-review/v1"
CONDITIONS = ("C0", "CLnew", "CD", "CLDnew", "CLnewNoCat", "CLDnewNoCat")
TASKS = ("hate", "group")
RESOURCE_FIELDS = (
    "ambiguity_stance", "definition_fit", "category_relation",
    "demo_correspondence", "stage1_resource_notes",
)
ASSESSMENT_FIELDS = (
    "gold_dispute", "stage2_candidate_explanation", "alternative_explanation",
    "falsifiable_followup", "patching_defer_reason", "ai_comparison",
)
DISPOSITIONS = ("input_control", "verify_first", "defer")


def empty_notes() -> dict[str, Any]:
    return {
        "resources": {**dict.fromkeys(RESOURCE_FIELDS, ""), "prior_ai_exposure": "unspecified"},
        "assessment": {
            **dict.fromkeys(ASSESSMENT_FIELDS, ""),
            "gold_verdict": "",
            "disposition": "",
        },
    }


def empty_record() -> dict[str, Any]:
    return {
        **empty_notes(), "status": "unreviewed",
        "resources_locked_at": None, "resources_sha256": None,
        "ai_revealed_at": None, "pre_ai_assessment": None,
        "updated_at": None, "confirmed_at": None,
    }


def normalize_notes(value: Any) -> dict[str, Any]:
    baseline = empty_notes()
    if not isinstance(value, dict) or set(value) != set(baseline):
        raise ReviewError("审核记录结构不正确，请刷新页面后重试。")
    result = copy.deepcopy(value)
    for section, fields in baseline.items():
        if not isinstance(value[section], dict) or set(value[section]) != set(fields):
            raise ReviewError("审核字段与当前模板不一致。")
        for field, text in value[section].items():
            limit = 1000 if section == "resources" else 2000
            if not isinstance(text, str) or len(text) > limit:
                raise ReviewError("单项记录过长或格式不正确。")
    if result["resources"]["prior_ai_exposure"] not in {"unspecified", "not_seen", "seen", "unsure"}:
        raise ReviewError("此前阅读 AI 意见的状态不正确。")
    assessment = result["assessment"]
    if assessment["gold_verdict"] not in {"", "agree", "dispute", "uncertain"}:
        raise ReviewError("请选择有效的 Gold 判断。")
    if assessment["disposition"] not in {"", *DISPOSITIONS}:
        raise ReviewError("请选择有效的下一阶段用途。")
    return result


def require_resources(notes: dict[str, Any]) -> None:
    labels = {
        "ambiguity_stance": "语义与立场", "definition_fit": "词典义项适配",
        "category_relation": "类别与对象关系", "demo_correspondence": "示例对应",
    }
    missing = [label for key, label in labels.items() if not notes[key].strip()]
    if missing:
        raise ReviewError("先完成资源初读：" + "、".join(missing) + "。可明确记录不确定或无相关词条。")


def require_assessment(notes: dict[str, Any]) -> None:
    if not notes["gold_verdict"]:
        raise ReviewError("请记录对原 Gold 的判断。")
    if notes["gold_verdict"] in {"dispute", "uncertain"} and not notes["gold_dispute"].strip():
        raise ReviewError("Gold 有争议或信息不足时，请写明理由。")
    if not notes["stage2_candidate_explanation"].strip() or not notes["alternative_explanation"].strip():
        raise ReviewError("请记录候选解释与替代解释；无法判断时可以明确说明。")
    if not notes["disposition"]:
        raise ReviewError("请选择下一阶段用途。")
    if notes["disposition"] == "input_control" and not notes["falsifiable_followup"].strip():
        raise ReviewError("进入输入对照前，请写明改什么、保持什么，以及观察什么。")
    if notes["disposition"] in {"verify_first", "defer"} and not notes["patching_defer_reason"].strip():
        raise ReviewError("请写明需要先核验或暂缓使用的原因。")


class PairedReviewStore:
    def __init__(self, *, data_dir: Path, session_path: Path, reviewer_id: str) -> None:
        self.data_dir = data_dir.resolve()
        if session_path.is_symlink():
            raise ReviewError("审核会话不能使用符号链接。")
        self.session_path = session_path.resolve()
        # A caller cannot accidentally replace a frozen report, card, or source file.
        experiment_dir = self.data_dir.parent.parent
        protected = [self.data_dir]
        if self.data_dir.parent.name in {"results", "runs"}:
            protected += [experiment_dir / "results", experiment_dir / "runs"]
        if any(self.session_path.is_relative_to(path) for path in protected):
            raise ReviewError("请将人工记录保存到独立 reviews 目录，不能覆盖封存结果。")
        self.reviewer_id = reviewer_id.strip()
        if not self.reviewer_id or len(self.reviewer_id) > 100:
            raise ReviewError("复核人不能为空，且不得超过 100 字符。")
        manifest_path = self.data_dir / "manifest.json"
        self.manifest = read_json(manifest_path)
        if self.manifest.get("status") != "complete":
            raise ReviewError("只接受已完整完成的配对分析。")
        self.source_sha256 = file_sha256(manifest_path)
        self.index = read_json(self._source("cases/cards_index.json"))
        initial = read_json(self._source("cases/initial_review_batch.json"))
        self.initial_ids = [row["query_id"] for row in initial]
        self.cards: dict[str, Any] = {}
        for entry in self.index:
            query_id = entry["query_id"]
            resource_name = Path(entry["resources_card"]).name
            prefix = resource_name.removesuffix("-1-resources.md")
            if resource_name == prefix:
                raise ReviewError("案例索引中的资源页名称不正确。")
            card = read_json(self._source("cases/card_data/" + prefix + ".json"))
            if (
                query_id in self.cards
                or card["selection"]["query_id"] != query_id
                or card["selection"]["split"] != "discovery"
                or card["selection"]["focus_task"] not in TASKS
            ):
                raise ReviewError("案例索引重复或包含非 discovery 数据。")
            self.cards[query_id] = card
        if len(set(self.initial_ids)) != len(self.initial_ids) or not set(self.initial_ids) <= self.cards.keys():
            raise ReviewError("首批复核名单与 discovery 不一致。")
        self.order = self.initial_ids + [key for key in self.cards if key not in self.initial_ids]
        self.ai_notes: dict[str, dict[str, str]] = {}
        ai_path = "cases/ai_review.csv"
        if ai_path in self.manifest["artifacts"]:
            with self._source(ai_path).open(encoding="utf-8", newline="") as handle:
                self.ai_notes = {row["query_id"]: row for row in csv.DictReader(handle)}
            if not set(self.ai_notes) <= self.cards.keys():
                raise ReviewError("AI 初读名单包含非 discovery 案例。")
        with _session_lock(self.session_path):
            if self.session_path.exists():
                self._read()
            else:
                stamp = _now_iso()
                write_json(self.session_path, _with_revision({
                    "schema_version": SCHEMA,
                    "source_manifest_sha256": self.source_sha256,
                    "source_identity": self.manifest["identity"],
                    "reviewer_id": self.reviewer_id,
                    "blind_review_claimed": False,
                    "created_at": stamp, "updated_at": stamp,
                    "records": {key: empty_record() for key in self.order},
                    "events": [],
                }))

    def _source(self, relative: str) -> Path:
        path = (self.data_dir / relative).resolve()
        expected = self.manifest["artifacts"].get(relative)
        if (
            not path.is_relative_to(self.data_dir)
            or not isinstance(expected, str)
            or not path.is_file()
            or file_sha256(path) != expected
        ):
            raise ReviewError("封存案例文件缺失或哈希不一致：" + relative)
        return path

    def _read(self) -> dict[str, Any]:
        session = read_json(self.session_path)
        if (
            session.get("schema_version") != SCHEMA
            or session.get("source_manifest_sha256") != self.source_sha256
            or session.get("reviewer_id") != self.reviewer_id
            or session.get("revision") != _with_revision(session)["revision"]
            or set(session.get("records", {})) != self.cards.keys()
        ):
            raise ReviewError("审核会话与来源、复核人或记录哈希不一致，请使用匹配的会话文件。")
        for row in session["records"].values():
            normalize_notes({key: row[key] for key in empty_notes()})
            if row["resources_locked_at"]:
                require_resources(row["resources"])
                if row["resources_sha256"] != canonical_sha256(row["resources"]):
                    raise ReviewError("资源初读快照发生变化。")
            elif row["assessment"] != empty_notes()["assessment"] or row["ai_revealed_at"]:
                raise ReviewError("资源初读完成前出现了后续阶段记录。")
            if row["status"] not in {"unreviewed", "draft", "confirmed"}:
                raise ReviewError("未知的人工复核状态。")
            if row["status"] == "confirmed":
                if not row["resources_locked_at"]:
                    raise ReviewError("确认记录缺少资源初读。")
                require_assessment(row["assessment"])
            if row["ai_revealed_at"]:
                require_assessment(row["pre_ai_assessment"])
        return session

    def _summary(self, session: dict[str, Any], key: str) -> dict[str, Any]:
        card, row = self.cards[key], session["records"][key]
        return {
            "item_id": key, "query_preview": card["query"]["content"][:100],
            "focus_task": card["selection"]["focus_task"],
            "initial_batch": key in self.initial_ids,
            "status": row["status"],
            "stage": "assessment" if row["resources_locked_at"] else "resources",
            "disposition": row["assessment"]["disposition"],
            "has_ai_review": key in self.ai_notes,
        }

    def _bootstrap(self, session: dict[str, Any]) -> dict[str, Any]:
        rows = session["records"]
        confirmed = sum(row["status"] == "confirmed" for row in rows.values())
        return {
            "schema_version": SCHEMA, "revision": session["revision"],
            "reviewer_id": self.reviewer_id,
            "source_identity": self.manifest["identity"],
            "initial_ids": self.initial_ids,
            "status": {
                "item_count": len(rows), "confirmed_count": confirmed,
                "initial_count": len(self.initial_ids),
                "initial_confirmed_count": sum(rows[key]["status"] == "confirmed" for key in self.initial_ids),
                "input_control_count": sum(row["status"] == "confirmed" and row["assessment"]["disposition"] == "input_control" for row in rows.values()),
            },
            "items": [self._summary(session, key) for key in self.order],
        }

    def bootstrap(self) -> dict[str, Any]:
        with _session_lock(self.session_path):
            return self._bootstrap(self._read())

    def _item(self, session: dict[str, Any], key: str) -> dict[str, Any]:
        card, row = self.cards[key], session["records"][key]
        focus = card["selection"]["focus_task"]
        resources = card["profile"]["resources"]
        result = {
            "revision": session["revision"], "item_id": key,
            "focus_task": focus, "query": card["query"]["content"],
            "lexicon_entries": card["lexicon_entries"],
            "resource_ids": {
                name: resources.get(name, [])
                for name in ("lq_ids", "ld_only_ids", "intersection_ids", "demo_ids")
            },
            "demonstrations": [
                {
                    "id": demo["id"], "content": demo["content"],
                    "answer": demo["projection"][focus],
                    "lexicon_ids": resources["demo_match_ids"].get(str(demo["id"]), []),
                }
                for demo in card["demonstrations"]
            ],
            "review": copy.deepcopy(row),
            "has_ai_review": key in self.ai_notes,
            "trajectory": None, "ai_review": None,
        }
        if row["resources_locked_at"]:
            result["trajectory"] = {
                "gold": card["query"]["projection"],
                "conditions": card["profile"]["conditions"],
                "core_mask": card["profile"]["core_mask"],
                "six_condition_mask": card["profile"]["six_condition_mask"],
                "primary_bucket": card["selection"]["primary_bucket"],
                "candidate_labels": card["selection"]["candidate_labels"],
            }
        if row["ai_revealed_at"]:
            result["ai_review"] = copy.deepcopy(self.ai_notes[key])
        return result

    def item_state(self, key: str) -> dict[str, Any]:
        if key not in self.cards:
            raise ReviewError("该案例不在 discovery 复核队列中。")
        with _session_lock(self.session_path):
            session = self._read()
            return {**self._item(session, key), "bootstrap": self._bootstrap(session)}

    def prompt(self, key: str, condition: str, task: str) -> dict[str, Any]:
        if key not in self.cards or condition not in CONDITIONS or task not in TASKS:
            raise ReviewError("提示请求不在本次复核范围内。")
        with _session_lock(self.session_path):
            if not self._read()["records"][key]["resources_locked_at"]:
                raise ReviewError("先保存资源初读，再查看完整条件提示。")
        for context in self.cards[key]["contexts"]:
            if context["condition"] == condition and context["task"] == task:
                return {
                    "query_id": key, "condition": condition, "task": task,
                    "prompt_text": context["prompt_text"],
                    "prompt_tokens": context["prompt_tokens"],
                }
        raise ReviewError("该条件提示缺失。")

    def mutate(
        self, *, expected_revision: str, item_id: str, action: str,
        notes: dict[str, Any] | None = None, reason: str = "",
    ) -> dict[str, Any]:
        if item_id not in self.cards:
            raise ReviewError("该案例不在 discovery 复核队列中。")
        if action not in {"save", "reveal", "reveal_ai", "confirm", "reopen"}:
            raise ReviewError("未知的复核操作。")
        with _session_lock(self.session_path):
            session = self._read()
            if expected_revision != session["revision"]:
                raise ReviewConflict("记录已在另一页面更新。请保留当前草稿，再载入最新记录。")
            row = session["records"][item_id]
            stamp = _now_iso()
            if action == "reopen":
                if row["status"] != "confirmed" or not isinstance(reason, str) or not reason.strip() or len(reason) > 1000:
                    raise ReviewError("重新打开已确认记录时，请简要填写更正原因。")
                session["events"].append({
                    "at": stamp, "item_id": item_id, "action": "reopen",
                    "reason": reason.strip(), "previous_record": copy.deepcopy(row),
                })
                row["status"], row["confirmed_at"] = "draft", None
            else:
                if row["status"] == "confirmed":
                    raise ReviewConflict("该记录已确认，请先重新打开。")
                normalized = normalize_notes(notes)
                if row["resources_locked_at"]:
                    if normalized["resources"] != row["resources"]:
                        raise ReviewError("揭示轨迹前的资源初读已保存；补充意见请写在后续复核记录中。")
                elif normalized["assessment"] != empty_notes()["assessment"]:
                    raise ReviewError("先完成资源初读，再填写轨迹复核。")
                row.update(normalized)
                if action == "reveal":
                    if row["resources_locked_at"]:
                        raise ReviewError("该案例的轨迹已经开放。")
                    require_resources(row["resources"])
                    row["resources_locked_at"] = stamp
                    row["resources_sha256"] = canonical_sha256(row["resources"])
                if action in {"reveal_ai", "confirm"}:
                    if not row["resources_locked_at"]:
                        raise ReviewError("先保存资源初读并查看轨迹。")
                    require_assessment(row["assessment"])
                if action == "reveal_ai":
                    if item_id not in self.ai_notes:
                        raise ReviewError("该案例没有 AI 初读。")
                    if row["ai_revealed_at"]:
                        raise ReviewError("该案例的 AI 初读已经开放。")
                    row["ai_revealed_at"] = stamp
                    row["pre_ai_assessment"] = copy.deepcopy(row["assessment"])
                row["status"] = "confirmed" if action == "confirm" else "draft"
                if action == "confirm":
                    row["confirmed_at"] = stamp
                if action != "save":
                    session["events"].append({
                        "at": stamp, "item_id": item_id, "action": action,
                        "reviewer_id": self.reviewer_id,
                    })
            row["updated_at"], session["updated_at"] = stamp, stamp
            session = _with_revision(session)
            write_json(self.session_path, session)
            return {"bootstrap": self._bootstrap(session), "current": self._item(session, item_id)}

    def snapshot(self, expected_revision: str) -> dict[str, Any]:
        with _session_lock(self.session_path):
            session = self._read()
            if expected_revision != session["revision"]:
                raise ReviewConflict("记录已更新，请刷新后再导出。")
            return {
                "schema_version": SCHEMA + "-export",
                "exported_at": _now_iso(), "status": self._bootstrap(session)["status"],
                "session": copy.deepcopy(session),
                "case_order": self.order, "initial_ids": self.initial_ids,
            }

    def export_csv(self, snapshot: dict[str, Any]) -> bytes:
        columns = [
            "query_id", "focus_task", "primary_bucket", "reviewer", "review_kind",
            "stage1_resource_notes", "ambiguity_stance", "definition_fit",
            "category_relation", "demo_correspondence", "gold_verdict",
            "stage2_candidate_explanation", "alternative_explanation", "gold_dispute",
            "falsifiable_followup", "patching_defer_reason", "disposition",
            "ai_comparison", "prior_ai_exposure", "resources_locked_at",
            "ai_revealed_at", "review_status", "confirmed_at",
        ]
        handle = io.StringIO(newline="")
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for key in self.order:
            row = snapshot["session"]["records"][key]
            prior = row["resources"]["prior_ai_exposure"]
            kind = "human_with_ai_nonblind" if row["ai_revealed_at"] or prior == "seen" else "human_nonblind"
            if row["status"] == "unreviewed":
                kind = "unreviewed"
            values = {
                "query_id": key, "focus_task": self.cards[key]["selection"]["focus_task"],
                "primary_bucket": self.cards[key]["selection"]["primary_bucket"] if row["resources_locked_at"] else "",
                "reviewer": self.reviewer_id, "review_kind": kind,
                **row["resources"], **row["assessment"],
                "resources_locked_at": row["resources_locked_at"] or "",
                "ai_revealed_at": row["ai_revealed_at"] or "",
                "review_status": row["status"], "confirmed_at": row["confirmed_at"] or "",
            }
            # JSON retains exact notes; spreadsheet exports treat user text as text.
            writer.writerow({
                key: "'" + value if isinstance(value, str) and value.startswith(("=", "+", "-", "@", "\t", "\r")) else value
                for key, value in values.items()
            })
        return ("\ufeff" + handle.getvalue()).encode("utf-8")
