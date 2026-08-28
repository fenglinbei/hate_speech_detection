from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from data.stage1_data import (
    Stage1DataError,
    _adjudication_composition,
    audit_data,
    export_data_adjudication,
    export_data_review_subset,
    finalize_data,
    merge_completed_adjudications,
    prepare_data_declaration,
    validate_data,
    validate_data_target,
)
from data.training_artifacts import build_payload_manifest, write_canonical_json
from review.blind_review import run_data_review
from review.data_review_artifact import seal_data_review


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SOURCE_CONFIG = REPOSITORY_ROOT / "config/stage1/context_factorial.json"
SOURCE_RUBRIC = REPOSITORY_ROOT / "config/stage1/data_adjudication_rubric.md"
SOURCE_BLIND_POLICY = REPOSITORY_ROOT / "config/stage1/blind_review.json"


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
        for row in sorted(rows, key=lambda row: row["issue_id"])
    )
    path.write_text(payload, encoding="utf-8")


class TemporaryStage1Workspace:
    def __init__(self, root: Path, *, enforce_expected_counts: bool = True):
        self.root = root
        self.config_path = root / "config/stage1/context_factorial.json"
        self.rubric_path = root / "config/stage1/data_adjudication_rubric.md"
        self.audit_ref = root / "refs/data_audit_ref.json"
        self.data_ref = root / "refs/data_ref.json"
        self.data_blind_review_ref = root / "refs/data_blind_review_ref.json"
        self.adjudication_file = root / "review/data_adjudication.jsonl"
        self.declaration_file = root / "review/data_reviewer_declaration.json"

        for split in ("train", "test"):
            source = REPOSITORY_ROOT / f"data/full/std/{split}.json"
            destination = root / f"data/full/std/{split}.json"
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
        config = json.loads(SOURCE_CONFIG.read_text(encoding="utf-8"))
        config["artifact_root"] = "artifacts"
        if not enforce_expected_counts:
            config["data"]["expected_counts"] = {
                "group_hate_blocking": None,
                "field_type_blocking": None,
                "blocking_total": None,
                "substring_warnings": None,
            }
        write_json(self.config_path, config)
        self.rubric_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(SOURCE_RUBRIC, self.rubric_path)
        self.blind_policy_path = root / "config/stage1/blind_review.json"
        shutil.copy2(SOURCE_BLIND_POLICY, self.blind_policy_path)

    def audit(self) -> tuple[dict, Path]:
        ref = audit_data(
            config_path=self.config_path,
            review_rubric_path=self.rubric_path,
            write_ref=self.audit_ref,
        )
        return ref, Path(ref["target_path"])

    def completed_rows(self, audit_target: Path) -> list[dict]:
        rows = []
        for issue in read_jsonl(audit_target / "issues.jsonl"):
            decision = "corrected"
            edits = []
            code = issue["issue_code"]
            if code == "group-hate-atypical":
                decision = "accepted"
                reason_code = "valid-independent-label-combination"
            elif code == "hateful-null-sentinel":
                location = issue["locations"][0]
                reason_code = "replace-legacy-null"
                edits = [
                    {
                        "location_index": 0,
                        "op": "set",
                        "json_pointer": f"/quadruples/{location['tuple_index']}/hateful",
                        "value": "hate",
                    }
                ]
            elif code == "group-hate-conflict":
                location = issue["locations"][0]
                reason_code = "resolve-group-hate-conflict"
                edits = [
                    {
                        "location_index": 0,
                        "op": "set",
                        "json_pointer": f"/quadruples/{location['tuple_index']}/hateful",
                        "value": "non-hate",
                    }
                ]
            elif code == "non-string-quad-field":
                location = issue["locations"][0]
                field = issue["review_context"]["field"]
                observed = issue["review_context"]["tuple_before"][field]
                reason_code = "coerce-numeric-annotation"
                edits = [
                    {
                        "location_index": 0,
                        "op": "set",
                        "json_pointer": location["json_pointer"],
                        "value": str(observed),
                    }
                ]
            elif code == "duplicate-record-id":
                # Only used by the duplicate-ID audit test, which does not finalize.
                reason_code = "correct-source-record-id"
                edits = [
                    {
                        "location_index": 1,
                        "op": "set",
                        "json_pointer": "/id",
                        "value": "9000001",
                    }
                ]
            else:  # pragma: no cover - catches additions to the issue contract
                raise AssertionError(f"Unhandled test issue code: {code}")
            rows.append(
                {
                    "schema_version": "stage1-data-adjudication-row/v1",
                    "data_audit_id": issue["data_audit_id"],
                    "issue_id": issue["issue_id"],
                    "issue_kind": issue["issue_kind"],
                    "decision": decision,
                    "edits": edits,
                    "reason_code": reason_code,
                    "reason": f"fixture decision for {code}",
                    "reviewer_id": "dual-blind-panel-v1",
                    "reviewed_at": "2026-08-22T09:00:00+08:00",
                }
            )
        return rows

    def seal_blind_review(
        self,
        *,
        issue_kinds: list[str] | None = None,
    ) -> tuple[dict, Path]:
        env_file = self.root / "review/test.env"
        env_file.parent.mkdir(parents=True, exist_ok=True)
        env_file.write_text(
            "GLM_API_KEY=fixture-glm-key\nDEEPSEEK_API_KEY=fixture-deepseek-key\n",
            encoding="utf-8",
        )

        def successful_transport(_url, request_payload, _headers, _timeout):
            prompt = request_payload["messages"][-1]["content"]
            item = json.loads(prompt.split("\n", 1)[1])
            if item["accept_allowed"]:
                judgement = {
                    "decision": "accepted",
                    "edits": [],
                    "reason_code": "valid-independent-label-combination",
                    "reason": "fixture reviewers independently accept the labels",
                    "confidence": 0.95,
                }
            else:
                issue_code = item["issue_code"]
                if issue_code == "non-string-quad-field":
                    field = item["review_context"]["field"]
                    edit_path = item["allowed_edit_paths"][0]
                    edit_value = str(
                        item["review_context"]["tuple_before"][field]
                    )
                    reason_code = "coerce-numeric-annotation"
                else:
                    edit_path = next(
                        path
                        for path in item["allowed_edit_paths"]
                        if path.endswith("/hateful")
                    )
                    edit_value = (
                        "hate"
                        if issue_code == "hateful-null-sentinel"
                        else "non-hate"
                    )
                    reason_code = (
                        "replace-legacy-null"
                        if issue_code == "hateful-null-sentinel"
                        else "resolve-group-hate-conflict"
                    )
                judgement = {
                    "decision": "corrected",
                    "edits": [
                        {
                            "location_index": 0,
                            "op": "set",
                            "json_pointer": edit_path,
                            "value": edit_value,
                        }
                    ],
                    "reason_code": reason_code,
                    "reason": "fixture reviewers independently correct the label",
                    "confidence": 0.95,
                }
            return {
                "id": f"fixture-{request_payload['model']}-{item['blind_item_id']}",
                "model": request_payload["model"],
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": json.dumps(judgement)},
                    }
                ],
            }

        summary = run_data_review(
            audit_ref=self.audit_ref,
            policy_path=self.blind_policy_path,
            env_file=env_file,
            output_dir=self.root / "review/runs",
            issue_kinds=issue_kinds or ["group-hate"],
            transport=successful_transport,
        )
        source_target = self.root / "review/runs" / summary["review_run_id"]
        locator = seal_data_review(
            audit_ref=self.audit_ref,
            review_target=source_target,
            output_root=self.root / "artifacts/data_blind_reviews",
            write_ref=self.data_blind_review_ref,
            workspace_root=self.root,
        )
        return locator, Path(locator["target_path"])

    def sign_and_finalize(self, audit_target: Path) -> tuple[dict, dict]:
        _, blind_target = self.seal_blind_review()
        rows = self.completed_rows(audit_target)
        auto_rows = {
            row["issue_id"]: row
            for row in read_jsonl(blind_target / "auto_adjudication.jsonl")
        }
        rows = [auto_rows.get(row["issue_id"], row) for row in rows]
        write_jsonl(self.adjudication_file, rows)
        declaration = prepare_data_declaration(
            audit_ref=self.audit_ref,
            adjudication_file=self.adjudication_file,
            reviewer_id="dual-blind-panel-v1",
            write_template=self.declaration_file,
        )
        declaration["attestation_confirmed"] = True
        write_json(self.declaration_file, declaration)
        data_ref = finalize_data(
            config_path=self.config_path,
            audit_ref=self.audit_ref,
            data_blind_review_ref=self.data_blind_review_ref,
            adjudication_file=self.adjudication_file,
            reviewer_declaration=self.declaration_file,
            write_ref=self.data_ref,
        )
        report = validate_data(data_ref=self.data_ref)
        return data_ref, report


class Stage1DataLifecycleTest(unittest.TestCase):
    def test_real_sources_audit_to_34_blocking_and_99_warning_locations(self):
        with tempfile.TemporaryDirectory() as temporary:
            workspace = TemporaryStage1Workspace(Path(temporary))
            ref, target = workspace.audit()

            self.assertEqual(ref["artifact_kind"], "data-audit")
            meta = json.loads((target / "audit.meta.json").read_text(encoding="utf-8"))
            report = json.loads((target / "audit_report.json").read_text(encoding="utf-8"))
            issues = read_jsonl(target / "issues.jsonl")

            self.assertEqual(meta["blocking_issue_count"], 34)
            self.assertEqual(meta["blocking_issue_counts_by_kind"], {"field-type": 4, "group-hate": 30})
            self.assertEqual(meta["blocking_issue_counts_by_split"], {"dev": 8, "test": 7, "train": 19})
            self.assertEqual(report["warnings"]["count"], 99)
            self.assertEqual(report["warnings"]["by_field"], {"argument": 66, "target": 33})
            self.assertEqual(
                sum(issue["issue_code"] == "group-hate-atypical" for issue in issues), 27
            )
            self.assertEqual(
                sum(issue["issue_code"] == "hateful-null-sentinel" for issue in issues), 2
            )
            self.assertEqual(
                sum(issue["issue_code"] == "group-hate-conflict" for issue in issues), 1
            )
            self.assertEqual(
                sum(issue["issue_code"] == "non-string-quad-field" for issue in issues), 4
            )

            split = json.loads((target / "split_manifest.proposed.json").read_text(encoding="utf-8"))
            self.assertTrue(all(isinstance(item, str) for item in split["train_ids"]))
            self.assertEqual(len(split["train_ids"]), 5781)
            self.assertEqual(len(split["dev_ids"]), 643)
            self.assertEqual(len(split["test_ids"]), 1605)
            self.assertEqual(
                split["train_ids_sha256"],
                "916e1dd8dec69b208066e600418d87e8e7da09438d331e28d731d9306764e7d6",
            )
            self.assertEqual(
                split["dev_ids_sha256"],
                "deaeeabe9b190973158e82d9d60cbd4078a6373d6a65d4108481bfeb05e8f253",
            )
            self.assertEqual(
                split["test_ids_sha256"],
                "04d63800eef192f6fec911137c4e96cee84e2beb12389e68fdf4715adf14dcc2",
            )

    def test_full_signed_lifecycle_normalizes_ids_nulls_groups_and_numeric_fields(self):
        with tempfile.TemporaryDirectory() as temporary:
            workspace = TemporaryStage1Workspace(Path(temporary))
            _, audit_target = workspace.audit()
            data_ref, validation = workspace.sign_and_finalize(audit_target)
            data_target = Path(data_ref["target_path"])

            self.assertTrue(validation["valid"])
            self.assertTrue(validation["upstream_adjudication_replayed"])
            self.assertTrue(validation["source_adjudication_replayed"])
            self.assertEqual(validation["split_counts"], {"train": 5781, "dev": 643, "test": 1605})
            queue_rows = read_jsonl(
                data_target / "human_adjudication_queue.jsonl"
            )
            self.assertEqual(
                sum(row["queue_source"] == "audit-human-only" for row in queue_rows),
                4,
            )
            self.assertTrue(
                read_jsonl(
                    Path(
                        json.loads(
                            workspace.data_blind_review_ref.read_text(encoding="utf-8")
                        )["target_path"]
                    )
                    / "auto_adjudication.jsonl"
                )
            )
            provenance = json.loads(
                (data_target / "provenance.json").read_text(encoding="utf-8")
            )
            for field in (
                "train_content_sha256",
                "dev_content_sha256",
                "test_content_sha256",
                "adjudication_log_sha256",
                "human_adjudication_queue_sha256",
                "adjudication_frame_sha256",
            ):
                self.assertRegex(provenance["data_id_inputs"][field], r"^[0-9a-f]{64}$")
            train = json.loads((data_target / "train.json").read_text(encoding="utf-8"))
            dev = json.loads((data_target / "dev.json").read_text(encoding="utf-8"))
            test = json.loads((data_target / "test.json").read_text(encoding="utf-8"))
            by_id = {record["id"]: record for record in train + dev + test}

            self.assertTrue(all(isinstance(record["id"], str) for record in train + dev + test))
            self.assertIsNone(by_id["2522"]["quadruples"][0]["target"])
            self.assertEqual(by_id["5302"]["quadruples"][0]["argument"], "414")
            self.assertEqual(by_id["7553"]["quadruples"][0]["argument"], "414")
            self.assertEqual(by_id["6585"]["quadruples"][0]["target"], "13")
            self.assertEqual(by_id["5804"]["quadruples"][0]["target"], "1")
            self.assertTrue(
                all(
                    quad["hateful"] in {"hate", "non-hate"}
                    and isinstance(quad["targeted_group"], list)
                    for record in train + dev + test
                    for quad in record["quadruples"]
                )
            )

            second_ref_path = workspace.root / "refs/data_ref_second.json"
            second_ref = finalize_data(
                config_path=workspace.config_path,
                audit_ref=workspace.audit_ref,
                data_blind_review_ref=workspace.data_blind_review_ref,
                adjudication_file=workspace.adjudication_file,
                reviewer_declaration=workspace.declaration_file,
                write_ref=second_ref_path,
            )
            self.assertEqual(second_ref["artifact_id"], data_ref["artifact_id"])
            self.assertEqual(
                second_ref["payload_manifest_sha256"], data_ref["payload_manifest_sha256"]
            )

    def test_finalize_rejects_a_blind_review_with_the_wrong_issue_scope(self):
        with tempfile.TemporaryDirectory() as temporary:
            workspace = TemporaryStage1Workspace(Path(temporary))
            _, audit_target = workspace.audit()
            workspace.seal_blind_review(issue_kinds=["field-type"])
            rows = workspace.completed_rows(audit_target)
            write_jsonl(workspace.adjudication_file, rows)
            declaration = prepare_data_declaration(
                audit_ref=workspace.audit_ref,
                adjudication_file=workspace.adjudication_file,
                reviewer_id="dual-blind-panel-v1",
                write_template=workspace.declaration_file,
            )
            declaration["attestation_confirmed"] = True
            write_json(workspace.declaration_file, declaration)
            with self.assertRaisesRegex(
                Stage1DataError, "complete group-hate audit scope"
            ):
                finalize_data(
                    config_path=workspace.config_path,
                    audit_ref=workspace.audit_ref,
                    data_blind_review_ref=workspace.data_blind_review_ref,
                    adjudication_file=workspace.adjudication_file,
                    reviewer_declaration=workspace.declaration_file,
                    write_ref=workspace.data_ref,
                )

    def test_retry_blind_review_derives_scope_from_complete_frozen_frame(self):
        with tempfile.TemporaryDirectory() as temporary:
            workspace = TemporaryStage1Workspace(Path(temporary))
            _, audit_target = workspace.audit()
            _, blind_target = workspace.seal_blind_review()

            run_meta = json.loads(
                (blind_target / "run.meta.json").read_text(encoding="utf-8")
            )
            run_meta.pop("issue_scope")
            run_meta.update(
                {
                    "parent_review_run_id": "dblind-" + "0" * 64,
                    "retry_policy": "failed-reviewer-items-only/v1",
                    "retry_issue_ids_sha256_by_reviewer": {},
                }
            )
            write_json(blind_target / "run.meta.json", run_meta)

            auto_rows = {
                row["issue_id"]: row
                for row in read_jsonl(blind_target / "auto_adjudication.jsonl")
            }
            completed_rows = [
                auto_rows.get(row["issue_id"], row)
                for row in workspace.completed_rows(audit_target)
            ]
            audit_dependency = json.loads(
                (blind_target / "data_audit_ref.json").read_text(encoding="utf-8")
            )
            issues = read_jsonl(audit_target / "issues.jsonl")
            rubric_meta = json.loads(
                (audit_target / "adjudication_rubric.meta.json").read_text(
                    encoding="utf-8"
                )
            )

            queue_rows, frame = _adjudication_composition(
                audit_dependency=audit_dependency,
                blind_review_dependency={"artifact_kind": "data-blind-review"},
                blind_review_target=blind_target,
                issues=issues,
                rubric_meta=rubric_meta,
                completed_rows=completed_rows,
            )

            expected_queue_count = len(
                read_jsonl(blind_target / "human_queue.jsonl")
            ) + sum(issue["issue_kind"] == "field-type" for issue in issues)
            self.assertEqual(len(queue_rows), expected_queue_count)
            self.assertEqual(
                frame["blind_scope"]["issue_kinds"], ["group-hate"]
            )

    def test_finalize_rejects_a_changed_sealed_auto_row(self):
        with tempfile.TemporaryDirectory() as temporary:
            workspace = TemporaryStage1Workspace(Path(temporary))
            _, audit_target = workspace.audit()
            _, blind_target = workspace.seal_blind_review()
            auto_rows = read_jsonl(blind_target / "auto_adjudication.jsonl")
            self.assertTrue(auto_rows)
            auto_map = {row["issue_id"]: row for row in auto_rows}
            rows = [
                auto_map.get(row["issue_id"], row)
                for row in workspace.completed_rows(audit_target)
            ]
            changed_id = auto_rows[0]["issue_id"]
            rows = [dict(row) for row in rows]
            next(row for row in rows if row["issue_id"] == changed_id)[
                "reason"
            ] += " changed after sealing"
            write_jsonl(workspace.adjudication_file, rows)
            declaration = prepare_data_declaration(
                audit_ref=workspace.audit_ref,
                adjudication_file=workspace.adjudication_file,
                reviewer_id="dual-blind-panel-v1",
                write_template=workspace.declaration_file,
            )
            declaration["attestation_confirmed"] = True
            write_json(workspace.declaration_file, declaration)
            with self.assertRaisesRegex(Stage1DataError, "changed or omitted blind auto"):
                finalize_data(
                    config_path=workspace.config_path,
                    audit_ref=workspace.audit_ref,
                    data_blind_review_ref=workspace.data_blind_review_ref,
                    adjudication_file=workspace.adjudication_file,
                    reviewer_declaration=workspace.declaration_file,
                    write_ref=workspace.data_ref,
                )

    def test_validator_rejects_same_id_with_changed_split_payload(self):
        with tempfile.TemporaryDirectory() as temporary:
            workspace = TemporaryStage1Workspace(Path(temporary))
            _, audit_target = workspace.audit()
            data_ref, _ = workspace.sign_and_finalize(audit_target)
            target = Path(data_ref["target_path"])
            train = json.loads((target / "train.json").read_text(encoding="utf-8"))
            train[0]["content"] += " tampered"
            (target / "train.json").write_text(
                json.dumps(
                    train,
                    ensure_ascii=False,
                    sort_keys=False,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                + "\n",
                encoding="utf-8",
            )
            write_canonical_json(
                target / "payload_manifest.json", build_payload_manifest(target)
            )
            with self.assertRaisesRegex(
                Stage1DataError, "train.json content"
            ):
                validate_data_target(target)

    def test_export_and_prepare_require_the_exact_completed_issue_set(self):
        with tempfile.TemporaryDirectory() as temporary:
            workspace = TemporaryStage1Workspace(Path(temporary))
            _, audit_target = workspace.audit()
            exported = workspace.root / "review/exported.jsonl"
            export_data_adjudication(audit_ref=workspace.audit_ref, output_path=exported)
            self.assertEqual(
                exported.read_bytes(), (audit_target / "adjudication_template.jsonl").read_bytes()
            )

            subset = workspace.root / "review/field_type.jsonl"
            packets = workspace.root / "review/field_type_packets.jsonl"
            report = export_data_review_subset(
                audit_ref=workspace.audit_ref,
                issue_kinds=["field-type"],
                reviewer_id="reviewer-01",
                output_path=subset,
                packet_output_path=packets,
            )
            self.assertEqual(report["row_count"], 4)
            self.assertTrue(all(row["reviewer_id"] == "reviewer-01" for row in read_jsonl(subset)))
            self.assertTrue(all(row["model_votes_hidden"] for row in read_jsonl(packets)))

            completed = workspace.completed_rows(audit_target)
            write_jsonl(workspace.adjudication_file, completed[:-1])
            with self.assertRaisesRegex(Stage1DataError, "issue set mismatch"):
                prepare_data_declaration(
                    audit_ref=workspace.audit_ref,
                    adjudication_file=workspace.adjudication_file,
                    reviewer_id="reviewer-01",
                    write_template=workspace.declaration_file,
                )

            completed = workspace.completed_rows(audit_target)
            group_file = workspace.root / "review/group.jsonl"
            field_file = workspace.root / "review/field.jsonl"
            merged_file = workspace.root / "review/merged.jsonl"
            write_jsonl(group_file, [row for row in completed if row["issue_kind"] == "group-hate"])
            write_jsonl(field_file, [row for row in completed if row["issue_kind"] == "field-type"])
            merged = merge_completed_adjudications(
                audit_ref=workspace.audit_ref,
                adjudication_files=[group_file, field_file],
                reviewer_id="dual-blind-panel-v1",
                output_path=merged_file,
            )
            self.assertEqual(merged["row_count"], 34)
            self.assertEqual(
                {row["issue_id"] for row in read_jsonl(merged_file)},
                {row["issue_id"] for row in completed},
            )

    def test_duplicate_ids_form_one_stable_multi_location_blocking_issue(self):
        with tempfile.TemporaryDirectory() as temporary:
            workspace = TemporaryStage1Workspace(
                Path(temporary), enforce_expected_counts=False
            )
            train_path = workspace.root / "data/full/std/train.json"
            test_path = workspace.root / "data/full/std/test.json"
            train = json.loads(train_path.read_text(encoding="utf-8"))
            test = json.loads(test_path.read_text(encoding="utf-8"))
            test[0]["id"] = train[0]["id"]
            write_json(test_path, test)

            _, target = workspace.audit()
            duplicate_issues = [
                issue
                for issue in read_jsonl(target / "issues.jsonl")
                if issue["issue_code"] == "duplicate-record-id"
            ]
            self.assertEqual(len(duplicate_issues), 1)
            issue = duplicate_issues[0]
            self.assertEqual(len(issue["locations"]), 2)
            self.assertEqual(
                {(location["source_key"], location["source_ordinal"]) for location in issue["locations"]},
                {("std-train", 0), ("std-test", 0)},
            )
            self.assertRegex(issue["issue_id"], r"^dissue:v1:[0-9a-f]{64}$")

    def test_config_cannot_redirect_stage1_to_legacy_raw_sources(self):
        with tempfile.TemporaryDirectory() as temporary:
            workspace = TemporaryStage1Workspace(Path(temporary))
            config = json.loads(workspace.config_path.read_text(encoding="utf-8"))
            config["data"]["sources"]["train"] = "data/full/raw/train.json"
            write_json(workspace.config_path, config)
            with self.assertRaisesRegex(Stage1DataError, "source-of-truth"):
                workspace.audit()


if __name__ == "__main__":
    unittest.main()
