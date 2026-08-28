import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from data.training_artifacts import (
    build_payload_manifest,
    sha256_file,
    write_canonical_json,
    write_canonical_jsonl,
    write_locator_ref,
)
from review.human_adjudication import (
    HumanAdjudicationError,
    HumanAdjudicationIncomplete,
    load_human_review_workspace,
    render_issue_for_human,
    review_interactively,
    validate_human_review_workspace,
    workspace_status,
)


def group_issue(issue_id: str = "dissue:v1:group") -> dict:
    return {
        "schema_version": "stage1-data-audit-issue/v1",
        "data_audit_id": "daudit-test",
        "issue_id": issue_id,
        "issue_kind": "group-hate",
        "issue_code": "group-hate-atypical",
        "issue_subtype": "specific-group-with-non-hate-label",
        "issue_rule_version": "stage1-data-issues/v1",
        "severity": "blocking",
        "locations": [
            {
                "source_key": "std-train",
                "source_file_sha256": "a" * 64,
                "source_ordinal": 3,
                "split": "train",
                "source_record_id": "4",
                "tuple_index": 0,
                "json_pointer": "/quadruples/0/hateful",
                "observed_value_sha256": "b" * 64,
            }
        ],
        "accept_allowed": True,
        "allowed_edit_paths": [
            "/quadruples/0/targeted_group",
            "/quadruples/0/hateful",
        ],
        "review_context": {
            "content": "synthetic content",
            "tuple_before": {
                "target": "x",
                "argument": "y",
                "targeted_group": "Racism",
                "hateful": "non-hate",
            },
            "expected_hateful_from_legacy_coupling": "LEAK_HINT",
        },
    }


def field_issue(issue_id: str = "dissue:v1:field") -> dict:
    return {
        "schema_version": "stage1-data-audit-issue/v1",
        "data_audit_id": "daudit-test",
        "issue_id": issue_id,
        "issue_kind": "field-type",
        "issue_code": "non-string-quad-field",
        "issue_subtype": "numeric-target",
        "issue_rule_version": "stage1-data-issues/v1",
        "severity": "blocking",
        "locations": [
            {
                "source_key": "std-train",
                "source_file_sha256": "a" * 64,
                "source_ordinal": 4,
                "split": "train",
                "source_record_id": "5",
                "tuple_index": 0,
                "json_pointer": "/quadruples/0/target",
                "observed_value_sha256": "c" * 64,
            }
        ],
        "accept_allowed": False,
        "allowed_edit_paths": ["/quadruples/0/target"],
        "review_context": {
            "content": "synthetic field content",
            "tuple_before": {
                "target": 414,
                "argument": "y",
                "targeted_group": "non-hate",
                "hateful": "non-hate",
            },
            "field": "target",
            "observed_type": "int",
            "allowed_correction_types": ["string", "null"],
        },
    }


def template(issue: dict) -> dict:
    return {
        "schema_version": "stage1-data-adjudication-row/v1",
        "data_audit_id": issue["data_audit_id"],
        "issue_id": issue["issue_id"],
        "issue_kind": issue["issue_kind"],
        "decision": "",
        "edits": [],
        "reason_code": "",
        "reason": "",
        "reviewer_id": "dual-blind-panel-v1",
        "reviewed_at": "",
    }


def rubric_meta() -> dict:
    return {
        "schema_version": "stage1-data-adjudication-rubric-meta/v1",
        "reason_codes": {
            "group-hate-atypical": {
                "accepted": ["valid-independent-label-combination"],
                "corrected": [
                    "correct-source-label",
                    "resolve-group-hate-conflict",
                ],
            },
            "non-string-quad-field": {
                "accepted": [],
                "corrected": [
                    "coerce-numeric-annotation",
                    "set-explicit-null",
                ],
            },
        },
    }


class Fixture:
    def __init__(
        self,
        root: Path,
        issue: dict,
        *,
        queue: bool = False,
        extra_audit_issues: list[dict] | None = None,
    ):
        self.root = root
        self.issue = issue
        self.audit_target = root / "artifacts" / "daudit-test"
        self.audit_target.mkdir(parents=True)
        write_canonical_jsonl(
            self.audit_target / "issues.jsonl",
            [issue, *(extra_audit_issues or [])],
            key="issue_id",
        )
        write_canonical_json(
            self.audit_target / "adjudication_rubric.meta.json", rubric_meta()
        )
        write_canonical_json(
            self.audit_target / "payload_manifest.json",
            build_payload_manifest(self.audit_target),
        )
        self.audit_ref = root / "data_audit_ref.json"
        write_locator_ref(
            self.audit_ref,
            artifact_kind="data-audit",
            artifact_id="daudit-test",
            target=self.audit_target,
            payload_manifest_sha256=sha256_file(
                self.audit_target / "payload_manifest.json"
            ),
        )
        self.row = template(issue)
        self.adjudication = root / "human_adjudication.jsonl"
        write_canonical_jsonl(self.adjudication, [self.row], key="issue_id")
        self.review_target = None
        if queue:
            self.packet = {
                "schema_version": "stage1-human-review-queue/v1",
                "issue_id": issue["issue_id"],
                "issue_code": issue["issue_code"],
                "issue_kind": issue["issue_kind"],
                "accept_allowed": issue["accept_allowed"],
                "allowed_edit_paths": issue["allowed_edit_paths"],
                "review_context": issue["review_context"],
                "queue_reasons": ["model_disagreement"],
                "model_votes_hidden": True,
            }
            self.review_target = root / "dblind-test"
            self.review_target.mkdir()
            write_canonical_json(
                self.review_target / "run.meta.json",
                {
                    "review_run_id": "dblind-test",
                    "audit_artifact_id": "daudit-test",
                    "audit_payload_manifest_sha256": sha256_file(
                        self.audit_target / "payload_manifest.json"
                    ),
                },
            )
            write_canonical_jsonl(
                self.review_target / "human_queue.jsonl",
                [self.packet],
                key="issue_id",
            )
            write_canonical_jsonl(
                self.review_target / "human_adjudication_template.jsonl",
                [self.row],
                key="issue_id",
            )
            write_canonical_jsonl(
                self.review_target / "auto_adjudication.jsonl", [], key="issue_id"
            )
            write_canonical_jsonl(
                self.review_target / "consensus.jsonl",
                [{"issue_id": issue["issue_id"]}],
                key="issue_id",
            )
        else:
            self.packet = {
                "schema_version": "stage1-local-human-review-packet/v1",
                "issue_id": issue["issue_id"],
                "issue_code": issue["issue_code"],
                "issue_kind": issue["issue_kind"],
                "accept_allowed": issue["accept_allowed"],
                "allowed_edit_paths": issue["allowed_edit_paths"],
                "locations": issue["locations"],
                "review_context": issue["review_context"],
                "model_votes_hidden": True,
                "saw_condition_outputs": False,
                "saw_model_scores": False,
            }
        self.packet_file = root / "human_packets.jsonl"
        write_canonical_jsonl(self.packet_file, [self.packet], key="issue_id")

    def load(self):
        return load_human_review_workspace(
            audit_ref=self.audit_ref,
            packet_file=self.packet_file,
            adjudication_file=self.adjudication,
            review_target=self.review_target,
        )

    def write_row(self, row: dict) -> None:
        write_canonical_jsonl(self.adjudication, [row], key="issue_id")


def scripted(values):
    answers = iter(values)
    return lambda _prompt: next(answers)


class WorkspaceValidationTests(unittest.TestCase):
    def test_pristine_status_and_strict_incomplete(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = Fixture(Path(directory), field_issue())
            workspace = fixture.load()
            report = workspace_status(workspace)
            self.assertEqual(report["complete_count"], 0)
            self.assertEqual(report["incomplete_count"], 1)
            with self.assertRaises(HumanAdjudicationIncomplete):
                validate_human_review_workspace(workspace)

    def test_partial_row_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = Fixture(Path(directory), field_issue())
            row = dict(fixture.row)
            row["reason"] = "half written"
            fixture.write_row(row)
            with self.assertRaisesRegex(HumanAdjudicationError, "partially edited"):
                fixture.load()

    def test_immutable_field_change_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = Fixture(Path(directory), field_issue())
            row = dict(fixture.row)
            row["reviewer_id"] = "another-reviewer"
            fixture.write_row(row)
            with self.assertRaisesRegex(HumanAdjudicationError, "reviewer_id"):
                fixture.load()

    def test_packet_tamper_and_unhidden_votes_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = Fixture(Path(directory), field_issue())
            packet = dict(fixture.packet)
            packet["model_votes_hidden"] = False
            write_canonical_jsonl(fixture.packet_file, [packet], key="issue_id")
            with self.assertRaisesRegex(HumanAdjudicationError, "hide model votes"):
                fixture.load()

    def test_queue_requires_and_matches_frozen_review_target(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = Fixture(Path(directory), group_issue(), queue=True)
            with self.assertRaisesRegex(HumanAdjudicationError, "--review-target"):
                load_human_review_workspace(
                    audit_ref=fixture.audit_ref,
                    packet_file=fixture.packet_file,
                    adjudication_file=fixture.adjudication,
                )
            packet = dict(fixture.packet)
            packet["queue_reasons"] = ["low_confidence"]
            write_canonical_jsonl(fixture.packet_file, [packet], key="issue_id")
            with self.assertRaisesRegex(HumanAdjudicationError, "frozen human queue"):
                fixture.load()

    def test_symlink_adjudication_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = Fixture(Path(directory), field_issue())
            link = Path(directory) / "link.jsonl"
            link.symlink_to(fixture.adjudication)
            with self.assertRaisesRegex(HumanAdjudicationError, "non-symlink"):
                load_human_review_workspace(
                    audit_ref=fixture.audit_ref,
                    packet_file=fixture.packet_file,
                    adjudication_file=link,
                )

    def test_local_packet_cannot_omit_an_issue_from_its_audit_kind(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = Fixture(
                Path(directory),
                field_issue(),
                extra_audit_issues=[field_issue("dissue:v1:field-omitted")],
            )
            with self.assertRaisesRegex(HumanAdjudicationError, "complete frozen"):
                fixture.load()


class InteractiveReviewTests(unittest.TestCase):
    def test_atomic_commit_opens_lock_descriptor_read_write(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = Fixture(Path(directory), group_issue(), queue=True)
            workspace = fixture.load()
            destination_open_flags = []
            real_open = os.open

            def recording_open(path, flags, *args, **kwargs):
                if Path(path) == fixture.adjudication:
                    destination_open_flags.append(flags)
                return real_open(path, flags, *args, **kwargs)

            with mock.patch(
                "review.human_adjudication.os.open", side_effect=recording_open
            ):
                report = review_interactively(
                    workspace,
                    input_fn=scripted(
                        ["a", "semantic annotation is supported", "y"]
                    ),
                    output_fn=lambda _line: None,
                )

            self.assertTrue(report["all_complete"])
            self.assertEqual(len(destination_open_flags), 1)
            self.assertEqual(destination_open_flags[0] & os.O_ACCMODE, os.O_RDWR)

    def test_accept_commits_atomically_and_hides_queue_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = Fixture(Path(directory), group_issue(), queue=True)
            before_mode = fixture.adjudication.stat().st_mode & 0o777
            output = []
            report = review_interactively(
                fixture.load(),
                input_fn=scripted(["a", "semantic annotation is supported", "y"]),
                output_fn=output.append,
            )
            self.assertTrue(report["all_complete"])
            completed = json.loads(fixture.adjudication.read_text(encoding="utf-8"))
            self.assertEqual(completed["decision"], "accepted")
            self.assertRegex(
                completed["reviewed_at"],
                r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}\+00:00$",
            )
            transcript = "\n".join(output)
            self.assertNotIn("LEAK_HINT", transcript)
            self.assertNotIn("model_disagreement", transcript)
            self.assertNotIn(fixture.issue["issue_id"], transcript)
            self.assertEqual(fixture.adjudication.stat().st_mode & 0o777, before_mode)

    def test_string_and_null_field_corrections(self):
        cases = [
            (["c", '"414"', "1", "numeric token is literal text", "y"], "414"),
            (["c", "null", "2", "annotation is absent", "y"], None),
        ]
        for answers, expected in cases:
            with self.subTest(expected=expected), tempfile.TemporaryDirectory() as directory:
                fixture = Fixture(Path(directory), field_issue())
                report = review_interactively(
                    fixture.load(), input_fn=scripted(answers), output_fn=lambda _line: None
                )
                self.assertTrue(report["all_complete"])
                completed = json.loads(fixture.adjudication.read_text(encoding="utf-8"))
                self.assertEqual(completed["edits"][0]["value"], expected)

    def test_skip_quit_and_unconfirmed_candidate_do_not_write(self):
        scenarios = [
            ["q"],
            ["s"],
            ["c", '"414"', "1", "not committed", "n", "q"],
        ]
        for answers in scenarios:
            with self.subTest(answers=answers), tempfile.TemporaryDirectory() as directory:
                fixture = Fixture(Path(directory), field_issue())
                before = fixture.adjudication.read_bytes()
                review_interactively(
                    fixture.load(), input_fn=scripted(answers), output_fn=lambda _line: None
                )
                self.assertEqual(fixture.adjudication.read_bytes(), before)

    def test_external_change_trips_compare_and_swap(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = Fixture(Path(directory), field_issue())
            workspace = fixture.load()
            with fixture.adjudication.open("ab") as handle:
                handle.write(b"\n")
            with self.assertRaisesRegex(HumanAdjudicationError, "changed concurrently"):
                review_interactively(
                    workspace,
                    input_fn=scripted(
                        ["c", '"414"', "1", "valid but raced", "y"]
                    ),
                    output_fn=lambda _line: None,
                )

    def test_invalid_correction_does_not_leak_real_issue_id(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = Fixture(Path(directory), group_issue(), queue=True)
            before = fixture.adjudication.read_bytes()
            output = []
            review_interactively(
                fixture.load(),
                input_fn=scripted(["c", "1", "1", "1", "no-op", "q"]),
                output_fn=output.append,
            )
            transcript = "\n".join(output)
            self.assertNotIn(fixture.issue["issue_id"], transcript)
            self.assertIn("未通过冻结规则", transcript)
            self.assertEqual(fixture.adjudication.read_bytes(), before)


class SemanticAndDisplayTests(unittest.TestCase):
    def test_invalid_field_value_reason_pair_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = Fixture(Path(directory), field_issue())
            row = dict(fixture.row)
            row.update(
                {
                    "decision": "corrected",
                    "edits": [
                        {
                            "location_index": 0,
                            "op": "set",
                            "json_pointer": "/quadruples/0/target",
                            "value": "414",
                        }
                    ],
                    "reason_code": "set-explicit-null",
                    "reason": "mismatched code",
                    "reviewed_at": "2026-08-22T00:00:00+00:00",
                }
            )
            fixture.write_row(row)
            with self.assertRaisesRegex(HumanAdjudicationError, "set-explicit-null"):
                fixture.load()

    def test_numeric_and_noncanonical_string_are_rejected(self):
        for value in (414, "", " padded ", "NULL"):
            with self.subTest(value=value), tempfile.TemporaryDirectory() as directory:
                fixture = Fixture(Path(directory), field_issue())
                row = dict(fixture.row)
                row.update(
                    {
                        "decision": "corrected",
                        "edits": [
                            {
                                "location_index": 0,
                                "op": "set",
                                "json_pointer": "/quadruples/0/target",
                                "value": value,
                            }
                        ],
                        "reason_code": "coerce-numeric-annotation",
                        "reason": "invalid fixture",
                        "reviewed_at": "2026-08-22T00:00:00+00:00",
                    }
                )
                fixture.write_row(row)
                with self.assertRaises(HumanAdjudicationError):
                    fixture.load()

    def test_group_correction_must_resolve_issue_and_not_be_noop(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = Fixture(Path(directory), group_issue(), queue=True)
            row = dict(fixture.row)
            row.update(
                {
                    "decision": "corrected",
                    "edits": [
                        {
                            "location_index": 0,
                            "op": "set",
                            "json_pointer": "/quadruples/0/targeted_group",
                            "value": "Sexism",
                        }
                    ],
                    "reason_code": "correct-source-label",
                    "reason": "still inconsistent",
                    "reviewed_at": "2026-08-22T00:00:00+00:00",
                }
            )
            fixture.write_row(row)
            with self.assertRaisesRegex(HumanAdjudicationError, "leaves a group"):
                fixture.load()

    def test_terminal_controls_are_escaped_and_hidden_fields_omitted(self):
        issue = group_issue()
        issue["review_context"]["content"] = "safe\x1b[31m\u202eevil"
        rendered = render_issue_for_human(
            issue, ordinal=1, total=1, blind_alias="H-deadbeef0000"
        )
        self.assertNotIn("\x1b", rendered)
        self.assertNotIn("\u202e", rendered)
        self.assertIn("\\u001b", rendered)
        self.assertIn("\\u202e", rendered)
        self.assertNotIn("LEAK_HINT", rendered)


if __name__ == "__main__":
    unittest.main()
