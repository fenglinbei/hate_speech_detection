import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from data.training_artifacts import (
    build_payload_manifest,
    canonical_sha256,
    sha256_file,
    write_canonical_json,
    write_canonical_jsonl,
    write_locator_ref,
)
from review.blind_review import (
    SYSTEM_PROMPT,
    _chat_url,
    _item_prompt,
    build_consensus,
    export_human_adjudication_workspace,
    merge_human_adjudication,
)
from review.d14_contract import D14_REQUEST_OPTIONS
from review.data_review_artifact import (
    DataReviewArtifactError,
    resolve_data_review_ref,
    seal_data_review,
    validate_data_review_ref,
)
from review.human_adjudication import load_human_review_workspace, workspace_status


REVIEWERS = [
    {
        "reviewer_id": "glm_high_parameter",
        "api_base": "https://open.bigmodel.cn/api/paas/v4",
        "model": "glm-5.3",
    },
    {
        "reviewer_id": "deepseek_flash",
        "api_base": "https://api.deepseek.com",
        "model": "deepseek-v4-flash",
    },
]


def _issue(issue_id: str) -> dict:
    return {
        "schema_version": "stage1-data-audit-issue/v1",
        "data_audit_id": "daudit-" + "a" * 64,
        "issue_id": issue_id,
        "issue_kind": "group-hate",
        "issue_code": "group-hate-atypical",
        "issue_subtype": "synthetic",
        "accept_allowed": True,
        "allowed_edit_paths": ["/quadruples/0/hateful"],
        "locations": [
            {
                "source_key": "train",
                "source_ordinal": 0,
                "json_pointer": "/quadruples/0/hateful",
            }
        ],
        "review_context": {"content": "synthetic", "tuple_before": {}},
    }


def _judgement(*, corrected: bool = False) -> dict:
    return {
        "decision": "corrected" if corrected else "accepted",
        "edits": (
            [
                {
                    "location_index": 0,
                    "op": "set",
                    "json_pointer": "/quadruples/0/hateful",
                    "value": "hate",
                }
            ]
            if corrected
            else []
        ),
        "reason_code": "correct-source-label" if corrected else "valid-label",
        "reason": "synthetic review",
        "confidence": 0.95,
    }


def _blind_map(issues: list[dict], seed: int) -> dict[str, str]:
    ordered = sorted(
        issues,
        key=lambda row: hashlib.sha256(f"{seed}:{row['issue_id']}".encode()).hexdigest(),
    )
    return {row["issue_id"]: f"B{index:04d}" for index, row in enumerate(ordered, 1)}


def _review_row(
    issue: dict,
    reviewer: dict,
    blind_item_id: str,
    *,
    corrected: bool = False,
) -> dict:
    judgement = _judgement(corrected=corrected)
    rubric_meta = {
        "reason_codes": {
            "group-hate-atypical": {
                "accepted": ["valid-label"],
                "corrected": ["correct-source-label"],
            }
        }
    }
    prompt = _item_prompt(blind_item_id, issue, rubric_meta)
    prompt_hash = hashlib.sha256(
        (SYSTEM_PROMPT + "\n" + prompt).encode("utf-8")
    ).hexdigest()
    payload = {
        "model": reviewer["model"],
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0,
        "top_p": 1,
        "stream": False,
        "max_tokens": 4096,
        "response_format": {"type": "json_object"},
        **D14_REQUEST_OPTIONS[reviewer["reviewer_id"]],
    }
    raw = {
        "id": f"req-{reviewer['reviewer_id']}-{issue['issue_id']}",
        "model": reviewer["model"],
        "choices": [
            {
                "finish_reason": "stop",
                "message": {"content": json.dumps(judgement, ensure_ascii=False)},
            }
        ],
    }
    return {
        "schema_version": "stage1-model-review/v1",
        "issue_id": issue["issue_id"],
        "blind_item_id": blind_item_id,
        "reviewer_id": reviewer["reviewer_id"],
        "provider_api_base": reviewer["api_base"],
        "requested_model": reviewer["model"],
        "returned_model": reviewer["model"],
        "provider_request_id": raw["id"],
        "prompt_sha256": prompt_hash,
        "request_without_secrets": {
            "url": _chat_url(reviewer["api_base"]),
            "payload": payload,
            "prompt_sha256": prompt_hash,
        },
        "raw_response": raw,
        "judgement": judgement,
        "normalizations": [],
        "error": None,
    }


class Fixture:
    def __init__(self, root: Path):
        self.root = root
        self.audit_id = "daudit-" + "a" * 64
        self.issue_ids = ["dissue:v1:" + "1" * 64, "dissue:v1:" + "2" * 64]
        self.issues = [_issue(value) for value in self.issue_ids]
        self.audit_target = root / "audits" / self.audit_id
        self.audit_target.mkdir(parents=True)
        write_canonical_jsonl(
            self.audit_target / "issues.jsonl", self.issues, key="issue_id"
        )
        self.rubric = {
            "schema_version": "stage1-data-adjudication-rubric-meta/v1",
            "reason_codes": {
                "group-hate-atypical": {
                    "accepted": ["valid-label"],
                    "corrected": ["correct-source-label"],
                }
            },
        }
        write_canonical_json(
            self.audit_target / "adjudication_rubric.meta.json", self.rubric
        )
        write_canonical_json(
            self.audit_target / "payload_manifest.json",
            build_payload_manifest(self.audit_target),
        )
        self.audit_ref = root / "audit_ref.json"
        write_locator_ref(
            self.audit_ref,
            artifact_kind="data-audit",
            artifact_id=self.audit_id,
            target=self.audit_target,
            payload_manifest_sha256=sha256_file(
                self.audit_target / "payload_manifest.json"
            ),
        )
        self.policy = {
            "schema_version": "stage1-blind-review-policy/v1",
            "temperature": 0,
            "top_p": 1,
            "max_attempts": 1,
            "max_tokens": 4096,
            "request_timeout_seconds": 120,
            "max_concurrency_per_reviewer": 1,
            "reviewers": [
                {
                    "reviewer_id": "glm_high_parameter",
                    "backend": "openai_compatible",
                    "api_base_env": "STAGE1_GLM_API_BASE",
                    "default_api_base": REVIEWERS[0]["api_base"],
                    "api_key_env": "GLM_API_KEY",
                    "model_env": "STAGE1_GLM_MODEL",
                    "default_model": REVIEWERS[0]["model"],
                },
                {
                    "reviewer_id": "deepseek_flash",
                    "backend": "openai_compatible",
                    "api_base_env": "STAGE1_DEEPSEEK_API_BASE",
                    "default_api_base": REVIEWERS[1]["api_base"],
                    "api_key_env": "DEEPSEEK_API_KEY",
                    "model_env": "STAGE1_DEEPSEEK_MODEL",
                    "default_model": REVIEWERS[1]["model"],
                },
            ],
            "blindness": {
                "reviewers_see_each_other": False,
                "show_model_predictions": False,
                "show_experimental_condition": False,
                "show_source_split": False,
                "stable_shuffle_seed": 42,
            },
            "consensus": {
                "auto_accept_only_exact_structured_agreement": True,
                "low_confidence_threshold": 0.8,
                "agreement_qc_fraction": 0.1,
                "agreement_qc_assignment": "sha256-review-item-id-v1",
                "human_queue_reasons": [
                    "model_disagreement",
                    "parse_failure",
                    "low_confidence",
                    "agreement_qc_sample",
                ],
            },
            "reviewer_request_options": {
                "glm_high_parameter": {
                    "thinking": {"type": "enabled"},
                    "reasoning_effort": "high",
                },
                "deepseek_flash": {
                    "thinking": {"type": "disabled"},
                    "reasoning_effort": "medium",
                },
            },
            "provenance": {
                "record_provider_request_id_when_available": True,
                "record_model_identifier": True,
                "record_prompt_sha256": True,
                "retain_raw_request_without_secrets": True,
                "retain_raw_response": True,
            },
        }
        self.policy_path = root / "config" / "stage1" / "blind_review.json"
        write_canonical_json(self.policy_path, self.policy)
        blind_map = _blind_map(self.issues, 42)
        self.model_rows = {}
        for reviewer in REVIEWERS:
            self.model_rows[reviewer["reviewer_id"]] = [
                _review_row(self.issues[0], reviewer, blind_map[self.issue_ids[0]]),
                _review_row(
                    self.issues[1],
                    reviewer,
                    blind_map[self.issue_ids[1]],
                    corrected=reviewer["reviewer_id"] == "deepseek_flash",
                ),
            ]
        consensus, queue, auto = build_consensus(
            self.issues,
            self.model_rows,
            low_confidence_threshold=0.8,
            agreement_qc_fraction=0.1,
        )
        scope = {
            "issue_kinds": ["group-hate"],
            "issue_count": 2,
            "scope_policy": "explicit-audit-subset/v1",
        }
        ordered_ids = [
            issue_id
            for issue_id, _ in sorted(
                (
                    (
                        issue["issue_id"],
                        hashlib.sha256(
                            f"42:{issue['issue_id']}".encode()
                        ).hexdigest(),
                    )
                    for issue in self.issues
                ),
                key=lambda pair: pair[1],
            )
        ]
        run_inputs = {
            "schema_version": "stage1-dual-blind-review-run/v1",
            "audit_artifact_id": self.audit_id,
            "audit_payload_manifest_sha256": sha256_file(
                self.audit_target / "payload_manifest.json"
            ),
            "policy_sha256": sha256_file(self.policy_path),
            "system_prompt_sha256": hashlib.sha256(
                SYSTEM_PROMPT.encode("utf-8")
            ).hexdigest(),
            "reviewers": REVIEWERS,
            "issue_scope": scope,
            "ordered_issue_ids_sha256": canonical_sha256(ordered_ids),
        }
        self.run_id = "dblind-" + canonical_sha256(run_inputs)
        self.review_target = root / "source" / self.run_id
        self.review_target.mkdir(parents=True)
        write_canonical_json(
            self.review_target / "run.meta.json",
            {**run_inputs, "review_run_id": self.run_id},
        )
        write_canonical_json(
            self.review_target / "policy.resolved.json", self.policy
        )
        write_canonical_json(
            self.review_target / "model_resolution.json", {"reviewers": REVIEWERS}
        )
        for reviewer_id, values in self.model_rows.items():
            write_canonical_jsonl(
                self.review_target / "model_reviews" / f"{reviewer_id}.jsonl",
                values,
                key="issue_id",
            )
        templates = [
            {
                "schema_version": "stage1-data-adjudication-row/v1",
                "data_audit_id": self.audit_id,
                "issue_id": packet["issue_id"],
                "issue_kind": packet["issue_kind"],
                "decision": "",
                "edits": [],
                "reason_code": "",
                "reason": "",
                "reviewer_id": "dual-blind-panel-v1",
                "reviewed_at": "",
            }
            for packet in queue
        ]
        for name, values in (
            ("consensus.jsonl", consensus),
            ("human_queue.jsonl", queue),
            ("auto_adjudication.jsonl", auto),
            ("human_adjudication_template.jsonl", templates),
        ):
            write_canonical_jsonl(self.review_target / name, values, key="issue_id")
        write_canonical_json(
            self.review_target / "summary.json",
            {
                "schema_version": "stage1-dual-blind-review-summary/v1",
                "review_run_id": self.run_id,
                "issue_count": 2,
                "auto_adjudication_count": len(auto),
                "human_queue_count": len(queue),
                "human_queue_reasons": {
                    reason: sum(reason in packet["queue_reasons"] for packet in queue)
                    for reason in self.policy["consensus"]["human_queue_reasons"]
                },
                "target_path": str(self.review_target.resolve()),
            },
        )
        self.output = root / "sealed"
        self.ref = root / "sealed_ref.json"

    def seal(self, *, ref: Path | None = None) -> dict:
        return seal_data_review(
            audit_ref=self.audit_ref,
            review_target=self.review_target,
            output_root=self.output,
            write_ref=ref or self.ref,
            workspace_root=self.root,
        )


class DataReviewArtifactTests(unittest.TestCase):
    def test_seal_validate_and_resolve_return_vote_free_report(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = Fixture(Path(directory))
            locator = fixture.seal()
            resolved_locator, target, report = resolve_data_review_ref(
                fixture.ref, workspace_root=fixture.root
            )
            self.assertEqual(locator, resolved_locator)
            self.assertEqual(report["issue_count"], 2)
            self.assertEqual(report["auto_adjudication_count"], 0)
            self.assertEqual(report["human_queue_count"], 2)
            self.assertFalse(
                {"consensus", "model_reviews", "votes", "raw_response"}.intersection(report)
            )
            self.assertEqual(target.name, report["data_blind_review_id"])

    def test_valid_source_content_tamper_changes_artifact_id(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = Fixture(Path(directory))
            first = fixture.seal(ref=fixture.root / "first_ref.json")
            path = fixture.review_target / "model_reviews" / "deepseek_flash.jsonl"
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            rows[0]["raw_response"]["usage"] = {"total_tokens": 1}
            write_canonical_jsonl(path, rows, key="issue_id")
            second = fixture.seal(ref=fixture.root / "second_ref.json")
            self.assertNotEqual(first["artifact_id"], second["artifact_id"])

    def test_output_tamper_fails_manifest_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = Fixture(Path(directory))
            locator = fixture.seal()
            target = Path(locator["target_path"])
            summary = json.loads((target / "summary.json").read_text(encoding="utf-8"))
            summary["issue_count"] = 99
            write_canonical_json(target / "summary.json", summary)
            with self.assertRaises(DataReviewArtifactError):
                validate_data_review_ref(fixture.ref, workspace_root=fixture.root)

    def test_consensus_queue_frame_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = Fixture(Path(directory))
            write_canonical_jsonl(
                fixture.review_target / "human_queue.jsonl", [], key="issue_id"
            )
            with self.assertRaisesRegex(DataReviewArtifactError, "queue"):
                fixture.seal()

    def test_source_symlink_and_unexpected_file_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = Fixture(Path(directory))
            extra = fixture.review_target / "model_reviews" / "unexpected.jsonl"
            extra.symlink_to(fixture.review_target / "consensus.jsonl")
            with self.assertRaisesRegex(DataReviewArtifactError, "symlink"):
                fixture.seal()
            extra.unlink()
            (fixture.review_target / "unexpected.json").write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(DataReviewArtifactError, "file frame"):
                fixture.seal()

    def test_unapproved_policy_or_model_resolution_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = Fixture(Path(directory))
            policy = json.loads(fixture.policy_path.read_text(encoding="utf-8"))
            policy["reviewers"][0]["default_model"] = "unapproved-model"
            write_canonical_json(fixture.policy_path, policy)
            with self.assertRaisesRegex(DataReviewArtifactError, "canonical D14"):
                fixture.seal()

    def test_model_provider_and_prompt_provenance_tampering_is_rejected(self):
        mutations = ("returned-model", "provider", "prompt")
        for label in mutations:
            with self.subTest(label=label), tempfile.TemporaryDirectory() as directory:
                fixture = Fixture(Path(directory))
                path = fixture.review_target / "model_reviews/deepseek_flash.jsonl"
                rows = [
                    json.loads(line)
                    for line in path.read_text(encoding="utf-8").splitlines()
                ]
                if label == "returned-model":
                    rows[0]["returned_model"] = "different-model"
                    rows[0]["raw_response"]["model"] = "different-model"
                    expected = "returned model"
                elif label == "provider":
                    rows[0]["provider_api_base"] = "https://changed.invalid/v1"
                    expected = "identity/frame"
                else:
                    payload = rows[0]["request_without_secrets"]["payload"]
                    payload["messages"][1]["content"] += " tampered"
                    changed_hash = hashlib.sha256(
                        (
                            payload["messages"][0]["content"]
                            + "\n"
                            + payload["messages"][1]["content"]
                        ).encode("utf-8")
                    ).hexdigest()
                    rows[0]["prompt_sha256"] = changed_hash
                    rows[0]["request_without_secrets"]["prompt_sha256"] = changed_hash
                    expected = "prompt hash"
                write_canonical_jsonl(path, rows, key="issue_id")
                with self.assertRaisesRegex(DataReviewArtifactError, expected):
                    fixture.seal()

    def test_sealed_payload_removes_source_absolute_target_path(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = Fixture(Path(directory))
            locator = fixture.seal()
            target = Path(locator["target_path"])
            summary = json.loads((target / "summary.json").read_text(encoding="utf-8"))
            self.assertNotIn("target_path", summary)
            source_bytes = str(fixture.review_target.resolve()).encode()
            for path in target.rglob("*"):
                self.assertFalse(path.is_symlink())
                if path.is_file():
                    self.assertNotIn(source_bytes, path.read_bytes())
            self.assertTrue(validate_data_review_ref(fixture.ref, workspace_root=fixture.root)["valid"])

    def test_sealed_ref_drives_blind_human_workspace_and_export(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = Fixture(Path(directory))
            locator = fixture.seal()
            target = Path(locator["target_path"])
            exported_rows = fixture.root / "human.jsonl"
            exported_packets = fixture.root / "packets.jsonl"
            export_human_adjudication_workspace(
                review_ref=fixture.ref,
                workspace_root=fixture.root,
                output=exported_rows,
                packet_output=exported_packets,
            )
            workspace = load_human_review_workspace(
                audit_ref=fixture.audit_ref,
                packet_file=exported_packets,
                adjudication_file=exported_rows,
                review_ref=fixture.ref,
                workspace_root=fixture.root,
            )
            self.assertEqual(workspace.review_target, target)
            self.assertEqual(workspace_status(workspace)["incomplete_count"], 2)

    def test_sealed_ref_drives_strict_human_merge(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = Fixture(Path(directory))
            locator = fixture.seal()
            target = Path(locator["target_path"])
            completed = fixture.root / "completed.jsonl"
            rows = [
                json.loads(line)
                for line in (target / "human_adjudication_template.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertEqual(len(rows), 2)
            for row in rows:
                row.update(
                    {
                        "decision": "accepted",
                        "edits": [],
                        "reason_code": "valid-label",
                        "reason": "human confirmed the independent labels",
                        "reviewed_at": "2026-08-22T00:00:00+00:00",
                    }
                )
            write_canonical_jsonl(completed, rows, key="issue_id")
            output = fixture.root / "merged.jsonl"
            report = merge_human_adjudication(
                audit_ref=fixture.audit_ref,
                review_ref=fixture.ref,
                workspace_root=fixture.root,
                human_completed=completed,
                output=output,
            )
            self.assertEqual(report["row_count"], 2)
            self.assertEqual(len(output.read_text(encoding="utf-8").splitlines()), 2)


if __name__ == "__main__":
    unittest.main()
