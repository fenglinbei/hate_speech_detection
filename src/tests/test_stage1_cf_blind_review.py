import copy
import json
import shutil
import tempfile
import threading
import unittest
from pathlib import Path
from typing import Any, Mapping

from data.counterfactual_lifecycle import (
    prepare_reviewer_declaration,
    propose_counterfactual_artifact,
)
from data.training_artifacts import (
    build_payload_manifest,
    load_json,
    load_jsonl,
    write_canonical_json,
    write_canonical_jsonl,
)
from review.cf_blind_review import (
    BlindReviewError,
    _validate_policy_document,
    _validate_transport_contract,
    build_cf_consensus,
    export_cf_human_review,
    merge_cf_human_review,
    run_cf_blind_review,
    validate_cf_blind_review_ref,
    validate_cf_blind_review_target,
)
from review.d14_contract import (
    D14_LIVE_EXECUTION_MODE,
    D14_SYNTHETIC_EXECUTION_MODE,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
PANEL_REVIEWER_ID = "dual-blind-panel-v1"
FINAL_REVIEW_KEYS = {
    "schema_version",
    "cf_proposal_id",
    "candidate_id",
    "decision",
    "reason_code",
    "note",
    "reviewer_id",
}


def _quad(target: str | None, argument: str, group: str, hateful: str) -> dict[str, Any]:
    return {
        "target": target,
        "argument": argument,
        "targeted_group": [group],
        "hateful": hateful,
    }


def _context_records() -> list[dict[str, Any]]:
    return [
        {
            "query": {
                "id": "10",
                "content": "坏甲说法，坏乙观点。",
                "gold": [
                    _quad("坏甲", "坏甲说法", "Racism", "hate"),
                    _quad("坏乙", "坏乙观点", "Sexism", "hate"),
                ],
            },
            "record_sha256": "a" * 64,
        }
    ]


def _train_records() -> list[dict[str, Any]]:
    return [
        {
            "id": "1",
            "content": "x",
            "quadruples": [_quad("x", "x", "Racism", "hate")],
        },
        {
            "id": "2",
            "content": "y",
            "quadruples": [_quad("y", "y", "Sexism", "hate")],
        },
        {
            "id": "3",
            "content": "z",
            "quadruples": [_quad(None, "z", "non-hate", "non-hate")],
        },
    ]


def _model_review(
    candidate_id: str,
    reviewer_id: str,
    *,
    decision: str = "pass",
    reason_code: str = "valid-local-foil",
    confidence: float = 0.95,
    error: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    judgement = None
    if error is None:
        judgement = {
            "decision": decision,
            "reason_code": reason_code,
            "note": "fixture judgement",
            "confidence": confidence,
        }
    return {
        "candidate_id": candidate_id,
        "reviewer_id": reviewer_id,
        "judgement": judgement,
        "error": dict(error) if error is not None else None,
    }


class FakeTransport:
    """Thread-safe OpenAI-compatible fake; it never performs network I/O."""

    def __init__(
        self,
        responses_by_model: Mapping[str, Mapping[str, Any] | str] | None = None,
    ) -> None:
        self.responses_by_model = dict(responses_by_model or {})
        self.calls: list[dict[str, Any]] = []
        self._lock = threading.Lock()

    def __call__(
        self,
        url: str,
        request_payload: dict[str, Any],
        headers: dict[str, str],
        timeout: int,
    ) -> dict[str, Any]:
        with self._lock:
            call_index = len(self.calls) + 1
            self.calls.append(
                {
                    "url": url,
                    "request_payload": copy.deepcopy(request_payload),
                    "headers": dict(headers),
                    "timeout": timeout,
                }
            )
        model = str(request_payload["model"])
        configured = self.responses_by_model.get(model)
        if configured == "malformed-json":
            content = "{this-is-not-json"
        else:
            judgement = {
                "decision": "pass",
                "reason_code": "valid-local-foil",
                "note": "fixture pass",
                "confidence": 0.95,
            }
            if isinstance(configured, Mapping):
                judgement.update(dict(configured))
            content = json.dumps(
                judgement,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        return {
            "id": f"fixture-request-{call_index}",
            "model": model,
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"content": content},
                }
            ],
        }


class CounterfactualBlindReviewTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.refs = self.root / "refs"
        self.proposal_ref = self.refs / "cf_proposal_ref.json"
        self.blind_review_ref = self.refs / "cf_blind_review_ref.json"
        self.blind_review_root = self.root / "cf_blind_reviews"
        self.secret_a = "fixture-secret-reviewer-a-71f9"
        self.secret_b = "fixture-secret-reviewer-b-a8c2"
        self.model_a = "fixture-cf-model-a"
        self.model_b = "fixture-cf-model-b"
        self.env_file = self.root / ".env"
        self.env_file.write_text(
            "\n".join(
                [
                    f"GLM_API_KEY={self.secret_a}",
                    f"DEEPSEEK_API_KEY={self.secret_b}",
                    "STAGE1_GLM_API_BASE=https://reviewer-a.invalid/v1",
                    "STAGE1_DEEPSEEK_API_BASE=https://reviewer-b.invalid/v1",
                    f"STAGE1_GLM_MODEL={self.model_a}",
                    f"STAGE1_DEEPSEEK_MODEL={self.model_b}",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        self._build_proposal()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _build_proposal(self) -> None:
        foil_policy = load_json(REPOSITORY_ROOT / "config/stage1/cf_foil_policy.json")
        foil_policy["sampling"]["review_candidate_target"] = 4
        foil_policy["sampling"]["field_quota"] = {"target": 2, "argument": 2}
        proposal = propose_counterfactual_artifact(
            config={"artifact_root": "unused-for-explicit-target"},
            foil_policy=foil_policy,
            review_rubric=REPOSITORY_ROOT / "config/stage1/cf_review_rubric.md",
            write_ref=self.proposal_ref,
            split="dev",
            engineering_context_records=_context_records(),
            engineering_train_records=_train_records(),
            target_root=self.root,
            workspace_root=self.root,
        )
        self.proposal_target = Path(proposal["target_path"])
        self.proposal_rows = load_jsonl(self.proposal_target / "candidates.dev.jsonl")
        self.required_rows = [row for row in self.proposal_rows if row["review_required"]]
        self.automatic_rows = [row for row in self.proposal_rows if not row["review_required"]]
        self.assertGreaterEqual(len(self.required_rows), 3)
        self.assertEqual({row["field"] for row in self.required_rows}, {"target", "argument"})
        self.assertEqual({row["field"] for row in self.automatic_rows}, {"targeted_group", "hateful"})

    def _write_review_policy(self) -> Path:
        policy = load_json(REPOSITORY_ROOT / "config/stage1/blind_review.json")
        path = self.root / "blind_review.json"
        path.write_text(
            json.dumps(policy, ensure_ascii=False, sort_keys=True), encoding="utf-8"
        )
        return path

    def _run(
        self,
        transport: FakeTransport,
    ) -> dict[str, Any]:
        return run_cf_blind_review(
            proposal_ref=self.proposal_ref,
            policy_path=self._write_review_policy(),
            env_file=self.env_file,
            output_dir=self.blind_review_root,
            write_ref=self.blind_review_ref,
            workspace_root=self.root,
            transport=transport,
            execution_mode=D14_SYNTHETIC_EXECUTION_MODE,
        )

    def _export(self, *, stem: str = "human") -> tuple[Path, Path, dict[str, Any]]:
        workspace = self.root / f"{stem}_workspace.jsonl"
        packets = self.root / f"{stem}_packets.jsonl"
        report = export_cf_human_review(
            blind_review_ref=self.blind_review_ref,
            output=workspace,
            packet_output=packets,
            workspace_root=self.root,
        )
        return workspace, packets, report

    def test_run_reviews_target_and_argument_twice_and_never_reviews_automatic_fields(self) -> None:
        transport = FakeTransport()
        proposal_before = {
            path.relative_to(self.proposal_target).as_posix(): path.read_bytes()
            for path in self.proposal_target.rglob("*")
            if path.is_file()
        }
        locator = self._run(transport)
        proposal_after = {
            path.relative_to(self.proposal_target).as_posix(): path.read_bytes()
            for path in self.proposal_target.rglob("*")
            if path.is_file()
        }
        self.assertEqual(proposal_after, proposal_before)

        self.assertEqual(len(transport.calls), 2 * len(self.required_rows))
        calls_by_model = {
            model: sum(call["request_payload"]["model"] == model for call in transport.calls)
            for model in (self.model_a, self.model_b)
        }
        self.assertEqual(
            calls_by_model,
            {self.model_a: len(self.required_rows), self.model_b: len(self.required_rows)},
        )
        serialized_requests = [
            json.dumps(call["request_payload"], ensure_ascii=False, sort_keys=True)
            for call in transport.calls
        ]
        for row in self.required_rows:
            self.assertEqual(
                sum(row["candidate_id"] in request for request in serialized_requests), 2
            )
        for row in self.automatic_rows:
            self.assertFalse(
                any(row["candidate_id"] in request for request in serialized_requests)
            )

        self.assertEqual(locator, load_json(self.blind_review_ref))
        validation = validate_cf_blind_review_ref(
            self.blind_review_ref, workspace_root=self.root
        )
        self.assertTrue(validation["valid"])

        human_workspace, _, export_report = self._export(stem="qc-human")
        human_templates = load_jsonl(human_workspace)
        self.assertEqual(export_report["row_count"], len(human_templates))
        completed_human = []
        for template in human_templates:
            row = dict(template)
            row.update(
                {
                    "decision": "pass",
                    "reason_code": "valid-local-foil",
                    "note": "independent QC review",
                    "reviewer_id": PANEL_REVIEWER_ID,
                }
            )
            completed_human.append(row)
        write_canonical_jsonl(
            human_workspace, completed_human, key="candidate_id"
        )
        merged = self.root / "merged_review.jsonl"
        declaration_output = self.root / "reviewer_declaration.json"
        merge_report = merge_cf_human_review(
            blind_review_ref=self.blind_review_ref,
            human_completed=human_workspace,
            output=merged,
            declaration_output=declaration_output,
            workspace_root=self.root,
        )
        self.assertEqual(merge_report["row_count"], len(self.proposal_rows))
        final_rows = load_jsonl(merged)
        self.assertEqual(
            {row["candidate_id"] for row in final_rows},
            {row["candidate_id"] for row in self.proposal_rows},
        )
        self.assertTrue(all(set(row) == FINAL_REVIEW_KEYS for row in final_rows))
        self.assertTrue(
            all(row["reviewer_id"] == PANEL_REVIEWER_ID for row in final_rows)
        )
        automatic_ids = {row["candidate_id"] for row in self.automatic_rows}
        for row in final_rows:
            if row["candidate_id"] in automatic_ids:
                self.assertEqual(
                    (row["decision"], row["reason_code"]),
                    ("not_required", "deterministic-label-foil"),
                )
            else:
                self.assertEqual(
                    (row["decision"], row["reason_code"]),
                    ("pass", "valid-local-foil"),
                )

        prepared = prepare_reviewer_declaration(
            blind_review_ref=self.blind_review_ref,
            review_file=merged,
            reviewer_id=PANEL_REVIEWER_ID,
            workspace_root=self.root,
        )
        self.assertEqual(load_json(declaration_output), prepared)
        self.assertFalse(prepared["attestation_confirmed"])
        self.assertEqual(
            prepared["cf_blind_review_dependency"]["artifact_id"],
            load_json(self.blind_review_ref)["artifact_id"],
        )
        blind_target = Path(load_json(self.blind_review_ref)["target_path"])
        self.assertEqual(
            prepared["human_queue_rows_sha256"],
            load_json(blind_target / "provenance.json")[
                "human_queue_rows_sha256"
            ],
        )
        self.assertEqual(
            prepared["auto_review_rows_sha256"],
            load_json(blind_target / "provenance.json")[
                "auto_review_rows_sha256"
            ],
        )

        persisted = self.blind_review_ref.read_bytes() + b"".join(
            path.read_bytes()
            for path in sorted(self.blind_review_root.rglob("*"))
            if path.is_file()
        )
        self.assertNotIn(self.secret_a.encode(), persisted)
        self.assertNotIn(self.secret_b.encode(), persisted)
        self.assertTrue(
            any(self.secret_a in call["headers"]["Authorization"] for call in transport.calls)
        )
        self.assertTrue(
            any(self.secret_b in call["headers"]["Authorization"] for call in transport.calls)
        )

    def test_build_consensus_routes_disagreement_parse_failure_low_confidence_and_qc(self) -> None:
        first, second, third = self.required_rows[:3]
        reviewers = {
            "deepseek_flash": [
                _model_review(first["candidate_id"], "deepseek_flash"),
                _model_review(
                    second["candidate_id"],
                    "deepseek_flash",
                    error={"type": "JSONDecodeError", "message": "malformed"},
                ),
                _model_review(
                    third["candidate_id"], "deepseek_flash", confidence=0.2
                ),
            ],
            "glm_high_parameter": [
                _model_review(
                    first["candidate_id"],
                    "glm_high_parameter",
                    decision="reject",
                    reason_code="unsupported-by-query",
                ),
                _model_review(second["candidate_id"], "glm_high_parameter"),
                _model_review(third["candidate_id"], "glm_high_parameter"),
            ],
        }
        _, queue, auto = build_cf_consensus(
            [first, second, third],
            reviewers,
            low_confidence_threshold=0.8,
            agreement_qc_fraction=0.0,
        )
        self.assertEqual(auto, [])
        queue_by_id = {row["candidate_id"]: row for row in queue}
        self.assertIn("model_disagreement", queue_by_id[first["candidate_id"]]["queue_reasons"])
        self.assertIn("parse_failure", queue_by_id[second["candidate_id"]]["queue_reasons"])
        self.assertIn("low_confidence", queue_by_id[third["candidate_id"]]["queue_reasons"])
        self.assertTrue(all(row["model_votes_hidden"] is True for row in queue))

        _, qc_queue, qc_auto = build_cf_consensus(
            [first],
            {
                "deepseek_flash": [
                    _model_review(first["candidate_id"], "deepseek_flash")
                ],
                "glm_high_parameter": [
                    _model_review(first["candidate_id"], "glm_high_parameter")
                ],
            },
            low_confidence_threshold=0.8,
            agreement_qc_fraction=1.0,
        )
        self.assertEqual(qc_auto, [])
        self.assertEqual(qc_queue[0]["queue_reasons"], ["agreement_qc_sample"])

    def test_disagreement_exports_blind_packets_and_human_merge_is_lifecycle_compatible(self) -> None:
        transport = FakeTransport(
            {
                self.model_b: {
                    "decision": "reject",
                    "reason_code": "unsupported-by-query",
                    "note": "fixture reject",
                    "confidence": 0.99,
                }
            }
        )
        self._run(transport)
        human_workspace, packets_path, report = self._export(stem="disagreement")
        self.assertEqual(report["row_count"], len(self.required_rows))

        packets = load_jsonl(packets_path)
        templates = load_jsonl(human_workspace)
        self.assertEqual(
            {row["candidate_id"] for row in packets},
            {row["candidate_id"] for row in self.required_rows},
        )
        self.assertEqual(
            {row["candidate_id"] for row in templates},
            {row["candidate_id"] for row in self.required_rows},
        )
        self.assertTrue(all(row["model_votes_hidden"] is True for row in packets))
        exported_bytes = packets_path.read_bytes() + human_workspace.read_bytes()
        for forbidden in (
            b"deepseek_flash",
            b"glm_high_parameter",
            self.model_a.encode(),
            self.model_b.encode(),
            b"raw_response",
            b"decision_projection",
            b"judgement",
        ):
            self.assertNotIn(forbidden, exported_bytes)

        completed = []
        for template in templates:
            row = dict(template)
            row.update(
                {
                    "decision": "pass",
                    "reason_code": "valid-local-foil",
                    "note": "independent human review",
                    "reviewer_id": PANEL_REVIEWER_ID,
                }
            )
            completed.append(row)
        write_canonical_jsonl(human_workspace, completed, key="candidate_id")
        merged = self.root / "human_merged_review.jsonl"
        declaration = self.root / "human_reviewer_declaration.json"
        merge_cf_human_review(
            blind_review_ref=self.blind_review_ref,
            human_completed=human_workspace,
            output=merged,
            declaration_output=declaration,
            workspace_root=self.root,
        )
        prepared = prepare_reviewer_declaration(
            blind_review_ref=self.blind_review_ref,
            review_file=merged,
            reviewer_id=PANEL_REVIEWER_ID,
            workspace_root=self.root,
        )
        self.assertEqual(load_json(declaration), prepared)

        missing = self.root / "missing_human_review.jsonl"
        write_canonical_jsonl(missing, completed[:-1], key="candidate_id")
        with self.assertRaises(BlindReviewError):
            merge_cf_human_review(
                blind_review_ref=self.blind_review_ref,
                human_completed=missing,
                output=self.root / "must_not_exist.jsonl",
                workspace_root=self.root,
            )

    def test_malformed_provider_response_enters_parse_failure_queue(self) -> None:
        self._run(FakeTransport({self.model_a: "malformed-json"}))
        _, packets_path, report = self._export(stem="parse-failure")
        self.assertEqual(report["row_count"], len(self.required_rows))
        packets = load_jsonl(packets_path)
        self.assertTrue(
            all("parse_failure" in row["queue_reasons"] for row in packets)
        )

    def test_low_confidence_exact_agreement_enters_human_queue(self) -> None:
        self._run(
            FakeTransport(
                {
                    self.model_a: {"confidence": 0.2},
                    self.model_b: {"confidence": 0.99},
                }
            )
        )
        _, packets_path, report = self._export(stem="low-confidence")
        self.assertEqual(report["row_count"], len(self.required_rows))
        self.assertTrue(
            all(
                "low_confidence" in row["queue_reasons"]
                for row in load_jsonl(packets_path)
            )
        )

    def test_d14_policy_seed_threshold_model_and_provider_are_fail_closed(self) -> None:
        policy = load_json(REPOSITORY_ROOT / "config/stage1/blind_review.json")
        mutations = {
            "seed": lambda value: value["blindness"].update(
                {"stable_shuffle_seed": 43}
            ),
            "threshold": lambda value: value["consensus"].update(
                {"low_confidence_threshold": 0.79}
            ),
            "model": lambda value: value["reviewers"][0].update(
                {"default_model": "glm-5.3-alias"}
            ),
            "provider": lambda value: value["reviewers"][1].update(
                {"default_api_base": "https://provider.invalid"}
            ),
        }
        for label, mutate in mutations.items():
            with self.subTest(label=label):
                candidate = copy.deepcopy(policy)
                mutate(candidate)
                with self.assertRaisesRegex(BlindReviewError, "canonical D14"):
                    _validate_policy_document(candidate)

    def test_scientific_transport_contract_rejects_every_injected_fake(self) -> None:
        fake = FakeTransport()
        with self.assertRaisesRegex(BlindReviewError, "built-in HTTP"):
            _validate_transport_contract(
                scientific_eligible=True,
                execution_mode=D14_LIVE_EXECUTION_MODE,
                transport=fake,
            )
        with self.assertRaisesRegex(BlindReviewError, "scientific/formal"):
            _validate_transport_contract(
                scientific_eligible=True,
                execution_mode=D14_SYNTHETIC_EXECUTION_MODE,
                transport=fake,
            )
        _validate_transport_contract(
            scientific_eligible=False,
            execution_mode=D14_SYNTHETIC_EXECUTION_MODE,
            transport=fake,
        )

    def test_target_validator_rejects_policy_provider_and_returned_model_tampering(self) -> None:
        locator = self._run(FakeTransport())
        source = Path(locator["target_path"])

        def copy_and_rehash(label: str) -> Path:
            target = self.root / f"tampered-{label}"
            shutil.copytree(source, target)
            return target

        def finalize_manifest(target: Path) -> None:
            write_canonical_json(
                target / "payload_manifest.json", build_payload_manifest(target)
            )

        policy_target = copy_and_rehash("policy")
        policy = load_json(policy_target / "policy.resolved.json")
        policy["consensus"]["low_confidence_threshold"] = 0.79
        write_canonical_json(policy_target / "policy.resolved.json", policy)
        finalize_manifest(policy_target)
        with self.assertRaisesRegex(BlindReviewError, "canonical D14"):
            validate_cf_blind_review_target(
                policy_target,
                workspace_root=self.root,
                proposal_target=self.proposal_target,
                require_directory_name=False,
            )

        provider_target = copy_and_rehash("provider")
        resolution = load_json(provider_target / "model_resolution.json")
        resolution["reviewers"][0]["api_base"] = "https://changed.invalid/v1"
        write_canonical_json(provider_target / "model_resolution.json", resolution)
        finalize_manifest(provider_target)
        with self.assertRaisesRegex(BlindReviewError, "identity/frame"):
            validate_cf_blind_review_target(
                provider_target,
                workspace_root=self.root,
                proposal_target=self.proposal_target,
                require_directory_name=False,
            )

        model_target = copy_and_rehash("returned-model")
        stream_path = model_target / "model_reviews/deepseek_flash.jsonl"
        rows = load_jsonl(stream_path)
        rows[0]["returned_model"] = "different-model"
        rows[0]["raw_response"]["model"] = "different-model"
        write_canonical_jsonl(stream_path, rows, key="candidate_id")
        finalize_manifest(model_target)
        with self.assertRaisesRegex(BlindReviewError, "returned model"):
            validate_cf_blind_review_target(
                model_target,
                workspace_root=self.root,
                proposal_target=self.proposal_target,
                require_directory_name=False,
            )

if __name__ == "__main__":
    unittest.main()
