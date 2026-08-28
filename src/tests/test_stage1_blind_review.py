import json
import tempfile
import unittest
from pathlib import Path

from data.training_artifacts import (
    build_payload_manifest,
    sha256_file,
    write_canonical_json,
    write_canonical_jsonl,
    write_locator_ref,
)
from review.blind_review import (
    BlindReviewError,
    _call_reviewer,
    _normalize_judgement_shape,
    _safe_parent_error,
    build_consensus,
    canonical_sha256,
    export_human_adjudication_workspace,
    load_env_file,
    load_review_policy,
    merge_human_adjudication,
    run_data_review,
)
from review.d14_contract import (
    d14_policy_document,
    validate_requested_returned_model,
)


def issue(issue_id="i1"):
    return {
        "issue_id": issue_id,
        "data_audit_id": "audit",
        "issue_kind": "group-hate",
        "issue_code": "group-hate-atypical",
        "accept_allowed": True,
        "allowed_edit_paths": ["/quadruples/0/hateful"],
        "review_context": {"content": "x"},
    }


def review(issue_id, reviewer_id, *, decision="accepted", confidence=0.95, error=None):
    judgement = None if error else {
        "decision": decision,
        "edits": [],
        "reason_code": "valid-independent-label-combination",
        "reason": "valid",
        "confidence": confidence,
    }
    return {"issue_id": issue_id, "reviewer_id": reviewer_id, "judgement": judgement, "error": error}


class ConsensusTests(unittest.TestCase):
    def test_exact_agreement_auto_adjudicates(self):
        consensus, queue, auto = build_consensus(
            [issue()],
            {"a": [review("i1", "a")], "b": [review("i1", "b")]},
            low_confidence_threshold=0.8,
            agreement_qc_fraction=0,
        )
        self.assertTrue(consensus[0]["agreed"])
        self.assertEqual(queue, [])
        self.assertEqual(auto[0]["decision"], "accepted")

    def test_disagreement_and_low_confidence_queue_without_votes(self):
        _, queue, auto = build_consensus(
            [issue("i1"), issue("i2")],
            {
                "a": [review("i1", "a"), review("i2", "a", confidence=0.5)],
                "b": [review("i1", "b", decision="corrected"), review("i2", "b")],
            },
            low_confidence_threshold=0.8,
            agreement_qc_fraction=0,
        )
        self.assertEqual(auto, [])
        self.assertEqual({row["issue_id"] for row in queue}, {"i1", "i2"})
        self.assertNotIn("decision", queue[0])
        self.assertTrue(all(row["model_votes_hidden"] for row in queue))

    def test_transport_failure_is_human_queue(self):
        _, queue, _ = build_consensus(
            [issue()],
            {"a": [review("i1", "a", error={"type": "x"})], "b": [review("i1", "b")]},
            low_confidence_threshold=0.8,
            agreement_qc_fraction=0,
        )
        self.assertIn("parse_failure", queue[0]["queue_reasons"])


class SecretLoadingTests(unittest.TestCase):
    def test_only_allowlisted_names_are_loaded(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / ".env"
            path.write_text("A='secret-a'\nB=secret-b\nUNUSED=do-not-load\n", encoding="utf-8")
            values = load_env_file(path, allowed_names={"A", "B"})
        self.assertEqual(values, {"A": "secret-a", "B": "secret-b"})

    def test_empty_allowlisted_key_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / ".env"
            path.write_text("A=\n", encoding="utf-8")
            with self.assertRaises(BlindReviewError):
                load_env_file(path, allowed_names={"A"})


class ModelEvidenceSecretBoundaryTests(unittest.TestCase):
    secret_a = "fixture-review-secret-a-71f9"
    secret_b = "fixture-review-secret-b-a8c2"

    def setUp(self):
        self.policy = {
            "temperature": 0,
            "top_p": 1,
            "max_tokens": 256,
            "request_timeout_seconds": 10,
            "reviewer_request_options": {},
        }
        self.resolved = {
            "reviewer_id": "reviewer-a",
            "api_base": "https://provider.invalid/v1",
            "api_key": self.secret_a,
            "model": "fixture-model",
        }
        self.review_issue = {
            **issue(),
            "issue_subtype": "atypical-combination",
        }
        self.rubric = {
            "reason_codes": {
                "group-hate-atypical": {
                    "accepted": ["valid-independent-label-combination"],
                    "corrected": ["correct-source-label"],
                }
            }
        }

    @staticmethod
    def _response(*, reason="ordinary model explanation", extra=None, model="fixture-model"):
        value = {
            "id": "request-fixture",
            "model": model,
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "content": json.dumps(
                            {
                                "decision": "accepted",
                                "edits": [],
                                "reason_code": "valid-independent-label-combination",
                                "reason": reason,
                                "confidence": 0.95,
                            }
                        )
                    },
                }
            ],
        }
        if extra is not None:
            value["provider_metadata"] = extra
        return value

    def _call(self, transport):
        return _call_reviewer(
            self.resolved,
            policy=self.policy,
            blind_item_id="B0001",
            issue=self.review_issue,
            rubric_meta=self.rubric,
            secrets=(self.secret_a, self.secret_b),
            transport=transport,
        )

    def test_nested_raw_secret_is_code_only_and_raw_response_is_dropped(self):
        row = self._call(
            lambda *_: self._response(
                extra={"debug": [{"echo": f"prefix-{self.secret_b}-suffix"}]}
            )
        )
        self.assertEqual(row["error"], {"code": "raw-response-contained-credential"})
        self.assertIsNone(row["raw_response"])
        self.assertIsNone(row["judgement"])
        self.assertNotIn(self.secret_a, json.dumps(row))
        self.assertNotIn(self.secret_b, json.dumps(row))

    def test_authorization_fragment_and_nested_exception_are_contained(self):
        for evidence in (
            {"trace": "Authorization: Bearer opaque-token-12345"},
            {"failure": RuntimeError(f"provider echoed {self.secret_a}")},
        ):
            with self.subTest(evidence_type=type(next(iter(evidence.values()))).__name__):
                row = self._call(lambda *_, evidence=evidence: self._response(extra=evidence))
                self.assertEqual(
                    row["error"], {"code": "raw-response-contained-credential"}
                )
                self.assertIsNone(row["raw_response"])

    def test_exception_text_is_never_persisted(self):
        def failing_transport(*_):
            raise RuntimeError(f"request failed with {self.secret_a}")

        row = self._call(failing_transport)
        self.assertEqual(row["error"], {"code": "raw-response-contained-credential"})
        self.assertIsNone(row["raw_response"])
        rendered = json.dumps(row)
        self.assertNotIn("RuntimeError", rendered)
        self.assertNotIn("request failed", rendered)
        self.assertNotIn(self.secret_a, rendered)

    def test_retry_lineage_never_copies_legacy_exception_text(self):
        self.assertEqual(
            _safe_parent_error(
                {"type": "RuntimeError", "message": f"failed with {self.secret_a}"}
            ),
            {"code": "parent-review-failure"},
        )
        self.assertEqual(
            _safe_parent_error({"code": "transport-failure"}),
            {"code": "transport-failure"},
        )

    def test_ordinary_auth_discussion_is_not_a_false_positive(self):
        reason = (
            "This ordinary explanation discusses authorization and the phrase "
            "Authorization: Bearer token, but contains no concrete credential."
        )
        row = self._call(lambda *_: self._response(reason=reason))
        self.assertIsNone(row["error"])
        self.assertEqual(row["judgement"]["reason"], reason)

    def test_policy_cannot_embed_a_key_or_authorization_header(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "policy.json"
            policy = self._complete_policy()
            policy["reviewers"][0]["api_key"] = self.secret_a
            path.write_text(json.dumps(policy), encoding="utf-8")
            with self.assertRaises(BlindReviewError) as caught:
                load_review_policy(path)
            self.assertNotIn(self.secret_a, str(caught.exception))

    def _complete_policy(self):
        return d14_policy_document()

    def _write_run_fixture(self, root, *, content="ordinary audit content"):
        audit_target = root / "audit-fixture"
        audit_target.mkdir()
        audit_issue = {
            **self.review_issue,
            "data_audit_id": "audit-fixture",
            "review_context": {"content": content},
        }
        write_canonical_jsonl(audit_target / "issues.jsonl", [audit_issue], key="issue_id")
        write_canonical_json(
            audit_target / "adjudication_rubric.meta.json",
            self.rubric,
        )
        write_canonical_json(
            audit_target / "payload_manifest.json",
            build_payload_manifest(audit_target),
        )
        audit_ref = root / "audit_ref.json"
        write_locator_ref(
            audit_ref,
            artifact_kind="data-audit",
            artifact_id="audit-fixture",
            target=audit_target,
            payload_manifest_sha256=sha256_file(audit_target / "payload_manifest.json"),
        )
        policy_path = root / "policy.json"
        write_canonical_json(policy_path, self._complete_policy())
        env_path = root / "fixture.env"
        env_path.write_text(
            f"GLM_API_KEY={self.secret_a}\nDEEPSEEK_API_KEY={self.secret_b}\n",
            encoding="utf-8",
        )
        return audit_ref, policy_path, env_path

    def test_run_persists_only_safe_codes_when_provider_echoes_secrets(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audit_ref, policy_path, env_path = self._write_run_fixture(root)

            def echoing_transport(_url, payload, headers, _timeout):
                echoed = headers["Authorization"].removeprefix("Bearer ")
                return self._response(
                    extra={"nested": [{"echo": echoed}]}, model=payload["model"]
                )

            summary = run_data_review(
                audit_ref=audit_ref,
                policy_path=policy_path,
                env_file=env_path,
                output_dir=root / "reviews",
                transport=echoing_transport,
            )
            target = root / "reviews" / summary["review_run_id"]
            persisted = b"".join(
                path.read_bytes() for path in sorted(target.rglob("*")) if path.is_file()
            )
            self.assertNotIn(self.secret_a.encode(), persisted)
            self.assertNotIn(self.secret_b.encode(), persisted)
            for path in sorted((target / "model_reviews").glob("*.jsonl")):
                row = json.loads(path.read_text(encoding="utf-8"))
                self.assertIsNone(row["raw_response"])
                self.assertEqual(
                    row["error"], {"code": "raw-response-contained-credential"}
                )

    def test_request_evidence_with_secret_fails_before_output_directory_exists(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audit_ref, policy_path, env_path = self._write_run_fixture(
                root, content=f"accidental source echo {self.secret_a}"
            )
            output_dir = root / "reviews"
            calls = []

            def transport(*args):
                calls.append(args)
                return self._response()

            with self.assertRaises(BlindReviewError) as caught:
                run_data_review(
                    audit_ref=audit_ref,
                    policy_path=policy_path,
                    env_file=env_path,
                    output_dir=output_dir,
                    transport=transport,
                )
            self.assertEqual(calls, [])
            self.assertFalse(output_dir.exists())
            self.assertNotIn(self.secret_a, str(caught.exception))


class ProviderShapeNormalizationTests(unittest.TestCase):
    def test_only_exact_glm_pointer_quirk_is_normalized(self):
        value, codes = _normalize_judgement_shape(
            {
                "decision": "corrected",
                "edits": [{"location_index": 0, "op": "set", "_pointer": "/x", "value": "hate"}],
                "reason_code": "x",
                "reason": "x",
                "confidence": 1,
            }
        )
        self.assertEqual(value["edits"][0]["json_pointer"], "/x")
        self.assertEqual(codes, ["edit-key:_pointer-to-json_pointer/v1"])

        untouched, codes = _normalize_judgement_shape({"edits": [{"path": "/x"}]})
        self.assertEqual(untouched["edits"], [{"path": "/x"}])
        self.assertEqual(codes, [])

    def test_model_identity_requires_two_nonempty_normalized_exact_ids(self):
        self.assertEqual(
            validate_requested_returned_model(" GLM-5.3 ", "glm-5.3"),
            "glm-5.3",
        )
        for returned in (None, "", "glm-5.3-alias"):
            with self.subTest(returned=returned), self.assertRaises(ValueError):
                validate_requested_returned_model("glm-5.3", returned)


class HumanWorkspaceTests(unittest.TestCase):
    def test_export_keeps_auto_rows_out_and_votes_hidden(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            row = {
                "schema_version": "stage1-data-adjudication-row/v1",
                "data_audit_id": "audit",
                "issue_id": "human",
                "issue_kind": "group-hate",
                "decision": "",
                "edits": [],
                "reason_code": "",
                "reason": "",
                "reviewer_id": "panel",
                "reviewed_at": "",
            }
            values = {
                "human_adjudication_template.jsonl": [row],
                "human_queue.jsonl": [{"issue_id": "human", "model_votes_hidden": True}],
                "auto_adjudication.jsonl": [{"issue_id": "auto"}],
                "consensus.jsonl": [{"issue_id": "human"}, {"issue_id": "auto"}],
            }
            for name, rows in values.items():
                (root / name).write_text(
                    "".join(json.dumps(item) + "\n" for item in rows), encoding="utf-8"
                )
            output = root / "workspace.jsonl"
            packets = root / "packets.jsonl"
            report = export_human_adjudication_workspace(
                review_target=root, output=output, packet_output=packets
            )
            self.assertEqual(report["row_count"], 1)
            self.assertEqual(json.loads(output.read_text())["issue_id"], "human")
            self.assertTrue(json.loads(packets.read_text())["model_votes_hidden"])

    def test_merge_rejects_changed_reviewer_instead_of_silently_overwriting(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audit_target = root / "audit"
            audit_target.mkdir()
            audit_issue = {
                "data_audit_id": "audit",
                "issue_id": "human",
                "issue_kind": "group-hate",
                "issue_code": "group-hate-atypical",
                "accept_allowed": True,
                "allowed_edit_paths": ["/quadruples/0/hateful"],
                "locations": [{"source_key": "std-train", "source_ordinal": 0}],
            }
            write_canonical_jsonl(
                audit_target / "issues.jsonl", [audit_issue], key="issue_id"
            )
            write_canonical_json(
                audit_target / "adjudication_rubric.meta.json",
                {
                    "reason_codes": {
                        "group-hate-atypical": {
                            "accepted": ["valid-independent-label-combination"],
                            "corrected": ["correct-source-label"],
                        }
                    }
                },
            )
            write_canonical_json(
                audit_target / "payload_manifest.json",
                build_payload_manifest(audit_target),
            )
            audit_ref = root / "audit_ref.json"
            payload_hash = sha256_file(audit_target / "payload_manifest.json")
            write_locator_ref(
                audit_ref,
                artifact_kind="data-audit",
                artifact_id="audit",
                target=audit_target,
                payload_manifest_sha256=payload_hash,
            )
            review_target = root / "dblind-test"
            review_target.mkdir()
            write_canonical_json(
                review_target / "run.meta.json",
                {
                    "review_run_id": "dblind-test",
                    "audit_artifact_id": "audit",
                    "audit_payload_manifest_sha256": payload_hash,
                    "issue_scope": {
                        "issue_kinds": ["group-hate"],
                        "issue_count": 1,
                        "scope_policy": "explicit-audit-subset/v1",
                    },
                    "ordered_issue_ids_sha256": canonical_sha256(["human"]),
                },
            )
            write_canonical_json(
                review_target / "policy.resolved.json",
                d14_policy_document(),
            )
            template = {
                "schema_version": "stage1-data-adjudication-row/v1",
                "data_audit_id": "audit",
                "issue_id": "human",
                "issue_kind": "group-hate",
                "decision": "",
                "edits": [],
                "reason_code": "",
                "reason": "",
                "reviewer_id": "dual-blind-panel-v1",
                "reviewed_at": "",
            }
            completed = dict(template)
            completed.update(
                {
                    "decision": "accepted",
                    "reason_code": "valid-independent-label-combination",
                    "reason": "valid",
                    "reviewer_id": "unexpected-reviewer",
                    "reviewed_at": "2026-08-22T00:00:00+00:00",
                }
            )
            values = {
                "human_adjudication_template.jsonl": [template],
                "human_queue.jsonl": [{"issue_id": "human"}],
                "auto_adjudication.jsonl": [],
                "consensus.jsonl": [{"issue_id": "human"}],
            }
            for name, rows in values.items():
                (review_target / name).write_text(
                    "".join(json.dumps(item) + "\n" for item in rows),
                    encoding="utf-8",
                )
            human = root / "human.jsonl"
            human.write_text(json.dumps(completed) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(BlindReviewError, "reviewer_id"):
                merge_human_adjudication(
                    audit_ref=audit_ref,
                    review_target=review_target,
                    human_completed=human,
                    output=root / "merged.jsonl",
                )
            completed["reviewer_id"] = "dual-blind-panel-v1"
            completed["reason_code"] = "not-in-rubric"
            human.write_text(json.dumps(completed) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(BlindReviewError, "merged human adjudication"):
                merge_human_adjudication(
                    audit_ref=audit_ref,
                    review_target=review_target,
                    human_completed=human,
                    output=root / "merged.jsonl",
                )
            completed["reason_code"] = "valid-independent-label-combination"
            human.write_text(json.dumps(completed) + "\n", encoding="utf-8")
            report = merge_human_adjudication(
                audit_ref=audit_ref,
                review_target=review_target,
                human_completed=human,
                output=root / "merged.jsonl",
            )
            self.assertEqual(report["row_count"], 1)


if __name__ == "__main__":
    unittest.main()
