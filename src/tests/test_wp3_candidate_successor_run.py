from __future__ import annotations

import copy
import json
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPOSITORY_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from build_lex.terminology_candidate_successor_run import (  # noqa: E402
    DEVELOPMENT_ARTIFACT_MARKING,
    PrivateResponseVault,
    ProviderCallError,
    ProviderCheckpoint,
    SuccessorRunError,
    TransportResponse,
    build_prompt_contract_from_current_sources,
    build_successor_plan_from_artifacts,
    build_verified_pricing_dependency,
    _assemble_successor_plan,
    _live_preflight_report,
    _metric,
    _normalizer_for_slot,
    _retention_decision,
    _successor_implementation_contract,
    _validate_formal_results_against_slots_and_vault,
    execute_provider_slot,
    load_provider_credentials,
    parse_chat_completion,
    preflight_offline,
    run_live_preflight,
    strict_json_object,
    validate_full_run_authorization,
    validate_development_evaluation,
    validate_successor_plan,
)
from data.training_artifacts import canonical_sha256  # noqa: E402
from build_lex.terminology_candidate_run_plan import CURRENT_PLAN_ID  # noqa: E402


def fixture_plan() -> dict:
    current = {
        "plan_revision": "wp3-s21b-provider-plan/v2",
        "plan_id": CURRENT_PLAN_ID,
        "total_logical_model_slot_count": 2544,
        "frame_dependency": {
            "artifact_kind": "wp3-s21-development-frame",
            "artifact_id": "wp3devframe-" + "2" * 64,
            "payload_manifest_sha256": "3" * 64,
            "case_count": 424,
            "task_projection_canonical_sha256": "4" * 64,
        },
        "preserved_raw_gold": {
            "artifact_kind": "wp3-s21-locked-raw-gold",
            "artifact_id": "wp3rawgold-" + "5" * 64,
            "payload_manifest_sha256": "6" * 64,
            "run_must_not_read": True,
            "first_join_stage": "development-evaluation/v1",
        },
    }
    g3 = {
        "artifact_kind": "wp3-s21-g3-development-run",
        "artifact_id": "wp3g3run-" + "7" * 64,
        "payload_manifest_sha256": "8" * 64,
        "profile_id": "wp3-g3-profile/full-v2",
        "profile_sha256": "9" * 64,
        "reference_id": "wp3g3ref-abc",
        "reference_sha256": "a" * 64,
        "romanizer_backend_id": (
            "pypinyin-0.55.0/phrase-aware-one-best-normal-ascii/v1"
        ),
        "case_count": 424,
        "enabled_families": [
            "mixed_script",
            "unicode_nfkc",
            "emoji",
            "known_variant",
            "pinyin_initials",
            "phonetic_variant",
            "separator_insertion",
            "orthographic_variant",
        ],
        "terminal_failure_count": 0,
        "offset_replay_count": 12,
        "offset_occurrence_count": 12,
        "offset_replay_rate": 1,
        "complete": True,
    }
    pricing = build_verified_pricing_dependency(
        evidence_path=(
            "src/tests/fixtures/wp3_provider_pricing/evidence.json"
        ),
        workspace_root=REPOSITORY_ROOT,
    )
    return _assemble_successor_plan(
        current_plan=current,
        prompt_contract=build_prompt_contract_from_current_sources(
            workspace_root=REPOSITORY_ROOT
        ),
        implementation_contract=_successor_implementation_contract(
            workspace_root=REPOSITORY_ROOT
        ),
        g3_completion_dependency=g3,
        pricing=pricing,
    )


class FakeTransport:
    def __init__(self, responses: list[TransportResponse]) -> None:
        self.responses = list(responses)
        self.calls = 0

    def post_json(self, **_kwargs):
        self.calls += 1
        return self.responses.pop(0)


def fixture_request(provider_id: str = "glm_flash") -> dict:
    return {
        "model": (
            "glm-5.3-flash"
            if provider_id == "glm_flash"
            else "deepseek-v4-flash"
        ),
        "messages": [{"role": "user", "content": "synthetic"}],
        "temperature": 1,
        "top_p": 0.95,
        "max_tokens": 4096,
        "stream": False,
        "response_format": {"type": "json_object"},
        "thinking": {"type": "enabled"},
        "reasoning_effort": "low",
    }


def fixture_success_response(
    provider_id: str = "glm_flash", *, prompt_tokens: int = 1, completion_tokens: int = 1
) -> TransportResponse:
    model = (
        "glm-5.3-flash"
        if provider_id == "glm_flash"
        else "deepseek-v4-flash"
    )
    body = json.dumps(
        {
            "id": f"request-{provider_id}",
            "model": model,
            "choices": [
                {"finish_reason": "stop", "message": {"content": "{}"}}
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        },
        separators=(",", ":"),
    ).encode()
    return TransportResponse(200, {"content-type": "application/json"}, body)


def fixture_empty_normalizer(*_args) -> dict:
    return {
        "state": "success_empty",
        "selection_truncated": False,
        "record_reason": "synthetic-empty",
        "observations": [],
    }


class SuccessorRunTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.plan = fixture_plan()

    def test_plan_and_offline_six_call_grid_are_content_bound(self) -> None:
        self.assertEqual(validate_successor_plan(self.plan), self.plan)
        report = preflight_offline(self.plan)
        self.assertEqual(report["call_count"], 6)
        self.assertEqual(report["network_calls_performed"], 0)
        changed = copy.deepcopy(self.plan)
        changed["resources"]["max_attempts_global"] = 3001
        with self.assertRaises(SuccessorRunError):
            validate_successor_plan(changed)

    def test_plan_rejects_claimed_or_tampered_pricing_without_snapshot_replay(self) -> None:
        claimed = copy.deepcopy(self.plan)
        claimed["pricing"] = copy.deepcopy(
            self.plan["pricing"]["verification_bundle"]["pricing_projection"]
        )
        claimed["plan_id"] = "wp3runplan-" + canonical_sha256(
            {key: value for key, value in claimed.items() if key != "plan_id"}
        )
        with self.assertRaisesRegex(SuccessorRunError, "dependency fields"):
            validate_successor_plan(claimed)

        tampered = copy.deepcopy(self.plan)
        tampered["pricing"]["verification_bundle"]["pricing_projection"][
            "provider_rates"
        ]["glm_flash"]["input_rmb_per_million"] = "1"
        tampered["plan_id"] = "wp3runplan-" + canonical_sha256(
            {key: value for key, value in tampered.items() if key != "plan_id"}
        )
        with self.assertRaisesRegex(SuccessorRunError, "failed verification"):
            validate_successor_plan(tampered)

    def test_plan_replays_exact_frozen_predecessor_and_current_code(self) -> None:
        current_path = (
            REPOSITORY_ROOT
            / "config/stage1/wp3_candidate_current_run_plan_v2.json"
        )
        current_schema = (
            REPOSITORY_ROOT
            / "schemas/wp3_candidate_current_run_plan_v1.schema.json"
        )
        with mock.patch(
            "build_lex.terminology_candidate_successor_run."
            "validate_g3_run_ref_for_planning",
            return_value=self.plan["g3_completion_dependency"],
        ):
            rebuilt = build_successor_plan_from_artifacts(
                current_plan_path=current_path,
                current_plan_schema_path=current_schema,
                g3_run_ref=Path("unused-by-mock"),
                pricing_evidence_path=(
                    REPOSITORY_ROOT
                    / "src/tests/fixtures/wp3_provider_pricing/evidence.json"
                ),
                workspace_root=REPOSITORY_ROOT,
            )
        self.assertEqual(rebuilt["supersedes_plan_id"], CURRENT_PLAN_ID)
        self.assertEqual(
            rebuilt["implementation_contract"],
            _successor_implementation_contract(workspace_root=REPOSITORY_ROOT),
        )

        with tempfile.TemporaryDirectory() as directory:
            tampered_path = Path(directory) / "current-plan.json"
            current = json.loads(current_path.read_text(encoding="utf-8"))
            current["total_logical_model_slot_count"] = 2543
            tampered_path.write_text(
                json.dumps(current, ensure_ascii=False), encoding="utf-8"
            )
            with mock.patch(
                "build_lex.terminology_candidate_successor_run."
                "validate_g3_run_ref_for_planning",
                return_value=self.plan["g3_completion_dependency"],
            ), self.assertRaisesRegex(
                SuccessorRunError, "frozen current provider plan failed validation"
            ):
                build_successor_plan_from_artifacts(
                    current_plan_path=tampered_path,
                    current_plan_schema_path=current_schema,
                    g3_run_ref=Path("unused-by-mock"),
                    pricing_evidence_path=(
                        REPOSITORY_ROOT
                        / "src/tests/fixtures/wp3_provider_pricing/evidence.json"
                    ),
                    workspace_root=REPOSITORY_ROOT,
                )

    def test_plan_validation_recomputes_prompt_schema_and_code_hashes(self) -> None:
        for section, key, field in (
            ("prompt_contract", "g1_surface_decode", "system_prompt_sha256"),
            ("prompt_contract", "g2_direct_mention", "response_schema_sha256"),
            ("implementation_contract", "generator_v2", "sha256"),
            ("implementation_contract", "successor_runner", "sha256"),
        ):
            with self.subTest(section=section, key=key, field=field):
                changed = copy.deepcopy(self.plan)
                changed[section][key][field] = "0" * 64
                identity = {
                    name: value
                    for name, value in changed.items()
                    if name != "plan_id"
                }
                changed["plan_id"] = "wp3runplan-" + canonical_sha256(identity)
                with self.assertRaises(SuccessorRunError):
                    validate_successor_plan(changed)

    def test_public_result_replays_from_exact_private_slot_response(self) -> None:
        task = {
            "task_id": "case-1",
            "blind_alias": "CASE-1",
            "content": "普通文本",
        }
        request_sha = "d" * 64
        binding = {
            "slot_id": "slot-1",
            "task_id": "case-1",
            "generator": "g2",
            "variant": "direct_mention",
            "provider_id": "glm_flash",
            "request_sha256": request_sha,
        }
        completion_content = json.dumps(
            {
                "mentions": [],
                "selection_truncated": False,
                "record_reason": "没有候选",
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
        body = json.dumps(
            {
                "id": "request-1",
                "model": "glm-5.3-flash",
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": completion_content},
                    }
                ],
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                },
            },
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        headers = {
            "content-type": "application/json",
            "x-request-id": "request-1",
        }
        normalizer = _normalizer_for_slot(
            task=task,
            generator="g2",
            variant="direct_mention",
            provider_id="glm_flash",
            prompt_version=self.plan["prompt_contract"]["g2_direct_mention"][
                "version"
            ],
        )
        result, _ = parse_chat_completion(
            TransportResponse(200, headers, body),
            plan=self.plan,
            provider_id="glm_flash",
            preflight=False,
            normalize=normalizer,
        )
        result.update(
            {
                "slot_id": "slot-1",
                "request_sha256": request_sha,
                "task_id": "case-1",
                "generator": "g2",
                "variant": "direct_mention",
            }
        )
        private = {
            "manifest": {"plan_id": self.plan["plan_id"]},
            "attempts": [
                {
                    "slot_id": "slot-1",
                    "provider_id": "glm_flash",
                    "attempt_no": 1,
                    "request_sha256": request_sha,
                    "response_body": body.decode("utf-8"),
                    "safe_headers": headers,
                    "error_code": None,
                }
            ],
        }
        with mock.patch(
            "build_lex.terminology_candidate_successor_run."
            "_expected_formal_slot_bindings",
            return_value={"slot-1": binding},
        ):
            _validate_formal_results_against_slots_and_vault(
                plan=self.plan,
                results=[result],
                public_tasks=[task],
                private=private,
            )
            changed = copy.deepcopy(result)
            changed["task_id"] = "case-other"
            with self.assertRaisesRegex(SuccessorRunError, "slot metadata"):
                _validate_formal_results_against_slots_and_vault(
                    plan=self.plan,
                    results=[changed],
                    public_tasks=[task],
                    private=private,
                )
            changed = copy.deepcopy(result)
            changed["record_reason"] = "伪造的公开归一化结果"
            with self.assertRaisesRegex(SuccessorRunError, "does not replay"):
                _validate_formal_results_against_slots_and_vault(
                    plan=self.plan,
                    results=[changed],
                    public_tasks=[task],
                    private=private,
                )

    def test_strict_json_rejects_padding_duplicate_and_nonfinite(self) -> None:
        self.assertEqual(strict_json_object('{"x":1}'), {"x": 1})
        for value in (' {"x":1}', '{"x":1,"x":2}', '{"x":NaN}', '```{"x":1}```'):
            with self.subTest(value=value), self.assertRaises(SuccessorRunError):
                strict_json_object(value)

    def test_checkpoint_marks_crash_reservation_ambiguous_and_counts_budget(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "checkpoint"
            checkpoint = ProviderCheckpoint.create(
                root, plan=self.plan, mode="formal"
            )
            header = json.loads(
                (root / "checkpoint.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                header["implementation_contract_sha256"],
                canonical_sha256(self.plan["implementation_contract"]),
            )
            reservation = checkpoint.reserve_attempt(
                slot_id="slot-1",
                provider_id="glm_flash",
                request_sha256="d" * 64,
                estimated_input_tokens=10,
            )
            self.assertTrue(reservation.attempt_id.startswith("wp3attempt-"))
            checkpoint.close()
            with ProviderCheckpoint.resume(
                root, plan=self.plan, mode="formal"
            ) as resumed:
                summary = resumed.summary()
                self.assertEqual(summary["attempt_status_counts"], {"ambiguous": 1})
                self.assertEqual(summary["budget"]["global_attempts"], 1)
                with self.assertRaisesRegex(SuccessorRunError, "request changed"):
                    resumed.reserve_attempt(
                        slot_id="slot-1",
                        provider_id="glm_flash",
                        request_sha256="e" * 64,
                        estimated_input_tokens=10,
                    )

    def test_checkpoint_repairs_only_the_single_event_head_crash_window(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "checkpoint"
            checkpoint = ProviderCheckpoint.create(root, plan=self.plan, mode="formal")
            checkpoint.reserve_attempt(
                slot_id="slot-head-window",
                provider_id="glm_flash",
                request_sha256="f" * 64,
                estimated_input_tokens=10,
            )
            checkpoint.close()
            (root / "HEAD.json").write_text(
                json.dumps({"sequence": 0, "event_sha256": "0" * 64}),
                encoding="utf-8",
            )
            with ProviderCheckpoint.resume(root, plan=self.plan, mode="formal") as resumed:
                # Resume first repairs HEAD to event 1, then durably classifies
                # the orphaned reservation as ambiguous in event 2.
                self.assertEqual(resumed.summary()["head"]["sequence"], 2)
                self.assertEqual(
                    resumed.summary()["attempt_status_counts"], {"ambiguous": 1}
                )

            (root / "HEAD.json").write_text(
                json.dumps({"sequence": 0, "event_sha256": "e" * 64}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(SuccessorRunError, "HEAD differs"):
                ProviderCheckpoint.resume(root, plan=self.plan, mode="formal")

    def test_successful_slot_resume_skips_transport(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with ProviderCheckpoint.create(
                root / "cp", plan=self.plan, mode="formal"
            ) as checkpoint:
                vault = PrivateResponseVault.create(
                    root / "vault", plan_id=self.plan["plan_id"]
                )
                first_transport = FakeTransport([fixture_success_response()])
                first = execute_provider_slot(
                    plan=self.plan,
                    checkpoint=checkpoint,
                    vault=vault,
                    transport=first_transport,
                    credentials={"glm_flash": "not-a-real-secret"},
                    slot_id="slot-success-resume",
                    provider_id="glm_flash",
                    request=fixture_request(),
                    estimated_input_tokens=4,
                    normalize=fixture_empty_normalizer,
                )
                self.assertEqual(first["execution_disposition"], "executed")
                resumed_transport = FakeTransport([])
                resumed = execute_provider_slot(
                    plan=self.plan,
                    checkpoint=checkpoint,
                    vault=vault,
                    transport=resumed_transport,
                    credentials={"glm_flash": "not-a-real-secret"},
                    slot_id="slot-success-resume",
                    provider_id="glm_flash",
                    request=fixture_request(),
                    estimated_input_tokens=4,
                    normalize=fixture_empty_normalizer,
                )
                self.assertEqual(
                    resumed["execution_disposition"], "skipped_success"
                )
                self.assertEqual(resumed_transport.calls, 0)

    def test_model_output_failure_is_terminal_and_not_retried(self) -> None:
        body = json.dumps(
            {
                "id": "request-1",
                "model": "glm-5.3-flash",
                "choices": [
                    {"finish_reason": "length", "message": {"content": "{}"}}
                ],
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                },
            },
            separators=(",", ":"),
        ).encode()
        response = TransportResponse(200, {"content-type": "application/json"}, body)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with ProviderCheckpoint.create(
                root / "cp", plan=self.plan, mode="formal"
            ) as checkpoint:
                vault = PrivateResponseVault.create(
                    root / "vault", plan_id=self.plan["plan_id"]
                )
                transport = FakeTransport([response])
                request = fixture_request()
                with self.assertRaises(ProviderCallError) as caught:
                    execute_provider_slot(
                        plan=self.plan,
                        checkpoint=checkpoint,
                        vault=vault,
                        transport=transport,
                        credentials={"glm_flash": "not-a-real-secret"},
                        slot_id="slot-output-failure",
                        provider_id="glm_flash",
                        request=request,
                        estimated_input_tokens=4,
                        normalize=lambda *_args: {},
                    )
                self.assertFalse(caught.exception.retryable)
                self.assertEqual(transport.calls, 1)
                resumed_transport = FakeTransport([])
                with self.assertRaises(ProviderCallError) as resumed_caught:
                    execute_provider_slot(
                        plan=self.plan,
                        checkpoint=checkpoint,
                        vault=vault,
                        transport=resumed_transport,
                        credentials={"glm_flash": "not-a-real-secret"},
                        slot_id="slot-output-failure",
                        provider_id="glm_flash",
                        request=request,
                        estimated_input_tokens=4,
                        normalize=lambda *_args: {},
                    )
                self.assertEqual(
                    resumed_caught.exception.code,
                    "model-protocol-or-schema-failure",
                )
                self.assertEqual(resumed_transport.calls, 0)

    def test_retryable_failures_keep_worst_case_budget_reservations(self) -> None:
        response = TransportResponse(
            500, {"content-type": "application/json"}, b'{"error":"busy"}'
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with ProviderCheckpoint.create(
                root / "cp", plan=self.plan, mode="formal"
            ) as checkpoint:
                vault = PrivateResponseVault.create(
                    root / "vault", plan_id=self.plan["plan_id"]
                )
                transport = FakeTransport([response, response])
                request = fixture_request()
                with self.assertRaises(ProviderCallError):
                    execute_provider_slot(
                        plan=self.plan,
                        checkpoint=checkpoint,
                        vault=vault,
                        transport=transport,
                        credentials={"glm_flash": "not-a-real-secret"},
                        slot_id="slot-retry-failure",
                        provider_id="glm_flash",
                        request=request,
                        estimated_input_tokens=4,
                        normalize=lambda *_args: {},
                    )
                budget = checkpoint.summary()["budget"]
                self.assertEqual(transport.calls, 2)
                self.assertEqual(budget["global_attempts"], 2)
                self.assertEqual(budget["reserved_output_tokens"], 8192)
                self.assertGreater(budget["reserved_cost_micrormb"], 0)

    def test_formal_resume_can_use_one_remaining_retry_after_retryable_failure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            request = fixture_request()
            with ProviderCheckpoint.create(
                root / "cp", plan=self.plan, mode="formal"
            ) as checkpoint:
                reservation = checkpoint.reserve_attempt(
                    slot_id="slot-retry-resume",
                    provider_id="glm_flash",
                    request_sha256=canonical_sha256(request),
                    estimated_input_tokens=len(json.dumps(request)) + 4096,
                )
                checkpoint.finish_attempt(
                    reservation,
                    outcome="terminal_failure",
                    error_code="retryable-http-status",
                    retryable=True,
                )
                vault = PrivateResponseVault.create(
                    root / "vault", plan_id=self.plan["plan_id"]
                )
                transport = FakeTransport([fixture_success_response()])
                result = execute_provider_slot(
                    plan=self.plan,
                    checkpoint=checkpoint,
                    vault=vault,
                    transport=transport,
                    credentials={"glm_flash": "not-a-real-secret"},
                    slot_id="slot-retry-resume",
                    provider_id="glm_flash",
                    request=request,
                    estimated_input_tokens=4,
                    normalize=fixture_empty_normalizer,
                )
                self.assertEqual(result["execution_disposition"], "executed")
                self.assertEqual(transport.calls, 1)
                self.assertEqual(
                    checkpoint.summary()["budget"]["global_attempts"], 2
                )

    def test_preflight_retryable_failure_is_never_sent_twice(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            request = fixture_request()
            with ProviderCheckpoint.create(
                root / "cp", plan=self.plan, mode="synthetic-preflight"
            ) as checkpoint:
                reservation = checkpoint.reserve_attempt(
                    slot_id="preflight:surface_decode:glm_flash",
                    provider_id="glm_flash",
                    request_sha256=canonical_sha256(request),
                    estimated_input_tokens=len(json.dumps(request)) + 4096,
                )
                checkpoint.finish_attempt(
                    reservation,
                    outcome="terminal_failure",
                    error_code="retryable-http-status",
                    retryable=True,
                )
                vault = PrivateResponseVault.create(
                    root / "vault", plan_id=self.plan["plan_id"]
                )
                transport = FakeTransport([])
                with self.assertRaisesRegex(
                    ProviderCallError, "slot-attempt-budget-exhausted"
                ):
                    execute_provider_slot(
                        plan=self.plan,
                        checkpoint=checkpoint,
                        vault=vault,
                        transport=transport,
                        credentials={"glm_flash": "not-a-real-secret"},
                        slot_id="preflight:surface_decode:glm_flash",
                        provider_id="glm_flash",
                        request=request,
                        estimated_input_tokens=4,
                        normalize=fixture_empty_normalizer,
                        preflight=True,
                    )
                self.assertEqual(transport.calls, 0)

    def test_live_preflight_report_has_alias_format_usage_cost_and_projection(self) -> None:
        rows = []
        for provider_id in ("glm_flash", "deepseek_flash"):
            alias = self.plan["providers"][provider_id]["requested_model"]
            for index in range(3):
                rows.append(
                    {
                        "provider_id": provider_id,
                        "prompt_key": f"prompt-{index}",
                        "state": "success_empty",
                        "returned_model": alias,
                        "usage": {
                            "prompt_tokens": 10,
                            "completion_tokens": 2,
                            "total_tokens": 12,
                        },
                        "execution_disposition": (
                            "skipped_success" if index == 0 else "executed"
                        ),
                    }
                )
        report = _live_preflight_report(
            plan=self.plan,
            results=rows,
            checkpoint_summary={"budget": {"global_attempts": 6}},
        )
        self.assertEqual(report["call_count"], 6)
        self.assertTrue(report["protocol_gate_passed"])
        self.assertEqual(report["calls_executed_this_invocation"], 4)
        self.assertEqual(report["successful_slots_skipped_this_invocation"], 2)
        self.assertEqual(report["format_success"]["overall"]["rate"], 1)
        self.assertEqual(report["usage"]["overall"]["total_tokens"], 72)
        self.assertEqual(
            report["returned_alias_counts"]["by_provider"]["glm_flash"],
            {"glm-5.3-flash": 3},
        )
        projection = report["formal_2544_slot_linear_projection"]
        self.assertEqual(projection["slot_count"], 2544)
        self.assertEqual(
            projection["usage"]["overall"]["completion_tokens"], 5088
        )
        self.assertEqual(
            projection["frozen_budget_cap"],
            {"aggregate_output_tokens": 4_000_000, "estimated_cost_rmb": 100},
        )
        self.assertGreater(report["actual_estimated_cost"]["micrormb"], 0)

    def test_live_preflight_report_keeps_protocol_failures_visible(self) -> None:
        rows = [
            {
                "provider_id": "glm_flash",
                "prompt_key": f"glm-{index}",
                "state": "success_empty",
                "returned_model": "glm-5.3-flash",
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 2,
                    "total_tokens": 12,
                },
                "execution_disposition": "executed",
            }
            for index in range(3)
        ] + [
            {
                "provider_id": "deepseek_flash",
                "prompt_key": f"deepseek-{index}",
                "state": "terminal_failure",
                "returned_model": None,
                "usage": None,
                "error_code": "model-protocol-or-schema-failure",
                "execution_disposition": "executed_failure",
            }
            for index in range(3)
        ]
        report = _live_preflight_report(
            plan=self.plan,
            results=rows,
            checkpoint_summary={"budget": {"global_attempts": 6}},
        )
        self.assertEqual(report["format_success"]["overall"]["rate"], 0.5)
        self.assertFalse(report["protocol_gate_passed"])
        self.assertEqual(
            report["format_success"]["by_provider"]["deepseek_flash"]["rate"],
            0,
        )
        self.assertNotIn("None", report["returned_alias_counts"]["overall"])
        self.assertEqual(report["returned_alias_counts"]["unknown_slot_count"], 3)
        self.assertEqual(report["usage_missing_slot_count"], 3)
        projection = report["formal_2544_slot_linear_projection"]
        self.assertEqual(
            projection["status"], "inconclusive-missing-provider-usage"
        )
        self.assertIsNone(projection["estimated_cost"])

    def test_live_preflight_resume_reuses_all_six_successes_without_calls(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with ProviderCheckpoint.create(
                root / "cp", plan=self.plan, mode="synthetic-preflight"
            ) as checkpoint:
                vault = PrivateResponseVault.create(
                    root / "vault", plan_id=self.plan["plan_id"]
                )
                first_transport = FakeTransport(
                    [
                        fixture_success_response(provider_id)
                        for _variant in range(3)
                        for provider_id in ("glm_flash", "deepseek_flash")
                    ]
                )
                with mock.patch(
                    "build_lex.terminology_candidate_successor_run."
                    "load_bound_g3_run",
                    return_value={},
                ), mock.patch(
                    "build_lex.terminology_candidate_successor_run."
                    "_normalizer_for_slot",
                    return_value=fixture_empty_normalizer,
                ):
                    first = run_live_preflight(
                        plan=self.plan,
                        checkpoint=checkpoint,
                        vault=vault,
                        transport=first_transport,
                        credentials={
                            "glm_flash": "not-a-real-glm-secret",
                            "deepseek_flash": "not-a-real-deepseek-secret",
                        },
                        g3_run_ref=Path("unused-by-mock"),
                    )
                    resumed_transport = FakeTransport([])
                    resumed = run_live_preflight(
                        plan=self.plan,
                        checkpoint=checkpoint,
                        vault=vault,
                        transport=resumed_transport,
                        credentials={
                            "glm_flash": "not-a-real-glm-secret",
                            "deepseek_flash": "not-a-real-deepseek-secret",
                        },
                        g3_run_ref=Path("unused-by-mock"),
                    )
                self.assertTrue(first["protocol_gate_passed"])
                self.assertEqual(first_transport.calls, 6)
                self.assertTrue(resumed["protocol_gate_passed"])
                self.assertEqual(resumed["calls_executed_this_invocation"], 0)
                self.assertEqual(
                    resumed["successful_slots_skipped_this_invocation"], 6
                )
                self.assertEqual(resumed_transport.calls, 0)

    def test_live_preflight_records_failure_and_continues_remaining_slots(self) -> None:
        failure_body = json.dumps(
            {
                "id": "failed-request",
                "model": "glm-5.3-flash",
                "choices": [
                    {"finish_reason": "length", "message": {"content": "{}"}}
                ],
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                },
            },
            separators=(",", ":"),
        ).encode()
        responses = [
            TransportResponse(
                200, {"content-type": "application/json"}, failure_body
            ),
            fixture_success_response("deepseek_flash"),
            fixture_success_response("glm_flash"),
            fixture_success_response("deepseek_flash"),
            fixture_success_response("glm_flash"),
            fixture_success_response("deepseek_flash"),
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with ProviderCheckpoint.create(
                root / "cp", plan=self.plan, mode="synthetic-preflight"
            ) as checkpoint:
                vault = PrivateResponseVault.create(
                    root / "vault", plan_id=self.plan["plan_id"]
                )
                transport = FakeTransport(responses)
                with mock.patch(
                    "build_lex.terminology_candidate_successor_run."
                    "load_bound_g3_run",
                    return_value={},
                ), mock.patch(
                    "build_lex.terminology_candidate_successor_run."
                    "_normalizer_for_slot",
                    return_value=fixture_empty_normalizer,
                ):
                    report = run_live_preflight(
                        plan=self.plan,
                        checkpoint=checkpoint,
                        vault=vault,
                        transport=transport,
                        credentials={
                            "glm_flash": "not-a-real-glm-secret",
                            "deepseek_flash": "not-a-real-deepseek-secret",
                        },
                        g3_run_ref=Path("unused-by-mock"),
                    )
                self.assertEqual(transport.calls, 6)
                self.assertFalse(report["protocol_gate_passed"])
                self.assertEqual(
                    report["format_success"]["overall"]["success_count"], 5
                )
                self.assertEqual(
                    report["checkpoint"]["budget"]["global_attempts"], 6
                )

    def test_completed_artifact_marking_is_immutable_but_nonsealed(self) -> None:
        self.assertEqual(
            DEVELOPMENT_ARTIFACT_MARKING,
            {
                "scientific_eligible": False,
                "sealed": False,
                "immutable": True,
            },
        )

    def test_evaluation_validator_replays_metric_and_retention_arithmetic(self) -> None:
        gold = {("case", index, index + 1, "x", 1) for index in range(10)}
        empty_metric = _metric(set(), gold)
        metric_names = {
            "g1:surface_decode:glm_flash",
            "g1:surface_decode:deepseek_flash",
            "g1:lexical_pragmatic:glm_flash",
            "g1:lexical_pragmatic:deepseek_flash",
            "g2:direct_mention:glm_flash",
            "g2:direct_mention:deepseek_flash",
            "g3:overall",
            "g1:glm_union",
            "g1:deepseek_union",
            "g2:provider_union",
            "baseline_union",
            "final_union",
            *(
                f"g3:{family}"
                for family in self.plan["g3_completion_dependency"][
                    "enabled_families"
                ]
            ),
        }
        model_groups = {
            f"{generator}:{variant}:{provider}"
            for generator, variant in (
                ("g1", "surface_decode"),
                ("g1", "lexical_pragmatic"),
                ("g2", "direct_mention"),
            )
            for provider in ("glm_flash", "deepseek_flash")
        }
        sources = {"g1_glm", "g1_deepseek", "g2_glm", "g2_deepseek", "g3"}
        report = {
            "schema_version": "wp3-s21-development-evaluation/v1",
            "scope": "development-only",
            "scientific_eligible": False,
            "sealed_benchmark": False,
            "run_id": "wp3run-" + "1" * 64,
            "raw_gold_id": "wp3rawgold-" + "2" * 64,
            "gold_mention_count": len(gold),
            "metrics": {
                name: copy.deepcopy(empty_metric) for name in metric_names
            },
            "truncated_slot_counts": {},
            "truncation_rates": {
                name: {
                    "truncated_slot_count": 0,
                    "slot_count": 424,
                    "rate": 0.0,
                }
                for name in model_groups
            },
            "unique_true_positives": {name: 0 for name in sources},
            "leave_one_source_out": {
                name: copy.deepcopy(empty_metric) for name in sources
            },
            "boundary_errors": {
                "too_wide": 0,
                "too_narrow": 0,
                "fragment": 0,
                "substring": 0,
                "wrong_occurrence": 0,
            },
            "g1_deepseek_retention": _retention_decision(
                gold_count=len(gold),
                baseline=set(),
                added=set(),
                gold=gold,
                complete=True,
            ),
            "completion": {
                "terminal_slot_count": 2544,
                "terminal_failure_count": 0,
                "g3_complete": True,
                "offset_replay_rate": 1,
            },
        }
        self.assertEqual(validate_development_evaluation(report), report)
        changed = copy.deepcopy(report)
        changed["metrics"]["final_union"]["precision"] = 0.1
        with self.assertRaisesRegex(SuccessorRunError, "arithmetic"):
            validate_development_evaluation(changed)

    def test_authorization_receipt_is_plan_and_budget_bound(self) -> None:
        identity = {
            "schema_version": "wp3-s21-full-run-authorization/v1",
            "plan_id": self.plan["plan_id"],
            "authorized_by": "reviewer",
            "authorized_at": "2026-08-30T00:00:00Z",
            "authorization_scope": "exact-2544-development-slot-grid-only/v1",
            "accepted_resource_contract_sha256": canonical_sha256(
                self.plan["resources"]
            ),
            "accepted_pricing_sha256": canonical_sha256(self.plan["pricing"]),
            "authorized": True,
        }
        receipt = {
            **identity,
            "receipt_id": "wp3runauth-" + canonical_sha256(identity),
        }
        self.assertEqual(
            validate_full_run_authorization(receipt, plan=self.plan), receipt
        )
        receipt["accepted_pricing_sha256"] = "0" * 64
        with self.assertRaises(SuccessorRunError):
            validate_full_run_authorization(receipt, plan=self.plan)

    def test_credentials_require_exact_frozen_base_and_git_ignored_env(self) -> None:
        values = load_provider_credentials(
            self.plan,
            workspace_root=REPOSITORY_ROOT,
            environ={"GLM_API_KEY": "g", "DEEPSEEK_API_KEY": "d"},
        )
        self.assertEqual(set(values), {"glm_flash", "deepseek_flash"})
        with self.assertRaisesRegex(SuccessorRunError, "override differs"):
            load_provider_credentials(
                self.plan,
                workspace_root=REPOSITORY_ROOT,
                environ={
                    "GLM_API_KEY": "g",
                    "DEEPSEEK_API_KEY": "d",
                    "WP3_GLM_API_BASE": "https://open.bigmodel.cn/api/paas/v4/",
                },
            )

    def test_cli_exposes_all_fixed_commands(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                str(REPOSITORY_ROOT / "scripts/stage1/wp3_candidate_successor_run.py"),
                "--help",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        for command in (
            "build-run-plan",
            "preflight-offline",
            "run-live-preflight",
            "status",
            "run",
            "finalize-run",
            "validate-run",
            "evaluate-dev",
            "validate-dev-evaluation",
        ):
            self.assertIn(command, result.stdout)


if __name__ == "__main__":
    unittest.main()
