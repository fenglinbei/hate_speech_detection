"""Frozen provider-plan overlay for the WP3 S2.1b current generator pilot.

The legacy frame, generator implementation, S2.1b v1 protocol, and locked raw
gold remain immutable.  This module validates the independent provider-plan v2,
renders provider-equivalent request bodies, and constructs the logical slot
grid.  It performs no network or model calls and publishes no run artifact.
"""

from __future__ import annotations

import copy
import hashlib
from collections import Counter
from collections.abc import Mapping, Sequence
from fractions import Fraction
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from build_lex.terminology_candidate_generators import (
    G1_LEXICAL_SYSTEM_PROMPT,
    G1_PROMPT_VERSIONS,
    G1_SURFACE_SYSTEM_PROMPT,
    G2_PROMPT_VERSION,
    G2_SYSTEM_PROMPT,
    CandidateGeneratorError,
    build_g1_request,
    build_g2_request,
)
from data.training_artifacts import (
    TrainingArtifactError,
    canonical_sha256,
    load_json,
    resolve_locator_ref,
    sha256_file,
    validate_json_schema,
)


PLAN_SCHEMA_VERSION = "wp3-s21-current-run-provider-plan/v1"
PLAN_REVISION = "wp3-s21b-provider-plan/v2"
PLAN_ID_PREFIX = "wp3runplan-"
CURRENT_PLAN_ID = (
    "wp3runplan-"
    "7c4b4c8bda5fd85a3ce9f26cae6069aa4bc8bb52497a2d5a6fb97194371ea90f"
)
PLAN_SCHEMA_ID = (
    "https://local.invalid/schemas/wp3_candidate_current_run_plan_v1.schema.json"
)
PLAN_SCHEMA_SHA256 = (
    "5e6b841e6f3755a2ca9a18ad95c57fdb2ea4ebb1696f9f4ff68aab7fe04157ee"
)
BASE_PROTOCOL_REVISION = "wp3-s21b-current-generator-diagnostic/v1"
BASE_PROTOCOL_CANONICAL_SHA256 = (
    "c3c1442c02567e15aac2a664b5ffeb2b5fb7db8b48b691997e5c16d71c51ce4b"
)
FRAME_ID = (
    "wp3devframe-"
    "0a28b5b91dff61db5d087a9dc64c095b01bf85fc64552fed5db05ea17fb47dab"
)
FRAME_PAYLOAD_SHA256 = (
    "8610262616ad993c7414b290e02daa74c4c8123cee6f4e6afd66cab5fe52415a"
)
FRAME_CASES_CANONICAL_SHA256 = (
    "392aa31e8fbc0c49b10cfe07fa9ac0f2fe4a1dcee0995afc923658d63678ce0b"
)
FRAME_CASE_ID_SET_CANONICAL_SHA256 = (
    "ae867eb3ec0fb6f3fed876633409d840146318bfa9bd8736f6d95099d237fbd6"
)
TASK_PROJECTION_CANONICAL_SHA256 = (
    "3660dbfdb518e469d62f69fabdfcb34f983269f3fb54e3bd1529419b3bd5b90d"
)
RAW_GOLD_ID = (
    "wp3rawgold-"
    "41f16aabde6fd253485d853319ad949c1507a4caf29ba0f658e7ea28a3d3a7b0"
)
RAW_GOLD_PAYLOAD_SHA256 = (
    "35b655724079678cc5831a944aed6294c179d74470f4d76c22b7af79ab500364"
)
LEGACY_IMPLEMENTATION_PATH = "src/build_lex/terminology_candidate_generators.py"
LEGACY_IMPLEMENTATION_SHA256 = (
    "66a11209e615bcd6be90977276665118b7f155f32ecf1388b4ae08ea7b37f1bf"
)
EXPECTED_CASE_COUNT = 424

PROMPT_CONTRACT = {
    "g1_surface_decode": {
        "version": "wp3-g1-surface-decode/v1",
        "sha256": "dfba0c116e76dea80efd98510194a5e038d46b19b9768e46bb03d32ece1be1ec",
    },
    "g1_lexical_pragmatic": {
        "version": "wp3-g1-lexical-pragmatic/v1",
        "sha256": "91b3d6552e736bf4fd6f8c8e7e217944e77cbeb25c84c6b938bb07c715bf47ec",
    },
    "g2_direct_mention": {
        "version": "wp3-g2-direct-mention/v1",
        "sha256": "ff7af4a9bc6e7bad06f965f5d4f0a95bee3ceb8fa37570055d38e2f64f23e459",
    },
}

REQUEST_PROFILE = {
    "api_shape": "openai-compatible-chat-completions/v1",
    "temperature": 1,
    "top_p": 0.95,
    "max_tokens": 4096,
    "stream": False,
    "response_format": {"type": "json_object"},
    "thinking": {"type": "enabled"},
    "reasoning_effort": "low",
    "provider_parameter_overrides_allowed": False,
    "equivalence_policy": "exact-request-body-except-model/v1",
    "reasoning_content_policy": "private-vault-only-not-normalizer-input/v1",
}

PROVIDERS = {
    "glm_flash": {
        "provider": "zhipu",
        "requested_model": "glm-5.3-flash",
        "transport": "openai-compatible-chat-completions-external",
        "api_base_env": "WP3_GLM_API_BASE",
        "default_api_base": "https://open.bigmodel.cn/api/paas/v4",
        "chat_completions_endpoint": (
            "https://open.bigmodel.cn/api/paas/v4/chat/completions"
        ),
        "allowed_origin": "https://open.bigmodel.cn",
        "api_key_env": "GLM_API_KEY",
        "returned_model_policy": "normalized-exact-requested-model/v1",
    },
    "deepseek_flash": {
        "provider": "deepseek",
        "requested_model": "deepseek-v4-flash",
        "transport": "openai-compatible-chat-completions-external",
        "api_base_env": "WP3_DEEPSEEK_API_BASE",
        "default_api_base": "https://api.deepseek.com",
        "chat_completions_endpoint": "https://api.deepseek.com/chat/completions",
        "allowed_origin": "https://api.deepseek.com",
        "api_key_env": "DEEPSEEK_API_KEY",
        "returned_model_policy": "normalized-exact-requested-model/v1",
    },
}

G1_RETENTION_CRITERIA = {
    "decision_state_policy": "incomplete-is-inconclusive-else-retain-or-drop/v1",
    "decision_rule": "retain-iff-(recall-gain-or-unique-tp)-and-f1-gates/v1",
    "completeness_scope": "all-2544-model-slots-plus-g3-and-offset-replay/v1",
    "metric_contract": "exact-occurrence-micro-unrounded-absolute-delta/v1",
    "baseline_union": "glm-g1-plus-dual-g2-plus-g3/v1",
    "added_union": "baseline-plus-deepseek-g1/v1",
    "min_full_union_exact_recall_gain": 0.01,
    "min_full_union_unique_exact_true_positives": 2,
    "min_unique_tp_distinct_cases": 2,
    "full_union_exact_f1_floor": 0.85,
    "max_full_union_exact_f1_drop": 0.005,
    "required_offset_replay_rate": 1,
    "required_terminal_slot_count": 2544,
    "required_terminal_failure_count": 0,
}


class CandidateRunPlanError(RuntimeError):
    """Raised when the current-run provider plan fails closed."""


def _object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise CandidateRunPlanError(f"{label} must be an object")
    return dict(value)


def _exact_int(value: Any, expected: int) -> bool:
    return not isinstance(value, bool) and isinstance(value, int) and value == expected


def _sha_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _validate_prompt_bindings(plan: Mapping[str, Any]) -> None:
    if plan.get("prompt_contract") != PROMPT_CONTRACT:
        raise CandidateRunPlanError("current-run prompt contract differs")
    current = {
        "g1_surface_decode": {
            "version": G1_PROMPT_VERSIONS["surface_decode"],
            "sha256": _sha_text(G1_SURFACE_SYSTEM_PROMPT),
        },
        "g1_lexical_pragmatic": {
            "version": G1_PROMPT_VERSIONS["lexical_pragmatic"],
            "sha256": _sha_text(G1_LEXICAL_SYSTEM_PROMPT),
        },
        "g2_direct_mention": {
            "version": G2_PROMPT_VERSION,
            "sha256": _sha_text(G2_SYSTEM_PROMPT),
        },
    }
    if current != PROMPT_CONTRACT:
        raise CandidateRunPlanError("frozen generator prompt implementation drifted")


def _validate_provider(provider_id: str, value: Any) -> None:
    provider = _object(value, f"provider {provider_id}")
    if canonical_sha256(provider) != canonical_sha256(PROVIDERS[provider_id]):
        raise CandidateRunPlanError(f"provider {provider_id} contract differs")
    base = urlsplit(provider["default_api_base"])
    endpoint = urlsplit(provider["chat_completions_endpoint"])
    origin = urlsplit(provider["allowed_origin"])
    if (
        base.scheme != "https"
        or endpoint.scheme != "https"
        or origin.scheme != "https"
        or not base.netloc
        or not endpoint.netloc
        or not origin.netloc
        or base.username is not None
        or endpoint.username is not None
        or origin.username is not None
        or f"{base.scheme}://{base.netloc}" != provider["allowed_origin"]
        or f"{endpoint.scheme}://{endpoint.netloc}" != provider["allowed_origin"]
        or endpoint.query
        or endpoint.fragment
    ):
        raise CandidateRunPlanError(f"provider {provider_id} origin differs")


def _validate_plan_identity(plan: Mapping[str, Any]) -> None:
    identity = {key: copy.deepcopy(value) for key, value in plan.items() if key != "plan_id"}
    expected = PLAN_ID_PREFIX + canonical_sha256(identity)
    if plan.get("plan_id") != expected:
        raise CandidateRunPlanError("current-run plan content-addressed ID differs")


def _validate_runtime_plan(plan: Mapping[str, Any]) -> None:
    providers = plan.get("providers")
    g1 = plan.get("g1")
    g2 = plan.get("g2")
    if (
        plan.get("schema_version") != PLAN_SCHEMA_VERSION
        or plan.get("plan_revision") != PLAN_REVISION
        or plan.get("plan_id") != CURRENT_PLAN_ID
        or canonical_sha256(plan.get("request_profile"))
        != canonical_sha256(REQUEST_PROFILE)
        or canonical_sha256(providers) != canonical_sha256(PROVIDERS)
        or not isinstance(g1, Mapping)
        or g1.get("passes") != ["surface_decode", "lexical_pragmatic"]
        or g1.get("providers") != ["glm_flash", "deepseek_flash"]
        or not isinstance(g2, Mapping)
        or g2.get("providers") != ["glm_flash", "deepseek_flash"]
        or not _exact_int(plan.get("total_logical_model_slot_count"), 2544)
    ):
        raise CandidateRunPlanError("runtime provider-plan projection differs")
    _validate_plan_identity(plan)


def load_current_run_plan(
    path: str | Path,
    *,
    workspace_root: str | Path,
    schema_path: str | Path,
    require_current_sources: bool = True,
) -> dict[str, Any]:
    """Load and deeply validate the provider-plan v2 overlay."""

    if schema_path is None:
        raise CandidateRunPlanError("current-run plan schema is required")
    try:
        value = load_json(path)
        plan = _object(value, "current-run provider plan")
        schema = _object(load_json(schema_path), "current-run plan schema")
        if (
            schema.get("$id") != PLAN_SCHEMA_ID
            or canonical_sha256(schema) != PLAN_SCHEMA_SHA256
        ):
            raise CandidateRunPlanError("current-run plan schema binding differs")
        validate_json_schema(plan, schema_path)
    except TrainingArtifactError as exc:
        raise CandidateRunPlanError(str(exc)) from exc

    expected_fields = {
        "schema_version",
        "plan_revision",
        "plan_id",
        "validation_contract",
        "accepted_date",
        "scope",
        "scientific_eligible",
        "sealed",
        "supersedes",
        "frame_dependency",
        "preserved_raw_gold",
        "implementation_dependency",
        "prompt_contract",
        "request_profile",
        "providers",
        "g1",
        "g2",
        "g3",
        "total_logical_model_slot_count",
        "execution",
    }
    if set(plan) != expected_fields:
        raise CandidateRunPlanError("current-run plan fields are not canonical")
    if (
        plan.get("schema_version") != PLAN_SCHEMA_VERSION
        or plan.get("plan_revision") != PLAN_REVISION
        or plan.get("accepted_date") != "2026-08-30"
        or plan.get("scope") != "development-only"
        or plan.get("scientific_eligible") is not False
        or plan.get("sealed") is not False
    ):
        raise CandidateRunPlanError("current-run plan identity or scope differs")

    validation_contract = _object(
        plan.get("validation_contract"), "validation_contract"
    )
    if validation_contract != {
        "schema_path": "schemas/wp3_candidate_current_run_plan_v1.schema.json",
        "schema_id": PLAN_SCHEMA_ID,
        "schema_canonical_sha256": PLAN_SCHEMA_SHA256,
    }:
        raise CandidateRunPlanError("current-run validation contract differs")

    supersedes = _object(plan.get("supersedes"), "supersedes")
    if supersedes != {
        "base_protocol_revision": BASE_PROTOCOL_REVISION,
        "base_protocol_canonical_sha256": BASE_PROTOCOL_CANONICAL_SHA256,
        "json_pointer": "/current_generator_pilot",
        "status": "superseded-before-execution",
        "raw_gold_semantics_unchanged": True,
    }:
        raise CandidateRunPlanError("current-run supersession boundary differs")
    frame = _object(plan.get("frame_dependency"), "frame_dependency")
    if frame != {
        "artifact_kind": "wp3-s21-development-frame",
        "artifact_id": FRAME_ID,
        "payload_manifest_sha256": FRAME_PAYLOAD_SHA256,
        "case_count": EXPECTED_CASE_COUNT,
        "cases_canonical_sha256": FRAME_CASES_CANONICAL_SHA256,
        "case_id_set_canonical_sha256": FRAME_CASE_ID_SET_CANONICAL_SHA256,
        "input_projection": "task-id-blind-alias-content-only/v1",
        "task_id_policy": "frame-case-id/v1",
        "task_projection_canonical_sha256": TASK_PROJECTION_CANONICAL_SHA256,
    }:
        raise CandidateRunPlanError("current-run frame dependency differs")
    raw_gold = _object(plan.get("preserved_raw_gold"), "preserved_raw_gold")
    if raw_gold != {
        "artifact_kind": "wp3-s21-locked-raw-gold",
        "artifact_id": RAW_GOLD_ID,
        "payload_manifest_sha256": RAW_GOLD_PAYLOAD_SHA256,
        "run_must_not_read": True,
        "first_join_stage": "development-evaluation/v1",
    }:
        raise CandidateRunPlanError("preserved raw-gold boundary differs")
    implementation = _object(
        plan.get("implementation_dependency"), "implementation_dependency"
    )
    if implementation != {
        "path": LEGACY_IMPLEMENTATION_PATH,
        "sha256": LEGACY_IMPLEMENTATION_SHA256,
        "reuse_scope": "prompt-request-shape-and-response-normalizers-only",
    }:
        raise CandidateRunPlanError("current-run implementation binding differs")
    _validate_prompt_bindings(plan)
    profile = _object(plan.get("request_profile"), "request_profile")
    if (
        profile != REQUEST_PROFILE
        or not _exact_int(profile.get("temperature"), 1)
        or isinstance(profile.get("top_p"), bool)
        or not isinstance(profile.get("top_p"), float)
        or profile.get("top_p") != 0.95
        or not _exact_int(profile.get("max_tokens"), 4096)
    ):
        raise CandidateRunPlanError("shared request profile differs")

    providers = _object(plan.get("providers"), "providers")
    if set(providers) != set(PROVIDERS):
        raise CandidateRunPlanError("current-run provider set differs")
    for provider_id in PROVIDERS:
        _validate_provider(provider_id, providers[provider_id])
    if "qwen" in str(plan).casefold():
        raise CandidateRunPlanError("current-run plan retains a superseded Qwen slot")

    g1 = _object(plan.get("g1"), "g1")
    if (
        set(g1)
        != {
            "passes",
            "providers",
            "aggregation",
            "logical_slots_per_provider",
            "logical_slot_count",
            "dual_provider_status",
            "post_pilot_retention_gate",
            "post_pilot_retention_criteria",
        }
        or g1.get("passes") != ["surface_decode", "lexical_pragmatic"]
        or g1.get("providers") != ["glm_flash", "deepseek_flash"]
        or g1.get("aggregation")
        != "pass-then-provider-exact-occurrence-union-no-vote/v1"
        or not _exact_int(g1.get("logical_slots_per_provider"), 848)
        or not _exact_int(g1.get("logical_slot_count"), 1696)
        or g1.get("dual_provider_status") != "development-factorial-diagnostic"
        or g1.get("post_pilot_retention_gate")
        != "provider-ablation-before-s22-or-full-fit/v1"
        or g1.get("post_pilot_retention_criteria") != G1_RETENTION_CRITERIA
    ):
        raise CandidateRunPlanError("G1 development factorial plan differs")
    g2 = _object(plan.get("g2"), "g2")
    if (
        set(g2)
        != {
            "providers",
            "aggregation",
            "logical_slots_per_provider",
            "logical_slot_count",
        }
        or g2.get("providers") != ["glm_flash", "deepseek_flash"]
        or g2.get("aggregation")
        != "provider-exact-occurrence-union-no-vote/v1"
        or not _exact_int(g2.get("logical_slots_per_provider"), 424)
        or not _exact_int(g2.get("logical_slot_count"), 848)
    ):
        raise CandidateRunPlanError("G2 provider-union plan differs")
    g3 = _object(plan.get("g3"), "g3")
    if g3 != {
        "execution": "offline-deterministic",
        "logical_model_slot_count": 0,
    } or not _exact_int(g3.get("logical_model_slot_count"), 0):
        raise CandidateRunPlanError("G3 offline boundary differs")
    if not _exact_int(plan.get("total_logical_model_slot_count"), 2544):
        raise CandidateRunPlanError("current-run logical slot total differs")
    execution = _object(plan.get("execution"), "execution")
    if set(execution) != {
        "model_calls_authorized",
        "network_calls_authorized",
        "paid_calls_authorized",
        "formal_artifact_publication_authorized",
        "full_fit_execution_authorized",
        "s22_materialization_authorized",
        "runner_status",
        "plan_lifecycle_status",
        "required_successor_change",
    } or any(
        execution.get(field) is not False
        for field in {
            "model_calls_authorized",
            "network_calls_authorized",
            "paid_calls_authorized",
            "formal_artifact_publication_authorized",
            "full_fit_execution_authorized",
            "s22_materialization_authorized",
        }
    ):
        raise CandidateRunPlanError("current-run execution must remain unauthorized")
    if execution.get("runner_status") != "not-implemented":
        raise CandidateRunPlanError("current-run runner status differs")
    if (
        execution.get("plan_lifecycle_status")
        != "planning-only-must-be-superseded-before-execution"
        or execution.get("required_successor_change")
        != "prompt-v2-run-plan-and-runner-contract/v1"
    ):
        raise CandidateRunPlanError("current-run lifecycle status differs")
    _validate_plan_identity(plan)

    if require_current_sources:
        root = Path(workspace_root).resolve()
        if sha256_file(root / LEGACY_IMPLEMENTATION_PATH) != LEGACY_IMPLEMENTATION_SHA256:
            raise CandidateRunPlanError("frozen generator implementation drifted")
        base_protocol = root / "config/stage1/wp3_candidate_generators_s21b_v1.json"
        if canonical_sha256(load_json(base_protocol)) != BASE_PROTOCOL_CANONICAL_SHA256:
            raise CandidateRunPlanError("base S2.1b protocol drifted")
    return plan


def _validate_bound_public_tasks(
    plan: Mapping[str, Any], tasks: Sequence[Mapping[str, Any]]
) -> list[dict[str, str]]:
    if isinstance(tasks, (str, bytes)) or len(tasks) != EXPECTED_CASE_COUNT:
        raise CandidateRunPlanError("bound task projection must contain 424 cases")
    projected: list[dict[str, str]] = []
    for value in tasks:
        task = _object(value, "bound public task")
        if set(task) != {"task_id", "blind_alias", "content"} or any(
            not isinstance(task.get(field), str) or not task[field]
            for field in ("task_id", "blind_alias", "content")
        ):
            raise CandidateRunPlanError("bound public task fields differ")
        projected.append(
            {
                "task_id": task["task_id"],
                "blind_alias": task["blind_alias"],
                "content": task["content"],
            }
        )
    projected.sort(key=lambda row: row["task_id"])
    task_ids = [row["task_id"] for row in projected]
    if (
        len(set(task_ids)) != EXPECTED_CASE_COUNT
        or canonical_sha256(task_ids) != FRAME_CASE_ID_SET_CANONICAL_SHA256
        or canonical_sha256(projected) != TASK_PROJECTION_CANONICAL_SHA256
        or plan["frame_dependency"]["task_projection_canonical_sha256"]
        != TASK_PROJECTION_CANONICAL_SHA256
    ):
        raise CandidateRunPlanError("bound task projection differs from frozen frame")
    return projected


def load_bound_public_tasks(
    plan: Mapping[str, Any], frame_ref: str | Path
) -> list[dict[str, str]]:
    """Load only the frozen frame's public cases and build the exact run projection."""

    _validate_runtime_plan(plan)
    try:
        locator, target = resolve_locator_ref(
            frame_ref, "wp3-s21-development-frame"
        )
        manifest = _object(load_json(target / "manifest.json"), "frame manifest")
        cases_value = load_json(target / "cases.json")
    except TrainingArtifactError as exc:
        raise CandidateRunPlanError(str(exc)) from exc
    dependency = plan["frame_dependency"]
    if (
        locator.get("artifact_id") != dependency["artifact_id"]
        or locator.get("payload_manifest_sha256")
        != dependency["payload_manifest_sha256"]
        or manifest.get("frame_id") != dependency["artifact_id"]
        or not _exact_int(manifest.get("case_count"), EXPECTED_CASE_COUNT)
        or manifest.get("cases_sha256") != FRAME_CASES_CANONICAL_SHA256
        or manifest.get("calibration_dev_test_intersection_count") != 0
        or manifest.get("task_label_contribution_count") != 0
        or manifest.get("source_policy") != "fit-content-only-no-task-fields/v1"
    ):
        raise CandidateRunPlanError("public execution frame binding differs")
    if not isinstance(cases_value, list) or canonical_sha256(cases_value) != (
        FRAME_CASES_CANONICAL_SHA256
    ):
        raise CandidateRunPlanError("public frame cases differ")
    tasks: list[dict[str, str]] = []
    for value in cases_value:
        case = _object(value, "public frame case")
        if set(case) != {"case_id", "blind_alias", "content"}:
            raise CandidateRunPlanError("public frame case fields differ")
        tasks.append(
            {
                "task_id": case["case_id"],
                "blind_alias": case["blind_alias"],
                "content": case["content"],
            }
        )
    return _validate_bound_public_tasks(plan, tasks)


def _wire_profile(plan: Mapping[str, Any]) -> dict[str, Any]:
    profile = _object(plan.get("request_profile"), "request_profile")
    return {
        "temperature": profile["temperature"],
        "top_p": profile["top_p"],
        "max_tokens": profile["max_tokens"],
        "stream": profile["stream"],
        "response_format": copy.deepcopy(profile["response_format"]),
        "thinking": copy.deepcopy(profile["thinking"]),
        "reasoning_effort": profile["reasoning_effort"],
    }


def build_current_request(
    plan: Mapping[str, Any],
    bound_tasks: Sequence[Mapping[str, Any]],
    *,
    task_id: str,
    generator: str,
    variant: str,
    provider_id: str,
) -> dict[str, Any]:
    """Render one request selected from the exact frozen public task projection."""

    _validate_runtime_plan(plan)
    tasks = _validate_bound_public_tasks(plan, bound_tasks)
    if not isinstance(task_id, str):
        raise CandidateRunPlanError("bound task ID is missing")
    task = next((row for row in tasks if row["task_id"] == task_id), None)
    if task is None:
        raise CandidateRunPlanError("task is not in the frozen public projection")
    providers = _object(plan.get("providers"), "providers")
    if provider_id not in providers:
        raise CandidateRunPlanError("provider is not in the current-run plan")
    if generator == "g1":
        if provider_id not in plan["g1"]["providers"]:
            raise CandidateRunPlanError("provider is not enabled for G1")
        try:
            request = build_g1_request(
                task,
                pass_name=variant,
                model=providers[provider_id]["requested_model"],
            )
        except CandidateGeneratorError as exc:
            raise CandidateRunPlanError(str(exc)) from exc
    elif generator == "g2":
        if variant != "direct_mention" or provider_id not in plan["g2"]["providers"]:
            raise CandidateRunPlanError("G2 variant or provider differs")
        try:
            request = build_g2_request(
                task, model=providers[provider_id]["requested_model"]
            )
        except CandidateGeneratorError as exc:
            raise CandidateRunPlanError(str(exc)) from exc
    else:
        raise CandidateRunPlanError("generator does not use a model request")
    request.update(_wire_profile(plan))
    return request


def request_equivalence_projection(request: Mapping[str, Any]) -> dict[str, Any]:
    """Remove only model identity from a rendered request for equality checks."""

    projected = copy.deepcopy(dict(request))
    if set(projected) != {
        "model",
        "messages",
        "temperature",
        "top_p",
        "max_tokens",
        "stream",
        "response_format",
        "thinking",
        "reasoning_effort",
    }:
        raise CandidateRunPlanError("rendered request fields are not canonical")
    projected.pop("model")
    return projected


def validate_returned_model(
    plan: Mapping[str, Any], *, provider_id: str, returned_model: Any
) -> str:
    _validate_runtime_plan(plan)
    providers = _object(plan.get("providers"), "providers")
    if provider_id not in providers or not isinstance(returned_model, str):
        raise CandidateRunPlanError("returned model identity is missing")
    requested = str(providers[provider_id]["requested_model"]).strip().casefold()
    returned = returned_model.strip().casefold()
    if not returned or returned != requested:
        raise CandidateRunPlanError("returned model differs from requested model")
    return returned


def validate_provider_api_base(
    plan: Mapping[str, Any], *, provider_id: str, api_base: Any
) -> str:
    _validate_runtime_plan(plan)
    providers = _object(plan.get("providers"), "providers")
    if provider_id not in providers or not isinstance(api_base, str):
        raise CandidateRunPlanError("provider API base is missing")
    candidate = api_base.strip().rstrip("/")
    expected = str(providers[provider_id]["default_api_base"]).rstrip("/")
    parsed = urlsplit(candidate)
    if (
        parsed.scheme != "https"
        or not parsed.netloc
        or parsed.username is not None
        or parsed.query
        or parsed.fragment
        or candidate != expected
    ):
        raise CandidateRunPlanError("provider API base differs")
    return expected


def _nonnegative_count(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise CandidateRunPlanError(f"{label} must be a nonnegative integer")
    return value


def evaluate_g1_deepseek_retention(
    plan: Mapping[str, Any],
    *,
    terminal_slot_count: int,
    terminal_failure_count: int,
    g3_complete: bool,
    offset_replay_count: int,
    offset_occurrence_count: int,
    gold_mention_count: int,
    baseline_exact_tp: int,
    baseline_proposal_count: int,
    added_exact_tp: int,
    added_proposal_count: int,
    deepseek_unique_exact_tp: int,
    unique_tp_distinct_cases: int,
) -> dict[str, Any]:
    """Apply the frozen three-state DeepSeek-G1 development ablation gate."""

    _validate_runtime_plan(plan)
    values = {
        "terminal_slot_count": terminal_slot_count,
        "terminal_failure_count": terminal_failure_count,
        "offset_replay_count": offset_replay_count,
        "offset_occurrence_count": offset_occurrence_count,
        "gold_mention_count": gold_mention_count,
        "baseline_exact_tp": baseline_exact_tp,
        "baseline_proposal_count": baseline_proposal_count,
        "added_exact_tp": added_exact_tp,
        "added_proposal_count": added_proposal_count,
        "deepseek_unique_exact_tp": deepseek_unique_exact_tp,
        "unique_tp_distinct_cases": unique_tp_distinct_cases,
    }
    counts = {
        name: _nonnegative_count(value, name) for name, value in values.items()
    }
    if g3_complete is not True:
        g3_complete = False
    gold_count = counts["gold_mention_count"]
    if gold_count == 0:
        raise CandidateRunPlanError("gold_mention_count must be positive")
    if counts["offset_replay_count"] > counts["offset_occurrence_count"]:
        raise CandidateRunPlanError("offset replay count exceeds occurrence count")

    criteria = plan["g1"]["post_pilot_retention_criteria"]
    replay_rate = (
        Fraction(counts["offset_replay_count"], counts["offset_occurrence_count"])
        if counts["offset_occurrence_count"]
        else None
    )
    incomplete_reasons: list[str] = []
    if counts["terminal_slot_count"] != criteria["required_terminal_slot_count"]:
        incomplete_reasons.append("logical_slot_grid_incomplete")
    if counts["terminal_failure_count"] != criteria["required_terminal_failure_count"]:
        incomplete_reasons.append("terminal_failure_present")
    if not g3_complete:
        incomplete_reasons.append("g3_incomplete")
    if replay_rate is None:
        incomplete_reasons.append("offset_replay_unobserved")
    elif replay_rate != Fraction(criteria["required_offset_replay_rate"], 1):
        incomplete_reasons.append("offset_replay_incomplete")
    if incomplete_reasons:
        return {
            "status": "inconclusive",
            "reasons": incomplete_reasons,
            "completeness_scope": criteria["completeness_scope"],
            "offset_replay_rate": (
                float(replay_rate) if replay_rate is not None else None
            ),
        }

    baseline_tp = counts["baseline_exact_tp"]
    added_tp = counts["added_exact_tp"]
    baseline_proposals = counts["baseline_proposal_count"]
    added_proposals = counts["added_proposal_count"]
    unique_tp = counts["deepseek_unique_exact_tp"]
    distinct_cases = counts["unique_tp_distinct_cases"]
    if (
        baseline_tp > min(gold_count, baseline_proposals)
        or added_tp > min(gold_count, added_proposals)
        or added_tp < baseline_tp
        or added_proposals < baseline_proposals
        or unique_tp != added_tp - baseline_tp
        or distinct_cases > unique_tp
    ):
        raise CandidateRunPlanError("G1 retention metric counts are inconsistent")

    recall_gain = Fraction(added_tp - baseline_tp, gold_count)
    baseline_f1 = Fraction(2 * baseline_tp, baseline_proposals + gold_count)
    added_f1 = Fraction(2 * added_tp, added_proposals + gold_count)
    f1_drop = baseline_f1 - added_f1
    gain_gate = recall_gain >= Fraction(
        str(criteria["min_full_union_exact_recall_gain"])
    ) or (
        unique_tp >= criteria["min_full_union_unique_exact_true_positives"]
        and distinct_cases >= criteria["min_unique_tp_distinct_cases"]
    )
    f1_floor_gate = added_f1 >= Fraction(str(criteria["full_union_exact_f1_floor"]))
    f1_drop_gate = f1_drop <= Fraction(
        str(criteria["max_full_union_exact_f1_drop"])
    )
    passed = gain_gate and f1_floor_gate and f1_drop_gate
    reasons = []
    if not gain_gate:
        reasons.append("marginal_gain_below_gate")
    if not f1_floor_gate:
        reasons.append("added_union_f1_below_floor")
    if not f1_drop_gate:
        reasons.append("added_union_f1_drop_exceeds_cap")
    return {
        "status": "retain" if passed else "drop",
        "reasons": reasons,
        "metric_contract": criteria["metric_contract"],
        "baseline_union": criteria["baseline_union"],
        "added_union": criteria["added_union"],
        "metrics": {
            "full_union_exact_recall_gain": float(recall_gain),
            "baseline_full_union_exact_f1": float(baseline_f1),
            "added_full_union_exact_f1": float(added_f1),
            "full_union_exact_f1_drop": float(f1_drop),
            "deepseek_unique_exact_true_positives": unique_tp,
            "unique_tp_distinct_cases": distinct_cases,
        },
    }


def build_logical_slot_grid(
    plan: Mapping[str, Any], tasks: Sequence[Mapping[str, Any]]
) -> list[dict[str, str]]:
    """Build the deterministic grid only for the exact frozen public tasks."""

    _validate_runtime_plan(plan)
    task_rows = _validate_bound_public_tasks(plan, tasks)
    rows: list[dict[str, str]] = []
    for task in task_rows:
        task_id = task["task_id"]
        for pass_name in plan["g1"]["passes"]:
            for provider_id in plan["g1"]["providers"]:
                rows.append(
                    {
                        "generator": "g1",
                        "variant": pass_name,
                        "provider_id": provider_id,
                        "task_id": task_id,
                    }
                )
        for provider_id in plan["g2"]["providers"]:
            rows.append(
                {
                    "generator": "g2",
                    "variant": "direct_mention",
                    "provider_id": provider_id,
                    "task_id": task_id,
                }
            )
    for row in rows:
        identity = {"plan_id": plan["plan_id"], **row}
        row["slot_id"] = "wp3slot-" + canonical_sha256(identity)[:32]
    counts = Counter(row["provider_id"] for row in rows)
    if len(rows) != 2544 or counts != {"glm_flash": 1272, "deepseek_flash": 1272}:
        raise CandidateRunPlanError("logical slot grid count differs")
    return rows


__all__ = [
    "CandidateRunPlanError",
    "PLAN_REVISION",
    "PROVIDERS",
    "REQUEST_PROFILE",
    "build_current_request",
    "build_logical_slot_grid",
    "evaluate_g1_deepseek_retention",
    "load_bound_public_tasks",
    "load_current_run_plan",
    "request_equivalence_projection",
    "validate_provider_api_base",
    "validate_returned_model",
]
