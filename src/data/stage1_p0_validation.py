"""Read-only, fail-closed Stage 1 P0 readiness validation.

The report produced here is intentionally a status report rather than a build
artifact.  It never creates or repairs lifecycle targets, never opens a
path-classified sealed reference or sealed target payload, and never
serialises locator ``target_path`` values.  This keeps the report deterministic
and portable while still deep-validating every published development artifact
for which a semantic validator is available.
"""

from __future__ import annotations

import hashlib
import importlib
import inspect
import json
import os
import re
import stat
import tempfile
import unicodedata
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from data.training_artifacts import (
    TrainingArtifactError,
    canonical_json_bytes,
    canonical_sha256,
    load_json,
    load_jsonl,
    portable_dependency,
    resolve_dependency_target,
    resolve_locator_ref,
    sha256_file,
    validate_dependency_ref,
    validate_json_schema,
    validate_payload_manifest,
)


REPORT_SCHEMA_VERSION = "stage1-p0-validation-report/v1"
REPORT_SCHEMA_NAME = "stage1_p0_validation_report_v1.schema.json"
MODES = ("engineering-smoke", "formal-readiness")
STATUSES = ("PASS", "BLOCKED", "PENDING", "FAIL")
STATUS_PRECEDENCE = {"PASS": 0, "PENDING": 1, "BLOCKED": 2, "FAIL": 3}
SAFE_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,255}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

SEALED_REF_NAMES = (
    "test_context_ref.json",
    "test_control_ref.json",
    "test_cf_proposal_ref.json",
    "test_cf_blind_review_ref.json",
    "test_cf_review_ref.json",
    "test_cf_ref.json",
    "test_generation_ref.json",
    "test_generation_run_ref.json",
    "test_evaluation_ref.json",
    "test_margin_ref.json",
    "test_analysis_ref.json",
)
_PUBLIC_LIFECYCLE_REF_NAMES = frozenset(
    SEALED_REF_NAMES
    + (
        "data_ref.json",
        "verification_receipt_ref.json",
        "lexicon_ref.json",
        "train_context_ref.json",
        "dev_context_ref.json",
        "context_ref.json",
        "control_ref.json",
        "cf_proposal_ref.json",
        "cf_blind_review_ref.json",
        "smoke_cf_blind_review_ref.json",
        "cf_review_ref.json",
        "cf_ref.json",
        "generation_ref.json",
        "generation_run_ref.json",
        "evaluation_ref.json",
        "margin_ref.json",
        "analysis_ref.json",
    )
)

_SEALED_PATH_MARKERS = ("sealedtest", "heldout", "holdout", "test", "sealed")
_SEALED_PATH_DOMAINS: dict[str, tuple[str, ...]] = {
    "query": ("query", "queries", "querypool", "queryset"),
    "context": ("context", "contexts"),
    "control": ("control", "controls", "placebo", "placebos"),
    "counterfactual": (
        "cf",
        "counterfactual",
        "counterfactuals",
        "proposal",
        "proposals",
        "cohort",
        "cohorts",
        "foil",
        "foils",
        "review",
        "reviews",
    ),
    "generation": (
        "generation",
        "generations",
        "generationrun",
        "generationruns",
        "prediction",
        "predictions",
        "response",
        "responses",
        "output",
        "outputs",
        "run",
        "runs",
    ),
    "evaluation": (
        "evaluation",
        "evaluations",
        "metric",
        "metrics",
        "score",
        "scores",
    ),
    "margin": ("margin", "margins", "marginrun", "marginruns"),
    "analysis": (
        "analysis",
        "analyses",
        "analysisrun",
        "analysisruns",
        "result",
        "results",
    ),
}
_SEALED_METADATA_MAX_BYTES = 1024 * 1024


class Stage1P0ValidationError(RuntimeError):
    """Raised when the validator itself cannot produce a safe report."""


@dataclass(frozen=True)
class ArtifactRequirement:
    check_id: str
    area: str
    phase: str
    ref_names: tuple[str, ...]
    expected_kinds: tuple[str, ...]
    validator_key: str
    missing_status: str = "PENDING"
    expected_scope: str | None = None
    expected_split: str | None = None
    expected_scientific_eligible: bool | None = None
    expected_sealing_status: str | None = None


@dataclass
class ValidationContext:
    workspace_root: Path
    mode: str
    validator_overrides: Mapping[str, Callable[..., Any]] = field(default_factory=dict)
    tokenizers: dict[str, Any] = field(default_factory=dict)
    tokenizer_failures: set[str] = field(default_factory=set)
    semantic_reports: dict[str, dict[str, Any]] = field(default_factory=dict)
    artifact_targets: dict[str, Path] = field(default_factory=dict)
    artifact_dependencies: dict[str, dict[str, Any]] = field(default_factory=dict)
    resolved_targets: set[Path] = field(default_factory=set)

    @property
    def stage1_root(self) -> Path:
        return self.workspace_root / "exps" / "causal_context" / "stage1_p0"

    @property
    def refs_root(self) -> Path:
        return self.stage1_root / "refs"

    @property
    def schema_root(self) -> Path:
        return self.workspace_root / "schemas"


def _canonical_report_hash(report_without_hash: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json_bytes(report_without_hash)).hexdigest()


def _logical_path(path: Path, workspace_root: Path) -> str:
    resolved = path.resolve()
    try:
        logical = resolved.relative_to(workspace_root.resolve()).as_posix()
    except ValueError as exc:
        raise Stage1P0ValidationError("path is outside the validation workspace") from exc
    if logical in {"", "."} or ".." in Path(logical).parts:
        raise Stage1P0ValidationError("path is not a portable workspace-relative path")
    return logical


def _check(
    *,
    check_id: str,
    area: str,
    phase: str,
    status: str,
    reason_code: str,
    ref_path: str | None = None,
    artifact: Mapping[str, Any] | None = None,
    metrics: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if status not in STATUSES:
        raise Stage1P0ValidationError(f"unsupported validation status: {status}")
    return {
        "check_id": check_id,
        "area": area,
        "phase": phase,
        "status": status,
        "reason_code": reason_code,
        "ref_path": ref_path,
        "artifact": dict(artifact) if artifact is not None else None,
        "metrics": dict(metrics or {}),
    }


def _safe_artifact_descriptor(
    locator: Mapping[str, Any], target: Path, context: ValidationContext
) -> dict[str, Any]:
    artifact_id = locator.get("artifact_id")
    artifact_kind = locator.get("artifact_kind")
    payload_hash = locator.get("payload_manifest_sha256")
    if (
        not isinstance(artifact_id, str)
        or not SAFE_IDENTIFIER_RE.fullmatch(artifact_id)
        or not isinstance(artifact_kind, str)
        or not SAFE_IDENTIFIER_RE.fullmatch(artifact_kind)
        or not isinstance(payload_hash, str)
        or not SHA256_RE.fullmatch(payload_hash)
    ):
        raise Stage1P0ValidationError("locator contains a non-portable artifact identity")
    return {
        "artifact_kind": artifact_kind,
        "artifact_id": artifact_id,
        "payload_manifest_sha256": payload_hash,
        "logical_target_path": _logical_path(target, context.workspace_root),
    }


def _validator_candidates(key: str) -> tuple[tuple[str, str], ...]:
    """Return lazily imported validators, including compatibility fallbacks."""

    table: dict[str, tuple[tuple[str, str], ...]] = {
        "data": (("data.stage1_data", "validate_data"),),
        "partition": (("data.train_partition", "validate_train_partition"),),
        "data_review": (
            ("review.data_review_artifact", "validate_data_review_ref"),
        ),
        "lexicon": (("build_lex.train_only", "validate_lexicon_ref"),),
        "environment": (("scripts.stage1.capture_environment", "validate_ref"),),
        "test_receipt": (("data.test_receipt", "validate_test_receipt"),),
        "model": (("model.stage1_registry", "validate_model_artifact"),),
        "context": (("data.build_context_manifest", "validate_context_ref"),),
        "evidence": (("data.training_evidence", "validate_training_evidence"),),
        "plan": (("data.training_plan", "validate_training_plan"),),
        "schedule": (("data.training_schedule", "validate_training_schedule"),),
        "control": (("data.control_manifest", "validate_control_ref"),),
        "cf_proposal": (("data.counterfactual_lifecycle", "validate_proposal_ref"),),
        "cf_blind_review": (
            ("review.cf_blind_review", "validate_cf_blind_review_ref"),
        ),
        "cf_review": (("data.counterfactual_lifecycle", "validate_review_target"),),
        "cf": (("data.counterfactual_lifecycle", "validate_cf_ref"),),
        # The registry is being evolved in parallel.  Resolve capabilities at
        # call time and retain the metrics wrapper as a compatibility fallback.
        "registry": (
            ("model.stage1_registry", "validate_model_registry"),
            ("metrics.stage1_artifacts", "validate_registry_ref"),
        ),
        # Accept both the new raw-generation lifecycle and the compatibility
        # wrapper used by the evaluator during the migration.
        "generation": (
            ("data.generation_lifecycle", "validate_generation_ref"),
            ("metrics.stage1_artifacts", "validate_generation_ref"),
        ),
        "evaluation": (
            ("metrics.stage1_artifacts", "validate_evaluation_ref_report"),
        ),
        "margin": (("metrics.stage1_artifacts", "validate_margin_ref"),),
        "analysis": (("metrics.stage1_artifacts", "validate_analysis_ref"),),
    }
    return table.get(key, ())


def _load_tokenizer(context: ValidationContext, *, ref_path: Path) -> Any:
    # Every selected artifact in the engineering report belongs to the exact
    # legacy-smoke chain, even when its generic locator name is
    # ``smoke_model_registry_ref.json`` or ``generation_run_ref.json``.  Keying
    # the tokenizer choice off the filename would silently inject the Qwen3
    # base tokenizer into validators for the registered legacy checkpoint.
    legacy = context.mode == "engineering-smoke"
    model_ref = context.refs_root / (
        "legacy_model_ref.json" if legacy else "base_model_ref.json"
    )
    tokenizer_path: Path | None = None
    model_document: Mapping[str, Any] | None = None
    if model_ref.is_file():
        try:
            _locator, model_target = resolve_locator_ref(
                model_ref, expected_kind="stage1-model"
            )
            model = load_json(model_target / "model.json")
            if not isinstance(model, Mapping):
                raise Stage1P0ValidationError("frozen model document is invalid")
            model_document = model
            logical_path = model.get("tokenizer_inventory", {}).get(
                "logical_repo_path"
            )
            if isinstance(logical_path, str) and logical_path:
                candidate = (context.workspace_root / logical_path).resolve()
                candidate.relative_to(context.workspace_root)
                tokenizer_path = candidate
        except Exception:
            tokenizer_path = None
    if tokenizer_path is None:
        tokenizer_path = context.workspace_root / "models" / "base" / "Qwen3-8B"
    cache_key = str(tokenizer_path)
    if cache_key in context.tokenizers:
        return context.tokenizers[cache_key]
    if cache_key in context.tokenizer_failures:
        raise Stage1P0ValidationError("frozen tokenizer is unavailable")
    if not tokenizer_path.is_dir():
        context.tokenizer_failures.add(cache_key)
        raise Stage1P0ValidationError("frozen tokenizer is unavailable")
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        context.tokenizer_failures.add(cache_key)
        raise Stage1P0ValidationError("frozen tokenizer is unavailable") from exc
    if model_document is None:
        context.tokenizer_failures.add(cache_key)
        raise Stage1P0ValidationError(
            "frozen tokenizer lacks a registered source contract"
        )
    try:
        from model.stage1_registry import (
            ResolvedModelSourceContract,
            verified_model_source_lease,
        )

        contract = ResolvedModelSourceContract(
            workspace_root=context.workspace_root.resolve(),
            checkpoint_inventory=model_document["checkpoint_inventory"],
            tokenizer_inventory=model_document["tokenizer_inventory"],
            base_inventory=model_document["base_inventory"],
        )
        # ``base`` is intentionally included: older base artifacts projected
        # a tokenizer whitelist while their base inventory froze the complete
        # same tree. New registrations freeze the tokenizer tree directly.
        with verified_model_source_lease(
            contract, source_names=("tokenizer", "base")
        ) as sources:
            if sources.tokenizer_path.resolve() != tokenizer_path.resolve():
                raise Stage1P0ValidationError(
                    "registered tokenizer path differs from its source lease"
                )
            tokenizer = AutoTokenizer.from_pretrained(
                sources.tokenizer_path,
                trust_remote_code=False,
                local_files_only=True,
            )
    except Exception as exc:
        context.tokenizer_failures.add(cache_key)
        raise Stage1P0ValidationError("frozen tokenizer is unavailable") from exc
    context.tokenizers[cache_key] = tokenizer
    return tokenizer


def _call_with_supported_kwargs(
    function: Callable[..., Any],
    *,
    key: str,
    ref_path: Path,
    target: Path,
    context: ValidationContext,
) -> Any:
    parameters = inspect.signature(function).parameters
    kwargs: dict[str, Any] = {}
    if "workspace_root" in parameters:
        kwargs["workspace_root"] = context.workspace_root
    if "require_scientific" in parameters:
        # The immutable two-pass smoke proof remains an explicitly
        # non-scientific engineering artifact even when it is checked as a
        # prerequisite of formal readiness.  Formal downstream generation,
        # which is selected through ``generation_run_ref.json``, must still
        # fail closed unless it is scientific.
        is_legacy_smoke_proof = (
            key == "generation"
            and ref_path.name == "legacy_smoke_generation_ref.json"
        )
        kwargs["require_scientific"] = (
            context.mode == "formal-readiness" and not is_legacy_smoke_proof
        )
    if "tokenizer" in parameters:
        kwargs["tokenizer"] = _load_tokenizer(context, ref_path=ref_path)
    if key == "control" and "context_ref" in parameters:
        context_names = (
            (
                "legacy_smoke_context_ref.json",
                "smoke_context_ref.json",
                "context_ref.json",
            )
            if context.mode == "engineering-smoke"
            else ("dev_context_ref.json",)
        )
        matching_context_ref = _find_ref(context, context_names)
        if matching_context_ref is not None:
            kwargs["context_ref"] = matching_context_ref
    if key == "cf_review":
        return function(target, **kwargs)
    if "data_ref" in parameters:
        kwargs["data_ref"] = ref_path
        return function(**kwargs)
    if "schedule_ref" in parameters:
        kwargs["schedule_ref"] = ref_path
        return function(**kwargs)
    positional_names = [
        name
        for name, parameter in parameters.items()
        if parameter.kind
        in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        and parameter.default is inspect.Parameter.empty
    ]
    if positional_names:
        return function(ref_path, **kwargs)
    keyword_aliases = {
        "registry": ("registry_ref",),
        "generation": ("generation_ref", "generation_run_ref", "ref_path"),
        "evaluation": ("evaluation_ref",),
        "margin": ("margin_ref",),
        "analysis": ("analysis_ref",),
        "model": ("model_ref",),
        "environment": ("environment_ref",),
        "context": ("ref_path", "context_ref"),
        "evidence": ("training_evidence_ref",),
        "plan": ("training_plan_ref",),
        "control": ("control_ref",),
        "cf_proposal": ("proposal_ref",),
        "cf": ("cf_ref",),
        "lexicon": ("ref_path",),
        "partition": ("partition_ref", "train_partition_ref", "ref_path"),
    }
    for alias in keyword_aliases.get(key, ()):
        if alias in parameters:
            kwargs[alias] = ref_path
            return function(**kwargs)
    return function(ref_path, **kwargs)


def _normalise_semantic_report(value: Any) -> dict[str, Any]:
    if isinstance(value, tuple):
        mappings = [item for item in value if isinstance(item, Mapping)]
        value = mappings[-1] if mappings else {}
    if not isinstance(value, Mapping):
        return {}
    # Retain only values used for cross-artifact gates.  Arbitrary validator
    # output can include paths or environment details and must not reach JSON.
    allowed = {
        "scope",
        "registry_scope",
        "split",
        "mode",
        "model_key",
        "model_role",
        "role",
        "seed",
        "query_count",
        "record_count",
        "row_count",
        "generation_run_id",
        "evaluation_id",
        "margin_run_id",
        "analysis_id",
        "cf_proposal_id",
        "cf_blind_review_id",
        "review_id",
        "cf_build_id",
        "scientific_eligible",
        "complete_paired_blocks",
        "execution_repetitions",
        "exact_rerun_match",
        "executor_backend",
        "executor_id",
        "sealing_status",
        "train_partition_id",
        "fit_count",
        "calibration_count",
    }
    result: dict[str, Any] = {}
    for key in sorted(allowed):
        if key not in value:
            continue
        item = value.get(key)
        if item is None or isinstance(item, (str, int, bool)):
            result[key] = item
    query_ids = value.get("query_ids")
    if "query_count" not in result and isinstance(query_ids, list) and all(
        isinstance(item, str) for item in query_ids
    ):
        result["query_count"] = len(query_ids)
    if "role" not in result and isinstance(result.get("model_role"), str):
        result["role"] = result["model_role"]
    ordered_conditions = value.get("ordered_conditions")
    if (
        isinstance(ordered_conditions, list)
        and ordered_conditions
        and len(ordered_conditions) == len(set(ordered_conditions))
        and all(
            item in {"C0", "CL", "CD", "CLD", "PL", "PD"}
            for item in ordered_conditions
        )
    ):
        result["ordered_conditions"] = list(ordered_conditions)
    cf_dependency = value.get("cf_dependency")
    if isinstance(cf_dependency, Mapping):
        try:
            result["cf_dependency"] = dict(
                validate_dependency_ref(
                    cf_dependency, expected_kind="counterfactual"
                )
            )
        except Exception:
            # A malformed dependency is deliberately omitted so the downstream
            # exact-lineage gate fails closed without leaking arbitrary report
            # fields into the public validation report.
            pass
    return result


def _semantic_validate(
    key: str,
    ref_path: Path,
    target: Path,
    context: ValidationContext,
) -> tuple[str, dict[str, Any]]:
    override = context.validator_overrides.get(key)
    if override is not None:
        result = _call_with_supported_kwargs(
            override,
            key=key,
            ref_path=ref_path,
            target=target,
            context=context,
        )
        return "override", _normalise_semantic_report(result)

    available = False
    rejected = False
    for module_name, attribute in _validator_candidates(key):
        try:
            module = importlib.import_module(module_name)
        except (ImportError, ModuleNotFoundError):
            continue
        function = getattr(module, attribute, None)
        if not callable(function):
            continue
        available = True
        try:
            result = _call_with_supported_kwargs(
                function,
                key=key,
                ref_path=ref_path,
                target=target,
                context=context,
            )
        except Stage1P0ValidationError:
            raise
        except Exception:
            rejected = True
            continue
        return f"{module_name}.{attribute}", _normalise_semantic_report(result)
    if rejected:
        raise TrainingArtifactError("all compatible semantic validators rejected artifact")
    if not available:
        raise Stage1P0ValidationError("semantic validator is unavailable")
    raise TrainingArtifactError("semantic validation did not complete")


def _find_ref(context: ValidationContext, names: Sequence[str]) -> Path | None:
    matches = [context.refs_root / name for name in names if (context.refs_root / name).is_file()]
    if not matches:
        return None
    return matches[0]


def _artifact_check(
    requirement: ArtifactRequirement, context: ValidationContext
) -> dict[str, Any]:
    ref_path = _find_ref(context, requirement.ref_names)
    expected_ref = f"exps/causal_context/stage1_p0/refs/{requirement.ref_names[0]}"
    if ref_path is None:
        return _check(
            check_id=requirement.check_id,
            area=requirement.area,
            phase=requirement.phase,
            status=requirement.missing_status,
            reason_code="required-ref-missing",
            ref_path=expected_ref,
        )

    ref_logical = _logical_path(ref_path, context.workspace_root)
    before_ref_hash = sha256_file(ref_path)
    try:
        locator, target = resolve_locator_ref(
            ref_path,
            expected_kind=requirement.expected_kinds,
        )
        target = target.resolve()
        target.relative_to(context.workspace_root)
        descriptor = _safe_artifact_descriptor(locator, target, context)
        context.resolved_targets.add(target)
        before_payload_hash = validate_payload_manifest(target)
    except Exception:
        return _check(
            check_id=requirement.check_id,
            area=requirement.area,
            phase=requirement.phase,
            status="FAIL",
            reason_code="locator-or-payload-invalid",
            ref_path=ref_logical,
        )

    try:
        validator_name, semantic = _semantic_validate(
            requirement.validator_key,
            ref_path,
            target,
            context,
        )
    except Stage1P0ValidationError:
        return _check(
            check_id=requirement.check_id,
            area=requirement.area,
            phase=requirement.phase,
            status="PENDING",
            reason_code="semantic-validator-unavailable",
            ref_path=ref_logical,
            artifact=descriptor,
        )
    except Exception:
        return _check(
            check_id=requirement.check_id,
            area=requirement.area,
            phase=requirement.phase,
            status="FAIL",
            reason_code="semantic-validator-rejected-artifact",
            ref_path=ref_logical,
            artifact=descriptor,
        )

    try:
        after_ref_hash = sha256_file(ref_path)
        after_payload_hash = validate_payload_manifest(target)
    except Exception:
        return _check(
            check_id=requirement.check_id,
            area=requirement.area,
            phase=requirement.phase,
            status="FAIL",
            reason_code="artifact-changed-during-validation",
            ref_path=ref_logical,
            artifact=descriptor,
        )
    if before_ref_hash != after_ref_hash or before_payload_hash != after_payload_hash:
        return _check(
            check_id=requirement.check_id,
            area=requirement.area,
            phase=requirement.phase,
            status="FAIL",
            reason_code="artifact-changed-during-validation",
            ref_path=ref_logical,
            artifact=descriptor,
        )

    observed_scope = semantic.get("scope")
    if observed_scope is None:
        observed_scope = semantic.get("registry_scope")
    if requirement.expected_scope is not None and observed_scope != requirement.expected_scope:
        return _check(
            check_id=requirement.check_id,
            area=requirement.area,
            phase=requirement.phase,
            status="FAIL",
            reason_code="artifact-scope-mismatch",
            ref_path=ref_logical,
            artifact=descriptor,
        )
    if (
        requirement.expected_split is not None
        and semantic.get("split") != requirement.expected_split
    ):
        return _check(
            check_id=requirement.check_id,
            area=requirement.area,
            phase=requirement.phase,
            status="FAIL",
            reason_code="artifact-split-mismatch",
            ref_path=ref_logical,
            artifact=descriptor,
        )
    if (
        requirement.expected_scientific_eligible is not None
        and semantic.get("scientific_eligible")
        is not requirement.expected_scientific_eligible
    ):
        return _check(
            check_id=requirement.check_id,
            area=requirement.area,
            phase=requirement.phase,
            status="FAIL",
            reason_code="artifact-scientific-eligibility-mismatch",
            ref_path=ref_logical,
            artifact=descriptor,
        )
    if (
        requirement.expected_sealing_status is not None
        and semantic.get("sealing_status") != requirement.expected_sealing_status
    ):
        return _check(
            check_id=requirement.check_id,
            area=requirement.area,
            phase=requirement.phase,
            status="FAIL",
            reason_code="artifact-sealing-status-mismatch",
            ref_path=ref_logical,
            artifact=descriptor,
        )
    context.semantic_reports[requirement.check_id] = semantic
    context.artifact_targets[requirement.check_id] = target
    context.artifact_dependencies[requirement.check_id] = portable_dependency(
        locator, target, context.workspace_root
    )
    return _check(
        check_id=requirement.check_id,
        area=requirement.area,
        phase=requirement.phase,
        status="PASS",
        reason_code="semantic-and-payload-validation-pass",
        ref_path=ref_logical,
        artifact=descriptor,
        metrics={"semantic_validator": validator_name},
    )


def _decision_check(context: ValidationContext) -> tuple[dict[str, Any], dict[str, Any]]:
    path = context.workspace_root / "config" / "stage1" / "decision_register.json"
    logical = "config/stage1/decision_register.json"
    if not path.is_file():
        check = _check(
            check_id="decision.register",
            area="decision",
            phase="p0-prerequisite",
            status="PENDING",
            reason_code="decision-register-missing",
            ref_path=logical,
        )
        return check, {"logical_path": logical, "sha256": None, "frozen": False}
    try:
        document = load_json(path)
        expected_decisions = {f"D{index}_{suffix}" for index, suffix in enumerate((
            "split",
            "wire_format",
            "anomaly_adjudication",
            "group_semantics",
            "sequence_budget",
            "controls",
            "cf_selector",
            "training_seeds",
            "margin_scope",
            "dev_gate",
            "placebo_matching",
            "base_model",
            "environment",
            "training_recipe",
            "blind_review_models",
            "terminology_library",
        ))}
        if (
            not isinstance(document, Mapping)
            or document.get("schema_version") != "stage1-decision-register/v1"
            or document.get("status") != "frozen"
            or document.get("source_plan")
            != "docs/research/experiment-plans/stage1-p0-implementation.md"
            or set(document.get("decisions", {})) != expected_decisions
        ):
            raise ValueError("decision register contract mismatch")
        decisions = document["decisions"]
        if (
            decisions["D0_split"].get("train_count") != 5781
            or decisions["D0_split"].get("dev_count") != 643
            or decisions["D0_split"].get("test_count") != 1605
            or decisions["D4_sequence_budget"].get("tail_truncation_allowed") is not False
            or decisions["D5_controls"].get("primary") != ["PL", "PD"]
            or decisions["D7_training_seeds"].get("formal") != [42, 43, 44]
            or decisions["D13_training_recipe"].get(
                "calibration_information_isolation"
            )
            != "full-information-isolated"
            or decisions["D13_training_recipe"].get("fit_evidence_scope")
            != "fit-only-demo-and-lexicon"
            or decisions["D13_training_recipe"].get("calibration_presentation")
            != "fixed-calibration-presentation"
            or decisions["D15_terminology_library"].get("resource_role")
            != "terminology-understanding-library/v1"
            or decisions["D15_terminology_library"].get("render_policy")
            != "category-free-terminology-evidence/v1"
            or decisions["D15_terminology_library"].get("task_label_visibility")
            != "absent"
            or decisions["D15_terminology_library"].get("forbidden_entry_fields")
            != [
                "annotation_count",
                "categories",
                "category",
                "category_counts",
                "category_purity",
                "hate_count",
                "hate_precision",
                "hateful",
                "label",
                "labels",
                "log_odds",
                "non_hate_count",
                "nonhate_penalty",
                "primary_category",
                "targeted_group",
            ]
            or decisions["D15_terminology_library"].get(
                "unknown_term_escalation"
            )
            != [
                "model-self-explanation",
                "versioned-web-evidence",
                "human-terminology-queue",
            ]
            or decisions["D15_terminology_library"].get(
                "live_web_during_formal_inference"
            )
            is not False
        ):
            raise ValueError("frozen P0 decision differs from implementation contract")
        digest = sha256_file(path)
    except Exception:
        check = _check(
            check_id="decision.register",
            area="decision",
            phase="p0-prerequisite",
            status="FAIL",
            reason_code="decision-register-invalid-or-not-frozen",
            ref_path=logical,
        )
        return check, {"logical_path": logical, "sha256": None, "frozen": False}
    check = _check(
        check_id="decision.register",
        area="decision",
        phase="p0-prerequisite",
        status="PASS",
        reason_code="frozen-decision-register-valid",
        ref_path=logical,
    )
    return check, {"logical_path": logical, "sha256": digest, "frozen": True}


def _validate_data_audit_target(
    target: Path, context: ValidationContext
) -> dict[str, int]:
    override = context.validator_overrides.get("audit")
    if override is not None:
        result = override(target, workspace_root=context.workspace_root)
        if not isinstance(result, Mapping):
            raise Stage1P0ValidationError(
                "data audit validator override returned a non-object"
            )
        return {
            "blocking_issue_count": int(result["blocking_issue_count"]),
            "warning_count": int(result["warning_count"]),
        }
    from data.stage1_data_audit_validation import validate_data_audit_target

    replay = validate_data_audit_target(
        target, workspace_root=context.workspace_root
    )
    return {
        "blocking_issue_count": int(replay["blocking_issue_count"]),
        "warning_count": int(replay["warning_count"]),
    }


def _validate_data_audit_target_structural_legacy(
    target: Path, context: ValidationContext
) -> dict[str, int]:
    """Retained only as an internal comparison aid for focused tests."""
    expected = {
        "adjudication_rubric.md",
        "adjudication_rubric.meta.json",
        "adjudication_template.jsonl",
        "audit.meta.json",
        "audit_report.json",
        "config.resolved.json",
        "issues.jsonl",
        "payload_manifest.json",
        "provenance.json",
        "source_inventory.json",
        "split_manifest.proposed.json",
    }
    actual = {path.name for path in target.iterdir() if path.is_file()}
    if actual != expected:
        raise Stage1P0ValidationError("data audit file set mismatch")
    meta = load_json(target / "audit.meta.json")
    rubric_meta = load_json(target / "adjudication_rubric.meta.json")
    report = load_json(target / "audit_report.json")
    provenance = load_json(target / "provenance.json")
    issues = load_jsonl(target / "issues.jsonl")
    template = load_jsonl(target / "adjudication_template.jsonl")
    validate_json_schema(meta, context.schema_root / "stage1_data_audit_v1.schema.json")
    validate_json_schema(
        rubric_meta,
        context.schema_root / "stage1_data_rubric_meta_v1.schema.json",
    )
    for issue in issues:
        validate_json_schema(
            issue,
            context.schema_root / "stage1_data_audit_issue_v1.schema.json",
        )
    audit_id = meta["data_audit_id"]
    if target.name != audit_id or "daudit-" + canonical_sha256(meta["audit_id_inputs"]) != audit_id:
        raise Stage1P0ValidationError("data audit ID mismatch")
    issue_ids = [row["issue_id"] for row in issues]
    if len(issue_ids) != len(set(issue_ids)) or canonical_sha256(issue_ids) != meta[
        "ordered_issue_ids_sha256"
    ]:
        raise Stage1P0ValidationError("data audit issue frame mismatch")
    if meta["blocking_issue_count"] != len(issues):
        raise Stage1P0ValidationError("data audit issue count mismatch")
    if any(row.get("data_audit_id") != audit_id for row in issues):
        raise Stage1P0ValidationError("data audit issue lineage mismatch")
    if [row.get("issue_id") for row in template] != issue_ids:
        raise Stage1P0ValidationError("data adjudication template frame mismatch")
    rubric_body_hash = sha256_file(target / "adjudication_rubric.md")
    rubric_meta_hash = canonical_sha256(rubric_meta)
    id_inputs = meta["audit_id_inputs"]
    if (
        rubric_meta["rubric_body_sha256"] != rubric_body_hash
        or id_inputs["rubric_body_sha256"] != rubric_body_hash
        or id_inputs["rubric_meta_sha256"] != rubric_meta_hash
        or provenance.get("rubric_body_sha256") != rubric_body_hash
        or provenance.get("rubric_meta_sha256") != rubric_meta_hash
        or provenance.get("data_audit_id") != audit_id
        or report.get("data_audit_id") != audit_id
        or report.get("blocking", {}).get("count") != len(issues)
        or report.get("warnings", {}).get("count") != meta["warning_count"]
    ):
        raise Stage1P0ValidationError("data audit rubric/report hash chain mismatch")
    return {
        "blocking_issue_count": len(issues),
        "warning_count": int(meta["warning_count"]),
    }


def _data_audit_check(
    context: ValidationContext,
) -> tuple[dict[str, Any], dict[str, int] | None, Path | None]:
    requirement = ArtifactRequirement(
        check_id="data.audit",
        area="data",
        phase="p0-prerequisite",
        ref_names=("data_audit_ref.json",),
        expected_kinds=("data-audit",),
        validator_key="audit-custom",
    )
    ref_path = _find_ref(context, requirement.ref_names)
    expected_ref = "exps/causal_context/stage1_p0/refs/data_audit_ref.json"
    if ref_path is None:
        return (
            _check(
                check_id=requirement.check_id,
                area=requirement.area,
                phase=requirement.phase,
                status="PENDING",
                reason_code="required-ref-missing",
                ref_path=expected_ref,
            ),
            None,
            None,
        )
    logical = _logical_path(ref_path, context.workspace_root)
    before_ref = sha256_file(ref_path)
    try:
        locator, target = resolve_locator_ref(ref_path, "data-audit")
        target = target.resolve()
        target.relative_to(context.workspace_root)
        descriptor = _safe_artifact_descriptor(locator, target, context)
        context.resolved_targets.add(target)
        before_payload = validate_payload_manifest(target)
        metrics = _validate_data_audit_target(target, context)
        if sha256_file(ref_path) != before_ref or validate_payload_manifest(target) != before_payload:
            raise Stage1P0ValidationError("audit changed during validation")
    except Exception:
        return (
            _check(
                check_id=requirement.check_id,
                area=requirement.area,
                phase=requirement.phase,
                status="FAIL",
                reason_code="data-audit-hash-chain-invalid",
                ref_path=logical,
            ),
            None,
            None,
        )
    return (
        _check(
            check_id=requirement.check_id,
            area=requirement.area,
            phase=requirement.phase,
            status="PASS",
            reason_code="data-audit-hash-chain-valid",
            ref_path=logical,
            artifact=descriptor,
            metrics=metrics,
        ),
        metrics,
        target,
    )


def _row_looks_complete(row: Mapping[str, Any]) -> bool:
    return (
        row.get("decision") in {"accepted", "corrected"}
        and isinstance(row.get("reason_code"), str)
        and bool(row["reason_code"])
        and isinstance(row.get("reason"), str)
        and bool(row["reason"])
        and isinstance(row.get("reviewer_id"), str)
        and bool(row["reviewer_id"])
        and isinstance(row.get("reviewed_at"), str)
        and bool(row["reviewed_at"])
    )


def _human_review_check(
    context: ValidationContext,
    audit_target: Path | None,
    data_check: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if data_check is not None and data_check.get("status") == "PASS":
        return _check(
            check_id="data.human_review",
            area="human-review",
            phase="p0-prerequisite",
            status="PASS",
            reason_code="signed-adjudication-frozen-in-data-target",
            ref_path="exps/causal_context/stage1_p0/refs/data_ref.json",
        )
    adjudication = context.stage1_root / "review_inputs" / "data_adjudication.jsonl"
    declaration = context.stage1_root / "review_inputs" / "data_reviewer_declaration.json"
    expected_count = 0
    expected_ids: list[str] = []
    if audit_target is not None:
        try:
            issues = load_jsonl(audit_target / "issues.jsonl")
            expected_ids = [str(row["issue_id"]) for row in issues]
            expected_count = len(expected_ids)
        except Exception:
            pass
    rows: list[dict[str, Any]] = []
    if adjudication.is_file():
        try:
            rows = load_jsonl(adjudication)
        except Exception:
            rows = []
    completed = sum(_row_looks_complete(row) for row in rows)
    row_ids = [str(row.get("issue_id", "")) for row in rows]
    exact_frame = bool(expected_ids) and row_ids == expected_ids
    declaration_present = declaration.is_file()
    attested = False
    if declaration_present:
        try:
            document = load_json(declaration)
            attested = document.get("attestation_confirmed") is True
        except Exception:
            attested = False
    metrics = {
        "expected_issue_count": expected_count,
        "review_row_count": len(rows),
        "completed_review_count": completed,
        "unresolved_review_count": max(expected_count - completed, 0),
        "exact_issue_frame": exact_frame,
        "reviewer_declaration_present": declaration_present,
        "attestation_confirmed": attested,
    }
    if expected_count == 0 and audit_target is None:
        status, reason = "PENDING", "data-audit-required-before-human-review"
    elif completed < expected_count or not exact_frame or not declaration_present or not attested:
        status, reason = "BLOCKED", "human-adjudication-incomplete-or-unsigned"
    else:
        status, reason = "PENDING", "signed-review-awaits-data-finalization"
    return _check(
        check_id="data.human_review",
        area="human-review",
        phase="p0-prerequisite",
        status=status,
        reason_code=reason,
        ref_path="exps/causal_context/stage1_p0/review_inputs/data_adjudication.jsonl",
        metrics=metrics,
    )


def _cf_review_workspace_check(
    context: ValidationContext,
    proposal_check: Mapping[str, Any],
    blind_review_check: Mapping[str, Any],
) -> dict[str, Any]:
    requirement = ArtifactRequirement(
        check_id="counterfactual.review",
        area="counterfactual",
        phase="p0-prerequisite",
        ref_names=("cf_review_ref.json",),
        expected_kinds=("cf-review",),
        validator_key="cf_review",
        missing_status="BLOCKED",
        expected_scientific_eligible=context.mode == "formal-readiness",
    )
    if (
        proposal_check.get("status") != "PASS"
        or blind_review_check.get("status") != "PASS"
    ):
        return _check(
            check_id=requirement.check_id,
            area=requirement.area,
            phase=requirement.phase,
            status="PENDING",
            reason_code="cf-proposal-and-blind-review-required-before-review",
            ref_path="exps/causal_context/stage1_p0/refs/cf_review_ref.json",
        )
    if _find_ref(context, requirement.ref_names) is not None:
        return _artifact_check(requirement, context)
    rows_path = context.stage1_root / "review_inputs" / "dev_cf_review.jsonl"
    declaration_path = context.stage1_root / "review_inputs" / "dev_reviewer_declaration.json"
    row_count = 0
    completed = 0
    if rows_path.is_file():
        try:
            rows = load_jsonl(rows_path)
            row_count = len(rows)
            completed = sum(
                row.get("decision") in {"pass", "reject", "not_required"}
                and bool(row.get("reviewer_id"))
                for row in rows
            )
        except Exception:
            pass
    attested = False
    if declaration_path.is_file():
        try:
            attested = load_json(declaration_path).get("attestation_confirmed") is True
        except Exception:
            pass
    return _check(
        check_id=requirement.check_id,
        area=requirement.area,
        phase=requirement.phase,
        status="BLOCKED",
        reason_code="cf-human-review-incomplete-or-not-finalized",
        ref_path="exps/causal_context/stage1_p0/refs/cf_review_ref.json",
        metrics={
            "review_row_count": row_count,
            "completed_review_count": completed,
            "reviewer_declaration_present": declaration_path.is_file(),
            "attestation_confirmed": attested,
        },
    )


def _normalised_path_tokens(name: str) -> tuple[tuple[str, ...], str]:
    folded = unicodedata.normalize("NFKC", name).casefold()
    tokens = tuple(re.findall(r"[a-z0-9]+", folded))
    return tokens, "".join(tokens)


def _sealed_component_category(name: str) -> str | None:
    """Classify a test-bearing path component without opening the entry."""

    tokens, compact = _normalised_path_tokens(name)
    if not compact:
        return None
    domains = {
        category: tuple(sorted(aliases, key=len, reverse=True))
        for category, aliases in _SEALED_PATH_DOMAINS.items()
    }

    def category_for_domains(values: Sequence[str]) -> str | None:
        value_set = set(values)
        for category, aliases in domains.items():
            if value_set.intersection(aliases):
                return category
        return None

    token_category = category_for_domains(tokens)
    if "test" in tokens:
        return token_category or "artifact"
    if "tests" in tokens and token_category is not None:
        return token_category
    if any(marker in tokens for marker in ("sealed", "holdout", "heldout")):
        if token_category is not None or len(tokens) == 1:
            return token_category or "artifact"

    if compact in _SEALED_PATH_MARKERS:
        return "artifact"
    metadata_tails = (
        "artifact",
        "artifacts",
        "target",
        "targets",
        "ref",
        "refs",
        "reference",
        "references",
        "sidecar",
        "sidecars",
    )
    for marker in _SEALED_PATH_MARKERS:
        if compact.startswith(marker):
            remainder = compact[len(marker) :]
            for category, aliases in domains.items():
                if any(remainder.startswith(alias) for alias in aliases):
                    return category
            if any(remainder.startswith(tail) for tail in metadata_tails):
                return "artifact"
        for category, aliases in domains.items():
            if any(compact.startswith(alias + marker) for alias in aliases):
                return category
    return None


def _allowed_preseal_test_path(parts: tuple[str, ...], *, is_dir: bool) -> bool:
    """Allow only the frozen data lifecycle's explicitly specified raw split file."""

    return (
        not is_dir
        and len(parts) == 3
        and parts[0] == "data"
        and re.fullmatch(r"data-[0-9a-f]{64}", parts[1]) is not None
        and parts[2] == "test.json"
    )


def _sealed_path_category(parts: tuple[str, ...], *, is_dir: bool) -> str | None:
    if _allowed_preseal_test_path(parts, is_dir=is_dir):
        return None
    for component in parts:
        category = _sealed_component_category(component)
        if category is not None:
            return category
    return None


def _discovery_label(
    parts: tuple[str, ...], *, kind: str, category: str = "artifact"
) -> str:
    """Return a stable non-disclosing label for an untrusted relative path."""

    if len(parts) == 2 and parts[0] == "refs" and parts[1] in SEALED_REF_NAMES:
        return parts[1]
    if len(parts) >= 2 and parts[0:2] == ("refs", "test"):
        return "test/"
    digest = hashlib.sha256(
        b"stage1-sealed-boundary-path/v1\0" + os.fsencode("/".join(parts))
    ).hexdigest()
    return f"test/{kind}/{category}/{digest}"


def _is_boundary_metadata_file(parts: tuple[str, ...]) -> bool:
    if not parts or Path(parts[-1]).suffix.casefold() != ".json":
        return False
    tokens, _compact = _normalised_path_tokens(parts[-1])
    return parts[0].casefold() == "refs" or bool(
        {"ref", "refs", "reference", "references", "dependency", "dependencies"}
        .intersection(tokens)
    )


def _scan_sealed_paths(
    context: ValidationContext,
) -> tuple[set[str], list[tuple[Path, tuple[str, ...]]]]:
    """Walk only directory metadata, never following links or opening targets."""

    root = context.stage1_root
    discoveries: set[str] = set()
    metadata_files: list[tuple[Path, tuple[str, ...]]] = []
    if not os.path.lexists(root):
        return discoveries, metadata_files
    if root.is_symlink():
        discoveries.add(
            _discovery_label(("stage1_p0",), kind="symlink", category="artifact")
        )
        return discoveries, metadata_files
    if not root.is_dir():
        discoveries.add(
            _discovery_label(("stage1_p0",), kind="special", category="artifact")
        )
        return discoveries, metadata_files

    pending: list[tuple[Path, tuple[str, ...]]] = [(root, ())]
    while pending:
        directory, prefix = pending.pop()
        try:
            with os.scandir(directory) as iterator:
                entries = sorted(iterator, key=lambda item: os.fsencode(item.name))
        except OSError:
            discoveries.add(
                _discovery_label(
                    prefix or ("stage1_p0",),
                    kind="unreadable",
                    category="artifact",
                )
            )
            continue
        for entry in entries:
            parts = prefix + (entry.name,)
            if entry.is_symlink():
                discoveries.add(
                    _discovery_label(parts, kind="symlink", category="artifact")
                )
                continue
            try:
                is_dir = entry.is_dir(follow_symlinks=False)
                is_file = entry.is_file(follow_symlinks=False)
            except OSError:
                discoveries.add(
                    _discovery_label(parts, kind="unreadable", category="artifact")
                )
                continue
            category = _sealed_path_category(parts, is_dir=is_dir)
            if category is not None:
                discoveries.add(_discovery_label(parts, kind="path", category=category))
                continue
            entry_path = directory / entry.name
            if is_dir:
                pending.append((entry_path, parts))
            elif is_file:
                if _is_boundary_metadata_file(parts):
                    metadata_files.append((entry_path, parts))
            else:
                discoveries.add(
                    _discovery_label(parts, kind="special", category="artifact")
                )
    return discoveries, metadata_files


def _read_boundary_metadata(path: Path) -> Any:
    """Read a bounded regular metadata file without following a final symlink."""

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size > _SEALED_METADATA_MAX_BYTES:
            raise Stage1P0ValidationError("boundary metadata is not a bounded regular file")
        chunks: list[bytes] = []
        remaining = _SEALED_METADATA_MAX_BYTES + 1
        while remaining:
            chunk = os.read(descriptor, min(65536, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        payload = b"".join(chunks)
        after = os.fstat(descriptor)
        current = os.stat(path, follow_symlinks=False)
        if (
            len(payload) > _SEALED_METADATA_MAX_BYTES
            or before.st_dev != after.st_dev
            or before.st_ino != after.st_ino
            or before.st_size != after.st_size
            or before.st_mtime_ns != after.st_mtime_ns
            or after.st_dev != current.st_dev
            or after.st_ino != current.st_ino
            or after.st_size != current.st_size
            or after.st_mtime_ns != current.st_mtime_ns
        ):
            raise Stage1P0ValidationError("boundary metadata changed while being read")
        return json.loads(payload.decode("utf-8"))
    finally:
        os.close(descriptor)


def _metadata_target_parts(raw_path: str, context: ValidationContext) -> tuple[str, ...]:
    """Project a locator path onto the Stage 1 root without resolving links."""

    try:
        path = Path(os.path.normpath(raw_path))
    except (OSError, ValueError):
        return ()
    if path.is_absolute():
        try:
            return path.relative_to(context.stage1_root).parts
        except ValueError:
            return ("test_external_target",)
    parts = path.parts
    stage1_prefix = ("exps", "causal_context", "stage1_p0")
    for index in range(len(parts) - len(stage1_prefix) + 1):
        if parts[index : index + len(stage1_prefix)] == stage1_prefix:
            return parts[index + len(stage1_prefix) :]
    if ".." in parts:
        return ("test_escaping_target",)
    return parts


def _metadata_sealed_category(
    value: Any, context: ValidationContext
) -> str | None:
    """Inspect only locator/dependency metadata for sealed lineage markers."""

    pending = [value]
    visited = 0
    while pending:
        current = pending.pop()
        visited += 1
        if visited > 100_000:
            return "artifact"
        if isinstance(current, Mapping):
            for raw_key, nested in current.items():
                _, key = _normalised_path_tokens(str(raw_key))
                if key in {"artifactkind", "kind"} and isinstance(nested, str):
                    category = _sealed_component_category(nested)
                    if category is not None:
                        return category
                if key in {"split", "sourcesplit"} and isinstance(nested, str):
                    if unicodedata.normalize("NFKC", nested).casefold() == "test":
                        return "artifact"
                if key in {"sealingstatus", "sealstatus"} and isinstance(nested, str):
                    marker = "".join(_normalised_path_tokens(nested)[0])
                    if marker in {"sealedtest", "testsealed"}:
                        return "artifact"
                if key in {"targetpath", "logicalrepopath", "logicalpath"} and isinstance(
                    nested, str
                ):
                    path_parts = _metadata_target_parts(nested, context)
                    category = _sealed_path_category(path_parts, is_dir=True)
                    if category is not None:
                        return category
                if isinstance(nested, (Mapping, list, tuple)):
                    pending.append(nested)
        elif isinstance(current, (list, tuple)):
            pending.extend(current)
    return None


def _metadata_discovery_label(parts: tuple[str, ...], category: str) -> str:
    if (
        len(parts) == 2
        and parts[0] == "refs"
        and parts[1] in _PUBLIC_LIFECYCLE_REF_NAMES
    ):
        return f"test/{parts[1]}"
    return _discovery_label(parts, kind="ref", category=category)


def _sealed_boundary_check(context: ValidationContext) -> tuple[dict[str, Any], dict[str, Any]]:
    discoveries, metadata_files = _scan_sealed_paths(context)
    for metadata_path, parts in sorted(metadata_files, key=lambda item: item[1]):
        try:
            metadata = _read_boundary_metadata(metadata_path)
        except Exception:
            discoveries.add(
                _discovery_label(parts, kind="unreadable", category="artifact")
            )
            continue
        category = _metadata_sealed_category(metadata, context)
        if category is not None:
            discoveries.add(_metadata_discovery_label(parts, category))

    discovered = sorted(discoveries)
    if discovered:
        check = _check(
            check_id="sealed.boundary",
            area="sealed-test",
            phase="sealed-boundary",
            status="FAIL",
            reason_code="sealed-test-boundary-violation",
            ref_path="exps/causal_context/stage1_p0",
            metrics={
                "discovered_ref_names": discovered,
                "discovered_entry_count": len(discovered),
                "scan_scope": "stage1-path-and-reference-metadata",
            },
        )
    else:
        check = _check(
            check_id="sealed.boundary",
            area="sealed-test",
            phase="sealed-boundary",
            status="PASS",
            reason_code="sealed-test-not-executed",
            ref_path="exps/causal_context/stage1_p0",
            metrics={
                "discovered_entry_count": 0,
                "scan_scope": "stage1-path-and-reference-metadata",
            },
        )
    return check, {
        "execution_performed": False,
        "target_content_read": False,
        "policy": "synthetic-only-not-executed",
        "discovered_ref_names": discovered,
    }


def _requirements(mode: str) -> tuple[ArtifactRequirement, ...]:
    dev_context_names = (
        (
            "legacy_smoke_context_ref.json",
            "smoke_context_ref.json",
            "context_ref.json",
        )
        if mode == "engineering-smoke"
        else ("dev_context_ref.json",)
    )
    smoke_control_names = (
        (
            "legacy_smoke_control_diagnostic_ref.json",
            "smoke_control_ref.json",
            "control_ref.json",
        )
        if mode == "engineering-smoke"
        else ("control_ref.json",)
    )
    cf_proposal_names = (
        ("smoke_cf_proposal_ref.json", "cf_proposal_ref.json")
        if mode == "engineering-smoke"
        else ("cf_proposal_ref.json",)
    )
    cf_blind_review_names = (
        ("smoke_cf_blind_review_ref.json", "cf_blind_review_ref.json")
        if mode == "engineering-smoke"
        else ("cf_blind_review_ref.json",)
    )
    plan_names = (
        ("smoke_plan_ref.json",)
        if mode == "engineering-smoke"
        else ("training_plan_ref.json",)
    )
    registry_names = (
        ("smoke_model_registry_ref.json",)
        if mode == "engineering-smoke"
        else ("model_registry_ref.json",)
    )
    expected_scope = "engineering-smoke" if mode == "engineering-smoke" else "formal"
    generation_expected_scope = "engineering" if mode == "engineering-smoke" else "formal"
    expected_scientific = mode == "formal-readiness"
    partition_requirement = (
        ArtifactRequirement(
            "data.train_partition",
            "data",
            "p0-prerequisite",
            ("train_partition_ref.json",),
            ("train-partition",),
            "partition",
        ),
    ) if mode == "formal-readiness" else ()
    train_context_requirement = (
        ArtifactRequirement(
            "context.train",
            "context",
            "p0-prerequisite",
            ("train_context_ref.json",),
            ("context",),
            "context",
            expected_split="train",
            expected_scientific_eligible=True,
        ),
    ) if mode == "formal-readiness" else ()
    common_requirements = (
        ArtifactRequirement(
            "data.blind_review",
            "human-review",
            "p0-prerequisite",
            ("data_blind_review_ref.json",),
            ("data-blind-review",),
            "data_review",
            missing_status="BLOCKED",
        ),
        ArtifactRequirement(
            "data.normalized",
            "data",
            "p0-prerequisite",
            ("data_ref.json",),
            ("data",),
            "data",
        ),
    ) + partition_requirement + (
        ArtifactRequirement(
            "data.lexicon",
            "data",
            "p0-prerequisite",
            ("lexicon_ref.json",),
            ("lexicon",),
            "lexicon",
        ),
        ArtifactRequirement(
            "infrastructure.environment",
            "infrastructure",
            "p0-prerequisite",
            ("environment_ref.json",),
            ("stage1-environment",),
            "environment",
        ),
        ArtifactRequirement(
            "tests.receipt",
            "tests",
            "p0-prerequisite",
            ("verification_receipt_ref.json",),
            ("stage1-p0-verification-receipt",),
            "test_receipt",
            missing_status="BLOCKED",
        ),
        ArtifactRequirement(
            "infrastructure.base_model",
            "infrastructure",
            "p0-prerequisite",
            ("base_model_ref.json",),
            ("stage1-model",),
            "model",
        ),
        ArtifactRequirement(
            "smoke.determinism",
            "downstream",
            "engineering-smoke",
            ("legacy_smoke_generation_ref.json",),
            ("generation-run",),
            "generation",
            expected_scope="engineering",
            expected_split="dev",
            expected_scientific_eligible=False,
            expected_sealing_status="unsealed-dev",
        ),
    ) + train_context_requirement + (
        ArtifactRequirement(
            "context.dev",
            "context",
            "p0-prerequisite",
            dev_context_names,
            ("context",),
            "context",
            expected_split="dev",
            expected_scientific_eligible=expected_scientific,
        ),
        ArtifactRequirement(
            "training.evidence",
            "training",
            "p0-prerequisite",
            ("training_evidence_ref.json",),
            ("training-evidence",),
            "evidence",
        ),
        ArtifactRequirement(
            "training.plan",
            "training",
            "p0-prerequisite",
            plan_names,
            ("training-plan",),
            "plan",
            expected_scope=expected_scope,
        ),
        ArtifactRequirement(
            "training.schedule",
            "training",
            "p0-prerequisite",
            ("schedule_ref.json",),
            ("training-schedule",),
            "schedule",
        ),
        ArtifactRequirement(
            "control.dev",
            "control",
            "p0-prerequisite",
            smoke_control_names,
            ("control",),
            "control",
            expected_split="dev",
        ),
        ArtifactRequirement(
            "counterfactual.proposal",
            "counterfactual",
            "p0-prerequisite",
            cf_proposal_names,
            ("cf-proposal",),
            "cf_proposal",
            expected_split="dev",
            expected_scientific_eligible=expected_scientific,
        ),
        ArtifactRequirement(
            "counterfactual.blind_review",
            "counterfactual",
            "p0-prerequisite",
            cf_blind_review_names,
            ("cf-blind-review",),
            "cf_blind_review",
            missing_status="BLOCKED",
        ),
        ArtifactRequirement(
            "counterfactual.final",
            "counterfactual",
            "p0-prerequisite",
            ("cf_ref.json",),
            ("counterfactual",),
            "cf",
            expected_split="dev",
            expected_scientific_eligible=expected_scientific,
        ),
    )
    legacy_requirement = (
        ArtifactRequirement(
            "model.legacy_smoke",
            "model",
            "engineering-smoke",
            ("legacy_model_ref.json",),
            ("stage1-model",),
            "model",
        ),
    ) if mode == "engineering-smoke" else ()
    downstream_requirements = (
        ArtifactRequirement(
            "model.registry",
            "model",
            "engineering-smoke" if mode == "engineering-smoke" else "post-training-formal",
            registry_names,
            ("stage1-model-registry",),
            "registry",
            expected_scope=expected_scope,
            expected_scientific_eligible=expected_scientific,
        ),
        ArtifactRequirement(
            "downstream.generation",
            "downstream",
            "engineering-smoke" if mode == "engineering-smoke" else "post-training-formal",
            ("generation_run_ref.json",),
            ("generation-run",),
            "generation",
            expected_scope=generation_expected_scope,
            expected_split="dev",
            expected_scientific_eligible=expected_scientific,
            expected_sealing_status="unsealed-dev",
        ),
        ArtifactRequirement(
            "downstream.evaluation",
            "downstream",
            "engineering-smoke" if mode == "engineering-smoke" else "post-training-formal",
            ("evaluation_ref.json",),
            ("evaluation",),
            "evaluation",
            expected_scope=expected_scope,
            expected_split="dev",
            expected_scientific_eligible=expected_scientific,
            expected_sealing_status="unsealed-dev",
        ),
        ArtifactRequirement(
            "downstream.margin",
            "downstream",
            "engineering-smoke" if mode == "engineering-smoke" else "post-training-formal",
            ("margin_ref.json",),
            ("margin",),
            "margin",
            expected_scope=expected_scope,
            expected_split="dev",
            expected_scientific_eligible=expected_scientific,
            expected_sealing_status="unsealed-dev",
        ),
        ArtifactRequirement(
            "downstream.analysis",
            "downstream",
            "engineering-smoke" if mode == "engineering-smoke" else "post-training-formal",
            ("analysis_ref.json",),
            ("analysis",),
            "analysis",
            expected_scope=expected_scope,
            expected_split="dev",
            expected_scientific_eligible=expected_scientific,
        ),
    )
    return common_requirements + legacy_requirement + downstream_requirements


def _dependency_identity(value: Any) -> tuple[str, str, str]:
    """Return the content identity shared by locator and dependency refs."""

    if not isinstance(value, Mapping):
        raise Stage1P0ValidationError("artifact dependency is not an object")
    artifact_kind = value.get("artifact_kind")
    artifact_id = value.get("artifact_id")
    payload_hash = value.get("payload_manifest_sha256")
    if (
        not isinstance(artifact_kind, str)
        or not SAFE_IDENTIFIER_RE.fullmatch(artifact_kind)
        or not isinstance(artifact_id, str)
        or not SAFE_IDENTIFIER_RE.fullmatch(artifact_id)
        or not isinstance(payload_hash, str)
        or not SHA256_RE.fullmatch(payload_hash)
    ):
        raise Stage1P0ValidationError("artifact dependency identity is malformed")
    return artifact_kind, artifact_id, payload_hash


def _context_policy_projection(target: Path) -> dict[str, Any]:
    """Project the split-invariant policy from an already validated context."""

    config = load_json(target / "config.resolved.json")
    prepared = load_json(target / "prepared_bundle.meta.json")
    meta_paths = sorted(target.glob("context_manifest.*.meta.json"))
    if (
        not isinstance(config, Mapping)
        or not isinstance(prepared, Mapping)
        or len(meta_paths) != 1
    ):
        raise Stage1P0ValidationError(
            "formal context lacks canonical policy payloads"
        )
    meta = load_json(meta_paths[0])
    retrieval = prepared.get("retrieval_provenance")
    id_inputs = meta.get("id_inputs") if isinstance(meta, Mapping) else None
    if not isinstance(retrieval, Mapping) or not isinstance(id_inputs, Mapping):
        raise Stage1P0ValidationError(
            "formal context lacks retrieval/builder policy lineage"
        )
    scorer = retrieval.get("scorer")
    builder_hash = id_inputs.get("builder_code_sha256")
    if (
        not isinstance(scorer, Mapping)
        or not isinstance(builder_hash, str)
        or not SHA256_RE.fullmatch(builder_hash)
    ):
        raise Stage1P0ValidationError(
            "formal context policy lineage is malformed"
        )
    return {
        "context_config_sha256": canonical_sha256(config),
        "retrieval_engine": {
            "schema_version": retrieval.get("schema_version"),
            "policy_version": retrieval.get("policy_version"),
            "scorer": dict(scorer),
        },
        "context_builder_code_sha256": builder_hash,
    }


def _formal_exact_dependency_chain_check(
    context: ValidationContext,
    checks_by_id: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Prove that the selected formal refs form one exact pre-training chain.

    Individual lifecycle validators deeply validate their own upstream targets.
    This additional gate prevents a set of independently valid top-level refs
    from silently selecting different data/partition/context/plan branches.
    """

    if context.mode != "formal-readiness":
        return _check(
            check_id="formal.lineage",
            area="training",
            phase="p0-prerequisite",
            status="PASS",
            reason_code="formal-lineage-not-applicable-to-engineering-smoke",
        )

    required_ids = (
        "data.blind_review",
        "data.normalized",
        "data.train_partition",
        "data.lexicon",
        "infrastructure.environment",
        "infrastructure.base_model",
        "context.train",
        "context.dev",
        "training.evidence",
        "training.plan",
        "training.schedule",
        "control.dev",
        "counterfactual.proposal",
        "counterfactual.blind_review",
        "counterfactual.review",
        "counterfactual.final",
        "model.registry",
    )
    if any(checks_by_id.get(item, {}).get("status") != "PASS" for item in required_ids):
        return _check(
            check_id="formal.lineage",
            area="training",
            phase="p0-prerequisite",
            status="PENDING",
            reason_code="formal-exact-dependency-chain-incomplete",
        )

    try:
        selected = {
            item: context.artifact_dependencies[item] for item in required_ids
        }
        targets = {item: context.artifact_targets[item] for item in required_ids}

        def require_file(
            consumer: str, filename: str, producer: str
        ) -> Mapping[str, Any]:
            document = load_json(targets[consumer] / filename)
            if _dependency_identity(document) != _dependency_identity(selected[producer]):
                raise ValueError(
                    f"{consumer}/{filename} does not consume selected {producer}"
                )
            return document

        def require_exact_file(
            consumer: str, filename: str, producer: str
        ) -> Mapping[str, Any]:
            document = require_file(consumer, filename, producer)
            if document != selected[producer]:
                raise ValueError(
                    f"{consumer}/{filename} is not the exact selected {producer} ref"
                )
            return document

        require_exact_file(
            "data.normalized",
            "data_blind_review_ref.json",
            "data.blind_review",
        )
        require_file("data.train_partition", "data_ref.json", "data.normalized")
        require_file("data.lexicon", "data_ref.json", "data.normalized")
        require_file(
            "data.lexicon", "train_partition_ref.json", "data.train_partition"
        )
        require_file("context.train", "data_ref.json", "data.normalized")
        require_file(
            "context.train", "train_partition_ref.json", "data.train_partition"
        )
        require_file("context.train", "lexicon_ref.json", "data.lexicon")
        require_file("context.dev", "data_ref.json", "data.normalized")
        require_file(
            "context.dev", "train_partition_ref.json", "data.train_partition"
        )
        require_file("context.dev", "lexicon_ref.json", "data.lexicon")
        if _context_policy_projection(
            targets["context.train"]
        ) != _context_policy_projection(targets["context.dev"]):
            raise ValueError(
                "selected train/dev contexts do not share one frozen policy"
            )
        require_file("training.evidence", "context_ref.json", "context.train")
        require_file(
            "training.evidence",
            "train_partition_ref.json",
            "data.train_partition",
        )
        require_file(
            "training.evidence",
            "base_model_ref.json",
            "infrastructure.base_model",
        )
        require_file(
            "training.plan", "training_evidence_ref.json", "training.evidence"
        )
        require_file(
            "training.plan", "train_partition_ref.json", "data.train_partition"
        )
        require_file(
            "training.plan", "base_model_ref.json", "infrastructure.base_model"
        )
        require_file(
            "training.plan", "environment_ref.json", "infrastructure.environment"
        )
        require_file(
            "training.schedule", "training_plan_ref.json", "training.plan"
        )
        require_file(
            "training.schedule",
            "training_evidence_ref.json",
            "training.evidence",
        )
        require_file(
            "training.schedule",
            "train_partition_ref.json",
            "data.train_partition",
        )
        require_file("control.dev", "context_ref.json", "context.dev")
        require_file(
            "counterfactual.proposal", "context_ref.json", "context.dev"
        )
        require_exact_file(
            "counterfactual.blind_review",
            "proposal_ref.json",
            "counterfactual.proposal",
        )
        require_exact_file(
            "counterfactual.review",
            "proposal_ref.json",
            "counterfactual.proposal",
        )
        require_exact_file(
            "counterfactual.review",
            "cf_blind_review_ref.json",
            "counterfactual.blind_review",
        )
        require_exact_file(
            "counterfactual.final",
            "context_ref.json",
            "context.dev",
        )
        require_exact_file(
            "counterfactual.final",
            "proposal_ref.json",
            "counterfactual.proposal",
        )
        require_exact_file(
            "counterfactual.final",
            "review_ref.json",
            "counterfactual.review",
        )
        require_file("model.registry", "training_plan_ref.json", "training.plan")

        # A registry binds model refs rather than one schedule ref.  Every
        # formal trained model must nevertheless consume the selected schedule,
        # base and environment branches, not merely some valid siblings.
        registry = load_json(targets["model.registry"] / "registry.json")
        entries = registry.get("entries") if isinstance(registry, Mapping) else None
        if not isinstance(entries, list) or not entries:
            raise ValueError("formal model registry has no entries")
        for entry in entries:
            if not isinstance(entry, Mapping):
                raise ValueError("formal model registry entry is malformed")
            model_dependency = validate_dependency_ref(
                entry.get("model_dependency", {}), expected_kind="stage1-model"
            )
            model_target = resolve_dependency_target(
                model_dependency, context.workspace_root
            )
            for filename, producer in (
                ("training_plan_ref.json", "training.plan"),
                ("schedule_ref.json", "training.schedule"),
                ("base_model_ref.json", "infrastructure.base_model"),
                ("environment_ref.json", "infrastructure.environment"),
            ):
                if _dependency_identity(load_json(model_target / filename)) != (
                    _dependency_identity(selected[producer])
                ):
                    raise ValueError(
                        f"registered model does not consume selected {producer}"
                    )
    except Exception:
        return _check(
            check_id="formal.lineage",
            area="training",
            phase="p0-prerequisite",
            status="FAIL",
            reason_code="formal-exact-dependency-lineage-mismatch",
        )

    return _check(
        check_id="formal.lineage",
        area="training",
        phase="p0-prerequisite",
        status="PASS",
        reason_code="formal-exact-dependency-lineage-valid",
    )


def _downstream_chain_check(
    context: ValidationContext, checks_by_id: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    ids = (
        "downstream.generation",
        "downstream.evaluation",
        "downstream.margin",
        "downstream.analysis",
    )
    phase = (
        "engineering-smoke"
        if context.mode == "engineering-smoke"
        else "post-training-formal"
    )
    statuses = [str(checks_by_id[item]["status"]) for item in ids]
    final_cf_status = str(
        checks_by_id.get("counterfactual.final", {}).get("status", "PENDING")
    )
    if any(status == "FAIL" for status in statuses) or final_cf_status == "FAIL":
        return _check(
            check_id="downstream.chain",
            area="downstream",
            phase=phase,
            status="FAIL",
            reason_code="downstream-artifact-invalid",
        )
    passed_ids = [item for item in ids if checks_by_id[item]["status"] == "PASS"]
    reports = {item: context.semantic_reports[item] for item in passed_ids}
    expected_scopes = {
        "downstream.generation": (
            "engineering" if context.mode == "engineering-smoke" else "formal"
        ),
        "downstream.evaluation": (
            "engineering-smoke" if context.mode == "engineering-smoke" else "formal"
        ),
        "downstream.margin": (
            "engineering-smoke" if context.mode == "engineering-smoke" else "formal"
        ),
        "downstream.analysis": (
            "engineering-smoke" if context.mode == "engineering-smoke" else "formal"
        ),
    }
    expected_scientific = context.mode == "formal-readiness"
    for item, report in reports.items():
        observed_scope = report.get("scope") or report.get("registry_scope")
        if (
            observed_scope != expected_scopes[item]
            or report.get("split") != "dev"
            or report.get("scientific_eligible") is not expected_scientific
            or (
                "sealing_status" in report
                and report.get("sealing_status") != "unsealed-dev"
            )
        ):
            return _check(
                check_id="downstream.chain",
                area="downstream",
                phase=phase,
                status="FAIL",
                reason_code="downstream-artifact-contract-mismatch",
            )

    generation = reports.get("downstream.generation")
    evaluation = reports.get("downstream.evaluation")
    if generation is not None and evaluation is not None and evaluation.get(
        "generation_run_id"
    ) != generation.get("generation_run_id"):
        return _check(
            check_id="downstream.chain",
            area="downstream",
            phase=phase,
            status="FAIL",
            reason_code="downstream-generation-evaluation-lineage-mismatch",
        )

    # Semantic validators prove that each artifact is internally valid.  The
    # readiness gate must additionally prove that the *selected top-level
    # refs* form one exact chain; an independently valid analysis must not be
    # allowed to point at a different evaluation/margin run with the same
    # model/query summary.
    try:
        generation_target = context.artifact_targets.get(
            "downstream.generation"
        )
        evaluation_target = context.artifact_targets.get(
            "downstream.evaluation"
        )
        margin_target = context.artifact_targets.get("downstream.margin")
        analysis_target = context.artifact_targets.get("downstream.analysis")
        generation_dependency = context.artifact_dependencies.get(
            "downstream.generation"
        )
        evaluation_dependency = context.artifact_dependencies.get(
            "downstream.evaluation"
        )
        margin_dependency = context.artifact_dependencies.get(
            "downstream.margin"
        )
        registry_dependency = context.artifact_dependencies.get("model.registry")
        context_dependency = context.artifact_dependencies.get("context.dev")
        control_dependency = context.artifact_dependencies.get("control.dev")
        final_cf_dependency = context.artifact_dependencies.get(
            "counterfactual.final"
        )

        if evaluation_target is not None and generation_dependency is not None:
            if load_json(evaluation_target / "generation_run_ref.json") != generation_dependency:
                raise ValueError("evaluation does not consume selected generation")
        if generation_target is not None and margin_target is not None:
            for filename in (
                "model_registry_ref.json",
                "context_ref.json",
                "control_ref.json",
            ):
                if load_json(generation_target / filename) != load_json(
                    margin_target / filename
                ):
                    raise ValueError("generation/margin dependency differs")
        if checks_by_id["downstream.margin"]["status"] == "PASS":
            if (
                margin_target is None
                or registry_dependency is None
                or context_dependency is None
                or control_dependency is None
                or final_cf_dependency is None
            ):
                raise ValueError("selected margin prerequisites are incomplete")
            for filename, dependency in (
                ("model_registry_ref.json", registry_dependency),
                ("context_ref.json", context_dependency),
                ("control_ref.json", control_dependency),
                ("cf_ref.json", final_cf_dependency),
            ):
                if load_json(margin_target / filename) != dependency:
                    raise ValueError(
                        f"margin {filename} does not consume selected dependency"
                    )
            margin_report = reports.get("downstream.margin")
            if (
                margin_report is None
                or margin_report.get("cf_dependency") != final_cf_dependency
            ):
                raise ValueError(
                    "margin report does not expose the selected final CF dependency"
                )
        if analysis_target is not None:
            if registry_dependency is None or load_json(
                analysis_target / "model_registry_ref.json"
            ) != registry_dependency:
                raise ValueError("analysis does not consume selected registry")
            if evaluation_dependency is not None:
                evaluation_entries = load_json(
                    analysis_target / "evaluation_refs.json"
                )
                if not isinstance(evaluation_entries, list) or {
                    "model_key": reports["downstream.evaluation"].get("model_key"),
                    "dependency": evaluation_dependency,
                } not in evaluation_entries:
                    raise ValueError("analysis does not consume selected evaluation")
            if margin_dependency is not None:
                margin_entries = load_json(analysis_target / "margin_refs.json")
                if not isinstance(margin_entries, list) or {
                    "model_key": reports["downstream.margin"].get("model_key"),
                    "dependency": margin_dependency,
                } not in margin_entries:
                    raise ValueError("analysis does not consume selected margin")
    except Exception:
        return _check(
            check_id="downstream.chain",
            area="downstream",
            phase=phase,
            status="FAIL",
            reason_code="downstream-exact-dependency-lineage-mismatch",
        )

    if generation is not None:
        for item in ("downstream.evaluation", "downstream.margin"):
            report = reports.get(item)
            if report is None:
                continue
            for key in ("model_key", "role", "seed", "query_count"):
                if (
                    key in report
                    and key in generation
                    and report.get(key) != generation.get(key)
                ):
                    return _check(
                        check_id="downstream.chain",
                        area="downstream",
                        phase=phase,
                        status="FAIL",
                        reason_code="downstream-model-or-query-frame-mismatch",
                    )
        analysis = reports.get("downstream.analysis")
        if analysis is not None and analysis.get("query_count") != generation.get(
            "query_count"
        ):
            return _check(
                check_id="downstream.chain",
                area="downstream",
                phase=phase,
                status="FAIL",
                reason_code="downstream-model-or-query-frame-mismatch",
            )

    query_count = generation.get("query_count") if generation is not None else None
    if generation is not None and context.mode == "engineering-smoke" and (
        not isinstance(query_count, int) or not 20 <= query_count <= 50
    ):
        return _check(
            check_id="downstream.chain",
            area="downstream",
            phase="engineering-smoke",
            status="FAIL",
            reason_code="real-dev-smoke-query-count-outside-20-50",
            metrics={"generation_query_count": query_count},
        )
    if final_cf_status != "PASS":
        return _check(
            check_id="downstream.chain",
            area="downstream",
            phase=phase,
            status="PENDING",
            reason_code="downstream-final-counterfactual-not-ready",
            metrics={"generation_query_count": query_count},
        )
    if not all(status == "PASS" for status in statuses):
        return _check(
            check_id="downstream.chain",
            area="downstream",
            phase=phase,
            status="PENDING",
            reason_code="downstream-chain-incomplete",
            metrics={"generation_query_count": query_count},
        )
    return _check(
        check_id="downstream.chain",
        area="downstream",
        phase=phase,
        status="PASS",
        reason_code="downstream-chain-valid",
        metrics={"generation_query_count": query_count},
    )


def _smoke_determinism_check(
    context: ValidationContext, checks_by_id: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    """Require an immutable two-pass real-backend dev smoke proof."""

    source = checks_by_id.get("smoke.determinism", {})
    status = str(source.get("status", "PENDING"))
    if status != "PASS":
        return _check(
            check_id="smoke.determinism_gate",
            area="downstream",
            phase="engineering-smoke",
            status="FAIL" if status == "FAIL" else "PENDING",
            reason_code=(
                "smoke-determinism-artifact-invalid"
                if status == "FAIL"
                else "smoke-determinism-artifact-not-ready"
            ),
            ref_path="exps/causal_context/stage1_p0/refs/legacy_smoke_generation_ref.json",
        )
    report = context.semantic_reports.get("smoke.determinism", {})
    conditions = report.get("ordered_conditions")
    query_count = report.get("query_count")
    executor_backend = report.get("executor_backend")
    executor_id = report.get("executor_id")
    valid = (
        report.get("split") == "dev"
        and report.get("scope") == "engineering"
        and report.get("scientific_eligible") is False
        and report.get("complete_paired_blocks") is True
        and conditions == ["C0", "CL", "CD", "CLD", "PL", "PD"]
        and isinstance(query_count, int)
        and not isinstance(query_count, bool)
        and 20 <= query_count <= 50
        and report.get("execution_repetitions") == 2
        and report.get("exact_rerun_match") is True
        and executor_backend in {"transformers", "vllm"}
        and executor_id
        in {"hf-local-transformers/v1", "vllm-local-registered/v1"}
    )
    if not valid:
        return _check(
            check_id="smoke.determinism_gate",
            area="downstream",
            phase="engineering-smoke",
            status="FAIL",
            reason_code="real-dev-two-pass-determinism-contract-mismatch",
            ref_path="exps/causal_context/stage1_p0/refs/legacy_smoke_generation_ref.json",
            metrics={
                "query_count": query_count,
                "execution_repetitions": report.get("execution_repetitions"),
                "exact_rerun_match": report.get("exact_rerun_match"),
                "executor_backend": executor_backend,
            },
        )
    return _check(
        check_id="smoke.determinism_gate",
        area="downstream",
        phase="engineering-smoke",
        status="PASS",
        reason_code="real-dev-two-pass-greedy-byte-determinism-valid",
        ref_path="exps/causal_context/stage1_p0/refs/legacy_smoke_generation_ref.json",
        metrics={
            "query_count": query_count,
            "execution_repetitions": 2,
            "exact_rerun_match": True,
            "executor_backend": executor_backend,
        },
    )


def _gap_from_check(check: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "check_id": check["check_id"],
        "area": check["area"],
        "phase": check["phase"],
        "status": check["status"],
        "reason_code": check["reason_code"],
        "ref_path": check["ref_path"],
    }


def _validate_report_safety(report: Mapping[str, Any], workspace_root: Path) -> None:
    forbidden_key_fragments = ("secret", "token", "password", "credential", "api_key")
    absolute_workspace = str(workspace_root.resolve())

    def walk(value: Any, key: str = "") -> None:
        lowered = key.lower()
        if any(fragment in lowered for fragment in forbidden_key_fragments):
            raise Stage1P0ValidationError("report contains a secret-like key")
        if isinstance(value, Mapping):
            for nested_key, nested_value in value.items():
                walk(nested_value, str(nested_key))
        elif isinstance(value, list):
            for nested_value in value:
                walk(nested_value, key)
        elif isinstance(value, str):
            if absolute_workspace in value or value.startswith("/"):
                raise Stage1P0ValidationError("report contains an absolute host path")

    walk(report)


def validate_stage1_p0(
    *,
    workspace_root: str | Path,
    mode: str,
    validator_overrides: Mapping[str, Callable[..., Any]] | None = None,
    schema_path: str | Path | None = None,
) -> dict[str, Any]:
    """Build a deterministic, read-only Stage 1 P0 status report."""

    if mode not in MODES:
        raise Stage1P0ValidationError(f"mode must be one of {MODES}")
    root = Path(workspace_root).resolve()
    context = ValidationContext(
        workspace_root=root,
        mode=mode,
        validator_overrides=dict(validator_overrides or {}),
    )
    sealed_check, sealed_summary = _sealed_boundary_check(context)
    decision_check, decision_summary = _decision_check(context)

    requirements = _requirements(mode)
    if sealed_check["status"] == "FAIL":
        audit_check = _check(
            check_id="data.audit",
            area="data",
            phase="p0-prerequisite",
            status="PENDING",
            reason_code="sealed-boundary-failed-validation-aborted",
            ref_path="exps/causal_context/stage1_p0/refs/data_audit_ref.json",
        )
        audit_target = None
        aborted = {
            requirement.check_id: _check(
                check_id=requirement.check_id,
                area=requirement.area,
                phase=requirement.phase,
                status="PENDING",
                reason_code="sealed-boundary-failed-validation-aborted",
                ref_path=(
                    "exps/causal_context/stage1_p0/refs/"
                    + requirement.ref_names[0]
                ),
            )
            for requirement in requirements
        }
        artifact_checks = [
            aborted[requirement.check_id]
            for requirement in requirements
            if requirement.check_id != "counterfactual.final"
        ]
        checks_by_id = {check["check_id"]: check for check in artifact_checks}
        human_check = _check(
            check_id="data.human_review",
            area="human-review",
            phase="p0-prerequisite",
            status="PENDING",
            reason_code="sealed-boundary-failed-validation-aborted",
            ref_path=(
                "exps/causal_context/stage1_p0/review_inputs/"
                "data_adjudication.jsonl"
            ),
        )
        cf_review_check = _check(
            check_id="counterfactual.review",
            area="counterfactual",
            phase="p0-prerequisite",
            status="PENDING",
            reason_code="sealed-boundary-failed-validation-aborted",
            ref_path="exps/causal_context/stage1_p0/refs/cf_review_ref.json",
        )
        final_cf_check = aborted["counterfactual.final"]
        chain_check = _check(
            check_id="downstream.chain",
            area="downstream",
            phase=(
                "engineering-smoke"
                if mode == "engineering-smoke"
                else "post-training-formal"
            ),
            status="PENDING",
            reason_code="sealed-boundary-failed-validation-aborted",
        )
        determinism_check = _check(
            check_id="smoke.determinism_gate",
            area="downstream",
            phase="engineering-smoke",
            status="PENDING",
            reason_code="sealed-boundary-failed-validation-aborted",
            ref_path="exps/causal_context/stage1_p0/refs/legacy_smoke_generation_ref.json",
        )
        formal_lineage_check = _formal_exact_dependency_chain_check(
            context, checks_by_id
        )
    else:
        audit_check, _audit_metrics, audit_target = _data_audit_check(context)
        artifact_checks = []
        checks_by_id: dict[str, dict[str, Any]] = {}
        formal_dependencies = {
            "data.normalized": ("data.blind_review",),
            "data.train_partition": ("data.normalized",),
            "data.lexicon": ("data.train_partition",),
            "context.train": ("data.lexicon", "data.train_partition"),
            "context.dev": ("data.lexicon", "data.train_partition"),
            "training.evidence": (
                "context.train",
                "infrastructure.base_model",
            ),
            "training.plan": (
                "training.evidence",
                "infrastructure.environment",
                "infrastructure.base_model",
            ),
            "training.schedule": ("training.plan",),
            "control.dev": ("context.dev",),
            "counterfactual.proposal": ("context.dev",),
            "counterfactual.blind_review": ("counterfactual.proposal",),
            "counterfactual.final": (
                "counterfactual.proposal",
                "counterfactual.blind_review",
                "counterfactual.review",
            ),
            "model.registry": ("training.plan", "training.schedule"),
            "downstream.generation": ("model.registry",),
            "downstream.evaluation": ("downstream.generation",),
            "downstream.margin": (
                "model.registry",
                "context.dev",
                "control.dev",
                "counterfactual.final",
            ),
            "downstream.analysis": (
                "downstream.evaluation",
                "downstream.margin",
            ),
        }
        cf_review_check: dict[str, Any] | None = None
        final_cf_check: dict[str, Any] | None = None
        for requirement in requirements:
            if requirement.check_id == "counterfactual.final":
                proposal_check = checks_by_id["counterfactual.proposal"]
                blind_review_check = checks_by_id[
                    "counterfactual.blind_review"
                ]
                cf_review_check = _cf_review_workspace_check(
                    context,
                    proposal_check,
                    blind_review_check,
                )
                if (
                    proposal_check["status"] == "PASS"
                    and blind_review_check["status"] == "PASS"
                    and cf_review_check["status"] == "PASS"
                ):
                    proposal_report = context.semantic_reports[
                        "counterfactual.proposal"
                    ]
                    review_report = context.semantic_reports[
                        "counterfactual.review"
                    ]
                    blind_review_report = context.semantic_reports[
                        "counterfactual.blind_review"
                    ]
                    if review_report.get("cf_proposal_id") != proposal_report.get(
                        "cf_proposal_id"
                    ) or blind_review_report.get(
                        "cf_proposal_id"
                    ) != proposal_report.get(
                        "cf_proposal_id"
                    ) or review_report.get(
                        "cf_blind_review_id"
                    ) != blind_review_report.get(
                        "cf_blind_review_id"
                    ):
                        cf_review_check = _check(
                            check_id="counterfactual.review",
                            area="counterfactual",
                            phase="p0-prerequisite",
                            status="FAIL",
                            reason_code=(
                                "counterfactual-review-blind-lineage-mismatch"
                            ),
                            ref_path=cf_review_check["ref_path"],
                            artifact=cf_review_check["artifact"],
                        )
                checks_by_id["counterfactual.review"] = cf_review_check
                dependencies = formal_dependencies["counterfactual.final"]
                if any(
                    checks_by_id.get(dependency, {}).get("status") != "PASS"
                    for dependency in dependencies
                ):
                    final_cf_check = _check(
                        check_id=requirement.check_id,
                        area=requirement.area,
                        phase=requirement.phase,
                        status="PENDING",
                        reason_code="counterfactual-upstream-prerequisite-not-ready",
                        ref_path=(
                            "exps/causal_context/stage1_p0/refs/"
                            + requirement.ref_names[0]
                        ),
                    )
                else:
                    final_cf_check = _artifact_check(requirement, context)
                if final_cf_check["status"] == "PASS":
                    proposal_report = context.semantic_reports[
                        "counterfactual.proposal"
                    ]
                    review_report = context.semantic_reports[
                        "counterfactual.review"
                    ]
                    blind_review_report = context.semantic_reports[
                        "counterfactual.blind_review"
                    ]
                    final_report = context.semantic_reports[
                        "counterfactual.final"
                    ]
                    if (
                        final_report.get("cf_proposal_id")
                        != proposal_report.get("cf_proposal_id")
                        or final_report.get("review_id")
                        != review_report.get("review_id")
                        or final_report.get("cf_blind_review_id")
                        != blind_review_report.get("cf_blind_review_id")
                    ):
                        final_cf_check = _check(
                            check_id="counterfactual.final",
                            area="counterfactual",
                            phase="p0-prerequisite",
                            status="FAIL",
                            reason_code="counterfactual-final-lineage-mismatch",
                            ref_path=final_cf_check["ref_path"],
                            artifact=final_cf_check["artifact"],
                        )
                checks_by_id["counterfactual.final"] = final_cf_check
                continue
            dependencies = formal_dependencies.get(requirement.check_id, ())
            if mode == "formal-readiness" and any(
                checks_by_id.get(dependency, {}).get("status") != "PASS"
                for dependency in dependencies
            ):
                check = _check(
                    check_id=requirement.check_id,
                    area=requirement.area,
                    phase=requirement.phase,
                    status="PENDING",
                    reason_code="formal-upstream-prerequisite-not-ready",
                    ref_path=(
                        "exps/causal_context/stage1_p0/refs/"
                        + requirement.ref_names[0]
                    ),
                )
            else:
                check = _artifact_check(requirement, context)
            artifact_checks.append(check)
            checks_by_id[check["check_id"]] = check
        if cf_review_check is None or final_cf_check is None:
            raise Stage1P0ValidationError(
                "counterfactual lifecycle requirements are incomplete"
            )
        human_check = _human_review_check(
            context,
            audit_target,
            checks_by_id.get("data.normalized"),
        )
        formal_lineage_check = _formal_exact_dependency_chain_check(
            context, checks_by_id
        )
        chain_check = _downstream_chain_check(context, checks_by_id)
        determinism_check = _smoke_determinism_check(context, checks_by_id)

    checks = [
        decision_check,
        audit_check,
        human_check,
        *artifact_checks,
        cf_review_check,
        final_cf_check,
        formal_lineage_check,
        determinism_check,
        chain_check,
        sealed_check,
    ]
    counts = Counter(str(check["status"]) for check in checks)
    summary = {status: counts.get(status, 0) for status in STATUSES}
    overall = max(STATUSES, key=lambda status: STATUS_PRECEDENCE[status] if counts.get(status) else -1)
    gaps = [_gap_from_check(check) for check in checks if check["status"] != "PASS"]
    report: dict[str, Any] = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "mode": mode,
        "status": overall,
        "ready": overall == "PASS",
        "status_precedence": ["FAIL", "BLOCKED", "PENDING", "PASS"],
        "decision_register": decision_summary,
        "sealed_test": sealed_summary,
        "summary": summary,
        "checks": checks,
        "gaps": gaps,
    }
    report["report_sha256"] = _canonical_report_hash(report)
    _validate_report_safety(report, root)
    selected_schema = Path(schema_path) if schema_path is not None else context.schema_root / REPORT_SCHEMA_NAME
    validate_json_schema(report, selected_schema)
    return report


def write_report_sidecar(
    report: Mapping[str, Any],
    destination: str | Path,
    *,
    workspace_root: str | Path,
) -> None:
    """Atomically write a report outside every immutable lifecycle target."""

    root = Path(workspace_root).resolve()
    output = Path(destination).resolve()
    stage1_root = root / "exps" / "causal_context" / "stage1_p0"
    refs_root = stage1_root / "refs"
    immutable_roots = {
        "data_audits",
        "data",
        "lexicons",
        "contexts",
        "training_evidence",
        "environments",
        "training_plans",
        "training_schedules",
        "controls",
        "cf_proposals",
        "reviews",
        "counterfactuals",
        "models",
        "model_registries",
        "generation_runs",
        "evaluations",
        "margins",
        "analyses",
        "test_contexts",
        "test_controls",
        "test_cf_proposals",
        "test_counterfactuals",
    }
    for check in report.get("checks", []):
        if not isinstance(check, Mapping):
            continue
        artifact = check.get("artifact")
        if not isinstance(artifact, Mapping):
            continue
        logical_target = artifact.get("logical_target_path")
        if not isinstance(logical_target, str):
            continue
        target = (root / logical_target).resolve()
        try:
            output.relative_to(target)
        except ValueError:
            continue
        raise Stage1P0ValidationError(
            "validation report sidecar cannot be written inside an immutable target"
        )
    try:
        relative = output.relative_to(stage1_root)
    except ValueError:
        relative = None
    if relative is not None:
        if not relative.parts or relative.parts[0] in immutable_roots or output == refs_root:
            raise Stage1P0ValidationError(
                "validation report sidecar cannot be written inside an immutable target root"
            )
        if relative.parts[0] == "refs":
            raise Stage1P0ValidationError(
                "validation report sidecar cannot be written into the lifecycle refs directory"
            )
    payload = canonical_json_bytes(dict(report)) + b"\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{output.name}.",
            dir=output.parent,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


__all__ = [
    "MODES",
    "REPORT_SCHEMA_VERSION",
    "STATUSES",
    "Stage1P0ValidationError",
    "validate_stage1_p0",
    "write_report_sidecar",
]
