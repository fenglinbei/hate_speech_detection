"""Freeze and validate the immutable Stage 1 training plan.

The source recipe is copied once, all referenced train/protocol configuration
is embedded as resolved JSON, and only portable artifact dependencies enter the
target.  A plan intentionally contains no schedule, future checkpoint, or
future trained-model binding.
"""

from __future__ import annotations

import copy
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from data.training_artifacts import (
    TrainingArtifactError,
    canonical_sha256,
    ensure_exact_file_set,
    finalize_target_atomic,
    load_json,
    new_staging_directory,
    portable_dependency,
    resolve_dependency_target,
    resolve_locator_ref,
    sha256_file,
    validate_dependency_ref,
    validate_json_schema,
    validate_payload_manifest,
    write_canonical_json,
    write_locator_ref,
)
from data.training_evidence import (
    ARTIFACT_KIND as EVIDENCE_ARTIFACT_KIND,
    context_policy_snapshot,
    validate_training_evidence_target,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SCHEMA = REPOSITORY_ROOT / "schemas/stage1_training_plan_v1.schema.json"
DEFAULT_DECISION_REGISTER = REPOSITORY_ROOT / "config/stage1/decision_register.json"
DEFAULT_TRAIN_CODE = REPOSITORY_ROOT / "src/finetune/train.py"
DEFAULT_RUNTIME_CODE = REPOSITORY_ROOT / "src/finetune/stage1_runtime.py"

ARTIFACT_KIND = "training-plan"
CONTEXT_ARTIFACT_KIND = "context"
ENVIRONMENT_ARTIFACT_KINDS = ("stage1-environment", "environment")
BASE_MODEL_ARTIFACT_KINDS = ("model", "stage1-model", "base-model")
NON_TRAINING_MODEL_ARTIFACT_KIND = "stage1-model"
PLAN_SCHEMA_VERSION = "stage1-training-plan/v1"
SOURCE_RECIPE_SCHEMA_VERSION = "stage1-training-source-recipe/v1"
PROTOCOL_SCHEMA_VERSION = "stage1-protocol-snapshot/v1"
PROVENANCE_SCHEMA_VERSION = "stage1-training-plan-provenance/v1"
MODEL_KEY_RE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.-]*/[A-Za-z0-9_][A-Za-z0-9_.-]*$")
SCOPES = ("formal", "pilot", "engineering-smoke")
FUTURE_BINDING_KEYS = {
    "schedule_ref",
    "schedule_build_id",
    "schedule_dependency",
    "model_ref",
    "model_artifact_id",
    "checkpoint_path",
    "checkpoint_ref",
    "training_receipt",
    "training_receipt_ref",
}
EARLY_STOPPING_POLICY = {
    "enabled": True,
    "metric": "eval_loss",
    "mode": "min",
    "minimum_epochs": 1,
    "maximum_epochs": 5,
    "patience_evaluations": 3,
    "threshold": 0.001,
    "tie_break": "earliest-global-step",
    "selection_data": "train-only-calibration",
    "scientific_dev_used_for_selection": False,
}
TRAIN_CALIBRATION_POLICY = {
    "assignment": "sha256-query-id-v1",
    "fraction": 0.1,
    "hash_modulus": 10000,
    "hash_threshold_exclusive": 1000,
    "id_fields": ["id", "query_id"],
    "salt": "stage1-train-calibration-v1",
}
TRAIN_PARTITION_CONFIG = {
    "artifact_kind": "train-partition",
    "policy": "immutable-explicit-ref/v1",
    "nominal_hash_policy": "assertion-only",
}
CALIBRATION_PRESENTATION_POLICY = {
    "policy": "calibration-epoch-1-wire/v1",
    "wire_epoch": 1,
    "demo_order_across_epochs": True,
    "source_mask_across_epochs": True,
}
EPOCH_SELECTION_POLICY = {
    "maximum_epochs": 5,
    "minimum_epochs": 1,
    "selection_data": "immutable-train-partition-calibration",
    "nominal_hash_policy": "sha256-query-id-v1-assertion-only",
    "fixed_presentation": "calibration-epoch-1-wire/v1",
    "metric": "eval_loss",
    "mode": "min",
    "patience_evaluations": 3,
    "threshold": 0.001,
    "tie_break": "earliest-global-step",
    "scientific_dev_used_for_selection": False,
}


class TrainingPlanError(TrainingArtifactError):
    """Raised when an immutable Stage 1 training plan is inconsistent."""


def _logical_repo_path(path: Path, workspace_root: Path) -> str:
    resolved = path.resolve()
    root = workspace_root.resolve()
    try:
        return resolved.relative_to(root).as_posix()
    except ValueError as exc:
        raise TrainingPlanError(f"source file is outside the workspace: {path}") from exc


def _resolve_repo_file(logical_path: Any, workspace_root: Path, *, label: str) -> tuple[Path, str]:
    if not isinstance(logical_path, str) or not logical_path:
        raise TrainingPlanError(f"{label} path is missing")
    candidate = Path(logical_path)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise TrainingPlanError(f"{label} path is not portable: {logical_path}")
    path = (workspace_root / candidate).resolve()
    _logical_repo_path(path, workspace_root)
    if not path.is_file() or path.is_symlink():
        raise TrainingPlanError(f"{label} file does not exist: {logical_path}")
    return path, candidate.as_posix()


def _resolved_file_snapshot(path_value: Any, workspace_root: Path, *, label: str) -> dict[str, Any]:
    path, logical_path = _resolve_repo_file(path_value, workspace_root, label=label)
    value = load_json(path)
    if not isinstance(value, dict):
        raise TrainingPlanError(f"{label} must contain a JSON object")
    return {
        "logical_repo_path": logical_path,
        "sha256": canonical_sha256(value),
        "resolved": value,
    }


def _protocol_snapshot(
    source_spec: Mapping[str, Any],
    context_config: Mapping[str, Any],
    workspace_root: Path,
    *,
    train_context_policy: Mapping[str, Any] | None,
    train_context_policy_sha256: str | None,
) -> dict[str, Any]:
    profiles = source_spec.get("protocol_profiles")
    if not isinstance(profiles, Mapping) or not profiles:
        raise TrainingPlanError("source recipe lacks protocol_profiles")
    frozen_profiles = {
        name: _resolved_file_snapshot(
            path, workspace_root, label=f"protocol profile {name}"
        )
        for name, path in sorted(profiles.items())
    }
    generation_conditions = source_spec.get("ordered_generation_conditions")
    margin_conditions = source_spec.get("ordered_margin_conditions")
    if not isinstance(generation_conditions, list) or not generation_conditions:
        raise TrainingPlanError("source recipe lacks ordered generation conditions")
    if not isinstance(margin_conditions, list) or not margin_conditions:
        raise TrainingPlanError("source recipe lacks ordered margin conditions")
    if len(generation_conditions) != len(set(generation_conditions)):
        raise TrainingPlanError("generation conditions are not unique")
    if len(margin_conditions) != len(set(margin_conditions)):
        raise TrainingPlanError("margin conditions are not unique")
    for name, snapshot in frozen_profiles.items():
        profile_conditions = snapshot["resolved"].get("ordered_conditions")
        expected = margin_conditions if name == "margin" else generation_conditions
        if profile_conditions is not None and profile_conditions != expected:
            raise TrainingPlanError(
                f"protocol profile {name} condition order disagrees with source recipe"
            )
    output_protocol = context_config.get("output_protocol")
    if not isinstance(output_protocol, Mapping):
        raise TrainingPlanError("context config lacks output_protocol")
    if output_protocol.get("schema_version") != "canonical-quad-json/v1":
        raise TrainingPlanError("Stage 1 plan requires canonical-quad-json/v1")
    return {
        "schema_version": PROTOCOL_SCHEMA_VERSION,
        "output_protocol": copy.deepcopy(dict(output_protocol)),
        "train_context_policy": (
            None
            if train_context_policy is None
            else copy.deepcopy(dict(train_context_policy))
        ),
        "train_context_policy_sha256": train_context_policy_sha256,
        "ordered_generation_conditions": copy.deepcopy(generation_conditions),
        "ordered_margin_conditions": copy.deepcopy(margin_conditions),
        "profiles": frozen_profiles,
    }


def _freeze_deepspeed_dependency(
    resolved: Mapping[str, Any], workspace_root: Path
) -> dict[str, Any]:
    training = resolved.get("training")
    if not isinstance(training, dict):
        raise TrainingPlanError("train config lacks training settings")
    deepspeed_path = training.get("deepspeed")
    snapshot = _resolved_file_snapshot(
        deepspeed_path, workspace_root, label="DeepSpeed profile"
    )
    return snapshot


def _validate_training_slot_config(
    *,
    slot: Mapping[str, Any],
    resolved: Mapping[str, Any],
    evidence_budget: Mapping[str, Any],
) -> None:
    model_key = slot["model_key"]
    role = slot["role"]
    seed = slot["seed"]
    epochs = slot["epochs"]
    if resolved.get("schema_version") != "stage1-train-config/v1":
        raise TrainingPlanError(f"slot {model_key} has the wrong train config schema")
    if resolved.get("model_role") != role:
        raise TrainingPlanError(f"slot {model_key} role disagrees with train config")
    if resolved.get("random_seed") != seed:
        raise TrainingPlanError(f"slot {model_key} random_seed was not resolved")
    if resolved.get("max_length") != evidence_budget.get("max_sequence_tokens"):
        raise TrainingPlanError(f"slot {model_key} max_length disagrees with evidence budget")
    data = resolved.get("data")
    context_policy = resolved.get("context_policy")
    training = resolved.get("training")
    checkpointing = resolved.get("checkpointing")
    if not all(isinstance(value, Mapping) for value in (data, context_policy, training, checkpointing)):
        raise TrainingPlanError(f"slot {model_key} train config sections are incomplete")
    if data.get("source_schema") != "stage1-training-schedule/v1" or data.get(
        "requires_frozen_schedule"
    ) is not True:
        raise TrainingPlanError(f"slot {model_key} does not require a frozen schedule")
    if data.get("train_data_path") is not None or data.get("val_data_path") is not None:
        raise TrainingPlanError(f"slot {model_key} embeds a mutable train/dev data path")
    if (
        data.get("selection_split") != "train-only-calibration"
        or data.get("calibration") != TRAIN_CALIBRATION_POLICY
    ):
        raise TrainingPlanError(
            f"slot {model_key} does not freeze the canonical train-only calibration split"
        )
    if data.get("partition") != TRAIN_PARTITION_CONFIG:
        raise TrainingPlanError(
            f"slot {model_key} does not require an immutable train partition"
        )
    if data.get("fixed_presentation") != CALIBRATION_PRESENTATION_POLICY:
        raise TrainingPlanError(
            f"slot {model_key} does not freeze calibration presentation"
        )
    if training.get("output_dir") is not None:
        raise TrainingPlanError(f"slot {model_key} embeds a future checkpoint path")
    if training.get("seed") != seed or training.get("data_seed") != seed:
        raise TrainingPlanError(f"slot {model_key} training seeds were not resolved")
    if training.get("num_train_epochs") != epochs:
        raise TrainingPlanError(f"slot {model_key} epoch count disagrees with train config")
    if checkpointing.get("final_checkpoint_rule") != slot["final_checkpoint_rule"]:
        raise TrainingPlanError(f"slot {model_key} checkpoint rule disagrees with train config")
    if resolved.get("early_stopping") != EARLY_STOPPING_POLICY:
        raise TrainingPlanError(
            f"slot {model_key} does not freeze the canonical early-stopping policy"
        )
    if EARLY_STOPPING_POLICY["maximum_epochs"] != epochs:
        raise TrainingPlanError(
            f"slot {model_key} epoch count differs from the early-stopping maximum"
        )
    lexicon_probability = context_policy.get("lexicon_dropout_probability")
    demo_probability = context_policy.get("demonstration_dropout_probability")
    if role == "M_LD" and (lexicon_probability != 0.0 or demo_probability != 0.0):
        raise TrainingPlanError("M_LD must not drop either evidence source")
    if role == "M_drop":
        if lexicon_probability != 0.5 or demo_probability != 0.5:
            raise TrainingPlanError("M_drop must independently drop L and D with p=0.5")
        if context_policy.get("dropout_independent") is not True:
            raise TrainingPlanError("M_drop source dropout must be independent")
    if context_policy.get("matched_permutation_across_roles") is not True:
        raise TrainingPlanError("model roles must share matched demo permutations")


def _matched_role_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Remove only the preregistered M_LD/M_drop treatment coordinates."""

    normalized = copy.deepcopy(dict(config))
    normalized.pop("model_role", None)
    normalized.pop("exp_name", None)
    policy = normalized.get("context_policy")
    if not isinstance(policy, dict):
        raise TrainingPlanError("matched role config lacks context_policy")
    for key in (
        "lexicon_dropout_probability",
        "demonstration_dropout_probability",
        "dropout_independent",
        "dropout_rng_policy",
    ):
        policy.pop(key, None)
    return normalized


def _validate_order_dropout_contract(
    slots: Sequence[Mapping[str, Any]],
    rng_policy: Mapping[str, Any] | None,
    *,
    scope: str,
) -> None:
    if scope == "engineering-smoke":
        if rng_policy is not None:
            raise TrainingPlanError("engineering-smoke cannot carry a training RNG policy")
        return
    if not isinstance(rng_policy, Mapping):
        raise TrainingPlanError("training plan lacks an order/dropout RNG policy")
    order_policy = rng_policy.get("order")
    matched = rng_policy.get("matched_permutation_across_roles")
    drop = rng_policy.get("m_drop")
    if not isinstance(order_policy, str) or not order_policy or matched is not True:
        raise TrainingPlanError("training plan has an invalid matched demo-order policy")
    if not isinstance(drop, Mapping):
        raise TrainingPlanError("training plan lacks the M_drop policy")
    expected_drop = {
        "lexicon_probability": 0.5,
        "demonstration_probability": 0.5,
        "independent_sources": True,
        "rng": "sha256-model-seed-epoch-query-source/v1",
    }
    if dict(drop) != expected_drop:
        raise TrainingPlanError("training plan M_drop policy is not canonical")
    if rng_policy.get("fit_presentation") != "per-seed-epoch-query/v1":
        raise TrainingPlanError("training plan fit presentation policy is not canonical")
    if (
        rng_policy.get("calibration_presentation")
        != CALIBRATION_PRESENTATION_POLICY
    ):
        raise TrainingPlanError(
            "training plan calibration presentation policy is not canonical"
        )

    paired: dict[int, dict[str, Mapping[str, Any]]] = {}
    for slot in slots:
        config = slot.get("train_config_resolved")
        if not isinstance(config, Mapping):
            raise TrainingPlanError(f"slot {slot.get('model_key')} lacks resolved config")
        policy = config.get("context_policy")
        if not isinstance(policy, Mapping):
            raise TrainingPlanError(f"slot {slot.get('model_key')} lacks context policy")
        if (
            policy.get("order_policy") != order_policy
            or policy.get("matched_permutation_across_roles") is not matched
        ):
            raise TrainingPlanError(
                f"slot {slot.get('model_key')} disagrees with plan demo-order policy"
            )
        role = slot.get("role")
        if role == "M_drop" and (
            policy.get("lexicon_dropout_probability")
            != drop["lexicon_probability"]
            or policy.get("demonstration_dropout_probability")
            != drop["demonstration_probability"]
            or policy.get("dropout_independent")
            is not drop["independent_sources"]
            or policy.get("dropout_rng_policy") != drop["rng"]
        ):
            raise TrainingPlanError(
                f"slot {slot.get('model_key')} disagrees with plan M_drop policy"
            )
        seed = slot.get("seed")
        if isinstance(seed, int) and role in {"M_LD", "M_drop"}:
            paired.setdefault(seed, {})[str(role)] = config

    for seed, roles in paired.items():
        if set(roles) != {"M_LD", "M_drop"}:
            raise TrainingPlanError(f"seed {seed} lacks a matched M_LD/M_drop pair")
        if _matched_role_config(roles["M_LD"]) != _matched_role_config(
            roles["M_drop"]
        ):
            raise TrainingPlanError(
                f"seed {seed} role configs differ outside preregistered context dropout"
            )


def _validate_epoch_selection_contract(value: Any, *, scope: str) -> None:
    if scope == "engineering-smoke":
        if value is not None:
            raise TrainingPlanError(
                "engineering-smoke cannot carry an epoch-selection policy"
            )
        return
    if value != EPOCH_SELECTION_POLICY:
        raise TrainingPlanError(
            "training plan epoch selection does not bind the immutable partition/fixed presentation policy"
        )


def _resolve_slot(
    source_slot: Mapping[str, Any],
    *,
    scope: str,
    workspace_root: Path,
    evidence_budget: Mapping[str, Any],
) -> dict[str, Any]:
    model_key = source_slot.get("model_key")
    role = source_slot.get("role")
    if not isinstance(model_key, str) or not MODEL_KEY_RE.fullmatch(model_key):
        raise TrainingPlanError(f"invalid model_key: {model_key!r}")
    if not isinstance(role, str) or not role:
        raise TrainingPlanError(f"slot {model_key} lacks role")
    training_required = source_slot.get("training_required", scope != "engineering-smoke")
    if not isinstance(training_required, bool):
        raise TrainingPlanError(f"slot {model_key} training_required is invalid")
    if not training_required:
        nullable = ("seed", "train_config", "epochs", "final_checkpoint_rule")
        if any(source_slot.get(field) is not None for field in nullable):
            raise TrainingPlanError(
                f"non-training slot {model_key} must use null training fields"
            )
        return {
            "model_key": model_key,
            "role": role,
            "seed": None,
            "training_required": False,
            "train_config_logical_path": None,
            "source_train_config_sha256": None,
            "resolved_train_config_sha256": None,
            "train_config_sha256": None,
            "train_config_resolved": None,
            "deepspeed_config_logical_path": None,
            "deepspeed_config_sha256": None,
            "deepspeed_config_resolved": None,
            "epochs": None,
            "final_checkpoint_rule": None,
        }
    if scope == "engineering-smoke":
        raise TrainingPlanError("engineering-smoke slots cannot require training")
    seed = source_slot.get("seed")
    epochs = source_slot.get("epochs")
    final_rule = source_slot.get("final_checkpoint_rule")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise TrainingPlanError(f"slot {model_key} has an invalid seed")
    if not isinstance(epochs, int) or isinstance(epochs, bool) or epochs <= 0:
        raise TrainingPlanError(f"slot {model_key} has an invalid epoch count")
    if not isinstance(final_rule, str) or not final_rule:
        raise TrainingPlanError(f"slot {model_key} lacks final checkpoint rule")
    config_path, logical_path = _resolve_repo_file(
        source_slot.get("train_config"), workspace_root, label=f"slot {model_key} train config"
    )
    source_config = load_json(config_path)
    if not isinstance(source_config, dict):
        raise TrainingPlanError(f"slot {model_key} train config must be an object")
    source_config_sha256 = canonical_sha256(source_config)
    resolved = copy.deepcopy(source_config)
    resolved["exp_name"] = f"stage1-{role.lower().replace('_', '-')}-seed-{seed}"
    resolved["random_seed"] = seed
    data = resolved.setdefault("data", {})
    training = resolved.setdefault("training", {})
    if not isinstance(data, dict) or not isinstance(training, dict):
        raise TrainingPlanError(f"slot {model_key} config sections are invalid")
    data["train_data_path"] = None
    training["output_dir"] = None
    training["seed"] = seed
    training["data_seed"] = seed
    training["num_train_epochs"] = epochs
    deepspeed_snapshot = _freeze_deepspeed_dependency(resolved, workspace_root)
    resolved_config_sha256 = canonical_sha256(resolved)
    frozen_slot = {
        "model_key": model_key,
        "role": role,
        "seed": seed,
        "training_required": True,
        "train_config_logical_path": logical_path,
        "source_train_config_sha256": source_config_sha256,
        "resolved_train_config_sha256": resolved_config_sha256,
        "train_config_sha256": resolved_config_sha256,
        "train_config_resolved": resolved,
        "deepspeed_config_logical_path": deepspeed_snapshot["logical_repo_path"],
        "deepspeed_config_sha256": deepspeed_snapshot["sha256"],
        "deepspeed_config_resolved": deepspeed_snapshot["resolved"],
        "epochs": epochs,
        "final_checkpoint_rule": final_rule,
    }
    _validate_training_slot_config(
        slot=frozen_slot,
        resolved=resolved,
        evidence_budget=evidence_budget,
    )
    return frozen_slot


def _ordered_subset(subset: Sequence[str], superset: Sequence[str]) -> bool:
    iterator = iter(superset)
    return all(any(value == candidate for candidate in iterator) for value in subset)


def _validate_slot_set(
    slots: Sequence[Mapping[str, Any]],
    pilot_slot_keys: Sequence[str],
    *,
    scope: str,
    decision_register: Mapping[str, Any],
    evidence_budget: Mapping[str, Any],
) -> None:
    keys = [slot.get("model_key") for slot in slots]
    if len(keys) != len(set(keys)):
        raise TrainingPlanError("ordered model slots contain duplicate model keys")
    if len(pilot_slot_keys) != len(set(pilot_slot_keys)) or not _ordered_subset(
        pilot_slot_keys, keys
    ):
        raise TrainingPlanError("pilot_slot_keys is not an ordered slot subset")
    if scope == "engineering-smoke":
        if pilot_slot_keys or any(slot.get("training_required") for slot in slots):
            raise TrainingPlanError("engineering-smoke plan cannot contain training slots")
        return
    if any(not slot.get("training_required") for slot in slots):
        raise TrainingPlanError(f"{scope} plan contains a non-training slot")
    for slot in slots:
        resolved = slot.get("train_config_resolved")
        if not isinstance(resolved, Mapping):
            raise TrainingPlanError(f"slot {slot.get('model_key')} lacks resolved config")
        if slot.get("train_config_sha256") != canonical_sha256(resolved):
            raise TrainingPlanError(f"slot {slot.get('model_key')} config hash mismatch")
        if slot.get("resolved_train_config_sha256") != slot.get("train_config_sha256"):
            raise TrainingPlanError(
                f"slot {slot.get('model_key')} resolved config hash alias mismatch"
            )
        deepspeed = slot.get("deepspeed_config_resolved")
        if not isinstance(deepspeed, Mapping):
            raise TrainingPlanError(
                f"slot {slot.get('model_key')} lacks frozen DeepSpeed config"
            )
        if slot.get("deepspeed_config_sha256") != canonical_sha256(deepspeed):
            raise TrainingPlanError(
                f"slot {slot.get('model_key')} DeepSpeed config hash mismatch"
            )
        if resolved["training"].get("deepspeed") != slot.get(
            "deepspeed_config_logical_path"
        ):
            raise TrainingPlanError(
                f"slot {slot.get('model_key')} DeepSpeed path/snapshot mismatch"
            )
        _validate_training_slot_config(
            slot=slot, resolved=resolved, evidence_budget=evidence_budget
        )
    if scope == "formal":
        decisions = decision_register.get("decisions", {})
        seed_decision = decisions.get("D7_training_seeds", {})
        expected_seeds = seed_decision.get("formal")
        if not isinstance(expected_seeds, list) or not expected_seeds:
            raise TrainingPlanError("decision register lacks formal training seeds")
        expected_coordinates = [
            (role, seed) for seed in expected_seeds for role in ("M_LD", "M_drop")
        ]
        actual_coordinates = [(slot["role"], slot["seed"]) for slot in slots]
        if actual_coordinates != expected_coordinates:
            raise TrainingPlanError(
                "formal slots must be ordered matched M_LD/M_drop pairs for frozen seeds"
            )
        expected_pilot = [f"M_LD/seed-{expected_seeds[0]}", f"M_drop/seed-{expected_seeds[0]}"]
        if list(pilot_slot_keys) != expected_pilot:
            raise TrainingPlanError("formal pilot_slot_keys disagrees with D7")


def _check_no_future_bindings(value: Any, *, path: str = "plan") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if key in FUTURE_BINDING_KEYS and item is not None:
                raise TrainingPlanError(f"future binding {path}.{key} is forbidden")
            if key == "output_dir" and item is not None:
                raise TrainingPlanError(f"future checkpoint path {path}.{key} is forbidden")
            _check_no_future_bindings(item, path=f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _check_no_future_bindings(item, path=f"{path}[{index}]")


def recompute_training_plan_id(plan: Mapping[str, Any]) -> str:
    id_inputs = plan.get("id_inputs")
    if not isinstance(id_inputs, Mapping):
        raise TrainingPlanError("training plan lacks id_inputs")
    return "tpl-" + canonical_sha256(id_inputs)


def _validate_protocol_snapshot(snapshot: Mapping[str, Any]) -> None:
    if snapshot.get("schema_version") != PROTOCOL_SCHEMA_VERSION:
        raise TrainingPlanError("protocol snapshot has the wrong schema")
    profiles = snapshot.get("profiles")
    if not isinstance(profiles, Mapping) or not profiles:
        raise TrainingPlanError("protocol snapshot lacks profiles")
    for name, profile in profiles.items():
        if not isinstance(profile, Mapping) or set(profile) != {
            "logical_repo_path",
            "sha256",
            "resolved",
        }:
            raise TrainingPlanError(f"protocol profile {name} snapshot is malformed")
        path = Path(str(profile["logical_repo_path"]))
        if path.is_absolute() or ".." in path.parts:
            raise TrainingPlanError(f"protocol profile {name} path is not portable")
        if profile["sha256"] != canonical_sha256(profile["resolved"]):
            raise TrainingPlanError(f"protocol profile {name} hash mismatch")
    policy = snapshot.get("train_context_policy")
    policy_hash = snapshot.get("train_context_policy_sha256")
    if policy is None or policy_hash is None:
        if policy is not None or policy_hash is not None:
            raise TrainingPlanError("protocol snapshot context policy nullability mismatch")
    elif not isinstance(policy, Mapping) or policy_hash != canonical_sha256(policy):
        raise TrainingPlanError("protocol snapshot train context policy hash mismatch")


def _expected_id_inputs(plan: Mapping[str, Any]) -> dict[str, Any]:
    expected = {
        "schema_version": PLAN_SCHEMA_VERSION,
        "scope": plan["scope"],
        "source_spec_sha256": plan["source_spec_sha256"],
        "protocol_snapshot_sha256": plan["protocol_snapshot_sha256"],
        "decision_register_sha256": plan["decision_register_sha256"],
        "context_dependency": plan["context_dependency"],
        "training_evidence_dependency": plan["training_evidence_dependency"],
        "base_model_dependency": plan["base_model_dependency"],
        "environment_dependency": plan["environment_dependency"],
        "non_training_model_dependencies": plan[
            "non_training_model_dependencies"
        ],
        "ordered_model_slots": plan["ordered_model_slots"],
        "pilot_slot_keys": plan["pilot_slot_keys"],
        "per_slot_train_config_sha256": plan["per_slot_train_config_sha256"],
        "epoch_and_checkpoint_selection_policy": plan[
            "epoch_and_checkpoint_selection_policy"
        ],
        "order_dropout_rng_policy": plan["order_dropout_rng_policy"],
        "train_code_sha256": plan["train_code_sha256"],
        "plan_builder_code_sha256": plan["plan_builder_code_sha256"],
    }
    if "runtime_code_sha256" in plan:
        expected["runtime_code_sha256"] = plan["runtime_code_sha256"]
    if "train_partition_dependency" in plan:
        expected["train_partition_dependency"] = plan[
            "train_partition_dependency"
        ]
    # `stage1-training-plan/v1` existed before sealed context-policy lineage
    # was added.  Preserve validation of already-published engineering-smoke
    # plans byte-for-byte; new plans include both nullable keys, while formal
    # validation below still requires their non-null exact policy snapshot.
    if (
        "train_context_policy" in plan
        or "train_context_policy_sha256" in plan
    ):
        expected["train_context_policy"] = plan.get("train_context_policy")
        expected["train_context_policy_sha256"] = plan.get(
            "train_context_policy_sha256"
        )
    return expected


def _project_non_training_model_bindings(
    bindings: Sequence[tuple[str, str | Path]], workspace_root: Path
) -> dict[str, dict[str, Any]]:
    """Resolve already-existing engineering models into portable dependencies."""

    projected: dict[str, dict[str, Any]] = {}
    for model_key, model_ref in bindings:
        if not isinstance(model_key, str) or not MODEL_KEY_RE.fullmatch(model_key):
            raise TrainingPlanError(f"invalid non-training model key: {model_key!r}")
        if model_key in projected:
            raise TrainingPlanError(
                f"duplicate non-training model binding: {model_key}"
            )
        locator, target = resolve_locator_ref(
            model_ref, expected_kind=NON_TRAINING_MODEL_ARTIFACT_KIND
        )
        projected[model_key] = portable_dependency(locator, target, workspace_root)
    return projected


def _validate_non_training_model_dependencies(
    dependencies: Any,
    *,
    slots: Sequence[Mapping[str, Any]],
    scope: str,
    context_config: Mapping[str, Any],
    context_target: Path,
    environment_dependency: Mapping[str, Any],
    workspace_root: Path,
) -> dict[str, dict[str, Any]]:
    """Validate the one-way engineering-plan -> legacy-model bindings."""

    if not isinstance(dependencies, Mapping):
        raise TrainingPlanError("non-training model dependencies must be an object")
    if scope != "engineering-smoke":
        if dependencies:
            raise TrainingPlanError(
                "pilot/formal plans cannot bind non-training model artifacts"
            )
        return {}

    expected_keys = [
        slot["model_key"] for slot in slots if slot.get("training_required") is False
    ]
    if len(expected_keys) != len(slots):
        raise TrainingPlanError(
            "engineering-smoke plan contains a training-required slot"
        )
    if set(dependencies) != set(expected_keys) or len(dependencies) != len(expected_keys):
        raise TrainingPlanError(
            "engineering non-training model binding set mismatch: "
            f"missing={sorted(set(expected_keys) - set(dependencies))}, "
            f"extra={sorted(set(dependencies) - set(expected_keys))}"
        )

    budget = context_config.get("budget")
    context_revision = budget.get("tokenizer_revision") if isinstance(budget, Mapping) else None
    if not isinstance(context_revision, str) or not context_revision:
        raise TrainingPlanError("engineering context lacks a frozen tokenizer revision")
    context_meta_paths = sorted(context_target.glob("context_manifest.*.meta.json"))
    if len(context_meta_paths) != 1:
        raise TrainingPlanError(
            "engineering context must contain exactly one split manifest meta file"
        )
    context_meta = load_json(context_meta_paths[0])
    meta_budget = context_meta.get("budget") if isinstance(context_meta, Mapping) else None
    meta_revision = (
        meta_budget.get("tokenizer_revision")
        if isinstance(meta_budget, Mapping)
        else None
    )
    if meta_revision != context_revision:
        raise TrainingPlanError(
            "engineering context config/meta tokenizer revisions disagree"
        )

    # Local import avoids a module-import cycle.  Legacy artifacts intentionally
    # have no plan dependency, so this remains a one-way provenance edge.
    from model.stage1_registry import validate_model_artifact_target

    slot_by_key = {slot["model_key"]: slot for slot in slots}
    ordered: dict[str, dict[str, Any]] = {}
    for model_key in expected_keys:
        dependency = dependencies[model_key]
        if not isinstance(dependency, Mapping):
            raise TrainingPlanError(
                f"non-training model dependency {model_key} is not an object"
            )
        frozen = validate_dependency_ref(
            dependency, expected_kind=NON_TRAINING_MODEL_ARTIFACT_KIND
        )
        model_target = resolve_dependency_target(frozen, workspace_root)
        model = validate_model_artifact_target(
            model_target, workspace_root=workspace_root, require_name=True
        )
        slot = slot_by_key[model_key]
        if frozen["artifact_id"] != model.get("model_artifact_id"):
            raise TrainingPlanError(
                f"non-training model dependency {model_key} has an artifact ID mismatch"
            )
        if (
            model.get("artifact_type") != "legacy-smoke-only"
            or model.get("scope") != "engineering-smoke"
            or model.get("scientific_eligible") is not False
            or model.get("model_key") != model_key
            or model.get("role") != slot.get("role")
            or model.get("seed") != slot.get("seed")
            or model.get("environment_dependency") != environment_dependency
        ):
            raise TrainingPlanError(
                f"non-training model {model_key} is not the matching legacy smoke artifact"
            )
        tokenizer_contract = model.get("tokenizer_contract")
        model_revision = (
            tokenizer_contract.get("tokenizer_revision")
            if isinstance(tokenizer_contract, Mapping)
            else None
        )
        if model_revision != context_revision:
            raise TrainingPlanError(
                f"engineering context tokenizer revision {context_revision!r} differs "
                f"from legacy model {model_key} revision {model_revision!r}"
            )
        ordered[model_key] = dict(frozen)
    return ordered


def _validate_training_base_dependency(
    base_dependency: Mapping[str, Any],
    environment_dependency: Mapping[str, Any],
    *,
    workspace_root: Path,
) -> dict[str, Any]:
    """Deep-validate the only base/environment pair a training plan may bind."""

    frozen_base = validate_dependency_ref(
        base_dependency, expected_kind="stage1-model"
    )
    frozen_environment = validate_dependency_ref(
        environment_dependency, expected_kind="stage1-environment"
    )
    base_target = resolve_dependency_target(frozen_base, workspace_root)
    try:
        from model.stage1_registry import validate_model_artifact_target

        model = validate_model_artifact_target(
            base_target,
            workspace_root=workspace_root,
            require_name=True,
        )
    except Exception as exc:
        raise TrainingPlanError(
            f"training base model/current environment validation failed: {exc}"
        ) from exc
    if (
        model.get("artifact_type") != "base"
        or model.get("model_artifact_id") != frozen_base["artifact_id"]
        or model.get("environment_dependency") != dict(frozen_environment)
    ):
        raise TrainingPlanError(
            "training base model identity/environment differs from the explicit plan refs"
        )
    return dict(model)


def _validate_slot_base_bindings(
    slots: Sequence[Mapping[str, Any]],
    base_model: Mapping[str, Any],
    context_config: Mapping[str, Any],
) -> None:
    checkpoint = base_model.get("checkpoint_inventory")
    tokenizer_contract = base_model.get("tokenizer_contract")
    budget = context_config.get("budget")
    if not all(
        isinstance(value, Mapping)
        for value in (checkpoint, tokenizer_contract, budget)
    ):
        raise TrainingPlanError(
            "base model or context lacks checkpoint/tokenizer budget lineage"
        )
    expected_model_path = checkpoint.get("logical_repo_path")
    expected_revision = tokenizer_contract.get("tokenizer_revision")
    if (
        not isinstance(expected_model_path, str)
        or not expected_model_path
        or budget.get("tokenizer_revision") != expected_revision
    ):
        raise TrainingPlanError(
            "context tokenizer revision differs from the registered training base"
        )
    for slot in slots:
        resolved = slot.get("train_config_resolved")
        if (
            slot.get("training_required") is not True
            or not isinstance(resolved, Mapping)
            or resolved.get("model_path") != expected_model_path
        ):
            raise TrainingPlanError(
                f"slot {slot.get('model_key')} model_path differs from the registered base"
            )


def _validate_target(
    target: Path,
    *,
    workspace_root: Path,
    schema_path: Path,
    require_directory_name: bool,
    tokenizer: Any | None = None,
) -> dict[str, Any]:
    plan_preview = load_json(target / "plan.resolved.json")
    has_partition_binding = isinstance(plan_preview, Mapping) and (
        "train_partition_dependency" in plan_preview
    )
    expected_files = {
            "source_spec.json",
            "plan.resolved.json",
            "protocol_snapshot.json",
            "decision_register.json",
            "training_evidence_ref.json",
            "base_model_ref.json",
            "environment_ref.json",
            "non_training_model_refs.json",
            "provenance.json",
            "payload_manifest.json",
        }
    if has_partition_binding:
        expected_files.add("train_partition_ref.json")
    ensure_exact_file_set(target, expected_files)
    validate_payload_manifest(target)
    plan = load_json(target / "plan.resolved.json")
    source_spec = load_json(target / "source_spec.json")
    protocol_snapshot = load_json(target / "protocol_snapshot.json")
    decision_register = load_json(target / "decision_register.json")
    evidence_dependency = load_json(target / "training_evidence_ref.json")
    partition_dependency = (
        load_json(target / "train_partition_ref.json")
        if has_partition_binding
        else None
    )
    base_dependency = load_json(target / "base_model_ref.json")
    environment_dependency = load_json(target / "environment_ref.json")
    non_training_model_dependencies = load_json(
        target / "non_training_model_refs.json"
    )
    provenance = load_json(target / "provenance.json")
    if not isinstance(plan, dict):
        raise TrainingPlanError("plan.resolved.json is not an object")
    validate_json_schema(plan, schema_path)
    if not isinstance(source_spec, dict) or source_spec.get("schema_version") != SOURCE_RECIPE_SCHEMA_VERSION:
        raise TrainingPlanError("source recipe snapshot is invalid")
    if not isinstance(protocol_snapshot, dict):
        raise TrainingPlanError("protocol snapshot is invalid")
    if not isinstance(decision_register, dict) or decision_register.get("status") != "frozen":
        raise TrainingPlanError("decision register is not frozen")
    _validate_protocol_snapshot(protocol_snapshot)
    scope = plan["scope"]
    if plan.get("training_evidence_dependency") != evidence_dependency:
        raise TrainingPlanError(
            "training_evidence_dependency mismatch between plan and payload file"
        )
    if plan.get("train_partition_dependency") != partition_dependency:
        raise TrainingPlanError(
            "train_partition_dependency mismatch between plan and payload file"
        )
    if plan.get("base_model_dependency") != base_dependency:
        raise TrainingPlanError(
            "base_model_dependency mismatch between plan and payload file"
        )
    if plan.get("environment_dependency") != environment_dependency:
        raise TrainingPlanError(
            "environment_dependency mismatch between plan and payload file"
        )
    if plan.get("non_training_model_dependencies") != non_training_model_dependencies:
        raise TrainingPlanError(
            "non_training_model_dependencies mismatch between plan and payload file"
        )
    if not isinstance(environment_dependency, dict):
        raise TrainingPlanError("environment_dependency file is not an object")
    validate_dependency_ref(
        environment_dependency, expected_kind=ENVIRONMENT_ARTIFACT_KINDS
    )
    resolve_dependency_target(environment_dependency, workspace_root)
    context_dependency = plan.get("context_dependency")
    if not isinstance(context_dependency, dict):
        raise TrainingPlanError("plan context dependency is invalid")
    validate_dependency_ref(context_dependency, expected_kind=CONTEXT_ARTIFACT_KIND)
    context_target = resolve_dependency_target(context_dependency, workspace_root)
    context_config = load_json(context_target / "config.resolved.json")
    if not isinstance(context_config, Mapping):
        raise TrainingPlanError("context resolved config is invalid")
    if scope == "engineering-smoke":
        if (
            evidence_dependency is not None
            or partition_dependency is not None
            or base_dependency is not None
        ):
            raise TrainingPlanError(
                "engineering-smoke plan must use null training evidence/partition/base dependencies"
            )
        evidence_records: list[dict[str, Any]] = []
        evidence_budget: Mapping[str, Any] = {}
        evidence_meta: Mapping[str, Any] | None = None
        base_model: Mapping[str, Any] | None = None
    else:
        if (
            not isinstance(evidence_dependency, dict)
            or not isinstance(partition_dependency, dict)
            or not isinstance(base_dependency, dict)
        ):
            raise TrainingPlanError(
                "pilot/formal plans require training evidence, partition, and base model dependencies"
            )
        validate_dependency_ref(
            evidence_dependency, expected_kind=EVIDENCE_ARTIFACT_KIND
        )
        validate_dependency_ref(
            partition_dependency, expected_kind="train-partition"
        )
        validate_dependency_ref(
            base_dependency, expected_kind=BASE_MODEL_ARTIFACT_KINDS
        )
        evidence_target = resolve_dependency_target(evidence_dependency, workspace_root)
        evidence_meta, evidence_records = validate_training_evidence_target(
            evidence_target,
            workspace_root=workspace_root,
            tokenizer=tokenizer,
        )
        partition_target = resolve_dependency_target(
            partition_dependency, workspace_root
        )
        from data.train_partition import validate_train_partition_target

        partition_report = validate_train_partition_target(
            partition_target,
            workspace_root=workspace_root,
            expected_data_dependency=evidence_meta.get("data_dependency"),
        )
        if partition_report.get("partition_dependency") != partition_dependency:
            raise TrainingPlanError(
                "plan partition dependency cannot be reproduced"
            )
        if evidence_meta.get("train_partition_dependency") != partition_dependency:
            raise TrainingPlanError(
                "plan partition dependency disagrees with training evidence"
            )
        if evidence_meta.get("base_model_dependency") != base_dependency:
            raise TrainingPlanError(
                "plan base model dependency disagrees with training evidence"
            )
        base_model = _validate_training_base_dependency(
            base_dependency,
            environment_dependency,
            workspace_root=workspace_root,
        )
        if context_dependency != evidence_meta["context_dependency"]:
            raise TrainingPlanError(
                "plan context dependency disagrees with training evidence"
            )
        evidence_budget = evidence_records[0]["budget"]
        for record in evidence_records[1:]:
            if (
                record["budget"]["max_sequence_tokens"]
                != evidence_budget["max_sequence_tokens"]
            ):
                raise TrainingPlanError(
                    "training evidence uses inconsistent sequence budgets"
                )
    if scope == "engineering-smoke":
        if (
            plan.get("train_context_policy") is not None
            or plan.get("train_context_policy_sha256") is not None
            or protocol_snapshot.get("train_context_policy") is not None
            or protocol_snapshot.get("train_context_policy_sha256") is not None
        ):
            raise TrainingPlanError(
                "engineering-smoke plan cannot claim scientific train context policy"
            )
    else:
        try:
            train_context_policy, train_context_policy_sha256 = (
                context_policy_snapshot(
                    context_target,
                    expected_split="train",
                    require_scientific=True,
                )
            )
        except Exception as exc:
            raise TrainingPlanError(
                f"train context policy validation failed: {exc}"
            ) from exc
        assert evidence_meta is not None
        if (
            plan.get("train_context_policy") != train_context_policy
            or plan.get("train_context_policy_sha256")
            != train_context_policy_sha256
            or evidence_meta.get("train_context_policy") != train_context_policy
            or evidence_meta.get("train_context_policy_sha256")
            != train_context_policy_sha256
            or protocol_snapshot.get("train_context_policy")
            != train_context_policy
            or protocol_snapshot.get("train_context_policy_sha256")
            != train_context_policy_sha256
        ):
            raise TrainingPlanError(
                "plan/evidence/protocol train context policy lineage mismatch"
            )
    if plan.get("source_spec_sha256") != canonical_sha256(source_spec):
        raise TrainingPlanError("source recipe snapshot hash mismatch")
    if plan.get("protocol_snapshot_sha256") != canonical_sha256(protocol_snapshot):
        raise TrainingPlanError("protocol snapshot hash mismatch")
    if plan.get("decision_register_sha256") != canonical_sha256(decision_register):
        raise TrainingPlanError("decision register hash mismatch")
    if plan.get("id_inputs") != _expected_id_inputs(plan):
        raise TrainingPlanError("plan id_inputs do not match resolved fields")
    plan_id = recompute_training_plan_id(plan)
    if plan.get("training_plan_id") != plan_id:
        raise TrainingPlanError("training plan ID is not reproducible")
    if require_directory_name and target.name != plan_id:
        raise TrainingPlanError("training plan directory name does not match plan ID")
    if source_spec.get("scope") not in {scope, "formal" if scope == "pilot" else scope}:
        raise TrainingPlanError("source recipe scope disagrees with frozen plan scope")
    if plan.get("scientific_eligible") is not (scope == "formal"):
        raise TrainingPlanError("plan scientific eligibility disagrees with scope")
    if plan.get("per_slot_train_config_sha256") != {
        slot["model_key"]: slot["train_config_sha256"]
        for slot in plan["ordered_model_slots"]
    }:
        raise TrainingPlanError("per-slot train config hash map mismatch")
    validated_non_training = _validate_non_training_model_dependencies(
        non_training_model_dependencies,
        slots=plan["ordered_model_slots"],
        scope=scope,
        context_config=context_config,
        context_target=context_target,
        environment_dependency=environment_dependency,
        workspace_root=workspace_root,
    )
    if validated_non_training != non_training_model_dependencies:
        raise TrainingPlanError("non-training model dependency ordering is not canonical")
    _validate_slot_set(
        plan["ordered_model_slots"],
        plan["pilot_slot_keys"],
        scope=scope,
        decision_register=decision_register,
        evidence_budget=evidence_budget,
    )
    if scope != "engineering-smoke":
        assert base_model is not None
        _validate_slot_base_bindings(
            plan["ordered_model_slots"], base_model, context_config
        )
    _validate_order_dropout_contract(
        plan["ordered_model_slots"],
        plan.get("order_dropout_rng_policy"),
        scope=scope,
    )
    _validate_epoch_selection_contract(
        plan.get("epoch_and_checkpoint_selection_policy"), scope=scope
    )
    _check_no_future_bindings(plan)
    expected_provenance = {
        "schema_version": PROVENANCE_SCHEMA_VERSION,
        "training_plan_id": plan_id,
        "id_inputs": plan["id_inputs"],
        "plan_builder_code_sha256": plan["plan_builder_code_sha256"],
        "train_code_sha256": plan["train_code_sha256"],
    }
    if "train_partition_dependency" in plan:
        expected_provenance["train_partition_dependency"] = plan[
            "train_partition_dependency"
        ]
    if "runtime_code_sha256" in plan:
        expected_provenance["runtime_code_sha256"] = plan[
            "runtime_code_sha256"
        ]
    if scope != "engineering-smoke" and not isinstance(
        plan.get("runtime_code_sha256"), str
    ):
        raise TrainingPlanError("pilot/formal plan lacks frozen runtime code")
    if (
        "train_context_policy" in plan
        or "train_context_policy_sha256" in plan
    ):
        expected_provenance["train_context_policy"] = plan.get(
            "train_context_policy"
        )
        expected_provenance["train_context_policy_sha256"] = plan.get(
            "train_context_policy_sha256"
        )
    if provenance != expected_provenance:
        raise TrainingPlanError("training plan provenance mismatch")
    return plan


def freeze_training_plan(
    *,
    source_spec_path: str | Path,
    scope: str,
    context_ref: str | Path,
    training_evidence_ref: str | Path | None = None,
    train_partition_ref: str | Path | None = None,
    base_model_ref: str | Path | None = None,
    environment_ref: str | Path,
    write_ref: str | Path,
    workspace_root: str | Path = REPOSITORY_ROOT,
    target_root: str | Path | None = None,
    decision_register_path: str | Path = DEFAULT_DECISION_REGISTER,
    train_code_path: str | Path = DEFAULT_TRAIN_CODE,
    runtime_code_path: str | Path = DEFAULT_RUNTIME_CODE,
    schema_path: str | Path = DEFAULT_SCHEMA,
    non_training_model_bindings: Sequence[tuple[str, str | Path]] = (),
    tokenizer: Any | None = None,
) -> dict[str, Any]:
    """Resolve a source recipe into a one-way immutable training plan."""

    if scope not in SCOPES:
        raise TrainingPlanError(f"unsupported training plan scope: {scope}")
    root = Path(workspace_root).resolve()
    schema = Path(schema_path)
    source_path = Path(source_spec_path)
    if not source_path.is_absolute():
        source_path = root / source_path
    _logical_repo_path(source_path, root)
    source_spec = load_json(source_path)
    if not isinstance(source_spec, dict) or source_spec.get("schema_version") != SOURCE_RECIPE_SCHEMA_VERSION:
        raise TrainingPlanError("invalid Stage 1 source recipe")
    _check_no_future_bindings(source_spec, path="source_spec")
    if source_spec.get("scope") not in {scope, "formal" if scope == "pilot" else scope}:
        raise TrainingPlanError("requested scope disagrees with source recipe")
    decision_path = Path(decision_register_path)
    if not decision_path.is_absolute():
        decision_path = root / decision_path
    _logical_repo_path(decision_path, root)
    decision_register = load_json(decision_path)
    if not isinstance(decision_register, dict) or decision_register.get("status") != "frozen":
        raise TrainingPlanError("decision register must be frozen")
    context_locator, context_target = resolve_locator_ref(
        context_ref, expected_kind=CONTEXT_ARTIFACT_KIND
    )
    context_dependency = portable_dependency(context_locator, context_target, root)
    if scope == "engineering-smoke":
        if (
            training_evidence_ref is not None
            or train_partition_ref is not None
            or base_model_ref is not None
        ):
            raise TrainingPlanError(
                "engineering-smoke plan cannot bind training evidence, partition, or a training base model"
            )
        evidence_dependency = None
        partition_dependency = None
        base_dependency = None
        evidence_records: list[dict[str, Any]] = []
        evidence_budget: Mapping[str, Any] = {}
    else:
        if (
            training_evidence_ref is None
            or train_partition_ref is None
            or base_model_ref is None
        ):
            raise TrainingPlanError(
                "pilot/formal plans require training evidence, partition, and base model refs"
            )
        evidence_locator, evidence_target = resolve_locator_ref(
            training_evidence_ref, expected_kind=EVIDENCE_ARTIFACT_KIND
        )
        evidence_dependency = portable_dependency(evidence_locator, evidence_target, root)
        partition_locator, partition_target = resolve_locator_ref(
            train_partition_ref, expected_kind="train-partition"
        )
        partition_dependency = portable_dependency(
            partition_locator, partition_target, root
        )
        evidence_meta, evidence_records = validate_training_evidence_target(
            evidence_target,
            workspace_root=root,
            tokenizer=tokenizer,
        )
        from data.train_partition import load_train_partition

        partition = load_train_partition(
            train_partition_ref,
            workspace_root=root,
            expected_data_dependency=evidence_meta.get("data_dependency"),
        )
        if partition.partition_dependency != partition_dependency:
            raise TrainingPlanError(
                "train partition ref cannot be reproduced from finalized data"
            )
        if evidence_meta.get("train_partition_dependency") != partition_dependency:
            raise TrainingPlanError(
                "train partition ref does not match training evidence lineage"
            )
        if evidence_meta.get("base_model_dependency") is None:
            raise TrainingPlanError(
                "training evidence does not bind a registered base model"
            )
        if evidence_meta["context_dependency"] != context_dependency:
            raise TrainingPlanError("context ref does not match training evidence lineage")
        base_locator, base_target = resolve_locator_ref(
            base_model_ref, expected_kind=BASE_MODEL_ARTIFACT_KINDS
        )
        base_dependency = portable_dependency(base_locator, base_target, root)
        if evidence_meta.get("base_model_dependency") != base_dependency:
            raise TrainingPlanError(
                "base model ref does not match training evidence lineage"
            )
        evidence_budget = evidence_records[0]["budget"]
        for record in evidence_records[1:]:
            if (
                record["budget"]["max_sequence_tokens"]
                != evidence_budget["max_sequence_tokens"]
            ):
                raise TrainingPlanError(
                    "training evidence uses inconsistent sequence budgets"
                )
    environment_locator, environment_target = resolve_locator_ref(
        environment_ref, expected_kind=ENVIRONMENT_ARTIFACT_KINDS
    )
    environment_dependency = portable_dependency(environment_locator, environment_target, root)
    context_config = load_json(context_target / "config.resolved.json")
    if not isinstance(context_config, dict):
        raise TrainingPlanError("context resolved config is invalid")
    training_base_model: Mapping[str, Any] | None = None
    if scope != "engineering-smoke":
        assert base_dependency is not None
        training_base_model = _validate_training_base_dependency(
            base_dependency,
            environment_dependency,
            workspace_root=root,
        )
    if scope == "engineering-smoke":
        train_context_policy = None
        train_context_policy_sha256 = None
    else:
        try:
            train_context_policy, train_context_policy_sha256 = (
                context_policy_snapshot(
                    context_target,
                    expected_split="train",
                    require_scientific=True,
                )
            )
        except Exception as exc:
            raise TrainingPlanError(
                f"train context policy validation failed: {exc}"
            ) from exc
        assert evidence_meta is not None
        if (
            evidence_meta.get("train_context_policy") != train_context_policy
            or evidence_meta.get("train_context_policy_sha256")
            != train_context_policy_sha256
        ):
            raise TrainingPlanError(
                "training evidence and context policy lineage disagree"
            )
    protocol_snapshot = _protocol_snapshot(
        source_spec,
        context_config,
        root,
        train_context_policy=train_context_policy,
        train_context_policy_sha256=train_context_policy_sha256,
    )
    source_slots = source_spec.get("ordered_model_slots")
    if not isinstance(source_slots, list) or not source_slots:
        raise TrainingPlanError("source recipe has no ordered model slots")
    slots = [
        _resolve_slot(
            slot,
            scope=scope,
            workspace_root=root,
            evidence_budget=evidence_budget,
        )
        for slot in source_slots
        if isinstance(slot, Mapping)
    ]
    if len(slots) != len(source_slots):
        raise TrainingPlanError("source recipe contains a non-object model slot")
    pilot_slot_keys = source_spec.get("pilot_slot_keys")
    if not isinstance(pilot_slot_keys, list) or not all(
        isinstance(value, str) for value in pilot_slot_keys
    ):
        raise TrainingPlanError("source recipe pilot_slot_keys is invalid")
    epoch_policy = source_spec.get("epoch_and_checkpoint_selection_policy")
    rng_policy = source_spec.get("order_dropout_rng_policy")
    if scope == "engineering-smoke":
        epoch_policy = None
        rng_policy = None
    elif not isinstance(epoch_policy, Mapping) or not isinstance(rng_policy, Mapping):
        raise TrainingPlanError("training recipe lacks epoch/RNG policy")
    _validate_slot_set(
        slots,
        pilot_slot_keys,
        scope=scope,
        decision_register=decision_register,
        evidence_budget=evidence_budget,
    )
    if scope != "engineering-smoke":
        assert training_base_model is not None
        _validate_slot_base_bindings(
            slots, training_base_model, context_config
        )
    _validate_order_dropout_contract(slots, rng_policy, scope=scope)
    _validate_epoch_selection_contract(epoch_policy, scope=scope)
    projected_non_training = _project_non_training_model_bindings(
        non_training_model_bindings, root
    )
    ordered_non_training = {
        slot["model_key"]: projected_non_training[slot["model_key"]]
        for slot in slots
        if slot["model_key"] in projected_non_training
    }
    validated_non_training = _validate_non_training_model_dependencies(
        ordered_non_training,
        slots=slots,
        scope=scope,
        context_config=context_config,
        context_target=context_target,
        environment_dependency=environment_dependency,
        workspace_root=root,
    )
    train_path = Path(train_code_path)
    if not train_path.is_absolute():
        train_path = root / train_path
    _logical_repo_path(train_path, root)
    if not train_path.is_file():
        raise TrainingPlanError("train.py source does not exist")
    runtime_path = Path(runtime_code_path)
    if not runtime_path.is_absolute():
        runtime_path = root / runtime_path
    _logical_repo_path(runtime_path, root)
    if not runtime_path.is_file():
        raise TrainingPlanError("stage1_runtime.py source does not exist")
    builder_hash = sha256_file(Path(__file__))
    plan_fields: dict[str, Any] = {
        "schema_version": PLAN_SCHEMA_VERSION,
        "training_plan_id": "tpl-" + "0" * 64,
        "scope": scope,
        "scientific_eligible": scope == "formal",
        "context_dependency": context_dependency,
        "train_context_policy": train_context_policy,
        "train_context_policy_sha256": train_context_policy_sha256,
        "training_evidence_dependency": evidence_dependency,
        "train_partition_dependency": partition_dependency,
        "base_model_dependency": base_dependency,
        "environment_dependency": environment_dependency,
        "non_training_model_dependencies": validated_non_training,
        "ordered_model_slots": slots,
        "pilot_slot_keys": copy.deepcopy(pilot_slot_keys),
        "per_slot_train_config_sha256": {
            slot["model_key"]: slot["train_config_sha256"] for slot in slots
        },
        "epoch_and_checkpoint_selection_policy": copy.deepcopy(epoch_policy),
        "order_dropout_rng_policy": copy.deepcopy(rng_policy),
        "source_spec_sha256": canonical_sha256(source_spec),
        "protocol_snapshot_sha256": canonical_sha256(protocol_snapshot),
        "decision_register_sha256": canonical_sha256(decision_register),
        "train_code_sha256": sha256_file(train_path),
        "runtime_code_sha256": sha256_file(runtime_path),
        "plan_builder_code_sha256": builder_hash,
    }
    plan_fields["id_inputs"] = _expected_id_inputs(plan_fields)
    plan_fields["training_plan_id"] = "tpl-" + canonical_sha256(plan_fields["id_inputs"])
    _check_no_future_bindings(plan_fields)
    validate_json_schema(plan_fields, schema)
    plan_id = plan_fields["training_plan_id"]
    output_parent = (
        Path(target_root).resolve()
        if target_root is not None
        else context_target.parent.parent / "training_plans"
    )
    target = output_parent / plan_id
    staging = new_staging_directory(output_parent, plan_id)
    try:
        write_canonical_json(staging / "source_spec.json", source_spec)
        write_canonical_json(staging / "plan.resolved.json", plan_fields)
        write_canonical_json(staging / "protocol_snapshot.json", protocol_snapshot)
        write_canonical_json(staging / "decision_register.json", decision_register)
        write_canonical_json(
            staging / "training_evidence_ref.json", evidence_dependency
        )
        write_canonical_json(
            staging / "train_partition_ref.json", partition_dependency
        )
        write_canonical_json(staging / "base_model_ref.json", base_dependency)
        write_canonical_json(staging / "environment_ref.json", environment_dependency)
        write_canonical_json(
            staging / "non_training_model_refs.json", validated_non_training
        )
        write_canonical_json(
            staging / "provenance.json",
            {
                "schema_version": PROVENANCE_SCHEMA_VERSION,
                "training_plan_id": plan_id,
                "id_inputs": plan_fields["id_inputs"],
                "train_context_policy": train_context_policy,
                "train_context_policy_sha256": train_context_policy_sha256,
                "plan_builder_code_sha256": builder_hash,
                "train_code_sha256": plan_fields["train_code_sha256"],
                "runtime_code_sha256": plan_fields["runtime_code_sha256"],
                "train_partition_dependency": partition_dependency,
            },
        )
        payload_hash = finalize_target_atomic(
            staging,
            target,
            validate_staging=lambda path: _validate_target(
                path,
                workspace_root=root,
                schema_path=schema,
                require_directory_name=False,
                tokenizer=tokenizer,
            ),
        )
    except Exception:
        if staging.exists():
            import shutil

            shutil.rmtree(staging)
        raise
    return write_locator_ref(
        write_ref,
        artifact_kind=ARTIFACT_KIND,
        artifact_id=plan_id,
        target=target,
        payload_manifest_sha256=payload_hash,
    )


def load_training_plan(
    training_plan_ref: str | Path,
    *,
    workspace_root: str | Path = REPOSITORY_ROOT,
    schema_path: str | Path = DEFAULT_SCHEMA,
    tokenizer: Any | None = None,
) -> tuple[dict[str, Any], Path, dict[str, Any]]:
    locator, target = resolve_locator_ref(training_plan_ref, expected_kind=ARTIFACT_KIND)
    plan = _validate_target(
        target,
        workspace_root=Path(workspace_root).resolve(),
        schema_path=Path(schema_path),
        require_directory_name=True,
        tokenizer=tokenizer,
    )
    if locator["artifact_id"] != plan["training_plan_id"]:
        raise TrainingPlanError("training plan locator ID mismatch")
    if locator["payload_manifest_sha256"] != validate_payload_manifest(target):
        raise TrainingPlanError("training plan locator payload mismatch")
    return locator, target, plan


def validate_training_plan(
    training_plan_ref: str | Path,
    *,
    workspace_root: str | Path = REPOSITORY_ROOT,
    schema_path: str | Path = DEFAULT_SCHEMA,
    tokenizer: Any | None = None,
) -> dict[str, Any]:
    locator, _target, plan = load_training_plan(
        training_plan_ref,
        workspace_root=workspace_root,
        schema_path=schema_path,
        tokenizer=tokenizer,
    )
    return {
        "schema_version": "stage1-training-plan-validation-report/v1",
        "valid": True,
        "training_plan_id": plan["training_plan_id"],
        "payload_manifest_sha256": locator["payload_manifest_sha256"],
        "scope": plan["scope"],
        "ordered_model_keys": [
            slot["model_key"] for slot in plan["ordered_model_slots"]
        ],
        "pilot_slot_keys": plan["pilot_slot_keys"],
    }


def validate_training_plan_target(
    target: str | Path,
    *,
    workspace_root: str | Path = REPOSITORY_ROOT,
    schema_path: str | Path = DEFAULT_SCHEMA,
    tokenizer: Any | None = None,
) -> dict[str, Any]:
    """Validate an already-resolved plan target without a locator file."""

    return _validate_target(
        Path(target),
        workspace_root=Path(workspace_root).resolve(),
        schema_path=Path(schema_path),
        require_directory_name=True,
        tokenizer=tokenizer,
    )


__all__ = [
    "ARTIFACT_KIND",
    "PLAN_SCHEMA_VERSION",
    "TrainingPlanError",
    "freeze_training_plan",
    "load_training_plan",
    "recompute_training_plan_id",
    "validate_training_plan",
    "validate_training_plan_target",
]
