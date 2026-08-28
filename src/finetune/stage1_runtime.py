"""Fail-closed runtime bridge from frozen Stage 1 artifacts to Trainer.

No model is loaded here.  The bridge first verifies that the explicitly
supplied plan/evidence/schedule/base/environment locator refs project to the
same portable dependencies, then exposes only the selected plan slot's frozen
schedule rows to the training loop.
"""

from __future__ import annotations

import copy
import hashlib
import math
import os
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence

from torch.utils.data import Dataset as TorchDataset
from transformers import TrainerCallback

from data.training_artifacts import (
    TrainingArtifactError,
    canonical_json_bytes,
    canonical_sha256,
    load_json,
    portable_dependency,
    resolve_locator_ref,
    sha256_file,
    validate_dependency_ref,
    validate_json_schema,
)
from data.training_schedule import (
    CALIBRATION_PRESENTATION_POLICY,
    FIXED_PRESENTATION_CONFIG,
    PARTITION_ARTIFACT_KIND,
    PARTITION_CONFIG,
    TrainingScheduleError,
    load_model_epoch_records,
    load_training_schedule,
)


RUNTIME_VALIDATION_MARKER = "stage1-schedule-runtime-refs-validated/v1"
TRAINING_RECEIPT_SCHEMA = "stage1-training-receipt/v1"
RECEIPT_RECORD_POLICY = "completed-epoch-fit-and-calibration-record-hashes/v1"


class Stage1RuntimeError(TrainingArtifactError):
    """Raised before model loading when immutable training inputs disagree."""


@dataclass(frozen=True)
class Stage1RuntimeBundle:
    workspace_root: Path
    plan_locator: dict[str, Any]
    plan_target: Path
    plan_dependency: dict[str, Any]
    plan: dict[str, Any]
    evidence_locator: dict[str, Any]
    evidence_target: Path
    evidence_dependency: dict[str, Any]
    evidence_meta: dict[str, Any]
    partition_locator: dict[str, Any]
    partition_target: Path
    partition_dependency: dict[str, Any]
    partition_meta: dict[str, Any]
    partition_records: list[dict[str, Any]]
    schedule_locator: dict[str, Any]
    schedule_target: Path
    schedule_dependency: dict[str, Any]
    schedule_meta: dict[str, Any]
    base_model_target: Path
    base_model_document: dict[str, Any]
    base_model_path: Path
    tokenizer_path: Path
    base_model_dependency: dict[str, Any]
    environment_dependency: dict[str, Any]
    slot: dict[str, Any]
    resolved_config: dict[str, Any]
    records_by_epoch: dict[int, list[dict[str, Any]]]


def _workspace_root_from_dependency_target(
    target: Path, dependency: Mapping[str, Any]
) -> Path:
    frozen = validate_dependency_ref(dependency)
    logical = Path(frozen["logical_repo_path"])
    candidate = target.resolve()
    for _ in logical.parts:
        candidate = candidate.parent
    if (candidate / logical).resolve() != target.resolve():
        raise Stage1RuntimeError(
            f"locator target does not match portable path {logical.as_posix()}"
        )
    return candidate


def _assert_locator_projection(
    ref_path: str | Path,
    expected_dependency: Mapping[str, Any],
    workspace_root: Path,
) -> tuple[dict[str, Any], Path, dict[str, Any]]:
    expected = validate_dependency_ref(expected_dependency)
    locator, target = resolve_locator_ref(ref_path, expected["artifact_kind"])
    projected = portable_dependency(locator, target, workspace_root)
    if projected != expected:
        raise Stage1RuntimeError(
            f"{expected['artifact_kind']} locator does not match the training plan dependency"
        )
    return locator, target, projected


def _delete_dotted(document: dict[str, Any], dotted: str) -> None:
    pieces = dotted.split(".")
    current: Any = document
    for piece in pieces[:-1]:
        if not isinstance(current, dict) or piece not in current:
            return
        current = current[piece]
    if isinstance(current, dict):
        current.pop(pieces[-1], None)


def _assert_template_matches_plan(
    source_config: Mapping[str, Any], resolved_config: Mapping[str, Any], slot: Mapping[str, Any]
) -> None:
    if source_config.get("schema_version") != "stage1-train-config/v1":
        raise Stage1RuntimeError("formal Stage 1 runtime requires a Stage 1 train config")
    if source_config.get("model_role") != slot.get("role"):
        raise Stage1RuntimeError("CLI train config role differs from --model-key plan slot")
    if canonical_sha256(source_config) != slot.get("source_train_config_sha256"):
        raise Stage1RuntimeError("CLI train config hash differs from the immutable plan slot")
    resolved_hash = canonical_sha256(resolved_config)
    if resolved_hash != slot.get("resolved_train_config_sha256") or resolved_hash != slot.get(
        "train_config_sha256"
    ):
        raise Stage1RuntimeError("resolved train config hash differs from the immutable plan slot")
    source = copy.deepcopy(dict(source_config))
    resolved = copy.deepcopy(dict(resolved_config))
    template_fields = source.get("template_fields_resolved_by_training_plan")
    if not isinstance(template_fields, list) or any(
        not isinstance(value, str) or not value for value in template_fields
    ):
        raise Stage1RuntimeError("Stage 1 train config lacks template field declarations")
    if resolved.get("template_fields_resolved_by_training_plan") != template_fields:
        raise Stage1RuntimeError("plan changed the train-config template field declaration")
    for dotted in template_fields:
        _delete_dotted(source, dotted)
        _delete_dotted(resolved, dotted)
    # The source templates intentionally remain blocked on disk.  Readiness is
    # a runtime proof, never a mutable source-config bit.
    source.pop("execution_status", None)
    resolved.pop("execution_status", None)
    if canonical_json_bytes(source) != canonical_json_bytes(resolved):
        raise Stage1RuntimeError(
            "CLI train config differs from the immutable plan outside declared template fields"
        )


def _validate_resolved_slot_config(config: Mapping[str, Any], slot: Mapping[str, Any]) -> None:
    if config.get("model_role") != slot.get("role"):
        raise Stage1RuntimeError("resolved train config role differs from plan slot")
    if config.get("random_seed") != slot.get("seed"):
        raise Stage1RuntimeError("resolved train config random_seed differs from plan slot")
    training = config.get("training")
    if not isinstance(training, Mapping):
        raise Stage1RuntimeError("resolved train config lacks training settings")
    if training.get("seed") != slot.get("seed") or training.get("data_seed") != slot.get("seed"):
        raise Stage1RuntimeError("resolved Trainer seeds differ from plan slot")
    if training.get("num_train_epochs") != slot.get("epochs"):
        raise Stage1RuntimeError("resolved epoch count differs from plan slot")
    data = config.get("data")
    if not isinstance(data, Mapping) or data.get("source_schema") != "stage1-training-schedule/v1":
        raise Stage1RuntimeError("resolved config does not require the Stage 1 schedule loader")
    if data.get("train_data_path") not in (None, "") or data.get("val_data_path") not in (None, ""):
        raise Stage1RuntimeError("formal schedule-aware training forbids static train/dev paths")
    if data.get("partition") != PARTITION_CONFIG:
        raise Stage1RuntimeError(
            "resolved config does not consume the immutable train partition"
        )
    if data.get("fixed_presentation") != FIXED_PRESENTATION_CONFIG:
        raise Stage1RuntimeError(
            "resolved config does not freeze calibration presentation"
        )
    maximum = config.get("max_length")
    if not isinstance(maximum, int) or isinstance(maximum, bool) or maximum <= 0:
        raise Stage1RuntimeError("resolved Stage 1 max_length must be a positive integer")


def _validate_partition_policy_assertion(
    config: Mapping[str, Any], partition_meta: Mapping[str, Any]
) -> None:
    """Prove the nominal train-config hash policy matches the frozen artifact.

    Cluster membership and the final labels remain authoritative in the
    partition artifact.  This check only preserves the preregistered nominal
    hash threshold as an assertion; it never re-partitions at runtime.
    """

    data = config.get("data")
    calibration = data.get("calibration") if isinstance(data, Mapping) else None
    policy = partition_meta.get("policy")
    assignment = policy.get("assignment") if isinstance(policy, Mapping) else None
    if not isinstance(calibration, Mapping) or not isinstance(assignment, Mapping):
        raise Stage1RuntimeError("partition nominal hash policy is unavailable")
    expected = {
        "assignment": assignment.get("assignment"),
        "fraction": assignment.get("nominal_fraction"),
        "hash_modulus": assignment.get("hash_modulus"),
        "hash_threshold_exclusive": assignment.get(
            "hash_threshold_exclusive"
        ),
        "salt": assignment.get("salt"),
    }
    if any(calibration.get(key) != value for key, value in expected.items()):
        raise Stage1RuntimeError(
            "train config nominal hash assertion differs from the frozen partition"
        )
    if calibration.get("id_fields") != ["id", "query_id"]:
        raise Stage1RuntimeError(
            "train config nominal hash assertion uses unexpected ID fields"
        )


def _deep_validate_runtime_base(
    *,
    base_target: Path,
    base_dependency: Mapping[str, Any],
    environment_dependency: Mapping[str, Any],
    workspace_root: Path,
) -> tuple[dict[str, Any], Path, Path]:
    """Validate the registered base and derive the only permitted load paths.

    The train configuration is deliberately not consulted here.  Both paths
    come from the re-hashed inventories inside the registered base artifact.
    Validating a base artifact also validates its frozen environment against
    the currently running Python/backend versions.
    """

    try:
        from model.stage1_registry import (
            ModelRegistryError,
            validate_model_artifact_target,
        )

        model = validate_model_artifact_target(
            base_target,
            workspace_root=workspace_root,
            require_name=True,
        )
    except (ModelRegistryError, OSError, ValueError) as exc:
        raise Stage1RuntimeError(
            f"registered base model/current environment validation failed: {exc}"
        ) from exc
    if (
        model.get("artifact_type") != "base"
        or model.get("model_artifact_id") != base_dependency.get("artifact_id")
        or model.get("environment_dependency") != dict(environment_dependency)
    ):
        raise Stage1RuntimeError(
            "base model artifact identity/environment differs from the training plan"
        )
    checkpoint = model.get("checkpoint_inventory")
    tokenizer = model.get("tokenizer_inventory")
    if not isinstance(checkpoint, Mapping) or not isinstance(tokenizer, Mapping):
        raise Stage1RuntimeError("registered base model lacks frozen load inventories")

    def _resolve_inventory_path(snapshot: Mapping[str, Any], label: str) -> Path:
        logical = snapshot.get("logical_repo_path")
        if (
            not isinstance(logical, str)
            or not logical
            or Path(logical).is_absolute()
            or ".." in Path(logical).parts
            or Path(logical).as_posix() != logical
        ):
            raise Stage1RuntimeError(f"registered base {label} path is not portable")
        candidate = (workspace_root / logical).resolve()
        try:
            candidate.relative_to(workspace_root)
        except ValueError as exc:
            raise Stage1RuntimeError(
                f"registered base {label} path escapes the workspace"
            ) from exc
        if not candidate.is_dir() or candidate.is_symlink():
            raise Stage1RuntimeError(
                f"registered base {label} path is no longer a regular directory"
            )
        return candidate

    return (
        dict(model),
        _resolve_inventory_path(checkpoint, "checkpoint"),
        _resolve_inventory_path(tokenizer, "tokenizer"),
    )


def revalidate_stage1_runtime_sources(bundle: Stage1RuntimeBundle) -> None:
    """Re-hash model/tokenizer and environment immediately around model load."""

    model, model_path, tokenizer_path = _deep_validate_runtime_base(
        base_target=bundle.base_model_target,
        base_dependency=bundle.base_model_dependency,
        environment_dependency=bundle.environment_dependency,
        workspace_root=bundle.workspace_root,
    )
    if (
        model != bundle.base_model_document
        or model_path != bundle.base_model_path
        or tokenizer_path != bundle.tokenizer_path
    ):
        raise Stage1RuntimeError(
            "registered base model changed after runtime inputs were resolved"
        )


@contextmanager
def _verified_stage1_model_source_load(
    *,
    workspace_root: Path,
    base_model_document: Mapping[str, Any],
    base_model_path: Path,
    tokenizer_path: Path,
    source_names: Sequence[str],
) -> Iterator[Any]:
    """Guard one formal constructor using only a validated base artifact."""

    from model.stage1_registry import (
        ModelRegistryError,
        ResolvedModelSourceContract,
        verified_model_source_lease,
    )

    try:
        contract = ResolvedModelSourceContract(
            workspace_root=workspace_root.resolve(),
            checkpoint_inventory=copy.deepcopy(
                base_model_document["checkpoint_inventory"]
            ),
            tokenizer_inventory=copy.deepcopy(
                base_model_document["tokenizer_inventory"]
            ),
            base_inventory=copy.deepcopy(base_model_document["base_inventory"]),
        )
        with verified_model_source_lease(
            contract, source_names=source_names
        ) as verified:
            if (
                verified.checkpoint_path.resolve() != base_model_path.resolve()
                or verified.tokenizer_path.resolve() != tokenizer_path.resolve()
            ):
                raise Stage1RuntimeError(
                    "formal training load paths differ from the verified base contract"
                )
            yield verified
    except ModelRegistryError as exc:
        raise Stage1RuntimeError(
            f"formal training source verification failed: {exc}"
        ) from exc


@contextmanager
def verified_stage1_runtime_source_load(
    bundle: Stage1RuntimeBundle,
    *,
    source_names: Sequence[str],
) -> Iterator[Any]:
    """Guard one formal tokenizer/model constructor with fresh source proofs."""

    with _verified_stage1_model_source_load(
        workspace_root=bundle.workspace_root,
        base_model_document=bundle.base_model_document,
        base_model_path=bundle.base_model_path,
        tokenizer_path=bundle.tokenizer_path,
        source_names=source_names,
    ) as verified:
        yield verified


def _load_verified_runtime_tokenizer(
    *,
    workspace_root: Path,
    base_model_document: Mapping[str, Any],
    base_model_path: Path,
    tokenizer_path: Path,
) -> Any:
    """Load the replay tokenizer from the registered base, never context config."""

    try:
        from transformers import AutoTokenizer
    except ImportError as exc:  # pragma: no cover - environment contract
        raise Stage1RuntimeError(
            "transformers is required for formal Stage 1 tokenizer replay"
        ) from exc
    with _verified_stage1_model_source_load(
        workspace_root=workspace_root,
        base_model_document=base_model_document,
        base_model_path=base_model_path,
        tokenizer_path=tokenizer_path,
        source_names=("tokenizer",),
    ) as sources:
        tokenizer = AutoTokenizer.from_pretrained(
            str(sources.tokenizer_path),
            local_files_only=True,
            # Formal tokenizer inventories intentionally contain tokenizer
            # assets, not arbitrary executable Python modules.
            trust_remote_code=False,
        )
    if getattr(tokenizer, "eos_token_id", None) is None:
        raise Stage1RuntimeError("registered formal tokenizer has no EOS token")
    return tokenizer


def resolve_stage1_runtime_inputs(
    source_config: Mapping[str, Any],
    *,
    training_plan_ref: str | Path,
    training_evidence_ref: str | Path,
    train_partition_ref: str | Path,
    schedule_ref: str | Path,
    base_model_ref: str | Path,
    environment_ref: str | Path,
    model_key: str,
    tokenizer: Any | None = None,
) -> Stage1RuntimeBundle:
    """Resolve all explicit refs and reject every cross-artifact mismatch."""

    from data.train_partition import load_train_partition
    from data.training_evidence import load_training_evidence
    from data.training_plan import load_training_plan

    # Resolve the workspace root from the plan's portable evidence dependency
    # before invoking deep validators.  This keeps synthetic/temp workspaces as
    # rigorous as the repository-root formal workflow.
    _, raw_plan_target = resolve_locator_ref(training_plan_ref, "training-plan")
    raw_plan = load_json(raw_plan_target / "plan.resolved.json")
    if not isinstance(raw_plan, Mapping):
        raise Stage1RuntimeError("training plan target lacks plan.resolved.json")
    expected_evidence = validate_dependency_ref(
        raw_plan.get("training_evidence_dependency", {}), expected_kind="training-evidence"
    )
    expected_partition = validate_dependency_ref(
        raw_plan.get("train_partition_dependency", {}),
        expected_kind=PARTITION_ARTIFACT_KIND,
    )
    _, raw_evidence_target = resolve_locator_ref(
        training_evidence_ref, "training-evidence"
    )
    workspace_root = _workspace_root_from_dependency_target(
        raw_evidence_target, expected_evidence
    )
    expected_base = validate_dependency_ref(
        raw_plan.get("base_model_dependency", {}), expected_kind="stage1-model"
    )
    _, base_target, base_dependency = _assert_locator_projection(
        base_model_ref, expected_base, workspace_root
    )
    expected_environment = validate_dependency_ref(
        raw_plan.get("environment_dependency", {}),
        expected_kind=("stage1-environment", "environment"),
    )
    _, _, environment_dependency = _assert_locator_projection(
        environment_ref, expected_environment, workspace_root
    )
    base_model_document, base_model_path, tokenizer_path = _deep_validate_runtime_base(
        base_target=base_target,
        base_dependency=base_dependency,
        environment_dependency=environment_dependency,
        workspace_root=workspace_root,
    )
    tokenizer_object = (
        tokenizer
        if tokenizer is not None
        else _load_verified_runtime_tokenizer(
            workspace_root=workspace_root,
            base_model_document=base_model_document,
            base_model_path=base_model_path,
            tokenizer_path=tokenizer_path,
        )
    )
    plan_locator, plan_target, plan = load_training_plan(
        training_plan_ref,
        workspace_root=workspace_root,
        tokenizer=tokenizer_object,
    )
    evidence_locator, evidence_target, evidence_meta, _ = load_training_evidence(
        training_evidence_ref,
        workspace_root=workspace_root,
        tokenizer=tokenizer_object,
    )
    evidence_dependency = portable_dependency(
        evidence_locator, evidence_target, workspace_root
    )
    if evidence_dependency != expected_evidence:
        raise Stage1RuntimeError("training evidence ref differs from the training plan")
    partition = load_train_partition(
        train_partition_ref,
        workspace_root=workspace_root,
        expected_data_dependency=evidence_meta.get("data_dependency"),
    )
    partition_dependency = partition.partition_dependency
    if partition_dependency != expected_partition:
        raise Stage1RuntimeError("train partition ref differs from the training plan")
    if evidence_meta.get("train_partition_dependency") != partition_dependency:
        raise Stage1RuntimeError(
            "training evidence points to a different train partition"
        )
    plan_dependency = portable_dependency(plan_locator, plan_target, workspace_root)
    frozen_train_code_hash = plan.get("train_code_sha256")
    runtime_train_code = workspace_root / "src/finetune/train.py"
    if (
        not isinstance(frozen_train_code_hash, str)
        or not runtime_train_code.is_file()
        or sha256_file(runtime_train_code) != frozen_train_code_hash
    ):
        raise Stage1RuntimeError("runtime train.py differs from the immutable training plan")
    frozen_runtime_code_hash = plan.get("runtime_code_sha256")
    if (
        not isinstance(frozen_runtime_code_hash, str)
        or sha256_file(__file__) != frozen_runtime_code_hash
    ):
        raise Stage1RuntimeError(
            "stage1_runtime.py differs from the immutable training plan"
        )

    schedule_locator, schedule_target, schedule_meta = load_training_schedule(
        schedule_ref, tokenizer=tokenizer_object
    )
    schedule_dependency = portable_dependency(
        schedule_locator, schedule_target, workspace_root
    )
    if schedule_meta.get("training_plan_dependency") != plan_dependency:
        raise Stage1RuntimeError("schedule points to a different training plan")
    if schedule_meta.get("training_evidence_dependency") != evidence_dependency:
        raise Stage1RuntimeError("schedule points to different training evidence")
    if schedule_meta.get("train_partition_dependency") != partition_dependency:
        raise Stage1RuntimeError("schedule points to a different train partition")
    if schedule_meta.get("preflight_status") != "complete-pass":
        raise Stage1RuntimeError("schedule is not fully preflighted")

    if validate_dependency_ref(
        plan.get("base_model_dependency", {}), expected_kind="stage1-model"
    ) != expected_base:
        raise Stage1RuntimeError("validated plan base model differs from raw projection")
    if validate_dependency_ref(
        plan.get("environment_dependency", {}),
        expected_kind=("stage1-environment", "environment"),
    ) != expected_environment:
        raise Stage1RuntimeError("validated plan environment differs from raw projection")

    slots = plan.get("ordered_model_slots")
    if not isinstance(slots, list):
        raise Stage1RuntimeError("training plan lacks ordered model slots")
    selected = [slot for slot in slots if slot.get("model_key") == model_key]
    if len(selected) != 1:
        raise Stage1RuntimeError("--model-key must resolve to exactly one plan slot")
    slot = copy.deepcopy(selected[0])
    if slot.get("training_required") is not True:
        raise Stage1RuntimeError("selected plan slot does not authorize training")
    resolved_config = slot.get("train_config_resolved")
    if not isinstance(resolved_config, Mapping):
        raise Stage1RuntimeError("selected plan slot lacks a resolved train config")
    _assert_template_matches_plan(source_config, resolved_config, slot)
    _validate_resolved_slot_config(resolved_config, slot)
    _validate_partition_policy_assertion(resolved_config, partition.meta)
    frozen_model_path = resolved_config.get("model_path")
    expected_model_path = base_model_document["checkpoint_inventory"][
        "logical_repo_path"
    ]
    if frozen_model_path != expected_model_path:
        raise Stage1RuntimeError(
            "resolved train config model_path differs from the registered base inventory"
        )
    frozen_tokenizer_revision = base_model_document.get("tokenizer_contract", {}).get(
        "tokenizer_revision"
    )
    if schedule_meta.get("tokenizer_revision") != frozen_tokenizer_revision:
        raise Stage1RuntimeError(
            "training schedule tokenizer revision differs from the registered base"
        )
    deepspeed_logical = slot.get("deepspeed_config_logical_path")
    if not isinstance(deepspeed_logical, str) or not deepspeed_logical:
        raise Stage1RuntimeError("plan slot lacks a frozen DeepSpeed config path")
    deepspeed_path = (workspace_root / deepspeed_logical).resolve()
    try:
        deepspeed_path.relative_to(workspace_root)
    except ValueError as exc:
        raise Stage1RuntimeError("DeepSpeed config path escapes the workspace") from exc
    deepspeed_document = load_json(deepspeed_path)
    if (
        not isinstance(deepspeed_document, Mapping)
        or canonical_sha256(deepspeed_document) != slot.get("deepspeed_config_sha256")
        or dict(deepspeed_document) != slot.get("deepspeed_config_resolved")
    ):
        raise Stage1RuntimeError("DeepSpeed config differs from the immutable plan slot")
    resolved = copy.deepcopy(dict(resolved_config))
    runtime_training = resolved.get("training")
    if not isinstance(runtime_training, dict):
        raise Stage1RuntimeError("resolved Stage 1 training settings are not mutable JSON")
    # TrainingArguments accepts either a path or a resolved object.  A path is
    # unsafe here: DeepSpeed would open it later relative to the launcher's
    # current working directory, after this resolver had validated the
    # workspace copy.  Hand the backend an independent copy of the exact
    # object embedded in the immutable plan instead.
    frozen_deepspeed = copy.deepcopy(dict(slot["deepspeed_config_resolved"]))
    runtime_training["deepspeed"] = frozen_deepspeed
    # From this point on legacy loader helpers can only see paths derived from
    # the validated registry artifact, never a caller-controlled config path.
    resolved["model_path"] = str(base_model_path)
    resolved["tokenizer_path"] = str(tokenizer_path)
    resolved["_stage1_runtime_validation"] = {
        "schema_version": RUNTIME_VALIDATION_MARKER,
        "training_plan_id": plan["training_plan_id"],
        "training_evidence_build_id": evidence_meta["training_evidence_build_id"],
        "train_partition_id": partition.meta["train_partition_id"],
        "schedule_build_id": schedule_meta["schedule_build_id"],
        "model_key": model_key,
        "base_model_artifact_id": base_dependency["artifact_id"],
        "base_model_file_tree_sha256": base_model_document[
            "checkpoint_inventory"
        ]["file_tree_sha256"],
        "tokenizer_file_tree_sha256": base_model_document[
            "tokenizer_inventory"
        ]["file_tree_sha256"],
        "deepspeed_config_sha256": canonical_sha256(frozen_deepspeed),
        "runtime_code_sha256": frozen_runtime_code_hash,
    }

    registry_keys = [entry.get("model_key") for entry in schedule_meta.get("slot_registry", [])]
    if registry_keys.count(model_key) != 1:
        raise Stage1RuntimeError("selected model key is missing or duplicated in schedule registry")
    records_by_epoch = {
        epoch: load_model_epoch_records(
            schedule_target, model_key=model_key, epoch=epoch
        )
        for epoch in range(1, int(slot["epochs"]) + 1)
    }
    first_ids = [row["query_id"] for row in records_by_epoch[1]]
    partition_ids = [row["query_id"] for row in partition.rows]
    partition_by_id = {row["query_id"]: row for row in partition.rows}
    if first_ids != partition_ids:
        raise Stage1RuntimeError(
            "selected schedule does not retain the complete frozen train partition"
        )
    calibration_wire: dict[str, tuple[Any, ...]] = {}
    for epoch, rows in records_by_epoch.items():
        if [row["query_id"] for row in rows] != first_ids:
            raise Stage1RuntimeError(
                f"selected schedule changes query registry at epoch {epoch}"
            )
        if any(row["model_key"] != model_key for row in rows):
            raise Stage1RuntimeError("selected schedule contains a foreign model key")
        for row in rows:
            expected_label = partition_by_id[row["query_id"]]["partition"]
            if row.get("partition") != expected_label:
                raise Stage1RuntimeError(
                    "selected schedule partition label differs from the frozen artifact"
                )
            expected_presentation_epoch = (
                1 if expected_label == "calibration" else epoch
            )
            if row.get("presentation_epoch") != expected_presentation_epoch:
                raise Stage1RuntimeError(
                    "selected schedule presentation epoch is not reproducible"
                )
            if expected_label == "calibration":
                wire = (
                    tuple(row.get("ordered_demo_ids", [])),
                    row.get("use_lexicon"),
                    row.get("use_demos"),
                    row.get("instruction"),
                    row.get("input"),
                    row.get("output"),
                    row.get("rendered_prompt_sha256"),
                    row.get("rendered_prompt_tokens"),
                    row.get("sequence_tokens"),
                )
                previous = calibration_wire.setdefault(row["query_id"], wire)
                if previous != wire:
                    raise Stage1RuntimeError(
                        "calibration presentation changes across schedule epochs"
                    )

    return Stage1RuntimeBundle(
        workspace_root=workspace_root,
        plan_locator=plan_locator,
        plan_target=plan_target,
        plan_dependency=plan_dependency,
        plan=plan,
        evidence_locator=evidence_locator,
        evidence_target=evidence_target,
        evidence_dependency=evidence_dependency,
        evidence_meta=evidence_meta,
        partition_locator=partition.locator,
        partition_target=partition.target,
        partition_dependency=partition_dependency,
        partition_meta=partition.meta,
        partition_records=[dict(row) for row in partition.rows],
        schedule_locator=schedule_locator,
        schedule_target=schedule_target,
        schedule_dependency=schedule_dependency,
        schedule_meta=schedule_meta,
        base_model_target=base_target,
        base_model_document=base_model_document,
        base_model_path=base_model_path,
        tokenizer_path=tokenizer_path,
        base_model_dependency=base_dependency,
        environment_dependency=environment_dependency,
        slot=slot,
        resolved_config=resolved,
        records_by_epoch=records_by_epoch,
    )


class ScheduleAwareTokenizedDataset(TorchDataset):
    """A fixed-length dataset whose immutable row rendering changes by epoch."""

    def __init__(
        self,
        records_by_epoch: Mapping[int, Sequence[Mapping[str, Any]]],
        *,
        selected_query_ids: Sequence[str],
        encoder: Callable[[dict[str, Any], int], dict[str, list[int]]],
        fixed_epoch: int | None = None,
    ) -> None:
        self._records = {
            int(epoch): [dict(row) for row in rows]
            for epoch, rows in records_by_epoch.items()
        }
        if not self._records or sorted(self._records) != list(
            range(1, len(self._records) + 1)
        ):
            raise Stage1RuntimeError("schedule dataset epochs must be contiguous and one-based")
        selected = list(selected_query_ids)
        if not selected or len(selected) != len(set(selected)):
            raise Stage1RuntimeError("schedule dataset query partition is empty or duplicated")
        self._selected = frozenset(selected)
        if fixed_epoch is not None and fixed_epoch not in self._records:
            raise Stage1RuntimeError("fixed schedule dataset epoch is unavailable")
        self._fixed_epoch = fixed_epoch
        self._encoder = encoder
        self._epoch = 0
        self._rows: list[dict[str, Any]] = []
        self._encoded: list[dict[str, list[int]]] = []
        baseline: list[str] | None = None
        for epoch, rows in sorted(self._records.items()):
            ids = [row["query_id"] for row in rows if row["query_id"] in self._selected]
            if set(ids) != self._selected or len(ids) != len(self._selected):
                raise Stage1RuntimeError(
                    f"schedule partition is incomplete or duplicated at epoch {epoch}"
                )
            if baseline is None:
                baseline = ids
            elif ids != baseline:
                raise Stage1RuntimeError("schedule partition query order changes across epochs")
        self.set_epoch(1)

    @property
    def epoch(self) -> int:
        return self._epoch

    @property
    def record_hashes(self) -> list[str]:
        return [row["record_sha256"] for row in self._rows]

    @property
    def query_ids(self) -> list[str]:
        return [row["query_id"] for row in self._rows]

    def hashes_for_epoch(self, epoch: int) -> list[str]:
        epoch = self._fixed_epoch or epoch
        return [
            row["record_sha256"]
            for row in self._records[epoch]
            if row["query_id"] in self._selected
        ]

    def set_epoch(self, epoch: int) -> None:
        epoch = self._fixed_epoch or epoch
        if epoch not in self._records:
            raise Stage1RuntimeError(f"schedule dataset lacks epoch {epoch}")
        if epoch == self._epoch:
            return
        self._rows = [
            row for row in self._records[epoch] if row["query_id"] in self._selected
        ]
        self._encoded = [
            self._encoder(row, index) for index, row in enumerate(self._rows)
        ]
        self._epoch = epoch

    def __len__(self) -> int:
        return len(self._encoded)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        return self._encoded[index]


@dataclass
class ScheduleEpochTracker(TrainerCallback):
    train_dataset: ScheduleAwareTokenizedDataset
    eval_dataset: ScheduleAwareTokenizedDataset
    completed_epochs: list[int] = field(default_factory=list)
    epoch_global_steps: dict[int, int] = field(default_factory=dict)
    evaluation_history: list[dict[str, Any]] = field(default_factory=list)

    def on_epoch_begin(self, args, state, control, **kwargs):  # type: ignore
        epoch = int(math.floor(float(state.epoch or 0.0))) + 1
        self.train_dataset.set_epoch(epoch)
        self.eval_dataset.set_epoch(1)
        return control

    def on_epoch_end(self, args, state, control, **kwargs):  # type: ignore
        epoch = int(round(float(state.epoch or self.train_dataset.epoch)))
        if epoch <= 0:
            epoch = self.train_dataset.epoch
        if epoch not in self.completed_epochs:
            self.completed_epochs.append(epoch)
        step = int(getattr(state, "global_step", 0))
        if step <= 0:
            raise Stage1RuntimeError("completed schedule epoch lacks a positive global step")
        previous = self.epoch_global_steps.get(epoch)
        if previous is not None and previous != step:
            raise Stage1RuntimeError("schedule epoch was recorded with conflicting global steps")
        self.epoch_global_steps[epoch] = step
        return control

    def on_evaluate(self, args, state, control, metrics, **kwargs):  # type: ignore
        del kwargs
        raw_name = str(getattr(args, "metric_for_best_model", "eval_loss"))
        metric_name = raw_name if raw_name.startswith("eval_") else f"eval_{raw_name}"
        value = metrics.get(metric_name)
        if value is None or not math.isfinite(float(value)):
            raise Stage1RuntimeError(
                f"checkpoint-selection audit requires finite {metric_name}"
            )
        raw_epoch = float(getattr(state, "epoch", 0.0) or 0.0)
        epoch = int(round(raw_epoch))
        step = int(getattr(state, "global_step", 0))
        if epoch <= 0 or not math.isclose(raw_epoch, epoch, abs_tol=1e-9):
            raise Stage1RuntimeError(
                "checkpoint-selection evaluation is not an exact epoch boundary"
            )
        if step <= 0:
            raise Stage1RuntimeError(
                "checkpoint-selection evaluation lacks a positive global step"
            )
        row = {
            "epoch": epoch,
            "global_step": step,
            "metric_name": metric_name,
            "metric_value": float(value),
        }
        if self.evaluation_history and (
            epoch <= self.evaluation_history[-1]["epoch"]
            or step <= self.evaluation_history[-1]["global_step"]
        ):
            raise Stage1RuntimeError(
                "checkpoint-selection evaluations are duplicated or out of order"
            )
        self.evaluation_history.append(row)
        return control


def build_schedule_aware_datasets(
    bundle: Stage1RuntimeBundle,
    *,
    encoder: Callable[[dict[str, Any], int], dict[str, list[int]]],
) -> tuple[ScheduleAwareTokenizedDataset, ScheduleAwareTokenizedDataset, list[dict[str, Any]], ScheduleEpochTracker]:
    all_ids = [row["query_id"] for row in bundle.records_by_epoch[1]]
    fit = [
        row["query_id"]
        for row in bundle.records_by_epoch[1]
        if row.get("partition") == "fit"
    ]
    calibration = [
        row["query_id"]
        for row in bundle.records_by_epoch[1]
        if row.get("partition") == "calibration"
    ]
    if (
        len(fit) != len(set(fit))
        or len(calibration) != len(set(calibration))
        or set(fit).intersection(calibration)
        or set(fit + calibration) != set(all_ids)
    ):
        raise Stage1RuntimeError(
            "frozen schedule partition is duplicated, overlapping, or incomplete"
        )
    if not fit or not calibration:
        raise Stage1RuntimeError("frozen partition produced an empty fit or calibration set")
    train_dataset = ScheduleAwareTokenizedDataset(
        bundle.records_by_epoch, selected_query_ids=fit, encoder=encoder
    )
    eval_dataset = ScheduleAwareTokenizedDataset(
        bundle.records_by_epoch,
        selected_query_ids=calibration,
        encoder=encoder,
        fixed_epoch=1,
    )
    eval_raw = [
        dict(row)
        for row in bundle.records_by_epoch[1]
        if row["query_id"] in set(calibration)
    ]
    tracker = ScheduleEpochTracker(train_dataset=train_dataset, eval_dataset=eval_dataset)
    return train_dataset, eval_dataset, eval_raw, tracker


def snapshot_selected_checkpoint_inventory(
    checkpoint_dir: str | Path,
    *,
    workspace_root: str | Path,
    expected_global_step: int,
) -> dict[str, Any]:
    """Freeze the exact stable regular-file tree selected by Trainer.

    The checkpoint writer and tokenizer save have completed before this helper
    is called.  The shared registry scanner opens every file without following
    symlinks and rejects a tree whose entries or stat signatures drift during
    hashing.
    """

    if (
        isinstance(expected_global_step, bool)
        or not isinstance(expected_global_step, int)
        or expected_global_step <= 0
    ):
        raise Stage1RuntimeError(
            "selected checkpoint inventory requires a positive global step"
        )
    checkpoint = Path(checkpoint_dir)
    expected_name = f"checkpoint-{expected_global_step}"
    if checkpoint.name != expected_name:
        raise Stage1RuntimeError(
            "selected checkpoint path does not match its global step"
        )
    try:
        # Imported lazily so ordinary schedule/runtime resolution remains
        # independent of the model registry module.
        from model.stage1_registry import inventory_regular_file_tree

        inventory = inventory_regular_file_tree(
            checkpoint,
            workspace_root=workspace_root,
            label="selected training checkpoint",
            inventory_policy="all-regular-files/v1",
        )
    except TrainingArtifactError as exc:
        raise Stage1RuntimeError(
            f"cannot freeze selected checkpoint inventory: {exc}"
        ) from exc
    if Path(str(inventory.get("logical_repo_path", ""))).name != expected_name:
        raise Stage1RuntimeError(
            "selected checkpoint inventory logical path differs from its global step"
        )
    return inventory


def build_training_receipt(
    bundle: Stage1RuntimeBundle,
    tracker: ScheduleEpochTracker,
    *,
    training_exit_global_step: int,
    selected_checkpoint_global_step: int,
    selected_checkpoint_dir: str | Path,
    train_code_sha256: str,
    trainer_best_metric: float,
    trainer_best_global_step: int,
) -> dict[str, Any]:
    completed = sorted(set(tracker.completed_epochs))
    if not completed or completed != list(range(1, max(completed) + 1)):
        raise Stage1RuntimeError("training receipt requires contiguous completed epochs")
    if max(completed) > int(bundle.slot["epochs"]):
        raise Stage1RuntimeError("completed epoch exceeds the immutable plan")
    if (
        isinstance(training_exit_global_step, bool)
        or not isinstance(training_exit_global_step, int)
        or training_exit_global_step <= 0
    ):
        raise Stage1RuntimeError("training receipt requires a positive exit global step")
    if (
        isinstance(selected_checkpoint_global_step, bool)
        or not isinstance(selected_checkpoint_global_step, int)
        or selected_checkpoint_global_step <= 0
        or selected_checkpoint_global_step > training_exit_global_step
    ):
        raise Stage1RuntimeError("training receipt has an invalid selected checkpoint step")
    selected_checkpoint_inventory = snapshot_selected_checkpoint_inventory(
        selected_checkpoint_dir,
        workspace_root=bundle.workspace_root,
        expected_global_step=selected_checkpoint_global_step,
    )
    if not isinstance(train_code_sha256, str) or len(train_code_sha256) != 64:
        raise Stage1RuntimeError("training receipt requires a valid train code hash")
    runtime_code_sha256 = bundle.plan.get("runtime_code_sha256")
    if (
        not isinstance(runtime_code_sha256, str)
        or runtime_code_sha256 != sha256_file(__file__)
    ):
        raise Stage1RuntimeError(
            "training receipt runtime code differs from the immutable plan"
        )
    epochs = []
    fit_query_ids = [
        row["query_id"]
        for row in bundle.records_by_epoch[1]
        if row.get("partition") == "fit"
    ]
    calibration_query_ids = [
        row["query_id"]
        for row in bundle.records_by_epoch[1]
        if row.get("partition") == "calibration"
    ]
    if (
        fit_query_ids != tracker.train_dataset.query_ids
        or calibration_query_ids != tracker.eval_dataset.query_ids
    ):
        raise Stage1RuntimeError(
            "runtime datasets differ from the frozen fit/calibration partition"
        )
    if (
        bundle.partition_meta.get("fit_count") != len(fit_query_ids)
        or bundle.partition_meta.get("calibration_count")
        != len(calibration_query_ids)
        or bundle.partition_meta.get("fit_ids_sha256")
        != canonical_sha256(fit_query_ids)
        or bundle.partition_meta.get("calibration_ids_sha256")
        != canonical_sha256(calibration_query_ids)
    ):
        raise Stage1RuntimeError(
            "runtime fit/calibration IDs differ from partition metadata"
        )
    calibration_baseline_hash: str | None = None
    for epoch in completed:
        fit_hashes = tracker.train_dataset.hashes_for_epoch(epoch)
        calibration_hashes = tracker.eval_dataset.hashes_for_epoch(epoch)
        schedule_rows = bundle.records_by_epoch[epoch]
        all_hashes = [row["record_sha256"] for row in schedule_rows]
        epoch_fit_ids = [
            row["query_id"] for row in schedule_rows if row.get("partition") == "fit"
        ]
        epoch_calibration_ids = [
            row["query_id"]
            for row in schedule_rows
            if row.get("partition") == "calibration"
        ]
        if (
            epoch_fit_ids != fit_query_ids
            or epoch_calibration_ids != calibration_query_ids
        ):
            raise Stage1RuntimeError(
                "receipt fit/calibration schedule partition changes across epochs"
            )
        calibration_digest = canonical_sha256(calibration_hashes)
        if calibration_baseline_hash is None:
            calibration_baseline_hash = calibration_digest
        elif calibration_digest != calibration_baseline_hash:
            raise Stage1RuntimeError(
                "calibration evaluation presentation changes across epochs"
            )
        epochs.append(
            {
                "epoch": epoch,
                "global_step": tracker.epoch_global_steps.get(epoch),
                "all_records_sha256": canonical_sha256(all_hashes),
                "fit_records_sha256": canonical_sha256(fit_hashes),
                "calibration_records_sha256": calibration_digest,
                "fit_record_count": len(fit_hashes),
                "calibration_record_count": len(calibration_hashes),
            }
        )
    if any(entry["global_step"] is None for entry in epochs):
        raise Stage1RuntimeError("training receipt lacks epoch-to-global-step lineage")
    selected_epochs = [
        entry["epoch"]
        for entry in epochs
        if entry["global_step"] == selected_checkpoint_global_step
    ]
    if len(selected_epochs) != 1:
        raise Stage1RuntimeError(
            "selected checkpoint step must equal exactly one completed epoch boundary"
        )
    resolved_slot = bundle.slot.get("train_config_resolved")
    policy = (
        resolved_slot.get("early_stopping")
        if isinstance(resolved_slot, Mapping)
        else None
    )
    required_policy = {
        "enabled",
        "metric",
        "mode",
        "minimum_epochs",
        "maximum_epochs",
        "patience_evaluations",
        "threshold",
        "tie_break",
        "selection_data",
        "scientific_dev_used_for_selection",
    }
    if (
        not isinstance(policy, Mapping)
        or set(policy) != required_policy
        or policy.get("enabled") is not True
        or policy.get("metric") != "eval_loss"
        or policy.get("mode") != "min"
        or policy.get("tie_break") != "earliest-global-step"
        or policy.get("selection_data") != "train-only-calibration"
        or policy.get("scientific_dev_used_for_selection") is not False
        or policy.get("maximum_epochs") != bundle.slot["epochs"]
    ):
        raise Stage1RuntimeError(
            "training receipt lacks the exact frozen early-stopping policy"
        )
    history = [dict(row) for row in tracker.evaluation_history]
    if [row.get("epoch") for row in history] != completed:
        raise Stage1RuntimeError(
            "checkpoint-selection history does not cover every completed epoch exactly once"
        )
    if [row.get("global_step") for row in history] != [
        entry["global_step"] for entry in epochs
    ]:
        raise Stage1RuntimeError(
            "checkpoint-selection history differs from epoch/global-step lineage"
        )
    threshold = float(policy["threshold"])
    patience_limit = int(policy["patience_evaluations"])
    minimum_epochs = int(policy["minimum_epochs"])
    if threshold < 0 or patience_limit <= 0 or minimum_epochs <= 0:
        raise Stage1RuntimeError("frozen early-stopping policy has invalid bounds")
    best_value: float | None = None
    best_step: int | None = None
    patience = 0
    audited_history: list[dict[str, Any]] = []
    for raw in history:
        if set(raw) != {"epoch", "global_step", "metric_name", "metric_value"}:
            raise Stage1RuntimeError(
                "checkpoint-selection history row is non-canonical"
            )
        value = float(raw["metric_value"])
        if raw["metric_name"] != "eval_loss" or not math.isfinite(value):
            raise Stage1RuntimeError(
                "checkpoint-selection history contains an invalid eval_loss"
            )
        qualifying = best_value is None or best_value - value > threshold
        if qualifying:
            best_value = value
            best_step = int(raw["global_step"])
            patience = 0
        else:
            patience += 1
        audited_history.append(
            {
                **raw,
                "metric_value": value,
                "qualifying_improvement": qualifying,
                "patience_counter_after": patience,
                "best_global_step_after": best_step,
            }
        )
    if best_value is None or best_step is None:
        raise Stage1RuntimeError("checkpoint-selection history is empty")
    if best_step != selected_checkpoint_global_step:
        raise Stage1RuntimeError(
            "selected checkpoint is not the threshold-aware earliest eval_loss winner"
        )
    if (
        isinstance(trainer_best_global_step, bool)
        or trainer_best_global_step != best_step
        or not math.isfinite(float(trainer_best_metric))
        or float(trainer_best_metric) != best_value
    ):
        raise Stage1RuntimeError(
            "Trainer best-metric state differs from the replayed checkpoint winner"
        )
    if training_exit_global_step != epochs[-1]["global_step"]:
        raise Stage1RuntimeError(
            "training exit step is not the final completed epoch boundary"
        )
    first_patience_stop_epoch = next(
        (
            row["epoch"]
            for row in audited_history
            if row["epoch"] >= minimum_epochs
            and row["patience_counter_after"] >= patience_limit
        ),
        None,
    )
    if completed[-1] < int(bundle.slot["epochs"]):
        if first_patience_stop_epoch != completed[-1]:
            raise Stage1RuntimeError(
                "training exit is not the first epoch satisfying the frozen patience rule"
            )
        stop_reason = "early-stopping-patience"
    else:
        if (
            first_patience_stop_epoch is not None
            and first_patience_stop_epoch < completed[-1]
        ):
            raise Stage1RuntimeError(
                "training continued after the frozen early-stopping boundary"
            )
        stop_reason = "maximum-epochs"
    checkpoint_selection_audit = {
        "schema_version": "stage1-checkpoint-selection-audit/v1",
        "policy": copy.deepcopy(dict(policy)),
        "evaluation_history": audited_history,
        "evaluation_history_sha256": canonical_sha256(audited_history),
        "selected_metric_value": best_value,
        "selected_global_step": best_step,
        "stop_reason": stop_reason,
    }
    receipt = {
        "schema_version": TRAINING_RECEIPT_SCHEMA,
        "training_plan_dependency": bundle.plan_dependency,
        "training_evidence_dependency": bundle.evidence_dependency,
        "train_partition_dependency": bundle.partition_dependency,
        "schedule_dependency": bundle.schedule_dependency,
        "base_model_dependency": bundle.base_model_dependency,
        "environment_dependency": bundle.environment_dependency,
        "model_key": bundle.slot["model_key"],
        "role": bundle.slot["role"],
        "seed": bundle.slot["seed"],
        "planned_epochs": bundle.slot["epochs"],
        "completed_epochs": completed,
        "completed_epoch_registry": epochs,
        "selected_checkpoint_epoch": selected_epochs[0],
        "selected_checkpoint_global_step": selected_checkpoint_global_step,
        "selected_checkpoint_inventory": selected_checkpoint_inventory,
        "training_exit_global_step": training_exit_global_step,
        "global_step": selected_checkpoint_global_step,
        "schedule_record_policy": RECEIPT_RECORD_POLICY,
        "fit_query_ids_sha256": canonical_sha256(fit_query_ids),
        "fit_query_count": len(fit_query_ids),
        "calibration_query_ids_sha256": canonical_sha256(
            calibration_query_ids
        ),
        "calibration_query_count": len(calibration_query_ids),
        "fixed_presentation": copy.deepcopy(FIXED_PRESENTATION_CONFIG),
        "final_checkpoint_rule": bundle.slot["final_checkpoint_rule"],
        "checkpoint_selection_audit": checkpoint_selection_audit,
        "train_config_sha256": bundle.slot["train_config_sha256"],
        "train_code_sha256": train_code_sha256,
        "runtime_code_sha256": runtime_code_sha256,
    }
    receipt["receipt_sha256"] = canonical_sha256(receipt)
    validate_json_schema(
        receipt,
        Path(__file__).resolve().parents[2]
        / "schemas"
        / "stage1_training_receipt_v1.schema.json",
    )
    return receipt


def write_immutable_training_receipt(path: str | Path, receipt: Mapping[str, Any]) -> None:
    destination = Path(path)
    payload = canonical_json_bytes(receipt) + b"\n"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.is_symlink() or destination.read_bytes() != payload:
            raise Stage1RuntimeError("refusing to overwrite a different training receipt")
        return
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{destination.name}.", dir=destination.parent, delete=False
        ) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, destination)
        except FileExistsError:
            if destination.is_symlink() or destination.read_bytes() != payload:
                raise Stage1RuntimeError(
                    "another process published a different training receipt"
                )
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


__all__ = [
    "RUNTIME_VALIDATION_MARKER",
    "ScheduleAwareTokenizedDataset",
    "ScheduleEpochTracker",
    "Stage1RuntimeBundle",
    "Stage1RuntimeError",
    "build_schedule_aware_datasets",
    "build_training_receipt",
    "revalidate_stage1_runtime_sources",
    "resolve_stage1_runtime_inputs",
    "snapshot_selected_checkpoint_inventory",
    "verified_stage1_runtime_source_load",
    "write_immutable_training_receipt",
]
