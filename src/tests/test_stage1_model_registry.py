from __future__ import annotations

import copy
import importlib.metadata
import json
import platform
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator
from unittest.mock import patch

from data.training_artifacts import (
    TrainingArtifactError,
    build_payload_manifest,
    canonical_sha256,
    load_json,
    load_jsonl,
    portable_dependency,
    resolve_locator_ref,
    validate_payload_manifest,
    write_bytes_atomic,
    write_canonical_json,
    write_locator_ref,
)
from data.training_plan import TrainingPlanError, freeze_training_plan, load_training_plan
from data.training_schedule import (
    build_training_schedule,
    load_model_epoch_records,
    tokenizer_revision_from_directory,
)
from model import stage1_registry as registry_module
from model.stage1_registry import (
    ModelRegistryError,
    _assert_receipt_checkpoint_binding,
    _validate_raw_receipt,
    _validate_receipt_cross_lineage,
    finalize_model_registry,
    inventory_regular_file_tree,
    register_base_model,
    register_legacy_model,
    register_trained_model,
    register_training_receipt,
    resolve_registered_model,
    resolve_registered_model_dependency,
    validate_model_artifact,
    validate_model_registry,
)
from scripts.stage1 import build_training_plan as plan_cli
from scripts.stage1 import register_model as model_cli
from scripts.stage1.capture_environment import (
    CRITICAL_DISTRIBUTIONS,
    build_environment_document,
    build_payload_manifest as build_environment_payload_manifest,
)
from tests.test_stage1_training_artifacts import (
    SyntheticTrainingWorkspace,
    deterministic_context_tokenizer_constructor,
    deterministic_formal_score_replayer,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
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


def _environment_ref(root: Path) -> Path:
    distributions = [
        {
            "installer": "conda",
            "name": "python",
            "version": platform.python_version(),
            "build": "synthetic_0",
            "build_number": 0,
            "subdir": "linux-64",
            "package_sha256_or_null": "a" * 64,
            "package_md5_or_null": None,
        }
    ]
    distributions.extend(
        {
            "installer": "python",
            "name": name,
            "version": importlib.metadata.version(name),
        }
        for name in CRITICAL_DISTRIBUTIONS
    )
    distributions.sort(key=lambda row: (row["installer"], row["name"]))
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - Stage 1 environment contract
        raise AssertionError("Stage 1 test environment requires torch") from exc
    torch_build = {
        "version": str(torch.__version__),
        "cuda_version": str(torch.version.cuda) if torch.version.cuda else None,
        "git_version": str(torch.version.git_version) if torch.version.git_version else None,
        "cudnn_version": None,
        "debug_build": bool(torch.version.debug),
    }
    document = build_environment_document(
        environment_spec_sha256="d" * 64,
        installed_distributions=distributions,
        torch_build=torch_build,
        driver_version="synthetic-driver",
        gpu_architecture=[
            {
                "name": "Synthetic L20",
                "compute_capability": "8.9",
                "memory_total_mib": 46068,
                "count": 4,
            }
        ],
        capture_code_sha256="e" * 64,
    )
    target = root / "artifacts/environments" / document["environment_build_id"]
    target.mkdir(parents=True)
    write_canonical_json(target / "environment.json", document)
    write_canonical_json(
        target / "provenance.json",
        {"schema_version": "stage1-environment-provenance/v1"},
    )
    write_canonical_json(
        target / "payload_manifest.json", build_environment_payload_manifest(target)
    )
    ref = root / "refs/environment_ref.json"
    write_locator_ref(
        ref,
        artifact_kind="stage1-environment",
        artifact_id=document["environment_build_id"],
        target=target,
        payload_manifest_sha256=validate_payload_manifest(target),
    )
    return ref


def _model_tree(root: Path, name: str = "base", *, sharded: bool = False) -> Path:
    target = root / "sources" / name
    target.mkdir(parents=True)
    write_canonical_json(
        target / "config.json",
        {"model_type": "qwen3", "architectures": ["Qwen3ForCausalLM"]},
    )
    write_canonical_json(
        target / "tokenizer_config.json",
        {
            "eos_token": "<eos>",
            "pad_token": "<pad>",
            "chat_template": (
                "{% if add_generation_prompt %}assistant{% endif %}"
                "{% if enable_thinking is false %}no-think{% endif %}"
            ),
        },
    )
    write_canonical_json(target / "tokenizer.json", {"version": "1.0"})
    if sharded:
        write_bytes_atomic(target / "model-00001-of-00002.safetensors", b"shard-one")
        write_bytes_atomic(target / "model-00002-of-00002.safetensors", b"shard-two")
        write_canonical_json(
            target / "model.safetensors.index.json",
            {
                "metadata": {},
                "weight_map": {
                    "a": "model-00001-of-00002.safetensors",
                    "b": "model-00002-of-00002.safetensors",
                },
            },
        )
    else:
        write_bytes_atomic(target / "model.safetensors", b"synthetic-weights")
    return target


def _legacy_model_tree(root: Path, name: str = "legacy-checkpoint") -> Path:
    target = root / "sources" / name
    target.mkdir(parents=True)
    write_canonical_json(
        target / "config.json",
        {"model_type": "qwen2", "architectures": ["Qwen2ForCausalLM"]},
    )
    write_canonical_json(
        target / "tokenizer_config.json",
        {
            "tokenizer_class": "Qwen2Tokenizer",
            "eos_token": "<|im_end|>",
            "pad_token": "<|endoftext|>",
        },
    )
    write_canonical_json(target / "tokenizer.json", {"version": "1.0"})
    (target / "chat_template.jinja").write_text(
        "{% for message in messages %}{{ message.content }}{% endfor %}"
        "{% if add_generation_prompt %}assistant{% endif %}",
        encoding="utf-8",
    )
    write_bytes_atomic(target / "model.safetensors", b"legacy-weights")
    write_bytes_atomic(
        target / "global_step10/optimizer-state.bin", b"legacy-optimizer-state"
    )
    return target


def _dependency(ref: Path, root: Path) -> dict:
    locator, target = resolve_locator_ref(ref)
    return portable_dependency(locator, target, root)


def _make_raw_receipt(
    *,
    training_plan_dependency: dict,
    training_evidence_dependency: dict,
    train_partition_dependency: dict,
    schedule_dependency: dict,
    base_model_dependency: dict,
    environment_dependency: dict,
    model_key: str,
    role: str,
    seed: int,
    train_config_sha256: str,
    train_code_sha256: str,
    runtime_code_sha256: str,
    policy: dict,
) -> dict:
    completed_epochs = [1, 2, 3, 4]
    completed_registry = [
        {
            "epoch": epoch,
            "global_step": epoch * 10,
            "all_records_sha256": str(epoch) * 64,
            "fit_records_sha256": str(epoch + 4) * 64,
            "calibration_records_sha256": "a" * 64,
            "fit_record_count": 1,
            "calibration_record_count": 1,
        }
        for epoch in completed_epochs
    ]
    evaluation_history = [
        {
            "epoch": epoch,
            "global_step": epoch * 10,
            "metric_name": "eval_loss",
            "metric_value": metric,
            "qualifying_improvement": epoch == 1,
            "patience_counter_after": epoch - 1,
            "best_global_step_after": 10,
        }
        for epoch, metric in zip(
            completed_epochs, (1.0, 1.01, 1.02, 1.03), strict=True
        )
    ]
    selection_audit = {
        "schema_version": "stage1-checkpoint-selection-audit/v1",
        "policy": policy,
        "evaluation_history": evaluation_history,
        "evaluation_history_sha256": canonical_sha256(evaluation_history),
        "selected_metric_value": 1.0,
        "selected_global_step": 10,
        "stop_reason": "early-stopping-patience",
    }
    receipt = {
        "schema_version": "stage1-training-receipt/v1",
        "training_plan_dependency": training_plan_dependency,
        "training_evidence_dependency": training_evidence_dependency,
        "train_partition_dependency": train_partition_dependency,
        "schedule_dependency": schedule_dependency,
        "base_model_dependency": base_model_dependency,
        "environment_dependency": environment_dependency,
        "model_key": model_key,
        "role": role,
        "seed": seed,
        "planned_epochs": 5,
        "completed_epochs": completed_epochs,
        "completed_epoch_registry": completed_registry,
        "selected_checkpoint_epoch": 1,
        "selected_checkpoint_global_step": 10,
        "training_exit_global_step": 40,
        "global_step": 10,
        "schedule_record_policy": "completed-epoch-fit-and-calibration-record-hashes/v1",
        "fit_query_ids_sha256": "b" * 64,
        "fit_query_count": 1,
        "calibration_query_ids_sha256": "c" * 64,
        "calibration_query_count": 1,
        "fixed_presentation": {
            "policy": "calibration-epoch-1-wire/v1",
            "wire_epoch": 1,
            "demo_order_across_epochs": True,
            "source_mask_across_epochs": True,
        },
        "final_checkpoint_rule": "best-eval-loss-threshold-earliest/v1",
        "checkpoint_selection_audit": selection_audit,
        "train_config_sha256": train_config_sha256,
        "train_code_sha256": train_code_sha256,
        "runtime_code_sha256": runtime_code_sha256,
    }
    receipt["receipt_sha256"] = canonical_sha256(receipt)
    return receipt


def _raw_receipt(
    workspace: SyntheticTrainingWorkspace,
    model_key: str,
    checkpoint: Path,
) -> dict:
    plan_locator, plan_target = resolve_locator_ref(
        workspace.refs / "training_plan_ref.json", "training-plan"
    )
    plan = json.loads((plan_target / "plan.resolved.json").read_text(encoding="utf-8"))
    slot = next(row for row in plan["ordered_model_slots"] if row["model_key"] == model_key)
    partition_target = Path(
        workspace.root / plan["train_partition_dependency"]["logical_repo_path"]
    )
    partition_rows = load_jsonl(partition_target / "partition.jsonl")
    fit_ids = [row["query_id"] for row in partition_rows if row["partition"] == "fit"]
    calibration_ids = [
        row["query_id"]
        for row in partition_rows
        if row["partition"] == "calibration"
    ]
    schedule_target = resolve_locator_ref(
        workspace.refs / "schedule_ref.json", "training-schedule"
    )[1]
    calibration_hashes = [
        row["record_sha256"]
        for row in load_model_epoch_records(
            schedule_target, model_key=model_key, epoch=1
        )
        if row["partition"] == "calibration"
    ]
    receipt = _make_raw_receipt(
        training_plan_dependency=portable_dependency(
            plan_locator, plan_target, workspace.root
        ),
        training_evidence_dependency=plan["training_evidence_dependency"],
        train_partition_dependency=plan["train_partition_dependency"],
        schedule_dependency=_dependency(
            workspace.refs / "schedule_ref.json", workspace.root
        ),
        base_model_dependency=plan["base_model_dependency"],
        environment_dependency=plan["environment_dependency"],
        model_key=model_key,
        role=slot["role"],
        seed=slot["seed"],
        train_config_sha256=slot["train_config_sha256"],
        train_code_sha256=plan["train_code_sha256"],
        runtime_code_sha256=plan["runtime_code_sha256"],
        policy=copy.deepcopy(slot["train_config_resolved"]["early_stopping"]),
    )
    receipt["fit_query_ids_sha256"] = canonical_sha256(fit_ids)
    receipt["fit_query_count"] = len(fit_ids)
    receipt["calibration_query_ids_sha256"] = canonical_sha256(calibration_ids)
    receipt["calibration_query_count"] = len(calibration_ids)
    receipt["selected_checkpoint_inventory"] = inventory_regular_file_tree(
        checkpoint,
        workspace_root=workspace.root,
        label="selected fixture checkpoint",
        inventory_policy="all-regular-files/v1",
    )
    for entry in receipt["completed_epoch_registry"]:
        schedule_rows = load_model_epoch_records(
            schedule_target, model_key=model_key, epoch=entry["epoch"]
        )
        entry.update(
            {
                "all_records_sha256": canonical_sha256(
                    [row["record_sha256"] for row in schedule_rows]
                ),
                "fit_records_sha256": canonical_sha256(
                    [
                        row["record_sha256"]
                        for row in schedule_rows
                        if row["partition"] == "fit"
                    ]
                ),
                "calibration_records_sha256": canonical_sha256(
                    calibration_hashes
                ),
                "fit_record_count": len(fit_ids),
                "calibration_record_count": len(calibration_ids),
            }
        )
    receipt["receipt_sha256"] = canonical_sha256(
        {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    )
    return receipt


def _standalone_raw_receipt() -> dict:
    def dependency(kind: str, marker: str) -> dict:
        return {
            "schema_version": "stage1-dependency-ref/v1",
            "artifact_kind": kind,
            "artifact_id": f"{kind}-{marker * 8}",
            "payload_manifest_sha256": marker * 64,
            "logical_repo_path": f"artifacts/{kind}/{marker * 8}",
        }

    return _make_raw_receipt(
        training_plan_dependency=dependency("training-plan", "1"),
        training_evidence_dependency=dependency("training-evidence", "2"),
        train_partition_dependency=dependency("train-partition", "9"),
        schedule_dependency=dependency("training-schedule", "3"),
        base_model_dependency=dependency("stage1-model", "4"),
        environment_dependency=dependency("stage1-environment", "5"),
        model_key="M_LD/seed-42",
        role="M_LD",
        seed=42,
        train_config_sha256="6" * 64,
        train_code_sha256="7" * 64,
        runtime_code_sha256="8" * 64,
        policy=copy.deepcopy(EARLY_STOPPING_POLICY),
    )


def _rehash_receipt(receipt: dict, *, history: bool = False) -> None:
    if history:
        audit = receipt["checkpoint_selection_audit"]
        audit["evaluation_history_sha256"] = canonical_sha256(
            audit["evaluation_history"]
        )
    receipt["receipt_sha256"] = canonical_sha256(
        {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    )


def _formal_model_keys() -> list[str]:
    return [
        f"{role}/seed-{seed}"
        for seed in (42, 43, 44)
        for role in ("M_LD", "M_drop")
    ]


def _lightweight_formal_plan(root: Path) -> tuple[Path, dict, Path, dict, dict]:
    """Create only the portable shell needed to exercise registry semantics."""

    keys = _formal_model_keys()
    slots = []
    for key in keys:
        role, seed_text = key.split("/seed-", maxsplit=1)
        slots.append(
            {
                "model_key": key,
                "role": role,
                "seed": int(seed_text),
                "training_required": True,
            }
        )
    plan = {
        "scope": "formal",
        "scientific_eligible": True,
        "ordered_model_slots": slots,
        "pilot_slot_keys": keys[:2],
    }
    plan_id = "tpl-" + canonical_sha256(
        {"schema_version": "stage1-lightweight-registry-plan/v1", "plan": plan}
    )
    target = root / "artifacts/training_plans" / plan_id
    target.mkdir(parents=True)
    write_canonical_json(target / "plan.resolved.json", plan)
    write_canonical_json(target / "payload_manifest.json", build_payload_manifest(target))
    ref = root / "refs/lightweight_training_plan_ref.json"
    write_locator_ref(
        ref,
        artifact_kind="training-plan",
        artifact_id=plan_id,
        target=target,
        payload_manifest_sha256=validate_payload_manifest(target),
    )
    locator, resolved_target = resolve_locator_ref(ref, "training-plan")
    dependency = portable_dependency(locator, resolved_target, root)
    return ref, locator, resolved_target, plan, dependency


def _lightweight_trained_model_refs(
    root: Path,
    *,
    plan_dependency: dict,
) -> tuple[Path, dict[str, Path], dict[Path, dict]]:
    """Create strict unique model locators backed by one tiny frozen source."""

    source = root / "sources/lightweight-formal-model"
    source.mkdir(parents=True)
    write_bytes_atomic(source / "model.safetensors", b"formal-model-weights")
    write_canonical_json(source / "tokenizer.json", {"version": "1.0"})
    inventory = inventory_regular_file_tree(
        source,
        workspace_root=root,
        label="lightweight formal model",
        inventory_policy="all-regular-files/v1",
    )
    tokenizer_content_revision = "tok-" + canonical_sha256(inventory)
    refs: dict[str, Path] = {}
    models_by_target: dict[Path, dict] = {}
    for key in _formal_model_keys():
        role, seed_text = key.split("/seed-", maxsplit=1)
        seed = int(seed_text)
        model_id = "mdl-" + canonical_sha256(
            {
                "schema_version": "stage1-lightweight-trained-model/v1",
                "model_key": key,
                "training_plan_dependency": plan_dependency,
            }
        )
        model = {
            "model_artifact_id": model_id,
            "artifact_type": "trained",
            "scope": "formal",
            "scientific_eligible": True,
            "model_key": key,
            "role": role,
            "seed": seed,
            "training_plan_dependency": copy.deepcopy(plan_dependency),
            "checkpoint_format": "full",
            "checkpoint_inventory": copy.deepcopy(inventory),
            "tokenizer_inventory": copy.deepcopy(inventory),
            "base_inventory": copy.deepcopy(inventory),
            "tokenizer_contract": {
                "tokenizer_revision": "qwen3-8b-stage1-v1",
                "tokenizer_content_revision": tokenizer_content_revision,
            },
        }
        target = root / "artifacts/models" / model_id
        target.mkdir(parents=True)
        write_canonical_json(target / "model.json", model)
        write_canonical_json(
            target / "payload_manifest.json", build_payload_manifest(target)
        )
        ref = root / "refs" / f"{key.replace('/', '_')}_lightweight_model_ref.json"
        write_locator_ref(
            ref,
            artifact_kind="stage1-model",
            artifact_id=model_id,
            target=target,
            payload_manifest_sha256=validate_payload_manifest(target),
        )
        refs[key] = ref
        models_by_target[target.resolve()] = model
    return source, refs, models_by_target


@contextmanager
def _lightweight_registry_validation_patches(
    *,
    root: Path,
    plan_ref: Path,
    plan_locator: dict,
    plan_target: Path,
    plan: dict,
    models_by_target: dict[Path, dict],
) -> Iterator[None]:
    """Patch only upstream deep-validation boundaries, with exact target maps."""

    resolved_root = root.resolve()
    resolved_plan_ref = plan_ref.resolve()
    resolved_plan_target = plan_target.resolve()

    def load_plan(
        supplied_ref: str | Path,
        *,
        workspace_root: str | Path,
        tokenizer=None,
    ) -> tuple[dict, Path, dict]:
        del tokenizer
        if Path(supplied_ref).resolve() != resolved_plan_ref:
            raise AssertionError("unexpected lightweight training-plan ref")
        if Path(workspace_root).resolve() != resolved_root:
            raise AssertionError("unexpected lightweight training-plan workspace")
        return (
            copy.deepcopy(plan_locator),
            resolved_plan_target,
            copy.deepcopy(plan),
        )

    def validate_plan(
        supplied_target: str | Path,
        *,
        workspace_root: str | Path,
        **_kwargs,
    ) -> dict:
        if Path(supplied_target).resolve() != resolved_plan_target:
            raise AssertionError("unexpected lightweight training-plan target")
        if Path(workspace_root).resolve() != resolved_root:
            raise AssertionError("unexpected lightweight training-plan workspace")
        return copy.deepcopy(plan)

    def validate_model(
        supplied_target: str | Path,
        *,
        workspace_root: str | Path,
        **_kwargs,
    ) -> dict:
        target = Path(supplied_target).resolve()
        if Path(workspace_root).resolve() != resolved_root:
            raise AssertionError("unexpected lightweight model workspace")
        if target not in models_by_target:
            raise AssertionError(f"unexpected lightweight model target: {target.name}")
        return copy.deepcopy(models_by_target[target])

    with patch(
        "data.training_plan.load_training_plan", side_effect=load_plan
    ), patch(
        "data.training_plan.validate_training_plan_target",
        side_effect=validate_plan,
    ), patch.object(
        registry_module, "_validate_model_target", side_effect=validate_model
    ):
        yield


class ModelInventoryTests(unittest.TestCase):
    def test_new_base_tokenizer_inventory_freezes_extra_python_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            environment_ref = _environment_ref(root)
            source = _model_tree(root)
            write_bytes_atomic(
                source / "tokenization_custom.py",
                b"class SyntheticTokenizer: pass\n",
            )
            model_ref = root / "refs/base_ref.json"
            register_base_model(
                model_dir=source,
                tokenizer_dir=None,
                model_name="synthetic",
                tokenizer_revision="synthetic-tokenizer/v1",
                environment_ref=environment_ref,
                write_ref=model_ref,
                workspace_root=root,
                target_root=root / "artifacts/models",
            )
            _, target = resolve_locator_ref(model_ref, "stage1-model")
            model = load_json(target / "model.json")
            tokenizer_inventory = model["tokenizer_inventory"]
            self.assertEqual(
                tokenizer_inventory["inventory_policy"],
                "all-regular-files/v1",
            )
            self.assertIn(
                "tokenization_custom.py",
                {row["path"] for row in tokenizer_inventory["files"]},
            )

    def test_receipt_checkpoint_binding_rejects_changed_bytes_and_same_name_copy(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            selected = root / "run-a/checkpoint-10"
            selected.mkdir(parents=True)
            write_bytes_atomic(selected / "model.safetensors", b"selected")
            write_canonical_json(selected / "config.json", {"model_type": "fixture"})
            expected = inventory_regular_file_tree(
                selected,
                workspace_root=root,
                label="selected checkpoint",
            )
            receipt = {
                "selected_checkpoint_global_step": 10,
                "selected_checkpoint_inventory": expected,
            }
            self.assertTrue(
                _assert_receipt_checkpoint_binding(
                    receipt, expected, required=True
                )
            )

            same_name = root / "run-b/checkpoint-10"
            same_name.mkdir(parents=True)
            write_bytes_atomic(same_name / "model.safetensors", b"replacement")
            write_canonical_json(
                same_name / "config.json", {"model_type": "fixture"}
            )
            replacement = inventory_regular_file_tree(
                same_name,
                workspace_root=root,
                label="replacement checkpoint",
            )
            with self.assertRaisesRegex(
                ModelRegistryError, "differs from the selected checkpoint receipt"
            ):
                _assert_receipt_checkpoint_binding(
                    receipt, replacement, required=True
                )

            write_bytes_atomic(selected / "model.safetensors", b"changed-in-place")
            changed = inventory_regular_file_tree(
                selected,
                workspace_root=root,
                label="changed checkpoint",
            )
            with self.assertRaisesRegex(
                ModelRegistryError, "differs from the selected checkpoint receipt"
            ):
                _assert_receipt_checkpoint_binding(receipt, changed, required=True)

    def test_base_registration_rejects_missing_shard_and_symlink(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            environment_ref = _environment_ref(root)
            source = _model_tree(root, sharded=True)
            (source / "model-00002-of-00002.safetensors").unlink()
            with self.assertRaisesRegex(ModelRegistryError, "incomplete|missing"):
                register_base_model(
                    model_dir=source,
                    tokenizer_dir=None,
                    model_name="synthetic",
                    tokenizer_revision="synthetic-tokenizer/v1",
                    environment_ref=environment_ref,
                    write_ref=root / "refs/base_ref.json",
                    workspace_root=root,
                    target_root=root / "artifacts/models",
                )
            write_bytes_atomic(source / "model-00002-of-00002.safetensors", b"shard-two")
            link = source / "unsafe-link"
            try:
                link.symlink_to(source / "config.json")
            except (OSError, NotImplementedError):  # pragma: no cover
                self.skipTest("symlinks are unavailable")
            with self.assertRaisesRegex(ModelRegistryError, "symlink"):
                register_base_model(
                    model_dir=source,
                    tokenizer_dir=None,
                    model_name="synthetic",
                    tokenizer_revision="synthetic-tokenizer/v1",
                    environment_ref=environment_ref,
                    write_ref=root / "refs/base_ref.json",
                    workspace_root=root,
                    target_root=root / "artifacts/models",
                )


class FormalResolverLeaseUnitTests(unittest.TestCase):
    def test_formal_resolver_mints_only_after_fresh_source_lease(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "sources/base-model"
            source.mkdir(parents=True)
            write_bytes_atomic(source / "model.safetensors", b"formal-base-weights")
            write_canonical_json(source / "tokenizer.json", {"version": "1.0"})
            inventory = inventory_regular_file_tree(
                source,
                workspace_root=root,
                inventory_policy="all-regular-files/v1",
            )
            registry = {
                "model_registry_id": "mreg-" + "a" * 64,
                "scientific_eligible": True,
                "entries": [
                    {
                        "model_key": "M_LD/seed-42",
                        "role": "M_LD",
                        "seed": 42,
                        "model_dependency": {},
                    }
                ],
            }
            model = {
                "checkpoint_format": "full",
                "checkpoint_inventory": inventory,
                "tokenizer_inventory": inventory,
                "base_inventory": inventory,
                "tokenizer_contract": {
                    "tokenizer_revision": "qwen3-8b-stage1-v1",
                    "tokenizer_content_revision": "tok-" + "b" * 64,
                },
                "model_artifact_id": "mdl-" + "c" * 64,
            }
            with patch.object(
                registry_module, "_validate_registry_target", return_value=registry
            ), patch.object(
                registry_module, "resolve_dependency_target", return_value=root
            ), patch.object(
                registry_module, "_validate_model_target", return_value=model
            ):
                resolved = registry_module._resolve_registered_model_target(
                    target=root,
                    model_key="M_LD/seed-42",
                    workspace_root=root,
                )
            self.assertTrue(resolved.scientific_eligible)
            self.assertEqual(resolved.checkpoint_path, source.resolve())
            self.assertIsInstance(
                resolved.source_contract, registry_module.ResolvedModelSourceContract
            )

    def test_exact_source_aliases_share_one_fresh_hash_per_lease_phase(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "sources/shared-model"
            source.mkdir(parents=True)
            write_bytes_atomic(source / "model.safetensors", b"shared-weights")
            write_canonical_json(source / "tokenizer.json", {"version": "1.0"})
            inventory = inventory_regular_file_tree(source, workspace_root=root)
            contract = registry_module.ResolvedModelSourceContract(
                workspace_root=root,
                checkpoint_inventory=copy.deepcopy(inventory),
                tokenizer_inventory=copy.deepcopy(inventory),
                base_inventory=copy.deepcopy(inventory),
            )

            real_rehash = registry_module._resolve_and_rehash_tree
            real_capture = registry_module._capture_source_lease_signature
            real_open = registry_module._open_source_lease_descriptors
            with patch.object(
                registry_module,
                "_resolve_and_rehash_tree",
                wraps=real_rehash,
            ) as rehash, patch.object(
                registry_module,
                "_capture_source_lease_signature",
                wraps=real_capture,
            ) as capture, patch.object(
                registry_module,
                "_open_source_lease_descriptors",
                wraps=real_open,
            ) as open_descriptors:
                with registry_module.verified_model_source_lease(contract) as paths:
                    self.assertEqual(paths.checkpoint_path, source.resolve())
                    self.assertEqual(paths.tokenizer_path, source.resolve())
                    self.assertEqual(paths.base_model_path, source.resolve())

            # One pre-load and one post-load fresh inventory, not one per role.
            self.assertEqual(rehash.call_count, 2)
            # Initial/pre-open/post-open plus pre/post-rehash captures, once per
            # distinct inventory at each safety boundary.
            self.assertEqual(capture.call_count, 5)
            self.assertEqual(open_descriptors.call_count, 1)
            self.assertEqual(
                set(open_descriptors.call_args.args[0]),
                {"checkpoint"},
            )

    def test_same_path_different_snapshots_do_not_share_lease_work(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "sources/overlapping-model"
            source.mkdir(parents=True)
            write_bytes_atomic(source / "model.safetensors", b"shared-weights")
            write_canonical_json(source / "tokenizer.json", {"version": "1.0"})
            complete_inventory = inventory_regular_file_tree(
                source,
                workspace_root=root,
                inventory_policy="all-regular-files/v1",
            )
            tokenizer_inventory = inventory_regular_file_tree(
                source,
                workspace_root=root,
                inventory_policy="tokenizer-files/v1",
            )
            self.assertEqual(
                complete_inventory["logical_repo_path"],
                tokenizer_inventory["logical_repo_path"],
            )
            self.assertNotEqual(complete_inventory, tokenizer_inventory)
            contract = registry_module.ResolvedModelSourceContract(
                workspace_root=root,
                checkpoint_inventory=complete_inventory,
                tokenizer_inventory=tokenizer_inventory,
                base_inventory=copy.deepcopy(complete_inventory),
            )

            real_rehash = registry_module._resolve_and_rehash_tree
            rehashed_policies: list[str] = []

            def counted_rehash(snapshot, workspace_root, rehash_cache=None):
                rehashed_policies.append(str(snapshot["inventory_policy"]))
                return real_rehash(snapshot, workspace_root, rehash_cache)

            real_open = registry_module._open_source_lease_descriptors
            with patch.object(
                registry_module,
                "_resolve_and_rehash_tree",
                side_effect=counted_rehash,
            ), patch.object(
                registry_module,
                "_open_source_lease_descriptors",
                wraps=real_open,
            ) as open_descriptors:
                with registry_module.verified_model_source_lease(contract) as paths:
                    self.assertEqual(paths.checkpoint_path, source.resolve())
                    self.assertEqual(paths.tokenizer_path, source.resolve())
                    self.assertEqual(paths.base_model_path, source.resolve())

            self.assertEqual(
                rehashed_policies.count("all-regular-files/v1"),
                2,
            )
            self.assertEqual(
                rehashed_policies.count("tokenizer-files/v1"),
                2,
            )
            self.assertEqual(len(rehashed_policies), 4)
            # Distinct snapshots receive distinct descriptor/lock sets even
            # though they deliberately resolve to the same lexical root.
            self.assertEqual(open_descriptors.call_count, 2)
            self.assertEqual(
                [set(call.args[0]) for call in open_descriptors.call_args_list],
                [{"checkpoint"}, {"tokenizer"}],
            )

    def test_public_locator_and_dependency_resolution_reject_checkpoint_tamper(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (
                plan_ref,
                plan_locator,
                plan_target,
                plan,
                plan_dependency,
            ) = _lightweight_formal_plan(root)
            source, refs, models_by_target = _lightweight_trained_model_refs(
                root,
                plan_dependency=plan_dependency,
            )
            registry_ref = root / "refs/lightweight_formal_registry_ref.json"
            with _lightweight_registry_validation_patches(
                root=root,
                plan_ref=plan_ref,
                plan_locator=plan_locator,
                plan_target=plan_target,
                plan=plan,
                models_by_target=models_by_target,
            ):
                finalize_model_registry(
                    scope="formal",
                    training_plan_ref=plan_ref,
                    model_bindings=[
                        (key, refs[key]) for key in _formal_model_keys()
                    ],
                    write_ref=registry_ref,
                    workspace_root=root,
                    target_root=root / "artifacts/model_registries",
                )
                resolved = resolve_registered_model(
                    registry_ref=registry_ref,
                    model_key="M_LD/seed-42",
                    workspace_root=root,
                )
                self.assertEqual(resolved.checkpoint_path, source.resolve())
                self.assertEqual(
                    resolved.tokenizer_revision, "qwen3-8b-stage1-v1"
                )
                registry_dependency = _dependency(registry_ref, root)
                replay = resolve_registered_model_dependency(
                    registry_dependency=registry_dependency,
                    model_key="M_drop/seed-44",
                    workspace_root=root,
                )
                self.assertEqual(replay.model_key, "M_drop/seed-44")

                write_bytes_atomic(
                    source / "model.safetensors", b"tampered-formal-model"
                )
                with self.assertRaisesRegex(
                    ModelRegistryError, "inventory changed|source changed"
                ):
                    resolve_registered_model(
                        registry_ref=registry_ref,
                        model_key="M_LD/seed-42",
                        workspace_root=root,
                    )


@patch(
    "data.build_context_manifest._default_formal_score_replayer",
    new=deterministic_formal_score_replayer,
)
@patch(
    "data.build_context_manifest._construct_formal_tokenizer",
    new=deterministic_context_tokenizer_constructor,
)
class LegacyEngineeringRegistryTests(unittest.TestCase):
    def _legacy_ref(self, root: Path) -> tuple[Path, Path, Path, str]:
        captured_environment_ref = _environment_ref(root)
        environment_ref = root / "refs/frozen_environment_ref.json"
        write_canonical_json(
            environment_ref,
            json.loads(captured_environment_ref.read_text(encoding="utf-8")),
        )
        source = _legacy_model_tree(root)
        revision = tokenizer_revision_from_directory(source)
        model_ref = root / "refs/legacy_model_ref.json"
        register_legacy_model(
            checkpoint=source,
            composition="auto",
            tokenizer_root="same",
            chat_template_source="chat-template-jinja",
            environment_ref=environment_ref,
            write_ref=model_ref,
            workspace_root=root,
            target_root=root / "artifacts/models",
        )
        return model_ref, environment_ref, source, revision

    def _engineering_plan(
        self,
        root: Path,
        *,
        legacy_ref: Path | None,
        environment_ref: Path,
        tokenizer_revision: str,
    ) -> tuple[SyntheticTrainingWorkspace, Path]:
        workspace = SyntheticTrainingWorkspace(root)
        _, context_target = resolve_locator_ref(workspace.context_ref, "context")
        context_config = json.loads(
            (context_target / "config.resolved.json").read_text(encoding="utf-8")
        )
        context_config["budget"]["tokenizer_revision"] = tokenizer_revision
        write_canonical_json(context_target / "config.resolved.json", context_config)
        context_meta_path = next(context_target.glob("context_manifest.*.meta.json"))
        context_meta = json.loads(context_meta_path.read_text(encoding="utf-8"))
        context_meta["budget"]["tokenizer_revision"] = tokenizer_revision
        write_canonical_json(context_meta_path, context_meta)
        workspace.refresh_manifest_and_locator(workspace.context_ref)
        source_spec = json.loads(workspace.source_spec.read_text(encoding="utf-8"))
        source_spec.update(
            {
                "scope": "engineering-smoke",
                "scientific_eligible": False,
                "ordered_model_slots": [
                    {
                        "model_key": "M_legacy/smoke",
                        "role": "legacy-smoke-only",
                        "seed": None,
                        "training_required": False,
                        "train_config": None,
                        "epochs": None,
                        "final_checkpoint_rule": None,
                    }
                ],
                "pilot_slot_keys": [],
            }
        )
        source_spec.pop("epoch_and_checkpoint_selection_policy", None)
        source_spec.pop("order_dropout_rng_policy", None)
        write_canonical_json(workspace.source_spec, source_spec)
        runtime_code = root / "src/finetune/stage1_runtime.py"
        runtime_code.parent.mkdir(parents=True, exist_ok=True)
        runtime_code.write_text("# synthetic frozen runtime code\n", encoding="utf-8")
        plan_ref = workspace.refs / "engineering_plan_ref.json"
        freeze_training_plan(
            source_spec_path=workspace.source_spec,
            scope="engineering-smoke",
            context_ref=workspace.context_ref,
            training_evidence_ref=None,
            base_model_ref=None,
            environment_ref=environment_ref,
            non_training_model_bindings=(
                [] if legacy_ref is None else [("M_legacy/smoke", legacy_ref)]
            ),
            write_ref=plan_ref,
            workspace_root=root,
            target_root=root / "artifacts/training_plans",
            decision_register_path=root / "config/stage1/decision_register.json",
            train_code_path=workspace.train_code,
            runtime_code_path=runtime_code,
        )
        return workspace, plan_ref

    def test_legacy_registration_is_truthful_full_tree_and_non_scientific(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model_ref, environment_ref, source, revision = self._legacy_ref(root)
            report = validate_model_artifact(model_ref, workspace_root=root)
            self.assertEqual(report["artifact_type"], "legacy-smoke-only")
            self.assertEqual(report["scope"], "engineering-smoke")
            self.assertFalse(report["scientific_eligible"])
            self.assertEqual(report["tokenizer_revision"], revision)
            _, target = resolve_locator_ref(model_ref, "stage1-model")
            model = json.loads((target / "model.json").read_text(encoding="utf-8"))
            self.assertEqual(model["model_key"], "M_legacy/smoke")
            self.assertEqual(model["role"], "legacy-smoke-only")
            self.assertIsNone(model["seed"])
            self.assertEqual(
                model["legacy_source_tree_sha256"],
                model["checkpoint_inventory"]["file_tree_sha256"],
            )
            self.assertEqual(
                model["checkpoint_inventory"], model["tokenizer_inventory"]
            )
            self.assertEqual(
                model["tokenizer_contract"]["tokenizer_revision"],
                model["tokenizer_contract"]["tokenizer_content_revision"],
            )
            self.assertEqual(
                model["tokenizer_contract"]["chat_template_source"],
                "chat_template.jinja",
            )
            self.assertFalse(
                model["tokenizer_contract"]["supports_enable_thinking_false"]
            )
            self.assertEqual(model["environment_dependency"], _dependency(environment_ref, root))
            self.assertEqual(
                {path.name for path in target.iterdir()},
                {"model.json", "environment_ref.json", "payload_manifest.json"},
            )
            self.assertTrue(
                any(
                    row["path"] == "global_step10/optimizer-state.bin"
                    for row in model["checkpoint_inventory"]["files"]
                )
            )

            with self.assertRaisesRegex(ModelRegistryError, "tokenizer_config.json"):
                register_legacy_model(
                    checkpoint=source,
                    composition="auto",
                    tokenizer_root="same",
                    chat_template_source="tokenizer-config",
                    environment_ref=environment_ref,
                    write_ref=root / "refs/bad_legacy_ref.json",
                    workspace_root=root,
                    target_root=root / "artifacts/models",
                )

    def test_engineering_plan_binds_legacy_without_fake_training_lineage(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            legacy_ref, environment_ref, source, revision = self._legacy_ref(root)
            workspace, plan_ref = self._engineering_plan(
                root,
                legacy_ref=legacy_ref,
                environment_ref=environment_ref,
                tokenizer_revision=revision,
            )
            _, plan_target, plan = load_training_plan(plan_ref, workspace_root=root)
            self.assertIsNone(plan["training_evidence_dependency"])
            self.assertIsNone(plan["base_model_dependency"])
            self.assertEqual(
                plan["non_training_model_dependencies"],
                {"M_legacy/smoke": _dependency(legacy_ref, root)},
            )
            self.assertIsNone(
                json.loads(
                    (plan_target / "training_evidence_ref.json").read_text(
                        encoding="utf-8"
                    )
                )
            )
            registry_ref = workspace.refs / "engineering_registry_ref.json"
            finalize_model_registry(
                scope="engineering-smoke",
                training_plan_ref=plan_ref,
                model_bindings=[("M_legacy/smoke", legacy_ref)],
                write_ref=registry_ref,
                workspace_root=root,
                target_root=root / "artifacts/model_registries",
            )
            report = validate_model_registry(registry_ref, workspace_root=root)
            self.assertFalse(report["scientific_eligible"])
            resolved = resolve_registered_model(
                registry_ref=registry_ref,
                model_key="M_legacy/smoke",
                workspace_root=root,
            )
            self.assertEqual(resolved.role, "legacy-smoke-only")
            self.assertEqual(resolved.tokenizer_revision, revision)
            self.assertEqual(resolved.checkpoint_path, source.resolve())

            original_validate = registry_module._validate_registry_target
            shard = source / "model.safetensors"
            original_shard = shard.read_bytes()

            def mutate_after_deep_validation(*args, **kwargs):
                validated = original_validate(*args, **kwargs)
                write_bytes_atomic(shard, b"changed-after-validator")
                return validated

            with patch.object(
                registry_module,
                "_validate_registry_target",
                side_effect=mutate_after_deep_validation,
            ), self.assertRaisesRegex(
                ModelRegistryError, "inventory changed|source changed"
            ):
                resolve_registered_model(
                    registry_ref=registry_ref,
                    model_key="M_legacy/smoke",
                    workspace_root=root,
                )
            write_bytes_atomic(shard, original_shard)

            write_bytes_atomic(
                source / "global_step10/optimizer-state.bin", b"tampered"
            )
            with self.assertRaisesRegex(ModelRegistryError, "inventory changed"):
                resolve_registered_model(
                    registry_ref=registry_ref,
                    model_key="M_legacy/smoke",
                    workspace_root=root,
                )

    def test_engineering_plan_rejects_missing_wrong_or_qwen3_context_binding(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            legacy_ref, environment_ref, _, revision = self._legacy_ref(root)
            with self.assertRaisesRegex(TrainingPlanError, "binding set mismatch"):
                self._engineering_plan(
                    root,
                    legacy_ref=None,
                    environment_ref=environment_ref,
                    tokenizer_revision=revision,
                )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            legacy_ref, environment_ref, _, _ = self._legacy_ref(root)
            with self.assertRaisesRegex(TrainingPlanError, "tokenizer revision"):
                self._engineering_plan(
                    root,
                    legacy_ref=legacy_ref,
                    environment_ref=environment_ref,
                    tokenizer_revision="qwen3-8b-stage1-v1",
                )

    def test_cli_contract_exposes_legacy_and_plan_binding_aliases(self):
        model_args = model_cli.build_parser().parse_args(
            [
                "register-legacy",
                "--checkpoint",
                "checkpoint",
                "--composition",
                "auto",
                "--tokenizer-root",
                "same",
                "--chat-template-source",
                "chat-template-jinja",
                "--environment-ref",
                "environment.json",
                "--write-ref",
                "legacy.json",
            ]
        )
        self.assertEqual(model_args.command, "register-legacy")
        registry_args = model_cli.build_parser().parse_args(
            [
                "finalize-registry",
                "--scope",
                "engineering-smoke",
                "--training-plan-ref",
                "plan.json",
                "--bind",
                "M_legacy/smoke=legacy.json",
                "--write-ref",
                "registry.json",
            ]
        )
        self.assertEqual(registry_args.binding[0][0], "M_legacy/smoke")
        plan_args = plan_cli.build_parser().parse_args(
            [
                "freeze",
                "--source-spec",
                "smoke.json",
                "--scope",
                "engineering-smoke",
                "--context-ref",
                "context.json",
                "--environment-ref",
                "environment.json",
                "--bind-non-training-model",
                "M_legacy/smoke=legacy.json",
                "--write-ref",
                "plan.json",
            ]
        )
        self.assertIsNone(plan_args.training_evidence_ref)
        self.assertIsNone(plan_args.base_model_ref)
        self.assertEqual(
            plan_args.bind_non_training_model[0][0], "M_legacy/smoke"
        )


@patch(
    "data.build_context_manifest._default_formal_score_replayer",
    new=deterministic_formal_score_replayer,
)
@patch(
    "data.build_context_manifest._construct_formal_tokenizer",
    new=deterministic_context_tokenizer_constructor,
)
class FormalModelRegistryTests(unittest.TestCase):
    def _formal_workspace(self, root: Path):
        workspace = SyntheticTrainingWorkspace(root)
        workspace.build_evidence()
        for filename in ("train_m_ld.json", "train_m_drop.json"):
            config_path = root / "config/stage1" / filename
            config = json.loads(config_path.read_text(encoding="utf-8"))
            config["early_stopping"] = copy.deepcopy(EARLY_STOPPING_POLICY)
            write_canonical_json(config_path, config)
        runtime_code = root / "src/finetune/stage1_runtime.py"
        runtime_code.write_text("# synthetic frozen runtime code\n", encoding="utf-8")
        freeze_training_plan(
            source_spec_path=workspace.source_spec,
            scope="formal",
            context_ref=workspace.context_ref,
            training_evidence_ref=workspace.refs / "training_evidence_ref.json",
            train_partition_ref=workspace.partition_ref,
            base_model_ref=workspace.base_ref,
            environment_ref=workspace.environment_ref,
            write_ref=workspace.refs / "training_plan_ref.json",
            workspace_root=root,
            decision_register_path=root / "config/stage1/decision_register.json",
            train_code_path=workspace.train_code,
            runtime_code_path=runtime_code,
            schema_path=REPOSITORY_ROOT / "schemas/stage1_training_plan_v1.schema.json",
            tokenizer=workspace.tokenizer,
        )
        build_training_schedule(
            training_plan_ref=workspace.refs / "training_plan_ref.json",
            training_evidence_ref=workspace.refs / "training_evidence_ref.json",
            train_partition_ref=workspace.partition_ref,
            tokenizer=workspace.tokenizer,
            write_ref=workspace.refs / "schedule_ref.json",
            workspace_root=root,
        )
        checkpoint = _model_tree(root, "checkpoints/checkpoint-10")
        return workspace, checkpoint

    def _register_receipt_document(
        self,
        workspace: SyntheticTrainingWorkspace,
        receipt: dict,
        stem: str,
    ) -> Path:
        receipt_path = workspace.root / "receipts" / f"{stem}.json"
        write_canonical_json(receipt_path, receipt)
        receipt_ref = workspace.refs / f"{stem}_receipt_ref.json"
        register_training_receipt(
            receipt_json=receipt_path,
            training_plan_ref=workspace.refs / "training_plan_ref.json",
            schedule_ref=workspace.refs / "schedule_ref.json",
            base_model_ref=workspace.base_ref,
            environment_ref=workspace.environment_ref,
            write_ref=receipt_ref,
            workspace_root=workspace.root,
            target_root=workspace.root / "artifacts/training_receipts",
            tokenizer=workspace.tokenizer,
        )
        return receipt_ref

    def _register_slot(
        self, workspace: SyntheticTrainingWorkspace, checkpoint: Path, model_key: str
    ) -> Path:
        safe = model_key.replace("/", "_")
        receipt_ref = self._register_receipt_document(
            workspace,
            _raw_receipt(workspace, model_key, checkpoint),
            safe,
        )
        model_ref = workspace.refs / f"{safe}_model_ref.json"
        register_trained_model(
            checkpoint_dir=checkpoint,
            tokenizer_dir=None,
            checkpoint_format="full",
            model_key=model_key,
            training_plan_ref=workspace.refs / "training_plan_ref.json",
            schedule_ref=workspace.refs / "schedule_ref.json",
            training_receipt_ref=receipt_ref,
            base_model_ref=workspace.base_ref,
            environment_ref=workspace.environment_ref,
            write_ref=model_ref,
            workspace_root=workspace.root,
            target_root=workspace.root / "artifacts/models",
            tokenizer=workspace.tokenizer,
        )
        return model_ref

    def test_raw_receipt_independently_replays_checkpoint_selection(self):
        original = _standalone_raw_receipt()
        _validate_raw_receipt(original)

        history_digest = copy.deepcopy(original)
        history_digest["checkpoint_selection_audit"]["evaluation_history"][1][
            "metric_value"
        ] = 1.5
        _rehash_receipt(history_digest)

        qualifying = copy.deepcopy(original)
        qualifying["checkpoint_selection_audit"]["evaluation_history"][1][
            "qualifying_improvement"
        ] = True
        _rehash_receipt(qualifying, history=True)

        selected = copy.deepcopy(original)
        selected["checkpoint_selection_audit"]["selected_global_step"] = 20
        _rehash_receipt(selected)

        exit_step = copy.deepcopy(original)
        exit_step["training_exit_global_step"] = 30
        _rehash_receipt(exit_step)

        cases = (
            ("history-digest", history_digest, "history digest"),
            ("qualifying", qualifying, "improvement audit"),
            ("selected", selected, "earliest winner"),
            ("exit", exit_step, "exit step"),
        )
        for name, receipt, message in cases:
            with self.subTest(name=name):
                with self.assertRaisesRegex(ModelRegistryError, message):
                    _validate_raw_receipt(receipt)

    def test_raw_receipt_rejects_non_finite_metrics_and_wrong_patience(self):
        original = _standalone_raw_receipt()

        non_finite_metric = copy.deepcopy(original)
        non_finite_metric["checkpoint_selection_audit"]["evaluation_history"][1][
            "metric_value"
        ] = float("nan")
        non_finite_selected = copy.deepcopy(original)
        non_finite_selected["checkpoint_selection_audit"][
            "selected_metric_value"
        ] = float("inf")
        non_finite_threshold = copy.deepcopy(original)
        non_finite_threshold["checkpoint_selection_audit"]["policy"][
            "threshold"
        ] = float("nan")
        for name, receipt in (
            ("history", non_finite_metric),
            ("selected", non_finite_selected),
            ("threshold", non_finite_threshold),
        ):
            with self.subTest(name=name):
                with self.assertRaisesRegex(ModelRegistryError, "finite"):
                    _validate_raw_receipt(receipt)

        wrong_patience = copy.deepcopy(original)
        wrong_patience["checkpoint_selection_audit"]["evaluation_history"][2][
            "patience_counter_after"
        ] = 99
        _rehash_receipt(wrong_patience, history=True)
        with self.assertRaisesRegex(ModelRegistryError, "patience audit"):
            _validate_raw_receipt(wrong_patience)

        wrong_stop = copy.deepcopy(original)
        wrong_stop["checkpoint_selection_audit"]["stop_reason"] = "maximum-epochs"
        _rehash_receipt(wrong_stop)
        with self.assertRaisesRegex(ModelRegistryError, "stop reason"):
            _validate_raw_receipt(wrong_stop)

    def test_receipt_policy_and_runtime_hash_must_match_frozen_plan(self):
        original = _standalone_raw_receipt()
        slot = {
            "model_key": original["model_key"],
            "role": original["role"],
            "seed": original["seed"],
            "training_required": True,
            "epochs": original["planned_epochs"],
            "final_checkpoint_rule": original["final_checkpoint_rule"],
            "train_config_sha256": original["train_config_sha256"],
            "train_config_resolved": {
                "early_stopping": copy.deepcopy(EARLY_STOPPING_POLICY)
            },
        }
        plan = {
            "ordered_model_slots": [slot],
            "train_code_sha256": original["train_code_sha256"],
            "runtime_code_sha256": original["runtime_code_sha256"],
            "training_evidence_dependency": original[
                "training_evidence_dependency"
            ],
            "train_partition_dependency": original[
                "train_partition_dependency"
            ],
        }
        schedule_meta = {
            "training_plan_dependency": original["training_plan_dependency"],
            "training_evidence_dependency": original[
                "training_evidence_dependency"
            ],
            "train_partition_dependency": original[
                "train_partition_dependency"
            ],
            "slot_registry": [{"model_key": original["model_key"]}],
        }

        def validate_cross_lineage(receipt: dict) -> None:
            _validate_receipt_cross_lineage(
                receipt,
                plan=plan,
                plan_dependency=original["training_plan_dependency"],
                schedule_meta=schedule_meta,
                schedule_dependency=original["schedule_dependency"],
                base_dependency=original["base_model_dependency"],
                environment_dependency=original["environment_dependency"],
            )

        validate_cross_lineage(original)
        plan["scope"] = "pilot"
        validate_cross_lineage(original)
        plan["scope"] = "formal"
        with self.assertRaisesRegex(
            ModelRegistryError, "lacks an exact selected checkpoint inventory"
        ):
            validate_cross_lineage(original)
        plan.pop("scope")
        wrong_runtime = copy.deepcopy(original)
        wrong_runtime["runtime_code_sha256"] = "0" * 64
        _rehash_receipt(wrong_runtime)
        with self.assertRaisesRegex(ModelRegistryError, "runtime code"):
            validate_cross_lineage(wrong_runtime)

        wrong_policy = copy.deepcopy(original)
        wrong_policy["checkpoint_selection_audit"]["policy"]["threshold"] = 0.002
        _rehash_receipt(wrong_policy)
        with self.assertRaisesRegex(ModelRegistryError, "early-stopping policy"):
            validate_cross_lineage(wrong_policy)

    def test_formal_trained_model_tokenizer_inventory_and_projection_rejection(self):
        with tempfile.TemporaryDirectory() as temporary:
            workspace, checkpoint = self._formal_workspace(Path(temporary))
            write_bytes_atomic(
                checkpoint / "tokenization_custom.py",
                b"class SyntheticTokenizer: pass\n",
            )
            model_ref = self._register_slot(
                workspace, checkpoint, "M_LD/seed-42"
            )
            _, first_target = resolve_locator_ref(
                model_ref, "stage1-model"
            )
            first_model = load_json(first_target / "model.json")
            tokenizer_inventory = first_model["tokenizer_inventory"]
            self.assertEqual(
                tokenizer_inventory["inventory_policy"],
                "all-regular-files/v1",
            )
            self.assertIn(
                "tokenization_custom.py",
                {row["path"] for row in tokenizer_inventory["files"]},
            )

            projected = copy.deepcopy(first_model)
            projected["tokenizer_inventory"] = inventory_regular_file_tree(
                checkpoint,
                workspace_root=workspace.root,
                label="projected tokenizer",
                inventory_policy="tokenizer-files/v1",
            )
            projected["id_inputs"] = registry_module._model_id_inputs(projected)
            projected["model_artifact_id"] = "mdl-" + canonical_sha256(
                projected["id_inputs"]
            )
            with self.assertRaisesRegex(
                ModelRegistryError, "scientific trained model tokenizer inventory"
            ):
                registry_module._validate_model_document(
                    projected,
                    workspace_root=workspace.root,
                    rehash_sources=False,
                )

    def test_formal_exact_six_slot_registry_finalizes_and_validates(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (
                plan_ref,
                plan_locator,
                plan_target,
                plan,
                plan_dependency,
            ) = _lightweight_formal_plan(root)
            _source, refs, models_by_target = _lightweight_trained_model_refs(
                root,
                plan_dependency=plan_dependency,
            )
            keys = _formal_model_keys()
            registry_ref = root / "refs/lightweight_formal_registry_ref.json"
            with _lightweight_registry_validation_patches(
                root=root,
                plan_ref=plan_ref,
                plan_locator=plan_locator,
                plan_target=plan_target,
                plan=plan,
                models_by_target=models_by_target,
            ):
                finalize_model_registry(
                    scope="formal",
                    training_plan_ref=plan_ref,
                    model_bindings=[(key, refs[key]) for key in keys],
                    write_ref=registry_ref,
                    workspace_root=root,
                    target_root=root / "artifacts/model_registries",
                )
                report = validate_model_registry(
                    registry_ref,
                    workspace_root=root,
                )
            self.assertTrue(report["scientific_eligible"])
            self.assertEqual(report["ordered_model_keys"], keys)
            _, registry_target = resolve_locator_ref(
                registry_ref, "stage1-model-registry"
            )
            registry = load_json(registry_target / "registry.json")
            self.assertEqual(
                registry["training_plan_dependency"], plan_dependency
            )
            self.assertEqual(
                load_json(registry_target / "training_plan_ref.json"),
                plan_dependency,
            )

    def test_pilot_registry_preserves_order_and_is_fail_closed_ineligible(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (
                plan_ref,
                plan_locator,
                plan_target,
                plan,
                plan_dependency,
            ) = _lightweight_formal_plan(root)
            _source, refs, models_by_target = _lightweight_trained_model_refs(
                root,
                plan_dependency=plan_dependency,
            )
            keys = _formal_model_keys()
            pilot_ref = root / "refs/lightweight_pilot_registry_ref.json"
            with _lightweight_registry_validation_patches(
                root=root,
                plan_ref=plan_ref,
                plan_locator=plan_locator,
                plan_target=plan_target,
                plan=plan,
                models_by_target=models_by_target,
            ):
                finalize_model_registry(
                    scope="pilot",
                    training_plan_ref=plan_ref,
                    model_bindings=[(key, refs[key]) for key in keys[:2]],
                    write_ref=pilot_ref,
                    workspace_root=root,
                    target_root=root / "artifacts/model_registries",
                )
                pilot_report = validate_model_registry(
                    pilot_ref,
                    workspace_root=root,
                )
            self.assertEqual(pilot_report["ordered_model_keys"], keys[:2])
            self.assertEqual(pilot_report["scope"], "pilot")
            self.assertFalse(pilot_report["scientific_eligible"])

    def test_registry_rejects_missing_extra_duplicate_and_legacy_slots(self):
        with tempfile.TemporaryDirectory() as temporary:
            workspace, checkpoint = self._formal_workspace(Path(temporary))
            keys = [
                f"{role}/seed-{seed}"
                for seed in (42, 43, 44)
                for role in ("M_LD", "M_drop")
            ]
            first_ref = self._register_slot(workspace, checkpoint, keys[0])
            with self.assertRaisesRegex(ModelRegistryError, "slot set mismatch"):
                finalize_model_registry(
                    scope="formal",
                    training_plan_ref=workspace.refs / "training_plan_ref.json",
                    model_bindings=[(keys[0], first_ref)],
                    write_ref=workspace.refs / "bad_registry_ref.json",
                    workspace_root=workspace.root,
                    tokenizer=workspace.tokenizer,
                )
            with self.assertRaisesRegex(ModelRegistryError, "duplicate model-key"):
                finalize_model_registry(
                    scope="formal",
                    training_plan_ref=workspace.refs / "training_plan_ref.json",
                    model_bindings=[(keys[0], first_ref), (keys[0], first_ref)],
                    write_ref=workspace.refs / "bad_registry_ref.json",
                    workspace_root=workspace.root,
                    tokenizer=workspace.tokenizer,
                )
            with self.assertRaisesRegex(ModelRegistryError, "slot set mismatch"):
                finalize_model_registry(
                    scope="formal",
                    training_plan_ref=workspace.refs / "training_plan_ref.json",
                    model_bindings=[(keys[0], first_ref), ("foreign/seed-9", first_ref)],
                    write_ref=workspace.refs / "bad_registry_ref.json",
                    workspace_root=workspace.root,
                    tokenizer=workspace.tokenizer,
                )
            # A legacy/generic model ref cannot enter formal registration at all.
            legacy_receipt = workspace.root / "receipts/legacy.json"
            write_canonical_json(
                legacy_receipt,
                _raw_receipt(workspace, "M_LD/seed-42", checkpoint),
            )
            with self.assertRaises(TrainingArtifactError):
                register_training_receipt(
                    receipt_json=legacy_receipt,
                    training_plan_ref=workspace.refs / "training_plan_ref.json",
                    schedule_ref=workspace.refs / "schedule_ref.json",
                    base_model_ref=workspace.root / "refs/base-model_ref.json",
                    environment_ref=workspace.environment_ref,
                    write_ref=workspace.refs / "legacy_receipt_ref.json",
                    workspace_root=workspace.root,
                    tokenizer=workspace.tokenizer,
                )

    def test_receipt_slot_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            workspace, checkpoint = self._formal_workspace(Path(temporary))
            receipt_path = workspace.root / "receipts/wrong.json"
            receipt = _raw_receipt(workspace, "M_LD/seed-42", checkpoint)
            receipt["model_key"] = "M_drop/seed-42"
            receipt["receipt_sha256"] = canonical_sha256(
                {key: value for key, value in receipt.items() if key != "receipt_sha256"}
            )
            write_canonical_json(receipt_path, receipt)
            with self.assertRaisesRegex(ModelRegistryError, "role/seed"):
                register_training_receipt(
                    receipt_json=receipt_path,
                    training_plan_ref=workspace.refs / "training_plan_ref.json",
                    schedule_ref=workspace.refs / "schedule_ref.json",
                    base_model_ref=workspace.base_ref,
                    environment_ref=workspace.environment_ref,
                    write_ref=workspace.refs / "wrong_receipt_ref.json",
                    workspace_root=workspace.root,
                    tokenizer=workspace.tokenizer,
                )

    def test_same_receipt_cannot_register_replacement_checkpoint_tree(self):
        with tempfile.TemporaryDirectory() as temporary:
            workspace, selected = self._formal_workspace(Path(temporary))
            receipt_ref = self._register_receipt_document(
                workspace,
                _raw_receipt(workspace, "M_LD/seed-42", selected),
                "checkpoint-bound",
            )
            replacement = _model_tree(
                workspace.root, "replacement/checkpoint-10"
            )
            write_bytes_atomic(
                replacement / "model.safetensors", b"different-selected-weights"
            )
            with self.assertRaisesRegex(
                ModelRegistryError,
                "differs from the selected checkpoint receipt",
            ):
                register_trained_model(
                    checkpoint_dir=replacement,
                    tokenizer_dir=None,
                    checkpoint_format="full",
                    model_key="M_LD/seed-42",
                    training_plan_ref=workspace.refs / "training_plan_ref.json",
                    schedule_ref=workspace.refs / "schedule_ref.json",
                    training_receipt_ref=receipt_ref,
                    base_model_ref=workspace.base_ref,
                    environment_ref=workspace.environment_ref,
                    write_ref=workspace.refs / "replacement-model-ref.json",
                    workspace_root=workspace.root,
                    target_root=workspace.root / "artifacts/models",
                    tokenizer=workspace.tokenizer,
                )

    def test_plan_schedule_and_registered_receipt_mismatches_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            workspace, checkpoint = self._formal_workspace(Path(temporary))
            original = _raw_receipt(workspace, "M_LD/seed-42", checkpoint)
            for field in ("training_plan_dependency", "schedule_dependency"):
                receipt = copy.deepcopy(original)
                receipt[field]["artifact_id"] = "foreign-" + "f" * 64
                receipt["receipt_sha256"] = canonical_sha256(
                    {key: value for key, value in receipt.items() if key != "receipt_sha256"}
                )
                receipt_path = workspace.root / "receipts" / f"bad-{field}.json"
                write_canonical_json(receipt_path, receipt)
                with self.assertRaisesRegex(ModelRegistryError, field):
                    register_training_receipt(
                        receipt_json=receipt_path,
                        training_plan_ref=workspace.refs / "training_plan_ref.json",
                        schedule_ref=workspace.refs / "schedule_ref.json",
                        base_model_ref=workspace.base_ref,
                        environment_ref=workspace.environment_ref,
                        write_ref=workspace.refs / f"bad-{field}-ref.json",
                        workspace_root=workspace.root,
                        tokenizer=workspace.tokenizer,
                    )

            receipt_path = workspace.root / "receipts/valid-left.json"
            write_canonical_json(receipt_path, original)
            receipt_ref = workspace.refs / "valid-left-receipt-ref.json"
            register_training_receipt(
                receipt_json=receipt_path,
                training_plan_ref=workspace.refs / "training_plan_ref.json",
                schedule_ref=workspace.refs / "schedule_ref.json",
                base_model_ref=workspace.base_ref,
                environment_ref=workspace.environment_ref,
                write_ref=receipt_ref,
                workspace_root=workspace.root,
                tokenizer=workspace.tokenizer,
            )
            with self.assertRaisesRegex(ModelRegistryError, "receipt and model slot"):
                register_trained_model(
                    checkpoint_dir=checkpoint,
                    tokenizer_dir=None,
                    checkpoint_format="full",
                    model_key="M_drop/seed-42",
                    training_plan_ref=workspace.refs / "training_plan_ref.json",
                    schedule_ref=workspace.refs / "schedule_ref.json",
                    training_receipt_ref=receipt_ref,
                    base_model_ref=workspace.base_ref,
                    environment_ref=workspace.environment_ref,
                    write_ref=workspace.refs / "bad-model-ref.json",
                    workspace_root=workspace.root,
                    tokenizer=workspace.tokenizer,
                )

    def test_adapter_checkpoint_uses_registered_base_and_frozen_tokenizer(self):
        with tempfile.TemporaryDirectory() as temporary:
            workspace, _ = self._formal_workspace(Path(temporary))
            adapter = workspace.root / "sources/adapters/checkpoint-10"
            adapter.mkdir(parents=True)
            write_canonical_json(
                adapter / "adapter_config.json", {"peft_type": "LORA"}
            )
            write_bytes_atomic(
                adapter / "adapter_model.safetensors", b"synthetic-adapter"
            )
            receipt_path = workspace.root / "receipts/adapter.json"
            write_canonical_json(
                receipt_path,
                _raw_receipt(workspace, "M_LD/seed-42", adapter),
            )
            receipt_ref = workspace.refs / "adapter_receipt_ref.json"
            register_training_receipt(
                receipt_json=receipt_path,
                training_plan_ref=workspace.refs / "training_plan_ref.json",
                schedule_ref=workspace.refs / "schedule_ref.json",
                base_model_ref=workspace.base_ref,
                environment_ref=workspace.environment_ref,
                write_ref=receipt_ref,
                workspace_root=workspace.root,
                tokenizer=workspace.tokenizer,
            )
            model_ref = workspace.refs / "adapter_model_ref.json"
            register_trained_model(
                checkpoint_dir=adapter,
                tokenizer_dir=workspace.root / "models/base/Qwen3-8B",
                checkpoint_format="adapter",
                model_key="M_LD/seed-42",
                training_plan_ref=workspace.refs / "training_plan_ref.json",
                schedule_ref=workspace.refs / "schedule_ref.json",
                training_receipt_ref=receipt_ref,
                base_model_ref=workspace.base_ref,
                environment_ref=workspace.environment_ref,
                write_ref=model_ref,
                workspace_root=workspace.root,
                tokenizer=workspace.tokenizer,
            )
            report = validate_model_artifact(
                model_ref,
                workspace_root=workspace.root,
                tokenizer=workspace.tokenizer,
            )
            self.assertEqual(report["checkpoint_format"], "adapter")


if __name__ == "__main__":
    unittest.main()
