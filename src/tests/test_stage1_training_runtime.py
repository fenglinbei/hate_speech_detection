from __future__ import annotations

import copy
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from finetune.stage1_runtime import (
    RUNTIME_VALIDATION_MARKER,
    ScheduleAwareTokenizedDataset,
    ScheduleEpochTracker,
    Stage1RuntimeError,
    build_training_receipt,
    resolve_stage1_runtime_inputs,
    snapshot_selected_checkpoint_inventory,
    write_immutable_training_receipt,
)
from data.training_artifacts import (
    canonical_sha256,
    sha256_file,
    write_bytes_atomic,
    write_canonical_json,
)
from data.training_schedule import build_training_schedule
from model.stage1_registry import inventory_regular_file_tree
from tests.test_stage1_training_artifacts import (
    SyntheticTrainingWorkspace,
    deterministic_context_tokenizer_constructor,
    deterministic_formal_score_replayer,
)


def _rows(epoch: int):
    return [
        {
            "query_id": query_id,
            "epoch": epoch,
            "partition": "calibration" if query_id == "2" else "fit",
            "presentation_epoch": 1 if query_id == "2" else epoch,
            "input": f"input-{query_id}-epoch-{epoch}",
            "record_sha256": character * 64,
        }
        for query_id, character in (("1", "a"), ("2", "b"), ("3", "c"))
    ]


def _encoder(row, index):
    return {
        "input_ids": [row["epoch"], int(row["query_id"])],
        "attention_mask": [1, 1],
        "labels": [-100, int(row["query_id"])],
    }


class FormalTrainingSourceLeaseTests(unittest.TestCase):
    @staticmethod
    def _fixture(root: Path):
        source_parent = root / "sources"
        source = source_parent / "base-model"
        source.mkdir(parents=True)
        write_bytes_atomic(source / "model.safetensors", b"verified-training-weights")
        write_canonical_json(source / "tokenizer.json", {"version": "verified"})
        inventory = inventory_regular_file_tree(
            source,
            workspace_root=root,
            inventory_policy="all-regular-files/v1",
        )
        bundle = SimpleNamespace(
            workspace_root=root,
            base_model_document={
                "checkpoint_inventory": inventory,
                "tokenizer_inventory": inventory,
                "base_inventory": inventory,
            },
            base_model_path=source,
            tokenizer_path=source,
        )
        attacker_parent = root / "attacker-tree"
        attacker_source = attacker_parent / "base-model"
        attacker_source.mkdir(parents=True)
        write_bytes_atomic(
            attacker_source / "model.safetensors", b"unverified-training-weights"
        )
        write_canonical_json(
            attacker_source / "tokenizer.json", {"version": "unverified"}
        )
        return bundle, source_parent, source, attacker_parent

    @staticmethod
    def _swap_ancestor_during_load(
        *,
        root: Path,
        source_parent: Path,
        attacker_parent: Path,
        observe: Path,
        observed: list[bytes],
    ) -> None:
        parked_parent = root / "verified-tree-parked"
        os.replace(source_parent, parked_parent)
        os.replace(attacker_parent, source_parent)
        try:
            observed.append(observe.read_bytes())
        finally:
            os.replace(source_parent, attacker_parent)
            os.replace(parked_parent, source_parent)

    def test_formal_tokenizer_load_rejects_ancestor_swap_and_restore(self):
        from finetune import train as training_module

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle, source_parent, source, attacker_parent = self._fixture(root)
            observed: list[bytes] = []

            def transient_tokenizer_load(*_args, **_kwargs):
                self._swap_ancestor_during_load(
                    root=root,
                    source_parent=source_parent,
                    attacker_parent=attacker_parent,
                    observe=source / "tokenizer.json",
                    observed=observed,
                )
                return SimpleNamespace(pad_token=None, eos_token="<eos>")

            with patch.object(
                training_module.AutoTokenizer,
                "from_pretrained",
                side_effect=transient_tokenizer_load,
            ), self.assertRaisesRegex(
                Stage1RuntimeError, "changed during backend load"
            ):
                training_module.load_tokenizer(
                    {"model_path": str(source)}, runtime_bundle=bundle
                )
            self.assertIn(b'"version":"unverified"', observed[0])

    def test_formal_replay_tokenizer_rejects_ancestor_swap_and_restore(self):
        from finetune import stage1_runtime as runtime_module

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle, source_parent, source, attacker_parent = self._fixture(root)
            observed: list[bytes] = []

            def transient_tokenizer_load(*_args, **_kwargs):
                self._swap_ancestor_during_load(
                    root=root,
                    source_parent=source_parent,
                    attacker_parent=attacker_parent,
                    observe=source / "tokenizer.json",
                    observed=observed,
                )
                return SimpleNamespace(eos_token_id=1)

            with patch(
                "transformers.AutoTokenizer.from_pretrained",
                side_effect=transient_tokenizer_load,
            ), self.assertRaisesRegex(
                Stage1RuntimeError, "changed during backend load"
            ):
                runtime_module._load_verified_runtime_tokenizer(
                    workspace_root=root,
                    base_model_document=bundle.base_model_document,
                    base_model_path=source,
                    tokenizer_path=source,
                )
            self.assertIn(b'"version":"unverified"', observed[0])

    def test_formal_model_load_rejects_ancestor_swap_and_restore(self):
        from finetune import train as training_module

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle, source_parent, source, attacker_parent = self._fixture(root)
            observed: list[bytes] = []

            def transient_model_load(*_args, **_kwargs):
                self._swap_ancestor_during_load(
                    root=root,
                    source_parent=source_parent,
                    attacker_parent=attacker_parent,
                    observe=source / "model.safetensors",
                    observed=observed,
                )
                return SimpleNamespace(config=SimpleNamespace(use_cache=True))

            with patch.object(
                training_module.AutoModelForCausalLM,
                "from_pretrained",
                side_effect=transient_model_load,
            ), patch.object(
                training_module, "get_train_backend", return_value="single"
            ), patch.object(
                training_module, "get_world_size", return_value=1
            ), patch.object(
                training_module, "build_device_map", return_value="cpu"
            ), self.assertRaisesRegex(
                Stage1RuntimeError, "changed during backend load"
            ):
                training_module.load_model(
                    {"model_path": str(source), "training": {"bf16": False}},
                    SimpleNamespace(deepspeed=None),
                    runtime_bundle=bundle,
                )
            self.assertEqual(observed, [b"unverified-training-weights"])


@patch(
    "data.build_context_manifest._default_formal_score_replayer",
    new=deterministic_formal_score_replayer,
)
@patch(
    "data.build_context_manifest._construct_formal_tokenizer",
    new=deterministic_context_tokenizer_constructor,
)
class ScheduleAwareRuntimeTests(unittest.TestCase):
    @staticmethod
    def _runtime_workspace(root: Path):
        workspace = SyntheticTrainingWorkspace(root)
        template_fields = [
            "exp_name",
            "random_seed",
            "training.output_dir",
            "training.seed",
            "training.data_seed",
            "training.num_train_epochs",
            "data.train_data_path",
        ]
        for filename in ("train_m_ld.json", "train_m_drop.json"):
            path = root / "config/stage1" / filename
            config = json.loads(path.read_text(encoding="utf-8"))
            config["template_fields_resolved_by_training_plan"] = template_fields
            write_canonical_json(path, config)
        workspace.build_evidence()
        workspace.build_plan(workspace.refs / "training_evidence_ref.json")
        build_training_schedule(
            training_plan_ref=workspace.refs / "training_plan_ref.json",
            training_evidence_ref=workspace.refs / "training_evidence_ref.json",
            train_partition_ref=workspace.partition_ref,
            tokenizer=workspace.tokenizer,
            write_ref=workspace.refs / "schedule_ref.json",
            workspace_root=workspace.root,
        )
        return workspace

    def test_runtime_unblocks_only_for_exact_matching_refs_and_model_key(self):
        with tempfile.TemporaryDirectory() as temporary:
            workspace = self._runtime_workspace(Path(temporary))
            source = json.loads(
                (workspace.root / "config/stage1/train_m_ld.json").read_text(
                    encoding="utf-8"
                )
            )
            bundle = resolve_stage1_runtime_inputs(
                source,
                training_plan_ref=workspace.refs / "training_plan_ref.json",
                training_evidence_ref=workspace.refs / "training_evidence_ref.json",
                train_partition_ref=workspace.partition_ref,
                schedule_ref=workspace.refs / "schedule_ref.json",
                base_model_ref=workspace.base_ref,
                environment_ref=workspace.environment_ref,
                model_key="M_LD/seed-42",
                tokenizer=workspace.tokenizer,
            )
            marker = bundle.resolved_config["_stage1_runtime_validation"]
            self.assertEqual(marker["schema_version"], RUNTIME_VALIDATION_MARKER)
            self.assertEqual(marker["model_key"], "M_LD/seed-42")
            self.assertIsInstance(bundle.resolved_config["training"]["deepspeed"], dict)
            self.assertEqual(
                bundle.resolved_config["training"]["deepspeed"],
                bundle.slot["deepspeed_config_resolved"],
            )
            self.assertEqual(
                marker["deepspeed_config_sha256"],
                canonical_sha256(bundle.slot["deepspeed_config_resolved"]),
            )
            self.assertEqual(sorted(bundle.records_by_epoch), [1, 2, 3, 4, 5])

            with self.assertRaisesRegex(Stage1RuntimeError, "exactly one plan slot"):
                resolve_stage1_runtime_inputs(
                    source,
                    training_plan_ref=workspace.refs / "training_plan_ref.json",
                    training_evidence_ref=workspace.refs / "training_evidence_ref.json",
                    train_partition_ref=workspace.partition_ref,
                    schedule_ref=workspace.refs / "schedule_ref.json",
                    base_model_ref=workspace.base_ref,
                    environment_ref=workspace.environment_ref,
                    model_key="M_LD/seed-999",
                    tokenizer=workspace.tokenizer,
                )

            wrong_role = json.loads(
                (workspace.root / "config/stage1/train_m_drop.json").read_text(
                    encoding="utf-8"
                )
            )
            with self.assertRaisesRegex(Stage1RuntimeError, "config role"):
                resolve_stage1_runtime_inputs(
                    wrong_role,
                    training_plan_ref=workspace.refs / "training_plan_ref.json",
                    training_evidence_ref=workspace.refs
                    / "training_evidence_ref.json",
                    train_partition_ref=workspace.partition_ref,
                    schedule_ref=workspace.refs / "schedule_ref.json",
                    base_model_ref=workspace.base_ref,
                    environment_ref=workspace.environment_ref,
                    model_key="M_LD/seed-42",
                    tokenizer=workspace.tokenizer,
                )

    def test_deepspeed_consume_binding_is_cwd_invariant_and_fail_closed(self):
        from finetune import train as training_module

        repository_root = Path(__file__).resolve().parents[2]
        config = json.loads(
            (repository_root / "config/stage1/train_m_ld.json").read_text(
                encoding="utf-8"
            )
        )
        frozen = json.loads(
            (repository_root / "config/stage1/deepspeed_zero3.json").read_text(
                encoding="utf-8"
            )
        )
        config["training"]["deepspeed"] = copy.deepcopy(frozen)
        config["_stage1_runtime_validation"] = {
            "schema_version": RUNTIME_VALIDATION_MARKER,
            "deepspeed_config_sha256": canonical_sha256(frozen),
        }

        observed: list[dict] = []
        training_argument_fields = training_module.TrainingArguments.__dataclass_fields__

        class CapturingTrainingArguments:
            __dataclass_fields__ = training_argument_fields

            def __new__(cls, **kwargs):
                observed.append(copy.deepcopy(kwargs))
                return SimpleNamespace(**kwargs)

        with tempfile.TemporaryDirectory(dir="/tmp") as attacker_temporary:
            attacker_root = Path(attacker_temporary)
            attacker_config = attacker_root / "config/stage1/deepspeed_zero3.json"
            attacker_config.parent.mkdir(parents=True)
            write_canonical_json(
                attacker_config,
                {"zero_optimization": {"stage": 0}, "attacker_controlled": True},
            )
            original_cwd = Path.cwd()
            try:
                with patch.object(
                    training_module,
                    "TrainingArguments",
                    CapturingTrainingArguments,
                ):
                    os.chdir(repository_root)
                    training_module.build_training_args(config)
                    os.chdir(attacker_root)
                    training_module.build_training_args(config)
            finally:
                os.chdir(original_cwd)

        self.assertEqual(len(observed), 2)
        self.assertEqual(observed[0]["deepspeed"], frozen)
        self.assertEqual(observed[1]["deepspeed"], frozen)
        self.assertEqual(observed[0]["deepspeed"], observed[1]["deepspeed"])
        self.assertNotIn("attacker_controlled", observed[1]["deepspeed"])

        path_rebound = copy.deepcopy(config)
        path_rebound["training"]["deepspeed"] = (
            "config/stage1/deepspeed_zero3.json"
        )
        with self.assertRaisesRegex(RuntimeError, "resolved object, not a path"):
            training_module.build_training_args(path_rebound)

        object_rebound = copy.deepcopy(config)
        object_rebound["training"]["deepspeed"]["zero_optimization"]["stage"] = 0
        with self.assertRaisesRegex(RuntimeError, "immutable runtime binding"):
            training_module.build_training_args(object_rebound)

        marker_rebound = copy.deepcopy(config)
        marker_rebound["_stage1_runtime_validation"].pop(
            "deepspeed_config_sha256"
        )
        with self.assertRaisesRegex(RuntimeError, "immutable runtime binding"):
            training_module.build_training_args(marker_rebound)

        legacy = copy.deepcopy(config)
        legacy.pop("schema_version")
        legacy.pop("_stage1_runtime_validation")
        legacy["training"]["deepspeed"] = "config/finetune/normal/ds_config.json"
        with patch.object(
            training_module,
            "TrainingArguments",
            CapturingTrainingArguments,
        ):
            training_module.build_training_args(legacy)
        self.assertEqual(
            observed[-1]["deepspeed"], "config/finetune/normal/ds_config.json"
        )

    def test_dataset_switches_rendering_by_epoch_without_changing_partition(self):
        dataset = ScheduleAwareTokenizedDataset(
            {1: _rows(1), 2: _rows(2)},
            selected_query_ids=["1", "3"],
            encoder=_encoder,
        )
        self.assertEqual(len(dataset), 2)
        self.assertEqual(dataset[0]["input_ids"], [1, 1])
        dataset.set_epoch(2)
        self.assertEqual(dataset[0]["input_ids"], [2, 1])
        self.assertEqual(dataset.record_hashes, ["a" * 64, "c" * 64])

    def test_epoch_tracker_keeps_train_and_calibration_in_sync(self):
        train = ScheduleAwareTokenizedDataset(
            {1: _rows(1), 2: _rows(2)}, selected_query_ids=["1", "3"], encoder=_encoder
        )
        evaluation = ScheduleAwareTokenizedDataset(
            {1: _rows(1), 2: _rows(2)},
            selected_query_ids=["2"],
            encoder=_encoder,
            fixed_epoch=1,
        )
        tracker = ScheduleEpochTracker(train_dataset=train, eval_dataset=evaluation)
        control = object()
        returned = tracker.on_epoch_begin(None, SimpleNamespace(epoch=1.0), control)
        self.assertIs(returned, control)
        self.assertEqual(train.epoch, 2)
        self.assertEqual(evaluation.epoch, 1)
        tracker.on_epoch_end(None, SimpleNamespace(epoch=2.0, global_step=20), control)
        self.assertEqual(tracker.completed_epochs, [2])
        self.assertEqual(tracker.epoch_global_steps, {2: 20})

    def test_receipt_binds_only_immutable_dependencies_and_consumed_hashes(self):
        checkpoint_workspace = tempfile.TemporaryDirectory()
        self.addCleanup(checkpoint_workspace.cleanup)
        checkpoint_root = Path(checkpoint_workspace.name)
        selected_checkpoint = checkpoint_root / "outputs/checkpoint-5"
        selected_checkpoint.mkdir(parents=True)
        (selected_checkpoint / "model.safetensors").write_bytes(b"selected-weights")
        write_canonical_json(
            selected_checkpoint / "config.json", {"model_type": "fixture"}
        )
        train = ScheduleAwareTokenizedDataset(
            {1: _rows(1), 2: _rows(2)}, selected_query_ids=["1", "3"], encoder=_encoder
        )
        evaluation = ScheduleAwareTokenizedDataset(
            {1: _rows(1), 2: _rows(2)},
            selected_query_ids=["2"],
            encoder=_encoder,
            fixed_epoch=1,
        )
        tracker = ScheduleEpochTracker(
            train_dataset=train,
            eval_dataset=evaluation,
            completed_epochs=[1, 2],
            epoch_global_steps={1: 5, 2: 10},
            evaluation_history=[
                {
                    "epoch": 1,
                    "global_step": 5,
                    "metric_name": "eval_loss",
                    "metric_value": 1.0,
                },
                {
                    "epoch": 2,
                    "global_step": 10,
                    "metric_name": "eval_loss",
                    "metric_value": 1.1,
                },
            ],
        )
        dependency = {
            "schema_version": "stage1-dependency-ref/v1",
            "artifact_kind": "fixture",
            "artifact_id": "fixture-id",
            "payload_manifest_sha256": "d" * 64,
            "logical_repo_path": "artifacts/fixture-id",
        }
        bundle = SimpleNamespace(
            workspace_root=checkpoint_root,
            plan={
                "runtime_code_sha256": sha256_file(
                    Path(__file__).resolve().parents[1]
                    / "finetune/stage1_runtime.py"
                )
            },
            plan_dependency={**dependency, "artifact_kind": "training-plan"},
            evidence_dependency={**dependency, "artifact_kind": "training-evidence"},
            partition_dependency={**dependency, "artifact_kind": "train-partition"},
            partition_meta={
                "fit_count": 2,
                "calibration_count": 1,
                "fit_ids_sha256": canonical_sha256(["1", "3"]),
                "calibration_ids_sha256": canonical_sha256(["2"]),
            },
            schedule_dependency={**dependency, "artifact_kind": "training-schedule"},
            base_model_dependency={**dependency, "artifact_kind": "model"},
            environment_dependency={**dependency, "artifact_kind": "environment"},
            slot={
                "model_key": "M_drop/seed-42",
                "role": "M_drop",
                "seed": 42,
                "epochs": 2,
                "final_checkpoint_rule": "best-eval-loss-threshold-earliest/v1",
                "train_config_sha256": "e" * 64,
                "train_config_resolved": {
                    "early_stopping": {
                        "enabled": True,
                        "metric": "eval_loss",
                        "mode": "min",
                        "minimum_epochs": 1,
                        "maximum_epochs": 2,
                        "patience_evaluations": 1,
                        "threshold": 0.0,
                        "tie_break": "earliest-global-step",
                        "selection_data": "train-only-calibration",
                        "scientific_dev_used_for_selection": False,
                    }
                },
            },
            records_by_epoch={1: _rows(1), 2: _rows(2)},
        )
        receipt = build_training_receipt(
            bundle,
            tracker,
            training_exit_global_step=10,
            selected_checkpoint_global_step=5,
            selected_checkpoint_dir=selected_checkpoint,
            train_code_sha256="f" * 64,
            trainer_best_metric=1.0,
            trainer_best_global_step=5,
        )
        self.assertEqual(receipt["completed_epochs"], [1, 2])
        self.assertEqual(
            receipt["checkpoint_selection_audit"]["selected_metric_value"],
            1.0,
        )
        self.assertNotIn("output_dir", receipt)
        self.assertNotIn("timestamp", receipt)
        self.assertEqual(receipt["selected_checkpoint_epoch"], 1)
        self.assertEqual(receipt["global_step"], 5)
        self.assertEqual(
            receipt["selected_checkpoint_inventory"]["logical_repo_path"],
            "outputs/checkpoint-5",
        )
        self.assertEqual(
            receipt["selected_checkpoint_inventory"]["file_count"], 2
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "training_receipt.json"
            write_immutable_training_receipt(path, receipt)
            write_immutable_training_receipt(path, receipt)
            changed = copy.deepcopy(receipt)
            changed["global_step"] = 11
            with self.assertRaisesRegex(Stage1RuntimeError, "overwrite"):
                write_immutable_training_receipt(path, changed)

    def test_receipt_rejects_noncontiguous_completed_epochs(self):
        train = ScheduleAwareTokenizedDataset(
            {1: _rows(1), 2: _rows(2)}, selected_query_ids=["1", "3"], encoder=_encoder
        )
        evaluation = ScheduleAwareTokenizedDataset(
            {1: _rows(1), 2: _rows(2)},
            selected_query_ids=["2"],
            encoder=_encoder,
            fixed_epoch=1,
        )
        tracker = ScheduleEpochTracker(
            train_dataset=train, eval_dataset=evaluation, completed_epochs=[2]
        )
        bundle = SimpleNamespace(slot={"epochs": 2})
        with self.assertRaisesRegex(Stage1RuntimeError, "contiguous"):
            build_training_receipt(
                bundle,
                tracker,
                training_exit_global_step=10,
                selected_checkpoint_global_step=10,
                selected_checkpoint_dir="checkpoint-10",
                train_code_sha256="f" * 64,
                trainer_best_metric=1.0,
                trainer_best_global_step=10,
            )

    def test_selected_checkpoint_inventory_rejects_symlinks(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "outputs/checkpoint-5"
            checkpoint.mkdir(parents=True)
            weights = root / "weights.bin"
            weights.write_bytes(b"weights")
            (checkpoint / "model.safetensors").symlink_to(weights)
            with self.assertRaisesRegex(Stage1RuntimeError, "symlink"):
                snapshot_selected_checkpoint_inventory(
                    checkpoint,
                    workspace_root=root,
                    expected_global_step=5,
                )


if __name__ == "__main__":
    unittest.main()
