from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import data.training_evidence as training_evidence_module

from data.build_context_manifest import build_prepared_context_artifact
from data.retrieval_bundle import (
    build_lexicon_catalog,
    prepare_context_bundle_from_scores,
)
from data.training_artifacts import (
    TrainingArtifactError,
    build_payload_manifest,
    canonical_json_bytes,
    canonical_sha256,
    finalize_target_atomic,
    new_staging_directory,
    portable_dependency,
    resolve_locator_ref,
    validate_payload_manifest,
    write_bytes_atomic,
    write_canonical_json,
    write_canonical_jsonl,
    write_locator_ref,
)
from data.train_partition import build_train_partition, load_train_partition
from data.training_evidence import (
    BUDGET_POLICY,
    EVIDENCE_REPLAY_POLICY,
    RENDERER_REVISION,
    TrainingEvidenceError,
    build_training_evidence,
    load_training_evidence,
    recompute_evidence_record_sha256,
    validate_training_evidence,
)
from data.training_plan import (
    TrainingPlanError,
    freeze_training_plan,
    load_training_plan,
    validate_training_plan,
)
from scripts.stage1 import build_training_evidence as evidence_cli
from scripts.stage1 import build_training_plan as plan_cli
from tests.stage1_semantic_fixtures import (
    make_embedding_model_fixture,
    make_formal_lexicon_artifact,
    make_semantic_data_artifact,
)
from tests.stage1_registered_model_fixtures import (
    make_current_environment_artifact,
    make_registered_base_model_artifact,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


class FakeTokenizer:
    eos_token_id = 99

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        # One deterministic token per Unicode scalar keeps Chinese prompt
        # fixtures within the production sequence budget without pretending
        # UTF-8 bytes are model tokens. ASCII behavior remains unchanged.
        return [ord(character) for character in text]

    def apply_chat_template(
        self,
        conversation: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        enable_thinking: bool = False,
    ) -> str:
        if tokenize or not add_generation_prompt or enable_thinking:
            raise AssertionError("fixture expects non-thinking generation-prompt rendering")
        return "\n".join(
            f"<{message['role']}>{message['content']}" for message in conversation
        ) + "\n<assistant>"


def make_id(prefix: str, atom: str) -> str:
    return prefix + canonical_sha256({"atom": atom})


def deterministic_formal_score_replayer(
    *,
    model_path,
    device_class,
    batch_size,
    train_texts,
    query_texts,
    lexicon_texts,
):
    del model_path, device_class, batch_size
    # Deliberately make retrieval prompt order differ from lex:v2 canonical ID
    # order so the set-encoding test cannot pass through an ordering accident.
    lexicon_row = np.arange(len(lexicon_texts), dtype=np.float64)[::-1]
    return (
        np.zeros((len(query_texts), len(train_texts)), dtype=np.float64),
        np.broadcast_to(
            lexicon_row, (len(query_texts), len(lexicon_texts))
        ).copy(),
    )


def deterministic_context_tokenizer_constructor(path):
    del path
    return FakeTokenizer()


class SyntheticTrainingWorkspace:
    def __init__(self, root: Path):
        self.root = root
        self.artifact_root = root / "artifacts"
        self.refs = root / "refs"
        self.tokenizer = FakeTokenizer()
        self.data_ref, self.data_splits = make_semantic_data_artifact(
            root, collapse_calibration_cluster=True
        )
        self.lexicon_ref = make_formal_lexicon_artifact(
            root, data_ref=self.data_ref
        )
        self.partition_ref = self.root / "train_partition_ref.json"
        self.partition = load_train_partition(
            self.partition_ref, workspace_root=self.root
        )
        self.embedding_model, self.embedding_model_hash = (
            make_embedding_model_fixture(root)
        )
        self.environment_ref = make_current_environment_artifact(root)
        self.base_ref = make_registered_base_model_artifact(
            root, environment_ref=self.environment_ref
        )
        self.context_ref, self.context_record = self._context_artifact()
        self._write_plan_sources()

    def _simple_artifact(self, kind: str, prefix: str, atom: str) -> Path:
        artifact_id = make_id(prefix, atom)
        target = self.artifact_root / f"{kind}s" / artifact_id
        target.mkdir(parents=True)
        write_canonical_json(
            target / "artifact.json",
            {"schema_version": f"synthetic-{kind}/v1", "artifact_id": artifact_id},
        )
        write_canonical_json(target / "payload_manifest.json", build_payload_manifest(target))
        ref_path = self.refs / f"{atom}_ref.json"
        write_locator_ref(
            ref_path,
            artifact_kind=kind,
            artifact_id=artifact_id,
            target=target,
            payload_manifest_sha256=validate_payload_manifest(target),
        )
        return ref_path

    def _dependency(self, ref_path: Path) -> dict:
        locator, target = resolve_locator_ref(ref_path)
        return portable_dependency(locator, target, self.root)

    def _context_artifact(self) -> tuple[Path, dict]:
        train_records = self.data_splits["train"]
        config = json.loads(
            (REPOSITORY_ROOT / "config/stage1/context_factorial.json").read_text(
                encoding="utf-8"
            )
        )
        config["retrieval"]["allocated_class_top_k"] = {
            label: (1 if label == "Racism" else 0)
            for label in config["retrieval"]["source_class_order"]
        }
        config["retrieval"]["demo_top_k"] = 1
        lexicon_locator = json.loads(self.lexicon_ref.read_text(encoding="utf-8"))
        lexicon_document = json.loads(
            (Path(lexicon_locator["target_path"]) / "lexicon.json").read_text(
                encoding="utf-8"
            )
        )
        lexicon_rows = build_lexicon_catalog(lexicon_document["terms"])
        demo_scores, lexicon_scores = deterministic_formal_score_replayer(
            model_path=self.embedding_model,
            device_class="cpu",
            batch_size=1,
            train_texts=[row["content"] for row in self.partition.fit_records],
            query_texts=[row["content"] for row in train_records],
            lexicon_texts=[row["rendered_block"] for row in lexicon_rows],
        )
        prepared_bundle = prepare_context_bundle_from_scores(
            train_records=train_records,
            query_records=train_records,
            lexicon_terms=lexicon_document["terms"],
            demo_scores=demo_scores,
            lexicon_scores=lexicon_scores,
            split="train",
            retrieval_config=config["retrieval"],
            data_dependency=self._dependency(self.data_ref),
            lexicon_dependency=self._dependency(self.lexicon_ref),
            scorer_provenance={
                "backend": "sentence-transformers-cosine/v1",
                "logical_model_path": self.embedding_model.relative_to(
                    self.root
                ).as_posix(),
                "model_file_tree_sha256": self.embedding_model_hash,
                "device_class": "cpu",
                "batch_size": 1,
            },
            fit_demo_records=self.partition.fit_records,
            calibration_ids=self.partition.calibration_ids,
            train_partition_dependency=self.partition.partition_dependency,
        )
        ref_path = self.refs / "context_ref.json"
        with patch(
            "data.build_context_manifest._construct_formal_tokenizer",
            new=deterministic_context_tokenizer_constructor,
        ), patch(
            "data.build_context_manifest._default_formal_score_replayer",
            new=deterministic_formal_score_replayer,
        ):
            locator = build_prepared_context_artifact(
                prepared_bundle=prepared_bundle,
                config=config,
                write_ref=ref_path,
                formal=True,
                data_ref=self.data_ref,
                train_partition_ref=self.partition_ref,
                lexicon_ref=self.lexicon_ref,
                target_root=self.artifact_root / "contexts",
                workspace_root=self.root,
            )
        target = Path(locator["target_path"])
        record = next(
            row
            for row in (
                json.loads(line)
                for line in (target / "context_manifest.train.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            )
            if row["query"]["id"] == "2"
        )
        return ref_path, record

    def _train_config(self, role: str) -> dict:
        drop = 0.5 if role == "M_drop" else 0.0
        return {
            "schema_version": "stage1-train-config/v1",
            "model_role": role,
            "execution_status": "blocked-pending-schedule-aware-loader",
            "exp_name": "source-template",
            "model_path": "models/base/Qwen3-8B",
            "random_seed": 42,
            "max_length": 2048,
            "data": {
                "source_schema": "stage1-training-schedule/v1",
                "requires_frozen_schedule": True,
                "train_data_path": None,
                "val_data_path": None,
                "selection_split": "train-only-calibration",
                "calibration": {
                    "assignment": "sha256-query-id-v1",
                    "fraction": 0.1,
                    "hash_modulus": 10000,
                    "hash_threshold_exclusive": 1000,
                    "id_fields": ["id", "query_id"],
                    "salt": "stage1-train-calibration-v1",
                },
                "partition": {
                    "artifact_kind": "train-partition",
                    "policy": "immutable-explicit-ref/v1",
                    "nominal_hash_policy": "assertion-only",
                },
                "fixed_presentation": {
                    "policy": "calibration-epoch-1-wire/v1",
                    "wire_epoch": 1,
                    "demo_order_across_epochs": True,
                    "source_mask_across_epochs": True,
                },
            },
            "context_policy": {
                "lexicon_dropout_probability": drop,
                "demonstration_dropout_probability": drop,
                "dropout_independent": True,
                "order_policy": "per-seed-epoch-query-sha256/v1",
                "matched_permutation_across_roles": True,
                **(
                    {"dropout_rng_policy": "sha256-model-seed-epoch-query-source/v1"}
                    if role == "M_drop"
                    else {}
                ),
            },
            "training": {
                "deepspeed": "config/stage1/deepspeed.json",
                "output_dir": f"future/{role}",
                "seed": 42,
                "data_seed": 42,
                "num_train_epochs": 5,
            },
            "checkpointing": {
                "final_checkpoint_rule": "best-eval-loss-threshold-earliest/v1"
            },
            "early_stopping": {
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
            },
        }

    def _write_plan_sources(self) -> None:
        config_root = self.root / "config/stage1"
        config_root.mkdir(parents=True)
        write_canonical_json(
            config_root / "deepspeed.json",
            {"schema_version": "synthetic-deepspeed/v1", "zero_optimization": {"stage": 3}},
        )
        for role, filename in (("M_LD", "train_m_ld.json"), ("M_drop", "train_m_drop.json")):
            write_canonical_json(config_root / filename, self._train_config(role))
        conditions = ["C0", "CL", "CD", "CLD", "PL", "PD"]
        write_canonical_json(
            config_root / "generation.json",
            {
                "schema_version": "stage1-generation-profile/v1",
                "ordered_conditions": conditions,
            },
        )
        write_canonical_json(
            config_root / "margin.json",
            {
                "schema_version": "stage1-margin-profile/v1",
                "ordered_conditions": conditions,
            },
        )
        write_canonical_json(
            config_root / "decision_register.json",
            {
                "schema_version": "stage1-decision-register/v1",
                "status": "frozen",
                "decisions": {"D7_training_seeds": {"formal": [42, 43, 44]}},
            },
        )
        slots = []
        for seed in (42, 43, 44):
            for role, filename in (("M_LD", "train_m_ld.json"), ("M_drop", "train_m_drop.json")):
                slots.append(
                    {
                        "model_key": f"{role}/seed-{seed}",
                        "role": role,
                        "seed": seed,
                        "train_config": f"config/stage1/{filename}",
                        "epochs": 5,
                        "final_checkpoint_rule": "best-eval-loss-threshold-earliest/v1",
                    }
                )
        self.source_spec = self.root / "exps/specs/stage1_context_factorial.json"
        write_canonical_json(
            self.source_spec,
            {
                "schema_version": "stage1-training-source-recipe/v1",
                "scope": "formal",
                "scientific_eligible": True,
                "execution_status": "blocked-pending-schedule-aware-loader",
                "ordered_model_slots": slots,
                "pilot_slot_keys": ["M_LD/seed-42", "M_drop/seed-42"],
                "epoch_and_checkpoint_selection_policy": {
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
                },
                "order_dropout_rng_policy": {
                    "order": "per-seed-epoch-query-sha256/v1",
                    "matched_permutation_across_roles": True,
                    "fit_presentation": "per-seed-epoch-query/v1",
                    "calibration_presentation": {
                        "policy": "calibration-epoch-1-wire/v1",
                        "wire_epoch": 1,
                        "demo_order_across_epochs": True,
                        "source_mask_across_epochs": True,
                    },
                    "m_drop": {
                        "lexicon_probability": 0.5,
                        "demonstration_probability": 0.5,
                        "independent_sources": True,
                        "rng": "sha256-model-seed-epoch-query-source/v1",
                    },
                },
                "protocol_profiles": {
                    "generation": "config/stage1/generation.json",
                    "margin": "config/stage1/margin.json",
                },
                "ordered_generation_conditions": conditions,
                "ordered_margin_conditions": conditions,
            },
        )
        self.train_code = self.root / "src/finetune/train.py"
        self.train_code.parent.mkdir(parents=True)
        self.train_code.write_text("# synthetic frozen train code\n", encoding="utf-8")
        self.runtime_code = self.root / "src/finetune/stage1_runtime.py"
        write_bytes_atomic(
            self.runtime_code,
            (REPOSITORY_ROOT / "src/finetune/stage1_runtime.py").read_bytes(),
        )

    def build_evidence(self, name: str = "training_evidence_ref.json") -> dict:
        return build_training_evidence(
            context_ref=self.context_ref,
            train_partition_ref=self.partition_ref,
            base_model_ref=self.base_ref,
            write_ref=self.refs / name,
            workspace_root=self.root,
            tokenizer=self.tokenizer,
            schema_path=REPOSITORY_ROOT / "schemas/stage1_training_evidence_v1.schema.json",
        )

    def build_plan(
        self,
        evidence_ref: Path,
        name: str = "training_plan_ref.json",
        *,
        context_ref: Path | None = None,
    ) -> dict:
        return freeze_training_plan(
            source_spec_path=self.source_spec,
            scope="formal",
            context_ref=context_ref or self.context_ref,
            training_evidence_ref=evidence_ref,
            train_partition_ref=self.partition_ref,
            base_model_ref=self.base_ref,
            environment_ref=self.environment_ref,
            write_ref=self.refs / name,
            workspace_root=self.root,
            decision_register_path=self.root / "config/stage1/decision_register.json",
            train_code_path=self.train_code,
            runtime_code_path=self.runtime_code,
            schema_path=REPOSITORY_ROOT / "schemas/stage1_training_plan_v1.schema.json",
            tokenizer=self.tokenizer,
        )

    @staticmethod
    def refresh_manifest_and_locator(ref_path: Path) -> None:
        locator = json.loads(ref_path.read_text(encoding="utf-8"))
        target = Path(locator["target_path"])
        write_canonical_json(target / "payload_manifest.json", build_payload_manifest(target))
        locator["payload_manifest_sha256"] = validate_payload_manifest(target)
        write_canonical_json(ref_path, locator)


@patch(
    "data.build_context_manifest._default_formal_score_replayer",
    new=deterministic_formal_score_replayer,
)
@patch(
    "data.build_context_manifest._construct_formal_tokenizer",
    new=deterministic_context_tokenizer_constructor,
)
class TrainingArtifactLifecycleTests(unittest.TestCase):
    def test_registered_tokenizer_is_local_trust_false_and_lease_scoped(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            artifact_id = "mdl-" + "1" * 64
            target = root / "artifacts/models" / artifact_id
            target.mkdir(parents=True)
            write_canonical_json(target / "artifact.json", {"id": artifact_id})
            write_canonical_json(
                target / "payload_manifest.json", build_payload_manifest(target)
            )
            dependency = {
                "schema_version": "stage1-dependency-ref/v1",
                "artifact_kind": "stage1-model",
                "artifact_id": artifact_id,
                "payload_manifest_sha256": validate_payload_manifest(target),
                "logical_repo_path": target.relative_to(root).as_posix(),
            }
            tokenizer_root = root / "registered-tokenizer"
            tokenizer_root.mkdir()
            events: list[str] = []
            constructor_calls: list[tuple[str, dict]] = []

            @contextmanager
            def fake_lease(contract, *, source_names):
                del contract
                self.assertEqual(source_names, ("tokenizer", "base"))
                events.append("enter")
                try:
                    yield SimpleNamespace(tokenizer_path=tokenizer_root)
                finally:
                    events.append("exit")

            model = {
                "artifact_type": "base",
                "checkpoint_format": "base",
                "checkpoint_inventory": {},
                "tokenizer_inventory": {},
                "base_inventory": {},
            }

            def construct(path, **kwargs):
                self.assertEqual(events, ["enter"])
                constructor_calls.append((path, kwargs))
                return FakeTokenizer()

            with (
                patch(
                    "model.stage1_registry.validate_model_artifact_target",
                    return_value=model,
                ),
                patch(
                    "model.stage1_registry.verified_model_source_lease",
                    new=fake_lease,
                ),
                patch(
                    "transformers.AutoTokenizer.from_pretrained",
                    side_effect=construct,
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "caller replay failed"):
                    with training_evidence_module._registered_evidence_tokenizer(
                        dependency, workspace_root=root
                    ) as resolved:
                        self.assertEqual(
                            resolved[0].encode("lease-covered"),
                            list("lease-covered".encode("utf-8")),
                        )
                        raise RuntimeError("caller replay failed")

            self.assertEqual(events, ["enter", "exit"])
            self.assertEqual(len(constructor_calls), 1)
            constructor_path, constructor_kwargs = constructor_calls[0]
            self.assertEqual(constructor_path, str(tokenizer_root))
            self.assertEqual(
                constructor_kwargs,
                {"local_files_only": True, "trust_remote_code": False},
            )

    def test_evidence_is_deterministic_train_only_canonical_set_encoding(self):
        with tempfile.TemporaryDirectory() as temporary:
            workspace = SyntheticTrainingWorkspace(Path(temporary))
            first = workspace.build_evidence()
            second = workspace.build_evidence("training_evidence_second_ref.json")
            locator, target, meta, records = load_training_evidence(
                workspace.refs / "training_evidence_ref.json",
                workspace_root=workspace.root,
                schema_path=REPOSITORY_ROOT / "schemas/stage1_training_evidence_v1.schema.json",
                tokenizer=workspace.tokenizer,
            )

            self.assertEqual(first["artifact_id"], second["artifact_id"])
            self.assertEqual(first["payload_manifest_sha256"], second["payload_manifest_sha256"])
            self.assertEqual(meta["split"], "train")
            self.assertEqual(meta["renderer_revision"], RENDERER_REVISION)
            self.assertEqual(meta["budget_policy"], BUDGET_POLICY)
            self.assertEqual(meta["replay_policy"], EVIDENCE_REPLAY_POLICY)
            self.assertEqual(len(records), 5781)
            record = records[1]
            self.assertEqual(record["query"]["id"], "2")
            self.assertEqual(
                record["query"]["actual_gold_tokens"],
                len(workspace.tokenizer.encode(record["query"]["gold_text"])),
            )
            self.assertEqual(
                record["demo_evidence"]["ids"],
                sorted(record["demo_evidence"]["ids"]),
            )
            self.assertEqual(
                record["lexicon_evidence"]["ids"],
                sorted(record["lexicon_evidence"]["ids"]),
            )
            self.assertNotEqual(
                record["lexicon_evidence"]["ids"],
                workspace.context_record["selection"]["lexicons"]["prompt_order_final"],
            )
            self.assertNotIn("prompt_order", json.dumps(record, ensure_ascii=False))
            embedded = json.loads((target / "context_ref.json").read_text(encoding="utf-8"))
            self.assertNotIn("target_path", embedded)
            self.assertFalse(Path(embedded["logical_repo_path"]).is_absolute())
            report = validate_training_evidence(
                workspace.refs / "training_evidence_ref.json",
                workspace_root=workspace.root,
                schema_path=REPOSITORY_ROOT / "schemas/stage1_training_evidence_v1.schema.json",
                tokenizer=workspace.tokenizer,
            )
            self.assertTrue(report["valid"])
            self.assertEqual(report["payload_manifest_sha256"], locator["payload_manifest_sha256"])

    def test_evidence_tamper_and_dependency_mismatch_fail_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            workspace = SyntheticTrainingWorkspace(Path(temporary))
            ref = workspace.build_evidence()
            target = Path(ref["target_path"])
            with (target / "training_evidence.train.jsonl").open("a", encoding="utf-8") as handle:
                handle.write("{}\n")
            with self.assertRaises(TrainingArtifactError):
                validate_training_evidence(
                    workspace.refs / "training_evidence_ref.json",
                    workspace_root=workspace.root,
                    tokenizer=workspace.tokenizer,
                )

        with tempfile.TemporaryDirectory() as temporary:
            workspace = SyntheticTrainingWorkspace(Path(temporary))
            ref = workspace.build_evidence()
            target = Path(ref["target_path"])
            dependency = json.loads((target / "context_ref.json").read_text(encoding="utf-8"))
            dependency["artifact_id"] = make_id("ctx-", "wrong")
            write_canonical_json(target / "context_ref.json", dependency)
            workspace.refresh_manifest_and_locator(
                workspace.refs / "training_evidence_ref.json"
            )
            with self.assertRaises(TrainingEvidenceError):
                validate_training_evidence(
                    workspace.refs / "training_evidence_ref.json",
                    workspace_root=workspace.root,
                    tokenizer=workspace.tokenizer,
                )

    def test_fully_resealed_forged_evidence_fails_exact_context_replay(self):
        with tempfile.TemporaryDirectory() as temporary:
            workspace = SyntheticTrainingWorkspace(Path(temporary))
            locator = workspace.build_evidence()
            target = Path(locator["target_path"])
            meta = json.loads(
                (target / "training_evidence.meta.json").read_text(
                    encoding="utf-8"
                )
            )
            provenance = json.loads(
                (target / "provenance.json").read_text(encoding="utf-8")
            )
            records = [
                json.loads(line)
                for line in (target / "training_evidence.train.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]

            # Forge a context-linked field, then update every evidence-local
            # hash, lifecycle ID, payload manifest, directory, and locator.
            # A self-consistency-only validator would accept this artifact.
            records[0]["source_context_record_sha256"] = "f" * 64
            records[0]["record_sha256"] = recompute_evidence_record_sha256(
                records[0]
            )
            records_content_sha256 = canonical_sha256(
                [
                    {
                        "query_id": record["query"]["id"],
                        "record_sha256": recompute_evidence_record_sha256(record),
                    }
                    for record in records
                ]
            )
            meta["records_content_sha256"] = records_content_sha256
            meta["id_inputs"]["records_content_sha256"] = records_content_sha256
            forged_id = "tevd-" + canonical_sha256(meta["id_inputs"])
            meta["training_evidence_build_id"] = forged_id
            for record in records:
                record["training_evidence_build_id"] = forged_id
            records_bytes = b"".join(
                canonical_json_bytes(record) + b"\n" for record in records
            )
            meta["records_sha256"] = hashlib.sha256(records_bytes).hexdigest()
            provenance["training_evidence_build_id"] = forged_id
            provenance["id_inputs"] = meta["id_inputs"]

            forged_target = target.parent / forged_id
            target.rename(forged_target)
            write_bytes_atomic(
                forged_target / "training_evidence.train.jsonl", records_bytes
            )
            write_canonical_json(
                forged_target / "training_evidence.meta.json", meta
            )
            write_canonical_json(forged_target / "provenance.json", provenance)
            write_canonical_json(
                forged_target / "payload_manifest.json",
                build_payload_manifest(forged_target),
            )
            forged_ref = workspace.refs / "forged_training_evidence_ref.json"
            write_locator_ref(
                forged_ref,
                artifact_kind="training-evidence",
                artifact_id=forged_id,
                target=forged_target,
                payload_manifest_sha256=validate_payload_manifest(forged_target),
            )

            with self.assertRaisesRegex(
                TrainingEvidenceError,
                "exact context manifest/catalog replay",
            ):
                validate_training_evidence(
                    forged_ref,
                    workspace_root=workspace.root,
                    tokenizer=workspace.tokenizer,
                )

    def test_plan_freezes_resolved_configs_without_future_bindings(self):
        with tempfile.TemporaryDirectory() as temporary:
            workspace = SyntheticTrainingWorkspace(Path(temporary))
            workspace.build_evidence()
            evidence_ref = workspace.refs / "training_evidence_ref.json"
            source_before = workspace.source_spec.read_bytes()
            first = workspace.build_plan(evidence_ref)
            second = workspace.build_plan(evidence_ref, "training_plan_second_ref.json")
            locator, target, plan = load_training_plan(
                workspace.refs / "training_plan_ref.json",
                workspace_root=workspace.root,
                schema_path=REPOSITORY_ROOT / "schemas/stage1_training_plan_v1.schema.json",
                tokenizer=workspace.tokenizer,
            )

            self.assertEqual(first["artifact_id"], second["artifact_id"])
            self.assertEqual(first["payload_manifest_sha256"], second["payload_manifest_sha256"])
            self.assertEqual(workspace.source_spec.read_bytes(), source_before)
            self.assertEqual(plan["scope"], "formal")
            self.assertEqual(len(plan["ordered_model_slots"]), 6)
            self.assertEqual(
                [slot["model_key"] for slot in plan["ordered_model_slots"][:2]],
                ["M_LD/seed-42", "M_drop/seed-42"],
            )
            for slot in plan["ordered_model_slots"]:
                resolved = slot["train_config_resolved"]
                self.assertEqual(
                    resolved["execution_status"],
                    "blocked-pending-schedule-aware-loader",
                )
                self.assertIsNone(resolved["training"]["output_dir"])
                self.assertIsNone(resolved["data"]["train_data_path"])
                self.assertEqual(resolved["training"]["seed"], slot["seed"])
                self.assertEqual(
                    slot["train_config_sha256"],
                    slot["resolved_train_config_sha256"],
                )
                self.assertEqual(
                    slot["resolved_train_config_sha256"], canonical_sha256(resolved)
                )
                self.assertRegex(slot["source_train_config_sha256"], r"^[0-9a-f]{64}$")
                self.assertNotIn("frozen_file_dependencies", resolved)
                self.assertEqual(
                    slot["deepspeed_config_sha256"],
                    canonical_sha256(slot["deepspeed_config_resolved"]),
                )
            plan_wire = (target / "plan.resolved.json").read_text(encoding="utf-8")
            self.assertNotIn("schedule_build_id", plan_wire)
            self.assertNotIn("checkpoint_path", plan_wire)
            self.assertNotIn('"model_ref"', plan_wire)
            for filename in (
                "training_evidence_ref.json",
                "base_model_ref.json",
                "environment_ref.json",
            ):
                embedded = json.loads((target / filename).read_text(encoding="utf-8"))
                self.assertNotIn("target_path", embedded)
            report = validate_training_plan(
                workspace.refs / "training_plan_ref.json",
                workspace_root=workspace.root,
                schema_path=REPOSITORY_ROOT / "schemas/stage1_training_plan_v1.schema.json",
                tokenizer=workspace.tokenizer,
            )
            self.assertTrue(report["valid"])
            self.assertEqual(report["training_plan_id"], locator["artifact_id"])

    def test_old_plan_is_immutable_and_new_config_produces_new_plan_id(self):
        with tempfile.TemporaryDirectory() as temporary:
            workspace = SyntheticTrainingWorkspace(Path(temporary))
            workspace.build_evidence()
            evidence_ref = workspace.refs / "training_evidence_ref.json"
            first = workspace.build_plan(evidence_ref)
            first_target_bytes = {
                path.relative_to(Path(first["target_path"])).as_posix(): path.read_bytes()
                for path in Path(first["target_path"]).rglob("*")
                if path.is_file()
            }
            # Matched roles may only differ in the preregistered context
            # dropout treatment, so change a shared hyperparameter in both.
            for filename in ("train_m_ld.json", "train_m_drop.json"):
                config_path = workspace.root / f"config/stage1/{filename}"
                config = json.loads(config_path.read_text(encoding="utf-8"))
                config["training"]["learning_rate"] = 2e-5
                write_canonical_json(config_path, config)
            second = workspace.build_plan(evidence_ref, "changed_training_plan_ref.json")

            self.assertNotEqual(first["artifact_id"], second["artifact_id"])
            self.assertEqual(
                first_target_bytes,
                {
                    path.relative_to(Path(first["target_path"])).as_posix(): path.read_bytes()
                    for path in Path(first["target_path"]).rglob("*")
                    if path.is_file()
                },
            )
            self.assertTrue(
                validate_training_plan(
                    workspace.refs / "training_plan_ref.json",
                    workspace_root=workspace.root,
                    schema_path=REPOSITORY_ROOT / "schemas/stage1_training_plan_v1.schema.json",
                    tokenizer=workspace.tokenizer,
                )["valid"]
            )

    def test_plan_rejects_wrong_context_and_tampered_evidence_dependency(self):
        with tempfile.TemporaryDirectory() as temporary:
            workspace = SyntheticTrainingWorkspace(Path(temporary))
            workspace.build_evidence()
            evidence_ref = workspace.refs / "training_evidence_ref.json"
            other_context_ref = workspace._simple_artifact("context", "ctx-", "other-context")
            with self.assertRaises(TrainingPlanError):
                workspace.build_plan(evidence_ref, context_ref=other_context_ref)

            plan_ref = workspace.build_plan(evidence_ref)
            target = Path(plan_ref["target_path"])
            dependency = json.loads(
                (target / "training_evidence_ref.json").read_text(encoding="utf-8")
            )
            dependency["payload_manifest_sha256"] = "f" * 64
            write_canonical_json(target / "training_evidence_ref.json", dependency)
            workspace.refresh_manifest_and_locator(workspace.refs / "training_plan_ref.json")
            with self.assertRaises(TrainingPlanError):
                validate_training_plan(
                    workspace.refs / "training_plan_ref.json",
                    workspace_root=workspace.root,
                    schema_path=REPOSITORY_ROOT / "schemas/stage1_training_plan_v1.schema.json",
                    tokenizer=workspace.tokenizer,
                )

    def test_source_recipe_future_binding_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            workspace = SyntheticTrainingWorkspace(Path(temporary))
            workspace.build_evidence()
            spec = json.loads(workspace.source_spec.read_text(encoding="utf-8"))
            spec["schedule_ref"] = "/tmp/future.json"
            write_canonical_json(workspace.source_spec, spec)
            with self.assertRaises(TrainingPlanError):
                workspace.build_plan(workspace.refs / "training_evidence_ref.json")

    def test_lifecycle_collision_and_cli_validation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            parent = root / "targets"
            artifact_id = make_id("x-", "collision")
            target = parent / artifact_id
            staging = new_staging_directory(parent, artifact_id)
            write_canonical_json(staging / "value.json", {"value": 1})
            finalize_target_atomic(staging, target)
            conflicting = new_staging_directory(parent, artifact_id)
            write_canonical_json(conflicting / "value.json", {"value": 2})
            with self.assertRaises(TrainingArtifactError):
                finalize_target_atomic(conflicting, target)

        with tempfile.TemporaryDirectory() as temporary:
            workspace = SyntheticTrainingWorkspace(Path(temporary))
            workspace.build_evidence()
            workspace.build_plan(workspace.refs / "training_evidence_ref.json")
            with patch(
                "transformers.AutoTokenizer.from_pretrained",
                return_value=workspace.tokenizer,
            ) as tokenizer_constructor:
                self.assertEqual(
                    evidence_cli.main(
                        [
                            "validate-evidence",
                            "--training-evidence-ref",
                            str(workspace.refs / "training_evidence_ref.json"),
                            "--workspace-root",
                            str(workspace.root),
                        ]
                    ),
                    0,
                )
                self.assertEqual(
                    plan_cli.main(
                        [
                            "validate",
                            "--training-plan-ref",
                            str(workspace.refs / "training_plan_ref.json"),
                            "--workspace-root",
                            str(workspace.root),
                        ]
                    ),
                    0,
                )
            self.assertGreaterEqual(tokenizer_constructor.call_count, 2)
            for call in tokenizer_constructor.call_args_list:
                self.assertTrue(call.kwargs["local_files_only"])
                self.assertFalse(call.kwargs["trust_remote_code"])


if __name__ == "__main__":
    unittest.main()
