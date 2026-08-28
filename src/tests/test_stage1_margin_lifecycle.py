from __future__ import annotations

import copy
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

from data.build_context_manifest import build_prepared_context_artifact
from data.control_manifest import build_control_artifact
from data.generation_lifecycle import MODEL_RESOLUTION_SCHEMA, RegisteredModelHandle
from data.training_artifacts import (
    TrainingArtifactError,
    canonical_sha256,
    finalize_target_atomic,
    load_json,
    new_staging_directory,
    portable_dependency,
    resolve_locator_ref,
    validate_json_schema,
    write_bytes_atomic,
    write_canonical_json,
    write_locator_ref,
)
from metrics import stage1_artifacts as artifacts
from metrics import stage1_margin_lifecycle as lifecycle
from metrics.stage1_margin import score_inputs
from model.stage1_registry import (
    ResolvedModelSourceContract,
    inventory_regular_file_tree,
)
from tests.test_stage1_generation_lifecycle import control_ready_bundle


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
MODEL_KEY = "M_LD/seed-42"


class LifecycleTokenizer:
    pad_token_id = 0
    eos_token_id = 1

    @staticmethod
    def _ids(text: str) -> list[int]:
        return [2 + (ord(character) % 120) for character in text]

    def apply_chat_template(
        self,
        conversation,
        *,
        tokenize,
        add_generation_prompt,
        enable_thinking=False,
    ):
        assert tokenize is False
        assert add_generation_prompt is True
        assert enable_thinking is False
        return "\n".join(
            f"{row['role']}:{row['content']}" for row in conversation
        ) + "\nassistant:"

    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return self._ids(text)

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        assert add_special_tokens is False
        result = {"input_ids": self._ids(text)}
        if return_offsets_mapping:
            result["offset_mapping"] = [
                (index, index + 1) for index in range(len(text))
            ]
        return result


class DeterministicModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()), requires_grad=False)

    def forward(self, input_ids, attention_mask, position_ids):
        del attention_mask
        vocabulary = torch.arange(128, device=input_ids.device, dtype=torch.float32)
        logits = vocabulary.view(1, 1, -1).expand(
            input_ids.shape[0], input_ids.shape[1], -1
        )
        logits = logits + position_ids.unsqueeze(-1).float() * 0.001 + self.anchor
        return SimpleNamespace(logits=logits)


class DeterministicMarginExecutor:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.model = DeterministicModel()

    def descriptor(self):
        return {
            "executor_id": "engineering-deterministic/v1",
            "executor_revision": "stage1-margin-test-executor/v1",
            "backend": "local-huggingface-causal-lm",
            "scientific_eligible": False,
        }

    def score(self, inputs):
        return score_inputs(
            model=self.model,
            score_inputs=inputs,
            pad_token_id=self.tokenizer.pad_token_id,
        )


class CalibrationFixtureExecutor:
    def __init__(self, singleton_gold_shift: float):
        self.singleton_gold_shift = singleton_gold_shift

    def score(self, inputs):
        result = []
        for item in inputs:
            mean = (
                self.singleton_gold_shift
                if len(inputs) == 1 and item == "gold"
                else 0.0
            )
            result.append(
                {
                    "token_ids": [2],
                    "token_logprobs": [mean],
                    "mean_logprob": mean,
                }
            )
        return result


class FrozenScoreExecutor:
    """Fast score fixture with batch/singleton-identical auditable outputs."""

    @staticmethod
    def score(inputs):
        rows = []
        for item in inputs:
            token_ids = [item.response_ids[index] for index in item.response_token_indices]
            token_logprobs = [-float((token_id % 17) + 1) / 100.0 for token_id in token_ids]
            total = sum(token_logprobs)
            rows.append(
                {
                    "schema_version": "stage1-field-score/v1",
                    "token_ids": token_ids,
                    "token_logprobs": token_logprobs,
                    "token_count": len(token_ids),
                    "character_span": list(item.character_span),
                    "response_token_indices": list(item.response_token_indices),
                    "global_token_indices": list(item.global_token_indices),
                    "sum_logprob": total,
                    "mean_logprob": total / len(token_logprobs),
                    "left_boundary_crossing": item.left_boundary_crossing,
                    "right_boundary_crossing": item.right_boundary_crossing,
                    "span_mask_version": "minimal-overlap-cover/v1",
                    "segmentation_version": "separate-no-special-tokens/v1",
                }
            )
        return rows


def _simple_target(
    root: Path, *, kind: str, prefix: str, label: str
) -> tuple[dict[str, object], Path, dict[str, object], Path]:
    artifact_id = prefix + canonical_sha256({"label": label})
    parent = root / "artifacts" / f"{kind}_targets"
    target = parent / artifact_id
    staging = new_staging_directory(parent, artifact_id)
    write_canonical_json(staging / "fixture.json", {"label": label})
    payload_hash = finalize_target_atomic(staging, target)
    ref = root / "refs" / f"{label}.ref.json"
    locator = write_locator_ref(
        ref,
        artifact_kind=kind,
        artifact_id=artifact_id,
        target=target,
        payload_manifest_sha256=payload_hash,
    )
    return locator, target, portable_dependency(locator, target, root), ref


class MarginLifecycleIntegrationTests(unittest.TestCase):
    def test_margin_load_rejects_checkpoint_swap_and_restore_inside_constructor(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "sources/checkpoint-10"
            source.mkdir(parents=True)
            shard = source / "model.safetensors"
            write_bytes_atomic(shard, b"verified-margin-weights")
            write_canonical_json(source / "tokenizer.json", {"version": "1.0"})
            inventory = inventory_regular_file_tree(
                source,
                workspace_root=root,
                inventory_policy="all-regular-files/v1",
            )
            contract = ResolvedModelSourceContract(
                workspace_root=root,
                checkpoint_inventory=inventory,
                tokenizer_inventory=inventory,
                base_inventory=inventory,
            )
            handle = RegisteredModelHandle(
                schema_version=MODEL_RESOLUTION_SCHEMA,
                locator={},
                target=root,
                dependency={},
                registry_id="mreg-" + "a" * 64,
                model_key=MODEL_KEY,
                role="M_LD",
                seed=42,
                scientific_eligible=True,
                tokenizer_revision="qwen3-8b-stage1-v1",
                tokenizer_content_revision="tok-" + "b" * 64,
                tokenizer=LifecycleTokenizer(),
                checkpoint_format="full",
                checkpoint_path=source,
                base_model_path=source,
                tokenizer_path=source,
                model_artifact_id="mdl-" + "c" * 64,
                source_contract=contract,
            )
            scorer_profile = load_json(
                REPOSITORY_ROOT / "config/stage1/margin_scorer.json"
            )
            original = shard.read_bytes()
            fake_model = SimpleNamespace(eval=lambda: None)

            def load_after_transient_swap(*_args, **_kwargs):
                write_bytes_atomic(shard, b"transient-margin-weights")
                write_bytes_atomic(shard, original)
                return fake_model

            with mock.patch(
                "transformers.AutoModelForCausalLM.from_pretrained",
                side_effect=load_after_transient_swap,
            ), self.assertRaisesRegex(
                lifecycle.MarginLifecycleError, "changed during backend load"
            ):
                lifecycle.LocalHFMarginExecutor.from_registered_model(
                    handle, scorer_profile
                )

    @staticmethod
    def _sealed_margin_upstreams(root: Path, tokenizer: LifecycleTokenizer):
        records = []
        context_runners = {condition: [] for condition in ("C0", "CL", "CD", "CLD")}
        control_runners = {condition: [] for condition in ("PL", "PD")}
        for ordinal in range(25):
            query_id = str(100 + ordinal)
            content = f"测试内容{ordinal}"
            gold = [
                {
                    "target": f"对象{ordinal}",
                    "argument": f"仇恨论点{ordinal}",
                    "targeted_group": ["Racism"],
                    "hateful": "hate",
                }
            ]
            query = {
                "id": query_id,
                "content": content,
                "gold": gold,
                "content_sha256": artifacts._hash_text(content),
                "gold_sha256": canonical_sha256(gold),
            }
            record = {"query": query}
            record["record_sha256"] = canonical_sha256(record)
            records.append(record)
            for condition in (*context_runners, *control_runners):
                messages = [
                    {"role": "system", "content": f"system-{condition}"},
                    {"role": "user", "content": content},
                ]
                rendered = lifecycle.chat_prompt_text(messages, tokenizer)
                lineage = {
                    "chat_prompt_sha256": lifecycle.text_sha256(rendered),
                    "chat_prompt_tokens": lifecycle.token_count(rendered, tokenizer),
                }
                item = {
                    "id": query_id,
                    "messages_list": [messages],
                    (
                        "control_manifest"
                        if condition in control_runners
                        else "context_manifest"
                    ): lineage,
                }
                destination = (
                    control_runners if condition in control_runners else context_runners
                )
                destination[condition].append(item)

        context_id = "tctx-" + canonical_sha256({"fixture": "sealed-margin-context"})
        context_parent = root / "artifacts/test_contexts"
        context_target = context_parent / context_id
        staging = new_staging_directory(context_parent, context_id)
        artifacts._write_ordered_jsonl(staging / "context_manifest.test.jsonl", records)
        for condition, rows in context_runners.items():
            write_canonical_json(
                staging / "conditions/runner" / condition / "test.json", rows
            )
        context_hash = finalize_target_atomic(staging, context_target)
        context_ref = root / "refs/test-context.ref.json"
        context_locator = write_locator_ref(
            context_ref,
            artifact_kind="test-context",
            artifact_id=context_id,
            target=context_target,
            payload_manifest_sha256=context_hash,
        )
        context_dependency = portable_dependency(context_locator, context_target, root)

        control_id = "tctl-" + canonical_sha256({"fixture": "sealed-margin-control"})
        control_parent = root / "artifacts/test_controls"
        control_target = control_parent / control_id
        staging = new_staging_directory(control_parent, control_id)
        write_canonical_json(staging / "context_ref.json", context_dependency)
        for condition, rows in control_runners.items():
            write_canonical_json(
                staging / "conditions/runner" / condition / "test.json", rows
            )
        control_hash = finalize_target_atomic(staging, control_target)
        control_ref = root / "refs/test-control.ref.json"
        control_locator = write_locator_ref(
            control_ref,
            artifact_kind="test-control",
            artifact_id=control_id,
            target=control_target,
            payload_manifest_sha256=control_hash,
        )

        cf_id = "cf-" + canonical_sha256({"fixture": "sealed-margin-cf"})
        cf_parent = root / "artifacts/counterfactuals"
        cf_target = cf_parent / cf_id
        staging = new_staging_directory(cf_parent, cf_id)
        write_canonical_json(staging / "context_ref.json", context_dependency)
        cf_rows = []
        for record in records:
            query = record["query"]
            gold = query["gold"][0]
            replacements = {
                "target": f"中性{gold['target']}",
                "argument": f"中性{gold['argument']}",
                "targeted_group": ["Sexism"],
                "hateful": "non-hate",
            }
            for field in artifacts.FIELDS:
                candidate_id = "cf:v1:" + canonical_sha256(
                    {"id": query["id"], "field": field}
                )
                row = {
                    "query_id": query["id"],
                    "tuple_index": 0,
                    "field": field,
                    "context_record_sha256": record["record_sha256"],
                    "gold_sha256": query["gold_sha256"],
                    "gold_value": copy.deepcopy(gold[field]),
                    "construction_status": "ok",
                    "selected_cf_id": candidate_id,
                    "candidates": [
                        {
                            "candidate_id": candidate_id,
                            "value": copy.deepcopy(replacements[field]),
                        }
                    ],
                }
                row["record_sha256"] = canonical_sha256(row)
                cf_rows.append(row)
        artifacts._write_ordered_jsonl(staging / "cf_manifest.test.jsonl", cf_rows)
        cf_hash = finalize_target_atomic(staging, cf_target)
        cf_ref = root / "refs/test-cf.ref.json"
        write_locator_ref(
            cf_ref,
            artifact_kind="counterfactual",
            artifact_id=cf_id,
            target=cf_target,
            payload_manifest_sha256=cf_hash,
        )
        return context_ref, control_ref, cf_ref, context_dependency

    def test_formal_margin_rejects_injected_resolver_before_model_resolution(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registry_id = "mreg-" + "a" * 64
            registry_locator = {
                "schema_version": "stage1-locator-ref/v1",
                "artifact_kind": "stage1-model-registry",
                "artifact_id": registry_id,
                "target_path": str(root),
                "payload_manifest_sha256": "b" * 64,
            }
            registry_report = {
                "model_registry_id": registry_id,
                "registry_scope": "formal",
                "scientific_eligible": True,
                "models": [
                    {
                        "model_key": MODEL_KEY,
                        "model_dependency": {
                            "artifact_id": "mdl-" + "c" * 64
                        },
                    }
                ],
            }
            rogue_resolver = mock.Mock()
            with mock.patch.object(
                lifecycle,
                "validate_registry_ref",
                return_value=(registry_locator, registry_report, root),
            ), self.assertRaisesRegex(
                lifecycle.MarginLifecycleError, "injected executors/resolvers"
            ):
                lifecycle.build_margin_artifact(
                    model_registry_ref=root / "registry.ref.json",
                    model_key=MODEL_KEY,
                    context_ref=root / "context.ref.json",
                    control_ref=root / "control.ref.json",
                    cf_ref=root / "cf.ref.json",
                    scorer_profile=REPOSITORY_ROOT
                    / "config/stage1/margin_scorer.json",
                    write_ref=root / "margin.ref.json",
                    split="dev",
                    workspace_root=root,
                    model_resolver=rogue_resolver,
                )
            rogue_resolver.assert_not_called()

    def test_calibration_uses_floor_when_all_pair_margins_are_identical(self):
        summary, rows = lifecycle._batch_calibration(
            CalibrationFixtureExecutor(0.0),
            [
                (
                    ("gold", "counterfactual"),
                    {
                        "id": "1",
                        "condition": "C0",
                        "tuple_index": 0,
                        "field": "target",
                    },
                )
            ],
        )
        self.assertEqual(summary["observed_pair_count"], 1)
        self.assertFalse(summary["requested_pair_count_reached"])
        self.assertEqual(
            summary["frozen_tolerance"],
            lifecycle.CALIBRATION_TOLERANCE_FLOOR,
        )
        self.assertEqual(rows[0]["abs_margin_delta"], 0.0)

    def test_calibration_blocks_tolerance_above_hard_cap(self):
        with self.assertRaisesRegex(
            lifecycle.MarginLifecycleError, "above 5e-3"
        ):
            lifecycle._batch_calibration(
                CalibrationFixtureExecutor(0.003),
                [
                    (
                        ("gold", "counterfactual"),
                        {
                            "id": "1",
                            "condition": "C0",
                            "tuple_index": 0,
                            "field": "target",
                        },
                    )
                ],
            )

    def test_control_kind_matrix_rejects_both_dev_test_swaps(self):
        self.assertEqual(
            artifacts._control_kind_for_split_sealing(
                split="dev", sealing_status="unsealed-dev"
            ),
            "control",
        )
        self.assertEqual(
            artifacts._control_kind_for_split_sealing(
                split="test", sealing_status="sealed-test"
            ),
            "test-control",
        )
        for split, sealing_status, wrong_kind in (
            ("dev", "unsealed-dev", "test-control"),
            ("test", "sealed-test", "control"),
        ):
            with self.subTest(split=split), self.assertRaisesRegex(
                artifacts.Stage1ArtifactError, "control kind"
            ):
                artifacts._assert_control_dependency_kind(
                    {"artifact_kind": wrong_kind},
                    split=split,
                    sealing_status=sealing_status,
                    label="margin",
                )

    def test_sealed_test_control_margin_builds_and_deep_validates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tokenizer = LifecycleTokenizer()
            context_ref, control_ref, cf_ref, context_dependency = (
                self._sealed_margin_upstreams(root, tokenizer)
            )
            _, _, plan_dependency, _ = _simple_target(
                root,
                kind="training-plan",
                prefix="plan-",
                label="sealed-margin-plan",
            )
            _, _, model_dependency, _ = _simple_target(
                root,
                kind="stage1-model",
                prefix="mdl-",
                label="sealed-margin-model",
            )
            model_source = root / "sources/formal-margin-model"
            model_source.mkdir(parents=True)
            write_bytes_atomic(
                model_source / "model.safetensors", b"formal-margin-weights"
            )
            write_canonical_json(
                model_source / "tokenizer.json", {"version": "1.0"}
            )
            custom_tokenizer_code = model_source / "tokenization_custom.py"
            write_bytes_atomic(
                custom_tokenizer_code,
                b"class FormalMarginTokenizer: pass\n",
            )
            model_inventory = inventory_regular_file_tree(
                model_source,
                workspace_root=root,
                inventory_policy="all-regular-files/v1",
            )
            legacy_tokenizer_inventory = inventory_regular_file_tree(
                model_source,
                workspace_root=root,
                inventory_policy="tokenizer-files/v1",
            )
            source_contract = ResolvedModelSourceContract(
                workspace_root=root,
                checkpoint_inventory=model_inventory,
                tokenizer_inventory=legacy_tokenizer_inventory,
                base_inventory=model_inventory,
            )
            self.assertEqual(
                artifacts._margin_tokenizer_lease_sources(
                    source_contract, require_full_coverage=True
                ),
                ("tokenizer", "base"),
            )
            registry_locator, registry_target, registry_dependency, registry_ref = (
                _simple_target(
                    root,
                    kind="stage1-model-registry",
                    prefix="mreg-",
                    label="sealed-margin-registry",
                )
            )
            handle = RegisteredModelHandle(
                schema_version=MODEL_RESOLUTION_SCHEMA,
                locator=registry_locator,
                target=registry_target,
                dependency=registry_dependency,
                registry_id=registry_dependency["artifact_id"],
                model_key=MODEL_KEY,
                role="M_LD",
                seed=42,
                scientific_eligible=True,
                tokenizer_revision="qwen3-8b-stage1-v1",
                tokenizer_content_revision="tok-" + "b" * 64,
                tokenizer=tokenizer,
                checkpoint_format="full",
                checkpoint_path=model_source,
                base_model_path=model_source,
                tokenizer_path=model_source,
                model_artifact_id=model_dependency["artifact_id"],
                source_contract=source_contract,
            )
            registry_report = {
                "model_registry_id": registry_dependency["artifact_id"],
                "registry_scope": "formal",
                "scientific_eligible": True,
                "training_plan_dependency": plan_dependency,
                "models": [
                    {
                        "model_key": MODEL_KEY,
                        "role": "M_LD",
                        "seed": 42,
                        "model_dependency": model_dependency,
                        "scientific_eligible": True,
                    }
                ],
                "payload_manifest_sha256": registry_dependency[
                    "payload_manifest_sha256"
                ],
            }
            context_report = {
                "split": "test",
                "scientific_eligible": True,
                "budget": {"tokenizer_revision": handle.tokenizer_revision},
            }
            control_report = {"split": "test"}
            cf_report = {"split": "test", "scientific_eligible": True}
            executor = FrozenScoreExecutor()
            patches = (
                mock.patch.object(
                    lifecycle,
                    "validate_registry_ref",
                    return_value=(registry_locator, registry_report, registry_target),
                ),
                mock.patch.object(
                    lifecycle, "resolve_generation_model", return_value=handle
                ),
                mock.patch.object(
                    lifecycle, "validate_context_target", return_value=context_report
                ),
                mock.patch.object(
                    lifecycle, "validate_control_target", return_value=control_report
                ),
                mock.patch.object(
                    lifecycle, "validate_cf_target", return_value=cf_report
                ),
                mock.patch.object(
                    lifecycle.LocalHFMarginExecutor,
                    "from_registered_model",
                    return_value=executor,
                ),
                mock.patch.object(
                    lifecycle, "resolve_registered_model_dependency", return_value=handle
                ),
                mock.patch.object(
                    artifacts, "validate_registry_target", return_value=registry_report
                ),
                mock.patch.object(
                    artifacts,
                    "resolve_registered_model_dependency",
                    return_value=handle,
                ),
                mock.patch.object(
                    artifacts, "validate_cf_target", return_value=cf_report
                ),
                mock.patch(
                    "data.build_context_manifest.validate_context_target",
                    return_value=context_report,
                ),
                mock.patch(
                    "data.control_manifest.validate_control_target",
                    return_value=control_report,
                ),
                mock.patch(
                    "transformers.AutoTokenizer.from_pretrained",
                    return_value=tokenizer,
                ),
            )
            with patches[0], patches[1], patches[2], patches[3], patches[4], patches[
                5
            ], patches[6], patches[7], patches[8], patches[9], patches[10], patches[
                11
            ], patches[12] as tokenizer_constructor:
                locator = lifecycle.build_margin_artifact(
                    model_registry_ref=registry_ref,
                    model_key=MODEL_KEY,
                    context_ref=context_ref,
                    control_ref=control_ref,
                    cf_ref=cf_ref,
                    scorer_profile=REPOSITORY_ROOT
                    / "config/stage1/margin_scorer.json",
                    write_ref=root / "refs/sealed-margin.ref.json",
                    split="test",
                    sealed=True,
                    target_root=root / "artifacts",
                    workspace_root=root,
                )
                target = Path(locator["target_path"])
                report = artifacts.validate_margin_target(
                    target, workspace_root=root
                )
                self.assertEqual(report["split"], "test")
                self.assertEqual(report["sealing_status"], "sealed-test")
                self.assertEqual(
                    report["control_dependency"]["artifact_kind"], "test-control"
                )
                # Formal context/control replay owns its frozen tokenizer
                # source; the registered scoring-model tokenizer is not
                # constructed or injected into either validator.
                tokenizer_constructor.assert_not_called()

                original_custom_code = custom_tokenizer_code.read_bytes()

                def context_replay_with_transient_drift(*_args, **_kwargs):
                    write_bytes_atomic(
                        custom_tokenizer_code,
                        b"class DriftedTokenizer: pass\n",
                    )
                    write_bytes_atomic(custom_tokenizer_code, original_custom_code)
                    return context_report

                with mock.patch(
                    "data.build_context_manifest.validate_context_target",
                    side_effect=context_replay_with_transient_drift,
                ), self.assertRaisesRegex(
                    artifacts.Stage1ArtifactError,
                    "changed during backend load",
                ):
                    artifacts.validate_margin_target(target, workspace_root=root)

                _, _, wrong_control_dependency, _ = _simple_target(
                    root,
                    kind="control",
                    prefix="ctl-",
                    label="wrong-sealed-margin-control",
                )
                tampered_parent = root / "tampered"
                tampered_target = tampered_parent / "resealed-wrong-control-kind"
                staging = new_staging_directory(
                    tampered_parent, "resealed-wrong-control-kind"
                )
                shutil.copytree(target, staging, dirs_exist_ok=True)
                (staging / "payload_manifest.json").unlink()
                write_canonical_json(
                    staging / "control_ref.json", wrong_control_dependency
                )
                finalize_target_atomic(staging, tampered_target)
                with self.assertRaisesRegex(
                    artifacts.Stage1ArtifactError, "control kind"
                ):
                    artifacts.validate_margin_target(
                        tampered_target,
                        workspace_root=root,
                        require_directory_name=False,
                    )

    def test_registered_engineering_model_scores_and_publishes_complete_target(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tokenizer = LifecycleTokenizer()
            context_ref = root / "refs/context.ref.json"
            build_prepared_context_artifact(
                prepared_bundle=control_ready_bundle(),
                config=REPOSITORY_ROOT / "config/stage1/context_factorial.json",
                tokenizer=tokenizer,
                write_ref=context_ref,
                formal=False,
                target_root=root / "artifacts/contexts",
            )
            context_locator, context_target = resolve_locator_ref(
                context_ref, expected_kind="context"
            )
            context_dependency = portable_dependency(
                context_locator, context_target, root
            )
            control_ref = root / "refs/control.ref.json"
            build_control_artifact(
                config={
                    "schema_version": "stage1-control-config/v1",
                    "profile_name": "margin-lifecycle-test",
                    "tokenizer": {
                        "revision": "qwen3-8b-stage1-v1",
                        "logical_path": "models/base/Qwen3-8B",
                    },
                },
                context_ref=context_ref,
                write_ref=control_ref,
                split="dev",
                tokenizer=tokenizer,
                tokenizer_revision="qwen3-8b-stage1-v1",
                target_root=root / "artifacts/controls",
            )

            _, _, plan_dependency, _ = _simple_target(
                root,
                kind="training-plan",
                prefix="plan-",
                label="margin-plan",
            )
            _, _, model_dependency, _ = _simple_target(
                root,
                kind="stage1-model",
                prefix="mdl-",
                label="margin-model",
            )
            registry_locator, registry_target, registry_dependency, registry_ref = (
                _simple_target(
                    root,
                    kind="stage1-model-registry",
                    prefix="mreg-",
                    label="margin-registry",
                )
            )
            handle = RegisteredModelHandle(
                schema_version=MODEL_RESOLUTION_SCHEMA,
                locator=registry_locator,
                target=registry_target,
                dependency=registry_dependency,
                registry_id=registry_dependency["artifact_id"],
                model_key=MODEL_KEY,
                role="M_LD",
                seed=42,
                scientific_eligible=False,
                tokenizer_revision="qwen3-8b-stage1-v1",
                tokenizer_content_revision="tok-" + "b" * 64,
                tokenizer=tokenizer,
                checkpoint_format="full",
                checkpoint_path=root,
                base_model_path=root,
                tokenizer_path=root,
                model_artifact_id=model_dependency["artifact_id"],
            )
            registry_report = {
                "model_registry_id": registry_dependency["artifact_id"],
                "registry_scope": "engineering-smoke",
                "scientific_eligible": False,
                "training_plan_dependency": plan_dependency,
                "models": [
                    {
                        "model_key": MODEL_KEY,
                        "role": "M_LD",
                        "seed": 42,
                        "model_dependency": model_dependency,
                        "scientific_eligible": False,
                    }
                ],
                "payload_manifest_sha256": registry_dependency[
                    "payload_manifest_sha256"
                ],
            }

            frame = artifacts._context_frame(context_target, split="dev")
            cf_id = "cf-" + canonical_sha256({"fixture": "margin-final-cf"})
            cf_parent = root / "artifacts/counterfactuals"
            cf_target = cf_parent / cf_id
            cf_staging = new_staging_directory(cf_parent, cf_id)
            write_canonical_json(cf_staging / "context_ref.json", context_dependency)
            replacement = {
                "target": "中性对象",
                "argument": "中性论点",
                "targeted_group": ["Sexism"],
                "hateful": "non-hate",
            }
            cf_rows = []
            for query in frame:
                for tuple_index, gold in enumerate(query["gold"]):
                    for field in artifacts.FIELDS:
                        selected_cf_id = "cf:v1:" + canonical_sha256(
                            {
                                "id": query["id"],
                                "tuple_index": tuple_index,
                                "field": field,
                            }
                        )
                        row = {
                            "query_id": query["id"],
                            "tuple_index": tuple_index,
                            "field": field,
                            "context_record_sha256": query[
                                "context_record_sha256"
                            ],
                            "gold_sha256": query["gold_sha256"],
                            "gold_value": copy.deepcopy(gold[field]),
                            "construction_status": "ok",
                            "selected_cf_id": selected_cf_id,
                            "candidates": [
                                {
                                    "candidate_id": selected_cf_id,
                                    "value": copy.deepcopy(replacement[field]),
                                }
                            ],
                        }
                        row["record_sha256"] = canonical_sha256(row)
                        cf_rows.append(row)
            artifacts._write_ordered_jsonl(
                cf_staging / "cf_manifest.dev.jsonl", cf_rows
            )
            cf_hash = finalize_target_atomic(cf_staging, cf_target)
            cf_ref = root / "refs/cf.ref.json"
            write_locator_ref(
                cf_ref,
                artifact_kind="counterfactual",
                artifact_id=cf_id,
                target=cf_target,
                payload_manifest_sha256=cf_hash,
            )

            executor = DeterministicMarginExecutor(tokenizer)
            upstream_report = {"split": "dev", "scientific_eligible": False}
            with mock.patch.object(
                lifecycle,
                "validate_registry_ref",
                return_value=(registry_locator, registry_report, registry_target),
            ), mock.patch.object(
                lifecycle, "validate_cf_target", return_value=upstream_report
            ), mock.patch.object(
                lifecycle,
                "resolve_registered_model_dependency",
                return_value=handle,
            ), mock.patch.object(
                artifacts,
                "validate_registry_target",
                return_value=registry_report,
            ), mock.patch.object(
                artifacts, "validate_cf_target", return_value=upstream_report
            ):
                locator = lifecycle.build_margin_artifact(
                    model_registry_ref=registry_ref,
                    model_key=MODEL_KEY,
                    context_ref=context_ref,
                    control_ref=control_ref,
                    cf_ref=cf_ref,
                    scorer_profile=REPOSITORY_ROOT
                    / "config/stage1/margin_scorer.json",
                    write_ref=root / "refs/margin.ref.json",
                    split="dev",
                    target_root=root / "artifacts",
                    workspace_root=root,
                    executor=executor,
                    model_resolver=lambda **_kwargs: handle,
                )
                target = Path(locator["target_path"])
                report = artifacts.validate_margin_target(
                    target,
                    workspace_root=root,
                    model_dependency_resolver=lambda **_kwargs: handle,
                    tokenizer_loader=lambda _path: tokenizer,
                )

            formal_registry_report = copy.deepcopy(registry_report)
            formal_registry_report["registry_scope"] = "formal"
            formal_registry_report["scientific_eligible"] = True
            resolver_callback = mock.Mock(return_value=handle)
            tokenizer_callback = mock.Mock(return_value=tokenizer)
            with mock.patch.object(
                artifacts,
                "validate_registry_target",
                return_value=formal_registry_report,
            ), self.assertRaisesRegex(
                artifacts.Stage1ArtifactError,
                "formal margin validation forbids injected",
            ):
                artifacts.validate_margin_target(
                    target,
                    workspace_root=root,
                    model_dependency_resolver=resolver_callback,
                    tokenizer_loader=tokenizer_callback,
                )
            resolver_callback.assert_not_called()
            tokenizer_callback.assert_not_called()

            self.assertEqual(report["query_ids"], ["10"])
            self.assertTrue(
                all(report["field_masks"][field] for field in artifacts.FIELDS)
            )
            self.assertEqual(
                {path.name for path in (target / "tuple_scores").iterdir()},
                {f"{condition}.jsonl" for condition in artifacts.CONDITIONS},
            )
            self.assertTrue((target / "query_contrasts.jsonl").is_file())
            self.assertTrue((target / "runtime_contract.json").is_file())
            profile = load_json(target / "scorer_profile.resolved.json")
            runtime_contract = load_json(target / "runtime_contract.json")
            meta = load_json(target / "margin.meta.json")
            expected_batch_contract = lifecycle._batch_contract(profile)
            self.assertEqual(profile["runtime"]["batch_unit"], "gold-cf-pair")
            self.assertEqual(profile["runtime"]["batch_size"], 1)
            self.assertEqual(
                runtime_contract["batch_contract"], expected_batch_contract
            )
            self.assertEqual(meta["batch_contract"], expected_batch_contract)

            for payload, schema_name in (
                (
                    runtime_contract,
                    "stage1_margin_runtime_contract_v1.schema.json",
                ),
                (meta, "stage1_margin_artifact_v1.schema.json"),
            ):
                tampered = copy.deepcopy(payload)
                tampered["batch_contract"]["batch_unit"] = "sequence"
                with self.subTest(schema_name=schema_name), self.assertRaises(
                    TrainingArtifactError
                ):
                    validate_json_schema(
                        tampered, REPOSITORY_ROOT / "schemas" / schema_name
                    )


if __name__ == "__main__":
    unittest.main()
