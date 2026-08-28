from __future__ import annotations

import copy
import json
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch

import jsonschema

from data.training_artifacts import (
    TrainingArtifactError,
    build_payload_manifest,
    canonical_sha256,
    validate_payload_manifest,
    write_canonical_json,
    write_canonical_jsonl,
    write_locator_ref,
)
import data.training_schedule as training_schedule_module
from data.training_schedule import (
    DEMO_ORDER_POLICY,
    DROP_RNG_POLICY,
    OVERFLOW_POLICY,
    SCHEDULE_RECORD_SCHEMA_VERSION,
    TrainingScheduleError,
    build_schedule_record,
    build_training_schedule,
    load_model_epoch_records,
    stateless_demo_order,
    stateless_source_use,
    validate_evidence_record_for_schedule,
    validate_plan_slots_for_schedule,
    validate_training_schedule,
)
from tests.test_stage1_training_artifacts import (
    SyntheticTrainingWorkspace,
    deterministic_context_tokenizer_constructor,
    deterministic_formal_score_replayer,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


class FakeTokenizer:
    eos_token_id = 7

    def apply_chat_template(
        self,
        conversation,
        *,
        tokenize,
        add_generation_prompt,
        enable_thinking=False,
    ):
        self.last_enable_thinking = enable_thinking
        text = "<chat>" + json.dumps(
            conversation, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        if add_generation_prompt:
            text += "<assistant>"
        return [ord(value) % 251 for value in text] if tokenize else text

    def encode(self, text, add_special_tokens=False):
        return [ord(value) % 251 for value in text]

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": self.encode(text, add_special_tokens=add_special_tokens)}


def _record(*, maximum: int = 10000) -> dict:
    query = {
        "id": "10",
        "partition": "fit",
        "content": "待分析文本",
        "content_sha256": "a" * 64,
        "gold": [],
        "gold_text": "[]",
        "gold_sha256": "b" * 64,
        "actual_gold_tokens": 2,
    }
    record = {
        "schema_version": "stage1-training-evidence-record/v1",
        "training_evidence_build_id": "tev-" + "c" * 64,
        "query": query,
        "lexicon_evidence": {
            "ids": ["lex-a", "lex-b"],
            "items": [
                {
                    "lexicon_id": "lex-a",
                    "rendered_block": "词典甲",
                    "rendered_block_sha256": "",
                },
                {
                    "lexicon_id": "lex-b",
                    "rendered_block": "词典乙",
                    "rendered_block_sha256": "",
                },
            ],
        },
        "demo_evidence": {
            "ids": ["demo-a", "demo-b", "demo-c"],
            "items": [
                {
                    "demo_id": demo_id,
                    "source_record_id": str(index),
                    "rendered_block": f"示例{index}",
                    "rendered_block_sha256": "",
                }
                for index, demo_id in enumerate(("demo-a", "demo-b", "demo-c"), 1)
            ],
        },
        "rendering": {
            "system_prompt": "system",
            "user_prompt_template": "词典：{lexicons}\n示例：{examples}\n文本：{text}",
            "thinking_mode": False,
        },
        "budget": {
            "max_sequence_tokens": maximum,
            "completion_reserve_tokens": 256,
            "eos_tokens": 1,
            "overflow_policy": OVERFLOW_POLICY,
        },
        "source_context_record_sha256": "d" * 64,
    }
    from data.context_manifest import text_sha256

    for section in ("lexicon_evidence", "demo_evidence"):
        for item in record[section]["items"]:
            item["rendered_block_sha256"] = text_sha256(item["rendered_block"])
    record["record_sha256"] = canonical_sha256(
        {
            key: value
            for key, value in record.items()
            if key != "training_evidence_build_id"
        }
    )
    return record


def _slot(role: str, seed: int = 42, epochs: int = 5) -> dict:
    probability = 0.0 if role == "M_LD" else 0.5
    policy = {
        "lexicon_dropout_probability": probability,
        "demonstration_dropout_probability": probability,
        "dropout_independent": True,
        "order_policy": DEMO_ORDER_POLICY,
        "matched_permutation_across_roles": True,
    }
    if role == "M_drop":
        policy["dropout_rng_policy"] = DROP_RNG_POLICY
    return {
        "model_key": f"{role}/seed-{seed}",
        "role": role,
        "seed": seed,
        "training_required": True,
        "train_config_logical_path": f"config/{role}.json",
        "train_config_sha256": "e" * 64,
        "train_config_resolved": {
            "schema_version": "stage1-train-config/v1",
            "model_role": role,
            "data": {
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
            "context_policy": policy,
        },
        "epochs": epochs,
        "final_checkpoint_rule": "best-eval-loss-threshold-earliest/v1",
    }


@patch(
    "data.build_context_manifest._default_formal_score_replayer",
    new=deterministic_formal_score_replayer,
)
@patch(
    "data.build_context_manifest._construct_formal_tokenizer",
    new=deterministic_context_tokenizer_constructor,
)
class TrainingScheduleTests(unittest.TestCase):
    def test_demo_order_is_stateless_and_input_order_independent(self):
        ids = ["demo-c", "demo-a", "demo-b"]
        first = stateless_demo_order(ids, seed=42, epoch=3, query_id="10")
        second = stateless_demo_order(reversed(ids), seed=42, epoch=3, query_id="10")
        self.assertEqual(first, second)
        self.assertCountEqual(first, ids)
        self.assertNotEqual(
            first,
            stateless_demo_order(ids, seed=42, epoch=4, query_id="10"),
        )

    def test_dropout_streams_are_deterministic_independent_and_half_probability(self):
        draws = [
            (
                stateless_source_use(seed=42, epoch=1, query_id=str(index), source="lexicon"),
                stateless_source_use(seed=42, epoch=1, query_id=str(index), source="demo"),
            )
            for index in range(500)
        ]
        self.assertEqual(len(set(draws)), 4)
        for source_index in (0, 1):
            kept = sum(draw[source_index] for draw in draws)
            self.assertGreater(kept, 200)
            self.assertLess(kept, 300)
        self.assertEqual(
            draws[17][0],
            stateless_source_use(seed=42, epoch=1, query_id="17", source="lexicon"),
        )
        with self.assertRaisesRegex(TrainingScheduleError, "exactly 0.5"):
            stateless_source_use(
                seed=42, epoch=1, query_id="17", source="lexicon", probability=0.4
            )

    def test_roles_share_order_and_only_m_drop_uses_masks(self):
        tokenizer = FakeTokenizer()
        evidence = _record()
        ld = build_schedule_record(
            evidence,
            _slot("M_LD"),
            epoch=2,
            query_ordinal=0,
            schedule_build_id="sch-" + "f" * 64,
            tokenizer=tokenizer,
        )
        drop = build_schedule_record(
            evidence,
            _slot("M_drop"),
            epoch=2,
            query_ordinal=0,
            schedule_build_id="sch-" + "f" * 64,
            tokenizer=tokenizer,
        )
        self.assertEqual(ld["ordered_demo_ids"], drop["ordered_demo_ids"])
        self.assertTrue(ld["use_lexicon"])
        self.assertTrue(ld["use_demos"])
        self.assertEqual(
            (drop["use_lexicon"], drop["use_demos"]),
            (
                stateless_source_use(seed=42, epoch=2, query_id="10", source="lexicon"),
                stateless_source_use(seed=42, epoch=2, query_id="10", source="demo"),
            ),
        )
        self.assertEqual(ld["schema_version"], SCHEDULE_RECORD_SCHEMA_VERSION)
        self.assertFalse(tokenizer.last_enable_thinking)
        self.assertEqual(ld["record_sha256"], canonical_sha256({k: v for k, v in ld.items() if k != "record_sha256"}))

    def test_full_prompt_gold_eos_preflight_fails_closed(self):
        record = _record(maximum=260)
        from data.context_manifest import text_sha256

        record["lexicon_evidence"]["items"][0]["rendered_block"] = "超" * 500
        record["lexicon_evidence"]["items"][0]["rendered_block_sha256"] = text_sha256(
            record["lexicon_evidence"]["items"][0]["rendered_block"]
        )
        record["record_sha256"] = canonical_sha256(
            {
                key: value
                for key, value in record.items()
                if key not in {"training_evidence_build_id", "record_sha256"}
            }
        )
        with self.assertRaisesRegex(TrainingScheduleError, "schedule overflow"):
            build_schedule_record(
                record,
                _slot("M_LD"),
                epoch=1,
                query_ordinal=0,
                schedule_build_id="sch-" + "f" * 64,
                tokenizer=FakeTokenizer(),
            )

    def test_calibration_presentation_is_fixed_to_epoch_one_wire(self):
        evidence = _record()
        evidence["query"]["partition"] = "calibration"
        evidence["record_sha256"] = canonical_sha256(
            {
                key: value
                for key, value in evidence.items()
                if key not in {"training_evidence_build_id", "record_sha256"}
            }
        )
        first = build_schedule_record(
            evidence,
            _slot("M_drop"),
            epoch=1,
            query_ordinal=0,
            schedule_build_id="sch-" + "f" * 64,
            tokenizer=FakeTokenizer(),
        )
        later = build_schedule_record(
            evidence,
            _slot("M_drop"),
            epoch=4,
            query_ordinal=0,
            schedule_build_id="sch-" + "f" * 64,
            tokenizer=FakeTokenizer(),
        )
        self.assertEqual(first["partition"], "calibration")
        self.assertEqual(later["presentation_epoch"], 1)
        for key in (
            "ordered_demo_ids",
            "use_lexicon",
            "use_demos",
            "instruction",
            "input",
            "rendered_prompt_sha256",
            "rendered_prompt_tokens",
            "sequence_tokens",
        ):
            self.assertEqual(first[key], later[key])

    def test_evidence_tamper_is_detected(self):
        record = _record()
        record["query"]["content"] = "tampered"
        with self.assertRaisesRegex(TrainingScheduleError, "record hash mismatch"):
            validate_evidence_record_for_schedule(record)

    def test_formal_plan_requires_matched_role_pairs(self):
        plan = {
            "schema_version": "stage1-training-plan/v1",
            "scope": "formal",
            "scientific_eligible": True,
            "ordered_model_slots": [_slot("M_LD"), _slot("M_drop")],
            "order_dropout_rng_policy": {
                "order": DEMO_ORDER_POLICY,
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
                    "rng": DROP_RNG_POLICY,
                },
            },
        }
        self.assertEqual(len(validate_plan_slots_for_schedule(plan)), 2)
        plan["ordered_model_slots"] = [_slot("M_LD")]
        with self.assertRaisesRegex(TrainingScheduleError, "exactly M_LD and M_drop"):
            validate_plan_slots_for_schedule(plan)

    def test_schedule_meta_schema_is_valid(self):
        schema_path = REPOSITORY_ROOT / "schemas" / "stage1_training_schedule_v1.schema.json"
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        jsonschema.Draft202012Validator.check_schema(schema)

    def test_resealed_schedule_tamper_is_rejected_without_explicit_tokenizer(self):
        """``None`` must load the frozen tokenizer and still exact-replay rows."""

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()

            def dependency(kind: str, artifact_id: str) -> tuple[Path, dict]:
                target = root / "artifacts" / kind / artifact_id
                target.mkdir(parents=True)
                write_canonical_json(
                    target / "artifact.json",
                    {"artifact_kind": kind, "artifact_id": artifact_id},
                )
                write_canonical_json(
                    target / "payload_manifest.json",
                    build_payload_manifest(target),
                )
                return target, {
                    "schema_version": "stage1-dependency-ref/v1",
                    "artifact_kind": kind,
                    "artifact_id": artifact_id,
                    "payload_manifest_sha256": validate_payload_manifest(target),
                    "logical_repo_path": target.relative_to(root).as_posix(),
                }

            plan_target, plan_dependency = dependency(
                "training-plan", "tpl-" + "1" * 64
            )
            evidence_target, evidence_dependency = dependency(
                "training-evidence", "tevd-" + "2" * 64
            )
            partition_target, partition_dependency = dependency(
                "train-partition", "tpart-" + "3" * 64
            )
            _base_target, base_model_dependency = dependency(
                "stage1-model", "mdl-" + "6" * 64
            )
            # The exact partition payload is supplied to the schedule validator
            # after the semantic target validator has authenticated this fixture.
            (partition_target / "payload_manifest.json").unlink()
            write_canonical_jsonl(
                partition_target / "partition.jsonl",
                [{"query_id": "10", "partition": "fit"}],
                key="query_id",
            )
            write_canonical_json(
                partition_target / "payload_manifest.json",
                build_payload_manifest(partition_target),
            )
            partition_dependency["payload_manifest_sha256"] = (
                validate_payload_manifest(partition_target)
            )

            slots = [_slot("M_LD", epochs=1), _slot("M_drop", epochs=1)]
            plan = {
                "schema_version": "stage1-training-plan/v1",
                "scope": "formal",
                "scientific_eligible": True,
                "ordered_model_slots": slots,
                "training_evidence_dependency": evidence_dependency,
                "train_partition_dependency": partition_dependency,
                "base_model_dependency": base_model_dependency,
                "order_dropout_rng_policy": {
                    "order": DEMO_ORDER_POLICY,
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
                        "rng": DROP_RNG_POLICY,
                    },
                },
            }
            evidence_record = _record()
            tokenizer = FakeTokenizer()
            tokenizer_revision = "fixture-tokenizer/v1"
            id_inputs = training_schedule_module._schedule_id_inputs(
                training_plan_dependency=plan_dependency,
                training_evidence_dependency=evidence_dependency,
                train_partition_dependency=partition_dependency,
                renderer_revision=training_schedule_module.RENDERER_REVISION,
                tokenizer_revision=tokenizer_revision,
                schedule_builder_code_sha256="4" * 64,
            )
            schedule_id = "sch-" + canonical_sha256(id_inputs)
            target = root / "schedules" / schedule_id
            target.mkdir(parents=True)
            write_canonical_json(
                target / "config.resolved.json",
                training_schedule_module._resolved_schedule_config(
                    tokenizer_revision
                ),
            )
            write_canonical_json(target / "training_plan_ref.json", plan_dependency)
            write_canonical_json(
                target / "training_evidence_ref.json", evidence_dependency
            )
            write_canonical_json(
                target / "train_partition_ref.json", partition_dependency
            )

            slot_registry = []
            all_hashes = []
            for slot_index, slot in enumerate(slots):
                row = build_schedule_record(
                    evidence_record,
                    slot,
                    epoch=1,
                    query_ordinal=0,
                    schedule_build_id=schedule_id,
                    tokenizer=tokenizer,
                )
                if slot_index == 0:
                    # Re-hash every enclosing layer so only exact tokenizer
                    # replay—not ordinary wire/digest validation—can reject it.
                    row["rendered_prompt_sha256"] = "5" * 64
                    row["record_sha256"] = canonical_sha256(
                        {
                            key: value
                            for key, value in row.items()
                            if key != "record_sha256"
                        }
                    )
                relative = (
                    Path("schedules")
                    / slot["role"]
                    / f"seed-{slot['seed']}"
                    / "epoch-1.jsonl"
                )
                write_canonical_jsonl(
                    target / relative,
                    [row],
                    key="query_ordinal",
                    numeric_key=True,
                )
                epoch_digest = canonical_sha256([row["record_sha256"]])
                slot_digest = canonical_sha256([epoch_digest])
                slot_meta_relative = relative.parent / "schedule.meta.json"
                write_canonical_json(
                    target / slot_meta_relative,
                    {
                        "schema_version": "stage1-training-schedule-slot/v1",
                        "schedule_build_id": schedule_id,
                        "model_key": slot["model_key"],
                        "role": slot["role"],
                        "seed": slot["seed"],
                        "epochs": 1,
                        "query_count_per_epoch": 1,
                        "epoch_registry": [
                            {
                                "epoch": 1,
                                "relative_path": relative.as_posix(),
                                "record_count": 1,
                                "records_sha256": epoch_digest,
                            }
                        ],
                        "slot_records_sha256": slot_digest,
                    },
                )
                slot_registry.append(
                    {
                        "model_key": slot["model_key"],
                        "role": slot["role"],
                        "seed": slot["seed"],
                        "epochs": 1,
                        "relative_meta_path": slot_meta_relative.as_posix(),
                        "slot_records_sha256": slot_digest,
                    }
                )
                all_hashes.append(row["record_sha256"])

            meta = {
                "schema_version": "stage1-training-schedule/v1",
                "schedule_build_id": schedule_id,
                "training_plan_dependency": plan_dependency,
                "training_evidence_dependency": evidence_dependency,
                "train_partition_dependency": partition_dependency,
                "renderer_revision": training_schedule_module.RENDERER_REVISION,
                "tokenizer_revision": tokenizer_revision,
                "rng_policy": training_schedule_module.SCHEDULE_RNG_POLICY,
                "slot_registry": slot_registry,
                "total_record_count": len(all_hashes),
                "all_schedule_records_sha256": canonical_sha256(all_hashes),
                "preflight_status": "complete-pass",
                "id_inputs": id_inputs,
            }
            write_canonical_json(target / "schedule.meta.json", meta)
            write_canonical_json(
                target / "provenance.json",
                {
                    "schema_version": "stage1-training-schedule-provenance/v1",
                    "schedule_build_id": schedule_id,
                    "schedule_builder_code_sha256": "4" * 64,
                    "id_inputs": id_inputs,
                },
            )
            write_canonical_json(
                target / "payload_manifest.json", build_payload_manifest(target)
            )
            schedule_ref = root / "schedule_ref.json"
            write_locator_ref(
                schedule_ref,
                artifact_kind="training-schedule",
                artifact_id=schedule_id,
                target=target,
                payload_manifest_sha256=validate_payload_manifest(target),
            )
            evidence_meta = {
                "renderer_revision": training_schedule_module.RENDERER_REVISION,
                "tokenizer_revision": tokenizer_revision,
                "train_partition_dependency": partition_dependency,
                "base_model_dependency": plan["base_model_dependency"],
            }

            with (
                patch(
                    "data.training_evidence.frozen_evidence_tokenizer_lease",
                    return_value=nullcontext(tokenizer),
                ) as frozen_lease,
                patch(
                    "data.training_plan.validate_training_plan_target",
                    return_value=plan,
                ),
                patch(
                    "data.training_schedule._load_evidence_target",
                    return_value=(evidence_meta, [evidence_record]),
                ),
                patch(
                    "data.train_partition.validate_train_partition_target",
                    return_value={"partition_dependency": partition_dependency},
                ),
            ):
                with self.assertRaisesRegex(
                    TrainingScheduleError, "cannot be re-rendered exactly"
                ):
                    validate_training_schedule(schedule_ref=schedule_ref)
            frozen_lease.assert_called_once_with(
                evidence_target, workspace_root=root, tokenizer=None
            )

    def test_complete_content_addressed_schedule_lifecycle(self):
        with tempfile.TemporaryDirectory() as temporary:
            workspace = SyntheticTrainingWorkspace(Path(temporary))
            workspace.build_evidence()
            workspace.build_plan(workspace.refs / "training_evidence_ref.json")
            first = build_training_schedule(
                training_plan_ref=workspace.refs / "training_plan_ref.json",
                training_evidence_ref=workspace.refs / "training_evidence_ref.json",
                tokenizer=workspace.tokenizer,
                write_ref=workspace.refs / "schedule_ref.json",
                workspace_root=workspace.root,
            )
            second = build_training_schedule(
                training_plan_ref=workspace.refs / "training_plan_ref.json",
                training_evidence_ref=workspace.refs / "training_evidence_ref.json",
                tokenizer=workspace.tokenizer,
                write_ref=workspace.refs / "schedule_second_ref.json",
                workspace_root=workspace.root,
            )
            self.assertEqual(first["artifact_id"], second["artifact_id"])
            self.assertEqual(
                first["payload_manifest_sha256"], second["payload_manifest_sha256"]
            )
            report = validate_training_schedule(
                schedule_ref=workspace.refs / "schedule_ref.json",
                tokenizer=workspace.tokenizer,
            )
            self.assertEqual(report["slot_count"], 6)
            # Production-semantic fixture: 5,781 queries × 6 model slots ×
            # 5 maximum epochs.  The previous 30-row expectation belonged to
            # the retired one-query pseudo-formal fixture.
            self.assertEqual(report["record_count"], 173_430)
            target = Path(first["target_path"])
            ld_epoch_one = load_model_epoch_records(
                target, model_key="M_LD/seed-42", epoch=1
            )
            drop_epoch_one = load_model_epoch_records(
                target, model_key="M_drop/seed-42", epoch=1
            )
            ld_epoch_one_by_id = {row["query_id"]: row for row in ld_epoch_one}
            drop_epoch_one_by_id = {
                row["query_id"]: row for row in drop_epoch_one
            }
            calibration_wire_fields = (
                "ordered_demo_ids",
                "use_lexicon",
                "use_demos",
                "instruction",
                "input",
                "rendered_prompt_sha256",
                "rendered_prompt_tokens",
                "sequence_tokens",
            )
            observed_fit_dropout_change = False
            for epoch in range(1, 6):
                ld_rows = (
                    ld_epoch_one
                    if epoch == 1
                    else load_model_epoch_records(
                        target, model_key="M_LD/seed-42", epoch=epoch
                    )
                )
                drop_rows = (
                    drop_epoch_one
                    if epoch == 1
                    else load_model_epoch_records(
                        target, model_key="M_drop/seed-42", epoch=epoch
                    )
                )
                self.assertEqual(
                    [row["query_id"] for row in ld_rows],
                    [row["query_id"] for row in drop_rows],
                )
                for ld, drop in zip(ld_rows, drop_rows, strict=True):
                    query_id = ld["query_id"]
                    self.assertEqual(ld["partition"], drop["partition"])
                    self.assertEqual(
                        ld["ordered_demo_ids"], drop["ordered_demo_ids"]
                    )
                    self.assertTrue(ld["use_lexicon"])
                    self.assertTrue(ld["use_demos"])
                    if ld["partition"] == "calibration":
                        self.assertEqual(ld["presentation_epoch"], 1)
                        self.assertEqual(drop["presentation_epoch"], 1)
                        for field in calibration_wire_fields:
                            self.assertEqual(
                                ld[field], ld_epoch_one_by_id[query_id][field]
                            )
                            self.assertEqual(
                                drop[field], drop_epoch_one_by_id[query_id][field]
                            )
                        continue

                    self.assertEqual(ld["presentation_epoch"], epoch)
                    self.assertEqual(drop["presentation_epoch"], epoch)
                    expected_order = stateless_demo_order(
                        ld_epoch_one_by_id[query_id]["ordered_demo_ids"],
                        seed=42,
                        epoch=epoch,
                        query_id=query_id,
                    )
                    self.assertEqual(ld["ordered_demo_ids"], expected_order)
                    self.assertEqual(
                        drop["use_lexicon"],
                        stateless_source_use(
                            seed=42,
                            epoch=epoch,
                            query_id=query_id,
                            source="lexicon",
                        ),
                    )
                    self.assertEqual(
                        drop["use_demos"],
                        stateless_source_use(
                            seed=42,
                            epoch=epoch,
                            query_id=query_id,
                            source="demo",
                        ),
                    )
                    if epoch > 1 and (
                        drop["use_lexicon"], drop["use_demos"]
                    ) != (
                        drop_epoch_one_by_id[query_id]["use_lexicon"],
                        drop_epoch_one_by_id[query_id]["use_demos"],
                    ):
                        observed_fit_dropout_change = True
            self.assertTrue(observed_fit_dropout_change)

    def test_schedule_rejects_wrong_evidence_revision_and_missing_epoch(self):
        with tempfile.TemporaryDirectory() as temporary:
            workspace = SyntheticTrainingWorkspace(Path(temporary))
            workspace.build_evidence()
            workspace.build_plan(workspace.refs / "training_evidence_ref.json")
            wrong_evidence = workspace._simple_artifact(
                "training-evidence", "tevd-", "wrong-training-evidence"
            )
            with self.assertRaisesRegex(TrainingScheduleError, "does not match"):
                build_training_schedule(
                    training_plan_ref=workspace.refs / "training_plan_ref.json",
                    training_evidence_ref=wrong_evidence,
                    tokenizer=workspace.tokenizer,
                    write_ref=workspace.refs / "wrong_schedule_ref.json",
                    workspace_root=workspace.root,
                )
            with self.assertRaisesRegex(TrainingScheduleError, "tokenizer revision"):
                build_training_schedule(
                    training_plan_ref=workspace.refs / "training_plan_ref.json",
                    training_evidence_ref=workspace.refs / "training_evidence_ref.json",
                    tokenizer=workspace.tokenizer,
                    tokenizer_revision="wrong-tokenizer/v1",
                    write_ref=workspace.refs / "wrong_revision_schedule_ref.json",
                    workspace_root=workspace.root,
                )

            locator = build_training_schedule(
                training_plan_ref=workspace.refs / "training_plan_ref.json",
                training_evidence_ref=workspace.refs / "training_evidence_ref.json",
                tokenizer=workspace.tokenizer,
                write_ref=workspace.refs / "schedule_ref.json",
                workspace_root=workspace.root,
            )
            target = Path(locator["target_path"])
            (target / "schedules/M_LD/seed-42/epoch-5.jsonl").unlink()
            workspace.refresh_manifest_and_locator(workspace.refs / "schedule_ref.json")
            with self.assertRaises(TrainingArtifactError):
                validate_training_schedule(
                    schedule_ref=workspace.refs / "schedule_ref.json"
                )


if __name__ == "__main__":
    unittest.main()
