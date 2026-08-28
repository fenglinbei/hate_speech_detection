from __future__ import annotations

import copy
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import jsonschema

from data.build_context_manifest import build_prepared_context_artifact
from data.generation_lifecycle import (
    MODEL_RESOLUTION_SCHEMA,
    FixtureExecutor,
    GenerationResult,
    GenerationLifecycleError,
    LocalHFExecutor,
    LocalVLLMExecutor,
    RegisteredModelHandle,
    _validate_formal_protocol_binding,
    build_generation_artifact,
    preflight_generation,
    validate_generation_profile,
    validate_generation_ref,
)
from data.context_manifest import text_sha256
from data.control_manifest import build_control_artifact
from metrics import stage1_artifacts as evaluation_artifacts
from metrics.stage1_metrics import evaluate_query
from data.training_artifacts import (
    build_payload_manifest,
    canonical_sha256,
    load_json,
    portable_dependency,
    resolve_dependency_target,
    resolve_locator_ref,
    sha256_file,
    write_bytes_atomic,
    write_canonical_json,
    write_canonical_jsonl,
    write_locator_ref,
)
from model.stage1_registry import (
    ResolvedModelSourceContract,
    inventory_regular_file_tree,
)
from tests.test_stage1_context_build import (
    FakeTokenizer,
    bundle,
    canonical_bundle_sha256,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
MODEL_KEY = "M_LD/seed-42"


class GenerationTokenizer(FakeTokenizer):
    eos_token_id = 10_000

    def decode(
        self,
        token_ids,
        *,
        skip_special_tokens,
        clean_up_tokenization_spaces,
    ):
        assert skip_special_tokens is True
        assert clean_up_tokenization_spaces is False
        return "decoded:" + ",".join(
            str(token_id)
            for token_id in token_ids
            if token_id != self.eos_token_id
        )


class ResolverFixture:
    def __init__(self, tokenizer: FakeTokenizer, *, eligible: bool = False):
        self.tokenizer = tokenizer
        self.eligible = eligible

    def _handle(self, locator, target, dependency):
        return RegisteredModelHandle(
            schema_version=MODEL_RESOLUTION_SCHEMA,
            locator=locator,
            target=target,
            dependency=dependency,
            registry_id=locator["artifact_id"],
            model_key=MODEL_KEY,
            role="M_LD",
            seed=42,
            scientific_eligible=self.eligible,
            tokenizer_revision="qwen3-8b-stage1-v1",
            tokenizer_content_revision="tok-" + "b" * 64,
            tokenizer=self.tokenizer,
            checkpoint_format="full",
            model_artifact_id="mdl-" + "c" * 64,
        )

    def direct(self, *, registry_ref, model_key, workspace_root):
        self._assert_key(model_key)
        locator, target = resolve_locator_ref(
            registry_ref, expected_kind="stage1-model-registry"
        )
        return self._handle(
            locator,
            target,
            portable_dependency(locator, target, workspace_root),
        )

    def dependency(self, *, registry_dependency, model_key, workspace_root):
        self._assert_key(model_key)
        target = resolve_dependency_target(registry_dependency, workspace_root)
        locator = {
            "schema_version": "stage1-locator-ref/v1",
            "artifact_kind": registry_dependency["artifact_kind"],
            "artifact_id": registry_dependency["artifact_id"],
            "target_path": str(target.resolve()),
            "payload_manifest_sha256": registry_dependency[
                "payload_manifest_sha256"
            ],
        }
        return self._handle(locator, target, registry_dependency)

    @staticmethod
    def _assert_key(model_key):
        if model_key != MODEL_KEY:
            raise AssertionError("unexpected model key")


class LegacyResolverFixture(ResolverFixture):
    def _handle(self, locator, target, dependency):
        handle = super()._handle(locator, target, dependency)
        return RegisteredModelHandle(
            **{
                **handle.__dict__,
                "model_key": "M_legacy/smoke",
                "role": "legacy-smoke-only",
                "seed": None,
                "scientific_eligible": False,
            }
        )

    @staticmethod
    def _assert_key(model_key):
        if model_key != "M_legacy/smoke":
            raise AssertionError("unexpected legacy model key")


class CountingFixture(FixtureExecutor):
    def __init__(self, outputs):
        super().__init__(outputs)
        self.calls = 0

    def generate(self, **kwargs):
        self.calls += 1
        return super().generate(**kwargs)


class FailingExecutor:
    def __init__(self):
        self.calls = 0

    def descriptor(self):
        return {
            "executor_id": "engineering-failure-fixture/v1",
            "executor_revision": "fixture/v1",
            "backend": "fixture",
            "scientific_eligible": False,
        }

    def generate(self, **kwargs):
        self.calls += 1
        if self.calls == 2:
            raise RuntimeError("synthetic failure")
        return "[]"


class LengthExecutor:
    def descriptor(self):
        return {
            "executor_id": "engineering-length-fixture/v1",
            "executor_revision": "fixture/v1",
            "backend": "fixture",
            "scientific_eligible": False,
        }

    def generate(self, *, profile, model, **kwargs):
        del kwargs
        count = int(profile["sampling"]["max_new_tokens"])
        token_ids = tuple(range(count))
        return GenerationResult(
            raw_output=model.tokenizer.decode(
                list(token_ids),
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            ),
            finish_reason="length",
            generated_token_ids=token_ids,
            backend_stop_reason=None,
        )


class RealLikeDeterministicExecutor:
    def __init__(self, *, drift_after: int | None = None):
        self.calls = 0
        self.drift_after = drift_after

    def descriptor(self):
        return {
            "executor_id": "hf-local-transformers/v1",
            "executor_revision": "stage1-generation-hf-executor/v1",
            "backend": "transformers",
            "scientific_eligible": True,
        }

    def generate(self, *, model, **kwargs):
        del kwargs
        self.calls += 1
        token = 2 if self.drift_after is not None and self.calls > self.drift_after else 1
        token_ids = (token, model.tokenizer.eos_token_id)
        return GenerationResult(
            raw_output=model.tokenizer.decode(
                token_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            ),
            finish_reason="eos",
            generated_token_ids=token_ids,
            backend_stop_reason=None,
        )


def profile(*, conditions=("C0", "CL", "CD", "CLD")):
    value = json.loads(
        (REPOSITORY_ROOT / "config/stage1/generation_greedy.json").read_text(
            encoding="utf-8"
        )
    )
    value["ordered_conditions"] = list(conditions)
    return value


def control_ready_bundle():
    value = copy.deepcopy(bundle())
    target_lexicon = value["lexicon_catalog"][0]
    target_demo = value["demo_catalog"][0]
    train_gold = copy.deepcopy(value["train_query_pool"][0]["quadruples"])
    for index, character in enumerate("3456789ab", start=2):
        lexicon_id = "lex:v2:" + character * 64
        lexicon_block = character * len(target_lexicon["rendered_block"])
        value["lexicon_catalog"].append(
            {
                "lexicon_id": lexicon_id,
                "term": f"candidate{index}",
                "definition": f"neutral definition {index}",
                "usage_notes": "",
                "ambiguity_notes": "",
                "evidence_kind": "terminology",
                "variants": [],
                "rendered_block": lexicon_block,
                "content_sha256": text_sha256(lexicon_block),
                "source_split": "train",
                "train_only": True,
            }
        )
        demo_id = "demo:v1:" + character * 64
        content = f"candidate-example-{index}"
        train_id = str(index)
        value["demo_catalog"].append(
            {
                "demo_id": demo_id,
                "source_record_id": train_id,
                "content": content,
                "output": target_demo["output"],
                "content_sha256": text_sha256(content),
                "gold_sha256": target_demo["gold_sha256"],
                "output_label": "non-hate",
                "rendered_block": character * len(target_demo["rendered_block"]),
                "source_split": "train",
                "train_only": True,
            }
        )
        value["train_query_pool"].append(
            {
                "id": train_id,
                "content": content,
                "quadruples": copy.deepcopy(train_gold),
                "content_sha256": text_sha256(content),
                "gold_sha256": target_demo["gold_sha256"],
                "source_split": "train",
            }
        )
    value["records"][0]["control_relevance"] = {
        "lexicons": [
            {
                "lexicon_id": row["lexicon_id"],
                "source_class": "terminology",
                "written_similarity": 0.9 if ordinal == 0 else ordinal / 100,
            }
            for ordinal, row in enumerate(value["lexicon_catalog"])
        ],
        "demos": [
            {
                "demo_id": row["demo_id"],
                "source_class": "non-hate",
                "written_similarity": 0.9 if ordinal == 0 else ordinal / 100,
            }
            for ordinal, row in enumerate(value["demo_catalog"])
        ],
    }
    value["bundle_sha256"] = canonical_bundle_sha256(value)
    return value


class GenerationLifecycleTests(unittest.TestCase):
    def test_local_hf_executor_cannot_be_constructed_from_arbitrary_objects(self):
        with self.assertRaisesRegex(GenerationLifecycleError, "registered-model"):
            LocalHFExecutor(object(), object())

    def test_formal_publication_rejects_every_caller_owned_executor_first(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(
                GenerationLifecycleError, "injected executors are forbidden"
            ):
                build_generation_artifact(
                    profile=root / "missing-profile.json",
                    context_ref=root / "missing-context.json",
                    model_registry_ref=root / "missing-registry.json",
                    model_key=MODEL_KEY,
                    workspace_root=root,
                    target_root=root / "generations",
                    write_ref=root / "generation-ref.json",
                    scope="formal",
                    executor=FixtureExecutor({("C0", "1"): "[]"}),
                )
            self.assertFalse((root / "generations").exists())

    def test_engineering_legacy_model_handle_is_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, registry_ref, resolver = self._workspace(root)
            locator, target = resolve_locator_ref(
                registry_ref, expected_kind="stage1-model-registry"
            )
            dependency = portable_dependency(locator, target, root)
            handle = resolver._handle(locator, target, dependency)
            legacy = RegisteredModelHandle(
                **{
                    **handle.__dict__,
                    "model_key": "M_legacy/smoke",
                    "role": "legacy-smoke-only",
                    "seed": None,
                    "scientific_eligible": False,
                }
            )
            from data.generation_lifecycle import _coerce_model_handle

            accepted = _coerce_model_handle(
                legacy,
                expected_locator=locator,
                expected_target=target,
                expected_dependency=dependency,
                model_key="M_legacy/smoke",
                require_scientific=False,
            )
            self.assertEqual(accepted.model_key, "M_legacy/smoke")
            self.assertEqual(accepted.role, "legacy-smoke-only")

    def test_engineering_legacy_artifact_passes_record_and_run_schemas(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            context_ref, registry_ref, resolver = self._workspace(root)
            legacy_resolver = LegacyResolverFixture(resolver.tokenizer)
            conditions = tuple(profile()["ordered_conditions"])
            locator = build_generation_artifact(
                profile=profile(),
                context_ref=context_ref,
                model_registry_ref=registry_ref,
                model_key="M_legacy/smoke",
                workspace_root=root,
                target_root=root / "artifacts/generations",
                write_ref=root / "refs/legacy-generation.json",
                scope="engineering",
                executor=FixtureExecutor(
                    {(condition, "10"): "[]" for condition in conditions}
                ),
                model_resolver=legacy_resolver.direct,
                model_dependency_resolver=legacy_resolver.dependency,
            )
            report = validate_generation_ref(
                root / "refs/legacy-generation.json",
                workspace_root=root,
                model_dependency_resolver=legacy_resolver.dependency,
            )
            self.assertTrue(report["valid"])
            self.assertEqual(report["model_key"], "M_legacy/smoke")
            self.assertEqual(report["model_role"], "legacy-smoke-only")
            self.assertEqual(Path(locator["target_path"]).name, report["generation_run_id"])

    def test_two_pass_real_backend_determinism_is_frozen_and_replayed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            context_ref, registry_ref, resolver = self._workspace(root)
            executor = RealLikeDeterministicExecutor()
            locator = build_generation_artifact(
                profile=profile(),
                context_ref=context_ref,
                model_registry_ref=registry_ref,
                model_key=MODEL_KEY,
                workspace_root=root,
                target_root=root / "artifacts/generations",
                write_ref=root / "refs/generation.json",
                scope="engineering",
                executor=executor,
                determinism_repetitions=2,
                model_resolver=resolver.direct,
                model_dependency_resolver=resolver.dependency,
            )
            self.assertEqual(executor.calls, 8)
            report = validate_generation_ref(
                root / "refs/generation.json",
                workspace_root=root,
                model_dependency_resolver=resolver.dependency,
            )
            self.assertEqual(report["execution_repetitions"], 2)
            self.assertTrue(report["exact_rerun_match"])
            self.assertEqual(report["executor_backend"], "transformers")
            receipt = load_json(Path(locator["target_path"]) / "determinism.json")
            self.assertEqual(receipt["mismatch_count"], 0)
            self.assertEqual(
                receipt["per_execution_raw_output_bytes_sha256"][0],
                receipt["per_execution_raw_output_bytes_sha256"][1],
            )

    def test_two_pass_drift_aborts_without_publishing_target_or_ref(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            context_ref, registry_ref, resolver = self._workspace(root)
            executor = RealLikeDeterministicExecutor(drift_after=4)
            with self.assertRaisesRegex(
                GenerationLifecycleError, "non-identical generation frame"
            ):
                build_generation_artifact(
                    profile=profile(),
                    context_ref=context_ref,
                    model_registry_ref=registry_ref,
                    model_key=MODEL_KEY,
                    workspace_root=root,
                    target_root=root / "artifacts/generations",
                    write_ref=root / "refs/generation.json",
                    scope="engineering",
                    executor=executor,
                    determinism_repetitions=2,
                    model_resolver=resolver.direct,
                    model_dependency_resolver=resolver.dependency,
                )
            self.assertFalse((root / "refs/generation.json").exists())
            self.assertEqual(list((root / "artifacts/generations").glob("gen-*")), [])

    def test_fixture_cannot_claim_two_pass_determinism(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            context_ref, registry_ref, resolver = self._workspace(root)
            executor = FixtureExecutor(
                {(condition, "10"): "[]" for condition in profile()["ordered_conditions"]}
            )
            with self.assertRaisesRegex(
                GenerationLifecycleError, "cannot serve as a determinism rerun"
            ):
                build_generation_artifact(
                    profile=profile(),
                    context_ref=context_ref,
                    model_registry_ref=registry_ref,
                    model_key=MODEL_KEY,
                    workspace_root=root,
                    target_root=root / "artifacts/generations",
                    write_ref=root / "refs/generation.json",
                    scope="engineering",
                    executor=executor,
                    determinism_repetitions=2,
                    model_resolver=resolver.direct,
                    model_dependency_resolver=resolver.dependency,
                )

    def _workspace(self, root: Path, *, eligible: bool = False):
        tokenizer = GenerationTokenizer()
        context_ref = root / "refs/context.json"
        build_prepared_context_artifact(
            prepared_bundle=bundle(),
            config=REPOSITORY_ROOT / "config/stage1/context_factorial.json",
            tokenizer=tokenizer,
            write_ref=context_ref,
            formal=False,
            target_root=root / "artifacts/contexts",
        )
        registry_id = "mreg-" + "a" * 64
        registry_target = root / "artifacts/model_registries" / registry_id
        registry_target.mkdir(parents=True)
        write_canonical_json(
            registry_target / "fixture.json",
            {"schema_version": "engineering-registry-fixture/v1"},
        )
        write_canonical_json(
            registry_target / "payload_manifest.json",
            build_payload_manifest(registry_target),
        )
        registry_ref = root / "refs/model_registry.json"
        write_locator_ref(
            registry_ref,
            artifact_kind="stage1-model-registry",
            artifact_id=registry_id,
            target=registry_target,
            payload_manifest_sha256=sha256_file(
                registry_target / "payload_manifest.json"
            ),
        )
        return context_ref, registry_ref, ResolverFixture(tokenizer, eligible=eligible)

    def _build(self, root: Path, *, executor=None, write_ref=None):
        context_ref, registry_ref, resolver = self._workspace(root)
        executor = executor or CountingFixture(
            {(condition, "10"): f"output-{condition}" for condition in profile()["ordered_conditions"]}
        )
        write_ref = write_ref or root / "refs/generation.json"
        locator = build_generation_artifact(
            profile=profile(),
            context_ref=context_ref,
            model_registry_ref=registry_ref,
            model_key=MODEL_KEY,
            workspace_root=root,
            target_root=root / "artifacts/generations",
            write_ref=write_ref,
            scope="engineering",
            executor=executor,
            model_resolver=resolver.direct,
            model_dependency_resolver=resolver.dependency,
        )
        return locator, context_ref, registry_ref, resolver, executor

    @staticmethod
    def _rehash_forged_generation(
        *, target: Path, rows: list[dict[str, object]], ref: Path, locator: dict
    ) -> None:
        for row in rows:
            row["raw_output_sha256"] = text_sha256(str(row["raw_output"]))
            frozen = {key: value for key, value in row.items() if key != "record_sha256"}
            row["record_sha256"] = canonical_sha256(frozen)
        write_canonical_jsonl(
            target / "generations.jsonl",
            rows,
            key="row_ordinal",
            numeric_key=True,
        )
        meta = load_json(target / "generation.meta.json")
        meta["records_sha256"] = canonical_sha256(rows)
        meta["record_hashes_sha256"] = canonical_sha256(
            [row["record_sha256"] for row in rows]
        )
        write_canonical_json(target / "generation.meta.json", meta)
        write_canonical_json(
            target / "payload_manifest.json", build_payload_manifest(target)
        )
        forged_locator = dict(locator)
        forged_locator["payload_manifest_sha256"] = sha256_file(
            target / "payload_manifest.json"
        )
        write_canonical_json(ref, forged_locator)

    def test_fixture_round_trip_complete_pairing_and_idempotence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            locator, context_ref, registry_ref, resolver, executor = self._build(root)
            report = validate_generation_ref(
                root / "refs/generation.json",
                workspace_root=root,
                model_dependency_resolver=resolver.dependency,
            )
            self.assertTrue(report["valid"])
            self.assertFalse(report["scientific_eligible"])
            self.assertEqual(report["row_count"], 4)
            rows = [
                json.loads(line)
                for line in (Path(locator["target_path"]) / "generations.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertEqual([row["condition"] for row in rows], ["C0", "CL", "CD", "CLD"])
            self.assertEqual({row["query_id"] for row in rows}, {"10"})
            self.assertTrue(all(row["attempt_count"] == 1 for row in rows))
            self.assertEqual(executor.calls, 4)

            second = build_generation_artifact(
                profile=profile(),
                context_ref=context_ref,
                model_registry_ref=registry_ref,
                model_key=MODEL_KEY,
                workspace_root=root,
                target_root=root / "artifacts/generations",
                write_ref=root / "refs/generation.json",
                scope="engineering",
                executor=executor,
                model_resolver=resolver.direct,
                model_dependency_resolver=resolver.dependency,
            )
            self.assertEqual(locator, second)
            self.assertEqual(executor.calls, 4, "idempotent rebuild reran generation")

    def test_length_finish_is_published_but_scores_as_invalid_in_fixed_denominator(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            locator, _, _, resolver, _ = self._build(
                root, executor=LengthExecutor()
            )
            report = validate_generation_ref(
                root / "refs/generation.json",
                workspace_root=root,
                model_dependency_resolver=resolver.dependency,
            )
            self.assertEqual(report["row_count"], 4)
            rows = [
                json.loads(line)
                for line in (Path(locator["target_path"]) / "generations.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertTrue(all(row["runner_status"] == "length" for row in rows))
            self.assertTrue(all(row["finish_reason"] == "length" for row in rows))
            self.assertTrue(all("backend_stop_reason" in row for row in rows))
            self.assertTrue(all(row["backend_stop_reason"] is None for row in rows))
            self.assertTrue(all(row["completion_tokens"] == 256 for row in rows))
            metric = evaluate_query(
                query_id=rows[0]["query_id"],
                condition=rows[0]["condition"],
                raw_output=rows[0]["raw_output"],
                gold=rows[0]["gold"],
                runner_status=rows[0]["runner_status"],
            )
            self.assertFalse(metric["strict_format_valid"])
            self.assertEqual(metric["scored_pred_tuple_count"], 0)

    def test_evaluator_rejects_stop_state_mismatch_and_keeps_length_invalid(self):
        prediction = {
            "id": "10",
            "raw_output": "[]",
            "runner_status": "length",
            "finish_reason": "length",
            "gold": [],
            "content_sha256": "a" * 64,
            "gold_sha256": "b" * 64,
            "prompt_sha256": "c" * 64,
            "context_record_sha256": "d" * 64,
        }
        generation = {
            "conditions": ["C0"],
            "predictions": {"C0": [prediction]},
        }
        evaluation_profile = evaluation_artifacts.resolve_evaluation_profile(
            REPOSITORY_ROOT / "config/stage1/evaluation_strict.json"
        )
        evaluated, summaries = evaluation_artifacts._evaluate_generation(
            generation, evaluation_profile
        )
        self.assertEqual(summaries["C0"]["query_count"], 1)
        self.assertEqual(evaluated["C0"][0]["finish_reason"], "length")
        self.assertFalse(evaluated["C0"][0]["strict_format_valid"])
        self.assertEqual(evaluated["C0"][0]["scored_pred_tuple_count"], 0)

        inconsistent = copy.deepcopy(generation)
        inconsistent["predictions"]["C0"][0]["runner_status"] = "ok"
        with self.assertRaisesRegex(
            evaluation_artifacts.Stage1ArtifactError,
            "finish_reason/runner_status mismatch",
        ):
            evaluation_artifacts._evaluate_generation(
                inconsistent, evaluation_profile
            )

    def test_output_and_meta_validate_against_schemas(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            locator, *_ = self._build(root)
            target = Path(locator["target_path"])
            run_schema = load_json(REPOSITORY_ROOT / "schemas/stage1_generation_run_v1.schema.json")
            row_schema = load_json(REPOSITORY_ROOT / "schemas/stage1_generation_record_v1.schema.json")
            jsonschema.validate(load_json(target / "generation.meta.json"), run_schema)
            for row in [json.loads(line) for line in (target / "generations.jsonl").read_text().splitlines()]:
                jsonschema.validate(row, row_schema)

    def test_full_six_condition_control_frame_is_exactly_paired(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tokenizer = FakeTokenizer()
            context_ref = root / "refs/context.json"
            build_prepared_context_artifact(
                prepared_bundle=control_ready_bundle(),
                config=REPOSITORY_ROOT / "config/stage1/context_factorial.json",
                tokenizer=tokenizer,
                write_ref=context_ref,
                formal=False,
                target_root=root / "artifacts/contexts",
            )
            registry_id = "mreg-" + "a" * 64
            registry_target = root / "artifacts/model_registries" / registry_id
            registry_target.mkdir(parents=True)
            write_canonical_json(registry_target / "fixture.json", {"fixture": True})
            write_canonical_json(
                registry_target / "payload_manifest.json",
                build_payload_manifest(registry_target),
            )
            registry_ref = root / "refs/model_registry.json"
            write_locator_ref(
                registry_ref,
                artifact_kind="stage1-model-registry",
                artifact_id=registry_id,
                target=registry_target,
                payload_manifest_sha256=sha256_file(registry_target / "payload_manifest.json"),
            )
            resolver = ResolverFixture(tokenizer)
            control_ref = root / "refs/control.json"
            build_control_artifact(
                config={
                    "schema_version": "stage1-control-config/v1",
                    "profile_name": "generation-test",
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
            conditions = ("C0", "CL", "CD", "CLD", "PL", "PD")
            executor = FixtureExecutor(
                {(condition, "10"): f"output-{condition}" for condition in conditions}
            )
            locator = build_generation_artifact(
                profile=profile(conditions=conditions),
                context_ref=context_ref,
                control_ref=control_ref,
                model_registry_ref=registry_ref,
                model_key=MODEL_KEY,
                workspace_root=root,
                target_root=root / "artifacts/generations",
                write_ref=root / "refs/generation.json",
                scope="engineering",
                executor=executor,
                model_resolver=resolver.direct,
                model_dependency_resolver=resolver.dependency,
            )
            rows = [
                json.loads(line)
                for line in (Path(locator["target_path"]) / "generations.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertEqual([row["condition"] for row in rows], list(conditions))
            self.assertTrue(all(row["query_id"] == "10" for row in rows))
            self.assertIsNone(rows[0]["control_record_sha256"])
            self.assertIsNotNone(rows[-1]["control_record_sha256"])

    def test_missing_fixture_row_fails_before_publication(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            context_ref, registry_ref, resolver = self._workspace(root)
            executor = FixtureExecutor({("C0", "10"): "[]"})
            ref = root / "refs/generation.json"
            with self.assertRaisesRegex(GenerationLifecycleError, "frame mismatch"):
                build_generation_artifact(
                    profile=profile(),
                    context_ref=context_ref,
                    model_registry_ref=registry_ref,
                    model_key=MODEL_KEY,
                    workspace_root=root,
                    target_root=root / "artifacts/generations",
                    write_ref=ref,
                    scope="engineering",
                    executor=executor,
                    model_resolver=resolver.direct,
                    model_dependency_resolver=resolver.dependency,
                )
            self.assertFalse(ref.exists())
            self.assertFalse((root / "artifacts/generations").exists())

    def test_executor_failure_publishes_no_partial_block_and_never_retries(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            context_ref, registry_ref, resolver = self._workspace(root)
            executor = FailingExecutor()
            ref = root / "refs/generation.json"
            with self.assertRaisesRegex(GenerationLifecycleError, "failed once"):
                build_generation_artifact(
                    profile=profile(),
                    context_ref=context_ref,
                    model_registry_ref=registry_ref,
                    model_key=MODEL_KEY,
                    workspace_root=root,
                    target_root=root / "artifacts/generations",
                    write_ref=ref,
                    scope="engineering",
                    executor=executor,
                    model_resolver=resolver.direct,
                    model_dependency_resolver=resolver.dependency,
                )
            self.assertEqual(executor.calls, 2)
            self.assertFalse(ref.exists())
            published = root / "artifacts/generations"
            self.assertFalse(any(path.name.startswith("gen-") for path in published.iterdir()))

    def test_tampered_output_is_rejected_even_after_payload_rehash(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            locator, _, _, resolver, _ = self._build(root)
            target = Path(locator["target_path"])
            rows = [
                json.loads(line)
                for line in (target / "generations.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            rows[0]["raw_output"] = "tampered"
            (target / "generations.jsonl").write_text(
                "".join(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n" for row in rows),
                encoding="utf-8",
            )
            write_canonical_json(target / "payload_manifest.json", build_payload_manifest(target))
            forged_locator = dict(locator)
            forged_locator["payload_manifest_sha256"] = sha256_file(
                target / "payload_manifest.json"
            )
            write_canonical_json(root / "refs/generation.json", forged_locator)
            with self.assertRaisesRegex(GenerationLifecycleError, "record hash mismatch|output hash mismatch"):
                validate_generation_ref(
                    root / "refs/generation.json",
                    workspace_root=root,
                    model_dependency_resolver=resolver.dependency,
                )

    def test_decode_semantics_survive_forged_record_meta_and_payload_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            locator, _, _, resolver, _ = self._build(
                root, executor=LengthExecutor()
            )
            target = Path(locator["target_path"])
            rows = [
                json.loads(line)
                for line in (target / "generations.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            rows[0]["raw_output"] = "forged-but-fully-rehashed"
            self._rehash_forged_generation(
                target=target,
                rows=rows,
                ref=root / "refs/generation.json",
                locator=locator,
            )
            with self.assertRaisesRegex(
                GenerationLifecycleError, "differs from frozen tokenizer decode"
            ):
                validate_generation_ref(
                    root / "refs/generation.json",
                    workspace_root=root,
                    model_dependency_resolver=resolver.dependency,
                )

    def test_eos_claim_without_tokenizer_evidence_fails_after_full_rehash(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            locator, _, _, resolver, _ = self._build(
                root, executor=LengthExecutor()
            )
            target = Path(locator["target_path"])
            rows = [
                json.loads(line)
                for line in (target / "generations.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            rows[0]["finish_reason"] = "eos"
            rows[0]["runner_status"] = "ok"
            self._rehash_forged_generation(
                target=target,
                rows=rows,
                ref=root / "refs/generation.json",
                locator=locator,
            )
            with self.assertRaisesRegex(
                GenerationLifecycleError, "lacks registered-tokenizer EOS evidence"
            ):
                validate_generation_ref(
                    root / "refs/generation.json",
                    workspace_root=root,
                    model_dependency_resolver=resolver.dependency,
                )

    def test_vllm_rejects_bool_or_coercible_token_ids_before_conversion(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, registry_ref, resolver = self._workspace(root)
            handle = resolver.direct(
                registry_ref=registry_ref,
                model_key=MODEL_KEY,
                workspace_root=root,
            )
            binding = {
                "registry_id": handle.registry_id,
                "model_artifact_id": handle.model_artifact_id,
                "model_key": handle.model_key,
                "checkpoint_format": handle.checkpoint_format,
                "tokenizer_revision": handle.tokenizer_revision,
                "tokenizer_content_revision": handle.tokenizer_content_revision,
            }
            for invalid_id in (True, "7"):
                completion = SimpleNamespace(
                    finish_reason="length",
                    stop_reason=None,
                    token_ids=[invalid_id] * 256,
                    text="ignored",
                )
                engine = SimpleNamespace(
                    generate=lambda *_args, **_kwargs: [
                        SimpleNamespace(outputs=[completion])
                    ]
                )
                executor = LocalVLLMExecutor(
                    engine,
                    object(),
                    _authority=LocalVLLMExecutor._AUTHORITY,
                    model_binding=binding,
                )
                with self.subTest(invalid_id=invalid_id), self.assertRaisesRegex(
                    GenerationLifecycleError, "original non-bool"
                ):
                    executor.generate(
                        messages=[],
                        query_id="10",
                        condition="C0",
                        profile=profile(),
                        model=handle,
                    )
            self.assertEqual(executor.descriptor()["vllm_use_v1"], "1")

            eos_token_ids = [7, handle.tokenizer.eos_token_id]
            eos_text = handle.tokenizer.decode(
                eos_token_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            eos_completion = SimpleNamespace(
                finish_reason="stop",
                stop_reason=None,
                token_ids=eos_token_ids,
                text=eos_text,
            )
            eos_engine = SimpleNamespace(
                generate=lambda *_args, **_kwargs: [
                    SimpleNamespace(outputs=[eos_completion])
                ]
            )
            eos_executor = LocalVLLMExecutor(
                eos_engine,
                object(),
                _authority=LocalVLLMExecutor._AUTHORITY,
                model_binding=binding,
            )
            eos_result = eos_executor.generate(
                messages=[],
                query_id="10",
                condition="C0",
                profile=profile(),
                model=handle,
            )
            self.assertEqual(eos_result.finish_reason, "eos")
            self.assertIsNone(eos_result.backend_stop_reason)

            # V1 checks the token cap before EOS, so an EOS sampled exactly at
            # max_tokens can retain backend finish_reason="length".  The
            # registered terminal token remains the authoritative evidence.
            boundary_ids = [7] * 255 + [handle.tokenizer.eos_token_id]
            boundary_completion = SimpleNamespace(
                finish_reason="length",
                stop_reason=None,
                token_ids=boundary_ids,
                text=handle.tokenizer.decode(
                    boundary_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                ),
            )
            boundary_executor = LocalVLLMExecutor(
                SimpleNamespace(
                    generate=lambda *_args, **_kwargs: [
                        SimpleNamespace(outputs=[boundary_completion])
                    ]
                ),
                object(),
                _authority=LocalVLLMExecutor._AUTHORITY,
                model_binding=binding,
            )
            boundary_result = boundary_executor.generate(
                messages=[],
                query_id="10",
                condition="C0",
                profile=profile(),
                model=handle,
            )
            self.assertEqual(boundary_result.finish_reason, "eos")
            self.assertIsNone(boundary_result.backend_stop_reason)

    def test_vllm_v1_environment_is_fixed_before_backend_construction(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, registry_ref, resolver = self._workspace(root)
            base_handle = resolver.direct(
                registry_ref=registry_ref,
                model_key=MODEL_KEY,
                workspace_root=root,
            )
            handle = RegisteredModelHandle(
                **{
                    **base_handle.__dict__,
                    "checkpoint_path": root,
                    "tokenizer_path": root,
                }
            )
            observed = {}

            def fake_llm(**kwargs):
                observed["vllm_use_v1"] = os.environ.get("VLLM_USE_V1")
                observed["llm_kwargs"] = kwargs
                return object()

            def fake_sampling_params(**kwargs):
                observed["sampling_kwargs"] = kwargs
                return object()

            fake_vllm = SimpleNamespace(
                LLM=fake_llm,
                SamplingParams=fake_sampling_params,
            )
            with patch.dict("sys.modules", {"vllm": fake_vllm}), patch.dict(
                os.environ, {"VLLM_USE_V1": "0"}
            ):
                executor = LocalVLLMExecutor.from_registered_model(
                    handle, profile()
                )
                self.assertEqual(os.environ["VLLM_USE_V1"], "1")
            self.assertEqual(observed["vllm_use_v1"], "1")
            self.assertTrue(observed["sampling_kwargs"]["skip_special_tokens"])
            self.assertEqual(executor.descriptor()["vllm_use_v1"], "1")

    def test_hf_load_rejects_checkpoint_swap_and_restore_inside_constructor(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "sources/checkpoint-10"
            source.mkdir(parents=True)
            shard = source / "model.safetensors"
            write_bytes_atomic(shard, b"verified-weights")
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
                tokenizer=GenerationTokenizer(),
                checkpoint_format="full",
                checkpoint_path=source,
                base_model_path=source,
                tokenizer_path=source,
                model_artifact_id="mdl-" + "c" * 64,
                source_contract=contract,
            )
            hf_profile = profile()
            hf_profile["backend"] = "transformers"
            hf_profile["model_runtime"]["tensor_parallel_size"] = 1
            original = shard.read_bytes()

            fake_tokenizer = SimpleNamespace()
            fake_model = SimpleNamespace(eval=lambda: None)

            def load_after_transient_swap(*_args, **_kwargs):
                write_bytes_atomic(shard, b"transient-unverified-weights")
                write_bytes_atomic(shard, original)
                return fake_model

            with patch(
                "transformers.AutoTokenizer.from_pretrained",
                return_value=fake_tokenizer,
            ), patch(
                "transformers.AutoModelForCausalLM.from_pretrained",
                side_effect=load_after_transient_swap,
            ), self.assertRaisesRegex(
                GenerationLifecycleError, "changed during backend load"
            ):
                LocalHFExecutor.from_registered_model(handle, hf_profile)

    def test_hf_load_rejects_ancestor_tree_swap_and_restore(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_parent = root / "sources"
            source = source_parent / "checkpoint-10"
            source.mkdir(parents=True)
            shard = source / "model.safetensors"
            write_bytes_atomic(shard, b"verified-ancestor-weights")
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
                tokenizer=GenerationTokenizer(),
                checkpoint_format="full",
                checkpoint_path=source,
                base_model_path=source,
                tokenizer_path=source,
                model_artifact_id="mdl-" + "c" * 64,
                source_contract=contract,
            )
            attacker_parent = root / "attacker-tree"
            attacker_source = attacker_parent / "checkpoint-10"
            attacker_source.mkdir(parents=True)
            write_bytes_atomic(
                attacker_source / "model.safetensors", b"unverified-ancestor-weights"
            )
            write_canonical_json(
                attacker_source / "tokenizer.json", {"version": "evil"}
            )
            parked_parent = root / "verified-tree-parked"
            observed: list[bytes] = []

            def load_from_transient_ancestor(*_args, **_kwargs):
                os.replace(source_parent, parked_parent)
                os.replace(attacker_parent, source_parent)
                try:
                    observed.append(shard.read_bytes())
                finally:
                    os.replace(source_parent, attacker_parent)
                    os.replace(parked_parent, source_parent)
                return SimpleNamespace(eval=lambda: None)

            hf_profile = profile()
            hf_profile["backend"] = "transformers"
            hf_profile["model_runtime"]["tensor_parallel_size"] = 1
            with patch(
                "transformers.AutoTokenizer.from_pretrained",
                return_value=SimpleNamespace(),
            ), patch(
                "transformers.AutoModelForCausalLM.from_pretrained",
                side_effect=load_from_transient_ancestor,
            ), self.assertRaisesRegex(
                GenerationLifecycleError, "changed during backend load"
            ):
                LocalHFExecutor.from_registered_model(handle, hf_profile)
            self.assertEqual(observed, [b"unverified-ancestor-weights"])

    def test_formal_scope_rejects_injected_registry_resolver(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            context_ref, registry_ref, resolver = self._workspace(root, eligible=True)
            descriptor = {
                "executor_id": "vllm-local/v1",
                "executor_revision": "vllm-fixture/v1",
                "backend": "vllm",
                "scientific_eligible": True,
            }
            with self.assertRaisesRegex(GenerationLifecycleError, "forbids injected"):
                preflight_generation(
                    profile=profile(conditions=("C0", "CL", "CD", "CLD", "PL", "PD")),
                    context_ref=context_ref,
                    control_ref=None,
                    model_registry_ref=registry_ref,
                    model_key=MODEL_KEY,
                    workspace_root=root,
                    scope="formal",
                    executor_descriptor=descriptor,
                    model_resolver=resolver.direct,
                )

    def test_retry_or_fallback_profile_is_rejected(self):
        retry = profile()
        retry["failure_policy"]["max_attempts"] = 2
        with self.assertRaisesRegex(GenerationLifecycleError, "exactly one"):
            validate_generation_profile(retry)
        fallback = profile()
        fallback["failure_policy"]["fallback_generation"] = True
        with self.assertRaisesRegex(GenerationLifecycleError, "forbidden"):
            validate_generation_profile(fallback)

    def test_formal_profile_is_bound_to_registry_training_plan_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frozen_profile = profile(conditions=("C0", "CL", "CD", "CLD", "PL", "PD"))
            train_context_policy = {
                "schema_version": "stage1-context-policy-lineage/v1",
                "fixture": "exact-policy",
            }
            train_context_policy_sha256 = canonical_sha256(train_context_policy)
            plan_id = "tpl-" + "f" * 64
            plan_target = root / "artifacts/training_plans" / plan_id
            plan_target.mkdir(parents=True)
            write_canonical_json(
                plan_target / "protocol_snapshot.json",
                {
                    "ordered_generation_conditions": frozen_profile["ordered_conditions"],
                    "train_context_policy": train_context_policy,
                    "train_context_policy_sha256": train_context_policy_sha256,
                    "profiles": {
                        "generation": {
                            "resolved": frozen_profile,
                            "sha256": canonical_sha256(frozen_profile),
                        }
                    },
                },
            )
            write_canonical_json(plan_target / "fixture.json", {"formal": True})
            write_canonical_json(
                plan_target / "payload_manifest.json", build_payload_manifest(plan_target)
            )
            plan_ref = root / "refs/plan.json"
            plan_locator = write_locator_ref(
                plan_ref,
                artifact_kind="training-plan",
                artifact_id=plan_id,
                target=plan_target,
                payload_manifest_sha256=sha256_file(plan_target / "payload_manifest.json"),
            )
            plan_dependency = portable_dependency(plan_locator, plan_target, root)

            registry_id = "mreg-" + "a" * 64
            registry_target = root / "artifacts/model_registries" / registry_id
            registry_target.mkdir(parents=True)
            write_canonical_json(
                registry_target / "registry.json",
                {"training_plan_dependency": plan_dependency},
            )
            write_canonical_json(
                registry_target / "payload_manifest.json",
                build_payload_manifest(registry_target),
            )
            registry_ref = root / "refs/registry.json"
            write_locator_ref(
                registry_ref,
                artifact_kind="stage1-model-registry",
                artifact_id=registry_id,
                target=registry_target,
                payload_manifest_sha256=sha256_file(registry_target / "payload_manifest.json"),
            )
            handle = ResolverFixture(FakeTokenizer(), eligible=True).direct(
                registry_ref=registry_ref,
                model_key=MODEL_KEY,
                workspace_root=root,
            )
            with patch(
                "data.training_plan.validate_training_plan_target",
                return_value={
                    "scope": "formal",
                    "scientific_eligible": True,
                    "train_context_policy": train_context_policy,
                    "train_context_policy_sha256": train_context_policy_sha256,
                },
            ):
                _validate_formal_protocol_binding(
                    handle, frozen_profile, workspace_root=root
                )
                changed = copy.deepcopy(frozen_profile)
                changed["sampling"]["seed"] = 43
                with self.assertRaisesRegex(
                    GenerationLifecycleError, "differs from.*frozen protocol"
                ):
                    _validate_formal_protocol_binding(
                        handle, changed, workspace_root=root
                    )

    def test_locator_is_immutable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            locator, context_ref, registry_ref, resolver, _ = self._build(root)
            other = FixtureExecutor({("C0", "10"): "different"})
            with self.assertRaisesRegex(GenerationLifecycleError, "locator is immutable"):
                build_generation_artifact(
                    profile=profile(conditions=("C0",)),
                    context_ref=context_ref,
                    model_registry_ref=registry_ref,
                    model_key=MODEL_KEY,
                    workspace_root=root,
                    target_root=root / "artifacts/generations",
                    write_ref=root / "refs/generation.json",
                    scope="engineering",
                    executor=other,
                    model_resolver=resolver.direct,
                    model_dependency_resolver=resolver.dependency,
                )
            self.assertEqual(load_json(root / "refs/generation.json"), locator)


if __name__ == "__main__":
    unittest.main()
