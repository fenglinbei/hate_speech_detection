from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from data.training_artifacts import (
    canonical_json_bytes,
    canonical_sha256,
    finalize_target_atomic,
    load_json,
    load_jsonl,
    new_staging_directory,
    portable_dependency,
    sha256_file,
    validate_payload_manifest,
    write_canonical_json,
    write_canonical_jsonl,
    write_locator_ref,
)
from metrics import stage1_artifacts as artifacts
from metrics.stage1_artifacts import (
    Stage1ArtifactError,
    build_factorial_analysis_artifact,
    build_free_evaluation_artifact,
    validate_analysis_ref,
    validate_evaluation_ref_report,
)
from metrics.stage1_metrics import aggregate_query_metrics, evaluate_query
from metrics.stage1_statistics import fixed_seed_margin_bootstrap
from utils.quadruple import serialize_quadruples


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def quad(target: str, argument: str) -> dict[str, object]:
    return {
        "target": target,
        "argument": argument,
        "targeted_group": ["Racism"],
        "hateful": "hate",
    }


class ArtifactWorkspace:
    def __init__(self, root: Path):
        self.root = root
        self.targets = root / "artifacts"
        self.refs = root / "refs"

    def simple(
        self, *, kind: str, prefix: str, label: str
    ) -> tuple[dict[str, object], Path, dict[str, object], Path]:
        artifact_id = prefix + canonical_sha256({"label": label})
        parent = self.targets / f"{kind.replace('/', '_')}_targets"
        target = parent / artifact_id
        staging = new_staging_directory(parent, artifact_id)
        write_canonical_json(staging / "marker.json", {"label": label})
        payload_hash = finalize_target_atomic(staging, target)
        ref = self.refs / f"{label}.ref.json"
        locator = write_locator_ref(
            ref,
            artifact_kind=kind,
            artifact_id=artifact_id,
            target=target,
            payload_manifest_sha256=payload_hash,
        )
        dependency = portable_dependency(locator, target, self.root)
        return locator, target, dependency, ref


class FrozenRepositoryProfileTests(unittest.TestCase):
    def test_shipped_profiles_resolve_under_the_runtime_contract(self):
        config_root = REPOSITORY_ROOT / "config/stage1"
        artifacts.resolve_evaluation_profile(config_root / "evaluation_strict.json")
        artifacts.resolve_margin_profile(config_root / "margin_scorer.json")
        for filename in (
            "analysis_smoke.json",
            "analysis_pilot.json",
            "analysis_confirmatory.json",
        ):
            artifacts.resolve_analysis_profile(config_root / filename)

    def test_margin_profile_rejects_ambiguous_or_tampered_batch_semantics(self):
        source = load_json(REPOSITORY_ROOT / "config/stage1/margin_scorer.json")
        mutations = {
            "missing-unit": lambda runtime: runtime.pop("batch_unit"),
            "sequence-unit": lambda runtime: runtime.__setitem__(
                "batch_unit", "sequence"
            ),
            "two-pair-batch": lambda runtime: runtime.__setitem__("batch_size", 2),
            "remote-code": lambda runtime: runtime.__setitem__(
                "trust_remote_code", True
            ),
        }
        for label, mutate in mutations.items():
            profile = copy.deepcopy(source)
            mutate(profile["runtime"])
            with self.subTest(label=label), self.assertRaises(Stage1ArtifactError):
                artifacts.resolve_margin_profile(profile)


class FreeEvaluationLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.workspace = ArtifactWorkspace(self.root)
        _, _, self.plan_dependency, _ = self.workspace.simple(
            kind="training-plan", prefix="plan-", label="plan"
        )
        _, _, self.model_dependency, _ = self.workspace.simple(
            kind="stage1-model", prefix="mdl-", label="model"
        )
        registry_locator, self.registry_target, self.registry_dependency, _ = (
            self.workspace.simple(
                kind="stage1-model-registry", prefix="mreg-", label="registry"
            )
        )
        del registry_locator
        _, _, self.context_dependency, _ = self.workspace.simple(
            kind="context", prefix="ctx-", label="context"
        )
        _, _, self.control_dependency, _ = self.workspace.simple(
            kind="control", prefix="ctl-", label="control"
        )
        self.registry_report = {
            "model_registry_id": self.registry_dependency["artifact_id"],
            "registry_scope": "engineering-smoke",
            "scientific_eligible": False,
            "training_plan_dependency": self.plan_dependency,
            "models": [
                {
                    "model_key": "M_legacy/smoke",
                    "role": "M_legacy",
                    "seed": None,
                    "model_dependency": self.model_dependency,
                    "scientific_eligible": False,
                }
            ],
            "payload_manifest_sha256": self.registry_dependency[
                "payload_manifest_sha256"
            ],
        }
        self.generation_ref, self.generation_target = self._generation()

    def tearDown(self):
        self.temporary.cleanup()

    def _generation(
        self,
        *,
        split: str = "dev",
        scope: str = "engineering",
        scientific_eligible: bool = False,
        context_dependency: dict[str, object] | None = None,
        control_dependency: dict[str, object] | None = None,
        label: str = "evaluation",
    ) -> tuple[Path, Path]:
        context_dependency = context_dependency or self.context_dependency
        control_dependency = control_dependency or self.control_dependency
        generation_id = "gen-" + canonical_sha256(
            {
                "fixture": label,
                "split": split,
                "context_kind": context_dependency["artifact_kind"],
                "control_kind": control_dependency["artifact_kind"],
            }
        )
        parent = self.workspace.targets / "generation_runs"
        target = parent / generation_id
        staging = new_staging_directory(parent, generation_id)
        profile = load_json(REPOSITORY_ROOT / "config/stage1/generation_greedy.json")
        query_ids = ["1", "2"]
        gold = [quad("甲", "甲的论点")]
        records = []
        for query_ordinal, query_id in enumerate(query_ids):
            for condition_ordinal, condition in enumerate(artifacts.CONDITIONS):
                raw = serialize_quadruples(gold) if query_id == "1" else "```json\n[]\n```"
                records.append(
                    {
                        "query_id": query_id,
                        "condition": condition,
                        "model_key": "M_legacy/smoke",
                        "raw_output": raw,
                        "runner_status": "ok",
                        "finish_reason": "fixture",
                        "gold": copy.deepcopy(gold),
                        "content_sha256": ("a" if query_id == "1" else "b") * 64,
                        "gold_sha256": "c" * 64,
                        "prompt_sha256": f"{condition_ordinal:x}" * 64,
                        "context_record_sha256": ("d" if query_id == "1" else "e") * 64,
                    }
                )
        meta = {
            "schema_version": "stage1-generation-run/v1",
            "generation_run_id": generation_id,
            "scope": scope,
            "scientific_eligible": scientific_eligible,
            "model_key": "M_legacy/smoke",
            "model_role": "M_legacy",
            "model_seed": None,
            "split": split,
            "ordered_conditions": list(artifacts.CONDITIONS),
            "ordered_query_ids": query_ids,
            "ordered_query_ids_sha256": canonical_sha256(query_ids),
            "dependencies": {
                "context": context_dependency,
                "model_registry": self.registry_dependency,
                "control": control_dependency,
            },
        }
        write_canonical_json(staging / "generation.meta.json", meta)
        write_canonical_json(staging / "generation_profile.resolved.json", profile)
        write_canonical_jsonl(
            staging / "generations.jsonl",
            [dict(row, row_ordinal=index) for index, row in enumerate(records)],
            key="row_ordinal",
            numeric_key=True,
        )
        payload_hash = finalize_target_atomic(staging, target)
        ref = self.workspace.refs / f"{label}.generation.ref.json"
        write_locator_ref(
            ref,
            artifact_kind="generation-run",
            artifact_id=generation_id,
            target=target,
            payload_manifest_sha256=payload_hash,
        )
        return ref, target

    def _upstream_report(self, target: Path, **_kwargs: object) -> dict[str, object]:
        return {
            "generation_run_id": target.name,
            "query_count": 2,
            "complete_paired_blocks": True,
            "payload_manifest_sha256": sha256_file(target / "payload_manifest.json"),
        }

    def test_free_target_is_replayable_idempotent_and_retains_invalid_denominator(self):
        evaluation_ref = self.workspace.refs / "evaluation.ref.json"
        patches = (
            mock.patch.object(
                artifacts, "validate_upstream_generation_target", self._upstream_report
            ),
            mock.patch.object(
                artifacts,
                "validate_registry_target",
                return_value=copy.deepcopy(self.registry_report),
            ),
        )
        with patches[0], patches[1]:
            first = build_free_evaluation_artifact(
                generation_run_ref=self.generation_ref,
                evaluation_profile=REPOSITORY_ROOT / "config/stage1/evaluation_strict.json",
                write_ref=evaluation_ref,
                split="dev",
                sealed=False,
                target_root=self.workspace.targets,
                workspace_root=self.root,
            )
            generation_hash_before = validate_payload_manifest(self.generation_target)
            second = build_free_evaluation_artifact(
                generation_run_ref=self.generation_ref,
                evaluation_profile=REPOSITORY_ROOT / "config/stage1/evaluation_strict.json",
                write_ref=evaluation_ref,
                split="dev",
                target_root=self.workspace.targets,
                workspace_root=self.root,
            )
            report = validate_evaluation_ref_report(
                evaluation_ref, workspace_root=self.root
            )
        self.assertEqual(first, second)
        self.assertEqual(generation_hash_before, validate_payload_manifest(self.generation_target))
        self.assertEqual(report["query_ids"], ["1", "2"])
        self.assertEqual(report["summaries"]["C0"]["query_count"], 2)
        self.assertEqual(report["summaries"]["C0"]["format"]["strict_format_rate"], 0.5)
        target = Path(first["target_path"])
        rows = load_jsonl(target / "per_query_metrics/C0.jsonl")
        self.assertEqual([row["id"] for row in rows], ["1", "2"])
        self.assertFalse(rows[1]["strict_format_valid"])
        self.assertGreater(rows[1]["tuple"]["hard"]["fn"], 0)
        id_inputs = load_json(target / "evaluation.meta.json")["id_inputs"]
        forbidden = {"summary", "parse_errors", "tp", "fp", "fn", "infra_exclusion_ids"}
        self.assertTrue(forbidden.isdisjoint(id_inputs))

    def test_free_assertions_fail_closed(self):
        with mock.patch.object(
            artifacts, "validate_upstream_generation_target", self._upstream_report
        ), mock.patch.object(
            artifacts,
            "validate_registry_target",
            return_value=copy.deepcopy(self.registry_report),
        ):
            with self.assertRaisesRegex(Stage1ArtifactError, "split assertion"):
                build_free_evaluation_artifact(
                    generation_run_ref=self.generation_ref,
                    evaluation_profile=REPOSITORY_ROOT / "config/stage1/evaluation_strict.json",
                    write_ref=self.workspace.refs / "bad.ref.json",
                    split="test",
                    target_root=self.workspace.targets,
                    workspace_root=self.root,
                )

    def test_sealed_test_control_generation_builds_replayable_evaluation(self):
        _, _, test_context_dependency, _ = self.workspace.simple(
            kind="test-context", prefix="tctx-", label="sealed-evaluation-context"
        )
        _, _, test_control_dependency, _ = self.workspace.simple(
            kind="test-control", prefix="tctl-", label="sealed-evaluation-control"
        )
        generation_ref, _ = self._generation(
            split="test",
            scope="formal",
            scientific_eligible=True,
            context_dependency=test_context_dependency,
            control_dependency=test_control_dependency,
            label="sealed-evaluation",
        )
        registry_report = copy.deepcopy(self.registry_report)
        registry_report["registry_scope"] = "formal"
        registry_report["scientific_eligible"] = True
        registry_report["models"][0]["scientific_eligible"] = True
        evaluation_ref = self.workspace.refs / "sealed.evaluation.ref.json"
        with mock.patch.object(
            artifacts, "validate_upstream_generation_target", self._upstream_report
        ), mock.patch.object(
            artifacts,
            "validate_registry_target",
            return_value=registry_report,
        ):
            locator = build_free_evaluation_artifact(
                generation_run_ref=generation_ref,
                evaluation_profile=REPOSITORY_ROOT
                / "config/stage1/evaluation_strict.json",
                write_ref=evaluation_ref,
                split="test",
                sealed=True,
                target_root=self.workspace.targets,
                workspace_root=self.root,
            )
            report = validate_evaluation_ref_report(
                evaluation_ref, workspace_root=self.root
            )
        self.assertEqual(report["split"], "test")
        self.assertEqual(report["sealing_status"], "sealed-test")
        self.assertEqual(
            report["control_dependency"]["artifact_kind"], "test-control"
        )
        self.assertEqual(locator["artifact_kind"], "evaluation")

    def test_generation_wrapper_rejects_dev_test_control_kind_swaps(self):
        _, _, test_context_dependency, _ = self.workspace.simple(
            kind="test-context", prefix="tctx-", label="swapped-test-context"
        )
        _, _, test_control_dependency, _ = self.workspace.simple(
            kind="test-control", prefix="tctl-", label="swapped-test-control"
        )
        sealed_with_dev_control, _ = self._generation(
            split="test",
            scope="formal",
            scientific_eligible=True,
            context_dependency=test_context_dependency,
            control_dependency=self.control_dependency,
            label="sealed-with-dev-control",
        )
        dev_with_test_control, _ = self._generation(
            control_dependency=test_control_dependency,
            label="dev-with-test-control",
        )
        formal_registry = copy.deepcopy(self.registry_report)
        formal_registry["registry_scope"] = "formal"
        formal_registry["scientific_eligible"] = True
        formal_registry["models"][0]["scientific_eligible"] = True
        cases = (
            (sealed_with_dev_control, "test", True, formal_registry),
            (dev_with_test_control, "dev", False, self.registry_report),
        )
        for generation_ref, split, sealed, registry_report in cases:
            with self.subTest(split=split), mock.patch.object(
                artifacts,
                "validate_upstream_generation_target",
                self._upstream_report,
            ), mock.patch.object(
                artifacts,
                "validate_registry_target",
                return_value=copy.deepcopy(registry_report),
            ), self.assertRaisesRegex(Stage1ArtifactError, "control kind"):
                build_free_evaluation_artifact(
                    generation_run_ref=generation_ref,
                    evaluation_profile=REPOSITORY_ROOT
                    / "config/stage1/evaluation_strict.json",
                    write_ref=self.workspace.refs / f"bad-{split}.evaluation.ref.json",
                    split=split,
                    sealed=sealed,
                    target_root=self.workspace.targets,
                    workspace_root=self.root,
                )


class FactorialAnalysisLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.workspace = ArtifactWorkspace(self.root)
        _, _, self.plan_dependency, _ = self.workspace.simple(
            kind="training-plan", prefix="plan-", label="analysis-plan"
        )
        _, _, self.model_dependency, _ = self.workspace.simple(
            kind="stage1-model", prefix="mdl-", label="analysis-model"
        )
        self.registry_locator, self.registry_target, self.registry_dependency, self.registry_ref = (
            self.workspace.simple(
                kind="stage1-model-registry", prefix="mreg-", label="analysis-registry"
            )
        )
        _, _, self.context_dependency, _ = self.workspace.simple(
            kind="context", prefix="ctx-", label="analysis-context"
        )
        _, _, self.control_dependency, _ = self.workspace.simple(
            kind="control", prefix="ctl-", label="analysis-control"
        )
        _, _, self.cf_dependency, _ = self.workspace.simple(
            kind="counterfactual", prefix="cf-", label="analysis-cf"
        )
        self.model_key = "M_legacy/smoke"
        self.registry_report = {
            "model_registry_id": self.registry_dependency["artifact_id"],
            "registry_scope": "engineering-smoke",
            "scientific_eligible": False,
            "training_plan_dependency": self.plan_dependency,
            "models": [
                {
                    "model_key": self.model_key,
                    "role": "M_legacy",
                    "seed": None,
                    "model_dependency": self.model_dependency,
                    "scientific_eligible": False,
                }
            ],
            "payload_manifest_sha256": self.registry_dependency[
                "payload_manifest_sha256"
            ],
        }
        self.evaluation_locator, self.evaluation_target, _, self.evaluation_ref = (
            self.workspace.simple(kind="evaluation", prefix="eval-", label="analysis-evaluation")
        )
        self.margin_locator, self.margin_target, _, self.margin_ref = self.workspace.simple(
            kind="margin", prefix="mgn-", label="analysis-margin"
        )
        self.query_ids = ["1", "2"]
        self.evaluation_report = self._evaluation_report()
        self.margin_report = self._margin_report()

    def tearDown(self):
        self.temporary.cleanup()

    def _evaluation_report(self) -> dict[str, object]:
        gold = [quad("甲", "论点")]
        records = {}
        for condition in artifacts.CONDITIONS:
            records[condition] = [
                evaluate_query(
                    query_id=query_id,
                    condition=condition,
                    raw_output=serialize_quadruples(gold),
                    gold=gold,
                    content_sha256=("a" if query_id == "1" else "b") * 64,
                    gold_sha256="c" * 64,
                    prompt_sha256="d" * 64,
                    context_record_sha256=("e" if query_id == "1" else "f") * 64,
                )
                for query_id in self.query_ids
            ]
        return {
            "evaluation_id": self.evaluation_locator["artifact_id"],
            "scope": "engineering-smoke",
            "model_key": self.model_key,
            "role": "M_legacy",
            "seed": None,
            "split": "dev",
            "sealing_status": "unsealed-dev",
            "scientific_eligible": False,
            "conditions": list(artifacts.CONDITIONS),
            "query_ids": self.query_ids,
            "records": records,
            "registry_dependency": self.registry_dependency,
            "training_plan_dependency": self.plan_dependency,
            "context_dependency": self.context_dependency,
            "control_dependency": self.control_dependency,
            "payload_manifest_sha256": self.evaluation_locator[
                "payload_manifest_sha256"
            ],
        }

    def _margin_report(self) -> dict[str, object]:
        records = {}
        for condition in artifacts.CONDITIONS:
            rows = []
            for query_id in self.query_ids:
                for field in artifacts.FIELDS:
                    rows.append({"id": query_id, "field": field, "margin_mean": 0.0})
            records[condition] = rows
        return {
            "margin_run_id": self.margin_locator["artifact_id"],
            "scope": "engineering-smoke",
            "model_key": self.model_key,
            "role": "M_legacy",
            "seed": None,
            "split": "dev",
            "sealing_status": "unsealed-dev",
            "scientific_eligible": False,
            "conditions": list(artifacts.CONDITIONS),
            "query_ids": self.query_ids,
            "field_masks": {field: [True, True] for field in artifacts.FIELDS},
            "records": records,
            "registry_dependency": self.registry_dependency,
            "training_plan_dependency": self.plan_dependency,
            "context_dependency": self.context_dependency,
            "control_dependency": self.control_dependency,
            "cf_dependency": self.cf_dependency,
            "payload_manifest_sha256": self.margin_locator["payload_manifest_sha256"],
        }

    def _patches(self):
        def registry_ref(*_args, **_kwargs):
            return (
                copy.deepcopy(self.registry_locator),
                copy.deepcopy(self.registry_report),
                self.registry_target,
            )

        def evaluation_ref(*_args, **_kwargs):
            return (
                copy.deepcopy(self.evaluation_locator),
                copy.deepcopy(self.evaluation_report),
                self.evaluation_target,
            )

        def margin_ref(*_args, **_kwargs):
            return (
                copy.deepcopy(self.margin_locator),
                copy.deepcopy(self.margin_report),
                self.margin_target,
            )

        return mock.patch.multiple(
            artifacts,
            validate_registry_ref=registry_ref,
            validate_registry_target=mock.Mock(return_value=copy.deepcopy(self.registry_report)),
            validate_evaluation_ref=evaluation_ref,
            validate_margin_ref=margin_ref,
            validate_evaluation_target=mock.Mock(return_value=copy.deepcopy(self.evaluation_report)),
            validate_margin_target=mock.Mock(return_value=copy.deepcopy(self.margin_report)),
        )

    def test_factorial_target_replays_existing_reducers_and_is_idempotent(self):
        run_map = {
            "schema_version": "stage1-run-ref-map/v1",
            "entries": [
                {
                    "model_key": self.model_key,
                    "evaluation_ref": str(self.evaluation_ref),
                    "margin_ref": str(self.margin_ref),
                }
            ],
        }
        analysis_ref = self.workspace.refs / "analysis.ref.json"
        with self._patches():
            first = build_factorial_analysis_artifact(
                model_registry_ref=self.registry_ref,
                run_ref_map=run_map,
                analysis_profile=REPOSITORY_ROOT / "config/stage1/analysis_smoke.json",
                decision_register=REPOSITORY_ROOT / "config/stage1/decision_register.json",
                write_ref=analysis_ref,
                split="dev",
                mode="engineering-smoke",
                n_bootstrap=200,
                seed=42,
                sealed=False,
                target_root=self.workspace.targets,
                workspace_root=self.root,
            )
            upstream_hashes = {
                path: validate_payload_manifest(path)
                for path in (self.registry_target, self.evaluation_target, self.margin_target)
            }
            second = build_factorial_analysis_artifact(
                model_registry_ref=self.registry_ref,
                run_ref_map=run_map,
                analysis_profile=REPOSITORY_ROOT / "config/stage1/analysis_smoke.json",
                decision_register=REPOSITORY_ROOT / "config/stage1/decision_register.json",
                write_ref=analysis_ref,
                split="dev",
                target_root=self.workspace.targets,
                workspace_root=self.root,
            )
            report = validate_analysis_ref(analysis_ref, workspace_root=self.root)
        self.assertEqual(first, second)
        self.assertEqual(report["query_count"], 2)
        for path, payload_hash in upstream_hashes.items():
            self.assertEqual(validate_payload_manifest(path), payload_hash)
        target = Path(first["target_path"])
        factorial = load_json(target / "factorial.json")
        for endpoint in artifacts.BEHAVIOR_ENDPOINTS:
            points = factorial["roles"]["M_legacy"][endpoint]
            self.assertTrue(all(value == 0.0 for value in points["factorial_points"].values()))
            self.assertTrue(all(value == 0.0 for value in points["placebo_points"].values()))
        hard_flip = factorial["flips"]["M_legacy"][self.model_key]["CL-C0"][
            "tuple/hard"
        ]
        self.assertEqual(hard_flip["n11"], 2)
        self.assertEqual(hard_flip["net_flip"], 0.0)
        gate = load_json(target / "scientific_gate.json")
        self.assertFalse(gate["confirmatory_gate_emitted"])
        id_inputs = load_json(target / "analysis.meta.json")["id_inputs"]
        self.assertNotIn("bootstrap_draws", id_inputs)
        self.assertNotIn("effects", id_inputs)
        self.assertNotIn("scientific_gate", id_inputs)

    def test_factorial_requires_exact_registry_order_and_profile_assertions(self):
        bad_map = {
            "schema_version": "stage1-run-ref-map/v1",
            "entries": [
                {
                    "model_key": "unexpected/model",
                    "evaluation_ref": str(self.evaluation_ref),
                    "margin_ref": str(self.margin_ref),
                }
            ],
        }
        with self._patches(), self.assertRaisesRegex(Stage1ArtifactError, "exactly cover"):
            build_factorial_analysis_artifact(
                model_registry_ref=self.registry_ref,
                run_ref_map=bad_map,
                analysis_profile=REPOSITORY_ROOT / "config/stage1/analysis_smoke.json",
                decision_register=REPOSITORY_ROOT / "config/stage1/decision_register.json",
                write_ref=self.workspace.refs / "bad-analysis.ref.json",
                split="dev",
                target_root=self.workspace.targets,
                workspace_root=self.root,
            )

    def test_confirmatory_gate_is_json_round_trip_stable(self):
        models = []
        evaluations = {}
        margins = {}
        for seed in (42, 43, 44):
            for role in ("M_LD", "M_drop"):
                key = f"{role}/seed-{seed}"
                models.append({"model_key": key, "role": role, "seed": seed})
                evaluation = copy.deepcopy(self.evaluation_report)
                evaluation.update({"model_key": key, "role": role, "seed": seed})
                margin = copy.deepcopy(self.margin_report)
                margin.update({"model_key": key, "role": role, "seed": seed})
                evaluations[key] = evaluation
                margins[key] = margin
        profile = load_json(REPOSITORY_ROOT / "config/stage1/analysis_confirmatory.json")
        profile["bootstrap"]["replicates"] = 2
        inputs = {
            "registry": {"models": models},
            "evaluations": evaluations,
            "margins": margins,
            "field_masks": {field: [True, True] for field in artifacts.FIELDS},
        }
        _, _, gate = artifacts._run_reducers(inputs, profile)
        self.assertTrue(gate["confirmatory_gate_emitted"])
        self.assertEqual(json.loads(canonical_json_bytes(gate)), gate)

    def test_compact_bootstrap_sufficient_stats_equal_full_metric_reducer(self):
        records = copy.deepcopy(self.evaluation_report["records"]["C0"])
        records[1] = evaluate_query(
            query_id="2",
            condition="C0",
            raw_output="not-json",
            gold=[quad("甲", "论点")],
            content_sha256="b" * 64,
            gold_sha256="c" * 64,
            prompt_sha256="d" * 64,
            context_record_sha256="f" * 64,
        )
        summary = aggregate_query_metrics(records)
        for endpoint in artifacts.BEHAVIOR_ENDPOINTS:
            expected = summary
            for component in endpoint.split("/"):
                expected = expected[component]
            actual = artifacts._reduce_endpoint_values(
                artifacts._endpoint_values(records, endpoint), endpoint
            )
            self.assertAlmostEqual(actual, float(expected))


class MarginBootstrapTests(unittest.TestCase):
    def test_shared_master_frame_and_zero_effect_degeneracy_policy(self):
        result = fixed_seed_margin_bootstrap(
            seed_cells={
                42: {"CL": [1.0, 2.0, 3.0], "C0": [1.0, 2.0, 3.0]},
                43: {"CL": [4.0, 5.0, 6.0], "C0": [4.0, 5.0, 6.0]},
            },
            left_condition="CL",
            right_condition="C0",
            eligibility_mask=[True, True, True],
            family="gold_margin_total_effect",
            bootstrap_seed=42,
            replicates=10,
        )
        self.assertEqual(result["raw"]["point"], 0.0)
        self.assertEqual(result["d_z"]["point"], 0.0)
        self.assertTrue(result["shared_query_draw_within_family"])
        self.assertEqual(result["fixed_seeds"], [42, 43])

    def test_margin_mask_is_prefrozen_and_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "at least two eligible"):
            fixed_seed_margin_bootstrap(
                seed_cells={42: {"CL": [1.0, 2.0], "C0": [0.0, 0.0]}},
                left_condition="CL",
                right_condition="C0",
                eligibility_mask=[True, False],
                family="gold_margin_total_effect",
                replicates=2,
            )


class MarginArtifactProtocolTests(unittest.TestCase):
    def test_strict_expected_margin_protocol_round_trip(self):
        from metrics import stage1_margin_lifecycle as margin_lifecycle

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workspace = ArtifactWorkspace(root)
            _, _, plan_dependency, _ = workspace.simple(
                kind="training-plan", prefix="plan-", label="margin-plan"
            )
            _, _, model_dependency, _ = workspace.simple(
                kind="stage1-model", prefix="mdl-", label="margin-model"
            )
            _, registry_target, registry_dependency, _ = workspace.simple(
                kind="stage1-model-registry", prefix="mreg-", label="margin-registry"
            )
            context_id = "ctx-" + canonical_sha256({"fixture": "margin-context"})
            context_parent = workspace.targets / "contexts"
            context_target = context_parent / context_id
            context_staging = new_staging_directory(context_parent, context_id)
            gold = [quad("甲", "论点")]
            context_rows = []
            for query_id, content, atom in (("1", "内容一", "a"), ("2", "内容二", "b")):
                context_rows.append(
                    {
                        "query": {
                            "id": query_id,
                            "content": content,
                            "gold": copy.deepcopy(gold),
                            "content_sha256": artifacts._hash_text(content),
                            "gold_sha256": "c" * 64,
                        },
                        "record_sha256": atom * 64,
                    }
                )
            artifacts._write_ordered_jsonl(
                context_staging / "context_manifest.dev.jsonl", context_rows
            )
            context_hash = finalize_target_atomic(context_staging, context_target)
            context_ref = workspace.refs / "margin-context.ref.json"
            context_locator = write_locator_ref(
                context_ref,
                artifact_kind="context",
                artifact_id=context_id,
                target=context_target,
                payload_manifest_sha256=context_hash,
            )
            context_dependency = portable_dependency(context_locator, context_target, root)

            control_id = "ctl-" + canonical_sha256({"fixture": "margin-control"})
            control_parent = workspace.targets / "control_targets"
            control_target = control_parent / control_id
            control_staging = new_staging_directory(control_parent, control_id)
            write_canonical_json(control_staging / "context_ref.json", context_dependency)
            write_canonical_json(control_staging / "fixture.json", {"split": "dev"})
            control_hash = finalize_target_atomic(control_staging, control_target)
            control_ref = workspace.refs / "margin-control.ref.json"
            control_locator = write_locator_ref(
                control_ref,
                artifact_kind="control",
                artifact_id=control_id,
                target=control_target,
                payload_manifest_sha256=control_hash,
            )
            control_dependency = portable_dependency(control_locator, control_target, root)

            cf_id = "cf-" + canonical_sha256({"fixture": "margin-cf"})
            cf_parent = workspace.targets / "counterfactuals"
            cf_target = cf_parent / cf_id
            cf_staging = new_staging_directory(cf_parent, cf_id)
            write_canonical_json(cf_staging / "context_ref.json", context_dependency)
            cf_rows = []
            candidates = {
                "target": "乙",
                "argument": "观点",
                "targeted_group": ["Sexism"],
                "hateful": "non-hate",
            }
            for context_row in context_rows:
                query = context_row["query"]
                for field in artifacts.FIELDS:
                    selected_cf_id = "cf:v1:" + canonical_sha256(
                        {"query_id": query["id"], "tuple_index": 0, "field": field}
                    )
                    cf_rows.append(
                        {
                            "query_id": query["id"],
                            "tuple_index": 0,
                            "field": field,
                            "context_record_sha256": context_row["record_sha256"],
                            "gold_sha256": query["gold_sha256"],
                            "gold_value": copy.deepcopy(gold[0][field]),
                            "construction_status": "ok",
                            "selected_cf_id": selected_cf_id,
                            "candidates": [
                                {
                                    "candidate_id": selected_cf_id,
                                    "value": copy.deepcopy(candidates[field]),
                                }
                            ],
                            "record_sha256": canonical_sha256(
                                {
                                    "query_id": query["id"],
                                    "tuple_index": 0,
                                    "field": field,
                                    "selected_cf_id": selected_cf_id,
                                }
                            ),
                        }
                    )
            artifacts._write_ordered_jsonl(
                cf_staging / "cf_manifest.dev.jsonl", cf_rows
            )
            cf_hash = finalize_target_atomic(cf_staging, cf_target)
            cf_ref = workspace.refs / "margin-cf.ref.json"
            cf_locator = write_locator_ref(
                cf_ref,
                artifact_kind="counterfactual",
                artifact_id=cf_id,
                target=cf_target,
                payload_manifest_sha256=cf_hash,
            )
            cf_dependency = portable_dependency(cf_locator, cf_target, root)

            model_key = "M_legacy/smoke"
            profile = load_json(REPOSITORY_ROOT / "config/stage1/margin_scorer.json")
            masks = {field: [True, True] for field in artifacts.FIELDS}
            frame = artifacts._context_frame(context_target, split="dev")
            cf_by_key = {
                (row["query_id"], row["tuple_index"], row["field"]): row
                for row in cf_rows
            }
            score_audit = {
                "token_ids": [2],
                "token_count": 1,
                "character_span": [0, 1],
                "token_span": [1, 2],
                "sum_logprob": -1.0,
                "mean_logprob": -1.0,
                "left_boundary_crossing": False,
                "right_boundary_crossing": False,
            }
            rows_by_condition = {}
            tuple_rows_by_condition = {}
            for condition in artifacts.CONDITIONS:
                rows = []
                tuple_rows = []
                for context_row in context_rows:
                    for field in artifacts.FIELDS:
                        query_id = context_row["query"]["id"]
                        cf_row = cf_by_key[(query_id, 0, field)]
                        tuple_rows.append(
                            {
                                "schema_version": "stage1-margin-tuple-score/v1",
                                "id": query_id,
                                "condition": condition,
                                "tuple_index": 0,
                                "field": field,
                                "model_key": model_key,
                                "status": "ok",
                                "selected_cf_id": cf_row["selected_cf_id"],
                                "gold": copy.deepcopy(score_audit),
                                "counterfactual": copy.deepcopy(score_audit),
                                "mean_margin": 0.0,
                                "sum_margin_sensitivity": 0.0,
                                "content_sha256": context_row["query"]["content_sha256"],
                                "gold_sha256": context_row["query"]["gold_sha256"],
                                "context_record_sha256": context_row["record_sha256"],
                                "cf_record_sha256": cf_row["record_sha256"],
                            }
                        )
                        rows.append(
                            {
                                "schema_version": "stage1-margin-record/v1",
                                "id": context_row["query"]["id"],
                                "condition": condition,
                                "field": field,
                                "model_key": model_key,
                                "status": "ok",
                                "margin_mean": 0.0,
                                "content_sha256": context_row["query"]["content_sha256"],
                                "gold_sha256": context_row["query"]["gold_sha256"],
                                "context_record_sha256": context_row["record_sha256"],
                                "cf_record_sha256": canonical_sha256(
                                    [cf_row["record_sha256"]]
                                ),
                            }
                        )
                rows_by_condition[condition] = rows
                tuple_rows_by_condition[condition] = tuple_rows
            dependencies = {
                "registry": registry_dependency,
                "training_plan": plan_dependency,
                "model": model_dependency,
                "context": context_dependency,
                "control": control_dependency,
                "cf": cf_dependency,
            }
            construction = {
                field: {
                    "constructed_tuple_count": 2,
                    "total_gold_tuple_count": 2,
                    "tuple_rate": 1.0,
                    "complete_case_query_count": 2,
                    "total_query_count": 2,
                    "complete_case_query_rate": 1.0,
                }
                for field in artifacts.FIELDS
            }
            scoring = {
                field: {
                    "expected_eligible_tuple_condition_cells": 12,
                    "observed_scored_tuple_condition_cells": 12,
                    "tuple_condition_rate": 1.0,
                    "expected_eligible_query_condition_cells": 12,
                    "observed_scored_query_condition_cells": 12,
                    "query_condition_rate": 1.0,
                }
                for field in artifacts.FIELDS
            }
            executor = {
                "executor_id": "engineering-deterministic/v1",
                "executor_revision": "stage1-margin-test-executor/v1",
                "backend": "local-huggingface-causal-lm",
                "scientific_eligible": False,
            }
            batch_contract = margin_lifecycle._batch_contract(profile)
            runtime_contract = {
                "schema_version": "stage1-margin-runtime-contract/v1",
                "executor": executor,
                "python_implementation": "CPython",
                "python_version": "3.11.0",
                "torch_version": "2.0.0",
                "transformers_version": "4.0.0",
                "peft_version": None,
                "accelerate_version": None,
                "cuda_runtime_version": None,
                "cuda_driver_version": None,
                "gpu_architectures": [],
                "batch_contract": batch_contract,
                "numeric_contract": {
                    "log_softmax_dtype": "float32",
                    "right_padding": True,
                    "position_ids": "attention-mask-cumsum-minus-one-zero-on-padding/v1",
                    "causal_shift": "logits-minus-last-vs-input-minus-first",
                },
            }
            resolved_model = SimpleNamespace(
                registry_id=registry_dependency["artifact_id"],
                model_artifact_id=model_dependency["artifact_id"],
                model_key=model_key,
                role="M_legacy",
                seed=None,
                scientific_eligible=False,
                tokenizer_path=root,
                tokenizer_revision="fixture-tokenizer",
                tokenizer_content_revision="fixture-tokenizer-content",
            )
            contracts = margin_lifecycle._contract_hashes(
                profile=profile,
                handle=resolved_model,
                runtime_contract=runtime_contract,
            )
            id_inputs = {
                "schema_version": "stage1-margin-id-inputs/v1",
                "training_plan_dependency": plan_dependency,
                "model_registry_dependency": registry_dependency,
                "model_key": model_key,
                "model_dependency": model_dependency,
                "context_dependency": context_dependency,
                "control_dependency": control_dependency,
                "cf_dependency": cf_dependency,
                "split": "dev",
                "sealing_status": "unsealed-dev",
                "ordered_conditions": list(artifacts.CONDITIONS),
                "expected_ordered_query_ids_sha256": canonical_sha256(["1", "2"]),
                "scorer_profile_sha256": canonical_sha256(profile),
                "eligibility_mask_schema": "stage1-prefrozen-field-mask/v1",
                "eligibility_mask_sha256": canonical_sha256(masks),
                **contracts,
            }
            margin_id = "mgn-" + canonical_sha256(id_inputs)
            contrast_rows = margin_lifecycle._contrast_rows(
                model_key=model_key,
                context_frame=frame,
                masks=masks,
                rows_by_condition=rows_by_condition,
            )
            calibration_rows = [
                {
                    "schema_version": "stage1-margin-batch-calibration-record/v1",
                    "id": query_id,
                    "condition": condition,
                    "tuple_index": 0,
                    "field": field,
                    "gold_token_ids": [2],
                    "counterfactual_token_ids": [2],
                    "batched_gold_mean_logprob": -1.0,
                    "batched_counterfactual_mean_logprob": -1.0,
                    "singleton_gold_mean_logprob": -1.0,
                    "singleton_counterfactual_mean_logprob": -1.0,
                    "batched_margin": 0.0,
                    "singleton_margin": 0.0,
                    "abs_margin_delta": 0.0,
                    "compared_token_count": 2,
                }
                for condition in artifacts.CONDITIONS
                for query_id in ("1", "2")
                for field in artifacts.FIELDS
            ]
            calibration_hash = canonical_sha256(calibration_rows)
            meta = {
                "schema_version": "stage1-margin-run/v1",
                "margin_run_id": margin_id,
                "model_key": model_key,
                "split": "dev",
                "sealing_status": "unsealed-dev",
                "scientific_eligible": False,
                "ordered_conditions": list(artifacts.CONDITIONS),
                "ordered_query_ids": ["1", "2"],
                "ordered_query_ids_sha256": canonical_sha256(["1", "2"]),
                "field_eligibility_masks": masks,
                "paired_block_complete": True,
                "expected_row_count_per_condition": 8,
                "observed_row_count_per_condition": {
                    condition: 8 for condition in artifacts.CONDITIONS
                },
                "expected_tuple_score_count_per_condition": 8,
                "observed_tuple_score_count_per_condition": {
                    condition: 8 for condition in artifacts.CONDITIONS
                },
                "construction_coverage": construction,
                "scoring_coverage": scoring,
                "per_condition_sufficient_stats_sha256": {
                    condition: canonical_sha256(rows)
                    for condition, rows in rows_by_condition.items()
                },
                "per_condition_tuple_scores_sha256": {
                    condition: canonical_sha256(rows)
                    for condition, rows in tuple_rows_by_condition.items()
                },
                "query_contrasts_sha256": canonical_sha256(contrast_rows),
                "batch_calibration": {
                    "schema_version": "stage1-margin-batch-calibration/v1",
                    "passed": True,
                    "traversal_policy": margin_lifecycle.CALIBRATION_TRAVERSAL,
                    "requested_pair_count": margin_lifecycle.CALIBRATION_PAIR_TARGET,
                    "observed_pair_count": len(calibration_rows),
                    "requested_pair_count_reached": False,
                    "calibration_records_sha256": calibration_hash,
                    "tolerance_floor": margin_lifecycle.CALIBRATION_TOLERANCE_FLOOR,
                    "tolerance_multiplier": margin_lifecycle.CALIBRATION_TOLERANCE_MULTIPLIER,
                    "tolerance_hard_cap": margin_lifecycle.CALIBRATION_TOLERANCE_HARD_CAP,
                    "observed_max_abs_margin_delta": 0.0,
                    "frozen_tolerance": margin_lifecycle.CALIBRATION_TOLERANCE_FLOOR,
                    "compared_token_count": 2 * len(calibration_rows),
                },
                "batch_calibration_records_sha256": calibration_hash,
                "batch_contract": batch_contract,
                "executor": executor,
                "runtime_contract_sha256": canonical_sha256(runtime_contract),
                "id_inputs": id_inputs,
            }
            summary = margin_lifecycle._summary(
                model_key=model_key,
                context_frame=frame,
                masks=masks,
                rows_by_condition=rows_by_condition,
                contrast_rows=contrast_rows,
                construction=construction,
                scoring=scoring,
            )
            provenance = {
                "schema_version": "stage1-margin-provenance/v1",
                "margin_run_id": margin_id,
                "dependencies": dependencies,
                "model_key": model_key,
                "model_resolver": {
                    "protocol": "model.stage1_registry.resolve_registered_model_dependency/v1",
                    "accepted_source": "embedded-stage1-model-registry-dependency-only",
                    "direct_model_path_allowed": False,
                },
                "executor": executor,
                "scorer_profile_sha256": canonical_sha256(profile),
                "runtime_contract_sha256": canonical_sha256(runtime_contract),
                "batch_calibration_contract_sha256": contracts[
                    "batch_calibration_contract_sha256"
                ],
                "scorer_code_sha256": contracts["scorer_code_sha256"],
                "observed_margins_in_lifecycle_id": False,
                "upstream_mutation_allowed": False,
            }
            margin_parent = workspace.targets / "margin_runs"
            margin_target = margin_parent / margin_id
            staging = new_staging_directory(margin_parent, margin_id)
            for filename, value in (
                ("model_registry_ref.json", registry_dependency),
                ("training_plan_ref.json", plan_dependency),
                ("model_ref.json", model_dependency),
                ("context_ref.json", context_dependency),
                ("control_ref.json", control_dependency),
                ("cf_ref.json", cf_dependency),
                ("scorer_profile.resolved.json", profile),
                ("runtime_contract.json", runtime_contract),
                ("margin.meta.json", meta),
                ("provenance.json", provenance),
                ("summary.json", summary),
            ):
                write_canonical_json(staging / filename, value)
            for condition, rows in rows_by_condition.items():
                artifacts._write_ordered_jsonl(
                    staging / "margins" / f"{condition}.jsonl", rows
                )
                artifacts._write_ordered_jsonl(
                    staging / "tuple_scores" / f"{condition}.jsonl",
                    tuple_rows_by_condition[condition],
                )
            artifacts._write_ordered_jsonl(
                staging / "query_contrasts.jsonl", contrast_rows
            )
            artifacts._write_ordered_jsonl(
                staging / "batch_calibration.jsonl", calibration_rows
            )
            finalize_target_atomic(staging, margin_target)
            registry_report = {
                "model_registry_id": registry_dependency["artifact_id"],
                "registry_scope": "engineering-smoke",
                "scientific_eligible": False,
                "training_plan_dependency": plan_dependency,
                "models": [
                    {
                        "model_key": model_key,
                        "role": "M_legacy",
                        "seed": None,
                        "model_dependency": model_dependency,
                        "scientific_eligible": False,
                    }
                ],
                "payload_manifest_sha256": registry_dependency[
                    "payload_manifest_sha256"
                ],
            }
            with mock.patch.object(
                artifacts,
                "validate_registry_target",
                return_value=registry_report,
            ), mock.patch.object(
                artifacts,
                "validate_cf_target",
                return_value={
                    "split": "dev",
                    "scientific_eligible": False,
                },
            ), mock.patch(
                "data.build_context_manifest.validate_context_target",
                return_value={
                    "split": "dev",
                    "scientific_eligible": False,
                    "budget": {"tokenizer_revision": "fixture-tokenizer"},
                },
            ), mock.patch(
                "data.control_manifest.validate_control_target",
                return_value={"split": "dev"},
            ):
                report = artifacts.validate_margin_target(
                    margin_target,
                    workspace_root=root,
                    model_dependency_resolver=lambda **_kwargs: resolved_model,
                    tokenizer_loader=lambda _path: object(),
                )
            self.assertEqual(report["query_ids"], ["1", "2"])
            self.assertEqual(report["field_masks"], masks)
            self.assertEqual(report["records"]["C0"][0]["margin_mean"], 0.0)


if __name__ == "__main__":
    unittest.main()
