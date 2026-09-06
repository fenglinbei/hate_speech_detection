from __future__ import annotations

import copy
import inspect
import json
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPOSITORY_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from build_lex.terminology_candidate_run_plan import (  # noqa: E402
    CandidateRunPlanError,
    build_current_request,
    build_logical_slot_grid,
    evaluate_g1_deepseek_retention,
    load_bound_public_tasks,
    load_current_run_plan,
    request_equivalence_projection,
    validate_provider_api_base,
    validate_returned_model,
)
from build_lex.terminology_candidate_development_v2 import (  # noqa: E402
    validate_locked_raw_gold,
)
from data.training_artifacts import (  # noqa: E402
    canonical_sha256,
    resolve_locator_ref,
    write_canonical_json,
)


PLAN = REPOSITORY_ROOT / "config/stage1/wp3_candidate_current_run_plan_v2.json"
SCHEMA = REPOSITORY_ROOT / "schemas/wp3_candidate_current_run_plan_v1.schema.json"
FRAME_REF = (
    REPOSITORY_ROOT
    / "exps/causal_context/stage1_p0/wp3_candidate_generators_v2/refs/development_frame_ref.json"
)
RAW_GOLD_REF = (
    REPOSITORY_ROOT
    / "exps/causal_context/stage1_p0/wp3_candidate_generators_v2/refs/s21b_locked_raw_gold_ref.json"
)
LEGACY_CONFIG = REPOSITORY_ROOT / "config/stage1/wp3_candidate_generators_v1.json"
BASE_PROTOCOL = REPOSITORY_ROOT / "config/stage1/wp3_candidate_generators_s21b_v1.json"


class CurrentRunPlanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.plan = load_current_run_plan(
            PLAN,
            workspace_root=REPOSITORY_ROOT,
            schema_path=SCHEMA,
            require_current_sources=True,
        )
        cls.tasks = (
            load_bound_public_tasks(cls.plan, FRAME_REF)
            if FRAME_REF.is_file()
            else None
        )

    def _write_plan(self, root: Path, value: dict) -> Path:
        path = root / "plan.json"
        write_canonical_json(path, value)
        return path

    @staticmethod
    def _rehash(value: dict) -> dict:
        updated = copy.deepcopy(value)
        identity = {key: item for key, item in updated.items() if key != "plan_id"}
        updated["plan_id"] = "wp3runplan-" + canonical_sha256(identity)
        return updated

    def _load_temp(self, path: Path) -> dict:
        return load_current_run_plan(
            path,
            workspace_root=REPOSITORY_ROOT,
            schema_path=SCHEMA,
            require_current_sources=False,
        )

    def test_plan_freezes_glm_deepseek_grid_and_unauthorized_execution(self) -> None:
        self.assertEqual(
            self.plan["providers"]["glm_flash"]["requested_model"],
            "glm-5.3-flash",
        )
        self.assertEqual(
            self.plan["providers"]["deepseek_flash"]["requested_model"],
            "deepseek-v4-flash",
        )
        self.assertNotIn("qwen", json.dumps(self.plan, ensure_ascii=False).lower())
        self.assertEqual(self.plan["g1"]["logical_slot_count"], 1696)
        self.assertEqual(self.plan["g2"]["logical_slot_count"], 848)
        self.assertEqual(self.plan["total_logical_model_slot_count"], 2544)
        self.assertEqual(
            self.plan["g1"]["post_pilot_retention_criteria"],
            {
                "decision_state_policy": (
                    "incomplete-is-inconclusive-else-retain-or-drop/v1"
                ),
                "decision_rule": (
                    "retain-iff-(recall-gain-or-unique-tp)-and-f1-gates/v1"
                ),
                "completeness_scope": (
                    "all-2544-model-slots-plus-g3-and-offset-replay/v1"
                ),
                "metric_contract": (
                    "exact-occurrence-micro-unrounded-absolute-delta/v1"
                ),
                "baseline_union": "glm-g1-plus-dual-g2-plus-g3/v1",
                "added_union": "baseline-plus-deepseek-g1/v1",
                "min_full_union_exact_recall_gain": 0.01,
                "min_full_union_unique_exact_true_positives": 2,
                "min_unique_tp_distinct_cases": 2,
                "full_union_exact_f1_floor": 0.85,
                "max_full_union_exact_f1_drop": 0.005,
                "required_offset_replay_rate": 1,
                "required_terminal_slot_count": 2544,
                "required_terminal_failure_count": 0,
            },
        )
        self.assertTrue(
            all(
                self.plan["execution"][key] is False
                for key in {
                    "model_calls_authorized",
                    "network_calls_authorized",
                    "paid_calls_authorized",
                    "formal_artifact_publication_authorized",
                    "full_fit_execution_authorized",
                    "s22_materialization_authorized",
                }
            )
        )
        self.assertEqual(self.plan["execution"]["runner_status"], "not-implemented")
        self.assertEqual(
            self.plan["execution"]["plan_lifecycle_status"],
            "planning-only-must-be-superseded-before-execution",
        )

    @unittest.skipUnless(FRAME_REF.is_file(), "materialized frame is unavailable")
    def test_two_providers_share_exact_request_parameters(self) -> None:
        assert self.tasks is not None
        task = self.tasks[0]
        for generator, variant in (
            ("g1", "surface_decode"),
            ("g1", "lexical_pragmatic"),
            ("g2", "direct_mention"),
        ):
            glm = build_current_request(
                self.plan,
                self.tasks,
                task_id=task["task_id"],
                generator=generator,
                variant=variant,
                provider_id="glm_flash",
            )
            deepseek = build_current_request(
                self.plan,
                self.tasks,
                task_id=task["task_id"],
                generator=generator,
                variant=variant,
                provider_id="deepseek_flash",
            )
            with self.subTest(generator=generator, variant=variant):
                self.assertEqual(
                    request_equivalence_projection(glm),
                    request_equivalence_projection(deepseek),
                )
                self.assertEqual(glm["model"], "glm-5.3-flash")
                self.assertEqual(deepseek["model"], "deepseek-v4-flash")
                self.assertEqual(glm["temperature"], 1)
                self.assertEqual(glm["top_p"], 0.95)
                self.assertEqual(glm["max_tokens"], 4096)
                self.assertEqual(glm["thinking"], {"type": "enabled"})
                self.assertEqual(glm["reasoning_effort"], "low")
                self.assertFalse(glm["stream"])
                self.assertEqual(glm["response_format"], {"type": "json_object"})

    @unittest.skipUnless(FRAME_REF.is_file(), "materialized frame is unavailable")
    def test_logical_slot_grid_is_complete_and_deterministic(self) -> None:
        assert self.tasks is not None
        first = build_logical_slot_grid(self.plan, self.tasks)
        second = build_logical_slot_grid(self.plan, list(reversed(self.tasks)))
        self.assertEqual(first, second)
        self.assertEqual(len(first), 2544)
        self.assertEqual(len({row["slot_id"] for row in first}), 2544)
        self.assertEqual(
            Counter(row["provider_id"] for row in first),
            {"glm_flash": 1272, "deepseek_flash": 1272},
        )
        self.assertEqual(
            Counter(row["generator"] for row in first), {"g1": 1696, "g2": 848}
        )
        with self.assertRaisesRegex(CandidateRunPlanError, "424 cases"):
            build_logical_slot_grid(self.plan, self.tasks[:-1])

        mutations = []
        changed_content = copy.deepcopy(self.tasks)
        changed_content[0]["content"] += "篡改"
        mutations.append(changed_content)
        changed_alias = copy.deepcopy(self.tasks)
        changed_alias[0]["blind_alias"] += "篡改"
        mutations.append(changed_alias)
        duplicate = copy.deepcopy(self.tasks)
        duplicate[0]["task_id"] = duplicate[1]["task_id"]
        mutations.append(duplicate)
        extra_field = copy.deepcopy(self.tasks)
        extra_field[0]["record_id"] = "forbidden"
        mutations.append(extra_field)
        for index, mutation in enumerate(mutations):
            with self.subTest(index=index), self.assertRaises(
                CandidateRunPlanError
            ):
                build_logical_slot_grid(self.plan, mutation)

    def test_provider_identity_and_origin_are_fail_closed(self) -> None:
        self.assertEqual(
            validate_returned_model(
                self.plan,
                provider_id="glm_flash",
                returned_model=" GLM-5.3-FLASH ",
            ),
            "glm-5.3-flash",
        )
        self.assertEqual(
            validate_provider_api_base(
                self.plan,
                provider_id="deepseek_flash",
                api_base="https://api.deepseek.com/",
            ),
            "https://api.deepseek.com",
        )
        with self.assertRaisesRegex(CandidateRunPlanError, "returned model differs"):
            validate_returned_model(
                self.plan,
                provider_id="glm_flash",
                returned_model="glm-5.3",
            )
        with self.assertRaisesRegex(CandidateRunPlanError, "API base differs"):
            validate_provider_api_base(
                self.plan,
                provider_id="deepseek_flash",
                api_base="https://provider.invalid/v1",
            )
        with self.assertRaisesRegex(CandidateRunPlanError, "API base differs"):
            validate_provider_api_base(
                self.plan,
                provider_id="deepseek_flash",
                api_base="https://api.deepseek.com/v1",
            )

    def test_plan_tampering_and_noncanonical_numbers_fail_closed(self) -> None:
        mutations = []
        qwen = copy.deepcopy(self.plan)
        qwen["providers"]["glm_flash"]["requested_model"] = "Qwen3.8-27B"
        mutations.append(self._rehash(qwen))
        profile = copy.deepcopy(self.plan)
        profile["request_profile"]["temperature"] = 0
        mutations.append(self._rehash(profile))
        missing_g1_provider = copy.deepcopy(self.plan)
        missing_g1_provider["g1"]["providers"] = ["glm_flash"]
        mutations.append(self._rehash(missing_g1_provider))
        endpoint = copy.deepcopy(self.plan)
        endpoint["providers"]["glm_flash"]["chat_completions_endpoint"] = (
            "https://open.bigmodel.cn/api/paas/v4/other"
        )
        mutations.append(self._rehash(endpoint))
        authorized = copy.deepcopy(self.plan)
        authorized["execution"]["model_calls_authorized"] = True
        mutations.append(self._rehash(authorized))
        float_count = copy.deepcopy(self.plan)
        float_count["g1"]["logical_slot_count"] = 1696.0
        mutations.append(self._rehash(float_count))
        numeric_bool = copy.deepcopy(self.plan)
        numeric_bool["request_profile"]["stream"] = 0
        mutations.append(self._rehash(numeric_bool))
        numeric_true = copy.deepcopy(self.plan)
        numeric_true["preserved_raw_gold"]["run_must_not_read"] = 1
        mutations.append(self._rehash(numeric_true))

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for index, mutation in enumerate(mutations):
                path = root / f"plan-{index}.json"
                write_canonical_json(path, mutation)
                with self.subTest(index=index), self.assertRaises(
                    CandidateRunPlanError
                ):
                    self._load_temp(path)

    def test_schema_and_plan_ids_are_hard_bound(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            permissive_schema = root / "schema.json"
            write_canonical_json(
                permissive_schema,
                {
                    "$id": "https://local.invalid/schemas/"
                    "wp3_candidate_current_run_plan_v1.schema.json"
                },
            )
            with self.assertRaisesRegex(CandidateRunPlanError, "schema binding differs"):
                load_current_run_plan(
                    PLAN,
                    workspace_root=REPOSITORY_ROOT,
                    schema_path=permissive_schema,
                    require_current_sources=False,
                )

            wrong_id = copy.deepcopy(self.plan)
            wrong_id["plan_id"] = "wp3runplan-" + "0" * 64
            with self.assertRaisesRegex(CandidateRunPlanError, "ID differs"):
                self._load_temp(self._write_plan(root, wrong_id))

            with self.assertRaises(CandidateRunPlanError):
                load_current_run_plan(
                    PLAN,
                    workspace_root=REPOSITORY_ROOT,
                    schema_path=None,
                    require_current_sources=False,
                )

    @unittest.skipUnless(FRAME_REF.is_file(), "materialized frame is unavailable")
    def test_execution_loader_has_no_raw_gold_input(self) -> None:
        assert self.tasks is not None
        self.assertNotIn(
            "raw_gold_ref", inspect.signature(load_current_run_plan).parameters
        )
        self.assertEqual(len(self.tasks), 424)
        self.assertTrue(
            all(set(task) == {"task_id", "blind_alias", "content"} for task in self.tasks)
        )

        mutated = copy.deepcopy(self.plan)
        mutated["providers"]["glm_flash"]["requested_model"] = "glm-5.3"
        mutated = self._rehash(mutated)
        with self.assertRaises(CandidateRunPlanError):
            build_current_request(
                mutated,
                self.tasks,
                task_id=self.tasks[0]["task_id"],
                generator="g2",
                variant="direct_mention",
                provider_id="glm_flash",
            )
        with self.assertRaisesRegex(CandidateRunPlanError, "not in the frozen"):
            build_current_request(
                self.plan,
                self.tasks,
                task_id="outside-frame",
                generator="g2",
                variant="direct_mention",
                provider_id="glm_flash",
            )

    def test_g1_retention_gate_is_three_state_and_formula_bound(self) -> None:
        common = {
            "terminal_slot_count": 2544,
            "terminal_failure_count": 0,
            "g3_complete": True,
            "offset_replay_count": 600,
            "offset_occurrence_count": 600,
            "gold_mention_count": 356,
            "baseline_exact_tp": 300,
            "baseline_proposal_count": 330,
            "added_exact_tp": 304,
            "added_proposal_count": 340,
            "deepseek_unique_exact_tp": 4,
            "unique_tp_distinct_cases": 3,
        }
        retained = evaluate_g1_deepseek_retention(self.plan, **common)
        self.assertEqual(retained["status"], "retain")
        self.assertGreaterEqual(
            retained["metrics"]["full_union_exact_recall_gain"], 0.01
        )

        incomplete = dict(common, terminal_slot_count=2543)
        self.assertEqual(
            evaluate_g1_deepseek_retention(self.plan, **incomplete)["status"],
            "inconclusive",
        )
        unobserved_replay = dict(
            common, offset_replay_count=0, offset_occurrence_count=0
        )
        unobserved_result = evaluate_g1_deepseek_retention(
            self.plan, **unobserved_replay
        )
        self.assertEqual(unobserved_result["status"], "inconclusive")
        self.assertIn("offset_replay_unobserved", unobserved_result["reasons"])

        unique_branch = dict(
            common,
            added_exact_tp=302,
            added_proposal_count=334,
            deepseek_unique_exact_tp=2,
            unique_tp_distinct_cases=2,
        )
        self.assertEqual(
            evaluate_g1_deepseek_retention(self.plan, **unique_branch)["status"],
            "retain",
        )

        noisy = dict(unique_branch, added_proposal_count=400)
        self.assertEqual(
            evaluate_g1_deepseek_retention(self.plan, **noisy)["status"],
            "drop",
        )

    @unittest.skipUnless(
        FRAME_REF.is_file() and RAW_GOLD_REF.is_file(),
        "active frame/raw gold is unavailable",
    )
    def test_active_raw_gold_remains_v1_and_strictly_valid(self) -> None:
        raw_locator, raw_target = resolve_locator_ref(
            RAW_GOLD_REF, "wp3-s21-locked-raw-gold"
        )
        _, frame_target = resolve_locator_ref(
            FRAME_REF, "wp3-s21-development-frame"
        )
        result = validate_locked_raw_gold(
            raw_target,
            frame_dir=frame_target,
            workspace_root=REPOSITORY_ROOT,
            legacy_generator_config_path=LEGACY_CONFIG,
            protocol_path=BASE_PROTOCOL,
            require_current_sources=True,
            require_current_validator=True,
        )
        self.assertEqual(result["raw_gold_id"], raw_locator["artifact_id"])
        self.assertEqual(
            result["payload_manifest_sha256"],
            raw_locator["payload_manifest_sha256"],
        )


if __name__ == "__main__":
    unittest.main()
