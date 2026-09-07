"""Registration-only checks without GPU, query gold or frozen artifact edits."""

import copy
import unittest

from diagnostics import general_model_nolabel as coverage
from data.stage1_data import canonical_json_sha256
from diagnostics.general_model_numeric_analysis import candidate_catalog
from diagnostics.general_model_package import PackageError, read_json


class CoverageRegistrationTests(unittest.TestCase):
    def setUp(self):
        self.config = read_json(coverage.DEFAULT_CONFIG)
        self.original = {"config": read_json(coverage.ROOT / "config/stage1/general_model_ld_numeric_v2.json")}

    def test_registered_config_and_resolved_true_batch_one(self):
        coverage.validate_config(self.config)
        resolved = coverage.resolved_config(self.config, self.original)
        self.assertEqual(resolved["runtime"]["dtype"], "float32")
        self.assertEqual(resolved["runtime"]["accelerated_batch_size"], 1)
        self.assertEqual(resolved["runtime"]["padding_policy"], "dynamic")
        self.assertEqual(resolved["runtime"]["batch_geometry"]["group"], "one-candidate-per-forward")
        self.assertEqual(len(resolved["analysis"]["primary_comparisons"]), 2)
        self.assertEqual(len(resolved["analysis"]["secondary_comparisons"]), 5)
        self.assertEqual(resolved["analysis"]["bootstrap"]["repetitions"], 10000)
        self.assertEqual(resolved["analysis"]["lexicon_strata"], ["Lq_hit", "Lq_no_hit"])

    def test_rejects_data_and_profile_overrides(self):
        mutations = (("test_access", True), ("query_gold_in_scoring", True),
                     ("conditions", ["C0", "CL", "CD", "CLD", "PL", "PD"]),
                     ("runtime", {}), ("model_key", "qwen3-14b"))
        for key, value in mutations:
            with self.subTest(key=key):
                config = copy.deepcopy(self.config)
                config[key] = value
                with self.assertRaises(PackageError):
                    coverage.validate_config(config)

    def test_rejects_tolerance_recalibration_or_batch_change(self):
        mutations = (("preflight", "epsilon", 0.005), ("preflight", "epsilon_recalibration", True),
                     ("execution", "production_batch_size", 4),
                     ("execution", "production_batch_size", True),
                     ("execution", "automatic_profile_search", True),
                     ("resource", "source_correspondence_visible", True))
        for section, key, value in mutations:
            with self.subTest(section=section, key=key):
                config = copy.deepcopy(self.config)
                config[section][key] = value
                with self.assertRaises(PackageError):
                    coverage.validate_config(config)

    def test_parent_is_not_mutated(self):
        before = copy.deepcopy(self.original)
        coverage.resolved_config(self.config, self.original)
        self.assertEqual(self.original, before)


class CoverageMatrixTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frame = [{"query_id": str(i), "lex_hit": i < 223} for i in range(643)]
        cls.cohorts = {"regression": [str(i) for i in range(8)],
                       "validation": [str(i) for i in range(8, 32)],
                       "boundary": ["32", "33", "34", "35"]}
        cls.catalog = candidate_catalog()
        for rows in cls.catalog.values():
            for row in rows:
                row["answer_tokens"] = 5
        cls.contexts = []
        for task in ("hate", "group"):
            for condition in coverage.CONDITIONS:
                for entry in cls.frame:
                    qid = entry["query_id"]
                    row = {"record_id": f"{qid}:{task}:{condition}", "query_id": qid,
                           "task": task, "condition": condition, "prompt_tokens": 10,
                           "prompt_sha256": "fixture", "prompt_token_ids_sha256": "fixture",
                           "overflow": False, "control_valid": True}
                    row["context_sha256"] = canonical_json_sha256(row)
                    cls.contexts.append(row)

    def test_accepts_exact_full_matrix_and_original_Lq_frame(self):
        rows = coverage.validate_matrix(self.contexts, self.frame, self.cohorts, self.catalog)
        self.assertEqual(len(rows), 7716)
        self.assertEqual(set(rows[0]), set(coverage.DESCRIPTOR_KEYS))

    def test_rejects_missing_reordered_or_tampered_contexts(self):
        changed = copy.deepcopy(self.contexts)
        changed[0]["prompt_tokens"] += 1
        cases = [self.contexts[:-1], [self.contexts[1], self.contexts[0], *self.contexts[2:]], changed]
        for rows in cases:
            with self.assertRaises(PackageError):
                coverage.validate_matrix(rows, self.frame, self.cohorts, self.catalog)

    def test_cannot_reclassify_hit_frame_by_new_union(self):
        frame = copy.deepcopy(self.frame)
        frame[-1]["lex_hit"] = True
        with self.assertRaises(PackageError):
            coverage.validate_matrix(self.contexts, frame, self.cohorts, self.catalog)

    def test_boundary_cohort_must_be_disjoint_and_bounded(self):
        for boundary in (["0"], ["32", "32"], ["32", "33", "34", "35", "36"], ["999"]):
            with self.subTest(boundary=boundary), self.assertRaises(PackageError):
                coverage.validate_matrix(self.contexts, self.frame,
                                         {**self.cohorts, "boundary": boundary}, self.catalog)

    def test_budget_includes_EOS_and_padding_challenge(self):
        changed = copy.deepcopy(self.contexts)
        changed[0]["prompt_tokens"] = 8192 - 5
        changed[0]["context_sha256"] = canonical_json_sha256(
            {key: value for key, value in changed[0].items() if key != "context_sha256"})
        with self.assertRaises(PackageError):
            coverage.validate_matrix(changed, self.frame, self.cohorts, self.catalog)


if __name__ == "__main__":
    unittest.main()
