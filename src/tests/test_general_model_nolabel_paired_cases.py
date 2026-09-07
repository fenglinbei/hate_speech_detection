"""Focused CPU checks for paired math, immutable joins and deterministic selection."""

import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from sklearn.metrics import f1_score

from diagnostics import general_model_nolabel_paired_cases as m
from diagnostics.general_model_numeric_analysis import candidate_catalog, candidate_scores, block_analysis


def profile(qid, mask="0011", *, extra=(False, False), group_mask=None, family=None, split=None):
    gold = {"hate": "hate", "group": ["Region"]}
    p = {"schema_version": m.SCHEMA, "query_id": str(qid), "lex_hit": int(qid) % 2 == 0,
         "gold": gold, "conditions": {}, "context_status": "complete",
         "family_key": family or m.family_key("query-" + str(qid)),
         "resources": {"lq_count": 1, "ld_only_count": 1, "union_count": 2,
                       "lq_ids": ["a"], "ld_ids": ["b"], "ld_only_ids": ["b"],
                       "lq_only_ids": ["a"], "intersection_ids": [], "union_ids": ["a", "b"],
                       "demo_ids": ["d"], "demo_match_ids": {"d": ["b"]},
                       "demo_hate_counts": {"hate": 1, "non-hate": 0},
                       "demo_group_counts": {label: int(label == "Region") for label in m.GROUP_LABELS},
                       "missing_reasons": []}}
    flags = dict(zip(m.CORE, (c == "1" for c in mask))) | dict(zip(("CLnew", "CLDnew"), extra))
    gflags = dict(zip(m.CORE, (c == "1" for c in (group_mask or mask)))) | dict(zip(("CLnew", "CLDnew"), extra))
    for c in m.CONDITIONS:
        p["conditions"][c] = {}
        for t in m.TASKS:
            correct = flags[c] if t == "hate" else gflags[c]
            pred = [gold[t] if correct else "non-hate"] if t == "hate" else (gold[t] if correct else [])
            p["conditions"][c][t] = {
                "correct": correct,
                "prediction": {"labels": pred, "ordinal": int(not correct), "top_score_gap": 1.,
                               "tied_top_count": 1, "within_two_epsilon": False},
                "readouts": {"answer_sum/gold/best_nongold_margin": 1. if correct else -1.,
                             "answer_sum/gold/mass": .8 if correct else .2,
                             "answer_sum/gold/nll": .1 if correct else 2.,
                             **{"answer_sum/margin/" + label: 1. for label in (("hate",) if t == "hate" else m.GROUP_LABELS)}},
                "score_mode_predictions": {mode: pred for mode in m.SCORE_MODES}, "score_mode_sensitive": False,
                "pred_group_size": len(pred) if t == "group" else None,
                "group_error_count": len(set(pred) ^ set(gold[t])) if t == "group" else None,
                "context": {"prompt_tokens": 100, "dictionary_tokens": 10, "demo_tokens": 20,
                            "context_sha256": "x", "prompt_sha256": "y", "missing_reason": None}}
    p["core_mask"] = {"hate": mask, "group": group_mask or mask}
    p["six_condition_mask"] = {t: "".join(str(int(p["conditions"][c][t]["correct"])) for c in m.CONDITIONS) for t in m.TASKS}
    p["split"] = split or m.split_family(p["family_key"])
    p["memberships"] = m.memberships(p, .001)
    p["candidate_labels"] = [r["bucket"] + ":" + r["focus_task"] for r in p["memberships"]]
    return p


def raw_candidates(task, winner=0):
    rows = candidate_catalog()[task]
    for row in rows:
        value = -1. if row["ordinal"] == winner else -3. - row["ordinal"]
        row.update(answer_token_ids=[12], token_logprobs=[value], scores=candidate_scores([value], -.25))
    return rows


class DecisionAndSelectionTests(unittest.TestCase):
    def test_all_16_patterns_and_overlapping_membership(self):
        for i in range(16):
            p = profile(i, f"{i:04b}")
            actual = "".join(str(int(p["conditions"][c]["hate"]["correct"])) for c in m.CORE)
            self.assertEqual(actual, f"{i:04b}")
            self.assertEqual(any(x["bucket"] == "H_joint_only" for x in p["memberships"]), i == 1)
        p = profile(2, "0011", group_mask="0001")
        self.assertIn("H_rescue:hate", p["candidate_labels"])
        self.assertIn("G_joint_only:group", p["candidate_labels"])

    def test_ordinal_tie_and_boundary_only_in_relevant_conditions(self):
        candidates = raw_candidates("hate")
        candidates[1]["scores"] = candidates[0]["scores"].copy()
        chosen = m.ranking(list(reversed(candidates)), .001)
        self.assertEqual(chosen["ordinal"], 0)
        self.assertEqual(chosen["tied_top_count"], 2)
        self.assertTrue(m.is_correct(chosen["labels"], "hate", "hate"))
        p = profile(2)
        p["conditions"]["C0"]["hate"]["prediction"]["top_score_gap"] = 0.
        self.assertTrue(next(r for r in m.memberships(p, .001) if r["bucket"] == "H_rescue")["numeric_eligible"])
        p["conditions"]["CD"]["hate"]["prediction"]["top_score_gap"] = .002
        self.assertFalse(next(r for r in m.memberships(p, .001) if r["bucket"] == "H_rescue")["numeric_eligible"])

    def test_group_still_wrong_can_reduce_error_and_set_order_is_ignored(self):
        self.assertTrue(m.is_correct(["Region", "Sexism"], ["Sexism", "Region"], "group"))
        self.assertTrue(m.is_correct([], [], "group"))
        p = profile(0, "0000")
        p["gold"]["group"] = ["Sexism"]
        for c, labels in (("CD", ["Racism", "Region"]), ("CLDnewNoCat", ["Region"])):
            d = p["conditions"][c]["group"]
            d["prediction"]["labels"] = labels
            d["group_error_count"] = len(set(labels) ^ {"Sexism"})
            d["pred_group_size"] = len(labels)
        result = m.transition_counts([p], "CD", "CLDnewNoCat", "group")
        self.assertEqual(result["00"], 1)
        self.assertEqual(result["still_wrong_but_error_decreased"], 1)
        self.assertEqual(result["group_error_delta_mean"], -1)

    def test_family_normalization_and_deterministic_selection(self):
        self.assertEqual(m.family_key("e\u0301\r\n行"), m.family_key("é\n行"))
        self.assertNotEqual(m.family_key("x"), m.family_key(" x"))
        profiles = [profile(i, f"{i % 16:04b}", extra=(i % 3 == 0, i % 5 == 0),
                            group_mask=f"{(i * 3) % 16:04b}") for i in range(200)]
        profiles.append(profile(201, "0011", family=profiles[3]["family_key"]))
        first = m.select_cases(profiles)
        self.assertEqual(first, m.select_cases(profiles))
        self.assertEqual(first["selected"], m.select_cases(list(reversed(profiles)))["selected"])
        selected = first["selected"]
        self.assertEqual(len(selected), len({p["query_id"] for p in selected}))
        self.assertEqual(len(selected), len({p["family_key"] for p in selected}))
        for p in selected:
            self.assertEqual(p["split"], m.split_family(p["family_key"]))

    def test_no_backfill_and_control_task_alternation(self):
        only = [profile(i, split="discovery") for i in range(2)]
        result = m.select_cases(only)
        h = next(r for r in result["quotas"] if (r["bucket"], r["split"]) == ("H_rescue", "discovery"))
        self.assertEqual(h["selected"], 2)
        self.assertEqual(h["shortfall"], 2)
        self.assertEqual(sum(p["split"] == "reserve" for p in result["selected"]), 0)
        controls = [profile(i, "1111", extra=(True, True), split="discovery") for i in range(8)]
        chosen = [p for p in m.select_cases(controls)["selected"] if p["primary_bucket"] == "Stable_correct"]
        self.assertEqual([p["focus_task"] for p in chosen], ["hate", "group", "hate", "group"])


class PairedMathTests(unittest.TestCase):
    def test_bootstrap_matches_direct_shared_resampling_for_all_endpoints(self):
        rng = np.random.default_rng(13)
        hg, gg = np.array([1, 1, 1, 0, 0, 0, 0]), rng.integers(0, 2, size=(7, 5))
        hp, gp = rng.integers(0, 2, size=(7, 6)), rng.integers(0, 2, size=(7, 6, 5))
        stats = np.zeros((7, 6, 7, 3), dtype=int)
        for i in range(7):
            for j in range(6):
                truth = np.r_[hg[i], 1 - hg[i], gg[i]].astype(bool)
                pred = np.r_[hp[i, j], 1 - hp[i, j], gp[i, j]].astype(bool)
                stats[i, j, :, 0] = truth & pred
                stats[i, j, :, 1] = ~truth & pred
                stats[i, j, :, 2] = truth & ~pred
        actual, first_hash = m.bootstrap(stats, repetitions=137, batch_size=128)
        tiny_batch, second_hash = m.bootstrap(stats, repetitions=137, batch_size=1)
        self.assertEqual(first_hash, second_hash)
        for a, b in zip(actual, tiny_batch):
            np.testing.assert_allclose([a["difference"], a["ci95_low"], a["ci95_high"]],
                                       [b["difference"], b["ci95_low"], b["ci95_high"]], atol=1e-12)
        indices = np.random.Generator(np.random.PCG64(42)).integers(0, 7, size=(137, 7))
        reference = {(t, contrast): [] for t in m.TASKS for contrast in m.CI_CONTRASTS}
        for ix in indices:
            scores = {
                "hate": [f1_score(hg[ix], hp[ix, j], labels=[1, 0], average="macro", zero_division=0) for j in range(6)],
                "group": [f1_score(gg[ix], gp[ix, j], average="micro", zero_division=0) for j in range(6)]}
            for task in m.TASKS:
                s = dict(zip(m.CONDITIONS, scores[task]))
                for contrast, coefficients in m.CI_CONTRASTS.items():
                    value = sum(w * s[c] for c, w in coefficients.items())
                    if contrast == "J_remove_by_D_F1":
                        value = (s["CLDnewNoCat"] - s["CLDnew"]) - (s["CLnewNoCat"] - s["CLnew"])
                    reference[task, contrast].append(value)
        for row in actual:
            np.testing.assert_allclose([row["ci95_low"], row["ci95_high"]],
                np.quantile(reference[row["task"], row["contrast"]], [.025, .975]), atol=1e-12)

    def test_identical_conditions_and_empty_single_class_zero_denominators(self):
        stats = np.zeros((4, 6, 7, 3), dtype=int)
        stats[:, :, 0, 0] = 1  # all hate; the absent non-hate class stays in macro-F1
        metrics = m.f1_from_counts(stats.sum(axis=0))
        np.testing.assert_array_equal(metrics["hate"], np.full(6, .5))
        np.testing.assert_array_equal(metrics["group"], np.zeros(6))
        for row in m.bootstrap(stats, repetitions=19)[0]:
            self.assertEqual([row["difference"], row["ci95_low"], row["ci95_high"]], [0, 0, 0])
        empty = m.point_metrics([], np.zeros((0, 6, 7, 3)))
        self.assertTrue(all(r["n"] == 0 and r["hate_macro_f1"] is None for r in empty))


class InputContractTests(unittest.TestCase):
    def test_case_export_uses_native_hate_and_group_demo_answers(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "output"
            profiles = [profile(0), profile(1, group_mask="0001")]
            selections = [{"query_id": str(i), "focus_task": task, "selection_ordinal": i,
                           "primary_bucket": bucket} for i, (task, bucket) in enumerate(
                               (("hate", "H_rescue"), ("group", "G_joint_only")))]
            paths = {name: root / (name + ".jsonl" if name != "lexicon" else "lexicon.json")
                     for name in ("queries", "contexts", "fit_catalog", "lexicon")}
            gold = {"hate": "hate", "group": ["Region"]}
            m.write_jsonl(paths["queries"], [{"id": str(i), "content": "查询文本", "projection": gold} for i in range(2)])
            m.write_jsonl(paths["contexts"], [{"query_id": str(i), "task": task, "condition": "CD",
                "trace": {"demo": [{"demo_id": "d", "ordinal": 0, "rendered_output": gold[task]}]}}
                for i in range(2) for task in m.TASKS])
            m.write_jsonl(paths["fit_catalog"], [{"id": "d", "content": "示例文本", "projection": gold}])
            m.write_json(paths["lexicon"], {"terms": []})
            m.write_jsonl(output / "cases/discovery.jsonl", selections)
            m.write_json(output / "cases/selection_manifest.json", {"initial_review_query_ids": ["0", "1"]})
            config = {"inputs": {name: {"sha256": "fixture"} for name in ("plan", "raw", "analysis", "queries", "lexicon")}}
            m.review_export(profiles, output, paths, config)
            self.assertIn('冻结答案："hate"', (output / "cases/cards/00-0-1-resources.md").read_text())
            self.assertIn('冻结答案：["Region"]', (output / "cases/cards/01-1-1-resources.md").read_text())
            self.assertEqual(m.read_json(output / "cases/review_access.json")["reserve_cards_exported"], 0)

    def test_hash_failure_missing_core_and_context_degradation(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "queries.dev.jsonl"
            p.write_text('{}\n')
            config = {"inputs": {"queries": {"path": str(p), "sha256": m.file_sha(p)},
                                  "contexts": {"path": str(Path(tmp) / "contexts.dev.jsonl"), "sha256": "a" * 64}}}
            _, _, missing = m.verify_inputs(config)
            self.assertEqual(missing, ["contexts"])
            p.write_text('{"changed":true}\n')
            with self.assertRaisesRegex(m.AnalysisError, "hash mismatch"):
                m.verify_inputs(config)
            p.unlink()
            with self.assertRaisesRegex(m.AnalysisError, "missing required"):
                m.verify_inputs(config)

    def test_raw_analysis_join_and_rejection_preserve_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            frame = [{"query_id": str(i), "lex_hit": False} for i in range(2)]
            gold = {p["query_id"]: {"hate": "hate", "group": []} for p in frame}
            blocks, descriptors, per_query = [], [], []
            for q in gold:
                conditions = {}
                for c in m.CONDITIONS:
                    conditions[c] = {}
                    for t in m.TASKS:
                        candidates = raw_candidates(t)
                        rid = f"{q}:{t}:{c}"
                        descriptors.append({"record_id": rid, "query_id": q, "task": t, "condition": c,
                                            "context_sha256": "x", "prompt_sha256": "y", "prompt_tokens": 100})
                        blocks.append({"query_id": q, "task": t, "condition": c, "record_id": rid,
                                       "plan_id": "fixture", "context_sha256": "x", "prompt_sha256": "y", "candidates": candidates})
                        conditions[c][t] = {"prediction": m.ranking(candidates, .001),
                                            "readouts": block_analysis(t, candidates, gold=gold[q][t])}
                per_query.append({"query_id": q, "lex_hit": False, "gold": gold[q], "conditions": conditions})
            paths = {name: root / (name + ".jsonl" if name in ("queries", "raw") else name + ".json")
                     for name in ("queries", "raw", "analysis")}
            m.write_jsonl(paths["queries"], [{"id": q, "content": "text" + q, "projection": g} for q, g in gold.items()])
            m.write_jsonl(paths["raw"], blocks)
            analysis = {"per_query": per_query, "conditions": list(m.CONDITIONS), "epsilon": .001}
            m.write_json(paths["analysis"], analysis)
            plan = {"frame": frame, "blocks": descriptors, "catalog": candidate_catalog()}
            config = {"query_count": 2, "block_count": 24, "candidate_count": 408,
                      "epsilon": .001, "plan_id": "fixture", "selection": {"seed": 20260907}}
            resources = {q: profile(q)["resources"] for q in gold}
            lengths = {(q, t, c): profile(q)["conditions"][c][t]["context"] for q in gold for t in m.TASKS for c in m.CONDITIONS}
            with patch.object(m, "check_binding", return_value=plan), patch.object(m, "load_context_metadata", return_value=(resources, lengths)):
                before = {name: m.file_sha(path) for name, path in paths.items()}
                profiles, audit = m.validate_and_profile(config, paths, [])
                self.assertTrue(audit["passed"])
                self.assertEqual(len(profiles), 2)
                self.assertEqual(before, {name: m.file_sha(path) for name, path in paths.items()})
                for altered in (blocks[:-1], blocks + [blocks[0]], [{**blocks[0], "query_id": "unexpected"}] + blocks[1:]):
                    m.write_jsonl(paths["raw"], altered)
                    with self.assertRaises(m.AnalysisError):
                        m.validate_and_profile(config, paths, [])
                m.write_jsonl(paths["raw"], blocks)
                bad = copy.deepcopy(analysis)
                bad["per_query"] = list(reversed(bad["per_query"]))
                m.write_json(paths["analysis"], bad)
                with self.assertRaisesRegex(m.AnalysisError, "order"):
                    m.validate_and_profile(config, paths, [])
                bad = copy.deepcopy(analysis)
                bad["per_query"][0]["gold"]["hate"] = "non-hate"
                m.write_json(paths["analysis"], bad)
                with self.assertRaisesRegex(m.AnalysisError, "Gold"):
                    m.validate_and_profile(config, paths, [])

    def test_cpu_import_and_runtime_import_blocker(self):
        script = """
import sys
from diagnostics.general_model_nolabel_paired_cases import CPUOnlyImports
assert not any(n in sys.modules for n in ('torch', 'transformers', 'accelerate'))
sys.meta_path.insert(0, CPUOnlyImports())
try:
    import torch
except RuntimeError as error:
    assert 'forbidden' in str(error)
else:
    raise AssertionError('model import was not blocked')
"""
        subprocess.run([sys.executable, "-c", script], check=True)


if __name__ == "__main__":
    unittest.main()
