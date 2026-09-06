import json
import unittest

from diagnostics.general_model_tasks import (
    GROUP_LABELS,
    POLICY_VERSION,
    evaluate_predictions,
    parse_prediction,
    project_gold,
)


def quad(target="alpha", argument="argument", group="Racism", hateful="hate"):
    return {"target": target, "argument": argument, "targeted_group": [group], "hateful": hateful}


class GoldProjectionTests(unittest.TestCase):
    def test_any_hate_and_all_tuple_group_union(self):
        result = project_gold([quad(group="Sexism", hateful="non-hate"), quad(group="Racism"),
                               quad(group="non-hate", hateful="non-hate")])
        self.assertEqual(result["hate"], "hate")
        self.assertEqual(result["group"], ["Racism", "Sexism"])
        self.assertEqual(len(result["extraction"]), 3)

    def test_rare_nonhate_group_is_preserved_without_inference(self):
        result = project_gold([quad(group="Sexism", hateful="non-hate")])
        self.assertEqual(result["hate"], "non-hate")
        self.assertEqual(result["group"], ["Sexism"])
        self.assertEqual(project_gold([quad(group="non-hate", hateful="non-hate")])["group"], [])

    def test_invalid_or_empty_gold_cannot_be_negative(self):
        for values in ([], [quad(hateful="NULL")], [{"targeted_group": ["Racism"]}]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                project_gold(values)


class PredictionParsingTests(unittest.TestCase):
    def assert_classification_state(self, result, state, recovery_rule=None):
        self.assertEqual(result["format_status"], state)
        self.assertEqual(result["strict_valid"], state == "canonical")
        self.assertEqual(result["valid"], state in {"canonical", "recovered"})
        self.assertEqual(result["status"], "ok" if state == "canonical" else state)
        self.assertEqual(result["recovery_rule"], recovery_rule)

    def test_canonical_hate_is_a_complete_json_string(self):
        for label in ("hate", "non-hate"):
            for raw in (json.dumps(label), " \n" + json.dumps(label) + "\t"):
                with self.subTest(raw=raw):
                    result = parse_prediction("hate", raw)
                    self.assertEqual(result["value"], label)
                    self.assert_classification_state(result, "canonical")

    def test_canonical_group_is_a_json_array_including_empty(self):
        for labels in ([], ["Sexism"], list(GROUP_LABELS), list(reversed(GROUP_LABELS))):
            with self.subTest(labels=labels):
                result = parse_prediction("group", json.dumps(labels))
                self.assertEqual(result["value"], [label for label in GROUP_LABELS if label in labels])
                self.assert_classification_state(result, "canonical")

    def test_result_alias_and_existing_hate_aliases_are_symmetric_recoveries(self):
        for label in ("hate", "non-hate"):
            for key in ("hate", "label", "hateful", "result"):
                with self.subTest(label=label, key=key):
                    result = parse_prediction("hate", json.dumps({key: label}))
                    self.assertEqual(result["value"], label)
                    self.assert_classification_state(result, "recovered", f"object-field:{key}")

    def test_result_and_existing_group_aliases_support_empty_and_multilabel(self):
        for labels in ([], ["Sexism"], ["Racism", "Sexism"]):
            for key in ("group", "groups", "labels", "targeted_group", "result"):
                with self.subTest(labels=labels, key=key):
                    result = parse_prediction("group", json.dumps({key: labels}))
                    self.assertEqual(result["value"], labels)
                    self.assert_classification_state(result, "recovered", f"object-field:{key}")

    def test_brace_single_label_recovery_is_symmetric_and_complete(self):
        for label in ("hate", "non-hate"):
            for raw in ('{"' + label + '"}', '{ \n"' + label + '"\t }'):
                with self.subTest(raw=raw):
                    result = parse_prediction("hate", raw)
                    self.assertEqual(result["value"], label)
                    self.assert_classification_state(result, "recovered", "brace-single-label")
        for raw in ("{'hate'}", "{'non-hate'}", '{"hate",}', '{"hate","non-hate"}',
                    '{{"hate"}}', '{"hate":}', '{"unknown"}', '{"hate"', '{"HATE"}',
                    '{"non-hate"} explanation', 'not {"hate"}', '{" hate "}'):
            with self.subTest(raw=raw):
                self.assert_classification_state(parse_prediction("hate", raw), "invalid")
        self.assert_classification_state(parse_prediction("group", '{"Sexism"}'), "invalid")

    def test_bare_labels_are_recovered_not_canonical(self):
        for label in ("hate", "non-hate"):
            result = parse_prediction("hate", label)
            self.assertEqual(result["value"], label)
            self.assert_classification_state(result, "recovered", "bare-label")
        for raw in ("Racism", "Sexism, Racism"):
            self.assert_classification_state(parse_prediction("group", raw), "recovered", "bare-labels")

    def test_complete_fences_and_group_string_recovery_are_auditable(self):
        cases = [
            ("hate", '```json\n"hate"\n```', "json-fence"),
            ("hate", '```\n"non-hate"\n```', "json-fence"),
            ("hate", '```JSON\n{"result":"hate"}\n```', "json-fence+object-field:result"),
            ("group", '```json\n[]\n```', "json-fence"),
            ("group", '"Sexism,Racism"', "group-string"),
            ("group", '{"result":"Sexism"}', "object-field:result+group-string"),
            ("group", '[" Sexism "]', "label-whitespace"),
            ("group", '```json\n{"result":"Sexism"}\n```', "json-fence+object-field:result+group-string"),
        ]
        for task, raw, rule in cases:
            with self.subTest(task=task, raw=raw):
                self.assert_classification_state(parse_prediction(task, raw), "recovered", rule)

    def test_explanation_negation_and_embedded_json_are_never_classification(self):
        for task, answer in (("hate", '{"label":"hate"}'), ("group", '["Sexism"]')):
            for raw in (f"Answer: {answer}", f"not {answer}", f"{answer} Done.",
                        f"It is not {answer}.", f"Explanation before {answer} after.",
                        f"Do not output {answer}", f"```json\n{answer}\n```\nExplanation",
                        f"The answer is:\n```json\n{answer}\n```", f"```json\n{answer}"):
                with self.subTest(task=task, raw=raw):
                    self.assert_classification_state(parse_prediction(task, raw), "invalid")

    def test_conflicting_repeated_unknown_and_nested_answers_are_rejected(self):
        hate = [
            '"hate" "non-hate"', '"hate" "hate"', '{"result":"hate"}{"result":"non-hate"}',
            '{"result":"hate","result":"non-hate"}', '{"result":"hate","result":"hate"}',
            '{"result":"hate","label":"hate"}', '{"result":"hate","label":"non-hate"}',
            '{"result":"hate","explanation":"x"}', '{"answer":"hate"}', '{"Result":"hate"}',
            '{"result":{"label":"hate"}}', '{"result":["hate"]}', '["hate"]', '"not hate"',
            '"Hate"', '" hate "', '"unknown"', 'true', '1', 'null', 'NaN', '{"result":NaN}',
            '```json\n"hate"\n```\n```json\n"non-hate"\n```',
        ]
        group = [
            '["Sexism"] []', '[] []', '{"result":["Sexism"],"groups":[]}',
            '{"result":[],"result":[]}', '{"result":["Sexism"],"result":["Racism"]}',
            '{"answer":["Sexism"]}', '{"result":{"group":["Sexism"]}}',
            '["Sexism","Sexism"]', '["Sexism"," Sexism "]', 'Sexism,Sexism',
            '["unknown"]', '["non-hate"]', '["Sexism",1]', '[true]', '""', '"not Sexism"',
            '["Sexism"', '{"result":["Sexism"]', 'true', '0', 'null', 'NaN',
        ]
        for task, cases in (("hate", hate), ("group", group)):
            for raw in cases:
                with self.subTest(task=task, raw=raw):
                    result = parse_prediction(task, raw)
                    self.assertIsNone(result["value"])
                    self.assert_classification_state(result, "invalid")

    def test_missing_and_nontext_classification_have_complete_format_metadata(self):
        for task in ("hate", "group"):
            for raw in (None, "", " \n\t"):
                with self.subTest(task=task, raw=raw):
                    self.assert_classification_state(parse_prediction(task, raw), "missing")
            for raw in (0, True, [], {}):
                with self.subTest(task=task, raw=raw):
                    self.assert_classification_state(parse_prediction(task, raw), "invalid")

    def test_length_termination_invalidates_even_complete_classification(self):
        for task, raw in (("hate", '"hate"'), ("hate", '{"non-hate"}'), ("group", "[]")):
            with self.subTest(task=task, raw=raw):
                result = parse_prediction(task, raw, termination="length")
                self.assert_classification_state(result, "invalid")
                self.assertEqual(result["errors"][0]["code"], "truncated_output")
                self.assertTrue(parse_prediction(task, raw, termination="eos")["valid"])
        extraction = '[{"argument":"x"}]'
        self.assertEqual(parse_prediction("extraction", extraction, termination="length"),
                         parse_prediction("extraction", extraction))

    def test_classification_recovery_has_no_substring_guessing(self):
        for raw in ("hate", '"hate"', '{"label":"hate"}', '```json\n{"hate":"hate"}\n```'):
            self.assertEqual(parse_prediction("hate", raw)["value"], "hate")
        for raw in ("I considered hate and non-hate.", "non-hateful", "1", 'null', '{"label":"hate","explanation":"x"}'):
            with self.subTest(raw=raw):
                self.assertFalse(parse_prediction("hate", raw)["valid"])

    def test_empty_group_is_valid_but_missing_unknown_is_not(self):
        self.assertEqual(parse_prediction("group", "[]")["value"], [])
        self.assertTrue(parse_prediction("group", "[]")["valid"])
        for raw in ("", "non-hate", '["unknown"]', 'null', '["Sexism","Sexism"]'):
            with self.subTest(raw=raw):
                result = parse_prediction("group", raw)
                self.assertFalse(result["valid"])
                self.assertIsNone(result["value"])
        self.assertEqual(parse_prediction("group", "Sexism,Racism")["value"], ["Racism", "Sexism"])

    def test_partial_extraction_preserves_null_and_missing(self):
        result = parse_prediction("extraction", '```json\n[{"target":null,"argument":5,"targeted_group":"Sexism"}]\n```')
        self.assertTrue(result["valid"])
        self.assertEqual(result["status"], "partial")
        self.assertEqual(result["value"], [{"target": None, "targeted_group": ["Sexism"]}])
        self.assertEqual(result["field_statuses"][0]["hateful"], "missing")
        self.assertEqual(result["field_statuses"][0]["argument"], "invalid")
        self.assertEqual(result["field_statuses"][0]["target"], "valid")

    def test_embedded_unique_json_and_multiple_answers(self):
        result = parse_prediction("extraction", 'Answer: [{"argument":"x"}] Done.')
        self.assertEqual(result["value"], [{"argument": "x"}])
        self.assertTrue(result["valid"])
        self.assertEqual(parse_prediction("extraction", '[{"argument":"x"}] Done.')["value"],
                         [{"argument": "x"}])
        for raw in ('Answer: [{"target":"x"}] or [{"target":"y"}]',
                    '[{"target":"x"}', '[{"target":"x","target":"y"}]',
                    '[{"argument":NaN}]'):
            with self.subTest(raw=raw):
                self.assertFalse(parse_prediction("extraction", raw)["valid"])

    def test_non_object_item_does_not_discard_other_fields(self):
        result = parse_prediction("extraction", '[3,{"argument":"x"}]')
        self.assertTrue(result["valid"])
        self.assertEqual(result["value"], [{}, {"argument": "x"}])


class FixedFrameScoringTests(unittest.TestCase):
    def test_policy_is_versioned(self):
        self.assertEqual(POLICY_VERSION, "general-model-task-scoring/v2")

    def test_strict_metrics_preserve_frame_without_overwriting_recovered_metrics(self):
        result = evaluate_predictions("hate", [
            {"query_id": "a", "gold": "hate", "prediction": '"hate"'},
            {"query_id": "b", "gold": "non-hate", "prediction": '{"result":"non-hate"}'},
            {"query_id": "c", "gold": "hate", "prediction": '"unknown"'},
            {"query_id": "d", "gold": "non-hate", "prediction": ""},
        ])
        metrics = result["metrics"]
        self.assertEqual(result["query_count"], 4)
        self.assertEqual(metrics["accuracy"], 0.5)
        self.assertEqual(metrics["canonical_rate"], 0.25)
        self.assertEqual(metrics["recovered_rate"], 0.25)
        self.assertEqual(metrics["invalid_rate"], 0.5)
        self.assertEqual(metrics["missing_count"], 1)
        self.assertEqual(metrics["missing_rate"], 0.25)
        self.assertEqual(metrics["format_status_counts"], {"canonical": 1, "recovered": 1, "invalid": 1, "missing": 1})
        self.assertEqual(metrics["strict_metrics"]["accuracy"], 0.25)
        self.assertEqual(metrics["strict_metrics"]["invalid_rate"], 0.75)
        self.assertEqual(metrics["strict_metrics"]["per_label"]["non-hate"]["fn"], 2)
        self.assertTrue(result["records"][1]["correct"])
        self.assertFalse(result["records"][1]["strict_correct"])
        self.assertEqual(result["records"][1]["sample_f1"], 1)
        self.assertEqual(result["records"][1]["strict_sample_f1"], 0)
        self.assertTrue(result["records"][1]["parsed"]["valid"])
        self.assertTrue(result["policy"]["fixed_query_denominator"])
        self.assertFalse(result["policy"]["classification"]["embedded_json_allowed"])

    def test_recovered_empty_group_is_valid_only_in_forgiving_metrics(self):
        result = evaluate_predictions("group", [
            {"query_id": "a", "gold": [], "prediction": "[]"},
            {"query_id": "b", "gold": [], "prediction": '{"result":[]}'},
            {"query_id": "c", "gold": ["Sexism"], "prediction": '"Sexism"'},
            {"query_id": "d", "gold": [], "prediction": ""},
        ])
        metrics = result["metrics"]
        self.assertEqual(metrics["exact_match"], 0.75)
        self.assertEqual(metrics["sample_f1"], 0.75)
        self.assertEqual(metrics["invalid_empty_gold_count"], 1)
        self.assertEqual(metrics["missing_count"], 1)
        self.assertEqual(metrics["missing_rate"], 0.25)
        self.assertEqual(metrics["strict_metrics"]["exact_match"], 0.25)
        self.assertEqual(metrics["strict_metrics"]["sample_f1"], 0.25)
        self.assertEqual(metrics["strict_metrics"]["invalid_empty_gold_count"], 2)
        self.assertEqual(metrics["strict_metrics"]["per_label"]["Sexism"]["fn"], 1)

    def test_recovery_does_not_consult_gold_or_imply_correctness(self):
        result = evaluate_predictions("hate", [
            {"query_id": "a", "gold": "hate", "prediction": '{"hate"}'},
            {"query_id": "b", "gold": "non-hate", "prediction": '{"hate"}'},
        ])
        self.assertEqual(result["records"][0]["parsed"], result["records"][1]["parsed"])
        self.assertEqual(result["metrics"]["valid_rate"], 1)
        self.assertEqual(result["metrics"]["accuracy"], 0.5)
        self.assertEqual(result["metrics"]["strict_metrics"]["accuracy"], 0)

    def test_evaluation_preserves_length_stop_as_unscorable(self):
        result = evaluate_predictions("hate", [
            {"query_id": "a", "gold": "hate", "prediction": '"hate"', "termination": "length"},
            {"query_id": "b", "gold": "non-hate", "prediction": '"non-hate"', "termination": "eos"},
        ])
        self.assertEqual(result["metrics"]["accuracy"], 0.5)
        self.assertEqual(result["metrics"]["canonical_rate"], 0.5)
        self.assertEqual(result["records"][0]["parsed"]["errors"][0]["code"], "truncated_output")

    def test_empty_frame_has_finite_zero_rates_in_both_scoring_modes(self):
        for task in ("hate", "group"):
            with self.subTest(task=task):
                result = evaluate_predictions(task, [])
                self.assertEqual(result["query_count"], 0)
                self.assertEqual(result["metrics"]["canonical_rate"], 0)
                self.assertEqual(result["metrics"]["recovered_rate"], 0)
                self.assertEqual(result["metrics"]["missing_count"], 0)
                self.assertEqual(result["metrics"]["missing_rate"], 0)
                self.assertEqual(result["metrics"]["strict_metrics"]["valid_rate"], 0)
                json.dumps(result, allow_nan=False)

    def test_binary_missing_stays_in_accuracy_and_recall(self):
        result = evaluate_predictions("hate", [
            {"query_id": "a", "gold": "hate", "prediction": "hate"},
            {"query_id": "b", "gold": "non-hate", "prediction": ""},
        ])
        self.assertEqual(result["metrics"]["accuracy"], 0.5)
        self.assertEqual(result["metrics"]["invalid_rate"], 0.5)
        self.assertEqual(result["metrics"]["per_label"]["non-hate"]["fn"], 1)
        self.assertIsNone(result["records"][1]["parsed"]["value"])

    def test_group_empty_and_missing_are_scored_differently(self):
        result = evaluate_predictions("group", [
            {"query_id": "a", "gold": [], "prediction": "[]"},
            {"query_id": "b", "gold": [], "prediction": ""},
            {"query_id": "c", "gold": ["Sexism"], "prediction": ""},
        ])
        self.assertAlmostEqual(result["metrics"]["exact_match"], 1 / 3)
        self.assertAlmostEqual(result["metrics"]["sample_f1"], 1 / 3)
        self.assertEqual(result["metrics"]["per_label"]["Sexism"]["fn"], 1)
        self.assertEqual(result["metrics"]["invalid_empty_gold_count"], 1)

    def test_extraction_field_independence_fixed_frame_and_null(self):
        gold = [quad(target=None)]
        result = evaluate_predictions("extraction", [
            {"query_id": "a", "gold": gold, "prediction": '[{"target":null,"hateful":"hate"}]'},
            {"query_id": "b", "gold": gold, "prediction": ""},
        ])
        fields = result["metrics"]["field_unbound"]
        self.assertEqual(fields["target"]["similarity"], 0.5)
        self.assertEqual(fields["target"]["answer_coverage"], 0.5)
        self.assertEqual(fields["target"]["fn"], 1)
        self.assertEqual(fields["argument"]["answer_coverage"], 0)
        self.assertEqual(fields["argument"]["fn"], 2)
        self.assertFalse(result["metrics"]["binding_scored"])
        json.dumps(result, allow_nan=False)

    def test_similarity_is_unbound_and_penalizes_extra_missing_slots(self):
        gold = [quad(target="abc"), quad(target="xyz")]
        result = evaluate_predictions("extraction", [{"query_id": "a", "gold": gold,
            "prediction": '[{"target":"xyz"},{"target":"abc"},{}]'}])
        target = result["metrics"]["field_unbound"]["target"]
        self.assertAlmostEqual(target["similarity"], 2 / 3)
        self.assertEqual(target["fp"], 1)
        self.assertEqual(target["exact_match"], 0)

    def test_invalid_field_does_not_erase_valid_group_or_infer_hate(self):
        gold = [quad(group="Sexism", hateful="non-hate")]
        result = evaluate_predictions("extraction", [{"query_id": "a", "gold": gold,
            "prediction": '[{"targeted_group":["Sexism"],"argument":false}]'}])
        fields = result["metrics"]["field_unbound"]
        self.assertEqual(fields["targeted_group"]["f1"], 1)
        self.assertEqual(fields["hateful"]["f1"], 0)

    def test_duplicate_ids_and_invalid_projected_gold_fail(self):
        row = {"query_id": "a", "gold": "hate", "prediction": "hate"}
        with self.assertRaises(ValueError):
            evaluate_predictions("hate", [row, row])
        with self.assertRaises(ValueError):
            evaluate_predictions("group", [{"query_id": "a", "gold": ["non-hate"], "prediction": "[]"}])


if __name__ == "__main__":
    unittest.main()
