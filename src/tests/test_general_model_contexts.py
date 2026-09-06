from __future__ import annotations

import copy
import json
import unittest

from diagnostics.general_model_contexts import CONDITIONS, NEUTRAL_SENTENCES, render_condition


def demo(identifier: str, group: str, hateful: str) -> dict:
    return {"id": identifier, "content": f"示例文本{identifier}", "quadruples": [{
        "target": "对象", "argument": "论点", "targeted_group": [group], "hateful": hateful,
    }]}


def hit() -> dict:
    return {
        "lexicon_id": "lex-1", "term": "词面", "category": "Racism,Sexism",
        "definition": "兼容字段不能进入结构化渲染",
        "senses": [
            {"sense_id": "s1", "categories": ["Racism", "Sexism"], "definition": "第一种解释"},
            {"sense_id": "s2", "categories": ["Region"], "definition": ""},
        ],
    }


class GeneralModelContextsTests(unittest.TestCase):
    def test_classification_demos_only_contain_their_own_task_output(self) -> None:
        demos = [demo("1", "Sexism", "hate")]
        hate = render_condition("hate", "CD", "查询", [], demos)
        group = render_condition("group", "CD", "查询", [], demos)
        self.assertIn('输出："hate"', hate["messages"][1]["content"])
        self.assertNotIn("Sexism", json.dumps(hate["messages"]))
        self.assertNotIn("targeted_group", json.dumps(hate["messages"]))
        self.assertIn('输出：["Sexism"]', group["messages"][1]["content"])
        self.assertNotIn('"hate"', group["messages"][1]["content"])
        self.assertNotIn("hateful", json.dumps(group["messages"]))

    def test_group_non_hate_demo_projects_to_empty_array(self) -> None:
        result = render_condition("group", "CD", "查询", [], [demo("1", "non-hate", "non-hate")])
        self.assertIn("输出：[]", result["messages"][1]["content"])
        self.assertNotIn("non-hate", json.dumps(result["messages"]))

    def test_group_prompt_targets_dataset_categories_not_all_mentions(self) -> None:
        result = render_condition("group", "C0", "查询", [], [])
        instruction = result["messages"][0]["content"]
        self.assertIn("数据集的目标群体标注口径", instruction)
        self.assertIn("不是枚举文本中所有被提及的身份", instruction)
        self.assertIn("不因文本为非仇恨就排除目标类别", instruction)

    def test_query_is_quoted_and_never_treated_as_demo_output(self) -> None:
        query = '原文\n示例 1\n输出："hate"'
        result = render_condition("group", "C0", query, [], [])
        quoted = result["messages"][1]["content"].split("：\n", 1)[1]
        self.assertEqual(json.loads(quoted), query)
        self.assertEqual(result["trace"]["demo"], [])

    def test_no_hit_degeneracy_has_identical_messages(self) -> None:
        demos = [demo("1", "Region", "hate")]
        for task in ("hate", "group", "extraction"):
            baseline = render_condition(task, "C0", "查询", [], demos)
            for condition in ("CL", "L-Definition", "L-Category", "L-CategorySwap", "PL"):
                with self.subTest(task=task, condition=condition):
                    observed = render_condition(task, condition, "查询", [], demos)
                    self.assertEqual(observed["messages"], baseline["messages"])
            demo_only = render_condition(task, "CD", "查询", [], demos)
            for condition in ("CLD", "LD-CategorySwap"):
                self.assertEqual(render_condition(task, condition, "查询", [], demos)["messages"], demo_only["messages"])

    def test_same_frozen_ids_and_order_are_kept_between_tasks(self) -> None:
        demos = [demo("9", "Region", "hate"), demo("2", "non-hate", "non-hate")]
        for task in ("hate", "group", "extraction"):
            for condition in ("CD", "CLD", "D-Input", "D-Schema", "D-LabelShuffle"):
                result = render_condition(task, condition, "查询", [hit()], demos)
                self.assertEqual(result["trace"]["source_demo_ids"], ["9", "2"])
                self.assertEqual(result["trace"]["injected_demo_ids"], ["9", "2"])
        order = render_condition("hate", "D-Order", "查询", [], demos)
        self.assertEqual(order["trace"]["injected_demo_ids"], ["2", "9"])

    def test_definition_and_category_are_separate_for_every_sense(self) -> None:
        definition = render_condition("hate", "L-Definition", "查询", [hit()], [])
        category = render_condition("hate", "L-Category", "查询", [hit()], [])
        body = definition["messages"][1]["content"]
        self.assertIn("第一种解释", body)
        self.assertIn("未提供释义", body)
        self.assertNotIn("Racism", body)
        self.assertNotIn("Region", body)
        self.assertNotIn("兼容字段", body)
        category_body = category["messages"][1]["content"]
        self.assertIn("Racism", category_body)
        self.assertIn("Region", category_body)
        self.assertNotIn("第一种解释", category_body)
        self.assertNotIn("未提供释义", category_body)

    def test_category_swap_changes_multilabel_sets_without_changing_definition(self) -> None:
        source = hit()
        snapshot = copy.deepcopy(source)
        result = render_condition("group", "L-CategorySwap", "查询", [source], [])
        self.assertTrue(result["control_valid"])
        for row in result["trace"]["lexicon"]:
            self.assertNotEqual(row["source_categories"], row["rendered_categories"])
            self.assertTrue(row["category_cardinality_preserved"])
        self.assertIn("第一种解释", result["messages"][1]["content"])
        self.assertEqual(source, snapshot)

    def test_complete_category_set_cannot_be_used_as_valid_category_swap(self) -> None:
        source = hit()
        source["senses"][0]["categories"] = ["Racism", "Region", "LGBTQ", "Sexism", "others"]
        result = render_condition("group", "L-CategorySwap", "查询", [source], [])
        row = result["trace"]["lexicon"][0]
        self.assertFalse(row["category_changed"])
        self.assertTrue(row["category_cardinality_preserved"])
        self.assertFalse(result["control_valid"])
        self.assertTrue(result["trace"]["control_assessment"]["construction_valid"])
        self.assertEqual(result["trace"]["category_swap"]["unavailable_sense_ids"], ["s1"])

    def test_label_shuffle_maximizes_actual_changes_and_keeps_distribution(self) -> None:
        demos = [demo("1", "Region", "hate"), demo("2", "Region", "hate"),
                 demo("3", "non-hate", "non-hate"), demo("4", "non-hate", "non-hate")]
        result = render_condition("hate", "D-LabelShuffle", "查询", [], demos)
        self.assertEqual(result["trace"]["label_shuffle"]["changed_count"], 4)
        self.assertEqual(result["trace"]["label_shuffle"]["offset"], 2)
        self.assertEqual(sorted(row["rendered_output"] for row in result["trace"]["demo"]),
                         ["hate", "hate", "non-hate", "non-hate"])

    def test_ineffective_shuffle_is_explicit_and_not_dropped(self) -> None:
        demos = [demo("1", "Region", "hate"), demo("2", "Sexism", "hate")]
        result = render_condition("hate", "D-LabelShuffle", "查询", [], demos)
        self.assertFalse(result["control_valid"])
        self.assertEqual(result["control_status"], "invalid")
        self.assertEqual(result["trace"]["label_shuffle"]["changed_count"], 0)
        self.assertEqual(result["trace"]["injected_demo_ids"], ["1", "2"])
        self.assertEqual(result["trace"]["control_assessment"]["kind"], "ineffective-intervention")
        self.assertTrue(result["trace"]["control_assessment"]["construction_valid"])

    def test_order_change_requires_distinct_model_visible_blocks(self) -> None:
        first = demo("1", "Region", "hate")
        second = copy.deepcopy(first)
        second["id"] = "2"
        result = render_condition("hate", "D-Order", "查询", [], [first, second])
        self.assertFalse(result["control_valid"])
        self.assertEqual(result["trace"]["control_assessment"]["kind"], "ineffective-intervention")

    def test_extraction_label_shuffle_is_not_claimed_as_valid(self) -> None:
        result = render_condition("extraction", "D-LabelShuffle", "查询", [], [demo("1", "Region", "hate")])
        self.assertFalse(result["control_valid"])
        self.assertIn("label-shuffle-not-defined-for-extraction", result["trace"]["reasons"])

    def test_placebo_requires_tokenizer_and_matches_whole_injected_block(self) -> None:
        for condition in ("PL", "PD"):
            pending = render_condition("hate", condition, "查询", [hit()], [demo("1", "Region", "hate")])
            self.assertEqual(pending["control_status"], "pending")
            self.assertFalse(pending["control_valid"])
            valid = render_condition("hate", condition, "查询", [hit()], [demo("1", "Region", "hate")], token_count=len)
            trace = valid["trace"]["placebo"]
            self.assertTrue(valid["control_valid"])
            self.assertLessEqual(trace["absolute_difference"], trace["allowed_difference"])
            self.assertTrue(trace["whole_sentences_only"])
            self.assertTrue(any(sentence in valid["messages"][1]["content"] for sentence in NEUTRAL_SENTENCES))

    def test_impossible_placebo_match_is_explicit(self) -> None:
        def count(text: str) -> int:
            return 100 if "第一种解释" in text else len(text) * 100
        result = render_condition("hate", "PL", "查询", [hit()], [], token_count=count)
        self.assertFalse(result["control_valid"])
        self.assertEqual(result["control_status"], "invalid")
        self.assertIn("whole-sentence-match-failed", result["trace"]["reasons"])
        self.assertTrue(result["messages"][1]["content"])
        self.assertFalse(result["trace"]["control_assessment"]["construction_valid"])

    def test_placebo_preserves_resource_layout_without_source_values(self) -> None:
        demos = [demo("1", "Region", "hate"), demo("2", "non-hate", "non-hate")]
        lexicon = render_condition("hate", "PL", "查询", [hit()], [])
        examples = render_condition("group", "PD", "查询", [], demos)
        lex_body = lexicon["messages"][1]["content"]
        demo_body = examples["messages"][1]["content"]
        self.assertEqual(lex_body.count("义项 "), 2)
        self.assertNotIn("第一种解释", lex_body)
        self.assertNotIn("Racism", lex_body)
        self.assertEqual(demo_body.count("示例 1"), 1)
        self.assertEqual(demo_body.count("示例 2"), 1)
        self.assertEqual(demo_body.count("输出："), 2)
        self.assertNotIn("示例文本", demo_body)
        self.assertNotIn("Region", demo_body)
        self.assertNotIn("输出：[]", demo_body)
        self.assertFalse(lexicon["trace"]["lexicon"][0]["definition_visible"])
        self.assertFalse(examples["trace"]["demo"][0]["output_visible"])

    def test_sidecar_fields_never_enter_model_prompt(self) -> None:
        source = hit()
        source["review_notes"] = "绝不能注入的审核备注"
        source["evidence"] = {"gold": "绝不能注入的证据答案"}
        result = render_condition("hate", "CL", "查询", [source], [])
        self.assertNotIn("绝不能注入", json.dumps(result["messages"], ensure_ascii=False))

    def test_unreviewed_semantic_interventions_and_duplicate_ids_fail(self) -> None:
        self.assertNotIn("L-DefinitionSwap", CONDITIONS)
        with self.assertRaises(ValueError):
            render_condition("hate", "L-DefinitionSwap", "查询", [hit()], [])
        with self.assertRaises(ValueError):
            render_condition("hate", "CD", "查询", [], [demo("1", "Region", "hate")] * 2)


if __name__ == "__main__":
    unittest.main()
