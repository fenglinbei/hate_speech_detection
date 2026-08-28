import json
import unittest

from prompt import (
    STAGE1_QUAD_JSON_EVIDENCE_PROMPT_USER_V2,
    STAGE1_QUAD_JSON_EXAMPLE_PROMPT_V1,
    STAGE1_QUAD_JSON_RAG_PROMPT_USER_V1,
    STAGE1_QUAD_JSON_SYSTEM_PROMPT_V1,
    STAGE1_QUAD_JSON_SYSTEM_PROMPT_V2,
)
from tools.convert import parsed_quad_to_canonical_json
from utils.parser import parse_quadruples as reexported_parse_quadruples
from utils.quadruple import (
    Quadruple,
    QuadrupleValidationError,
    adapt_source_quad,
    canonicalize_quadruple,
    canonicalize_quadruples,
    parse_quadruples,
    serialize_quadruples,
    serialize_with_spans,
)


class SourceAdapterTests(unittest.TestCase):
    def test_adapts_legacy_null_aliases_order_and_nfc(self):
        quad = adapt_source_quad(
            {
                "target": "  e\u0301  ",
                "argument": " NULL ",
                "targeted_group": "Sexism, racism",
                "hateful": "HATE",
            }
        )

        self.assertEqual(
            quad,
            Quadruple(
                target="é",
                argument=None,
                targeted_group=("Racism", "Sexism"),
                hateful="hate",
            ),
        )

    def test_unknown_hateful_annotation_is_never_inferred(self):
        for hateful in ("NULL", None):
            with self.subTest(hateful=hateful), self.assertRaises(QuadrupleValidationError) as caught:
                adapt_source_quad(
                    {
                        "target": "人",
                        "argument": "坏",
                        "targeted_group": ["Racism"],
                        "hateful": hateful,
                    }
                )

            self.assertIn("unknown_annotation", {issue.code for issue in caught.exception.issues})

    def test_numeric_text_field_is_rejected(self):
        with self.assertRaises(QuadrupleValidationError) as caught:
            adapt_source_quad(
                {
                    "target": 42,
                    "argument": None,
                    "targeted_group": ["non-hate"],
                    "hateful": "non-hate",
                }
            )

        self.assertIn("field_type", {issue.code for issue in caught.exception.issues})


class CanonicalSerializationTests(unittest.TestCase):
    def test_exact_wire_and_round_trip(self):
        values = [
            {
                "target": "甲",
                "argument": None,
                "targeted_group": ["Sexism", "Racism"],
                "hateful": "hate",
            },
            {
                "target": None,
                "argument": "普通表达",
                "targeted_group": ["non-hate"],
                "hateful": "non-hate",
            },
        ]
        expected = (
            '[{"target":"甲","argument":null,"targeted_group":["Racism","Sexism"],"hateful":"hate"},'
            '{"target":null,"argument":"普通表达","targeted_group":["non-hate"],"hateful":"non-hate"}]'
        )

        wire = serialize_quadruples(values)
        parsed = parse_quadruples(wire)

        self.assertEqual(wire, expected)
        self.assertTrue(parsed.strict_format_valid)
        self.assertTrue(parsed.canonical_wire_equal)
        self.assertEqual(serialize_quadruples(parsed.quadruples), wire)

    def test_empty_array_is_valid(self):
        result = parse_quadruples("[]")

        self.assertTrue(result.syntax_valid)
        self.assertTrue(result.schema_valid)
        self.assertTrue(result.strict_format_valid)
        self.assertTrue(result.canonical_wire_equal)
        self.assertEqual(result.quadruples, [])

    def test_legal_json_whitespace_is_strict_but_not_canonical_wire(self):
        result = parse_quadruples(" \n [] \t")

        self.assertTrue(result.strict_format_valid)
        self.assertFalse(result.canonical_wire_equal)
        self.assertEqual(result.canonical_text, "[]")

    def test_value_spans_cover_complete_json_literals(self):
        values = [
            {
                "target": 'a"猫',
                "argument": None,
                "targeted_group": ["Sexism", "Racism"],
                "hateful": "hate",
            }
        ]

        wire, spans = serialize_with_spans(values)

        self.assertEqual(wire, serialize_quadruples(values))
        self.assertEqual(wire[slice(*spans[(0, "target")])], json.dumps('a"猫', ensure_ascii=False))
        self.assertEqual(wire[slice(*spans[(0, "argument")])], "null")
        self.assertEqual(
            wire[slice(*spans[(0, "targeted_group")])],
            '["Racism","Sexism"]',
        )
        self.assertEqual(wire[slice(*spans[(0, "hateful")])], '"hate"')

    def test_cross_field_disagreement_is_warning_without_rewrite(self):
        raw = '[{"target":null,"argument":"x","targeted_group":["non-hate"],"hateful":"hate"}]'

        result = parse_quadruples(raw)

        self.assertTrue(result.strict_format_valid)
        self.assertIn("group_hateful_mismatch", result.warning_codes)
        self.assertEqual(result.quadruples[0].hateful, "hate")
        self.assertEqual(result.quadruples[0].targeted_group, ("non-hate",))


class StrictAndRecoveryParserTests(unittest.TestCase):
    def test_strict_rejection_matrix(self):
        invalid_cases = {
            "legacy_pipe": "x | y | Racism | hate [END]",
            "fenced": "```json\n[]\n```",
            "trailing_text": "[] done",
            "legacy_null": '[{"target":"NULL","argument":null,"targeted_group":["Racism"],"hateful":"hate"}]',
            "legacy_null_hateful": '[{"target":null,"argument":"x","targeted_group":["Racism"],"hateful":"NULL"}]',
            "group_alias": '[{"target":null,"argument":"x","targeted_group":["racism"],"hateful":"hate"}]',
            "group_string": '[{"target":null,"argument":"x","targeted_group":"Racism","hateful":"hate"}]',
            "mixed_non_hate": '[{"target":null,"argument":"x","targeted_group":["Racism","non-hate"],"hateful":"hate"}]',
            "unknown_key": '[{"target":null,"argument":"x","targeted_group":["Racism"],"hateful":"hate","extra":1}]',
            "duplicate_key": '[{"target":"x","target":"y","argument":"a","targeted_group":["Racism"],"hateful":"hate"}]',
        }

        for name, raw in invalid_cases.items():
            with self.subTest(name=name):
                result = parse_quadruples(raw)
                self.assertFalse(result.strict_format_valid)
                self.assertFalse(result.recoverable_parse_valid)
                self.assertEqual(result.quadruples, [])

        duplicate = parse_quadruples(invalid_cases["duplicate_key"])
        self.assertTrue(duplicate.syntax_valid)
        self.assertIn("duplicate_key", duplicate.error_codes)
        self.assertIn(
            "legacy_null_sentinel",
            parse_quadruples(invalid_cases["legacy_null_hateful"]).error_codes,
        )

    def test_recovery_is_diagnostic_and_preserves_strict_flags(self):
        raw = """answer:
```json
[{"target":"NULL","argument":" x ","targeted_group":["sexism","Racism"],"hateful":"HATE"}]
```
"""

        result = parse_quadruples(raw, mode="recover")

        self.assertIs(result.raw, raw)
        self.assertFalse(result.syntax_valid)
        self.assertFalse(result.schema_valid)
        self.assertFalse(result.strict_format_valid)
        self.assertTrue(result.recoverable_parse_valid)
        self.assertFalse(result.canonical_wire_equal)
        self.assertEqual(
            result.quadruples[0],
            Quadruple(None, "x", ("Racism", "Sexism"), "hate"),
        )
        self.assertIn("recovered_json_fence", result.warning_codes)
        self.assertIn("alias_normalized", result.warning_codes)

    def test_recovery_does_not_fallback_to_legacy_pipe_parser(self):
        result = parse_quadruples("x | y | Racism | hate [END]", mode="recover")

        self.assertFalse(result.strict_format_valid)
        self.assertFalse(result.recoverable_parse_valid)
        self.assertEqual(result.quadruples, [])


class CompatibilitySurfaceTests(unittest.TestCase):
    def test_parser_reexports_authority(self):
        self.assertIs(reexported_parse_quadruples, parse_quadruples)

    def test_convert_wrapper_uses_source_adapter_and_canonical_serializer(self):
        wire = parsed_quad_to_canonical_json(
            [
                {
                    "target": "NULL",
                    "argument": "观点",
                    "targeted_group": "Sexism, Racism",
                    "hateful": "hate",
                }
            ]
        )

        self.assertEqual(
            wire,
            '[{"target":null,"argument":"观点","targeted_group":["Racism","Sexism"],"hateful":"hate"}]',
        )

    def test_stage1_prompts_keep_json_contract_and_placeholders(self):
        self.assertIn("canonical-quad-json/v1", STAGE1_QUAD_JSON_SYSTEM_PROMPT_V1)
        self.assertIn("只", STAGE1_QUAD_JSON_SYSTEM_PROMPT_V1)
        self.assertIn("[]", STAGE1_QUAD_JSON_SYSTEM_PROMPT_V1)
        self.assertIn("{lexicons}", STAGE1_QUAD_JSON_RAG_PROMPT_USER_V1)
        self.assertIn("{examples}", STAGE1_QUAD_JSON_RAG_PROMPT_USER_V1)
        self.assertIn("{text}", STAGE1_QUAD_JSON_RAG_PROMPT_USER_V1)
        self.assertIn("{retrieve_content}", STAGE1_QUAD_JSON_EXAMPLE_PROMPT_V1)
        self.assertIn("{retrieve_output}", STAGE1_QUAD_JSON_EXAMPLE_PROMPT_V1)
        self.assertIn("canonical-quad-json/v1", STAGE1_QUAD_JSON_SYSTEM_PROMPT_V2)
        self.assertIn("独立判断", STAGE1_QUAD_JSON_SYSTEM_PROMPT_V2)
        self.assertIn("不含任务类别", STAGE1_QUAD_JSON_EVIDENCE_PROMPT_USER_V2)
        self.assertIn("词条命中", STAGE1_QUAD_JSON_EVIDENCE_PROMPT_USER_V2)
        self.assertIn("{lexicons}", STAGE1_QUAD_JSON_EVIDENCE_PROMPT_USER_V2)
        self.assertIn("{examples}", STAGE1_QUAD_JSON_EVIDENCE_PROMPT_USER_V2)
        self.assertIn("{text}", STAGE1_QUAD_JSON_EVIDENCE_PROMPT_USER_V2)

    def test_programmatic_canonicalizer_is_strict(self):
        canonical = canonicalize_quadruple(
            {
                "target": " x ",
                "argument": None,
                "targeted_group": ["Sexism", "Racism"],
                "hateful": "hate",
            }
        )
        self.assertEqual(canonical, Quadruple("x", None, ("Racism", "Sexism"), "hate"))

        with self.assertRaises(QuadrupleValidationError):
            canonicalize_quadruples(
                [
                    {
                        "target": "NULL",
                        "argument": None,
                        "targeted_group": ["Racism"],
                        "hateful": "hate",
                    }
                ]
            )


if __name__ == "__main__":
    unittest.main()
