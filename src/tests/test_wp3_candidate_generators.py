from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

import jsonschema


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPOSITORY_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from build_lex.terminology_candidate_generators import (  # noqa: E402
    G1_PROMPT_VERSIONS,
    G2_PROMPT_VERSION,
    CandidateGeneratorError,
    build_g1_request,
    build_g2_request,
    build_model_source,
    build_public_task,
    generate_g3_observations,
    load_generator_config,
    merge_observations,
    normalize_form_reference,
    normalize_g1_response,
    normalize_g2_response,
)


CONFIG_PATH = REPOSITORY_ROOT / "config/stage1/wp3_candidate_generators_v1.json"
OBSERVATION_SCHEMA_PATH = REPOSITORY_ROOT / "schemas/wp3_candidate_observation_v1.schema.json"
CANDIDATE_SCHEMA_PATH = REPOSITORY_ROOT / "schemas/wp3_candidate_mention_v1.schema.json"


def model_source(
    response: dict,
    *,
    provider: str = "qwen",
    model: str = "Qwen3.8-27B",
    prompt_version: str = G2_PROMPT_VERSION,
) -> dict:
    return build_model_source(
        provider=provider,
        model=model,
        prompt_version=prompt_version,
        response=response,
    )


class ConfigAndRequestTests(unittest.TestCase):
    def test_config_binds_frozen_handbook_and_is_offline(self) -> None:
        config = load_generator_config(CONFIG_PATH, workspace_root=REPOSITORY_ROOT)
        self.assertEqual(
            config["handbook"]["version"],
            "wp3-terminology-evidence-handbook/v1.0",
        )
        self.assertFalse(any(config["execution"].values()))

    def test_public_task_discards_task_fields_and_requests_use_content_only(self) -> None:
        task = build_public_task(
            {
                "id": "fit-1",
                "content": "J生虫和txl都需要解码",
                "label": "must-not-leak",
                "targeted_group": ["must-not-leak"],
                "quadruples": [{"hateful": "must-not-leak"}],
            },
            blind_alias="术语-001",
        )
        self.assertEqual(set(task), {"task_id", "blind_alias", "content"})
        for request in (
            build_g1_request(task, pass_name="surface_decode", model="Qwen3.8-27B"),
            build_g2_request(task, model="deepseek-v4-flash"),
        ):
            wire = json.dumps(request, ensure_ascii=False)
            self.assertIn("J生虫和txl都需要解码", wire)
            for forbidden in (
                "targeted_group",
                '"hateful"',
                '"label"',
                '"argument"',
            ):
                self.assertNotIn(forbidden, wire)

    def test_tampered_handbook_hash_is_rejected(self) -> None:
        config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        config["handbook"]["sha256"] = "0" * 64
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory) / "config.json"
            temporary.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaisesRegex(CandidateGeneratorError, "handbook hash"):
                load_generator_config(temporary, workspace_root=REPOSITORY_ROOT)


class G1NormalizationTests(unittest.TestCase):
    def test_repeated_surface_uses_occurrence_and_reconstructs_rewrite(self) -> None:
        content = "J生虫和J生虫"
        parsed = {
            "rewritten_text": "寄生虫和J生虫",
            "edits": [
                {
                    "source_surface": "J生虫",
                    "occurrence_ordinal": 1,
                    "replacement": "寄生虫",
                    "mechanism": "mixed_script",
                    "requires_context": False,
                    "reason": "混写需要形式解码",
                }
            ],
            "record_reason": "存在一个局部形式编码",
        }
        rows = normalize_g1_response(
            parsed,
            record_id="fit-1",
            content=content,
            pass_name="surface_decode",
            source=model_source(
                parsed,
                prompt_version=G1_PROMPT_VERSIONS["surface_decode"],
            ),
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(content[rows[0]["start"] : rows[0]["end"]], "J生虫")
        self.assertEqual(rows[0]["occurrence_ordinal"], 1)
        self.assertEqual(rows[0]["replacement"], "寄生虫")

    def test_unreported_change_overlap_and_model_offsets_are_rejected(self) -> None:
        content = "舔狗舔狗"
        base = {
            "rewritten_text": "单方面讨好者舔狗",
            "edits": [
                {
                    "source_surface": "舔狗",
                    "occurrence_ordinal": 1,
                    "replacement": "单方面讨好者",
                    "mechanism": "slang",
                    "requires_context": False,
                    "reason": "网络俚语",
                }
            ],
            "record_reason": "存在俚语",
        }
        source = model_source(
            base,
            prompt_version=G1_PROMPT_VERSIONS["lexical_pragmatic"],
        )

        changed = copy.deepcopy(base)
        changed["rewritten_text"] = "额外改写"
        with self.assertRaisesRegex(CandidateGeneratorError, "reconstruct"):
            normalize_g1_response(
                changed,
                record_id="fit-2",
                content=content,
                pass_name="lexical_pragmatic",
                source=source,
            )

        overlap = copy.deepcopy(base)
        overlap["edits"].append(
            {
                "source_surface": "舔狗舔",
                "occurrence_ordinal": 1,
                "replacement": "重叠",
                "mechanism": "fixed_expression",
                "requires_context": True,
                "reason": "故意构造重叠",
            }
        )
        overlap["rewritten_text"] = "任意"
        with self.assertRaisesRegex(CandidateGeneratorError, "overlap"):
            normalize_g1_response(
                overlap,
                record_id="fit-2",
                content=content,
                pass_name="lexical_pragmatic",
                source=source,
            )

        with_offsets = copy.deepcopy(base)
        with_offsets["edits"][0]["start"] = 0
        with self.assertRaisesRegex(CandidateGeneratorError, "fields"):
            normalize_g1_response(
                with_offsets,
                record_id="fit-2",
                content=content,
                pass_name="lexical_pragmatic",
                source=source,
            )


class G2NormalizationTests(unittest.TestCase):
    def test_second_occurrence_is_kept_separate(self) -> None:
        content = "舔狗不是狗，舔狗是网络表达"
        parsed = {
            "mentions": [
                {
                    "surface": "舔狗",
                    "occurrence_ordinal": 2,
                    "mechanism": "slang",
                    "requires_context": False,
                    "reason": "需要网络语知识",
                }
            ],
            "record_reason": "提议第二次出现",
        }
        rows = normalize_g2_response(
            parsed,
            record_id="fit-3",
            content=content,
            source=model_source(parsed),
        )
        self.assertEqual(rows[0]["start"], content.rfind("舔狗"))
        self.assertEqual(rows[0]["occurrence_ordinal"], 2)

    def test_numeric_model_offsets_are_not_part_of_response_contract(self) -> None:
        parsed = {
            "mentions": [
                {
                    "surface": "txl",
                    "occurrence_ordinal": 1,
                    "mechanism": "abbreviation",
                    "requires_context": False,
                    "reason": "缩写",
                    "start": 99,
                }
            ],
            "record_reason": "测试",
        }
        with self.assertRaisesRegex(CandidateGeneratorError, "fields"):
            normalize_g2_response(
                parsed,
                record_id="fit-4",
                content="这里有txl",
                source=model_source(parsed),
            )


class G3AndUnionTests(unittest.TestCase):
    @staticmethod
    def reference() -> list[dict]:
        return [
            {
                "canonical": "同性恋",
                "pinyin": ["tong", "xing", "lian"],
                "initials": "txl",
                "variants": [],
            },
            {
                "canonical": "母人",
                "pinyin": ["mu", "ren"],
                "initials": None,
                "variants": [],
            },
        ]

    @staticmethod
    def romanizer(surface: str) -> list[str]:
        mapping = {
            "幕": "mu",
            "刃": "ren",
            "母": "mu",
            "人": "ren",
            "同": "tong",
            "性": "xing",
            "恋": "lian",
        }
        return [mapping.get(char, "unknown") for char in surface]

    def test_rules_find_form_triggers_without_emitting_all_ngrams(self) -> None:
        content = "ai紫 J生虫 ｔｘｌ 😀 txl 幕刃 全国女性"
        rows = generate_g3_observations(
            record_id="fit-5",
            content=content,
            reference_rows=self.reference(),
            romanizer=self.romanizer,
        )
        surfaces = {row["surface"] for row in rows}
        self.assertTrue(
            {"ai紫", "J生虫", "ｔｘｌ", "😀", "txl", "幕刃"}.issubset(surfaces)
        )
        self.assertNotIn("全国", surfaces)
        self.assertNotIn("女性", surfaces)
        by_surface = {}
        for row in rows:
            by_surface.setdefault(row["surface"], []).append(row)
        self.assertTrue(
            any(row["replacement"] == "txl" for row in by_surface["ｔｘｌ"])
        )
        self.assertTrue(
            any(row["replacement"] == "同性恋" for row in by_surface["txl"])
        )
        self.assertTrue(
            any(row["replacement"] == "母人" for row in by_surface["幕刃"])
        )

    def test_union_preserves_generator_provenance_and_nested_boundaries(self) -> None:
        content = "J生虫和easy girl"
        g1_response = {
            "rewritten_text": "寄生虫和easy girl",
            "edits": [
                {
                    "source_surface": "J生虫",
                    "occurrence_ordinal": 1,
                    "replacement": "寄生虫",
                    "mechanism": "mixed_script",
                    "requires_context": False,
                    "reason": "混写",
                }
            ],
            "record_reason": "存在混写",
        }
        g1 = normalize_g1_response(
            g1_response,
            record_id="fit-6",
            content=content,
            pass_name="surface_decode",
            source=model_source(
                g1_response,
                prompt_version=G1_PROMPT_VERSIONS["surface_decode"],
            ),
        )
        g2_response = {
            "mentions": [
                {
                    "surface": "J生虫",
                    "occurrence_ordinal": 1,
                    "mechanism": "mixed_script",
                    "requires_context": False,
                    "reason": "直接提议",
                },
                {
                    "surface": "easy",
                    "occurrence_ordinal": 1,
                    "mechanism": "other_nontransparent",
                    "requires_context": True,
                    "reason": "故意保留错误子串供后续 R 审核",
                },
                {
                    "surface": "easy girl",
                    "occurrence_ordinal": 1,
                    "mechanism": "fixed_expression",
                    "requires_context": True,
                    "reason": "完整短语",
                },
            ],
            "record_reason": "测试并集与嵌套",
        }
        g2 = normalize_g2_response(
            g2_response,
            record_id="fit-6",
            content=content,
            source=model_source(
                g2_response,
                provider="deepseek",
                model="deepseek-v4-flash",
            ),
        )
        g3 = generate_g3_observations(record_id="fit-6", content=content)
        candidates = merge_observations(
            [*g1, *g2, *g3],
            contents_by_record_id={"fit-6": content},
        )
        by_surface = {row["surface"]: row for row in candidates}
        self.assertEqual(
            by_surface["J生虫"]["generators"],
            ["g1_rewrite", "g2_direct", "g3_form_rule"],
        )
        self.assertEqual(by_surface["J生虫"]["replacement_hypotheses"], ["寄生虫"])
        self.assertIn("easy", by_surface)
        self.assertIn("easy girl", by_surface)
        self.assertNotEqual(
            by_surface["easy"]["candidate_id"],
            by_surface["easy girl"]["candidate_id"],
        )

    def test_form_reference_rejects_task_fields(self) -> None:
        bad = self.reference()[0] | {"target": "forbidden"}
        with self.assertRaisesRegex(CandidateGeneratorError, "fields"):
            normalize_form_reference([bad])


class JsonSchemaTests(unittest.TestCase):
    def test_generated_observation_and_candidate_match_published_schemas(self) -> None:
        content = "txl"
        parsed = {
            "mentions": [
                {
                    "surface": "txl",
                    "occurrence_ordinal": 1,
                    "mechanism": "abbreviation",
                    "requires_context": False,
                    "reason": "拼音首字母缩写",
                }
            ],
            "record_reason": "存在缩写",
        }
        observations = normalize_g2_response(
            parsed,
            record_id="fit-schema",
            content=content,
            source=model_source(parsed),
        )
        candidates = merge_observations(
            observations,
            contents_by_record_id={"fit-schema": content},
        )
        observation_schema = json.loads(
            OBSERVATION_SCHEMA_PATH.read_text(encoding="utf-8")
        )
        candidate_schema = json.loads(
            CANDIDATE_SCHEMA_PATH.read_text(encoding="utf-8")
        )
        jsonschema.Draft202012Validator(observation_schema).validate(observations[0])
        jsonschema.Draft202012Validator(candidate_schema).validate(candidates[0])


if __name__ == "__main__":
    unittest.main()
