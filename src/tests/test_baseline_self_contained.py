import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from data.build_data import load_global_demo_examples
from data.config import Config
from baselines.ids_runner import build_examples as build_ids_examples
from prompt import COLD_BINARY_EXAMPLE_PROMPT, HATEXPLAIN_EXAMPLE_PROMPT, RAG_PROMPT_EXAMPLE_V2
from utils.parser import extract_triplets, parse_binary_label, parse_hatexplain_output, parse_llm_output_trip


class ExplicitCotParserTest(unittest.TestCase):
    def test_structured_final_triples_marker_is_preferred(self):
        output = "分析：先判断上下文。\n### 最终三元组：群体 | 被污名化 | Region [END]"
        triples = extract_triplets(output)
        parsed = parse_llm_output_trip(triples)
        self.assertEqual(parsed[0]["target"], "群体")
        self.assertEqual(parsed[0]["targeted_group"], "Region")

    def test_cold_final_label_marker_is_preferred(self):
        output = "分析：这里提到了 non-hate 的边界，但最终是群体攻击。\n最终标签：hate"
        self.assertEqual(parse_binary_label(output), "hate")

    def test_hatexplain_final_json_marker_is_preferred(self):
        output = (
            "Analysis: the word normal appears in the reasoning.\n"
            'FINAL_JSON: {"label":"hatespeech","target_groups":["Asian"],"rationales":["slur"]}'
        )
        parsed = parse_hatexplain_output(output)
        self.assertEqual(parsed["label"], "hatespeech")
        self.assertEqual(parsed["target_groups"], ["Asian"])


class ExplicitCotPromptConfigTest(unittest.TestCase):
    def _config_prompt(self, root: Path, prompt_name: str, task_type: str = "structured") -> str:
        config_path = root / "config.json"
        config_path.write_text(
            json.dumps(
                {
                    "task_type": task_type,
                    "data_paths": {
                        "raw_data_path": "train.json",
                        "test_data_path": "test.json",
                        "train_output_path": "train.jsonl",
                        "val_output_path": "val.jsonl",
                        "test_output_path": "test_out.json",
                    },
                    "prompt_templates": {
                        "prompt_template": prompt_name,
                        "example_template": "RAG_PROMPT_EXAMPLE_V2",
                        "system_prompt": "DEFAULT_SYSTEM_PTOMPT_EN",
                    },
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        return Config(str(config_path)).prompt_template

    def test_config_resolves_all_explicit_cot_prompts(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            structured = self._config_prompt(root, "STRUCTURED_EXPLICIT_COT_RAG_PROMPT_USER")
            cold = self._config_prompt(root, "COLD_BINARY_EXPLICIT_COT_RAG_PROMPT_USER", "cold_binary")
            hatexplain = self._config_prompt(root, "HATEXPLAIN_EXPLICIT_COT_RAG_PROMPT_USER", "hatexplain")

        self.assertIn("### 最终三元组：", structured)
        self.assertIn("最终标签：", cold)
        self.assertIn("FINAL_JSON:", hatexplain)


class DppGlobalDemoRenderTest(unittest.TestCase):
    def test_global_demos_render_for_all_tasks(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            demos_path = root / "demos.json"
            demos = [
                {
                    "id": "s1",
                    "content": "地域攻击",
                    "quadruples": [
                        {"target": "地域", "argument": "被攻击", "targeted_group": "Region", "hateful": "hate"}
                    ],
                },
                {
                    "id": "h1",
                    "content": "bad slur",
                    "annotation": {
                        "label": "hatespeech",
                        "target_groups": ["Asian"],
                        "rationales": [{"text": "slur"}],
                    },
                },
            ]
            demos_path.write_text(json.dumps(demos, ensure_ascii=False), encoding="utf-8")

            structured, _ = load_global_demo_examples(str(demos_path), RAG_PROMPT_EXAMPLE_V2, task_type="structured")
            cold, _ = load_global_demo_examples(str(demos_path), COLD_BINARY_EXAMPLE_PROMPT, task_type="cold_binary")
            hatexplain, _ = load_global_demo_examples(str(demos_path), HATEXPLAIN_EXAMPLE_PROMPT, task_type="hatexplain")

        self.assertIn("地域 | 被攻击 | Region", structured[0])
        self.assertIn("标签：hate", cold[0])
        self.assertIn('"label":"hatespeech"', hatexplain[1])


class IdsDemoRenderTest(unittest.TestCase):
    def test_retrieved_outputs_render_for_all_tasks(self):
        structured = build_ids_examples(
            ["文本"],
            ["target | argument | Region | hate [END]"],
            RAG_PROMPT_EXAMPLE_V2,
            "structured",
        )
        cold = build_ids_examples(
            ["文本"],
            ["target | argument | Region | hate [END]"],
            COLD_BINARY_EXAMPLE_PROMPT,
            "cold_binary",
        )
        hatexplain = build_ids_examples(
            ["text"],
            ['{"label":"normal","target_groups":[],"rationales":[]}'],
            HATEXPLAIN_EXAMPLE_PROMPT,
            "hatexplain",
        )

        self.assertIn("target | argument | Region [END]", structured)
        self.assertIn("标签：hate", cold)
        self.assertIn('"label":"normal"', hatexplain)


class SelfContainedSpecGenerationTest(unittest.TestCase):
    def _run_spec(self, spec_name: str, root: Path) -> Path:
        repo = Path(__file__).resolve().parents[2]
        source = repo / "exps" / "specs" / spec_name
        spec = json.loads(source.read_text(encoding="utf-8"))
        output_root = root / spec_name.replace(".json", "")
        spec["output_root"] = str(output_root)
        spec_path = root / spec_name
        spec_path.write_text(json.dumps(spec, ensure_ascii=False), encoding="utf-8")
        subprocess.run(
            [sys.executable, "scripts/exps/expctl.py", "gen", "--spec", str(spec_path)],
            cwd=repo,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        return output_root

    def test_baseline_specs_generate_self_contained_configs(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            for spec_name in [
                "baselines_structured.json",
                "baselines_cold_binary.json",
                "baselines_hatexplain.json",
            ]:
                output_root = self._run_spec(spec_name, root)
                exp_dirs = sorted(path for path in output_root.iterdir() if path.is_dir())
                self.assertEqual(len(exp_dirs), 3)
                by_method = {}
                for exp_dir in exp_dirs:
                    runner = json.loads((exp_dir / "runner_config.json").read_text(encoding="utf-8"))
                    build = json.loads((exp_dir / "build_config.json").read_text(encoding="utf-8"))
                    method = runner["baseline"]["method"]
                    by_method[method] = (exp_dir, build, runner)
                    self.assertTrue(str(runner["tester"]["output_dir"]).startswith(str(exp_dir)))
                    self.assertTrue(str(build["data_paths"]["test_output_path"]).startswith(str(exp_dir)))
                    self.assertTrue(str(build["cache_settings"]["build_cache_dir"]).startswith(str(exp_dir)))
                    self.assertTrue(str(build["cache_settings"]["retrieval_cache_dir"]).startswith(str(exp_dir)))

                self.assertIn("ids", by_method)
                self.assertIn("dpp", by_method)
                self.assertIn("explicit_cot", by_method)
                self.assertEqual(by_method["ids"][2]["baseline"]["ids"]["q"], 3)
                dpp_exp, dpp_build, dpp_runner = by_method["dpp"]
                self.assertEqual(dpp_runner["baseline"]["dpp"]["k"], 10)
                self.assertTrue(dpp_build["global_demo_settings"]["global_demos_path"].startswith(str(dpp_exp)))
                self.assertFalse(by_method["explicit_cot"][2]["tester"]["run"]["llm_params"]["enable_thinking"])


if __name__ == "__main__":
    unittest.main()
