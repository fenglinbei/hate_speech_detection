import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from data.build_data import make_data
from data.config import Config
from data.hatebase_adapter import convert_hatebase
from data.hatexplain_adapter import convert_dataset
from metrics.metric_llm import HateXplainMetrics
from prompt import (
    HATEXPLAIN_PROMPT_USER,
    HATEXPLAIN_RAG_PROMPT_USER_WO_EXAMPLES,
    HATEXPLAIN_RAG_PROMPT_USER_WO_LEX,
)
from utils.parser import parse_hatexplain_output


class _NoopLogger:
    def __getattr__(self, _name):
        return lambda *args, **kwargs: None


try:
    import loguru  # noqa: F401
except ModuleNotFoundError:
    sys.modules["loguru"] = types.SimpleNamespace(logger=_NoopLogger())

try:
    import sentence_transformers  # noqa: F401
except ModuleNotFoundError:
    sys.modules["sentence_transformers"] = types.SimpleNamespace(SentenceTransformer=object)

try:
    import sklearn.metrics.pairwise  # noqa: F401
except ModuleNotFoundError:
    sklearn_mod = types.ModuleType("sklearn")
    metrics_mod = types.ModuleType("sklearn.metrics")
    pairwise_mod = types.ModuleType("sklearn.metrics.pairwise")
    cluster_mod = types.ModuleType("sklearn.cluster")

    def _cosine_similarity(a, b):
        a = np.asarray(a, dtype=np.float32)
        b = np.asarray(b, dtype=np.float32)
        a = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)
        b = b / np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-12)
        return a @ b.T

    pairwise_mod.cosine_similarity = _cosine_similarity
    cluster_mod.KMeans = object
    metrics_mod.pairwise = pairwise_mod
    sklearn_mod.cluster = cluster_mod
    sklearn_mod.metrics = metrics_mod
    sys.modules["sklearn"] = sklearn_mod
    sys.modules["sklearn.cluster"] = cluster_mod
    sys.modules["sklearn.metrics"] = metrics_mod
    sys.modules["sklearn.metrics.pairwise"] = pairwise_mod

reranker_mod = types.ModuleType("rag.reranker")
reranker_mod.Reranker = object
sys.modules.setdefault("rag.reranker", reranker_mod)

from rag.core import LexiconRetriever


def write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


class HateXplainAdapterTest(unittest.TestCase):
    def test_convert_dataset_uses_split_order_and_drops_ties(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            dataset_path = root / "dataset.json"
            split_path = root / "post_id_divisions.json"
            output_dir = root / "std"

            dataset = {
                "a_twitter": {
                    "post_id": "a_twitter",
                    "post_tokens": ["bad", "asian", "slur"],
                    "annotators": [
                        {"label": "hatespeech", "annotator_id": 1, "target": ["Asian"]},
                        {"label": "hatespeech", "annotator_id": 2, "target": ["Asian"]},
                        {"label": "offensive", "annotator_id": 3, "target": ["Asian"]},
                    ],
                    "rationales": [[0, 1, 1], [0, 1, 1], [1, 0, 0]],
                },
                "b_gab": {
                    "post_id": "b_gab",
                    "post_tokens": ["one", "two"],
                    "annotators": [
                        {"label": "hatespeech", "annotator_id": 1, "target": ["African"]},
                        {"label": "offensive", "annotator_id": 2, "target": ["African"]},
                        {"label": "normal", "annotator_id": 3, "target": ["None"]},
                    ],
                    "rationales": [[1, 0], [0, 1]],
                },
            }
            write_json(dataset_path, dataset)
            write_json(split_path, {"train": ["a_twitter", "b_gab"], "val": [], "test": []})

            report = convert_dataset(dataset_path, split_path, output_dir, tie_policy="drop")

            train = json.loads((output_dir / "train.json").read_text(encoding="utf-8"))
            self.assertEqual([row["id"] for row in train], ["a_twitter"])
            self.assertEqual(train[0]["annotation"]["label"], "hatespeech")
            self.assertEqual(train[0]["annotation"]["target_groups"], ["Asian"])
            self.assertEqual(train[0]["annotation"]["rationales"][0]["token_indices"], [1, 2])
            self.assertEqual(report["counts"]["dropped_tie_records"], 1)


class HateBaseAdapterTest(unittest.TestCase):
    def test_convert_hatebase_pages_to_project_lexicon(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            page = {
                "version": "4.4",
                "datetime": "2020",
                "important": "",
                "query": {},
                "number_of_results": 1,
                "number_of_pages": 1,
                "page": "1",
                "result": [
                    {
                        "vocabulary_id": "x1",
                        "term": "Camel Jacker",
                        "hateful_meaning": "Person of Middle Eastern descent.",
                        "nonhateful_meaning": "",
                        "is_unambiguous": False,
                        "is_unambiguous_in": "",
                        "average_offensiveness": 90,
                        "language": "eng",
                        "plural_of": None,
                        "variant_of": "camel jockey",
                        "transliteration_of": None,
                        "is_about_nationality": False,
                        "is_about_ethnicity": True,
                        "is_about_religion": False,
                        "is_about_gender": False,
                        "is_about_sexual_orientation": False,
                        "is_about_disability": False,
                        "is_about_class": False,
                    }
                ],
            }
            write_json(root / "page_1.json", page)

            output = root / "processed" / "hatebase_en.json"
            report = convert_hatebase(root, output, expected_total=1, expected_pages=1)

            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(report["terms"], 1)
            self.assertEqual(payload["terms"][0]["category"], "ethnicity")
            self.assertEqual(payload["terms"][0]["variants"], ["camel jockey"])


class FakeSentenceTransformer:
    encoded_inputs = []

    def __init__(self, *_args, **_kwargs):
        pass

    def to(self, _device):
        return self

    def encode(self, texts, **_kwargs):
        if isinstance(texts, str):
            texts = [texts]
        self.__class__.encoded_inputs.extend(texts)
        return FakeTensor(np.ones((len(texts), 3), dtype=np.float32))


class FakeTensor:
    is_cuda = False

    def __init__(self, values):
        self.values = values

    def numpy(self):
        return self.values


class HateBaseRetrieverTest(unittest.TestCase):
    def test_exact_match_uses_word_boundary_variants_and_query_instruction(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            lexicon_path = root / "hatebase.json"
            write_json(
                lexicon_path,
                {
                    "terms": [
                        {
                            "term": "Camel Jacker",
                            "variants": ["camel jockey"],
                            "category": "ethnicity",
                            "categories": ["ethnicity"],
                            "definition": "Person of Middle Eastern descent.",
                            "nonhateful_meaning": "",
                            "average_offensiveness": 90,
                        }
                    ]
                },
            )
            FakeSentenceTransformer.encoded_inputs = []
            with patch("rag.core.SentenceTransformer", FakeSentenceTransformer):
                retriever = LexiconRetriever(
                    model_path="fake",
                    model_name="fake",
                    data_path=str(lexicon_path),
                    lexicon_schema="hatebase",
                    match_mode="word_boundary",
                    case_sensitive=False,
                    include_variants=True,
                    query_instruction="Represent this sentence for searching relevant passages: ",
                    cache_dir=str(root / "cache"),
                    enable_cache=False,
                )
                exact = retriever.including_retrieve("That phrase camel jockey appears.", top_k=-1, use_cache=False)
                self.assertEqual(len(exact), 1)
                self.assertEqual(retriever.including_retrieve("camel jockeying", top_k=-1, use_cache=False), [])
                retriever.similarity_retrieve("hello", top_k=1, use_cache=False)

            self.assertIn("Represent this sentence for searching relevant passages: hello", FakeSentenceTransformer.encoded_inputs)
            self.assertTrue(any(text.startswith("###\nTerm:") for text in FakeSentenceTransformer.encoded_inputs))


class HateXplainBuildParserMetricTest(unittest.TestCase):
    def test_prompt_variants_omit_only_missing_resource_sections(self):
        lexicons = "###\nTerm: slur\nCategory: ethnicity"
        examples = 'Text:\nexample text\nJSON:\n{"label":"normal","target_groups":[],"rationales":[]}'

        no_examples = (
            HATEXPLAIN_RAG_PROMPT_USER_WO_EXAMPLES
            .replace("{lexicons}", lexicons)
            .replace("{text}", "sample text")
        )
        self.assertIn("Background lexicon:", no_examples)
        self.assertIn("Term: slur", no_examples)
        self.assertNotIn("Examples:", no_examples)
        self.assertIn("A lexicon match is only background knowledge", no_examples)
        self.assertIn('"label" must be one of: "hatespeech", "offensive", "normal"', no_examples)

        no_lex = (
            HATEXPLAIN_RAG_PROMPT_USER_WO_LEX
            .replace("{examples}", examples)
            .replace("{text}", "sample text")
        )
        self.assertIn("Examples:", no_lex)
        self.assertIn("example text", no_lex)
        self.assertNotIn("Background lexicon:", no_lex)
        self.assertNotIn("lexicon match", no_lex)
        self.assertIn('"target_groups" must be a list of target communities', no_lex)

        no_resources = HATEXPLAIN_PROMPT_USER.replace("{text}", "sample text")
        self.assertNotIn("Background lexicon:", no_resources)
        self.assertNotIn("Examples:", no_resources)
        self.assertNotIn("lexicon match", no_resources)
        self.assertIn('"rationales" must be a list of short text spans', no_resources)

    def test_build_parser_and_metric_use_hatexplain_annotation(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            out_dir = root / "out"
            cache_dir = root / "cache"
            record = {
                "id": "1",
                "content": "bad asian slur",
                "annotation": {
                    "label": "hatespeech",
                    "target_groups": ["Asian"],
                    "rationales": [{"token_indices": [1, 2], "text": "asian slur"}],
                },
            }
            write_json(root / "train.json", [record])
            write_json(root / "val.json", [record])
            write_json(root / "test.json", [record])
            config_path = root / "config.json"
            write_json(
                config_path,
                {
                    "task_type": "hatexplain",
                    "data_paths": {
                        "raw_data_path": str(root / "train.json"),
                        "val_data_path": str(root / "val.json"),
                        "test_data_path": str(root / "test.json"),
                        "train_output_path": str(out_dir / "train.jsonl"),
                        "val_output_path": str(out_dir / "val.jsonl"),
                        "val_runner_output_path": str(out_dir / "val_runner.json"),
                        "test_output_path": str(out_dir / "test.json"),
                        "lexicon_data_path": "",
                        "tokenizer_path": None,
                    },
                    "prompt_templates": {
                        "prompt_template": "HATEXPLAIN_RAG_PROMPT_USER",
                        "example_template": "HATEXPLAIN_EXAMPLE_PROMPT",
                        "system_prompt": "HATEXPLAIN_SYSTEM_PROMPT",
                    },
                    "retrieval_settings": {"use_srag": False, "use_lex": False},
                    "training_settings": {"auto_length": False, "split_ratio": 0.9},
                    "cache_settings": {"enable_build_cache": False, "build_cache_dir": str(cache_dir)},
                },
            )

            make_data(Config(str(config_path)))
            train_row = json.loads((out_dir / "train.jsonl").read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual(json.loads(train_row["output"])["label"], "hatespeech")
            test_row = json.loads((out_dir / "test.json").read_text(encoding="utf-8"))[0]
            self.assertEqual(test_row["gt_annotation"]["label"], "hatespeech")

            parsed = parse_hatexplain_output('```json\n{"label":"hatespeech","target_groups":["Asian"],"rationales":["asian slur"]}\n```')
            self.assertEqual(parsed["target_groups"], ["Asian"])

            metrics = HateXplainMetrics().run(datas_list=[{
                "gt_annotation": record["annotation"],
                "pred_annotation": parsed,
                "status": "success",
            }])
            self.assertEqual(metrics["accuracy"], 1.0)
            self.assertEqual(metrics["target_group"]["micro_f1"], 1.0)
            self.assertEqual(metrics["rationale"]["micro_f1"], 1.0)


if __name__ == "__main__":
    unittest.main()
