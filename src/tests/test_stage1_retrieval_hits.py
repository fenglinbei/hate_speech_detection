import sys
import json
import tempfile
import types
import unittest
from pathlib import Path


class _NoopLogger:
    def __getattr__(self, _name):
        return lambda *args, **kwargs: None


def install_import_stubs():
    sys.modules.setdefault("loguru", types.SimpleNamespace(logger=_NoopLogger()))
    sys.modules.setdefault("numpy", types.ModuleType("numpy"))
    sys.modules.setdefault(
        "sentence_transformers",
        types.SimpleNamespace(SentenceTransformer=object),
    )
    sklearn = types.ModuleType("sklearn")
    metrics = types.ModuleType("sklearn.metrics")
    pairwise = types.ModuleType("sklearn.metrics.pairwise")
    pairwise.cosine_similarity = lambda *_args, **_kwargs: None
    cluster = types.ModuleType("sklearn.cluster")
    cluster.KMeans = object
    metrics.pairwise = pairwise
    sklearn.metrics = metrics
    sklearn.cluster = cluster
    sys.modules.setdefault("sklearn", sklearn)
    sys.modules.setdefault("sklearn.cluster", cluster)
    sys.modules.setdefault("sklearn.metrics", metrics)
    sys.modules.setdefault("sklearn.metrics.pairwise", pairwise)
    tqdm_module = types.ModuleType("tqdm")
    tqdm_module.tqdm = lambda value=None, **_kwargs: value if value is not None else []
    sys.modules.setdefault("tqdm", tqdm_module)
    reranker = types.ModuleType("rag.reranker")
    reranker.Reranker = object
    sys.modules.setdefault("rag.reranker", reranker)


class _Values(list):
    def tolist(self):
        return list(self)


class Stage1RetrievalHitTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        install_import_stubs()
        from rag.core import Retriever
        from rag.core import MultiClassRetriever
        from rag.core import LexiconRetriever

        cls.Retriever = Retriever
        cls.MultiClassRetriever = MultiClassRetriever
        cls.LexiconRetriever = LexiconRetriever

    def make_retriever(self):
        retriever = self.Retriever.__new__(self.Retriever)
        retriever.items = [
            {
                "id": "1",
                "content": "candidate one",
                "quadruples": [
                    {
                        "target": "NULL",
                        "argument": "arg",
                        "targeted_group": ["Racism"],
                        "hateful": "hate",
                    }
                ],
            },
            {
                "id": "2",
                "content": "candidate two",
                "quadruples": [
                    {
                        "target": None,
                        "argument": "ok",
                        "targeted_group": ["non-hate"],
                        "hateful": "non-hate",
                    }
                ],
            },
        ]
        retriever.texts = [item["content"] for item in retriever.items]
        retriever.test2item = {item["content"]: item for item in retriever.items}
        retriever.model_name = "fixture-model"
        retriever.task_type = "structured"
        retriever.stratify_field = "targeted_group"
        calls = []

        def score_rows(queries, **_kwargs):
            calls.append(tuple(queries))
            return [(_Values([1, 0]), _Values([0.7, 0.9]), 0.7) for _ in queries]

        retriever._structured_score_rows = score_rows
        return retriever, calls

    def test_retrieve_hits_contains_stable_ids_hashes_scores_and_provenance(self):
        retriever, calls = self.make_retriever()

        hits = retriever.retrieve_hits(
            "query",
            top_k=2,
            source_class="Racism",
            use_cache=False,
        )

        self.assertEqual(len(calls), 1)
        self.assertEqual([hit.source_record_id for hit in hits], ["1", "2"])
        self.assertEqual([hit.rank for hit in hits], [0, 1])
        self.assertEqual([hit.score for hit in hits], [0.9, 0.7])
        self.assertTrue(all(hit.id.startswith("demo:v1:") for hit in hits))
        self.assertTrue(all(len(hit.content_sha256) == 64 for hit in hits))
        self.assertEqual(hits[0].source_class, "Racism")
        self.assertEqual(hits[0].method, "cosine")
        self.assertIn('"target":null', hits[0].output)
        self.assertEqual(hits[0].provenance["retriever_model"], "fixture-model")

    def test_batch_is_one_score_pass_and_legacy_method_is_untouched(self):
        retriever, calls = self.make_retriever()

        results = retriever.retrieve_batch_hits(
            ["q1", "q2"],
            top_k=1,
            source_class="Racism",
            use_cache=False,
        )

        self.assertEqual(calls, [("q1", "q2")])
        self.assertEqual(len(results), 2)
        self.assertTrue(all(len(items) == 1 for items in results))
        self.assertIn("retrieve", self.Retriever.__dict__)

    def test_multiclass_trace_api_keeps_class_lists_separate(self):
        first, _calls = self.make_retriever()
        second, _calls = self.make_retriever()
        retriever = self.MultiClassRetriever.__new__(self.MultiClassRetriever)
        retriever.target_groups = ["Racism", "Region"]
        retriever.retrievers = {"Racism": first, "Region": second}

        rows = retriever.retrieve_batch_hits(
            ["q1", "q2"],
            top_k={"Racism": 2, "Region": 1},
            use_cache=False,
        )

        self.assertEqual(len(rows), 2)
        self.assertEqual(set(rows[0]), {"Racism", "Region"})
        self.assertEqual(len(rows[0]["Racism"]), 2)
        self.assertEqual(len(rows[0]["Region"]), 1)
        self.assertTrue(all(hit.source_class == "Region" for hit in rows[0]["Region"]))

    def test_exact_lexicon_hits_merge_term_and_variant_evidence(self):
        retriever = self.LexiconRetriever.__new__(self.LexiconRetriever)
        retriever.lexicon_schema = "hatebase"
        retriever.match_mode = "word_boundary"
        retriever.case_sensitive = False
        retriever.entries = [
            {
                "lexicon_id": "lex:v1:" + "a" * 64,
                "term": "badword",
                "category": "Racism",
                "definition": "fixture",
                "variants": ["bad-word"],
            }
        ]
        retriever.texts = ["rendered block"]
        retriever.word2index = {"badword": 0, "bad-word": 0}

        hits = retriever.including_retrieve_hits("BADWORD and bad-word", top_k=-1)

        self.assertEqual(len(hits), 1)
        self.assertIsNone(hits[0].score)
        self.assertEqual(hits[0].method, "word_boundary")
        self.assertEqual(hits[0].provenance["match_terms"], ["badword", "bad-word"])
        self.assertEqual(hits[0].provenance["match_spans"], [[0, 7], [12, 20]])

    def test_repaired_lexicon_routes_legacy_api_through_controlled_matcher(self):
        from rag.controlled_lexicon_matcher import ControlledLexiconMatcher

        terms = [
            {
                "lexicon_id": "lex-short",
                "term": "批",
                "category": "others",
                "definition": "fixture",
                "senses": [{"sense_id": "sense-short", "definition": "fixture"}],
                "variants": [],
                "match_policy": {
                    "exclude_any": [
                        {"rule_id": "ordinary", "target": "right", "pattern": "^(判|评)"}
                    ]
                },
            }
        ]
        policy_sha = ControlledLexiconMatcher(
            terms,
            lexicon_sha256="a" * 64,
        ).policy_sha256
        payload = {
            "matcher_policy_version": "annotated-lexicon-controlled-longest/v1",
            "matcher_policy_sha256": policy_sha,
            "terms": terms,
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "lexicon.json"
            path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            retriever = self.LexiconRetriever.__new__(self.LexiconRetriever)
            retriever.lexicon_schema = "cold"
            retriever.include_variants = False
            retriever.load_datas(str(path))
            hits = retriever.including_retrieve_hits("批判和一批", top_k=-1)
            trace = retriever.including_match_trace("批判和一批")

        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0].method, "controlled_longest")
        self.assertEqual(hits[0].provenance["match_spans"], [[4, 5]])
        self.assertEqual(hits[0].provenance["matcher_policy_sha256"], policy_sha)
        self.assertEqual(trace["selected_spans"][0]["span"], [4, 5])


if __name__ == "__main__":
    unittest.main()
