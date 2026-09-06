import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from data.context_selector import QuotaUnsatisfiedError
from diagnostics.general_model_retrieval import (
    CacheIntegrityError,
    QUERY_PREFIX,
    RetrievalError,
    _BGEEncoder,
    build_retrieval,
    prepare_demo_pool,
    resolve_policy,
    select_from_scores,
)

try:
    import numpy as np
except ImportError:
    np = None


POLICY = {
    "source_class_order": ["Region", "Sexism"],
    "allocated_class_top_k": {"Region": 1, "Sexism": 1},
    "demo_top_k": 2,
    "candidate_multiplier": 3,
}


def quad(groups, *, argument="argument", hateful="hate"):
    return {"target": "target", "argument": argument, "targeted_group": groups, "hateful": hateful}


def demo(source_id, content, groups, **kwargs):
    return {"id": source_id, "content": content, "quadruples": [quad(groups, **kwargs)]}


class GeneralModelRetrievalSelectionTest(unittest.TestCase):
    def test_multiquad_membership_is_union_without_changing_gold(self):
        source = demo("1", "mixed", ["Region", "Racism"])
        source["quadruples"].append(quad(["non-hate"], hateful="non-hate"))
        before = copy.deepcopy(source)
        pool, audit = prepare_demo_pool([source], POLICY)
        self.assertEqual(source, before)
        self.assertEqual(pool[0]["source_classes"], ["non-hate", "Region", "Racism"])
        self.assertEqual(pool[0]["quadruples"], source["quadruples"])
        self.assertEqual(audit["retained_multiquad_count"], 1)

    def test_duplicate_content_minimum_numeric_id_and_conflicting_gold_exclusion(self):
        sources = [
            demo("10", "same\r\ncontent", ["Sexism", "Region"]),
            demo("2", "same\ncontent", ["Region", "Sexism"]),
            demo("3", "conflict", ["Region"]),
            demo("4", "conflict", ["Region"], argument="different gold"),
        ]
        pool, audit = prepare_demo_pool(sources, POLICY)
        self.assertEqual([row["id"] for row in pool], ["2"])
        self.assertEqual(pool[0]["content"], "same\ncontent")
        self.assertEqual(audit["duplicate_clusters"][0]["removed_ids"], ["10"])
        self.assertEqual(audit["conflicting_gold_clusters"][0]["source_ids"], ["3", "4"])
        self.assertEqual(audit["conflicting_gold_excluded_count"], 2)
        self.assertEqual(audit["same_gold_removed_count"], 1)

    def test_duplicate_source_ids_fail(self):
        with self.assertRaisesRegex(RetrievalError, "duplicate demo source ID"):
            prepare_demo_pool([demo("1", "first", ["Region"]), demo(1, "second", ["Sexism"])], POLICY)

    def test_null_target_and_argument_follow_canonical_gold_contract(self):
        source = demo("1", "no explicit target", ["Region"])
        source["quadruples"][0].update(target=None, argument=None)
        pool, _ = prepare_demo_pool([source], POLICY)
        self.assertIsNone(pool[0]["quadruples"][0]["target"])
        self.assertIsNone(pool[0]["quadruples"][0]["argument"])

    def test_noncanonical_configuration_names_are_not_silently_ignored(self):
        for key, value in (("max_length", 128), ("torch_threads", 8)):
            with self.assertRaisesRegex(RetrievalError, "max_embedding_tokens and torch_num_threads"):
                resolve_policy({**POLICY, key: value})
        resolved = resolve_policy({**POLICY, "torch_num_threads": 8})
        self.assertEqual(resolved["torch_num_threads"], 8)

    def test_multiclass_global_dedup_refills_and_excludes_query_overlap(self):
        sources = [
            demo("1", "shared", ["Region", "Sexism"]),
            demo("2", "region", ["Region"]),
            demo("3", "sexism", ["Sexism"]),
            demo("4", "other content", ["Region"]),
            demo("5", "query\r\ncontent", ["Sexism"]),
        ]
        pool, _ = prepare_demo_pool(sources, POLICY)
        query = {"id": "4", "content": "query\ncontent"}
        result = select_from_scores(pool, [query], [[0.9, 0.8, 0.7, 0.99, 0.98]], policy=POLICY)
        chosen = result["selected_by_query"]["4"]
        self.assertEqual([row["id"] for row in chosen], ["1", "3"])
        self.assertEqual([row["quota_class"] for row in chosen], ["Region", "Sexism"])
        self.assertEqual([row["prompt_rank"] for row in chosen], [0, 1])
        trace = result["traces_by_query"]["4"]
        self.assertEqual({item["reason"] for item in trace["excluded"]}, {"source_record_id_overlap", "content_sha256_overlap"})
        self.assertEqual(trace["duplicate_occurrences_merged"], 1)
        self.assertEqual(len(chosen[0]["quadruples"]), 1)

    def test_fixed_tenfold_candidates_fill_quotas_after_shared_top_three_are_consumed(self):
        shared_non_hate = demo("1", "shared non-hate", ["non-hate"], hateful="non-hate")
        shared_non_hate["quadruples"].append(quad(["others"]))
        sources = [
            shared_non_hate,
            demo("2", "shared region", ["Region", "others"]),
            demo("3", "shared racism", ["Racism", "others"]),
            demo("4", "sexism first", ["Sexism"]),
            demo("5", "lgbtq", ["LGBTQ"]),
            demo("6", "others refill", ["others"]),
            demo("7", "non-hate second", ["non-hate"], hateful="non-hate"),
            demo("8", "non-hate third", ["non-hate"], hateful="non-hate"),
            demo("9", "non-hate fourth", ["non-hate"], hateful="non-hate"),
            demo("10", "sexism second", ["Sexism"]),
        ] + [demo(str(source_id), f"others reserve {source_id}", ["others"]) for source_id in range(11, 18)]
        pool, _ = prepare_demo_pool(sources, {})
        queries = [{"id": "query", "content": "query without gold"}]
        scores = [[0.99 - index * 0.01 for index in range(len(pool))]]
        with self.assertRaises(QuotaUnsatisfiedError) as failure:
            select_from_scores(pool, queries, scores, policy={"candidate_multiplier": 3})
        self.assertEqual(failure.exception.source_class, "others")
        self.assertEqual(failure.exception.quota_round, 0)
        self.assertEqual(failure.exception.candidate_count, 3)

        result = select_from_scores(pool, queries, scores, policy={"candidate_multiplier": 10})
        chosen = result["selected_by_query"]["query"]
        self.assertEqual([row["id"] for row in chosen], [str(value) for value in range(1, 11)])
        self.assertEqual(len({row["id"] for row in chosen}), 10)
        quota_counts = {group: sum(row["quota_class"] == group for row in chosen)
                        for group in ("non-hate", "Region", "Racism", "Sexism", "LGBTQ", "others")}
        self.assertEqual(quota_counts, {"non-hate": 4, "Region": 1, "Racism": 1, "Sexism": 2, "LGBTQ": 1, "others": 1})
        poisoned_queries = [{**queries[0], "quadruples": "must never be read by retrieval"}]
        self.assertEqual(result, select_from_scores(pool, poisoned_queries, scores, policy={"candidate_multiplier": 10}))

    def test_prompt_order_follows_score_not_quota_assignment(self):
        pool, _ = prepare_demo_pool([demo("1", "region", ["Region"]), demo("2", "sexism", ["Sexism"])], POLICY)
        result = select_from_scores(pool, [{"id": "q", "content": "query"}], [[0.6, 0.9]], policy=POLICY)
        self.assertEqual([row["id"] for row in result["selected_by_query"]["q"]], ["2", "1"])
        self.assertEqual(result["traces_by_query"]["q"]["quota_assignments"][0]["assigned_quota_class"], "Region")

    def test_nonfinite_scores_and_unfillable_quota_fail(self):
        pool, _ = prepare_demo_pool([demo("1", "both", ["Region", "Sexism"])], POLICY)
        queries = [{"id": "q", "content": "query"}]
        with self.assertRaises(QuotaUnsatisfiedError):
            select_from_scores(pool, queries, [[0.9]], policy=POLICY)
        with self.assertRaisesRegex(RetrievalError, "finite"):
            select_from_scores(pool, queries, [[float("nan")]], policy=POLICY)

    def test_wrong_score_dimensions_and_gpu_are_rejected_before_loading(self):
        pool, _ = prepare_demo_pool([demo("1", "one", ["Region"])], POLICY)
        with self.assertRaisesRegex(RetrievalError, "dimensions"):
            select_from_scores(pool, [{"id": "q", "content": "query"}], [[0.1, 0.2]], policy=POLICY)
        with self.assertRaisesRegex(RetrievalError, "CPU FP32"):
            build_retrieval([], [], model_path=Path("unused"), cache_root=Path("unused"), policy=POLICY, device="cuda")


@unittest.skipIf(np is None, "numpy is required for safe embedding cache tests")
class GeneralModelRetrievalCacheTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.model = self.root / "model"
        self.model.mkdir()
        (self.model / "config.json").write_text("{}", encoding="utf-8")
        (self.model / "tokenizer.json").write_text("{}", encoding="utf-8")
        (self.model / "model.safetensors").write_bytes(b"fake weights")
        self.cache = self.root / "cache"
        self.sources = [demo("1", "first", ["Region"]), demo("2", "second", ["Sexism"])]
        self.queries = [{"id": "q", "content": "query"}]
        self.calls = []

        def encode(texts):
            self.calls.append(list(texts))
            return np.array([[1.0, 0.0] for _ in texts], dtype=np.float32), [600 if text.startswith(QUERY_PREFIX) else 12 for text in texts]

        self.encoder_patch = patch("diagnostics.general_model_retrieval._BGEEncoder")
        self.encoder = self.encoder_patch.start()
        self.addCleanup(self.encoder_patch.stop)
        self.encoder.return_value.encode.side_effect = encode
        self.version_patch = patch("diagnostics.general_model_retrieval.importlib.metadata.version", return_value="test-version")
        self.version_patch.start()
        self.addCleanup(self.version_patch.stop)

    def build(self):
        return build_retrieval(self.sources, self.queries, model_path=self.model, cache_root=self.cache, policy=POLICY)

    def test_cached_replay_is_identical_without_reencoding_and_records_truncation(self):
        first = self.build()
        second = self.build()
        self.assertEqual(first, second)
        self.assertEqual(len(self.calls), 2)
        self.assertEqual(self.encoder.call_count, 1)
        self.assertEqual(self.calls, [["first", "second"], [QUERY_PREFIX + "query"]])
        truncation = first["summary"]["embedding_truncation"]["queries"]
        self.assertEqual(truncation["truncated_count"], 1)
        self.assertEqual(truncation["records"][0]["embedding_tokens"], 512)
        self.assertEqual(truncation["records"][0]["untruncated_tokens"], 600)

    def test_weight_and_tokenizer_tree_changes_invalidate_both_caches(self):
        first = self.build()
        (self.model / "model.safetensors").write_bytes(b"updated weights")
        second = self.build()
        (self.model / "tokenizer.json").write_text('{"changed":true}', encoding="utf-8")
        third = self.build()
        ids = {result["summary"]["demo_cache"]["cache_id"] for result in (first, second, third)}
        self.assertEqual(len(ids), 3)
        self.assertEqual(len(self.calls), 6)

    def test_query_content_change_only_reencodes_query_cache(self):
        first = self.build()
        self.queries[0]["content"] = "changed query"
        second = self.build()
        self.assertEqual(first["summary"]["demo_cache"], second["summary"]["demo_cache"])
        self.assertNotEqual(first["summary"]["query_cache"]["cache_id"], second["summary"]["query_cache"]["cache_id"])
        self.assertEqual(len(self.calls), 3)

    def test_cache_array_corruption_fails_without_reencoding(self):
        first = self.build()
        array_path = self.cache / first["summary"]["demo_cache"]["cache_id"] / "vectors.npy"
        array_path.write_bytes(b"corrupt")
        with self.assertRaisesRegex(CacheIntegrityError, "checksum"):
            self.build()
        self.assertEqual(len(self.calls), 2)

    def test_cache_metadata_corruption_fails_without_reencoding(self):
        first = self.build()
        path = self.cache / first["summary"]["query_cache"]["cache_id"] / "metadata.json"
        metadata = json.loads(path.read_text(encoding="utf-8"))
        metadata["untruncated_token_counts"] = [3]
        path.write_text(json.dumps(metadata), encoding="utf-8")
        with self.assertRaisesRegex(CacheIntegrityError, "checksum"):
            self.build()
        self.assertEqual(len(self.calls), 2)

    def test_encoder_load_is_local_cpu_fp32_and_uses_normalized_cls(self):
        self.version_patch.stop()
        try:
            import torch
        except ImportError:
            self.skipTest("torch is required for the CPU encoder contract test")
        tokenizer = MagicMock()
        tokenizer.side_effect = lambda texts, **kwargs: (
            {"length": [600] * len(texts)} if kwargs.get("return_length")
            else {"input_ids": torch.zeros((len(texts), 2), dtype=torch.long)}
        )
        model = MagicMock()
        model.to.return_value = model
        model.eval.return_value = model
        model.return_value = SimpleNamespace(last_hidden_state=torch.tensor([[[3.0, 4.0], [0.0, 5.0]]]))
        tokenizer_loader = MagicMock(return_value=tokenizer)
        model_loader = MagicMock(return_value=model)
        fake_transformers = SimpleNamespace(
            AutoTokenizer=SimpleNamespace(from_pretrained=tokenizer_loader),
            AutoModel=SimpleNamespace(from_pretrained=model_loader),
        )
        before_threads = torch.get_num_threads()
        with patch.dict("sys.modules", {"transformers": fake_transformers}):
            encoder = _BGEEncoder(self.model, {"torch_num_threads": 4, "batch_size": 32})
            try:
                vectors, lengths = encoder.encode(["long text"])
            finally:
                encoder.close()
        self.assertEqual(torch.get_num_threads(), before_threads)
        self.assertEqual(lengths, [600])
        np.testing.assert_allclose(vectors, [[0.6, 0.8]])
        model.to.assert_called_once_with("cpu")
        self.assertIs(model_loader.call_args.kwargs["torch_dtype"], torch.float32)
        for loader in (tokenizer_loader, model_loader):
            self.assertTrue(loader.call_args.kwargs["local_files_only"])
            self.assertFalse(loader.call_args.kwargs["trust_remote_code"])
        self.assertTrue(model_loader.call_args.kwargs["use_safetensors"])
        self.assertEqual(tokenizer.call_args_list[-1].kwargs["max_length"], 512)
        self.assertTrue(tokenizer.call_args_list[-1].kwargs["truncation"])


if __name__ == "__main__":
    unittest.main()
