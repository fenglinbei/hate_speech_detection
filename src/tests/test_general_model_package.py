"""Synthetic-only package identity, isolation and replay regression checks."""

import copy
import hashlib
import io
import json
import tempfile
import unittest
from contextlib import ExitStack, redirect_stdout
from pathlib import Path
from unittest import mock

from diagnostics import general_model_package as package
from rag.controlled_lexicon_matcher import ControlledLexiconMatcher, MATCHER_POLICY_VERSION


def _quad(group="Racism", hateful="hate"):
    return {"target": "target", "argument": "argument", "targeted_group": [group], "hateful": hateful}


class _Tokenizer:
    """Deterministic text-only tokenizer fixture; never loads model files."""

    def encode(self, text, add_special_tokens=False):
        return [ord(character) for character in text]

    def apply_chat_template(self, messages, **kwargs):
        return json.dumps(messages, ensure_ascii=False, separators=(",", ":")) + "\nassistant:"


class _SyntheticPackage:
    def __init__(self, root):
        self.root = root
        self.config = copy.deepcopy(package.read_json(package.ROOT / "config/stage1/general_model_ld_run_v1.json"))
        self.train = [
            {"id": "1", "content": "fit first", "quadruples": [_quad()]},
            {"id": "2", "content": "fit second", "quadruples": [_quad("non-hate", "non-hate")]},
            {"id": "3", "content": "calibration isolated", "quadruples": [_quad("Sexism", "non-hate")]},
        ]
        self.dev = [
            {"id": "11", "content": "marker first dev", "quadruples": [_quad()]},
            {"id": "12", "content": "second dev", "quadruples": [_quad("Sexism", "non-hate")]},
            {"id": "13", "content": "third dev", "quadruples": [_quad("non-hate", "non-hate")]},
        ]
        self.data_target = root / "sources/data"
        self.part_target = root / "sources/partition"
        self.data_target.mkdir(parents=True)
        self.part_target.mkdir(parents=True)
        (root / "refs").mkdir()
        self.config["sources"] = {
            "data_ref": "refs/data.json", "data_id": "synthetic-data-v1",
            "data_payload_sha256": "", "partition_ref": "refs/partition.json",
            "partition_id": "synthetic-partition-v1", "partition_payload_sha256": "",
            "lexicon": "lexicon.json", "lexicon_sha256": "", "lexicon_manifest": "lexicon-manifest.json",
            "expected_counts": {"train": 3, "fit": 2, "calibration": 1, "dev": 3, "lexicon": 1},
        }
        self.config["matrix"].update({
            "preflight_query_count": 2, "extraction_dev_count": 3,
            "primary_conditions": ["C0", "CL", "CD", "CLD", "PL", "PD"],
            "core_conditions": ["C0", "CL", "CD", "CLD", "PL", "PD"],
        })
        self.config["retrieval"].update({
            "demo_top_k": 2, "source_class_order": ["Racism", "non-hate"],
            "allocated_class_top_k": {"Racism": 1, "non-hate": 1},
        })
        entries = [{"lexicon_id": "synthetic-term", "term": "marker", "variants": [],
                    "category": ["Racism"], "definition": "fixture definition"}]
        policy_hash = ControlledLexiconMatcher(entries, lexicon_sha256="0" * 64).policy_sha256
        lexicon = {"terms": entries, "lexicon_build_id": "synthetic-lexicon-v1",
                   "matcher_policy_version": MATCHER_POLICY_VERSION, "matcher_policy_sha256": policy_hash}
        package.write_json(root / "lexicon.json", lexicon)
        lex_hash = package.sha256_file(root / "lexicon.json")
        self.config["sources"]["lexicon_sha256"] = lex_hash
        package.write_json(root / "lexicon-manifest.json", {
            "artifact_sha256": {"lexicon": lex_hash}, "lexicon_build_id": "synthetic-lexicon-v1",
        })
        for name in package.CODE_PATHS:
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("synthetic source identity: " + name + "\n", encoding="utf-8")
        protocol = root / self.config["protocol"]
        protocol.parent.mkdir(parents=True, exist_ok=True)
        protocol.write_text("Synthetic development-only protocol.\n", encoding="utf-8")
        self.config_path = root / "config.json"
        self.output = root / "output"
        self.write_sources()

    def _seal_source(self, kind):
        target = self.data_target if kind == "data" else self.part_target
        members = ["train.json", "dev.json"] if kind == "data" else ["partition.jsonl", "partition.meta.json", "data_ref.json"]
        files = [{"path": name, "size": (target / name).stat().st_size,
                  "sha256": package.sha256_file(target / name)} for name in members]
        if kind == "data":
            # This sealed member deliberately has no physical file: any attempt
            # to read or hash its payload would fail this development fixture.
            files.append({"path": "test.json", "size": 123, "sha256": "a" * 64})
        package.write_json(target / "payload_manifest.json", {"files": files})
        prefix = "data" if kind == "data" else "partition"
        sources = self.config["sources"]
        sources[f"{prefix}_payload_sha256"] = package.sha256_file(target / "payload_manifest.json")
        ref = {"artifact_id": sources[f"{prefix}_id"],
               "payload_manifest_sha256": sources[f"{prefix}_payload_sha256"],
               "target_path": target.relative_to(self.root).as_posix()}
        package.write_json(self.root / sources[f"{prefix}_ref"], ref)
        return ref

    def write_sources(self):
        package.write_json(self.data_target / "train.json", self.train)
        package.write_json(self.data_target / "dev.json", self.dev)
        data_ref = self._seal_source("data")
        partitions = [{"query_id": row["id"], "partition": "calibration" if row["id"] == "3" else "fit",
                       "content_sha256": hashlib.sha256(row["content"].replace("\r\n", "\n").encode()).hexdigest()}
                      for row in self.train]
        package.write_jsonl(self.part_target / "partition.jsonl", partitions)
        package.write_json(self.part_target / "partition.meta.json", {"data_dependency": data_ref})
        package.write_json(self.part_target / "data_ref.json", data_ref)
        self._seal_source("partition")
        package.write_json(self.config_path, self.config)

    def retrieval(self, fit, queries, **kwargs):
        return {
            "selected_by_query": {str(row["id"]): copy.deepcopy(fit) for row in queries},
            "traces_by_query": {str(row["id"]): {"synthetic": True} for row in queries},
            "summary": {"synthetic": True, "source_count": len(fit)},
        }

    def mocks(self):
        stack = ExitStack()
        stack.enter_context(redirect_stdout(io.StringIO()))
        stack.enter_context(mock.patch.object(package, "_environment", return_value={
            "python": "synthetic", "packages": {"regex": "2026.4.4"},
        }))
        stack.enter_context(mock.patch.object(package, "_model_inventory", return_value=[
            {**model, "available": index == 0, "runtime_verified": False}
            for index, model in enumerate(self.config["models"])
        ]))
        stack.enter_context(mock.patch.object(package, "tokenizer_for_primary", return_value=_Tokenizer()))
        stack.enter_context(mock.patch("diagnostics.general_model_retrieval.build_retrieval", side_effect=self.retrieval))
        return stack

    def build(self):
        package.write_json(self.config_path, self.config)
        with self.mocks():
            result = package.build_dev(self.config_path, self.output, root=self.root)
        return Path(result["path"]), result

    def validate(self, path, replay=False):
        with self.mocks():
            return package.validate_package(path, root=self.root, replay=replay)

    def reseal_package(self, target):
        files = package._payload_files(target)
        package_id = "gmlpkg-" + package.canonical_json_sha256({"schema_version": package.SCHEMA, "files": files})
        package.write_json(target / "manifest.json", {
            "schema_version": package.SCHEMA, "package_id": package_id,
            "files": files, "scope": "development-package",
        })


class _PackageFixtureTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="general-model-package-test-")
        self.addCleanup(self.temporary.cleanup)
        self.fixture = _SyntheticPackage(Path(self.temporary.name))


class PackageInputTests(_PackageFixtureTests):
    def test_default_v2_config_preserves_model_inputs_and_decoding(self):
        legacy = package.load_config(package.ROOT / "config/stage1/general_model_ld_run_v1.json")
        current = package.load_config()
        self.assertEqual(package.DEFAULT_OUTPUT, package.ROOT / "exps/causal_context/general_model_ld_v2")
        self.assertEqual(current["experiment_id"], "general-model-ld-mechanism-v2")
        for name in ("sources", "models", "retrieval", "runtime", "controls", "protocol"):
            self.assertEqual(current[name], legacy[name], name)
        self.assertEqual(current["analysis"]["scoring_policy_version"], "general-model-task-scoring/v2")
        self.assertEqual(current["matrix"]["preflight_validation_query_count"], 24)
        legacy_matrix = {name: value for name, value in current["matrix"].items() if name != "preflight_validation_query_count"}
        self.assertEqual(legacy_matrix, legacy["matrix"])

    def test_two_cohort_config_requires_v2_scoring_policy(self):
        self.fixture.config["matrix"]["preflight_validation_query_count"] = 1
        package.write_json(self.fixture.config_path, self.fixture.config)
        with self.assertRaisesRegex(package.PackageError, "v2 scoring policy"):
            package.load_config(self.fixture.config_path)

    def test_development_allowlist_denies_sealed_and_traversal_members_before_io(self):
        for name in ("test.json", "sealed/test.json", "../train.json", "predictions.test.jsonl"):
            with self.subTest(name=name), mock.patch.object(package, "sha256_file") as hash_file:
                with self.assertRaisesRegex(package.PackageError, "allowlist"):
                    package._member(self.fixture.data_target, {}, name, {}, self.fixture.root)
                hash_file.assert_not_called()

    def test_load_reads_fit_dev_and_metadata_without_sealed_payload(self):
        self.assertFalse((self.fixture.data_target / "test.json").exists())
        result = package.load_inputs(self.fixture.config, root=self.fixture.root)
        self.assertEqual([row["id"] for row in result["fit"]], ["1", "2"])
        self.assertEqual([row["id"] for row in result["dev"]], ["11", "12", "13"])
        self.assertNotIn("calibration", result)
        self.assertFalse(result["test_content_read"])
        self.assertTrue(all("test.json" not in name for name in result["source_files"]))

    def test_runtime_and_model_roster_changes_are_rejected(self):
        changes = [("test_access", True), ("scope", "formal-test")]
        for key, value in changes:
            config = copy.deepcopy(self.fixture.config)
            config[key] = value
            package.write_json(self.fixture.config_path, config)
            with self.subTest(key=key), self.assertRaises(package.PackageError):
                package.load_config(self.fixture.config_path)
        for key, value in (
            ("enable_thinking", True), ("do_sample", True), ("local_files_only", False),
            ("dtype", "float32"), ("batch_size", 2), ("overflow_policy", "truncate"),
            ("determinism_repetitions", 1), ("classification_valid_rate_min", 0),
            ("max_new_tokens", {"hate": 64, "group": 128, "extraction": -1}),
            ("max_new_tokens", {"hate": True, "group": 128, "extraction": 512}),
        ):
            config = copy.deepcopy(self.fixture.config)
            config["runtime"][key] = value
            package.write_json(self.fixture.config_path, config)
            with self.subTest(key=key), self.assertRaises(package.PackageError):
                package.load_config(self.fixture.config_path)
        config = copy.deepcopy(self.fixture.config)
        config["models"][1]["model_id"] = "different-model"
        package.write_json(self.fixture.config_path, config)
        with self.assertRaisesRegex(package.PackageError, "model roster"):
            package.load_config(self.fixture.config_path)

    def test_all_six_core_conditions_are_required_in_primary_grid(self):
        for name in ("core_conditions", "primary_conditions"):
            config = copy.deepcopy(self.fixture.config)
            config["matrix"][name].remove("PL")
            package.write_json(self.fixture.config_path, config)
            with self.subTest(name=name), self.assertRaisesRegex(package.PackageError, "core conditions"):
                package.load_config(self.fixture.config_path)

    def test_source_ref_identity_mismatch_fails(self):
        config = copy.deepcopy(self.fixture.config)
        config["sources"]["data_id"] = "different-data"
        with self.assertRaisesRegex(package.PackageError, "frozen identity"):
            package.load_inputs(config, root=self.fixture.root)

    def test_changed_source_bytes_fail_even_if_json_still_parses(self):
        path = self.fixture.data_target / "dev.json"
        path.write_bytes(path.read_bytes() + b"\n")
        with self.assertRaisesRegex(package.PackageError, "source member changed"):
            package.load_inputs(self.fixture.config, root=self.fixture.root)

    def test_partition_bound_to_different_data_is_rejected(self):
        meta_path = self.fixture.part_target / "partition.meta.json"
        meta = package.read_json(meta_path)
        meta["data_dependency"]["artifact_id"] = "different-data"
        package.write_json(meta_path, meta)
        self.fixture._seal_source("partition")
        with self.assertRaisesRegex(package.PackageError, "different data"):
            package.load_inputs(self.fixture.config, root=self.fixture.root)

    def test_duplicate_partition_member_cannot_replace_missing_membership(self):
        path = self.fixture.part_target / "partition.jsonl"
        rows = package.read_jsonl(path)
        rows[-1] = copy.deepcopy(rows[0])
        package.write_jsonl(path, rows)
        self.fixture._seal_source("partition")
        with self.assertRaisesRegex(package.PackageError, "cover train exactly once"):
            package.load_inputs(self.fixture.config, root=self.fixture.root)

    def test_crlf_normalized_content_cannot_cross_fit_calibration(self):
        self.fixture.train[0]["content"] = "shared\r\ncontent"
        self.fixture.train[2]["content"] = "shared\ncontent"
        self.fixture.write_sources()
        with self.assertRaisesRegex(package.PackageError, "crosses fit/calibration"):
            package.load_inputs(self.fixture.config, root=self.fixture.root)

    def test_dev_ids_must_not_overlap_train(self):
        self.fixture.dev[0]["id"] = "1"
        self.fixture.write_sources()
        with self.assertRaisesRegex(package.PackageError, "overlapping data IDs"):
            package.load_inputs(self.fixture.config, root=self.fixture.root)

    def test_fixed_lexicon_hash_and_source_member_symlink_are_rejected(self):
        path = self.fixture.root / "lexicon.json"
        path.write_bytes(path.read_bytes() + b"\n")
        with self.assertRaisesRegex(package.PackageError, "lexicon bytes changed"):
            package.load_inputs(self.fixture.config, root=self.fixture.root)
        member = self.fixture.data_target / "dev.json"
        original = member.with_name("original-dev.json")
        member.rename(original)
        member.symlink_to(original.name)
        with self.assertRaisesRegex(package.PackageError, "source member unavailable"):
            package.load_inputs(self.fixture.config, root=self.fixture.root)

    def test_workspace_escape_is_rejected(self):
        with self.assertRaisesRegex(package.PackageError, "escapes workspace"):
            package.repo_path(self.fixture.root, "../outside.json")


class PackageFrameAndReplayTests(_PackageFixtureTests):
    def test_legacy_frame_keeps_exact_fields_and_order(self):
        traces = {row["id"]: {"selected_hits": ["synthetic"] if row["id"] == "11" else []} for row in self.fixture.dev}
        frames = package.select_dev_frames(self.fixture.dev, traces, self.fixture.config["matrix"])
        self.assertEqual(frames, {
            "preflight_query_ids": ["11", "13"],
            "extraction_query_ids": ["11", "13", "12"],
            "sampling_seed": 42,
            "selection": "pre-output-label-group-count-lex-hit-round-robin-and-sha256/v1",
        })

    def test_frame_is_order_invariant_and_ignores_prediction_sidecars(self):
        queries = copy.deepcopy(self.fixture.dev)
        traces = {row["id"]: {"selected_hits": ["synthetic"] if row["id"] == "11" else []} for row in queries}
        policy = self.fixture.config["matrix"]
        first = package.select_dev_frames(queries, traces, policy)
        queries.reverse()
        for row in queries:
            row["prediction"] = "opposite of any existing prediction"
            row["condition_accuracy"] = 1
        second = package.select_dev_frames(queries, traces, policy)
        self.assertEqual(first, second)
        self.assertEqual(len(set(first["preflight_query_ids"])), 2)
        self.assertEqual(set(first["extraction_query_ids"]), {"11", "12", "13"})
        self.assertTrue(set(first["preflight_query_ids"]).issubset(first["extraction_query_ids"]))

    def test_insufficient_preflight_frame_fails_without_fallback_sampling(self):
        traces = {row["id"]: {"selected_hits": []} for row in self.fixture.dev}
        policy = {**self.fixture.config["matrix"], "preflight_query_count": 4}
        with self.assertRaisesRegex(package.PackageError, "not enough dev queries"):
            package.select_dev_frames(self.fixture.dev, traces, policy)

    def test_invalid_frame_counts_and_missing_traces_fail(self):
        traces = {row["id"]: {"selected_hits": []} for row in self.fixture.dev}
        for key in ("preflight_query_count", "extraction_dev_count", "preflight_validation_query_count"):
            for invalid in (0, -1, True, 1.5, "2", None):
                with self.subTest(key=key, invalid=invalid), self.assertRaisesRegex(package.PackageError, "positive integer"):
                    package.select_dev_frames(self.fixture.dev, traces, {**self.fixture.config["matrix"], key: invalid})
        with self.assertRaisesRegex(package.PackageError, "sampling_seed"):
            package.select_dev_frames(self.fixture.dev, traces, {**self.fixture.config["matrix"], "sampling_seed": True})
        with self.assertRaisesRegex(package.PackageError, "every regression query"):
            package.select_dev_frames(self.fixture.dev, traces, {**self.fixture.config["matrix"], "extraction_dev_count": 1})
        with self.assertRaisesRegex(package.PackageError, "extraction diagnostic frame"):
            package.select_dev_frames(self.fixture.dev, traces, {**self.fixture.config["matrix"], "extraction_dev_count": 4})
        with self.assertRaisesRegex(package.PackageError, "duplicate dev query IDs"):
            package.select_dev_frames(self.fixture.dev + [self.fixture.dev[0]], traces, self.fixture.config["matrix"])
        del traces["11"]
        with self.assertRaisesRegex(package.PackageError, "missing lexicon trace"):
            package.select_dev_frames(self.fixture.dev, traces, self.fixture.config["matrix"])

    def test_two_cohorts_are_disjoint_and_require_enough_available_dev(self):
        traces = {row["id"]: {"selected_hits": []} for row in self.fixture.dev}
        policy = {**self.fixture.config["matrix"], "preflight_validation_query_count": 1}
        frames = package.select_dev_frames(self.fixture.dev, traces, policy)
        regression = frames["preflight_regression_query_ids"]
        validation = frames["preflight_validation_query_ids"]
        self.assertEqual(len(regression), 2)
        self.assertEqual(len(validation), 1)
        self.assertFalse(set(regression).intersection(validation))
        self.assertEqual(frames["preflight_query_ids"], regression + validation)
        with self.assertRaisesRegex(package.PackageError, "not enough dev queries"):
            package.select_dev_frames(self.fixture.dev, traces, {**policy, "preflight_validation_query_count": 2})
        with mock.patch.object(package, "_stratified_frame", side_effect=[["11", "12"], ["12"]]):
            with self.assertRaisesRegex(package.PackageError, "cohorts overlap"):
                package.select_dev_frames(self.fixture.dev, traces, policy)

    def test_v2_fixes_new_24_without_changing_original_8_or_extraction_64(self):
        queries = [{"id": str(index), "content": f"query {index}", "quadruples": [_quad(
            "Racism" if index % 3 else "non-hate", "hate" if index % 2 else "non-hate"
        )]} for index in range(100)]
        traces = {row["id"]: {"selected_hits": ["synthetic"] if int(row["id"]) % 4 else []} for row in queries}
        legacy_policy = {"preflight_query_count": 8, "extraction_dev_count": 64, "sampling_seed": 42}
        policy = {**legacy_policy, "preflight_validation_query_count": 24}
        legacy = package.select_dev_frames(queries, traces, legacy_policy)
        frames = package.select_dev_frames(queries, traces, policy)
        self.assertEqual(frames["preflight_regression_query_ids"], legacy["preflight_query_ids"])
        self.assertEqual(frames["extraction_query_ids"], legacy["extraction_query_ids"])
        self.assertEqual(len(frames["preflight_query_ids"]), 32)
        self.assertEqual(len(set(frames["preflight_validation_query_ids"])), 24)
        self.assertEqual(frames["validation_selection_namespace"], "preflight-validation/v2")
        self.assertEqual(frames["selection"], "pre-output-label-group-count-lex-hit-round-robin-and-sha256/v2")
        queries.reverse()
        for row in queries:
            row["prediction"] = "changed prediction"
            row["condition_accuracy"] = 0
        self.assertEqual(frames, package.select_dev_frames(queries, traces, policy))
        with mock.patch.object(package, "_hash_order", wraps=package._hash_order) as hash_order:
            package.select_dev_frames(queries, traces, policy)
        validation_calls = [call.args for call in hash_order.call_args_list if call.args[1] == "preflight-validation/v2"]
        self.assertEqual(len(validation_calls), 92)
        self.assertTrue(all(call[2] == 42 and call[0] not in frames["preflight_regression_query_ids"] for call in validation_calls))

    def test_two_cohort_package_replays_and_rejects_resealed_validation_order(self):
        self.fixture.config["matrix"].update({"preflight_query_count": 1, "preflight_validation_query_count": 2})
        self.fixture.config["analysis"]["scoring_policy_version"] = "general-model-task-scoring/v2"
        target, _ = self.fixture.build()
        self.assertTrue(self.fixture.validate(target, replay=True)["render_replayed"])
        path = target / "frames.dev.json"
        frames = package.read_json(path)
        frames["preflight_validation_query_ids"].reverse()
        frames["preflight_query_ids"] = frames["preflight_regression_query_ids"] + frames["preflight_validation_query_ids"]
        package.write_json(path, frames)
        self.fixture.reseal_package(target)
        with self.assertRaisesRegex(package.PackageError, "sample selection replay differs"):
            self.fixture.validate(target, replay=True)

    def test_build_repeats_exactly_and_replays_without_real_models(self):
        target, first = self.fixture.build()
        second_target, second = self.fixture.build()
        self.assertEqual(target, second_target)
        self.assertEqual(first["package_id"], second["package_id"])
        self.assertFalse(first["generation_preflight_passed"])
        self.assertFalse(first["formal_test_ready"])
        result = self.fixture.validate(self.fixture.output / "package_ref.json", replay=True)
        self.assertTrue(result["render_replayed"])
        self.assertEqual(result["context_count"], 54)
        fit = package.read_jsonl(target / "fit_catalog.jsonl")
        self.assertEqual({row["id"] for row in fit}, {"1", "2"})

    def test_unsealed_payload_changes_are_detected(self):
        target, _ = self.fixture.build()
        path = target / "contexts.dev.jsonl"
        path.write_bytes(path.read_bytes() + b"\n")
        with self.assertRaisesRegex(package.PackageError, "payload or identity changed"):
            self.fixture.validate(target)

    def test_resealed_wrong_projection_is_detected_from_source(self):
        target, _ = self.fixture.build()
        path = target / "queries.dev.jsonl"
        rows = package.read_jsonl(path)
        rows[0]["projection"]["hate"] = "non-hate"
        package.write_jsonl(path, rows)
        self.fixture.reseal_package(target)
        with self.assertRaisesRegex(package.PackageError, "task projections differ"):
            self.fixture.validate(target)

    def test_resealed_calibration_demo_is_rejected(self):
        target, _ = self.fixture.build()
        path = target / "retrieval.dev.jsonl"
        rows = package.read_jsonl(path)
        rows[0]["demos"][0] = copy.deepcopy(self.fixture.train[2])
        package.write_jsonl(path, rows)
        self.fixture.reseal_package(target)
        with self.assertRaisesRegex(package.PackageError, "example pool"):
            self.fixture.validate(target)

    def test_resealed_preflight_order_is_rejected_by_selection_replay(self):
        target, _ = self.fixture.build()
        path = target / "frames.dev.json"
        frames = package.read_json(path)
        frames["preflight_query_ids"].reverse()
        package.write_json(path, frames)
        self.fixture.reseal_package(target)
        with self.assertRaisesRegex(package.PackageError, "sample selection replay differs"):
            self.fixture.validate(target, replay=True)

    def test_resealed_prompt_with_new_local_hash_is_rejected_by_render_replay(self):
        target, _ = self.fixture.build()
        path = target / "contexts.dev.jsonl"
        rows = package.read_jsonl(path)
        rows[0]["prompt_text"] += "changed prompt"
        rows[0]["prompt_sha256"] = hashlib.sha256(rows[0]["prompt_text"].encode()).hexdigest()
        rows[0]["context_sha256"] = package.canonical_json_sha256({
            key: value for key, value in rows[0].items() if key != "context_sha256"
        })
        package.write_jsonl(path, rows)
        self.fixture.reseal_package(target)
        with self.assertRaisesRegex(package.PackageError, "condition/prompt/token-budget replay differs"):
            self.fixture.validate(target, replay=True)

    def test_runtime_source_tree_drift_is_rejected(self):
        target, _ = self.fixture.build()
        path = self.fixture.root / package.CODE_PATHS[0]
        path.write_text("changed source identity", encoding="utf-8")
        with self.assertRaisesRegex(package.PackageError, "runtime source tree differs"):
            self.fixture.validate(target)


if __name__ == "__main__":
    unittest.main()
