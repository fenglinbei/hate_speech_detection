"""CPU registration identity tests for separately materialized model replications."""

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from diagnostics import general_model_coverage_replication as replication
from diagnostics.general_model_package import PackageError


class ReplicationPlanTests(unittest.TestCase):
    def config(self, key="qwen3-14b"):
        suffix = "14b" if key == "qwen3-14b" else "27b"
        return json.loads((replication.ROOT / f"config/stage1/general_model_ld_coverage_{suffix}_v1.json").read_text())

    def test_two_registered_models_have_separate_namespaces(self):
        for key in replication.MODELS:
            config = self.config(key)
            replication.validate_config(config)
            self.assertNotEqual(config["output_root"], "exps/causal_context/general_model_ld_coverage_v1")

    def test_model_substitution_or_extra_override_is_rejected(self):
        for key, value in (("model_key", "qwen3-8b"), ("model_path", "/tmp/other-weights"),
                           ("output_root", "exps/causal_context/general_model_ld_coverage_v1"),
                           ("automatic_profile_search", True)):
            with self.subTest(key=key):
                config = self.config()
                config[key] = value
                with self.assertRaises(PackageError):
                    replication.validate_config(config)

    def toy_source(self, directory):
        names = ("config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json",
                 "merges.txt", "vocab.json", "model-01.safetensors")
        for name in names:
            (directory / name).write_bytes(b"opaque test content")
        (directory / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"layer": "model-01.safetensors"}}))

    def test_complete_weight_hash_inventory_and_tamper_detection(self):
        with tempfile.TemporaryDirectory() as temporary, patch.object(replication.base, "progress"):
            path = Path(temporary)
            self.toy_source(path)
            files = replication.model_inventory(path)
            model = {"path": str(path), "source_files": files}
            replication.check_model_inventory(model)
            altered = copy.deepcopy(model)
            del altered["source_files"]["model-01.safetensors"]
            with self.assertRaisesRegex(PackageError, "incomplete"):
                replication.check_model_inventory(altered)
            (path / "model-01.safetensors").write_bytes(b"altered test content")
            with self.assertRaises(PackageError):
                replication.check_model_inventory(model)

    def test_indirect_or_escaping_weight_source_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary, patch.object(replication.base, "progress"):
            path = Path(temporary)
            self.toy_source(path)
            (path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"layer": "../outside.safetensors"}}))
            with self.assertRaises(PackageError):
                replication.model_inventory(path)
            (path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"layer": "model-01.safetensors"}}))
            (path / "model-01.safetensors").unlink()
            (path / "model-01.safetensors").symlink_to(path / "vocab.json")
            with self.assertRaises(PackageError):
                replication.model_inventory(path)

    def test_added_monolithic_weight_or_adapter_cannot_change_loading_precedence(self):
        with tempfile.TemporaryDirectory() as temporary, patch.object(replication.base, "progress"):
            path = Path(temporary)
            self.toy_source(path)
            model = {"path": str(path), "source_files": replication.model_inventory(path)}
            for name in ("model.safetensors", "adapter_config.json"):
                with self.subTest(name=name):
                    (path / name).write_bytes(b"unregistered alternative")
                    with self.assertRaisesRegex(PackageError, "alternate"):
                        replication.check_model_inventory(model)
                    (path / name).unlink()
            del model["source_files"]["tokenizer_config.json"]
            with self.assertRaisesRegex(PackageError, "incomplete"):
                replication.check_model_inventory(model)

    def test_resolved_configuration_preserves_every_registration_field_without_aliasing(self):
        config = self.config()
        parent = {"config": {"analysis": {"bootstrap": {"seed": 42}}}}
        resolved = replication.resolved_config(config, parent)
        self.assertEqual({key: resolved[key] for key in config}, config)
        self.assertIs(resolved["test_access"], False)
        self.assertIs(resolved["query_gold_in_scoring"], False)
        resolved["analysis"]["bootstrap"]["seed"] = 0
        self.assertEqual(parent["config"]["analysis"]["bootstrap"]["seed"], 42)

    def test_runtime_never_uses_quantization_or_offload(self):
        runtime = replication.RUNTIME
        self.assertEqual(runtime["dtype"], "float32")
        self.assertEqual(runtime["batch_size"], 1)
        for key in ("tf32", "use_cache", "cpu_offload", "quantization", "automatic_profile_search", "enable_thinking"):
            self.assertIs(runtime[key], False)


if __name__ == "__main__":
    unittest.main()
