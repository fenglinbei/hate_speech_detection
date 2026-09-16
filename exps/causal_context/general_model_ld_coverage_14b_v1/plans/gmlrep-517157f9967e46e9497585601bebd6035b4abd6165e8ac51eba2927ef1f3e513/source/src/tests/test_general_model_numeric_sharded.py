"""CPU-only synthetic and installed-backend tests; no checkpoint weights load."""

import copy
import hashlib
import json
import math
import os
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

from diagnostics import general_model_numeric_sharded as sharded
from diagnostics.general_model_numeric_analysis import candidate_catalog


class FakeCuda:
    def __init__(self):
        self.calls = []

    def synchronize(self, device):
        self.calls.append(("sync", device))

    def reset_peak_memory_stats(self, device):
        self.calls.append(("reset", device))

    def max_memory_allocated(self, device):
        return 100 + device

    def max_memory_reserved(self, device):
        return 200 + device

    def device(self, device):
        self.calls.append(("device", device))
        return nullcontext()

    def empty_cache(self):
        self.calls.append(("empty_cache", None))


class TorchProxy:
    def __init__(self):
        self.cuda = FakeCuda()

    def __getattr__(self, name):
        return getattr(torch, name)


class Tokenizer:
    eos_token_id = 127
    pad_token_id = 0

    def encode(self, text, add_special_tokens=False):
        return [ord(character) + 1 for character in text]


class Backbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = torch.nn.Embedding(128, 8)
        with torch.no_grad():
            self.embed_tokens.weight.copy_(torch.arange(1024).reshape(128, 8).float() / 1000)
        self.calls = []

    def forward(self, input_ids, attention_mask, use_cache):
        assert use_cache is False
        self.calls.append({"input_ids": input_ids.clone(), "mask": attention_mask.clone(), "use_cache": use_cache})
        return SimpleNamespace(last_hidden_state=self.embed_tokens(input_ids).cumsum(dim=1))


def runner_fixture(devices=(0, 1)):
    runner = sharded.ShardedRunner.__new__(sharded.ShardedRunner)
    runner.torch = TorchProxy()
    runner.tokenizer = Tokenizer()
    runner.eos_token_id, runner.pad_token_id = 127, 0
    runner.eos_ids = {127}
    runner.device = torch.device("cpu")
    runner.devices = list(devices)
    runner.replica_shift = 0
    runner.device_map = {"model.embed_tokens": devices[0], "lm_head": devices[-1]}
    runner.hardware = [{"uuid": f"GPU-synthetic-{device}"} for device in devices]
    runner.padding_extra = 0
    runner.backbone = Backbone()
    runner.head = torch.nn.Linear(8, 128, bias=False)
    with torch.no_grad():
        runner.head.weight.copy_(torch.arange(1024).reshape(128, 8).float().sin() / 100)
    runner.model = SimpleNamespace(model=runner.backbone, lm_head=runner.head)
    runner._closed = False
    text = "prompt:"
    tokens = runner.tokenizer.encode(text)
    context = {"prompt_text": text, "prompt_tokens": len(tokens),
               "prompt_sha256": hashlib.sha256(text.encode()).hexdigest(),
               "prompt_token_ids_sha256": sharded._hash(tokens)}
    return runner, context


class DeviceMapTests(unittest.TestCase):
    def test_14b_fixed_twenty_layer_stages_and_disjoint_physical_remapping(self):
        maps = sharded.build_device_maps({"model_type": "qwen3", "num_hidden_layers": 40})
        self.assertEqual(set(maps["baseline"].values()), {0, 1})
        self.assertEqual(set(maps["replica"].values()), {2, 3})
        self.assertEqual(maps["baseline"]["model.layers.19"], 0)
        self.assertEqual(maps["baseline"]["model.layers.20"], 1)
        self.assertEqual(maps["baseline"]["model.rotary_emb"], 0)
        self.assertEqual(maps["baseline"]["lm_head"], 1)
        self.assertTrue(all(maps["baseline"][name] != maps["replica"][name] for name in maps["baseline"]))

    def test_27b_whole_vl_checkpoint_map_keeps_vision_and_rotates_every_stage(self):
        maps = sharded.build_device_maps({"model_type": "qwen3_5", "text_config": {"num_hidden_layers": 64}})
        for index in range(64):
            name = f"model.language_model.layers.{index}"
            self.assertEqual(maps["baseline"][name], index // 16)
            self.assertEqual(maps["replica"][name], (index // 16 + 1) % 4)
        self.assertEqual(maps["baseline"]["model.visual"], 0)
        self.assertEqual(maps["replica"]["model.visual"], 1)
        self.assertEqual(maps["baseline"]["lm_head"], 3)
        self.assertEqual(maps["replica"]["lm_head"], 0)
        self.assertFalse(any(value in ("auto", "cpu", "disk") for value in maps["baseline"].values()))

    def test_unknown_model_mismatched_type_and_uneven_layers_fail(self):
        for config, kind in (({"model_type": "qwen2", "num_hidden_layers": 40}, None),
                             ({"model_type": "qwen3", "num_hidden_layers": 39}, None),
                             ({"model_type": "qwen3", "num_hidden_layers": 40}, "qwen3_5")):
            with self.subTest(config=config), self.assertRaises(sharded.NumericKernelError):
                sharded.build_device_maps(config, kind)


class ScoringTests(unittest.TestCase):
    def test_teacher_forcing_shift_eos_separation_and_same_logits_cpu64_reference(self):
        runner, context = runner_fixture()
        candidate = candidate_catalog()["hate"][0]
        result = sharded.score_batch(runner, [{"context": context, "candidate": candidate}], reference=True)[0]
        ids = runner.tokenizer.encode(context["prompt_text"] + candidate["canonical_answer"]) + [127]
        with torch.inference_mode():
            hidden = runner.backbone(torch.tensor([ids]), torch.ones(1, len(ids), dtype=torch.long), False).last_hidden_state
            logits = runner.head(hidden[0, context["prompt_tokens"] - 1:len(ids) - 1])
            targets = torch.tensor(runner.tokenizer.encode(candidate["canonical_answer"]) + [127])
            expected = logits.log_softmax(-1).gather(-1, targets[:, None]).squeeze(-1)
            expected64 = logits.double().log_softmax(-1).gather(-1, targets[:, None]).squeeze(-1)
        self.assertEqual(len(result["token_logprobs"]), len(targets) - 1)
        self.assertAlmostEqual(result["answer_sum"], math.fsum(expected[:-1].tolist()), places=5)
        self.assertAlmostEqual(result["eos_logprob"], expected[-1].item(), places=6)
        self.assertAlmostEqual(result["reference_scores"]["total_with_eos"], math.fsum(expected64.tolist()), places=12)
        self.assertEqual(result["reference_arithmetic_dtype"], "cpu.torch.float64")
        self.assertTrue(result["fp32_operator_dispatch_checked"])
        self.assertGreater(result["fp32_operator_count"], 0)
        self.assertTrue(all(call["use_cache"] is False for call in runner.backbone.calls))

    def test_all_devices_synchronized_and_peaks_recorded_not_only_input_device(self):
        runner, context = runner_fixture((0, 1, 2, 3))
        result = sharded.score_batch(runner, [{"context": context, "candidate": candidate_catalog()["hate"][0]}])[0]
        for device in runner.devices:
            self.assertEqual(runner.torch.cuda.calls.count(("sync", device)), 2)
            self.assertEqual(runner.torch.cuda.calls.count(("reset", device)), 1)
            self.assertEqual(result["peak_memory_by_device"][str(device)],
                             {"allocated_bytes": 100 + device, "reserved_bytes": 200 + device})
        self.assertEqual(result["physical_gpu_indices"], [0, 1, 2, 3])

    def test_padding_and_execution_permutation_preserve_answers_with_true_batch_one(self):
        runner, context = runner_fixture()
        catalog = candidate_catalog()["hate"]
        before = {row["candidate_id"]: sharded.score_batch(runner, [{"context": context, "candidate": row}])[0]
                  for row in catalog}
        runner.padding_extra = 64
        after = {row["candidate_id"]: sharded.score_batch(runner, [{"context": context, "candidate": row}])[0]
                 for row in reversed(catalog)}
        for key in before:
            self.assertEqual(before[key]["token_logprobs"], after[key]["token_logprobs"])
            self.assertEqual(after[key]["padded_sequence_tokens"], before[key]["padded_sequence_tokens"] + 64)
        with self.assertRaises(sharded.NumericKernelError):
            sharded.score_batch(runner, [{"context": context, "candidate": row} for row in catalog])
        runner.padding_extra = 1
        with self.assertRaises(sharded.NumericKernelError):
            sharded.score_batch(runner, [{"context": context, "candidate": catalog[0]}])

    def test_prefix_shares_conditionals_exactly_and_matches_full_sequence(self):
        runner, context = runner_fixture()
        catalog = candidate_catalog()["group"][:4]
        prefix = sharded.score_prefix_block(runner, context, catalog, reference=True)
        unique = {tuple(runner.tokenizer.encode(candidate["canonical_answer"])[:index])
                  for candidate in catalog for index in range(len(runner.tokenizer.encode(candidate["canonical_answer"])) + 1)}
        self.assertEqual(len(runner.backbone.calls), len(unique))
        seen = {}
        for candidate, score in zip(catalog, prefix, strict=True):
            full = sharded.score_batch(runner, [{"context": context, "candidate": candidate}], reference=True)[0]
            self.assertEqual(score["prefix_unique_forward_count"], len(unique))
            self.assertIsNone(score["padded_sequence_tokens"])
            targets = score["answer_token_ids"] + [127]
            values = score["token_logprobs"] + [score["eos_logprob"]]
            for index, (target, value) in enumerate(zip(targets, values, strict=True)):
                key = (tuple(targets[:index]), target)
                if key in seen:
                    self.assertEqual(value, seen[key])
                seen[key] = value
            for key in ("answer_sum", "answer_mean", "eos_logprob", "total_with_eos", "mean_with_eos"):
                self.assertAlmostEqual(score[key], full[key], places=5)
        runner.padding_extra = 64
        with self.assertRaises(sharded.NumericKernelError):
            sharded.score_prefix_block(runner, context, catalog)

    def test_nonfinite_or_lower_precision_hidden_states_are_rejected(self):
        runner, _ = runner_fixture()
        for hidden in (torch.ones(2, 8, dtype=torch.float16), torch.full((2, 8), float("nan"))):
            with self.assertRaises(sharded.NumericKernelError):
                sharded._target_values(runner, hidden, [1, 2], reference=True)

    def test_operator_guard_rejects_output_downcast_but_cpu_reference_outside_is_valid(self):
        with sharded.fp32_operator_guard(torch, all_devices=True) as guard:
            self.assertEqual((torch.ones(2) + 1).dtype, torch.float32)
        self.assertGreater(guard.checked_operations, 0)
        with self.assertRaises(sharded.NumericKernelError), sharded.fp32_operator_guard(torch, all_devices=True):
            torch.ones(2).to(torch.bfloat16)
        self.assertEqual(torch.ones(2).double().logsumexp(0).dtype, torch.float64)

    def test_close_releases_model_then_clears_every_device_cache_idempotently(self):
        runner, _ = runner_fixture((0, 1, 2, 3))
        runner.close()
        runner.close()
        self.assertIsNone(runner.model)
        self.assertIsNone(runner.backbone)
        self.assertIsNone(runner.head)
        self.assertEqual(runner.torch.cuda.calls.count(("empty_cache", None)), 4)


class LoadingAndBackendTests(unittest.TestCase):
    def test_only_exact_declared_unused_mtp_keys_may_be_unexpected(self):
        keys = {"model.language_model.embed_tokens.weight", "lm_head.weight", "mtp.fc.weight"}
        info = {"missing_keys": [], "mismatched_keys": [], "unexpected_keys": ["mtp.fc.weight"]}
        self.assertEqual(sharded.validate_loading_info(info, keys, "qwen3_5")["unused_checkpoint_mtp_keys"], ["mtp.fc.weight"])
        self.assertEqual(sharded.validate_loading_info({**info, "unexpected_keys": []}, keys, "qwen3_5")
                         ["unused_checkpoint_mtp_keys"], ["mtp.fc.weight"])
        for wrong in ({**info, "missing_keys": ["lm_head.weight"]},
                      {**info, "mismatched_keys": ["lm_head.weight"]},
                      {**info, "unexpected_keys": ["model.visual.weight"]}):
            with self.assertRaises(sharded.NumericKernelError):
                sharded.validate_loading_info(wrong, keys, "qwen3_5")
        with self.assertRaises(sharded.NumericKernelError):
            sharded.validate_loading_info(info, keys, "qwen3")

    def test_installed_qwen35_native_torch_delta_is_explicitly_checked(self):
        from transformers.models.qwen3_5 import modeling_qwen3_5 as native
        config = native.Qwen3_5TextConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=4,
            num_attention_heads=4, num_key_value_heads=2, head_dim=8,
            linear_num_key_heads=2, linear_num_value_heads=2, linear_key_head_dim=8,
            linear_value_head_dim=8, layer_types=["linear_attention"] * 3 + ["full_attention"], vocab_size=128,
            rope_parameters={"rope_type": "default", "rope_theta": 10000, "partial_rotary_factor": 1.0,
                             "mrope_section": [2, 1, 1], "mrope_interleaved": True})
        config._attn_implementation = "eager"
        backbone = native.Qwen3_5TextModel(config).float().eval()
        result = sharded.validate_native_delta(backbone, "qwen3_5")
        self.assertEqual(result["native_linear_attention_layers"], 3)
        with torch.inference_mode(), sharded.fp32_operator_guard(torch, all_devices=True) as guard:
            hidden = backbone(input_ids=torch.tensor([[1, 2, 3, 4, 5]]),
                              attention_mask=torch.ones(1, 5, dtype=torch.long), use_cache=False).last_hidden_state
        self.assertEqual(tuple(hidden.shape), (1, 5, 32))
        self.assertEqual(hidden.dtype, torch.float32)
        self.assertGreater(guard.checked_operations, 100)
        with mock.patch.object(native, "chunk_gated_delta_rule", object()), self.assertRaises(sharded.NumericKernelError):
            sharded.validate_native_delta(backbone, "qwen3_5")


def sealed_identity_fixture(directory, *, model_type="qwen3", replica_shift=0):
    from diagnostics.general_model_coverage_replication import RUNTIME

    config = {"model_type": model_type, "num_hidden_layers": 4}
    if model_type == "qwen3_5":
        config["text_config"] = {"num_hidden_layers": 8,
                                  "layer_types": ["linear_attention"] * 3 + ["full_attention"]}
    maps = sharded.build_device_maps(config)
    placement = "replica" if replica_shift else "baseline"
    mapping = maps[placement]
    rows = [{"name": name + (".inv_freq" if name.endswith("rotary_emb") else ".weight"),
             "kind": "buffer" if name.endswith("rotary_emb") else "parameter",
             "device": f"cuda:{device}", "dtype": "torch.float32", "shape": [2, 2]}
            for name, device in mapping.items()]
    keys = {row["name"]: "model-1.safetensors" for row in rows if row["kind"] == "parameter"}
    if model_type == "qwen3_5":
        keys["mtp.fc.weight"] = "model-1.safetensors"
    for name, content in (("config.json", config), ("model.safetensors.index.json", {"weight_map": keys})):
        with (directory / name).open("w") as handle:
            json.dump(content, handle)
    plan = {"model": {"key": "synthetic", "model_type": model_type, "path": str(directory), "source_files": {}},
            "config": {"runtime": copy.deepcopy(RUNTIME)}, "environment": sharded._environment(),
            "eos_token_id": 127, "pad_token_id": 0, "generation_eos_token_ids": [127, 0], "device_maps": maps}
    prefix = "model" if model_type == "qwen3" else "model.language_model"
    identity = {
        "execution": "whole-layer-sharded-fp32", "model": plan["model"], "numeric_runtime": plan["config"]["runtime"],
        "placement": placement, "replica_shift": replica_shift, "device_map": mapping,
        "device_map_sha256": sharded._hash(mapping), "backbone": prefix,
        "input_device": f"cuda:{mapping[prefix + '.embed_tokens']}", "head_device": f"cuda:{mapping['lm_head']}",
        "text_only_forward": True, "vision_forward_used": False, "scoring_eos_token": "<|im_end|>",
        "scoring_eos_token_id": 127, "tokenizer_pad_token_id": 0, "model_generation_eos_token_ids": [127, 0],
        "transformer_dtype": "torch.float32", "lm_head_dtype": "torch.float32", "fp32_operator_dispatch_guard": True,
        "projection": "answer-and-eos-prediction-positions-only", "tf32_matmul": False, "tf32_cudnn": False,
        "deterministic_algorithms": True, "cudnn_benchmark": False, "bf16_reduced_precision_reduction": False,
        "fp16_reduced_precision_reduction": False, "default_dtype": "torch.float32", "cpu_threads": 4,
        "use_cache": False, "logprob_arithmetic": "float32", "aggregation": "float64",
        "cublas_workspace_config": ":4096:8", "environment": plan["environment"], "cuda_version": "12.4",
        "hardware": [{"physical_gpu_index": device, "logical_device": f"cuda:{device}", "uuid": f"GPU-{device}",
                      "name": "NVIDIA L20", "torch_name": "NVIDIA L20", "nvml_total_memory_mib": 46068,
                      "total_memory_bytes": 46068 * 1024 ** 2, "torch_total_memory_bytes": 0,
                      "torch_capacity_reporting": "zero-under-HAMI", "capability": [8, 9]}
                     for device in sorted(set(mapping.values()))],
        "tensor_placement": {"tensor_count": len(rows), "tensor_layout_sha256": sharded._hash(rows), "tensors": rows,
                             "all_parameters_and_buffers_on_registered_gpus": True, "all_floating_model_tensors_fp32": True},
        "checkpoint_loading": sharded.validate_loading_info({"unexpected_keys": []}, keys, model_type),
        "native_delta": ({"applicable": False, "native_linear_attention_layers": 0} if model_type == "qwen3" else
                         {"applicable": True, "native_linear_attention_layers": 3,
                          "implementation": "transformers-native-torch-chunk-gated-delta-rule", "fused_kernels": False}),
    }
    return plan, identity


class SealedIdentityTests(unittest.TestCase):
    def test_both_fixed_topologies_and_physical_challenges_validate_without_cuda(self):
        with tempfile.TemporaryDirectory() as temporary:
            for model_type in ("qwen3", "qwen3_5"):
                for shift in (0, 1):
                    plan, identity = sealed_identity_fixture(Path(temporary), model_type=model_type, replica_shift=shift)
                    with self.subTest(model_type=model_type, shift=shift), \
                            mock.patch.object(torch.cuda, "is_available", side_effect=AssertionError("must remain CPU-only")):
                        sharded.validate_runtime_identity(plan, identity, shift)

    def test_hami_zero_torch_capacity_does_not_waive_nvml_uuid_or_numerical_flags(self):
        with tempfile.TemporaryDirectory() as temporary:
            plan, original = sealed_identity_fixture(Path(temporary))
            for kind in ("nvml", "uuid", "fp16", "deterministic", "tf32"):
                identity = copy.deepcopy(original)
                if kind == "nvml":
                    identity["hardware"][0]["nvml_total_memory_mib"] = 0
                elif kind == "uuid":
                    identity["hardware"][1]["uuid"] = identity["hardware"][0]["uuid"]
                else:
                    identity[{"fp16": "fp16_reduced_precision_reduction", "deterministic": "deterministic_algorithms",
                              "tf32": "tf32_matmul"}[kind]] = kind != "deterministic"
                with self.subTest(kind=kind), self.assertRaises(sharded.NumericKernelError):
                    sharded.validate_runtime_identity(plan, identity)

    def test_rehashed_offload_lower_precision_or_missing_visual_tensor_still_fails(self):
        with tempfile.TemporaryDirectory() as temporary:
            plan, original = sealed_identity_fixture(Path(temporary), model_type="qwen3_5")
            for kind in ("offload", "dtype", "visual"):
                identity = copy.deepcopy(original)
                proof = identity["tensor_placement"]
                if kind == "visual":
                    proof["tensors"] = [row for row in proof["tensors"] if not row["name"].startswith("model.visual")]
                else:
                    proof["tensors"][0]["device" if kind == "offload" else "dtype"] = "cpu" if kind == "offload" else "torch.bfloat16"
                proof["tensor_layout_sha256"] = sharded._hash(proof["tensors"])
                proof["tensor_count"] = len(proof["tensors"])
                with self.subTest(kind=kind), self.assertRaises(sharded.NumericKernelError):
                    sharded.validate_runtime_identity(plan, identity)

    def test_hardware_records_nvml_capacity_separately_from_hami_torch_zero(self):
        cuda = SimpleNamespace(get_device_properties=lambda device: SimpleNamespace(
            uuid=f"GPU-{device}", name="NVIDIA L20", total_memory=0), get_device_capability=lambda device: (8, 9))
        text = "0, GPU-0, NVIDIA L20, 46068\n1, GPU-1, NVIDIA L20, 46068\n"
        with mock.patch.dict(os.environ, {}, clear=True), mock.patch.object(sharded.subprocess, "check_output", return_value=text):
            result = sharded._hardware(SimpleNamespace(cuda=cuda), [0, 1])
        self.assertEqual(result[0]["total_memory_bytes"], 46068 * 1024 ** 2)
        self.assertEqual(result[0]["torch_total_memory_bytes"], 0)
        self.assertEqual(result[0]["torch_capacity_reporting"], "zero-under-HAMI")


class FullArchitectureMetaTests(unittest.TestCase):
    def test_real_14b_and_27b_module_names_cover_original_checkpoint_without_loading_weights(self):
        from accelerate import init_empty_weights
        from transformers import AutoConfig, Qwen3ForCausalLM, Qwen3_5ForConditionalGeneration

        for path, model_type, model_class in (
                (Path("/data/models/Qwen3-14B"), "qwen3", Qwen3ForCausalLM),
                (Path("/data/models/Qwen/Qwen3.8-27B"), "qwen3_5", Qwen3_5ForConditionalGeneration)):
            with self.subTest(model_type=model_type):
                config = AutoConfig.from_pretrained(path, local_files_only=True, trust_remote_code=False)
                config._attn_implementation = "eager"
                if model_type == "qwen3_5":
                    config.text_config._attn_implementation = config.vision_config._attn_implementation = "eager"
                with init_empty_weights(include_buffers=True):
                    model = model_class(config)
                self.assertTrue(all(parameter.device.type == "meta" for parameter in model.parameters()))
                keys = json.loads((path / "model.safetensors.index.json").read_text())["weight_map"]
                expected = {key for key in keys if not (model_type == "qwen3_5" and key.startswith("mtp."))}
                self.assertEqual({name for name, _ in model.named_parameters()}, expected)
                maps = sharded.build_device_maps(json.loads((path / "config.json").read_text()), model_type)
                for mapping in maps.values():
                    for name, _ in [*model.named_parameters(), *model.named_buffers()]:
                        self.assertIsInstance(sharded._mapped_device(name, mapping), int)


if __name__ == "__main__":
    unittest.main()
