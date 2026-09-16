"""Registered FP32 layer-sharded scoring for the 14B and 27B extensions.

Only whole decoder layers move between GPUs. There is no tensor parallelism,
quantization, CPU/disk offload, KV cache, or automatic placement selection.
"""

from __future__ import annotations

import copy
import csv
import gc
import hashlib
import importlib.metadata
import json
import math
import os
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

from diagnostics.general_model_numeric_kernel import MAX_SEQUENCE_TOKENS, NumericKernelError, _prepare
from diagnostics.general_model_numeric_kernel_v2 import _result


ASSISTANT_EOS_TOKEN = "<|im_end|>"
MODEL_TYPES = {"qwen3": 2, "qwen3_5": 4}


def _hash(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _environment():
    names = ("torch", "transformers", "accelerate", "safetensors", "numpy", "scipy", "regex", "tokenizers")
    return {"python": sys.version.split()[0], "executable": str(Path(sys.executable).resolve()),
            "packages": {name: importlib.metadata.version(name) for name in names}}


def build_device_maps(model_config: dict, model_type: str | None = None) -> dict:
    model_type = model_type or model_config.get("model_type")
    if model_type not in MODEL_TYPES or model_config.get("model_type") != model_type:
        raise NumericKernelError("unsupported registered sharded architecture")
    text = model_config["text_config"] if model_type == "qwen3_5" else model_config
    count, stages = text["num_hidden_layers"], MODEL_TYPES[model_type]
    if type(count) is not int or count <= 0 or count % stages:
        raise NumericKernelError("decoder layers must divide exactly into the fixed stages")
    prefix = "model.language_model" if model_type == "qwen3_5" else "model"
    placements = {"baseline": [0, 1] if stages == 2 else [0, 1, 2, 3],
                  "replica": [2, 3] if stages == 2 else [1, 2, 3, 0]}
    maps = {}
    for name, devices in placements.items():
        mapping = {prefix + ".embed_tokens": devices[0], prefix + ".rotary_emb": devices[0],
                   prefix + ".norm": devices[-1], "lm_head": devices[-1]}
        mapping.update({f"{prefix}.layers.{index}": devices[index // (count // stages)] for index in range(count)})
        if model_type == "qwen3_5":
            mapping["model.visual"] = devices[0]
        maps[name] = mapping
    return maps


def _mapped_device(name, mapping):
    matches = [key for key in mapping if name == key or name.startswith(key + ".")]
    if not matches:
        raise NumericKernelError(f"tensor has no registered device placement: {name}")
    return mapping[max(matches, key=len)]


def validate_tensor_placement(model, mapping, torch) -> dict:
    rows = []
    for kind, iterator in (("parameter", model.named_parameters()), ("buffer", model.named_buffers())):
        for name, tensor in iterator:
            expected = _mapped_device(name, mapping)
            if tensor.device.type != "cuda" or tensor.device.index != expected:
                raise NumericKernelError(f"tensor is offloaded or misplaced: {name}: {tensor.device}")
            if tensor.is_floating_point() and tensor.dtype != torch.float32:
                raise NumericKernelError(f"non-FP32 model tensor: {name}: {tensor.dtype}")
            rows.append({"kind": kind, "name": name, "device": str(tensor.device),
                         "dtype": str(tensor.dtype), "shape": list(tensor.shape)})
    if not any(row["kind"] == "parameter" for row in rows):
        raise NumericKernelError("loaded model has no parameters")
    return {"tensor_count": len(rows), "all_parameters_and_buffers_on_registered_gpus": True,
            "all_floating_model_tensors_fp32": True, "tensor_layout_sha256": _hash(rows), "tensors": rows}


def validate_loading_info(info, checkpoint_keys, model_type):
    allowed = {key for key in checkpoint_keys if model_type == "qwen3_5" and key.startswith("mtp.")}
    if (info.get("missing_keys") or info.get("mismatched_keys") or info.get("error_msgs")
            or not set(info.get("unexpected_keys", ())) <= allowed):
        raise NumericKernelError("checkpoint loading differs from the registered full model (except unused MTP)")
    return {"missing_keys": [], "mismatched_keys": [], "unused_checkpoint_mtp_keys": sorted(allowed),
            "checkpoint_conversion": False, "multitoken_prediction_used": False}


def validate_native_delta(backbone, model_type):
    if model_type != "qwen3_5":
        return {"applicable": False, "native_linear_attention_layers": 0}
    from transformers.models.qwen3_5 import modeling_qwen3_5 as native

    if any(getattr(native, key) is not None for key in
           ("causal_conv1d_fn", "chunk_gated_delta_rule", "fused_recurrent_gated_delta_rule", "FusedRMSNormGated")):
        raise NumericKernelError("registered DeltaNet requires the unfused native torch implementation")
    count = 0
    for layer in backbone.layers:
        linear = getattr(layer, "linear_attn", None)
        if linear is None:
            continue
        if (linear.causal_conv1d_fn is not None
                or linear.causal_conv1d_update is not native.torch_causal_conv1d_update
                or linear.chunk_gated_delta_rule is not native.torch_chunk_gated_delta_rule
                or linear.recurrent_gated_delta_rule is not native.torch_recurrent_gated_delta_rule
                or type(linear.norm) is not native.Qwen3_5RMSNormGated):
            raise NumericKernelError("loaded DeltaNet layer selected an unregistered implementation")
        count += 1
    expected = sum(kind == "linear_attention" for kind in backbone.config.layer_types)
    if count != expected or count == 0:
        raise NumericKernelError("hybrid linear-attention layer inventory differs")
    return {"applicable": True, "native_linear_attention_layers": count,
            "implementation": "transformers-native-torch-chunk-gated-delta-rule", "fused_kernels": False}


@contextmanager
def fp32_operator_guard(torch, *, all_devices=False):
    from torch.utils._python_dispatch import TorchDispatchMode

    class Guard(TorchDispatchMode):
        def __init__(self):
            super().__init__()
            self.checked_operations = 0

        def check(self, value):
            if isinstance(value, torch.Tensor):
                if (all_devices or value.device.type == "cuda") and value.is_floating_point() and value.dtype != torch.float32:
                    raise NumericKernelError(f"reduced or mixed precision tensor in scoring operator: {value.dtype}")
            elif isinstance(value, (list, tuple)):
                for item in value:
                    self.check(item)
            elif isinstance(value, dict):
                for item in value.values():
                    self.check(item)

        def __torch_dispatch__(self, function, types, args=(), kwargs=None):
            self.check(args)
            self.check(kwargs)
            output = function(*args, **(kwargs or {}))
            self.check(output)
            self.checked_operations += 1
            return output

    with Guard() as guard:
        yield guard


def _hardware(torch, devices):
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None and visible != "0,1,2,3":
        raise NumericKernelError("physical placement requires unmasked CUDA or explicit 0,1,2,3 visibility")
    raw = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,name,memory.total",
                                   "--format=csv,noheader,nounits"], text=True)
    physical = {int(parts[0].strip()): (parts[1].strip(), parts[2].strip(), int(parts[3].strip()))
                for parts in csv.reader(raw.splitlines())}
    result = []
    for device in devices:
        properties = torch.cuda.get_device_properties(device)
        uuid, name, memory_mib = physical[device]
        observed = str(getattr(properties, "uuid", ""))
        if observed.lower().removeprefix("gpu-") != uuid.lower().removeprefix("gpu-"):
            raise NumericKernelError("CUDA logical device does not match its registered physical UUID")
        result.append({"physical_gpu_index": device, "logical_device": f"cuda:{device}",
                       "uuid": uuid, "name": name, "torch_name": properties.name,
                       "total_memory_bytes": memory_mib * 1024 ** 2, "nvml_total_memory_mib": memory_mib,
                       "torch_total_memory_bytes": properties.total_memory,
                       "torch_capacity_reporting": "zero-under-HAMI" if properties.total_memory == 0 else "reported",
                       "capability": list(torch.cuda.get_device_capability(device))})
    if len({row["uuid"] for row in result}) != len(devices):
        raise NumericKernelError("sharded stages do not occupy distinct GPUs")
    return result


def validate_runtime_identity(plan: dict, identity: dict, replica_shift: int = 0) -> None:
    """Validate a sealed runtime without importing torch or initializing CUDA."""
    if type(replica_shift) is not int or replica_shift not in (0, 1):
        raise NumericKernelError("unregistered physical remapping")
    specification = plan["model"]
    config = json.loads((Path(specification["path"]) / "config.json").read_text())
    maps = build_device_maps(config, specification["model_type"])
    placement = "replica" if replica_shift else "baseline"
    mapping = maps[placement]
    prefix = "model" if specification["model_type"] == "qwen3" else "model.language_model"
    expected = {
        "execution": "whole-layer-sharded-fp32", "model": specification,
        "numeric_runtime": plan["config"]["runtime"], "placement": placement, "replica_shift": replica_shift,
        "device_map": mapping, "device_map_sha256": _hash(mapping),
        "backbone": prefix, "input_device": f"cuda:{mapping[prefix + '.embed_tokens']}",
        "head_device": f"cuda:{mapping['lm_head']}", "text_only_forward": True, "vision_forward_used": False,
        "scoring_eos_token": ASSISTANT_EOS_TOKEN, "scoring_eos_token_id": plan["eos_token_id"],
        "tokenizer_pad_token_id": plan["pad_token_id"],
        "model_generation_eos_token_ids": plan["generation_eos_token_ids"],
        "transformer_dtype": "torch.float32", "lm_head_dtype": "torch.float32",
        "fp32_operator_dispatch_guard": True, "projection": "answer-and-eos-prediction-positions-only",
        "tf32_matmul": False, "tf32_cudnn": False, "use_cache": False,
        "deterministic_algorithms": True, "cudnn_benchmark": False,
        "bf16_reduced_precision_reduction": False, "fp16_reduced_precision_reduction": False,
        "default_dtype": "torch.float32", "cpu_threads": 4,
        "logprob_arithmetic": "float32", "aggregation": "float64",
        "cublas_workspace_config": ":4096:8", "environment": plan["environment"],
    }
    if plan["device_maps"] != maps or any(identity.get(key) != value for key, value in expected.items()):
        raise NumericKernelError("sealed sharded runtime differs from its registered model/placement/profile")
    devices = sorted(set(mapping.values()))
    hardware = identity.get("hardware", [])
    if ([row.get("physical_gpu_index") for row in hardware] != devices
            or len({row.get("uuid") for row in hardware}) != len(devices)):
        raise NumericKernelError("sealed runtime has missing or duplicated physical GPU identities")
    for row in hardware:
        if (row.get("logical_device") != f"cuda:{row['physical_gpu_index']}"
                or not str(row.get("uuid", "")).startswith("GPU-")
                or "L20" not in str(row.get("name", "")) or "L20" not in str(row.get("torch_name", ""))
                or type(row.get("nvml_total_memory_mib")) is not int or row["nvml_total_memory_mib"] <= 0
                or row.get("total_memory_bytes") != row["nvml_total_memory_mib"] * 1024 ** 2
                or type(row.get("torch_total_memory_bytes")) is not int or row["torch_total_memory_bytes"] < 0
                or row.get("torch_capacity_reporting") != ("zero-under-HAMI" if row["torch_total_memory_bytes"] == 0 else "reported")
                or not isinstance(row.get("capability"), list) or len(row["capability"]) != 2):
            raise NumericKernelError("sealed runtime hardware does not describe the registered L20 stages")
    tensors = identity.get("tensor_placement", {})
    rows = tensors.get("tensors", [])
    if (not rows or tensors.get("tensor_count") != len(rows) or tensors.get("tensor_layout_sha256") != _hash(rows)
            or tensors.get("all_parameters_and_buffers_on_registered_gpus") is not True
            or tensors.get("all_floating_model_tensors_fp32") is not True
            or len({row["name"] for row in rows}) != len(rows)):
        raise NumericKernelError("sealed tensor placement evidence is incomplete")
    for row in rows:
        integer_buffer = row["kind"] == "buffer" and row["dtype"] in (
            "torch.bool", "torch.uint8", "torch.int8", "torch.int16", "torch.int32", "torch.int64")
        if (row["kind"] not in ("parameter", "buffer") or row["device"] != f"cuda:{_mapped_device(row['name'], mapping)}"
                or (row["dtype"] != "torch.float32" and not integer_buffer)):
            raise NumericKernelError("sealed tensor evidence contains offload or non-FP32 weights")
    keys = json.loads((Path(specification["path"]) / "model.safetensors.index.json").read_text())["weight_map"]
    unused = {key for key in keys if specification["model_type"] == "qwen3_5" and key.startswith("mtp.")}
    if (not set(keys) - unused <= {row["name"] for row in rows}
            or any(row["name"] not in keys for row in rows if row["kind"] == "parameter")
            or identity.get("checkpoint_loading") != {
                "missing_keys": [], "mismatched_keys": [], "unused_checkpoint_mtp_keys": sorted(unused),
                "checkpoint_conversion": False, "multitoken_prediction_used": False}):
        raise NumericKernelError("sealed parameter inventory differs from the full checkpoint")
    delta = {"applicable": False, "native_linear_attention_layers": 0}
    if specification["model_type"] == "qwen3_5":
        delta = {"applicable": True, "native_linear_attention_layers":
                 sum(kind == "linear_attention" for kind in config["text_config"]["layer_types"]),
                 "implementation": "transformers-native-torch-chunk-gated-delta-rule", "fused_kernels": False}
    if identity.get("native_delta") != delta or not isinstance(identity.get("cuda_version"), str):
        raise NumericKernelError("sealed native attention or CUDA runtime evidence differs")


class ShardedRunner:
    def __init__(self, plan: dict, replica_shift: int = 0):
        if type(replica_shift) is not int or replica_shift not in (0, 1):
            raise NumericKernelError("only the registered baseline/physical-remapping placements are allowed")
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        if os.environ["CUBLAS_WORKSPACE_CONFIG"] != ":4096:8":
            raise NumericKernelError("unregistered cuBLAS workspace configuration")
        import torch
        from transformers import AutoTokenizer, Qwen3ForCausalLM, Qwen3_5ForConditionalGeneration

        runtime = copy.deepcopy(plan["config"]["runtime"])
        required = {"dtype": "float32", "attention_implementation": "eager", "use_cache": False,
                    "seed": 42, "cpu_threads": 4, "max_sequence_tokens": MAX_SEQUENCE_TOKENS,
                    "padding_policy": "dynamic", "padding_side": "right", "enable_thinking": False,
                    "batch_size": 1, "tf32": False, "bf16_reduced_precision_reduction": False,
                    "local_files_only": True, "trust_remote_code": False, "automatic_profile_search": False,
                    "cpu_offload": False, "quantization": False}
        if any(runtime.get(key) != value for key, value in required.items()):
            raise NumericKernelError("sharded numerical runtime differs from the registered FP32 profile")
        if _environment() != plan["environment"]:
            raise NumericKernelError("sharded environment differs from the registered environment")
        if not torch.cuda.is_available():
            raise NumericKernelError("sharded FP32 scoring requires CUDA")
        torch.set_num_threads(4)
        torch.set_default_dtype(torch.float32)
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
        specification = plan["model"]
        path = Path(specification["path"])
        config = json.loads((path / "config.json").read_text())
        maps = build_device_maps(config, specification["model_type"])
        if plan["device_maps"] != maps:
            raise NumericKernelError("device maps differ from the frozen whole-layer allocation")
        placement = "replica" if replica_shift else "baseline"
        self.device_map = maps[placement]
        self.devices = sorted(set(self.device_map.values()))
        self.hardware = _hardware(torch, self.devices)
        self.numeric_runtime = runtime
        self.replica_shift = replica_shift
        self.padding_extra = 0
        self.torch = torch
        self.model_type = specification["model_type"]
        self.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False, use_fast=True)
        self.eos_token_id, self.pad_token_id = plan["eos_token_id"], plan["pad_token_id"]
        if (type(self.eos_token_id) is not int or type(self.pad_token_id) is not int
                or min(self.eos_token_id, self.pad_token_id) < 0
                or self.tokenizer.encode(ASSISTANT_EOS_TOKEN, add_special_tokens=False) != [self.eos_token_id]
                or self.tokenizer.eos_token_id != self.eos_token_id or self.tokenizer.pad_token_id != self.pad_token_id):
            raise NumericKernelError("registered assistant EOS/padding differs from the tokenizer")
        self.eos_ids = {self.eos_token_id}
        model_class = Qwen3ForCausalLM if self.model_type == "qwen3" else Qwen3_5ForConditionalGeneration
        self.model, loading = model_class.from_pretrained(
            path, local_files_only=True, trust_remote_code=False, use_safetensors=True,
            dtype=torch.float32, attn_implementation="eager", device_map=self.device_map,
            output_loading_info=True)
        self.model.eval()
        self.model.config.use_cache = False
        if hasattr(self.model.config, "text_config"):
            self.model.config.text_config.use_cache = False
        keys = json.loads((path / "model.safetensors.index.json").read_text())["weight_map"]
        loaded = validate_loading_info(loading, keys, self.model_type)
        tensor_audit = validate_tensor_placement(self.model, self.device_map, torch)
        self.backbone = self.model.model if self.model_type == "qwen3" else self.model.model.language_model
        self.head = self.model.lm_head
        self.device = self.backbone.embed_tokens.weight.device
        self.head_device = self.head.weight.device
        native = validate_native_delta(self.backbone, self.model_type)
        attention_configs = [self.model.config, self.backbone.config]
        if any(getattr(config, "_attn_implementation", None) != "eager" for config in attention_configs):
            raise NumericKernelError("full-attention backend is not eager")
        generation_eos = self.model.generation_config.eos_token_id
        self.identity = {
            "execution": "whole-layer-sharded-fp32", "model": copy.deepcopy(specification),
            "numeric_runtime": runtime, "placement": placement, "replica_shift": replica_shift,
            "device_map": self.device_map, "device_map_sha256": _hash(self.device_map),
            "hardware": self.hardware, "tensor_placement": tensor_audit,
            "checkpoint_loading": loaded, "native_delta": native,
            "backbone": "model" if self.model_type == "qwen3" else "model.language_model",
            "input_device": str(self.device), "head_device": str(self.head_device),
            "text_only_forward": True, "vision_forward_used": False,
            "scoring_eos_token": ASSISTANT_EOS_TOKEN, "scoring_eos_token_id": self.eos_token_id,
            "tokenizer_pad_token_id": self.pad_token_id, "model_generation_eos_token_ids":
                [generation_eos] if isinstance(generation_eos, int) else generation_eos,
            "transformer_dtype": "torch.float32", "lm_head_dtype": "torch.float32",
            "fp32_operator_dispatch_guard": True, "projection": "answer-and-eos-prediction-positions-only",
            "tf32_matmul": torch.backends.cuda.matmul.allow_tf32, "tf32_cudnn": torch.backends.cudnn.allow_tf32,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "bf16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
            "fp16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
            "default_dtype": str(torch.get_default_dtype()), "cpu_threads": torch.get_num_threads(), "use_cache": False,
            "logprob_arithmetic": "float32", "aggregation": "float64",
            "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
            "environment": _environment(),
            "cuda_version": torch.version.cuda,
        }
        validate_runtime_identity(plan, self.identity, replica_shift)
        self._closed = False

    def close(self):
        if getattr(self, "_closed", False):
            return
        self._closed = True
        torch = self.torch
        _synchronize(self)
        self.backbone = self.head = self.model = None
        gc.collect()
        for device in self.devices:
            with torch.cuda.device(device):
                torch.cuda.empty_cache()


def _synchronize(runner, *, reset=False):
    for device in runner.devices:
        runner.torch.cuda.synchronize(device)
        if reset:
            runner.torch.cuda.reset_peak_memory_stats(device)


def _peaks(runner):
    return {str(device): {"allocated_bytes": runner.torch.cuda.max_memory_allocated(device),
                          "reserved_bytes": runner.torch.cuda.max_memory_reserved(device)} for device in runner.devices}


def _annotate(runner, result, elapsed, peaks, operators):
    result.update(physical_gpu_indices=list(runner.devices),
                  physical_gpu_uuids=[row["uuid"] for row in runner.hardware],
                  model_device_map_sha256=_hash(runner.device_map), replica_shift=runner.replica_shift,
                  peak_memory_by_device=peaks, forward_seconds=elapsed,
                  fp32_operator_dispatch_checked=True, fp32_operator_count=operators,
                  timing_scope="all-sharded-devices-synchronized-forward-projection-normalization-and-optional-reference")
    return result


def _target_values(runner, hidden, targets, *, reference):
    torch = runner.torch
    if hidden.dtype != torch.float32 or runner.head.weight.dtype != torch.float32:
        raise NumericKernelError("hidden states and output head must already be FP32")
    with fp32_operator_guard(torch) as guard:
        logits = runner.head(hidden.to(device=runner.head.weight.device, dtype=torch.float32))
        if logits.dtype != torch.float32 or not torch.isfinite(logits).all().item():
            raise NumericKernelError("target logits are non-finite or not FP32")
        ids = torch.tensor(targets, dtype=torch.long, device=logits.device)
        values = logits.gather(-1, ids[:, None]).squeeze(-1) - logits.logsumexp(-1)
        if not torch.isfinite(values).all().item():
            raise NumericKernelError("non-finite target logprob")
    refs = None
    if reference:
        logits64 = logits.to(device="cpu", dtype=torch.float64)
        refs = (logits64.gather(-1, ids.cpu()[:, None]).squeeze(-1) - logits64.logsumexp(-1)).tolist()
    return values.cpu().tolist(), refs, guard.checked_operations


def score_batch(runner, items: list[dict], *, reference=False) -> list[dict]:
    if not items:
        return []
    if len(items) != 1:
        raise NumericKernelError("sharded production requires true candidate batch one")
    torch = runner.torch
    eos = runner.eos_token_id
    row = _prepare(runner, items[0], eos, MAX_SEQUENCE_TOKENS)
    extra = runner.padding_extra
    if type(extra) is not int or extra not in (0, 64):
        raise NumericKernelError("only the registered +64 padding challenge is allowed")
    length = len(row["sequence"]) + extra
    if length > MAX_SEQUENCE_TOKENS:
        raise NumericKernelError("padding challenge exceeds the registered sequence limit")
    inputs = torch.full((1, length), runner.pad_token_id, dtype=torch.long, device=runner.device)
    inputs[0, :len(row["sequence"])] = torch.tensor(row["sequence"], dtype=torch.long, device=runner.device)
    mask = torch.zeros_like(inputs)
    mask[0, :len(row["sequence"])] = 1
    _synchronize(runner, reset=True)
    started = time.monotonic()
    with torch.inference_mode():
        with fp32_operator_guard(torch) as guard:
            hidden = runner.backbone(input_ids=inputs, attention_mask=mask, use_cache=False).last_hidden_state
        if hidden.ndim != 3 or tuple(hidden.shape[:2]) != tuple(inputs.shape):
            raise NumericKernelError("text backbone returned a different teacher-forcing geometry")
        start = row["prompt_tokens"] - 1
        targets = row["answer_token_ids"] + [eos]
        selected = hidden[0, start:start + len(targets)]
        values, refs, projected_ops = _target_values(runner, selected, targets, reference=reference)
    _synchronize(runner)
    elapsed, peaks = time.monotonic() - started, _peaks(runner)
    result = _result(row, values, refs, eos=eos, padded_length=length, batch_size=1,
                     logits_dtype="torch.float32", forward_seconds=elapsed,
                     peak=max(value["allocated_bytes"] for value in peaks.values()))
    result["padding_challenge_extra"] = extra
    return [_annotate(runner, result, elapsed, peaks, guard.checked_operations + projected_ops)]


def score_prefix_block(runner, context: dict, catalog: list[dict], *, reference=False) -> list[dict]:
    if runner.padding_extra:
        raise NumericKernelError("prefix scoring must remain unpadded")
    torch, eos = runner.torch, runner.eos_token_id
    prepared = [_prepare(runner, {"context": context, "candidate": candidate}, eos, MAX_SEQUENCE_TOKENS)
                for candidate in catalog]
    if not prepared:
        raise NumericKernelError("prefix catalog cannot be empty")
    prefix_targets = {}
    for row in prepared:
        for index, target in enumerate(row["answer_token_ids"] + [eos]):
            prefix_targets.setdefault(tuple(row["answer_token_ids"][:index]), set()).add(target)
    prompt = prepared[0]["sequence"][:prepared[0]["prompt_tokens"]]
    scores, operators = {}, 0
    _synchronize(runner, reset=True)
    started = time.monotonic()
    with torch.inference_mode():
        for prefix, target_set in prefix_targets.items():
            inputs = torch.tensor([prompt + list(prefix)], dtype=torch.long, device=runner.device)
            with fp32_operator_guard(torch) as guard:
                hidden = runner.backbone(input_ids=inputs, attention_mask=torch.ones_like(inputs),
                                         use_cache=False).last_hidden_state[:, -1, :]
            targets = sorted(target_set)
            if hidden.dtype != torch.float32 or runner.head.weight.dtype != torch.float32:
                raise NumericKernelError("prefix hidden states and output head must already be FP32")
            # One projection and normalizer per unique prefix, shared by branches.
            with fp32_operator_guard(torch) as head_guard:
                logits = runner.head(hidden.to(device=runner.head.weight.device, dtype=torch.float32))[0]
                if logits.dtype != torch.float32 or not torch.isfinite(logits).all().item():
                    raise NumericKernelError("prefix logits are non-finite or not FP32")
                ids = torch.tensor(targets, dtype=torch.long, device=logits.device)
                values = (logits[ids] - logits.logsumexp(0)).cpu().tolist()
            if not all(math.isfinite(value) for value in values):
                raise NumericKernelError("non-finite prefix logprob")
            refs = None
            if reference:
                logits64 = logits.to(device="cpu", dtype=torch.float64)
                refs = (logits64[ids.cpu()] - logits64.logsumexp(0)).tolist()
            for index, target in enumerate(targets):
                scores[(prefix, target)] = (values[index], refs[index] if reference else None)
            operators += guard.checked_operations + head_guard.checked_operations
    _synchronize(runner)
    elapsed, peaks = time.monotonic() - started, _peaks(runner)
    results = []
    for row in prepared:
        entries = [scores[(tuple(row["answer_token_ids"][:index]), target)]
                   for index, target in enumerate(row["answer_token_ids"] + [eos])]
        result = _result(row, [value[0] for value in entries], [value[1] for value in entries] if reference else None,
                         eos=eos, padded_length=None, batch_size=1, logits_dtype="torch.float32",
                         forward_seconds=elapsed / len(prepared),
                         peak=max(value["allocated_bytes"] for value in peaks.values()), prefix=True)
        result.update(prefix_unique_forward_count=len(prefix_targets), prefix_padding=False)
        results.append(_annotate(runner, result, elapsed / len(prepared), peaks, operators))
    return results
