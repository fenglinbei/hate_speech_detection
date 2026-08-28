from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
import shutil
from pathlib import Path
from typing import Any, Optional, Union

import pandas as pd
import torch
from datasets import Dataset
from transformers import (  # type: ignore
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)

from data.training_schedule import (
    FIXED_PRESENTATION_CONFIG,
    PARTITION_CONFIG,
    validate_training_schedule,
)
from data.training_artifacts import TrainingArtifactError, canonical_json_bytes
from finetune.stage1_runtime import (
    RUNTIME_VALIDATION_MARKER,
    Stage1RuntimeBundle,
    Stage1RuntimeError,
    build_schedule_aware_datasets,
    build_training_receipt,
    revalidate_stage1_runtime_sources,
    resolve_stage1_runtime_inputs,
    verified_stage1_runtime_source_load,
    write_immutable_training_receipt,
)
from metrics.metric_llm import LLMmetrics
from prompt import *
from utils.log import init_logger

logger = init_logger(level="INFO", show_console=True)

HF_WEIGHT_FILES = (
    "pytorch_model.bin",
    "pytorch_model.bin.index.json",
    "model.safetensors",
    "model.safetensors.index.json",
)

HF_WEIGHT_GLOBS = (
    "pytorch_model-*.bin",
    "model-*.safetensors",
)

LORA_ADAPTER_CONFIG_FILE = "adapter_config.json"

LORA_ADAPTER_WEIGHT_FILES = (
    "adapter_model.bin",
    "adapter_model.bin.index.json",
    "adapter_model.safetensors",
    "adapter_model.safetensors.index.json",
)

LORA_ADAPTER_WEIGHT_GLOBS = (
    "adapter_model-*.bin",
    "adapter_model-*.safetensors",
)

TRAINING_STATE_DIR_PATTERNS = (
    "global_step*",
)

TRAINING_STATE_FILE_PATTERNS = (
    "latest",
    "optimizer.pt",
    "scheduler.pt",
    "scaler.pt",
    "rng_state*.pth",
    "trainer_state.json",
    "training_args.bin",
    "zero_to_fp32.py",
)

CUSTOM_TRAINING_KEYS = (
    "save_inference_only",
)

DEFAULT_LORA_TARGET_MODULES = (
    "q_proj",
    "v_proj",
)

STAGE1_TRAIN_SCHEMA_VERSION = "stage1-train-config/v1"
STAGE1_EARLY_STOPPING_POLICY = {
    "metric": "eval_loss",
    "mode": "min",
    "minimum_epochs": 1,
    "maximum_epochs": 5,
    "patience_evaluations": 3,
    "threshold": 0.001,
    "tie_break": "earliest-global-step",
    "selection_data": "train-only-calibration",
    "scientific_dev_used_for_selection": False,
}


def is_stage1_config(config: dict) -> bool:
    return config.get("schema_version") == STAGE1_TRAIN_SCHEMA_VERSION


def tracking_enabled(config: dict) -> bool:
    tracking = config.get("tracking", {}).get("swanlab", {})
    return bool(tracking.get("enabled", False))


def load_swanlab_tracking():
    try:
        import swanlab
        from swanlab.integration.transformers import SwanLabCallback
    except ImportError as exc:
        raise ImportError(
            "SwanLab tracking was enabled, but the optional 'swanlab' package is not installed"
        ) from exc
    return swanlab, SwanLabCallback


def configure_reproducibility(config: dict) -> int:
    training = config.get("training", {})
    seed = int(config.get("random_seed", training.get("seed", 42)))
    if is_stage1_config(config):
        for key in ("seed", "data_seed"):
            if int(training.get(key, seed)) != seed:
                raise ValueError(f"Stage 1 training.{key} must equal random_seed={seed}")
    random.seed(seed)
    set_seed(seed, deterministic=True)
    return seed


def get_early_stopping_settings(config: dict) -> dict[str, Any] | None:
    raw = config.get("early_stopping")
    if not raw or not str_to_bool(raw.get("enabled", False)):
        return None
    settings = dict(raw)
    if is_stage1_config(config):
        for key, expected in STAGE1_EARLY_STOPPING_POLICY.items():
            if settings.get(key) != expected:
                raise ValueError(
                    f"Stage 1 early_stopping.{key} must be {expected!r}, got {settings.get(key)!r}"
                )
    return settings


def validate_train_only_calibration_config(
    data_config: dict[str, Any], *, require_immutable_partition: bool = False
) -> dict[str, Any]:
    if data_config.get("selection_split") != "train-only-calibration":
        raise ValueError("Stage 1 checkpoint selection must use train-only-calibration")
    if data_config.get("val_data_path") not in (None, ""):
        raise ValueError("scientific dev data cannot be supplied as val_data_path for Stage 1 selection")

    calibration = dict(data_config.get("calibration") or {})
    if calibration.get("assignment") != "sha256-query-id-v1":
        raise ValueError("Stage 1 calibration assignment must be sha256-query-id-v1")
    modulus = int(calibration.get("hash_modulus", 10000))
    threshold = int(calibration.get("hash_threshold_exclusive", 1000))
    fraction = float(calibration.get("fraction", threshold / modulus))
    if not (0 < threshold < modulus):
        raise ValueError("calibration hash threshold must be strictly between zero and its modulus")
    if not math.isclose(fraction, threshold / modulus, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("calibration fraction must equal hash_threshold_exclusive/hash_modulus")
    if not math.isclose(fraction, 0.1, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("Stage 1 train-only calibration fraction must be exactly 0.1")
    id_fields = calibration.get("id_fields")
    if not isinstance(id_fields, list) or not id_fields or not all(
        isinstance(field, str) and field for field in id_fields
    ):
        raise ValueError("Stage 1 calibration id_fields must be a non-empty string list")
    if not isinstance(calibration.get("salt"), str) or not calibration["salt"]:
        raise ValueError("Stage 1 calibration salt must be a non-empty string")
    if require_immutable_partition and data_config.get("partition") != PARTITION_CONFIG:
        raise ValueError(
            "Stage 1 fit/calibration membership must come from an immutable partition ref"
        )
    if (
        require_immutable_partition
        and data_config.get("fixed_presentation") != FIXED_PRESENTATION_CONFIG
    ):
        raise ValueError(
            "Stage 1 calibration presentation must use the frozen epoch-1 wire"
        )
    return calibration


def validate_stage1_data_contract(config: dict, *, require_runtime_ready: bool = False) -> None:
    if not is_stage1_config(config):
        return
    settings = get_early_stopping_settings(config)
    if settings is None:
        raise ValueError("Stage 1 requires the frozen early-stopping policy")
    data_config = config.get("data", {})
    validate_train_only_calibration_config(
        data_config, require_immutable_partition=True
    )
    if settings["selection_data"] != data_config["selection_split"]:
        raise ValueError("early-stopping selection_data and data.selection_split must match")
    if data_config.get("source_schema") != "stage1-training-schedule/v1":
        raise ValueError("Stage 1 training data must resolve from stage1-training-schedule/v1")
    if data_config.get("train_data_path") not in (None, ""):
        raise ValueError("formal Stage 1 schedule-aware training forbids a static train_data_path")
    if require_runtime_ready:
        runtime = config.get("_stage1_runtime_validation")
        if not isinstance(runtime, dict) or runtime.get("schema_version") != RUNTIME_VALIDATION_MARKER:
            raise RuntimeError(
                "formal Stage 1 training is blocked until the schedule-aware loader validates "
                "matching plan/evidence/schedule/base/environment refs and model key"
            )
        _validated_stage1_deepspeed_object(config, runtime)


def _validated_stage1_deepspeed_object(
    config: dict, runtime_marker: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Return an isolated copy of the consume-time frozen DeepSpeed object."""

    runtime = (
        runtime_marker
        if runtime_marker is not None
        else config.get("_stage1_runtime_validation")
    )
    if not isinstance(runtime, dict) or runtime.get("schema_version") != RUNTIME_VALIDATION_MARKER:
        raise RuntimeError("formal Stage 1 DeepSpeed config lacks a validated runtime marker")
    training = config.get("training")
    deepspeed = training.get("deepspeed") if isinstance(training, dict) else None
    if not isinstance(deepspeed, dict):
        raise RuntimeError(
            "formal Stage 1 DeepSpeed handoff must be the validated resolved object, not a path"
        )
    expected_sha256 = runtime.get("deepspeed_config_sha256")
    try:
        frozen_bytes = canonical_json_bytes(deepspeed)
    except (TypeError, ValueError, TrainingArtifactError) as exc:
        raise RuntimeError("formal Stage 1 DeepSpeed object is not canonical JSON") from exc
    observed_sha256 = hashlib.sha256(frozen_bytes).hexdigest()
    if expected_sha256 != observed_sha256:
        raise RuntimeError(
            "formal Stage 1 DeepSpeed object differs from the immutable runtime binding"
        )
    # Return the object decoded from the exact bytes that were hashed.  This
    # closes the small validation-to-copy window in which a caller-owned dict
    # could otherwise be mutated after hashing but before the backend copy.
    frozen_object = json.loads(frozen_bytes)
    if not isinstance(frozen_object, dict):  # defensive: the input check above is exact
        raise RuntimeError("formal Stage 1 DeepSpeed binding is not a JSON object")
    return frozen_object


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_rank() -> int:
    return int(os.environ.get("RANK", "0"))


def get_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "-1"))


def get_world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def is_main_process() -> bool:
    return get_rank() == 0


def get_train_backend() -> str:
    return os.environ.get("TRAIN_BACKEND", "single").strip().lower()


def build_device_map(config):
    return config.get("device_map", "auto")


def str_to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def csv_or_list(value: Any, default: tuple[str, ...]) -> list[str]:
    if value is None:
        return list(default)
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    raise TypeError(f"Expected string or list for LoRA target_modules, got {type(value).__name__}")


def get_lora_settings(config: dict) -> dict:
    raw = config.get("lora", False)
    if isinstance(raw, dict):
        enabled = str_to_bool(raw.get("enabled", True))
        payload = dict(raw)
    else:
        enabled = str_to_bool(raw)
        payload = {}

    return {
        "enabled": enabled,
        "r": int(payload.get("r", 8)),
        "alpha": int(payload.get("alpha", payload.get("lora_alpha", 32))),
        "dropout": float(payload.get("dropout", payload.get("lora_dropout", 0.1))),
        "target_modules": csv_or_list(payload.get("target_modules"), DEFAULT_LORA_TARGET_MODULES),
        "bias": payload.get("bias", "none"),
        "merge_on_save": str_to_bool(payload.get("merge_on_save", True)),
    }


def is_lora_enabled(config: dict) -> bool:
    return bool(get_lora_settings(config)["enabled"])


def to_str(x):
    if x is None:
        return ""
    try:
        if isinstance(x, float) and pd.isna(x):
            return ""
    except Exception:
        pass

    if isinstance(x, list):
        return "\n".join(map(str, x))
    if isinstance(x, dict):
        return json.dumps(x, ensure_ascii=False)
    return str(x)


def build_messages(example) -> list[dict]:
    inst = to_str(example.get("instruction"))
    inp = to_str(example.get("input"))

    if inst.strip():
        return [
            {"role": "system", "content": inst},
            {"role": "user", "content": inp},
        ]
    return [{"role": "user", "content": inp}]


def latest_checkpoint_dir(model_root: Union[str, Path]) -> Optional[Path]:
    root = Path(model_root)
    if not root.exists():
        return None
    checkpoints = [p for p in root.glob("checkpoint-*") if p.is_dir()]
    if not checkpoints:
        return None

    def checkpoint_sort_key(path: Path):
        prefix = "checkpoint-"
        suffix = path.name[len(prefix) :] if path.name.startswith(prefix) else path.name
        try:
            return (0, int(suffix))
        except ValueError:
            return (1, suffix)

    return sorted(checkpoints, key=checkpoint_sort_key)[-1]


def checkpoint_global_step(checkpoint_dir: Union[str, Path]) -> int:
    name = Path(checkpoint_dir).name
    prefix = "checkpoint-"
    if not name.startswith(prefix) or not name[len(prefix) :].isdigit():
        raise RuntimeError(
            f"selected checkpoint lacks a canonical checkpoint-<global_step> name: {name}"
        )
    step = int(name[len(prefix) :])
    if step <= 0:
        raise RuntimeError("selected checkpoint global step must be positive")
    return step


def checkpoint_has_hf_weights(checkpoint_dir: Union[str, Path]) -> bool:
    path = Path(checkpoint_dir)
    return any((path / filename).exists() for filename in HF_WEIGHT_FILES) or any(
        next(path.glob(pattern), None) is not None for pattern in HF_WEIGHT_GLOBS
    )


def checkpoint_has_lora_adapter(checkpoint_dir: Union[str, Path]) -> bool:
    path = Path(checkpoint_dir)
    has_config = (path / LORA_ADAPTER_CONFIG_FILE).exists()
    has_weights = any((path / filename).exists() for filename in LORA_ADAPTER_WEIGHT_FILES) or any(
        next(path.glob(pattern), None) is not None for pattern in LORA_ADAPTER_WEIGHT_GLOBS
    )
    return has_config and has_weights


def checkpoint_has_inference_weights(checkpoint_dir: Union[str, Path]) -> bool:
    return checkpoint_has_hf_weights(checkpoint_dir) or checkpoint_has_lora_adapter(checkpoint_dir)


def get_save_inference_only(config: dict) -> bool:
    checkpointing = config.get("checkpointing", {})
    training = config.get("training", {})
    return bool(
        training.get(
            "save_inference_only",
            checkpointing.get("save_inference_only", False),
        )
    )


def remove_training_state_from_checkpoint(checkpoint_dir: Union[str, Path]) -> list[str]:
    checkpoint = Path(checkpoint_dir)
    if not checkpoint.exists():
        return []
    if not checkpoint_has_inference_weights(checkpoint):
        logger.warning(
            "Skip inference-only cleanup for {} because no HuggingFace or LoRA adapter weights were found.",
            checkpoint,
        )
        return []

    removed: list[str] = []

    for pattern in TRAINING_STATE_DIR_PATTERNS:
        for path in checkpoint.glob(pattern):
            if not path.exists():
                continue
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
            removed.append(path.name)

    for pattern in TRAINING_STATE_FILE_PATTERNS:
        for path in checkpoint.glob(pattern):
            if not path.exists() or path.is_dir():
                continue
            path.unlink()
            removed.append(path.name)

    return sorted(set(removed))


def barrier_if_needed() -> None:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()


def cleanup_distributed() -> None:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


class Stage1EarlyStoppingCallback(EarlyStoppingCallback):
    """Fail closed on invalid loss and honor the frozen minimum-epoch rule."""

    def __init__(self, *, patience: int, threshold: float, minimum_epochs: int):
        super().__init__(
            early_stopping_patience=patience,
            early_stopping_threshold=threshold,
        )
        self.minimum_epochs = minimum_epochs

    def on_evaluate(self, args, state, control, metrics, **kwargs):  # type: ignore
        metric_name = args.metric_for_best_model
        if not metric_name.startswith("eval_"):
            metric_name = f"eval_{metric_name}"
        metric_value = metrics.get(metric_name)
        if metric_value is None or not math.isfinite(float(metric_value)):
            raise FloatingPointError(
                f"Stage 1 checkpoint selection requires finite {metric_name}; got {metric_value!r}"
            )

        was_stopping = control.should_training_stop
        super().on_evaluate(args, state, control, metrics, **kwargs)
        current_epoch = float(state.epoch or 0.0)
        if not was_stopping and current_epoch < self.minimum_epochs:
            control.should_training_stop = False
        return control


class CustomTrainer(Trainer):
    def __init__(
        self,
        *args,
        eval_tokenizer,
        eval_config: dict,
        llm_metrics: LLMmetrics,
        eval_raw_dataset=None,
        max_retries: int = 0,
        eval_num: int = 100,
        prompt_template: str = "",
        early_stopping_config: dict | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.eval_config = eval_config
        self.eval_raw_dataset = eval_raw_dataset
        self.eval_tokenizer = eval_tokenizer
        self.llm_metrics = llm_metrics
        self.max_retries = max_retries
        self.eval_num = eval_num
        self.prompt_template = prompt_template
        self.early_stopping_config = early_stopping_config

    def _determine_best_metric(self, metrics, trial):  # type: ignore
        if not self.early_stopping_config:
            return super()._determine_best_metric(metrics, trial)

        metric_name = self.args.metric_for_best_model
        if not metric_name.startswith("eval_"):
            metric_name = f"eval_{metric_name}"
        if metric_name not in metrics:
            raise KeyError(
                f"metric_for_best_model={metric_name!r} is absent; available metrics: {sorted(metrics)}"
            )
        value = float(metrics[metric_name])
        if not math.isfinite(value):
            raise FloatingPointError(f"checkpoint selection metric {metric_name} is not finite: {value!r}")

        best = self.state.best_metric
        threshold = float(self.early_stopping_config["threshold"])
        mode = self.early_stopping_config["mode"]
        improved = best is None
        if best is not None:
            best = float(best)
            improvement = best - value if mode == "min" else value - best
            improved = improvement > threshold

        if improved:
            self.state.best_metric = value
            self.state.best_global_step = self.state.global_step
            return True
        return False

    def evaluate(self, **kwargs):  # type: ignore
        metrics = super().evaluate(**kwargs)
        self.log(metrics)
        return metrics


class InferenceOnlyCheckpointCallback(TrainerCallback):
    def on_save(self, args, state, control, **kwargs):  # type: ignore
        barrier_if_needed()
        if is_main_process():
            checkpoint_dir = Path(args.output_dir) / f"checkpoint-{state.global_step}"
            removed = remove_training_state_from_checkpoint(checkpoint_dir)
            if removed:
                logger.info(
                    "Inference-only checkpoint cleanup removed {} entries from {}",
                    len(removed),
                    checkpoint_dir,
                )
        barrier_if_needed()
        return control


def predict(messages, model, tokenizer, config):
    device = "cuda"
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(device)
    attention_mask = model_inputs["attention_mask"]

    generated_ids = model.generate(
        model_inputs.input_ids,
        attention_mask=attention_mask,
        max_new_tokens=config.get("max_length", 512),
        repetition_penalty=config.get("repetition_penalty", 1.15),
        temperature=config.get("temperature"),
        top_p=config.get("top_p"),
        top_k=config.get("top_k"),
        min_p=config.get("min_p"),
        pad_token_id=tokenizer.eos_token_id,
    )
    generated_ids = [
        output_ids[len(input_ids) :] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]


def validate_early_stopping_training_args(
    config: dict,
    training_config: dict,
    save_inference_only: bool,
) -> dict[str, Any] | None:
    settings = get_early_stopping_settings(config)
    if settings is None:
        return None

    maximum_epochs = int(settings["maximum_epochs"])
    epochs = float(training_config.get("num_train_epochs", maximum_epochs))
    if epochs > maximum_epochs or epochs < int(settings["minimum_epochs"]):
        raise ValueError(
            f"num_train_epochs must be within [{settings['minimum_epochs']}, {maximum_epochs}], got {epochs}"
        )
    if str(training_config.get("eval_strategy", "no")) == "no":
        raise ValueError("early stopping requires a non-'no' eval_strategy")
    if training_config.get("save_strategy") != training_config.get("eval_strategy"):
        raise ValueError("load-best early stopping requires matching save_strategy and eval_strategy")
    if training_config.get("load_best_model_at_end") is not True:
        raise ValueError("early stopping requires load_best_model_at_end=true")
    if training_config.get("metric_for_best_model") != settings["metric"]:
        raise ValueError("metric_for_best_model must match the frozen early-stopping metric")
    expected_greater_is_better = settings["mode"] == "max"
    if training_config.get("greater_is_better") is not expected_greater_is_better:
        raise ValueError("greater_is_better conflicts with the frozen early-stopping mode")
    if save_inference_only:
        raise ValueError(
            "save_inference_only is incompatible with DeepSpeed load-best checkpoint selection"
        )
    return settings


def build_training_args(config: dict) -> TrainingArguments:
    training_config = copy.deepcopy(dict(config["training"]))
    save_inference_only = get_save_inference_only(config)
    if is_stage1_config(config):
        validate_stage1_data_contract(config, require_runtime_ready=True)
        training_config["deepspeed"] = _validated_stage1_deepspeed_object(config)
        if training_config.get("fsdp") not in (None, "", []):
            raise ValueError("Stage 1 is frozen to DeepSpeed ZeRO-3; FSDP is not allowed")
        if is_lora_enabled(config):
            raise ValueError("Stage 1 is frozen to full fine-tuning; LoRA is not allowed")
    validate_early_stopping_training_args(config, training_config, save_inference_only)
    for key in CUSTOM_TRAINING_KEYS:
        training_config.pop(key, None)

    training_arg_fields = getattr(TrainingArguments, "__dataclass_fields__", {})
    if "save_only_model" in training_config and "save_only_model" not in training_arg_fields:
        logger.warning("Ignoring unsupported TrainingArguments field: save_only_model")
        training_config.pop("save_only_model", None)

    if save_inference_only:
        if "save_safetensors" in training_arg_fields:
            training_config["save_safetensors"] = True
        else:
            logger.warning("TrainingArguments.save_safetensors is unavailable; using the default checkpoint format.")
        if "save_only_model" in training_arg_fields:
            training_config["save_only_model"] = True
        else:
            logger.warning(
                "TrainingArguments.save_only_model is unavailable; "
                "DeepSpeed/Trainer state will be removed after each checkpoint save."
            )

    return TrainingArguments(
        **training_config,
        group_by_length=True,
    )


def enable_input_require_grads(model) -> None:
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
        return

    input_embeddings = model.get_input_embeddings()

    def make_inputs_require_grad(_module, _input, output):
        output.requires_grad_(True)

    input_embeddings.register_forward_hook(make_inputs_require_grad)


def apply_lora_if_enabled(model, config: dict, training_args: TrainingArguments):
    lora = get_lora_settings(config)
    if not lora["enabled"]:
        return model

    if lora["r"] <= 0:
        raise ValueError(f"LoRA rank must be positive, got {lora['r']}")
    if not lora["target_modules"]:
        raise ValueError("LoRA target_modules cannot be empty")

    try:
        from peft import LoraConfig, get_peft_model
    except ImportError as exc:
        raise ImportError(
            "LoRA training requires the 'peft' package. "
            "Install project dependencies again, or run: pip install 'peft>=0.14.0'"
        ) from exc

    if training_args.gradient_checkpointing:
        enable_input_require_grads(model)

    peft_config = LoraConfig(
        r=lora["r"],
        lora_alpha=lora["alpha"],
        lora_dropout=lora["dropout"],
        target_modules=lora["target_modules"],
        bias=lora["bias"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)

    if is_main_process():
        print(
            "[INFO] LoRA enabled:",
            {
                "r": lora["r"],
                "alpha": lora["alpha"],
                "dropout": lora["dropout"],
                "target_modules": lora["target_modules"],
                "bias": lora["bias"],
                "merge_on_save": lora["merge_on_save"],
            },
        )
        model.print_trainable_parameters()

    return model


def load_model(
    config: dict,
    training_args: TrainingArguments,
    *,
    runtime_bundle: Stage1RuntimeBundle | None = None,
):
    train_backend = get_train_backend()
    distributed = get_world_size() > 1
    deepspeed_enabled = training_args.deepspeed is not None
    dtype = torch.bfloat16 if config["training"].get("bf16", False) else torch.float32

    model_kwargs = {
        "torch_dtype": dtype,
        "attn_implementation": "flash_attention_2",
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
    }

    if deepspeed_enabled:
        if is_main_process():
            print("[INFO] DeepSpeed owns model placement; loading without device_map")
    elif train_backend == "single" or not distributed:
        model_kwargs["device_map"] = build_device_map(config)
    elif is_main_process():
        print(
            f"[INFO] distributed backend={train_backend}; loading without device_map "
            "so the distributed Trainer backend owns placement"
        )

    if runtime_bundle is None:
        model = AutoModelForCausalLM.from_pretrained(
            config["model_path"], **model_kwargs
        )
    else:
        with verified_stage1_runtime_source_load(
            runtime_bundle, source_names=("checkpoint",)
        ) as sources:
            model = AutoModelForCausalLM.from_pretrained(
                str(sources.checkpoint_path), local_files_only=True, **model_kwargs
            )
    if runtime_bundle is not None:
        # Detect source-tree/environment drift during the potentially long
        # sharded load before any gradient update is allowed.
        revalidate_stage1_runtime_sources(runtime_bundle)
    model.config.use_cache = False
    model = apply_lora_if_enabled(model, config, training_args)

    if is_main_process():
        print(
            "[INFO] TrainingArguments:",
            {
                "output_dir": training_args.output_dir,
                "deepspeed": training_args.deepspeed,
                "fsdp": str(training_args.fsdp),
                "per_device_train_batch_size": training_args.per_device_train_batch_size,
                "gradient_accumulation_steps": training_args.gradient_accumulation_steps,
                "gradient_checkpointing": training_args.gradient_checkpointing,
                "save_safetensors": getattr(training_args, "save_safetensors", None),
                "save_only_model": getattr(training_args, "save_only_model", None),
                "lora_enabled": is_lora_enabled(config),
            },
        )

    return model


def load_tokenizer(
    config: dict,
    *,
    runtime_bundle: Stage1RuntimeBundle | None = None,
):
    """Load the tokenizer, leasing registered sources for formal Stage 1."""

    kwargs = {"use_fast": False, "trust_remote_code": True}
    if runtime_bundle is None:
        tokenizer = AutoTokenizer.from_pretrained(
            config.get("tokenizer_path", config["model_path"]), **kwargs
        )
    else:
        with verified_stage1_runtime_source_load(
            runtime_bundle, source_names=("tokenizer",)
        ) as sources:
            tokenizer = AutoTokenizer.from_pretrained(
                str(sources.tokenizer_path),
                use_fast=False,
                local_files_only=True,
                trust_remote_code=False,
            )
    tokenizer.pad_token = (
        tokenizer.eos_token if tokenizer.pad_token is None else tokenizer.pad_token
    )
    return tokenizer


def file_sha256(path: Union[str, Path]) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_train_data_path(config: dict) -> Path:
    raw_path = config.get("data", {}).get("train_data_path")
    if not raw_path:
        raise ValueError(
            "data.train_data_path is unresolved; materialize the frozen Stage 1 training schedule first"
        )
    path = Path(raw_path)
    if not path.is_file():
        raise FileNotFoundError(f"training data does not exist: {path}")
    return path


def query_id_for_calibration(row: dict[str, Any], id_fields: list[str]) -> str:
    for field in id_fields:
        value = row.get(field)
        if isinstance(value, bool):
            continue
        if isinstance(value, (str, int)):
            normalized = str(value).strip()
            if normalized:
                return normalized
    raise ValueError(
        "train-only calibration requires a non-empty string/integer query ID in one of: "
        + ", ".join(id_fields)
    )


def calibration_bucket(query_id: str, calibration: dict[str, Any]) -> int:
    if calibration.get("assignment") != "sha256-query-id-v1":
        raise ValueError("unsupported train-only calibration assignment policy")
    modulus = int(calibration.get("hash_modulus", 10000))
    if modulus <= 1:
        raise ValueError("calibration hash_modulus must be greater than one")
    identity = {
        "assignment": "sha256-query-id-v1",
        "query_id": query_id,
        "salt": str(calibration.get("salt", "")),
    }
    wire = json.dumps(
        identity,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return int(hashlib.sha256(wire).hexdigest(), 16) % modulus


def split_train_only_calibration(
    train_df: pd.DataFrame,
    data_config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    calibration = validate_train_only_calibration_config(data_config)
    modulus = int(calibration.get("hash_modulus", 10000))
    threshold = int(calibration.get("hash_threshold_exclusive", 1000))

    id_fields = [str(field) for field in calibration["id_fields"]]
    calibration_mask = []
    for row in train_df.to_dict(orient="records"):
        query_id = query_id_for_calibration(row, id_fields)
        calibration_mask.append(calibration_bucket(query_id, calibration) < threshold)

    calibration_df = train_df.loc[calibration_mask].reset_index(drop=True)
    fit_df = train_df.loc[[not selected for selected in calibration_mask]].reset_index(drop=True)
    if fit_df.empty or calibration_df.empty:
        raise ValueError("hash calibration produced an empty fit or calibration partition")
    return fit_df, calibration_df


def tokenized_cache_paths(config: dict, training_args: TrainingArguments) -> tuple[Path, Path]:
    data_config = config["data"]
    train_data_path = require_train_data_path(config)
    cache_payload = {
        "train_data_path": str(train_data_path),
        "train_data_sha256": file_sha256(train_data_path),
        "val_data_path": data_config.get("val_data_path"),
        "selection_split": data_config.get("selection_split"),
        "calibration": data_config.get("calibration"),
        "model_path": config["model_path"],
        "max_length": config.get("max_length", 512),
        "prompt_template": config.get("prompt_template", ""),
        "system_prompt": config.get("system_prompt", ""),
        "train_code_sha256": file_sha256(__file__),
    }
    digest = hashlib.sha256(
        json.dumps(cache_payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    cache_dir = Path(config.get("tokenized_cache_dir") or Path(training_args.output_dir).parent / "cache" / "tokenized")
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"train_{digest}.arrow", cache_dir / f"eval_{digest}.arrow"


def example_record_id(example: dict[str, Any], fallback_index: int | None = None) -> str:
    for field in ("id", "query_id", "record_id"):
        value = example.get(field)
        if isinstance(value, (str, int)) and not isinstance(value, bool) and str(value).strip():
            return str(value).strip()
    return f"row-index:{fallback_index}" if fallback_index is not None else "unknown-record"


def gold_response_text(value: Any) -> str:
    if isinstance(value, (list, dict)):
        return json.dumps(
            value,
            ensure_ascii=False,
            separators=(",", ":"),
        )
    return to_str(value)


def encode_training_example(
    example: dict[str, Any],
    tokenizer,
    *,
    max_length: int,
    fallback_index: int | None = None,
) -> dict[str, list[int]]:
    record_id = example_record_id(example, fallback_index)
    messages = build_messages(example)
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    response_text = gold_response_text(example.get("output"))
    if not response_text.strip():
        raise ValueError(f"training record {record_id} has an empty gold response")
    if tokenizer.eos_token_id is None:
        raise ValueError("tokenizer must define eos_token_id for supervised training")

    instruction = tokenizer(text, add_special_tokens=False)
    response = tokenizer(response_text, add_special_tokens=False)
    prompt_ids = list(instruction["input_ids"])
    response_ids = list(response["input_ids"])
    input_ids = prompt_ids + response_ids + [tokenizer.eos_token_id]
    attention_mask = [1] * len(input_ids)
    labels = [-100] * len(prompt_ids) + response_ids + [tokenizer.eos_token_id]

    if len(input_ids) > max_length:
        raise ValueError(
            "training record "
            f"{record_id} requires {len(input_ids)} tokens "
            f"(prompt={len(prompt_ids)}, gold+eos={len(response_ids) + 1}) but max_length={max_length}; "
            "gold truncation is forbidden"
        )
    if not any(label != -100 for label in labels):
        raise ValueError(f"training record {record_id} has no supervised gold tokens")
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def stage1_calibration_query_ids(
    runtime_bundle: Stage1RuntimeBundle, data_config: dict[str, Any]
) -> list[str]:
    calibration = validate_train_only_calibration_config(
        data_config, require_immutable_partition=True
    )
    threshold = int(calibration["hash_threshold_exclusive"])
    rows = runtime_bundle.records_by_epoch[1]
    partition_by_id = {
        row["query_id"]: row for row in runtime_bundle.partition_records
    }
    if [row["query_id"] for row in rows] != list(partition_by_id):
        raise ValueError(
            "frozen schedule registry differs from the immutable train partition"
        )
    for row in runtime_bundle.partition_records:
        expected_bucket = calibration_bucket(
            row["cluster_representative_query_id"], calibration
        )
        expected_label = (
            "calibration" if expected_bucket < threshold else "fit"
        )
        if row["bucket"] != expected_bucket or row["partition"] != expected_label:
            raise ValueError(
                "frozen train partition disagrees with the nominal hash-policy assertion"
            )
    frozen = [
        row["query_id"]
        for row in rows
        if row.get("partition") == "calibration"
    ]
    partition_calibration = [
        row["query_id"]
        for row in runtime_bundle.partition_records
        if row["partition"] == "calibration"
    ]
    if frozen != partition_calibration:
        raise ValueError(
            "frozen train partition disagrees with the nominal hash-policy assertion"
        )
    if not frozen or len(frozen) == len(rows):
        raise ValueError("hash calibration produced an empty fit or calibration partition")
    return frozen


def build_datasets(
    config: dict,
    tokenizer,
    training_args: TrainingArguments,
    *,
    runtime_bundle: Stage1RuntimeBundle | None = None,
):
    max_length = config.get("max_length", 512)
    data_config = config["data"]
    if is_stage1_config(config):
        if runtime_bundle is None:
            raise RuntimeError("Stage 1 schedule-aware loader lacks validated runtime inputs")

        def encode_schedule_row(row: dict[str, Any], index: int):
            return encode_training_example(
                row,
                tokenizer,
                max_length=max_length,
                fallback_index=index,
            )

        # This replay is only an assertion over the immutable partition.  The
        # dataset builder consumes the frozen schedule labels directly.
        stage1_calibration_query_ids(runtime_bundle, data_config)
        train_dataset, eval_dataset, eval_raw, tracker = build_schedule_aware_datasets(
            runtime_bundle, encoder=encode_schedule_row
        )
        if is_main_process():
            print(
                "schedule-aware fit/calibration rows:",
                len(train_dataset),
                len(eval_dataset),
            )
            print(
                "schedule/model key:",
                runtime_bundle.schedule_meta["schedule_build_id"],
                runtime_bundle.slot["model_key"],
            )
        return train_dataset, eval_dataset, eval_raw, tracker

    preprocessing_num_proc = int(config.get("preprocessing_num_proc", 8))
    train_cache_file, eval_cache_file = tokenized_cache_paths(config, training_args)

    def process_func(batch, indices):
        input_ids_list, attention_list, labels_list = [], [], []

        for position, source_index in enumerate(indices):
            example = {column: values[position] for column, values in batch.items()}
            encoded = encode_training_example(
                example,
                tokenizer,
                max_length=max_length,
                fallback_index=int(source_index),
            )
            input_ids_list.append(encoded["input_ids"])
            attention_list.append(encoded["attention_mask"])
            labels_list.append(encoded["labels"])

        return {
            "input_ids": input_ids_list,
            "attention_mask": attention_list,
            "labels": labels_list,
        }

    source_train_df = pd.read_json(require_train_data_path(config), lines=True)
    if data_config.get("selection_split") == "train-only-calibration":
        train_df, eval_df = split_train_only_calibration(source_train_df, data_config)
    else:
        val_data_path = data_config.get("val_data_path")
        if not val_data_path:
            raise ValueError("data.val_data_path is required outside Stage 1 train-only calibration")
        train_df = source_train_df
        eval_df = pd.read_json(val_data_path, lines=True)
    eval_raw = eval_df.to_dict(orient="records")

    train_ds = Dataset.from_pandas(train_df)
    eval_ds = Dataset.from_pandas(eval_df)

    with training_args.main_process_first(desc="tokenize datasets"):
        train_dataset = train_ds.map(
            process_func,
            remove_columns=train_ds.column_names,
            batched=True,
            with_indices=True,
            num_proc=preprocessing_num_proc,
            cache_file_name=str(train_cache_file),
            load_from_cache_file=True,
            new_fingerprint=f"train-{train_cache_file.stem}",
        )
        eval_dataset = eval_ds.map(
            process_func,
            remove_columns=eval_ds.column_names,
            batched=True,
            with_indices=True,
            num_proc=preprocessing_num_proc,
            cache_file_name=str(eval_cache_file),
            load_from_cache_file=True,
            new_fingerprint=f"eval-{eval_cache_file.stem}",
        )

    if is_main_process():
        print("features:", train_dataset.features)
        ex0 = train_dataset[0]["input_ids"]
        print("type(input_ids[0]):", type(ex0), "sample:", ex0 if isinstance(ex0, int) else ex0[:10])

        bad = []
        for i in range(min(2000, len(train_dataset))):
            x = train_dataset[i]["input_ids"]
            if isinstance(x, int):
                bad.append(i)
                if len(bad) <= 5:
                    print("bad idx:", i, "value:", x)
        print("bad count:", len(bad))
        print("tokenized train cache:", train_cache_file)
        print("tokenized eval cache:", eval_cache_file)
        print("fit/calibration rows:", len(train_dataset), len(eval_dataset))

    return train_dataset, eval_dataset, eval_raw, None


def ensure_loadable_checkpoint(
    trainer: Trainer,
    tokenizer,
    save_inference_only: bool = False,
    lora_enabled: bool = False,
) -> Path:
    output_dir = Path(trainer.args.output_dir)
    best_checkpoint = getattr(trainer.state, "best_model_checkpoint", None)
    if trainer.args.load_best_model_at_end:
        if not best_checkpoint:
            raise RuntimeError("load_best_model_at_end completed without a best_model_checkpoint")
        selected = Path(best_checkpoint)
        if not selected.is_dir():
            raise RuntimeError(f"best model checkpoint does not exist: {selected}")
    else:
        selected = latest_checkpoint_dir(output_dir)

    if selected is None:
        selected = output_dir / "checkpoint-final"
        trainer.save_model(str(selected))
        barrier_if_needed()

    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(str(selected))

    barrier_if_needed()

    if save_inference_only:
        if trainer.is_world_process_zero():
            removed = remove_training_state_from_checkpoint(selected)
            if removed:
                logger.info(
                    "Inference-only checkpoint cleanup removed {} entries from {}",
                    len(removed),
                    selected,
                )
        barrier_if_needed()

    if trainer.is_world_process_zero():
        if lora_enabled:
            if not checkpoint_has_lora_adapter(selected):
                raise RuntimeError(
                    f"No LoRA adapter weights found in {selected}. "
                    "Expected adapter_config.json plus adapter_model weights."
                )
        elif not checkpoint_has_hf_weights(selected):
            raise RuntimeError(
                f"No HuggingFace model weights found in {selected}. "
                "For DeepSpeed ZeRO-3, ensure stage3_gather_16bit_weights_on_model_save=true."
            )
    return selected


def run(
    config: dict,
    *,
    stage1_runtime_refs: dict[str, str] | None = None,
    write_receipt: str | Path | None = None,
):
    runtime_bundle: Stage1RuntimeBundle | None = None
    if is_stage1_config(config):
        validate_stage1_data_contract(config)
        required = {
            "training_plan_ref",
            "training_evidence_ref",
            "train_partition_ref",
            "schedule_ref",
            "base_model_ref",
            "environment_ref",
            "model_key",
        }
        if not isinstance(stage1_runtime_refs, dict) or set(stage1_runtime_refs) != required:
            raise RuntimeError(
                "formal Stage 1 training is blocked: the schedule-aware loader requires "
                "plan/evidence/schedule/base/environment refs and model key"
            )
        if write_receipt is None:
            raise RuntimeError("formal Stage 1 training requires --write-receipt")
        receipt_path = Path(write_receipt)
        if receipt_path.name != "training_receipt.json":
            raise ValueError("--write-receipt must end in training_receipt.json")
        if receipt_path.parent.resolve() == Path.cwd().resolve():
            raise ValueError("training output/receipt must use a dedicated subdirectory")
        runtime_bundle = resolve_stage1_runtime_inputs(
            config,
            **stage1_runtime_refs,
        )
        config = copy.deepcopy(runtime_bundle.resolved_config)
        # Output location is a runtime locator, never part of the portable plan.
        config["training"]["output_dir"] = str(receipt_path.parent)
        validate_stage1_data_contract(config, require_runtime_ready=True)
    elif stage1_runtime_refs is not None or write_receipt is not None:
        raise ValueError("Stage 1 immutable refs/receipt cannot be supplied to a legacy config")

    seed = configure_reproducibility(config)
    local_rank = get_local_rank()
    if local_rank >= 0 and torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    if is_main_process():
        print("CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES"))
        print("TRAIN_BACKEND =", get_train_backend())
        print("TRAIN_PROFILE =", os.environ.get("TRAIN_PROFILE", ""))
        print("RANDOM_SEED =", seed)
        print("WORLD_SIZE =", get_world_size())
        print("torch.cuda.device_count() =", torch.cuda.device_count())
        for i in range(torch.cuda.device_count()):
            print(i, torch.cuda.get_device_name(i))

    tokenizer = load_tokenizer(config, runtime_bundle=runtime_bundle)

    if runtime_bundle is not None:
        # Re-render and re-tokenize the complete plan registry under the actual
        # tokenizer before a model is loaded or any gradient update is possible.
        validate_training_schedule(
            schedule_ref=stage1_runtime_refs["schedule_ref"], tokenizer=tokenizer
        )

    save_inference_only = get_save_inference_only(config)
    lora_settings = get_lora_settings(config)
    training_args = build_training_args(config)
    early_stopping_settings = get_early_stopping_settings(config)
    use_tracking = tracking_enabled(config)
    swanlab_module = None
    swanlab_callback_class = None

    if use_tracking and is_main_process():
        swanlab_module, swanlab_callback_class = load_swanlab_tracking()

    if use_tracking and is_main_process():
        os.environ["SWANLAB_PROJECT"] = config.get("project_name", "qwen3-8b-sft-hsd")
        swanlab_module.config.update(
            {
                "model": config["model_name"],
                "system_prompt": get_prompt(config["system_prompt"]),
                "prompt": get_prompt(config["prompt_template"]),
                "data_max_length": config.get("max_length", 512),
                "use_bf16": config["training"].get("bf16", False),
                "train_backend": get_train_backend(),
                "train_profile": os.environ.get("TRAIN_PROFILE", ""),
                "save_inference_only": save_inference_only,
                "lora_enabled": lora_settings["enabled"],
                "lora_r": lora_settings["r"] if lora_settings["enabled"] else None,
                "lora_alpha": lora_settings["alpha"] if lora_settings["enabled"] else None,
                "lora_target_modules": ",".join(lora_settings["target_modules"]) if lora_settings["enabled"] else None,
            }
        )

    train_dataset, eval_dataset, eval_raw, schedule_tracker = build_datasets(
        config,
        tokenizer,
        training_args,
        runtime_bundle=runtime_bundle,
    )
    if runtime_bundle is not None:
        # Re-hash after tokenizer/schedule preparation and immediately before
        # opening checkpoint shards.
        revalidate_stage1_runtime_sources(runtime_bundle)
    model = load_model(config, training_args, runtime_bundle=runtime_bundle)
    llm_metrics = LLMmetrics()

    callbacks = []
    if schedule_tracker is not None:
        callbacks.append(schedule_tracker)
    if save_inference_only:
        callbacks.append(InferenceOnlyCheckpointCallback())
    if early_stopping_settings:
        callbacks.append(
            Stage1EarlyStoppingCallback(
                patience=int(early_stopping_settings["patience_evaluations"]),
                threshold=float(early_stopping_settings["threshold"]),
                minimum_epochs=int(early_stopping_settings["minimum_epochs"]),
            )
        )
    if use_tracking and is_main_process():
        callbacks.append(
            swanlab_callback_class(
                project=os.environ["SWANLAB_PROJECT"],
                experiment_name=config["exp_name"],
            )
        )

    trainer = CustomTrainer(
        model=model,
        eval_tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            padding=True,
            pad_to_multiple_of=8,
        ),
        eval_raw_dataset=eval_raw,
        llm_metrics=llm_metrics,
        max_retries=config["eval"].get("max_retries", 0),
        eval_num=config["eval"].get("eval_num", 100),
        eval_config=config["eval"],
        prompt_template=get_prompt(config["prompt_template"]),
        early_stopping_config=early_stopping_settings,
        callbacks=callbacks,
    )

    trainer.train()
    selected_checkpoint = ensure_loadable_checkpoint(
        trainer,
        tokenizer,
        save_inference_only=save_inference_only,
        lora_enabled=lora_settings["enabled"],
    )

    if runtime_bundle is not None:
        if schedule_tracker is None:  # pragma: no cover - guarded by build_datasets
            raise RuntimeError("Stage 1 training completed without a schedule epoch tracker")
        barrier_if_needed()
        if trainer.is_world_process_zero():
            receipt = build_training_receipt(
                runtime_bundle,
                schedule_tracker,
                training_exit_global_step=int(trainer.state.global_step),
                selected_checkpoint_global_step=checkpoint_global_step(
                    selected_checkpoint
                ),
                selected_checkpoint_dir=selected_checkpoint,
                train_code_sha256=runtime_bundle.plan["train_code_sha256"],
                trainer_best_metric=float(trainer.state.best_metric),
                trainer_best_global_step=int(trainer.state.best_global_step),
            )
            write_immutable_training_receipt(write_receipt, receipt)
        barrier_if_needed()

    if use_tracking and is_main_process():
        swanlab_module.finish()


def get_prompt(prompt_name_or_prompt: str):
    try:
        return eval(prompt_name_or_prompt)
    except Exception:
        return prompt_name_or_prompt


def apply_cli_overrides(config: dict, args: argparse.Namespace) -> None:
    training = config.setdefault("training", {})

    if args.deepspeed:
        training["deepspeed"] = args.deepspeed
    if args.per_device_train_batch_size is not None:
        training["per_device_train_batch_size"] = args.per_device_train_batch_size
        if args.per_device_eval_batch_size is None:
            training["per_device_eval_batch_size"] = args.per_device_train_batch_size
    if args.per_device_eval_batch_size is not None:
        training["per_device_eval_batch_size"] = args.per_device_eval_batch_size
    if args.gradient_accumulation_steps is not None:
        training["gradient_accumulation_steps"] = args.gradient_accumulation_steps


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM Fine-tuning Script")
    parser.add_argument("--config", type=str, default="config.json", help="Path to config file")
    parser.add_argument("--deepspeed", type=str, default=None, help="Path to DeepSpeed config override")
    parser.add_argument("--per_device_train_batch_size", type=int, default=None)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=None)
    parser.add_argument("--local_rank", "--local-rank", dest="local_rank", type=int, default=None)
    parser.add_argument("--training-plan-ref", type=str)
    parser.add_argument("--training-evidence-ref", type=str)
    parser.add_argument("--train-partition-ref", type=str)
    parser.add_argument("--schedule-ref", type=str)
    parser.add_argument("--base-model-ref", type=str)
    parser.add_argument("--environment-ref", type=str)
    parser.add_argument("--model-key", type=str)
    parser.add_argument("--write-receipt", type=str)
    args = parser.parse_args()

    if args.local_rank is not None and "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(args.local_rank)

    config = load_config(args.config)
    runtime_refs = None
    write_receipt = None
    if is_stage1_config(config):
        forbidden_overrides = {
            "deepspeed": args.deepspeed,
            "per_device_train_batch_size": args.per_device_train_batch_size,
            "per_device_eval_batch_size": args.per_device_eval_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
        }
        changed = [key for key, value in forbidden_overrides.items() if value is not None]
        if changed:
            parser.error(
                "formal Stage 1 hyperparameters come from the immutable training plan; "
                f"CLI overrides are forbidden: {', '.join(changed)}"
            )
        runtime_refs = {
            "training_plan_ref": args.training_plan_ref,
            "training_evidence_ref": args.training_evidence_ref,
            "train_partition_ref": args.train_partition_ref,
            "schedule_ref": args.schedule_ref,
            "base_model_ref": args.base_model_ref,
            "environment_ref": args.environment_ref,
            "model_key": args.model_key,
        }
        missing = [key for key, value in runtime_refs.items() if not value]
        if missing or not args.write_receipt:
            parser.error(
                "formal Stage 1 requires --training-plan-ref --training-evidence-ref "
                "--train-partition-ref --schedule-ref --base-model-ref --environment-ref --model-key "
                "--write-receipt"
            )
        write_receipt = args.write_receipt
    else:
        supplied_stage1 = any(
            value
            for value in (
                args.training_plan_ref,
                args.training_evidence_ref,
                args.train_partition_ref,
                args.schedule_ref,
                args.base_model_ref,
                args.environment_ref,
                args.model_key,
                args.write_receipt,
            )
        )
        if supplied_stage1:
            parser.error("immutable Stage 1 refs cannot be mixed with a legacy config")
        apply_cli_overrides(config, args)
        config.setdefault("transfer_data", True)
        config.setdefault("exp_name", "default-exp")

    try:
        run(
            config,
            stage1_runtime_refs=runtime_refs,
            write_receipt=write_receipt,
        )
    finally:
        cleanup_distributed()
