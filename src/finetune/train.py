from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
from pathlib import Path
from typing import Any, Optional, Union

import pandas as pd
import swanlab
import torch
from datasets import Dataset
from modelscope import AutoTokenizer
from swanlab.integration.transformers import SwanLabCallback
from transformers import AutoModelForCausalLM, DataCollatorForSeq2Seq, Trainer, TrainerCallback, TrainingArguments  # type: ignore

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


def build_training_args(config: dict) -> TrainingArguments:
    training_config = dict(config["training"])
    save_inference_only = get_save_inference_only(config)
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


def load_model(config: dict, training_args: TrainingArguments):
    train_backend = get_train_backend()
    distributed = get_world_size() > 1
    dtype = torch.bfloat16 if config["training"].get("bf16", False) else torch.float32

    model_kwargs = {
        "torch_dtype": dtype,
        "attn_implementation": "flash_attention_2",
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
    }

    if train_backend == "single" or not distributed:
        model_kwargs["device_map"] = build_device_map(config)
    elif is_main_process():
        print(
            f"[INFO] distributed backend={train_backend}; loading without device_map "
            "so Trainer/DeepSpeed/FSDP owns placement"
        )

    model = AutoModelForCausalLM.from_pretrained(config["model_path"], **model_kwargs)
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


def tokenized_cache_paths(config: dict, training_args: TrainingArguments) -> tuple[Path, Path]:
    data_config = config["data"]
    cache_payload = {
        "train_data_path": data_config["train_data_path"],
        "val_data_path": data_config["val_data_path"],
        "model_path": config["model_path"],
        "max_length": config.get("max_length", 512),
        "prompt_template": config.get("prompt_template", ""),
        "system_prompt": config.get("system_prompt", ""),
    }
    digest = hashlib.sha1(json.dumps(cache_payload, sort_keys=True).encode("utf-8")).hexdigest()[:12]
    cache_dir = Path(config.get("tokenized_cache_dir") or Path(training_args.output_dir).parent / "cache" / "tokenized")
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"train_{digest}.arrow", cache_dir / f"eval_{digest}.arrow"


def build_datasets(config: dict, tokenizer, training_args: TrainingArguments):
    max_length = config.get("max_length", 512)
    data_config = config["data"]
    preprocessing_num_proc = int(config.get("preprocessing_num_proc", 8))
    train_cache_file, eval_cache_file = tokenized_cache_paths(config, training_args)

    def process_func(batch):
        input_ids_list, attention_list, labels_list = [], [], []

        for inst, inp, out in zip(batch["instruction"], batch["input"], batch["output"]):
            messages = build_messages({"instruction": inst, "input": inp})
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            instruction = tokenizer(text, add_special_tokens=False)
            response = tokenizer(to_str(out), add_special_tokens=False)

            input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.eos_token_id]
            attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]
            labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.eos_token_id]

            if len(input_ids) > max_length:
                input_ids = input_ids[:max_length]
                attention_mask = attention_mask[:max_length]
                labels = labels[:max_length]

            input_ids_list.append(input_ids)
            attention_list.append(attention_mask)
            labels_list.append(labels)

        return {
            "input_ids": input_ids_list,
            "attention_mask": attention_list,
            "labels": labels_list,
        }

    train_df = pd.read_json(data_config["train_data_path"], lines=True)
    eval_df = pd.read_json(data_config["val_data_path"], lines=True)
    eval_raw = [row for _, row in eval_df.iterrows()]

    train_ds = Dataset.from_pandas(train_df)
    eval_ds = Dataset.from_pandas(eval_df)

    with training_args.main_process_first(desc="tokenize datasets"):
        train_dataset = train_ds.map(
            process_func,
            remove_columns=train_ds.column_names,
            batched=True,
            num_proc=preprocessing_num_proc,
            cache_file_name=str(train_cache_file),
            load_from_cache_file=True,
            new_fingerprint=f"train-{train_cache_file.stem}",
        )
        eval_dataset = eval_ds.map(
            process_func,
            remove_columns=eval_ds.column_names,
            batched=True,
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

    return train_dataset, eval_dataset, eval_raw


def ensure_loadable_checkpoint(
    trainer: Trainer,
    tokenizer,
    save_inference_only: bool = False,
    lora_enabled: bool = False,
) -> None:
    output_dir = Path(trainer.args.output_dir)
    latest = latest_checkpoint_dir(output_dir)

    if latest is None:
        latest = output_dir / "checkpoint-final"
        trainer.save_model(str(latest))
        barrier_if_needed()

    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(str(latest))

    barrier_if_needed()

    if save_inference_only:
        if trainer.is_world_process_zero():
            removed = remove_training_state_from_checkpoint(latest)
            if removed:
                logger.info(
                    "Inference-only checkpoint cleanup removed {} entries from {}",
                    len(removed),
                    latest,
                )
        barrier_if_needed()

    if trainer.is_world_process_zero():
        if lora_enabled:
            if not checkpoint_has_lora_adapter(latest):
                raise RuntimeError(
                    f"No LoRA adapter weights found in {latest}. "
                    "Expected adapter_config.json plus adapter_model weights."
                )
        elif not checkpoint_has_hf_weights(latest):
            raise RuntimeError(
                f"No HuggingFace model weights found in {latest}. "
                "For DeepSpeed ZeRO-3, ensure stage3_gather_16bit_weights_on_model_save=true."
            )


def run(config: dict):
    local_rank = get_local_rank()
    if local_rank >= 0 and torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    if is_main_process():
        print("CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES"))
        print("TRAIN_BACKEND =", get_train_backend())
        print("TRAIN_PROFILE =", os.environ.get("TRAIN_PROFILE", ""))
        print("WORLD_SIZE =", get_world_size())
        print("torch.cuda.device_count() =", torch.cuda.device_count())
        for i in range(torch.cuda.device_count()):
            print(i, torch.cuda.get_device_name(i))

    tokenizer = AutoTokenizer.from_pretrained(
        config["model_path"],
        use_fast=False,
        trust_remote_code=True,
    )
    tokenizer.pad_token = tokenizer.eos_token if tokenizer.pad_token is None else tokenizer.pad_token

    save_inference_only = get_save_inference_only(config)
    lora_settings = get_lora_settings(config)
    training_args = build_training_args(config)

    if is_main_process():
        os.environ["SWANLAB_PROJECT"] = config.get("project_name", "qwen3-8b-sft-hsd")
        swanlab.config.update(  # type: ignore
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

    train_dataset, eval_dataset, eval_raw = build_datasets(config, tokenizer, training_args)
    model = load_model(config, training_args)
    llm_metrics = LLMmetrics()

    callbacks = []
    if save_inference_only:
        callbacks.append(InferenceOnlyCheckpointCallback())
    if is_main_process():
        callbacks.append(
            SwanLabCallback(
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
        callbacks=callbacks,
    )

    trainer.train()
    ensure_loadable_checkpoint(
        trainer,
        tokenizer,
        save_inference_only=save_inference_only,
        lora_enabled=lora_settings["enabled"],
    )

    if is_main_process():
        swanlab.finish()


def get_prompt(prompt_name_or_prompt: str):
    try:
        return eval(prompt_name_or_prompt)
    except Exception:
        return prompt_name_or_prompt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM Fine-tuning Script")
    parser.add_argument("--config", type=str, default="config.json", help="Path to config file")
    args = parser.parse_args()

    config = load_config(args.config)
    config.setdefault("transfer_data", True)
    config.setdefault("exp_name", "default-exp")

    if "random_seed" in config:
        random.seed(config["random_seed"])
        torch.manual_seed(config["random_seed"])

    try:
        run(config)
    finally:
        cleanup_distributed()
