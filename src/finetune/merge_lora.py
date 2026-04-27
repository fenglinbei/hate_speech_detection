from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import torch
from modelscope import AutoTokenizer
from transformers import AutoModelForCausalLM


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_base_model(config: dict):
    dtype = torch.bfloat16 if config.get("training", {}).get("bf16", False) else torch.float32
    kwargs: dict[str, Any] = {
        "torch_dtype": dtype,
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
        "device_map": config.get("merge_device_map", "auto"),
    }
    attn_implementation = config.get("merge_attn_implementation", config.get("attn_implementation", "flash_attention_2"))
    if attn_implementation:
        kwargs["attn_implementation"] = attn_implementation
    return AutoModelForCausalLM.from_pretrained(config["model_path"], **kwargs)


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge a PEFT LoRA adapter into its base model.")
    parser.add_argument("--config", required=True, help="Training config JSON used for the LoRA run.")
    parser.add_argument("--adapter", required=True, help="LoRA adapter checkpoint directory.")
    parser.add_argument("--output", required=True, help="Output directory for the merged HuggingFace checkpoint.")
    parser.add_argument("--max-shard-size", default="5GB", help="Shard size passed to save_pretrained.")
    parser.add_argument("--overwrite", action="store_true", help="Replace output directory if it already exists.")
    args = parser.parse_args()

    config = load_config(args.config)
    adapter_dir = Path(args.adapter)
    output_dir = Path(args.output)

    if not (adapter_dir / "adapter_config.json").exists():
        raise FileNotFoundError(f"Missing LoRA adapter_config.json in {adapter_dir}")
    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"{output_dir} already exists; pass --overwrite to replace it")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        from peft import PeftModel
    except ImportError as exc:
        raise ImportError("Merging LoRA checkpoints requires 'peft>=0.14.0'.") from exc

    base_model = load_base_model(config)
    peft_model = PeftModel.from_pretrained(base_model, str(adapter_dir), is_trainable=False)
    merged_model = peft_model.merge_and_unload(safe_merge=True)
    merged_model.save_pretrained(
        str(output_dir),
        safe_serialization=True,
        max_shard_size=args.max_shard_size,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        config["model_path"],
        use_fast=False,
        trust_remote_code=True,
    )
    tokenizer.save_pretrained(str(output_dir))

    metadata = {
        "base_model": config["model_path"],
        "adapter": str(adapter_dir),
        "merged": True,
    }
    with open(output_dir / "lora_merge_info.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"[OK] merged LoRA checkpoint: {output_dir}")


if __name__ == "__main__":
    main()
