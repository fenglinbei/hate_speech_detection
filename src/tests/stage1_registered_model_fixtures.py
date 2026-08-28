"""Small but semantically valid Stage 1 model/environment test artifacts."""

from __future__ import annotations

import importlib.metadata
import platform
from pathlib import Path

from data.training_artifacts import (
    validate_payload_manifest,
    write_bytes_atomic,
    write_canonical_json,
    write_locator_ref,
)
from model.stage1_registry import register_base_model
from scripts.stage1.capture_environment import (
    CRITICAL_DISTRIBUTIONS,
    build_environment_document,
    build_payload_manifest as build_environment_payload_manifest,
)


TOKENIZER_REVISION = "qwen3-8b-stage1-v1"
BASE_MODEL_LOGICAL_PATH = "models/base/Qwen3-8B"


def make_current_environment_artifact(root: Path) -> Path:
    """Freeze the current critical Python backend versions into a tiny target."""

    distributions = [
        {
            "installer": "conda",
            "name": "python",
            "version": platform.python_version(),
            "build": "synthetic_0",
            "build_number": 0,
            "subdir": "linux-64",
            "package_sha256_or_null": "a" * 64,
            "package_md5_or_null": None,
        }
    ]
    distributions.extend(
        {
            "installer": "python",
            "name": name,
            "version": importlib.metadata.version(name),
        }
        for name in CRITICAL_DISTRIBUTIONS
    )
    distributions.sort(key=lambda row: (row["installer"], row["name"]))
    import torch

    document = build_environment_document(
        environment_spec_sha256="b" * 64,
        installed_distributions=distributions,
        torch_build={
            "version": str(torch.__version__),
            "cuda_version": str(torch.version.cuda) if torch.version.cuda else None,
            "git_version": (
                str(torch.version.git_version)
                if torch.version.git_version
                else None
            ),
            "cudnn_version": None,
            "debug_build": bool(torch.version.debug),
        },
        driver_version="synthetic-driver",
        gpu_architecture=[
            {
                "name": "Synthetic L20",
                "compute_capability": "8.9",
                "memory_total_mib": 46068,
                "count": 4,
            }
        ],
        capture_code_sha256="c" * 64,
    )
    target = root / "artifacts/environments" / document["environment_build_id"]
    target.mkdir(parents=True)
    write_canonical_json(target / "environment.json", document)
    write_canonical_json(
        target / "provenance.json",
        {"schema_version": "stage1-environment-provenance/v1"},
    )
    write_canonical_json(
        target / "payload_manifest.json",
        build_environment_payload_manifest(target),
    )
    ref = root / "refs/environment_ref.json"
    write_locator_ref(
        ref,
        artifact_kind="stage1-environment",
        artifact_id=document["environment_build_id"],
        target=target,
        payload_manifest_sha256=validate_payload_manifest(target),
    )
    return ref


def make_registered_base_model_artifact(
    root: Path, *, environment_ref: Path
) -> Path:
    """Register a minimal local Qwen3-shaped model tree through production code."""

    source = root / BASE_MODEL_LOGICAL_PATH
    # Context fixtures may create the registered tokenizer root first.  The
    # base-model fixture intentionally completes/overwrites that same canonical
    # source tree before registration.
    source.mkdir(parents=True, exist_ok=True)
    write_canonical_json(
        source / "config.json",
        {"model_type": "qwen3", "architectures": ["Qwen3ForCausalLM"]},
    )
    write_canonical_json(
        source / "tokenizer_config.json",
        {
            "eos_token": "<eos>",
            "pad_token": "<pad>",
            "chat_template": (
                "{% if add_generation_prompt %}assistant{% endif %}"
                "{% if enable_thinking is false %}no-think{% endif %}"
            ),
        },
    )
    write_canonical_json(source / "tokenizer.json", {"version": "1.0"})
    write_bytes_atomic(source / "model.safetensors", b"synthetic-weights")
    ref = root / "refs/base_model_ref.json"
    register_base_model(
        model_dir=source,
        tokenizer_dir=None,
        model_name="synthetic-qwen3-8b",
        tokenizer_revision=TOKENIZER_REVISION,
        environment_ref=environment_ref,
        write_ref=ref,
        workspace_root=root,
        target_root=root / "artifacts/models",
    )
    return ref


__all__ = [
    "BASE_MODEL_LOGICAL_PATH",
    "TOKENIZER_REVISION",
    "make_current_environment_artifact",
    "make_registered_base_model_artifact",
]
