"""Stage 1 model artifact and registry lifecycle."""

from .stage1_registry import (
    ModelRegistryError,
    ResolvedRegisteredModel,
    finalize_model_registry,
    register_base_model,
    register_legacy_model,
    register_trained_model,
    register_training_receipt,
    resolve_registered_model,
    resolve_registered_model_dependency,
    validate_model_artifact,
    validate_model_artifact_target,
    validate_model_registry,
    validate_model_registry_target,
    validate_training_receipt_artifact,
)

__all__ = [
    "ModelRegistryError",
    "ResolvedRegisteredModel",
    "finalize_model_registry",
    "register_base_model",
    "register_legacy_model",
    "register_trained_model",
    "register_training_receipt",
    "resolve_registered_model",
    "resolve_registered_model_dependency",
    "validate_model_artifact",
    "validate_model_artifact_target",
    "validate_model_registry",
    "validate_model_registry_target",
    "validate_training_receipt_artifact",
]
