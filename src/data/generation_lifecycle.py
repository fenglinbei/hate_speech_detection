"""Immutable Stage 1 free-generation lifecycle.

The lifecycle consumes only validated context/control locators and a model
handle returned by the Stage 1 model registry.  It never accepts a checkpoint
or tokenizer path from a caller.  Every query is executed as one complete
condition block, exactly once per condition; a failure leaves no published
target or locator.

Real inference is deliberately hidden behind an executor protocol.  The
fixture executor is engineering-only, while :class:`LocalHFExecutor` is an
explicit, local-files-only reference implementation for a registered model.
"""

from __future__ import annotations

import copy
import hashlib
import os
import re
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Protocol

from data.build_context_manifest import validate_context_target
from data.context_manifest import canonical_sha256 as context_canonical_sha256
from data.control_manifest import validate_control_target
from data.counterfactual_lifecycle import validate_cf_target
from data.counterfactual_manifest import FINAL_ARTIFACT_KIND as CF_ARTIFACT_KIND
from data.training_evidence import context_policy_snapshot
from data.training_artifacts import (
    TrainingArtifactError,
    canonical_json_bytes,
    canonical_sha256,
    ensure_exact_file_set,
    finalize_target_atomic,
    load_json,
    load_jsonl,
    new_staging_directory,
    portable_dependency,
    resolve_dependency_target,
    resolve_locator_ref,
    sha256_file,
    validate_dependency_ref,
    validate_json_schema,
    validate_payload_manifest,
    write_canonical_json,
    write_canonical_jsonl,
)


GENERATION_ARTIFACT_KIND = "generation-run"
GENERATION_RUN_SCHEMA = "stage1-generation-run/v1"
GENERATION_RECORD_SCHEMA = "stage1-generation-record/v1"
GENERATION_PROVENANCE_SCHEMA = "stage1-generation-provenance/v1"
GENERATION_ID_INPUT_SCHEMA = "stage1-generation-id-inputs/v1"
GENERATION_VALIDATION_SCHEMA = "stage1-generation-validation-report/v1"
MODEL_RESOLUTION_SCHEMA = "stage1-generation-model-resolution/v1"
TRAVERSAL_POLICY = "query-major-condition-minor/v1"
ATTEMPT_POLICY = "exactly-once-no-retry/v1"
DETERMINISM_SCHEMA = "stage1-generation-determinism/v1"
DETERMINISM_POLICY = "full-frame-exact-record-and-raw-bytes/v1"
FORMAL_CONDITIONS = ("C0", "CL", "CD", "CLD", "PL", "PD")
BASE_CONDITIONS = frozenset({"C0", "CL", "CD", "CLD"})
CONTROL_CONDITIONS = frozenset({"PL", "PD"})
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class GenerationLifecycleError(RuntimeError):
    """Raised when a generation run cannot be proven complete and immutable."""


@dataclass(frozen=True)
class RegisteredModelHandle:
    """Validated model-registry result used by generation.

    ``model_path`` and ``tokenizer_path`` are outputs of the registry resolver,
    never CLI inputs.  Tests may leave them unset when using a fixture executor.
    """

    schema_version: str
    locator: Mapping[str, Any]
    target: Path
    dependency: Mapping[str, Any]
    registry_id: str
    model_key: str
    role: str
    seed: int | None
    scientific_eligible: bool
    tokenizer_revision: str
    tokenizer_content_revision: str
    tokenizer: Any
    checkpoint_format: str
    checkpoint_path: Path | None = None
    base_model_path: Path | None = None
    tokenizer_path: Path | None = None
    model_artifact_id: str = ""
    registry_report: Mapping[str, Any] | None = None
    source_contract: Any | None = None


@dataclass(frozen=True)
class _BackendSourcePaths:
    checkpoint_path: Path
    tokenizer_path: Path
    base_model_path: Path


@contextmanager
def verified_registered_model_source_load(
    *,
    source_contract: Any | None,
    scientific_eligible: bool,
    checkpoint_path: Path,
    tokenizer_path: Path,
    base_model_path: Path,
    source_names: Sequence[str],
) -> Iterator[Any]:
    """Yield registry-derived roots inside a fresh pre/post inventory lease."""

    expected = _BackendSourcePaths(
        checkpoint_path=checkpoint_path.resolve(),
        tokenizer_path=tokenizer_path.resolve(),
        base_model_path=base_model_path.resolve(),
    )
    if source_contract is None:
        if scientific_eligible:
            raise GenerationLifecycleError(
                "scientific model handle lacks its verified source-tree contract"
            )
        # Engineering fixtures and explicitly non-scientific legacy handles do
        # not claim the formal immutable-source guarantee.
        yield expected
        return
    from model.stage1_registry import (
        ModelRegistryError,
        verified_model_source_lease,
    )
    try:
        with verified_model_source_lease(
            source_contract, source_names=source_names
        ) as verified:
            actual = _BackendSourcePaths(
                checkpoint_path=verified.checkpoint_path.resolve(),
                tokenizer_path=verified.tokenizer_path.resolve(),
                base_model_path=verified.base_model_path.resolve(),
            )
            if actual != expected:
                raise GenerationLifecycleError(
                    "registered model paths differ from the verified source contract"
                )
            yield verified
    except ModelRegistryError as exc:
        raise GenerationLifecycleError(
            f"registered model source verification failed: {exc}"
        ) from exc


class RegisteredModelResolver(Protocol):
    def __call__(
        self,
        *,
        registry_ref: str | Path,
        model_key: str,
        workspace_root: str | Path,
    ) -> Any: ...


class GenerationExecutor(Protocol):
    """Exactly-once executor interface used by the immutable builder."""

    def descriptor(self) -> Mapping[str, Any]: ...

    def generate(
        self,
        *,
        messages: Sequence[Mapping[str, str]],
        query_id: str,
        condition: str,
        profile: Mapping[str, Any],
        model: RegisteredModelHandle,
    ) -> "GenerationResult": ...


@dataclass(frozen=True)
class GenerationResult:
    """Structured backend result retained for denominator-safe evaluation."""

    raw_output: str
    finish_reason: str
    generated_token_ids: tuple[int, ...]
    backend_stop_reason: str | int | None = None


@dataclass(frozen=True)
class GenerationUnit:
    query_ordinal: int
    condition_ordinal: int
    row_ordinal: int
    query_id: str
    condition: str
    content: str
    gold: list[Any]
    messages: list[dict[str, str]]
    prompt_sha256: str
    prompt_tokens: int
    context_record_sha256: str
    control_record_sha256: str | None


@dataclass(frozen=True)
class PreparedGeneration:
    generation_run_id: str
    scope: str
    scientific_eligible: bool
    split: str
    sealing_status: str
    train_context_policy_sha256: str | None
    evaluation_context_policy_sha256: str | None
    profile: Mapping[str, Any]
    model: RegisteredModelHandle
    dependencies: Mapping[str, Any]
    ordered_conditions: tuple[str, ...]
    ordered_query_ids: tuple[str, ...]
    units: tuple[GenerationUnit, ...]
    executor_descriptor: Mapping[str, Any]
    id_inputs: Mapping[str, Any]
    context_target: Path
    control_target: Path | None


class FixtureExecutor:
    """Engineering-only deterministic outputs keyed by ``(condition, query)``."""

    def __init__(self, outputs: Mapping[tuple[str, str], str]):
        frozen: dict[tuple[str, str], str] = {}
        for key, value in outputs.items():
            if (
                not isinstance(key, tuple)
                or len(key) != 2
                or not all(isinstance(part, str) and part for part in key)
                or not isinstance(value, str)
            ):
                raise GenerationLifecycleError(
                    "fixture outputs must map non-empty (condition, query_id) tuples to text"
                )
            frozen[(key[0], key[1])] = value
        self._outputs = frozen
        serializable = [
            {"condition": condition, "query_id": query_id, "raw_output": output}
            for (condition, query_id), output in sorted(frozen.items())
        ]
        self._fixture_sha256 = canonical_sha256(serializable)

    @classmethod
    def from_rows(cls, rows: Sequence[Mapping[str, Any]]) -> "FixtureExecutor":
        outputs: dict[tuple[str, str], str] = {}
        for ordinal, raw in enumerate(rows):
            row = dict(raw)
            if set(row) != {"condition", "query_id", "raw_output"}:
                raise GenerationLifecycleError(
                    f"fixture row {ordinal} must contain exactly condition/query_id/raw_output"
                )
            key = (str(row["condition"]), str(row["query_id"]))
            if key in outputs:
                raise GenerationLifecycleError(f"duplicate fixture output: {key}")
            if not isinstance(row["raw_output"], str):
                raise GenerationLifecycleError(f"fixture output is not text: {key}")
            outputs[key] = row["raw_output"]
        return cls(outputs)

    def descriptor(self) -> Mapping[str, Any]:
        return {
            "executor_id": "engineering-fixture/v1",
            "executor_revision": "stage1-generation-fixture/v1",
            "backend": "fixture",
            "scientific_eligible": False,
            "fixture_sha256": self._fixture_sha256,
        }

    def assert_exact_frame(self, expected: set[tuple[str, str]]) -> None:
        actual = set(self._outputs)
        if actual != expected:
            missing = sorted(expected - actual)
            extra = sorted(actual - expected)
            raise GenerationLifecycleError(
                f"fixture frame mismatch: missing={missing[:5]}, extra={extra[:5]}"
            )

    def generate(
        self,
        *,
        messages: Sequence[Mapping[str, str]],
        query_id: str,
        condition: str,
        profile: Mapping[str, Any],
        model: RegisteredModelHandle,
    ) -> GenerationResult:
        del messages, profile, model
        try:
            return GenerationResult(
                raw_output=self._outputs[(condition, query_id)],
                finish_reason="fixture",
                generated_token_ids=(),
                backend_stop_reason="fixture",
            )
        except KeyError as exc:  # should already be caught by assert_exact_frame
            raise GenerationLifecycleError(
                f"fixture output is missing for {(condition, query_id)}"
            ) from exc


class LocalHFExecutor:
    """Explicit local Hugging Face reference executor.

    Construction loads only registry-derived local paths.  This implementation
    intentionally supports the ``transformers`` backend and tensor parallel 1;
    a vLLM profile remains blocked until a separately versioned executor is
    supplied.  No retry or fallback branch exists.
    """

    _AUTHORITY = object()

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        *,
        _authority: object | None = None,
        model_binding: Mapping[str, Any] | None = None,
    ):
        if _authority is not self._AUTHORITY or not isinstance(model_binding, Mapping):
            raise GenerationLifecycleError(
                "LocalHFExecutor must be minted from a validated registered-model handle"
            )
        self._model = model
        self._tokenizer = tokenizer
        self._model_binding = copy.deepcopy(dict(model_binding))

    @classmethod
    def from_registered_model(
        cls,
        handle: RegisteredModelHandle,
        profile: Mapping[str, Any],
    ) -> "LocalHFExecutor":
        resolved = validate_generation_profile(profile)
        runtime = resolved["model_runtime"]
        if resolved["backend"] != "transformers":
            raise GenerationLifecycleError(
                "LocalHFExecutor requires a generation profile with backend=transformers"
            )
        if runtime["tensor_parallel_size"] != 1:
            raise GenerationLifecycleError(
                "LocalHFExecutor requires tensor_parallel_size=1"
            )
        if (
            handle.checkpoint_path is None
            or handle.base_model_path is None
            or handle.tokenizer_path is None
        ):
            raise GenerationLifecycleError(
                "model registry did not provide local model/tokenizer load roots"
            )
        for path in (
            handle.checkpoint_path,
            handle.base_model_path,
            handle.tokenizer_path,
        ):
            if not path.is_dir() or path.is_symlink():
                raise GenerationLifecycleError(
                    "registered model/tokenizer load root is missing or is a symlink"
                )
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise GenerationLifecycleError(
                "torch and transformers are required for explicit local HF inference"
            ) from exc
        dtype = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }[runtime["dtype"]]
        source_names = (
            ("checkpoint", "tokenizer", "base")
            if handle.checkpoint_format == "adapter"
            else ("checkpoint", "tokenizer")
        )
        with verified_registered_model_source_load(
            source_contract=handle.source_contract,
            scientific_eligible=handle.scientific_eligible,
            checkpoint_path=handle.checkpoint_path,
            tokenizer_path=handle.tokenizer_path,
            base_model_path=handle.base_model_path,
            source_names=source_names,
        ) as sources:
            tokenizer = AutoTokenizer.from_pretrained(
                str(sources.tokenizer_path),
                local_files_only=True,
                trust_remote_code=runtime["trust_remote_code"],
            )
            load_root = (
                sources.base_model_path
                if handle.checkpoint_format == "adapter"
                else sources.checkpoint_path
            )
            model = AutoModelForCausalLM.from_pretrained(
                str(load_root),
                local_files_only=True,
                trust_remote_code=runtime["trust_remote_code"],
                torch_dtype=dtype,
                device_map="auto",
            )
            if handle.checkpoint_format == "adapter":
                try:
                    from peft import PeftModel
                except ImportError as exc:  # pragma: no cover - environment dependent
                    raise GenerationLifecycleError(
                        "peft is required for a registered adapter checkpoint"
                    ) from exc
                model = PeftModel.from_pretrained(
                    model,
                    str(sources.checkpoint_path),
                    local_files_only=True,
                    is_trainable=False,
                )
            model.eval()
        binding = {
            "registry_id": handle.registry_id,
            "model_artifact_id": handle.model_artifact_id,
            "model_key": handle.model_key,
            "checkpoint_format": handle.checkpoint_format,
            "tokenizer_revision": handle.tokenizer_revision,
            "tokenizer_content_revision": handle.tokenizer_content_revision,
        }
        return cls(
            model,
            tokenizer,
            _authority=cls._AUTHORITY,
            model_binding=binding,
        )

    def descriptor(self) -> Mapping[str, Any]:
        return {
            "executor_id": "hf-local-transformers/v1",
            "executor_revision": "stage1-generation-hf-executor/v1",
            "backend": "transformers",
            "scientific_eligible": True,
            "model_binding_sha256": canonical_sha256(self._model_binding),
        }

    def generate(
        self,
        *,
        messages: Sequence[Mapping[str, str]],
        query_id: str,
        condition: str,
        profile: Mapping[str, Any],
        model: RegisteredModelHandle,
    ) -> GenerationResult:
        del query_id, condition
        expected_binding = {
            "registry_id": model.registry_id,
            "model_artifact_id": model.model_artifact_id,
            "model_key": model.model_key,
            "checkpoint_format": model.checkpoint_format,
            "tokenizer_revision": model.tokenizer_revision,
            "tokenizer_content_revision": model.tokenizer_content_revision,
        }
        if expected_binding != self._model_binding:
            raise GenerationLifecycleError(
                "HF executor is bound to a different registered model"
            )
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise GenerationLifecycleError("torch is required for HF inference") from exc
        sampling = profile["sampling"]
        runtime = profile["model_runtime"]
        try:
            prompt = self._tokenizer.apply_chat_template(
                list(messages),
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=runtime["thinking_mode"],
            )
        except TypeError:
            prompt = self._tokenizer.apply_chat_template(
                list(messages), tokenize=False, add_generation_prompt=True
            )
        encoded = self._tokenizer(
            prompt, return_tensors="pt", add_special_tokens=False
        )
        device = next(self._model.parameters()).device
        encoded = {key: value.to(device) for key, value in encoded.items()}
        torch.manual_seed(sampling["seed"])
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(sampling["seed"])
        with torch.inference_mode():
            generated = self._model.generate(
                **encoded,
                do_sample=False,
                max_new_tokens=sampling["max_new_tokens"],
                num_return_sequences=1,
                eos_token_id=self._tokenizer.eos_token_id,
                pad_token_id=(
                    self._tokenizer.pad_token_id
                    if self._tokenizer.pad_token_id is not None
                    else self._tokenizer.eos_token_id
                ),
                use_cache=True,
            )
        input_length = int(encoded["input_ids"].shape[-1])
        completion = generated[0, input_length:]
        token_ids = tuple(int(value) for value in completion.detach().cpu().tolist())
        eos_ids = _registered_eos_token_ids(self._tokenizer)
        if eos_ids.intersection(token_ids):
            finish_reason = "eos"
        elif len(token_ids) >= sampling["max_new_tokens"]:
            finish_reason = "length"
        else:
            raise GenerationLifecycleError(
                "HF generation stopped without EOS or the frozen length boundary"
            )
        text = self._tokenizer.decode(
            list(token_ids),
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return GenerationResult(
            raw_output=text,
            finish_reason=finish_reason,
            generated_token_ids=token_ids,
            # transformers.generate does not expose a backend stop_reason.
            # EOS itself remains provable from the final generated token ID.
            backend_stop_reason=None,
        )


class LocalVLLMExecutor:
    """Registry-bound vLLM executor for the frozen four-GPU formal profile."""

    _AUTHORITY = object()

    def __init__(
        self,
        engine: Any,
        sampling_params: Any,
        *,
        _authority: object | None = None,
        model_binding: Mapping[str, Any] | None = None,
    ) -> None:
        if _authority is not self._AUTHORITY or not isinstance(model_binding, Mapping):
            raise GenerationLifecycleError(
                "LocalVLLMExecutor must be minted from a validated registered-model handle"
            )
        self._engine = engine
        self._sampling_params = sampling_params
        self._model_binding = copy.deepcopy(dict(model_binding))

    @classmethod
    def from_registered_model(
        cls,
        handle: RegisteredModelHandle,
        profile: Mapping[str, Any],
    ) -> "LocalVLLMExecutor":
        resolved = validate_generation_profile(profile)
        runtime = resolved["model_runtime"]
        sampling = resolved["sampling"]
        if resolved["backend"] != "vllm":
            raise GenerationLifecycleError(
                "LocalVLLMExecutor requires a generation profile with backend=vllm"
            )
        if (
            handle.checkpoint_path is None
            or handle.tokenizer_path is None
            or handle.checkpoint_format not in {"base", "full"}
        ):
            raise GenerationLifecycleError(
                "formal vLLM execution requires a registered full/base checkpoint"
            )
        for path in (handle.checkpoint_path, handle.tokenizer_path):
            if not path.is_dir() or path.is_symlink():
                raise GenerationLifecycleError(
                    "registered vLLM model/tokenizer root is missing or is a symlink"
                )
        # vLLM chooses its engine implementation while importing.  Set the
        # frozen switch before the first import and carry it in the executor
        # descriptor so the published meta/ID inputs make the choice auditable.
        os.environ["VLLM_USE_V1"] = "1"
        try:
            from vllm import LLM, SamplingParams
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise GenerationLifecycleError(
                "vllm is required for the frozen formal generation backend"
            ) from exc
        params = SamplingParams(
            n=1,
            best_of=1,
            temperature=float(sampling["temperature"]),
            top_p=float(sampling["top_p"]),
            top_k=int(sampling["top_k"]),
            min_p=float(sampling["min_p"]),
            seed=int(sampling["seed"]),
            max_tokens=int(sampling["max_new_tokens"]),
            ignore_eos=False,
            skip_special_tokens=True,
        )
        with verified_registered_model_source_load(
            source_contract=handle.source_contract,
            scientific_eligible=handle.scientific_eligible,
            checkpoint_path=handle.checkpoint_path,
            tokenizer_path=handle.tokenizer_path,
            base_model_path=handle.base_model_path or handle.checkpoint_path,
            source_names=("checkpoint", "tokenizer"),
        ) as sources:
            engine = LLM(
                model=str(sources.checkpoint_path),
                tokenizer=str(sources.tokenizer_path),
                tensor_parallel_size=int(runtime["tensor_parallel_size"]),
                dtype=str(runtime["dtype"]),
                max_model_len=int(runtime["max_model_len"]),
                trust_remote_code=bool(runtime["trust_remote_code"]),
                seed=int(sampling["seed"]),
            )
        binding = {
            "registry_id": handle.registry_id,
            "model_artifact_id": handle.model_artifact_id,
            "model_key": handle.model_key,
            "checkpoint_format": handle.checkpoint_format,
            "tokenizer_revision": handle.tokenizer_revision,
            "tokenizer_content_revision": handle.tokenizer_content_revision,
        }
        return cls(
            engine,
            params,
            _authority=cls._AUTHORITY,
            model_binding=binding,
        )

    def descriptor(self) -> Mapping[str, Any]:
        return {
            "executor_id": "vllm-local-registered/v1",
            "executor_revision": "stage1-generation-vllm-executor/v1",
            "backend": "vllm",
            "scientific_eligible": True,
            "model_binding_sha256": canonical_sha256(self._model_binding),
            "vllm_use_v1": "1",
        }

    def generate(
        self,
        *,
        messages: Sequence[Mapping[str, str]],
        query_id: str,
        condition: str,
        profile: Mapping[str, Any],
        model: RegisteredModelHandle,
    ) -> GenerationResult:
        del query_id, condition
        expected_binding = {
            "registry_id": model.registry_id,
            "model_artifact_id": model.model_artifact_id,
            "model_key": model.model_key,
            "checkpoint_format": model.checkpoint_format,
            "tokenizer_revision": model.tokenizer_revision,
            "tokenizer_content_revision": model.tokenizer_content_revision,
        }
        if expected_binding != self._model_binding:
            raise GenerationLifecycleError(
                "vLLM executor is bound to a different registered model"
            )
        runtime = profile["model_runtime"]
        try:
            prompt = model.tokenizer.apply_chat_template(
                list(messages),
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=runtime["thinking_mode"],
            )
        except TypeError:
            prompt = model.tokenizer.apply_chat_template(
                list(messages), tokenize=False, add_generation_prompt=True
            )
        outputs = self._engine.generate(
            [prompt], self._sampling_params, use_tqdm=False
        )
        if len(outputs) != 1 or len(outputs[0].outputs) != 1:
            raise GenerationLifecycleError(
                "vLLM did not return exactly one completion"
            )
        completion = outputs[0].outputs[0]
        finish = completion.finish_reason
        if not isinstance(finish, str):
            raise GenerationLifecycleError(
                "vLLM finish_reason must be an original string value"
            )
        if not hasattr(completion, "stop_reason"):
            raise GenerationLifecycleError("vLLM completion lacks backend stop_reason")
        backend_stop_reason = completion.stop_reason
        if (
            isinstance(backend_stop_reason, bool)
            or not isinstance(backend_stop_reason, (str, int, type(None)))
            or isinstance(backend_stop_reason, str)
            and not backend_stop_reason
            or isinstance(backend_stop_reason, int)
            and backend_stop_reason < 0
        ):
            raise GenerationLifecycleError(
                "vLLM stop_reason must be an original non-bool integer, string, or null"
            )
        raw_token_ids = completion.token_ids
        if (
            not isinstance(raw_token_ids, Sequence)
            or isinstance(raw_token_ids, (str, bytes, bytearray))
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value < 0
                for value in raw_token_ids
            )
        ):
            raise GenerationLifecycleError(
                "vLLM token_ids must contain original non-bool non-negative integers"
            )
        token_ids = tuple(raw_token_ids)
        eos_ids = _registered_eos_token_ids(model.tokenizer)
        if finish == "stop":
            # Under vLLM V1, tokenizer EOS has stop_reason=None; a non-null
            # stop_reason identifies an explicit stop token/string instead.
            if (
                backend_stop_reason is not None
                or not token_ids
                or token_ids[-1] not in eos_ids
            ):
                raise GenerationLifecycleError(
                    "vLLM stop is not proven to be tokenizer EOS"
                )
            finish_reason = "eos"
        elif finish == "length":
            # vLLM V1 checks the max-token boundary before its EOS predicate.
            # Consequently, an EOS sampled exactly at ``max_tokens`` is
            # reported by the backend as ``length`` even though the returned
            # token frame proves a semantic EOS termination.  Prefer the
            # registered-tokenizer evidence in that single boundary case.
            if (
                backend_stop_reason is None
                and token_ids
                and token_ids[-1] in eos_ids
            ):
                finish_reason = "eos"
            else:
                finish_reason = "length"
        else:
            raise GenerationLifecycleError(
                f"vLLM returned unsupported finish reason {finish!r}"
            )
        text = _decode_generated_token_ids(model.tokenizer, token_ids)
        if not isinstance(completion.text, str) or completion.text != text:
            raise GenerationLifecycleError(
                "vLLM completion text differs from frozen tokenizer decode"
            )
        return GenerationResult(
            raw_output=text,
            finish_reason=finish_reason,
            generated_token_ids=token_ids,
            backend_stop_reason=backend_stop_reason,
        )


def _schema_path(filename: str) -> Path:
    return Path(__file__).resolve().parents[2] / "schemas" / filename


def _require_exact_keys(value: Mapping[str, Any], expected: set[str], where: str) -> None:
    if set(value) != expected:
        raise GenerationLifecycleError(
            f"{where} fields mismatch: missing={sorted(expected-set(value))}, "
            f"extra={sorted(set(value)-expected)}"
        )


def validate_generation_profile(profile: Mapping[str, Any]) -> dict[str, Any]:
    """Resolve the deterministic decoding contract without defaults."""

    if not isinstance(profile, Mapping):
        raise GenerationLifecycleError("generation profile must be an object")
    expected_top = {
        "schema_version",
        "profile_name",
        "backend",
        "ordered_conditions",
        "model_runtime",
        "sampling",
        "failure_policy",
    }
    _require_exact_keys(profile, expected_top, "generation profile")
    if profile.get("schema_version") != "stage1-generation-profile/v1":
        raise GenerationLifecycleError("unsupported generation profile schema")
    if not isinstance(profile.get("profile_name"), str) or not profile["profile_name"]:
        raise GenerationLifecycleError("generation profile_name must be non-empty")
    if profile.get("backend") not in {"vllm", "transformers"}:
        raise GenerationLifecycleError("generation backend must be vllm or transformers")
    conditions = profile.get("ordered_conditions")
    if (
        not isinstance(conditions, list)
        or not conditions
        or any(condition not in FORMAL_CONDITIONS for condition in conditions)
        or len(conditions) != len(set(conditions))
    ):
        raise GenerationLifecycleError("ordered generation conditions are invalid")

    runtime = profile.get("model_runtime")
    if not isinstance(runtime, Mapping):
        raise GenerationLifecycleError("model_runtime must be an object")
    _require_exact_keys(
        runtime,
        {
            "dtype",
            "tensor_parallel_size",
            "max_model_len",
            "max_prompt_tokens",
            "completion_reserve_tokens",
            "overflow_policy",
            "trust_remote_code",
            "thinking_mode",
        },
        "model_runtime",
    )
    if runtime.get("dtype") not in {"bfloat16", "float16", "float32"}:
        raise GenerationLifecycleError("unsupported model runtime dtype")
    for key in (
        "tensor_parallel_size",
        "max_model_len",
        "max_prompt_tokens",
        "completion_reserve_tokens",
    ):
        if not isinstance(runtime.get(key), int) or isinstance(runtime.get(key), bool) or runtime[key] <= 0:
            raise GenerationLifecycleError(f"model_runtime.{key} must be a positive integer")
    if runtime["max_prompt_tokens"] + runtime["completion_reserve_tokens"] > runtime["max_model_len"]:
        raise GenerationLifecycleError("prompt plus completion reserve exceeds max_model_len")
    if runtime.get("overflow_policy") != "hard-fail-no-truncation":
        raise GenerationLifecycleError("generation overflow policy must hard-fail")
    if not isinstance(runtime.get("trust_remote_code"), bool):
        raise GenerationLifecycleError("trust_remote_code must be boolean")
    if runtime.get("thinking_mode") is not False:
        raise GenerationLifecycleError("Stage 1 generation must disable thinking mode")

    sampling = profile.get("sampling")
    if not isinstance(sampling, Mapping):
        raise GenerationLifecycleError("sampling must be an object")
    _require_exact_keys(
        sampling,
        {
            "do_sample",
            "temperature",
            "top_p",
            "top_k",
            "min_p",
            "max_new_tokens",
            "n",
            "best_of",
            "seed",
            "stop_policy",
        },
        "sampling",
    )
    if sampling.get("do_sample") is not False:
        raise GenerationLifecycleError("formal decoding must disable sampling")
    if (
        isinstance(sampling.get("temperature"), bool)
        or isinstance(sampling.get("top_p"), bool)
        or not isinstance(sampling.get("temperature"), (int, float))
        or not isinstance(sampling.get("top_p"), (int, float))
        or sampling.get("temperature") != 0
        or sampling.get("top_p") != 1
    ):
        raise GenerationLifecycleError("deterministic decoding requires temperature=0/top_p=1")
    if (
        isinstance(sampling.get("top_k"), bool)
        or isinstance(sampling.get("min_p"), bool)
        or not isinstance(sampling.get("top_k"), (int, float))
        or not isinstance(sampling.get("min_p"), (int, float))
        or sampling.get("top_k") != -1
        or sampling.get("min_p") != 0
    ):
        raise GenerationLifecycleError("deterministic decoding requires top_k=-1/min_p=0")
    if (
        isinstance(sampling.get("n"), bool)
        or isinstance(sampling.get("best_of"), bool)
        or not isinstance(sampling.get("n"), int)
        or not isinstance(sampling.get("best_of"), int)
        or sampling.get("n") != 1
        or sampling.get("best_of") != 1
    ):
        raise GenerationLifecycleError("generation must produce exactly one completion")
    if not isinstance(sampling.get("seed"), int) or isinstance(sampling.get("seed"), bool):
        raise GenerationLifecycleError("generation seed must be an integer")
    if (
        not isinstance(sampling.get("max_new_tokens"), int)
        or isinstance(sampling.get("max_new_tokens"), bool)
        or sampling["max_new_tokens"] <= 0
    ):
        raise GenerationLifecycleError("max_new_tokens must be positive")
    if sampling["max_new_tokens"] != runtime["completion_reserve_tokens"]:
        raise GenerationLifecycleError("max_new_tokens must equal the completion reserve")
    if sampling.get("stop_policy") != "tokenizer-eos-only":
        raise GenerationLifecycleError("generation may stop only on tokenizer EOS")

    failure = profile.get("failure_policy")
    if not isinstance(failure, Mapping):
        raise GenerationLifecycleError("failure_policy must be an object")
    _require_exact_keys(
        failure,
        {
            "max_attempts",
            "fallback_generation",
            "preserve_raw_failure",
            "require_complete_paired_block",
        },
        "failure_policy",
    )
    if (
        not isinstance(failure.get("max_attempts"), int)
        or isinstance(failure.get("max_attempts"), bool)
        or failure.get("max_attempts") != 1
    ):
        raise GenerationLifecycleError("generation retry count must be exactly one")
    if failure.get("fallback_generation") is not False:
        raise GenerationLifecycleError("fallback generation is forbidden")
    if failure.get("preserve_raw_failure") is not True:
        raise GenerationLifecycleError("raw failures must be preserved")
    if failure.get("require_complete_paired_block") is not True:
        raise GenerationLifecycleError("complete paired blocks are mandatory")
    return copy.deepcopy(dict(profile))


def _normalize_executor_descriptor(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise GenerationLifecycleError("executor descriptor must be an object")
    required = {
        "executor_id",
        "executor_revision",
        "backend",
        "scientific_eligible",
    }
    if not required.issubset(value):
        raise GenerationLifecycleError("executor descriptor lacks required fields")
    optional = {"fixture_sha256", "model_binding_sha256", "vllm_use_v1"}
    _require_exact_keys(value, required | (set(value) & optional), "executor descriptor")
    for key in ("executor_id", "executor_revision", "backend"):
        if not isinstance(value.get(key), str) or not value[key]:
            raise GenerationLifecycleError(f"executor descriptor {key} is invalid")
    if not isinstance(value.get("scientific_eligible"), bool):
        raise GenerationLifecycleError("executor scientific_eligible must be boolean")
    if "fixture_sha256" in value and not SHA256_RE.fullmatch(str(value["fixture_sha256"])):
        raise GenerationLifecycleError("executor fixture hash is invalid")
    if "model_binding_sha256" in value and not SHA256_RE.fullmatch(
        str(value["model_binding_sha256"])
    ):
        raise GenerationLifecycleError("executor model binding hash is invalid")
    if "vllm_use_v1" in value and value["vllm_use_v1"] != "1":
        raise GenerationLifecycleError("executor VLLM_USE_V1 audit value must be '1'")
    if "vllm_use_v1" in value and value.get("backend") != "vllm":
        raise GenerationLifecycleError(
            "executor VLLM_USE_V1 audit value is only valid for vLLM"
        )
    if (
        value.get("backend") == "vllm"
        and value.get("scientific_eligible") is True
        and value.get("vllm_use_v1") != "1"
    ):
        raise GenerationLifecycleError(
            "scientific vLLM executor must audit VLLM_USE_V1=1"
        )
    return copy.deepcopy(dict(value))


def _coerce_model_handle(
    handle: RegisteredModelHandle,
    *,
    expected_locator: Mapping[str, Any],
    expected_target: Path,
    expected_dependency: Mapping[str, Any],
    model_key: str,
    require_scientific: bool,
) -> RegisteredModelHandle:
    if not isinstance(handle, RegisteredModelHandle):
        raise GenerationLifecycleError(
            "model resolver must return RegisteredModelHandle"
        )
    if handle.schema_version != MODEL_RESOLUTION_SCHEMA:
        raise GenerationLifecycleError("unsupported registered-model resolution schema")
    if (
        dict(handle.locator) != dict(expected_locator)
        or handle.target.resolve() != expected_target.resolve()
    ):
        raise GenerationLifecycleError("model resolver target disagrees with supplied locator")
    if dict(handle.dependency) != dict(expected_dependency):
        raise GenerationLifecycleError("model resolver dependency projection mismatch")
    if handle.registry_id != expected_locator["artifact_id"]:
        raise GenerationLifecycleError("model resolver registry ID mismatch")
    if handle.model_key != model_key or not (
        re.fullmatch(r"(?:M_LD|M_drop)/seed-[0-9]+", handle.model_key)
        or handle.model_key == "M_legacy/smoke"
    ):
        raise GenerationLifecycleError("registered model_key is invalid")
    expected_role = (
        "legacy-smoke-only"
        if handle.model_key == "M_legacy/smoke"
        else handle.model_key.split("/", 1)[0]
    )
    if handle.role != expected_role:
        raise GenerationLifecycleError("registered model role/model_key mismatch")
    if not isinstance(handle.scientific_eligible, bool):
        raise GenerationLifecycleError("registered model eligibility is invalid")
    if require_scientific and handle.scientific_eligible is not True:
        raise GenerationLifecycleError("formal generation requires a scientific model artifact")
    if require_scientific and handle.source_contract is None:
        raise GenerationLifecycleError(
            "formal generation requires a verified model source-tree contract"
        )
    if not isinstance(handle.tokenizer_revision, str) or not handle.tokenizer_revision:
        raise GenerationLifecycleError("registered model lacks tokenizer revision")
    if not isinstance(handle.tokenizer_content_revision, str) or not re.fullmatch(
        r"tok-[0-9a-f]{64}", handle.tokenizer_content_revision
    ):
        raise GenerationLifecycleError("registered model lacks tokenizer content revision")
    if handle.tokenizer is None:
        raise GenerationLifecycleError("registered model resolver did not load its tokenizer")
    if handle.checkpoint_format not in {"base", "full", "adapter"}:
        raise GenerationLifecycleError("registered checkpoint format is invalid")
    if not isinstance(handle.model_artifact_id, str) or not handle.model_artifact_id:
        raise GenerationLifecycleError("registered model artifact ID is invalid")
    return handle


def _load_registered_tokenizer(
    path: Path, loader: Callable[[Path], Any] | None
) -> Any:
    if loader is not None:
        return loader(path)
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise GenerationLifecycleError(
            "transformers is required to load the registered tokenizer"
        ) from exc
    return AutoTokenizer.from_pretrained(
        str(path),
        local_files_only=True,
        # Registry-backed tokenizers must never execute source-tree Python.
        # Model runtimes may expose a separate, explicitly frozen flag, but
        # prompt/token replay needs tokenizer assets only.
        trust_remote_code=False,
    )


def _handle_from_registry_result(
    raw: Any,
    *,
    locator: Mapping[str, Any],
    target: Path,
    dependency: Mapping[str, Any],
    tokenizer_loader: Callable[[Path], Any] | None,
) -> RegisteredModelHandle:
    tokenizer_path = Path(raw.tokenizer_path).resolve()
    checkpoint_path = Path(raw.checkpoint_path).resolve()
    base_model_path = Path(raw.base_model_path).resolve()
    scientific_eligible = bool(raw.scientific_eligible)
    source_contract = getattr(raw, "source_contract", None)
    with verified_registered_model_source_load(
        source_contract=source_contract,
        scientific_eligible=scientific_eligible,
        checkpoint_path=checkpoint_path,
        tokenizer_path=tokenizer_path,
        base_model_path=base_model_path,
        source_names=("tokenizer",),
    ) as sources:
        tokenizer = _load_registered_tokenizer(
            sources.tokenizer_path, tokenizer_loader
        )
    return RegisteredModelHandle(
        schema_version=MODEL_RESOLUTION_SCHEMA,
        locator=dict(locator),
        target=target,
        dependency=dict(dependency),
        registry_id=str(raw.registry_id),
        model_key=str(raw.model_key),
        role=str(raw.role),
        seed=raw.seed,
        scientific_eligible=scientific_eligible,
        tokenizer_revision=str(raw.tokenizer_revision),
        tokenizer_content_revision=str(raw.tokenizer_content_revision),
        tokenizer=tokenizer,
        checkpoint_format=str(raw.checkpoint_format),
        checkpoint_path=checkpoint_path,
        base_model_path=base_model_path,
        tokenizer_path=tokenizer_path,
        model_artifact_id=str(raw.model_artifact_id),
        source_contract=source_contract,
    )


def resolve_generation_model(
    registry_ref: str | Path,
    *,
    model_key: str,
    workspace_root: str | Path,
    require_scientific: bool,
    resolver: RegisteredModelResolver | None = None,
    tokenizer_loader: Callable[[Path], Any] | None = None,
) -> RegisteredModelHandle:
    """Resolve through the model-registry protocol or fail closed.

    The default import intentionally names one narrow protocol.  Generation
    does not attempt to interpret model manifests itself.
    """

    if resolver is None:
        try:
            from model.stage1_registry import resolve_registered_model
        except (ImportError, AttributeError) as exc:
            raise GenerationLifecycleError(
                "Stage 1 model registry resolver is unavailable; expected "
                "model.stage1_registry.resolve_registered_model"
            ) from exc
        resolver = resolve_registered_model
    try:
        locator, target = resolve_locator_ref(
            registry_ref, expected_kind="stage1-model-registry"
        )
        dependency = portable_dependency(locator, target, workspace_root)
        raw = resolver(
            registry_ref=registry_ref,
            model_key=model_key,
            workspace_root=workspace_root,
        )
    except GenerationLifecycleError:
        raise
    except Exception as exc:
        raise GenerationLifecycleError(f"registered model resolution failed: {exc}") from exc
    try:
        handle = (
            raw
            if isinstance(raw, RegisteredModelHandle)
            else _handle_from_registry_result(
                raw,
                locator=locator,
                target=target,
                dependency=dependency,
                tokenizer_loader=tokenizer_loader,
            )
        )
    except GenerationLifecycleError:
        raise
    except Exception as exc:
        raise GenerationLifecycleError(
            f"model registry returned an incomplete resolution: {exc}"
        ) from exc
    return _coerce_model_handle(
        handle,
        expected_locator=locator,
        expected_target=target,
        expected_dependency=dependency,
        model_key=model_key,
        require_scientific=require_scientific,
    )


def resolve_generation_model_dependency(
    registry_dependency: Mapping[str, Any],
    *,
    model_key: str,
    workspace_root: str | Path,
    require_scientific: bool,
    resolver: Callable[..., Any] | None = None,
    tokenizer_loader: Callable[[Path], Any] | None = None,
) -> RegisteredModelHandle:
    """Read-only model resolution from an embedded registry dependency."""

    try:
        dependency = validate_dependency_ref(
            registry_dependency, expected_kind="stage1-model-registry"
        )
        target = resolve_dependency_target(dependency, workspace_root)
    except TrainingArtifactError as exc:
        raise GenerationLifecycleError(
            f"invalid registered model dependency: {exc}"
        ) from exc
    locator = {
        "schema_version": "stage1-locator-ref/v1",
        "artifact_kind": dependency["artifact_kind"],
        "artifact_id": dependency["artifact_id"],
        "target_path": str(target.resolve()),
        "payload_manifest_sha256": dependency["payload_manifest_sha256"],
    }
    if resolver is None:
        try:
            from model.stage1_registry import resolve_registered_model_dependency
        except (ImportError, AttributeError) as exc:
            raise GenerationLifecycleError(
                "Stage 1 model dependency resolver is unavailable; expected "
                "model.stage1_registry.resolve_registered_model_dependency"
            ) from exc
        resolver = resolve_registered_model_dependency
    try:
        raw = resolver(
            registry_dependency=dependency,
            model_key=model_key,
            workspace_root=workspace_root,
        )
        handle = (
            raw
            if isinstance(raw, RegisteredModelHandle)
            else _handle_from_registry_result(
                raw,
                locator=locator,
                target=target,
                dependency=dependency,
                tokenizer_loader=tokenizer_loader,
            )
        )
    except GenerationLifecycleError:
        raise
    except Exception as exc:
        raise GenerationLifecycleError(
            f"registered model dependency resolution failed: {exc}"
        ) from exc
    return _coerce_model_handle(
        handle,
        expected_locator=locator,
        expected_target=target,
        expected_dependency=dependency,
        model_key=model_key,
        require_scientific=require_scientific,
    )


def _messages(item: Mapping[str, Any], *, condition: str, query_id: str) -> list[dict[str, str]]:
    messages_list = item.get("messages_list")
    if (
        not isinstance(messages_list, list)
        or len(messages_list) != 1
        or not isinstance(messages_list[0], list)
        or len(messages_list[0]) != 2
    ):
        raise GenerationLifecycleError(
            f"{condition}/{query_id} must contain exactly one system+user prompt"
        )
    messages: list[dict[str, str]] = []
    for expected_role, raw in zip(("system", "user"), messages_list[0], strict=True):
        if (
            not isinstance(raw, Mapping)
            or raw.get("role") != expected_role
            or not isinstance(raw.get("content"), str)
            or not raw["content"]
            or set(raw) != {"role", "content"}
        ):
            raise GenerationLifecycleError(
                f"{condition}/{query_id} prompt messages are not canonical system+user"
            )
        messages.append({"role": expected_role, "content": raw["content"]})
    return messages


def _runner_prompt_lineage(
    item: Mapping[str, Any], condition: str
) -> tuple[str, int, str, str | None]:
    context = item.get("context_manifest")
    if not isinstance(context, Mapping):
        raise GenerationLifecycleError(f"{condition} runner lacks context lineage")
    context_hash = context.get("record_sha256")
    if not isinstance(context_hash, str) or not SHA256_RE.fullmatch(context_hash):
        raise GenerationLifecycleError(f"{condition} context record hash is invalid")
    control_hash: str | None = None
    lineage: Mapping[str, Any] = context
    if condition in CONTROL_CONDITIONS:
        control = item.get("control_manifest")
        if not isinstance(control, Mapping):
            raise GenerationLifecycleError(f"{condition} runner lacks control lineage")
        control_hash = control.get("record_sha256")
        if not isinstance(control_hash, str) or not SHA256_RE.fullmatch(control_hash):
            raise GenerationLifecycleError(f"{condition} control record hash is invalid")
        if control.get("status") not in {"ok", "empty-target"}:
            raise GenerationLifecycleError(
                f"{condition} is unavailable for a complete paired block"
            )
        lineage = control
    prompt_hash = lineage.get("chat_prompt_sha256")
    prompt_tokens = lineage.get("chat_prompt_tokens")
    if not isinstance(prompt_hash, str) or not SHA256_RE.fullmatch(prompt_hash):
        raise GenerationLifecycleError(f"{condition} prompt hash is invalid")
    if not isinstance(prompt_tokens, int) or isinstance(prompt_tokens, bool) or prompt_tokens <= 0:
        raise GenerationLifecycleError(f"{condition} prompt token count is invalid")
    return prompt_hash, prompt_tokens, context_hash, control_hash


def _context_snapshot(
    target: Path,
    *,
    tokenizer: Any,
    workspace_root: str | Path,
    scientific: bool = False,
) -> tuple[dict[str, Any], str, list[dict[str, Any]]]:
    try:
        if scientific:
            meta = validate_context_target(
                target,
                workspace_root=workspace_root,
            )
        else:
            meta = validate_context_target(
                target,
                tokenizer=tokenizer,
                workspace_root=workspace_root,
            )
    except Exception as exc:
        raise GenerationLifecycleError(f"context target validation failed: {exc}") from exc
    split = meta.get("split")
    if split not in {"train", "dev", "test"}:
        raise GenerationLifecycleError("context split is invalid")
    records = load_jsonl(target / f"context_manifest.{split}.jsonl")
    return dict(meta), str(split), records


def _runner_items(target: Path, condition: str, split: str) -> list[dict[str, Any]]:
    path = target / "conditions" / "runner" / condition / f"{split}.json"
    raw = load_json(path)
    if not isinstance(raw, list) or any(not isinstance(row, dict) for row in raw):
        raise GenerationLifecycleError(f"invalid runner adapter for {condition}/{split}")
    return [dict(row) for row in raw]


def _validate_model_tokenizer_lineage(
    context_meta: Mapping[str, Any], handle: RegisteredModelHandle
) -> None:
    budget = context_meta.get("budget")
    revision = budget.get("tokenizer_revision") if isinstance(budget, Mapping) else None
    if revision != handle.tokenizer_revision:
        raise GenerationLifecycleError(
            "context tokenizer revision differs from registered model tokenizer"
        )


def _validate_formal_protocol_binding(
    handle: RegisteredModelHandle,
    profile: Mapping[str, Any],
    *,
    workspace_root: str | Path,
) -> dict[str, Any]:
    """Bind formal decoding to the registry's frozen training-plan protocol."""

    try:
        from data.training_plan import validate_training_plan_target

        registry = load_json(handle.target / "registry.json")
        if not isinstance(registry, Mapping):
            raise GenerationLifecycleError("model registry payload is not an object")
        plan_dependency = validate_dependency_ref(
            registry.get("training_plan_dependency", {}),
            expected_kind="training-plan",
        )
        plan_target = resolve_dependency_target(plan_dependency, workspace_root)
        plan = validate_training_plan_target(
            plan_target, workspace_root=workspace_root
        )
        if plan.get("scope") != "formal" or plan.get("scientific_eligible") is not True:
            raise GenerationLifecycleError(
                "formal generation requires a formal scientific training plan"
            )
        snapshot = load_json(plan_target / "protocol_snapshot.json")
        profiles = snapshot.get("profiles") if isinstance(snapshot, Mapping) else None
        generation = profiles.get("generation") if isinstance(profiles, Mapping) else None
        frozen_profile = generation.get("resolved") if isinstance(generation, Mapping) else None
        if not isinstance(frozen_profile, Mapping):
            raise GenerationLifecycleError(
                "formal training plan lacks its frozen generation profile"
            )
        if dict(profile) != dict(frozen_profile):
            raise GenerationLifecycleError(
                "generation profile differs from the model registry's frozen protocol"
            )
        if generation.get("sha256") != canonical_sha256(frozen_profile):
            raise GenerationLifecycleError("frozen generation profile hash mismatch")
        if snapshot.get("ordered_generation_conditions") != profile.get(
            "ordered_conditions"
        ):
            raise GenerationLifecycleError(
                "formal condition order differs from the frozen protocol"
            )
        train_policy = plan.get("train_context_policy")
        train_policy_hash = plan.get("train_context_policy_sha256")
        if (
            not isinstance(train_policy, Mapping)
            or not isinstance(train_policy_hash, str)
            or not SHA256_RE.fullmatch(train_policy_hash)
            or canonical_sha256(train_policy) != train_policy_hash
            or snapshot.get("train_context_policy") != train_policy
            or snapshot.get("train_context_policy_sha256") != train_policy_hash
        ):
            raise GenerationLifecycleError(
                "formal training plan lacks a replayable train context policy"
            )
        return {
            "training_plan_dependency": dict(plan_dependency),
            "train_context_policy": copy.deepcopy(dict(train_policy)),
            "train_context_policy_sha256": train_policy_hash,
        }
    except GenerationLifecycleError:
        raise
    except Exception as exc:
        raise GenerationLifecycleError(
            f"formal generation protocol binding failed: {exc}"
        ) from exc


def _derive_sealing_status(
    *,
    context_locator: Mapping[str, Any],
    context_meta: Mapping[str, Any],
    context_target: Path,
    sealed: bool | None,
    enforce_assertion: bool = True,
) -> str:
    """Derive dev/test sealing from validated context lineage.

    ``sealed`` is deliberately only an operator assertion.  It never chooses
    the status and therefore cannot turn a dev artifact into a sealed one.
    """

    split = context_meta.get("split")
    artifact_kind = context_meta.get("artifact_kind")
    locator_kind = context_locator.get("artifact_kind")
    id_inputs = context_meta.get("id_inputs")
    if not isinstance(id_inputs, Mapping):
        raise GenerationLifecycleError("context sealing lineage lacks ID inputs")
    frozen_policy = id_inputs.get("frozen_policy_ref")
    frozen_file = context_target / "frozen_policy_ref.json"
    if (
        split == "dev"
        and artifact_kind == "context"
        and locator_kind == "context"
        and frozen_policy is None
        and not frozen_file.exists()
    ):
        status = "unsealed-dev"
    elif (
        split == "test"
        and artifact_kind == "test-context"
        and locator_kind == "test-context"
        and isinstance(frozen_policy, Mapping)
        and frozen_file.is_file()
    ):
        status = "sealed-test"
    else:
        raise GenerationLifecycleError(
            "generation context kind/split/frozen-policy lineage is not a valid "
            "unsealed dev or sealed test input"
        )
    if enforce_assertion:
        expected_assertion = status == "sealed-test"
        if sealed is not None and sealed is not expected_assertion:
            raise GenerationLifecycleError(
                f"--sealed assertion disagrees with derived {status} context lineage"
            )
        if expected_assertion and sealed is not True:
            raise GenerationLifecycleError("sealed test generation requires --sealed")
    return status


def prepare_generation(
    *,
    profile: str | Path | Mapping[str, Any],
    context_ref: str | Path,
    model_registry_ref: str | Path,
    model_key: str,
    workspace_root: str | Path,
    scope: str,
    executor_descriptor: Mapping[str, Any],
    sealed: bool | None = None,
    control_ref: str | Path | None = None,
    cf_ref: str | Path | None = None,
    determinism_repetitions: int = 1,
    model_resolver: RegisteredModelResolver | None = None,
    tokenizer_loader: Callable[[Path], Any] | None = None,
) -> PreparedGeneration:
    """Validate dependencies and freeze the exact query/condition traversal."""

    if scope not in {"formal", "engineering"}:
        raise GenerationLifecycleError("generation scope must be formal or engineering")
    if (
        isinstance(determinism_repetitions, bool)
        or not isinstance(determinism_repetitions, int)
        or determinism_repetitions not in {1, 2}
    ):
        raise GenerationLifecycleError(
            "determinism_repetitions must be exactly 1 or 2"
        )
    if scope == "formal" and (
        model_resolver is not None or tokenizer_loader is not None
    ):
        raise GenerationLifecycleError(
            "formal generation forbids injected model/tokenizer resolvers"
        )
    resolved_profile = validate_generation_profile(
        load_json(profile) if isinstance(profile, (str, Path)) else profile
    )
    descriptor = _normalize_executor_descriptor(executor_descriptor)
    if scope == "formal":
        if tuple(resolved_profile["ordered_conditions"]) != FORMAL_CONDITIONS:
            raise GenerationLifecycleError(
                "formal Stage 1 generation requires the complete frozen condition order"
            )
        if descriptor["scientific_eligible"] is not True:
            raise GenerationLifecycleError("engineering executor cannot publish a formal run")
        if descriptor["backend"] != resolved_profile["backend"]:
            raise GenerationLifecycleError("formal executor backend differs from frozen profile")
        if (
            resolved_profile["backend"] == "vllm"
            and descriptor.get("vllm_use_v1") != "1"
        ):
            raise GenerationLifecycleError(
                "formal vLLM executor must audit VLLM_USE_V1=1"
            )
    handle = resolve_generation_model(
        model_registry_ref,
        model_key=model_key,
        workspace_root=workspace_root,
        require_scientific=scope == "formal",
        resolver=model_resolver,
        tokenizer_loader=tokenizer_loader,
    )
    formal_binding: dict[str, Any] | None = None
    if scope == "formal":
        formal_binding = _validate_formal_protocol_binding(
            handle, resolved_profile, workspace_root=workspace_root
        )
    try:
        context_locator, context_target = resolve_locator_ref(
            context_ref, expected_kind=("context", "test-context")
        )
        context_dependency = portable_dependency(
            context_locator, context_target, workspace_root
        )
    except TrainingArtifactError as exc:
        raise GenerationLifecycleError(f"invalid context locator: {exc}") from exc
    context_meta, split, context_records = _context_snapshot(
        context_target,
        tokenizer=handle.tokenizer,
        workspace_root=workspace_root,
        scientific=scope == "formal",
    )
    sealing_status = _derive_sealing_status(
        context_locator=context_locator,
        context_meta=context_meta,
        context_target=context_target,
        sealed=sealed,
    )
    _validate_model_tokenizer_lineage(context_meta, handle)
    if scope == "formal":
        if context_meta.get("scientific_eligible") is not True:
            raise GenerationLifecycleError(
                "formal generation requires a scientific context artifact"
            )
        assert formal_binding is not None
        evaluation_policy, evaluation_policy_sha256 = context_policy_snapshot(
            context_target,
            expected_split=split,
            require_scientific=True,
            context_meta=context_meta,
        )
        if (
            evaluation_policy != formal_binding["train_context_policy"]
            or evaluation_policy_sha256
            != formal_binding["train_context_policy_sha256"]
        ):
            raise GenerationLifecycleError(
                "evaluation context policy differs from the registry-derived "
                "training plan"
            )
        train_context_policy_sha256: str | None = formal_binding[
            "train_context_policy_sha256"
        ]
    else:
        if sealing_status != "unsealed-dev":
            raise GenerationLifecycleError(
                "engineering generation is restricted to unsealed dev context"
            )
        evaluation_policy_sha256 = None
        train_context_policy_sha256 = None

    ordered_conditions = tuple(resolved_profile["ordered_conditions"])
    needs_control = bool(set(ordered_conditions) & CONTROL_CONDITIONS)
    control_target: Path | None = None
    control_dependency: dict[str, Any] | None = None
    if needs_control and control_ref is None:
        raise GenerationLifecycleError("PL/PD conditions require a verified control ref")
    if not needs_control and control_ref is not None:
        raise GenerationLifecycleError(
            "control ref is forbidden when no control condition is requested"
        )
    if control_ref is not None:
        try:
            expected_control_kind = (
                "test-control"
                if sealing_status == "sealed-test"
                else "control"
            )
            control_locator, control_target = resolve_locator_ref(
                control_ref, expected_kind=expected_control_kind
            )
            control_dependency = portable_dependency(
                control_locator, control_target, workspace_root
            )
            if scope == "formal":
                control_report = validate_control_target(
                    control_target,
                    context_target=context_target,
                    workspace_root=workspace_root,
                )
            else:
                control_report = validate_control_target(
                    control_target,
                    tokenizer=handle.tokenizer,
                    context_target=context_target,
                    workspace_root=workspace_root,
                )
        except (TrainingArtifactError, Exception) as exc:
            if isinstance(exc, GenerationLifecycleError):
                raise
            raise GenerationLifecycleError(f"control target validation failed: {exc}") from exc
        if control_report.get("split") != split:
            raise GenerationLifecycleError("control/context split mismatch")
        if control_report.get("context_build_id") != context_locator["artifact_id"]:
            raise GenerationLifecycleError("control/context dependency mismatch")

    cf_dependency: dict[str, Any] | None = None
    if sealing_status == "sealed-test" and cf_ref is None:
        raise GenerationLifecycleError(
            "sealed test generation requires the finalized test CF ref"
        )
    if scope == "engineering" and cf_ref is not None:
        raise GenerationLifecycleError(
            "engineering generation cannot claim a finalized CF dependency"
        )
    if cf_ref is not None:
        try:
            cf_locator, cf_target = resolve_locator_ref(
                cf_ref, expected_kind=CF_ARTIFACT_KIND
            )
            cf_dependency = portable_dependency(
                cf_locator, cf_target, workspace_root
            )
            cf_report = validate_cf_target(
                cf_target, workspace_root=workspace_root
            )
            cf_context_dependency = load_json(cf_target / "context_ref.json")
        except Exception as exc:
            raise GenerationLifecycleError(
                f"finalized CF validation failed: {exc}"
            ) from exc
        if cf_report.get("split") != split:
            raise GenerationLifecycleError("finalized CF/context split mismatch")
        if cf_context_dependency != context_dependency:
            raise GenerationLifecycleError(
                "finalized CF does not consume the selected context"
            )
        if scope == "formal" and cf_report.get("scientific_eligible") is not True:
            raise GenerationLifecycleError(
                "formal generation requires a scientific finalized CF artifact"
            )

    query_ids = [str(record.get("query", {}).get("id")) for record in context_records]
    if (
        not query_ids
        or any(not value or value == "None" for value in query_ids)
        or len(query_ids) != len(set(query_ids))
    ):
        raise GenerationLifecycleError("context ordered query frame is invalid")
    condition_items: dict[str, list[dict[str, Any]]] = {}
    for condition in ordered_conditions:
        source = control_target if condition in CONTROL_CONDITIONS else context_target
        assert source is not None
        items = _runner_items(source, condition, split)
        item_ids = [str(item.get("id")) for item in items]
        if item_ids != query_ids:
            raise GenerationLifecycleError(
                f"{condition} runner does not contain the exact paired query traversal"
            )
        condition_items[condition] = items

    runtime = resolved_profile["model_runtime"]
    units: list[GenerationUnit] = []
    for query_ordinal, query_id in enumerate(query_ids):
        canonical_content: str | None = None
        canonical_gold: list[Any] | None = None
        context_hash: str | None = None
        for condition_ordinal, condition in enumerate(ordered_conditions):
            item = condition_items[condition][query_ordinal]
            content = item.get("content")
            gold = item.get("gt_quadruples")
            if not isinstance(content, str) or not isinstance(gold, list):
                raise GenerationLifecycleError(
                    f"{condition}/{query_id} lacks immutable content/gold"
                )
            if canonical_content is None:
                canonical_content = content
                canonical_gold = copy.deepcopy(gold)
            elif content != canonical_content or gold != canonical_gold:
                raise GenerationLifecycleError(
                    f"paired conditions disagree on query content/gold: {query_id}"
                )
            messages = _messages(item, condition=condition, query_id=query_id)
            prompt_hash, prompt_tokens, row_context_hash, control_hash = (
                _runner_prompt_lineage(item, condition)
            )
            if context_hash is None:
                context_hash = row_context_hash
            elif context_hash != row_context_hash:
                raise GenerationLifecycleError(
                    f"paired conditions disagree on context record: {query_id}"
                )
            if prompt_tokens > runtime["max_prompt_tokens"]:
                raise GenerationLifecycleError(
                    f"prompt overflow at {condition}/{query_id}: {prompt_tokens}"
                )
            if prompt_tokens + runtime["completion_reserve_tokens"] > runtime["max_model_len"]:
                raise GenerationLifecycleError(
                    f"sequence overflow at {condition}/{query_id}"
                )
            units.append(
                GenerationUnit(
                    query_ordinal=query_ordinal,
                    condition_ordinal=condition_ordinal,
                    row_ordinal=query_ordinal * len(ordered_conditions) + condition_ordinal,
                    query_id=query_id,
                    condition=condition,
                    content=content,
                    gold=copy.deepcopy(gold),
                    messages=messages,
                    prompt_sha256=prompt_hash,
                    prompt_tokens=prompt_tokens,
                    context_record_sha256=row_context_hash,
                    control_record_sha256=control_hash,
                )
            )

    dependencies: dict[str, Any] = {
        "context": context_dependency,
        "model_registry": dict(handle.dependency),
    }
    if formal_binding is not None:
        dependencies["training_plan"] = formal_binding[
            "training_plan_dependency"
        ]
    if control_dependency is not None:
        dependencies["control"] = control_dependency
    if cf_dependency is not None:
        dependencies["counterfactual"] = cf_dependency
    builder_hash = sha256_file(__file__)
    id_inputs = {
        "schema_version": GENERATION_ID_INPUT_SCHEMA,
        "scope": scope,
        "dependencies": dependencies,
        "model_registry_id": handle.registry_id,
        "model_artifact_id": handle.model_artifact_id,
        "model_key": handle.model_key,
        "model_role": handle.role,
        "model_seed": handle.seed,
        "checkpoint_format": handle.checkpoint_format,
        "tokenizer_revision": handle.tokenizer_revision,
        "tokenizer_content_revision": handle.tokenizer_content_revision,
        "split": split,
        "sealing_status": sealing_status,
        "train_context_policy_sha256": train_context_policy_sha256,
        "evaluation_context_policy_sha256": evaluation_policy_sha256,
        "ordered_conditions": list(ordered_conditions),
        "ordered_query_ids_sha256": canonical_sha256(query_ids),
        "query_count": len(query_ids),
        "row_traversal_policy": TRAVERSAL_POLICY,
        "determinism": {
            "schema_version": DETERMINISM_SCHEMA,
            "comparison_policy": DETERMINISM_POLICY,
            "execution_repetitions": determinism_repetitions,
        },
        "generation_profile": resolved_profile,
        "executor": descriptor,
        "builder_code_sha256": builder_hash,
    }
    generation_id = "gen-" + canonical_sha256(id_inputs)
    return PreparedGeneration(
        generation_run_id=generation_id,
        scope=scope,
        scientific_eligible=scope == "formal",
        split=split,
        sealing_status=sealing_status,
        train_context_policy_sha256=train_context_policy_sha256,
        evaluation_context_policy_sha256=evaluation_policy_sha256,
        profile=resolved_profile,
        model=handle,
        dependencies=dependencies,
        ordered_conditions=ordered_conditions,
        ordered_query_ids=tuple(query_ids),
        units=tuple(units),
        executor_descriptor=descriptor,
        id_inputs=id_inputs,
        context_target=context_target,
        control_target=control_target,
    )


def preflight_generation(**kwargs: Any) -> dict[str, Any]:
    prepared = prepare_generation(**kwargs)
    supported = {
        "transformers": "hf-local-transformers/v1",
        "vllm": "vllm-local-registered/v1",
    }
    execution_supported = prepared.executor_descriptor.get(
        "executor_id"
    ) == supported.get(str(prepared.profile.get("backend")))
    return {
        "schema_version": "stage1-generation-preflight/v1",
        "inputs_valid": True,
        "execution_supported": execution_supported,
        "ready_to_publish": execution_supported,
        "generation_run_id": prepared.generation_run_id,
        "scope": prepared.scope,
        "scientific_eligible": prepared.scientific_eligible,
        "model_registry_id": prepared.model.registry_id,
        "model_artifact_id": prepared.model.model_artifact_id,
        "model_key": prepared.model.model_key,
        "model_role": prepared.model.role,
        "split": prepared.split,
        "sealing_status": prepared.sealing_status,
        "train_context_policy_sha256": prepared.train_context_policy_sha256,
        "evaluation_context_policy_sha256": (
            prepared.evaluation_context_policy_sha256
        ),
        "ordered_conditions": list(prepared.ordered_conditions),
        "query_count": len(prepared.ordered_query_ids),
        "row_count": len(prepared.units),
        "ordered_query_ids_sha256": canonical_sha256(
            list(prepared.ordered_query_ids)
        ),
        "row_traversal_policy": TRAVERSAL_POLICY,
        "attempt_policy": ATTEMPT_POLICY,
    }


def _text_sha256(text: str) -> str:
    return hashlib.sha256(text.replace("\r\n", "\n").encode("utf-8")).hexdigest()


def _raw_output_frame_sha256(records: Sequence[Mapping[str, Any]]) -> str:
    """Hash exact normalized raw-output bytes with unambiguous length framing."""

    digest = hashlib.sha256()
    for record in records:
        raw = record.get("raw_output")
        if not isinstance(raw, str):
            raise GenerationLifecycleError(
                "determinism frame contains a non-text raw output"
            )
        payload = raw.replace("\r\n", "\n").encode("utf-8")
        digest.update(len(payload).to_bytes(8, byteorder="big", signed=False))
        digest.update(payload)
    return digest.hexdigest()


def _build_determinism_document(
    prepared: PreparedGeneration,
    executions: Sequence[Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    contract = prepared.id_inputs.get("determinism")
    if not isinstance(contract, Mapping):
        raise GenerationLifecycleError("generation ID lacks determinism contract")
    repetitions = contract.get("execution_repetitions")
    if repetitions != len(executions) or repetitions not in {1, 2}:
        raise GenerationLifecycleError("determinism execution count mismatch")
    if not executions or any(len(rows) != len(prepared.units) for rows in executions):
        raise GenerationLifecycleError("determinism execution frame is incomplete")
    canonical_frames = [canonical_json_bytes(list(rows)) for rows in executions]
    exact_match = all(frame == canonical_frames[0] for frame in canonical_frames[1:])
    if repetitions == 2 and not exact_match:
        raise GenerationLifecycleError(
            "fixed-seed greedy rerun produced a non-identical generation frame"
        )
    records_hashes = [canonical_sha256(list(rows)) for rows in executions]
    raw_hashes = [_raw_output_frame_sha256(rows) for rows in executions]
    record_hash_registry = [
        canonical_sha256([row["record_sha256"] for row in rows])
        for rows in executions
    ]
    return {
        "schema_version": DETERMINISM_SCHEMA,
        "generation_run_id": prepared.generation_run_id,
        "comparison_policy": DETERMINISM_POLICY,
        "execution_repetitions": repetitions,
        "exact_match": exact_match,
        "mismatch_count": 0 if exact_match else 1,
        "per_execution_records_sha256": records_hashes,
        "per_execution_record_hashes_sha256": record_hash_registry,
        "per_execution_raw_output_bytes_sha256": raw_hashes,
    }


def _registered_eos_token_ids(tokenizer: Any) -> frozenset[int]:
    """Return the exact EOS-ID set exposed by the registered tokenizer."""

    raw = getattr(tokenizer, "eos_token_id", None)
    if isinstance(raw, (list, tuple, set, frozenset)):
        values = list(raw)
    else:
        values = [raw]
    if not values or any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in values
    ):
        raise GenerationLifecycleError(
            "registered tokenizer does not expose valid non-bool EOS token IDs"
        )
    return frozenset(values)


def _decode_generated_token_ids(tokenizer: Any, token_ids: Sequence[int]) -> str:
    """Apply the frozen, replayable Stage 1 completion decode contract."""

    try:
        decoded = tokenizer.decode(
            list(token_ids),
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
    except Exception as exc:
        raise GenerationLifecycleError(
            "registered tokenizer cannot replay the frozen completion decode"
        ) from exc
    if not isinstance(decoded, str):
        raise GenerationLifecycleError("registered tokenizer decode did not return text")
    return decoded


def _validate_generated_output_semantics(
    *,
    raw_output: str,
    finish_reason: str,
    generated_token_ids: Sequence[int],
    backend_stop_reason: str | int | None,
    tokenizer: Any,
    maximum: int,
    scope: str,
    backend: str,
) -> None:
    """Cross-check stop evidence, token budget, and frozen decoded text."""

    if (
        isinstance(backend_stop_reason, bool)
        or not isinstance(backend_stop_reason, (str, int, type(None)))
        or isinstance(backend_stop_reason, str)
        and not backend_stop_reason
        or isinstance(backend_stop_reason, int)
        and backend_stop_reason < 0
    ):
        raise GenerationLifecycleError("generation backend stop reason is invalid")
    if finish_reason == "fixture":
        if scope != "engineering":
            raise GenerationLifecycleError("formal generation contains a fixture result")
        if generated_token_ids or backend_stop_reason != "fixture":
            raise GenerationLifecycleError(
                "fixture result must retain an empty token frame and fixture stop reason"
            )
        return

    eos_ids = _registered_eos_token_ids(tokenizer)
    decoded = _decode_generated_token_ids(tokenizer, generated_token_ids)
    if decoded != raw_output:
        raise GenerationLifecycleError(
            "generation raw output differs from frozen tokenizer decode"
        )
    eos_positions = [
        index
        for index, token_id in enumerate(generated_token_ids)
        if token_id in eos_ids
    ]
    if eos_positions and eos_positions != [len(generated_token_ids) - 1]:
        raise GenerationLifecycleError(
            "generation contains tokens after tokenizer EOS"
        )
    stop_reason_is_eos = (
        isinstance(backend_stop_reason, int)
        and not isinstance(backend_stop_reason, bool)
        and backend_stop_reason in eos_ids
    )
    if finish_reason == "eos":
        if not eos_positions or eos_positions[-1] != len(generated_token_ids) - 1:
            raise GenerationLifecycleError(
                "EOS finish reason lacks registered-tokenizer EOS evidence"
            )
        if scope == "formal" and backend == "vllm" and backend_stop_reason is not None:
            raise GenerationLifecycleError(
                "formal vLLM EOS result has a non-EOS backend stop reason"
            )
        if isinstance(backend_stop_reason, int) and not stop_reason_is_eos:
            raise GenerationLifecycleError(
                "backend stop token differs from registered tokenizer EOS"
            )
    elif finish_reason == "length":
        if len(generated_token_ids) != maximum:
            raise GenerationLifecycleError(
                "length finish reason does not equal the frozen completion boundary"
            )
        if eos_positions or stop_reason_is_eos:
            raise GenerationLifecycleError(
                "length finish reason conflicts with tokenizer EOS evidence"
            )
        if scope == "formal" and backend == "vllm" and backend_stop_reason is not None:
            raise GenerationLifecycleError(
                "formal vLLM length result has an unexpected backend stop reason"
            )
    else:
        raise GenerationLifecycleError("generation finish reason is unsupported")
    if len(generated_token_ids) > maximum:
        raise GenerationLifecycleError("generation exceeded the frozen completion budget")


def _record_from_output(
    prepared: PreparedGeneration,
    unit: GenerationUnit,
    result: GenerationResult | str,
) -> dict[str, Any]:
    if isinstance(result, str):
        if prepared.scope != "engineering":
            raise GenerationLifecycleError(
                f"formal executor returned an unstructured result at "
                f"{unit.condition}/{unit.query_id}"
            )
        result = GenerationResult(
            raw_output=result,
            finish_reason="fixture",
            generated_token_ids=(),
            backend_stop_reason="fixture",
        )
    if not isinstance(result, GenerationResult) or not isinstance(
        result.raw_output, str
    ):
        raise GenerationLifecycleError(
            f"executor returned an invalid result at {unit.condition}/{unit.query_id}"
        )
    if result.finish_reason not in {"eos", "length", "fixture"}:
        raise GenerationLifecycleError(
            f"executor returned an unknown finish reason at "
            f"{unit.condition}/{unit.query_id}"
        )
    if prepared.scope == "formal" and result.finish_reason == "fixture":
        raise GenerationLifecycleError("formal executor returned a fixture result")
    token_ids = list(result.generated_token_ids)
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in token_ids
    ):
        raise GenerationLifecycleError("executor returned invalid generated token IDs")
    completion_tokens = len(token_ids)
    maximum = int(prepared.profile["sampling"]["max_new_tokens"])
    if completion_tokens > maximum:
        raise GenerationLifecycleError("executor exceeded the frozen completion budget")
    _validate_generated_output_semantics(
        raw_output=result.raw_output,
        finish_reason=result.finish_reason,
        generated_token_ids=token_ids,
        backend_stop_reason=result.backend_stop_reason,
        tokenizer=prepared.model.tokenizer,
        maximum=maximum,
        scope=prepared.scope,
        backend=str(prepared.profile["backend"]),
    )
    runner_status = "length" if result.finish_reason == "length" else "ok"
    record = {
        "schema_version": GENERATION_RECORD_SCHEMA,
        "generation_run_id": prepared.generation_run_id,
        "row_ordinal": unit.row_ordinal,
        "query_ordinal": unit.query_ordinal,
        "condition_ordinal": unit.condition_ordinal,
        "model_key": prepared.model.model_key,
        "model_role": prepared.model.role,
        "split": prepared.split,
        "condition": unit.condition,
        "query_id": unit.query_id,
        "content": unit.content,
        "content_sha256": _text_sha256(unit.content),
        "gold": copy.deepcopy(unit.gold),
        "gold_sha256": context_canonical_sha256(unit.gold),
        "context_record_sha256": unit.context_record_sha256,
        "prompt_sha256": unit.prompt_sha256,
        "prompt_tokens": unit.prompt_tokens,
        "control_record_sha256": unit.control_record_sha256,
        "attempt_count": 1,
        "runner_status": runner_status,
        "finish_reason": result.finish_reason,
        "backend_stop_reason": result.backend_stop_reason,
        "generated_token_ids": token_ids,
        "completion_tokens": completion_tokens,
        "total_tokens": unit.prompt_tokens + completion_tokens,
        "raw_output": result.raw_output,
        "raw_output_sha256": _text_sha256(result.raw_output),
    }
    record["record_sha256"] = canonical_sha256(record)
    return record


def _build_meta(
    prepared: PreparedGeneration,
    records: Sequence[Mapping[str, Any]],
    determinism: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": GENERATION_RUN_SCHEMA,
        "generation_run_id": prepared.generation_run_id,
        "scope": prepared.scope,
        "scientific_eligible": prepared.scientific_eligible,
        "model_registry_id": prepared.model.registry_id,
        "model_artifact_id": prepared.model.model_artifact_id,
        "model_key": prepared.model.model_key,
        "model_role": prepared.model.role,
        "model_seed": prepared.model.seed,
        "checkpoint_format": prepared.model.checkpoint_format,
        "tokenizer_revision": prepared.model.tokenizer_revision,
        "tokenizer_content_revision": prepared.model.tokenizer_content_revision,
        "split": prepared.split,
        "sealing_status": prepared.sealing_status,
        "train_context_policy_sha256": prepared.train_context_policy_sha256,
        "evaluation_context_policy_sha256": (
            prepared.evaluation_context_policy_sha256
        ),
        "ordered_conditions": list(prepared.ordered_conditions),
        "ordered_query_ids": list(prepared.ordered_query_ids),
        "ordered_query_ids_sha256": canonical_sha256(
            list(prepared.ordered_query_ids)
        ),
        "row_traversal_policy": TRAVERSAL_POLICY,
        "attempt_policy": ATTEMPT_POLICY,
        "query_count": len(prepared.ordered_query_ids),
        "row_count": len(records),
        "complete_paired_blocks": True,
        "dependencies": copy.deepcopy(dict(prepared.dependencies)),
        "generation_profile_sha256": canonical_sha256(prepared.profile),
        "decoding_parameters_sha256": canonical_sha256(
            {
                "model_runtime": prepared.profile["model_runtime"],
                "sampling": prepared.profile["sampling"],
                "failure_policy": prepared.profile["failure_policy"],
            }
        ),
        "executor": copy.deepcopy(dict(prepared.executor_descriptor)),
        "records_sha256": canonical_sha256(list(records)),
        "record_hashes_sha256": canonical_sha256(
            [record["record_sha256"] for record in records]
        ),
        "determinism": copy.deepcopy(dict(determinism)),
        "id_inputs": copy.deepcopy(dict(prepared.id_inputs)),
    }


def _build_provenance(
    prepared: PreparedGeneration, determinism: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "schema_version": GENERATION_PROVENANCE_SCHEMA,
        "generation_run_id": prepared.generation_run_id,
        "dependencies": copy.deepcopy(dict(prepared.dependencies)),
        "model_registry_id": prepared.model.registry_id,
        "model_artifact_id": prepared.model.model_artifact_id,
        "model_key": prepared.model.model_key,
        "model_role": prepared.model.role,
        "split": prepared.split,
        "sealing_status": prepared.sealing_status,
        "train_context_policy_sha256": prepared.train_context_policy_sha256,
        "evaluation_context_policy_sha256": (
            prepared.evaluation_context_policy_sha256
        ),
        "ordered_conditions": list(prepared.ordered_conditions),
        "row_traversal_policy": TRAVERSAL_POLICY,
        "attempt_policy": ATTEMPT_POLICY,
        "max_attempts": 1,
        "fallback_generation": False,
        "sample_selection_can_change_after_failure": False,
        "executor": copy.deepcopy(dict(prepared.executor_descriptor)),
        "determinism_sha256": canonical_sha256(determinism),
        "builder_code_sha256": prepared.id_inputs["builder_code_sha256"],
    }


def _strict_write_ref(path: str | Path, locator: Mapping[str, Any]) -> None:
    destination = Path(path)
    if destination.is_symlink():
        raise GenerationLifecycleError("generation locator cannot be a symlink")
    if destination.exists():
        if load_json(destination) != dict(locator):
            raise GenerationLifecycleError(
                "generation locator is immutable and already points elsewhere"
            )
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_json_bytes(locator) + b"\n"
    try:
        descriptor = os.open(
            destination,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o644,
        )
    except FileExistsError:
        if load_json(destination) != dict(locator):
            raise GenerationLifecycleError(
                "generation locator publication raced with a different locator"
            )
        return
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def build_generation_artifact(
    *,
    profile: str | Path | Mapping[str, Any],
    context_ref: str | Path,
    model_registry_ref: str | Path,
    model_key: str,
    workspace_root: str | Path,
    target_root: str | Path,
    write_ref: str | Path,
    scope: str,
    executor: GenerationExecutor | None = None,
    sealed: bool | None = None,
    control_ref: str | Path | None = None,
    cf_ref: str | Path | None = None,
    determinism_repetitions: int = 1,
    model_resolver: RegisteredModelResolver | None = None,
    model_dependency_resolver: Callable[..., Any] | None = None,
    tokenizer_loader: Callable[[Path], Any] | None = None,
) -> dict[str, Any]:
    """Execute and atomically publish one complete generation run."""

    if scope == "formal":
        if executor is not None:
            raise GenerationLifecycleError(
                "formal publication constructs its executor internally; injected executors are forbidden"
            )
        if any(
            value is not None
            for value in (
                model_resolver,
                model_dependency_resolver,
                tokenizer_loader,
            )
        ):
            raise GenerationLifecycleError(
                "formal publication forbids injected model/tokenizer resolvers"
            )
        resolved_profile = validate_generation_profile(
            load_json(profile) if isinstance(profile, (str, Path)) else profile
        )
        backend = resolved_profile["backend"]
        bootstrap_descriptor = {
            "executor_id": (
                "hf-local-transformers/v1"
                if backend == "transformers"
                else "vllm-local-registered/v1"
            ),
            "executor_revision": (
                "stage1-generation-hf-executor/v1"
                if backend == "transformers"
                else "stage1-generation-vllm-executor/v1"
            ),
            "backend": backend,
            "scientific_eligible": True,
        }
        if backend == "vllm":
            bootstrap_descriptor["vllm_use_v1"] = "1"
        bootstrap = prepare_generation(
            profile=profile,
            context_ref=context_ref,
            control_ref=control_ref,
            cf_ref=cf_ref,
            model_registry_ref=model_registry_ref,
            model_key=model_key,
            workspace_root=workspace_root,
            scope=scope,
            executor_descriptor=bootstrap_descriptor,
            sealed=sealed,
            determinism_repetitions=determinism_repetitions,
        )
        executor = (
            LocalHFExecutor.from_registered_model(
                bootstrap.model, bootstrap.profile
            )
            if backend == "transformers"
            else LocalVLLMExecutor.from_registered_model(
                bootstrap.model, bootstrap.profile
            )
        )
        descriptor = _normalize_executor_descriptor(executor.descriptor())
        # Resolve and re-hash the registry/model trees again after loading.  The
        # second prepared frame is the only one whose ID can be published.
        prepared = prepare_generation(
            profile=profile,
            context_ref=context_ref,
            control_ref=control_ref,
            cf_ref=cf_ref,
            model_registry_ref=model_registry_ref,
            model_key=model_key,
            workspace_root=workspace_root,
            scope=scope,
            executor_descriptor=descriptor,
            sealed=sealed,
            determinism_repetitions=determinism_repetitions,
        )
    else:
        if executor is None:
            raise GenerationLifecycleError(
                "engineering publication requires an explicit executor"
            )
        descriptor = _normalize_executor_descriptor(executor.descriptor())
        prepared = prepare_generation(
            profile=profile,
            context_ref=context_ref,
            control_ref=control_ref,
            cf_ref=cf_ref,
            model_registry_ref=model_registry_ref,
            model_key=model_key,
            workspace_root=workspace_root,
            scope=scope,
            executor_descriptor=descriptor,
            sealed=sealed,
            determinism_repetitions=determinism_repetitions,
            model_resolver=model_resolver,
            tokenizer_loader=tokenizer_loader,
        )
    assert executor is not None
    expected_frame = {
        (unit.condition, unit.query_id) for unit in prepared.units
    }
    if isinstance(executor, FixtureExecutor):
        if scope != "engineering":
            raise GenerationLifecycleError("fixture execution is engineering-only")
        if determinism_repetitions != 1:
            raise GenerationLifecycleError(
                "fixture execution cannot serve as a determinism rerun"
            )
        executor.assert_exact_frame(expected_frame)

    raw_parent = Path(target_root)
    if raw_parent.is_symlink():
        raise GenerationLifecycleError("generation target root cannot be a symlink")
    parent = raw_parent.resolve()
    workspace = Path(workspace_root).resolve()
    try:
        parent.relative_to(workspace)
    except ValueError as exc:
        raise GenerationLifecycleError(
            "generation target root must live below workspace_root"
        ) from exc
    parent.mkdir(parents=True, exist_ok=True)
    target = parent / prepared.generation_run_id
    if target.exists():
        report = validate_generation_target(
            target,
            workspace_root=workspace_root,
            model_dependency_resolver=model_dependency_resolver,
            tokenizer_loader=tokenizer_loader,
            require_scientific=scope == "formal",
        )
        if report["generation_run_id"] != prepared.generation_run_id:
            raise GenerationLifecycleError("existing generation target ID mismatch")
        locator = {
            "schema_version": "stage1-locator-ref/v1",
            "artifact_kind": GENERATION_ARTIFACT_KIND,
            "artifact_id": prepared.generation_run_id,
            "target_path": str(target),
            "payload_manifest_sha256": report["payload_manifest_sha256"],
        }
        _strict_write_ref(write_ref, locator)
        return locator

    staging = new_staging_directory(parent, prepared.generation_run_id)
    try:
        executions: list[list[dict[str, Any]]] = []
        for repetition in range(determinism_repetitions):
            repetition_records: list[dict[str, Any]] = []
            for unit in prepared.units:
                try:
                    output = executor.generate(
                        messages=unit.messages,
                        query_id=unit.query_id,
                        condition=unit.condition,
                        profile=prepared.profile,
                        model=prepared.model,
                    )
                except Exception as exc:
                    raise GenerationLifecycleError(
                        f"generation repetition {repetition + 1} failed once at "
                        f"{unit.condition}/{unit.query_id}; the complete block "
                        "will not be published"
                    ) from exc
                repetition_records.append(
                    _record_from_output(prepared, unit, output)
                )
            if len(repetition_records) != len(prepared.units):
                raise GenerationLifecycleError(
                    "generation did not produce the complete frame"
                )
            executions.append(repetition_records)
        determinism = _build_determinism_document(prepared, executions)
        records = executions[0]
        write_canonical_json(staging / "generation_profile.resolved.json", prepared.profile)
        write_canonical_json(staging / "context_ref.json", prepared.dependencies["context"])
        write_canonical_json(
            staging / "model_registry_ref.json",
            prepared.dependencies["model_registry"],
        )
        if "training_plan" in prepared.dependencies:
            write_canonical_json(
                staging / "training_plan_ref.json",
                prepared.dependencies["training_plan"],
            )
        if "control" in prepared.dependencies:
            write_canonical_json(staging / "control_ref.json", prepared.dependencies["control"])
        if "counterfactual" in prepared.dependencies:
            write_canonical_json(
                staging / "cf_ref.json", prepared.dependencies["counterfactual"]
            )
        write_canonical_jsonl(
            staging / "generations.jsonl", records, key="row_ordinal", numeric_key=True
        )
        write_canonical_json(staging / "determinism.json", determinism)
        write_canonical_json(
            staging / "generation.meta.json",
            _build_meta(prepared, records, determinism),
        )
        write_canonical_json(
            staging / "provenance.json",
            _build_provenance(prepared, determinism),
        )
        payload_hash = finalize_target_atomic(
            staging,
            target,
            validate_staging=lambda path: validate_generation_target(
                path,
                workspace_root=workspace_root,
                model_dependency_resolver=model_dependency_resolver,
                tokenizer_loader=tokenizer_loader,
                require_scientific=scope == "formal",
                require_directory_name=False,
            ),
        )
    except Exception:
        if staging.exists():
            import shutil

            shutil.rmtree(staging)
        raise
    report = validate_generation_target(
        target,
        workspace_root=workspace_root,
        model_dependency_resolver=model_dependency_resolver,
        tokenizer_loader=tokenizer_loader,
        require_scientific=scope == "formal",
    )
    if report["payload_manifest_sha256"] != payload_hash:
        raise GenerationLifecycleError("published generation payload hash changed")
    locator = {
        "schema_version": "stage1-locator-ref/v1",
        "artifact_kind": GENERATION_ARTIFACT_KIND,
        "artifact_id": prepared.generation_run_id,
        "target_path": str(target),
        "payload_manifest_sha256": payload_hash,
    }
    _strict_write_ref(write_ref, locator)
    return locator


def _validate_record_hash(record: Mapping[str, Any]) -> None:
    try:
        validate_json_schema(
            record, _schema_path("stage1_generation_record_v1.schema.json")
        )
    except TrainingArtifactError as exc:
        raise GenerationLifecycleError(f"invalid generation record: {exc}") from exc
    frozen = {key: value for key, value in record.items() if key != "record_sha256"}
    if record.get("record_sha256") != canonical_sha256(frozen):
        raise GenerationLifecycleError("generation record hash mismatch")
    if record.get("content_sha256") != _text_sha256(str(record.get("content"))):
        raise GenerationLifecycleError("generation content hash mismatch")
    if record.get("gold_sha256") != context_canonical_sha256(record.get("gold")):
        raise GenerationLifecycleError("generation gold hash mismatch")
    if record.get("raw_output_sha256") != _text_sha256(str(record.get("raw_output"))):
        raise GenerationLifecycleError("generation output hash mismatch")
    token_ids = record.get("generated_token_ids")
    if (
        not isinstance(token_ids, list)
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in token_ids
        )
        or record.get("completion_tokens") != len(token_ids)
        or record.get("total_tokens")
        != record.get("prompt_tokens", 0) + len(token_ids)
    ):
        raise GenerationLifecycleError("generation token usage is internally inconsistent")
    finish_reason = record.get("finish_reason")
    expected_status = "length" if finish_reason == "length" else "ok"
    if record.get("runner_status") != expected_status:
        raise GenerationLifecycleError("generation finish reason/status mismatch")


def _resolve_embedded_dependency(
    target: Path,
    filename: str,
    *,
    workspace_root: str | Path,
    expected_kind: str | Sequence[str],
) -> tuple[dict[str, Any], Path]:
    dependency = load_json(target / filename)
    try:
        frozen = validate_dependency_ref(dependency, expected_kind=expected_kind)
        upstream = resolve_dependency_target(frozen, workspace_root)
    except TrainingArtifactError as exc:
        raise GenerationLifecycleError(
            f"invalid embedded dependency {filename}: {exc}"
        ) from exc
    return frozen, upstream


def validate_generation_target(
    target_dir: str | Path,
    *,
    workspace_root: str | Path,
    model_dependency_resolver: Callable[..., Any] | None = None,
    tokenizer_loader: Callable[[Path], Any] | None = None,
    require_scientific: bool = False,
    require_directory_name: bool = True,
) -> dict[str, Any]:
    """Read-only full replay validation of a published generation run."""

    target = Path(target_dir)
    try:
        payload_hash = validate_payload_manifest(target)
    except TrainingArtifactError as exc:
        raise GenerationLifecycleError(f"generation payload validation failed: {exc}") from exc
    meta = load_json(target / "generation.meta.json")
    profile = validate_generation_profile(
        load_json(target / "generation_profile.resolved.json")
    )
    try:
        validate_json_schema(meta, _schema_path("stage1_generation_run_v1.schema.json"))
    except TrainingArtifactError as exc:
        raise GenerationLifecycleError(f"generation meta schema failed: {exc}") from exc
    run_id = meta.get("generation_run_id")
    if not isinstance(run_id, str) or not re.fullmatch(r"gen-[0-9a-f]{64}", run_id):
        raise GenerationLifecycleError("generation run ID is invalid")
    if require_directory_name and target.name != run_id:
        raise GenerationLifecycleError("generation directory name differs from run ID")
    id_inputs = meta.get("id_inputs")
    if not isinstance(id_inputs, Mapping) or "gen-" + canonical_sha256(id_inputs) != run_id:
        raise GenerationLifecycleError("generation run ID cannot be recomputed")
    if id_inputs.get("builder_code_sha256") != sha256_file(__file__):
        raise GenerationLifecycleError(
            "generation builder code differs from the frozen run contract"
        )
    if id_inputs.get("generation_profile") != profile:
        raise GenerationLifecycleError("profile differs from generation ID inputs")
    if meta.get("generation_profile_sha256") != canonical_sha256(profile):
        raise GenerationLifecycleError("generation profile hash mismatch")
    expected_decoding_hash = canonical_sha256(
        {
            "model_runtime": profile["model_runtime"],
            "sampling": profile["sampling"],
            "failure_policy": profile["failure_policy"],
        }
    )
    if meta.get("decoding_parameters_sha256") != expected_decoding_hash:
        raise GenerationLifecycleError("decoding parameter hash mismatch")
    if tuple(meta.get("ordered_conditions", [])) != tuple(profile["ordered_conditions"]):
        raise GenerationLifecycleError("condition order differs from frozen profile")
    descriptor = _normalize_executor_descriptor(meta.get("executor", {}))
    if descriptor != id_inputs.get("executor"):
        raise GenerationLifecycleError("executor descriptor differs from generation ID inputs")
    if meta.get("scope") != id_inputs.get("scope"):
        raise GenerationLifecycleError("generation scope differs from ID inputs")
    if meta.get("row_traversal_policy") != TRAVERSAL_POLICY:
        raise GenerationLifecycleError("unsupported generation traversal policy")
    if meta.get("attempt_policy") != ATTEMPT_POLICY:
        raise GenerationLifecycleError("unsupported generation attempt policy")
    determinism_contract = id_inputs.get("determinism")
    if determinism_contract not in (
        {
            "schema_version": DETERMINISM_SCHEMA,
            "comparison_policy": DETERMINISM_POLICY,
            "execution_repetitions": 1,
        },
        {
            "schema_version": DETERMINISM_SCHEMA,
            "comparison_policy": DETERMINISM_POLICY,
            "execution_repetitions": 2,
        },
    ):
        raise GenerationLifecycleError(
            "generation determinism contract is invalid"
        )
    if meta.get("complete_paired_blocks") is not True:
        raise GenerationLifecycleError("generation target is not a complete paired block")
    scope = meta.get("scope")
    if scope not in {"formal", "engineering"}:
        raise GenerationLifecycleError("generation scope is invalid")
    if meta.get("scientific_eligible") is not (scope == "formal"):
        raise GenerationLifecycleError("generation scientific eligibility/scope mismatch")
    if require_scientific and meta.get("scientific_eligible") is not True:
        raise GenerationLifecycleError("scientific generation run is required")
    if scope == "formal" and (
        model_dependency_resolver is not None or tokenizer_loader is not None
    ):
        raise GenerationLifecycleError(
            "formal validation forbids injected model/tokenizer resolvers"
        )

    context_dependency, context_target = _resolve_embedded_dependency(
        target,
        "context_ref.json",
        workspace_root=workspace_root,
        expected_kind=("context", "test-context"),
    )
    model_dependency, _ = _resolve_embedded_dependency(
        target,
        "model_registry_ref.json",
        workspace_root=workspace_root,
        expected_kind="stage1-model-registry",
    )
    plan_dependency: dict[str, Any] | None = None
    if "training_plan" in meta.get("dependencies", {}):
        plan_dependency, _ = _resolve_embedded_dependency(
            target,
            "training_plan_ref.json",
            workspace_root=workspace_root,
            expected_kind="training-plan",
        )
    control_dependency: dict[str, Any] | None = None
    control_target: Path | None = None
    if "control" in meta.get("dependencies", {}):
        control_dependency, control_target = _resolve_embedded_dependency(
            target,
            "control_ref.json",
            workspace_root=workspace_root,
            expected_kind=("control", "test-control"),
        )
    cf_dependency: dict[str, Any] | None = None
    cf_target: Path | None = None
    if "counterfactual" in meta.get("dependencies", {}):
        cf_dependency, cf_target = _resolve_embedded_dependency(
            target,
            "cf_ref.json",
            workspace_root=workspace_root,
            expected_kind=CF_ARTIFACT_KIND,
        )
    dependencies = {
        "context": context_dependency,
        "model_registry": model_dependency,
    }
    if plan_dependency is not None:
        dependencies["training_plan"] = plan_dependency
    if control_dependency is not None:
        dependencies["control"] = control_dependency
    if cf_dependency is not None:
        dependencies["counterfactual"] = cf_dependency
    if meta.get("dependencies") != dependencies or id_inputs.get("dependencies") != dependencies:
        raise GenerationLifecycleError("generation dependency registry mismatch")
    needs_control = bool(set(meta["ordered_conditions"]) & CONTROL_CONDITIONS)
    if needs_control is not (control_dependency is not None):
        raise GenerationLifecycleError(
            "generation control dependency disagrees with its condition family"
        )

    handle = resolve_generation_model_dependency(
        model_dependency,
        model_key=str(meta["model_key"]),
        workspace_root=workspace_root,
        require_scientific=scope == "formal",
        resolver=model_dependency_resolver,
        tokenizer_loader=tokenizer_loader,
    )
    expected_model_identity = {
        "model_registry_id": handle.registry_id,
        "model_artifact_id": handle.model_artifact_id,
        "model_key": handle.model_key,
        "model_role": handle.role,
        "model_seed": handle.seed,
        "checkpoint_format": handle.checkpoint_format,
        "tokenizer_revision": handle.tokenizer_revision,
        "tokenizer_content_revision": handle.tokenizer_content_revision,
    }
    for key, expected in expected_model_identity.items():
        if meta.get(key) != expected or id_inputs.get(key) != expected:
            raise GenerationLifecycleError(
                f"registered model identity differs from generation meta: {key}"
            )
    if scope == "formal":
        expected_executor_binding = canonical_sha256(
            {
                "registry_id": handle.registry_id,
                "model_artifact_id": handle.model_artifact_id,
                "model_key": handle.model_key,
                "checkpoint_format": handle.checkpoint_format,
                "tokenizer_revision": handle.tokenizer_revision,
                "tokenizer_content_revision": handle.tokenizer_content_revision,
            }
        )
        expected_executor = {
            "transformers": (
                "hf-local-transformers/v1",
                "stage1-generation-hf-executor/v1",
            ),
            "vllm": (
                "vllm-local-registered/v1",
                "stage1-generation-vllm-executor/v1",
            ),
        }[profile["backend"]]
        if (
            descriptor.get("executor_id") != expected_executor[0]
            or descriptor.get("executor_revision") != expected_executor[1]
            or descriptor.get("model_binding_sha256")
            != expected_executor_binding
            or profile["backend"] == "vllm"
            and descriptor.get("vllm_use_v1") != "1"
        ):
            raise GenerationLifecycleError(
                "formal executor is not bound to the resolved registry model"
            )
    formal_binding: dict[str, Any] | None = None
    if scope == "formal":
        formal_binding = _validate_formal_protocol_binding(
            handle, profile, workspace_root=workspace_root
        )
        if plan_dependency != formal_binding["training_plan_dependency"]:
            raise GenerationLifecycleError(
                "generation training-plan dependency differs from its model registry"
            )
    elif plan_dependency is not None:
        raise GenerationLifecycleError(
            "engineering generation cannot claim a formal training-plan binding"
        )
    context_meta, split, context_records = _context_snapshot(
        context_target,
        tokenizer=handle.tokenizer,
        workspace_root=workspace_root,
        scientific=scope == "formal",
    )
    sealing_status = _derive_sealing_status(
        context_locator={"artifact_kind": context_dependency["artifact_kind"]},
        context_meta=context_meta,
        context_target=context_target,
        sealed=None,
        enforce_assertion=False,
    )
    _validate_model_tokenizer_lineage(context_meta, handle)
    if split != meta.get("split") or split != id_inputs.get("split"):
        raise GenerationLifecycleError("generation/context split mismatch")
    if (
        meta.get("sealing_status") != sealing_status
        or id_inputs.get("sealing_status") != sealing_status
    ):
        raise GenerationLifecycleError(
            "generation sealing status differs from derived context lineage"
        )
    expected_control_kind = (
        "test-control" if sealing_status == "sealed-test" else "control"
    )
    if control_dependency is not None and control_dependency.get(
        "artifact_kind"
    ) != expected_control_kind:
        raise GenerationLifecycleError(
            "generation control kind differs from derived sealing status"
        )
    if sealing_status == "sealed-test" and cf_dependency is None:
        raise GenerationLifecycleError(
            "sealed test generation lacks its finalized test CF dependency"
        )
    if scope == "engineering" and cf_dependency is not None:
        raise GenerationLifecycleError(
            "engineering generation claims a finalized CF dependency"
        )
    if cf_target is not None:
        try:
            cf_report = validate_cf_target(
                cf_target, workspace_root=workspace_root
            )
        except Exception as exc:
            raise GenerationLifecycleError(
                f"generation finalized CF replay failed: {exc}"
            ) from exc
        if (
            cf_report.get("split") != split
            or load_json(cf_target / "context_ref.json") != context_dependency
        ):
            raise GenerationLifecycleError(
                "generation finalized CF/context lineage mismatch"
            )
        if scope == "formal" and cf_report.get("scientific_eligible") is not True:
            raise GenerationLifecycleError(
                "formal generation finalized CF is scientifically ineligible"
            )
    if scope == "formal" and context_meta.get("scientific_eligible") is not True:
        raise GenerationLifecycleError("formal generation has engineering context")
    if scope == "formal":
        assert formal_binding is not None
        evaluation_policy, evaluation_policy_hash = context_policy_snapshot(
            context_target,
            expected_split=split,
            require_scientific=True,
            context_meta=context_meta,
        )
        if (
            evaluation_policy != formal_binding["train_context_policy"]
            or evaluation_policy_hash
            != formal_binding["train_context_policy_sha256"]
            or meta.get("train_context_policy_sha256")
            != formal_binding["train_context_policy_sha256"]
            or id_inputs.get("train_context_policy_sha256")
            != formal_binding["train_context_policy_sha256"]
            or meta.get("evaluation_context_policy_sha256")
            != evaluation_policy_hash
            or id_inputs.get("evaluation_context_policy_sha256")
            != evaluation_policy_hash
        ):
            raise GenerationLifecycleError(
                "generation context policy differs from its registry-derived plan"
            )
    elif (
        sealing_status != "unsealed-dev"
        or meta.get("train_context_policy_sha256") is not None
        or id_inputs.get("train_context_policy_sha256") is not None
        or meta.get("evaluation_context_policy_sha256") is not None
        or id_inputs.get("evaluation_context_policy_sha256") is not None
    ):
        raise GenerationLifecycleError(
            "engineering generation must remain unsealed and scientifically ineligible"
        )
    if control_target is not None:
        try:
            if scope == "formal":
                report = validate_control_target(
                    control_target,
                    context_target=context_target,
                    workspace_root=workspace_root,
                )
            else:
                report = validate_control_target(
                    control_target,
                    tokenizer=handle.tokenizer,
                    context_target=context_target,
                    workspace_root=workspace_root,
                )
        except Exception as exc:
            raise GenerationLifecycleError(f"control replay failed: {exc}") from exc
        if report.get("split") != split:
            raise GenerationLifecycleError("generation/control split mismatch")

    expected_queries = [str(row["query"]["id"]) for row in context_records]
    conditions = tuple(meta["ordered_conditions"])
    if meta.get("ordered_query_ids") != expected_queries:
        raise GenerationLifecycleError("ordered query IDs differ from context")
    if meta.get("ordered_query_ids_sha256") != canonical_sha256(expected_queries):
        raise GenerationLifecycleError("ordered query frame hash mismatch")
    condition_items: dict[str, list[dict[str, Any]]] = {}
    for condition in conditions:
        source = control_target if condition in CONTROL_CONDITIONS else context_target
        if source is None:
            raise GenerationLifecycleError(
                f"generation condition {condition} lacks its control dependency"
            )
        items = _runner_items(source, condition, split)
        if [str(item.get("id")) for item in items] != expected_queries:
            raise GenerationLifecycleError(
                f"generation condition {condition} no longer replays the paired frame"
            )
        condition_items[condition] = items
    records = load_jsonl(target / "generations.jsonl")
    if meta.get("query_count") != len(expected_queries):
        raise GenerationLifecycleError("generation query count mismatch")
    if meta.get("row_count") != len(records) or len(records) != len(expected_queries) * len(conditions):
        raise GenerationLifecycleError("generation row frame is incomplete")
    for row_ordinal, record in enumerate(records):
        _validate_record_hash(record)
        _validate_generated_output_semantics(
            raw_output=record["raw_output"],
            finish_reason=record["finish_reason"],
            generated_token_ids=record.get("generated_token_ids", []),
            backend_stop_reason=record.get("backend_stop_reason"),
            tokenizer=handle.tokenizer,
            maximum=int(profile["sampling"]["max_new_tokens"]),
            scope=scope,
            backend=profile["backend"],
        )
        query_ordinal, condition_ordinal = divmod(row_ordinal, len(conditions))
        expected_query = expected_queries[query_ordinal]
        expected_condition = conditions[condition_ordinal]
        expected_identity = {
            "generation_run_id": run_id,
            "row_ordinal": row_ordinal,
            "query_ordinal": query_ordinal,
            "condition_ordinal": condition_ordinal,
            "model_key": meta["model_key"],
            "model_role": meta["model_role"],
            "split": split,
            "condition": expected_condition,
            "query_id": expected_query,
        }
        for key, expected in expected_identity.items():
            if record.get(key) != expected:
                raise GenerationLifecycleError(
                    f"generation traversal mismatch at row {row_ordinal}: {key}"
                )
        if record.get("attempt_count") != 1:
            raise GenerationLifecycleError("generation row violates exactly-once policy")
        if scope == "formal" and record.get("finish_reason") == "fixture":
            raise GenerationLifecycleError("formal generation contains a fixture result")
        if record.get("finish_reason") == "length" and record.get(
            "completion_tokens"
        ) != profile["sampling"]["max_new_tokens"]:
            raise GenerationLifecycleError(
                "length result does not reach the frozen completion boundary"
            )
        if record.get("completion_tokens", 0) > profile["sampling"][
            "max_new_tokens"
        ]:
            raise GenerationLifecycleError(
                "generation row exceeds the frozen completion budget"
            )
        if (
            record.get("prompt_tokens", 0) > profile["model_runtime"]["max_prompt_tokens"]
            or record.get("prompt_tokens", 0)
            + profile["model_runtime"]["completion_reserve_tokens"]
            > profile["model_runtime"]["max_model_len"]
        ):
            raise GenerationLifecycleError("generation row exceeds the frozen token budget")
        runner = condition_items[expected_condition][query_ordinal]
        prompt_hash, prompt_tokens, context_hash, control_hash = _runner_prompt_lineage(
            runner, expected_condition
        )
        expected_lineage = {
            "content": runner.get("content"),
            "gold": runner.get("gt_quadruples"),
            "context_record_sha256": context_hash,
            "control_record_sha256": control_hash,
            "prompt_sha256": prompt_hash,
            "prompt_tokens": prompt_tokens,
        }
        for key, expected in expected_lineage.items():
            if record.get(key) != expected:
                raise GenerationLifecycleError(
                    f"generation row differs from frozen runner at row {row_ordinal}: {key}"
                )
    if meta.get("records_sha256") != canonical_sha256(records):
        raise GenerationLifecycleError("generation record collection hash mismatch")
    if meta.get("record_hashes_sha256") != canonical_sha256(
        [record["record_sha256"] for record in records]
    ):
        raise GenerationLifecycleError("generation record hash registry mismatch")
    determinism = load_json(target / "determinism.json")
    expected_repetitions = determinism_contract["execution_repetitions"]
    expected_records_hash = canonical_sha256(records)
    expected_record_hashes_hash = canonical_sha256(
        [record["record_sha256"] for record in records]
    )
    expected_raw_hash = _raw_output_frame_sha256(records)
    expected_determinism = {
        "schema_version": DETERMINISM_SCHEMA,
        "generation_run_id": run_id,
        "comparison_policy": DETERMINISM_POLICY,
        "execution_repetitions": expected_repetitions,
        "exact_match": True,
        "mismatch_count": 0,
        "per_execution_records_sha256": [
            expected_records_hash
        ] * expected_repetitions,
        "per_execution_record_hashes_sha256": [
            expected_record_hashes_hash
        ] * expected_repetitions,
        "per_execution_raw_output_bytes_sha256": [
            expected_raw_hash
        ] * expected_repetitions,
    }
    if determinism != expected_determinism or meta.get("determinism") != determinism:
        raise GenerationLifecycleError(
            "generation determinism receipt cannot be replayed"
        )
    provenance = load_json(target / "provenance.json")
    expected_provenance = {
        "schema_version": GENERATION_PROVENANCE_SCHEMA,
        "generation_run_id": run_id,
        "dependencies": dependencies,
        "model_registry_id": meta["model_registry_id"],
        "model_artifact_id": meta["model_artifact_id"],
        "model_key": meta["model_key"],
        "model_role": meta["model_role"],
        "split": split,
        "sealing_status": sealing_status,
        "train_context_policy_sha256": meta[
            "train_context_policy_sha256"
        ],
        "evaluation_context_policy_sha256": meta[
            "evaluation_context_policy_sha256"
        ],
        "ordered_conditions": list(conditions),
        "row_traversal_policy": TRAVERSAL_POLICY,
        "attempt_policy": ATTEMPT_POLICY,
        "max_attempts": 1,
        "fallback_generation": False,
        "sample_selection_can_change_after_failure": False,
        "executor": meta["executor"],
        "determinism_sha256": canonical_sha256(determinism),
        "builder_code_sha256": id_inputs["builder_code_sha256"],
    }
    if provenance != expected_provenance:
        raise GenerationLifecycleError("generation provenance mismatch")
    if scope == "formal":
        if tuple(conditions) != FORMAL_CONDITIONS:
            raise GenerationLifecycleError("formal run lacks the complete condition family")
        if meta["executor"].get("scientific_eligible") is not True:
            raise GenerationLifecycleError("formal run used an engineering executor")
        if meta["executor"].get("backend") != profile["backend"]:
            raise GenerationLifecycleError("formal executor/profile backend mismatch")

    expected_files = {
        "generation_profile.resolved.json",
        "context_ref.json",
        "model_registry_ref.json",
        "generations.jsonl",
        "determinism.json",
        "generation.meta.json",
        "provenance.json",
        "payload_manifest.json",
    }
    if control_dependency is not None:
        expected_files.add("control_ref.json")
    if cf_dependency is not None:
        expected_files.add("cf_ref.json")
    if plan_dependency is not None:
        expected_files.add("training_plan_ref.json")
    try:
        ensure_exact_file_set(target, expected_files)
    except TrainingArtifactError as exc:
        raise GenerationLifecycleError(str(exc)) from exc
    return {
        "schema_version": GENERATION_VALIDATION_SCHEMA,
        "valid": True,
        "generation_run_id": run_id,
        "scope": scope,
        "scientific_eligible": scope == "formal",
        "model_key": meta["model_key"],
        "model_role": meta["model_role"],
        "split": split,
        "sealing_status": sealing_status,
        "ordered_conditions": list(conditions),
        "query_count": len(expected_queries),
        "row_count": len(records),
        "complete_paired_blocks": True,
        "execution_repetitions": expected_repetitions,
        "exact_rerun_match": determinism["exact_match"],
        "executor_backend": descriptor["backend"],
        "executor_id": descriptor["executor_id"],
        "payload_manifest_sha256": payload_hash,
    }


def validate_generation_ref(
    ref_path: str | Path,
    *,
    workspace_root: str | Path,
    model_dependency_resolver: Callable[..., Any] | None = None,
    tokenizer_loader: Callable[[Path], Any] | None = None,
    require_scientific: bool = False,
) -> dict[str, Any]:
    try:
        locator, target = resolve_locator_ref(
            ref_path, expected_kind=GENERATION_ARTIFACT_KIND
        )
    except TrainingArtifactError as exc:
        raise GenerationLifecycleError(f"invalid generation locator: {exc}") from exc
    report = validate_generation_target(
        target,
        workspace_root=workspace_root,
        model_dependency_resolver=model_dependency_resolver,
        tokenizer_loader=tokenizer_loader,
        require_scientific=require_scientific,
    )
    if report["generation_run_id"] != locator["artifact_id"]:
        raise GenerationLifecycleError("generation locator ID mismatch")
    if report["payload_manifest_sha256"] != locator["payload_manifest_sha256"]:
        raise GenerationLifecycleError("generation locator payload hash mismatch")
    return report


__all__ = [
    "ATTEMPT_POLICY",
    "FORMAL_CONDITIONS",
    "GENERATION_ARTIFACT_KIND",
    "GENERATION_RECORD_SCHEMA",
    "GENERATION_RUN_SCHEMA",
    "MODEL_RESOLUTION_SCHEMA",
    "TRAVERSAL_POLICY",
    "FixtureExecutor",
    "GenerationExecutor",
    "GenerationResult",
    "GenerationLifecycleError",
    "LocalHFExecutor",
    "LocalVLLMExecutor",
    "PreparedGeneration",
    "RegisteredModelHandle",
    "RegisteredModelResolver",
    "build_generation_artifact",
    "preflight_generation",
    "prepare_generation",
    "resolve_generation_model",
    "verified_registered_model_source_load",
    "validate_generation_profile",
    "validate_generation_ref",
    "validate_generation_target",
]
