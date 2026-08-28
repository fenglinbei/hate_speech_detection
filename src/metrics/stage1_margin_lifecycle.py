"""Immutable local-HF lifecycle for Stage-1 teacher-forced field margins.

The lifecycle deliberately separates prediction-before-scoring preparation
from model execution.  Registry/context/control/CF targets are fully resolved,
the exact query/tuple/field frame and eligibility masks are frozen, and every
gold/counterfactual span is tokenized before the model is loaded.  Only then is
the registered local checkpoint loaded and scored.  No path-based model input,
retry, truncation, or partial target publication is supported.
"""

from __future__ import annotations

import copy
import importlib.metadata
import math
import shutil
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

from data.build_context_manifest import validate_context_target
from data.context_manifest import chat_prompt_text, text_sha256, token_count
from data.control_manifest import validate_control_target
from data.counterfactual_lifecycle import validate_cf_target
from data.generation_lifecycle import (
    GenerationLifecycleError,
    RegisteredModelHandle,
    resolve_generation_model,
    verified_registered_model_source_load,
)
from data.training_artifacts import (
    TrainingArtifactError,
    canonical_sha256,
    finalize_target_atomic,
    load_json,
    load_jsonl,
    new_staging_directory,
    portable_dependency,
    resolve_locator_ref,
    sha256_file,
    write_canonical_json,
    write_locator_ref,
)
from metrics.stage1_artifacts import (
    CONDITIONS,
    FIELDS,
    MARGIN_KIND,
    Stage1ArtifactError,
    _assert_target_unchanged,
    _control_kind_for_split_sealing,
    _context_frame,
    _ordered_id_hash,
    _scope_matrix,
    _write_ordered_jsonl,
    resolve_margin_profile,
    validate_margin_target,
    validate_registry_ref,
)
from metrics.stage1_margin import (
    SEGMENTATION_VERSION,
    SPAN_MASK_VERSION,
    ScoreInput,
    prepare_field_pair,
    score_inputs,
)
from model.stage1_registry import (
    ModelRegistryError,
    resolve_registered_model_dependency,
)
from utils.quadruple import canonicalize_quadruples


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
TUPLE_SCORE_SCHEMA = "stage1-margin-tuple-score/v1"
CONTRAST_SCHEMA = "stage1-margin-query-contrasts/v1"
CALIBRATION_SCHEMA = "stage1-margin-batch-calibration/v1"
CALIBRATION_RECORD_SCHEMA = "stage1-margin-batch-calibration-record/v1"
CALIBRATION_TRAVERSAL = "condition-query-tuple-field-first-100/v1"
CALIBRATION_PAIR_TARGET = 100
CALIBRATION_TOLERANCE_FLOOR = 1e-4
CALIBRATION_TOLERANCE_MULTIPLIER = 2.0
CALIBRATION_TOLERANCE_HARD_CAP = 5e-3
BATCH_CONTRACT_SCHEMA = "stage1-margin-batch-contract/v1"
BATCH_UNIT = "gold-cf-pair"
BATCH_SIZE = 1
BATCH_SEQUENCES_PER_UNIT = 2
LOCAL_EXECUTOR_DESCRIPTOR = {
    "executor_id": "hf-local-transformers/v1",
    "executor_revision": "stage1-margin-hf-executor/v1",
    "backend": "local-huggingface-causal-lm",
    "scientific_eligible": True,
}


class MarginLifecycleError(Stage1ArtifactError):
    """Raised when a scorer input or immutable margin artifact is invalid."""


class MarginExecutor(Protocol):
    tokenizer: Any

    def descriptor(self) -> Mapping[str, Any]: ...

    def score(self, inputs: Sequence[ScoreInput]) -> list[dict[str, Any]]: ...


class LocalHFMarginExecutor:
    """Reference local-HF scorer loaded only from a registered model handle."""

    def __init__(self, *, model: Any, tokenizer: Any, pad_token_id: int):
        self.model = model
        self.tokenizer = tokenizer
        self.pad_token_id = int(pad_token_id)

    @classmethod
    def from_registered_model(
        cls,
        handle: RegisteredModelHandle,
        profile: Mapping[str, Any],
    ) -> "LocalHFMarginExecutor":
        runtime = profile["runtime"]
        if (
            handle.checkpoint_path is None
            or handle.base_model_path is None
            or handle.tokenizer_path is None
        ):
            raise MarginLifecycleError(
                "registered model lacks checkpoint/base/tokenizer load roots"
            )
        for path in (
            handle.checkpoint_path,
            handle.base_model_path,
            handle.tokenizer_path,
        ):
            if not path.is_dir() or path.is_symlink():
                raise MarginLifecycleError(
                    "registered model/tokenizer root is missing or a symlink"
                )
        try:
            import torch
            from transformers import AutoModelForCausalLM
        except ImportError as exc:  # pragma: no cover - environment contract
            raise MarginLifecycleError(
                "torch and transformers are required for local-HF margin scoring"
            ) from exc
        dtype = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }[runtime["dtype"]]
        try:
            source_names = (
                ("checkpoint", "base")
                if handle.checkpoint_format == "adapter"
                else ("checkpoint",)
            )
            with verified_registered_model_source_load(
                source_contract=handle.source_contract,
                scientific_eligible=handle.scientific_eligible,
                checkpoint_path=handle.checkpoint_path,
                tokenizer_path=handle.tokenizer_path,
                base_model_path=handle.base_model_path,
                source_names=source_names,
            ) as sources:
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
                    device_map=runtime["device_map"],
                )
                if handle.checkpoint_format == "adapter":
                    try:
                        from peft import PeftModel
                    except ImportError as exc:  # pragma: no cover - environment contract
                        raise MarginLifecycleError(
                            "peft is required for a registered adapter checkpoint"
                        ) from exc
                    model = PeftModel.from_pretrained(
                        model,
                        str(sources.checkpoint_path),
                        local_files_only=True,
                        is_trainable=False,
                    )
                model.eval()
        except MarginLifecycleError:
            raise
        except GenerationLifecycleError as exc:
            raise MarginLifecycleError(str(exc)) from exc
        except Exception as exc:
            raise MarginLifecycleError(
                f"registered local-HF model load failed: {exc}"
            ) from exc
        pad = getattr(handle.tokenizer, "pad_token_id", None)
        if pad is None:
            pad = getattr(handle.tokenizer, "eos_token_id", None)
        if pad is None:
            raise MarginLifecycleError(
                "registered tokenizer defines neither pad_token_id nor eos_token_id"
            )
        return cls(model=model, tokenizer=handle.tokenizer, pad_token_id=int(pad))

    def descriptor(self) -> Mapping[str, Any]:
        return copy.deepcopy(LOCAL_EXECUTOR_DESCRIPTOR)

    def score(self, inputs: Sequence[ScoreInput]) -> list[dict[str, Any]]:
        return score_inputs(
            model=self.model,
            score_inputs=inputs,
            pad_token_id=self.pad_token_id,
        )


def _version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _batch_contract(profile: Mapping[str, Any]) -> dict[str, Any]:
    """Resolve the only unambiguous scoring-batch interpretation.

    ``batch_size`` counts gold/counterfactual pairs, not individual token
    sequences.  A normal scoring call therefore contains exactly two
    sequences in the frozen gold-then-counterfactual order.  Calibration
    compares that call with two one-sequence calls.
    """

    runtime = profile.get("runtime")
    if not isinstance(runtime, Mapping) or runtime.get("batch_unit") != BATCH_UNIT:
        raise MarginLifecycleError(
            "margin batch_size is ambiguous without runtime.batch_unit=gold-cf-pair"
        )
    if runtime.get("batch_size") != BATCH_SIZE:
        raise MarginLifecycleError("margin scorer requires exactly one gold/CF pair per batch")
    return {
        "schema_version": BATCH_CONTRACT_SCHEMA,
        "batch_unit": BATCH_UNIT,
        "batch_size": BATCH_SIZE,
        "sequences_per_unit": BATCH_SEQUENCES_PER_UNIT,
        "pair_member_order": ["gold", "counterfactual"],
        "calibration_comparison": "one-pair-vs-two-singletons",
    }


def _runtime_contract(
    executor: Mapping[str, Any], profile: Mapping[str, Any]
) -> dict[str, Any]:
    """Capture deterministic runtime identity without host/device identifiers."""

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - environment contract
        raise MarginLifecycleError("torch is required for the margin runtime contract") from exc
    cuda_available = bool(torch.cuda.is_available())
    architectures: list[str] = []
    driver_version: int | None = None
    if cuda_available:
        architectures = sorted(
            {
                f"sm_{major}{minor}"
                for major, minor in (
                    torch.cuda.get_device_capability(index)
                    for index in range(torch.cuda.device_count())
                )
            }
        )
        for candidate in (
            getattr(torch.cuda, "driver_version", None),
            getattr(getattr(torch, "_C", object()), "_cuda_getDriverVersion", None),
        ):
            if callable(candidate):
                try:
                    driver_version = int(candidate())
                    break
                except Exception:
                    continue
    return {
        "schema_version": "stage1-margin-runtime-contract/v1",
        "executor": copy.deepcopy(dict(executor)),
        "python_implementation": __import__("platform").python_implementation(),
        "python_version": __import__("platform").python_version(),
        "torch_version": str(torch.__version__),
        "transformers_version": _version("transformers"),
        "peft_version": _version("peft"),
        "accelerate_version": _version("accelerate"),
        "cuda_runtime_version": str(torch.version.cuda) if torch.version.cuda else None,
        "cuda_driver_version": driver_version,
        "gpu_architectures": architectures,
        "batch_contract": _batch_contract(profile),
        "numeric_contract": {
            "log_softmax_dtype": "float32",
            "right_padding": True,
            "position_ids": "attention-mask-cumsum-minus-one-zero-on-padding/v1",
            "causal_shift": "logits-minus-last-vs-input-minus-first",
        },
    }


def _scorer_code_sha256() -> str:
    return canonical_sha256(
        {
            "lifecycle_sha256": sha256_file(__file__),
            "scorer_sha256": sha256_file(Path(__file__).with_name("stage1_margin.py")),
            "serializer_sha256": sha256_file(
                REPOSITORY_ROOT / "src/utils/quadruple.py"
            ),
        }
    )


def _contract_hashes(
    *,
    profile: Mapping[str, Any],
    handle: RegisteredModelHandle,
    runtime_contract: Mapping[str, Any],
) -> dict[str, str]:
    return {
        "serializer_schema_sha256": canonical_sha256(
            {
                "wire_schema": "canonical-quad-json/v1",
                "complete_value_literal_spans": True,
                "serializer_source_sha256": sha256_file(
                    REPOSITORY_ROOT / "src/utils/quadruple.py"
                ),
            }
        ),
        "tokenizer_contract_sha256": canonical_sha256(
            {
                "profile": profile["tokenization"],
                "tokenizer_revision": handle.tokenizer_revision,
                "tokenizer_content_revision": handle.tokenizer_content_revision,
            }
        ),
        "span_mask_contract_sha256": canonical_sha256(
            {
                "span_mask_version": SPAN_MASK_VERSION,
                "segmentation_version": SEGMENTATION_VERSION,
            }
        ),
        "aggregation_contract_sha256": canonical_sha256(profile["aggregation"]),
        "runtime_contract_sha256": canonical_sha256(runtime_contract),
        "batch_calibration_contract_sha256": canonical_sha256(
            {
                "schema_version": CALIBRATION_SCHEMA,
                "record_schema_version": CALIBRATION_RECORD_SCHEMA,
                "comparison": "batched-vs-singleton-field-margin-absolute-delta",
                "traversal_policy": CALIBRATION_TRAVERSAL,
                "requested_pair_count": CALIBRATION_PAIR_TARGET,
                "tolerance_floor": CALIBRATION_TOLERANCE_FLOOR,
                "tolerance_multiplier": CALIBRATION_TOLERANCE_MULTIPLIER,
                "tolerance_hard_cap": CALIBRATION_TOLERANCE_HARD_CAP,
                "batch_contract": _batch_contract(profile),
                "require_token_ids_exact": True,
                "require_score_shapes_exact": True,
            }
        ),
        "scorer_code_sha256": _scorer_code_sha256(),
    }


def _runner_items(target: Path, condition: str, split: str) -> list[dict[str, Any]]:
    raw = load_json(target / "conditions" / "runner" / condition / f"{split}.json")
    if not isinstance(raw, list) or any(not isinstance(row, dict) for row in raw):
        raise MarginLifecycleError(f"invalid runner adapter for {condition}/{split}")
    return [dict(row) for row in raw]


def _messages(item: Mapping[str, Any], *, condition: str, query_id: str) -> list[dict[str, str]]:
    messages_list = item.get("messages_list")
    if (
        not isinstance(messages_list, list)
        or len(messages_list) != 1
        or not isinstance(messages_list[0], list)
        or len(messages_list[0]) != 2
    ):
        raise MarginLifecycleError(
            f"{condition}/{query_id} must contain one canonical system+user prompt"
        )
    messages: list[dict[str, str]] = []
    for expected_role, raw in zip(("system", "user"), messages_list[0], strict=True):
        if (
            not isinstance(raw, Mapping)
            or set(raw) != {"role", "content"}
            or raw.get("role") != expected_role
            or not isinstance(raw.get("content"), str)
            or not raw["content"]
        ):
            raise MarginLifecycleError(
                f"{condition}/{query_id} has non-canonical prompt messages"
            )
        messages.append({"role": expected_role, "content": raw["content"]})
    return messages


def _prompt(
    item: Mapping[str, Any], *, condition: str, query_id: str, tokenizer: Any
) -> str:
    rendered = chat_prompt_text(
        _messages(item, condition=condition, query_id=query_id), tokenizer
    )
    lineage = item.get("control_manifest") if condition in {"PL", "PD"} else item.get(
        "context_manifest"
    )
    if not isinstance(lineage, Mapping):
        raise MarginLifecycleError(f"{condition}/{query_id} lacks prompt lineage")
    if lineage.get("chat_prompt_sha256") != text_sha256(rendered) or lineage.get(
        "chat_prompt_tokens"
    ) != token_count(rendered, tokenizer):
        raise MarginLifecycleError(
            f"{condition}/{query_id} rendered prompt differs from frozen adapter"
        )
    return rendered


def _selected_candidate(row: Mapping[str, Any]) -> Any:
    selected = row.get("selected_cf_id")
    candidates = row.get("candidates")
    if not isinstance(candidates, list):
        raise MarginLifecycleError("CF row lacks candidate list")
    matches = [candidate for candidate in candidates if candidate.get("candidate_id") == selected]
    if row.get("construction_status") == "ok":
        if len(matches) != 1:
            raise MarginLifecycleError("constructed CF row lacks one selected candidate")
        return copy.deepcopy(matches[0]["value"])
    if selected is not None or matches:
        raise MarginLifecycleError("ineligible CF row unexpectedly selects a candidate")
    return None


def _same_dependency_identity(
    left: Mapping[str, Any], right: Mapping[str, Any]
) -> bool:
    """Compare immutable identity while tolerating equivalent portable roots.

    The control lifecycle predates the shared workspace-relative dependency
    helper and stores its context under the canonical ``contexts/<id>`` name.
    The artifact ID, kind, and payload hash are the authoritative binding; the
    margin target itself still stores a resolvable workspace-relative ref.
    """

    keys = ("schema_version", "artifact_kind", "artifact_id", "payload_manifest_sha256")
    return all(left.get(key) == right.get(key) for key in keys)


def _cf_frame(
    *,
    cf_target: Path,
    split: str,
    context_frame: Sequence[Mapping[str, Any]],
) -> tuple[
    dict[tuple[str, int, str], dict[str, Any]],
    dict[str, list[bool]],
    dict[str, int],
]:
    rows = load_jsonl(cf_target / f"cf_manifest.{split}.jsonl")
    actual: dict[tuple[str, int, str], dict[str, Any]] = {}
    expected: list[tuple[str, int, str]] = []
    tuple_counts: dict[str, int] = {}
    frame_by_id = {str(row["id"]): row for row in context_frame}
    for query in context_frame:
        query_id = str(query["id"])
        gold = canonicalize_quadruples(query["gold"])
        tuple_counts[query_id] = len(gold)
        for tuple_index in range(len(gold)):
            for field in FIELDS:
                expected.append((query_id, tuple_index, field))
    for raw in rows:
        key = (str(raw.get("query_id", "")), int(raw.get("tuple_index", -1)), str(raw.get("field", "")))
        if key in actual:
            raise MarginLifecycleError(f"duplicate CF unit: {key}")
        actual[key] = dict(raw)
    if set(actual) != set(expected):
        raise MarginLifecycleError("CF units do not exactly cover every gold tuple/field")
    for query_id, tuple_index, field in expected:
        row = actual[(query_id, tuple_index, field)]
        frozen = frame_by_id[query_id]
        if (
            row.get("context_record_sha256") != frozen["context_record_sha256"]
            or row.get("gold_sha256") != frozen["gold_sha256"]
        ):
            raise MarginLifecycleError("CF/context hash lineage differs")
        gold = canonicalize_quadruples(frozen["gold"])
        expected_gold: Any = getattr(gold[tuple_index], field)
        if field == "targeted_group":
            expected_gold = list(expected_gold)
        if row.get("gold_value") != expected_gold:
            raise MarginLifecycleError("CF gold value differs from canonical query gold")
        _selected_candidate(row)
    masks = {
        field: [
            all(
                actual[(str(query["id"]), tuple_index, field)].get(
                    "construction_status"
                )
                == "ok"
                for tuple_index in range(tuple_counts[str(query["id"])])
            )
            for query in context_frame
        ]
        for field in FIELDS
    }
    for field in ("targeted_group", "hateful"):
        if not all(masks[field]):
            raise MarginLifecycleError(
                "group/hate CF construction coverage must be exactly 100%"
            )
    return actual, masks, tuple_counts


def _prepare_pair(
    *,
    tokenizer: Any,
    prompt: str,
    gold: Sequence[Mapping[str, Any]],
    tuple_index: int,
    field: str,
    cf_row: Mapping[str, Any],
    max_sequence_tokens: int,
) -> tuple[ScoreInput, ScoreInput]:
    try:
        return prepare_field_pair(
            tokenizer=tokenizer,
            rendered_chat_prompt=prompt,
            gold=gold,
            tuple_index=tuple_index,
            field=field,
            candidate_value=_selected_candidate(cf_row),
            max_sequence_tokens=max_sequence_tokens,
        )
    except OverflowError as exc:
        raise MarginLifecycleError(
            "margin token preflight overflowed; truncation and post-hoc exclusion are forbidden"
        ) from exc
    except (ValueError, IndexError, TypeError) as exc:
        raise MarginLifecycleError(f"margin score-input preparation failed: {exc}") from exc


def _score_projection(score: Mapping[str, Any]) -> dict[str, Any]:
    response_indices = score.get("response_token_indices")
    if not isinstance(response_indices, list) or not response_indices:
        raise MarginLifecycleError("field score lacks response token indices")
    projected = {
        "token_ids": [int(value) for value in score["token_ids"]],
        "token_count": int(score["token_count"]),
        "character_span": [int(value) for value in score["character_span"]],
        "token_span": [int(response_indices[0]), int(response_indices[-1]) + 1],
        "sum_logprob": float(score["sum_logprob"]),
        "mean_logprob": float(score["mean_logprob"]),
        "left_boundary_crossing": bool(score["left_boundary_crossing"]),
        "right_boundary_crossing": bool(score["right_boundary_crossing"]),
    }
    if projected["token_count"] != len(projected["token_ids"]) or any(
        not math.isfinite(projected[key]) for key in ("sum_logprob", "mean_logprob")
    ):
        raise MarginLifecycleError("field score audit projection is invalid")
    return projected


def _batch_calibration(
    executor: MarginExecutor,
    entries: Sequence[
        tuple[tuple[ScoreInput, ScoreInput], Mapping[str, Any]]
    ],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not entries or len(entries) > CALIBRATION_PAIR_TARGET:
        raise MarginLifecycleError("batch calibration frame has an invalid size")
    rows: list[dict[str, Any]] = []
    for pair, identity in entries:
        batched = executor.score(pair)
        if len(batched) != 2:
            raise MarginLifecycleError(
                "margin executor did not return a two-score calibration pair"
            )
        singleton_batches = [executor.score([item]) for item in pair]
        if any(len(batch) != 1 for batch in singleton_batches):
            raise MarginLifecycleError(
                "margin executor did not return one singleton calibration score"
            )
        singles = [batch[0] for batch in singleton_batches]
        compared_tokens = 0
        token_ids: list[list[int]] = []
        for batched_score, singleton_score in zip(batched, singles, strict=True):
            batch_ids = batched_score.get("token_ids")
            singleton_ids = singleton_score.get("token_ids")
            batch_values = batched_score.get("token_logprobs")
            singleton_values = singleton_score.get("token_logprobs")
            if (
                not isinstance(batch_ids, list)
                or not batch_ids
                or batch_ids != singleton_ids
                or not isinstance(batch_values, list)
                or not isinstance(singleton_values, list)
                or len(batch_values) != len(singleton_values)
                or len(batch_ids) != len(batch_values)
            ):
                raise MarginLifecycleError(
                    "batch calibration token IDs or score shapes differ"
                )
            if any(
                not math.isfinite(float(value))
                for value in (*batch_values, *singleton_values)
            ):
                raise MarginLifecycleError("batch calibration has non-finite scores")
            compared_tokens += len(batch_ids)
            token_ids.append([int(value) for value in batch_ids])
        batched_gold = float(batched[0]["mean_logprob"])
        batched_cf = float(batched[1]["mean_logprob"])
        singleton_gold = float(singles[0]["mean_logprob"])
        singleton_cf = float(singles[1]["mean_logprob"])
        if any(
            not math.isfinite(value)
            for value in (batched_gold, batched_cf, singleton_gold, singleton_cf)
        ):
            raise MarginLifecycleError("batch calibration has a non-finite mean")
        batched_margin = batched_gold - batched_cf
        singleton_margin = singleton_gold - singleton_cf
        delta = abs(batched_margin - singleton_margin)
        rows.append(
            {
                "schema_version": CALIBRATION_RECORD_SCHEMA,
                **copy.deepcopy(dict(identity)),
                "gold_token_ids": token_ids[0],
                "counterfactual_token_ids": token_ids[1],
                "batched_gold_mean_logprob": batched_gold,
                "batched_counterfactual_mean_logprob": batched_cf,
                "singleton_gold_mean_logprob": singleton_gold,
                "singleton_counterfactual_mean_logprob": singleton_cf,
                "batched_margin": batched_margin,
                "singleton_margin": singleton_margin,
                "abs_margin_delta": delta,
                "compared_token_count": compared_tokens,
            }
        )
    maximum = max(float(row["abs_margin_delta"]) for row in rows)
    frozen_tolerance = max(
        CALIBRATION_TOLERANCE_FLOOR,
        CALIBRATION_TOLERANCE_MULTIPLIER * maximum,
    )
    if not math.isfinite(frozen_tolerance) or frozen_tolerance > (
        CALIBRATION_TOLERANCE_HARD_CAP
    ):
        raise MarginLifecycleError(
            "batched/unbatched margin calibration requires a tolerance above "
            f"5e-3: observed_max_abs_margin_delta={maximum:.17g}, "
            f"required_tolerance={frozen_tolerance:.17g}"
        )
    return {
        "schema_version": CALIBRATION_SCHEMA,
        "passed": True,
        "traversal_policy": CALIBRATION_TRAVERSAL,
        "requested_pair_count": CALIBRATION_PAIR_TARGET,
        "observed_pair_count": len(rows),
        "requested_pair_count_reached": len(rows) == CALIBRATION_PAIR_TARGET,
        "calibration_records_sha256": canonical_sha256(rows),
        "tolerance_floor": CALIBRATION_TOLERANCE_FLOOR,
        "tolerance_multiplier": CALIBRATION_TOLERANCE_MULTIPLIER,
        "tolerance_hard_cap": CALIBRATION_TOLERANCE_HARD_CAP,
        "observed_max_abs_margin_delta": maximum,
        "frozen_tolerance": frozen_tolerance,
        "compared_token_count": sum(
            int(row["compared_token_count"]) for row in rows
        ),
    }, rows


def _tuple_score_row(
    *,
    query: Mapping[str, Any],
    condition: str,
    tuple_index: int,
    field: str,
    model_key: str,
    cf_row: Mapping[str, Any],
    gold_score: Mapping[str, Any],
    foil_score: Mapping[str, Any],
) -> dict[str, Any]:
    mean_margin = float(gold_score["mean_logprob"]) - float(
        foil_score["mean_logprob"]
    )
    sum_margin = float(gold_score["sum_logprob"]) - float(foil_score["sum_logprob"])
    if not math.isfinite(mean_margin) or not math.isfinite(sum_margin):
        raise MarginLifecycleError("margin score is non-finite")
    return {
        "schema_version": TUPLE_SCORE_SCHEMA,
        "id": str(query["id"]),
        "condition": condition,
        "tuple_index": tuple_index,
        "field": field,
        "model_key": model_key,
        "status": "ok",
        "selected_cf_id": cf_row["selected_cf_id"],
        "gold": _score_projection(gold_score),
        "counterfactual": _score_projection(foil_score),
        "mean_margin": mean_margin,
        "sum_margin_sensitivity": sum_margin,
        "content_sha256": query["content_sha256"],
        "gold_sha256": query["gold_sha256"],
        "context_record_sha256": query["context_record_sha256"],
        "cf_record_sha256": cf_row["record_sha256"],
    }


def _aggregate_rows(
    *,
    condition: str,
    model_key: str,
    context_frame: Sequence[Mapping[str, Any]],
    masks: Mapping[str, Sequence[bool]],
    tuple_scores: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    grouped: defaultdict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in tuple_scores:
        grouped[(str(row["id"]), str(row["field"]))].append(row)
    result = []
    for query_index, query in enumerate(context_frame):
        query_id = str(query["id"])
        for field in FIELDS:
            eligible = bool(masks[field][query_index])
            rows = grouped[(query_id, field)]
            if eligible and not rows:
                raise MarginLifecycleError("eligible query/field has no tuple scores")
            if not eligible and rows:
                raise MarginLifecycleError("ineligible query/field has tuple scores")
            result.append(
                {
                    "schema_version": "stage1-margin-record/v1",
                    "id": query_id,
                    "condition": condition,
                    "field": field,
                    "model_key": model_key,
                    "status": "ok" if eligible else "ineligible-pre-frozen",
                    "margin_mean": (
                        sum(float(row["mean_margin"]) for row in rows) / len(rows)
                        if eligible
                        else None
                    ),
                    "content_sha256": query["content_sha256"],
                    "gold_sha256": query["gold_sha256"],
                    "context_record_sha256": query["context_record_sha256"],
                    "cf_record_sha256": (
                        canonical_sha256([row["cf_record_sha256"] for row in rows])
                        if eligible
                        else None
                    ),
                }
            )
    return result


def _contrast_rows(
    *,
    model_key: str,
    context_frame: Sequence[Mapping[str, Any]],
    masks: Mapping[str, Sequence[bool]],
    rows_by_condition: Mapping[str, Sequence[Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    values = {
        condition: {
            (str(row["id"]), str(row["field"])): row
            for row in rows_by_condition[condition]
        }
        for condition in CONDITIONS
    }
    result = []
    for query_index, query in enumerate(context_frame):
        query_id = str(query["id"])
        for field in FIELDS:
            eligible = bool(masks[field][query_index])
            cell = {
                condition: values[condition][(query_id, field)]["margin_mean"]
                for condition in CONDITIONS
            }
            if eligible:
                numeric = {key: float(value) for key, value in cell.items()}
                contrasts = {
                    "TE_L": numeric["CL"] - numeric["C0"],
                    "TE_D": numeric["CD"] - numeric["C0"],
                    "I_L_D": numeric["CLD"] - numeric["CL"] - numeric["CD"] + numeric["C0"],
                    "TE_L_rel": numeric["CL"] - numeric["PL"],
                    "TE_D_rel": numeric["CD"] - numeric["PD"],
                    "L_shape": numeric["PL"] - numeric["C0"],
                    "D_shape": numeric["PD"] - numeric["C0"],
                }
            else:
                if any(value is not None for value in cell.values()):
                    raise MarginLifecycleError("ineligible contrast has a scored condition")
                contrasts = {
                    key: None
                    for key in (
                        "TE_L", "TE_D", "I_L_D", "TE_L_rel", "TE_D_rel",
                        "L_shape", "D_shape",
                    )
                }
            result.append(
                {
                    "schema_version": CONTRAST_SCHEMA,
                    "id": query_id,
                    "field": field,
                    "model_key": model_key,
                    "status": "ok" if eligible else "ineligible-pre-frozen",
                    **contrasts,
                    "content_sha256": query["content_sha256"],
                    "gold_sha256": query["gold_sha256"],
                    "context_record_sha256": query["context_record_sha256"],
                }
            )
    return result


def _coverage(
    masks: Mapping[str, Sequence[bool]],
    *,
    context_frame: Sequence[Mapping[str, Any]],
    tuple_counts: Mapping[str, int],
    cf_rows: Mapping[tuple[str, int, str], Mapping[str, Any]],
    rows_by_condition: Mapping[str, Sequence[Mapping[str, Any]]],
    tuple_rows_by_condition: Mapping[str, Sequence[Mapping[str, Any]]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    total_tuple_count = sum(tuple_counts[str(query["id"])] for query in context_frame)
    construction = {
        field: {
            "constructed_tuple_count": sum(
                cf_rows[(str(query["id"]), tuple_index, field)].get(
                    "construction_status"
                )
                == "ok"
                for query in context_frame
                for tuple_index in range(tuple_counts[str(query["id"])])
            ),
            "total_gold_tuple_count": total_tuple_count,
            "tuple_rate": sum(
                cf_rows[(str(query["id"]), tuple_index, field)].get(
                    "construction_status"
                )
                == "ok"
                for query in context_frame
                for tuple_index in range(tuple_counts[str(query["id"])])
            )
            / total_tuple_count,
            "complete_case_query_count": sum(masks[field]),
            "total_query_count": len(masks[field]),
            "complete_case_query_rate": sum(masks[field]) / len(masks[field]),
        }
        for field in FIELDS
    }
    scoring: dict[str, Any] = {}
    for field in FIELDS:
        eligible_tuples = sum(
            tuple_counts[str(query["id"])]
            for query_index, query in enumerate(context_frame)
            if masks[field][query_index]
        )
        expected_tuple_cells = len(CONDITIONS) * eligible_tuples
        observed_tuple_cells = sum(
            row["field"] == field
            for condition in CONDITIONS
            for row in tuple_rows_by_condition[condition]
        )
        expected_query_cells = len(CONDITIONS) * sum(masks[field])
        observed_query_cells = sum(
            row["field"] == field and row["status"] == "ok"
            for condition in CONDITIONS
            for row in rows_by_condition[condition]
        )
        scoring[field] = {
            "expected_eligible_tuple_condition_cells": expected_tuple_cells,
            "observed_scored_tuple_condition_cells": observed_tuple_cells,
            "tuple_condition_rate": (
                observed_tuple_cells / expected_tuple_cells
                if expected_tuple_cells
                else 0.0
            ),
            "expected_eligible_query_condition_cells": expected_query_cells,
            "observed_scored_query_condition_cells": observed_query_cells,
            "query_condition_rate": (
                observed_query_cells / expected_query_cells
                if expected_query_cells
                else 0.0
            ),
        }
    return construction, scoring


def _summary(
    *,
    model_key: str,
    context_frame: Sequence[Mapping[str, Any]],
    masks: Mapping[str, Sequence[bool]],
    rows_by_condition: Mapping[str, Sequence[Mapping[str, Any]]],
    contrast_rows: Sequence[Mapping[str, Any]],
    construction: Mapping[str, Any],
    scoring: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": "stage1-margin-summary/v1",
        "model_key": model_key,
        "query_count": len(context_frame),
        "ordered_conditions": list(CONDITIONS),
        "construction_coverage": copy.deepcopy(dict(construction)),
        "scoring_coverage": copy.deepcopy(dict(scoring)),
        "per_condition": {
            condition: {
                field: {
                    "eligible_query_count": sum(masks[field]),
                    "mean_margin": (
                        sum(
                            float(row["margin_mean"])
                            for row in rows_by_condition[condition]
                            if row["field"] == field and row["status"] == "ok"
                        )
                        / sum(masks[field])
                        if sum(masks[field])
                        else None
                    ),
                }
                for field in FIELDS
            }
            for condition in CONDITIONS
        },
        "per_contrast": {
            contrast: {
                field: {
                    "eligible_query_count": sum(masks[field]),
                    "mean": (
                        sum(
                            float(row[contrast])
                            for row in contrast_rows
                            if row["field"] == field and row["status"] == "ok"
                        )
                        / sum(masks[field])
                        if sum(masks[field])
                        else None
                    ),
                }
                for field in FIELDS
            }
            for contrast in (
                "TE_L", "TE_D", "I_L_D", "TE_L_rel", "TE_D_rel", "L_shape", "D_shape"
            )
        },
    }


def build_margin_artifact(
    *,
    model_registry_ref: str | Path,
    model_key: str,
    context_ref: str | Path,
    control_ref: str | Path,
    cf_ref: str | Path,
    scorer_profile: str | Path | Mapping[str, Any],
    write_ref: str | Path,
    split: str,
    conditions: Sequence[str] | None = None,
    sealed: bool | None = None,
    target_root: str | Path | None = None,
    workspace_root: str | Path = REPOSITORY_ROOT,
    executor: MarginExecutor | None = None,
    model_resolver: Any | None = None,
    tokenizer_loader: Any | None = None,
) -> dict[str, Any]:
    """Score and atomically publish one complete model/condition margin block."""

    profile = resolve_margin_profile(scorer_profile)
    asserted_conditions = list(conditions) if conditions is not None else list(CONDITIONS)
    if asserted_conditions != profile["ordered_conditions"]:
        raise MarginLifecycleError("--conditions assertion differs from scorer profile")
    if split not in {"dev", "test"}:
        raise MarginLifecycleError("margin split must be dev or test")
    sealing = "sealed-test" if split == "test" else "unsealed-dev"
    if sealed is not None and sealed is not (split == "test"):
        raise MarginLifecycleError("--sealed assertion differs from derived sealing status")
    expected_control_kind = _control_kind_for_split_sealing(
        split=split, sealing_status=sealing
    )
    registry_locator, registry, registry_target = validate_registry_ref(
        model_registry_ref, workspace_root=workspace_root
    )
    entries = [row for row in registry["models"] if row["model_key"] == model_key]
    if len(entries) != 1:
        raise MarginLifecycleError("model key does not identify exactly one registry slot")
    scientific = registry["registry_scope"] == "formal"
    if bool(registry["scientific_eligible"]) is not scientific:
        raise MarginLifecycleError("registry scope/scientific eligibility mismatch")
    if scientific and any(
        value is not None for value in (executor, model_resolver, tokenizer_loader)
    ):
        raise MarginLifecycleError(
            "formal margin scoring constructs the registry-bound model internally; "
            "injected executors/resolvers/tokenizer loaders are forbidden"
        )
    try:
        handle = resolve_generation_model(
            model_registry_ref,
            model_key=model_key,
            workspace_root=workspace_root,
            require_scientific=scientific,
            resolver=model_resolver,
            tokenizer_loader=tokenizer_loader,
        )
    except GenerationLifecycleError as exc:
        raise MarginLifecycleError(str(exc)) from exc
    if handle.dependency != portable_dependency(
        registry_locator, registry_target, workspace_root
    ) or handle.model_artifact_id != entries[0]["model_dependency"]["artifact_id"]:
        raise MarginLifecycleError("registered runtime model binding differs from registry")
    if executor is not None and executor.tokenizer is not handle.tokenizer:
        raise MarginLifecycleError("injected engineering executor must use the registered tokenizer")

    try:
        context_locator, context_target = resolve_locator_ref(
            context_ref, expected_kind=("context", "test-context")
        )
        control_locator, control_target = resolve_locator_ref(
            control_ref, expected_kind=expected_control_kind
        )
        cf_locator, cf_target = resolve_locator_ref(
            cf_ref, expected_kind="counterfactual"
        )
    except TrainingArtifactError as exc:
        raise MarginLifecycleError(str(exc)) from exc
    context_dependency = portable_dependency(context_locator, context_target, workspace_root)
    control_dependency = portable_dependency(control_locator, control_target, workspace_root)
    cf_dependency = portable_dependency(cf_locator, cf_target, workspace_root)
    try:
        if scientific:
            context_meta = validate_context_target(
                context_target, workspace_root=workspace_root
            )
            control_report = validate_control_target(
                control_target,
                context_target=context_target,
                workspace_root=workspace_root,
            )
        else:
            context_meta = validate_context_target(
                context_target,
                tokenizer=handle.tokenizer,
                workspace_root=workspace_root,
            )
            control_report = validate_control_target(
                control_target,
                tokenizer=handle.tokenizer,
                context_target=context_target,
                workspace_root=workspace_root,
            )
        cf_report = validate_cf_target(cf_target, workspace_root=workspace_root)
    except Exception as exc:
        raise MarginLifecycleError(f"margin upstream validation failed: {exc}") from exc
    if context_meta.get("split") != split or control_report.get("split") != split or cf_report.get(
        "split"
    ) != split:
        raise MarginLifecycleError("margin context/control/CF split mismatch")
    if not _same_dependency_identity(
        load_json(control_target / "context_ref.json"), context_dependency
    ) or not _same_dependency_identity(
        load_json(cf_target / "context_ref.json"), context_dependency
    ):
        raise MarginLifecycleError("margin context/control/CF lineage mismatch")
    if context_meta.get("budget", {}).get("tokenizer_revision") != handle.tokenizer_revision:
        raise MarginLifecycleError("context tokenizer revision differs from registered model")
    expected_context_kind = "test-context" if split == "test" else "context"
    if context_locator["artifact_kind"] != expected_context_kind:
        raise MarginLifecycleError("margin split/context sealing kind mismatch")
    _scope_matrix(
        scope=registry["registry_scope"],
        split=split,
        sealing_status=sealing,
        scientific_eligible=scientific,
    )
    if scientific and (
        context_meta.get("scientific_eligible") is not True
        or cf_report.get("scientific_eligible") is not True
    ):
        raise MarginLifecycleError("formal margin run has an engineering upstream")

    context_frame = _context_frame(context_target, split=split)
    query_ids = [str(row["id"]) for row in context_frame]
    cf_rows, masks, tuple_counts = _cf_frame(
        cf_target=cf_target, split=split, context_frame=context_frame
    )
    prompts: dict[str, dict[str, str]] = {}
    for condition in CONDITIONS:
        source = control_target if condition in {"PL", "PD"} else context_target
        items = _runner_items(source, condition, split)
        if [str(item.get("id", "")) for item in items] != query_ids:
            raise MarginLifecycleError(
                f"{condition} runner changed the ordered master query frame"
            )
        prompts[condition] = {
            query_id: _prompt(
                item, condition=condition, query_id=query_id, tokenizer=handle.tokenizer
            )
            for query_id, item in zip(query_ids, items, strict=True)
        }

    maximum = int(profile["runtime"]["max_sequence_tokens"])
    calibration_entries: list[
        tuple[tuple[ScoreInput, ScoreInput], dict[str, Any]]
    ] = []
    # Tokenize the complete frozen frame before model execution.  Any overflow
    # blocks publication; it is never converted into a condition-specific mask.
    for condition in CONDITIONS:
        for query_index, query in enumerate(context_frame):
            query_id = str(query["id"])
            for tuple_index in range(tuple_counts[query_id]):
                for field in FIELDS:
                    if not masks[field][query_index]:
                        continue
                    pair = _prepare_pair(
                        tokenizer=handle.tokenizer,
                        prompt=prompts[condition][query_id],
                        gold=query["gold"],
                        tuple_index=tuple_index,
                        field=field,
                        cf_row=cf_rows[(query_id, tuple_index, field)],
                        max_sequence_tokens=maximum,
                    )
                    if len(calibration_entries) < CALIBRATION_PAIR_TARGET:
                        calibration_entries.append(
                            (
                                pair,
                                {
                                    "id": query_id,
                                    "condition": condition,
                                    "tuple_index": tuple_index,
                                    "field": field,
                                },
                            )
                        )
    if not calibration_entries:
        raise MarginLifecycleError("margin scorer frame has no eligible tuple/field")
    if scientific and len(calibration_entries) != CALIBRATION_PAIR_TARGET:
        raise MarginLifecycleError(
            "formal margin scoring requires exactly 100 deterministic calibration pairs"
        )

    intended_executor = (
        dict(executor.descriptor()) if executor is not None else copy.deepcopy(LOCAL_EXECUTOR_DESCRIPTOR)
    )
    if set(intended_executor) != {
        "executor_id", "executor_revision", "backend", "scientific_eligible"
    } or intended_executor.get("backend") != "local-huggingface-causal-lm":
        raise MarginLifecycleError("margin executor descriptor is non-canonical")
    if scientific and intended_executor != LOCAL_EXECUTOR_DESCRIPTOR:
        raise MarginLifecycleError("formal margin run requires the frozen local-HF executor")
    runtime_contract = _runtime_contract(intended_executor, profile)
    batch_contract = _batch_contract(profile)
    contracts = _contract_hashes(
        profile=profile, handle=handle, runtime_contract=runtime_contract
    )
    dependencies = {
        "registry": portable_dependency(registry_locator, registry_target, workspace_root),
        "training_plan": registry["training_plan_dependency"],
        "model": entries[0]["model_dependency"],
        "context": context_dependency,
        "control": control_dependency,
        "cf": cf_dependency,
    }
    id_inputs = {
        "schema_version": "stage1-margin-id-inputs/v1",
        "training_plan_dependency": dependencies["training_plan"],
        "model_registry_dependency": dependencies["registry"],
        "model_key": model_key,
        "model_dependency": dependencies["model"],
        "context_dependency": dependencies["context"],
        "control_dependency": dependencies["control"],
        "cf_dependency": dependencies["cf"],
        "split": split,
        "sealing_status": sealing,
        "ordered_conditions": list(CONDITIONS),
        "expected_ordered_query_ids_sha256": _ordered_id_hash(query_ids),
        "scorer_profile_sha256": canonical_sha256(profile),
        "eligibility_mask_schema": "stage1-prefrozen-field-mask/v1",
        "eligibility_mask_sha256": canonical_sha256(masks),
        **contracts,
    }
    margin_id = "mgn-" + canonical_sha256(id_inputs)

    root = (
        Path(target_root).resolve()
        if target_root is not None
        else registry_target.parent.parent
    )
    parent = root / "margin_runs"
    target = parent / margin_id
    if target.exists():
        try:
            report = validate_margin_target(
                target,
                workspace_root=workspace_root,
                model_dependency_resolver=(
                    (lambda **_kwargs: handle) if executor is not None else None
                ),
                tokenizer_loader=(
                    (lambda _path: handle.tokenizer) if executor is not None else None
                ),
            )
        except Exception as exc:
            raise MarginLifecycleError(
                f"existing margin target is not the requested immutable payload: {exc}"
            ) from exc
        return write_locator_ref(
            write_ref,
            artifact_kind=MARGIN_KIND,
            artifact_id=margin_id,
            target=target,
            payload_manifest_sha256=report["payload_manifest_sha256"],
        )

    active_executor = executor or LocalHFMarginExecutor.from_registered_model(
        handle, profile
    )
    calibration, calibration_rows = _batch_calibration(
        active_executor, calibration_entries
    )
    rows_by_condition: dict[str, list[dict[str, Any]]] = {}
    tuple_rows_by_condition: dict[str, list[dict[str, Any]]] = {}
    for condition in CONDITIONS:
        tuple_rows: list[dict[str, Any]] = []
        for query_index, query in enumerate(context_frame):
            query_id = str(query["id"])
            for tuple_index in range(tuple_counts[query_id]):
                for field in FIELDS:
                    if not masks[field][query_index]:
                        continue
                    cf_row = cf_rows[(query_id, tuple_index, field)]
                    pair = _prepare_pair(
                        tokenizer=handle.tokenizer,
                        prompt=prompts[condition][query_id],
                        gold=query["gold"],
                        tuple_index=tuple_index,
                        field=field,
                        cf_row=cf_row,
                        max_sequence_tokens=maximum,
                    )
                    scored = active_executor.score(pair)
                    if len(scored) != 2:
                        raise MarginLifecycleError(
                            "margin executor did not return a gold/CF score pair"
                        )
                    tuple_rows.append(
                        _tuple_score_row(
                            query=query,
                            condition=condition,
                            tuple_index=tuple_index,
                            field=field,
                            model_key=model_key,
                            cf_row=cf_row,
                            gold_score=scored[0],
                            foil_score=scored[1],
                        )
                    )
        tuple_rows_by_condition[condition] = tuple_rows
        rows_by_condition[condition] = _aggregate_rows(
            condition=condition,
            model_key=model_key,
            context_frame=context_frame,
            masks=masks,
            tuple_scores=tuple_rows,
        )
    contrast_rows = _contrast_rows(
        model_key=model_key,
        context_frame=context_frame,
        masks=masks,
        rows_by_condition=rows_by_condition,
    )
    construction, scoring = _coverage(
        masks,
        context_frame=context_frame,
        tuple_counts=tuple_counts,
        cf_rows=cf_rows,
        rows_by_condition=rows_by_condition,
        tuple_rows_by_condition=tuple_rows_by_condition,
    )
    summary = _summary(
        model_key=model_key,
        context_frame=context_frame,
        masks=masks,
        rows_by_condition=rows_by_condition,
        contrast_rows=contrast_rows,
        construction=construction,
        scoring=scoring,
    )
    meta = {
        "schema_version": "stage1-margin-run/v1",
        "margin_run_id": margin_id,
        "model_key": model_key,
        "split": split,
        "sealing_status": sealing,
        "scientific_eligible": scientific,
        "ordered_conditions": list(CONDITIONS),
        "ordered_query_ids": query_ids,
        "ordered_query_ids_sha256": _ordered_id_hash(query_ids),
        "field_eligibility_masks": masks,
        "paired_block_complete": True,
        "expected_row_count_per_condition": len(query_ids) * len(FIELDS),
        "observed_row_count_per_condition": {
            condition: len(rows_by_condition[condition]) for condition in CONDITIONS
        },
        "expected_tuple_score_count_per_condition": sum(
            tuple_counts[str(query["id"])]
            for query_index, query in enumerate(context_frame)
            for field in FIELDS
            if masks[field][query_index]
        ),
        "observed_tuple_score_count_per_condition": {
            condition: len(tuple_rows_by_condition[condition])
            for condition in CONDITIONS
        },
        "construction_coverage": construction,
        "scoring_coverage": scoring,
        "per_condition_sufficient_stats_sha256": {
            condition: canonical_sha256(rows_by_condition[condition])
            for condition in CONDITIONS
        },
        "per_condition_tuple_scores_sha256": {
            condition: canonical_sha256(tuple_rows_by_condition[condition])
            for condition in CONDITIONS
        },
        "query_contrasts_sha256": canonical_sha256(contrast_rows),
        "batch_calibration": calibration,
        "batch_calibration_records_sha256": canonical_sha256(calibration_rows),
        "batch_contract": batch_contract,
        "executor": intended_executor,
        "runtime_contract_sha256": canonical_sha256(runtime_contract),
        "id_inputs": id_inputs,
    }
    provenance = {
        "schema_version": "stage1-margin-provenance/v1",
        "margin_run_id": margin_id,
        "dependencies": copy.deepcopy(dependencies),
        "model_key": model_key,
        "model_resolver": {
            "protocol": "model.stage1_registry.resolve_registered_model_dependency/v1",
            "accepted_source": "embedded-stage1-model-registry-dependency-only",
            "direct_model_path_allowed": False,
        },
        "executor": intended_executor,
        "scorer_profile_sha256": canonical_sha256(profile),
        "runtime_contract_sha256": canonical_sha256(runtime_contract),
        "batch_calibration_contract_sha256": contracts[
            "batch_calibration_contract_sha256"
        ],
        "scorer_code_sha256": contracts["scorer_code_sha256"],
        "observed_margins_in_lifecycle_id": False,
        "upstream_mutation_allowed": False,
    }

    upstream_targets = (
        (registry_target, registry_locator["payload_manifest_sha256"], "registry"),
        (context_target, context_locator["payload_manifest_sha256"], "context"),
        (control_target, control_locator["payload_manifest_sha256"], "control"),
        (cf_target, cf_locator["payload_manifest_sha256"], "CF"),
    )
    for upstream, payload_hash_before, label in upstream_targets:
        _assert_target_unchanged(upstream, payload_hash_before, label=label)
    # Re-run the registry's source-tree rehash after scoring but before any
    # target is published, so a concurrent checkpoint/tokenizer mutation
    # cannot leave even an unreferenced final margin directory behind.
    try:
        resolved_after = resolve_registered_model_dependency(
            registry_dependency=dependencies["registry"],
            model_key=model_key,
            workspace_root=workspace_root,
        )
    except (TrainingArtifactError, ModelRegistryError) as exc:
        raise MarginLifecycleError(str(exc)) from exc
    if resolved_after.model_artifact_id != dependencies["model"]["artifact_id"]:
        raise MarginLifecycleError("registered model changed while margin scoring ran")

    staging = new_staging_directory(parent, margin_id)
    try:
        for filename, value in (
            ("model_registry_ref.json", dependencies["registry"]),
            ("training_plan_ref.json", dependencies["training_plan"]),
            ("model_ref.json", dependencies["model"]),
            ("context_ref.json", dependencies["context"]),
            ("control_ref.json", dependencies["control"]),
            ("cf_ref.json", dependencies["cf"]),
            ("scorer_profile.resolved.json", profile),
            ("runtime_contract.json", runtime_contract),
            ("batch_calibration.jsonl", calibration_rows),
            ("margin.meta.json", meta),
            ("provenance.json", provenance),
            ("summary.json", summary),
        ):
            if filename.endswith(".jsonl"):
                _write_ordered_jsonl(staging / filename, value)
            else:
                write_canonical_json(staging / filename, value)
        for condition in CONDITIONS:
            _write_ordered_jsonl(
                staging / "margins" / f"{condition}.jsonl",
                rows_by_condition[condition],
            )
            _write_ordered_jsonl(
                staging / "tuple_scores" / f"{condition}.jsonl",
                tuple_rows_by_condition[condition],
            )
        _write_ordered_jsonl(staging / "query_contrasts.jsonl", contrast_rows)
        payload_hash = finalize_target_atomic(
            staging,
            target,
            validate_staging=lambda candidate: validate_margin_target(
                candidate,
                workspace_root=workspace_root,
                require_directory_name=False,
                model_dependency_resolver=(
                    (lambda **_kwargs: handle) if executor is not None else None
                ),
                tokenizer_loader=(
                    (lambda _path: handle.tokenizer) if executor is not None else None
                ),
            ),
        )
    except (TrainingArtifactError, OSError) as exc:
        if staging.exists():
            shutil.rmtree(staging)
        raise MarginLifecycleError(str(exc)) from exc
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return write_locator_ref(
        write_ref,
        artifact_kind=MARGIN_KIND,
        artifact_id=margin_id,
        target=target,
        payload_manifest_sha256=payload_hash,
    )


__all__ = [
    "LocalHFMarginExecutor",
    "MarginExecutor",
    "MarginLifecycleError",
    "build_margin_artifact",
]
