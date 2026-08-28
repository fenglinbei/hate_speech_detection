"""Local-HF Stage-1 teacher-forced field margin scorer.

This module scores complete canonical JSON value literals.  Prompt and response
are tokenized as two independent segments, matching the Stage-1 SFT boundary;
the response is never pre-truncated at a character boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from utils.quadruple import (
    QUAD_KEYS,
    Quadruple,
    canonicalize_quadruples,
    serialize_with_spans,
)


MARGIN_VERSION = "stage1-margin/v1"
SPAN_MASK_VERSION = "minimal-overlap-cover/v1"
SEGMENTATION_VERSION = "separate-no-special-tokens/v1"


@dataclass(frozen=True)
class ScoreInput:
    prompt_ids: tuple[int, ...]
    response_ids: tuple[int, ...]
    response_offsets: tuple[tuple[int, int], ...]
    character_span: tuple[int, int]
    response_token_indices: tuple[int, ...]
    global_token_indices: tuple[int, ...]
    left_boundary_crossing: bool
    right_boundary_crossing: bool
    response_text: str

    @property
    def cropped_input_ids(self) -> tuple[int, ...]:
        last = self.global_token_indices[-1]
        return (self.prompt_ids + self.response_ids)[: last + 1]


def _one_tokenizer_value(value: Any, key: str) -> list[int]:
    if isinstance(value, Mapping):
        value = value.get(key)
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError(f"tokenizer output {key!r} must be a sequence")
    if value and isinstance(value[0], Sequence) and not isinstance(value[0], (str, bytes, bytearray)):
        if len(value) != 1:
            raise ValueError("batched tokenizer output is not allowed here")
        value = value[0]
    return [int(item) for item in value]


def _tokenize_prompt(tokenizer: Any, prompt: str) -> list[int]:
    encoded = tokenizer(prompt, add_special_tokens=False)
    return _one_tokenizer_value(encoded, "input_ids")


def _tokenize_response(tokenizer: Any, response: str) -> tuple[list[int], list[tuple[int, int]]]:
    encoded = tokenizer(
        response,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    ids = _one_tokenizer_value(encoded, "input_ids")
    raw_offsets = encoded.get("offset_mapping") if isinstance(encoded, Mapping) else None
    if raw_offsets is None:
        raise ValueError("a fast tokenizer with response-local offset mapping is required")
    if (
        raw_offsets
        and isinstance(raw_offsets[0], Sequence)
        and raw_offsets[0]
        and isinstance(raw_offsets[0][0], Sequence)
    ):
        if len(raw_offsets) != 1:
            raise ValueError("batched offset mapping is not allowed here")
        raw_offsets = raw_offsets[0]
    offsets = [(int(start), int(end)) for start, end in raw_offsets]
    if len(ids) != len(offsets):
        raise ValueError("response token IDs and offsets have different lengths")
    return ids, offsets


def minimal_overlap_cover(
    offsets: Sequence[tuple[int, int]],
    character_span: tuple[int, int],
) -> tuple[tuple[int, ...], bool, bool]:
    """Select every response token having non-empty overlap with the span."""

    start, end = character_span
    if start < 0 or end <= start:
        raise ValueError("character span must be non-empty and non-negative")
    indices: list[int] = []
    left_crossing = False
    right_crossing = False
    for index, (token_start, token_end) in enumerate(offsets):
        if token_start < 0 or token_end < token_start:
            raise ValueError("invalid tokenizer offset")
        if token_start == token_end:
            continue
        if max(token_start, start) < min(token_end, end):
            indices.append(index)
            left_crossing = left_crossing or token_start < start < token_end
            right_crossing = right_crossing or token_start < end < token_end
    if not indices:
        raise ValueError("field character span has no overlapping response token")
    return tuple(indices), left_crossing, right_crossing


def prepare_score_input(
    *,
    tokenizer: Any,
    rendered_chat_prompt: str,
    canonical_response: str,
    character_span: tuple[int, int],
    max_sequence_tokens: int,
) -> ScoreInput:
    """Tokenize the full response once, freeze its mask, then crop by token."""

    if max_sequence_tokens <= 0:
        raise ValueError("max_sequence_tokens must be positive")
    prompt_ids = _tokenize_prompt(tokenizer, rendered_chat_prompt)
    response_ids, offsets = _tokenize_response(tokenizer, canonical_response)
    response_indices, left_crossing, right_crossing = minimal_overlap_cover(offsets, character_span)
    global_indices = tuple(len(prompt_ids) + index for index in response_indices)
    if global_indices[0] == 0:
        raise ValueError("the first sequence token cannot be scored by a causal LM")
    cropped_length = global_indices[-1] + 1
    if cropped_length > max_sequence_tokens:
        raise OverflowError(
            f"required scoring prefix has {cropped_length} tokens; limit is {max_sequence_tokens}"
        )
    return ScoreInput(
        prompt_ids=tuple(prompt_ids),
        response_ids=tuple(response_ids),
        response_offsets=tuple(offsets),
        character_span=character_span,
        response_token_indices=response_indices,
        global_token_indices=global_indices,
        left_boundary_crossing=left_crossing,
        right_boundary_crossing=right_crossing,
        response_text=canonical_response,
    )


def replace_one_field(
    gold: Sequence[Quadruple | Mapping[str, Any]],
    *,
    tuple_index: int,
    field: str,
    candidate_value: Any,
) -> list[Quadruple]:
    """Canonicalize a foil and prove exactly one requested field changed."""

    if field not in QUAD_KEYS:
        raise ValueError(f"unknown quadruple field: {field}")
    gold_quads = canonicalize_quadruples(gold)
    if not 0 <= tuple_index < len(gold_quads):
        raise IndexError("tuple_index is outside the gold response")
    mappings = [
        {
            "target": quad.target,
            "argument": quad.argument,
            "targeted_group": list(quad.targeted_group),
            "hateful": quad.hateful,
        }
        for quad in gold_quads
    ]
    mappings[tuple_index][field] = candidate_value
    foil = canonicalize_quadruples(mappings)
    changes = []
    for index, (before, after) in enumerate(zip(gold_quads, foil, strict=True)):
        for name in QUAD_KEYS:
            if getattr(before, name) != getattr(after, name):
                changes.append((index, name))
    if changes != [(tuple_index, field)]:
        raise ValueError(
            "counterfactual must change exactly the requested field and differ from gold; "
            f"observed changes={changes}"
        )
    return foil


def prepare_field_pair(
    *,
    tokenizer: Any,
    rendered_chat_prompt: str,
    gold: Sequence[Quadruple | Mapping[str, Any]],
    tuple_index: int,
    field: str,
    candidate_value: Any,
    max_sequence_tokens: int = 2048,
) -> tuple[ScoreInput, ScoreInput]:
    """Prepare gold and one-field foil inputs under the identical prompt."""

    gold_quads = canonicalize_quadruples(gold)
    foil_quads = replace_one_field(
        gold_quads,
        tuple_index=tuple_index,
        field=field,
        candidate_value=candidate_value,
    )
    gold_response, gold_spans = serialize_with_spans(gold_quads)
    foil_response, foil_spans = serialize_with_spans(foil_quads)
    gold_input = prepare_score_input(
        tokenizer=tokenizer,
        rendered_chat_prompt=rendered_chat_prompt,
        canonical_response=gold_response,
        character_span=gold_spans[(tuple_index, field)],
        max_sequence_tokens=max_sequence_tokens,
    )
    foil_input = prepare_score_input(
        tokenizer=tokenizer,
        rendered_chat_prompt=rendered_chat_prompt,
        canonical_response=foil_response,
        character_span=foil_spans[(tuple_index, field)],
        max_sequence_tokens=max_sequence_tokens,
    )
    return gold_input, foil_input


def score_inputs(
    *,
    model: Any,
    score_inputs: Sequence[ScoreInput],
    pad_token_id: int,
    device: Any | None = None,
) -> list[dict[str, Any]]:
    """Right-pad and score a batch with explicit attention/position IDs."""

    if not score_inputs:
        return []
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - environment contract includes torch
        raise RuntimeError("torch is required for Stage 1 margin scoring") from exc

    sequences = [list(item.cropped_input_ids) for item in score_inputs]
    maximum = max(len(sequence) for sequence in sequences)
    input_ids = torch.full((len(sequences), maximum), int(pad_token_id), dtype=torch.long)
    attention_mask = torch.zeros((len(sequences), maximum), dtype=torch.long)
    for row, sequence in enumerate(sequences):
        input_ids[row, : len(sequence)] = torch.tensor(sequence, dtype=torch.long)
        attention_mask[row, : len(sequence)] = 1
    position_ids = (attention_mask.cumsum(dim=-1) - 1).masked_fill(attention_mask == 0, 0)

    if device is None:
        try:
            device = next(model.parameters()).device
        except (StopIteration, AttributeError):
            device = None
    if device is not None:
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        position_ids = position_ids.to(device)

    with torch.inference_mode():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
        if logits.ndim != 3 or logits.shape[:2] != input_ids.shape:
            raise ValueError("causal LM returned logits with an unexpected shape")
        log_probs = torch.log_softmax(logits[:, :-1, :].float(), dim=-1)

    results: list[dict[str, Any]] = []
    for row, item in enumerate(score_inputs):
        token_logprobs: list[float] = []
        token_ids: list[int] = []
        for global_index in item.global_token_indices:
            if global_index <= 0 or global_index >= len(sequences[row]):
                raise AssertionError("field token index is outside the cropped causal sequence")
            token_id = int(input_ids[row, global_index].item())
            value = float(log_probs[row, global_index - 1, token_id].item())
            token_ids.append(token_id)
            token_logprobs.append(value)
        total = sum(token_logprobs)
        results.append(
            {
                "schema_version": "stage1-field-score/v1",
                "token_ids": token_ids,
                "token_logprobs": token_logprobs,
                "token_count": len(token_ids),
                "character_span": list(item.character_span),
                "response_token_indices": list(item.response_token_indices),
                "global_token_indices": list(item.global_token_indices),
                "sum_logprob": total,
                "mean_logprob": total / len(token_logprobs),
                "left_boundary_crossing": item.left_boundary_crossing,
                "right_boundary_crossing": item.right_boundary_crossing,
                "span_mask_version": SPAN_MASK_VERSION,
                "segmentation_version": SEGMENTATION_VERSION,
            }
        )
    return results


def score_field_margin(
    *,
    model: Any,
    tokenizer: Any,
    rendered_chat_prompt: str,
    gold: Sequence[Quadruple | Mapping[str, Any]],
    tuple_index: int,
    field: str,
    candidate_value: Any,
    pad_token_id: int | None = None,
    max_sequence_tokens: int = 2048,
    device: Any | None = None,
) -> dict[str, Any]:
    gold_input, foil_input = prepare_field_pair(
        tokenizer=tokenizer,
        rendered_chat_prompt=rendered_chat_prompt,
        gold=gold,
        tuple_index=tuple_index,
        field=field,
        candidate_value=candidate_value,
        max_sequence_tokens=max_sequence_tokens,
    )
    resolved_pad = pad_token_id
    if resolved_pad is None:
        resolved_pad = getattr(tokenizer, "pad_token_id", None)
    if resolved_pad is None:
        resolved_pad = getattr(tokenizer, "eos_token_id", None)
    if resolved_pad is None:
        raise ValueError("tokenizer must define pad_token_id or eos_token_id")
    gold_score, foil_score = score_inputs(
        model=model,
        score_inputs=[gold_input, foil_input],
        pad_token_id=int(resolved_pad),
        device=device,
    )
    return {
        "schema_version": "stage1-field-margin/v1",
        "margin_version": MARGIN_VERSION,
        "tuple_index": tuple_index,
        "field": field,
        "gold": gold_score,
        "counterfactual": foil_score,
        "mean_margin": gold_score["mean_logprob"] - foil_score["mean_logprob"],
        "sum_margin_sensitivity": gold_score["sum_logprob"] - foil_score["sum_logprob"],
    }


__all__ = [
    "MARGIN_VERSION",
    "SEGMENTATION_VERSION",
    "SPAN_MASK_VERSION",
    "ScoreInput",
    "minimal_overlap_cover",
    "prepare_field_pair",
    "prepare_score_input",
    "replace_one_field",
    "score_field_margin",
    "score_inputs",
]
