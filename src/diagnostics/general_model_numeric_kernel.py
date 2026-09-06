"""Full-sequence teacher forcing for the registered classification candidates."""

from __future__ import annotations

import hashlib
import math
import time
from typing import Any

from data.stage1_data import canonical_json_sha256


MAX_SEQUENCE_TOKENS = 8192
TARGET_TIME_CHUNK = 8


class NumericKernelError(ValueError):
    """A frozen input or a numerical scoring invariant was violated."""


def _token_ids(tokenizer: Any, text: str) -> list[int]:
    values = tokenizer.encode(text, add_special_tokens=False)
    if not isinstance(values, list) or any(type(value) is not int or value < 0 for value in values):
        raise NumericKernelError("tokenizer returned invalid token IDs")
    return values


def _prepare(runner: Any, item: dict, eos_id: int, limit: int) -> dict:
    context, candidate = item["context"], item["candidate"]
    prompt, answer = context["prompt_text"], candidate["canonical_answer"]
    if not isinstance(prompt, str) or not isinstance(answer, str):
        raise NumericKernelError("prompt and canonical answer must be strings")
    if context.get("overflow", False) or not context.get("control_valid", True):
        raise NumericKernelError("context failed its registered control or length check")
    prompt_ids = _token_ids(runner.tokenizer, prompt)
    answer_ids = _token_ids(runner.tokenizer, answer)
    if not prompt_ids or not answer_ids:
        raise NumericKernelError("prompt and answer tokenization must be nonempty")
    prompt_hash = canonical_json_sha256(prompt_ids)
    if (hashlib.sha256(prompt.encode("utf-8")).hexdigest() != context["prompt_sha256"]
            or prompt_hash != context["prompt_token_ids_sha256"]
            or len(prompt_ids) != context["prompt_tokens"]):
        raise NumericKernelError("prompt text or token identity differs from the frozen context")
    answer_hash = canonical_json_sha256(answer_ids)
    expected = {
        "answer_token_ids": answer_ids,
        "answer_token_ids_sha256": answer_hash,
        "answer_tokens": len(answer_ids),
        "canonical_answer_sha256": hashlib.sha256(answer.encode("utf-8")).hexdigest(),
    }
    for key, observed in expected.items():
        if key in candidate and candidate[key] != observed:
            raise NumericKernelError(f"candidate {key} differs from its frozen token identity")
    if eos_id in answer_ids:
        raise NumericKernelError("canonical answer contains the reserved EOS token")
    if _token_ids(runner.tokenizer, prompt + answer) != prompt_ids + answer_ids:
        raise NumericKernelError("prompt/answer token boundary is not concatenation-stable")
    sequence = prompt_ids + answer_ids + [eos_id]
    if len(sequence) > limit:
        raise NumericKernelError(f"teacher-forcing sequence overflow: {len(sequence)} > {limit}")
    return {
        **expected, "sequence": sequence, "prompt_tokens": len(prompt_ids),
        "prompt_token_ids_sha256": prompt_hash,
    }


def _scores(values: list[float]) -> dict:
    answer_values, eos = values[:-1], values[-1]
    answer_sum, total = math.fsum(answer_values), math.fsum(values)
    return {
        "token_logprobs": answer_values,
        "eos_logprob": eos,
        "answer_sum": answer_sum,
        "answer_mean": answer_sum / len(answer_values),
        "total_with_eos": total,
        "mean_with_eos": total / len(values),
    }


def score_batch(runner: Any, items: list[dict], *, reference: bool = False) -> list[dict]:
    """Score each ``{context, candidate}`` with one right-padded causal forward pass.

    ``reference`` compares FP32 log-probability arithmetic against CPU float64
    arithmetic on the same model logits. It does not change model precision.
    """
    import torch

    if not items:
        return []
    if len(items) not in (1, 2, 3, 4):
        raise NumericKernelError("registered scoring batches contain at most four sequences")
    eos_id = runner.tokenizer.eos_token_id
    if type(eos_id) is not int or eos_id < 0 or set(runner.eos_ids) != {eos_id}:
        raise NumericKernelError("numeric scoring requires one unambiguous tokenizer/model EOS")
    pad_id = runner.tokenizer.pad_token_id
    if type(pad_id) is not int or pad_id < 0:
        raise NumericKernelError("numeric scoring requires an explicit padding token")
    configured_limit = runner.config["runtime"]["max_sequence_tokens"]
    if type(configured_limit) is not int or configured_limit <= 0:
        raise NumericKernelError("invalid registered sequence limit")
    limit = min(configured_limit, MAX_SEQUENCE_TOKENS)
    prepared = [_prepare(runner, item, eos_id, limit) for item in items]
    max_length = max(len(row["sequence"]) for row in prepared)
    input_ids = torch.full((len(items), max_length), pad_id, dtype=torch.long, device=runner.device)
    attention_mask = torch.zeros_like(input_ids)
    for index, row in enumerate(prepared):
        length = len(row["sequence"])
        input_ids[index, :length] = torch.tensor(row["sequence"], dtype=torch.long, device=runner.device)
        attention_mask[index, :length] = 1

    results = []
    is_cuda = torch.device(runner.device).type == "cuda"
    if is_cuda:
        torch.cuda.synchronize(runner.device)
        torch.cuda.reset_peak_memory_stats(runner.device)
    with torch.inference_mode():
        forward_started = time.monotonic()
        logits = runner.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits
        if is_cuda:
            torch.cuda.synchronize(runner.device)
        forward_seconds = time.monotonic() - forward_started
        if logits.ndim != 3 or tuple(logits.shape[:2]) != tuple(input_ids.shape):
            raise NumericKernelError("model logits do not match the full teacher-forcing batch")
        for index, row in enumerate(prepared):
            start = row["prompt_tokens"] - 1
            target_ids = row["answer_token_ids"] + [eos_id]
            if max(target_ids) >= logits.shape[-1]:
                raise NumericKernelError("candidate target is outside the model vocabulary")
            values: list[float] = []
            reference_values: list[float] = []
            normalization_seconds = 0.0
            reference_seconds = 0.0
            # Only answer/EOS positions need vocabulary normalizers; prompt and
            # right-padding logits never enter the score or its finite check.
            for offset in range(0, len(target_ids), TARGET_TIME_CHUNK):
                normalization_started = time.monotonic()
                end = min(offset + TARGET_TIME_CHUNK, len(target_ids))
                selected = logits[index, start + offset:start + end, :]
                if not torch.isfinite(selected).all().item():
                    raise NumericKernelError("non-finite logits at an answer/EOS target position")
                targets = torch.tensor(target_ids[offset:end], dtype=torch.long, device=selected.device)
                selected32 = selected.float()
                logprobs = selected32.gather(-1, targets[:, None]).squeeze(-1) - selected32.logsumexp(dim=-1)
                if not torch.isfinite(logprobs).all().item():
                    raise NumericKernelError("non-finite target token log-probabilities")
                values.extend(logprobs.cpu().tolist())
                normalization_seconds += time.monotonic() - normalization_started
                if reference:
                    reference_started = time.monotonic()
                    selected64 = selected.to(device="cpu", dtype=torch.float64)
                    targets_cpu = targets.cpu()
                    ref = selected64.gather(-1, targets_cpu[:, None]).squeeze(-1) - selected64.logsumexp(dim=-1)
                    if not torch.isfinite(ref).all().item():
                        raise NumericKernelError("non-finite CPU float64 reference")
                    reference_values.extend(ref.tolist())
                    reference_seconds += time.monotonic() - reference_started
            result = {key: value for key, value in row.items() if key != "sequence"}
            result.update(_scores(values))
            result.update({
                "eos_token_id": eos_id,
                "sequence_tokens": len(row["sequence"]),
                "padded_sequence_tokens": max_length,
                "batch_size": len(items),
                "finite_target_logits_checked": True,
                "token_boundary_checked": True,
                "causal_shift": 1,
                "use_cache": False,
                "padding_side": "right",
                "model_logits_dtype": str(logits.dtype),
                "logprob_arithmetic_dtype": "torch.float32",
                "reference_checked": reference,
                "forward_seconds": forward_seconds,
                "normalization_seconds": normalization_seconds,
                "reference_seconds": reference_seconds,
            })
            if reference:
                reference_scores = _scores(reference_values)
                differences = {key: result[key] - reference_scores[key] for key in (
                    "eos_logprob", "answer_sum", "answer_mean", "total_with_eos", "mean_with_eos",
                )}
                token_differences = [a - b for a, b in zip(values[:-1], reference_values[:-1], strict=True)]
                result.update({
                    "reference_scores": reference_scores,
                    "reference_token_logprobs": reference_values[:-1],
                    "reference_eos_logprob": reference_values[-1],
                    "reference_differences": {**differences, "token_logprobs": token_differences},
                    "reference_abs_error_max": max(abs(value) for value in [
                        *differences.values(), *token_differences,
                    ]),
                    "reference_arithmetic_dtype": "cpu.torch.float64",
                })
            results.append(result)
    peak = torch.cuda.max_memory_allocated(runner.device) if is_cuda else None
    for result in results:
        result["peak_memory_allocated_bytes"] = peak
    return results
