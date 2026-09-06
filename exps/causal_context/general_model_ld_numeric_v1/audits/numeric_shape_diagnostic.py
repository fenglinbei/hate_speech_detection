"""Read-only sealed-score shape diagnostic plus a CPU-only tiny Qwen3 probe."""

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM
from transformers.models.qwen3 import modeling_qwen3
from transformers import masking_utils

from data.stage1_data import canonical_json_sha256, sha256_file
from diagnostics.general_model_numeric_kernel import score_batch


ROOT = Path(__file__).resolve().parents[4]
RUN = ROOT / "exps/causal_context/general_model_ld_numeric_v1/runs/numeric-01"
PASSES = ("regression-b1-r0", "regression-b4-r0", "regression-b4-tail2")


def read(path):
    return json.loads(path.read_text())


def group_summary(groups):
    repeated = []
    for key, values in groups.items():
        if len(values) < 2:
            continue
        minimum = min(values, key=lambda row: row["token_logprob"])
        maximum = max(values, key=lambda row: row["token_logprob"])
        spread = maximum["token_logprob"] - minimum["token_logprob"]
        repeated.append({"record_id": key[0], "answer_prefix_through_target": list(key[1]),
                         "answer_token_index": len(key[1]) - 1,
                         "spread": spread, "member_count": len(values),
                         "minimum": minimum, "maximum": maximum})
    repeated.sort(key=lambda row: row["spread"], reverse=True)
    return {
        "repeated_group_count": len(repeated),
        "exact_zero_spread_count": sum(row["spread"] == 0 for row in repeated),
        "above_1e_4_count": sum(row["spread"] > 1e-4 for row in repeated),
        "above_5e_3_count": sum(row["spread"] > 5e-3 for row in repeated),
        "maximum_spread": max((row["spread"] for row in repeated), default=0.0),
        "representative_largest_groups": repeated[:5],
    }


class Tokenizer:
    eos_token_id = 127
    pad_token_id = 0

    def encode(self, value, *, add_special_tokens):
        assert add_special_tokens is False
        return [ord(char) - 31 for char in value]


def tiny_probe(dtype):
    torch.manual_seed(42)
    config = Qwen3Config(vocab_size=128, hidden_size=64, intermediate_size=128,
                         num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                         head_dim=16, max_position_embeddings=256, pad_token_id=0,
                         eos_token_id=127, attention_dropout=0.0, use_cache=False,
                         attn_implementation="eager")
    model = Qwen3ForCausalLM(config).to(device="cpu", dtype=dtype).eval()
    tokenizer = Tokenizer()
    runner = SimpleNamespace(model=model, tokenizer=tokenizer, eos_ids={127},
                             device=torch.device("cpu"),
                             config={"runtime": {"max_sequence_tokens": 8192}})
    items = []
    prompt = "PROMPT:"
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    for answer in ('["R"]', '["Region"]', '["R","others"]', '["Region","others"]'):
        context = {"prompt_text": prompt, "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                   "prompt_token_ids_sha256": canonical_json_sha256(prompt_ids),
                   "prompt_tokens": len(prompt_ids), "control_valid": True, "overflow": False}
        items.append({"context": context, "candidate": {"canonical_answer": answer}})
    capture = {}

    def hook(module, args, kwargs):
        capture.update(mask=kwargs["attention_mask"].detach().clone(),
                       positions=kwargs["position_ids"].detach().clone(),
                       cache_position=kwargs["cache_position"].detach().clone(),
                       past_key_value=kwargs["past_key_value"], use_cache=kwargs["use_cache"])

    handle = model.model.layers[0].register_forward_pre_hook(hook, with_kwargs=True)
    batched = score_batch(runner, items)
    handle.remove()
    length = batched[0]["padded_sequence_tokens"]
    expected = torch.zeros((4, 1, length, length), dtype=torch.bool)
    for batch, row in enumerate(batched):
        for query in range(length):
            expected[batch, 0, query, :min(query + 1, row["sequence_tokens"])] = True
    mask_valid = torch.equal(capture["mask"] == 0, expected)
    position_valid = torch.equal(capture["positions"], torch.arange(length)[None, :])
    cache_valid = torch.equal(capture["cache_position"], torch.arange(length))
    singleton = [score_batch(runner, [item])[0] for item in items]
    max_error = max(abs(a - b) for row_a, row_b in zip(singleton, batched)
                    for a, b in zip(row_a["token_logprobs"] + [row_a["eos_logprob"]],
                                    row_b["token_logprobs"] + [row_b["eos_logprob"]]))
    return {"device": "cpu", "dtype": str(dtype), "attention": config._attn_implementation,
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "candidate_lengths_including_prompt_and_eos": [row["sequence_tokens"] for row in batched],
            "right_padding_and_causal_mask_exact": mask_valid,
            "default_position_ids_match_unpadded_prefix_positions": position_valid,
            "cache_position_is_zero_based_arange": cache_valid,
            "past_key_value_is_none": capture["past_key_value"] is None,
            "use_cache": capture["use_cache"],
            "batch1_vs_batch4_max_target_logprob_difference": max_error,
            "limits": "CPU random tiny model does not establish correctness or a numerical bound for the trained 8B CUDA run."}


def main():
    torch.set_num_threads(4)
    report = read(RUN / "preflight/preflight_report.json")
    output = {"schema_version": "general-model-numeric-shape-diagnostic/v1",
              "plan_id": report["plan_id"], "run": str(RUN),
              "preflight_report_sha256": sha256_file(RUN / "preflight/preflight_report.json"),
              "original_preflight_status": {key: report[key] for key in ("passed", "failure", "E8", "validation_executed")},
              "gpu_model_execution": False,
              "implicit_cuda_device_probe_observed": True,
              "cuda_probe_note": "Torch/transformers initialization attempted a CUDA device availability probe and received driver error 304; all model parameters, inputs and forwards in this probe were on CPU.",
              "frozen_source_modified": False, "original_gate_modified": False,
              "source_identity": {str(path): sha256_file(path) for path in (
                  Path(modeling_qwen3.__file__), Path(masking_utils.__file__),
                  ROOT / "src/diagnostics/general_model_numeric_kernel.py")},
              "passes": {}}
    for name in PASSES:
        path = RUN / "preflight" / name / "scores.jsonl"
        digest = sha256_file(path)
        assert digest == report["files"][f"{name}/scores.jsonl"]
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        all_shapes = defaultdict(list)
        same_shape = defaultdict(list)
        for row in rows:
            for candidate in row["candidates"]:
                for index, value in enumerate(candidate["token_logprobs"]):
                    prefix = tuple(candidate["answer_token_ids"][:index + 1])
                    key = (row["record_id"], prefix)
                    entry = {"candidate_id": candidate["candidate_id"], "token_logprob": value,
                             "target_token_id": candidate["answer_token_ids"][index],
                             "absolute_input_target_position": candidate["prompt_tokens"] + index,
                             "absolute_logit_position": candidate["prompt_tokens"] + index - 1,
                             "prompt_token_ids_sha256": candidate["prompt_token_ids_sha256"],
                             "sequence_tokens": candidate["sequence_tokens"],
                             "padded_sequence_tokens": candidate["padded_sequence_tokens"],
                             "effective_batch_size": candidate["effective_batch_size"],
                             "batch_ordinal": candidate["batch_ordinal"],
                             "batch_member_ordinal": candidate["batch_member_ordinal"]}
                    all_shapes[key].append(entry)
                    same_shape[(*key, candidate["padded_sequence_tokens"], candidate["effective_batch_size"])].append(entry)
        output["passes"][name] = {"scores_sha256": digest, "block_count": len(rows),
                                  "same_prefix_across_shapes": group_summary(all_shapes),
                                  "same_prefix_same_padded_length_and_batch_size": group_summary(same_shape)}
    output["cpu_tiny_qwen3"] = [tiny_probe(dtype) for dtype in (torch.float32, torch.bfloat16)]
    output["interpretation"] = {
        "supported": "Recorded target logprob differences are associated with full sequence/padded batch geometry, including within batch1. Fixed-shape same-prefix repeated groups agree exactly in these sealed scores.",
        "not_established": "No trained-model logits or hidden states were retained; the layer/operator/root cause and contribution of low precision versus backend behavior are not localized by this audit.",
        "baseline_repeat_limit": "Identical batch1 repetition tests one fixed geometry per candidate and does not establish comparable scores for common prefixes across candidate lengths.",
    }
    destination = Path(__file__).with_name("numeric-01-shape-diagnostic.json")
    destination.write_text(json.dumps(output, indent=2, ensure_ascii=True) + "\n")
    print(json.dumps({"output": str(destination), "cpu_tiny_qwen3": output["cpu_tiny_qwen3"]}, indent=2))


if __name__ == "__main__":
    main()
