"""Read-only CPU verification; never loads model weights or writes review records."""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import re
from importlib.metadata import version
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(data):
    return hashlib.sha256(data).hexdigest()


def render_user(query, lexicon, demos):
    """Normative visible input renderer; only text and demo answers are arguments."""
    require(isinstance(query, str) and bool(query), "query must be nonempty text")
    require(isinstance(lexicon, list) and isinstance(demos, list), "material arrays required")
    lexicon_blocks = []
    for i, entry in enumerate(lexicon, 1):
        require(isinstance(entry, str) and bool(entry), "empty lexicon sense")
        lexicon_blocks.append(f"词条{i}：\n{entry}")
    demo_blocks = []
    for i, demo in enumerate(demos, 1):
        require(set(demo) == {"text", "answer"}, "demo renderer accepts only text and answer")
        require(isinstance(demo["text"], str) and bool(demo["text"]), "empty demo")
        require(demo["answer"] in ("有", "无"), "demo answer outside fixed mapping")
        demo_blocks.append(f"示例{i}：\n文本：{demo['text']}\n答案：{demo['answer']}")
    lexicon_text = "\n\n".join(lexicon_blocks) if lexicon_blocks else "（无）"
    demo_text = "\n\n".join(demo_blocks) if demo_blocks else "（无）"
    return f"【参考词典】\n{lexicon_text}\n\n【参考示例】\n{demo_text}\n\n【待判断文本】\n{query}"


def literal_spans(raw, forms):
    spans = []
    for form in forms:
        start = 0
        while True:
            at = raw.find(form, start)
            if at < 0:
                break
            spans.append({"start": at, "end": at + len(form), "text": form})
            start = at + 1
    return spans


def validate_relation(record, validator):
    validator.validate(record)
    for role in ("source", "target"):
        material = record[role]
        require(sha(material["raw_text"].encode("utf-8")) == material["text_sha256"],
                f"{role} text hash mismatch")
    for field in ("sense_fit", "semantic_reference_fit", "rule_fit", "lexical_overlap"):
        dimension = record[field]
        if dimension is None:
            continue
        for role in ("source", "target"):
            raw = record[role]["raw_text"]
            for span in dimension[f"{role}_spans"]:
                require(0 <= span["start"] < span["end"] <= len(raw), "span bounds invalid")
                require(raw[span["start"]:span["end"]] == span["text"], "span text mismatch")
    overlap = record["lexical_overlap"]
    if overlap["value"] in ("present", "absent"):
        forms = overlap["focal_forms"]
        shared = [f for f in forms if f in record["source"]["raw_text"]
                  and f in record["target"]["raw_text"]]
        require(overlap["shared_forms"] == shared, "shared forms mismatch")
        require(overlap["value"] == ("present" if shared else "absent"), "overlap value mismatch")
        for role in ("source", "target"):
            require(overlap[f"{role}_spans"] == literal_spans(record[role]["raw_text"], forms),
                    "literal occurrence inventory mismatch")
    for provenance in (record["provenance"], record["source_quality"]["provenance"]):
        for pointer in provenance["accepted_fields"]:
            value = record
            for component in pointer[1:].split("/"):
                component = component.replace("~1", "/").replace("~0", "~")
                require(isinstance(value, dict) and component in value, "invalid accepted field pointer")
                value = value[component]


def logsumexp(values):
    largest = max(values)
    return largest + math.log(sum(math.exp(v - largest) for v in values))


def synthetic_readout(logits, bound):
    """Three-token arithmetic fixture: [other, 有, 无]; not an inference engine."""
    require(all(math.isfinite(v) for v in logits), "nonfinite logits")
    require(bound is None or (math.isfinite(bound) and bound >= 0), "invalid bound")
    z_yes, z_no = logits[1:]
    normalizer = logsumexp(logits)
    margin = z_no - z_yes
    require(math.isclose(margin, (z_no - normalizer) - (z_yes - normalizer),
                         rel_tol=0, abs_tol=1e-12), "log-probability identity failed")
    prediction = "无" if margin > 0 else "有" if margin < 0 else None
    resolution = "unqualified" if bound is None else (
        "resolved_no" if margin > bound else "resolved_yes" if margin < -bound
        else "numerical_unresolved")
    return {"m": margin, "raw_prediction": prediction, "resolution": resolution,
            "legal_mass": math.exp(logsumexp([z_yes, z_no]) - normalizer)}


def expression_bound(terms, bounds):
    coefficients = {}
    for score_id, coefficient in terms:
        coefficients[score_id] = coefficients.get(score_id, 0) + coefficient
    retained = {key: coefficient for key, coefficient in coefficients.items() if coefficient != 0}
    if any(bounds[key] is None for key in retained):
        return None
    return sum(abs(coefficient) * bounds[key] for key, coefficient in retained.items())


def main():
    from jsonschema import Draft202012Validator, FormatChecker, ValidationError

    manifest = read_json(HERE / "manifest.json")
    for group, base in (("artifacts", HERE), ("external_sources", ROOT), ("tokenizer_sources", ROOT)):
        for entry in manifest[group]:
            path = (base / entry["path"]).resolve()
            require(path.is_relative_to(ROOT), "source path escapes repository")
            data = path.read_bytes()
            require(len(data) == entry["bytes"] and sha(data) == entry["sha256"],
                    f"pinned source changed: {entry['path']}")
    actual = sorted(p.name for p in HERE.iterdir() if p.is_file() and p.name != "manifest.json")
    require(actual == sorted(entry["path"] for entry in manifest["artifacts"]), "artifact inventory differs")
    for document in HERE.glob("*.md"):
        for target in re.findall(r"\[[^\]]+\]\(([^)]+)\)", document.read_text()):
            require((document.parent / target).is_file(), f"broken document link: {target}")

    system = (HERE / "model-task.txt").read_text(encoding="utf-8")
    parent_prompt = ROOT / manifest["model_prompt_source"]
    require(system.encode("utf-8") == parent_prompt.read_bytes(), "model task differs from approved v2")
    require(system.endswith("\n") and "\r" not in system, "model task newline convention changed")
    spec = read_json(HERE / "scoring-spec.json")
    require(spec["candidate_tokens"] == {"有": 18830, "无": 42192}, "candidate identity changed")
    require(spec["classification_threshold"] == 0 and not spec["include_eos"]
            and not spec["include_quotes"] and not spec["ncc_subtraction"], "primary score rules changed")
    require(spec["primary_margin"] == "z[42192]-z[18830]", "primary orientation changed")

    schema = read_json(HERE / "relation-record.schema.json")
    Draft202012Validator.check_schema(schema)
    validator = Draft202012Validator(schema, format_checker=FormatChecker())
    fixture = read_json(HERE / "relation-examples.json")
    require(fixture["scientific_material"] is False and fixture["human_adoption"] is False,
            "fixtures must not become scientific or human records")
    rows = fixture["records"]
    identities = {}
    relation_ids = set()
    for row in rows:
        validate_relation(row, validator)
        require(row["relation_id"] not in relation_ids, "duplicate relation id")
        relation_ids.add(row["relation_id"])
        for role in ("source", "target"):
            material = row[role]
            key = material["material_id"]
            require(key not in identities or identities[key] == material, "material identity conflict")
            identities[key] = material
        for p in (row["provenance"], row["source_quality"]["provenance"]):
            require(p["adoption"] == "none" and p["review_kind"] == "ai_note", "fixture falsely adopted")
    for row in rows:
        require(set(row["introduced_lexicon_relation_ids"]) <= relation_ids, "unresolved lexicon edge")
    require({row["relation_kind"] for row in rows} == {
        "lexicon_to_query", "lexicon_to_demo", "demo_to_query"}, "missing relationship coverage")
    fit_values = {row[field]["value"] for row in rows
                  for field in ("sense_fit", "semantic_reference_fit", "rule_fit") if row[field] is not None}
    require(fit_values == {"direct", "partial", "none", "unclear", None}, "missing rating coverage")

    invalid = []
    bad = copy.deepcopy(rows[0]); bad["source"]["text_sha256"] = "0" * 64
    invalid.append(("changed_text_hash", bad))
    bad = copy.deepcopy(rows[0]); bad["sense_fit"]["target_spans"][0]["end"] += 1
    invalid.append(("invalid_unicode_span", bad))
    bad = copy.deepcopy(rows[0]); bad["lexical_overlap"]["value"] = "absent"; bad["lexical_overlap"]["shared_forms"] = []
    invalid.append(("contradictory_overlap", bad))
    bad = copy.deepcopy(rows[0]); bad["provenance"]["adoption"] = "bulk"
    invalid.append(("invented_human_adoption", bad))
    bad = copy.deepcopy(rows[4]); bad["semantic_reference_fit"]["limitation"] = None
    invalid.append(("partial_without_scope", bad))
    bad = copy.deepcopy(rows[0]); bad["model_prediction"] = "无"
    invalid.append(("model_result_in_relation_record", bad))
    rejected = []
    for name, bad in invalid:
        try:
            validate_relation(bad, validator)
        except (ValueError, ValidationError):
            rejected.append(name)
        else:
            raise ValueError(f"invalid relation accepted: {name}")

    arithmetic = [
        ([0.0, 1.0, 3.0], 0.25, 2.0, "无", "resolved_no"),
        ([0.0, 3.0, 1.0], 0.25, -2.0, "有", "resolved_yes"),
        ([0.0, 2.0, 2.0], 0.25, 0.0, None, "numerical_unresolved"),
        ([0.0, 1.0, 1.25], 0.25, 0.25, "无", "numerical_unresolved"),
        ([0.0, 1.0, 3.0], None, 2.0, "无", "unqualified"),
        ([100.0, 1.0, 3.0], 0.25, 2.0, "无", "resolved_no"),
    ]
    arithmetic_results = []
    for logits, bound, margin, prediction, resolution in arithmetic:
        result = synthetic_readout(logits, bound)
        require((result["m"], result["raw_prediction"], result["resolution"]) ==
                (margin, prediction, resolution), "synthetic score expectation differs")
        arithmetic_results.append(result)
    require(arithmetic_results[-1]["legal_mass"] < 1e-30, "pair preference confused with legal mass")
    require(synthetic_readout([1000.0, 1001.0, 1003.0], 0.25)["m"] == 2.0,
            "shared logit shift changed margin")
    require(expression_bound([("a", 1), ("b", -1)], {"a": 0.25, "b": 0.5}) == 0.75,
            "paired bound propagation failed")
    require(expression_bound([("a", 1), ("a", -1), ("b", 1)], {"a": None, "b": 0.5}) == 0.5,
            "physical cancellation must precede missing-bound propagation")
    require(expression_bound([("a", 1)], {"a": None}) is None, "unknown bound became zero")
    try:
        synthetic_readout([0.0, float("nan"), 1.0], None)
    except ValueError:
        pass
    else:
        raise ValueError("nonfinite scores accepted")

    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["HF_HUB_OFFLINE"] = "1"
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(ROOT / spec["local_model_path"]),
                                            local_files_only=True, trust_remote_code=False)
    for label, token_id in spec["candidate_tokens"].items():
        require(tokenizer.encode(label, add_special_tokens=False) == [token_id], "candidate not one token")
    prompt_fixture = read_json(HERE / "prompt-fixtures.json")
    require(prompt_fixture["scientific_material"] is False and prompt_fixture["human_adoption"] is False,
            "prompt fixtures falsely marked as adopted scientific materials")
    prompt_checks = []
    for condition in prompt_fixture["conditions"]:
        user = render_user(condition["query"], condition["lexicon"], condition["demos"])
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False,
                                              add_generation_prompt=True, enable_thinking=False)
        token_ids = tokenizer.encode(prompt, add_special_tokens=False)
        direct_ids = tokenizer.apply_chat_template(messages, tokenize=True,
                                                  add_generation_prompt=True, enable_thinking=False)
        require(direct_ids == token_ids, "rendered and direct template tokenization differ")
        for label, token_id in spec["candidate_tokens"].items():
            require(tokenizer.encode(prompt + label, add_special_tokens=False) == token_ids + [token_id],
                    f"candidate boundary unstable: {condition['id']} / {label}")
        prompt_checks.append({"condition_id": condition["id"], "prompt_sha256": sha(prompt.encode()),
                              "prompt_tokens": len(token_ids), "last_input_token_index": len(token_ids)-1,
                              "next_token_position": len(token_ids), "both_candidates_stable": True})
    boundary_text = "  原文🙂e\u0301\n尾行  "
    require(render_user(boundary_text, [], []).endswith(boundary_text), "renderer normalized raw text")

    print(json.dumps({
        "status": "passed", "scope": "protocol_files_schema_rendering_tokenizer_and_synthetic_arithmetic_only",
        "manifest_sha256": sha((HERE / "manifest.json").read_bytes()),
        "artifact_count": len(manifest["artifacts"]),
        "external_source_count": len(manifest["external_sources"]),
        "tokenizer_source_count": len(manifest["tokenizer_sources"]),
        "relation_fixtures": len(rows), "negative_schema_checks_rejected": rejected,
        "synthetic_arithmetic_cases": len(arithmetic), "shared_term_bound_checks": 3,
        "model_task_sha256": sha(system.encode()),
        "chat_template_sha256": sha(tokenizer.chat_template.encode()),
        "candidate_tokens": spec["candidate_tokens"], "prompt_checks": prompt_checks,
        "candidate_boundary_checks": len(prompt_checks) * 2,
        "runtime_versions": {name: version(name) for name in ("transformers", "tokenizers", "jsonschema")},
        "weights_loaded": False, "model_forward_performed": False, "gpu_qualification": False,
        "scientific_material_frame_checked": False, "human_records_created_or_modified": False,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
