"""Retokenize frozen eight-condition semantics for registered replication models."""

from __future__ import annotations

import copy
import hashlib
import json
from functools import lru_cache
from pathlib import Path

from data.stage1_data import canonical_json_sha256, sha256_file
from diagnostics import general_model_contexts as renderer
from diagnostics.general_model_contexts import render_condition
from diagnostics.general_model_coverage_package import (
    CONDITIONS, FROZEN_ENTRY_COUNT, FROZEN_LEXICON_SHA256, MAX_SEQUENCE_TOKENS,
    PADDING_EXTRA, RENDER_CONDITIONS, TASKS, _indexed, select_boundary_queries,
)
from diagnostics.general_model_numeric_analysis import candidate_catalog
from diagnostics.general_model_package import ROOT, PackageError, read_json, read_jsonl, repo_path


VERSION = "general-model-coverage-replication-inputs/v1"
MODEL_KEYS = ("qwen3-14b", "qwen3.8-27b")
COMPACT_MATERIAL_VERSION = "ordinary-materials-compact-complete-sentences/v1"
COMPACT_SENTENCES = ("\u706f\u4eae\u3002", "\u706f\u706d\u3002", "\u95e8\u5173\u7740\u3002", "\u706f\u5f00\u7740\u3002")


def compact_placebo_policy():
    material = {"base_sentences": list(COMPACT_SENTENCES), "append_sentences": list(renderer.NEUTRAL_SENTENCES),
                "neutral_terms": list(renderer.NEUTRAL_TERMS), "task_schemas": renderer._SCHEMAS}
    return {"material_version": COMPACT_MATERIAL_VERSION, "material_sha256": canonical_json_sha256(material),
            **material, "activation": "original-PL-or-PD-invalid-only",
            "selection": "minimum-complete-block-absolute-token-error-then-base-sentence-index",
            "base_layout": "one-fixed-complete-sentence-per-source-unit-or-sense",
            "append_algorithm": "frozen-general-model-contexts._placebo/v1",
            "tolerance": {"absolute_tokens": 8, "relative": .02},
            "source_units_and_task_placeholders_preserved": True, "whole_sentences_only": True}


def _compact_material(task, condition, hits, demos, addition, sentence):
    blocks = []
    if condition == "PL":
        for index, hit in enumerate(hits):
            lines = [f"\u8bcd\u6761\uff1a{renderer.NEUTRAL_TERMS[index % len(renderer.NEUTRAL_TERMS)]}"]
            for sense_index, _ in enumerate(hit.get("senses") or [{}], start=1):
                definition = sentence + (addition if index == 0 and sense_index == 1 else "")
                lines.extend([f"\u4e49\u9879 {sense_index}\uff1a", '\u7c7b\u522b\uff1a["<\u7c7b\u522b>"]',
                              f"\u5b9a\u4e49\uff1a{definition}"])
            blocks.append("\n".join(lines))
        return "\u8bcd\u5178\u53c2\u8003\uff1a\n" + "\n\n".join(blocks)
    for index, demo in enumerate(demos):
        content = sentence + (addition if index == 0 else "")
        output = json.loads(renderer._SCHEMAS[task])
        if task == "extraction":
            output *= len(demo["quadruples"])
        blocks.append(f"\u793a\u4f8b {index + 1}\n\u6587\u672c\uff1a{content}\n\u8f93\u51fa\uff1a{renderer._json(output)}")
    return "\u53c2\u8003\u793a\u4f8b\uff1a\n" + "\n\n".join(blocks)


def render_with_placebo_adapter(task, condition, query, hits, demos, *, token_count, placebo_tolerance):
    rendered = render_condition(task, condition, query, hits, demos,
                                token_count=token_count, placebo_tolerance=placebo_tolerance)
    if condition not in {"PL", "PD"} or rendered["control_status"] != "invalid":
        return rendered
    if placebo_tolerance != {"absolute_tokens": 8, "relative": .02}:
        raise PackageError("compact placebo adapter cannot change the registered tolerance")
    if condition == "PL":
        source, _ = renderer._render_lexicon(hits, "Full")
    else:
        outputs = [renderer.project_gold(demo["quadruples"])[task] for demo in demos]
        source = renderer._render_demo_blocks(task, demos, outputs, "Full")
    choices = []
    for index, sentence in enumerate(COMPACT_SENTENCES):
        block, trace = renderer._placebo(source, lambda addition, sentence=sentence: _compact_material(
            task, condition, hits, demos, addition, sentence), token_count, placebo_tolerance)
        choices.append((block, trace, index))
    block, trace, index = min(choices, key=lambda choice: (choice[1]["absolute_difference"], choice[2]))
    policy = compact_placebo_policy()
    rendered = copy.deepcopy(rendered)
    original = rendered["trace"]["placebo"]
    rendered["trace"]["placebo"] = {
        **trace, "tolerance": dict(placebo_tolerance), "material_version": policy["material_version"],
        "material_sha256": policy["material_sha256"], "compact_adapter_applied": True,
        "original_failure": original, "selected_base_sentence_index": index, "selection": policy["selection"],
        "compact_candidates": [{"base_sentence_index": i, **{key: candidate[key] for key in
            ("source_tokens", "placebo_tokens", "allowed_difference", "absolute_difference", "construction_steps", "status")}}
            for _, candidate, i in choices]}
    rendered["control_status"], rendered["control_valid"] = trace["status"], trace["status"] == "valid"
    assessment = rendered["trace"]["control_assessment"]
    assessment.update(construction_valid=rendered["control_valid"], intervention_effective=bool(source and block != source),
                      kind="valid" if rendered["control_valid"] else "control-construction-failed")
    rendered["trace"]["reasons"] = [reason for reason in rendered["trace"]["reasons"] if reason != original["reason"]]
    if not rendered["control_valid"]:
        rendered["trace"]["reasons"].append(trace["reason"])
    prefix = "lexicon" if condition == "PL" else "demo"
    rendered["trace"]["injected_blocks"].update({f"{prefix}_sha256": hashlib.sha256(block.encode()).hexdigest(),
                                                f"{prefix}_tokens": trace["placebo_tokens"]})
    query_block = "\u5f85\u5224\u65ad\u6587\u672c\uff08JSON \u5b57\u7b26\u4e32\uff09\uff1a\n" + renderer._json(query)
    rendered["messages"][1]["content"] = "\n\n".join([block, query_block])
    return rendered


def _catalog(tokenizer):
    result = candidate_catalog()
    for candidates in result.values():
        for candidate in candidates:
            answer = candidate["canonical_answer"]
            ids = tokenizer.encode(answer, add_special_tokens=False)
            if not ids or any(type(value) is not int or value < 0 for value in ids):
                raise PackageError("replication canonical answer tokenization is invalid")
            candidate.update(answer_token_ids=ids, answer_tokens=len(ids),
                             answer_token_ids_sha256=canonical_json_sha256(ids),
                             answer_sha256=hashlib.sha256(answer.encode()).hexdigest())
    return result


def _eos_metadata(tokenizer, generation_eos_token_ids):
    eos, pad = tokenizer.eos_token_id, tokenizer.pad_token_id
    if any(type(value) is not int or value < 0 for value in (eos, pad)):
        raise PackageError("replication requires explicit tokenizer EOS and padding IDs")
    supplied = generation_eos_token_ids
    if type(supplied) is int:
        supplied = [supplied]
    if supplied is None:
        supplied = []
    if not isinstance(supplied, (list, tuple)) or any(type(value) is not int or value < 0 for value in supplied):
        raise PackageError("invalid generation EOS metadata")
    return eos, pad, {
        "canonical_eos_policy": "tokenizer.eos_token_id; answer-only scores are primary",
        "canonical_eos_token_id": eos, "pad_token_id": pad,
        "generation_eos_token_ids": list(supplied),
        "canonical_eos_in_generation_stop_ids": eos in supplied,
        "multiple_generation_stop_ids_are_not_canonical_score_ambiguity": True,
    }


def build_contexts(parent_plan, old_contexts, package_dir, tokenizer, *, model_key,
                   generation_eos_token_ids=None):
    """Return contexts/catalog/inventory/EOS/pad without matching or gold access.

    parent_plan is the verified 8B coverage plan with its effective plan_dir.
    package_dir is the original fit/retrieval package, not a model directory.
    """
    if model_key not in MODEL_KEYS:
        raise PackageError("replication model is outside the registered roster")
    parent_dir, package_dir = Path(parent_plan["plan_dir"]), Path(package_dir)
    inventory_path = parent_dir / "resource_inventory.json"
    parent_contexts_path = parent_dir / "contexts.dev.jsonl"
    source_inventory_sha = sha256_file(inventory_path)
    if (source_inventory_sha != parent_plan["input_files"]["resource_inventory.json"]
            or sha256_file(parent_contexts_path) != parent_plan["input_files"]["contexts.dev.jsonl"]):
        raise PackageError("frozen 8B resource inventory changed")
    source_inventory = read_json(inventory_path)
    resources = _indexed(source_inventory["per_query"], "query_id", "frozen coverage inventory")
    frame = _indexed(parent_plan["frame"], "query_id", "frozen coverage frame")
    old = _indexed(old_contexts, "record_id", "frozen 8B contexts")
    expected = {f"{qid}:{task}:{condition}" for qid in frame for task in TASKS for condition in CONDITIONS}
    if set(old) != expected or set(resources) != set(frame):
        raise PackageError("replication source query or eight-condition frame changed")
    descriptors = _indexed(parent_plan["blocks"], "record_id", "parent block descriptors")
    if set(descriptors) != expected:
        raise PackageError("replication parent descriptors are incomplete")
    for identifier, row in old.items():
        if (canonical_json_sha256({k: v for k, v in row.items() if k != "context_sha256"}) != row["context_sha256"]
                or any(row.get(key) != value for key, value in descriptors[identifier].items())):
            raise PackageError("replication parent context identity changed")
    sources = source_inventory["source_sha256"]
    paths = {name: package_dir / name for name in
             ("config.resolved.json", "retrieval.dev.jsonl", "lexicon.dev.jsonl", "fit_catalog.jsonl")}
    for path in paths.values():
        if str(path) not in sources or sha256_file(path) != sources[str(path)]:
            raise PackageError("replication original package payload changed")
    config = read_json(paths["config.resolved.json"])
    if config["controls"] != {"absolute_tokens": 8, "relative": .02}:
        raise PackageError("replication placebo tolerance changed")
    lexicon_path = repo_path(ROOT, config["sources"]["lexicon"])
    if sha256_file(lexicon_path) != FROZEN_LEXICON_SHA256:
        raise PackageError("replication frozen lexicon changed")
    lexicon = read_json(lexicon_path)
    entries = _indexed(lexicon["terms"], "lexicon_id", "frozen lexicon")
    if len(entries) != FROZEN_ENTRY_COUNT:
        raise PackageError("replication frozen lexicon count changed")
    retrieval = _indexed(read_jsonl(paths["retrieval.dev.jsonl"]), "query_id", "frozen retrieval")
    query_traces = _indexed(read_jsonl(paths["lexicon.dev.jsonl"]), "query_id", "frozen query trace")
    fit = _indexed(read_jsonl(paths["fit_catalog.jsonl"]), "id", "frozen fit")
    if set(retrieval) != set(frame) or set(query_traces) != set(frame):
        raise PackageError("replication query resource membership changed")
    eos, pad, eos_metadata = _eos_metadata(tokenizer, generation_eos_token_ids)
    catalog = _catalog(tokenizer)
    qids = sorted(frame, key=int)
    inventory_rows = []
    material = {}
    for qid in qids:
        source = resources[qid]
        lq, union = source["lq_ids"], source["union_ids"]
        if (lq != sorted(set(lq)) or union != sorted(set(union)) or not set(lq) <= set(union)
                or not set(union) <= set(entries) or frame[qid]["lex_hit"] != bool(lq)):
            raise PackageError("frozen coverage membership or ID ordering changed")
        demos = retrieval[qid]["demos"]
        if [str(row["id"]) for row in demos] != source["demo_ids"] or len(demos) != 10:
            raise PackageError("replication frozen demo membership/order changed")
        for ordinal, demo in enumerate(demos):
            identifier = str(demo["id"])
            if (identifier not in fit or demo["prompt_rank"] != ordinal
                    or any(demo[key] != fit[identifier][key] for key in ("content", "quadruples"))):
                raise PackageError("replication frozen demo content/answer changed")
        for task in TASKS:
            if (old[f"{qid}:{task}:CLnew"]["trace"]["source_lexicon_ids"] != union
                    or old[f"{qid}:{task}:CLq"]["trace"]["source_lexicon_ids"] != lq):
                raise PackageError("frozen context and resource ID inventory disagree")
        material[qid] = (query_traces[qid]["trace"]["query"], [entries[i] for i in lq],
                         [entries[i] for i in union], demos)
        metadata = copy.deepcopy(source)
        metadata.update(dictionary_tokens=0, lq_dictionary_tokens=0, max_full_sequence_tokens=0,
                        max_prompt_tokens=0, condition_tokens={})
        inventory_rows.append(metadata)
    by_query = {row["query_id"]: row for row in inventory_rows}

    @lru_cache(maxsize=8192)
    def token_count(text):
        return len(tokenizer.encode(text, add_special_tokens=False))

    contexts, boundaries, adapted_records = [], set(), []
    boundary_checks = unchanged_messages = 0
    for task in TASKS:
        for condition in CONDITIONS:
            for qid in qids:
                query, lq, union, demos = material[qid]
                hits = lq if condition in ("CLq", "CLqD") else union
                rendered = render_with_placebo_adapter(task, RENDER_CONDITIONS[condition], query, hits, demos,
                                                      token_count=token_count, placebo_tolerance=config["controls"])
                if rendered["control_status"] != "valid" or not rendered["control_valid"]:
                    raise PackageError(f"replication placebo construction failed: {qid}:{task}:{condition}")
                if rendered["trace"].get("placebo", {}).get("compact_adapter_applied"):
                    adapted_records.append(f"{qid}:{task}:{condition}")
                if condition not in ("PLnew", "PD"):
                    if rendered["messages"] != old[f"{qid}:{task}:{condition}"]["messages"]:
                        raise PackageError("replication changed non-placebo message semantics")
                    unchanged_messages += 1
                rendered["trace"].update(condition=condition, coverage_renderer_version=VERSION,
                                           resource_policy="reuse-frozen-8B-global-Lq-union-Ld-IDs/v1",
                                           source_correspondence_visible=False,
                                           source_context_sha256=old[f"{qid}:{task}:{condition}"]["context_sha256"])
                prompt = tokenizer.apply_chat_template(rendered["messages"], tokenize=False,
                                                       add_generation_prompt=True, enable_thinking=False)
                token_ids = tokenizer.encode(prompt, add_special_tokens=False)
                prompt_sha = hashlib.sha256(prompt.encode()).hexdigest()
                max_answer = max(candidate["answer_tokens"] for candidate in catalog[task])
                full_length = len(token_ids) + max_answer + 1
                if full_length + PADDING_EXTRA > MAX_SEQUENCE_TOKENS:
                    raise PackageError(f"replication input plus padding exceeds 8192: {qid}:{task}:{condition}")
                identity = (prompt_sha, task)
                if identity not in boundaries:
                    for candidate in catalog[task]:
                        if (tokenizer.encode(prompt + candidate["canonical_answer"], add_special_tokens=False)
                                != token_ids + candidate["answer_token_ids"]):
                            raise PackageError("replication prompt/candidate token boundary changed")
                    boundaries.add(identity)
                boundary_checks += len(catalog[task])
                context = {"query_id": qid, "task": task, "condition": condition,
                           "record_id": f"{qid}:{task}:{condition}", "model_key": model_key,
                           "messages": rendered["messages"], "trace": rendered["trace"],
                           "control_valid": True, "control_status": "valid", "prompt_text": prompt,
                           "prompt_sha256": prompt_sha, "prompt_tokens": len(token_ids),
                           "prompt_token_ids_sha256": canonical_json_sha256(token_ids),
                           "max_new_tokens": max_answer + 1, "overflow": False}
                context["context_sha256"] = canonical_json_sha256(context)
                contexts.append(context)
                metadata = by_query[qid]
                lex_tokens = rendered["trace"]["injected_blocks"]["lexicon_tokens"]
                metadata["condition_tokens"][f"{task}:{condition}"] = {
                    "prompt_tokens": len(token_ids), "dictionary_tokens": lex_tokens,
                    "max_full_sequence_tokens": full_length}
                metadata["max_full_sequence_tokens"] = max(metadata["max_full_sequence_tokens"], full_length)
                metadata["max_prompt_tokens"] = max(metadata["max_prompt_tokens"], len(token_ids))
                if condition == "CLnew":
                    metadata["dictionary_tokens"] = lex_tokens
                elif condition == "CLq":
                    metadata["lq_dictionary_tokens"] = lex_tokens
    original = {name: copy.deepcopy(parent_plan["cohorts"][name]) for name in ("regression", "validation")}
    boundary_ids, boundary_selection = select_boundary_queries(inventory_rows, original)
    if (sha256_file(inventory_path) != source_inventory_sha
            or sha256_file(parent_contexts_path) != parent_plan["input_files"]["contexts.dev.jsonl"]
            or sha256_file(lexicon_path) != FROZEN_LEXICON_SHA256
            or any(sha256_file(path) != sources[str(path)] for path in paths.values())):
        raise PackageError("replication source changed during CPU construction")
    inventory = {"schema_version": VERSION, "model_key": model_key, "per_query": inventory_rows,
                 "parent_plan_id": parent_plan["plan_id"], "source_inventory_sha256": source_inventory_sha,
                 "source_contexts_sha256": parent_plan["input_files"]["contexts.dev.jsonl"],
                 "frozen_resource_payload_sha256": {str(path): sources[str(path)] for path in paths.values()},
                 "lexicon_sha256": FROZEN_LEXICON_SHA256, "catalog_sha256": canonical_json_sha256(catalog),
                 "conditions": list(CONDITIONS), "tasks": list(TASKS), "eos": eos_metadata,
                 "placebo_adapter_policy": compact_placebo_policy(),
                 "placebo_adapter_applied_record_ids": adapted_records,
                 "boundary_checks": boundary_checks, "unique_prompt_task_boundary_checks": len(boundaries),
                 "counts": {**copy.deepcopy(source_inventory["counts"]), "contexts": len(contexts),
                            "candidates": boundary_checks, "unchanged_non_placebo_messages": unchanged_messages},
                 "limits": {"max_sequence_tokens": MAX_SEQUENCE_TOKENS, "padding_extra": PADDING_EXTRA,
                            "max_union_count": max(row["union_count"] for row in inventory_rows),
                            "max_dictionary_tokens": max(row["dictionary_tokens"] for row in inventory_rows),
                            "max_prompt_tokens": max(row["max_prompt_tokens"] for row in inventory_rows),
                            "max_full_sequence_tokens": max(row["max_full_sequence_tokens"] for row in inventory_rows)},
                 "original_cohorts": original, "boundary_query_ids": boundary_ids,
                 "boundary_selection": boundary_selection,
                 "gates": {"non_placebo_messages_byte_identical": True, "frozen_demo_payload_reused": True,
                           "frozen_lexical_ID_sets_reused": True, "control_construction": True,
                           "all_candidate_boundaries": True, "all_sequences_with_eos_and_padding_within_budget": True},
                 "matching_reexecuted": False, "retrieval_reexecuted": False, "query_gold_loaded": False,
                 "test_content_read": False, "model_forward_executed": False}
    return contexts, catalog, inventory, eos, pad
