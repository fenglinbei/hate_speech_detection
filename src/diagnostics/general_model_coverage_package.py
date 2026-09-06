"""Gold-free, CPU-only inputs for the globally deduplicated Lq union Ld study."""

from __future__ import annotations

import copy
import hashlib
from functools import lru_cache
from pathlib import Path

from data.stage1_data import canonical_json_sha256, sha256_file
from diagnostics.general_model_contexts import render_condition
from diagnostics.general_model_numeric_analysis import candidate_catalog
from diagnostics.general_model_package import ROOT, PackageError, read_json, read_jsonl, repo_path
from rag.controlled_lexicon_matcher import ControlledLexiconMatcher


VERSION = "general-model-coverage-package/v1"
CONDITIONS = ("C0", "CLnew", "CD", "CLDnew", "PLnew", "PD", "CLq", "CLqD")
TASKS = ("hate", "group")
RENDER_CONDITIONS = dict(zip(CONDITIONS, ("C0", "CL", "CD", "CLD", "PL", "PD", "CL", "CLD"), strict=True))
FROZEN_LEXICON_SHA256 = "31240193eeba29f712560e2e80e89ee9dd5c3c969aa180d10b95bf2451882385"
FROZEN_ENTRY_COUNT = 833
MAX_SEQUENCE_TOKENS = 8192
PADDING_EXTRA = 64


def _indexed(rows: list[dict], field: str, name: str) -> dict[str, dict]:
    result = {}
    for row in rows:
        value = row.get(field)
        if value is None or not str(value).strip() or str(value) in result:
            raise PackageError(f"{name}: missing or duplicate {field}")
        result[str(value)] = row
    return result


def _hits_from_frozen(hits: list[dict], entries: dict[str, dict]) -> list[dict]:
    selected = _indexed(hits, "lexicon_id", "matched entries")
    for identifier, hit in selected.items():
        if identifier not in entries or any(hit.get(k) != v for k, v in entries[identifier].items()):
            raise PackageError("matched entry payload differs from the frozen lexicon")
    return [entries[identifier] for identifier in sorted(selected)]


def select_boundary_queries(per_query: list[dict], cohorts: dict) -> tuple[list[str], list[dict]]:
    """Choose each representative first, then deduplicate without replacements."""
    if not per_query:
        raise PackageError("cannot select boundaries from an empty frame")
    _indexed(per_query, "query_id", "boundary frame")
    original = [str(qid) for name in ("regression", "validation") for qid in cohorts[name]]
    if len(original) != len(set(original)):
        raise PackageError("original preflight cohorts overlap or contain duplicates")
    frame_ids = {str(row["query_id"]) for row in per_query}
    if not set(original).issubset(frame_ids):
        raise PackageError("original preflight is outside the new frame")
    selected, records = [], []
    specifications = (
        ("empty_union", "union_count", True),
        ("maximum_union_count", "union_count", False),
        ("maximum_dictionary_tokens", "dictionary_tokens", False),
        ("maximum_complete_input_tokens", "max_full_sequence_tokens", False),
    )
    for name, field, empty in specifications:
        target = 0 if empty else max(row[field] for row in per_query)
        eligible = sorted(str(row["query_id"]) for row in per_query if row[field] == target)
        representative = eligible[0] if eligible else None
        reason = "no-empty-union" if representative is None else "selected"
        if representative in original:
            reason = "already-in-original-preflight"
        elif representative in selected:
            reason = "already-selected-boundary"
        elif representative is not None:
            selected.append(representative)
        records.append({"boundary": name, "field": field, "value": target,
                        "eligible_query_ids": eligible, "representative_query_id": representative,
                        "included": reason == "selected", "reason": reason})
    return selected, records


def _catalog(parent_plan: dict, tokenizer) -> dict:
    catalog = candidate_catalog()
    for candidates in catalog.values():
        for candidate in candidates:
            answer = candidate["canonical_answer"]
            token_ids = tokenizer.encode(answer, add_special_tokens=False)
            if not token_ids or any(type(value) is not int or value < 0 for value in token_ids):
                raise PackageError("invalid canonical answer tokenization")
            candidate.update(answer_token_ids=token_ids, answer_tokens=len(token_ids),
                             answer_token_ids_sha256=canonical_json_sha256(token_ids),
                             answer_sha256=hashlib.sha256(answer.encode()).hexdigest())
    if catalog != parent_plan["catalog"]:
        raise PackageError("canonical candidate catalog changed")
    if (type(tokenizer.eos_token_id) is not int or tokenizer.eos_token_id < 0
            or tokenizer.eos_token_id != parent_plan["eos_token_id"]
            or tokenizer.pad_token_id != parent_plan["pad_token_id"]):
        raise PackageError("EOS or padding token identity changed")
    return catalog


def build_contexts(parent_plan: dict, old_contexts: list[dict], package_dir: Path,
                   tokenizer) -> tuple[list[dict], dict]:
    """Return new contexts and inventory; never open query gold or write files."""
    package_dir = Path(package_dir)
    names = ("config.resolved.json", "retrieval.dev.jsonl", "lexicon.dev.jsonl", "fit_catalog.jsonl")
    sources = {str(package_dir / name): sha256_file(package_dir / name) for name in names}
    config = read_json(package_dir / "config.resolved.json")
    lexicon_path = repo_path(ROOT, config["sources"]["lexicon"])
    sources[str(lexicon_path)] = sha256_file(lexicon_path)
    if (sources[str(lexicon_path)] != FROZEN_LEXICON_SHA256
            or config["sources"]["lexicon_sha256"] != FROZEN_LEXICON_SHA256):
        raise PackageError("the registered frozen lexicon identity changed")
    lexicon = read_json(lexicon_path)
    entries = _indexed(lexicon["terms"], "lexicon_id", "frozen lexicon")
    if len(entries) != FROZEN_ENTRY_COUNT:
        raise PackageError("the frozen lexicon entry count changed")
    if config["controls"] != {"absolute_tokens": 8, "relative": 0.02}:
        raise PackageError("neutral-control token tolerance changed")
    if config["retrieval"]["demo_top_k"] != 10:
        raise PackageError("the frozen ten-demo policy changed")
    matcher = ControlledLexiconMatcher(lexicon["terms"], lexicon_sha256=FROZEN_LEXICON_SHA256,
                                       policy_sha256=lexicon["matcher_policy_sha256"])
    retrieval = _indexed(read_jsonl(package_dir / "retrieval.dev.jsonl"), "query_id", "retrieval")
    lex_traces = _indexed(read_jsonl(package_dir / "lexicon.dev.jsonl"), "query_id", "query lexicon")
    fit = _indexed(read_jsonl(package_dir / "fit_catalog.jsonl"), "id", "fit catalog")
    frame = _indexed(parent_plan["frame"], "query_id", "parent frame")
    if set(retrieval) != set(frame) or set(lex_traces) != set(frame):
        raise PackageError("resource frames differ from the parent development frame")
    old = _indexed(old_contexts, "record_id", "parent contexts")
    expected_old = {f"{qid}:{task}:{condition}" for qid in frame for task in TASKS
                    for condition in ("C0", "CL", "CD", "CLD", "PL", "PD")}
    if set(old) != expected_old:
        raise PackageError("parent context matrix differs")
    for row in old_contexts:
        if canonical_json_sha256({k: v for k, v in row.items() if k != "context_sha256"}) != row["context_sha256"]:
            raise PackageError("parent context hash changed")
    catalog = _catalog(parent_plan, tokenizer)

    @lru_cache(maxsize=8192)
    def token_count(text: str) -> int:
        return len(tokenizer.encode(text, add_special_tokens=False))

    resources, demo_matches, per_query = {}, {}, []
    qids = sorted(frame, key=int)
    for qid in qids:
        trace = lex_traces[qid]["trace"]
        query = trace["query"]
        replay = matcher.match(query)
        if replay != trace:
            raise PackageError("frozen query matching does not replay")
        lq = _hits_from_frozen(trace["selected_hits"], entries)
        if frame[qid]["lex_hit"] != bool(lq):
            raise PackageError("Lq-hit membership changed")
        demos = retrieval[qid]["demos"]
        demo_ids = [str(demo["id"]) for demo in demos]
        if len(demos) != 10 or len(set(demo_ids)) != 10:
            raise PackageError("each frozen D must contain ten distinct examples")
        ld_ids, matching_demos = set(), 0
        for ordinal, demo in enumerate(demos):
            did = str(demo["id"])
            if did not in fit or any(demo[k] != fit[did][k] for k in ("content", "quadruples")):
                raise PackageError("selected demo text or answer differs from fit")
            if demo.get("prompt_rank") != ordinal:
                raise PackageError("frozen demo order changed")
            if did not in demo_matches:
                demo_matches[did] = matcher.match(demo["content"])
            matched = _hits_from_frozen(demo_matches[did]["selected_hits"], entries)
            matching_demos += bool(matched)
            ld_ids.update(hit["lexicon_id"] for hit in matched)
        lq_ids = {hit["lexicon_id"] for hit in lq}
        union_ids = sorted(lq_ids | ld_ids)
        resources[qid] = (query, lq, [entries[identifier] for identifier in union_ids], demos)
        per_query.append({
            "query_id": qid, "lex_hit": bool(lq), "lq_hit": bool(lq), "ld_hit": bool(ld_ids),
            "union_hit": bool(union_ids), "lq_ids": sorted(lq_ids), "ld_ids": sorted(ld_ids),
            "union_ids": union_ids, "intersection_ids": sorted(lq_ids & ld_ids),
            "lq_only_ids": sorted(lq_ids - ld_ids), "ld_only_ids": sorted(ld_ids - lq_ids),
            "lq_count": len(lq_ids), "ld_count": len(ld_ids), "union_count": len(union_ids),
            "demo_ids": demo_ids, "matching_demo_count": matching_demos,
            "demo_match_ids": {did: sorted(hit["lexicon_id"] for hit in demo_matches[did]["selected_hits"])
                               for did in demo_ids},
            "dictionary_tokens": 0, "lq_dictionary_tokens": 0, "max_full_sequence_tokens": 0,
            "max_prompt_tokens": 0, "condition_tokens": {},
        })
    inventory_by_id = {row["query_id"]: row for row in per_query}
    contexts, verified_boundaries = [], set()
    boundary_checks = 0
    for task in TASKS:
        for condition in CONDITIONS:
            renderer_condition = RENDER_CONDITIONS[condition]
            for qid in qids:
                query, lq, union, demos = resources[qid]
                hits = lq if condition in ("CLq", "CLqD") else union
                rendered = render_condition(task, renderer_condition, query, hits, demos,
                                            token_count=token_count, placebo_tolerance=config["controls"])
                if not rendered["control_valid"] or rendered["control_status"] != "valid":
                    raise PackageError(f"neutral-control construction failed: {qid}:{task}:{condition}")
                if rendered["trace"]["source_demo_ids"] != inventory_by_id[qid]["demo_ids"]:
                    raise PackageError("rendered source demo IDs or order changed")
                if condition in ("C0", "CD", "PD") and rendered["messages"] != old[f"{qid}:{task}:{condition}"]["messages"]:
                    raise PackageError("unchanged baseline/demo condition is not byte-identical")
                rendered["trace"].update(condition=condition, coverage_renderer_version=VERSION,
                                           resource_policy="global-Lq-union-Ld-frozen-ID-order/v1",
                                           source_correspondence_visible=False)
                prompt = tokenizer.apply_chat_template(rendered["messages"], tokenize=False,
                                                       add_generation_prompt=True, enable_thinking=False)
                token_ids = tokenizer.encode(prompt, add_special_tokens=False)
                prompt_sha = hashlib.sha256(prompt.encode()).hexdigest()
                max_answer = max(candidate["answer_tokens"] for candidate in catalog[task])
                full_length = len(token_ids) + max_answer + 1
                if full_length + PADDING_EXTRA > MAX_SEQUENCE_TOKENS:
                    raise PackageError(f"candidate sequence plus padding exceeds budget: {qid}:{task}:{condition}")
                identity = (prompt_sha, task)
                if identity not in verified_boundaries:
                    for candidate in catalog[task]:
                        combined = tokenizer.encode(prompt + candidate["canonical_answer"], add_special_tokens=False)
                        if combined != token_ids + candidate["answer_token_ids"]:
                            raise PackageError("prompt/candidate token boundary changed")
                    verified_boundaries.add(identity)
                boundary_checks += len(catalog[task])
                row = {
                    "query_id": qid, "task": task, "condition": condition,
                    "record_id": f"{qid}:{task}:{condition}", "model_key": "qwen3-8b",
                    "messages": rendered["messages"], "trace": rendered["trace"],
                    "control_valid": True, "control_status": "valid", "prompt_text": prompt,
                    "prompt_sha256": prompt_sha, "prompt_tokens": len(token_ids),
                    "prompt_token_ids_sha256": canonical_json_sha256(token_ids),
                    "max_new_tokens": max_answer + 1, "overflow": False,
                }
                row["context_sha256"] = canonical_json_sha256(row)
                contexts.append(row)
                metadata = inventory_by_id[qid]
                lex_tokens = rendered["trace"]["injected_blocks"]["lexicon_tokens"]
                metadata["condition_tokens"][f"{task}:{condition}"] = {
                    "prompt_tokens": len(token_ids), "dictionary_tokens": lex_tokens,
                    "max_full_sequence_tokens": full_length,
                }
                metadata["max_full_sequence_tokens"] = max(metadata["max_full_sequence_tokens"], full_length)
                metadata["max_prompt_tokens"] = max(metadata["max_prompt_tokens"], len(token_ids))
                if condition == "CLnew":
                    metadata["dictionary_tokens"] = lex_tokens
                elif condition == "CLq":
                    metadata["lq_dictionary_tokens"] = lex_tokens
    boundary_ids, boundary_selection = select_boundary_queries(per_query, parent_plan["cohorts"])
    if any(sha256_file(Path(path)) != digest for path, digest in sources.items()):
        raise PackageError("a coverage source changed during construction")
    all_lq = {identifier for row in per_query for identifier in row["lq_ids"]}
    all_ld = {identifier for row in per_query for identifier in row["ld_ids"]}
    inventory = {
        "schema_version": VERSION, "per_query": per_query, "source_sha256": sources,
        "lexicon_sha256": FROZEN_LEXICON_SHA256, "matcher_identity": matcher.cache_identity,
        "ordering": "lexicon-id-string-ascending", "dictionary_deduplication": "whole-prompt-entry-ID",
        "source_correspondence_visible": False, "demo_selection_reused": True,
        "query_gold_loaded": False, "test_content_read": False, "model_forward_executed": False,
        "conditions": list(CONDITIONS), "tasks": list(TASKS), "boundary_checks": boundary_checks,
        "unique_prompt_task_boundary_checks": len(verified_boundaries),
        "counts": {"queries": len(qids), "contexts": len(contexts), "candidates": boundary_checks,
                   "frozen_entries": len(entries), "lq_hit_queries": sum(row["lq_hit"] for row in per_query),
                   "ld_hit_queries": sum(row["ld_hit"] for row in per_query),
                   "union_hit_queries": sum(row["union_hit"] for row in per_query),
                   "unique_demos": len(demo_matches), "lq_entry_types": len(all_lq),
                   "ld_entry_types": len(all_ld), "union_entry_types": len(all_lq | all_ld)},
        "limits": {"max_sequence_tokens": MAX_SEQUENCE_TOKENS, "padding_extra": PADDING_EXTRA,
                   "max_union_count": max(row["union_count"] for row in per_query),
                   "max_dictionary_tokens": max(row["dictionary_tokens"] for row in per_query),
                   "max_prompt_tokens": max(row["max_prompt_tokens"] for row in per_query),
                   "max_full_sequence_tokens": max(row["max_full_sequence_tokens"] for row in per_query)},
        "gates": {"frozen_query_matches_replayed": True, "selected_demos_verified_against_fit": True,
                  "control_construction": True, "all_candidate_boundaries": True,
                  "all_sequences_with_eos_and_padding_within_budget": True, "no_truncation": True},
        "original_cohorts": copy.deepcopy(parent_plan["cohorts"]), "boundary_query_ids": boundary_ids,
        "boundary_selection": boundary_selection,
        "boundary_selection_policy": "metadata-representative-first-string-ID-ties-then-deduplicate-no-replacement/v1",
    }
    return contexts, inventory
