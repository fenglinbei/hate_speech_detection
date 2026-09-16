"""Remove only explicit dictionary category fields from frozen merged-L inputs."""

from __future__ import annotations

import copy
import hashlib
from pathlib import Path

from data.stage1_data import canonical_json_sha256, sha256_file
from diagnostics.general_model_contexts import _render_lexicon
from diagnostics.general_model_coverage_package import _catalog, select_boundary_queries
from diagnostics.general_model_package import ROOT, PackageError, read_json


CONDITIONS = ("C0", "CLnew", "CD", "CLDnew", "CLnewNoCat", "CLDnewNoCat")
SOURCE_CONDITIONS = {c: c for c in CONDITIONS[:4]} | {
    "CLnewNoCat": "CLnew", "CLDnewNoCat": "CLDnew"}
LEXICON_SHA256 = "31240193eeba29f712560e2e80e89ee9dd5c3c969aa180d10b95bf2451882385"


def category_removal(messages: list[dict], hits: list[dict]) -> tuple[list[dict], list[dict], str]:
    """Replay the full prefix before replacing it; never globally delete text lines."""
    full, _ = _render_lexicon(hits, "Full")
    definition, trace = _render_lexicon(hits, "Definition")
    if len(messages) != 2 or [m["role"] for m in messages] != ["system", "user"]:
        raise PackageError("expected the frozen two-message prompt")
    result = copy.deepcopy(messages)
    if full:
        if not result[1]["content"].startswith(full + "\n\n"):
            raise PackageError("dictionary is not the exact frozen full prefix")
        result[1]["content"] = definition + result[1]["content"][len(full):]
    elif result[1]["content"].startswith("词典参考：\n"):
        raise PackageError("unexpected dictionary block for empty hits")
    return result, trace, definition


def build_contexts(parent_plan: dict, old_contexts: list[dict], package_dir: Path,
                   tokenizer) -> tuple[list[dict], dict]:
    """Construct six cells from sealed coverage inputs without deserializing Gold."""
    path = ROOT / "data/lexicon/annotated_lexicon_mechanism_frozen_v1.json"
    if sha256_file(path) != LEXICON_SHA256:
        raise PackageError("frozen lexicon changed")
    entries = {r["lexicon_id"]: r for r in read_json(path)["terms"]}
    if len(entries) != 833:
        raise PackageError("expected all 833 frozen entries")
    old = {row["record_id"]: row for row in old_contexts}
    qids = [r["query_id"] for r in parent_plan["frame"]]
    expected = {f"{q}:{t}:{c}" for q in qids for t in ("hate", "group")
                for c in parent_plan["config"]["conditions"]}
    if len(old) != len(old_contexts) or set(old) != expected:
        raise PackageError("parent input matrix is incomplete or duplicated")
    for row in old_contexts:
        if canonical_json_sha256({k:v for k,v in row.items() if k != "context_sha256"}) != row["context_sha256"]:
            raise PackageError("parent context hash differs")
    catalog = _catalog(parent_plan, tokenizer)
    inventory = {}
    for q in qids:
        ids = old[f"{q}:hate:CLnew"]["trace"]["injected_lexicon_ids"]
        if ids != sorted(set(ids)) or not set(ids) <= entries.keys():
            raise PackageError("merged dictionary membership/order is not frozen ID order")
        for t in ("hate", "group"):
            for c in ("CLnew", "CLDnew"):
                if old[f"{q}:{t}:{c}"]["trace"]["injected_lexicon_ids"] != ids:
                    raise PackageError("dictionary membership differs between source conditions")
        inventory[q] = {"query_id":q, "union_ids":ids, "union_count":len(ids),
                        "dictionary_tokens":0, "max_full_sequence_tokens":0, "condition_tokens":{}}
    contexts = []
    boundary_checks = 0
    for t in ("hate", "group"):
        for c in CONDITIONS:
            for q in qids:
                source = old[f"{q}:{t}:{SOURCE_CONDITIONS[c]}"]
                row = copy.deepcopy(source)
                if c.endswith("NoCat"):
                    hits = [entries[i] for i in inventory[q]["union_ids"]]
                    messages, trace, dictionary = category_removal(source["messages"], hits)
                    row.update(condition=c, record_id=f"{q}:{t}:{c}", messages=messages)
                    row["trace"].update(condition=c, lexicon=trace,
                        category_removal={"source_record_id":source["record_id"],
                                          "source_context_sha256":source["context_sha256"],
                                          "removed_fields":len(trace), "definitions_preserved":True,
                                          "demo_and_query_suffix_preserved":True})
                    row["trace"]["injected_blocks"].update(
                        lexicon_sha256=hashlib.sha256(dictionary.encode()).hexdigest(),
                        lexicon_tokens=len(tokenizer.encode(dictionary, add_special_tokens=False)))
                    prompt = tokenizer.apply_chat_template(messages, tokenize=False,
                        add_generation_prompt=True, enable_thinking=False)
                    ids = tokenizer.encode(prompt, add_special_tokens=False)
                    row.update(prompt_text=prompt, prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
                               prompt_tokens=len(ids), prompt_token_ids_sha256=canonical_json_sha256(ids))
                    row["context_sha256"] = canonical_json_sha256({k:v for k,v in row.items() if k != "context_sha256"})
                else:
                    if row != source:
                        raise PackageError("baseline context changed")
                token_ids = tokenizer.encode(row["prompt_text"], add_special_tokens=False)
                if (len(token_ids) != row["prompt_tokens"]
                        or canonical_json_sha256(token_ids) != row["prompt_token_ids_sha256"]):
                    raise PackageError("context token identity does not replay")
                for candidate in catalog[t]:
                    if tokenizer.encode(row["prompt_text"]+candidate["canonical_answer"], add_special_tokens=False) != token_ids+candidate["answer_token_ids"]:
                        raise PackageError("prompt/answer token boundary differs")
                    boundary_checks += 1
                full_len = len(token_ids)+max(x["answer_tokens"] for x in catalog[t])+1
                if full_len+64 > 8192:
                    raise PackageError("candidate and padding exceed capacity")
                meta = inventory[q]
                meta["condition_tokens"][f"{t}:{c}"] = row["prompt_tokens"]
                meta["max_full_sequence_tokens"] = max(meta["max_full_sequence_tokens"], full_len)
                if c == "CLnew":
                    meta["dictionary_tokens"] = row["trace"]["injected_blocks"]["lexicon_tokens"]
                contexts.append(row)
    boundary, selection = select_boundary_queries(list(inventory.values()), parent_plan["cohorts"])
    if sha256_file(path) != LEXICON_SHA256:
        raise PackageError("lexicon changed during construction")
    return contexts, {"schema_version":"general-model-nolabel-package/v1",
        "conditions":list(CONDITIONS), "per_query":list(inventory.values()),
        "lexicon_sha256":LEXICON_SHA256, "source_parent_plan_id":parent_plan["plan_id"],
        "boundary_query_ids":boundary, "boundary_selection":selection,
        "counts":{"queries":len(qids), "contexts":len(contexts), "candidates":boundary_checks},
        "query_gold_loaded":False, "test_content_read":False, "model_forward_executed":False,
        "baseline_contexts_byte_identical":True, "only_explicit_dictionary_category_fields_removed":True,
        "definitions_and_demos_unchanged":True, "length_matching":False}
