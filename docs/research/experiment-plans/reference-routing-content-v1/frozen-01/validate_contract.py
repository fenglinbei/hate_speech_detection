"""Read-only protocol checks. No model import, weights, network, GPU or writes.

The deterministic index builders are used before sealing and reproduced here.
Toy algebra checks are NOT hook qualification or scientific model evidence.
"""
from __future__ import annotations

from collections import Counter
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()


def pin(path):
    raw = path.read_bytes()
    return {"path": str(path.relative_to(ROOT)), "bytes": len(raw), "sha256": sha_bytes(raw)}


def canonical_hash(value):
    return sha_bytes(json.dumps(value, ensure_ascii=False, sort_keys=True,
                               separators=(",", ":")).encode("utf-8"))


def inherited_rows(spec):
    source = ROOT / spec["stage_a"]["source_dir"] / "scoring-inputs.jsonl"
    return [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines()]


def partition_keys(row):
    query = set(row["roles"]["query_all"])
    focal = set(row["patch_position_sets"]["focal"])
    demo_text, demo_answer, demo_all = set(), set(), set()
    for span in row["spans"]:
        if span["kind"] == "demo_text":
            demo_text.update(span["token_positions"])
        elif span["kind"] == "demo_answer":
            demo_answer.update(span["token_positions"])
        elif span["kind"] == "demo":
            demo_all.update(span["token_positions"])
    groups = {
        "query_focal": focal,
        "query_before_focal": {p for p in query if p < min(focal)},
        "query_after_focal": {p for p in query if p > max(focal)},
        "demo_text": demo_text,
        "demo_answer": demo_answer,
        "demo_structure": demo_all - demo_text - demo_answer,
    }
    used = set().union(*groups.values())
    groups["other_prompt"] = set(range(row["prompt_tokens"])) - used
    flat = [p for group in groups.values() for p in group]
    require(sorted(flat) == list(range(row["prompt_tokens"])), "key partition not disjoint/exhaustive")
    require(focal <= query, "focal outside query")
    return {key: sorted(value) for key, value in groups.items()}


def expected_stage_a_index(spec):
    rows = inherited_rows(spec)
    entries = []
    for row in rows:
        entries.append({
            "request_id": row["request_id"],
            "query_id": row["query_id"],
            "condition": row["condition"],
            "source_row_canonical_sha256": canonical_hash(row),
            "prompt_sha256": row["prompt_sha256"],
            "input_ids_sha256": row["input_ids_sha256"],
            "prompt_tokens": row["prompt_tokens"],
            "pre_answer": row["roles"]["pre_answer"][0],
            "focal_positions": row["patch_position_sets"]["focal"],
            "preceding_positions": row["patch_position_sets"]["pre"],
            "key_partition": partition_keys(row),
        })
    return {
        "source": spec["stage_a"]["source_dir"] + "/scoring-inputs.jsonl",
        "byte_identical_inherited_inputs_only": True,
        "query_gold_in_worker_index": False,
        "inputs": entries,
    }


def expected_stage_a_jobs(spec):
    rows = inherited_rows(spec)
    by_id = {r["request_id"]: r for r in rows}
    jobs = []
    for row in rows:
        rid = row["request_id"]
        jobs.append({"job_id": rid + "/N", "kind": "native", "recipient": rid,
                     "background": "N", "upstream": None, "av_override": None})
    for row in rows:
        if row["condition"] == "M00":
            continue
        rid = row["request_id"]
        donor_id = f'jmix-{row["query_id"]}-M00'
        donor = by_id[donor_id]
        focal = {"donor": donor_id, "layer": 17, "site": "decoder_block_output",
                 "positions": row["patch_position_sets"]["focal"],
                 "donor_positions": donor["patch_position_sets"]["focal"], "strength": 1}
        preceding = dict(focal, positions=row["patch_position_sets"]["pre"],
                         donor_positions=donor["patch_position_sets"]["pre"])
        jobs.append({"job_id": rid + "/U", "kind": "upstream", "recipient": rid,
                     "background": "U", "upstream": focal, "av_override": None})
        jobs.append({"job_id": rid + "/P", "kind": "preceding", "recipient": rid,
                     "background": "P", "upstream": preceding, "av_override": None})
        for cell in ("00", "01", "10", "11"):
            av = {"layer": 18, "position": row["roles"]["pre_answer"][0], "heads": "all",
                  "site": "attention_after_o_proj_before_residual", "cell": cell,
                  "A_from": rid + ("/U" if cell[0] == "1" else "/N"),
                  "V_from": rid + ("/U" if cell[1] == "1" else "/N")}
            jobs.append({"job_id": rid + "/AV" + cell,
                         "kind": "av_u_self" if cell == "11" else "av_hybrid",
                         "recipient": rid, "background": "U", "upstream": focal, "av_override": av})
        jobs.append({"job_id": rid + "/N_AV00_self", "kind": "av_native_self",
                     "recipient": rid, "background": "N", "upstream": None,
                     "av_override": dict(av, cell="00", A_from=rid + "/N", V_from=rid + "/N")})
        own = dict(focal, donor=rid, donor_positions=focal["positions"])
        jobs.append({"job_id": rid + "/N_focal_self", "kind": "focal_native_self",
                     "recipient": rid, "background": "N", "upstream": own, "av_override": None})
    return {"kind": "registered_score_configurations_not_a_launch_plan",
            "real_forward_count": None, "jobs": jobs}


def expected_material_slots(spec):
    terms = [{"term_family_id": f"T{i:02d}", "focal_form": term, "history": "previously_exposed"}
             for i, term in enumerate(spec["stage_b"]["known_terms"], 1)]
    terms += [{"term_family_id": tid, "focal_form": None, "history": "new_term_identity_pending"}
              for tid in spec["stage_b"]["new_term_slots"]]
    slots = []
    for split, selected, repetitions in (("development", terms[:3], 2), ("confirmation", terms, 1)):
        for term in selected:
            for stratum, intended in spec["stage_b"]["functional_strata"].items():
                for repeat in range(1, repetitions + 1):
                    slots.append({
                        "slot_id": f'{split}-{term["term_family_id"]}-{stratum}-{repeat}',
                        "split": split, "term_family_id": term["term_family_id"],
                        "stratum": stratum, "design_intended_reference": intended,
                        "raw_text": None, "text_sha256": None, "construction_family_id": None,
                        "human_reference": None, "accepted_fields": [], "decision_ref": None,
                        "relation_ids": [], "eligible_for_model_execution": False,
                    })
    return {"kind": "design_slots_not_material_adoption", "terms": terms, "slots": slots}


def condition_recipes(spec):
    recipes = dict(spec["stage_b"]["base_conditions"])
    for order in spec["stage_b"]["single_slot_replacement_orders"]:
        for k in spec["stage_b"]["replacement_slots"]:
            replacement = list(recipes[order])
            replacement[k - 1] = f"B{k}"
            recipes[f"{order}_replace_{k}"] = replacement
    return recipes


def linear_combination(terms, values, bounds):
    """Collapse physical record IDs before bounding reused score errors."""
    coefficients = {}
    for record, coefficient in terms:
        coefficients[record] = coefficients.get(record, Fraction(0)) + Fraction(coefficient)
    result = sum((a * values[r] for r, a in coefficients.items()), Fraction(0))
    bound = sum((abs(a) * bounds[r] for r, a in coefficients.items()), Fraction(0))
    return result, bound


def select_offset(points):
    """Toy/reference arithmetic: (margin, y, weight, engineering bound)."""
    scores = sorted({p[0] for p in points})
    candidates = {Fraction(0), -max(scores) - 1, -min(scores) + 1}
    candidates.update(-(a + b) / 2 for a, b in zip(scores, scores[1:]))
    def correct_weight(offset):
        return sum(w for m, y, w, b in points if y * (m + offset) > b)
    return min(candidates, key=lambda b: (-correct_weight(b), abs(b), b))


def algebra_checks():
    passed = []
    F = Fraction
    values = {"N": F(-3), "U": F(2), "M00": F(5), "C": F(1), "R": F(-2)}
    bounds = {r: F(1, 1000000) for r in values}
    require(linear_combination([("N", 1), ("N", -1)], values, bounds) == (0, 0),
            "same record must cancel before bounding")
    passed.append("physical_record_ID_error_cancellation")
    gain, error = linear_combination([("N", F(3, 2)), ("M00", F(-1, 2)), ("N", -1)], values, bounds)
    require(gain == F(1, 2) * (values["N"] - values["M00"]) and error == F(1, 1000000),
            "CAD gain/error identity")
    passed.append("CAD_G_identity_and_shared_score_bound")
    for shift in (F(7), F(-13, 8)):
        # Unbalanced raw sample counts: class balancing still cancels a constant.
        no = [F(2), F(-4), F(7)]
        yes = [F(5), F(-1)]
        delta_no = sum((x + shift) - x for x in no) / len(no)
        delta_yes = sum(-((x + shift) - x) for x in yes) / len(yes)
        require((delta_no + delta_yes) / 2 == 0, "class-balanced G invariant")
    passed.append("constant_offset_zero_class_balanced_G_even_with_unequal_classes")
    e_native = values["C"] - values["R"]
    e_cad = F(3, 2) * values["C"] - F(1, 2) * values["M00"] - (F(3, 2) * values["R"] - F(1, 2) * values["M00"])
    require(e_cad == F(3, 2) * e_native, "paired-reference CAD multiplier")
    require((values["C"] + 7) - (values["R"] + 7) == e_native, "offset E invariance")
    require(values["M00"] - values["M00"] == 0, "NOREF reference contribution")
    passed.append("reference_E_CAD_constant_offset_and_NOREF_identities")
    # Exact finite A/V example with nonzero interaction (not a model score).
    a0, a1 = [F(1, 4), F(3, 4)], [F(3, 4), F(1, 4)]
    v0, v1 = [F(1), F(5)], [F(3), F(5)]
    dot = lambda a, v: sum(x * y for x, y in zip(a, v))
    cells = {"00": dot(a0, v0), "01": dot(a0, v1), "10": dot(a1, v0), "11": dot(a1, v1)}
    ea, ev = cells["11"] - cells["01"], cells["11"] - cells["10"]
    joint = cells["11"] - cells["00"]
    inter = cells["11"] - cells["10"] - cells["01"] + cells["00"]
    require(joint == ea + ev - inter and inter != 0, "A/V factorial interaction")
    passed.append("exact_fraction_AV_factorial_including_nonzero_interaction")
    # GQA toy: 2 query heads share 1 KV head, only focal K/V row 1 changes.
    q = [[1.0, 0.5], [-0.25, 1.0]]
    k0 = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.0, 0.0]]
    k1 = [list(v) for v in k0]
    k1[1] = [1.5, -1.0]
    softmax = lambda x: [math.exp(v - max(x)) / sum(math.exp(z - max(x)) for z in x) for v in x]
    for head in q:
        s0 = [sum(x * y for x, y in zip(head, key)) for key in k0]
        s1 = [sum(x * y for x, y in zip(head, key)) for key in k1]
        aa, ab = softmax(s0), softmax(s1)
        require(s0[0] - s0[2] == s1[0] - s1[2], "unpatched-key score contrast")
        require(math.isclose(aa[0] / aa[2], ab[0] / ab[2], rel_tol=1e-14), "unpatched-key attention ratio")
    passed.append("per_head_outside_focal_ratio_invariant_toy_GQA")
    points = [(F(-5), 1, F(1, 4), F(0)), (F(-1), 1, F(1, 4), F(0)),
              (F(-7), -1, F(1, 4), F(0)), (F(-3), -1, F(1, 4), F(0))]
    require(select_offset(points) == 2, "offset tie break must prefer least absolute shift")
    # Exact tie zero must never count as correct; only an offset outside error wins.
    require(select_offset([(F(0), 1, F(1), F(1, 10))]) == 1, "unresolved calibration candidate")
    passed.append("development_offset_grid_tie_break_and_unresolved_handling")
    return passed


def validate():
    spec = read_json(HERE / "protocol.json")
    manifest = read_json(HERE / "manifest.json")
    require(manifest["protocol_id"] == spec["protocol_id"] == "reference-routing-content/v1", "protocol identity")
    pin_count = 0
    for key in ("package_files", "source_files", "protected_historical_selectors"):
        for record in manifest[key]:
            require(pin(ROOT / record["path"]) == record, f'changed pinned file: {record["path"]}')
            pin_count += 1
    actual_package = {str(p.relative_to(ROOT)) for p in HERE.iterdir() if p.is_file() and p.name != "manifest.json"}
    require(actual_package == {p["path"] for p in manifest["package_files"]}, "package closed-file inventory")
    require(spec["status"]["protocol_frozen"] and spec["status"]["model_forwards_performed"] == 0, "freeze status")
    for field in ("stage_b_material_texts_frozen", "stage_b_human_adoption_complete", "executable_runtime_implemented",
                  "gpu_numerically_qualified", "gpu_execution_authorized"):
        require(spec["status"][field] is False, "must not overclaim readiness: " + field)
    require(spec["intervention"]["upstream_layer"] == 17 and spec["intervention"]["decomposition_layer"] == 18,
            "zero-based adjacent layers")
    config = read_json(ROOT / "models/base/Qwen3-8B/config.json")
    for recorded, actual in (("layers", "num_hidden_layers"), ("query_heads", "num_attention_heads"),
                            ("kv_heads", "num_key_value_heads"), ("hidden_size", "hidden_size"), ("head_dim", "head_dim")):
        require(spec["model"][recorded] == config[actual], "model dimension " + recorded)
    require(config["attention_dropout"] == 0 and config["attention_bias"] is False, "attention contract")
    parent = ROOT / "docs/research/experiment-plans/task-applicability-scoring-v1/frozen-01"
    require((parent / "model-task.txt").read_bytes() == (ROOT / spec["stage_a"]["source_dir"] / "model-task.txt").read_bytes(),
            "inherited task bytes differ")
    expected_index = expected_stage_a_index(spec)
    require(read_json(HERE / "stage-a-index.json") == expected_index, "stage A index reproduction")
    rows = inherited_rows(spec)
    require(len(rows) == 18 and len({r["request_id"] for r in rows}) == 18, "18 unique inherited prompts")
    require({(r["query_id"], r["condition"]) for r in rows} ==
            {(q, c) for q in spec["stage_a"]["queries"] for c in spec["stage_a"]["conditions"]}, "full inherited crossing")
    by_id = {r["request_id"]: r for r in rows}
    for row in rows:
        require(sha_bytes(row["prompt_text"].encode("utf-8")) == row["prompt_sha256"], "prompt digest")
        require(row["prompt_tokens"] == len(row["input_ids"]), "token length")
        require(row["roles"]["pre_answer"] == [row["prompt_tokens"] - 1], "original p position")
        require(len(row["patch_position_sets"]["focal"]) == len(row["patch_position_sets"]["pre"]) == 2, "U/P count")
        require(max(row["patch_position_sets"]["focal"]) < row["roles"]["pre_answer"][0], "T must exclude p")
    jobs_doc = read_json(HERE / "stage-a-jobs.json")
    require(jobs_doc == expected_stage_a_jobs(spec), "stage A job reproduction")
    jobs = jobs_doc["jobs"]
    require(Counter(j["kind"] for j in jobs) == spec["stage_a"]["job_counts"], "registered configuration counts")
    require(len({j["job_id"] for j in jobs}) == len(jobs) == 114, "unique job IDs")
    job_ids = {j["job_id"] for j in jobs}
    for job in jobs:
        if job["upstream"]:
            patch = job["upstream"]
            rr, dd = by_id[job["recipient"]], by_id[patch["donor"]]
            require(rr["query_id"] == dd["query_id"], "same-query donor")
            require(len(patch["positions"]) == len(patch["donor_positions"]), "donor position mapping")
            require(all(0 <= p < rr["prompt_tokens"] for p in patch["positions"]), "recipient positions")
            require(all(0 <= p < dd["prompt_tokens"] for p in patch["donor_positions"]), "donor positions")
            if job["kind"] != "preceding":
                require([rr["input_ids"][p] for p in patch["positions"]] ==
                        [dd["input_ids"][p] for p in patch["donor_positions"]], "focal token identity")
        if job["av_override"]:
            av = job["av_override"]
            require(av["A_from"] in job_ids and av["V_from"] in job_ids, "existing AV sources")
            require(av["A_from"].rsplit("/", 1)[0] == av["V_from"].rsplit("/", 1)[0] == job["recipient"],
                    "AV sources must share recipient prompt, never M00")
    material = read_json(HERE / "material-slots.json")
    require(material == expected_material_slots(spec), "48 material slot reproduction")
    slots = material["slots"]
    require(Counter(s["split"] for s in slots) == {"development": 24, "confirmation": 24}, "split counts")
    for split in ("development", "confirmation"):
        require(Counter(s["design_intended_reference"] for s in slots if s["split"] == split) == {"无": 12, "有": 12},
                "design class balance")
    for s in slots:
        require(s["raw_text"] is None and s["human_reference"] is None and s["decision_ref"] is None
                and s["accepted_fields"] == [] and not s["eligible_for_model_execution"], "no invented adoption")
    review = read_json(HERE / "review-contract.json")
    require(review["new_relation_records"] == review["new_human_decisions"] == [], "no invented relation decisions")
    require((ROOT / review["parent_schema"]).is_file(), "real relation schema")
    recipes = condition_recipes(spec)
    require(len(recipes) == 14 and recipes["M00"] == [], "14 prompt configurations")
    labels = lambda sources: ["无" if int(s[1:]) % 2 else "有" for s in sources]
    for name, sources in recipes.items():
        if name == "M00":
            continue
        require(len(sources) == len(set(sources)) == 4, "four distinct demo IDs")
        require(labels(sources) == ["无", "有", "无", "有"], "label-slot invariance")
        if "_replace_" in name:
            order, slot = name.split("_replace_")
            differing = [k for k, (x, y) in enumerate(zip(recipes[order], sources), 1) if x != y]
            require(differing == [int(slot)], "single-slot replacement only")
    require(14 * 48 == spec["stage_b"]["new_unique_prompt_configurations"], "new prompt count")
    require(24 * (1 + 13 * 4) == spec["stage_b"]["development_science_score_configurations"], "dev N/U/Q2/Q4 plus M00")
    require(24 * (1 + 13 * 3) == spec["stage_b"]["confirmation_science_score_configurations_max"], "confirm N/U/Qselected plus M00")
    require(spec["methods"]["QAS"]["layers"] == list(range(18, 36)), "fixed QAS layer set")
    require(spec["methods"]["QAS"]["factor_grid"] == [1, 2, 4], "fixed QAS parameter budget")
    require(spec["methods"]["N_CAL"]["selected_offset"] is None and
            spec["methods"]["U_CAL"]["selected_offset"] is None and
            spec["calibration"]["selected_baseline"] is None, "no imagined development fitting")
    checks = algebra_checks()
    links = 0
    for path in HERE.glob("*.md"):
        for target in re.findall(r"\]\(([^)]+)\)", path.read_text(encoding="utf-8")):
            if "://" in target or target.startswith("#"):
                continue
            linked = (path.parent / target.split("#", 1)[0]).resolve()
            require(linked.exists(), f"broken local link {path.name}: {target}")
            links += 1
    return {
        "status": "PASS", "protocol_id": spec["protocol_id"], "pinned_files_checked": pin_count,
        "package_files": len(manifest["package_files"]), "source_files": len(manifest["source_files"]),
        "protected_historical_selectors": len(manifest["protected_historical_selectors"]),
        "inherited_prompts": 18, "registered_stage_a_score_configurations": 114,
        "stage_a_factorial_pairs": 12, "new_material_slots": 48,
        "new_adopted_materials": 0, "stage_b_conditions_per_query": 14,
        "local_links_checked": links, "algebra_fixture_checks": checks,
        "cpu_only": True, "real_model_forwards": 0, "model_hook_qualification": False,
        "GPU_qualified": False, "scope": "normative_protocol_and_CPU_contract_only",
    }


if __name__ == "__main__":
    print(json.dumps(validate(), ensure_ascii=False, indent=2))
