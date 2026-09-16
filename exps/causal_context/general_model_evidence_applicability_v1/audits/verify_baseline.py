#!/usr/bin/env python3
"""Read-only baseline registration; decodes only indexed discovery case resources.

No model imports, matcher rebuild, private review access, network, or source writes.
Full-dev score/input and reserve selection files may be byte-hashed, never decoded.
Prints a receipt to stdout, or creates --output exclusively in the audit directory
or /tmp. Existing output files, including symlinks, are never overwritten.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


IMPLEMENTATION_BASE_COMMIT = "10580e71a51ebad71bb83ece37f325d78e10103f"
TASKS = ("hate", "group")
CONDITIONS = ("C0", "CLnew", "CD", "CLDnew", "CLnewNoCat", "CLDnewNoCat")
GROUP_LABELS = ("Racism", "Region", "LGBTQ", "Sexism", "others")
AUDIT_RELATIVE_DIR = Path("exps/causal_context/general_model_evidence_applicability_v1/audits")


def digest(path, decompressed=False):
    value = hashlib.sha256()
    opener = gzip.open if decompressed else open
    with opener(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def check_projection(record):
    """Verify both task fields and their existing canonical quadruple projection."""
    projection = record["projection"]
    assert isinstance(projection, dict) and all(task in projection for task in TASKS)
    assert isinstance(projection["hate"], str) and projection["hate"] in ("hate", "non-hate")
    group = projection["group"]
    assert isinstance(group, list) and all(isinstance(label, str) for label in group)
    assert len(group) == len(set(group)) and set(group) <= set(GROUP_LABELS)
    assert group == [label for label in GROUP_LABELS if label in group]
    quads = record["quadruples"]
    assert isinstance(quads, list) and quads
    all_groups = set()
    for quad in quads:
        assert isinstance(quad, dict)
        assert quad["hateful"] in ("hate", "non-hate")
        labels = quad["targeted_group"]
        assert isinstance(labels, list) and labels
        assert all(isinstance(label, str) for label in labels)
        assert len(labels) == len(set(labels))
        assert set(labels) <= set(GROUP_LABELS) | {"non-hate"}
        assert "non-hate" not in labels or labels == ["non-hate"]
        all_groups.update(labels)
    assert projection["hate"] == ("hate" if any(q["hateful"] == "hate" for q in quads) else "non-hate")
    # Group is projected independently of hateful, exactly as the bound source does.
    assert projection["group"] == [label for label in GROUP_LABELS if label in all_groups]


def check_resource_closure(resources):
    """Check the frozen entry-set graph, without claiming occurrence-level spans."""
    fields = ("lq_ids", "ld_ids", "ld_only_ids", "lq_only_ids", "intersection_ids", "union_ids")
    sets = {}
    for field in fields:
        values = resources[field]
        assert isinstance(values, list) and all(isinstance(value, str) for value in values)
        assert len(values) == len(set(values))
        sets[field] = set(values)
    demo_sets = []
    for values in resources["demo_match_ids"].values():
        assert isinstance(values, list) and all(isinstance(value, str) for value in values)
        assert len(values) == len(set(values))
        demo_sets.append(set(values))
    assert sets["ld_ids"] == set().union(*demo_sets)
    assert sets["union_ids"] == sets["lq_ids"] | sets["ld_ids"]
    assert sets["ld_only_ids"] == sets["ld_ids"] - sets["lq_ids"]
    assert sets["lq_only_ids"] == sets["lq_ids"] - sets["ld_ids"]
    assert sets["intersection_ids"] == sets["lq_ids"] & sets["ld_ids"]
    for count_field, id_field in (("lq_count", "lq_ids"), ("ld_count", "ld_ids"),
                                 ("ld_only_count", "ld_only_ids"), ("union_count", "union_ids")):
        assert type(resources[count_field]) is int
        assert resources[count_field] == len(sets[id_field])


def check_output_path(path, root, protected=()):
    """Resolve the parent but reject an existing final component before following it."""
    target = path.expanduser().absolute()
    if target.exists() or target.is_symlink():
        raise ValueError("output_already_exists; choose a new receipt filename")
    target = target.parent.resolve() / target.name
    # Do not resolve an audit-directory symlink into an outside write permission.
    allowed = (root / AUDIT_RELATIVE_DIR, Path("/tmp").resolve())
    if not any(target.is_relative_to(directory) and target != directory for directory in allowed):
        raise ValueError("output_outside_authorized_audit_directory_or_tmp")
    if target == Path(__file__).resolve() or target in {Path(p).resolve() for p in protected}:
        raise ValueError("output_overlaps_protected_input_or_script")
    return target


def emit_receipt(result, output, root, protected):
    payload = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if output is None:
        print(payload, end="")
        return
    target = check_output_path(output, root, protected)
    target.parent.mkdir(parents=True, exist_ok=True)
    target = check_output_path(target, root, protected)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(target, flags, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        stream.write(payload)
    print(f"Receipt created: {target}")


def main():
    if sys.flags.optimize:
        raise RuntimeError("optimized_python_is_not_supported; rerun without -O/-OO or PYTHONOPTIMIZE")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, help="Repository root; defaults to __file__.parents[4] in the installed audits directory")
    parser.add_argument("--map-source-root", type=Path, help="Explicit historical absolute source root to map to --root; content hashes remain mandatory")
    parser.add_argument("--output", type=Path, help="Create a new JSON receipt in the experiment audits directory or /tmp; default: stdout")
    args = parser.parse_args()
    if args.root is None:
        try:
            root = Path(__file__).resolve().parents[4]
        except IndexError:
            parser.error("cannot infer repository root from this script location; pass --root")
    else:
        root = args.root.resolve()
    if args.output is not None:
        check_output_path(args.output, root)
    base = root / "exps/causal_context/general_model_ld_nolabel_paired_cases_v1/results/paired-cases-02"
    checked = {}

    def verify(path, expected):
        actual = digest(path)
        if actual != expected:
            raise ValueError(f"source_identity_mismatch: {path}")
        checked[str(path.relative_to(root))] = actual
        return actual

    def read(path):
        return json.loads(path.read_text(encoding="utf-8"))

    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    tracked_status = subprocess.check_output(["git", "status", "--porcelain=v1", "--untracked-files=no"],
                                             cwd=root, text=True)
    bound_source_hashes = {}
    for relative in ("src/diagnostics/general_model_contexts.py", "src/diagnostics/general_model_tasks.py",
                     "src/utils/quadruple.py"):
        baseline_bytes = subprocess.check_output(["git", "show", f"{IMPLEMENTATION_BASE_COMMIT}:{relative}"], cwd=root)
        baseline_hash = hashlib.sha256(baseline_bytes).hexdigest()
        verify(root / relative, baseline_hash)
        bound_source_hashes[relative] = baseline_hash
    manifest_hash = verify(base / "manifest.json", "4f9b48b76e2102aa49e65deafb55ac6566c798fee346fe1be6e5cd2c5ba771d2")
    manifest = read(base / "manifest.json")
    identity = "a8c1fc6a97f83b339f02e36c286da5954d233a568b47ec235e4a55b217842201"
    assert manifest["identity"] == identity
    assert manifest["execution_commit"] == "f80fe355263c267606c9601ca5d12aabee26da10"
    assert manifest["status"] == "complete" and manifest["source_run"] == "nolabel-01"
    verify(base / "export_manifest.json", "1b3f9b71fb2f36e39e1ce45e06a92744c1aeca2e463861e549d592778ac682ee")
    export = read(base / "export_manifest.json")
    assert export["source_manifest_sha256"] == manifest_hash
    assert export["source_identity"] == identity

    # The reserve selector is verified as opaque bytes only; no reserve rows are parsed.
    published_names = ["config.frozen.json", "cases/cards_index.json", "cases/discovery.jsonl",
                       "cases/reserve.jsonl", "cases/selection_manifest.json", "REPORT.md",
                       "tables/classification.csv", "tables/exploratory_ci.csv"]
    for name in published_names:
        verify(base / name, export["artifacts"][name])
    source = manifest["source_receipt"]
    historical_root = Path(source["lexicon"]["path"]).parents[2]
    if args.map_source_root is not None and args.map_source_root != historical_root:
        raise ValueError("source_root_mapping_does_not_match_frozen_root")

    def source_path(name):
        path = Path(source[name]["path"])
        if args.map_source_root is not None:
            return root / path.relative_to(args.map_source_root)
        if not path.is_relative_to(root):
            raise ValueError("historical_source_path_outside_root; an explicit --map-source-root is required")
        return path

    source_names = ["lexicon", "package_manifest", "fit_catalog", "raw", "raw_manifest",
                    "report_manifest"]
    source_status = {}
    for name in source_names:
        path = source_path(name)
        actual = verify(path, source[name]["sha256"])
        source_status[name] = {"available": True, "sha256": actual,
                               "payload_decoded": name == "package_manifest"}
    assert source_status["lexicon"]["sha256"] == "31240193eeba29f712560e2e80e89ee9dd5c3c969aa180d10b95bf2451882385"

    package_path = source_path("package_manifest")
    package = read(package_path)
    package_files = {row["path"]: row["sha256"] for row in package["files"]}
    package_config_path = package_path.parent / "config.resolved.json"
    verify(package_config_path, package_files["config.resolved.json"])
    dataset = read(package_config_path)["sources"]
    dataset_id = dataset["data_id"]
    input_hashes = {}
    for name, metadata in export["additional_inputs"].items():
        path = base / name
        compressed_hash = verify(path, export["artifacts"][name])
        decoded_hash = digest(path, decompressed=metadata["encoding"] == "gzip")
        assert decoded_hash == metadata["source_sha256"]
        input_hashes[name] = {"file_sha256": compressed_hash,
                              "uncompressed_sha256": decoded_hash,
                              "payload_decoded": False}

    selection = read(base / "cases/selection_manifest.json")
    assert selection["passed"] and selection["deterministic_replay_passed"]
    assert selection["selected_counts"] == {"discovery": 32, "reserve": 16}
    assert selection["reserve_cards_exported"] is False
    index = read(base / "cases/cards_index.json")
    rows = [json.loads(line) for line in (base / "cases/discovery.jsonl").read_text().splitlines()]
    ids = [row["query_id"] for row in index]
    assert len(ids) == len(set(ids)) == 32
    assert len(rows) == 32
    assert {row["query_id"] for row in rows} == set(ids)
    assert all(row["split"] == "discovery" for row in rows)

    counts = Counter()
    demos_by_identity, demo_by_id, entries, senses = {}, {}, {}, {}
    text_families = defaultdict(set)
    unique_demo_entry_edges = set()
    unique_query_entry_edges = set()
    unique_demo_sense_edges = set()
    per_query = []
    for row in index:
        relative = row["resources_card"].replace("/cards/", "/card_data/").replace("-1-resources.md", ".json")
        assert relative.startswith("cases/card_data/")
        card_path = (base / relative).resolve()
        assert card_path.is_relative_to(base / "cases/card_data")
        verify(card_path, export["artifacts"][relative])
        card = read(card_path)
        q = str(card["query"]["id"])
        assert q == row["query_id"]
        assert card["selection"]["split"] == "discovery"
        assert card["source_sha256"]["lexicon"] == source["lexicon"]["sha256"]
        check_projection(card["query"])
        counts["query_projection_task_fields_verified"] += 2
        resources = card["profile"]["resources"]
        check_resource_closure(resources)
        counts["query_resource_graphs_closed"] += 1
        local_entries = {entry["lexicon_id"]: entry for entry in card["lexicon_entries"]}
        assert len(local_entries) == len(card["lexicon_entries"])
        assert set(local_entries) == set(resources["union_ids"])
        local_demos = [str(demo["id"]) for demo in card["demonstrations"]]
        assert local_demos == resources["demo_ids"] and len(local_demos) == 10
        assert len(local_demos) == len(set(local_demos))
        assert set(local_demos) == set(resources["demo_match_ids"])
        assert not resources["missing_reasons"]
        counts["query_demo_relationships"] += len(local_demos)
        counts["query_to_entry_aggregate_edges"] += len(resources["lq_ids"])
        counts["demo_to_entry_aggregate_edges_with_query_context"] += sum(map(len, resources["demo_match_ids"].values()))
        counts["query_entry_render_instances"] += len(local_entries)
        counts["query_sense_render_instances"] += sum(len(e["senses"]) for e in local_entries.values())
        counts["query_to_sense_aggregate_edges"] += sum(len(local_entries[e]["senses"]) for e in resources["lq_ids"])
        counts["demo_to_sense_aggregate_edges_with_query_context"] += sum(len(local_entries[e]["senses"]) for values in resources["demo_match_ids"].values() for e in values)
        for eid in resources["lq_ids"]:
            unique_query_entry_edges.add((q, eid))
        for entry_id, entry in local_entries.items():
            payload = canonical(entry)
            assert entries.get(entry_id, payload) == payload
            entries[entry_id] = payload
            for sense in entry["senses"]:
                key = (entry_id, sense["sense_id"])
                sense_payload = canonical(sense)
                assert senses.get(key, sense_payload) == sense_payload
                senses[key] = sense_payload
        for demo in card["demonstrations"]:
            check_projection(demo)
            counts["demo_projection_task_fields_verified_with_query_context"] += 2
            demo_id = str(demo["id"])
            content_hash = hashlib.sha256(demo["content"].encode("utf-8")).hexdigest()
            key = (dataset_id, demo_id, content_hash)
            # Full payload equality also verifies task answers and original quadruples.
            payload = canonical(demo)
            if demo_id in demo_by_id:
                assert demo_by_id[demo_id] == (content_hash, payload), "same_id_payload_conflict"
            demo_by_id[demo_id] = (content_hash, payload)
            demos_by_identity[key] = payload
            text_families[content_hash].add(demo_id)
            for eid in resources["demo_match_ids"][demo_id]:
                unique_demo_entry_edges.add((key, eid))
                for sense in local_entries[eid]["senses"]:
                    unique_demo_sense_edges.add((key, eid, sense["sense_id"]))
        contexts = card["contexts"]
        expected_frames = {(task, condition) for task in TASKS for condition in CONDITIONS}
        assert len(contexts) == len(expected_frames) == 12
        assert {(c["task"], c["condition"]) for c in contexts} == expected_frames
        counts["exact_six_by_two_context_frames_verified"] += 1
        for context in contexts:
            assert str(context["query_id"]) == q
            trace = context["trace"]
            assert trace["task"] == context["task"] and trace["condition"] == context["condition"]
            assert trace["source_demo_ids"] == local_demos
            assert trace["source_lexicon_ids"] == resources["union_ids"]
            expected_demos = local_demos if context["condition"] in ("CD", "CLDnew", "CLDnewNoCat") else []
            expected_entries = resources["union_ids"] if context["condition"] in ("CLnew", "CLDnew", "CLnewNoCat", "CLDnewNoCat") else []
            assert trace["injected_demo_ids"] == expected_demos
            assert trace["injected_lexicon_ids"] == expected_entries
            assert hashlib.sha256(context["prompt_text"].encode()).hexdigest() == context["prompt_sha256"]
            counts["prompt_text_hashes_verified"] += 1
            if context["condition"] == "CLDnewNoCat":
                rendered = {(v["lexicon_id"], v["sense_id"]) for v in context["trace"]["lexicon"] if v["definition_visible"]}
                expected = {(eid, s["sense_id"]) for eid,e in local_entries.items() for s in e["senses"]}
                assert rendered == expected
        per_query.append({"query_id": q, "demo_count": len(local_demos),
                          "entry_count": len(local_entries),
                          "sense_count": sum(len(e["senses"]) for e in local_entries.values()),
                          "query_entry_edges": len(resources["lq_ids"]),
                          "demo_entry_edges": sum(map(len, resources["demo_match_ids"].values()))})

    counts.update({"discovery_queries": len(ids), "reserved_queries_from_frozen_summary": 16,
                   "unique_demonstrations": len(demos_by_identity), "unique_lexicon_entries": len(entries),
                   "unique_lexicon_senses": len(senses),
                   "unique_demo_to_entry_aggregate_edges": len(unique_demo_entry_edges),
                   "unique_demo_to_sense_aggregate_edges": len(unique_demo_sense_edges),
                   "same_text_multiple_id_families": sum(len(v)>1 for v in text_families.values()),
                   "same_id_payload_conflicts": 0})
    counts["query_task_review_units"] = len(ids) * 2
    counts["demo_task_label_review_units"] = len(demos_by_identity) * 2
    counts["query_demo_task_relation_review_units"] = counts["query_demo_relationships"] * 2
    # Check the protected files again; this audit has no writes to any source tree.
    assert all(digest(root / path) == sha for path,sha in checked.items())
    result = {"schema_version": "evidence-applicability-baseline-audit/v1",
              "audited_at": datetime.now(timezone.utc).isoformat(), "passed": True,
              "implementation_base_commit": IMPLEMENTATION_BASE_COMMIT,
              "current_head": head, "audit_execution_commit": head,
              "tracked_worktree_modified": bool(tracked_status.strip()),
              "bound_source_sha256_equal_to_implementation_base": bound_source_hashes,
              "audit_script_sha256": digest(Path(__file__).resolve()),
              "scientific_source_commit": manifest["execution_commit"],
              "source_scoring_run": "nolabel-01", "source_paired_run": "paired-cases-02",
              "source_path_mapping": ({"from": str(args.map_source_root), "to": str(root)}
                                      if args.map_source_root is not None else None),
              "paired_source_identity": identity, "paired_manifest_sha256": manifest_hash,
              "source_dataset_id": dataset_id, "source_partition_id": dataset["partition_id"],
              "source_dataset_identity_bound_via": str(package_config_path.relative_to(root)),
              "dedup_key": ["source_dataset_id", "demo_id", "content_sha256"],
              "counts": dict(counts), "per_query_resource_inventory": per_query,
              "span_hit_count": None, "span_hit_status": "not_reconstructed; cards provide aggregate entry provenance and rendered senses, not source occurrence spans",
              "new_human_reviews_created": 0, "human_review_completion_assessed": False,
              "private_review_sessions_accessed": False, "reserve_records_decoded": False,
              "reserve_body_reviewed": False, "full_dev_payloads_decoded": False,
              "model_forward_executed": False, "deployment_changed": False,
              "sources_unchanged_after_audit": True, "verified_file_sha256": checked,
              "full_dev_input_byte_hashes": input_hashes, "local_source_availability": source_status,
              "scope_limitations": ["Raw scores checked by hash only; no candidate-score reconstruction performed.",
                                    "No source span or normalized-offset reconstruction performed.",
                                    "The 64/560/640 task units are review scope, not completed human records.",
                                    "No new selection or freezing of historical inputs is required for section 3.1."]}
    emit_receipt(result, args.output, root, [root / path for path in checked])


if __name__ == "__main__":
    main()
