"""Independent selection, provenance and immutability checks for the freeze."""

from __future__ import annotations

import copy
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch

from jsonschema import Draft202012Validator, ValidationError

from build_lex.annotated_lexicon_repair import (
    LexiconRepairError, canonical_sha256, file_sha256, read_json, write_json,
)
from build_lex.lexicon_mechanism_freeze import (
    EXPECTED_SESSION_REVISION, EXPECTED_SESSION_SHA256, FILE_NAMES,
    MISSING_DEFINITION, build_mechanism_freeze, render_entry,
    select_review_entry, verify_freeze, write_freeze,
)


ROOT = Path(__file__).resolve().parents[2]
REPAIR = ROOT / "exps/causal_context/annotated_lexicon_repair_v1"
RECORDS = ROOT / "exps/causal_context/lexicon_mechanism_frozen_v1"
INPUTS = {
    "lexicon_path": ROOT / "data/lexicon/annotated_lexicon.json",
    "frame_path": REPAIR / "repair_operation/frame.json",
    "session_path": RECORDS / "review_session_snapshot.json",
    "span_reference_path": REPAIR / "span_gold/reference.json",
}
MODULE = "build_lex.lexicon_mechanism_freeze"
MISSING_TERMS = {"呆比", "宁", "掉的一批", "盖盖", "蝲蝲蛄", "褐兰州", "黑人喃"}


def item_and_decision(*, term="测试", status="draft", resolution=None, changed=False, notes=""):
    proposal = {
        "lexicon_id": "lex-test", "term": term, "variants": [],
        "senses": [{"sense_id": "lex-test:sense:01", "definition": "提案释义", "categories": ["others"]}],
        "match_policy": {"require_any": [], "exclude_any": []},
    }
    decision = {"entry": copy.deepcopy(proposal), "status": status,
                "resolution": resolution, "notes": notes}
    if changed:
        decision["entry"]["senses"][0]["definition"] = "用户改写"
    return {"term": term, "proposed_entry": proposal}, decision


class FreezeSelectionTests(unittest.TestCase):
    def test_confirmed_entry_wins_over_proposal_and_is_copied(self):
        for resolution in ("approve", "revise"):
            with self.subTest(resolution=resolution):
                item, decision = item_and_decision(status="confirmed", resolution=resolution, changed=True)
                selected, source = select_review_entry(item, decision)
                self.assertEqual(source, "human_confirmed_preserved")
                self.assertEqual(selected, decision["entry"])
                selected["senses"][0]["definition"] = "不应回写"
                self.assertEqual(decision["entry"]["senses"][0]["definition"], "用户改写")
                self.assertEqual(item["proposed_entry"]["senses"][0]["definition"], "提案释义")

    def test_untouched_draft_uses_separate_batch_authority(self):
        item, decision = item_and_decision()
        selected, source = select_review_entry(item, decision)
        self.assertEqual(selected, item["proposed_entry"])
        self.assertEqual(source, "user_batch_default_accept")
        self.assertIsNone(decision["resolution"])
        self.assertEqual(decision["status"], "draft")

    def test_saved_draft_changes_are_not_replaced_by_proposal(self):
        item, decision = item_and_decision(changed=True, resolution="revise")
        selected, source = select_review_entry(item, decision)
        self.assertEqual(source, "user_saved_draft_preserved")
        self.assertEqual(selected, decision["entry"])

    def test_known_edited_defer_is_preserved_without_rewriting_history(self):
        item, decision = item_and_decision(
            term="它们", status="deferred", resolution="defer", changed=True,
            notes="可以先保留，其他留到语义层完成",
        )
        before = copy.deepcopy(decision)
        selected, source = select_review_entry(item, decision)
        self.assertEqual(source, "user_edited_deferred_preserved")
        self.assertEqual(selected, decision["entry"])
        self.assertEqual(decision, before)

    def test_arbitrary_deferrals_are_not_automatically_approved(self):
        cases = (
            {"term": "另一条", "notes": "可以先保留", "changed": True},
            {"term": "它们", "notes": "暂缓", "changed": True},
            {"term": "它们", "notes": "可以先保留", "changed": False},
        )
        for case in cases:
            with self.subTest(**case):
                item, decision = item_and_decision(status="deferred", resolution="defer", **case)
                with self.assertRaises(LexiconRepairError):
                    select_review_entry(item, decision)

    def test_rejection_and_other_statuses_cannot_be_batch_accepted(self):
        for status, resolution in (
            ("confirmed", "reject"), ("draft", "reject"), ("draft", "defer"),
            ("confirmed", None), ("confirmed", "defer"), ("unknown", "approve"),
            ("deferred", "approve"),
        ):
            with self.subTest(status=status, resolution=resolution):
                item, decision = item_and_decision(status=status, resolution=resolution, changed=True)
                with self.assertRaises(LexiconRepairError):
                    select_review_entry(item, decision)

    def test_render_preserves_missing_senses_and_keeps_labels_out_of_definition(self):
        entry = {
            "term": "测试", "notes": "不得注入的审核备注", "rationale": "不得注入的理由",
            "senses": [
                {"sense_id": "1", "definition": "原释义", "categories": ["Region", "others"]},
                {"sense_id": "2", "definition": "", "categories": ["Sexism"]},
                {"sense_id": "3", "definition": " \t", "categories": ["Racism", "LGBTQ", "Region"]},
            ],
        }
        before = copy.deepcopy(entry)
        rendered = render_entry(entry)
        self.assertEqual(rendered["category"], "Racism, Sexism, LGBTQ, Region, others")
        self.assertEqual(rendered["definition"], f"[1] 原释义\n[2] {MISSING_DEFINITION}\n[3] {MISSING_DEFINITION}")
        self.assertEqual(entry, before)
        for forbidden in ("Region", "Sexism", "LGBTQ", "Racism", "others", entry["notes"], entry["rationale"]):
            self.assertNotIn(forbidden, rendered["definition"])


class RealFreezeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frame = read_json(INPUTS["frame_path"])
        cls.session = read_json(INPUTS["session_path"])
        cls.source = read_json(INPUTS["lexicon_path"])
        cls.original_bytes = {name: path.read_bytes() for name, path in INPUTS.items()}
        # One complete build really verifies the pinned engine, resource-limited
        # compilation, full source hashes and all matching diagnostics.
        cls.bundle = build_mechanism_freeze(**INPUTS)
        cls.terms = {entry["term"]: entry for entry in cls.bundle["lexicon"]["terms"]}
        cls.operations = {entry["term"]: entry for entry in cls.bundle["decisions"]["operations"]}

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.temporary = Path(temporary.name)

    def quick_build(self, **overrides):
        """Re-use only the proven isolation result; retain other validation."""
        runtime = self.bundle["validation"]["isolated_regex_check"]
        with patch(MODULE + ".check_regex_runtime", return_value=copy.deepcopy(runtime)), \
                patch(MODULE + ".validate_match_policy", side_effect=copy.deepcopy):
            return build_mechanism_freeze(**{**INPUTS, **overrides})

    def test_real_inventory_and_pinned_isolation(self):
        counts = self.bundle["manifest"]["counts"]
        for name, expected in {
            "source_records": 829, "source_unique_terms": 818, "duplicate_source_groups": 11,
            "runtime_entries": 833, "new_entries": 15, "senses": 839,
            "missing_definitions": 7, "entries_with_rules": 15, "rule_count": 16,
        }.items():
            with self.subTest(name=name):
                self.assertEqual(counts[name], expected)
        self.assertEqual(counts["operation_selections"], {
            "human_confirmed_preserved": 24, "user_edited_deferred_preserved": 1,
            "user_batch_default_accept": 31,
        })
        self.assertEqual(counts["historical_review_status"], {"confirmed": 24, "deferred": 1, "draft": 31})
        isolated = self.bundle["validation"]["isolated_regex_check"]
        self.assertEqual(isolated["regex_version"], "2026.4.4")
        self.assertEqual(isolated["validation_isolation"], "resource-limited-subprocess/v1")

    def test_frozen_schema_allows_explicit_missing_definition_not_missing_categories(self):
        schema = read_json(ROOT / "schemas/annotated_lexicon_mechanism_frozen_v1.schema.json")
        Draft202012Validator.check_schema(schema)
        validator = Draft202012Validator(schema)
        validator.validate(self.bundle["lexicon"])
        invalid = copy.deepcopy(self.bundle["lexicon"])
        invalid["terms"][0]["senses"][0]["categories"] = []
        with self.assertRaises(ValidationError):
            validator.validate(invalid)
        invalid = copy.deepcopy(self.bundle["lexicon"])
        invalid["quality"]["fully_human_reviewed"] = True
        with self.assertRaises(ValidationError):
            validator.validate(invalid)

    def test_every_edited_and_default_entry_has_exact_preserved_provenance(self):
        changed = []
        for item in self.frame["items"]:
            term = item["term"]
            decision = self.session["decisions"][item["item_id"]]
            operation = self.operations[term]
            with self.subTest(term=term):
                self.assertEqual(operation["effective_entry"], decision["entry"])
                self.assertEqual(operation["historical_status"], decision["status"])
                self.assertEqual(operation["historical_resolution"], decision["resolution"])
                self.assertEqual(operation["historical_notes"], decision["notes"])
                self.assertEqual(operation["effective_entry_sha256"], canonical_sha256(decision["entry"]))
                for field, value in decision["entry"].items():
                    self.assertEqual(self.terms[term][field], value)
                if decision["status"] == "draft":
                    self.assertEqual(operation["selection_source"], "user_batch_default_accept")
                    self.assertEqual(decision["entry"], item["proposed_entry"])
                if operation["entry_changed_from_proposal"]:
                    changed.append(term)
        self.assertEqual(len(changed), 14)
        self.assertIn("它们", changed)
        self.assertEqual(self.operations["它们"]["historical_resolution"], "defer")
        self.assertTrue(self.terms["它"]["match_policy"]["require_any"])
        for term in ("同志", "尼"):
            self.assertEqual(self.terms[term]["match_policy"], {"require_any": [], "exclude_any": []})

    def test_source_rows_are_all_accounted_for_and_untouched_entries_are_verbatim(self):
        mapped = [index for entry in self.terms.values() for index in entry["source_record_indices"]]
        self.assertEqual(Counter(mapped), Counter(range(829)))
        self.assertEqual(sum(len(entry["source_record_indices"]) > 1 for entry in self.terms.values()), 11)
        self.assertEqual(sum(entry["freeze_selection_source"] == "original_preserved" for entry in self.terms.values()), 777)
        for entry in self.terms.values():
            for index in entry["source_record_indices"]:
                self.assertEqual(self.source["terms"][index]["term"], entry["term"])
            if entry["freeze_selection_source"] == "original_preserved":
                self.assertEqual(len(entry["senses"]), len(entry["source_record_indices"]))
                for sense, index in zip(entry["senses"], entry["source_record_indices"]):
                    original = self.source["terms"][index]
                    self.assertEqual(sense["definition"], original["definition"])
                    self.assertEqual(sense["categories"], list(dict.fromkeys(label.strip() for label in original["category"].split(","))))

    def test_empty_definitions_remain_explicit_and_yp_is_no_longer_empty(self):
        observed = {entry["term"] for entry in self.terms.values() if any(not sense["definition"].strip() for sense in entry["senses"])}
        self.assertEqual(observed, MISSING_TERMS)
        for term in MISSING_TERMS:
            self.assertEqual(self.terms[term]["senses"][0]["definition"], "")
            self.assertEqual(self.terms[term]["definition"], "[1] " + MISSING_DEFINITION)
        self.assertEqual(len(self.terms["YP"]["senses"]), 2)
        self.assertTrue(all(sense["definition"] for sense in self.terms["YP"]["senses"]))
        self.assertFalse(self.bundle["lexicon"]["quality"]["fully_human_reviewed"])
        self.assertFalse(self.bundle["validation"]["span_diagnostic"]["repair_acceptance_claimed"])

    def test_all_inputs_and_review_history_remain_unchanged(self):
        self.assertEqual(self.bundle["review_session"], self.session)
        self.assertEqual(self.session["revision"], EXPECTED_SESSION_REVISION)
        self.assertIsNone(self.session["finalized_bundle_id"])
        self.assertFalse(self.bundle["decisions"]["historical_session_modified"])
        self.assertEqual(file_sha256(INPUTS["session_path"]), EXPECTED_SESSION_SHA256)
        for name, path in INPUTS.items():
            with self.subTest(name=name):
                self.assertEqual(path.read_bytes(), self.original_bytes[name])

    def test_identical_inputs_rebuild_identical_bundle_and_all_hashes(self):
        rebuilt = self.quick_build()
        self.assertEqual(rebuilt, self.bundle)
        for key in self.bundle:
            self.assertEqual(canonical_sha256(rebuilt[key]), canonical_sha256(self.bundle[key]))
        manifest = rebuilt["manifest"]
        for artifact, key in (("lexicon", "lexicon"), ("decisions", "decisions"),
                              ("validation", "validation"), ("review_session_snapshot", "review_session")):
            self.assertEqual(manifest["artifact_sha256"][artifact], canonical_sha256(rebuilt[key]))

    def test_input_tampering_fails_closed(self):
        for argument in INPUTS:
            with self.subTest(argument=argument):
                payload = read_json(INPUTS[argument])
                if argument == "lexicon_path":
                    payload["terms"][0]["definition"] += "篡改"
                elif argument == "frame_path":
                    payload["items"][0]["proposed_entry"]["senses"][0]["definition"] += "篡改"
                elif argument == "session_path":
                    payload["decisions"][next(iter(payload["decisions"]))]["notes"] += "篡改"
                else:
                    payload["rows"][0]["query_content"] += "篡改"
                target = self.temporary / (argument + ".json")
                write_json(target, payload)
                with self.assertRaises(LexiconRepairError):
                    self.quick_build(**{argument: target})

    def test_writing_is_idempotent_readonly_and_preserves_history_bytes(self):
        output, records = self.temporary / "lexicon.json", self.temporary / "records"
        paths = write_freeze(self.bundle, output_path=output, record_root=records)
        first = {name: Path(path).read_bytes() for name, path in paths.items()}
        self.assertEqual(paths, write_freeze(self.bundle, output_path=output, record_root=records))
        for key, path in paths.items():
            target = Path(path)
            self.assertEqual(target.read_bytes(), first[key])
            self.assertEqual(file_sha256(target), canonical_sha256(self.bundle[key]))
            self.assertEqual(target.stat().st_mode & 0o777, 0o444)
        self.assertEqual(first["review_session"], self.original_bytes["session_path"])

    def test_existing_different_artifact_blocks_whole_bundle_before_writes(self):
        output, records = self.temporary / "lexicon.json", self.temporary / "records"
        conflict = records / FILE_NAMES["decisions"]
        write_json(conflict, {"do_not_overwrite": True})
        before = conflict.read_bytes()
        with self.assertRaisesRegex(LexiconRepairError, "refusing to overwrite"):
            write_freeze(self.bundle, output_path=output, record_root=records)
        self.assertFalse(output.exists())
        self.assertFalse((records / FILE_NAMES["manifest"]).exists())
        self.assertEqual(conflict.read_bytes(), before)

    def test_verify_rebuilt_bundle_is_readonly_and_detects_each_modified_output(self):
        for altered_key in self.bundle:
            with self.subTest(altered_key=altered_key):
                case_root = self.temporary / altered_key
                output, records = case_root / "lexicon.json", case_root / "records"
                paths = write_freeze(self.bundle, output_path=output, record_root=records)
                before = {key: (Path(path).read_bytes(), Path(path).stat().st_mtime_ns) for key, path in paths.items()}
                self.assertEqual(paths, verify_freeze(self.bundle, output_path=output, record_root=records))
                for key, path in paths.items():
                    self.assertEqual((Path(path).read_bytes(), Path(path).stat().st_mtime_ns), before[key])
                corrupted = copy.deepcopy(self.bundle[altered_key])
                corrupted["test_only_tamper"] = True
                write_json(Path(paths[altered_key]), corrupted)
                with self.assertRaisesRegex(LexiconRepairError, "checksum differs"):
                    verify_freeze(self.bundle, output_path=output, record_root=records)
                self.assertTrue(read_json(Path(paths[altered_key]))["test_only_tamper"])

    def test_verify_missing_outputs_fails_without_creating_them(self):
        output, records = self.temporary / "missing.json", self.temporary / "missing-records"
        with self.assertRaisesRegex(LexiconRepairError, "missing or unsafe"):
            verify_freeze(self.bundle, output_path=output, record_root=records)
        self.assertFalse(output.exists())
        self.assertFalse(records.exists())

    def test_symlink_and_existing_directory_are_not_output_targets(self):
        records = self.temporary / "records"
        target = self.temporary / "lexicon.json"
        target.symlink_to(INPUTS["lexicon_path"])
        with self.assertRaisesRegex(LexiconRepairError, "symlink"):
            write_freeze(self.bundle, output_path=target, record_root=records)
        self.assertFalse(records.exists())
        directory_target = self.temporary / "directory-target"
        directory_target.mkdir()
        with self.assertRaisesRegex(LexiconRepairError, "refusing to overwrite"):
            write_freeze(self.bundle, output_path=directory_target, record_root=records)
        self.assertFalse(records.exists())


if __name__ == "__main__":
    unittest.main()
