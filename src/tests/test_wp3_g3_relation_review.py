from __future__ import annotations

import copy
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPOSITORY_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from build_lex import terminology_g3_relation_review as relation  # noqa: E402
from build_lex.terminology_g3_form_reference import (  # noqa: E402
    G3FormReferenceError,
    create_form_review_session,
    read_form_review_session,
    save_form_decision,
    validate_form_review_frame,
)
from data.training_artifacts import (  # noqa: E402
    build_payload_manifest,
    canonical_sha256,
    sha256_file,
    write_canonical_json,
)


class G3RelationReviewV2Tests(unittest.TestCase):
    def _source(self, root: Path, *, include_method: bool = False):
        bundle = root / "source-bundle"
        (bundle / "normalized").mkdir(parents=True)
        (bundle / "candidate_projections").mkdir()

        direct_text = (
            "冻结页开头。“YYDS”是“永远的神”的拼音首字母缩写。"
            "另有“YYDS”是“永远滴神”的拼音缩写。冻结页结尾。\n"
            "大小写示例：“mp3”是“MP3”的书写变体。\n"
        )
        direct_path = bundle / "normalized/direct-page.txt"
        direct_path.write_text(direct_text, encoding="utf-8")

        rows = []
        for ordinal in range(1, 186):
            source_type = "abbreviation" if ordinal <= 52 else "homophonic pun"
            if ordinal == 53:
                meme = "亚子"
                meaning = "“亚子”是“样子”的谐音，仍须人工核对。"
            else:
                meme = f"候选{ordinal}"
                meaning = f"候选{ordinal}只提供语义背景，未明确写出形式关系。"
            rows.append(
                {
                    "source_row_ordinal": ordinal,
                    "meme": meme,
                    "meaning": meaning,
                    "origin": None,
                    "type_cn": "缩写" if source_type == "abbreviation" else "谐音",
                    "type_en": source_type,
                }
            )
        projection = {
            "schema_version": "wp3-g3-chime-form-candidate-projection/v1",
            "source_component_id": "chime-data-json",
            "raw_sha256": "1" * 64,
            "source_record_count": 1458,
            "allowed_types": ["abbreviation", "homophonic pun"],
            "candidate_count": 185,
            "rows": rows,
        }
        projection_path = bundle / "candidate_projections/chime-data-json.json"
        write_canonical_json(projection_path, projection)
        components = [
            {
                "component_id": "direct-page",
                "source_id": "china-daily-yyds-nbcs-hhh",
                "source_role": "direct_evidence",
                "acquisition_mode": "direct_fetch",
                "publisher": "China Daily",
                "requested_url": "https://example.test/direct",
                "final_url": "https://example.test/direct",
                "snapshot_url": "https://example.test/direct",
                "normalized_file": "normalized/direct-page.txt",
                "normalized_sha256": sha256_file(direct_path),
                "candidate_projection": None,
                "evidence_eligible": True,
                "human_review_required": True,
            },
            {
                "component_id": "chime-data-json",
                "source_id": "chime-data",
                "source_role": "candidate_pool",
                "acquisition_mode": "user_supplied_archive",
                "publisher": "GitHub raw content",
                "requested_url": "https://example.test/chime.json",
                "final_url": "https://example.test/chime.json",
                "snapshot_url": "https://example.test/chime.json",
                "normalized_file": None,
                "normalized_sha256": None,
                "candidate_projection": {
                    "schema_version": "wp3-g3-chime-form-candidate-projection/v1",
                    "file": "candidate_projections/chime-data-json.json",
                    "size_bytes": projection_path.stat().st_size,
                    "sha256": sha256_file(projection_path),
                    "count": 185,
                    "allowed_types": ["abbreviation", "homophonic pun"],
                },
                "evidence_eligible": True,
                "human_review_required": True,
            },
        ]
        if include_method:
            method_path = bundle / "normalized/method-page.txt"
            method_path.write_text(
                "“ABC”是“甲乙丙”的拼音缩写。\n", encoding="utf-8"
            )
            components.append(
                {
                    "component_id": "method-page",
                    "source_id": "method-paper",
                    "source_role": "method_only",
                    "acquisition_mode": "direct_fetch",
                    "publisher": "Method publisher",
                    "requested_url": "https://example.test/method",
                    "final_url": "https://example.test/method",
                    "snapshot_url": "https://example.test/method",
                    "normalized_file": "normalized/method-page.txt",
                    "normalized_sha256": sha256_file(method_path),
                    "candidate_projection": None,
                    "evidence_eligible": False,
                    "human_review_required": False,
                }
            )
        return {
            "source_bundle_id": "wp3g3sources-" + "2" * 64,
            "target": str(bundle),
            "payload_manifest_sha256": "3" * 64,
            "components": components,
        }

    def _seeds(self, root: Path, seeds=None) -> Path:
        if seeds is None:
            seeds = [
                {
                    "seed_id": "g3seed-yyds-standard",
                    "source_id": "china-daily-yyds-nbcs-hhh",
                    "surface": "YYDS",
                    "canonical": "永远的神",
                    "proposed_family": "phonetic_variant",
                    "phonetic_scan_enabled": False,
                }
            ]
        path = root / "seeds.json"
        write_canonical_json(
            path,
            {
                "schema_version": "wp3-g3-form-relation-seeds/v2",
                "seed_role": "pair-location-hints-only-not-evidence",
                "seeds": seeds,
            },
        )
        return path

    def test_offline_extraction_chime_185_and_pair_seed_relocation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self._source(root)
            seeds = self._seeds(root)
            with patch.object(
                relation, "_validate_source_bundle_v2", return_value=source
            ):
                result = relation.extract_form_relation_claims_v2(
                    source_bundle_dir=source["target"],
                    workspace_root=REPOSITORY_ROOT,
                    seed_paths=[seeds],
                )
            self.assertEqual(result["report"]["chime_candidate_row_count"], 185)
            self.assertEqual(
                result["report"]["disposition_counts"][
                    "no_explicit_form_relation"
                ],
                184,
            )
            pairs = {(row["surface"], row["canonical"]) for row in result["items"]}
            self.assertEqual(pairs, {("YYDS", "永远的神"), ("亚子", "样子")})
            for item in result["items"]:
                selected = [
                    row
                    for row in result["evidence"]
                    if row["evidence_id"] in item["evidence_ids"]
                ]
                self.assertTrue(selected)
                for evidence in selected:
                    self.assertIn(item["surface"], evidence["quote"])
                    self.assertIn(item["canonical"], evidence["quote"])
                    self.assertIn(evidence["source_role"], {"direct_evidence", "candidate_pool"})
                    self.assertIn("acquisition_mode", evidence)
            wire = json.dumps(
                {
                    "claims": result["claims"],
                    "items": result["items"],
                    "evidence": result["evidence"],
                    "report": result["report"],
                },
                ensure_ascii=False,
            ).casefold()
            for forbidden in (
                '"meaning"',
                '"origin"',
                '"examples"',
                '"profanity"',
                '"offense"',
                '"gold"',
                '"abc"',
            ):
                self.assertNotIn(forbidden, wire)

    def test_conflicting_canonicals_remain_independent_items(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self._source(root)
            seeds = self._seeds(
                root,
                [
                    {
                        "seed_id": "g3seed-yyds-standard",
                        "source_id": "china-daily-yyds-nbcs-hhh",
                        "surface": "YYDS",
                        "canonical": "永远的神",
                        "proposed_family": "phonetic_variant",
                        "phonetic_scan_enabled": False,
                    },
                    {
                        "seed_id": "g3seed-yyds-di",
                        "source_id": "china-daily-yyds-nbcs-hhh",
                        "surface": "YYDS",
                        "canonical": "永远滴神",
                        "proposed_family": "phonetic_variant",
                        "phonetic_scan_enabled": False,
                    },
                ],
            )
            with patch.object(
                relation, "_validate_source_bundle_v2", return_value=source
            ):
                result = relation.extract_form_relation_claims_v2(
                    source_bundle_dir=source["target"],
                    workspace_root=REPOSITORY_ROOT,
                    seed_paths=[seeds],
                )
            yyds = [row for row in result["items"] if row["surface"] == "YYDS"]
            self.assertEqual({row["canonical"] for row in yyds}, {"永远的神", "永远滴神"})
            self.assertEqual(len({row["item_id"] for row in yyds}), 2)

    def test_case_only_pair_uses_two_distinct_exact_evidence_spans(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self._source(root)
            seeds = self._seeds(
                root,
                [
                    {
                        "seed_id": "g3seed-mp3-case",
                        "source_id": "china-daily-yyds-nbcs-hhh",
                        "surface": "mp3",
                        "canonical": "MP3",
                        "proposed_family": "orthographic_variant",
                        "phonetic_scan_enabled": False,
                    }
                ],
            )
            with patch.object(
                relation, "_validate_source_bundle_v2", return_value=source
            ):
                result = relation.extract_form_relation_claims_v2(
                    source_bundle_dir=source["target"],
                    workspace_root=REPOSITORY_ROOT,
                    seed_paths=[seeds],
                )
            matching = [
                row
                for row in result["items"]
                if row["surface"] == "mp3" and row["canonical"] == "MP3"
            ]
            self.assertEqual(len(matching), 1)
            evidence = next(
                row
                for row in result["evidence"]
                if row["evidence_id"] in matching[0]["evidence_ids"]
            )
            self.assertIn("“mp3”是“MP3”", evidence["quote"])

    def test_method_role_cannot_emit_relation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self._source(root, include_method=True)
            seeds = self._seeds(
                root,
                [
                    {
                        "seed_id": "g3seed-method-only",
                        "source_id": "method-paper",
                        "surface": "ABC",
                        "canonical": "甲乙丙",
                        "proposed_family": "phonetic_variant",
                        "phonetic_scan_enabled": False,
                    }
                ],
            )
            with patch.object(
                relation, "_validate_source_bundle_v2", return_value=source
            ):
                result = relation.extract_form_relation_claims_v2(
                    source_bundle_dir=source["target"],
                    workspace_root=REPOSITORY_ROOT,
                    seed_paths=[seeds],
                )
            self.assertNotIn(
                ("ABC", "甲乙丙"),
                {(row["surface"], row["canonical"]) for row in result["items"]},
            )
            self.assertEqual(
                result["report"]["seed_dispositions"][0]["disposition"],
                "source_role_not_relation_evidence",
            )

    def test_frame_dispatch_session_and_independent_replay(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self._source(root)
            seeds = self._seeds(root)
            with patch.object(
                relation, "_validate_source_bundle_v2", return_value=source
            ):
                frame = relation.build_form_relation_review_frame_v2(
                    source_bundle_dir=source["target"],
                    workspace_root=REPOSITORY_ROOT,
                    output_root=root / "frames",
                    seed_paths=[seeds],
                )
                validated = validate_form_review_frame(
                    frame["target"],
                    source_bundle_dir=source["target"],
                    workspace_root=REPOSITORY_ROOT,
                )
                self.assertEqual(validated["frame_id"], frame["frame_id"])
                session_path = root / "working/session.json"
                session = create_form_review_session(
                    frame_dir=frame["target"],
                    source_bundle_dir=source["target"],
                    workspace_root=REPOSITORY_ROOT,
                    session_path=session_path,
                    reviewer_id="reviewer-v2",
                )
                item = frame["items"][0]
                saved = save_form_decision(
                    frame_dir=frame["target"],
                    source_bundle_dir=source["target"],
                    workspace_root=REPOSITORY_ROOT,
                    session_path=session_path,
                    item_id=item["item_id"],
                    decision={
                        "action": "accept",
                        "surface": item["surface"],
                        "canonical": item["canonical"],
                        "family": item["proposed_family"],
                        "phonetic_scan_enabled": item["phonetic_scan_enabled"],
                        "evidence_ids": item["evidence_ids"],
                        "notes": "",
                    },
                    confirm=True,
                    expected_revision=session["revision"],
                )
                self.assertEqual(
                    read_form_review_session(session_path)["revision"],
                    saved["revision"],
                )

                tampered = root / f".{frame['frame_id']}.tampered"
                shutil.copytree(frame["target"], tampered)
                evidence = json.loads((tampered / "evidence.json").read_text())
                evidence[0]["snapshot_start"] += 1
                write_canonical_json(tampered / "evidence.json", evidence)
                write_canonical_json(
                    tampered / "payload_manifest.json",
                    build_payload_manifest(tampered),
                )
                with self.assertRaises(G3FormReferenceError):
                    relation.validate_form_relation_review_frame_v2(
                        tampered,
                        source_bundle_dir=source["target"],
                        workspace_root=REPOSITORY_ROOT,
                    )

    def test_chime_projection_type_count_tamper_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self._source(root)
            projection_path = (
                Path(source["target"])
                / "candidate_projections/chime-data-json.json"
            )
            projection = json.loads(projection_path.read_text())
            projection["rows"][0]["type_en"] = "homophonic pun"
            write_canonical_json(projection_path, projection)
            component = next(
                row
                for row in source["components"]
                if row["component_id"] == "chime-data-json"
            )
            component["candidate_projection"]["sha256"] = sha256_file(projection_path)
            component["candidate_projection"]["size_bytes"] = projection_path.stat().st_size
            with patch.object(
                relation, "_validate_source_bundle_v2", return_value=source
            ):
                with self.assertRaisesRegex(G3FormReferenceError, "52/133"):
                    relation.extract_form_relation_claims_v2(
                        source_bundle_dir=source["target"],
                        workspace_root=REPOSITORY_ROOT,
                    )

    def test_v1_projection_discards_old_quote_and_evidence_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "old.json"
            write_canonical_json(
                path,
                {
                    "schema_version": "wp3-g3-form-extraction/v1",
                    "extractor": "human-prepared-form-relations/v1",
                    "evidence": [
                        {
                            "evidence_id": "g3ev-old",
                            "source_id": "source-one",
                            "quote": "OLD QUOTE MUST NOT CROSS",
                            "occurrence_ordinal": 1,
                            "relation_note": "old",
                        }
                    ],
                    "items": [
                        {
                            "item_id": "g3form-one",
                            "surface": "ABC",
                            "canonical": "甲乙丙",
                            "proposed_family": "phonetic_variant",
                            "phonetic_scan_enabled": False,
                            "evidence_ids": ["g3ev-old"],
                        }
                    ],
                },
            )
            seeds = relation.pair_seeds_from_v1_extraction(path)
            wire = json.dumps(seeds, ensure_ascii=False)
            self.assertNotIn("OLD QUOTE", wire)
            self.assertNotIn("g3ev-old", wire)
            self.assertEqual(seeds[0]["source_id"], "source-one")


if __name__ == "__main__":
    unittest.main()
