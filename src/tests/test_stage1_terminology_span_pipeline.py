from __future__ import annotations

import copy
import hashlib
import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPOSITORY_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from build_lex.terminology_resolution import (  # noqa: E402
    HUMAN_RESOLUTION_VERSION,
    RESOLUTION_GOLD_VERSION,
    ResolutionCheckpoint,
    TerminologyResolutionError,
    _build_merged_resolution,
    _registrable_domain,
    _sources_sufficient,
    build_resolution_execution_contract,
    calibrate_resolution_gate,
    finalize_library,
    merge_human_resolution,
    normalize_entry,
    publish_stage1_library,
    run_resolution,
    validate_library,
    validate_resolution_artifact,
    validate_stage1_published_library,
)
from build_lex.terminology_span_pipeline import (  # noqa: E402
    FEATURE_NAMES,
    GATE_SCHEMA_VERSION,
    RANKER_POLICY_VERSION,
    SpanResultStore,
    SpanProviderError,
    TerminologySpanError,
    _canonical_sha,
    _forbidden_key_paths,
    build_census_documents,
    build_full_tasks,
    build_pilot_review_package,
    build_qwen_request,
    clopper_pearson_interval,
    exact_occurrences,
    load_pipeline_config,
    normalize_human_spans,
    normalize_qwen_annotation,
    qwen_provider_config,
    qwen_contract,
    validate_inconclusive_pilot_gate,
    validate_pilot_decision,
    validate_tune_review_prerequisite,
    validation_extension_tasks,
    write_census_artifact,
    write_pilot_decision,
)
from build_lex.train_only import (  # noqa: E402
    PILOT_GATED_TERMINOLOGY_MANIFEST_VERSION,
    resolve_train_input,
    validate_lexicon_target,
)
from data.training_artifacts import write_canonical_json  # noqa: E402


CONFIG_PATH = REPOSITORY_ROOT / "config/stage1/terminology_span_pipeline.json"


def entry(term: str, definition: str = "测试释义") -> dict:
    return {
        "term": term,
        "definition": definition,
        "usage_notes": "结合上下文理解。",
        "ambiguity_notes": "可能存在其他字面义。",
        "variants": [],
    }


class SpanOffsetTests(unittest.TestCase):
    def test_unicode_repetition_overlap_and_authoritative_offsets(self) -> None:
        content = "哈😀哈😀哈；舔狗又说舔狗"
        self.assertEqual(exact_occurrences("哈哈哈", "哈哈"), [(0, 2), (1, 3)])
        normalized = normalize_qwen_annotation(
            {
                "spans": [
                    {
                        "surface": "😀哈",
                        "occurrence_ordinal": 2,
                        "reason": "emoji 与中文混合表达",
                    },
                    {
                        "surface": "舔狗",
                        "occurrence_ordinal": 2,
                        "reason": "网络用语",
                    },
                ],
                "record_reason": "存在两个需理解表达",
            },
            content,
        )
        by_surface = {row["surface"]: row for row in normalized["spans"]}
        self.assertEqual(content[by_surface["😀哈"]["start"] : by_surface["😀哈"]["end"]], "😀哈")
        self.assertEqual(by_surface["舔狗"]["start"], content.rfind("舔狗"))

    def test_model_offsets_are_never_accepted(self) -> None:
        with self.assertRaises(SpanProviderError):
            normalize_qwen_annotation(
                {
                    "spans": [
                        {
                            "surface": "男同",
                            "occurrence_ordinal": 1,
                            "reason": "身份简称",
                            "start": 999,
                            "end": 1001,
                        }
                    ],
                    "record_reason": "测试",
                },
                "这里提到男同",
            )

    def test_nested_and_repeated_human_spans(self) -> None:
        rows = normalize_human_spans(
            "基佬一词和基佬文化",
            [
                {"surface": "基佬", "occurrence_ordinal": 1},
                {"surface": "基佬文化", "occurrence_ordinal": 1},
                {"surface": "基佬", "occurrence_ordinal": 2},
            ],
        )
        self.assertEqual(len(rows), 3)
        self.assertTrue(any(left["start"] == right["start"] and left["end"] < right["end"] for left in rows for right in rows))


class CensusAndContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_pipeline_config(CONFIG_PATH)

    def test_prompt_payload_contains_content_only(self) -> None:
        task = {"task_id": "q-1", "blind_alias": "Q-0001", "content": "男同不是异常类别"}
        payload = build_qwen_request(task, qwen_provider_config(self.config, environ={}))
        wire = json.dumps(payload, ensure_ascii=False)
        self.assertIn("男同不是异常类别", wire)
        for forbidden in ("targeted_group", '"label"', '"category"', '"hateful"'):
            self.assertNotIn(forbidden, wire)
        contract = qwen_contract(
            qwen_provider_config(self.config, environ={}),
            config_sha256=_canonical_sha(self.config),
        )
        self.assertRegex(contract["span_implementation_sha256"], r"^[0-9a-f]{64}$")

    def test_full_text_is_never_silently_truncated(self) -> None:
        config = copy.deepcopy(self.config)
        config["expected_fit_count"] = 1
        config["extraction"]["max_text_chars_per_record"] = 3
        frozen = resolve_train_input(
            train_records=[
                {"id": "long", "content": "超过三字的完整原文", "quadruples": []}
            ],
            formal=False,
        )
        with self.assertRaisesRegex(TerminologySpanError, "instead of truncating"):
            build_census_documents(frozen, config)

    def test_a1_review_does_not_require_or_expose_locked_a2(self) -> None:
        tasks = [
            {
                "task_id": f"task-{index}",
                "blind_alias": f"Q-{index:03d}",
                "content": f"调试文本 {index}",
            }
            for index in range(300)
        ]
        audit_tasks = [
            {
                "task_id": f"task-{index}",
                "record_id": f"record-{index}",
                "phase": "tune" if index < 200 else "validation",
                "sampling_stratum": "random_baseline",
            }
            for index in range(300)
        ]
        review_order = [
            {
                "case_id": f"case-{index}",
                "blind_alias": f"B-{index:03d}",
                **audit_tasks[index],
                "repeat_group": None,
                "occurrence": 0,
            }
            for index in range(300)
        ]
        review_order.extend(
            {
                "case_id": f"repeat-{index}",
                "blind_alias": f"R-{index:03d}",
                **audit_tasks[index],
                "repeat_group": f"group-{index}",
                "occurrence": 1,
            }
            for index in range(60)
        )
        census = {
            "census_id": "spancensus-" + "a" * 64,
            "metadata": {
                "config_sha256": _canonical_sha(self.config),
                "fit_data_sha256": "b" * 64,
            },
            "pilot_public": {"tasks": tasks},
            "pilot_audit": {"tasks": audit_tasks, "review_order": review_order},
            "records": [
                {"record_id": f"record-{index}", "rule_proposals": []}
                for index in range(300)
            ],
        }

        class TuneOnlyStore:
            contract_sha256 = "c" * 64

            @staticmethod
            def terminal_rows():
                return {
                    f"task-{index}": {
                        "annotation": {"spans": []},
                        "terminal_status": "empty",
                    }
                    for index in range(200)
                }

            @staticmethod
            def failures():
                return {}

        with tempfile.TemporaryDirectory() as directory, patch(
            "build_lex.terminology_span_pipeline.validate_census_artifact",
            return_value=census,
        ), patch(
            "build_lex.terminology_span_pipeline._open_span_store",
            return_value=TuneOnlyStore(),
        ):
            result = build_pilot_review_package(
                census_dir=Path(directory) / "census",
                checkpoint_path=Path(directory) / "pilot.sqlite3",
                config=self.config,
                templates_dir=REPOSITORY_ROOT / "tools/terminology_span_review",
                output_root=Path(directory) / "review",
                phase="tune",
            )
            cases = json.loads(
                (Path(result["directory"]) / "cases.json").read_text(encoding="utf-8")
            )
            self.assertEqual(len(cases), 200)
            self.assertFalse(any(int(row["content"].rsplit(" ", 1)[-1]) >= 200 for row in cases))
            with zipfile.ZipFile(result["archive"]) as archive:
                self.assertFalse(
                    any("source_map" in name or "/.private/" in name for name in archive.namelist())
                )
            private_maps = list((Path(directory) / "review" / ".private").glob("*.json"))
            self.assertEqual(len(private_maps), 1)
            self.assertEqual(private_maps[0].stat().st_mode & 0o777, 0o600)
            annotations_path = Path(directory) / "tune.annotations.json"
            write_canonical_json(
                annotations_path,
                {
                    "schema_version": "terminology-span-human-review/v1",
                    "package_id": result["package_id"],
                    "reviewer_id": "reviewer",
                    "annotations": [
                        {
                            "case_id": row["case_id"],
                            "needs_explanation": False,
                            "spans": [],
                            "issue_tags": [],
                            "notes": "",
                        }
                        for row in cases
                    ],
                },
            )
            prerequisite = validate_tune_review_prerequisite(
                census_dir=Path(directory) / "census",
                checkpoint_path=Path(directory) / "pilot.sqlite3",
                config=self.config,
                review_package_dir=result["directory"],
                annotation_path=annotations_path,
            )
            self.assertEqual(prerequisite["qwen_contract_sha256"], "c" * 64)

    def test_census_and_extension_blocks_are_disjoint(self) -> None:
        records = []
        for index in range(520):
            content = f"记录{index}包含词{index % 17}、tag{index % 11}和😀"
            records.append(
                {
                    "id": str(index),
                    "content": content,
                    # Deliberately present in source input: A0 must not expose it.
                    "label": "hate" if index % 2 else "non-hate",
                    "quadruples": [
                        {
                            "target": None,
                            "argument": content,
                            "targeted_group": ["non-hate"],
                            "hateful": "non-hate",
                        }
                    ],
                }
            )
        frozen = resolve_train_input(train_records=records, formal=False)
        config = copy.deepcopy(self.config)
        config["expected_fit_count"] = len(records)
        documents = build_census_documents(frozen, config)
        self.assertEqual(len(documents["pilot_public"]["tasks"]), 300)
        self.assertEqual(len(documents["pilot_audit"]["review_order"]), 360)
        self.assertFalse(_forbidden_key_paths(documents["records"]))
        self.assertFalse(_forbidden_key_paths(documents["pilot_public"]))
        with tempfile.TemporaryDirectory() as directory:
            census_dir = write_census_artifact(documents, output_root=directory)
            _, block_one, _ = validation_extension_tasks(
                frozen=frozen, census_dir=census_dir, block=1
            )
            _, block_two, _ = validation_extension_tasks(
                frozen=frozen, census_dir=census_dir, block=2
            )
            base = {row["task_id"] for row in documents["pilot_public"]["tasks"]}
            first = {row["task_id"] for row in block_one}
            second = {row["task_id"] for row in block_two}
            self.assertEqual((len(first), len(second)), (100, 100))
            self.assertFalse(base & first or base & second or first & second)
            self.assertEqual(build_full_tasks(frozen=frozen, census=documents), build_full_tasks(frozen=frozen, census=documents))

    def test_pass_approval_is_invalidated_by_config_drift(self) -> None:
        gate = {
            "schema_version": GATE_SCHEMA_VERSION,
            "gate_kind": "span-pilot",
            "status": "PASS",
            "census_id": "spancensus-" + "a" * 64,
            "config_sha256": _canonical_sha(self.config),
            "qwen_contract_sha256": "b" * 64,
            "review_package_id": "review-1",
            "annotation_sha256": "c" * 64,
            "review_bindings_sha256": "d" * 64,
            "ranker": {
                "policy": RANKER_POLICY_VERSION,
                "feature_names": list(FEATURE_NAMES),
                "coefficients": [0.0] * len(FEATURE_NAMES),
                "intercept": 0.0,
            },
            "auto_threshold": {"threshold": 0.5},
        }
        gate["gate_sha256"] = _canonical_sha(gate)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "decision.json"
            write_pilot_decision(
                gate=gate,
                decision="PASS",
                reviewer_id="reviewer",
                notes="checked",
                output_path=path,
            )
            validate_pilot_decision(
                path,
                config=self.config,
                qwen_contract_sha256="b" * 64,
            )
            changed = copy.deepcopy(self.config)
            changed["qwen"]["max_tokens"] += 1
            with self.assertRaises(TerminologySpanError):
                validate_pilot_decision(path, config=changed)

    def test_fail_decision_cannot_unlock_full_scan(self) -> None:
        gate = {
            "schema_version": GATE_SCHEMA_VERSION,
            "gate_kind": "span-pilot",
            "status": "FAIL",
            "census_id": "spancensus-" + "a" * 64,
            "config_sha256": _canonical_sha(self.config),
            "qwen_contract_sha256": "b" * 64,
            "review_package_id": "review-1",
            "annotation_sha256": "c" * 64,
            "review_bindings_sha256": "d" * 64,
        }
        gate["gate_sha256"] = _canonical_sha(gate)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "decision.json"
            write_pilot_decision(
                gate=gate,
                decision="FAIL",
                reviewer_id="reviewer",
                notes="failed",
                output_path=path,
            )
            with self.assertRaises(TerminologySpanError):
                validate_pilot_decision(path, require_pass=True)

    def test_extension_requires_only_a2_denominator_reasons(self) -> None:
        census = {
            "census_id": "spancensus-" + "a" * 64,
            "metadata": {"config_sha256": _canonical_sha(self.config)},
        }
        base_gate = {
            "schema_version": GATE_SCHEMA_VERSION,
            "gate_kind": "span-pilot",
            "status": "INCONCLUSIVE",
            "reasons": ["insufficient_recall_denominator"],
            "census_id": census["census_id"],
            "config_sha256": _canonical_sha(self.config),
            "qwen_contract_sha256": "b" * 64,
            "validation_extension_blocks": [],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gate.json"
            gate = dict(base_gate)
            gate["gate_sha256"] = _canonical_sha(gate)
            write_canonical_json(path, gate)
            accepted = validate_inconclusive_pilot_gate(
                path,
                census=census,
                config=self.config,
                qwen_contract_sha256="b" * 64,
            )
            self.assertEqual(accepted["validation_extension_blocks"], [])

            gate = dict(base_gate)
            gate["reasons"] = ["insufficient_tune_ranker_classes"]
            gate["gate_sha256"] = _canonical_sha(gate)
            write_canonical_json(path, gate)
            with self.assertRaisesRegex(TerminologySpanError, "locked A2"):
                validate_inconclusive_pilot_gate(
                    path,
                    census=census,
                    config=self.config,
                    qwen_contract_sha256="b" * 64,
                )

    def test_ambiguous_reserved_attempt_can_be_human_resolved_after_cap(self) -> None:
        provider = qwen_provider_config(self.config, environ={})
        contract = qwen_contract(
            provider, config_sha256=_canonical_sha(self.config)
        )
        content = "这里有舔狗一词"
        task = {
            "task_id": "task-ambiguous",
            "blind_alias": "Q-ambiguous",
            "content": content,
        }
        with tempfile.TemporaryDirectory() as directory:
            store = SpanResultStore(
                Path(directory) / "span.sqlite3",
                contract=contract,
                fit_data_sha256="a" * 64,
            )
            for attempt in range(1, provider.max_attempts_per_task + 1):
                reservation = store.reserve(
                    task_id=task["task_id"],
                    request_sha256="b" * 64,
                    max_attempts=provider.max_attempts_per_task,
                )
                self.assertIsNotNone(reservation)
                if attempt < provider.max_attempts_per_task:
                    store.finish_failure(
                        reservation,
                        error="transport failed",
                        retryable=True,
                        http_status=None,
                        response=None,
                    )
            inserted = store.add_exception_resolutions(
                tasks={task["task_id"]: task},
                resolution_document={
                    "schema_version": "terminology-span-exception-resolution/v1",
                    "reviewer_id": "reviewer",
                    "resolutions": [
                        {
                            "task_id": task["task_id"],
                            "spans": [
                                {"surface": "舔狗", "occurrence_ordinal": 1}
                            ],
                            "notes": "三次物理尝试已耗尽，人工补充",
                        }
                    ],
                },
            )
            self.assertEqual(inserted, 1)
            self.assertEqual(
                store.terminal_rows()[task["task_id"]]["terminal_status"],
                "exception_resolved",
            )

    def test_exact_confidence_interval(self) -> None:
        lower, upper = clopper_pearson_interval(190, 200)
        self.assertGreater(lower, 0.90)
        self.assertLess(upper, 1.0)
        _, zero_error_upper = clopper_pearson_interval(0, 100)
        self.assertLess(zero_error_upper, 0.05)

    def test_main_experiment_validator_dispatches_pilot_gated_publication(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "lex-test"
            target.mkdir()
            for name, value in (
                ("data_ref.json", {}),
                ("lexicon.json", {}),
                (
                    "manifest.json",
                    {"schema_version": PILOT_GATED_TERMINOLOGY_MANIFEST_VERSION},
                ),
                ("provenance.json", {}),
                ("payload_manifest.json", {}),
            ):
                write_canonical_json(target / name, value)
            expected = {"lexicon_build_id": "lex-" + "d" * 64}
            with patch(
                "build_lex.terminology_resolution.validate_stage1_published_library",
                return_value=expected,
            ) as validator:
                self.assertEqual(
                    validate_lexicon_target(
                        target,
                        workspace_root=directory,
                    ),
                    expected,
                )
            validator.assert_called_once_with(target, workspace_root=directory)

    def test_stage1_publication_revalidates_complete_lifecycle(self) -> None:
        workspace = REPOSITORY_ROOT
        with tempfile.TemporaryDirectory(
            prefix=".terminology-publication-test-", dir=workspace
        ) as directory:
            root = Path(directory)
            frame_id = "termspan-" + "1" * 64
            resolution_id = "termres-" + "2" * 64
            library_id = "termlib-" + "3" * 64
            pilot_sha = "4" * 64
            audit_sha = "5" * 64
            gate_sha = "6" * 64
            fit_sha = "7" * 64
            content = "这里只是合成术语文本"
            frame = {
                "frame_id": frame_id,
                "metadata": {
                    "qwen_contract_sha256": "8" * 64,
                    "pilot_decision_sha256": pilot_sha,
                    "fit_data_sha256": fit_sha,
                    "record_count": 1,
                },
                "records": [
                    {
                        "record_id": "record-1",
                        "content": content,
                        "content_sha256": hashlib.sha256(
                            content.encode("utf-8")
                        ).hexdigest(),
                    }
                ],
            }
            candidate_id = "termcand-" + "9" * 20
            resolution = {
                "resolution_id": resolution_id,
                "metadata": {
                    "formal_complete": True,
                    "span_frame_id": frame_id,
                    "full_audit_decision_sha256": audit_sha,
                    "resolution_gate_sha256": gate_sha,
                },
                "rows": [
                    {
                        "candidate_id": candidate_id,
                        "rank": 1,
                        "term": "合成术语",
                        "route": "human_required",
                        "entry": None,
                        "diagnostic": {},
                    }
                ],
                "queue": [{"candidate_id": candidate_id}],
                "qc": [],
            }
            resolution["execution_contract"] = build_resolution_execution_contract(
                config=self.config,
                qwen=object(),
                deepseek=object(),
                searcher=object(),
                similarity=object(),
            )
            human = {
                "schema_version": HUMAN_RESOLUTION_VERSION,
                "resolution_id": resolution_id,
                "reviewer_id": "reviewer",
                "decisions": [
                    {
                        "candidate_id": candidate_id,
                        "action": "exclude",
                        "entry": None,
                        "notes": "合成测试排除项",
                    }
                ],
                "qc_reviews": [],
            }
            merged = _build_merged_resolution(resolution, human)
            library = {
                "library_id": library_id,
                "terms": [],
                "metadata": {
                    "resolution_id": resolution_id,
                    "merged_sha256": merged["merged_sha256"],
                },
            }
            pilot = {"decision_sha256": pilot_sha}
            audit = {"decision_sha256": audit_sha}
            gate = {
                "gate_sha256": gate_sha,
                "execution_contract_sha256": resolution["execution_contract"][
                    "contract_sha256"
                ],
            }
            data_dependency = {
                "schema_version": "stage1-dependency-ref/v1",
                "artifact_kind": "data",
                "artifact_id": "data-" + "a" * 64,
                "payload_manifest_sha256": "b" * 64,
            }
            partition_dependency = {
                "schema_version": "stage1-dependency-ref/v1",
                "artifact_kind": "train-partition",
                "artifact_id": "tpart-" + "c" * 64,
                "payload_manifest_sha256": "d" * 64,
            }
            frozen = SimpleNamespace(
                records=({"id": "record-1", "content": content},),
                data_build_id=data_dependency["artifact_id"],
                train_data_sha256=fit_sha,
                train_ids_sha256="e" * 64,
                source_train_data_sha256="f" * 64,
                source_train_ids_sha256="0" * 64,
                source_mode="data_ref+train_partition",
                data_ref=data_dependency,
                train_partition_ref=partition_dependency,
                train_partition_dependency=partition_dependency,
            )
            paths = {
                "frame": root / "frame",
                "resolution": root / "resolution",
                "library": root / "library",
            }
            for name, path in paths.items():
                path.mkdir()
                write_canonical_json(path / "manifest.json", {"kind": name})
            pilot_path = root / "pilot.json"
            audit_path = root / "audit.json"
            gate_path = root / "gate.json"
            merged_path = root / "merged.json"
            human_path = root / "human.json"
            for path, value in (
                (pilot_path, pilot),
                (audit_path, audit),
                (gate_path, gate),
                (merged_path, merged),
                (human_path, human),
            ):
                write_canonical_json(path, value)

            with (
                patch("build_lex.train_only.resolve_train_input", return_value=frozen),
                patch(
                    "build_lex.train_only._resolve_portable_train_input",
                    return_value=frozen,
                ),
                patch(
                    "build_lex.terminology_resolution.validate_full_span_frame",
                    return_value=frame,
                ),
                patch(
                    "build_lex.terminology_resolution.validate_pilot_decision",
                    return_value=pilot,
                ),
                patch(
                    "build_lex.terminology_resolution.validate_full_audit_decision",
                    return_value=audit,
                ),
                patch(
                    "build_lex.terminology_resolution.validate_resolution_gate",
                    return_value=gate,
                ),
                patch(
                    "build_lex.terminology_resolution.validate_resolution_artifact",
                    return_value=resolution,
                ),
                patch(
                    "build_lex.terminology_resolution.validate_library",
                    return_value=library,
                ),
            ):
                locator = publish_stage1_library(
                    library_dir=paths["library"],
                    span_frame_dir=paths["frame"],
                    pilot_decision_path=pilot_path,
                    full_audit_decision_path=audit_path,
                    resolution_gate_path=gate_path,
                    resolution_dir=paths["resolution"],
                    merged_path=merged_path,
                    human_resolution_path=human_path,
                    config=self.config,
                    data_ref=root / "unused-data-ref.json",
                    train_partition_ref=root / "unused-partition-ref.json",
                    workspace_root=workspace,
                    target_root=root / "published",
                    write_ref=root / "lexicon_ref.json",
                )
                report = validate_stage1_published_library(
                    locator["target_path"], workspace_root=workspace
                )
                self.assertEqual(report["lexicon_build_id"], locator["artifact_id"])
                self.assertTrue(report["fit_only_verified"])
                self.assertEqual(report["lexicon_ids"], [])


class FakeQwen:
    def complete_json(self, stage: str, payload: dict) -> dict:
        if stage == "qwen_self_explanation":
            return {"malformed": True}
        if stage == "qwen_web_rewrite":
            return entry(payload["term"], "由两份独立网页证据支持的释义")
        raise AssertionError(stage)


class FakeDeepSeek:
    def complete_json(self, stage: str, payload: dict) -> dict:
        if stage == "deepseek_web_review":
            return {
                "supported": True,
                "material_conflict": False,
                "evidence_ids": [row["evidence_id"] for row in payload["evidence"]],
                "reason": "证据直接支持",
            }
        raise AssertionError(stage)


class FakeSearch:
    def search(self, query: str) -> list[dict[str, str]]:
        return [
            {"url": "https://one.example/a", "title": "一", "snippet": query},
            {"url": "https://two.test/b", "title": "二", "snippet": query},
        ]


class FakeFetch:
    def fetch(self, url: str) -> dict:
        return {
            "content_sha256": ("a" if "one.example" in url else "b") * 64,
            "text_excerpt": "网页中的术语解释证据",
        }


class ResolutionGateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_pipeline_config(CONFIG_PATH)

    def test_entry_is_strictly_category_free(self) -> None:
        value = entry("舔狗")
        self.assertEqual(normalize_entry(value, expected_term="舔狗")["term"], "舔狗")
        value["category"] = "Sexism"
        with self.assertRaises(TerminologyResolutionError):
            normalize_entry(value, expected_term="舔狗")

    def test_empty_registry_requires_two_independent_domains(self) -> None:
        self.assertFalse(
            _sources_sufficient([{"domain": "example.com"}], self.config)[0]
        )
        self.assertTrue(
            _sources_sufficient(
                [{"domain": "example.com"}, {"domain": "example.org"}],
                self.config,
            )[0]
        )

    def test_registrable_domain_uses_frozen_public_suffix_list(self) -> None:
        expected = self.config["resolution"]["public_suffix_sha1"]
        self.assertEqual(
            _registrable_domain(
                "https://a.b.example.co.uk/path", expected_psl_sha1=expected
            ),
            "example.co.uk",
        )

    def test_resolution_checkpoint_enforces_physical_provider_cap(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = ResolutionCheckpoint(
                Path(directory) / "resolution.sqlite3",
                binding={"test": "binding"},
                attempt_budgets={"search": 1, "deepseek": 1},
            )
            calls = 0

            def fail_search():
                nonlocal calls
                calls += 1
                raise RuntimeError("transport failure")

            with self.assertRaisesRegex(
                TerminologyResolutionError, "physical-attempt cap exhausted"
            ):
                checkpoint.execute(
                    "termcand-test",
                    "search_1",
                    {"query": "测试"},
                    fail_search,
                )
            self.assertEqual(calls, 1)

    def test_qc_failure_disables_only_the_affected_auto_route(self) -> None:
        resolution = {
            "resolution_id": "termres-" + "a" * 64,
            "metadata": {"formal_complete": True},
            "rows": [
                {
                    "candidate_id": "termcand-" + "1" * 20,
                    "term": "自解释词",
                    "route": "auto_self",
                    "entry": entry("自解释词"),
                },
                {
                    "candidate_id": "termcand-" + "2" * 20,
                    "term": "联网词",
                    "route": "auto_web",
                    "entry": entry("联网词"),
                },
            ],
            "queue": [],
            "qc": [{"candidate_id": "termcand-" + "1" * 20}],
        }
        human = {
            "schema_version": HUMAN_RESOLUTION_VERSION,
            "resolution_id": resolution["resolution_id"],
            "reviewer_id": "reviewer",
            "decisions": [
                {
                    "candidate_id": "termcand-" + "1" * 20,
                    "action": "exclude",
                    "entry": None,
                    "notes": "自解释路线抽检失败",
                }
            ],
            "qc_reviews": [
                {
                    "candidate_id": "termcand-" + "1" * 20,
                    "material_error": True,
                    "notes": "实质错误",
                }
            ],
        }
        merged = _build_merged_resolution(resolution, human)
        self.assertEqual(merged["disabled_auto_routes"], ["auto_self"])
        self.assertEqual(
            [row["candidate_id"] for row in merged["entries"]],
            ["termcand-" + "2" * 20],
        )

    def test_resolution_branch_is_enabled_only_after_calibration(self) -> None:
        cases = []
        for branch in ("self_explanation", "web_evidence"):
            for index in range(100):
                cases.append(
                    {
                        "case_id": f"{branch}-{index}",
                        "branch": branch,
                        "auto_accept": True,
                        "human_acceptable": True,
                        "notes": "",
                    }
                )
        contract = build_resolution_execution_contract(
            config=self.config,
            qwen=object(),
            deepseek=object(),
            searcher=object(),
            similarity=object(),
        )
        gold = {
            "schema_version": RESOLUTION_GOLD_VERSION,
            "span_frame_id": "frame-1",
            "execution_contract_sha256": contract["contract_sha256"],
            "reviewer_id": "reviewer",
            "cases": cases,
        }
        with tempfile.TemporaryDirectory() as directory:
            gold_path = Path(directory) / "gold.json"
            contract_path = Path(directory) / "contract.json"
            gate_path = Path(directory) / "gate.json"
            write_canonical_json(gold_path, gold)
            write_canonical_json(contract_path, contract)
            gate = calibrate_resolution_gate(
                gold_path=gold_path,
                execution_contract_path=contract_path,
                span_frame_id="frame-1",
                config=self.config,
                output_path=gate_path,
            )
            self.assertTrue(gate["branches"]["self_explanation"]["enabled"])
            self.assertTrue(gate["branches"]["web_evidence"]["enabled"])

    def test_invalid_self_explanation_falls_back_to_web_and_human_complement(self) -> None:
        frame = {
            "frame_id": "frame-1",
            "records": [
                {
                    "record_id": "1",
                    "content": "这个词是舔狗",
                }
            ],
            "candidates": [
                {
                    "candidate_id": "termcand-ready",
                    "rank": 1,
                    "term": "舔狗",
                    "resolution_status": "ready",
                    "sample_occurrences": [
                        {
                            "record_id": "1",
                            "surface": "舔狗",
                            "start": 5,
                            "end": 7,
                        }
                    ],
                },
                {
                    "candidate_id": "termcand-human",
                    "rank": 2,
                    "term": "普通词",
                    "resolution_status": "human_span_required",
                    "sample_occurrences": [],
                },
            ],
        }
        audit = {"decision_sha256": "a" * 64}
        gate = {
            "gate_sha256": "b" * 64,
            "minimum_bge_similarity": 0.7,
            "branches": {
                "self_explanation": {"enabled": True},
                "web_evidence": {"enabled": True},
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with (
                patch(
                    "build_lex.terminology_resolution.validate_full_span_frame",
                    return_value=frame,
                ),
                patch(
                    "build_lex.terminology_resolution.validate_full_audit_decision",
                    return_value=audit,
                ),
                patch(
                    "build_lex.terminology_resolution.validate_resolution_gate",
                    return_value=gate,
                ),
            ):
                result_dir = run_resolution(
                    span_frame_dir=root / "frame",
                    full_audit_decision_path=root / "audit.json",
                    resolution_gate_path=root / "gate.json",
                    config=self.config,
                    checkpoint_path=root / "resolution.sqlite3",
                    output_root=root / "resolutions",
                    qwen=FakeQwen(),
                    deepseek=FakeDeepSeek(),
                    searcher=FakeSearch(),
                    fetcher=FakeFetch(),
                    similarity=lambda _left, _right: 1.0,
                )
            result = validate_resolution_artifact(result_dir)
            rows = {row["candidate_id"]: row for row in result["rows"]}
            self.assertEqual(rows["termcand-ready"]["route"], "auto_web")
            self.assertIn("error", rows["termcand-ready"]["diagnostic"]["self_explanation"])
            self.assertEqual(rows["termcand-human"]["route"], "human_required")
            self.assertEqual({row["candidate_id"] for row in result["queue"]}, {"termcand-human"})
            self.assertTrue(result["metadata"]["formal_complete"])

            human = {
                "schema_version": HUMAN_RESOLUTION_VERSION,
                "resolution_id": result["resolution_id"],
                "reviewer_id": "human-reviewer",
                "decisions": [
                    {
                        "candidate_id": "termcand-human",
                        "action": "exclude",
                        "entry": None,
                        "notes": "不具备额外理解价值",
                    }
                ],
                "qc_reviews": [
                    {
                        "candidate_id": "termcand-ready",
                        "material_error": False,
                        "notes": "证据充分",
                    }
                ],
            }
            human_path = root / "human.json"
            merged_path = root / "merged.json"
            write_canonical_json(human_path, human)
            merged = merge_human_resolution(
                resolution_dir=result_dir,
                human_path=human_path,
                output_path=merged_path,
            )
            self.assertEqual(merged["unresolved_count"], 0)
            self.assertEqual(len(merged["entries"]), 1)
            library_dir = finalize_library(
                resolution_dir=result_dir,
                merged_path=merged_path,
                output_root=root / "libraries",
            )
            library = validate_library(library_dir)
            self.assertEqual(library["entry_count"], 1)
            self.assertEqual(
                set(library["terms"][0]),
                {
                    "term",
                    "definition",
                    "usage_notes",
                    "ambiguity_notes",
                    "variants",
                },
            )
            self.assertTrue(library["catalog"][0]["lexicon_id"].startswith("lex:v2:"))


if __name__ == "__main__":
    unittest.main()
