import copy
import hashlib
import json
import sys
import tempfile
import unittest
from contextlib import contextmanager
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import jsonschema

import data.control_manifest as control_manifest_module
from data.context_manifest import finalize_context_budget
from data.control_manifest import (
    ControlManifestError,
    ControlUnavailableError,
    build_control_artifact,
    build_control_record,
    canonical_sha256,
    demo_query_overlap,
    lexical_normalize,
    lexicon_query_overlap,
    render_control_condition_item,
    resolve_control_config,
    round_written_similarity,
    validate_control_record,
    validate_control_ref,
    _prepare_token_costs,
    _search_match,
)
from model.stage1_registry import inventory_regular_file_tree


class CharacterTokenizer:
    def apply_chat_template(
        self, conversation, *, tokenize, add_generation_prompt, enable_thinking=False
    ):
        assert not tokenize and add_generation_prompt and not enable_thinking
        return "".join(f"<{row['role']}>{row['content']}" for row in conversation) + "<assistant>"

    def encode(self, text, add_special_tokens=False):
        assert not add_special_tokens
        return list(text)


class BatchedCharacterTokenizer(CharacterTokenizer):
    def __call__(self, texts, *, add_special_tokens, padding, truncation):
        assert not add_special_tokens and not padding and not truncation
        return {"input_ids": [list(text) for text in texts]}


TOKENIZER = CharacterTokenizer()
SYSTEM = "quad-json"
USER = "L={lexicons}\nD={examples}\nQ={text}"
CONTEXT_ID = "ctx-" + "c" * 64


def control_config(**changes):
    value = {
        "schema_version": "stage1-control-config/v1",
        "profile_name": "fixture",
        "source_class_order": ["terminology", "A"],
        "tokenizer": {"revision": "character/v1", "logical_path": "models/base/Qwen3-8B"},
    }
    value.update(changes)
    return value


def fixture_catalogs():
    lexicon_ids = ["lex:v2:" + character * 64 for character in "abcdef0123"]
    demo_ids = ["demo:v1:" + character * 64 for character in "456789abcd"]
    lexicons = {}
    demos = {}
    for index, item_id in enumerate(lexicon_ids):
        lexicons[item_id] = {
            "lexicon_id": item_id,
            "term": f"term{index:02d}",
            "definition": f"definition {index}",
            "usage_notes": "",
            "ambiguity_notes": "",
            "evidence_kind": "terminology",
            "variants": [f"variant{index:02d}"],
            "rendered_block": "LEXT00" if index == 0 else f"LEX{index:03d}",
            "source_split": "train",
        }
    for index, item_id in enumerate(demo_ids):
        demos[item_id] = {
            "demo_id": item_id,
            "source_record_id": str(100 + index),
            "content": "targetdemo" if index == 0 else f"alpha{index:02d}",
            "output_label": "A",
            "rendered_block": "DEMOT0" if index == 0 else f"DEM{index:03d}",
            "source_split": "train",
        }
    return lexicons, demos, lexicon_ids, demo_ids


def context_record(lexicons, demos, lexicon_ids, demo_ids, *, selected=True):
    selected_lexicons = [lexicon_ids[0]] if selected else []
    selected_demos = [demo_ids[0]] if selected else []
    lex_scores = []
    demo_scores = []
    for index, item_id in enumerate(lexicon_ids):
        lex_scores.append(
            {
                "lexicon_id": item_id,
                "source_class": "terminology",
                "written_similarity": 0.9 if index == 0 else index / 100,
            }
        )
    for index, item_id in enumerate(demo_ids):
        demo_scores.append(
            {
                "demo_id": item_id,
                "source_class": "A",
                "written_similarity": 0.9 if index == 0 else index / 100,
            }
        )
    record = {
        "context_build_id": CONTEXT_ID,
        "query": {"id": "99", "content": "QUERYTEXT", "gold": []},
        "selection": {
            "lexicons": {"prompt_order_before_budget": selected_lexicons},
            "demos": {
                "prompt_order_before_budget": selected_demos,
                "quota_assignments": (
                    [{"demo_id": demo_ids[0], "assigned_quota_class": "A", "quota_round": 0}]
                    if selected
                    else []
                ),
            },
        },
    }
    if selected:
        record["control_relevance"] = {"lexicons": lex_scores, "demos": demo_scores}
    return finalize_context_budget(
        record,
        lexicon_catalog=lexicons,
        demo_catalog=demos,
        system_prompt=SYSTEM,
        user_prompt_template=USER,
        tokenizer=TOKENIZER,
        max_sequence_tokens=1000,
        completion_reserve_tokens=20,
    )


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        b"".join(
            json.dumps(
                row,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
            + b"\n"
            for row in rows
        )
    )


def file_sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def payload_manifest(target):
    files = []
    for path in sorted(target.rglob("*"), key=lambda value: value.relative_to(target).as_posix()):
        if path.is_file() and path.name != "payload_manifest.json":
            files.append(
                {
                    "path": path.relative_to(target).as_posix(),
                    "sha256": file_sha(path),
                    "size": path.stat().st_size,
                }
            )
    return {"schema_version": "stage1-payload-manifest/v1", "files": files}


def write_context_artifact(root, record, lexicons, demos):
    target = root / "contexts" / CONTEXT_ID
    records_path = target / "context_manifest.dev.jsonl"
    write_jsonl(records_path, [record])
    write_jsonl(target / "catalogs" / "lexicon_pool.jsonl", list(lexicons.values()))
    write_jsonl(target / "catalogs" / "demo_pool.train.jsonl", list(demos.values()))
    write_json(
        target / "context_manifest.dev.meta.json",
        {
            "schema_version": "stage1-context-manifest/v1",
            "context_build_id": CONTEXT_ID,
            "record_count": 1,
            "records_sha256": file_sha(records_path),
            "sources": {
                "demo_pool": {"split": "train"},
                "lexicon_pool": {"train_only_verified": True},
            },
            "budget": {"tokenizer_revision": "character/v1"},
        },
    )
    write_json(target / "payload_manifest.json", payload_manifest(target))
    locator = {
        "schema_version": "stage1-locator-ref/v1",
        "artifact_kind": "context",
        "artifact_id": CONTEXT_ID,
        "target_path": str(target.resolve()),
        "payload_manifest_sha256": file_sha(target / "payload_manifest.json"),
    }
    ref = root / "refs" / "context_ref.json"
    write_json(ref, locator)
    return ref, target


def make_context_scientific(root, context_ref, target, tokenizer_root):
    dependency = {
        "schema_version": "stage1-dependency-ref/v1",
        "artifact_kind": "train-partition",
        "artifact_id": "tpart-" + "a" * 64,
        "payload_manifest_sha256": "b" * 64,
        "logical_repo_path": "train_partitions/tpart-" + "a" * 64,
    }
    inventory = inventory_regular_file_tree(
        tokenizer_root,
        workspace_root=root,
        label="fixture tokenizer",
        inventory_policy="all-regular-files/v1",
    )
    meta_path = target / "context_manifest.dev.meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["scientific_eligible"] = True
    meta["sources"]["demo_pool"]["partition"] = "fit"
    meta["sources"]["lexicon_pool"]["partition"] = "fit"
    meta["runtime_source_identity"] = {
        "tokenizer": {
            "declared_revision": "character/v1",
            "inventory": inventory,
            "constructor_policy": dict(
                control_manifest_module.TOKENIZER_CONSTRUCTOR_POLICY
            ),
        }
    }
    write_json(meta_path, meta)
    write_json(target / "train_partition_ref.json", dependency)
    write_json(
        target / "prepared_bundle.meta.json",
        {
            "retrieval_provenance": {
                "fit_only_demo_pool": True,
                "calibration_demo_excluded": True,
                "train_partition_dependency": dependency,
            }
        },
    )
    write_json(target / "payload_manifest.json", payload_manifest(target))
    locator = json.loads(context_ref.read_text(encoding="utf-8"))
    locator["payload_manifest_sha256"] = file_sha(target / "payload_manifest.json")
    write_json(context_ref, locator)
    return inventory


class LexicalPolicyTests(unittest.TestCase):
    def test_nfkc_ascii_lower_and_unicode_punctuation_separator_deletion(self):
        self.assertEqual(lexical_normalize("ＡB-C， DÉ"), "abcdÉ")
        self.assertEqual(lexical_normalize("É"), "É")
        self.assertTrue(
            lexicon_query_overlap(
                "ＡＢ，词", {"term": "ab词", "variants": []}
            )
        )
        self.assertTrue(demo_query_overlap("abcd", {"content": "xxab-cdyy"}))
        self.assertTrue(demo_query_overlap("abc", {"content": "xxabcxx"}))
        self.assertFalse(demo_query_overlap("abc", {"content": "xyz"}))

    def test_score_rounding_and_post_outcome_config_guard(self):
        self.assertEqual(str(round_written_similarity("0.123456785")), "0.12345678")
        self.assertEqual(str(round_written_similarity("0.123456795")), "0.12345680")
        for value in ("NaN", "Infinity"):
            with self.assertRaises(ControlManifestError):
                round_written_similarity(value)
        with self.assertRaises(ControlManifestError):
            resolve_control_config({**control_config(), "prediction_path": "forbidden.json"})


class MatchingAndRenderingTests(unittest.TestCase):
    def setUp(self):
        self.lexicons, self.demos, self.lexicon_ids, self.demo_ids = fixture_catalogs()
        self.context = context_record(
            self.lexicons, self.demos, self.lexicon_ids, self.demo_ids
        )

    def test_pl_pd_match_quotas_tokens_and_render_only_one_source(self):
        control = build_control_record(
            self.context,
            lexicon_catalog=self.lexicons,
            demo_catalog=self.demos,
            tokenizer=TOKENIZER,
            config=control_config(),
        )
        for condition in ("PL", "PD"):
            shape = control[condition]
            self.assertEqual(shape["status"], "ok")
            self.assertEqual(shape["used_tier_percent"], 10)
            self.assertEqual(
                shape["target_quota"],
                {"terminology": 1} if condition == "PL" else {"A": 1},
            )
            self.assertEqual(shape["target_block_tokens"], shape["replacement_block_tokens"])
            self.assertEqual(shape["token_delta_ratio"], 0.0)

    def test_batched_tokenizer_preserves_exact_match_and_tie_break(self):
        scalar = build_control_record(
            self.context,
            lexicon_catalog=self.lexicons,
            demo_catalog=self.demos,
            tokenizer=TOKENIZER,
            config=control_config(),
        )
        batched = build_control_record(
            self.context,
            lexicon_catalog=self.lexicons,
            demo_catalog=self.demos,
            tokenizer=BatchedCharacterTokenizer(),
            config=control_config(),
        )
        self.assertEqual(batched, scalar)
        control = batched
        self.assertEqual(control["PL"]["quota_source"], "catalog-evidence-kind")
        self.assertEqual(control["PD"]["quota_source"], "context-assigned-quota-class")

        item = render_control_condition_item(
            self.context,
            control,
            lexicon_catalog=self.lexicons,
            demo_catalog=self.demos,
            tokenizer=TOKENIZER,
            condition="PL",
            base_condition="CLD",
        )
        self.assertEqual(item["id"], self.context["query"]["id"])
        self.assertEqual(item["content"], self.context["query"]["content"])
        self.assertEqual(item["gt_quadruples"], self.context["query"]["gold"])
        self.assertEqual(item["context_manifest"]["demo_ids"], self.context["conditions"]["CLD"]["demo_ids"])
        self.assertNotEqual(item["context_manifest"]["lexicon_ids"], self.context["conditions"]["CLD"]["lexicon_ids"])
        original_user = self.context["conditions"]["CLD"]["messages"][1]["content"]
        rendered_user = item["messages_list"][0][1]["content"]
        self.assertIn(self.demos[self.demo_ids[0]]["rendered_block"], rendered_user)
        self.assertIn(self.demos[self.demo_ids[0]]["rendered_block"], original_user)

        validate_control_record(
            control,
            self.context,
            lexicon_catalog=self.lexicons,
            demo_catalog=self.demos,
            tokenizer=TOKENIZER,
            config=control_config(),
            allow_unavailable=False,
        )

    def test_large_combinatorial_frame_uses_exact_token_dp(self):
        catalog = {}
        candidates = []
        for index in range(50):
            item_id = f"demo:v1:{index:064x}"
            row = {
                "demo_id": item_id,
                "rendered_block": "XX",
                "content_sha256": f"{index + 100:064x}",
            }
            catalog[item_id] = row
            candidates.append(
                {
                    "item_id": item_id,
                    "source_class": "A",
                    "written_similarity": Decimal(index) / Decimal(100),
                    "catalog": row,
                }
            )
        result = _search_match(
            candidates=candidates,
            quotas={"A": 5},
            catalog=catalog,
            tokenizer=BatchedCharacterTokenizer(),
            token_costs=_prepare_token_costs(catalog, BatchedCharacterTokenizer()),
            target_tokens=len("\n\n".join(["XX"] * 5)),
            max_delta=Decimal("0.01"),
            max_states=1_000_000,
            class_order=["A"],
            target_class_slots=["A"] * 5,
        )
        self.assertGreater(result["search_space_upper_bound"], 1_000_000)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["replacement_ids"], [f"demo:v1:{index:064x}" for index in range(5)])

    def test_tier_expansion_and_boundary_ties_are_auditable(self):
        lexicons = copy.deepcopy(self.lexicons)
        lexicons[self.lexicon_ids[1]]["rendered_block"] = "TOO-LONG"
        context = context_record(lexicons, self.demos, self.lexicon_ids, self.demo_ids)
        control = build_control_record(
            context,
            lexicon_catalog=lexicons,
            demo_catalog=self.demos,
            tokenizer=TOKENIZER,
            config=control_config(),
        )
        self.assertEqual(control["PL"]["used_tier_percent"], 20)
        self.assertEqual([row["status"] for row in control["PL"]["tier_attempts"]], ["no-token-match", "ok"])

        tied = copy.deepcopy(context)
        tied["control_relevance"]["lexicons"][2]["written_similarity"] = 0.01
        tied["record_sha256"] = canonical_sha256(
            {key: value for key, value in tied.items() if key != "record_sha256"}
        )
        tied_control = build_control_record(
            tied,
            lexicon_catalog=lexicons,
            demo_catalog=self.demos,
            tokenizer=TOKENIZER,
            config=control_config(),
        )
        self.assertEqual(tied_control["PL"]["used_tier_percent"], 10)
        boundary = tied_control["PL"]["tier_attempts"][0]["class_boundaries"][
            "terminology"
        ]
        self.assertEqual(boundary["nearest_rank_count"], 1)
        self.assertEqual(boundary["included_count"], 2)

    def test_pd_exclusions_expand_to_thirty_percent(self):
        demos = copy.deepcopy(self.demos)
        demos[self.demo_ids[1]]["source_record_id"] = "99"
        demos[self.demo_ids[2]]["content"] = "xxQUERYTEXTxx"
        context = context_record(self.lexicons, demos, self.lexicon_ids, self.demo_ids)
        control = build_control_record(
            context,
            lexicon_catalog=self.lexicons,
            demo_catalog=demos,
            tokenizer=TOKENIZER,
            config=control_config(),
        )
        self.assertEqual(control["PD"]["status"], "ok")
        self.assertEqual(control["PD"]["used_tier_percent"], 30)
        evidence = {row["demo_id"]: row for row in control["PD"]["candidate_evidence"]}
        self.assertIn("query-source-record-id", evidence[self.demo_ids[1]]["exclusion_reasons"])
        self.assertIn("query-lexical-overlap", evidence[self.demo_ids[2]]["exclusion_reasons"])

    def test_empty_and_unavailable_are_distinct_and_unavailable_cannot_render(self):
        empty_context = context_record(
            self.lexicons, self.demos, self.lexicon_ids, self.demo_ids, selected=False
        )
        empty = build_control_record(
            empty_context,
            lexicon_catalog=self.lexicons,
            demo_catalog=self.demos,
            tokenizer=TOKENIZER,
            config=control_config(),
        )
        self.assertEqual(empty["PL"]["status"], "empty")
        self.assertEqual(empty["PD"]["status"], "empty")
        item = render_control_condition_item(
            empty_context,
            empty,
            lexicon_catalog=self.lexicons,
            demo_catalog=self.demos,
            tokenizer=TOKENIZER,
            condition="PL",
        )
        self.assertEqual(
            item["control_manifest"]["chat_prompt_sha256"],
            empty_context["conditions"]["CL"]["chat_prompt_sha256"],
        )

        impossible_lexicons = copy.deepcopy(self.lexicons)
        for index in range(1, len(self.lexicon_ids)):
            impossible_lexicons[self.lexicon_ids[index]]["rendered_block"] = "X" * (20 + index)
        impossible_context = context_record(
            impossible_lexicons, self.demos, self.lexicon_ids, self.demo_ids
        )
        unavailable = build_control_record(
            impossible_context,
            lexicon_catalog=impossible_lexicons,
            demo_catalog=self.demos,
            tokenizer=TOKENIZER,
            config=control_config(),
        )
        self.assertEqual(unavailable["PL"]["status"], "unavailable")
        self.assertEqual(unavailable["PL"]["failure_reason"], "no-token-match")
        with self.assertRaises(ControlUnavailableError):
            render_control_condition_item(
                impossible_context,
                unavailable,
                lexicon_catalog=impossible_lexicons,
                demo_catalog=self.demos,
                tokenizer=TOKENIZER,
                condition="PL",
            )


class ArtifactLifecycleTests(unittest.TestCase):
    def test_frozen_constructor_is_local_and_remote_code_disabled(self):
        loader = mock.Mock()
        loader.from_pretrained.return_value = TOKENIZER
        with mock.patch.dict(
            sys.modules,
            {"transformers": SimpleNamespace(AutoTokenizer=loader)},
        ):
            result = control_manifest_module._construct_control_tokenizer(
                Path("/registered/tokenizer")
            )
        self.assertIs(result, TOKENIZER)
        loader.from_pretrained.assert_called_once_with(
            "/registered/tokenizer",
            local_files_only=True,
            trust_remote_code=False,
        )

    def test_frozen_engineering_control_ignores_compatibility_injection(self):
        lexicons, demos, lexicon_ids, demo_ids = fixture_catalogs()
        record = context_record(lexicons, demos, lexicon_ids, demo_ids)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tokenizer_root = root / "sources" / "tokenizer"
            tokenizer_root.mkdir(parents=True)
            (tokenizer_root / "tokenizer.json").write_text(
                '{"version":"fixture"}\n', encoding="utf-8"
            )
            context_ref, _ = write_context_artifact(
                root, record, lexicons, demos
            )
            config = control_config(
                tokenizer={
                    "revision": "character/v1",
                    "logical_path": "sources/tokenizer",
                }
            )

            @contextmanager
            def source_lease(contract, *, source_names):
                self.assertEqual(source_names, ("tokenizer",))
                yield SimpleNamespace(tokenizer_path=tokenizer_root)

            with mock.patch.object(
                control_manifest_module,
                "verified_model_source_lease",
                side_effect=source_lease,
            ), mock.patch.object(
                control_manifest_module,
                "_construct_control_tokenizer",
                return_value=TOKENIZER,
            ) as constructor:
                control_ref = root / "refs" / "control_ref.json"
                build_control_artifact(
                    config=config,
                    context_ref=context_ref,
                    write_ref=control_ref,
                    split="dev",
                    target_root=root / "controls",
                    workspace_root=root,
                )
                report = validate_control_ref(
                    control_ref,
                    # Compatibility callers may still supply their registered
                    # tokenizer, but a frozen-tree artifact must ignore it.
                    tokenizer=object(),
                    context_ref=context_ref,
                    workspace_root=root,
                )
            self.assertTrue(report["valid"])
            self.assertEqual(constructor.call_count, 2)

    def test_scientific_control_uses_full_tree_lease_and_rejects_injection(self):
        lexicons, demos, lexicon_ids, demo_ids = fixture_catalogs()
        record = context_record(lexicons, demos, lexicon_ids, demo_ids)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tokenizer_root = root / "sources" / "tokenizer"
            tokenizer_root.mkdir(parents=True)
            (tokenizer_root / "tokenizer.json").write_text(
                '{"version":"fixture"}\n', encoding="utf-8"
            )
            context_ref, context_target = write_context_artifact(
                root, record, lexicons, demos
            )
            inventory = make_context_scientific(
                root, context_ref, context_target, tokenizer_root
            )
            config = control_config(
                tokenizer={
                    "revision": "character/v1",
                    "logical_path": "sources/tokenizer",
                }
            )
            with self.assertRaisesRegex(
                ControlManifestError, "forbids caller-injected tokenizer"
            ):
                build_control_artifact(
                    config=config,
                    context_ref=context_ref,
                    write_ref=root / "refs" / "rejected.json",
                    split="dev",
                    tokenizer=TOKENIZER,
                    target_root=root / "controls",
                    workspace_root=root,
                )

            active = {"value": False}

            class LeaseTokenizer(CharacterTokenizer):
                def apply_chat_template(self, *args, **kwargs):
                    assert active["value"]
                    return super().apply_chat_template(*args, **kwargs)

                def encode(self, *args, **kwargs):
                    assert active["value"]
                    return super().encode(*args, **kwargs)

            lease_tokenizer = LeaseTokenizer()

            @contextmanager
            def source_lease(contract, *, source_names):
                self.assertEqual(source_names, ("tokenizer",))
                self.assertEqual(
                    contract.tokenizer_inventory["inventory_policy"],
                    "all-regular-files/v1",
                )
                self.assertFalse(active["value"])
                active["value"] = True
                try:
                    yield SimpleNamespace(tokenizer_path=tokenizer_root)
                finally:
                    active["value"] = False

            def constructor(path):
                self.assertTrue(active["value"])
                self.assertEqual(path, tokenizer_root)
                return lease_tokenizer

            control_ref = root / "refs" / "control_ref.json"
            with mock.patch.object(
                control_manifest_module,
                "verified_model_source_lease",
                side_effect=source_lease,
            ), mock.patch.object(
                control_manifest_module,
                "_construct_control_tokenizer",
                side_effect=constructor,
            ):
                locator = build_control_artifact(
                    config=config,
                    context_ref=context_ref,
                    write_ref=control_ref,
                    split="dev",
                    target_root=root / "controls",
                    workspace_root=root,
                )
                report = validate_control_ref(
                    control_ref,
                    context_ref=context_ref,
                    workspace_root=root,
                )
            self.assertTrue(report["valid"])
            self.assertFalse(active["value"])
            target = Path(locator["target_path"])
            meta = json.loads(
                (target / "control_manifest.meta.json").read_text(encoding="utf-8")
            )
            provenance = json.loads(
                (target / "provenance.json").read_text(encoding="utf-8")
            )
            identity = meta["tokenizer_source_identity"]
            self.assertEqual(identity["inventory"], inventory)
            self.assertEqual(
                identity,
                meta["id_inputs"]["tokenizer_source_identity"],
            )
            self.assertEqual(identity, provenance["tokenizer_source_identity"])
            self.assertEqual(
                identity["constructor_policy"],
                {
                    "backend": "transformers-auto-tokenizer/v1",
                    "local_files_only": True,
                    "trust_remote_code": False,
                },
            )
            with self.assertRaisesRegex(
                ControlManifestError, "forbids caller-injected tokenizer"
            ):
                validate_control_ref(
                    control_ref,
                    tokenizer=TOKENIZER,
                    context_ref=context_ref,
                    workspace_root=root,
                )

    def test_scientific_context_must_prove_fit_only_partition_lineage(self):
        lexicons, demos, lexicon_ids, demo_ids = fixture_catalogs()
        record = context_record(lexicons, demos, lexicon_ids, demo_ids)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            context_ref, target = write_context_artifact(
                root, record, lexicons, demos
            )
            meta_path = target / "context_manifest.dev.meta.json"
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            meta["scientific_eligible"] = True
            meta["sources"]["demo_pool"]["partition"] = "fit"
            meta["sources"]["lexicon_pool"]["partition"] = "fit"
            write_json(meta_path, meta)
            write_json(
                target / "train_partition_ref.json",
                {
                    "schema_version": "stage1-dependency-ref/v1",
                    "artifact_kind": "train-partition",
                    "artifact_id": "tpart-" + "a" * 64,
                    "payload_manifest_sha256": "b" * 64,
                    "logical_repo_path": "train_partitions/tpart-" + "a" * 64,
                },
            )
            write_json(target / "payload_manifest.json", payload_manifest(target))
            locator = json.loads(context_ref.read_text(encoding="utf-8"))
            locator["payload_manifest_sha256"] = file_sha(
                target / "payload_manifest.json"
            )
            write_json(context_ref, locator)

            with self.assertRaisesRegex(
                ControlManifestError, "frozen retrieval provenance"
            ):
                build_control_artifact(
                    config=control_config(),
                    context_ref=context_ref,
                    write_ref=root / "refs/control_ref.json",
                    split="dev",
                    tokenizer=TOKENIZER,
                    tokenizer_revision="character/v1",
                    target_root=root / "controls",
                )

    def test_content_addressed_artifact_rebuild_schema_and_tamper_validation(self):
        lexicons, demos, lexicon_ids, demo_ids = fixture_catalogs()
        record = context_record(lexicons, demos, lexicon_ids, demo_ids)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            context_ref, _ = write_context_artifact(root, record, lexicons, demos)
            control_ref = root / "refs" / "control_ref.json"
            first = build_control_artifact(
                config=control_config(),
                context_ref=context_ref,
                write_ref=control_ref,
                split="dev",
                tokenizer=TOKENIZER,
                tokenizer_revision="character/v1",
                target_root=root / "controls",
            )
            second = build_control_artifact(
                config=control_config(),
                context_ref=context_ref,
                write_ref=control_ref,
                split="dev",
                tokenizer=TOKENIZER,
                tokenizer_revision="character/v1",
                target_root=root / "controls",
            )
            self.assertEqual(first, second)
            report = validate_control_ref(
                control_ref, tokenizer=TOKENIZER, context_ref=context_ref
            )
            self.assertTrue(report["valid"])
            target = Path(first["target_path"])
            embedded = json.loads((target / "context_ref.json").read_text(encoding="utf-8"))
            self.assertEqual(embedded["schema_version"], "stage1-dependency-ref/v1")
            self.assertNotIn("target_path", embedded)

            rows = [
                json.loads(line)
                for line in (target / "control_manifest.dev.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            schema = json.loads(
                (Path(__file__).resolve().parents[2] / "schemas" / "stage1_control_manifest_v1.schema.json").read_text(
                    encoding="utf-8"
                )
            )
            jsonschema.validate(rows[0], schema)

            rows[0]["PL"]["replacement_ids"] = []
            write_jsonl(target / "control_manifest.dev.jsonl", rows)
            with self.assertRaises(ControlManifestError):
                validate_control_ref(control_ref, tokenizer=TOKENIZER, context_ref=context_ref)


if __name__ == "__main__":
    unittest.main()
