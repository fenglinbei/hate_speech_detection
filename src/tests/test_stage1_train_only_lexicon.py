import hashlib
import json
import shutil
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch

from build_lex import llm_lexicon_builder
from build_lex.llm_lexicon_builder import CandidateStats, CorpusCandidateResult
from build_lex.stage1_preflight import preflight_formal_lexicon
from build_lex.train_only import (
    TrainOnlyLexiconError,
    _validate_stage_response,
    _validated_provider_usage,
    build_train_only_lexicon,
    resolve_train_input,
    validate_lexicon_ref,
)
from build_lex.llm_lexicon_builder import record_label_and_categories
from data.train_partition import build_train_partition, load_train_partition
from tests.stage1_semantic_fixtures import make_semantic_data_artifact


def canonical_bytes(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_bytes(value) + b"\n")


def normalized_record(record_id, content):
    return {
        "id": str(record_id),
        "content": content,
        "quadruples": [
            {
                "target": None,
                "argument": content,
                "targeted_group": ["Racism"],
                "hateful": "hate",
            }
        ],
    }


def make_fake_builder(support_id="1"):
    def fake_builder(_dataset, config, **_kwargs):
        input_paths = config["data_paths"]["input_paths"]
        records = json.loads(Path(input_paths[0]).read_text(encoding="utf-8"))
        output_dir = Path(config["data_paths"]["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        term = {
            "term": "火星人",
            "definition": "指来自火星的虚构人物，也可在语境中比喻与常人不同的人。",
            "usage_notes": "具体褒贬由所在句子决定。",
            "ambiguity_notes": "可能是科幻设定，也可能是比喻用法。",
            "variants": [],
            "metadata": {
                "support": {
                    "sample_ids": [support_id],
                    "total_count": 1,
                }
            },
        }
        write_json(
            output_dir / "lexicon.json",
            {
                "source": "legacy",
                "dataset": "full",
                "input_paths": input_paths,
                "language": "zh",
                "resource_role": "terminology-understanding-library/v1",
                "terms": [term],
            },
        )
        for filename, row in (
            ("candidates.jsonl", {"term": "火星人", "support_sample_ids": [support_id], "contexts": records[:1]}),
            ("web_evidence.jsonl", {"term": "火星人", "query": "火星人 歧视", "url": "https://example.test"}),
            ("llm_judgements.jsonl", {"term": "火星人", "raw": {"include": True}}),
            ("rejected.jsonl", {}),
        ):
            (output_dir / filename).write_text(
                json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        return {"terms": [term], "total_records": len(records)}

    return fake_builder


def make_data_ref(root):
    return make_semantic_data_artifact(root)[0]


def make_partition_ref(root, data_ref):
    ref = root / "train_partition_ref.json"
    if not ref.is_file():
        build_train_partition(
            data_ref=data_ref,
            write_ref=ref,
            workspace_root=root,
            target_root=root / "train_partitions",
        )
    return ref


def formal_config():
    config_path = Path(__file__).resolve().parents[2] / "config" / "stage1" / "lexicon_train_only.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["web_settings"]["cache_enabled"] = False
    config["web_settings"]["cache_path"] = None
    config["web_settings"]["max_results"] = 3
    config["web_settings"]["timeout"] = 30
    config["llm_settings"]["retries"] = 2
    config["llm_settings"]["retry_sleep"] = 1.0
    return config


def make_formal_preflight(root, data_ref, *, config=None):
    config = formal_config() if config is None else config
    config_path = root / "formal_config.json"
    env_path = root / ".env"
    write_json(config_path, config)
    env_path.write_text(
        "TAVILY_API_KEY=tavily-fixture\nDEEPSEEK_API_KEY=deepseek-fixture\n",
        encoding="utf-8",
    )
    partition_ref = make_partition_ref(root, data_ref)
    result = preflight_formal_lexicon(
        dataset="full",
        config_path=config_path,
        data_ref=data_ref,
        train_partition_ref=partition_ref,
        env_file=env_path,
        repository_root=root,
        environ={},
    )
    if not result.passed:
        raise AssertionError(result.to_public_dict())
    return config, result


class FixtureWebSearcher:
    def __init__(
        self,
        formal_checkpoint=None,
        counters=None,
        fail_after_reserve_call=None,
    ):
        self.formal_checkpoint = formal_checkpoint
        self.counters = counters
        self.fail_after_reserve_call = fail_after_reserve_call

    def search(
        self,
        query,
        *,
        checkpoint_rank=None,
        checkpoint_slot=None,
        checkpoint_term=None,
    ):
        if self.counters is not None:
            self.counters["web_calls"] = self.counters.get("web_calls", 0) + 1
        results = [
            {
                "id": hashlib.sha256(query.encode("utf-8")).hexdigest()[:16],
                "query": query,
                "title": "fixture evidence",
                "snippet": "fixture snippet",
                "url": "https://example.test/evidence",
                "source": "fixture",
            }
        ]
        if self.formal_checkpoint is not None:
            slot = f"query_{checkpoint_slot}"
            request = {
                "method": "POST",
                "url": "https://api.tavily.com/search",
                "params": {"query": query},
                "timeout": 30,
            }
            cached = self.formal_checkpoint.get_slot_success(
                "tavily", checkpoint_rank, slot, request_payload=request
            )
            if cached is not None:
                return cached.response["results"]
            reservation = self.formal_checkpoint.reserve_attempt(
                "tavily", checkpoint_rank, slot, request
            )
            if (
                self.counters is not None
                and self.fail_after_reserve_call is not None
                and self.counters["web_calls"] == self.fail_after_reserve_call
            ):
                raise RuntimeError("simulated process interruption after reservation")
            self.formal_checkpoint.finish_attempt(
                reservation,
                status="success",
                response={"results": results},
                capture={
                    "term": checkpoint_term,
                    "query": query,
                    "results": results,
                    "error": None,
                },
                detail={
                    "error_type": None,
                    "http_status": 200,
                    "retryable": False,
                },
            )
        return results

    def close(self):
        return None


class FixtureJudgementClient:
    def __init__(
        self,
        recorder,
        secret_in_capture=None,
        formal_checkpoint=None,
        counters=None,
    ):
        self.recorder = recorder
        self.secret_in_capture = secret_in_capture
        self.formal_checkpoint = formal_checkpoint
        self.counters = counters

    def complete_json(self, stage, payload, *, checkpoint_rank=None):
        if self.counters is not None:
            self.counters["llm_calls"] = self.counters.get("llm_calls", 0) + 1
        terminology_mode = (
            payload.get("decision_contract", {}).get("resource_role")
            == "terminology-understanding-library/v1"
        )
        if stage == "context_judge":
            parsed = {
                "supported": True,
                "confidence": 0.99,
                "reason": self.secret_in_capture or "fixture",
            }
            if not terminology_mode:
                parsed.update(
                    {"category": "Racism", "categories": ["Racism"]}
                )
        elif stage == "web_evidence_judge":
            evidence_ids = list(
                payload.get("decision_contract", {}).get(
                    "allowed_web_evidence_ids", []
                )
            )[:1]
            parsed = {
                "supported": bool(evidence_ids),
                "confidence": 0.99,
                "reason": (
                    "fixture"
                    if evidence_ids
                    else "no supplied Web evidence in disabled-Web mode"
                ),
                "evidence_ids": evidence_ids,
            }
        else:
            evidence_ids = list(
                payload.get("decision_contract", {}).get(
                    "allowed_web_evidence_ids", []
                )
            )[:1]
            if terminology_mode:
                parsed = {
                    "include": True,
                    "definition": "指来自火星的虚构人物，也可比喻与常人不同的人。",
                    "usage_notes": "具体褒贬由所在句子决定。",
                    "ambiguity_notes": "可能是科幻设定，也可能是比喻用法。",
                    "variants": [],
                    "confidence": 0.99,
                    "reason": "fixture",
                    "evidence_ids": evidence_ids,
                }
            else:
                parsed = {
                    "include": True,
                    "category": "Racism",
                    "categories": ["Racism"],
                    "definition": "对火星人的歧视性称呼。",
                    "nonhateful_meaning": "",
                    "variants": [],
                    "confidence": 0.99,
                    "reason": "fixture",
                    "evidence_ids": evidence_ids,
                }
        request_payload = {
            "model": "deepseek-v4-flash",
            "messages": [{"role": "user", "content": "fixture"}],
            "max_tokens": 1024,
            "stream": False,
            "response_format": {"type": "json_object"},
            "temperature": 0,
            "thinking": {"type": "disabled"},
        }
        raw_response = {
            "model": "deepseek-v4-flash-0731",
            "usage": {
                "prompt_tokens": 100,
                "prompt_cache_hit_tokens": 20,
                "prompt_cache_miss_tokens": 80,
                "completion_tokens": 30,
                "total_tokens": 130,
            },
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"content": json.dumps(parsed, ensure_ascii=False)},
                }
            ],
        }
        if self.formal_checkpoint is not None:
            cached = self.formal_checkpoint.get_slot_success(
                "deepseek",
                checkpoint_rank,
                stage,
                request_payload=request_payload,
            )
            if cached is not None:
                return cached.response["parsed_response"]
            reservation = self.formal_checkpoint.reserve_attempt(
                "deepseek", checkpoint_rank, stage, request_payload
            )
            self.formal_checkpoint.finish_attempt(
                reservation,
                status="success",
                response={"parsed_response": parsed},
                capture={
                    "stage": stage,
                    "term": payload["candidate"]["term"],
                    "attempt": reservation.attempt,
                    "request_payload": request_payload,
                    "raw_response": raw_response,
                    "parsed_response": parsed,
                    "error": None,
                },
                detail={
                    "error_type": None,
                    "http_status": 200,
                    "retryable": False,
                },
            )
        self.recorder.record_llm_call(
            stage=stage,
            payload=payload,
            request_payload=request_payload,
            attempt=1,
            raw_response=raw_response,
            parsed_response=parsed,
        )
        return parsed


def formal_build(
    root,
    *,
    data_ref=None,
    config=None,
    preflight=None,
    candidates=None,
    candidate_support_id="1",
    secret_in_capture=None,
    provider_counters=None,
    fail_web_after_reserve_call=None,
    **overrides,
):
    data_ref = make_data_ref(root / "data") if data_ref is None else data_ref
    if preflight is None:
        config, preflight = make_formal_preflight(root, data_ref, config=config)
    support_content = "train text"
    if str(candidate_support_id) != "1":
        partition = load_train_partition(
            root / "train_partition_ref.json", workspace_root=root
        )
        support_content = {
            record["id"]: record["content"]
            for record in partition.train_records
        }[str(candidate_support_id)]
    candidate = CandidateStats(
        term="火星人",
        dataset="full",
        language="zh",
        total_count=1,
        source_counts=Counter({"content": 1}),
        track_counts=Counter({"contrastive_phrase": 1}),
        support_sample_ids=[str(candidate_support_id)],
        sample_contexts=[
            {
                "id": str(candidate_support_id),
                "source": "content",
                "content": support_content,
            }
        ],
    )
    corpus = CorpusCandidateResult(
        candidates=[candidate] if candidates is None else candidates,
        total_records=1,
        hate_records=0,
        non_hate_records=0,
        input_paths=[],
    )

    def judgement_factory(
        _settings, debug_recorder=None, *, formal_checkpoint=None
    ):
        return FixtureJudgementClient(
            debug_recorder,
            secret_in_capture=secret_in_capture,
            formal_checkpoint=formal_checkpoint,
            counters=provider_counters,
        )

    kwargs = {
        "builder": llm_lexicon_builder.build_lexicon,
        "data_ref": data_ref,
        "train_partition_ref": root / "train_partition_ref.json",
        "formal": True,
        "target_root": root / "lexicons",
        "write_ref": root / "lexicon_ref.json",
        "build_authorization": preflight.build_authorization,
        "workspace_root": root,
    }
    kwargs.update(overrides)
    fixture_searcher = (
        llm_lexicon_builder.DisabledWebSearcher()
        if config["web_settings"]["backend"] == "disabled"
        else FixtureWebSearcher(
            counters=provider_counters,
            fail_after_reserve_call=fail_web_after_reserve_call,
        )
    )

    def web_factory(_settings, *, formal_checkpoint=None):
        if isinstance(fixture_searcher, FixtureWebSearcher):
            fixture_searcher.formal_checkpoint = formal_checkpoint
        return fixture_searcher

    class FixtureProgress:
        def set_postfix(self, **_kwargs):
            return None

        def update(self, _amount):
            return None

        def close(self):
            return None

    def fixture_tqdm(value=None, **_kwargs):
        return value if value is not None else FixtureProgress()

    with (
        patch.object(llm_lexicon_builder, "build_candidates", return_value=corpus),
        patch.object(
            llm_lexicon_builder,
            "select_candidates",
            side_effect=lambda value, _settings: value.candidates,
        ),
        patch.object(
            llm_lexicon_builder,
            "create_judgement_client",
            side_effect=judgement_factory,
        ),
        patch.object(
            llm_lexicon_builder,
            "create_web_searcher",
            side_effect=web_factory,
        ),
        patch.object(llm_lexicon_builder, "tqdm", side_effect=fixture_tqdm),
    ):
        locator = build_train_only_lexicon("full", config, **kwargs)
    return locator, config, preflight


def rewrite_payload_manifest(target):
    payload = {
        "schema_version": "stage1-payload-manifest/v1",
        "files": [
            {"path": path.name, "size": path.stat().st_size, "sha256": sha256_file(path)}
            for path in sorted(target.iterdir())
            if path.is_file() and path.name != "payload_manifest.json"
        ],
    }
    write_json(target / "payload_manifest.json", payload)


class Stage1TrainOnlyLexiconTest(unittest.TestCase):
    def test_provider_model_and_usage_are_strictly_audited(self):
        raw = {
            "model": "deepseek-v4-flash-0731",
            "usage": {
                "prompt_tokens": 10,
                "prompt_cache_hit_tokens": 3,
                "prompt_cache_miss_tokens": 7,
                "completion_tokens": 4,
                "total_tokens": 14,
            },
        }
        model, usage = _validated_provider_usage(
            raw, requested_model="deepseek-v4-flash"
        )
        self.assertEqual(model, "deepseek-v4-flash-0731")
        self.assertEqual(usage["total_tokens"], 14)

        wrong_model = json.loads(json.dumps(raw))
        wrong_model["model"] = "deepseek-v4-flash-pro"
        with self.assertRaisesRegex(TrainOnlyLexiconError, "model family"):
            _validated_provider_usage(
                wrong_model, requested_model="deepseek-v4-flash"
            )

        inconsistent = json.loads(json.dumps(raw))
        inconsistent["usage"]["total_tokens"] = 15
        with self.assertRaisesRegex(TrainOnlyLexiconError, "inconsistent"):
            _validated_provider_usage(
                inconsistent, requested_model="deepseek-v4-flash"
            )

    def test_stage_validator_rejects_noncanonical_and_non_others_negative_categories(self):
        valid_negative = {
            "supported": False,
            "category": "others",
            "categories": ["others"],
            "confidence": 0.9,
            "reason": "candidate itself is not hateful",
        }
        _validate_stage_response("context_judge", valid_negative)

        for category, categories in (
            ("none", ["none"]),
            ("LGBTQ", ["LGBTQ"]),
            ("others", ["others", "LGBTQ"]),
        ):
            invalid = dict(valid_negative, category=category, categories=categories)
            with self.assertRaises(TrainOnlyLexiconError):
                _validate_stage_response("context_judge", invalid)

    def test_stage_validator_confines_citations_to_web_evidence_namespace(self):
        final = {
            "include": True,
            "category": "LGBTQ",
            "categories": ["LGBTQ"],
            "definition": "fixture",
            "nonhateful_meaning": "",
            "variants": [],
            "confidence": 0.9,
            "reason": "fixture",
            "evidence_ids": ["web-1"],
        }
        _validate_stage_response(
            "final_lexicon_judge",
            final,
            allowed_evidence_ids=["web-1", "web-2"],
        )

        with self.assertRaisesRegex(TrainOnlyLexiconError, "Web evidence namespace"):
            _validate_stage_response(
                "final_lexicon_judge",
                dict(final, evidence_ids=["train-sample-17"]),
                allowed_evidence_ids=["web-1", "web-2"],
            )
        with self.assertRaisesRegex(TrainOnlyLexiconError, "invalid evidence_ids"):
            _validate_stage_response(
                "final_lexicon_judge",
                dict(final, evidence_ids=["web-1", "web-1"]),
                allowed_evidence_ids=["web-1"],
            )

    def test_terminology_stage_validator_rejects_nested_task_fields(self):
        response = {
            "supported": True,
            "confidence": 0.9,
            "reason": "fixture",
            "metadata": {"primary_category": "LGBTQ"},
        }
        with self.assertRaisesRegex(
            TrainOnlyLexiconError, "metadata.primary_category"
        ):
            _validate_stage_response(
                "context_judge",
                response,
                resource_role="terminology-understanding-library/v1",
            )

    def test_negative_final_response_uses_others_category(self):
        negative = {
            "include": False,
            "category": "others",
            "categories": ["others"],
            "definition": "",
            "nonhateful_meaning": "neutral identity label",
            "variants": [],
            "confidence": 0.95,
            "reason": "context toxicity is not attributable to the candidate",
            "evidence_ids": [],
        }
        _validate_stage_response(
            "final_lexicon_judge",
            negative,
            allowed_evidence_ids=["web-1"],
        )

    def test_explicit_zero_retry_policy_is_not_replaced_by_legacy_defaults(self):
        client = llm_lexicon_builder.OpenAICompatibleJudgementClient(
            {
                "provider": "deepseek",
                "api_base": "https://api.deepseek.com",
                "api_key": "fixture",
                "model": "fixture-model",
                "retries": 0,
                "retry_sleep": 0,
                "thinking": {"type": "disabled"},
            }
        )
        self.assertEqual(client.retries, 0)
        self.assertEqual(client.retry_sleep, 0)

    def test_normalized_group_arrays_do_not_infer_hateful(self):
        canonical = normalized_record("1", "neutral")
        canonical["quadruples"][0]["hateful"] = "non-hate"

        is_hate, categories = record_label_and_categories("full", canonical)
        legacy_is_hate, _ = record_label_and_categories(
            "full",
            {
                "quadruples": [
                    {
                        "targeted_group": "Racism",
                        "hateful": "non-hate",
                    }
                ]
            },
        )

        self.assertFalse(is_hate)
        self.assertEqual(categories, ["Racism"])
        self.assertTrue(legacy_is_hate)

    def test_stage1_config_has_no_legacy_split_paths(self):
        config_path = Path(__file__).resolve().parents[2] / "config" / "stage1" / "lexicon_train_only.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))

        self.assertEqual(
            config["schema_version"],
            "stage1-train-only-terminology-library-config/v1",
        )
        self.assertEqual(
            config["resource_role"], "terminology-understanding-library/v1"
        )
        self.assertNotIn("data_paths", config)
        self.assertTrue(config["runtime_settings"]["resume"])
        self.assertTrue(config["runtime_settings"]["resume_require_config_match"])
        self.assertEqual(
            config["runtime_settings"]["formal_checkpoint_policy"],
            "provider-slot-checkpoint/v1",
        )
        self.assertEqual(
            config["runtime_settings"]["ambiguous_attempt_policy"],
            "count-and-retry-within-budget/v1",
        )
        self.assertEqual(config["runtime_settings"]["max_llm_http_attempts"], 9000)
        self.assertEqual(
            config["web_settings"]["transport_retry_policy"],
            {
                "id": "tavily-transient/v1",
                "retries": 2,
                "base_sleep_seconds": 1.0,
            },
        )
        self.assertEqual(
            config["web_settings"]["physical_attempt_budget"],
            {
                "scope_id": "stage1-p0-wp3-formal-full-tavily-key2/v1",
                "cap": 3100,
            },
        )

    def test_formal_mode_requires_frozen_data_ref(self):
        with self.assertRaises(TrainOnlyLexiconError):
            resolve_train_input(train_records=[normalized_record("1", "x")], formal=True)

    def test_direct_records_engineering_build_is_stable_and_secret_free(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target_root = root / "lexicons"
            ref_path = root / "lexicon_ref.json"
            config = {
                "candidate_settings": {"min_count": 1},
                "llm_settings": {"backend": "fake", "model": "judge-v1", "api_key": "must-not-leak"},
                "web_settings": {"backend": "fake"},
            }
            kwargs = {
                "dataset": "full",
                "config": config,
                "builder": make_fake_builder(),
                "train_records": [normalized_record("1", "train text")],
                "formal": False,
                "target_root": target_root,
                "write_ref": ref_path,
            }

            first = build_train_only_lexicon(**kwargs)
            second = build_train_only_lexicon(**kwargs)
            manifest = validate_lexicon_ref(ref_path)
            target = Path(first["target_path"])
            lexicon = json.loads((target / "lexicon.json").read_text(encoding="utf-8"))
            provenance_text = (target / "provenance.json").read_text(encoding="utf-8")

            self.assertEqual(first["artifact_id"], second["artifact_id"])
            self.assertTrue(manifest["train_only_verified"])
            self.assertEqual(manifest["source_split"], "train")
            self.assertRegex(
                lexicon["terms"][0]["lexicon_id"], r"^lex:v2:[0-9a-f]{64}$"
            )
            self.assertNotIn("category", lexicon["terms"][0])
            self.assertNotIn("must-not-leak", provenance_text)
            self.assertNotIn("input_paths", (target / "lexicon.json").read_text(encoding="utf-8"))

    def test_formal_data_ref_uses_only_train_and_validates_lineage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_ref = make_data_ref(root / "data")
            locator, built_config, built_preflight = formal_build(root, data_ref=data_ref)
            manifest = validate_lexicon_ref(
                root / "lexicon_ref.json", workspace_root=root
            )
            provenance = json.loads(
                (Path(locator["target_path"]) / "provenance.json").read_text(encoding="utf-8")
            )

            self.assertEqual(manifest["data_build_id"], json.loads(data_ref.read_text())["artifact_id"])
            self.assertEqual(
                manifest["train_record_count"],
                len(
                    load_train_partition(
                        root / "train_partition_ref.json",
                        workspace_root=root,
                    ).fit_ids
                ),
            )
            self.assertEqual(manifest["source_partition"], "fit")
            self.assertEqual(provenance["calibration_contribution_count"], 0)
            self.assertEqual(provenance["data_dependency"]["split"], "train")
            self.assertNotIn("dev_ids", provenance)
            self.assertNotIn("test_ids", provenance)
            self.assertEqual(
                list((root / "lexicons").glob(".formal-billing-ledger.*.jsonl")),
                [],
            )
            self.assertEqual(manifest["evidence_audit"]["candidate_count"], 1)
            self.assertEqual(manifest["evidence_audit"]["logical_web_request_count"], 3)
            self.assertEqual(manifest["evidence_audit"]["physical_web_attempt_count"], 3)
            self.assertEqual(manifest["evidence_audit"]["web_retry_attempt_count"], 0)
            self.assertEqual(manifest["evidence_audit"]["ambiguous_web_attempt_count"], 0)
            self.assertEqual(
                manifest["evidence_audit"]["authorized_web_physical_attempt_cap"],
                3100,
            )
            self.assertEqual(manifest["evidence_audit"]["successful_llm_capture_count"], 3)
            self.assertEqual(
                manifest["evidence_audit"]["provider_response_models"],
                {"deepseek-v4-flash-0731": 3},
            )
            self.assertEqual(
                manifest["evidence_audit"]["provider_usage_totals"],
                {
                    "response_count": 3,
                    "prompt_tokens": 300,
                    "prompt_cache_hit_tokens": 60,
                    "prompt_cache_miss_tokens": 240,
                    "completion_tokens": 90,
                    "total_tokens": 390,
                },
            )
            self.assertEqual(manifest["formal_authorization"]["dataset"], "full")
            self.assertEqual(
                set(manifest["formal_authorization"]["protocol_code_sha256s"]),
                {
                    "src/build_lex/formal_checkpoint.py",
                    "src/build_lex/stage1_preflight.py",
                    "src/build_lex/train_only.py",
                    "src/build_lex/web_search.py",
                },
            )
            target = Path(locator["target_path"])
            embedded_data = json.loads(
                (target / "data_ref.json").read_text(encoding="utf-8")
            )
            embedded_partition = json.loads(
                (target / "train_partition_ref.json").read_text(encoding="utf-8")
            )
            portable_fields = {
                "schema_version",
                "artifact_kind",
                "artifact_id",
                "payload_manifest_sha256",
                "logical_repo_path",
            }
            self.assertEqual(set(embedded_data), portable_fields)
            self.assertEqual(set(embedded_partition), portable_fields)
            self.assertNotIn(
                "data_ref_locator_sha256", manifest["formal_authorization"]
            )
            self.assertNotIn(
                "train_partition_ref_locator_sha256",
                manifest["formal_authorization"],
            )
            for filename in (
                "candidates.jsonl",
                "web_evidence.jsonl",
                "llm_judgements.jsonl",
                "rejected.jsonl",
                "debug_llm_calls.jsonl",
                "debug_search_calls.jsonl",
                "debug_tavily_attempts.jsonl",
            ):
                self.assertTrue((target / filename).is_file())
            tavily_attempts = [
                json.loads(line)
                for line in (target / "debug_tavily_attempts.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertEqual(len(tavily_attempts), 3)
            safe_attempt_fields = {
                "provider",
                "budget_scope_id",
                "rank",
                "slot",
                "attempt",
                "status",
                "http_status",
                "error_type",
                "retryable",
                "result_count",
                "request_dispatched",
            }
            for slot, attempt in enumerate(tavily_attempts, start=1):
                self.assertEqual(set(attempt), safe_attempt_fields)
                self.assertEqual(attempt["provider"], "tavily")
                self.assertEqual(
                    attempt["budget_scope_id"],
                    "stage1-p0-wp3-formal-full-tavily-key2/v1",
                )
                self.assertEqual(attempt["rank"], 1)
                self.assertEqual(attempt["slot"], slot)
                self.assertEqual(attempt["attempt"], 1)
                self.assertEqual(attempt["status"], "success")
                self.assertIsNone(attempt["error_type"])
                self.assertIs(attempt["retryable"], False)
                self.assertEqual(attempt["result_count"], 1)
                self.assertIs(attempt["request_dispatched"], True)
                self.assertTrue(
                    {"term", "query", "content", "request_payload"}.isdisjoint(
                        attempt
                    )
                )
            published_text = "\n".join(
                path.read_text(encoding="utf-8")
                for path in target.iterdir()
                if path.is_file()
            )
            self.assertNotIn("tavily-fixture", published_text)
            self.assertNotIn("deepseek-fixture", published_text)
            self.assertNotIn('"target_path"', published_text)
            second_target = root / "second-build"
            with self.assertRaisesRegex(TrainOnlyLexiconError, "already been consumed"):
                build_train_only_lexicon(
                    "full",
                    built_config,
                    builder=llm_lexicon_builder.build_lexicon,
                    data_ref=data_ref,
                    train_partition_ref=root / "train_partition_ref.json",
                    formal=True,
                    target_root=second_target,
                    build_authorization=built_preflight.build_authorization,
                    workspace_root=root,
                )
            self.assertFalse(second_target.exists())

    def test_formal_restart_reuses_committed_candidate_and_partial_provider_slots(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_ref = make_data_ref(root / "data")

            def candidate(term):
                return CandidateStats(
                    term=term,
                    dataset="full",
                    language="zh",
                    total_count=1,
                    source_counts=Counter({"content": 1}),
                    track_counts=Counter({"contrastive_phrase": 1}),
                    support_sample_ids=["1"],
                    sample_contexts=[
                        {
                            "id": "1",
                            "source": "content",
                            "content": "train text",
                        }
                    ],
                )

            candidates = [candidate("火星人甲"), candidate("火星人乙")]
            counters = {}
            with self.assertRaisesRegex(RuntimeError, "simulated process interruption"):
                formal_build(
                    root,
                    data_ref=data_ref,
                    candidates=candidates,
                    provider_counters=counters,
                    fail_web_after_reserve_call=4,
                )
            self.assertEqual(counters, {"web_calls": 4, "llm_calls": 3})
            self.assertFalse((root / "lexicon_ref.json").exists())

            locator, _config, _preflight = formal_build(
                root,
                data_ref=data_ref,
                candidates=candidates,
                provider_counters=counters,
                fail_web_after_reserve_call=4,
            )
            self.assertEqual(counters, {"web_calls": 7, "llm_calls": 6})
            target = Path(locator["target_path"])
            manifest = validate_lexicon_ref(
                root / "lexicon_ref.json", workspace_root=root
            )
            self.assertEqual(manifest["evidence_audit"]["candidate_count"], 2)
            self.assertEqual(
                manifest["evidence_audit"]["physical_web_attempt_count"], 7
            )
            self.assertEqual(
                manifest["evidence_audit"]["web_retry_attempt_count"], 1
            )
            self.assertEqual(
                manifest["evidence_audit"]["ambiguous_web_attempt_count"], 1
            )
            self.assertEqual(
                len((target / "debug_tavily_attempts.jsonl").read_text().splitlines()),
                7,
            )

    def test_formal_provider_scope_is_unique_across_target_roots(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_ref = make_data_ref(root / "data")
            counters = {}
            formal_build(
                root,
                data_ref=data_ref,
                provider_counters=counters,
                target_root=root / "lexicons-a",
                write_ref=root / "lexicon-a-ref.json",
            )
            self.assertEqual(counters, {"web_calls": 3, "llm_calls": 3})

            formal_build(
                root,
                data_ref=data_ref,
                provider_counters=counters,
                target_root=root / "lexicons-b",
                write_ref=root / "lexicon-b-ref.json",
            )
            self.assertEqual(counters, {"web_calls": 3, "llm_calls": 3})
            checkpoint_roots = [
                path
                for path in (
                    root
                    / "exps"
                    / "causal_context"
                    / "stage1_p0"
                    / "lexicons"
                    / ".formal-checkpoints"
                ).iterdir()
                if path.is_dir()
            ]
            self.assertEqual(len(checkpoint_roots), 1)

    def test_active_scope_refuses_to_recreate_a_lost_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_ref = make_data_ref(root / "data")
            formal_build(root, data_ref=data_ref)
            checkpoint_parent = (
                root
                / "exps"
                / "causal_context"
                / "stage1_p0"
                / "lexicons"
                / ".formal-checkpoints"
            )
            checkpoint_roots = [
                path for path in checkpoint_parent.iterdir() if path.is_dir()
            ]
            self.assertEqual(len(checkpoint_roots), 1)
            shutil.rmtree(checkpoint_roots[0])

            with self.assertRaisesRegex(
                TrainOnlyLexiconError, "lost its durable checkpoint"
            ):
                formal_build(
                    root,
                    data_ref=data_ref,
                    target_root=root / "second-target",
                    write_ref=root / "second-ref.json",
                )
            self.assertFalse((root / "second-ref.json").exists())

    def test_dev_or_test_support_id_is_a_hard_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaises(TrainOnlyLexiconError):
                build_train_only_lexicon(
                    "full",
                    {},
                    builder=make_fake_builder("dev-9"),
                    train_records=[normalized_record("train-1", "train")],
                    formal=False,
                    target_root=root / "lexicons",
                    forbidden_dev_test_ids=["dev-9"],
                )

    def test_formal_authorization_is_required_and_fake_builder_is_rejected_before_mkdir(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target_without_auth = root / "without-auth"
            with self.assertRaisesRegex(TrainOnlyLexiconError, "same-process preflight"):
                build_train_only_lexicon(
                    "full",
                    formal_config(),
                    builder=make_fake_builder(),
                    data_ref=root / "missing-ref.json",
                    train_partition_ref=root / "missing-partition-ref.json",
                    formal=True,
                    target_root=target_without_auth,
                )
            self.assertFalse(target_without_auth.exists())

            data_ref = make_data_ref(root / "data")
            config, preflight = make_formal_preflight(root, data_ref)
            target_with_fake = root / "with-fake"
            with self.assertRaisesRegex(TrainOnlyLexiconError, "frozen llm_lexicon_builder"):
                build_train_only_lexicon(
                    "full",
                    config,
                    builder=make_fake_builder(),
                    data_ref=data_ref,
                    train_partition_ref=root / "train_partition_ref.json",
                    formal=True,
                    target_root=target_with_fake,
                    build_authorization=preflight.build_authorization,
                    workspace_root=root,
                )
            self.assertFalse(target_with_fake.exists())

    def test_formal_authorization_blocks_config_data_and_client_mixing_before_mkdir(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_ref = make_data_ref(root / "data")
            config, preflight = make_formal_preflight(root, data_ref)

            drifted = json.loads(json.dumps(config))
            drifted["llm_settings"]["max_tokens"] += 1
            drift_target = root / "drift-target"
            with self.assertRaisesRegex(TrainOnlyLexiconError, "config drifted"):
                build_train_only_lexicon(
                    "full",
                    drifted,
                    builder=llm_lexicon_builder.build_lexicon,
                    data_ref=data_ref,
                    train_partition_ref=root / "train_partition_ref.json",
                    formal=True,
                    target_root=drift_target,
                    build_authorization=preflight.build_authorization,
                    workspace_root=root,
                )
            self.assertFalse(drift_target.exists())

            mixed_target = root / "mixed-target"
            with self.assertRaisesRegex(TrainOnlyLexiconError, "data_ref path"):
                build_train_only_lexicon(
                    "full",
                    config,
                    builder=llm_lexicon_builder.build_lexicon,
                    data_ref=root / "another_ref.json",
                    train_partition_ref=root / "train_partition_ref.json",
                    formal=True,
                    target_root=mixed_target,
                    build_authorization=preflight.build_authorization,
                    workspace_root=root,
                )
            self.assertFalse(mixed_target.exists())

            injected_target = root / "injected-target"
            with self.assertRaisesRegex(TrainOnlyLexiconError, "forbid injected"):
                build_train_only_lexicon(
                    "full",
                    config,
                    builder=llm_lexicon_builder.build_lexicon,
                    data_ref=data_ref,
                    train_partition_ref=root / "train_partition_ref.json",
                    formal=True,
                    target_root=injected_target,
                    judge_client=object(),
                    build_authorization=preflight.build_authorization,
                    workspace_root=root,
                )
            self.assertFalse(injected_target.exists())

    def test_formal_authorization_blocks_protocol_source_drift_before_mkdir(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_ref = make_data_ref(root / "data")
            config, preflight = make_formal_preflight(root, data_ref)
            target = root / "protocol-drift-target"

            with (
                patch(
                    "build_lex.train_only._expected_formal_protocol_hashes",
                    return_value={"src/build_lex/web_search.py": "0" * 64},
                ),
                self.assertRaisesRegex(
                    TrainOnlyLexiconError, "protocol binding"
                ),
            ):
                build_train_only_lexicon(
                    "full",
                    config,
                    builder=llm_lexicon_builder.build_lexicon,
                    data_ref=data_ref,
                    train_partition_ref=root / "train_partition_ref.json",
                    formal=True,
                    target_root=target,
                    build_authorization=preflight.build_authorization,
                    workspace_root=root,
                )
            self.assertFalse(target.exists())

    def test_formal_build_rejects_data_ref_drift_before_mkdir(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_ref = make_data_ref(root / "data")
            config, preflight = make_formal_preflight(root, data_ref)
            data_ref.write_text("{\"mutated\":true}\n", encoding="utf-8")

            with self.assertRaisesRegex(TrainOnlyLexiconError, "data_ref drifted"):
                formal_build(
                    root,
                    data_ref=data_ref,
                    config=config,
                    preflight=preflight,
                )
            self.assertFalse((root / "lexicons").exists())

    def test_formal_build_rejects_train_payload_drift_before_mkdir(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_ref = make_data_ref(root / "data")
            config, preflight = make_formal_preflight(root, data_ref)
            locator = json.loads(data_ref.read_text(encoding="utf-8"))
            train_path = Path(locator["target_path"]) / "train.json"
            train_path.write_text("[]\n", encoding="utf-8")

            with self.assertRaisesRegex(TrainOnlyLexiconError, "train data drifted"):
                formal_build(
                    root,
                    data_ref=data_ref,
                    config=config,
                    preflight=preflight,
                )
            self.assertFalse((root / "lexicons").exists())

    def test_formal_zero_candidates_is_an_explicit_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaisesRegex(TrainOnlyLexiconError, "zero candidates"):
                formal_build(root, candidates=[])
            self.assertFalse((root / "lexicon_ref.json").exists())

    def test_calibration_candidate_contribution_is_a_hard_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_ref = make_data_ref(root / "data")
            make_formal_preflight(root, data_ref)
            partition = load_train_partition(
                root / "train_partition_ref.json", workspace_root=root
            )
            with self.assertRaisesRegex(
                TrainOnlyLexiconError, "train-only|forbidden|non-train"
            ):
                formal_build(
                    root,
                    data_ref=data_ref,
                    candidate_support_id=partition.calibration_ids[0],
                )
            self.assertFalse((root / "lexicon_ref.json").exists())

    def test_formal_disabled_web_policy_still_captures_local_query_coverage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = formal_config()
            config["web_settings"] = {
                "backend": "disabled",
                "cache_enabled": False,
                "cache_path": None,
                "require_direct_term_match": False,
                "dedupe_by_url": True,
                "transport_retry_policy": {
                    "id": "tavily-transient/v1",
                    "retries": 2,
                    "base_sleep_seconds": 1.0,
                },
                "physical_attempt_budget": {
                    "scope_id": "stage1-p0-wp3-formal-full-tavily-key2/v1",
                    "cap": 3100,
                },
            }
            locator, _config, built_preflight = formal_build(root, config=config)
            manifest = validate_lexicon_ref(
                root / "lexicon_ref.json", workspace_root=root
            )
            audit = manifest["evidence_audit"]
            self.assertEqual(audit["web_backend"], "disabled")
            self.assertEqual(audit["query_count"], 3)
            self.assertEqual(audit["search_capture_count"], 3)
            self.assertEqual(audit["logical_web_request_count"], 0)
            self.assertEqual(audit["physical_web_attempt_count"], 0)
            self.assertEqual(audit["authorized_web_physical_attempt_cap"], 0)
            self.assertEqual(
                built_preflight.request_budget,
                {
                    "logical_web_requests": 0,
                    "max_transport_attempts_per_logical": 3,
                    "theoretical_web_physical_attempts": 0,
                    "global_web_physical_attempt_cap": 3100,
                    "web_retry_reserve": 3100,
                    "max_llm_requests": 9000,
                },
            )
            web_row = json.loads(
                (Path(locator["target_path"]) / "web_evidence.jsonl").read_text(encoding="utf-8")
            )
            self.assertEqual(web_row["evidence"], [])

    def test_formal_published_capture_is_structurally_revalidated(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            locator, _config, _preflight = formal_build(root)
            target = Path(locator["target_path"])
            capture_path = target / "debug_llm_calls.jsonl"
            rows = [json.loads(line) for line in capture_path.read_text(encoding="utf-8").splitlines()]
            rows[0]["raw_response"] = {}
            capture_path.write_text(
                "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
                encoding="utf-8",
            )
            rewrite_payload_manifest(target)
            locator_payload = json.loads((root / "lexicon_ref.json").read_text(encoding="utf-8"))
            locator_payload["payload_manifest_sha256"] = sha256_file(target / "payload_manifest.json")
            write_json(root / "lexicon_ref.json", locator_payload)

            with self.assertRaises(TrainOnlyLexiconError):
                validate_lexicon_ref(
                    root / "lexicon_ref.json", workspace_root=root
                )

    def test_formal_target_is_portable_across_workspace_roots(self):
        with tempfile.TemporaryDirectory() as tmp:
            outer = Path(tmp)
            first, _first_config, _first_preflight = formal_build(
                outer / "workspace-a"
            )
            second, _second_config, _second_preflight = formal_build(
                outer / "workspace-b"
            )
            self.assertEqual(first["artifact_id"], second["artifact_id"])
            for filename in ("data_ref.json", "train_partition_ref.json"):
                self.assertEqual(
                    (Path(first["target_path"]) / filename).read_bytes(),
                    (Path(second["target_path"]) / filename).read_bytes(),
                )

    def test_formal_target_rejects_locator_or_implicit_workspace(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            locator, _config, _preflight = formal_build(root)
            with self.assertRaisesRegex(
                TrainOnlyLexiconError, "explicit workspace_root"
            ):
                validate_lexicon_ref(root / "lexicon_ref.json")

            target = Path(locator["target_path"])
            embedded = json.loads(
                (target / "data_ref.json").read_text(encoding="utf-8")
            )
            embedded["target_path"] = "/workspace-specific/data"
            write_json(target / "data_ref.json", embedded)
            rewrite_payload_manifest(target)
            ref = json.loads(
                (root / "lexicon_ref.json").read_text(encoding="utf-8")
            )
            ref["payload_manifest_sha256"] = sha256_file(
                target / "payload_manifest.json"
            )
            write_json(root / "lexicon_ref.json", ref)
            with self.assertRaisesRegex(
                TrainOnlyLexiconError, "non-portable dependency"
            ):
                validate_lexicon_ref(
                    root / "lexicon_ref.json", workspace_root=root
                )

    def test_formal_capture_containing_preflight_credential_is_never_published(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaisesRegex(TrainOnlyLexiconError, "credential value"):
                formal_build(root, secret_in_capture="deepseek-fixture")
            self.assertFalse((root / "lexicon_ref.json").exists())
            published = [
                path
                for path in (root / "lexicons").iterdir()
                if path.is_dir() and path.name.startswith("lex-")
            ]
            self.assertEqual(published, [])
            ledgers = list(
                (root / "lexicons").glob(".formal-billing-ledger.*.jsonl")
            )
            self.assertEqual(ledgers, [])
            checkpoint_files = [
                path
                for runtime_directory in (
                    root
                    / "exps"
                    / "causal_context"
                    / "stage1_p0"
                    / "lexicons"
                ).glob(".formal-checkpoint*")
                for path in runtime_directory.rglob("*")
                if path.is_file()
            ]
            self.assertTrue(checkpoint_files)
            checkpoint_text = "\n".join(
                path.read_text(encoding="utf-8", errors="replace")
                for path in checkpoint_files
            )
            self.assertNotIn("deepseek-fixture", checkpoint_text)
            self.assertNotIn("tavily-fixture", checkpoint_text)

    def test_formal_capture_containing_credential_hash_is_never_published(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            credential_hash = hashlib.sha256(
                b"deepseek-fixture"
            ).hexdigest()
            with self.assertRaisesRegex(
                TrainOnlyLexiconError, "credential value"
            ):
                formal_build(root, secret_in_capture=credential_hash)
            self.assertFalse((root / "lexicon_ref.json").exists())
            self.assertNotIn(
                credential_hash,
                "\n".join(
                    path.read_text(encoding="utf-8", errors="replace")
                    for path in (
                        root
                        / "exps"
                        / "causal_context"
                        / "stage1_p0"
                        / "lexicons"
                    ).rglob("*")
                    if path.is_file()
                ),
            )


if __name__ == "__main__":
    unittest.main()
