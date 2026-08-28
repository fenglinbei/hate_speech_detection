import contextlib
import hashlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from build_lex.llm_lexicon_builder import main as lexicon_builder_main
from build_lex.stage1_preflight import (
    FormalLexiconPreflightError,
    parse_allowlisted_env_file,
    preflight_formal_lexicon,
)
from data.train_partition import build_train_partition, validate_train_partition
from tests.stage1_semantic_fixtures import make_semantic_data_artifact


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
FORMAL_CONFIG = REPOSITORY_ROOT / "config" / "stage1" / "lexicon_train_only.json"


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


def make_data_ref(root):
    records = {
        "train": [normalized_record("1", "train fixture")],
        "dev": [normalized_record("2", "dev fixture")],
        "test": [normalized_record("3", "test fixture")],
    }
    split = {
        "schema_version": "stage1-split/v1",
        "policy": "fixture",
        "id_policy": "fixture",
        "source_sha256": "f" * 64,
        "train_ids": ["1"],
        "dev_ids": ["2"],
        "test_ids": ["3"],
        "train_ids_sha256": hashlib.sha256(canonical_bytes(["1"])).hexdigest(),
        "dev_ids_sha256": hashlib.sha256(canonical_bytes(["2"])).hexdigest(),
        "test_ids_sha256": hashlib.sha256(canonical_bytes(["3"])).hexdigest(),
    }
    data_id = "data-" + "a" * 64
    target = root / data_id
    target.mkdir(parents=True)
    for split_name, values in records.items():
        write_json(target / f"{split_name}.json", values)
    write_json(target / "split_manifest.json", split)
    payload = {
        "schema_version": "stage1-payload-manifest/v1",
        "files": [
            {"path": path.name, "size": path.stat().st_size, "sha256": sha256_file(path)}
            for path in sorted(target.iterdir())
            if path.is_file()
        ],
    }
    write_json(target / "payload_manifest.json", payload)
    locator = {
        "schema_version": "stage1-locator-ref/v1",
        "artifact_kind": "data",
        "artifact_id": data_id,
        "target_path": str(target),
        "payload_manifest_sha256": sha256_file(target / "payload_manifest.json"),
    }
    ref = root / "data_ref.json"
    write_json(ref, locator)
    return ref


def make_formal_data_partition(root):
    data_ref = make_semantic_data_artifact(root / "semantic_data")[0]
    partition_ref = root / "train_partition_ref.json"
    build_train_partition(
        data_ref=data_ref,
        write_ref=partition_ref,
        workspace_root=root,
        target_root=root / "train_partitions",
    )
    return data_ref, partition_ref


def valid_config():
    config = json.loads(FORMAL_CONFIG.read_text(encoding="utf-8"))
    config["web_settings"]["cache_enabled"] = False
    # Explicit null cancels the legacy default during recursive config merge.
    config["web_settings"]["cache_path"] = None
    config["web_settings"]["max_results"] = 3
    config["web_settings"]["timeout"] = 30
    config["llm_settings"]["retries"] = 2
    config["llm_settings"]["retry_sleep"] = 1
    return config


def write_env(path, *, tavily="tavily-test-secret", deepseek="deepseek-test-secret"):
    path.write_text(
        f"TAVILY_API_KEY={tavily}\nDEEPSEEK_API_KEY={deepseek}\n",
        encoding="utf-8",
    )


def blocker_codes(result):
    return {finding.code for finding in result.blockers}


class Stage1LexiconPreflightTest(unittest.TestCase):
    def test_allowlisted_env_parser_is_literal_and_ignores_unrelated_keys(self):
        with tempfile.TemporaryDirectory() as tmp:
            env_path = Path(tmp) / ".env"
            env_path.write_text(
                "UNRELATED=$(touch should-never-run)\n"
                "TAVILY_API_KEY='$NOT_EXPANDED'\n"
                "export DEEPSEEK_API_KEY=literal-value # comment\n",
                encoding="utf-8",
            )

            values = parse_allowlisted_env_file(env_path)

            self.assertEqual(values["TAVILY_API_KEY"], "$NOT_EXPANDED")
            self.assertEqual(values["DEEPSEEK_API_KEY"], "literal-value")
            self.assertNotIn("UNRELATED", values)
            self.assertFalse((Path(tmp) / "should-never-run").exists())

    def test_duplicate_allowlisted_env_assignment_fails_without_value_disclosure(self):
        with tempfile.TemporaryDirectory() as tmp:
            env_path = Path(tmp) / ".env"
            env_path.write_text(
                "DEEPSEEK_API_KEY=first-secret\nDEEPSEEK_API_KEY=second-secret\n",
                encoding="utf-8",
            )
            with self.assertRaises(FormalLexiconPreflightError) as raised:
                parse_allowlisted_env_file(env_path)
            message = str(raised.exception)
            self.assertNotIn("first-secret", message)
            self.assertNotIn("second-secret", message)

    def test_valid_preflight_is_read_only_and_does_not_construct_runtime_clients(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path = root / "lexicon.json"
            env_path = root / ".env"
            checkpoint_path = root / "provider-slot-checkpoint.json"
            data_ref, partition_ref = make_formal_data_partition(root)
            write_json(config_path, valid_config())
            write_env(env_path)
            checkpoint_path.write_text("checkpoint-must-not-be-read\n", encoding="utf-8")
            before = {
                path.relative_to(root).as_posix(): path.read_bytes()
                for path in root.rglob("*")
                if path.is_file()
            }

            original_path_open = Path.open

            def guarded_path_open(path, *args, **kwargs):
                if path == checkpoint_path:
                    raise AssertionError("preflight must not read the runtime checkpoint")
                return original_path_open(path, *args, **kwargs)

            with (
                patch("build_lex.llm_lexicon_builder.build_candidates") as mine,
                patch("build_lex.web_search.SQLiteKVCache") as cache,
                patch("build_lex.web_search.requests.post") as http_post,
                patch("build_lex.web_search.requests.get") as http_get,
                patch.object(Path, "open", guarded_path_open),
            ):
                result = preflight_formal_lexicon(
                    config_path=config_path,
                    data_ref=data_ref,
                    train_partition_ref=partition_ref,
                    env_file=env_path,
                    repository_root=root,
                    environ={},
                )

            after = {
                path.relative_to(root).as_posix(): path.read_bytes()
                for path in root.rglob("*")
                if path.is_file()
            }
            self.assertTrue(result.passed, result.to_public_dict())
            self.assertEqual(result.data_build_id, json.loads(data_ref.read_text())["artifact_id"])
            self.assertEqual(
                result.train_record_count,
                validate_train_partition(partition_ref, workspace_root=root)[
                    "fit_count"
                ],
            )
            self.assertEqual(result.train_partition_id, json.loads(partition_ref.read_text())["artifact_id"])
            self.assertRegex(result.train_data_sha256 or "", r"^[0-9a-f]{64}$")
            self.assertRegex(result.train_ids_sha256 or "", r"^[0-9a-f]{64}$")
            self.assertEqual(result.cache_policy, "disabled")
            self.assertEqual(
                result.request_budget,
                {
                    "logical_web_requests": 3000,
                    "max_transport_attempts_per_logical": 3,
                    "theoretical_web_physical_attempts": 9000,
                    "global_web_physical_attempt_cap": 3100,
                    "web_retry_reserve": 100,
                    "max_llm_requests": 9000,
                },
            )
            self.assertEqual(
                result.to_public_dict()["schema_version"],
                "stage1-formal-lexicon-preflight/v3",
            )
            self.assertEqual(result.frozen_config(), valid_config())
            self.assertEqual(repr(result.build_authorization), "<formal Stage-1 lexicon build authorization>")
            self.assertEqual(before, after)
            mine.assert_not_called()
            cache.assert_not_called()
            http_post.assert_not_called()
            http_get.assert_not_called()

            public_text = json.dumps(result.to_public_dict(), ensure_ascii=False)
            result_repr = repr(result)
            for secret in ("tavily-test-secret", "deepseek-test-secret"):
                self.assertNotIn(secret, public_text)
                self.assertNotIn(secret, result_repr)

    def test_live_web_evidence_filters_are_required_for_formal_build(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = valid_config()
            config["web_settings"]["require_direct_term_match"] = False
            config["web_settings"]["dedupe_by_url"] = False
            config["candidate_settings"]["use_jieba"] = True
            config["candidate_settings"]["keep_all_content_ngrams"] = False
            config["candidate_settings"]["suppressed_reject_hints"].append(
                "singleton_ngram"
            )
            config_path = root / "lexicon.json"
            env_path = root / ".env"
            data_ref, partition_ref = make_formal_data_partition(root)
            write_json(config_path, config)
            write_env(env_path)

            result = preflight_formal_lexicon(
                config_path=config_path,
                data_ref=data_ref,
                train_partition_ref=partition_ref,
                env_file=env_path,
                repository_root=root,
                environ={},
            )

            self.assertIn("WEB_EVIDENCE_POLICY_DISABLED", blocker_codes(result))
            self.assertIn("UNFROZEN_OPTIONAL_SEGMENTER", blocker_codes(result))
            self.assertIn(
                "NARROW_TERMINOLOGY_CANDIDATE_FRAME", blocker_codes(result)
            )

    def test_live_web_evidence_text_limits_are_required_and_bounded(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            env_path = root / ".env"
            data_ref, partition_ref = make_formal_data_partition(root)
            write_env(env_path)
            cases = {
                "missing": ("max_title_chars", None),
                "zero": ("max_snippet_chars", 0),
                "bool": ("max_source_chars", True),
                "over": ("max_url_chars", 2049),
            }
            for name, (key, value) in cases.items():
                with self.subTest(name=name):
                    config = valid_config()
                    if value is None:
                        config["web_settings"].pop(key)
                    else:
                        config["web_settings"][key] = value
                    config_path = root / f"lexicon-{name}.json"
                    write_json(config_path, config)
                    result = preflight_formal_lexicon(
                        config_path=config_path,
                        data_ref=data_ref,
                        train_partition_ref=partition_ref,
                        env_file=env_path,
                        repository_root=root,
                        environ={},
                    )
                    self.assertIn(
                        "WEB_EVIDENCE_TEXT_LIMIT_INVALID", blocker_codes(result)
                    )

    def test_tavily_transport_and_physical_attempt_budget_are_exactly_frozen(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            env_path = root / ".env"
            data_ref, partition_ref = make_formal_data_partition(root)
            write_env(env_path)
            missing = object()
            cases = (
                (
                    "retry-policy-missing",
                    "transport_retry_policy",
                    missing,
                    {
                        "CONFIG_SECTION_KEYS_MISSING",
                        "WEB_TRANSPORT_RETRY_POLICY_INVALID",
                    },
                ),
                (
                    "retry-count-bool",
                    "transport_retry_policy",
                    {
                        "id": "tavily-transient/v1",
                        "retries": True,
                        "base_sleep_seconds": 1.0,
                    },
                    {"WEB_TRANSPORT_RETRY_POLICY_INVALID"},
                ),
                (
                    "retry-policy-extra-key",
                    "transport_retry_policy",
                    {
                        "id": "tavily-transient/v1",
                        "retries": 2,
                        "base_sleep_seconds": 1.0,
                        "jitter": True,
                    },
                    {"WEB_TRANSPORT_RETRY_POLICY_INVALID"},
                ),
                (
                    "physical-budget-missing",
                    "physical_attempt_budget",
                    missing,
                    {
                        "CONFIG_SECTION_KEYS_MISSING",
                        "WEB_PHYSICAL_ATTEMPT_BUDGET_INVALID",
                    },
                ),
                (
                    "physical-cap-bool",
                    "physical_attempt_budget",
                    {
                        "scope_id": "stage1-p0-wp3-formal-full-tavily-key2/v1",
                        "cap": True,
                    },
                    {"WEB_PHYSICAL_ATTEMPT_BUDGET_INVALID"},
                ),
                (
                    "physical-scope-drift",
                    "physical_attempt_budget",
                    {
                        "scope_id": "stage1-p0-wp3-formal-full-tavily-key3/v1",
                        "cap": 3100,
                    },
                    {"WEB_PHYSICAL_ATTEMPT_BUDGET_INVALID"},
                ),
            )
            for name, field, value, expected_codes in cases:
                with self.subTest(name=name):
                    config = valid_config()
                    if value is missing:
                        config["web_settings"].pop(field)
                    else:
                        config["web_settings"][field] = value
                    config_path = root / f"config-{name}.json"
                    write_json(config_path, config)
                    result = preflight_formal_lexicon(
                        config_path=config_path,
                        data_ref=data_ref,
                        train_partition_ref=partition_ref,
                        env_file=env_path,
                        repository_root=root,
                        environ={},
                    )
                    self.assertTrue(
                        expected_codes.issubset(blocker_codes(result)),
                        result.to_public_dict(),
                    )

    def test_formal_checkpoint_runtime_policy_is_required_and_exact(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = valid_config()
            config["schema_version"] = "stage1-train-only-lexicon-config/v1"
            config["runtime_settings"].update(
                {
                    "resume": False,
                    "resume_require_config_match": False,
                    "formal_checkpoint_policy": "legacy-checkpoint/v1",
                    "ambiguous_attempt_policy": "ignore/v1",
                    "max_llm_http_attempts": True,
                }
            )
            config_path = root / "config.json"
            env_path = root / ".env"
            data_ref, partition_ref = make_formal_data_partition(root)
            write_json(config_path, config)
            write_env(env_path)

            result = preflight_formal_lexicon(
                config_path=config_path,
                data_ref=data_ref,
                train_partition_ref=partition_ref,
                env_file=env_path,
                repository_root=root,
                environ={},
            )

            self.assertTrue(
                {
                    "CONFIG_SCHEMA_INVALID",
                    "RESUME_REQUIRED",
                    "RESUME_GUARD_INVALID",
                    "FORMAL_CHECKPOINT_POLICY_INVALID",
                    "AMBIGUOUS_ATTEMPT_POLICY_INVALID",
                    "MAX_LLM_HTTP_ATTEMPTS_INVALID",
                }.issubset(blocker_codes(result)),
                result.to_public_dict(),
            )

    def test_new_checkpoint_fields_remain_strictly_required_and_allowlisted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = valid_config()
            config["runtime_settings"].pop("formal_checkpoint_policy")
            config["runtime_settings"]["checkpoint_path"] = "must-not-be-configurable"
            config_path = root / "config.json"
            env_path = root / ".env"
            data_ref, partition_ref = make_formal_data_partition(root)
            write_json(config_path, config)
            write_env(env_path)

            result = preflight_formal_lexicon(
                config_path=config_path,
                data_ref=data_ref,
                train_partition_ref=partition_ref,
                env_file=env_path,
                repository_root=root,
                environ={},
            )

            self.assertTrue(
                {
                    "CONFIG_SECTION_KEYS_MISSING",
                    "CONFIG_SECTION_KEYS_UNEXPECTED",
                    "FORMAL_CHECKPOINT_POLICY_INVALID",
                }.issubset(blocker_codes(result)),
                result.to_public_dict(),
            )

    def test_formal_model_and_output_budget_are_frozen(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = valid_config()
            config["llm_settings"]["model"] = "deepseek-v4-pro"
            config["llm_settings"]["max_tokens"] = 2048
            config["llm_settings"]["thinking"] = {"type": "enabled"}
            config["llm_settings"]["reasoning_effort"] = "high"
            config["llm_settings"]["retries"] = 3
            config["llm_settings"]["retry_sleep"] = 2
            config["llm_settings"]["output_language"] = "en"
            config_path = root / "lexicon.json"
            env_path = root / ".env"
            data_ref, partition_ref = make_formal_data_partition(root)
            write_json(config_path, config)
            write_env(env_path)

            result = preflight_formal_lexicon(
                config_path=config_path,
                data_ref=data_ref,
                train_partition_ref=partition_ref,
                env_file=env_path,
                repository_root=root,
                environ={},
            )

            self.assertIn("LLM_MODEL_INVALID", blocker_codes(result))
            self.assertIn("LLM_MAX_TOKENS_INVALID", blocker_codes(result))
            self.assertIn("LLM_THINKING_INVALID", blocker_codes(result))
            self.assertIn("LLM_RETRIES_INVALID", blocker_codes(result))
            self.assertIn("LLM_RETRY_SLEEP_INVALID", blocker_codes(result))
            self.assertIn("LLM_OUTPUT_LANGUAGE_INVALID", blocker_codes(result))

    def test_missing_key_blocks_before_target_or_cache_exists(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path = root / "lexicon.json"
            env_path = root / ".env"
            data_ref, partition_ref = make_formal_data_partition(root)
            target_root = root / "lexicons"
            ref_path = root / "lexicon_ref.json"
            write_json(config_path, valid_config())
            env_path.write_text("DEEPSEEK_API_KEY=deepseek-test-secret\n", encoding="utf-8")
            argv = [
                "llm_lexicon_builder.py",
                "--dataset",
                "full",
                "--config",
                str(config_path),
                "--data-ref",
                str(data_ref),
                "--train-partition-ref",
                str(partition_ref),
                "--target-root",
                str(target_root),
                "--write-ref",
                str(ref_path),
                "--env-file",
                str(env_path),
            ]
            stderr = io.StringIO()
            with (
                patch.dict(os.environ, {}, clear=True),
                patch.object(sys, "argv", argv),
                patch(
                    "build_lex.llm_lexicon_builder.SRC_ROOT", root / "src"
                ),
                patch("build_lex.llm_lexicon_builder.build_candidates") as mine,
                patch("build_lex.web_search.SQLiteKVCache") as cache,
                patch("build_lex.web_search.requests.post") as http_post,
                contextlib.redirect_stderr(stderr),
            ):
                with self.assertRaises(SystemExit) as raised:
                    lexicon_builder_main()

            self.assertEqual(raised.exception.code, 2)
            self.assertFalse(target_root.exists())
            self.assertFalse(ref_path.exists())
            mine.assert_not_called()
            cache.assert_not_called()
            http_post.assert_not_called()
            self.assertIn("CREDENTIAL_MISSING", stderr.getvalue())
            self.assertNotIn("deepseek-test-secret", stderr.getvalue())

    def test_successful_formal_cli_uses_preflighted_config_without_runtime_side_effects(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path = root / "lexicon.json"
            env_path = root / ".env"
            data_ref, partition_ref = make_formal_data_partition(root)
            write_json(config_path, valid_config())
            write_env(env_path)
            argv = [
                "llm_lexicon_builder.py",
                "--dataset",
                "full",
                "--config",
                str(config_path),
                "--data-ref",
                str(data_ref),
                "--train-partition-ref",
                str(partition_ref),
                "--target-root",
                str(root / "lexicons"),
                "--write-ref",
                str(root / "lexicon_ref.json"),
                "--env-file",
                str(env_path),
            ]
            with (
                patch.dict(os.environ, {}, clear=True),
                patch.object(sys, "argv", argv),
                patch(
                    "build_lex.llm_lexicon_builder.SRC_ROOT", root / "src"
                ),
                patch("build_lex.llm_lexicon_builder.load_build_config") as reload_config,
                patch(
                    "build_lex.train_only.build_train_only_lexicon",
                    return_value={"artifact_id": "lex-fixture"},
                ) as build,
                contextlib.redirect_stdout(io.StringIO()),
            ):
                lexicon_builder_main()

            reload_config.assert_not_called()
            build_config = build.call_args.args[1]
            self.assertIs(build_config["web_settings"]["cache_enabled"], False)
            self.assertIsNone(build_config["web_settings"]["cache_path"])
            self.assertEqual(build_config, valid_config())
            self.assertIsNotNone(build.call_args.kwargs["build_authorization"])
            self.assertEqual(
                Path(build.call_args.kwargs["train_partition_ref"]), partition_ref
            )

    def test_conflicting_process_and_file_credentials_fail_without_disclosure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path = root / "config.json"
            env_path = root / ".env"
            write_json(config_path, valid_config())
            write_env(env_path, tavily="file-secret")
            data_ref, partition_ref = make_formal_data_partition(root)
            result = preflight_formal_lexicon(
                config_path=config_path,
                data_ref=data_ref,
                train_partition_ref=partition_ref,
                env_file=env_path,
                repository_root=root,
                environ={"TAVILY_API_KEY": "process-secret"},
            )
            self.assertIn("CREDENTIAL_SOURCE_CONFLICT", blocker_codes(result))
            public = json.dumps(result.to_public_dict(), ensure_ascii=False)
            self.assertNotIn("file-secret", public)
            self.assertNotIn("process-secret", public)

    def test_implicit_and_legacy_cache_policies_are_hard_blocks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_ref, partition_ref = make_formal_data_partition(root)
            env_path = root / ".env"
            write_env(env_path)

            implicit_path = root / "implicit.json"
            implicit = valid_config()
            implicit["web_settings"].pop("cache_enabled")
            implicit["web_settings"].pop("cache_path")
            write_json(implicit_path, implicit)
            implicit_result = preflight_formal_lexicon(
                config_path=implicit_path,
                data_ref=data_ref,
                env_file=env_path,
                repository_root=root,
                environ={},
            )
            self.assertIn("CACHE_POLICY_UNRESOLVED", blocker_codes(implicit_result))

            legacy_path = root / "legacy.json"
            legacy = valid_config()
            legacy["web_settings"].update(
                {
                    "cache_enabled": True,
                    "cache_path": "data/lexicon/generated/full/web_cache.sqlite3",
                }
            )
            write_json(legacy_path, legacy)
            legacy_result = preflight_formal_lexicon(
                config_path=legacy_path,
                data_ref=data_ref,
                env_file=env_path,
                repository_root=root,
                environ={},
            )
            self.assertTrue(
                {"FORMAL_CACHE_ENABLED", "LEGACY_CACHE_PATH"}.issubset(blocker_codes(legacy_result))
            )

    def test_existing_and_git_tracked_cache_paths_are_hard_blocks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            cache_path = root / "runtime_cache.sqlite3"
            cache_path.write_bytes(b"fixture")
            subprocess.run(
                ["git", "-C", str(root), "add", "runtime_cache.sqlite3"],
                check=True,
                stdout=subprocess.DEVNULL,
            )
            config = valid_config()
            config["web_settings"].update(
                {"cache_enabled": True, "cache_path": str(cache_path)}
            )
            config_path = root / "config.json"
            env_path = root / ".env"
            write_json(config_path, config)
            write_env(env_path)
            result = preflight_formal_lexicon(
                config_path=config_path,
                data_ref=make_data_ref(root / "data"),
                env_file=env_path,
                repository_root=root,
                environ={},
            )
            self.assertTrue(
                {"TRACKED_CACHE_PATH", "SHARED_CACHE_PATH"}.issubset(blocker_codes(result))
            )

    def test_unapproved_endpoint_and_corrupt_data_ref_both_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = valid_config()
            config["web_settings"]["api_base"] = "http://example.test/search?key=bad"
            config_path = root / "config.json"
            env_path = root / ".env"
            data_ref, partition_ref = make_formal_data_partition(root)
            write_json(config_path, config)
            write_env(env_path)
            locator = json.loads(data_ref.read_text(encoding="utf-8"))
            (Path(locator["target_path"]) / "train.json").write_text("[]\n", encoding="utf-8")

            result = preflight_formal_lexicon(
                config_path=config_path,
                data_ref=data_ref,
                train_partition_ref=partition_ref,
                env_file=env_path,
                repository_root=root,
                environ={},
            )
            self.assertTrue(
                {"ENDPOINT_NOT_ALLOWLISTED", "DATA_REF_INVALID"}.issubset(blocker_codes(result))
            )

    def test_recursive_config_placeholder_is_blocked_without_cross_provider_secret_disclosure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sentinel = "cross-provider-secret-sentinel"
            config = valid_config()
            config["llm_settings"]["model"] = "prefix-$TAVILY_API_KEY"
            config["candidate_settings"]["suppressed_reject_hints"].append(
                "${DEEPSEEK_API_KEY}"
            )
            config_path = root / "config.json"
            write_json(config_path, config)
            result = preflight_formal_lexicon(
                config_path=config_path,
                data_ref=make_data_ref(root / "data"),
                env_file=None,
                repository_root=root,
                environ={
                    "TAVILY_API_KEY": sentinel,
                    "DEEPSEEK_API_KEY": sentinel + "-llm",
                },
            )

            self.assertIn("CONFIG_ENV_PLACEHOLDER_FORBIDDEN", blocker_codes(result))
            public = json.dumps(result.to_public_dict(), ensure_ascii=False)
            self.assertNotIn(sentinel, public)
            self.assertNotIn(sentinel, repr(result))
            with self.assertRaises(FormalLexiconPreflightError):
                _ = result.build_authorization

    def test_cross_provider_placeholder_cli_never_reaches_stderr_or_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sentinel = "never-publish-cross-provider-sentinel"
            config = valid_config()
            config["llm_settings"]["model"] = "$TAVILY_API_KEY"
            config_path = root / "config.json"
            write_json(config_path, config)
            data_ref = make_data_ref(root / "data")
            target_root = root / "lexicons"
            argv = [
                "llm_lexicon_builder.py",
                "--dataset",
                "full",
                "--config",
                str(config_path),
                "--data-ref",
                str(data_ref),
                "--target-root",
                str(target_root),
                "--write-ref",
                str(root / "lexicon_ref.json"),
                "--env-file",
                str(root / "absent.env"),
            ]
            stderr = io.StringIO()
            with (
                patch.dict(
                    os.environ,
                    {
                        "TAVILY_API_KEY": sentinel,
                        "DEEPSEEK_API_KEY": sentinel + "-deepseek",
                    },
                    clear=True,
                ),
                patch.object(sys, "argv", argv),
                contextlib.redirect_stderr(stderr),
            ):
                with self.assertRaises(SystemExit) as raised:
                    lexicon_builder_main()

            self.assertEqual(raised.exception.code, 2)
            self.assertIn("CONFIG_ENV_PLACEHOLDER_FORBIDDEN", stderr.getvalue())
            self.assertNotIn(sentinel, stderr.getvalue())
            self.assertFalse(target_root.exists())
            self.assertFalse((root / "lexicon_ref.json").exists())

    def test_nested_web_and_llm_retry_protocol_types_are_strict(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = valid_config()
            config["web_settings"]["api_extra_params"] = {
                "max_results": True,
                "search_depth": "unbounded",
            }
            config["llm_settings"].update(
                {
                    "thinking": {"type": "disabled", "extra": True},
                    "reasoning_effort": "high",
                    "retries": True,
                    "retry_sleep": -1,
                }
            )
            config_path = root / "config.json"
            env_path = root / ".env"
            write_json(config_path, config)
            write_env(env_path)
            result = preflight_formal_lexicon(
                config_path=config_path,
                data_ref=make_data_ref(root / "data"),
                env_file=env_path,
                repository_root=root,
                environ={},
            )
            self.assertTrue(
                {
                    "WEB_MAX_RESULTS_INVALID",
                    "WEB_SEARCH_DEPTH_INVALID",
                    "LLM_THINKING_INVALID",
                    "LLM_RETRIES_INVALID",
                    "LLM_RETRY_SLEEP_INVALID",
                }.issubset(blocker_codes(result))
            )

    def test_formal_preflight_binds_only_normalized_full_dataset(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path = root / "config.json"
            env_path = root / ".env"
            write_json(config_path, valid_config())
            write_env(env_path)
            result = preflight_formal_lexicon(
                dataset="state",
                config_path=config_path,
                data_ref=make_data_ref(root / "data"),
                env_file=env_path,
                repository_root=root,
                environ={},
            )
            self.assertIn("DATASET_UNSUPPORTED", blocker_codes(result))

    def test_activation_is_separate_allowlisted_and_never_overwrites(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path = root / "config.json"
            env_path = root / ".env"
            write_json(config_path, valid_config())
            write_env(env_path)
            data_ref, partition_ref = make_formal_data_partition(root)
            result = preflight_formal_lexicon(
                config_path=config_path,
                data_ref=data_ref,
                train_partition_ref=partition_ref,
                env_file=env_path,
                repository_root=root,
                environ={},
            )
            target = {"UNRELATED": "keep"}
            result.activate_credentials(target)
            self.assertEqual(target["UNRELATED"], "keep")
            self.assertEqual(set(target), {"UNRELATED", "TAVILY_API_KEY", "DEEPSEEK_API_KEY"})
            with self.assertRaises(FormalLexiconPreflightError):
                result.activate_credentials({"TAVILY_API_KEY": "different"})


if __name__ == "__main__":
    unittest.main()
