import json
import hashlib
import os
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

from build_lex.formal_checkpoint import (
    CheckpointSpec,
    FormalCheckpointError,
    FormalLexiconCheckpoint,
)


HASH = "a" * 64


def make_spec(
    *,
    term="火星人",
    tavily_cap=4,
    deepseek_cap=6,
    config_hash=HASH,
    active_provider_slots=None,
):
    return CheckpointSpec.build(
        intent={
            "authorization_sha256": HASH,
            "builder_code_sha256": "b" * 64,
            "config_sha256": config_hash,
            "data_build_id": "data-fixture",
            "protocol_code_sha256s": {"builder.py": "c" * 64},
            "train_records_sha256": "d" * 64,
        },
        provider_caps={
            "tavily": {"cap": tavily_cap, "scope_id": "tavily-key2/v1"},
            "deepseek": {"cap": deepseek_cap, "scope_id": "deepseek/v1"},
        },
        candidate_frame=[{"rank": 1, "term": term, "support_sample_ids": ["1"]}],
        max_slot_attempts=3,
        active_provider_slots=active_provider_slots,
    )


def finish_success(checkpoint, provider, slot, request, *, capture, response):
    reservation = checkpoint.reserve_attempt(provider, 1, slot, request)
    checkpoint.finish_attempt(
        reservation,
        status="success",
        response=response,
        capture=capture,
        detail={"error_type": None, "http_status": 200, "retryable": False},
    )


class FormalCheckpointTest(unittest.TestCase):
    def test_resume_reuses_slots_and_materializes_canonical_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "checkpoint"
            spec = make_spec(tavily_cap=3, deepseek_cap=3)
            checkpoint = FormalLexiconCheckpoint.create(root, spec)
            queries = ["q1", "q2", "q3"]
            evidence = []
            for index, query in enumerate(queries, start=1):
                result = {
                    "id": f"e{index}",
                    "query": query,
                    "title": "title",
                    "snippet": "snippet",
                    "url": "https://example.test",
                    "source": "fixture",
                }
                evidence.append(result)
                finish_success(
                    checkpoint,
                    "tavily",
                    f"query_{index}",
                    {"query": query},
                    response={"results": [result]},
                    capture={
                        "term": "火星人",
                        "query": query,
                        "results": [result],
                        "error": None,
                    },
                )
            stages = (
                "context_judge",
                "web_evidence_judge",
                "final_lexicon_judge",
            )
            parsed_by_stage = {}
            for stage in stages:
                parsed = {"stage": stage, "ok": True}
                parsed_by_stage[stage] = parsed
                request = {"model": "deepseek-v4-flash", "messages": [{"role": "user", "content": stage}]}
                finish_success(
                    checkpoint,
                    "deepseek",
                    stage,
                    request,
                    response={"parsed_response": parsed},
                    capture={
                        "stage": stage,
                        "term": "火星人",
                        "attempt": 1,
                        "request_payload": request,
                        "raw_response": {"choices": []},
                        "parsed_response": parsed,
                        "error": None,
                    },
                )
            candidate = dict(spec.candidate_frame[0])
            web_row = {
                "rank": 1,
                "term": "火星人",
                "queries": queries,
                "evidence": evidence,
            }
            judgement_row = {"rank": 1, "term": "火星人", **parsed_by_stage}
            checkpoint.commit_candidate(
                1,
                {
                    "candidate": candidate,
                    "web_evidence": web_row,
                    "llm_judgement": judgement_row,
                    "term": {"term": "火星人"},
                    "rejected": None,
                },
            )
            checkpoint.close()

            resumed = FormalLexiconCheckpoint.resume(root, spec)
            cached = resumed.get_slot_success(
                "deepseek",
                1,
                "context_judge",
                request_payload={
                    "model": "deepseek-v4-flash",
                    "messages": [{"role": "user", "content": "context_judge"}],
                },
            )
            self.assertEqual(cached.response["parsed_response"]["ok"], True)
            output = Path(tmp) / "materialized"
            summary = resumed.materialize(output)
            resumed.close()

            self.assertEqual(summary["committed_prefix"], 1)
            self.assertEqual(
                summary["provider_attempt_counts"], {"deepseek": 3, "tavily": 3}
            )
            self.assertEqual(summary["attempt_head"]["reservation_count"], 6)
            self.assertEqual(
                summary["attempt_head"]["provider_counts"],
                {"deepseek": 3, "tavily": 3},
            )
            self.assertEqual(
                summary["provider_attempt_scopes"]["tavily"], "tavily-key2/v1"
            )
            self.assertEqual(
                summary["candidate_commit_head"]["commit_sha256"],
                json.loads(
                    (root / "candidates" / "000001.json").read_text(
                        encoding="utf-8"
                    )
                )["commit_sha256"],
            )
            self.assertEqual(
                len((output / "debug" / "tavily_attempts.jsonl").read_text().splitlines()),
                3,
            )
            safe_row = json.loads(
                (output / "debug" / "tavily_attempts.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()[0]
            )
            self.assertNotIn("query", safe_row)
            self.assertNotIn("term", safe_row)
            self.assertEqual(os.stat(root).st_mode & 0o777, 0o700)
            self.assertEqual(os.stat(root / "manifest.json").st_mode & 0o777, 0o600)

    def test_pending_reservation_becomes_ambiguous_and_can_retry(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "checkpoint"
            spec = make_spec(tavily_cap=2)
            checkpoint = FormalLexiconCheckpoint.create(root, spec)
            first = checkpoint.reserve_attempt("tavily", 1, "query_1", {"query": "q"})
            self.assertEqual(first.attempt, 1)
            checkpoint.close()

            resumed = FormalLexiconCheckpoint.resume(root, spec)
            self.assertEqual(resumed.summary()["ambiguous_attempt_count"], 1)
            self.assertTrue(resumed.can_retry("tavily", 1, "query_1"))
            second = resumed.reserve_attempt("tavily", 1, "query_1", {"query": "q"})
            self.assertEqual(second.attempt, 2)
            resumed.finish_attempt(
                second,
                status="success",
                response={"results": []},
                capture={"term": "火星人", "query": "q", "results": [], "error": None},
                detail={"error_type": None, "http_status": 200, "retryable": False},
            )
            resumed.close()

    def test_global_cap_is_reserved_before_another_attempt(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = FormalLexiconCheckpoint.create(
                Path(tmp) / "checkpoint", make_spec(tavily_cap=1)
            )
            first = checkpoint.reserve_attempt("tavily", 1, "query_1", {"query": "q1"})
            checkpoint.finish_attempt(
                first,
                status="retryable_failure",
                detail={"error_type": "Timeout", "http_status": None, "retryable": True},
            )
            with self.assertRaisesRegex(FormalCheckpointError, "global physical-attempt"):
                checkpoint.reserve_attempt("tavily", 1, "query_1", {"query": "q1"})
            checkpoint.close()

    def test_candidate_or_config_drift_is_rejected_at_same_checkpoint_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "checkpoint"
            checkpoint = FormalLexiconCheckpoint.create(root, make_spec())
            checkpoint.close()
            for drifted in (
                make_spec(term="火星怪"),
                make_spec(config_hash="e" * 64),
            ):
                with self.assertRaisesRegex(FormalCheckpointError, "drifted"):
                    FormalLexiconCheckpoint.resume(root, drifted)

    def test_single_writer_lock_and_symlink_are_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "checkpoint"
            first = FormalLexiconCheckpoint.create(root, make_spec())
            with self.assertRaisesRegex(FormalCheckpointError, "active writer"):
                FormalLexiconCheckpoint.resume(root, make_spec())
            first.close()
            manifest = root / "manifest.json"
            manifest.unlink()
            manifest.symlink_to(root / "writer.lock")
            with self.assertRaisesRegex(FormalCheckpointError, "unsafe"):
                FormalLexiconCheckpoint.resume(root, make_spec())

    def test_attempt_tail_deletion_cannot_roll_back_physical_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "checkpoint"
            spec = make_spec(tavily_cap=2)
            checkpoint = FormalLexiconCheckpoint.create(root, spec)
            for slot in ("query_1", "query_2"):
                reservation = checkpoint.reserve_attempt(
                    "tavily", 1, slot, {"query": slot}
                )
                checkpoint.finish_attempt(
                    reservation,
                    status="retryable_failure",
                    detail={"error_type": "Timeout", "retryable": True},
                )
            checkpoint.close()

            (root / "attempts" / "000000002.json").unlink()
            with self.assertRaisesRegex(FormalCheckpointError, "rolled back"):
                FormalLexiconCheckpoint.resume(root, spec)

    def test_attempt_written_before_head_is_counted_conservatively(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "checkpoint"
            spec = make_spec(tavily_cap=2)
            checkpoint = FormalLexiconCheckpoint.create(root, spec)
            with mock.patch.object(
                checkpoint,
                "_write_attempt_head",
                side_effect=OSError("simulated SIGKILL window"),
            ):
                with self.assertRaisesRegex(OSError, "SIGKILL"):
                    checkpoint.reserve_attempt(
                        "tavily", 1, "query_1", {"query": "q"}
                    )
            checkpoint.close()

            resumed = FormalLexiconCheckpoint.resume(root, spec)
            self.assertEqual(
                resumed.summary()["provider_attempt_counts"]["tavily"], 1
            )
            self.assertEqual(resumed.summary()["ambiguous_attempt_count"], 1)
            resumed.close()

    def test_atomic_temp_orphans_are_removed_but_other_entries_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "checkpoint"
            spec = make_spec()
            checkpoint = FormalLexiconCheckpoint.create(root, spec)
            checkpoint.close()
            orphans = (
                root / "attempts" / ".000000001.json.ABCDEF12",
                root / "attempts" / ".HEAD.json.ABCDEF12",
                root / "slots" / ".tavily-000001-1.json.ABCDEF12",
                root / "candidates" / ".000001.json.ABCDEF12",
            )
            for path in orphans:
                path.write_text("partial", encoding="utf-8")
                path.chmod(0o600)

            resumed = FormalLexiconCheckpoint.resume(root, spec)
            resumed.close()
            self.assertTrue(all(not path.exists() for path in orphans))

            unexpected = root / "attempts" / "notes.txt"
            unexpected.write_text("not checkpoint state", encoding="utf-8")
            unexpected.chmod(0o600)
            with self.assertRaisesRegex(FormalCheckpointError, "unexpected entry"):
                FormalLexiconCheckpoint.resume(root, spec)

    def test_success_outcome_without_slot_index_is_recovered(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "checkpoint"
            spec = make_spec()
            checkpoint = FormalLexiconCheckpoint.create(root, spec)
            reservation = checkpoint.reserve_attempt(
                "tavily", 1, "query_1", {"query": "q"}
            )
            with mock.patch.object(
                checkpoint,
                "_write_slot_success",
                side_effect=OSError("simulated crash after outcome"),
            ):
                with self.assertRaisesRegex(OSError, "after outcome"):
                    checkpoint.finish_attempt(
                        reservation,
                        status="success",
                        response={"results": []},
                        capture={
                            "term": "火星人",
                            "query": "q",
                            "results": [],
                            "error": None,
                        },
                        detail={
                            "error_type": None,
                            "http_status": 200,
                            "retryable": False,
                        },
                    )
            checkpoint.close()
            self.assertFalse((root / "slots" / "tavily-000001-1.json").exists())

            resumed = FormalLexiconCheckpoint.resume(root, spec)
            cached = resumed.get_slot_success(
                "tavily", 1, "query_1", request_payload={"query": "q"}
            )
            self.assertIsNotNone(cached)
            self.assertEqual(cached.attempt, 1)
            resumed.close()

    def test_commit_requires_all_bound_slots_by_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = FormalLexiconCheckpoint.create(
                Path(tmp) / "checkpoint", make_spec()
            )
            with self.assertRaisesRegex(FormalCheckpointError, "every active provider slot"):
                checkpoint.commit_candidate(
                    1,
                    {
                        "candidate": dict(checkpoint.spec.candidate_frame[0]),
                        "web_evidence": {},
                        "llm_judgement": {},
                        "term": {"term": "火星人"},
                        "rejected": None,
                    },
                )
            checkpoint.close()

    def test_explicit_inactive_slots_are_bound_into_manifest_and_commit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "checkpoint"
            spec = make_spec(
                tavily_cap=0,
                deepseek_cap=0,
                active_provider_slots={"tavily": (), "deepseek": ()},
            )
            checkpoint = FormalLexiconCheckpoint.create(root, spec)
            checkpoint.commit_candidate(
                1,
                {
                    "candidate": dict(spec.candidate_frame[0]),
                    "web_evidence": {"queries": [], "evidence": []},
                    "llm_judgement": {},
                    "term": {"term": "火星人"},
                    "rejected": None,
                },
            )
            checkpoint.close()

            manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
            commit = json.loads(
                (root / "candidates" / "000001.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                manifest["active_provider_slots"], {"deepseek": [], "tavily": []}
            )
            self.assertEqual(
                commit["provider_slot_sha256s"], {"deepseek": {}, "tavily": {}}
            )
            resumed = FormalLexiconCheckpoint.resume(root, spec)
            self.assertEqual(resumed.committed_prefix, 1)
            resumed.close()

    def test_nested_secret_fields_and_corrupt_direct_slot_reads_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "checkpoint"
            spec = make_spec()
            checkpoint = FormalLexiconCheckpoint.create(root, spec)
            with self.assertRaisesRegex(FormalCheckpointError, "forbidden secret field"):
                checkpoint.reserve_attempt(
                    "deepseek",
                    1,
                    "context_judge",
                    {"messages": [{"authorization": "Bearer secret"}]},
                )
            finish_success(
                checkpoint,
                "tavily",
                "query_1",
                {"query": "q"},
                response={"results": []},
                capture={
                    "term": "火星人",
                    "query": "q",
                    "results": [],
                    "error": None,
                },
            )
            slot_path = root / "slots" / "tavily-000001-1.json"
            slot = json.loads(slot_path.read_text(encoding="utf-8"))
            slot["response"] = {"results": [{"tampered": True}]}
            slot_path.write_text(
                json.dumps(slot, ensure_ascii=False, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            slot_path.chmod(0o600)
            with self.assertRaisesRegex(FormalCheckpointError, "corrupt"):
                checkpoint.get_slot_success("tavily", 1, "query_1", {"query": "q"})
            checkpoint.close()

    def test_permission_drift_is_rejected_without_being_repaired(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "checkpoint"
            spec = make_spec()
            checkpoint = FormalLexiconCheckpoint.create(root, spec)
            checkpoint.close()
            attempts = root / "attempts"
            attempts.chmod(0o750)
            with self.assertRaisesRegex(FormalCheckpointError, "owner-only"):
                FormalLexiconCheckpoint.resume(root, spec)
            self.assertEqual(attempts.stat().st_mode & 0o777, 0o750)

    def test_secret_hash_in_frozen_frame_is_rejected_before_create(self):
        with tempfile.TemporaryDirectory() as tmp:
            secret = "tavily-private-fixture"
            root = Path(tmp) / "checkpoint"
            spec = make_spec(term=hashlib.sha256(secret.encode("utf-8")).hexdigest())
            with self.assertRaisesRegex(FormalCheckpointError, "credential value"):
                FormalLexiconCheckpoint.create(
                    root, spec, forbidden_values=(secret,)
                )
            self.assertFalse(root.exists())

    def test_threaded_reservations_are_serialized_and_forked_handle_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "checkpoint"
            spec = make_spec(tavily_cap=3, deepseek_cap=3)
            checkpoint = FormalLexiconCheckpoint.create(root, spec)
            errors = []

            def worker(provider, slot):
                try:
                    finish_success(
                        checkpoint,
                        provider,
                        slot,
                        {"request": f"{provider}:{slot}"},
                        response={"ok": True},
                        capture={"ok": True},
                    )
                except Exception as exc:  # pragma: no cover - asserted below
                    errors.append(exc)

            threads = [
                threading.Thread(target=worker, args=(provider, slot))
                for provider, slots in (
                    ("tavily", ("query_1", "query_2", "query_3")),
                    (
                        "deepseek",
                        (
                            "context_judge",
                            "web_evidence_judge",
                            "final_lexicon_judge",
                        ),
                    ),
                )
                for slot in slots
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            self.assertEqual(errors, [])
            self.assertEqual(
                checkpoint.summary()["attempt_head"]["reservation_count"], 6
            )
            with mock.patch(
                "build_lex.formal_checkpoint.os.getpid",
                return_value=os.getpid() + 1,
            ):
                with self.assertRaisesRegex(FormalCheckpointError, "fork"):
                    checkpoint.summary()
            checkpoint.close()


if __name__ == "__main__":
    unittest.main()
