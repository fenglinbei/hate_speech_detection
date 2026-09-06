"""Synthetic CPU tests; no real query gold, model execution or dev scores."""

import argparse
import contextlib
import copy
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import independent_replication_audit as audit
from test_independent_coverage_audit import geometry_fixture as coverage_fixture


def fixture(label="reference", cohort="boundary"):
    suffix = dict(audit.PASSES).get(label, "r0")
    rows, contexts, identity, plan = coverage_fixture(cohort=cohort, suffix=suffix)
    plan["plan_id"] = "synthetic-plan"
    shift = int(label == "replica")
    devices = [2, 3] if shift else [0, 1]
    runtime = {"device_map": {"stage0": devices[0], "stage1": devices[1]},
               "device_map_sha256": "map-sha", "hardware": [
                   {"physical_gpu_index": device, "uuid": f"GPU-synthetic-{device}"} for device in devices]}
    identity.update(plan_id=plan["plan_id"], runtime=runtime)
    ordinal = 0
    for row in rows:
        row.update(runtime_sha256=audit.canonical_hash(runtime), scoring_profile=identity["scoring_profile"], attempt_ordinal=1)
        execution = list(row["candidates"])
        if label == "members":
            execution = list(reversed(execution if row["task"] == "hate" else execution[1:] + execution[:1]))
        for candidate in execution:
            candidate.pop("physical_gpu_index")
            candidate.update(physical_gpu_indices=devices,
                physical_gpu_uuids=[item["uuid"] for item in runtime["hardware"]],
                model_device_map_sha256=runtime["device_map_sha256"], replica_shift=shift,
                fp32_operator_dispatch_checked=True, fp32_operator_count=100,
                peak_memory_by_device={str(device): {"allocated_bytes": 20, "reserved_bytes": 30} for device in devices},
                padding_challenge_extra=identity["scoring_profile"]["padding_extra"])
            candidate["batch_ordinal"] = 0 if label == "prefix" else ordinal
            ordinal += 1
            if identity["reference"]:
                candidate.update(reference_arithmetic_dtype="cpu.torch.float64",
                    reference_token_logprobs=candidate["token_logprobs"], reference_eos_logprob=candidate["eos_logprob"],
                    reference_scores={**candidate["scores"], "token_logprobs": candidate["token_logprobs"]})
    return rows, contexts, identity, plan


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def checkpoint(path, rows, identity):
    with sqlite3.connect(path / "checkpoint.sqlite3") as connection:
        connection.execute("CREATE TABLE meta(key TEXT,value TEXT)")
        connection.execute("CREATE TABLE blocks(key TEXT,payload TEXT,sha256 TEXT)")
        connection.execute("CREATE TABLE attempts(ordinal INTEGER,invocation TEXT,records TEXT,status TEXT)")
        connection.execute("INSERT INTO meta VALUES ('identity',?)", (json.dumps(identity),))
        for index, row in enumerate(rows, 1):
            row["attempt_ordinal"] = index
            payload = json.dumps(row)
            connection.execute("INSERT INTO blocks VALUES (?,?,?)",
                (row["record_id"], payload, audit.hashlib.sha256(payload.encode()).hexdigest()))
            connection.execute("INSERT INTO attempts VALUES (?,?,?,'committed')",
                (index, "one", json.dumps([row["record_id"]])))
    save(path / "invocations.jsonl", {"invocation_id": "one", "identity_sha256": audit.canonical_hash(identity), "reused_blocks": 0})


class GeometryTests(unittest.TestCase):
    def test_all_registered_pass_profiles_and_boundary_legacy_cohort(self):
        for label, _ in audit.PASSES:
            with self.subTest(label=label):
                values = fixture(label)
                before = copy.deepcopy(values)
                proof = audit.validate_geometry(*values)
                self.assertTrue(proof["whole_layer_sharding_verified"])
                self.assertEqual(values, before)
        self.assertTrue(audit.validate_geometry(*fixture("dev", "dev"))["passed"])

    def test_candidate_attribution_precision_and_geometry_mutations_fail(self):
        mutations = {"physical_gpu_uuids": ["GPU-wrong", "GPU-wrong2"],
            "physical_gpu_indices": [0], "model_device_map_sha256": "other", "fp32_operator_count": 0,
            "fp32_operator_dispatch_checked": False, "replica_shift": 1, "batch_ordinal": 99,
            "batch_size": 4, "model_logits_dtype": "torch.bfloat16", "padding_challenge_extra": 64,
            "peak_memory_by_device": {"0": {"allocated_bytes": 40, "reserved_bytes": 20}}}
        for key, value in mutations.items():
            with self.subTest(key=key):
                rows, *rest = fixture()
                rows[0]["candidates"][0][key] = value
                with self.assertRaises(AssertionError):
                    audit.validate_geometry(rows, *rest)

    def test_global_ordinal_cannot_restart_at_next_block(self):
        rows, *rest = fixture()
        rows[1]["candidates"][0]["batch_ordinal"] = 0
        with self.assertRaises(AssertionError):
            audit.validate_geometry(rows, *rest)

    def test_old_single_gpu_evidence_is_not_accepted(self):
        rows, *rest = fixture()
        del rows[0]["candidates"][0]["physical_gpu_indices"]
        rows[0]["candidates"][0]["physical_gpu_index"] = 0
        with self.assertRaises(AssertionError):
            audit.validate_geometry(rows, *rest)

    def test_shared_prefix_corruption_rejected(self):
        rows, *rest = fixture("prefix")
        rows[0]["candidates"][1]["token_logprobs"][0] -= 1
        with self.assertRaises(AssertionError):
            audit.validate_geometry(rows, *rest)

    def test_physical_remapping_requires_new_uuids(self):
        baseline = fixture()[2]["runtime"]
        replica = fixture("replica")[2]["runtime"]
        self.assertTrue(audit.validate_remapping(baseline, replica)["passed"])
        replica["hardware"][0]["uuid"] = baseline["hardware"][0]["uuid"]
        with self.assertRaises(AssertionError):
            audit.validate_remapping(baseline, replica)


class SealAndCheckpointTests(unittest.TestCase):
    def test_unsealed_run_refuses_before_gold_or_plan_access_and_preserves_failure(self):
        for state in ("interrupted", "running", "raw_complete", "preflight_passed", "failed"):
            with self.subTest(state=state), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                terminal = root / "run_manifest.json"
                save(terminal, {"schema_version": "general-model-coverage-replication-run/v1", "status": state})
                args = argparse.Namespace(run=root, plan=root / "missing-plan", output=root / "audit",
                    allow_gold_after_seal=True, gold_file=root / "MUST_NOT_READ_GOLD")
                with mock.patch.object(audit, "read", wraps=audit.read) as reads:
                    with self.assertRaises(AssertionError):
                        audit.audit(args)
                    self.assertEqual(reads.call_args_list, [mock.call(terminal)])
                self.assertTrue((args.output / "audit_failure.json").is_file())
                with self.assertRaises(AssertionError):
                    audit.audit(args)

    def test_flat_checkpoint_matches_rows_and_preserves_interrupted_attempt(self):
        rows, _, identity, _ = fixture()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            checkpoint(path, rows, identity)
            with sqlite3.connect(path / "checkpoint.sqlite3") as connection:
                connection.execute("INSERT INTO attempts VALUES (999,'one',?,'started')", (json.dumps([rows[0]["record_id"]]),))
            proof = audit.validate_checkpoint(path, identity, rows, {})
            self.assertEqual(proof["uncommitted_attempts"], 1)
            with sqlite3.connect(path / "checkpoint.sqlite3") as connection:
                connection.execute("UPDATE blocks SET sha256='corrupt' WHERE key=?", (rows[0]["record_id"],))
            with self.assertRaises(AssertionError):
                audit.validate_checkpoint(path, identity, rows, {})

    def test_checkpoint_identity_and_attempt_misattribution_rejected(self):
        for alter in ("identity", "attempt"):
            with self.subTest(alter=alter), tempfile.TemporaryDirectory() as temporary:
                path = Path(temporary)
                rows, _, identity, _ = fixture()
                checkpoint(path, rows, identity)
                if alter == "identity":
                    identity = {**identity, "plan_id": "wrong"}
                else:
                    rows[0]["attempt_ordinal"] = 2
                with self.assertRaises(AssertionError):
                    audit.validate_checkpoint(path, identity, rows, {})


class RawOnlyAnalysisSealTests(unittest.TestCase):
    def make_run(self, root):
        run, plan_dir = root / "run", root / "plan"
        plan = {"schema_version": "general-model-coverage-replication-plan/v1", "plan_id": "synthetic-plan",
            "parent_plan_id": "synthetic-parent", "model": {"key": "qwen3-14b"}, "numeric_policy": dict(audit.POLICY),
            "frame": [{"query_id": str(index), "lex_hit": index < 223} for index in range(643)], "blocks": []}
        save(plan_dir / "plan.json", plan)
        save(root / "plan_ref.json", {"target_path": str(plan_dir)})
        save(run / "preflight/preflight_report.json", {})
        save(run / "runtime-baseline.json", {})
        save(run / "runtime-replica.json", {})
        raw = {"blocks": 10288, "candidates": 174896}
        save(run / "dev-b1/manifest.json", raw)
        analysis_file = run / "analysis/analysis.json"
        analysis_file.parent.mkdir(parents=True)
        analysis_file.write_bytes(b"opaque science bytes; must never be parsed in raw-only mode")
        manifest = {"schema_version": "general-model-coverage-replication-analysis/v1", "model_key": "qwen3-14b",
            "plan_id": plan["plan_id"], "raw_manifest_sha256": audit.digest(run / "dev-b1/manifest.json"),
            "analysis_sha256": audit.digest(analysis_file), "gold_join_after_raw_sealed": True, "test_content_read": False,
            "production_geometry": {"passed": True, "blocks": 10288, "candidates": 174896,
                "scoring_profile": audit.profile_for("dev"), "true_batch_one": True, "prefix_is_reference_only": False,
                "execution_order_verified": True, "within_batch_row_position_claimed": False}}
        save(run / "analysis/manifest.json", manifest)
        terminal = {"schema_version": "general-model-coverage-replication-run/v1", "status": "complete",
            "model_key": "qwen3-14b", "plan_id": plan["plan_id"], "execution": "model-parallel-fp32",
            "full_dev_started": True, "analysis_published": True, "raw_blocks": 10288,
            "query_gold_loaded_during_scoring": False, "test_content_read": False, "automatic_profile_search": False,
            "preflight_report_sha256": audit.digest(run / "preflight/preflight_report.json"),
            "raw_manifest_sha256": audit.digest(run / "dev-b1/manifest.json"),
            "analysis_manifest_sha256": audit.digest(run / "analysis/manifest.json")}
        save(run / "run_manifest.json", terminal)
        gold_file = root / "synthetic-gold.jsonl"
        gold_file.write_bytes(b"gold must not be read")
        args = argparse.Namespace(run=run, plan=root / "plan_ref.json", output=root / "audit",
            allow_gold_after_seal=False, gold_file=gold_file)
        return args, plan, raw

    @contextlib.contextmanager
    def mocked_raw_verification(self, args, plan, raw):
        original_read_text = Path.read_text
        forbidden = {args.gold_file, args.run / "analysis/analysis.json"}

        def guarded_read_text(path, *positional, **keywords):
            self.assertNotIn(path, forbidden, "raw-only mode deserialized gold/science")
            return original_read_text(path, *positional, **keywords)

        with mock.patch("diagnostics.general_model_coverage_replication.load_plan", return_value=(plan, [])), \
             mock.patch.object(audit, "independent_preflight", return_value={"passed": True}), \
             mock.patch.object(audit, "validate_pass", return_value=([], raw, {}, {})), \
             mock.patch.object(audit.math_audit, "_raw_validation", return_value=({}, {}, [], [], {"candidates": 174896})), \
             mock.patch.object(audit.math_audit, "verify_analysis", side_effect=AssertionError("science math must not run")), \
             mock.patch.object(Path, "read_text", guarded_read_text):
            yield

    def test_valid_raw_only_binds_analysis_hashes_without_science_or_gold_deserialization(self):
        with tempfile.TemporaryDirectory() as temporary:
            args, plan, raw = self.make_run(Path(temporary))
            with self.mocked_raw_verification(args, plan, raw):
                receipt = audit.audit(args)
            self.assertTrue(receipt["audit_passed"])
            self.assertFalse(receipt["query_gold_read"])
            self.assertFalse(receipt["scientific_tables_written"])
            self.assertFalse(receipt["all_ci_verified"])
            for name in ("manifest.json", "analysis.json"):
                self.assertEqual(receipt["source_hashes"][str(args.run / "analysis" / name)],
                                 audit.digest(args.run / "analysis" / name))
            self.assertNotIn(str(args.gold_file), receipt["source_hashes"])

    def test_default_raw_only_rejects_missing_or_tampered_analysis(self):
        for mutation in ("missing-manifest", "missing-science", "manifest-sha", "science-sha", "plan-id", "raw-binding"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temporary:
                args, plan, raw = self.make_run(Path(temporary))
                manifest_path = args.run / "analysis/manifest.json"
                science_path = args.run / "analysis/analysis.json"
                if mutation.startswith("missing"):
                    (manifest_path if mutation == "missing-manifest" else science_path).unlink()
                elif mutation == "science-sha":
                    science_path.write_bytes(b"changed science")
                else:
                    manifest = audit.read(manifest_path)
                    manifest["plan_id" if mutation == "plan-id" else "raw_manifest_sha256"] = "changed"
                    save(manifest_path, manifest)
                    if mutation != "manifest-sha":
                        terminal_path = args.run / "run_manifest.json"
                        terminal = audit.read(terminal_path)
                        terminal["analysis_manifest_sha256"] = audit.digest(manifest_path)
                        save(terminal_path, terminal)
                with self.mocked_raw_verification(args, plan, raw):
                    with self.assertRaises((AssertionError, FileNotFoundError)):
                        audit.audit(args)
                self.assertFalse((args.output / "audit.json").exists())
                self.assertFalse(audit.read(args.output / "audit_failure.json")["audit_passed"])


class PreflightInventoryTests(unittest.TestCase):
    def test_exact_18_check_inventory_limits_and_hashes(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            directory = run / "preflight"
            directory.mkdir()
            cohorts = {name: [f"{name}-{i}" for i in range(n)] for name, n in
                       (("regression", 8), ("validation", 24), ("boundary", 4))}
            plan = {"plan_id": "synthetic-plan", "cohorts": cohorts, "blocks": []}
            runtimes = [{"fake": 0}, {"fake": 1}]
            for runtime, name in zip(runtimes, ("runtime-baseline.json", "runtime-replica.json")):
                save(run / name, runtime)
            report = {"schema_version": "general-model-coverage-replication-preflight/v1", "plan_id": plan["plan_id"],
                "numeric_policy": dict(audit.POLICY), "passed": True, "complete": True,
                "query_gold_loaded": False, "test_content_read": False, "scientific_effect_checked": False,
                "placement_challenge": "same-layer-partition-different-physical-GPUs",
                "inherited_E8_is_not_new_model_calibration": True,
                "runtime_sha256": {str(i): audit.digest(run / name) for i, name in enumerate(
                    ("runtime-baseline.json", "runtime-replica.json"))}, "checks": {}, "files": {}}
            for cohort in audit.COHORTS:
                for label, suffix in audit.PASSES:
                    key = f"{cohort}-{label}"
                    limit = .0001 if label in ("reference", "repeat") else audit.POLICY["epsilon"]
                    report["checks"][key] = {"max_abs_error": 0., "limit": limit, "passed": True}
                    files = {f"{cohort}-b1-{suffix}/manifest.json": {}, f"{cohort}-b1-{suffix}/scores.jsonl": {},
                        key + "-differences.json": {}, key + "-geometry-proof.json": {
                            "passed": True, "blocks": 1, "candidates": 2, "scoring_profile": audit.profile_for(label),
                            "true_batch_one": True, "prefix_is_reference_only": label == "prefix",
                            "execution_order_verified": label != "prefix", "within_batch_row_position_claimed": False}}
                    for name, value in files.items():
                        save(directory / name, value)
                        report["files"][name] = audit.digest(directory / name)
            with mock.patch.object(audit, "validate_runtime"), mock.patch.object(audit, "validate_remapping", return_value={"passed": True}), \
                 mock.patch.object(audit, "validate_pass", return_value=([{}], {}, {"candidates": 2}, {})), \
                 mock.patch.object(audit.math_audit, "compare", return_value={"max_abs_error": 0., "stored_difference_file_verified": True}):
                self.assertEqual(audit.independent_preflight(plan, run, report, runtimes, {})["sealed_pass_count"], 18)
                bad = copy.deepcopy(report)
                del bad["checks"]["boundary-replica"]
                with self.assertRaises(AssertionError):
                    audit.independent_preflight(plan, run, bad, runtimes, {})
                bad = copy.deepcopy(report)
                bad["checks"]["regression-reference"]["limit"] *= 2
                with self.assertRaises(AssertionError):
                    audit.independent_preflight(plan, run, bad, runtimes, {})
                bad = copy.deepcopy(report)
                bad["files"]["regression-reference-differences.json"] = "wrong"
                with self.assertRaises(AssertionError):
                    audit.independent_preflight(plan, run, bad, runtimes, {})


if __name__ == "__main__":
    unittest.main()
