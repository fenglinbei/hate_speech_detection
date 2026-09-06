import copy
import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from data.stage1_data import canonical_json_sha256, sha256_file
from diagnostics import general_model_numeric_v2 as numeric
from diagnostics.general_model_numeric_pool import (
    PersistentNumericPool, assignment_descriptor, hardware_identity, merge_shards,
    partition_groups, pass_identity, validate_sealed_pass,
)
from diagnostics.general_model_package import PackageError, read_json, read_jsonl, write_json, write_jsonl
from tests.test_general_model_numeric import catalog_fixture, context, plan_fixture, score_fixture


def runtime_identity(devices=(0, 1, 2, 3)):
    return {"execution": "data-parallel-identical-fp32", "numeric_runtime": {"dtype": "float32"},
            "device_indices": list(devices),
            "replicas": [{"physical_gpu_index": index, "actual_numeric_identity": {"dtype": "float32"},
                          "hardware": {"uuid": f"GPU-fixture-{index}"}} for index in devices]}


def frame():
    return [context(str(index), task=task, condition=condition)
            for task in ("hate", "group") for condition in ("C0", "CL") for index in range(1, 6)]


class PartitionTests(unittest.TestCase):
    def test_shift_moves_every_atomic_group_without_changing_members_or_local_order(self):
        contexts, catalog = frame(), catalog_fixture()
        baseline = partition_groups(contexts, catalog, 4, [0, 1, 2, 3])
        shifted = partition_groups(contexts, catalog, 4, [0, 1, 2, 3], 1)
        def owners(assignments):
            return {ordinal: (assignment["physical_gpu_index"], group, local)
                    for assignment in assignments
                    for local, (ordinal, group) in enumerate(zip(assignment["group_ordinals"], assignment["groups"]))}
        original, changed = owners(baseline), owners(shifted)
        self.assertEqual(original.keys(), changed.keys())
        for key in original:
            gpu, members, local = original[key]
            moved_gpu, moved_members, moved_local = changed[key]
            self.assertEqual(moved_gpu, (gpu + 1) % 4)
            self.assertEqual(moved_members, members)
            self.assertEqual(moved_local, local)
        self.assertEqual(sum(len(row["contexts"]) for row in baseline), len(contexts))
        self.assertTrue(all(len(group) in (1, 2) for row in baseline for group in row["groups"]))

    def test_arbitrary_physical_indices_use_order_not_modulo_gpu_number(self):
        assignments = partition_groups(frame(), catalog_fixture(), 4, [7, 2, 9, 4], 1)
        first = next(row for row in assignments if 0 in row["group_ordinals"])
        self.assertEqual(first["physical_gpu_index"], 2)

    def test_duplicate_records_invalid_shift_and_device_aliases_are_rejected(self):
        for contexts, devices, shift in (([context(), context()], [0, 1], 0),
                                        ([context()], [0, 0], 0), ([context()], [0, 1], 2),
                                        ([context()], [], 0)):
            with self.assertRaises(PackageError):
                partition_groups(contexts, catalog_fixture(), 4, devices, shift)


class MergeTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.plan = plan_fixture()
        self.contexts = frame()
        self.plan["blocks"] = self.contexts
        self.runtime = runtime_identity()

    def shards(self, *, shift=0, name="regression-b4-r0"):
        output = self.root / name
        output.mkdir()
        assignments = partition_groups(self.contexts, self.plan["catalog"], 4, [0, 1, 2, 3], shift)
        profile = numeric.scoring_profile(replica_shift=shift)
        identity = pass_identity(self.plan, self.runtime, [row["record_id"] for row in self.contexts],
                                 name, 4, False, profile)
        identity["data_parallel_assignment_sha256"] = canonical_json_sha256(assignment_descriptor(assignments))
        write_json(output / "pool_binding.json", {"identity": identity, "assignments": assignment_descriptor(assignments)})
        completed = []
        for assignment in assignments:
            index = assignment["physical_gpu_index"]
            path = output / "shards" / str(index) / name
            runner = SimpleNamespace(identity=self.runtime, padding_extra=0)
            def scorer(instance, items, *, reference=False, physical=index):
                return [{**score_fixture(item["candidate"], reference=reference),
                         "physical_gpu_index": physical, "physical_gpu_uuid": f"GPU-fixture-{physical}"}
                        for item in items]
            with redirect_stdout(io.StringIO()):
                _, receipt = numeric.score_pass(runner, assignment["contexts"], self.plan, path,
                                               batch_size=4, scorer=scorer, replica_shift=shift)
            completed.append({"physical_gpu_index": index, "receipt": receipt,
                              "manifest_sha256": sha256_file(path / "manifest.json"),
                              "scores_sha256": sha256_file(path / "scores.jsonl")})
        return output, identity, assignments, completed

    def merge(self, data):
        output, identity, assignments, completed = data
        return merge_shards(self.contexts, self.plan, output, identity, assignments, completed)

    def test_exact_canonical_merge_can_be_verified_again_without_overwrite(self):
        data = self.shards()
        rows, receipt = self.merge(data)
        self.assertEqual([row["record_id"] for row in rows], [row["record_id"] for row in self.contexts])
        self.assertEqual(receipt["blocks"], len(self.contexts))
        self.assertEqual(len(receipt["shards"]), 4)
        self.assertEqual(receipt["identity"]["runtime"], self.runtime)
        self.assertFalse(receipt["query_gold_loaded"])
        self.assertEqual(self.merge(data), (rows, receipt))

    def test_replica_shift_changes_producer_but_not_recorded_candidate_geometry(self):
        baseline, _ = self.merge(self.shards())
        shifted, _ = self.merge(self.shards(shift=1, name="regression-b4-replica"))
        for original, changed in zip(baseline, shifted):
            for a, b in zip(original["candidates"], changed["candidates"]):
                self.assertNotEqual(a["physical_gpu_index"], b["physical_gpu_index"])
                self.assertNotEqual(a["physical_gpu_uuid"], b["physical_gpu_uuid"])
                for key in ("candidate_id", "batch_members", "batch_member_ordinal", "batch_ordinal", "effective_batch_size"):
                    self.assertEqual(a[key], b[key])

    def test_missing_and_duplicate_completion_prevent_top_seal(self):
        data = self.shards()
        for replies in (data[3][:-1], data[3] + [data[3][0]]):
            with self.assertRaisesRegex(PackageError, "coverage"):
                self.merge((*data[:3], replies))
        self.assertFalse((data[0] / "manifest.json").exists())

    def test_stale_completion_hash_is_rejected(self):
        data = self.shards()
        data[3][0]["manifest_sha256"] = "stale"
        with self.assertRaisesRegex(PackageError, "hash"):
            self.merge(data)

    def test_shard_payload_tampering_is_rejected(self):
        data = self.shards()
        path = data[0] / "shards/0" / data[0].name / "scores.jsonl"
        path.write_bytes(path.read_bytes() + b"\n")
        with self.assertRaisesRegex(PackageError, "hash"):
            self.merge(data)

    def test_wrong_physical_producer_cannot_be_hidden_by_refreshing_hashes(self):
        data = self.shards()
        directory = data[0] / "shards/0" / data[0].name
        rows = read_jsonl(directory / "scores.jsonl")
        rows[0]["candidates"][0]["physical_gpu_uuid"] = "GPU-wrong"
        write_jsonl(directory / "scores.jsonl", rows)
        receipt = read_json(directory / "manifest.json")
        receipt["scores_sha256"] = sha256_file(directory / "scores.jsonl")
        write_json(directory / "manifest.json", receipt)
        data[3][0].update(receipt=receipt, manifest_sha256=sha256_file(directory / "manifest.json"),
                          scores_sha256=receipt["scores_sha256"])
        with self.assertRaisesRegex(PackageError, "physical assignment"):
            self.merge(data)

    def test_existing_top_scores_and_manifest_are_immutable(self):
        data = self.shards()
        self.merge(data)
        path = data[0] / "manifest.json"
        receipt = read_json(path)
        receipt["blocks"] -= 1
        write_json(path, receipt)
        with self.assertRaisesRegex(PackageError, "sealed data-parallel manifest"):
            self.merge(data)

    def test_readonly_terminal_validator_rechecks_shards_and_assignment(self):
        data = self.shards()
        _, receipt = self.merge(data)
        self.assertEqual(validate_sealed_pass(data[0], self.plan), receipt)
        shard_manifest = data[0] / "shards/0" / data[0].name / "manifest.json"
        stale = read_json(shard_manifest)
        stale["status"] = "partial"
        write_json(shard_manifest, stale)
        with self.assertRaisesRegex(PackageError, "hash"):
            validate_sealed_pass(data[0], self.plan)


class PoolLifecycleTests(unittest.TestCase):
    def bare_pool(self):
        pool = PersistentNumericPool.__new__(PersistentNumericPool)
        pool._closed = False
        pool._workers = []
        return pool

    def test_fake_worker_stale_job_and_wrong_device_replies_fail(self):
        for result in ({"kind": "complete", "job_id": 4, "physical_gpu_index": 0},
                       {"kind": "complete", "job_id": 5, "physical_gpu_index": 1},
                       {"kind": "error", "error": "fixture"}):
            pool = self.bare_pool()
            connection = MagicMock()
            connection.recv.return_value = result
            worker = {"physical_gpu_index": 0, "connection": connection,
                      "process": SimpleNamespace(exitcode=None)}
            with patch("diagnostics.general_model_numeric_pool.wait", return_value=[connection]):
                with self.assertRaises(PackageError):
                    pool._collect([worker], "complete", job_id=5)

    def test_normal_close_sends_stop_and_joins_each_owned_worker_once(self):
        pool = self.bare_pool()
        for index in range(4):
            process, connection = MagicMock(), MagicMock()
            process.is_alive.return_value = False
            pool._workers.append({"physical_gpu_index": index, "connection": connection, "process": process})
        pool.close()
        pool.close()
        for worker in pool._workers:
            worker["connection"].send.assert_called_once_with({"kind": "stop"})
            worker["connection"].close.assert_called_once()
            worker["process"].terminate.assert_not_called()
            self.assertGreaterEqual(worker["process"].join.call_count, 1)

    def test_exception_close_terminates_and_joins_owned_workers(self):
        pool = self.bare_pool()
        process, connection = MagicMock(), MagicMock()
        process.is_alive.side_effect = [True, False, False]
        pool._workers = [{"physical_gpu_index": 0, "process": process, "connection": connection}]
        pool.close(terminate=True)
        process.terminate.assert_called_once()
        connection.send.assert_not_called()
        connection.close.assert_called_once()
        self.assertGreaterEqual(process.join.call_count, 1)

    def test_hardware_verification_accepts_bare_uuid_and_virtualized_zero_capacity(self):
        properties = SimpleNamespace(uuid="1234-abcd", name="NVIDIA L20", total_memory=0, major=8, minor=9)
        torch = SimpleNamespace(cuda=SimpleNamespace(get_device_properties=lambda index: properties,
                                                    device_count=lambda: 1))
        response = SimpleNamespace(stdout="1, GPU-1234-abcd, NVIDIA L20, 46068\n")
        with patch("diagnostics.general_model_numeric_pool.subprocess.run", return_value=response), \
                patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": "1"}):
            hardware = hardware_identity(torch, 1)
            self.assertEqual(hardware["uuid"], "GPU-1234-abcd")
            self.assertEqual(hardware["total_memory_bytes"], 0)
            self.assertEqual(hardware["nvidia_smi_total_memory_mib"], 46068)
            properties.uuid = "different"
            with self.assertRaisesRegex(PackageError, "UUID"):
                hardware_identity(torch, 1)


if __name__ == "__main__":
    unittest.main()
