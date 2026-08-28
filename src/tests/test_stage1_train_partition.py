from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from data.stage1_data import validate_data
from data.train_partition import (
    FROZEN_POLICY,
    TrainPartitionError,
    build_train_partition,
    calibration_bucket,
    load_train_partition,
    validate_train_partition,
    validate_train_partition_target,
)
from data.training_artifacts import (
    build_payload_manifest,
    canonical_sha256,
    load_json,
    load_jsonl,
    validate_payload_manifest,
    write_bytes_atomic,
    write_canonical_json,
    write_canonical_jsonl,
    write_locator_ref,
)
from tests.stage1_semantic_fixtures import (
    make_semantic_data_artifact,
    readdress_semantic_data_artifact,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _ordered_records_bytes(records: list[dict]) -> bytes:
    return (
        json.dumps(
            records,
            ensure_ascii=False,
            sort_keys=False,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )


def _reseal_data(data_ref: Path) -> None:
    readdress_semantic_data_artifact(data_ref)
    validate_data(data_ref=data_ref)


def _production_id_semantic_data(root: Path) -> Path:
    """Keep the production validator frame and use its actual frozen IDs."""

    data_ref, _ = make_semantic_data_artifact(root)
    locator = load_json(data_ref)
    target = Path(locator["target_path"])
    source_train = json.loads(
        (REPOSITORY_ROOT / "data/full/std/train.json").read_text(encoding="utf-8")
    )
    source_test = json.loads(
        (REPOSITORY_ROOT / "data/full/std/test.json").read_text(encoding="utf-8")
    )
    source_ids = {
        "train": [str(record["id"]) for record in source_train[:5781]],
        "dev": [str(record["id"]) for record in source_train[5781:]],
        "test": [str(record["id"]) for record in source_test],
    }
    for split in ("train", "dev", "test"):
        records = load_json(target / f"{split}.json")
        if len(records) != len(source_ids[split]):
            raise AssertionError("semantic data fixture split count changed")
        for record, identifier in zip(records, source_ids[split], strict=True):
            record["id"] = identifier
        write_bytes_atomic(target / f"{split}.json", _ordered_records_bytes(records))
    _reseal_data(data_ref)
    return data_ref


def _set_train_content(data_ref: Path, updates: dict[str, str]) -> None:
    target = Path(load_json(data_ref)["target_path"])
    records = load_json(target / "train.json")
    found: set[str] = set()
    for record in records:
        if record["id"] in updates:
            record["content"] = updates[record["id"]]
            found.add(record["id"])
    if found != set(updates):
        raise AssertionError(f"fixture train IDs not found: {set(updates) - found}")
    write_bytes_atomic(target / "train.json", _ordered_records_bytes(records))
    _reseal_data(data_ref)


def _reseal_partition_target(target: Path) -> None:
    write_canonical_json(target / "payload_manifest.json", build_payload_manifest(target))


class Stage1TrainPartitionTest(unittest.TestCase):
    def test_production_frame_builds_expected_current_counts_and_replays(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_ref = _production_id_semantic_data(root)
            partition_ref = root / "refs/train_partition_ref.json"
            locator = build_train_partition(
                data_ref=data_ref,
                write_ref=partition_ref,
                workspace_root=root,
            )
            report = validate_train_partition(partition_ref, workspace_root=root)

            self.assertEqual(locator["artifact_kind"], "train-partition")
            self.assertEqual(report["train_count"], 5781)
            # This is an observation of the currently frozen real ID frame, not
            # a count embedded in the validator or assignment policy.
            self.assertEqual(report["calibration_count"], 616)
            self.assertEqual(report["fit_count"], 5165)
            self.assertEqual(len(report["fit_ids"]), 5165)
            self.assertEqual(len(report["calibration_ids"]), 616)
            self.assertFalse(set(report["fit_ids"]) & set(report["calibration_ids"]))
            self.assertEqual(
                set(report["fit_ids"]) | set(report["calibration_ids"]),
                {
                    record["id"]
                    for record in load_json(
                        Path(load_json(data_ref)["target_path"]) / "train.json"
                    )
                },
            )
            compact_data_dependency = {
                key: report["data_dependency"][key]
                for key in (
                    "schema_version",
                    "artifact_kind",
                    "artifact_id",
                    "payload_manifest_sha256",
                )
            }
            target_report = validate_train_partition_target(
                locator["target_path"],
                workspace_root=root,
                expected_data_dependency=compact_data_dependency,
            )
            self.assertEqual(
                target_report["partition_dependency"],
                report["partition_dependency"],
            )
            self.assertEqual(
                set(Path(locator["target_path"]).iterdir()),
                {
                    Path(locator["target_path"]) / "partition.jsonl",
                    Path(locator["target_path"]) / "partition.meta.json",
                    Path(locator["target_path"]) / "data_ref.json",
                    Path(locator["target_path"]) / "config.resolved.json",
                    Path(locator["target_path"]) / "payload_manifest.json",
                },
            )

            second_ref = root / "refs/train_partition_ref_second.json"
            second = build_train_partition(
                data_ref=data_ref,
                write_ref=second_ref,
                workspace_root=root,
            )
            self.assertEqual(second["artifact_id"], locator["artifact_id"])
            self.assertEqual(
                second["payload_manifest_sha256"], locator["payload_manifest_sha256"]
            )

            bundle = load_train_partition(partition_ref, workspace_root=root)
            self.assertEqual(bundle.partition_dependency, report["partition_dependency"])
            self.assertEqual(len(bundle.train_records), 5781)
            self.assertEqual(len(bundle.fit_records), 5165)
            self.assertEqual(len(bundle.calibration_records), 616)
            self.assertEqual(
                [record["id"] for record in bundle.fit_records], list(bundle.fit_ids)
            )
            self.assertEqual(
                [record["id"] for record in bundle.calibration_records],
                list(bundle.calibration_ids),
            )

    def test_crlf_content_cluster_uses_minimum_numeric_id_once(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_ref = _production_id_semantic_data(root)
            assignment = FROZEN_POLICY["assignment"]
            self.assertGreaterEqual(calibration_bucket("2", assignment), 1000)
            self.assertLess(calibration_bucket("22", assignment), 1000)
            original_data_id = load_json(data_ref)["artifact_id"]
            _set_train_content(
                data_ref,
                {
                    "2": "same\r\nnormalized content",
                    "22": "same\nnormalized content",
                },
            )
            self.assertNotEqual(load_json(data_ref)["artifact_id"], original_data_id)
            partition_ref = root / "refs/train_partition_ref.json"
            locator = build_train_partition(
                data_ref=data_ref,
                write_ref=partition_ref,
                workspace_root=root,
            )
            rows = {
                row["query_id"]: row
                for row in load_jsonl(Path(locator["target_path"]) / "partition.jsonl")
            }
            expected_hash = hashlib.sha256(
                "same\nnormalized content".encode("utf-8")
            ).hexdigest()
            for query_id in ("2", "22"):
                self.assertEqual(rows[query_id]["content_sha256"], expected_hash)
                self.assertEqual(rows[query_id]["cluster_representative_query_id"], "2")
                self.assertEqual(
                    rows[query_id]["bucket"], calibration_bucket("2", assignment)
                )
                self.assertEqual(rows[query_id]["partition"], "fit")
            report = validate_train_partition(partition_ref, workspace_root=root)
            by_partition = {
                partition: {
                    row["content_sha256"]
                    for row in rows.values()
                    if row["partition"] == partition
                }
                for partition in ("fit", "calibration")
            }
            self.assertFalse(by_partition["fit"] & by_partition["calibration"])
            self.assertEqual(report["fit_count"] + report["calibration_count"], 5781)

    def test_validator_rejects_forged_cluster_bucket_and_dependency(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_ref = _production_id_semantic_data(root)
            partition_ref = root / "refs/train_partition_ref.json"
            locator = build_train_partition(
                data_ref=data_ref,
                write_ref=partition_ref,
                workspace_root=root,
            )
            target = Path(locator["target_path"])
            original = {
                name: (target / name).read_bytes()
                for name in (
                    "partition.jsonl",
                    "partition.meta.json",
                    "data_ref.json",
                    "config.resolved.json",
                    "payload_manifest.json",
                )
            }

            def restore() -> None:
                for name, payload in original.items():
                    write_bytes_atomic(target / name, payload)

            with self.subTest("forged-cluster-representative"):
                rows = load_jsonl(target / "partition.jsonl")
                rows[0]["cluster_representative_query_id"] = rows[1]["query_id"]
                write_canonical_jsonl(
                    target / "partition.jsonl", rows, key="query_id", numeric_key=True
                )
                _reseal_partition_target(target)
                with self.assertRaisesRegex(TrainPartitionError, "row replay mismatch"):
                    validate_train_partition_target(target, workspace_root=root)
                restore()

            with self.subTest("forged-bucket"):
                rows = load_jsonl(target / "partition.jsonl")
                rows[0]["bucket"] = (rows[0]["bucket"] + 1) % 10000
                write_canonical_jsonl(
                    target / "partition.jsonl", rows, key="query_id", numeric_key=True
                )
                _reseal_partition_target(target)
                with self.assertRaisesRegex(TrainPartitionError, "row replay mismatch"):
                    validate_train_partition_target(target, workspace_root=root)
                restore()

            with self.subTest("forged-data-dependency"):
                meta = load_json(target / "partition.meta.json")
                dependency = copy.deepcopy(meta["data_dependency"])
                dependency["payload_manifest_sha256"] = "f" * 64
                meta["data_dependency"] = dependency
                meta["id_inputs"]["data_dependency"] = copy.deepcopy(dependency)
                write_canonical_json(target / "partition.meta.json", meta)
                write_canonical_json(target / "data_ref.json", dependency)
                _reseal_partition_target(target)
                with self.assertRaisesRegex(
                    TrainPartitionError, "data dependency failed deep validation"
                ):
                    validate_train_partition_target(target, workspace_root=root)
                restore()

            validate_train_partition_target(target, workspace_root=root)

    def test_builder_rejects_a_non_deep_valid_data_ref(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_ref = _production_id_semantic_data(root)
            target = Path(load_json(data_ref)["target_path"])
            records = load_json(target / "train.json")
            records[0]["id"] = "0002"
            write_bytes_atomic(target / "train.json", _ordered_records_bytes(records))
            write_canonical_json(
                target / "payload_manifest.json", build_payload_manifest(target)
            )
            locator = load_json(data_ref)
            write_locator_ref(
                data_ref,
                artifact_kind="data",
                artifact_id=locator["artifact_id"],
                target=target,
                payload_manifest_sha256=validate_payload_manifest(target),
            )
            with self.assertRaisesRegex(TrainPartitionError, "deep validation"):
                build_train_partition(
                    data_ref=data_ref,
                    write_ref=root / "refs/partition.json",
                    workspace_root=root,
                )


if __name__ == "__main__":
    unittest.main()
