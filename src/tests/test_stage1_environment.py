from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import yaml

from scripts.stage1 import capture_environment as environment


class FakeDistribution:
    def __init__(self, name: str, version: str, direct_url: dict | None = None):
        self.metadata = {"Name": name}
        self.version = version
        self.direct_url = direct_url

    def read_text(self, filename: str) -> str | None:
        if filename != "direct_url.json" or self.direct_url is None:
            return None
        return json.dumps(self.direct_url)


def package_snapshot(version: str = "1.0") -> list[dict[str, str]]:
    return [
        {"installer": "python", "name": name, "version": version}
        for name in sorted(environment.CRITICAL_DISTRIBUTIONS)
    ]


def conda_snapshot(build: str = "h123_0") -> list[dict]:
    return [
        {
            "installer": "conda",
            "name": "python",
            "version": "3.11.15",
            "build": build,
            "build_number": 0,
            "subdir": "linux-64",
            "package_sha256_or_null": "4" * 64,
            "package_md5_or_null": None,
        }
    ]


def environment_document(version: str = "1.0", conda_build: str = "h123_0") -> dict:
    return environment.build_environment_document(
        environment_spec_sha256="1" * 64,
        installed_distributions=environment.merge_distribution_snapshots(
            package_snapshot(version),
            conda_snapshot(conda_build),
        ),
        torch_build={
            "version": "2.6.0+cu124",
            "cuda_version": "12.4",
            "git_version": "2" * 40,
            "cudnn_version": 91002,
            "debug_build": False,
        },
        driver_version="580.105.08",
        gpu_architecture=[
            {
                "name": "NVIDIA L20",
                "compute_capability": "8.9",
                "memory_total_mib": 46068,
                "count": 4,
            }
        ],
        capture_code_sha256="3" * 64,
        python_implementation="CPython",
        python_version="3.11.15",
    )


class EnvironmentIdentityTests(unittest.TestCase):
    def test_conda_snapshot_keeps_build_checksum_but_drops_channel_locator(self):
        with tempfile.TemporaryDirectory() as temporary_root:
            prefix = Path(temporary_root)
            conda_meta = prefix / "conda-meta"
            conda_meta.mkdir()
            (conda_meta / "cuda-nvcc-12.8.93-0.json").write_text(
                json.dumps(
                    {
                        "name": "cuda-nvcc",
                        "version": "12.8.93",
                        "build": "0",
                        "build_number": 0,
                        "subdir": "linux-64",
                        "sha256": "5" * 64,
                        "md5": None,
                        "channel": "https://user:secret@example.invalid/private",
                        "url": "https://user:secret@example.invalid/private/cuda-nvcc.conda",
                    }
                ),
                encoding="utf-8",
            )

            snapshot = environment.collect_conda_distributions(prefix)

        self.assertEqual(snapshot[0]["installer"], "conda")
        self.assertEqual(snapshot[0]["build"], "0")
        self.assertEqual(snapshot[0]["package_sha256_or_null"], "5" * 64)
        wire = json.dumps(snapshot, sort_keys=True)
        self.assertNotIn("channel", wire)
        self.assertNotIn("url", wire)
        self.assertNotIn("secret", wire)

    def test_package_snapshot_normalizes_sorts_and_retains_vcs_commit(self):
        snapshot = environment.collect_installed_distributions(
            [
                FakeDistribution("Z_pkg", "2"),
                FakeDistribution(
                    "A.Pkg",
                    "1",
                    {
                        "url": "https://github.com/example/project.git",
                        "vcs_info": {"vcs": "git", "commit_id": "a" * 40},
                    },
                ),
            ]
        )

        self.assertEqual([entry["name"] for entry in snapshot], ["a-pkg", "z-pkg"])
        self.assertEqual({entry["installer"] for entry in snapshot}, {"python"})
        self.assertEqual(snapshot[0]["direct_url"]["vcs_info"]["commit_id"], "a" * 40)

        with self.assertRaises(environment.EnvironmentCaptureError):
            environment.collect_installed_distributions(
                [FakeDistribution("local", "1", {"url": "file:///private/build"})]
            )

        conda_backed = environment.collect_installed_distributions(
            [FakeDistribution("local", "1", {"url": "file:///private/build"})],
            conda_identities={("local", "1")},
        )
        self.assertNotIn("direct_url", conda_backed[0])

    def test_gpu_probe_is_aggregated_without_runtime_identity(self):
        driver, architecture = environment.parse_nvidia_smi(
            "\n".join(
                ["NVIDIA L20, 8.9, 46068, 580.105.08"] * 4
            )
        )

        self.assertEqual(driver, "580.105.08")
        self.assertEqual(
            architecture,
            [
                {
                    "name": "NVIDIA L20",
                    "compute_capability": "8.9",
                    "memory_total_mib": 46068,
                    "count": 4,
                }
            ],
        )
        self.assertNotIn("uuid", json.dumps(architecture).lower())

    def test_build_id_is_reproducible_and_sensitive_to_package_versions(self):
        first = environment_document()
        repeated = environment_document()
        changed = environment_document("1.1")
        changed_conda_build = environment_document(conda_build="h999_1")

        self.assertEqual(first, repeated)
        self.assertEqual(
            first["environment_build_id"],
            environment.recompute_environment_build_id(first),
        )
        self.assertNotEqual(first["environment_build_id"], changed["environment_build_id"])
        self.assertNotEqual(
            first["environment_build_id"],
            changed_conda_build["environment_build_id"],
        )

    def test_unsorted_packages_and_mutable_container_tags_are_rejected(self):
        with self.assertRaises(environment.EnvironmentCaptureError):
            environment.build_environment_document(
                environment_spec_sha256="1" * 64,
                installed_distributions=list(reversed(package_snapshot())),
                torch_build={"cuda_version": "12.8"},
                driver_version="580.105.08",
                gpu_architecture=[{"name": "NVIDIA L20"}],
                capture_code_sha256="3" * 64,
            )

        with self.assertRaises(environment.EnvironmentCaptureError):
            environment.build_environment_document(
                environment_spec_sha256="1" * 64,
                installed_distributions=package_snapshot(),
                torch_build={"cuda_version": "12.8"},
                driver_version="580.105.08",
                gpu_architecture=[{"name": "NVIDIA L20"}],
                capture_code_sha256="3" * 64,
                container_image_digest="stage1:latest",
            )


class EnvironmentMaterializationTests(unittest.TestCase):
    def test_phase1_requirements_match_final_yaml_except_flash_attention(self):
        repo_root = Path(__file__).resolve().parents[2]
        spec = yaml.safe_load((repo_root / "environment/stage1-p0.yml").read_text(encoding="utf-8"))
        pip_requirements = spec["dependencies"][-1]["pip"]
        phase1_requirements = [
            line.strip()
            for line in (repo_root / "environment/stage1-p0-phase1.txt").read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.startswith("#")
        ]

        without_flash = [item for item in pip_requirements if not item.startswith("flash-attn @")]
        self.assertEqual(without_flash, phase1_requirements)
        flash = [item for item in pip_requirements if item.startswith("flash-attn @")]
        self.assertEqual(len(flash), 1)
        self.assertIn("torch2.6cxx11abiFALSE-cp311", flash[0])
        self.assertIn("#sha256=58853b28", flash[0])


class EnvironmentArtifactTests(unittest.TestCase):
    @staticmethod
    def write_target(target: Path, document: dict, provenance: dict | None = None) -> None:
        target.mkdir()
        environment._write_canonical_json(target / "environment.json", document)
        environment._write_canonical_json(
            target / "provenance.json",
            provenance
            or {
                "schema_version": "stage1-environment-provenance/v1",
                "capture_policy_version": environment.CAPTURE_POLICY_VERSION,
            },
        )
        environment._write_canonical_json(
            target / "payload_manifest.json",
            environment.build_payload_manifest(target),
        )

    def test_atomic_staging_directory_can_be_validated_before_rename(self):
        document = environment_document()
        with tempfile.TemporaryDirectory() as temporary_root:
            staging = Path(temporary_root) / ".staging"
            self.write_target(staging, document)

            self.assertEqual(
                environment.validate_target(staging, require_directory_name=False)[
                    "environment_build_id"
                ],
                document["environment_build_id"],
            )
            with self.assertRaises(environment.EnvironmentCaptureError):
                environment.validate_target(staging)

    def test_capture_atomically_publishes_target_and_locator(self):
        with tempfile.TemporaryDirectory() as temporary_root:
            root = Path(temporary_root)
            spec = root / "environment.yml"
            spec.write_text("name: stage1-p0\n", encoding="utf-8")
            target_root = root / "targets"
            locator_path = root / "environment_ref.json"
            torch_build = {
                "version": "2.6.0+cu124",
                "cuda_version": "12.4",
                "git_version": "2" * 40,
                "cudnn_version": 91002,
                "debug_build": False,
            }

            with (
                mock.patch.object(
                    environment,
                    "collect_installed_distributions",
                    return_value=package_snapshot(),
                ),
                mock.patch.object(environment, "collect_torch_build", return_value=torch_build),
                mock.patch.object(
                    environment,
                    "collect_conda_distributions",
                    return_value=conda_snapshot(),
                ),
                mock.patch.object(
                    environment,
                    "_run_nvidia_smi",
                    return_value="NVIDIA L20, 8.9, 46068, 580.105.08\n" * 4,
                ),
            ):
                locator = environment.capture(
                    environment_spec=spec,
                    target_root=target_root,
                    write_ref=locator_path,
                    expected_prefix=Path(environment.sys.prefix),
                )

            target = Path(locator["target_path"])
            self.assertEqual(target.name, locator["artifact_id"])
            self.assertTrue(locator_path.is_file())
            self.assertFalse(any(path.name.startswith(".env-") for path in target_root.iterdir()))
            self.assertEqual(
                environment.validate_ref(locator_path)["environment_build_id"],
                locator["artifact_id"],
            )

    def test_target_validates_then_detects_payload_tampering(self):
        document = environment_document()
        with tempfile.TemporaryDirectory() as temporary_root:
            target = Path(temporary_root) / document["environment_build_id"]
            self.write_target(target, document)

            self.assertEqual(
                environment.validate_target(target)["environment_build_id"],
                document["environment_build_id"],
            )

            with (target / "provenance.json").open("a", encoding="utf-8") as handle:
                handle.write(" ")
            with self.assertRaises(environment.EnvironmentCaptureError):
                environment.validate_target(target)

    def test_nested_nonportable_provenance_key_is_rejected(self):
        document = environment_document()
        with tempfile.TemporaryDirectory() as temporary_root:
            target = Path(temporary_root) / document["environment_build_id"]
            self.write_target(
                target,
                document,
                {"schema_version": "stage1-environment-provenance/v1", "runtime": {"hostname": "x"}},
            )

            with self.assertRaises(environment.EnvironmentCaptureError):
                environment.validate_target(target)


if __name__ == "__main__":
    unittest.main()
