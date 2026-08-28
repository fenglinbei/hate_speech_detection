#!/usr/bin/env python3
"""Capture and validate the immutable Stage 1 training environment.

The artifact deliberately excludes hostnames, GPU UUIDs, timestamps, absolute
environment prefixes, and arbitrary environment variables.  It is intended to
run from the freshly materialized ``.conda/stage1-p0`` environment.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping
from urllib.parse import urlsplit


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SPEC = REPO_ROOT / "environment" / "stage1-p0.yml"
DEFAULT_PREFIX = REPO_ROOT / ".conda" / "stage1-p0"
DEFAULT_TARGET_ROOT = REPO_ROOT / "exps" / "causal_context" / "stage1_p0" / "environments"
DEFAULT_SCHEMA = REPO_ROOT / "schemas" / "stage1_environment_manifest_v1.schema.json"

SCHEMA_VERSION = "stage1-environment/v1"
CAPTURE_POLICY_VERSION = "stage1-environment-capture/v1"
PAYLOAD_MANIFEST_VERSION = "stage1-payload-manifest/v1"
LOCATOR_REF_VERSION = "stage1-locator-ref/v1"
ARTIFACT_KIND = "stage1-environment"
ENVIRONMENT_ID_RE = re.compile(r"^env-[0-9a-f]{64}$")
OCI_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
NORMALIZED_NAME_RE = re.compile(r"[-_.]+")
CONDA_NAME_RE = re.compile(r"^[a-z0-9_][a-z0-9_.-]*$")
HEX_32_RE = re.compile(r"^[0-9a-f]{32}$")
HEX_64_RE = re.compile(r"^[0-9a-f]{64}$")

CRITICAL_DISTRIBUTIONS = (
    "accelerate",
    "datasets",
    "deepspeed",
    "flash-attn",
    "numpy",
    "pandas",
    "scikit-learn",
    "torch",
    "transformers",
    "vllm",
)

ID_INPUT_KEYS = (
    "schema_version",
    "python_implementation_version",
    "sorted_installed_distributions",
    "torch_build",
    "cuda_runtime_driver",
    "gpu_architecture",
    "container_image_digest_or_null",
    "conda_explicit_lock_sha256",
    "critical_backend_versions",
    "environment_spec_sha256",
    "capture_policy_version",
    "capture_code_sha256",
)


class EnvironmentCaptureError(RuntimeError):
    """Raised when a portable environment artifact cannot be produced."""


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_distribution_name(name: str) -> str:
    return NORMALIZED_NAME_RE.sub("-", name).lower()


def _validated_direct_url(
    raw: str | None,
    distribution_name: str,
    *,
    allow_conda_local_build_url: bool = False,
) -> Any | None:
    if not raw:
        return None
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise EnvironmentCaptureError(
            f"invalid direct_url.json for distribution {distribution_name}"
        ) from exc
    if not isinstance(value, dict) or not isinstance(value.get("url"), str):
        raise EnvironmentCaptureError(
            f"invalid direct URL metadata for distribution {distribution_name}"
        )
    parsed = urlsplit(value["url"])
    if parsed.scheme == "file" and allow_conda_local_build_url:
        return None
    if parsed.scheme in {"", "file"} or parsed.username or parsed.password or parsed.query:
        raise EnvironmentCaptureError(
            f"non-portable or credential-bearing direct URL for distribution {distribution_name}"
        )
    allowed_top_level = {"url", "archive_info", "dir_info", "vcs_info", "subdirectory"}
    if set(value) - allowed_top_level:
        raise EnvironmentCaptureError(
            f"unsupported direct URL metadata for distribution {distribution_name}"
        )
    info_keys = [key for key in ("archive_info", "dir_info", "vcs_info") if key in value]
    if len(info_keys) > 1:
        raise EnvironmentCaptureError(
            f"ambiguous direct URL metadata for distribution {distribution_name}"
        )
    subdirectory = value.get("subdirectory")
    if subdirectory is not None and (
        not isinstance(subdirectory, str)
        or not subdirectory
        or subdirectory.startswith(("/", "\\"))
        or ".." in Path(subdirectory).parts
    ):
        raise EnvironmentCaptureError(f"invalid direct URL subdirectory for distribution {distribution_name}")
    if "archive_info" in value:
        archive_info = value["archive_info"]
        if not isinstance(archive_info, dict) or set(archive_info) - {"hash", "hashes"}:
            raise EnvironmentCaptureError(f"invalid archive_info for distribution {distribution_name}")
        if "hash" in archive_info and not isinstance(archive_info["hash"], str):
            raise EnvironmentCaptureError(f"invalid archive hash for distribution {distribution_name}")
        hashes = archive_info.get("hashes", {})
        if not isinstance(hashes, dict) or not all(
            isinstance(key, str) and isinstance(item, str) for key, item in hashes.items()
        ):
            raise EnvironmentCaptureError(f"invalid archive hashes for distribution {distribution_name}")
    if "vcs_info" in value:
        vcs_info = value["vcs_info"]
        if not isinstance(vcs_info, dict) or set(vcs_info) - {"vcs", "commit_id", "requested_revision"}:
            raise EnvironmentCaptureError(f"invalid vcs_info for distribution {distribution_name}")
        if not all(isinstance(vcs_info.get(key), str) and vcs_info[key] for key in ("vcs", "commit_id")):
            raise EnvironmentCaptureError(f"incomplete VCS identity for distribution {distribution_name}")
        if "requested_revision" in vcs_info and not isinstance(vcs_info["requested_revision"], str):
            raise EnvironmentCaptureError(f"invalid VCS revision for distribution {distribution_name}")
    if "dir_info" in value:
        dir_info = value["dir_info"]
        if not isinstance(dir_info, dict) or set(dir_info) - {"editable"}:
            raise EnvironmentCaptureError(f"invalid dir_info for distribution {distribution_name}")
    return value


def collect_installed_distributions(
    distributions: Iterable[importlib.metadata.Distribution] | None = None,
    *,
    conda_identities: set[tuple[str, str]] | None = None,
) -> list[dict[str, Any]]:
    by_name: dict[str, dict[str, Any]] = {}
    for distribution in distributions or importlib.metadata.distributions():
        raw_name = distribution.metadata.get("Name")
        if not raw_name:
            raise EnvironmentCaptureError("installed distribution is missing its Name metadata")
        name = normalize_distribution_name(str(raw_name))
        entry: dict[str, Any] = {
            "installer": "python",
            "name": name,
            "version": str(distribution.version),
        }
        direct_url = _validated_direct_url(
            distribution.read_text("direct_url.json"),
            name,
            allow_conda_local_build_url=(name, str(distribution.version)) in (conda_identities or set()),
        )
        if direct_url is not None:
            entry["direct_url"] = direct_url
        previous = by_name.get(name)
        if previous is not None and previous != entry:
            raise EnvironmentCaptureError(f"conflicting installed distributions for normalized name {name}")
        by_name[name] = entry
    return [by_name[name] for name in sorted(by_name)]


def collect_conda_distributions(prefix: Path) -> list[dict[str, Any]]:
    conda_meta = prefix / "conda-meta"
    if not conda_meta.is_dir():
        raise EnvironmentCaptureError(f"expected a Conda prefix with conda-meta at {prefix}")
    by_name: dict[str, dict[str, Any]] = {}
    for record_path in sorted(conda_meta.glob("*.json")):
        try:
            record = json.loads(record_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise EnvironmentCaptureError(f"invalid Conda package record: {record_path.name}") from exc
        raw_name = record.get("name")
        version = record.get("version")
        build = record.get("build")
        build_number = record.get("build_number")
        subdir = record.get("subdir")
        if not isinstance(raw_name, str) or not isinstance(version, str) or not isinstance(build, str):
            raise EnvironmentCaptureError(f"incomplete Conda package record: {record_path.name}")
        if not isinstance(build_number, int) or not isinstance(subdir, str) or not subdir:
            raise EnvironmentCaptureError(f"invalid Conda build metadata: {record_path.name}")
        sha256 = record.get("sha256")
        md5 = record.get("md5")
        if sha256 is not None and (not isinstance(sha256, str) or not HEX_64_RE.fullmatch(sha256)):
            raise EnvironmentCaptureError(f"invalid Conda SHA-256: {record_path.name}")
        if md5 is not None and (not isinstance(md5, str) or not HEX_32_RE.fullmatch(md5)):
            raise EnvironmentCaptureError(f"invalid Conda MD5: {record_path.name}")
        if sha256 is None and md5 is None:
            raise EnvironmentCaptureError(f"Conda package record has no content checksum: {record_path.name}")
        name = raw_name.lower()
        if not CONDA_NAME_RE.fullmatch(name):
            raise EnvironmentCaptureError(f"invalid Conda package name: {raw_name!r}")
        entry = {
            "installer": "conda",
            "name": name,
            "version": version,
            "build": build,
            "build_number": build_number,
            "subdir": subdir,
            "package_sha256_or_null": sha256,
            "package_md5_or_null": md5,
        }
        previous = by_name.get(name)
        if previous is not None and previous != entry:
            raise EnvironmentCaptureError(f"multiple Conda records found for package {name}")
        by_name[name] = entry
    if not by_name:
        raise EnvironmentCaptureError("Conda package snapshot is empty")
    return [by_name[name] for name in sorted(by_name)]


def distribution_sort_key(entry: Mapping[str, Any]) -> tuple[str, str]:
    return str(entry.get("installer")), str(entry.get("name"))


def merge_distribution_snapshots(
    python_distributions: list[dict[str, Any]],
    conda_distributions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    combined = python_distributions + conda_distributions
    keys = [distribution_sort_key(entry) for entry in combined]
    if len(keys) != len(set(keys)):
        raise EnvironmentCaptureError("duplicate installer/name pair in environment package snapshot")
    return sorted(combined, key=distribution_sort_key)


def conda_explicit_lock_sha256(distributions: list[dict[str, Any]]) -> str:
    conda_records = [
        entry
        for entry in distributions
        if entry.get("installer") == "conda"
    ]
    if not conda_records:
        raise EnvironmentCaptureError("environment snapshot contains no Conda explicit records")
    return sha256_bytes(canonical_json_bytes(conda_records))


def critical_backend_versions(distributions: list[dict[str, Any]]) -> dict[str, str]:
    versions = {
        entry["name"]: entry["version"]
        for entry in distributions
        if entry.get("installer", "python") == "python"
    }
    missing = [name for name in CRITICAL_DISTRIBUTIONS if name not in versions]
    if missing:
        raise EnvironmentCaptureError(
            "critical Stage 1 distributions are missing: " + ", ".join(missing)
        )
    return {name: versions[name] for name in CRITICAL_DISTRIBUTIONS}


def collect_torch_build() -> dict[str, Any]:
    try:
        import torch
    except ImportError as exc:
        raise EnvironmentCaptureError("torch is required to capture the Stage 1 environment") from exc
    if not torch.cuda.is_available():
        raise EnvironmentCaptureError("CUDA is not available in the Stage 1 environment")
    cudnn_version = torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None
    return {
        "version": str(torch.__version__),
        "cuda_version": str(torch.version.cuda) if torch.version.cuda is not None else None,
        "git_version": str(torch.version.git_version) if torch.version.git_version else None,
        "cudnn_version": int(cudnn_version) if cudnn_version is not None else None,
        "debug_build": bool(torch.version.debug),
    }


def _run_nvidia_smi() -> str:
    command = [
        "nvidia-smi",
        "--query-gpu=name,compute_cap,memory.total,driver_version",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(command, check=True, capture_output=True, text=True, timeout=30)
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise EnvironmentCaptureError(
            "nvidia-smi GPU architecture probe failed; run capture in the real GPU job context"
        ) from exc
    return completed.stdout


def parse_nvidia_smi(output: str) -> tuple[str, list[dict[str, Any]]]:
    grouped: dict[tuple[str, str, int], int] = {}
    drivers: set[str] = set()
    for line in output.splitlines():
        if not line.strip():
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 4:
            raise EnvironmentCaptureError(f"unexpected nvidia-smi row: {line!r}")
        name, compute_capability, memory_text, driver_version = parts
        try:
            memory_total_mib = int(memory_text)
        except ValueError as exc:
            raise EnvironmentCaptureError(f"invalid GPU memory value: {memory_text!r}") from exc
        if not name or not compute_capability or memory_total_mib <= 0 or not driver_version:
            raise EnvironmentCaptureError(f"incomplete nvidia-smi row: {line!r}")
        drivers.add(driver_version)
        key = (name, compute_capability, memory_total_mib)
        grouped[key] = grouped.get(key, 0) + 1
    if not grouped:
        raise EnvironmentCaptureError("nvidia-smi returned no GPUs")
    if len(drivers) != 1:
        raise EnvironmentCaptureError("all GPUs must report the same NVIDIA driver version")
    architecture = [
        {
            "name": name,
            "compute_capability": compute_capability,
            "memory_total_mib": memory_total_mib,
            "count": grouped[(name, compute_capability, memory_total_mib)],
        }
        for name, compute_capability, memory_total_mib in sorted(grouped)
    ]
    return next(iter(drivers)), architecture


def build_environment_document(
    *,
    environment_spec_sha256: str,
    installed_distributions: list[dict[str, Any]],
    torch_build: Mapping[str, Any],
    driver_version: str,
    gpu_architecture: list[dict[str, Any]],
    capture_code_sha256: str,
    container_image_digest: str | None = None,
    python_implementation: str | None = None,
    python_version: str | None = None,
) -> dict[str, Any]:
    if container_image_digest is not None and not OCI_DIGEST_RE.fullmatch(container_image_digest):
        raise EnvironmentCaptureError(
            "container image digest must be an immutable sha256:<64-lowercase-hex> digest"
        )
    distribution_keys = [distribution_sort_key(entry) for entry in installed_distributions]
    if distribution_keys != sorted(distribution_keys) or len(distribution_keys) != len(set(distribution_keys)):
        raise EnvironmentCaptureError(
            "installed distributions must be uniquely sorted by installer and normalized name"
        )
    architecture_keys = [
        (
            entry.get("name"),
            entry.get("compute_capability"),
            entry.get("memory_total_mib"),
        )
        for entry in gpu_architecture
    ]
    if architecture_keys != sorted(architecture_keys) or len(architecture_keys) != len(set(architecture_keys)):
        raise EnvironmentCaptureError("GPU architecture rows must be uniquely sorted")
    document: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "python_implementation_version": {
            "implementation": python_implementation or platform.python_implementation(),
            "version": python_version or platform.python_version(),
        },
        "sorted_installed_distributions": installed_distributions,
        "torch_build": dict(torch_build),
        "cuda_runtime_driver": {
            "torch_cuda_version": torch_build.get("cuda_version"),
            "nvidia_driver_version": driver_version,
        },
        "gpu_architecture": gpu_architecture,
        "container_image_digest_or_null": container_image_digest,
        "conda_explicit_lock_sha256": conda_explicit_lock_sha256(installed_distributions),
        "critical_backend_versions": critical_backend_versions(installed_distributions),
        "environment_spec_sha256": environment_spec_sha256,
        "capture_policy_version": CAPTURE_POLICY_VERSION,
        "capture_code_sha256": capture_code_sha256,
    }
    id_input = {key: document[key] for key in ID_INPUT_KEYS}
    document["environment_build_id"] = "env-" + sha256_bytes(canonical_json_bytes(id_input))
    document["reproducibility_grade"] = (
        "container-pinned" if container_image_digest else "isolated-conda-environment"
    )
    return document


def recompute_environment_build_id(document: Mapping[str, Any]) -> str:
    try:
        id_input = {key: document[key] for key in ID_INPUT_KEYS}
    except KeyError as exc:
        raise EnvironmentCaptureError(f"environment document is missing ID input {exc.args[0]}") from exc
    return "env-" + sha256_bytes(canonical_json_bytes(id_input))


def _write_canonical_json(path: Path, value: Any) -> None:
    path.write_bytes(canonical_json_bytes(value) + b"\n")


def build_payload_manifest(target_dir: Path) -> dict[str, Any]:
    entries = []
    for path in sorted(target_dir.rglob("*")):
        if not path.is_file() or path.name == "payload_manifest.json":
            continue
        relative = path.relative_to(target_dir).as_posix()
        entries.append({"path": relative, "size": path.stat().st_size, "sha256": sha256_file(path)})
    return {"schema_version": PAYLOAD_MANIFEST_VERSION, "files": entries}


def _validate_with_json_schema(document: Mapping[str, Any], schema_path: Path) -> None:
    try:
        import jsonschema
    except ImportError as exc:
        raise EnvironmentCaptureError(
            "jsonschema is required for fail-closed Stage 1 environment validation"
        ) from exc
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    try:
        jsonschema.validate(document, schema)
    except jsonschema.ValidationError as exc:
        location = "/".join(str(item) for item in exc.absolute_path) or "<root>"
        raise EnvironmentCaptureError(
            f"environment schema validation failed at {location}: {exc.message}"
        ) from exc


def validate_target(
    target_dir: Path,
    schema_path: Path = DEFAULT_SCHEMA,
    *,
    require_directory_name: bool = True,
) -> dict[str, Any]:
    environment_path = target_dir / "environment.json"
    provenance_path = target_dir / "provenance.json"
    manifest_path = target_dir / "payload_manifest.json"
    for path in (environment_path, provenance_path, manifest_path):
        if not path.is_file():
            raise EnvironmentCaptureError(f"missing environment artifact payload: {path.name}")
    document = json.loads(environment_path.read_text(encoding="utf-8"))
    _validate_with_json_schema(document, schema_path)
    build_id = document.get("environment_build_id")
    if not isinstance(build_id, str) or not ENVIRONMENT_ID_RE.fullmatch(build_id):
        raise EnvironmentCaptureError("invalid environment_build_id")
    if recompute_environment_build_id(document) != build_id:
        raise EnvironmentCaptureError("environment_build_id does not match its canonical inputs")
    distributions = document.get("sorted_installed_distributions", [])
    keys = [distribution_sort_key(entry) for entry in distributions if isinstance(entry, dict)]
    if len(keys) != len(distributions) or keys != sorted(keys) or len(keys) != len(set(keys)):
        raise EnvironmentCaptureError(
            "installed distributions are not uniquely sorted by installer and normalized name"
        )
    if document.get("critical_backend_versions") != critical_backend_versions(distributions):
        raise EnvironmentCaptureError("critical backend versions do not match the full package snapshot")
    if document.get("conda_explicit_lock_sha256") != conda_explicit_lock_sha256(distributions):
        raise EnvironmentCaptureError("Conda explicit lock hash does not match package records")
    for entry in distributions:
        if entry.get("installer") == "python" and "direct_url" in entry:
            _validated_direct_url(
                json.dumps(entry["direct_url"], ensure_ascii=False),
                entry["name"],
            )
    container_digest = document.get("container_image_digest_or_null")
    if container_digest is not None and not OCI_DIGEST_RE.fullmatch(str(container_digest)):
        raise EnvironmentCaptureError("invalid immutable container image digest")
    if require_directory_name and target_dir.name != build_id:
        raise EnvironmentCaptureError("environment target directory name does not match build ID")
    stored_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_manifest = build_payload_manifest(target_dir)
    if stored_manifest != expected_manifest:
        raise EnvironmentCaptureError("payload_manifest.json does not match target payload")
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    forbidden = {"hostname", "gpu_uuid", "captured_at", "timestamp", "job_id", "environment_prefix"}

    def all_mapping_keys(value: Any) -> set[str]:
        if isinstance(value, dict):
            return set(value).union(*(all_mapping_keys(item) for item in value.values()))
        if isinstance(value, list):
            return set().union(*(all_mapping_keys(item) for item in value))
        return set()

    if forbidden.intersection(all_mapping_keys(provenance)):
        raise EnvironmentCaptureError("environment provenance contains non-portable runtime fields")
    return document


def _atomic_write_ref(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix=f".{path.name}.", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(canonical_json_bytes(value) + b"\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def capture(
    *,
    environment_spec: Path,
    target_root: Path,
    write_ref: Path,
    expected_prefix: Path,
    container_image_digest: str | None = None,
) -> dict[str, Any]:
    if Path(sys.prefix).resolve() != expected_prefix.resolve():
        raise EnvironmentCaptureError(
            f"capture must run from {expected_prefix}; current prefix is different"
        )
    if not environment_spec.is_file():
        raise EnvironmentCaptureError(f"environment spec does not exist: {environment_spec}")
    conda_distributions = collect_conda_distributions(expected_prefix)
    conda_identities = {
        (normalize_distribution_name(entry["name"]), entry["version"])
        for entry in conda_distributions
    }
    distributions = merge_distribution_snapshots(
        collect_installed_distributions(conda_identities=conda_identities),
        conda_distributions,
    )
    torch_build = collect_torch_build()
    driver_version, architecture = parse_nvidia_smi(_run_nvidia_smi())
    document = build_environment_document(
        environment_spec_sha256=sha256_file(environment_spec),
        installed_distributions=distributions,
        torch_build=torch_build,
        driver_version=driver_version,
        gpu_architecture=architecture,
        capture_code_sha256=sha256_file(Path(__file__)),
        container_image_digest=container_image_digest,
    )
    build_id = document["environment_build_id"]
    target_root.mkdir(parents=True, exist_ok=True)
    final_dir = target_root / build_id
    temporary_dir = Path(tempfile.mkdtemp(prefix=f".{build_id}.", dir=target_root))
    try:
        _write_canonical_json(temporary_dir / "environment.json", document)
        _write_canonical_json(
            temporary_dir / "provenance.json",
            {
                "schema_version": "stage1-environment-provenance/v1",
                "environment_spec_logical_path": "environment/stage1-p0.yml",
                "capture_script_logical_path": "scripts/stage1/capture_environment.py",
                "capture_policy_version": CAPTURE_POLICY_VERSION,
                "capture_code_sha256": document["capture_code_sha256"],
            },
        )
        _write_canonical_json(temporary_dir / "payload_manifest.json", build_payload_manifest(temporary_dir))
        validate_target(temporary_dir, require_directory_name=False)
        if final_dir.exists():
            existing = validate_target(final_dir)
            if existing != document:
                raise EnvironmentCaptureError(f"existing target {build_id} has different content")
            shutil.rmtree(temporary_dir)
        else:
            os.replace(temporary_dir, final_dir)
    finally:
        if temporary_dir.exists():
            shutil.rmtree(temporary_dir)
    manifest_hash = sha256_file(final_dir / "payload_manifest.json")
    locator = {
        "schema_version": LOCATOR_REF_VERSION,
        "artifact_kind": ARTIFACT_KIND,
        "artifact_id": build_id,
        "payload_manifest_sha256": manifest_hash,
        "target_path": str(final_dir.resolve()),
    }
    _atomic_write_ref(write_ref, locator)
    return locator


def validate_ref(environment_ref: Path) -> dict[str, Any]:
    locator = json.loads(environment_ref.read_text(encoding="utf-8"))
    if locator.get("schema_version") != LOCATOR_REF_VERSION:
        raise EnvironmentCaptureError("unsupported environment locator-ref schema")
    if locator.get("artifact_kind") != ARTIFACT_KIND:
        raise EnvironmentCaptureError("locator ref is not a Stage 1 environment")
    build_id = locator.get("artifact_id")
    if not isinstance(build_id, str) or not ENVIRONMENT_ID_RE.fullmatch(build_id):
        raise EnvironmentCaptureError("invalid environment artifact ID in locator ref")
    target_dir = Path(locator.get("target_path", ""))
    document = validate_target(target_dir)
    if document["environment_build_id"] != build_id:
        raise EnvironmentCaptureError("locator ref artifact ID does not match target")
    manifest_hash = sha256_file(target_dir / "payload_manifest.json")
    if locator.get("payload_manifest_sha256") != manifest_hash:
        raise EnvironmentCaptureError("locator ref payload hash does not match target")
    return document


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    capture_parser = subparsers.add_parser("capture", help="capture an immutable environment target")
    capture_parser.add_argument("--environment-spec", type=Path, default=DEFAULT_SPEC)
    capture_parser.add_argument("--expected-prefix", type=Path, default=DEFAULT_PREFIX)
    capture_parser.add_argument("--target-root", type=Path, default=DEFAULT_TARGET_ROOT)
    capture_parser.add_argument("--container-image-digest", default=None)
    capture_parser.add_argument("--write-ref", type=Path, required=True)
    validate_parser = subparsers.add_parser("validate", help="validate an immutable environment target")
    validate_parser.add_argument("--environment-ref", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "capture":
            locator = capture(
                environment_spec=args.environment_spec,
                target_root=args.target_root,
                write_ref=args.write_ref,
                expected_prefix=args.expected_prefix,
                container_image_digest=args.container_image_digest,
            )
            print(json.dumps(locator, ensure_ascii=False, sort_keys=True))
        else:
            document = validate_ref(args.environment_ref)
            print(json.dumps({"environment_build_id": document["environment_build_id"]}, sort_keys=True))
    except (EnvironmentCaptureError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"[stage1-environment] {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
