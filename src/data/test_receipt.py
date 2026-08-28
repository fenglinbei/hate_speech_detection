"""Immutable, machine-verifiable Stage 1 P0 test receipts.

The builder executes the repository-owned command manifest directly (never via
``shell=True`` and never by importing pre-computed results).  A successful run
freezes the exact command frames, the current Stage 1 source/config/schema
inventory, the Python runtime identity, normalised logs, and exit codes in a
content-addressed target.  Publication is fail-closed: one non-zero command,
timeout, malformed manifest, or source change during execution publishes
nothing.
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import math
import os
import py_compile
import re
import subprocess
import sys
import tempfile
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from data.training_artifacts import (
    TrainingArtifactError,
    canonical_sha256,
    ensure_exact_file_set,
    finalize_target_atomic,
    load_json,
    new_staging_directory,
    resolve_locator_ref,
    sha256_file,
    validate_json_schema,
    validate_payload_manifest,
    write_bytes_atomic,
    write_canonical_json,
    write_locator_ref,
)


SCHEMA_VERSION = "stage1-test-receipt/v1"
COMMAND_MANIFEST_SCHEMA_VERSION = "stage1-test-command-manifest/v1"
SOURCE_INVENTORY_SCHEMA_VERSION = "stage1-test-source-inventory/v1"
RESULTS_SCHEMA_VERSION = "stage1-test-results/v1"
PROVENANCE_SCHEMA_VERSION = "stage1-test-receipt-provenance/v1"
ARTIFACT_KIND = "stage1-p0-verification-receipt"
ARTIFACT_ID_PREFIX = "vrec-"
EXECUTION_POLICY = "actual-subprocess-no-import/v1"
SOURCE_POLICY = "stage1-source-config-schema-regular-files/v1"
LOG_NORMALIZATION_POLICY = "utf8-replace-crlf-lf-strip-ansi-portable-paths/v1"
DEFAULT_MANIFEST_LOGICAL_PATH = "config/stage1/p0_test_commands.json"
DEFAULT_SCHEMA_LOGICAL_PATH = "schemas/stage1_test_receipt_v1.schema.json"
DEFAULT_TARGET_LOGICAL_ROOT = "exps/causal_context/stage1_p0/verification_receipts"
DEFAULT_REF_LOGICAL_PATH = "exps/causal_context/stage1_p0/refs/verification_receipt_ref.json"
MAX_LOG_BYTES = 64 * 1024 * 1024

COMMAND_ID_RE = re.compile(r"^[a-z][a-z0-9_.-]{1,95}$")
GROUP_RE = re.compile(r"^[a-z][a-z0-9_.-]{0,63}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")

_INVENTORY_DIRECTORY_RULES: tuple[tuple[str, frozenset[str]], ...] = (
    ("src", frozenset({".py"})),
    ("scripts/stage1", frozenset({".py"})),
    ("schemas", frozenset({".json"})),
    ("config/stage1", frozenset({".json", ".yaml", ".yml", ".md", ".txt"})),
)
_INVENTORY_FILE_RULES = (
    "environment/stage1-p0.yml",
)


class TestReceiptError(TrainingArtifactError):
    """Raised when a test receipt cannot be built or validated."""


TestReceiptError.__test__ = False


def _safe_logical_path(value: Any, *, allow_dot: bool = False) -> str:
    if not isinstance(value, str) or not value:
        raise TestReceiptError("logical path must be a non-empty string")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or value != path.as_posix():
        raise TestReceiptError("logical path is not portable")
    if value == "." and allow_dot:
        return value
    if value in {"", "."}:
        raise TestReceiptError("logical path cannot name the workspace root")
    return value


def _workspace_path(root: Path, logical_path: str, *, allow_dot: bool = False) -> Path:
    logical = _safe_logical_path(logical_path, allow_dot=allow_dot)
    target = (root / logical).resolve()
    try:
        target.relative_to(root.resolve())
    except ValueError as exc:
        raise TestReceiptError("logical path escapes workspace") from exc
    return target


def _inventory_files(workspace_root: Path) -> list[Path]:
    root = workspace_root.resolve()
    files: set[Path] = set()
    for logical_directory, suffixes in _INVENTORY_DIRECTORY_RULES:
        directory = _workspace_path(root, logical_directory)
        if not directory.is_dir() or directory.is_symlink():
            raise TestReceiptError(
                f"required source inventory directory is missing: {logical_directory}"
            )
        for candidate in directory.rglob("*"):
            if candidate.is_symlink():
                raise TestReceiptError(
                    f"source inventory cannot contain symlink: {candidate.relative_to(root)}"
                )
            if candidate.is_file() and candidate.suffix.lower() in suffixes:
                files.add(candidate.resolve())
    for logical_file in _INVENTORY_FILE_RULES:
        candidate = _workspace_path(root, logical_file)
        if not candidate.is_file() or candidate.is_symlink():
            raise TestReceiptError(
                f"required source inventory file is missing: {logical_file}"
            )
        files.add(candidate.resolve())
    if not files:
        raise TestReceiptError("source inventory is empty")
    return sorted(files, key=lambda item: item.relative_to(root).as_posix())


def build_source_inventory(workspace_root: str | Path) -> dict[str, Any]:
    """Hash the complete frozen Stage 1 source/config/schema selection."""

    root = Path(workspace_root).resolve()
    entries = [
        {
            "logical_path": path.relative_to(root).as_posix(),
            "size": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in _inventory_files(root)
    ]
    selectors = {
        "directories": [
            {"logical_path": logical, "suffixes": sorted(suffixes)}
            for logical, suffixes in _INVENTORY_DIRECTORY_RULES
        ],
        "files": list(_INVENTORY_FILE_RULES),
    }
    projection = {
        "schema_version": SOURCE_INVENTORY_SCHEMA_VERSION,
        "inventory_policy": SOURCE_POLICY,
        "selectors": selectors,
        "file_count": len(entries),
        "total_bytes": sum(int(item["size"]) for item in entries),
        "files": entries,
    }
    return {**projection, "file_tree_sha256": canonical_sha256(projection)}


def _validate_source_inventory(document: Any) -> dict[str, Any]:
    required = {
        "schema_version",
        "inventory_policy",
        "selectors",
        "file_count",
        "total_bytes",
        "files",
        "file_tree_sha256",
    }
    if not isinstance(document, Mapping) or set(document) != required:
        raise TestReceiptError("source inventory has non-canonical fields")
    if document.get("schema_version") != SOURCE_INVENTORY_SCHEMA_VERSION:
        raise TestReceiptError("source inventory schema mismatch")
    if document.get("inventory_policy") != SOURCE_POLICY:
        raise TestReceiptError("source inventory policy mismatch")
    expected_selectors = {
        "directories": [
            {"logical_path": logical, "suffixes": sorted(suffixes)}
            for logical, suffixes in _INVENTORY_DIRECTORY_RULES
        ],
        "files": list(_INVENTORY_FILE_RULES),
    }
    if document.get("selectors") != expected_selectors:
        raise TestReceiptError("source inventory selectors mismatch")
    entries = document.get("files")
    if not isinstance(entries, list) or not entries:
        raise TestReceiptError("source inventory file list is empty")
    paths: list[str] = []
    for item in entries:
        if not isinstance(item, Mapping) or set(item) != {
            "logical_path",
            "size",
            "sha256",
        }:
            raise TestReceiptError("source inventory entry is malformed")
        logical = _safe_logical_path(item.get("logical_path"))
        size = item.get("size")
        digest = item.get("sha256")
        if not isinstance(size, int) or isinstance(size, bool) or size < 0:
            raise TestReceiptError("source inventory size is invalid")
        if not isinstance(digest, str) or not SHA256_RE.fullmatch(digest):
            raise TestReceiptError("source inventory digest is invalid")
        paths.append(logical)
    if paths != sorted(paths) or len(paths) != len(set(paths)):
        raise TestReceiptError("source inventory paths are not canonical")
    projection = {key: document[key] for key in required - {"file_tree_sha256"}}
    if document.get("file_count") != len(entries):
        raise TestReceiptError("source inventory file count mismatch")
    if document.get("total_bytes") != sum(int(item["size"]) for item in entries):
        raise TestReceiptError("source inventory byte count mismatch")
    if document.get("file_tree_sha256") != canonical_sha256(projection):
        raise TestReceiptError("source inventory tree hash mismatch")
    return dict(document)


def _validate_command_manifest(document: Any) -> dict[str, Any]:
    if not isinstance(document, Mapping) or set(document) != {
        "schema_version",
        "execution_policy",
        "log_normalization_policy",
        "commands",
    }:
        raise TestReceiptError("test command manifest has non-canonical fields")
    if document.get("schema_version") != COMMAND_MANIFEST_SCHEMA_VERSION:
        raise TestReceiptError("test command manifest schema mismatch")
    if document.get("execution_policy") != EXECUTION_POLICY:
        raise TestReceiptError("test command execution policy mismatch")
    if document.get("log_normalization_policy") != LOG_NORMALIZATION_POLICY:
        raise TestReceiptError("test command log policy mismatch")
    commands = document.get("commands")
    if not isinstance(commands, list) or not commands:
        raise TestReceiptError("test command manifest is empty")
    identities: list[str] = []
    normalized: list[dict[str, Any]] = []
    for command in commands:
        if not isinstance(command, Mapping) or set(command) != {
            "command_id",
            "group",
            "argv",
            "working_directory",
            "timeout_seconds",
        }:
            raise TestReceiptError("test command frame has non-canonical fields")
        command_id = command.get("command_id")
        group = command.get("group")
        argv = command.get("argv")
        working_directory = command.get("working_directory")
        timeout_seconds = command.get("timeout_seconds")
        if not isinstance(command_id, str) or not COMMAND_ID_RE.fullmatch(command_id):
            raise TestReceiptError("test command ID is invalid")
        if not isinstance(group, str) or not GROUP_RE.fullmatch(group):
            raise TestReceiptError("test command group is invalid")
        if (
            not isinstance(argv, list)
            or len(argv) < 2
            or argv[0] != "{python}"
            or not all(isinstance(token, str) and token and "\x00" not in token for token in argv)
        ):
            raise TestReceiptError("test command argv must use the frozen Python token")
        _safe_logical_path(working_directory, allow_dot=True)
        if (
            not isinstance(timeout_seconds, int)
            or isinstance(timeout_seconds, bool)
            or not 1 <= timeout_seconds <= 7200
        ):
            raise TestReceiptError("test command timeout is invalid")
        identities.append(command_id)
        normalized.append(dict(command))
    if len(identities) != len(set(identities)):
        raise TestReceiptError("test command IDs are not unique")
    return {
        "schema_version": COMMAND_MANIFEST_SCHEMA_VERSION,
        "execution_policy": EXECUTION_POLICY,
        "log_normalization_policy": LOG_NORMALIZATION_POLICY,
        "commands": normalized,
    }


def load_required_command_manifest(workspace_root: str | Path) -> dict[str, Any]:
    root = Path(workspace_root).resolve()
    path = _workspace_path(root, DEFAULT_MANIFEST_LOGICAL_PATH)
    if not path.is_file() or path.is_symlink():
        raise TestReceiptError("required P0 command manifest is missing")
    return _validate_command_manifest(load_json(path))


def runtime_identity() -> dict[str, Any]:
    executable = Path(sys.executable).resolve()
    if not executable.is_file():
        raise TestReceiptError("current Python executable is not a regular file")
    return {
        "schema_version": "stage1-test-python-runtime/v1",
        "implementation": sys.implementation.name,
        "implementation_version": platform_python_version(),
        "cache_tag": sys.implementation.cache_tag,
        "executable_sha256": sha256_file(executable),
    }


def platform_python_version() -> str:
    return ".".join(str(item) for item in sys.version_info[:3])


def _normalise_log(payload: bytes, *, workspace_root: Path) -> bytes:
    text = payload.decode("utf-8", errors="replace").replace("\r\n", "\n").replace("\r", "\n")
    text = ANSI_ESCAPE_RE.sub("", text)
    replacements = {
        str(workspace_root.resolve()): "<WORKSPACE>",
        str(Path(sys.executable).resolve()): "{python}",
    }
    for source in sorted(replacements, key=len, reverse=True):
        text = text.replace(source, replacements[source])
    return text.encode("utf-8")


def _subprocess_environment(workspace_root: Path) -> dict[str, str]:
    secret_fragments = ("SECRET", "TOKEN", "PASSWORD", "API_KEY", "ACCESS_KEY")
    environment = {
        key: value
        for key, value in os.environ.items()
        if not any(fragment in key.upper() for fragment in secret_fragments)
    }
    python_path = os.pathsep.join(
        (str(workspace_root / "src"), str(workspace_root))
    )
    environment.update(
        {
            "PYTHONPATH": python_path,
            "PYTHONDONTWRITEBYTECODE": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        }
    )
    return environment


def _execute_command(command: Mapping[str, Any], workspace_root: Path) -> tuple[dict[str, Any], bytes, bytes]:
    cwd = _workspace_path(
        workspace_root,
        str(command["working_directory"]),
        allow_dot=True,
    )
    if not cwd.is_dir() or cwd.is_symlink():
        raise TestReceiptError(f"test command cwd is invalid: {command['command_id']}")
    argv = [sys.executable, *[str(item) for item in command["argv"]][1:]]
    started = time.monotonic_ns()
    timed_out = False
    try:
        completed = subprocess.run(
            argv,
            cwd=cwd,
            env=_subprocess_environment(workspace_root),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=False,
            check=False,
            timeout=int(command["timeout_seconds"]),
        )
        exit_code = int(completed.returncode)
        raw_stdout = completed.stdout
        raw_stderr = completed.stderr
    except subprocess.TimeoutExpired as exc:
        exit_code = 124
        timed_out = True
        raw_stdout = exc.stdout or b""
        raw_stderr = (exc.stderr or b"") + b"\ncommand timed out\n"
    duration_ms = max(0, math.ceil((time.monotonic_ns() - started) / 1_000_000))
    stdout = _normalise_log(raw_stdout, workspace_root=workspace_root)
    stderr = _normalise_log(raw_stderr, workspace_root=workspace_root)
    if len(stdout) > MAX_LOG_BYTES or len(stderr) > MAX_LOG_BYTES:
        raise TestReceiptError(f"test command log exceeds frozen cap: {command['command_id']}")
    result = {
        "command_id": command["command_id"],
        "command_frame_sha256": canonical_sha256(command),
        "runner_execution": True,
        "exit_code": exit_code,
        "timed_out": timed_out,
        "duration_ms": duration_ms,
        "stdout_size": len(stdout),
        "stdout_sha256": hashlib.sha256(stdout).hexdigest(),
        "stderr_size": len(stderr),
        "stderr_sha256": hashlib.sha256(stderr).hexdigest(),
    }
    return result, stdout, stderr


def _receipt_schema_path(workspace_root: Path) -> Path:
    path = _workspace_path(workspace_root, DEFAULT_SCHEMA_LOGICAL_PATH)
    if not path.is_file() or path.is_symlink():
        raise TestReceiptError("test receipt JSON schema is missing")
    return path


def _builder_code_sha256() -> str:
    return sha256_file(Path(__file__).resolve())


def _execution_projection(
    *,
    command_manifest: Mapping[str, Any],
    source_inventory: Mapping[str, Any],
    results: Mapping[str, Any],
    runtime: Mapping[str, Any],
    jobs: int,
    schema_sha256: str,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "command_manifest_sha256": canonical_sha256(command_manifest),
        "source_inventory_sha256": canonical_sha256(source_inventory),
        "runtime_identity": dict(runtime),
        "execution_policy": {
            "policy": EXECUTION_POLICY,
            "max_parallel_commands": jobs,
            "log_normalization_policy": LOG_NORMALIZATION_POLICY,
            "maximum_log_bytes_per_stream": MAX_LOG_BYTES,
            "expected_exit_code": 0,
        },
        "ordered_results_sha256": canonical_sha256(results),
        "builder_code_sha256": _builder_code_sha256(),
        "receipt_schema_sha256": schema_sha256,
    }


def build_test_receipt(
    *,
    workspace_root: str | Path,
    target_root: str | Path | None = None,
    ref_path: str | Path | None = None,
    jobs: int = 1,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Execute the frozen P0 manifest and publish a receipt only on success."""

    root = Path(workspace_root).resolve()
    if not isinstance(jobs, int) or isinstance(jobs, bool) or not 1 <= jobs <= 16:
        raise TestReceiptError("jobs must be an integer in [1, 16]")
    command_manifest = load_required_command_manifest(root)
    before_inventory = build_source_inventory(root)
    runtime = runtime_identity()
    commands = list(command_manifest["commands"])
    collected: dict[str, tuple[dict[str, Any], bytes, bytes]] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as executor:
        futures = {
            executor.submit(_execute_command, command, root): str(command["command_id"])
            for command in commands
        }
        for future in concurrent.futures.as_completed(futures):
            collected[futures[future]] = future.result()
    ordered = [collected[str(command["command_id"])] for command in commands]
    result_rows = [item[0] for item in ordered]
    if any(result["exit_code"] != 0 or result["timed_out"] for result in result_rows):
        failures = [
            str(result["command_id"])
            for result in result_rows
            if result["exit_code"] != 0 or result["timed_out"]
        ]
        raise TestReceiptError(
            "test commands failed; immutable receipt was not published: "
            + ", ".join(failures)
        )
    after_inventory = build_source_inventory(root)
    if after_inventory != before_inventory:
        raise TestReceiptError(
            "Stage 1 source/config/schema changed during test execution"
        )
    results = {
        "schema_version": RESULTS_SCHEMA_VERSION,
        "command_manifest_sha256": canonical_sha256(command_manifest),
        "ordered_command_ids": [str(command["command_id"]) for command in commands],
        "all_passed": True,
        "results": result_rows,
    }
    schema_path = _receipt_schema_path(root)
    id_inputs = _execution_projection(
        command_manifest=command_manifest,
        source_inventory=before_inventory,
        results=results,
        runtime=runtime,
        jobs=jobs,
        schema_sha256=sha256_file(schema_path),
    )
    receipt_id = ARTIFACT_ID_PREFIX + canonical_sha256(id_inputs)
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "test_receipt_id": receipt_id,
        "status": "completed",
        "all_passed": True,
        "command_count": len(commands),
        "total_duration_ms": sum(int(row["duration_ms"]) for row in result_rows),
        "id_inputs": id_inputs,
        "provenance": {
            "schema_version": PROVENANCE_SCHEMA_VERSION,
            "test_receipt_id": receipt_id,
            "command_manifest_sha256": canonical_sha256(command_manifest),
            "source_inventory_sha256": canonical_sha256(before_inventory),
            "results_sha256": canonical_sha256(results),
            "builder_code_sha256": _builder_code_sha256(),
            "receipt_schema_sha256": sha256_file(schema_path),
        },
    }
    validate_json_schema(receipt, schema_path)
    destination_root = (
        Path(target_root).resolve()
        if target_root is not None
        else _workspace_path(root, DEFAULT_TARGET_LOGICAL_ROOT)
    )
    try:
        destination_root.relative_to(root)
    except ValueError as exc:
        raise TestReceiptError("test receipt target root must be inside workspace") from exc
    destination = destination_root / receipt_id
    staging = new_staging_directory(destination_root, receipt_id)
    try:
        write_canonical_json(staging / "receipt.json", receipt)
        write_canonical_json(staging / "command_manifest.json", command_manifest)
        write_canonical_json(staging / "source_inventory.json", before_inventory)
        write_canonical_json(staging / "results.json", results)
        for command, (_result, stdout, stderr) in zip(commands, ordered, strict=True):
            command_id = str(command["command_id"])
            write_bytes_atomic(staging / "logs" / f"{command_id}.stdout.txt", stdout)
            write_bytes_atomic(staging / "logs" / f"{command_id}.stderr.txt", stderr)
        payload_hash = finalize_target_atomic(
            staging,
            destination,
            validate_staging=lambda target: _validate_test_receipt_target(
                target,
                workspace_root=root,
                require_current_sources=True,
                allow_staging_name=True,
            ),
        )
    except Exception:
        if staging.exists():
            import shutil

            shutil.rmtree(staging)
        raise
    selected_ref = (
        Path(ref_path).resolve()
        if ref_path is not None
        else _workspace_path(root, DEFAULT_REF_LOGICAL_PATH)
    )
    try:
        selected_ref.relative_to(root)
    except ValueError as exc:
        raise TestReceiptError("test receipt locator must be inside workspace") from exc
    locator = write_locator_ref(
        selected_ref,
        artifact_kind=ARTIFACT_KIND,
        artifact_id=receipt_id,
        target=destination,
        payload_manifest_sha256=payload_hash,
    )
    return locator, receipt


def _validate_result_row(
    row: Any,
    command: Mapping[str, Any],
    target: Path,
    *,
    workspace_root: Path,
) -> dict[str, Any]:
    required = {
        "command_id",
        "command_frame_sha256",
        "runner_execution",
        "exit_code",
        "timed_out",
        "duration_ms",
        "stdout_size",
        "stdout_sha256",
        "stderr_size",
        "stderr_sha256",
    }
    if not isinstance(row, Mapping) or set(row) != required:
        raise TestReceiptError("test result row has non-canonical fields")
    command_id = str(command["command_id"])
    if row.get("command_id") != command_id:
        raise TestReceiptError("test result order/identity mismatch")
    if row.get("command_frame_sha256") != canonical_sha256(command):
        raise TestReceiptError("test result command frame hash mismatch")
    if row.get("runner_execution") is not True:
        raise TestReceiptError("test result is not an actual runner execution")
    if row.get("exit_code") != 0 or row.get("timed_out") is not False:
        raise TestReceiptError("test result is not successful")
    duration = row.get("duration_ms")
    if not isinstance(duration, int) or isinstance(duration, bool) or duration < 0:
        raise TestReceiptError("test result duration is invalid")
    for stream in ("stdout", "stderr"):
        size = row.get(f"{stream}_size")
        digest = row.get(f"{stream}_sha256")
        if not isinstance(size, int) or isinstance(size, bool) or not 0 <= size <= MAX_LOG_BYTES:
            raise TestReceiptError("test result log size is invalid")
        if not isinstance(digest, str) or not SHA256_RE.fullmatch(digest):
            raise TestReceiptError("test result log hash is invalid")
        log_path = target / "logs" / f"{command_id}.{stream}.txt"
        if not log_path.is_file() or log_path.is_symlink():
            raise TestReceiptError("test result log is missing")
        payload = log_path.read_bytes()
        if len(payload) != size or hashlib.sha256(payload).hexdigest() != digest:
            raise TestReceiptError("test result log payload mismatch")
        if _normalise_log(payload, workspace_root=workspace_root) != payload:
            raise TestReceiptError("test result log is not canonically normalized")
    return dict(row)


def _validate_test_receipt_target(
    target: str | Path,
    *,
    workspace_root: str | Path,
    require_current_sources: bool,
    allow_staging_name: bool = False,
) -> dict[str, Any]:
    directory = Path(target).resolve()
    root = Path(workspace_root).resolve()
    try:
        directory.relative_to(root)
    except ValueError as exc:
        raise TestReceiptError("test receipt target is outside workspace") from exc
    validate_payload_manifest(directory)
    receipt = load_json(directory / "receipt.json")
    manifest = _validate_command_manifest(load_json(directory / "command_manifest.json"))
    required_manifest = load_required_command_manifest(root)
    if manifest != required_manifest:
        raise TestReceiptError("receipt did not execute the frozen P0 command manifest")
    inventory = _validate_source_inventory(load_json(directory / "source_inventory.json"))
    commands = list(manifest["commands"])
    expected_files = {
        "receipt.json",
        "command_manifest.json",
        "source_inventory.json",
        "results.json",
        "payload_manifest.json",
    }
    for command in commands:
        command_id = str(command["command_id"])
        expected_files.add(f"logs/{command_id}.stdout.txt")
        expected_files.add(f"logs/{command_id}.stderr.txt")
    ensure_exact_file_set(directory, expected_files)
    results = load_json(directory / "results.json")
    if not isinstance(results, Mapping) or set(results) != {
        "schema_version",
        "command_manifest_sha256",
        "ordered_command_ids",
        "all_passed",
        "results",
    }:
        raise TestReceiptError("test results document has non-canonical fields")
    if results.get("schema_version") != RESULTS_SCHEMA_VERSION:
        raise TestReceiptError("test results schema mismatch")
    command_ids = [str(command["command_id"]) for command in commands]
    if results.get("command_manifest_sha256") != canonical_sha256(manifest):
        raise TestReceiptError("test results manifest hash mismatch")
    if results.get("ordered_command_ids") != command_ids or results.get("all_passed") is not True:
        raise TestReceiptError("test results completeness mismatch")
    rows = results.get("results")
    if not isinstance(rows, list) or len(rows) != len(commands):
        raise TestReceiptError("test results row set mismatch")
    validated_rows = [
        _validate_result_row(row, command, directory, workspace_root=root)
        for row, command in zip(rows, commands, strict=True)
    ]
    schema_path = _receipt_schema_path(root)
    if not isinstance(receipt, Mapping):
        raise TestReceiptError("test receipt document is malformed")
    validate_json_schema(receipt, schema_path)
    current_runtime = runtime_identity()
    execution_policy = receipt.get("id_inputs", {}).get("execution_policy")
    if not isinstance(execution_policy, Mapping):
        raise TestReceiptError("test receipt execution policy is malformed")
    jobs = execution_policy.get("max_parallel_commands")
    if not isinstance(jobs, int) or isinstance(jobs, bool) or not 1 <= jobs <= 16:
        raise TestReceiptError("test receipt parallelism is invalid")
    expected_id_inputs = _execution_projection(
        command_manifest=manifest,
        source_inventory=inventory,
        results=results,
        runtime=current_runtime,
        jobs=jobs,
        schema_sha256=sha256_file(schema_path),
    )
    if receipt.get("id_inputs") != expected_id_inputs:
        raise TestReceiptError("test receipt identity inputs mismatch current contract")
    receipt_id = ARTIFACT_ID_PREFIX + canonical_sha256(expected_id_inputs)
    valid_directory_name = directory.name == receipt_id or (
        allow_staging_name and directory.name.startswith(f".{receipt_id}.")
    )
    if not valid_directory_name or receipt.get("test_receipt_id") != receipt_id:
        raise TestReceiptError("test receipt content-addressed ID mismatch")
    if receipt.get("status") != "completed" or receipt.get("all_passed") is not True:
        raise TestReceiptError("test receipt is incomplete")
    if receipt.get("command_count") != len(commands):
        raise TestReceiptError("test receipt command count mismatch")
    if receipt.get("total_duration_ms") != sum(int(row["duration_ms"]) for row in validated_rows):
        raise TestReceiptError("test receipt total duration mismatch")
    expected_provenance = {
        "schema_version": PROVENANCE_SCHEMA_VERSION,
        "test_receipt_id": receipt_id,
        "command_manifest_sha256": canonical_sha256(manifest),
        "source_inventory_sha256": canonical_sha256(inventory),
        "results_sha256": canonical_sha256(results),
        "builder_code_sha256": _builder_code_sha256(),
        "receipt_schema_sha256": sha256_file(schema_path),
    }
    if receipt.get("provenance") != expected_provenance:
        raise TestReceiptError("test receipt provenance mismatch")
    if require_current_sources and build_source_inventory(root) != inventory:
        raise TestReceiptError("current Stage 1 source/config/schema differs from tested inventory")
    return {
        "test_receipt_id": receipt_id,
        "command_count": len(commands),
        "all_passed": True,
        "source_file_count": inventory["file_count"],
    }


def validate_test_receipt(
    test_receipt_ref: str | Path,
    *,
    workspace_root: str | Path,
) -> dict[str, Any]:
    """Deep-validate a locator, payload, command frame, logs, and live sources."""

    try:
        locator, target = resolve_locator_ref(test_receipt_ref, expected_kind=ARTIFACT_KIND)
        report = _validate_test_receipt_target(
            target,
            workspace_root=workspace_root,
            require_current_sources=True,
        )
        if locator.get("artifact_id") != report["test_receipt_id"]:
            raise TestReceiptError("test receipt locator ID mismatch")
        return report
    except TestReceiptError:
        raise
    except TrainingArtifactError as exc:
        raise TestReceiptError("test receipt locator or payload is invalid") from exc


def check_python_sources(workspace_root: str | Path) -> dict[str, int]:
    """Run ``py_compile`` for every frozen Python source without polluting the tree."""

    root = Path(workspace_root).resolve()
    sources = [path for path in _inventory_files(root) if path.suffix == ".py"]
    with tempfile.TemporaryDirectory(prefix="stage1-pycompile-") as temporary:
        output_root = Path(temporary)
        for index, source in enumerate(sources):
            try:
                py_compile.compile(
                    str(source),
                    cfile=str(output_root / f"{index}.pyc"),
                    doraise=True,
                )
            except py_compile.PyCompileError as exc:
                raise TestReceiptError(f"Python compile failed: {source.relative_to(root)}") from exc
    return {"python_source_count": len(sources)}


def check_json_documents(workspace_root: str | Path) -> dict[str, int]:
    root = Path(workspace_root).resolve()
    paths = [
        path
        for path in _inventory_files(root)
        if path.suffix.lower() == ".json"
        and (path.is_relative_to(root / "config/stage1") or path.is_relative_to(root / "schemas"))
    ]
    for path in paths:
        load_json(path)
    return {"json_document_count": len(paths)}


def check_json_schemas(workspace_root: str | Path) -> dict[str, int]:
    root = Path(workspace_root).resolve()
    schema_paths = sorted((root / "schemas").glob("*.json"))
    if not schema_paths:
        raise TestReceiptError("schema directory is empty")
    try:
        import jsonschema
    except ImportError as exc:
        raise TestReceiptError("jsonschema is required for schema checks") from exc
    for path in schema_paths:
        schema = load_json(path)
        validator_class = jsonschema.validators.validator_for(schema)
        validator_class.check_schema(schema)
    return {"json_schema_count": len(schema_paths)}


__all__ = [
    "ARTIFACT_KIND",
    "COMMAND_MANIFEST_SCHEMA_VERSION",
    "DEFAULT_MANIFEST_LOGICAL_PATH",
    "DEFAULT_REF_LOGICAL_PATH",
    "DEFAULT_TARGET_LOGICAL_ROOT",
    "TestReceiptError",
    "build_source_inventory",
    "build_test_receipt",
    "check_json_documents",
    "check_json_schemas",
    "check_python_sources",
    "load_required_command_manifest",
    "runtime_identity",
    "validate_test_receipt",
]
