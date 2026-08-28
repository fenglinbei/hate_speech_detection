"""Shared immutable-artifact primitives for Stage 1 training artifacts.

Only external locator references may contain an absolute path.  Payloads embed
the canonical five-field dependency projection returned by
``portable_dependency``.  Builders stage a complete target, validate it, write
its payload manifest, atomically rename it, and only then publish the locator.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable


LOCATOR_REF_SCHEMA_VERSION = "stage1-locator-ref/v1"
DEPENDENCY_REF_SCHEMA_VERSION = "stage1-dependency-ref/v1"
PAYLOAD_MANIFEST_SCHEMA_VERSION = "stage1-payload-manifest/v1"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class TrainingArtifactError(RuntimeError):
    """Raised when a Stage 1 immutable training artifact is invalid."""


def canonical_json_bytes(value: Any) -> bytes:
    """Encode canonical UTF-8 JSON, rejecting NaN and non-JSON values."""

    try:
        rendered = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise TrainingArtifactError(f"value is not canonical JSON: {exc}") from exc
    return rendered.encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: str | Path) -> Any:
    source = Path(path)
    try:
        with source.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TrainingArtifactError(f"cannot read JSON {source}: {exc}") from exc


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    rows: list[dict[str, Any]] = []
    try:
        with source.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    raise TrainingArtifactError(
                        f"blank JSONL row in {source} at line {line_number}"
                    )
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise TrainingArtifactError(
                        f"JSONL row in {source} at line {line_number} is not an object"
                    )
                rows.append(value)
    except TrainingArtifactError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TrainingArtifactError(f"cannot read JSONL {source}: {exc}") from exc
    return rows


def _atomic_write_bytes(path: str | Path, payload: bytes) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{destination.name}.",
            dir=destination.parent,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def write_canonical_json(path: str | Path, value: Any) -> None:
    _atomic_write_bytes(path, canonical_json_bytes(value) + b"\n")


def write_bytes_atomic(path: str | Path, payload: bytes) -> None:
    """Atomically write already-canonical bytes."""

    _atomic_write_bytes(path, payload)


def canonical_jsonl_bytes(
    rows: Iterable[Mapping[str, Any]], *, key: str, numeric_key: bool = False
) -> bytes:
    materialized = [dict(row) for row in rows]
    try:
        if numeric_key:
            materialized.sort(key=lambda row: int(str(row[key])))
        else:
            materialized.sort(key=lambda row: str(row[key]))
    except (KeyError, TypeError, ValueError) as exc:
        raise TrainingArtifactError(f"invalid JSONL primary key {key!r}: {exc}") from exc
    identities = [str(row[key]) for row in materialized]
    if len(identities) != len(set(identities)):
        raise TrainingArtifactError(f"JSONL primary key {key!r} is not unique")
    return b"".join(canonical_json_bytes(row) + b"\n" for row in materialized)


def write_canonical_jsonl(
    path: str | Path,
    rows: Iterable[Mapping[str, Any]],
    *,
    key: str,
    numeric_key: bool = False,
) -> None:
    _atomic_write_bytes(
        path, canonical_jsonl_bytes(rows, key=key, numeric_key=numeric_key)
    )


def _safe_payload_files(target: Path) -> list[Path]:
    if not target.is_dir() or target.is_symlink():
        raise TrainingArtifactError(f"artifact target is not a regular directory: {target}")
    files: list[Path] = []
    for path in sorted(target.rglob("*")):
        if path.is_symlink():
            raise TrainingArtifactError(f"artifact payload cannot contain symlinks: {path}")
        if path.is_file() and path.name != "payload_manifest.json":
            files.append(path)
    return files


def build_payload_manifest(target: str | Path) -> dict[str, Any]:
    directory = Path(target)
    files = [
        {
            "path": path.relative_to(directory).as_posix(),
            "size": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in _safe_payload_files(directory)
    ]
    return {"schema_version": PAYLOAD_MANIFEST_SCHEMA_VERSION, "files": files}


def validate_payload_manifest(target: str | Path) -> str:
    directory = Path(target)
    manifest_path = directory / "payload_manifest.json"
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise TrainingArtifactError(f"artifact lacks payload_manifest.json: {directory}")
    stored = load_json(manifest_path)
    if not isinstance(stored, dict) or stored.get("schema_version") != PAYLOAD_MANIFEST_SCHEMA_VERSION:
        raise TrainingArtifactError(f"invalid payload manifest schema in {directory}")
    if stored != build_payload_manifest(directory):
        raise TrainingArtifactError(f"payload manifest mismatch in {directory}")
    return sha256_file(manifest_path)


def new_staging_directory(parent: str | Path, artifact_id: str) -> Path:
    target_parent = Path(parent)
    target_parent.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=f".{artifact_id}.", dir=target_parent))


def finalize_target_atomic(
    staging: str | Path,
    target: str | Path,
    *,
    validate_staging: Callable[[Path], Any] | None = None,
) -> str:
    """Publish a staged target without ever overwriting an existing lifecycle ID."""

    temporary = Path(staging)
    destination = Path(target)
    if temporary.parent.resolve() != destination.parent.resolve():
        raise TrainingArtifactError("staging and final targets must share a parent filesystem")
    write_canonical_json(
        temporary / "payload_manifest.json", build_payload_manifest(temporary)
    )
    staged_hash = validate_payload_manifest(temporary)
    if validate_staging is not None:
        validate_staging(temporary)
        if validate_payload_manifest(temporary) != staged_hash:
            raise TrainingArtifactError("staging validator mutated the immutable target")
    if destination.exists():
        existing_hash = validate_payload_manifest(destination)
        if existing_hash != staged_hash:
            raise TrainingArtifactError(
                f"lifecycle ID collision: {destination.name} has a different payload"
            )
        shutil.rmtree(temporary)
        return existing_hash
    os.replace(temporary, destination)
    return staged_hash


def write_locator_ref(
    ref_path: str | Path,
    *,
    artifact_kind: str,
    artifact_id: str,
    target: str | Path,
    payload_manifest_sha256: str,
) -> dict[str, Any]:
    destination = Path(target).resolve()
    if not destination.is_dir() or destination.name != artifact_id:
        raise TrainingArtifactError("locator target or artifact ID is invalid")
    if not SHA256_RE.fullmatch(payload_manifest_sha256):
        raise TrainingArtifactError("locator payload hash is invalid")
    if validate_payload_manifest(destination) != payload_manifest_sha256:
        raise TrainingArtifactError("locator payload hash does not match target")
    locator = {
        "schema_version": LOCATOR_REF_SCHEMA_VERSION,
        "artifact_kind": artifact_kind,
        "artifact_id": artifact_id,
        "target_path": str(destination),
        "payload_manifest_sha256": payload_manifest_sha256,
    }
    write_canonical_json(ref_path, locator)
    return locator


def resolve_locator_ref(
    ref_path: str | Path, expected_kind: str | Sequence[str] | None = None
) -> tuple[dict[str, Any], Path]:
    locator = load_json(ref_path)
    required = {
        "schema_version",
        "artifact_kind",
        "artifact_id",
        "target_path",
        "payload_manifest_sha256",
    }
    if not isinstance(locator, dict) or set(locator) != required:
        raise TrainingArtifactError(f"malformed locator ref: {ref_path}")
    if locator.get("schema_version") != LOCATOR_REF_SCHEMA_VERSION:
        raise TrainingArtifactError(f"unsupported locator ref schema: {ref_path}")
    artifact_kind = locator.get("artifact_kind")
    if not isinstance(artifact_kind, str) or not artifact_kind:
        raise TrainingArtifactError(f"invalid artifact kind in locator: {ref_path}")
    if expected_kind is not None:
        expected = {expected_kind} if isinstance(expected_kind, str) else set(expected_kind)
        if artifact_kind not in expected:
            raise TrainingArtifactError(
                f"expected artifact kind {sorted(expected)!r}, got {artifact_kind!r}"
            )
    artifact_id = locator.get("artifact_id")
    target_path = locator.get("target_path")
    payload_hash = locator.get("payload_manifest_sha256")
    if not isinstance(artifact_id, str) or not artifact_id:
        raise TrainingArtifactError("locator artifact ID is invalid")
    if not isinstance(target_path, str) or not Path(target_path).is_absolute():
        raise TrainingArtifactError("locator target path must be absolute")
    if not isinstance(payload_hash, str) or not SHA256_RE.fullmatch(payload_hash):
        raise TrainingArtifactError("locator payload hash is invalid")
    target = Path(target_path)
    if not target.is_dir() or target.is_symlink() or target.name != artifact_id:
        raise TrainingArtifactError("locator target is missing or has the wrong artifact ID")
    if validate_payload_manifest(target) != payload_hash:
        raise TrainingArtifactError("locator payload hash does not match target")
    return dict(locator), target


def validate_dependency_ref(
    dependency: Mapping[str, Any], *, expected_kind: str | Sequence[str] | None = None
) -> dict[str, Any]:
    required = {
        "schema_version",
        "artifact_kind",
        "artifact_id",
        "payload_manifest_sha256",
        "logical_repo_path",
    }
    if set(dependency) != required:
        raise TrainingArtifactError("embedded dependency ref has non-canonical fields")
    if dependency.get("schema_version") != DEPENDENCY_REF_SCHEMA_VERSION:
        raise TrainingArtifactError("embedded ref is not a portable dependency")
    kind = dependency.get("artifact_kind")
    if not isinstance(kind, str) or not kind:
        raise TrainingArtifactError("embedded dependency artifact kind is invalid")
    if expected_kind is not None:
        expected = {expected_kind} if isinstance(expected_kind, str) else set(expected_kind)
        if kind not in expected:
            raise TrainingArtifactError(
                f"expected dependency kind {sorted(expected)!r}, got {kind!r}"
            )
    artifact_id = dependency.get("artifact_id")
    payload_hash = dependency.get("payload_manifest_sha256")
    logical_path = dependency.get("logical_repo_path")
    if not isinstance(artifact_id, str) or not artifact_id:
        raise TrainingArtifactError("embedded dependency artifact ID is invalid")
    if not isinstance(payload_hash, str) or not SHA256_RE.fullmatch(payload_hash):
        raise TrainingArtifactError("embedded dependency payload hash is invalid")
    if not isinstance(logical_path, str) or not logical_path:
        raise TrainingArtifactError("embedded dependency logical path is invalid")
    path = Path(logical_path)
    if path.is_absolute() or ".." in path.parts or logical_path != path.as_posix():
        raise TrainingArtifactError("embedded dependency logical path is not portable")
    return dict(dependency)


def portable_dependency(
    locator: Mapping[str, Any], target: str | Path, workspace_root: str | Path
) -> dict[str, Any]:
    directory = Path(target).resolve()
    root = Path(workspace_root).resolve()
    try:
        logical_path = directory.relative_to(root).as_posix()
    except ValueError as exc:
        raise TrainingArtifactError("artifact target must live below the workspace root") from exc
    dependency = {
        "schema_version": DEPENDENCY_REF_SCHEMA_VERSION,
        "artifact_kind": locator["artifact_kind"],
        "artifact_id": locator["artifact_id"],
        "payload_manifest_sha256": locator["payload_manifest_sha256"],
        "logical_repo_path": logical_path,
    }
    return validate_dependency_ref(dependency)


def resolve_dependency_target(
    dependency: Mapping[str, Any], workspace_root: str | Path
) -> Path:
    """Resolve and verify a portable dependency below ``workspace_root``."""

    frozen = validate_dependency_ref(dependency)
    root = Path(workspace_root).resolve()
    target = (root / frozen["logical_repo_path"]).resolve()
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise TrainingArtifactError("dependency target escapes the workspace root") from exc
    if not target.is_dir() or target.is_symlink():
        raise TrainingArtifactError("dependency target does not exist")
    if target.name != frozen["artifact_id"]:
        raise TrainingArtifactError("dependency artifact ID does not match target directory")
    if validate_payload_manifest(target) != frozen["payload_manifest_sha256"]:
        raise TrainingArtifactError("dependency payload hash does not match target")
    return target


def ensure_exact_file_set(target: str | Path, expected: set[str]) -> None:
    directory = Path(target)
    actual = {
        path.relative_to(directory).as_posix()
        for path in directory.rglob("*")
        if path.is_file()
    }
    if actual != expected:
        raise TrainingArtifactError(
            f"artifact file set mismatch: missing={sorted(expected-actual)}, "
            f"extra={sorted(actual-expected)}"
        )


@lru_cache(maxsize=64)
def _compiled_json_schema(
    resolved_path: str, mtime_ns: int, size: int
) -> Any:
    """Compile one immutable schema revision once per process.

    ``mtime_ns`` and ``size`` are cache-key inputs on purpose: a schema edit in
    a long-lived process cannot silently reuse an older validator.  Stage 1
    validators call this function once per record, so compiling Draft 2020-12
    thousands of times would otherwise dominate validation time.
    """

    del mtime_ns, size
    try:
        import jsonschema
    except ImportError as exc:  # fail closed by design
        raise TrainingArtifactError(
            "jsonschema is required for Stage 1 artifact validation"
        ) from exc
    schema = load_json(resolved_path)
    validator_class = jsonschema.validators.validator_for(schema)
    try:
        validator_class.check_schema(schema)
    except jsonschema.SchemaError as exc:
        raise TrainingArtifactError(
            f"invalid JSON schema {resolved_path}: {exc.message}"
        ) from exc
    return validator_class(schema)


def validate_json_schema(document: Mapping[str, Any], schema_path: str | Path) -> None:
    try:
        import jsonschema
    except ImportError as exc:  # fail closed by design
        raise TrainingArtifactError(
            "jsonschema is required for Stage 1 artifact validation"
        ) from exc
    source = Path(schema_path).resolve()
    try:
        stat_result = source.stat()
        validator = _compiled_json_schema(
            str(source), stat_result.st_mtime_ns, stat_result.st_size
        )
        validator.validate(document)
    except OSError as exc:
        raise TrainingArtifactError(
            f"cannot inspect JSON schema {source}: {exc}"
        ) from exc
    except jsonschema.ValidationError as exc:
        location = "/".join(str(item) for item in exc.absolute_path) or "<root>"
        raise TrainingArtifactError(
            f"schema validation failed at {location}: {exc.message}"
        ) from exc


__all__ = [
    "DEPENDENCY_REF_SCHEMA_VERSION",
    "LOCATOR_REF_SCHEMA_VERSION",
    "PAYLOAD_MANIFEST_SCHEMA_VERSION",
    "SHA256_RE",
    "TrainingArtifactError",
    "build_payload_manifest",
    "canonical_json_bytes",
    "canonical_jsonl_bytes",
    "canonical_sha256",
    "ensure_exact_file_set",
    "finalize_target_atomic",
    "load_json",
    "load_jsonl",
    "new_staging_directory",
    "portable_dependency",
    "resolve_dependency_target",
    "resolve_locator_ref",
    "sha256_file",
    "validate_dependency_ref",
    "validate_json_schema",
    "validate_payload_manifest",
    "write_canonical_json",
    "write_canonical_jsonl",
    "write_bytes_atomic",
    "write_locator_ref",
]
