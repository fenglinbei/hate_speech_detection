"""Immutable, content-addressed Stage 1 model and registry lifecycle.

The artifacts created here contain metadata, not copied model weights.  Every
source tree is represented by a portable workspace-relative path and a full
regular-file inventory.  Validation and runtime resolution always re-hash the
source trees, so a locator cannot silently drift to another checkpoint.

No function in this module imports transformers or loads a model.
"""

from __future__ import annotations

import copy
import fcntl
import hashlib
import importlib.metadata
import math
import os
import platform
import re
import shutil
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from data.training_artifacts import (
    TrainingArtifactError,
    canonical_json_bytes,
    canonical_sha256,
    ensure_exact_file_set,
    finalize_target_atomic,
    load_json,
    load_jsonl,
    new_staging_directory,
    portable_dependency,
    resolve_dependency_target,
    resolve_locator_ref,
    sha256_file,
    validate_dependency_ref,
    validate_json_schema,
    validate_payload_manifest,
    write_canonical_json,
    write_locator_ref,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
MODEL_SCHEMA = REPOSITORY_ROOT / "schemas/stage1_model_artifact_v1.schema.json"
RECEIPT_ARTIFACT_SCHEMA = (
    REPOSITORY_ROOT / "schemas/stage1_training_receipt_artifact_v1.schema.json"
)
RECEIPT_SCHEMA = REPOSITORY_ROOT / "schemas/stage1_training_receipt_v1.schema.json"
REGISTRY_SCHEMA = REPOSITORY_ROOT / "schemas/stage1_model_registry_v1.schema.json"
ENVIRONMENT_SCHEMA = REPOSITORY_ROOT / "schemas/stage1_environment_manifest_v1.schema.json"

MODEL_ARTIFACT_KIND = "stage1-model"
RECEIPT_ARTIFACT_KIND = "stage1-training-receipt"
REGISTRY_ARTIFACT_KIND = "stage1-model-registry"
MODEL_SCHEMA_VERSION = "stage1-model-artifact/v1"
RECEIPT_ARTIFACT_SCHEMA_VERSION = "stage1-training-receipt-artifact/v1"
REGISTRY_SCHEMA_VERSION = "stage1-model-registry/v1"
TREE_SCHEMA_VERSION = "stage1-regular-file-tree/v1"
MODEL_ID_PREFIX = "mdl-"
RECEIPT_ID_PREFIX = "trc-"
REGISTRY_ID_PREFIX = "mreg-"
LEGACY_MODEL_KEY = "M_legacy/smoke"
LEGACY_MODEL_ROLE = "legacy-smoke-only"
MODEL_KEY_RE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.-]*/[A-Za-z0-9_][A-Za-z0-9_.-]*$")
SHARD_RE = re.compile(r"^(?P<prefix>.+)-(?P<index>[0-9]{5})-of-(?P<count>[0-9]{5})\.(?:safetensors|bin)$")
TOKENIZER_FILENAMES = {
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "vocab.txt",
    "merges.txt",
    "special_tokens_map.json",
    "added_tokens.json",
    "chat_template.jinja",
    "tokenizer.model",
    "sentencepiece.bpe.model",
    "spiece.model",
}
ENVIRONMENT_ID_INPUT_KEYS = (
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


class ModelRegistryError(TrainingArtifactError):
    """Raised before a mutable or inconsistent model can enter Stage 1."""


@dataclass(frozen=True)
class ResolvedModelSourceContract:
    """Immutable expected inventories used to guard a real backend load.

    The contract contains no caller-selected load paths.  Every path is
    resolved again from a workspace-relative inventory immediately before a
    backend reads it.
    """

    workspace_root: Path
    checkpoint_inventory: Mapping[str, Any]
    tokenizer_inventory: Mapping[str, Any]
    base_inventory: Mapping[str, Any]


@dataclass(frozen=True)
class VerifiedModelSourcePaths:
    """Load roots made available only inside a verified source-tree lease."""

    checkpoint_path: Path
    tokenizer_path: Path
    base_model_path: Path


@dataclass(frozen=True)
class ResolvedRegisteredModel:
    """Paths returned only after registry and source-tree revalidation."""

    registry_id: str
    model_key: str
    role: str
    seed: int | None
    checkpoint_format: str
    checkpoint_path: Path
    tokenizer_path: Path
    base_model_path: Path
    tokenizer_revision: str
    tokenizer_content_revision: str
    model_artifact_id: str
    scientific_eligible: bool
    source_contract: ResolvedModelSourceContract


def _logical_directory(path: str | Path, workspace_root: str | Path, *, label: str) -> tuple[Path, str]:
    root = Path(workspace_root).resolve()
    candidate = Path(path)
    if candidate.is_symlink():
        raise ModelRegistryError(f"{label} cannot be a symlink: {candidate}")
    source = candidate.resolve()
    try:
        logical = source.relative_to(root).as_posix()
    except ValueError as exc:
        raise ModelRegistryError(f"{label} must be below workspace_root") from exc
    if not logical or logical == ".":
        raise ModelRegistryError(f"{label} cannot be the workspace root")
    if not source.is_dir() or source.is_symlink():
        raise ModelRegistryError(f"{label} is not a regular directory: {source}")
    return source, logical


def _scan_tree(root: Path) -> tuple[list[str], dict[str, tuple[int, int, int, int]]]:
    """Return sorted files and stability signatures, rejecting every non-regular entry."""

    if not root.is_dir() or root.is_symlink():
        raise ModelRegistryError(f"model source is not a regular directory: {root}")
    files: list[str] = []
    signatures: dict[str, tuple[int, int, int, int]] = {}
    for current, directory_names, file_names in os.walk(root, topdown=True, followlinks=False):
        current_path = Path(current)
        for name in sorted(directory_names):
            path = current_path / name
            status = os.lstat(path)
            if stat.S_ISLNK(status.st_mode):
                raise ModelRegistryError(f"model source cannot contain symlinks: {path}")
            if not stat.S_ISDIR(status.st_mode):
                raise ModelRegistryError(f"model source contains a non-directory entry: {path}")
        for name in sorted(file_names):
            path = current_path / name
            relative = path.relative_to(root).as_posix()
            status = os.lstat(path)
            if stat.S_ISLNK(status.st_mode):
                raise ModelRegistryError(f"model source cannot contain symlinks: {path}")
            if not stat.S_ISREG(status.st_mode):
                raise ModelRegistryError(f"model source contains a non-regular file: {path}")
            files.append(relative)
            signatures[relative] = (
                int(status.st_dev),
                int(status.st_ino),
                int(status.st_size),
                int(status.st_mtime_ns),
            )
    files.sort()
    if not files:
        raise ModelRegistryError(f"model source tree is empty: {root}")
    return files, signatures


def _stable_sha256(path: Path, expected: tuple[int, int, int, int]) -> tuple[int, str]:
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ModelRegistryError(f"cannot open regular model file {path}: {exc}") from exc
    digest = hashlib.sha256()
    try:
        before = os.fstat(descriptor)
        before_signature = (
            int(before.st_dev),
            int(before.st_ino),
            int(before.st_size),
            int(before.st_mtime_ns),
        )
        if not stat.S_ISREG(before.st_mode) or before_signature != expected:
            raise ModelRegistryError(f"model file changed while inventory started: {path}")
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
        after = os.fstat(descriptor)
        after_signature = (
            int(after.st_dev),
            int(after.st_ino),
            int(after.st_size),
            int(after.st_mtime_ns),
        )
        if after_signature != before_signature:
            raise ModelRegistryError(f"model file changed while hashing: {path}")
    finally:
        os.close(descriptor)
    return expected[2], digest.hexdigest()


def inventory_regular_file_tree(
    path: str | Path,
    *,
    workspace_root: str | Path,
    label: str = "model source",
    inventory_policy: str = "all-regular-files/v1",
) -> dict[str, Any]:
    """Hash a stable regular-file tree and return a portable full inventory."""

    root, logical = _logical_directory(path, workspace_root, label=label)
    paths, before = _scan_tree(root)
    if inventory_policy == "tokenizer-files/v1":
        selected_paths = [path for path in paths if Path(path).name in TOKENIZER_FILENAMES]
    elif inventory_policy == "all-regular-files/v1":
        selected_paths = paths
    else:
        raise ModelRegistryError(f"unsupported inventory policy: {inventory_policy}")
    if not selected_paths:
        raise ModelRegistryError(f"{label} has no files selected by {inventory_policy}")
    files = []
    for relative in selected_paths:
        size, digest = _stable_sha256(root / relative, before[relative])
        files.append({"path": relative, "size": size, "sha256": digest})
    after_paths, after = _scan_tree(root)
    if paths != after_paths or before != after:
        raise ModelRegistryError(f"{label} changed while its inventory was being hashed")
    return {
        "schema_version": TREE_SCHEMA_VERSION,
        "inventory_policy": inventory_policy,
        "logical_repo_path": logical,
        "file_count": len(files),
        "total_bytes": sum(item["size"] for item in files),
        "files": files,
        "file_tree_sha256": canonical_sha256(files),
    }


def _validate_tree_shape(snapshot: Mapping[str, Any]) -> None:
    if snapshot.get("schema_version") != TREE_SCHEMA_VERSION:
        raise ModelRegistryError("source inventory has the wrong schema")
    if snapshot.get("inventory_policy") not in {
        "all-regular-files/v1",
        "tokenizer-files/v1",
    }:
        raise ModelRegistryError("source inventory has an invalid inventory policy")
    logical = snapshot.get("logical_repo_path")
    if not isinstance(logical, str) or not logical:
        raise ModelRegistryError("source inventory lacks a logical path")
    logical_path = Path(logical)
    if logical_path.is_absolute() or ".." in logical_path.parts or logical_path.as_posix() != logical:
        raise ModelRegistryError("source inventory logical path is not portable")
    files = snapshot.get("files")
    if not isinstance(files, list) or not files:
        raise ModelRegistryError("source inventory cannot be empty")
    expected_paths: list[str] = []
    total = 0
    for item in files:
        if not isinstance(item, Mapping) or set(item) != {"path", "size", "sha256"}:
            raise ModelRegistryError("source inventory file row is malformed")
        relative = item.get("path")
        path = Path(str(relative))
        if (
            not isinstance(relative, str)
            or not relative
            or path.is_absolute()
            or ".." in path.parts
            or path.as_posix() != relative
        ):
            raise ModelRegistryError("source inventory file path is not portable")
        size = item.get("size")
        digest = item.get("sha256")
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise ModelRegistryError("source inventory has an invalid file size")
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ModelRegistryError("source inventory has an invalid file digest")
        expected_paths.append(relative)
        total += size
    if expected_paths != sorted(expected_paths) or len(expected_paths) != len(set(expected_paths)):
        raise ModelRegistryError("source inventory paths are not sorted and unique")
    if snapshot.get("file_count") != len(files) or snapshot.get("total_bytes") != total:
        raise ModelRegistryError("source inventory count or byte total mismatch")
    if snapshot.get("file_tree_sha256") != canonical_sha256(files):
        raise ModelRegistryError("source inventory tree digest mismatch")


def _resolve_tree_path(
    snapshot: Mapping[str, Any],
    workspace_root: str | Path,
) -> Path:
    _validate_tree_shape(snapshot)
    root = Path(workspace_root).resolve()
    candidate = root / str(snapshot["logical_repo_path"])
    if candidate.is_symlink():
        raise ModelRegistryError("model source path became a symlink")
    target = candidate.resolve()
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise ModelRegistryError("model source escapes workspace_root") from exc
    return target


def _resolve_and_rehash_tree(
    snapshot: Mapping[str, Any],
    workspace_root: str | Path,
    rehash_cache: dict[str, Path] | None = None,
) -> Path:
    _validate_tree_shape(snapshot)
    cache_key = canonical_sha256(snapshot)
    if rehash_cache is not None and cache_key in rehash_cache:
        return rehash_cache[cache_key]
    target = _resolve_tree_path(snapshot, workspace_root)
    current = inventory_regular_file_tree(
        target,
        workspace_root=workspace_root,
        label=str(snapshot["logical_repo_path"]),
        inventory_policy=str(snapshot["inventory_policy"]),
    )
    if current != dict(snapshot):
        raise ModelRegistryError(
            f"model source inventory changed: {snapshot['logical_repo_path']}"
        )
    if rehash_cache is not None:
        rehash_cache[cache_key] = target
    return target


_SOURCE_NAMES = ("checkpoint", "tokenizer", "base")


def _source_contract_snapshot(
    contract: ResolvedModelSourceContract, source_name: str
) -> Mapping[str, Any]:
    if not isinstance(contract, ResolvedModelSourceContract):
        raise ModelRegistryError("registered model lacks a typed source contract")
    snapshots = {
        "checkpoint": contract.checkpoint_inventory,
        "tokenizer": contract.tokenizer_inventory,
        "base": contract.base_inventory,
    }
    try:
        snapshot = snapshots[source_name]
    except KeyError as exc:  # pragma: no cover - guarded by public normalizer
        raise ModelRegistryError(f"unsupported model source lease member: {source_name}") from exc
    if not isinstance(snapshot, Mapping):
        raise ModelRegistryError(f"model source contract {source_name} inventory is malformed")
    _validate_tree_shape(snapshot)
    return snapshot


def _normalized_source_names(source_names: Sequence[str]) -> tuple[str, ...]:
    names = tuple(source_names)
    if not names or len(names) != len(set(names)) or any(
        name not in _SOURCE_NAMES for name in names
    ):
        raise ModelRegistryError("model source lease members must be unique known sources")
    return tuple(name for name in _SOURCE_NAMES if name in names)


def _lease_inventory_identity(snapshot: Mapping[str, Any]) -> bytes:
    """Return the exact, collision-free grouping identity for one inventory.

    Lease work may be shared only when the complete canonical inventory is
    identical.  In particular, the canonical bytes bind the logical path,
    inventory policy, tree digest, counts, byte total, and every file row; a
    common physical path or a common tree digest alone is never sufficient.
    """

    _validate_tree_shape(snapshot)
    return canonical_json_bytes(snapshot)


def _lease_stat_signature(status: os.stat_result) -> tuple[int, ...]:
    """Metadata that changes on content, namespace, permission, or link drift."""

    return (
        int(status.st_dev),
        int(status.st_ino),
        int(status.st_mode),
        int(status.st_nlink),
        int(status.st_size),
        int(status.st_mtime_ns),
        int(status.st_ctime_ns),
    )


def _capture_source_lease_signature(
    target: Path,
    snapshot: Mapping[str, Any],
    *,
    workspace_root: Path,
) -> dict[tuple[str, str], tuple[int, ...]]:
    """Capture path and inode state without following any symlink.

    ``ctime`` and parent-directory state are intentionally retained in this
    runtime-only signature.  A shard that is swapped in for loading and then
    restored has the same final digest, but it cannot restore inode/directory
    ctime as an unprivileged writer.
    """

    root = workspace_root.resolve()
    logical = Path(str(snapshot["logical_repo_path"]))
    lexical_target = root / logical
    if lexical_target.resolve() != target:
        raise ModelRegistryError("model source lease root no longer resolves consistently")
    if target.is_symlink() or not target.is_dir():
        raise ModelRegistryError("model source lease root is missing or a symlink")
    signature: dict[tuple[str, str], tuple[int, ...]] = {}
    ancestor = root
    ancestor_status = os.lstat(ancestor)
    if not stat.S_ISDIR(ancestor_status.st_mode) or stat.S_ISLNK(ancestor_status.st_mode):
        raise ModelRegistryError("model source workspace root changed or became a symlink")
    signature[("ancestor", ".")] = _lease_stat_signature(ancestor_status)
    for part in logical.parts[:-1]:
        ancestor = ancestor / part
        ancestor_status = os.lstat(ancestor)
        if not stat.S_ISDIR(ancestor_status.st_mode) or stat.S_ISLNK(
            ancestor_status.st_mode
        ):
            raise ModelRegistryError(
                f"model source ancestor changed or became a symlink: {ancestor}"
            )
        signature[("ancestor", ancestor.relative_to(root).as_posix())] = (
            _lease_stat_signature(ancestor_status)
        )
    signature[("directory", ".")] = _lease_stat_signature(os.lstat(target))
    directories: set[Path] = {Path(".")}
    for item in snapshot["files"]:
        relative = Path(str(item["path"]))
        parent = relative.parent
        while parent != Path("."):
            directories.add(parent)
            parent = parent.parent
    for relative in sorted(directories, key=lambda value: value.as_posix()):
        if relative == Path("."):
            continue
        status = os.lstat(target / relative)
        if not stat.S_ISDIR(status.st_mode) or stat.S_ISLNK(status.st_mode):
            raise ModelRegistryError(
                f"model source lease directory changed: {target / relative}"
            )
        signature[("directory", relative.as_posix())] = _lease_stat_signature(status)
    for item in snapshot["files"]:
        relative = str(item["path"])
        path = target / relative
        status = os.lstat(path)
        if not stat.S_ISREG(status.st_mode) or stat.S_ISLNK(status.st_mode):
            raise ModelRegistryError(f"model source lease file changed: {path}")
        signature[("file", relative)] = _lease_stat_signature(status)
    return signature


def _open_source_lease_descriptors(
    targets: Mapping[str, Path],
    snapshots: Mapping[str, Mapping[str, Any]],
    signatures: Mapping[str, Mapping[tuple[str, str], tuple[int, ...]]],
    *,
    workspace_root: Path,
) -> list[int]:
    """Pin verified inodes and take cooperative read locks for the load window."""

    descriptors: list[int] = []
    opened: set[Path] = set()
    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        for source_name, target in targets.items():
            root = workspace_root.resolve()
            directory_flags = flags
            if hasattr(os, "O_DIRECTORY"):
                directory_flags |= os.O_DIRECTORY
            ancestor_entries = sorted(
                (
                    (relative, signature)
                    for (kind, relative), signature in signatures[source_name].items()
                    if kind == "ancestor"
                ),
                key=lambda item: (len(Path(item[0]).parts), item[0]),
            )
            for relative, expected_signature in ancestor_entries:
                ancestor = root if relative == "." else root / relative
                if ancestor in opened:
                    continue
                descriptor = os.open(ancestor, directory_flags)
                descriptors.append(descriptor)
                fcntl.flock(descriptor, fcntl.LOCK_SH | fcntl.LOCK_NB)
                if _lease_stat_signature(os.fstat(descriptor)) != expected_signature:
                    raise ModelRegistryError(
                        f"model source ancestor changed while lease opened: {ancestor}"
                    )
                opened.add(ancestor)
            lexical_target = root / str(snapshots[source_name]["logical_repo_path"])
            if lexical_target not in opened:
                descriptor = os.open(lexical_target, directory_flags)
                descriptors.append(descriptor)
                fcntl.flock(descriptor, fcntl.LOCK_SH | fcntl.LOCK_NB)
                if _lease_stat_signature(os.fstat(descriptor)) != signatures[source_name][
                    ("directory", ".")
                ]:
                    raise ModelRegistryError("model source root changed while lease opened")
                opened.add(lexical_target)
            source_directory_entries = sorted(
                (
                    (relative, signature)
                    for (kind, relative), signature in signatures[source_name].items()
                    if kind == "directory" and relative != "."
                ),
                key=lambda item: (len(Path(item[0]).parts), item[0]),
            )
            for relative, expected_signature in source_directory_entries:
                directory = lexical_target / relative
                if directory in opened:
                    continue
                descriptor = os.open(directory, directory_flags)
                descriptors.append(descriptor)
                fcntl.flock(descriptor, fcntl.LOCK_SH | fcntl.LOCK_NB)
                if _lease_stat_signature(os.fstat(descriptor)) != expected_signature:
                    raise ModelRegistryError(
                        f"model source directory changed while lease opened: {directory}"
                    )
                opened.add(directory)
            for item in snapshots[source_name]["files"]:
                relative = str(item["path"])
                path = target / relative
                if path in opened:
                    continue
                try:
                    path.relative_to(target)
                except ValueError as exc:  # pragma: no cover - shape checks guard this
                    raise ModelRegistryError("model source lease file escaped its root") from exc
                descriptor = os.open(path, flags)
                descriptors.append(descriptor)
                fcntl.flock(descriptor, fcntl.LOCK_SH | fcntl.LOCK_NB)
                if _lease_stat_signature(os.fstat(descriptor)) != signatures[source_name][
                    ("file", relative)
                ]:
                    raise ModelRegistryError(
                        f"model source file changed while lease opened: {path}"
                    )
                opened.add(path)
    except Exception:
        for descriptor in reversed(descriptors):
            os.close(descriptor)
        raise
    return descriptors


class VerifiedModelSourceLease:
    """Fresh-hash and monitor registered load roots for one backend load.

    This does not pretend POSIX paths are immutable.  It establishes a narrow
    fail-closed verification window: fresh full inventories immediately before
    the backend load, pinned read descriptors plus inode/ctime signatures while
    it runs, and fresh full inventories immediately after it returns.  Writers
    that cooperate with ``flock`` are excluded; non-cooperating writes are
    detected by the post-load hash/signature checks, including swap-and-restore
    namespace changes.
    """

    def __init__(
        self,
        contract: ResolvedModelSourceContract,
        *,
        source_names: Sequence[str] = _SOURCE_NAMES,
    ) -> None:
        self._contract = contract
        self._source_names = _normalized_source_names(source_names)
        self._targets: dict[str, Path] = {}
        self._snapshots: dict[str, Mapping[str, Any]] = {}
        self._signatures: dict[
            str, dict[tuple[str, str], tuple[int, ...]]
        ] = {}
        self._representatives: list[str] = []
        self._descriptors: list[int] = []
        self._entered = False

    def __enter__(self) -> VerifiedModelSourcePaths:
        if self._entered:
            raise ModelRegistryError("model source lease cannot be re-entered")
        root = Path(self._contract.workspace_root).resolve()
        identities: dict[bytes, str] = {}
        self._targets = {}
        self._snapshots = {}
        self._signatures = {}
        self._representatives = []
        try:
            for source_name in self._source_names:
                snapshot = _source_contract_snapshot(self._contract, source_name)
                self._snapshots[source_name] = snapshot
                identity = _lease_inventory_identity(snapshot)
                representative = identities.get(identity)
                if representative is None:
                    representative = source_name
                    identities[identity] = representative
                    self._representatives.append(representative)
                    target = _resolve_tree_path(snapshot, root)
                    self._targets[representative] = target
                    self._signatures[representative] = (
                        _capture_source_lease_signature(
                            target, snapshot, workspace_root=root
                        )
                    )
                else:
                    target = self._targets[representative]
                self._targets[source_name] = target
            # Deliberately pass no cache: every distinct inventory in every
            # lease is a fresh on-disk proof.  Exact source aliases share only
            # within this one lease.
            for source_name in self._representatives:
                current = _resolve_and_rehash_tree(
                    self._snapshots[source_name], root, rehash_cache=None
                )
                if current != self._targets[source_name]:  # pragma: no cover
                    raise ModelRegistryError("model source lease resolved inconsistent roots")
            for source_name in self._representatives:
                if (
                    _capture_source_lease_signature(
                        self._targets[source_name],
                        self._snapshots[source_name],
                        workspace_root=root,
                    )
                    != self._signatures[source_name]
                ):
                    raise ModelRegistryError(
                        f"model source changed before backend load: {source_name}"
                    )
            # Open one descriptor/lock set per distinct inventory.  Separate
            # snapshots intentionally do not share even when their lexical
            # roots or some selected files overlap.
            for source_name in self._representatives:
                self._descriptors.extend(
                    _open_source_lease_descriptors(
                        {source_name: self._targets[source_name]},
                        {source_name: self._snapshots[source_name]},
                        {source_name: self._signatures[source_name]},
                        workspace_root=root,
                    )
                )
            for source_name in self._representatives:
                if (
                    _capture_source_lease_signature(
                        self._targets[source_name],
                        self._snapshots[source_name],
                        workspace_root=root,
                    )
                    != self._signatures[source_name]
                ):
                    raise ModelRegistryError(
                        f"model source changed while backend lease opened: {source_name}"
                    )
        except Exception:
            self._close_descriptors()
            raise
        self._entered = True
        checkpoint = _resolve_tree_path(self._contract.checkpoint_inventory, root)
        tokenizer = _resolve_tree_path(self._contract.tokenizer_inventory, root)
        base = _resolve_tree_path(self._contract.base_inventory, root)
        return VerifiedModelSourcePaths(
            checkpoint_path=checkpoint,
            tokenizer_path=tokenizer,
            base_model_path=base,
        )

    def _close_descriptors(self) -> None:
        for descriptor in reversed(self._descriptors):
            try:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            finally:
                os.close(descriptor)
        self._descriptors = []

    def _post_load_verify(self) -> None:
        errors: list[str] = []
        root = Path(self._contract.workspace_root).resolve()
        for source_name in self._representatives:
            try:
                current_signature = _capture_source_lease_signature(
                    self._targets[source_name],
                    self._snapshots[source_name],
                    workspace_root=root,
                )
                if current_signature != self._signatures[source_name]:
                    errors.append(f"{source_name} inode/ctime signature changed")
            except Exception as exc:  # keep going so the fresh hash still runs
                errors.append(f"{source_name} signature check failed: {exc}")
        for source_name in self._representatives:
            try:
                _resolve_and_rehash_tree(
                    self._snapshots[source_name], root, rehash_cache=None
                )
            except Exception as exc:
                errors.append(f"{source_name} fresh inventory failed: {exc}")
        for source_name in self._representatives:
            try:
                current_signature = _capture_source_lease_signature(
                    self._targets[source_name],
                    self._snapshots[source_name],
                    workspace_root=root,
                )
                if current_signature != self._signatures[source_name]:
                    errors.append(f"{source_name} changed during post-load inventory")
            except Exception as exc:
                errors.append(f"{source_name} final signature check failed: {exc}")
        if errors:
            raise ModelRegistryError(
                "registered model source changed during backend load: "
                + "; ".join(errors)
            )

    def __exit__(self, exc_type: Any, exc: BaseException | None, traceback: Any) -> bool:
        verification_error: BaseException | None = None
        try:
            if self._entered:
                self._post_load_verify()
        except BaseException as caught:  # preserve fail-closed behavior on load errors too
            verification_error = caught
        finally:
            self._close_descriptors()
            self._entered = False
        if verification_error is not None:
            if exc is not None:
                raise verification_error from exc
            raise verification_error
        return False


def verified_model_source_lease(
    contract: ResolvedModelSourceContract,
    *,
    source_names: Sequence[str] = _SOURCE_NAMES,
) -> VerifiedModelSourceLease:
    """Create a one-shot verified load lease for registered model sources."""

    return VerifiedModelSourceLease(contract, source_names=source_names)


def _inventory_paths(snapshot: Mapping[str, Any]) -> set[str]:
    _validate_tree_shape(snapshot)
    return {str(item["path"]) for item in snapshot["files"]}


def _validate_shards(snapshot: Mapping[str, Any], *, checkpoint_format: str) -> None:
    paths = _inventory_paths(snapshot)
    root_name = str(snapshot["logical_repo_path"])
    indexes = sorted(
        path
        for path in paths
        if path.endswith(".index.json")
        and Path(path).name
        in {
            "model.safetensors.index.json",
            "pytorch_model.bin.index.json",
            "adapter_model.safetensors.index.json",
            "adapter_model.bin.index.json",
        }
    )
    if checkpoint_format in {"base", "full"}:
        if "config.json" not in paths:
            raise ModelRegistryError(f"full model tree lacks config.json: {root_name}")
        weight_candidates = {
            "model.safetensors",
            "pytorch_model.bin",
            "model.safetensors.index.json",
            "pytorch_model.bin.index.json",
        }
        if not paths.intersection(weight_candidates):
            raise ModelRegistryError(f"full model tree lacks supported weights: {root_name}")
    elif checkpoint_format == "adapter":
        if "adapter_config.json" not in paths:
            raise ModelRegistryError("adapter checkpoint lacks adapter_config.json")
        candidates = {
            "adapter_model.safetensors",
            "adapter_model.bin",
            "adapter_model.safetensors.index.json",
            "adapter_model.bin.index.json",
        }
        if not paths.intersection(candidates):
            raise ModelRegistryError("adapter checkpoint lacks supported adapter weights")
    else:
        raise ModelRegistryError(f"unsupported checkpoint format: {checkpoint_format}")

    # Contiguous shard names are independently required even when an index is
    # accidentally absent or incomplete.
    groups: dict[tuple[str, int], set[int]] = {}
    for relative in paths:
        match = SHARD_RE.fullmatch(Path(relative).name)
        if match is None:
            continue
        count = int(match.group("count"))
        index = int(match.group("index"))
        groups.setdefault((match.group("prefix"), count), set()).add(index)
    for (prefix, count), observed in groups.items():
        expected = set(range(1, count + 1))
        if observed != expected:
            raise ModelRegistryError(
                f"checkpoint shard set {prefix!r} is incomplete: "
                f"missing={sorted(expected-observed)} extra={sorted(observed-expected)}"
            )
    if indexes and not any(path.endswith(".safetensors") or path.endswith(".bin") for path in paths):
        raise ModelRegistryError("checkpoint index has no weight shard files")


def _validate_index_references(
    snapshot: Mapping[str, Any], *, workspace_root: str | Path
) -> None:
    paths = _inventory_paths(snapshot)
    root = (Path(workspace_root).resolve() / str(snapshot["logical_repo_path"])).resolve()
    for relative in sorted(paths):
        if Path(relative).name not in {
            "model.safetensors.index.json",
            "pytorch_model.bin.index.json",
            "adapter_model.safetensors.index.json",
            "adapter_model.bin.index.json",
        }:
            continue
        index = load_json(root / relative)
        if not isinstance(index, Mapping) or not isinstance(index.get("weight_map"), Mapping):
            raise ModelRegistryError(f"checkpoint index lacks weight_map: {relative}")
        referenced = set(index["weight_map"].values())
        if not referenced or any(not isinstance(value, str) for value in referenced):
            raise ModelRegistryError(f"checkpoint index has invalid shard references: {relative}")
        for value in referenced:
            path = Path(value)
            if path.is_absolute() or ".." in path.parts or path.as_posix() != value:
                raise ModelRegistryError(f"checkpoint index contains unsafe shard path: {value}")
        missing = sorted(referenced - paths)
        if missing:
            raise ModelRegistryError(
                f"checkpoint index {relative} references missing shards: {missing}"
            )


def _validate_tokenizer(snapshot: Mapping[str, Any]) -> None:
    paths = _inventory_paths(snapshot)
    if "tokenizer_config.json" not in paths:
        raise ModelRegistryError("tokenizer tree lacks tokenizer_config.json")
    vocabulary = {
        "tokenizer.json",
        "tokenizer.model",
        "spiece.model",
        "vocab.json",
        "vocab.txt",
    }
    if not paths.intersection(vocabulary):
        raise ModelRegistryError("tokenizer tree lacks a supported vocabulary artifact")


def _snapshot_cache(
    paths: Sequence[tuple[str | Path, str]],
    workspace_root: str | Path,
    *,
    tokenizer_inventory_policy: str = "all-regular-files/v1",
) -> dict[str, dict[str, Any]]:
    cached_by_path: dict[Path, dict[str, Any]] = {}
    result: dict[str, dict[str, Any]] = {}
    for path, label in paths:
        resolved, _ = _logical_directory(path, workspace_root, label=label)
        if label == "tokenizer" and resolved in cached_by_path:
            full = cached_by_path[resolved]
            if tokenizer_inventory_policy == "all-regular-files/v1":
                result[label] = copy.deepcopy(full)
                continue
            files = [
                copy.deepcopy(row)
                for row in full["files"]
                if Path(row["path"]).name in TOKENIZER_FILENAMES
            ]
            if not files:
                raise ModelRegistryError("tokenizer source has no tokenizer files")
            projected = {
                "schema_version": TREE_SCHEMA_VERSION,
                "inventory_policy": "tokenizer-files/v1",
                "logical_repo_path": full["logical_repo_path"],
                "file_count": len(files),
                "total_bytes": sum(row["size"] for row in files),
                "files": files,
                "file_tree_sha256": canonical_sha256(files),
            }
            result[label] = projected
            continue
        if resolved not in cached_by_path:
            policy = (
                tokenizer_inventory_policy
                if label == "tokenizer"
                else "all-regular-files/v1"
            )
            cached_by_path[resolved] = inventory_regular_file_tree(
                resolved,
                workspace_root=workspace_root,
                label=label,
                inventory_policy=policy,
            )
        result[label] = copy.deepcopy(cached_by_path[resolved])
    return result


def _dependency_projection(
    ref: str | Path,
    *,
    expected_kind: str | Sequence[str],
    workspace_root: str | Path,
) -> tuple[dict[str, Any], Path, dict[str, Any]]:
    locator, target = resolve_locator_ref(ref, expected_kind)
    return locator, target, portable_dependency(locator, target, workspace_root)


def _tokenizer_contract(
    snapshot: Mapping[str, Any],
    *,
    workspace_root: str | Path,
    tokenizer_revision: str,
    chat_template_source: str = "auto",
    require_enable_thinking: bool = True,
) -> dict[str, Any]:
    """Freeze the tokenizer revision and the non-thinking chat-template contract."""

    from data.training_schedule import tokenizer_revision_from_directory

    _validate_tree_shape(snapshot)
    if not isinstance(tokenizer_revision, str) or not tokenizer_revision:
        raise ModelRegistryError("tokenizer revision label cannot be empty")
    root = (Path(workspace_root).resolve() / str(snapshot["logical_repo_path"])).resolve()
    config_path = root / "tokenizer_config.json"
    config = load_json(config_path)
    if not isinstance(config, Mapping):
        raise ModelRegistryError("tokenizer_config.json must contain an object")
    source_aliases = {
        "auto": "auto",
        "tokenizer-config": "tokenizer_config.json",
        "tokenizer_config.json": "tokenizer_config.json",
        "chat-template-jinja": "chat_template.jinja",
        "chat_template.jinja": "chat_template.jinja",
    }
    selected_source = source_aliases.get(chat_template_source)
    if selected_source is None:
        raise ModelRegistryError(
            f"unsupported chat template source: {chat_template_source}"
        )
    config_template = config.get("chat_template")
    jinja_path = root / "chat_template.jinja"
    if selected_source == "tokenizer_config.json":
        template = config_template
        template_source = "tokenizer_config.json"
    elif selected_source == "chat_template.jinja":
        template = (
            jinja_path.read_text(encoding="utf-8") if jinja_path.is_file() else None
        )
        template_source = "chat_template.jinja"
    elif isinstance(config_template, str) and config_template:
        template = config_template
        template_source = "tokenizer_config.json"
    else:
        template = (
            jinja_path.read_text(encoding="utf-8") if jinja_path.is_file() else None
        )
        template_source = "chat_template.jinja"
    if not isinstance(template, str) or not template:
        raise ModelRegistryError(
            f"tokenizer lacks a non-empty chat template in {template_source}"
        )
    supports_generation_prompt = "add_generation_prompt" in template
    supports_enable_thinking = "enable_thinking" in template
    if not supports_generation_prompt:
        raise ModelRegistryError(
            "Stage 1 tokenizer template must expose add_generation_prompt"
        )
    if require_enable_thinking and not supports_enable_thinking:
        raise ModelRegistryError(
            "Stage 1 formal tokenizer template must expose enable_thinking"
        )
    eos = config.get("eos_token")
    pad = config.get("pad_token")
    if not isinstance(eos, (str, Mapping)) or not isinstance(pad, (str, Mapping)):
        raise ModelRegistryError("Stage 1 tokenizer must freeze EOS and PAD tokens")
    return {
        "schema_version": "stage1-tokenizer-contract/v1",
        "tokenizer_revision": tokenizer_revision,
        "tokenizer_content_revision": tokenizer_revision_from_directory(root),
        "tokenizer_config_sha256": sha256_file(config_path),
        "chat_template_source": template_source,
        "chat_template_sha256": hashlib.sha256(template.encode("utf-8")).hexdigest(),
        "supports_add_generation_prompt": supports_generation_prompt,
        "supports_enable_thinking_false": supports_enable_thinking,
        "eos_token": copy.deepcopy(eos),
        "pad_token": copy.deepcopy(pad),
    }


def _validate_environment_dependency(
    dependency: Mapping[str, Any],
    workspace_root: str | Path,
    *,
    validate_current_runtime: bool,
) -> dict[str, Any]:
    """Deep-validate the frozen environment and its current critical subset."""

    frozen = validate_dependency_ref(dependency, expected_kind="stage1-environment")
    target = resolve_dependency_target(frozen, workspace_root)
    ensure_exact_file_set(
        target, {"environment.json", "provenance.json", "payload_manifest.json"}
    )
    validate_payload_manifest(target)
    document = load_json(target / "environment.json")
    if not isinstance(document, Mapping):
        raise ModelRegistryError("environment.json is not an object")
    validate_json_schema(document, ENVIRONMENT_SCHEMA)
    expected_environment_id = "env-" + canonical_sha256(
        {key: document[key] for key in ENVIRONMENT_ID_INPUT_KEYS}
    )
    if document.get("environment_build_id") != expected_environment_id:
        raise ModelRegistryError("environment build ID is not reproducible")
    distributions = document.get("sorted_installed_distributions", [])
    python_versions = {
        row["name"]: row["version"]
        for row in distributions
        if isinstance(row, Mapping) and row.get("installer") == "python"
    }
    if document.get("critical_backend_versions") != {
        name: python_versions.get(name)
        for name in document.get("critical_backend_versions", {})
    }:
        raise ModelRegistryError("environment critical versions differ from package snapshot")
    provenance = load_json(target / "provenance.json")

    def _all_keys(value: Any) -> set[str]:
        if isinstance(value, Mapping):
            return set(value).union(*(_all_keys(item) for item in value.values()))
        if isinstance(value, list):
            return set().union(*(_all_keys(item) for item in value))
        return set()

    forbidden = {"hostname", "gpu_uuid", "captured_at", "timestamp", "job_id", "environment_prefix"}
    if forbidden.intersection(_all_keys(provenance)):
        raise ModelRegistryError("environment provenance contains non-portable runtime fields")
    if document.get("environment_build_id") != frozen["artifact_id"]:
        raise ModelRegistryError("environment dependency ID differs from environment payload")
    if not validate_current_runtime:
        return document
    expected_python = document.get("python_implementation_version", {})
    current_python = {
        "implementation": platform.python_implementation(),
        "version": platform.python_version(),
    }
    if expected_python != current_python:
        raise ModelRegistryError(
            f"current Python runtime differs from frozen environment: "
            f"expected={expected_python!r} current={current_python!r}"
        )
    expected_versions = document.get("critical_backend_versions")
    if not isinstance(expected_versions, Mapping) or not expected_versions:
        raise ModelRegistryError("environment lacks critical backend versions")
    current_versions: dict[str, str] = {}
    for distribution in sorted(expected_versions):
        try:
            current_versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError as exc:
            raise ModelRegistryError(
                f"current runtime lacks frozen critical backend: {distribution}"
            ) from exc
    if dict(expected_versions) != current_versions:
        raise ModelRegistryError(
            "current critical backend versions differ from the frozen environment"
        )
    return document


def _model_id_inputs(model: Mapping[str, Any]) -> dict[str, Any]:
    inputs = {
        key: copy.deepcopy(model[key])
        for key in (
            "schema_version",
            "artifact_type",
            "scope",
            "scientific_eligible",
            "model_name",
            "model_key",
            "role",
            "seed",
            "checkpoint_format",
            "checkpoint_inventory",
            "tokenizer_inventory",
            "base_inventory",
            "tokenizer_contract",
            "training_plan_dependency",
            "schedule_dependency",
            "training_receipt_dependency",
            "base_model_dependency",
            "environment_dependency",
        )
    }
    if model.get("artifact_type") == "legacy-smoke-only":
        inputs["legacy_source_tree_sha256"] = model[
            "legacy_source_tree_sha256"
        ]
    if "train_partition_dependency" in model:
        inputs["train_partition_dependency"] = copy.deepcopy(
            model["train_partition_dependency"]
        )
    return inputs


def _validate_model_document(
    model: Mapping[str, Any],
    *,
    workspace_root: str | Path,
    rehash_sources: bool,
    rehash_cache: dict[str, Path] | None = None,
) -> None:
    validate_json_schema(model, MODEL_SCHEMA)
    if model.get("id_inputs") != _model_id_inputs(model):
        raise ModelRegistryError("model artifact ID inputs are not canonical")
    expected_id = MODEL_ID_PREFIX + canonical_sha256(model["id_inputs"])
    if model.get("model_artifact_id") != expected_id:
        raise ModelRegistryError("model artifact ID is not reproducible")
    for name in ("checkpoint_inventory", "tokenizer_inventory", "base_inventory"):
        snapshot = model.get(name)
        if not isinstance(snapshot, Mapping):
            raise ModelRegistryError(f"model artifact lacks {name}")
        _validate_tree_shape(snapshot)
        if rehash_sources:
            _resolve_and_rehash_tree(snapshot, workspace_root, rehash_cache)
    artifact_type = model.get("artifact_type")
    if (
        artifact_type == "trained"
        and (
            model.get("scope") == "formal"
            or model.get("scientific_eligible") is True
        )
        and model.get("tokenizer_inventory", {}).get("inventory_policy")
        != "all-regular-files/v1"
    ):
        raise ModelRegistryError(
            "scientific trained model tokenizer inventory must cover the complete file tree"
        )
    _validate_shards(
        model["checkpoint_inventory"], checkpoint_format=model["checkpoint_format"]
    )
    _validate_index_references(model["checkpoint_inventory"], workspace_root=workspace_root)
    _validate_tokenizer(model["tokenizer_inventory"])
    stored_contract = model.get("tokenizer_contract")
    if not isinstance(stored_contract, Mapping):
        raise ModelRegistryError("model lacks a tokenizer contract")
    if model.get("tokenizer_contract") != _tokenizer_contract(
        model["tokenizer_inventory"],
        workspace_root=workspace_root,
        tokenizer_revision=stored_contract.get("tokenizer_revision"),
        chat_template_source=str(stored_contract.get("chat_template_source")),
        require_enable_thinking=model.get("artifact_type") != "legacy-smoke-only",
    ):
        raise ModelRegistryError("model tokenizer contract is not reproducible")
    if (
        artifact_type != "legacy-smoke-only"
        and "legacy_source_tree_sha256" in model
    ):
        raise ModelRegistryError(
            "non-legacy model contains a legacy source tree hash"
        )
    if artifact_type == "base":
        if model.get("scope") != "base" or model.get("scientific_eligible") is not False:
            raise ModelRegistryError("base model scope/eligibility is invalid")
        nullable = (
            "model_key",
            "role",
            "seed",
            "training_plan_dependency",
            "schedule_dependency",
            "training_receipt_dependency",
            "base_model_dependency",
        )
        if any(model.get(key) is not None for key in nullable):
            raise ModelRegistryError("base model artifact contains training lineage")
        if model.get("checkpoint_format") != "base":
            raise ModelRegistryError("base model artifact has the wrong checkpoint format")
        if model.get("checkpoint_inventory") != model.get("base_inventory"):
            raise ModelRegistryError("base model checkpoint/base inventories differ")
        validate_dependency_ref(
            model.get("environment_dependency", {}), expected_kind="stage1-environment"
        )
        _validate_environment_dependency(
            model["environment_dependency"], workspace_root, validate_current_runtime=True
        )
        return
    if artifact_type == "legacy-smoke-only":
        if (
            model.get("scope") != "engineering-smoke"
            or model.get("scientific_eligible") is not False
            or model.get("model_key") != LEGACY_MODEL_KEY
            or model.get("role") != LEGACY_MODEL_ROLE
            or model.get("seed") is not None
        ):
            raise ModelRegistryError("legacy model identity/scope is invalid")
        if model.get("checkpoint_format") != "full":
            raise ModelRegistryError(
                "legacy smoke checkpoint must be a self-contained full model"
            )
        if model.get("checkpoint_inventory", {}).get(
            "inventory_policy"
        ) != "all-regular-files/v1" or model.get("tokenizer_inventory", {}).get(
            "inventory_policy"
        ) != "all-regular-files/v1":
            raise ModelRegistryError(
                "legacy checkpoint and tokenizer must use full-tree inventories"
            )
        if model.get("base_inventory") != model.get("checkpoint_inventory"):
            raise ModelRegistryError(
                "legacy full checkpoint base/checkpoint inventories differ"
            )
        if model.get("legacy_source_tree_sha256") != model.get(
            "checkpoint_inventory", {}
        ).get("file_tree_sha256"):
            raise ModelRegistryError("legacy source tree hash is invalid")
        if stored_contract.get("tokenizer_revision") != stored_contract.get(
            "tokenizer_content_revision"
        ):
            raise ModelRegistryError(
                "legacy tokenizer revision must equal its truthful content revision"
            )
        nullable = (
            "training_plan_dependency",
            "schedule_dependency",
            "training_receipt_dependency",
            "base_model_dependency",
        )
        if any(model.get(key) is not None for key in nullable):
            raise ModelRegistryError("legacy model contains forbidden training lineage")
        validate_dependency_ref(
            model.get("environment_dependency", {}), expected_kind="stage1-environment"
        )
        _validate_environment_dependency(
            model["environment_dependency"],
            workspace_root,
            validate_current_runtime=True,
        )
        return
    if artifact_type != "trained":
        raise ModelRegistryError("unknown model artifact type")
    if model.get("scope") not in {"pilot", "formal"}:
        raise ModelRegistryError("trained model has an invalid scope")
    if model.get("scientific_eligible") is not (model.get("scope") == "formal"):
        raise ModelRegistryError("trained model eligibility disagrees with scope")
    if not isinstance(model.get("model_key"), str) or not MODEL_KEY_RE.fullmatch(model["model_key"]):
        raise ModelRegistryError("trained model has an invalid model_key")
    for key, kinds in (
        ("training_plan_dependency", "training-plan"),
        ("schedule_dependency", "training-schedule"),
        ("training_receipt_dependency", RECEIPT_ARTIFACT_KIND),
        ("train_partition_dependency", "train-partition"),
        ("base_model_dependency", MODEL_ARTIFACT_KIND),
        ("environment_dependency", "stage1-environment"),
    ):
        validate_dependency_ref(model.get(key, {}), expected_kind=kinds)
    base_target = resolve_dependency_target(model["base_model_dependency"], workspace_root)
    base = _validate_model_target(
        base_target,
        workspace_root=workspace_root,
        require_name=True,
        rehash_cache=rehash_cache,
    )
    if base.get("artifact_type") != "base":
        raise ModelRegistryError("trained model base dependency is not a registered base model")
    if model.get("base_inventory") != base.get("base_inventory"):
        raise ModelRegistryError("trained model embeds a different base inventory")
    if model.get("tokenizer_contract") != base.get("tokenizer_contract"):
        raise ModelRegistryError("trained model tokenizer/template contract differs from base")
    if base.get("environment_dependency") != model.get("environment_dependency"):
        raise ModelRegistryError("trained model/base environment dependencies differ")
    _validate_environment_dependency(
        model["environment_dependency"], workspace_root, validate_current_runtime=True
    )


def _validate_model_target(
    target: Path,
    *,
    workspace_root: str | Path,
    require_name: bool,
    rehash_cache: dict[str, Path] | None = None,
    tokenizer: Any | None = None,
) -> dict[str, Any]:
    if rehash_cache is None:
        rehash_cache = {}
    validate_payload_manifest(target)
    model = load_json(target / "model.json")
    if not isinstance(model, Mapping):
        raise ModelRegistryError("model.json is not an object")
    artifact_type = model.get("artifact_type")
    expected = {"model.json", "payload_manifest.json"}
    if artifact_type == "base":
        expected.add("environment_ref.json")
    if artifact_type == "legacy-smoke-only":
        expected.add("environment_ref.json")
    if artifact_type == "trained":
        expected.update(
            {
                "training_plan_ref.json",
                "schedule_ref.json",
                "training_receipt_ref.json",
                "train_partition_ref.json",
                "base_model_ref.json",
                "environment_ref.json",
            }
        )
    ensure_exact_file_set(target, expected)
    _validate_model_document(
        model,
        workspace_root=workspace_root,
        rehash_sources=True,
        rehash_cache=rehash_cache,
    )
    if require_name and target.name != model["model_artifact_id"]:
        raise ModelRegistryError("model target directory does not match artifact ID")
    if artifact_type == "trained":
        files = {
            "training_plan_ref.json": "training_plan_dependency",
            "schedule_ref.json": "schedule_dependency",
            "training_receipt_ref.json": "training_receipt_dependency",
            "train_partition_ref.json": "train_partition_dependency",
            "base_model_ref.json": "base_model_dependency",
            "environment_ref.json": "environment_dependency",
        }
        for filename, key in files.items():
            if load_json(target / filename) != model[key]:
                raise ModelRegistryError(f"model {filename} differs from model.json")
        receipt_target = resolve_dependency_target(
            model["training_receipt_dependency"], workspace_root
        )
        receipt_meta, receipt = _validate_receipt_target(
            receipt_target,
            workspace_root=workspace_root,
            require_name=True,
            rehash_cache=rehash_cache,
            tokenizer=tokenizer,
        )
        _assert_receipt_checkpoint_binding(
            receipt,
            model["checkpoint_inventory"],
            required=model.get("scope") == "formal",
        )
        if (
            receipt_meta["training_plan_dependency"] != model["training_plan_dependency"]
            or receipt_meta["schedule_dependency"] != model["schedule_dependency"]
            or receipt_meta["train_partition_dependency"]
            != model["train_partition_dependency"]
            or receipt_meta["base_model_dependency"] != model["base_model_dependency"]
            or receipt_meta["environment_dependency"] != model["environment_dependency"]
            or receipt["model_key"] != model["model_key"]
            or receipt["role"] != model["role"]
            or receipt["seed"] != model["seed"]
        ):
            raise ModelRegistryError("trained model and receipt lineage disagree")
        schedule_target = resolve_dependency_target(model["schedule_dependency"], workspace_root)
        schedule_meta = load_json(schedule_target / "schedule.meta.json")
        if (
            not isinstance(schedule_meta, Mapping)
            or schedule_meta.get("tokenizer_revision")
            != model["tokenizer_contract"]["tokenizer_revision"]
        ):
            raise ModelRegistryError("trained model tokenizer revision differs from schedule")
    elif artifact_type in {"base", "legacy-smoke-only"}:
        if load_json(target / "environment_ref.json") != model["environment_dependency"]:
            raise ModelRegistryError(
                f"{artifact_type} model environment_ref.json differs from model.json"
            )
    return dict(model)


def register_base_model(
    *,
    model_dir: str | Path,
    tokenizer_dir: str | Path | None,
    model_name: str,
    tokenizer_revision: str,
    environment_ref: str | Path,
    write_ref: str | Path,
    workspace_root: str | Path = REPOSITORY_ROOT,
    target_root: str | Path | None = None,
) -> dict[str, Any]:
    """Register a local immutable base model without copying its weights."""

    if not isinstance(model_name, str) or not model_name.strip():
        raise ModelRegistryError("model_name cannot be empty")
    root = Path(workspace_root).resolve()
    _, _, environment_dependency = _dependency_projection(
        environment_ref, expected_kind="stage1-environment", workspace_root=root
    )
    _validate_environment_dependency(
        environment_dependency, root, validate_current_runtime=True
    )
    tokenizer_source = model_dir if tokenizer_dir is None else tokenizer_dir
    snapshots = _snapshot_cache(
        ((model_dir, "base"), (model_dir, "checkpoint"), (tokenizer_source, "tokenizer")),
        root,
    )
    _validate_shards(snapshots["base"], checkpoint_format="base")
    _validate_index_references(snapshots["base"], workspace_root=root)
    _validate_tokenizer(snapshots["tokenizer"])
    fields: dict[str, Any] = {
        "schema_version": MODEL_SCHEMA_VERSION,
        "model_artifact_id": MODEL_ID_PREFIX + "0" * 64,
        "artifact_type": "base",
        "scope": "base",
        "scientific_eligible": False,
        "model_name": model_name.strip(),
        "model_key": None,
        "role": None,
        "seed": None,
        "checkpoint_format": "base",
        "checkpoint_inventory": snapshots["checkpoint"],
        "tokenizer_inventory": snapshots["tokenizer"],
        "base_inventory": snapshots["base"],
        "tokenizer_contract": _tokenizer_contract(
            snapshots["tokenizer"],
            workspace_root=root,
            tokenizer_revision=tokenizer_revision,
        ),
        "training_plan_dependency": None,
        "schedule_dependency": None,
        "training_receipt_dependency": None,
        "base_model_dependency": None,
        "environment_dependency": environment_dependency,
    }
    fields["id_inputs"] = _model_id_inputs(fields)
    fields["model_artifact_id"] = MODEL_ID_PREFIX + canonical_sha256(fields["id_inputs"])
    _validate_model_document(fields, workspace_root=root, rehash_sources=False)
    parent = (
        Path(target_root).resolve() if target_root is not None else root / "exps/causal_context/stage1_p0/models"
    )
    target = parent / fields["model_artifact_id"]
    staging = new_staging_directory(parent, fields["model_artifact_id"])
    try:
        write_canonical_json(staging / "model.json", fields)
        write_canonical_json(
            staging / "environment_ref.json", environment_dependency
        )
        payload_hash = finalize_target_atomic(
            staging,
            target,
            validate_staging=lambda path: _validate_model_target(
                path, workspace_root=root, require_name=False
            ),
        )
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return write_locator_ref(
        write_ref,
        artifact_kind=MODEL_ARTIFACT_KIND,
        artifact_id=fields["model_artifact_id"],
        target=target,
        payload_manifest_sha256=payload_hash,
    )


def _resolve_legacy_composition(
    checkpoint: Mapping[str, Any], composition: str
) -> str:
    if composition not in {"auto", "full"}:
        raise ModelRegistryError(
            "legacy smoke registration supports only auto or self-contained full composition"
        )
    paths = _inventory_paths(checkpoint)
    full = "config.json" in paths and bool(
        paths.intersection(
            {
                "model.safetensors",
                "pytorch_model.bin",
                "model.safetensors.index.json",
                "pytorch_model.bin.index.json",
            }
        )
    )
    adapter = "adapter_config.json" in paths and bool(
        paths.intersection(
            {
                "adapter_model.safetensors",
                "adapter_model.bin",
                "adapter_model.safetensors.index.json",
                "adapter_model.bin.index.json",
            }
        )
    )
    if composition == "auto" and full and not adapter:
        return "full"
    if composition == "full" and full and not adapter:
        return "full"
    if adapter:
        raise ModelRegistryError(
            "legacy adapter checkpoints require a separately frozen base and are not "
            "eligible for the self-contained engineering smoke slot"
        )
    raise ModelRegistryError("legacy checkpoint is not a self-contained full model")


def register_legacy_model(
    *,
    checkpoint: str | Path,
    composition: str = "auto",
    tokenizer_root: str | Path | None = None,
    chat_template_source: str = "auto",
    environment_ref: str | Path,
    write_ref: str | Path,
    workspace_root: str | Path = REPOSITORY_ROOT,
    target_root: str | Path | None = None,
) -> dict[str, Any]:
    """Register the immutable, non-scientific legacy compatibility checkpoint.

    No weights are copied.  Both the checkpoint and tokenizer source are
    represented by complete regular-file inventories, and the legacy tokenizer
    revision is derived from its content rather than supplied by the operator.
    """

    from data.training_schedule import tokenizer_revision_from_directory

    root = Path(workspace_root).resolve()
    checkpoint_source, checkpoint_logical = _logical_directory(
        checkpoint, root, label="legacy checkpoint"
    )
    if tokenizer_root is None or str(tokenizer_root) == "same":
        tokenizer_source = checkpoint_source
    else:
        tokenizer_source = tokenizer_root
    _, _, environment_dependency = _dependency_projection(
        environment_ref, expected_kind="stage1-environment", workspace_root=root
    )
    _validate_environment_dependency(
        environment_dependency, root, validate_current_runtime=True
    )
    snapshots = _snapshot_cache(
        (
            (checkpoint_source, "checkpoint"),
            (tokenizer_source, "tokenizer"),
        ),
        root,
        tokenizer_inventory_policy="all-regular-files/v1",
    )
    checkpoint_format = _resolve_legacy_composition(
        snapshots["checkpoint"], composition
    )
    _validate_shards(
        snapshots["checkpoint"], checkpoint_format=checkpoint_format
    )
    _validate_index_references(snapshots["checkpoint"], workspace_root=root)
    _validate_tokenizer(snapshots["tokenizer"])
    truthful_revision = tokenizer_revision_from_directory(
        root / snapshots["tokenizer"]["logical_repo_path"]
    )
    tokenizer_contract = _tokenizer_contract(
        snapshots["tokenizer"],
        workspace_root=root,
        tokenizer_revision=truthful_revision,
        chat_template_source=chat_template_source,
        require_enable_thinking=False,
    )
    fields: dict[str, Any] = {
        "schema_version": MODEL_SCHEMA_VERSION,
        "model_artifact_id": MODEL_ID_PREFIX + "0" * 64,
        "artifact_type": "legacy-smoke-only",
        "scope": "engineering-smoke",
        "scientific_eligible": False,
        "model_name": f"legacy-smoke:{Path(checkpoint_logical).name}",
        "model_key": LEGACY_MODEL_KEY,
        "role": LEGACY_MODEL_ROLE,
        "seed": None,
        "checkpoint_format": checkpoint_format,
        "checkpoint_inventory": snapshots["checkpoint"],
        "tokenizer_inventory": snapshots["tokenizer"],
        "base_inventory": copy.deepcopy(snapshots["checkpoint"]),
        "tokenizer_contract": tokenizer_contract,
        "legacy_source_tree_sha256": snapshots["checkpoint"][
            "file_tree_sha256"
        ],
        "training_plan_dependency": None,
        "schedule_dependency": None,
        "training_receipt_dependency": None,
        "base_model_dependency": None,
        "environment_dependency": environment_dependency,
    }
    fields["id_inputs"] = _model_id_inputs(fields)
    fields["model_artifact_id"] = MODEL_ID_PREFIX + canonical_sha256(
        fields["id_inputs"]
    )
    _validate_model_document(fields, workspace_root=root, rehash_sources=False)
    parent = (
        Path(target_root).resolve()
        if target_root is not None
        else root / "exps/causal_context/stage1_p0/models"
    )
    target = parent / fields["model_artifact_id"]
    staging = new_staging_directory(parent, fields["model_artifact_id"])
    rehash_cache = {
        canonical_sha256(snapshot): (
            root / str(snapshot["logical_repo_path"])
        ).resolve()
        for snapshot in (snapshots["checkpoint"], snapshots["tokenizer"])
    }
    try:
        write_canonical_json(staging / "model.json", fields)
        write_canonical_json(
            staging / "environment_ref.json", environment_dependency
        )
        payload_hash = finalize_target_atomic(
            staging,
            target,
            validate_staging=lambda path: _validate_model_target(
                path,
                workspace_root=root,
                require_name=False,
                rehash_cache=rehash_cache,
            ),
        )
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return write_locator_ref(
        write_ref,
        artifact_kind=MODEL_ARTIFACT_KIND,
        artifact_id=fields["model_artifact_id"],
        target=target,
        payload_manifest_sha256=payload_hash,
    )


def _receipt_without_hash(receipt: Mapping[str, Any]) -> dict[str, Any]:
    value = copy.deepcopy(dict(receipt))
    value.pop("receipt_sha256", None)
    return value


def _validated_receipt_checkpoint_inventory(
    receipt: Mapping[str, Any],
    *,
    required: bool,
) -> dict[str, Any] | None:
    value = receipt.get("selected_checkpoint_inventory")
    if value is None:
        if required:
            raise ModelRegistryError(
                "formal receipt lacks an exact selected checkpoint inventory"
            )
        return None
    if not isinstance(value, Mapping):
        raise ModelRegistryError(
            "training receipt selected checkpoint inventory is malformed"
        )
    _validate_tree_shape(value)
    if value.get("inventory_policy") != "all-regular-files/v1":
        raise ModelRegistryError(
            "training receipt checkpoint inventory must cover the complete file tree"
        )
    step = receipt.get("selected_checkpoint_global_step")
    if (
        isinstance(step, bool)
        or not isinstance(step, int)
        or step <= 0
        or Path(str(value.get("logical_repo_path", ""))).name
        != f"checkpoint-{step}"
    ):
        raise ModelRegistryError(
            "training receipt checkpoint inventory path differs from selected step"
        )
    return copy.deepcopy(dict(value))


def _assert_receipt_checkpoint_binding(
    receipt: Mapping[str, Any],
    checkpoint_inventory: Mapping[str, Any],
    *,
    required: bool,
) -> bool:
    """Compare one freshly hashed checkpoint tree to the immutable receipt."""

    expected = _validated_receipt_checkpoint_inventory(receipt, required=required)
    _validate_tree_shape(checkpoint_inventory)
    if checkpoint_inventory.get("inventory_policy") != "all-regular-files/v1":
        raise ModelRegistryError(
            "registered checkpoint inventory must cover the complete file tree"
        )
    if expected is None:
        return False
    if dict(checkpoint_inventory) != expected:
        raise ModelRegistryError(
            "registered checkpoint inventory differs from the selected checkpoint receipt"
        )
    return True


def _validate_raw_receipt(receipt: Mapping[str, Any]) -> None:
    validate_json_schema(receipt, RECEIPT_SCHEMA)
    completed = receipt.get("completed_epochs")
    rows = receipt.get("completed_epoch_registry")
    planned_epochs = receipt.get("planned_epochs")
    if (
        isinstance(planned_epochs, bool)
        or not isinstance(planned_epochs, int)
        or planned_epochs <= 0
    ):
        raise ModelRegistryError("training receipt planned epochs are invalid")
    if (
        not isinstance(completed, list)
        or not completed
        or any(
            isinstance(epoch, bool) or not isinstance(epoch, int) or epoch <= 0
            for epoch in completed
        )
        or completed != list(range(1, len(completed) + 1))
    ):
        raise ModelRegistryError("training receipt completed epochs are not contiguous")
    if not isinstance(rows, list) or [row.get("epoch") for row in rows] != completed:
        raise ModelRegistryError("training receipt epoch registry is incomplete or unordered")
    global_steps = [row.get("global_step") for row in rows]
    if any(
        isinstance(step, bool) or not isinstance(step, int) or step <= 0
        for step in global_steps
    ) or any(left >= right for left, right in zip(global_steps, global_steps[1:])):
        raise ModelRegistryError("training receipt epoch global steps are invalid")
    if completed[-1] > planned_epochs:
        raise ModelRegistryError("training receipt exceeds planned epochs")
    fit_count = receipt.get("fit_query_count")
    calibration_count = receipt.get("calibration_query_count")
    if (
        isinstance(fit_count, bool)
        or not isinstance(fit_count, int)
        or fit_count <= 0
        or isinstance(calibration_count, bool)
        or not isinstance(calibration_count, int)
        or calibration_count <= 0
        or any(row.get("fit_record_count") != fit_count for row in rows)
        or any(
            row.get("calibration_record_count") != calibration_count
            for row in rows
        )
    ):
        raise ModelRegistryError(
            "training receipt fit/calibration counts are inconsistent"
        )
    if any(
        not isinstance(receipt.get(field), str)
        or re.fullmatch(r"[0-9a-f]{64}", str(receipt.get(field))) is None
        for field in ("fit_query_ids_sha256", "calibration_query_ids_sha256")
    ):
        raise ModelRegistryError(
            "training receipt fit/calibration ID hashes are invalid"
        )
    fixed = receipt.get("fixed_presentation")
    expected_fixed = {
        "policy": "calibration-epoch-1-wire/v1",
        "wire_epoch": 1,
        "demo_order_across_epochs": True,
        "source_mask_across_epochs": True,
    }
    if fixed != expected_fixed:
        raise ModelRegistryError(
            "training receipt calibration presentation policy is not frozen"
        )
    calibration_digests = [
        row.get("calibration_records_sha256") for row in rows
    ]
    if len(set(calibration_digests)) != 1:
        raise ModelRegistryError(
            "training receipt calibration presentation changes across epochs"
        )

    audit = receipt.get("checkpoint_selection_audit")
    if not isinstance(audit, Mapping):
        raise ModelRegistryError("training receipt checkpoint-selection audit is missing")
    policy = audit.get("policy")
    history = audit.get("evaluation_history")
    if not isinstance(policy, Mapping) or not isinstance(history, list):
        raise ModelRegistryError("training receipt checkpoint-selection audit is malformed")
    threshold = policy.get("threshold")
    selected_metric = audit.get("selected_metric_value")
    numeric_values = {
        "early-stopping threshold": threshold,
        "selected checkpoint metric": selected_metric,
    }
    for label, value in numeric_values.items():
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            raise ModelRegistryError(f"training receipt {label} must be finite")
    threshold_value = float(threshold)
    if threshold_value < 0:
        raise ModelRegistryError("training receipt early-stopping threshold is invalid")

    minimum_epochs = policy.get("minimum_epochs")
    maximum_epochs = policy.get("maximum_epochs")
    patience_limit = policy.get("patience_evaluations")
    for label, value in (
        ("minimum epochs", minimum_epochs),
        ("maximum epochs", maximum_epochs),
        ("patience", patience_limit),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ModelRegistryError(f"training receipt early-stopping {label} is invalid")
    if minimum_epochs > maximum_epochs or maximum_epochs != planned_epochs:
        raise ModelRegistryError(
            "training receipt early-stopping epoch bounds differ from planned epochs"
        )

    history_epochs = [row.get("epoch") for row in history]
    history_steps = [row.get("global_step") for row in history]
    if history_epochs != completed or history_steps != global_steps:
        raise ModelRegistryError(
            "training receipt evaluation history does not exactly cover epoch/step lineage"
        )
    metric_values: list[float] = []
    for row in history:
        value = row.get("metric_value")
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            raise ModelRegistryError(
                "training receipt evaluation history contains a non-finite metric"
            )
        metric_values.append(float(value))
    if audit.get("evaluation_history_sha256") != canonical_sha256(history):
        raise ModelRegistryError(
            "training receipt evaluation history digest is not reproducible"
        )

    best_value: float | None = None
    best_step: int | None = None
    patience_counter = 0
    first_stop_index: int | None = None
    for index, (row, metric_value) in enumerate(zip(history, metric_values, strict=True)):
        qualifying = best_value is None or best_value - metric_value > threshold_value
        if qualifying:
            best_value = metric_value
            best_step = int(row["global_step"])
            patience_counter = 0
        else:
            patience_counter += 1
        if row.get("qualifying_improvement") is not qualifying:
            raise ModelRegistryError(
                "training receipt checkpoint improvement audit is inconsistent"
            )
        if row.get("patience_counter_after") != patience_counter:
            raise ModelRegistryError("training receipt patience audit is inconsistent")
        if row.get("best_global_step_after") != best_step:
            raise ModelRegistryError(
                "training receipt threshold-aware winner audit is inconsistent"
            )
        if (
            first_stop_index is None
            and int(row["epoch"]) >= minimum_epochs
            and patience_counter >= patience_limit
        ):
            first_stop_index = index
    if best_value is None or best_step is None:  # guarded by the receipt schema
        raise ModelRegistryError("training receipt evaluation history is empty")

    selected_step = receipt.get("selected_checkpoint_global_step")
    selected_epoch = receipt.get("selected_checkpoint_epoch")
    selected_rows = [
        row
        for row in rows
        if row.get("global_step") == best_step and row.get("epoch") == selected_epoch
    ]
    if (
        selected_step != best_step
        or audit.get("selected_global_step") != best_step
        or len(selected_rows) != 1
        or float(selected_metric) != best_value
        or receipt.get("global_step") != best_step
    ):
        raise ModelRegistryError(
            "training receipt selected checkpoint is not the threshold-aware earliest winner"
        )
    exit_step = receipt.get("training_exit_global_step")
    if exit_step != global_steps[-1] or selected_step > exit_step:
        raise ModelRegistryError(
            "training receipt exit step differs from the final completed epoch"
        )

    last_index = len(history) - 1
    if completed[-1] < planned_epochs:
        expected_stop_reason = "early-stopping-patience"
        if first_stop_index != last_index:
            raise ModelRegistryError(
                "training receipt early exit does not occur at the first patience boundary"
            )
    else:
        expected_stop_reason = "maximum-epochs"
        if first_stop_index is not None and first_stop_index < last_index:
            raise ModelRegistryError(
                "training receipt continued after the first patience boundary"
            )
    if audit.get("stop_reason") != expected_stop_reason:
        raise ModelRegistryError("training receipt stop reason is inconsistent")

    for field in ("train_code_sha256", "runtime_code_sha256"):
        digest = receipt.get(field)
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise ModelRegistryError(f"training receipt {field} is not a SHA-256 digest")
    _validated_receipt_checkpoint_inventory(receipt, required=False)
    if receipt.get("receipt_sha256") != canonical_sha256(_receipt_without_hash(receipt)):
        raise ModelRegistryError("training receipt digest is not reproducible")


def _receipt_id_inputs(meta: Mapping[str, Any]) -> dict[str, Any]:
    inputs = {
        key: copy.deepcopy(meta[key])
        for key in (
            "schema_version",
            "scope",
            "scientific_eligible",
            "receipt_sha256",
            "model_key",
            "role",
            "seed",
            "training_plan_dependency",
            "schedule_dependency",
            "train_partition_dependency",
            "base_model_dependency",
            "environment_dependency",
        )
    }
    # Older non-scientific pilot receipt artifacts predate exact checkpoint
    # binding.  Preserve their original ID projection while every newly
    # wrapped receipt explicitly binds either a tree hash or null.
    if "selected_checkpoint_file_tree_sha256" in meta:
        inputs["selected_checkpoint_file_tree_sha256"] = copy.deepcopy(
            meta["selected_checkpoint_file_tree_sha256"]
        )
    return inputs


def _validate_receipt_cross_lineage(
    receipt: Mapping[str, Any],
    *,
    plan: Mapping[str, Any],
    plan_dependency: Mapping[str, Any],
    schedule_meta: Mapping[str, Any],
    schedule_dependency: Mapping[str, Any],
    base_dependency: Mapping[str, Any],
    environment_dependency: Mapping[str, Any],
    workspace_root: str | Path | None = None,
    tokenizer: Any | None = None,
) -> None:
    selected = [
        slot for slot in plan.get("ordered_model_slots", []) if slot.get("model_key") == receipt.get("model_key")
    ]
    if len(selected) != 1 or selected[0].get("training_required") is not True:
        raise ModelRegistryError("receipt model_key is not one training plan slot")
    slot = selected[0]
    _validated_receipt_checkpoint_inventory(
        receipt,
        required=plan.get("scope") == "formal",
    )
    if receipt.get("role") != slot.get("role") or receipt.get("seed") != slot.get("seed"):
        raise ModelRegistryError("receipt role/seed differs from training plan slot")
    if receipt.get("planned_epochs") != slot.get("epochs"):
        raise ModelRegistryError("receipt planned epochs differ from training plan slot")
    if receipt.get("final_checkpoint_rule") != slot.get("final_checkpoint_rule"):
        raise ModelRegistryError("receipt checkpoint rule differs from training plan slot")
    if receipt.get("train_config_sha256") != slot.get("train_config_sha256"):
        raise ModelRegistryError("receipt train config differs from training plan slot")
    resolved_config = slot.get("train_config_resolved")
    frozen_early_stopping = (
        resolved_config.get("early_stopping")
        if isinstance(resolved_config, Mapping)
        else None
    )
    receipt_audit = receipt.get("checkpoint_selection_audit")
    receipt_policy = (
        receipt_audit.get("policy") if isinstance(receipt_audit, Mapping) else None
    )
    if (
        not isinstance(frozen_early_stopping, Mapping)
        or not isinstance(receipt_policy, Mapping)
        or dict(receipt_policy) != dict(frozen_early_stopping)
    ):
        raise ModelRegistryError(
            "receipt early-stopping policy differs from training plan slot"
        )
    for label, value in (
        ("receipt train", receipt.get("train_code_sha256")),
        ("plan train", plan.get("train_code_sha256")),
        ("receipt runtime", receipt.get("runtime_code_sha256")),
        ("plan runtime", plan.get("runtime_code_sha256")),
    ):
        if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
            raise ModelRegistryError(f"{label} code hash is not a SHA-256 digest")
    if receipt.get("train_code_sha256") != plan.get("train_code_sha256"):
        raise ModelRegistryError("receipt train code differs from training plan")
    if receipt.get("runtime_code_sha256") != plan.get("runtime_code_sha256"):
        raise ModelRegistryError("receipt runtime code differs from training plan")
    expected = {
        "training_plan_dependency": plan_dependency,
        "training_evidence_dependency": plan.get("training_evidence_dependency"),
        "train_partition_dependency": plan.get("train_partition_dependency"),
        "schedule_dependency": schedule_dependency,
        "base_model_dependency": base_dependency,
        "environment_dependency": environment_dependency,
    }
    for key, value in expected.items():
        if receipt.get(key) != value:
            raise ModelRegistryError(f"receipt {key} differs from explicit immutable ref")
    if schedule_meta.get("training_plan_dependency") != plan_dependency:
        raise ModelRegistryError("schedule points to a different training plan")
    if schedule_meta.get("training_evidence_dependency") != plan.get(
        "training_evidence_dependency"
    ):
        raise ModelRegistryError("schedule points to different training evidence")
    if schedule_meta.get("train_partition_dependency") != plan.get(
        "train_partition_dependency"
    ):
        raise ModelRegistryError("schedule points to a different train partition")
    schedule_slots = [
        row for row in schedule_meta.get("slot_registry", []) if row.get("model_key") == receipt.get("model_key")
    ]
    if len(schedule_slots) != 1:
        raise ModelRegistryError("schedule does not bind the receipt model slot exactly once")

    # Keep a narrow seam for unit-testing the pure binding checks above.  All
    # registry entry points pass an explicit workspace root and therefore
    # always execute the independent artifact replay below.
    if workspace_root is None:
        return

    # Independently replay the complete scientific lineage instead of trusting
    # the receipt's self-reported partition counts or presentation hashes.
    from data.train_partition import validate_train_partition_target
    from data.training_evidence import validate_training_evidence_target
    from data.training_schedule import (
        _validate_schedule_target,
        load_model_epoch_records,
    )

    root = Path(workspace_root).resolve()
    partition_dependency = validate_dependency_ref(
        plan.get("train_partition_dependency", {}), expected_kind="train-partition"
    )
    partition_target = resolve_dependency_target(partition_dependency, root)
    partition_report = validate_train_partition_target(
        partition_target, workspace_root=root
    )
    if partition_report.get("partition_dependency") != partition_dependency:
        raise ModelRegistryError("receipt partition dependency cannot be reproduced")
    evidence_dependency = validate_dependency_ref(
        plan.get("training_evidence_dependency", {}),
        expected_kind="training-evidence",
    )
    evidence_target = resolve_dependency_target(evidence_dependency, root)
    evidence_meta, evidence_records = validate_training_evidence_target(
        evidence_target,
        workspace_root=root,
        tokenizer=tokenizer,
    )
    if (
        evidence_meta.get("train_partition_dependency") != partition_dependency
        or evidence_meta.get("data_dependency")
        != partition_report.get("data_dependency")
    ):
        raise ModelRegistryError(
            "receipt plan/evidence/partition lineage is inconsistent"
        )
    schedule_target = resolve_dependency_target(schedule_dependency, root)
    _validate_schedule_target(
        schedule_target,
        workspace_root=root,
        tokenizer=tokenizer,
        require_directory_name=True,
    )

    partition_rows = load_jsonl(partition_target / "partition.jsonl")
    fit_ids = [
        row["query_id"] for row in partition_rows if row["partition"] == "fit"
    ]
    calibration_ids = [
        row["query_id"]
        for row in partition_rows
        if row["partition"] == "calibration"
    ]
    if (
        receipt.get("fit_query_count") != len(fit_ids)
        or receipt.get("calibration_query_count") != len(calibration_ids)
        or receipt.get("fit_query_ids_sha256") != canonical_sha256(fit_ids)
        or receipt.get("calibration_query_ids_sha256")
        != canonical_sha256(calibration_ids)
        or evidence_meta.get("fit_query_ids_sha256")
        != canonical_sha256(fit_ids)
        or evidence_meta.get("calibration_query_ids_sha256")
        != canonical_sha256(calibration_ids)
        or [record["query"]["id"] for record in evidence_records]
        != [row["query_id"] for row in partition_rows]
    ):
        raise ModelRegistryError(
            "receipt/evidence fit-calibration IDs differ from the frozen partition"
        )
    data_config = (
        resolved_config.get("data")
        if isinstance(resolved_config, Mapping)
        else None
    )
    plan_rng = plan.get("order_dropout_rng_policy")
    expected_fixed = {
        "policy": "calibration-epoch-1-wire/v1",
        "wire_epoch": 1,
        "demo_order_across_epochs": True,
        "source_mask_across_epochs": True,
    }
    if (
        not isinstance(data_config, Mapping)
        or data_config.get("fixed_presentation") != expected_fixed
        or not isinstance(plan_rng, Mapping)
        or plan_rng.get("calibration_presentation") != expected_fixed
        or receipt.get("fixed_presentation") != expected_fixed
    ):
        raise ModelRegistryError(
            "receipt fixed calibration presentation differs from plan/config"
        )

    by_epoch = {
        epoch: load_model_epoch_records(
            schedule_target,
            model_key=str(receipt["model_key"]),
            epoch=epoch,
        )
        for epoch in range(1, int(receipt["planned_epochs"]) + 1)
    }
    partition_labels = {row["query_id"]: row["partition"] for row in partition_rows}
    baseline_wire: dict[str, tuple[Any, ...]] = {}
    baseline_calibration_hashes: list[str] | None = None
    for epoch, schedule_rows in by_epoch.items():
        if [row["query_id"] for row in schedule_rows] != list(partition_labels):
            raise ModelRegistryError(
                "receipt schedule registry differs from the frozen partition"
            )
        for row in schedule_rows:
            label = partition_labels[row["query_id"]]
            if row.get("partition") != label:
                raise ModelRegistryError(
                    "receipt schedule partition label differs from the frozen artifact"
                )
            if label == "calibration":
                wire = (
                    tuple(row.get("ordered_demo_ids", [])),
                    row.get("use_lexicon"),
                    row.get("use_demos"),
                    row.get("instruction"),
                    row.get("input"),
                    row.get("output"),
                    row.get("rendered_prompt_sha256"),
                    row.get("rendered_prompt_tokens"),
                    row.get("sequence_tokens"),
                )
                previous = baseline_wire.setdefault(row["query_id"], wire)
                if previous != wire or row.get("presentation_epoch") != 1:
                    raise ModelRegistryError(
                        "receipt calibration presentation is not fixed across epochs"
                    )
        calibration_hashes = [
            row["record_sha256"]
            for row in by_epoch[1]
            if row["partition"] == "calibration"
        ]
        if baseline_calibration_hashes is None:
            baseline_calibration_hashes = calibration_hashes
        elif baseline_calibration_hashes != calibration_hashes:
            raise ModelRegistryError(
                "receipt calibration schedule hashes are not fixed"
            )

    receipt_epochs = {
        row["epoch"]: row for row in receipt["completed_epoch_registry"]
    }
    assert baseline_calibration_hashes is not None
    for epoch, registry_row in receipt_epochs.items():
        schedule_rows = by_epoch[epoch]
        all_hashes = [row["record_sha256"] for row in schedule_rows]
        fit_hashes = [
            row["record_sha256"]
            for row in schedule_rows
            if row["partition"] == "fit"
        ]
        expected_values = {
            "all_records_sha256": canonical_sha256(all_hashes),
            "fit_records_sha256": canonical_sha256(fit_hashes),
            "calibration_records_sha256": canonical_sha256(
                baseline_calibration_hashes
            ),
            "fit_record_count": len(fit_ids),
            "calibration_record_count": len(calibration_ids),
        }
        if any(registry_row.get(key) != value for key, value in expected_values.items()):
            raise ModelRegistryError(
                "training receipt epoch hashes cannot be replayed from the schedule"
            )


def register_training_receipt(
    *,
    receipt_json: str | Path,
    training_plan_ref: str | Path,
    schedule_ref: str | Path,
    base_model_ref: str | Path,
    environment_ref: str | Path,
    write_ref: str | Path,
    workspace_root: str | Path = REPOSITORY_ROOT,
    target_root: str | Path | None = None,
    tokenizer: Any | None = None,
) -> dict[str, Any]:
    """Wrap a Trainer receipt in a portable immutable dependency artifact."""

    from data.training_plan import load_training_plan
    from data.training_schedule import load_training_schedule

    root = Path(workspace_root).resolve()
    receipt = load_json(receipt_json)
    if not isinstance(receipt, Mapping):
        raise ModelRegistryError("training receipt JSON is not an object")
    _validate_raw_receipt(receipt)
    plan_locator, _, plan = load_training_plan(
        training_plan_ref,
        workspace_root=root,
        tokenizer=tokenizer,
    )
    plan_target = Path(plan_locator["target_path"])
    plan_dependency = portable_dependency(plan_locator, plan_target, root)
    schedule_locator, schedule_target, schedule_meta = load_training_schedule(
        schedule_ref, tokenizer=tokenizer
    )
    schedule_dependency = portable_dependency(schedule_locator, schedule_target, root)
    _, base_target, base_dependency = _dependency_projection(
        base_model_ref, expected_kind=MODEL_ARTIFACT_KIND, workspace_root=root
    )
    base = _validate_model_target(
        base_target,
        workspace_root=root,
        require_name=True,
        tokenizer=tokenizer,
    )
    if base.get("artifact_type") != "base":
        raise ModelRegistryError("formal receipt requires a registered base model")
    _, _, environment_dependency = _dependency_projection(
        environment_ref,
        expected_kind="stage1-environment",
        workspace_root=root,
    )
    if plan.get("scope") not in {"pilot", "formal"}:
        raise ModelRegistryError("training receipts cannot be registered for engineering plans")
    if plan.get("base_model_dependency") != base_dependency:
        raise ModelRegistryError("base model ref differs from training plan")
    if plan.get("environment_dependency") != environment_dependency:
        raise ModelRegistryError("environment ref differs from training plan")
    _validate_receipt_cross_lineage(
        receipt,
        plan=plan,
        plan_dependency=plan_dependency,
        schedule_meta=schedule_meta,
        schedule_dependency=schedule_dependency,
        base_dependency=base_dependency,
        environment_dependency=environment_dependency,
        workspace_root=root,
        tokenizer=tokenizer,
    )
    selected_checkpoint_inventory = _validated_receipt_checkpoint_inventory(
        receipt,
        required=plan.get("scope") == "formal",
    )
    selected_checkpoint_tree_hash = (
        selected_checkpoint_inventory["file_tree_sha256"]
        if selected_checkpoint_inventory is not None
        else None
    )
    meta: dict[str, Any] = {
        "schema_version": RECEIPT_ARTIFACT_SCHEMA_VERSION,
        "training_receipt_id": RECEIPT_ID_PREFIX + "0" * 64,
        "scope": plan["scope"],
        "scientific_eligible": (
            plan["scope"] == "formal"
            and selected_checkpoint_inventory is not None
        ),
        "receipt_sha256": receipt["receipt_sha256"],
        "selected_checkpoint_file_tree_sha256": selected_checkpoint_tree_hash,
        "model_key": receipt["model_key"],
        "role": receipt["role"],
        "seed": receipt["seed"],
        "training_plan_dependency": plan_dependency,
        "schedule_dependency": schedule_dependency,
        "train_partition_dependency": plan["train_partition_dependency"],
        "base_model_dependency": base_dependency,
        "environment_dependency": environment_dependency,
    }
    meta["id_inputs"] = _receipt_id_inputs(meta)
    meta["training_receipt_id"] = RECEIPT_ID_PREFIX + canonical_sha256(meta["id_inputs"])
    validate_json_schema(meta, RECEIPT_ARTIFACT_SCHEMA)
    parent = (
        Path(target_root).resolve()
        if target_root is not None
        else root / "exps/causal_context/stage1_p0/training_receipts"
    )
    target = parent / meta["training_receipt_id"]
    staging = new_staging_directory(parent, meta["training_receipt_id"])
    try:
        write_canonical_json(staging / "receipt.json", receipt)
        write_canonical_json(staging / "receipt_artifact.json", meta)
        write_canonical_json(staging / "training_plan_ref.json", plan_dependency)
        write_canonical_json(staging / "schedule_ref.json", schedule_dependency)
        write_canonical_json(
            staging / "train_partition_ref.json",
            plan["train_partition_dependency"],
        )
        write_canonical_json(staging / "base_model_ref.json", base_dependency)
        write_canonical_json(staging / "environment_ref.json", environment_dependency)
        payload_hash = finalize_target_atomic(
            staging,
            target,
            validate_staging=lambda path: _validate_receipt_target(
                path,
                workspace_root=root,
                require_name=False,
                tokenizer=tokenizer,
            ),
        )
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return write_locator_ref(
        write_ref,
        artifact_kind=RECEIPT_ARTIFACT_KIND,
        artifact_id=meta["training_receipt_id"],
        target=target,
        payload_manifest_sha256=payload_hash,
    )


def _validate_receipt_target(
    target: Path,
    *,
    workspace_root: str | Path,
    require_name: bool,
    rehash_cache: dict[str, Path] | None = None,
    tokenizer: Any | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    from data.training_plan import validate_training_plan_target
    from data.training_schedule import load_training_schedule

    ensure_exact_file_set(
        target,
        {
            "receipt.json",
            "receipt_artifact.json",
            "training_plan_ref.json",
            "schedule_ref.json",
            "train_partition_ref.json",
            "base_model_ref.json",
            "environment_ref.json",
            "payload_manifest.json",
        },
    )
    validate_payload_manifest(target)
    meta = load_json(target / "receipt_artifact.json")
    receipt = load_json(target / "receipt.json")
    if not isinstance(meta, Mapping) or not isinstance(receipt, Mapping):
        raise ModelRegistryError("receipt artifact payload is malformed")
    validate_json_schema(meta, RECEIPT_ARTIFACT_SCHEMA)
    _validate_raw_receipt(receipt)
    if meta.get("id_inputs") != _receipt_id_inputs(meta):
        raise ModelRegistryError("receipt artifact ID inputs are not canonical")
    receipt_id = RECEIPT_ID_PREFIX + canonical_sha256(meta["id_inputs"])
    if meta.get("training_receipt_id") != receipt_id:
        raise ModelRegistryError("receipt artifact ID is not reproducible")
    if require_name and target.name != receipt_id:
        raise ModelRegistryError("receipt target directory does not match artifact ID")
    files = {
        "training_plan_ref.json": ("training_plan_dependency", "training-plan"),
        "schedule_ref.json": ("schedule_dependency", "training-schedule"),
        "train_partition_ref.json": (
            "train_partition_dependency",
            "train-partition",
        ),
        "base_model_ref.json": ("base_model_dependency", MODEL_ARTIFACT_KIND),
        "environment_ref.json": ("environment_dependency", "stage1-environment"),
    }
    for filename, (key, kind) in files.items():
        dependency = load_json(target / filename)
        validate_dependency_ref(dependency, expected_kind=kind)
        if dependency != meta.get(key):
            raise ModelRegistryError(f"receipt {filename} differs from metadata")
    plan_target = resolve_dependency_target(meta["training_plan_dependency"], workspace_root)
    plan = validate_training_plan_target(
        plan_target,
        workspace_root=workspace_root,
        tokenizer=tokenizer,
    )
    schedule_target = resolve_dependency_target(meta["schedule_dependency"], workspace_root)
    # Build a temporary locator-free projection check via the deeply validated target meta.
    schedule_meta = load_json(schedule_target / "schedule.meta.json")
    if not isinstance(schedule_meta, Mapping):
        raise ModelRegistryError("receipt schedule dependency is malformed")
    base_target = resolve_dependency_target(meta["base_model_dependency"], workspace_root)
    base = _validate_model_target(
        base_target,
        workspace_root=workspace_root,
        require_name=True,
        rehash_cache=rehash_cache,
        tokenizer=tokenizer,
    )
    if base.get("artifact_type") != "base":
        raise ModelRegistryError("receipt base dependency is not a registered base")
    resolve_dependency_target(meta["environment_dependency"], workspace_root)
    _validate_receipt_cross_lineage(
        receipt,
        plan=plan,
        plan_dependency=meta["training_plan_dependency"],
        schedule_meta=schedule_meta,
        schedule_dependency=meta["schedule_dependency"],
        base_dependency=meta["base_model_dependency"],
        environment_dependency=meta["environment_dependency"],
        workspace_root=workspace_root,
        tokenizer=tokenizer,
    )
    selected_checkpoint_inventory = _validated_receipt_checkpoint_inventory(
        receipt,
        required=plan.get("scope") == "formal",
    )
    selected_checkpoint_tree_hash = (
        selected_checkpoint_inventory["file_tree_sha256"]
        if selected_checkpoint_inventory is not None
        else None
    )
    if (
        "selected_checkpoint_file_tree_sha256" in meta
        and meta.get("selected_checkpoint_file_tree_sha256")
        != selected_checkpoint_tree_hash
    ):
        raise ModelRegistryError(
            "receipt artifact checkpoint tree hash differs from raw receipt"
        )
    expected_scientific_eligibility = (
        plan.get("scope") == "formal"
        and selected_checkpoint_inventory is not None
        and "selected_checkpoint_file_tree_sha256" in meta
    )
    if (
        meta.get("scope") != plan.get("scope")
        or meta.get("scientific_eligible")
        is not expected_scientific_eligibility
        or meta.get("receipt_sha256") != receipt.get("receipt_sha256")
        or meta.get("model_key") != receipt.get("model_key")
        or meta.get("role") != receipt.get("role")
        or meta.get("seed") != receipt.get("seed")
    ):
        raise ModelRegistryError("receipt metadata differs from receipt/plan")
    return dict(meta), dict(receipt)


def register_trained_model(
    *,
    checkpoint_dir: str | Path,
    tokenizer_dir: str | Path | None,
    checkpoint_format: str,
    model_key: str,
    training_plan_ref: str | Path,
    schedule_ref: str | Path,
    training_receipt_ref: str | Path,
    base_model_ref: str | Path,
    environment_ref: str | Path,
    write_ref: str | Path,
    workspace_root: str | Path = REPOSITORY_ROOT,
    target_root: str | Path | None = None,
    tokenizer: Any | None = None,
) -> dict[str, Any]:
    """Register one selected full/adapter checkpoint against one frozen slot."""

    from data.training_plan import load_training_plan
    from data.training_schedule import load_training_schedule

    if not isinstance(model_key, str) or not MODEL_KEY_RE.fullmatch(model_key):
        raise ModelRegistryError("invalid model_key")
    if checkpoint_format not in {"full", "adapter"}:
        raise ModelRegistryError("trained checkpoint format must be full or adapter")
    root = Path(workspace_root).resolve()
    tokenizer_source = checkpoint_dir if tokenizer_dir is None else tokenizer_dir
    snapshots = _snapshot_cache(
        ((checkpoint_dir, "checkpoint"), (tokenizer_source, "tokenizer")), root
    )
    _validate_shards(snapshots["checkpoint"], checkpoint_format=checkpoint_format)
    _validate_index_references(snapshots["checkpoint"], workspace_root=root)
    _validate_tokenizer(snapshots["tokenizer"])
    plan_locator, plan_target, plan = load_training_plan(
        training_plan_ref,
        workspace_root=root,
        tokenizer=tokenizer,
    )
    if plan.get("scope") not in {"pilot", "formal"}:
        raise ModelRegistryError("trained model cannot bind an engineering plan")
    plan_dependency = portable_dependency(plan_locator, plan_target, root)
    slots = [slot for slot in plan["ordered_model_slots"] if slot["model_key"] == model_key]
    if len(slots) != 1 or slots[0].get("training_required") is not True:
        raise ModelRegistryError("model_key is not exactly one trainable plan slot")
    slot = slots[0]
    schedule_locator, schedule_target, schedule_meta = load_training_schedule(
        schedule_ref, tokenizer=tokenizer
    )
    schedule_dependency = portable_dependency(schedule_locator, schedule_target, root)
    if schedule_meta.get("training_plan_dependency") != plan_dependency:
        raise ModelRegistryError("schedule differs from the model training plan")
    if schedule_meta.get("train_partition_dependency") != plan.get(
        "train_partition_dependency"
    ):
        raise ModelRegistryError("schedule differs from the model train partition")
    receipt_locator, receipt_target = resolve_locator_ref(
        training_receipt_ref, RECEIPT_ARTIFACT_KIND
    )
    receipt_dependency = portable_dependency(receipt_locator, receipt_target, root)
    receipt_meta, receipt = _validate_receipt_target(
        receipt_target,
        workspace_root=root,
        require_name=True,
        tokenizer=tokenizer,
    )
    require_exact_checkpoint = plan.get("scope") == "formal"
    _assert_receipt_checkpoint_binding(
        receipt,
        snapshots["checkpoint"],
        required=require_exact_checkpoint,
    )
    _, base_target, base_dependency = _dependency_projection(
        base_model_ref, expected_kind=MODEL_ARTIFACT_KIND, workspace_root=root
    )
    base = _validate_model_target(
        base_target,
        workspace_root=root,
        require_name=True,
        tokenizer=tokenizer,
    )
    if base.get("artifact_type") != "base":
        raise ModelRegistryError("trained model requires a registered base model")
    _, _, environment_dependency = _dependency_projection(
        environment_ref,
        expected_kind="stage1-environment",
        workspace_root=root,
    )
    expected_lineage = {
        "training_plan_dependency": plan_dependency,
        "schedule_dependency": schedule_dependency,
        "train_partition_dependency": plan["train_partition_dependency"],
        "base_model_dependency": base_dependency,
        "environment_dependency": environment_dependency,
    }
    for key, dependency in expected_lineage.items():
        if receipt_meta.get(key) != dependency:
            raise ModelRegistryError(f"receipt {key} differs from explicit model ref")
    frozen_tokenizer_revision = schedule_meta.get("tokenizer_revision")
    if (
        not isinstance(frozen_tokenizer_revision, str)
        or base["tokenizer_contract"]["tokenizer_revision"] != frozen_tokenizer_revision
    ):
        raise ModelRegistryError("base tokenizer revision differs from training schedule")
    if (
        receipt.get("model_key") != model_key
        or receipt.get("role") != slot.get("role")
        or receipt.get("seed") != slot.get("seed")
    ):
        raise ModelRegistryError("receipt and model slot role/seed/model-key disagree")
    selected_step = receipt.get("selected_checkpoint_global_step")
    checkpoint_name = Path(checkpoint_dir).resolve().name
    match = re.fullmatch(r"checkpoint-([1-9][0-9]*)", checkpoint_name)
    if match is None:
        raise ModelRegistryError(
            "selected checkpoint directory must use checkpoint-<global_step>"
        )
    if int(match.group(1)) != selected_step:
        raise ModelRegistryError("checkpoint directory step differs from receipt selection")
    # Re-hash immediately before deriving the model ID.  The first snapshot
    # may have been separated from this point by deep plan/schedule/receipt
    # replay, so it is not sufficient as a TOCTOU guard by itself.
    prepublish_checkpoint_inventory = inventory_regular_file_tree(
        checkpoint_dir,
        workspace_root=root,
        label="checkpoint",
        inventory_policy="all-regular-files/v1",
    )
    if prepublish_checkpoint_inventory != snapshots["checkpoint"]:
        raise ModelRegistryError(
            "checkpoint inventory changed during trained-model registration"
        )
    _assert_receipt_checkpoint_binding(
        receipt,
        prepublish_checkpoint_inventory,
        required=require_exact_checkpoint,
    )
    fields: dict[str, Any] = {
        "schema_version": MODEL_SCHEMA_VERSION,
        "model_artifact_id": MODEL_ID_PREFIX + "0" * 64,
        "artifact_type": "trained",
        "scope": plan["scope"],
        "scientific_eligible": plan["scope"] == "formal",
        "model_name": model_key,
        "model_key": model_key,
        "role": slot["role"],
        "seed": slot["seed"],
        "checkpoint_format": checkpoint_format,
        "checkpoint_inventory": snapshots["checkpoint"],
        "tokenizer_inventory": snapshots["tokenizer"],
        "base_inventory": copy.deepcopy(base["base_inventory"]),
        "tokenizer_contract": _tokenizer_contract(
            snapshots["tokenizer"],
            workspace_root=root,
            tokenizer_revision=frozen_tokenizer_revision,
        ),
        "training_plan_dependency": plan_dependency,
        "schedule_dependency": schedule_dependency,
        "training_receipt_dependency": receipt_dependency,
        "train_partition_dependency": plan["train_partition_dependency"],
        "base_model_dependency": base_dependency,
        "environment_dependency": environment_dependency,
    }
    fields["id_inputs"] = _model_id_inputs(fields)
    if (
        fields["tokenizer_contract"]["tokenizer_content_revision"]
        != base["tokenizer_contract"]["tokenizer_content_revision"]
        or fields["tokenizer_contract"]["chat_template_sha256"]
        != base["tokenizer_contract"]["chat_template_sha256"]
    ):
        raise ModelRegistryError("trained checkpoint tokenizer/template drifted from base")
    fields["model_artifact_id"] = MODEL_ID_PREFIX + canonical_sha256(fields["id_inputs"])
    _validate_model_document(fields, workspace_root=root, rehash_sources=False)
    parent = (
        Path(target_root).resolve()
        if target_root is not None
        else root / "exps/causal_context/stage1_p0/models"
    )
    target = parent / fields["model_artifact_id"]
    staging = new_staging_directory(parent, fields["model_artifact_id"])
    dependencies = {
        "training_plan_ref.json": plan_dependency,
        "schedule_ref.json": schedule_dependency,
        "training_receipt_ref.json": receipt_dependency,
        "train_partition_ref.json": plan["train_partition_dependency"],
        "base_model_ref.json": base_dependency,
        "environment_ref.json": environment_dependency,
    }
    try:
        write_canonical_json(staging / "model.json", fields)
        for filename, dependency in dependencies.items():
            write_canonical_json(staging / filename, dependency)
        payload_hash = finalize_target_atomic(
            staging,
            target,
            validate_staging=lambda path: _validate_model_target(
                path,
                workspace_root=root,
                require_name=False,
                tokenizer=tokenizer,
            ),
        )
        postpublish_checkpoint_inventory = inventory_regular_file_tree(
            checkpoint_dir,
            workspace_root=root,
            label="checkpoint",
            inventory_policy="all-regular-files/v1",
        )
        if postpublish_checkpoint_inventory != fields["checkpoint_inventory"]:
            raise ModelRegistryError(
                "checkpoint inventory changed while trained-model registration was published"
            )
        _assert_receipt_checkpoint_binding(
            receipt,
            postpublish_checkpoint_inventory,
            required=require_exact_checkpoint,
        )
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return write_locator_ref(
        write_ref,
        artifact_kind=MODEL_ARTIFACT_KIND,
        artifact_id=fields["model_artifact_id"],
        target=target,
        payload_manifest_sha256=payload_hash,
    )


def validate_training_receipt_artifact(
    receipt_ref: str | Path,
    *,
    workspace_root: str | Path = REPOSITORY_ROOT,
    tokenizer: Any | None = None,
) -> dict[str, Any]:
    locator, target = resolve_locator_ref(receipt_ref, RECEIPT_ARTIFACT_KIND)
    meta, receipt = _validate_receipt_target(
        target,
        workspace_root=Path(workspace_root).resolve(),
        require_name=True,
        tokenizer=tokenizer,
    )
    if locator["artifact_id"] != meta["training_receipt_id"]:
        raise ModelRegistryError("receipt locator artifact ID mismatch")
    return {
        "schema_version": "stage1-training-receipt-validation-report/v1",
        "valid": True,
        "training_receipt_id": meta["training_receipt_id"],
        "model_key": receipt["model_key"],
        "role": receipt["role"],
        "seed": receipt["seed"],
        "selected_checkpoint_global_step": receipt[
            "selected_checkpoint_global_step"
        ],
        "scientific_eligible": meta["scientific_eligible"],
    }


def validate_model_artifact(
    model_ref: str | Path,
    *,
    workspace_root: str | Path = REPOSITORY_ROOT,
    tokenizer: Any | None = None,
) -> dict[str, Any]:
    locator, target = resolve_locator_ref(model_ref, MODEL_ARTIFACT_KIND)
    model = _validate_model_target(
        target,
        workspace_root=Path(workspace_root).resolve(),
        require_name=True,
        tokenizer=tokenizer,
    )
    if locator["artifact_id"] != model["model_artifact_id"]:
        raise ModelRegistryError("model locator artifact ID mismatch")
    return {
        "schema_version": "stage1-model-validation-report/v1",
        "valid": True,
        "model_artifact_id": model["model_artifact_id"],
        "artifact_type": model["artifact_type"],
        "model_key": model["model_key"],
        "checkpoint_format": model["checkpoint_format"],
        "scope": model["scope"],
        "tokenizer_revision": model["tokenizer_contract"]["tokenizer_revision"],
        "scientific_eligible": model["scientific_eligible"],
    }


def validate_model_artifact_target(
    target: str | Path,
    *,
    workspace_root: str | Path = REPOSITORY_ROOT,
    require_name: bool = True,
    tokenizer: Any | None = None,
) -> dict[str, Any]:
    """Deep read-only validation used by one-way plan model dependencies."""

    return _validate_model_target(
        Path(target),
        workspace_root=Path(workspace_root).resolve(),
        require_name=require_name,
        tokenizer=tokenizer,
    )


def _registry_id_inputs(registry: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: copy.deepcopy(registry[key])
        for key in (
            "schema_version",
            "scope",
            "scientific_eligible",
            "training_plan_dependency",
            "ordered_model_keys",
            "entries",
        )
    }


def _expected_registry_keys(plan: Mapping[str, Any], scope: str) -> list[str]:
    if scope == "engineering-smoke":
        if plan.get("scope") != "engineering-smoke":
            raise ModelRegistryError("engineering registry requires an engineering plan")
        return [slot["model_key"] for slot in plan["ordered_model_slots"]]
    if scope == "pilot":
        if plan.get("scope") not in {"pilot", "formal"}:
            raise ModelRegistryError("pilot registry requires a pilot/formal plan")
        keys = list(plan.get("pilot_slot_keys", []))
        if not keys:
            raise ModelRegistryError("pilot plan has no frozen pilot slot set")
        return keys
    if scope == "formal":
        if plan.get("scope") != "formal" or plan.get("scientific_eligible") is not True:
            raise ModelRegistryError("formal registry requires a scientific-eligible formal plan")
        keys = [slot["model_key"] for slot in plan["ordered_model_slots"]]
        expected = [
            f"{role}/seed-{seed}"
            for seed in (42, 43, 44)
            for role in ("M_LD", "M_drop")
        ]
        if keys != expected or len(keys) != 6:
            raise ModelRegistryError("formal registry requires exactly six frozen Stage 1 slots")
        return keys
    raise ModelRegistryError(f"unsupported registry scope: {scope}")


def finalize_model_registry(
    *,
    scope: str,
    training_plan_ref: str | Path,
    model_bindings: Sequence[tuple[str, str | Path]],
    write_ref: str | Path,
    workspace_root: str | Path = REPOSITORY_ROOT,
    target_root: str | Path | None = None,
    tokenizer: Any | None = None,
) -> dict[str, Any]:
    """Freeze an exact plan-slot to model-artifact bijection."""

    from data.training_plan import load_training_plan

    root = Path(workspace_root).resolve()
    plan_locator, plan_target, plan = load_training_plan(
        training_plan_ref,
        workspace_root=root,
        tokenizer=tokenizer,
    )
    plan_dependency = portable_dependency(plan_locator, plan_target, root)
    expected_keys = _expected_registry_keys(plan, scope)
    provided_keys = [key for key, _ in model_bindings]
    if len(provided_keys) != len(set(provided_keys)):
        raise ModelRegistryError("registry contains a duplicate model-key binding")
    if set(provided_keys) != set(expected_keys) or len(provided_keys) != len(expected_keys):
        raise ModelRegistryError(
            f"registry slot set mismatch: missing={sorted(set(expected_keys)-set(provided_keys))}, "
            f"extra={sorted(set(provided_keys)-set(expected_keys))}"
        )
    refs = dict(model_bindings)
    slots = {slot["model_key"]: slot for slot in plan["ordered_model_slots"]}
    entries: list[dict[str, Any]] = []
    seen_artifacts: set[str] = set()
    for model_key in expected_keys:
        locator, target = resolve_locator_ref(refs[model_key], MODEL_ARTIFACT_KIND)
        model = _validate_model_target(
            target,
            workspace_root=root,
            require_name=True,
            tokenizer=tokenizer,
        )
        dependency = portable_dependency(locator, target, root)
        if model["model_artifact_id"] in seen_artifacts:
            raise ModelRegistryError("one model artifact cannot satisfy multiple registry slots")
        seen_artifacts.add(model["model_artifact_id"])
        slot = slots[model_key]
        if scope == "engineering-smoke":
            planned = plan.get("non_training_model_dependencies", {}).get(
                model_key
            )
            if (
                slot.get("training_required") is not False
                or model.get("artifact_type") != "legacy-smoke-only"
                or model.get("scope") != "engineering-smoke"
                or model.get("scientific_eligible") is not False
                or model.get("model_key") != model_key
                or model.get("role") != slot.get("role")
                or model.get("seed") != slot.get("seed")
            ):
                raise ModelRegistryError(
                    "engineering slots require their matching legacy-smoke-only artifact"
                )
            if dependency != planned:
                raise ModelRegistryError(
                    "engineering slot differs from the legacy model frozen in its plan"
                )
            role = model["role"]
            seed = model["seed"]
        else:
            if model.get("artifact_type") != "trained":
                raise ModelRegistryError("pilot/formal slots require registered trained models")
            if (
                model.get("model_key") != model_key
                or model.get("role") != slot.get("role")
                or model.get("seed") != slot.get("seed")
                or model.get("training_plan_dependency") != plan_dependency
            ):
                raise ModelRegistryError("registered model differs from plan slot lineage")
            if scope == "formal" and (
                model.get("scope") != "formal" or model.get("scientific_eligible") is not True
            ):
                raise ModelRegistryError("formal slot model is not scientific eligible")
            role = model["role"]
            seed = model["seed"]
        entries.append(
            {
                "model_key": model_key,
                "role": role,
                "seed": seed,
                "model_dependency": dependency,
                "model_artifact_id": model["model_artifact_id"],
                "checkpoint_format": model["checkpoint_format"],
                "model_scientific_eligible": model["scientific_eligible"],
            }
        )
    scientific_eligible = scope == "formal" and all(
        entry["model_scientific_eligible"] for entry in entries
    )
    registry: dict[str, Any] = {
        "schema_version": REGISTRY_SCHEMA_VERSION,
        "model_registry_id": REGISTRY_ID_PREFIX + "0" * 64,
        "scope": scope,
        "scientific_eligible": scientific_eligible,
        "training_plan_dependency": plan_dependency,
        "ordered_model_keys": expected_keys,
        "entries": entries,
    }
    registry["id_inputs"] = _registry_id_inputs(registry)
    registry["model_registry_id"] = REGISTRY_ID_PREFIX + canonical_sha256(
        registry["id_inputs"]
    )
    validate_json_schema(registry, REGISTRY_SCHEMA)
    parent = (
        Path(target_root).resolve()
        if target_root is not None
        else root / "exps/causal_context/stage1_p0/model_registries"
    )
    target = parent / registry["model_registry_id"]
    staging = new_staging_directory(parent, registry["model_registry_id"])
    try:
        write_canonical_json(staging / "registry.json", registry)
        write_canonical_json(staging / "training_plan_ref.json", plan_dependency)
        payload_hash = finalize_target_atomic(
            staging,
            target,
            validate_staging=lambda path: _validate_registry_target(
                path,
                workspace_root=root,
                require_name=False,
                tokenizer=tokenizer,
            ),
        )
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return write_locator_ref(
        write_ref,
        artifact_kind=REGISTRY_ARTIFACT_KIND,
        artifact_id=registry["model_registry_id"],
        target=target,
        payload_manifest_sha256=payload_hash,
    )


def _validate_registry_target(
    target: Path,
    *,
    workspace_root: str | Path,
    require_name: bool,
    rehash_cache: dict[str, Path] | None = None,
    tokenizer: Any | None = None,
) -> dict[str, Any]:
    from data.training_plan import validate_training_plan_target

    if rehash_cache is None:
        rehash_cache = {}
    ensure_exact_file_set(
        target, {"registry.json", "training_plan_ref.json", "payload_manifest.json"}
    )
    validate_payload_manifest(target)
    registry = load_json(target / "registry.json")
    if not isinstance(registry, Mapping):
        raise ModelRegistryError("registry.json is not an object")
    validate_json_schema(registry, REGISTRY_SCHEMA)
    if registry.get("id_inputs") != _registry_id_inputs(registry):
        raise ModelRegistryError("registry ID inputs are not canonical")
    registry_id = REGISTRY_ID_PREFIX + canonical_sha256(registry["id_inputs"])
    if registry.get("model_registry_id") != registry_id:
        raise ModelRegistryError("registry ID is not reproducible")
    if require_name and target.name != registry_id:
        raise ModelRegistryError("registry target directory does not match artifact ID")
    plan_dependency = load_json(target / "training_plan_ref.json")
    validate_dependency_ref(plan_dependency, expected_kind="training-plan")
    if plan_dependency != registry.get("training_plan_dependency"):
        raise ModelRegistryError("registry training plan dependency mismatch")
    plan_target = resolve_dependency_target(plan_dependency, workspace_root)
    plan = validate_training_plan_target(
        plan_target,
        workspace_root=workspace_root,
        tokenizer=tokenizer,
    )
    expected_keys = _expected_registry_keys(plan, registry["scope"])
    if registry.get("ordered_model_keys") != expected_keys:
        raise ModelRegistryError("registry ordered slot set differs from plan")
    entries = registry.get("entries")
    if not isinstance(entries, list) or [row.get("model_key") for row in entries] != expected_keys:
        raise ModelRegistryError("registry entries are missing, extra, duplicated, or unordered")
    if len({row.get("model_artifact_id") for row in entries}) != len(entries):
        raise ModelRegistryError("registry reuses one artifact for multiple slots")
    slots = {slot["model_key"]: slot for slot in plan["ordered_model_slots"]}
    all_eligible = True
    for entry in entries:
        dependency = validate_dependency_ref(
            entry.get("model_dependency", {}), expected_kind=MODEL_ARTIFACT_KIND
        )
        model_target = resolve_dependency_target(dependency, workspace_root)
        model = _validate_model_target(
            model_target,
            workspace_root=workspace_root,
            require_name=True,
            rehash_cache=rehash_cache,
            tokenizer=tokenizer,
        )
        if (
            entry.get("model_artifact_id") != model.get("model_artifact_id")
            or entry.get("checkpoint_format") != model.get("checkpoint_format")
            or entry.get("model_scientific_eligible") is not model.get("scientific_eligible")
        ):
            raise ModelRegistryError("registry entry differs from model artifact")
        slot = slots[entry["model_key"]]
        if registry["scope"] == "engineering-smoke":
            planned = plan.get("non_training_model_dependencies", {}).get(
                entry["model_key"]
            )
            if (
                model.get("artifact_type") != "legacy-smoke-only"
                or model.get("scope") != "engineering-smoke"
                or model.get("scientific_eligible") is not False
                or slot.get("training_required") is not False
                or model.get("model_key") != entry.get("model_key")
                or model.get("role") != slot.get("role")
                or model.get("seed") != slot.get("seed")
            ):
                raise ModelRegistryError(
                    "engineering registry contains a trained, base, or foreign model"
                )
            if dependency != planned:
                raise ModelRegistryError(
                    "engineering registry differs from its plan-frozen legacy model"
                )
            if entry.get("role") != model.get("role") or entry.get(
                "seed"
            ) != model.get("seed"):
                raise ModelRegistryError(
                    "engineering registry role/seed differs from legacy artifact"
                )
        else:
            if (
                model.get("artifact_type") != "trained"
                or model.get("model_key") != entry.get("model_key")
                or model.get("role") != slot.get("role")
                or model.get("seed") != slot.get("seed")
                or model.get("training_plan_dependency") != plan_dependency
                or entry.get("role") != model.get("role")
                or entry.get("seed") != model.get("seed")
            ):
                raise ModelRegistryError("registry trained entry differs from its plan slot")
        all_eligible = all_eligible and bool(model.get("scientific_eligible"))
    expected_eligibility = registry["scope"] == "formal" and all_eligible
    if registry.get("scientific_eligible") is not expected_eligibility:
        raise ModelRegistryError("registry scientific eligibility is not fail-closed")
    if registry["scope"] == "formal" and not expected_eligibility:
        raise ModelRegistryError("formal registry contains an ineligible model")
    return dict(registry)


def validate_model_registry(
    registry_ref: str | Path,
    *,
    workspace_root: str | Path = REPOSITORY_ROOT,
    tokenizer: Any | None = None,
) -> dict[str, Any]:
    locator, target = resolve_locator_ref(registry_ref, REGISTRY_ARTIFACT_KIND)
    registry = _validate_registry_target(
        target,
        workspace_root=Path(workspace_root).resolve(),
        require_name=True,
        tokenizer=tokenizer,
    )
    if locator["artifact_id"] != registry["model_registry_id"]:
        raise ModelRegistryError("registry locator artifact ID mismatch")
    return {
        "schema_version": "stage1-model-registry-validation-report/v1",
        "valid": True,
        "model_registry_id": registry["model_registry_id"],
        "scope": registry["scope"],
        "scientific_eligible": registry["scientific_eligible"],
        "ordered_model_keys": registry["ordered_model_keys"],
    }


def validate_model_registry_target(
    target: str | Path,
    *,
    workspace_root: str | Path = REPOSITORY_ROOT,
    require_name: bool = True,
    tokenizer: Any | None = None,
) -> dict[str, Any]:
    """Deep read-only validator for replay from an embedded dependency target."""

    return _validate_registry_target(
        Path(target),
        workspace_root=Path(workspace_root).resolve(),
        require_name=require_name,
        tokenizer=tokenizer,
    )


def resolve_registered_model(
    *,
    registry_ref: str | Path,
    model_key: str,
    workspace_root: str | Path = REPOSITORY_ROOT,
    tokenizer: Any | None = None,
) -> ResolvedRegisteredModel:
    """Resolve the only permitted Stage 1 inference/scoring model source."""

    root = Path(workspace_root).resolve()
    locator, target = resolve_locator_ref(registry_ref, REGISTRY_ARTIFACT_KIND)
    resolved = _resolve_registered_model_target(
        target=target,
        model_key=model_key,
        workspace_root=root,
        tokenizer=tokenizer,
    )
    if locator["artifact_id"] != resolved.registry_id:
        raise ModelRegistryError("registry locator artifact ID mismatch")
    return resolved


def resolve_registered_model_dependency(
    *,
    registry_dependency: Mapping[str, Any],
    model_key: str,
    workspace_root: str | Path = REPOSITORY_ROOT,
    tokenizer: Any | None = None,
) -> ResolvedRegisteredModel:
    """Resolve a registry embedded as a portable dependency (artifact replay)."""

    root = Path(workspace_root).resolve()
    dependency = validate_dependency_ref(
        registry_dependency, expected_kind=REGISTRY_ARTIFACT_KIND
    )
    target = resolve_dependency_target(dependency, root)
    resolved = _resolve_registered_model_target(
        target=target,
        model_key=model_key,
        workspace_root=root,
        tokenizer=tokenizer,
    )
    if dependency["artifact_id"] != resolved.registry_id:
        raise ModelRegistryError("registry dependency artifact ID mismatch")
    return resolved


def _resolve_registered_model_target(
    *,
    target: Path,
    model_key: str,
    workspace_root: Path,
    tokenizer: Any | None = None,
) -> ResolvedRegisteredModel:
    rehash_cache: dict[str, Path] = {}
    registry = _validate_registry_target(
        target,
        workspace_root=workspace_root,
        require_name=True,
        rehash_cache=rehash_cache,
        tokenizer=tokenizer,
    )
    matches = [entry for entry in registry["entries"] if entry["model_key"] == model_key]
    if len(matches) != 1:
        raise ModelRegistryError("model_key must resolve to exactly one registry entry")
    model_target = resolve_dependency_target(
        matches[0]["model_dependency"], workspace_root
    )
    model = _validate_model_target(
        model_target,
        workspace_root=workspace_root,
        require_name=True,
        rehash_cache=rehash_cache,
        tokenizer=tokenizer,
    )
    source_contract = ResolvedModelSourceContract(
        workspace_root=workspace_root.resolve(),
        checkpoint_inventory=copy.deepcopy(model["checkpoint_inventory"]),
        tokenizer_inventory=copy.deepcopy(model["tokenizer_inventory"]),
        base_inventory=copy.deepcopy(model["base_inventory"]),
    )
    # Do not reuse the deep validator's cache here.  The handle is minted only
    # after a new load-style lease has freshly inventoried every returned root
    # and verified that none changed across that final check.
    with verified_model_source_lease(source_contract) as sources:
        resolved = ResolvedRegisteredModel(
            registry_id=registry["model_registry_id"],
            model_key=model_key,
            role=str(matches[0]["role"]),
            seed=matches[0]["seed"],
            checkpoint_format=model["checkpoint_format"],
            checkpoint_path=sources.checkpoint_path,
            tokenizer_path=sources.tokenizer_path,
            base_model_path=sources.base_model_path,
            tokenizer_revision=model["tokenizer_contract"]["tokenizer_revision"],
            tokenizer_content_revision=model["tokenizer_contract"][
                "tokenizer_content_revision"
            ],
            model_artifact_id=model["model_artifact_id"],
            scientific_eligible=bool(registry["scientific_eligible"]),
            source_contract=source_contract,
        )
    return resolved


__all__ = [
    "MODEL_ARTIFACT_KIND",
    "RECEIPT_ARTIFACT_KIND",
    "REGISTRY_ARTIFACT_KIND",
    "ModelRegistryError",
    "ResolvedModelSourceContract",
    "ResolvedRegisteredModel",
    "VerifiedModelSourceLease",
    "VerifiedModelSourcePaths",
    "finalize_model_registry",
    "inventory_regular_file_tree",
    "register_base_model",
    "register_legacy_model",
    "register_trained_model",
    "register_training_receipt",
    "resolve_registered_model",
    "resolve_registered_model_dependency",
    "verified_model_source_lease",
    "validate_model_artifact",
    "validate_model_artifact_target",
    "validate_model_registry",
    "validate_model_registry_target",
    "validate_training_receipt_artifact",
]
