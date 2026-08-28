"""Durable, intent-bound checkpointing for the formal Stage-1 lexicon build.

The checkpoint is deliberately private runtime state, not a scientific
artifact.  It stores corpus-derived provider payloads with owner-only
permissions so a process restart can reuse completed provider slots without
silently exceeding the frozen physical-attempt budgets.
"""

from __future__ import annotations

import copy
import fcntl
import hashlib
import json
import os
import re
import shutil
import stat
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence


CHECKPOINT_SCHEMA_VERSION = "stage1-formal-lexicon-checkpoint/v2"
ATTEMPT_SCHEMA_VERSION = "stage1-formal-provider-attempt/v2"
ATTEMPT_HEAD_SCHEMA_VERSION = "stage1-formal-provider-attempt-head/v1"
SLOT_SCHEMA_VERSION = "stage1-formal-provider-slot/v2"
CANDIDATE_COMMIT_SCHEMA_VERSION = "stage1-formal-candidate-commit/v2"

PROVIDER_SLOTS = {
    "tavily": ("query_1", "query_2", "query_3"),
    "deepseek": (
        "context_judge",
        "web_evidence_judge",
        "final_lexicon_judge",
    ),
}
_HASH_RE = re.compile(r"[0-9a-f]{64}\Z")
_GENESIS_RESERVATION_SHA256 = "0" * 64
_ATOMIC_TEMP_SUFFIX_RE = re.compile(r"[A-Za-z0-9_-]{6,32}\Z")
_SECRET_KEYS = frozenset(
    {
        "api_key",
        "apikey",
        "authorization",
        "access_token",
        "auth_token",
        "password",
        "secret",
        "token",
    }
)


class FormalCheckpointError(RuntimeError):
    """Raised when checkpoint state is unsafe, inconsistent, or exhausted."""


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            _plain_copy(value),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise FormalCheckpointError("checkpoint payload is not canonical JSON") from exc


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _plain_copy(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain_copy(inner) for key, inner in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain_copy(inner) for inner in value]
    return copy.deepcopy(value)


def _reject_secret_fields(value: Any, *, label: str) -> None:
    if isinstance(value, Mapping):
        for key, inner in value.items():
            if str(key).casefold() in _SECRET_KEYS:
                raise FormalCheckpointError(f"{label} contains a forbidden secret field")
            _reject_secret_fields(inner, label=label)
    elif isinstance(value, (list, tuple)):
        for inner in value:
            _reject_secret_fields(inner, label=label)


def _iter_strings(value: Any):
    if isinstance(value, Mapping):
        for key, inner in value.items():
            yield str(key)
            yield from _iter_strings(inner)
    elif isinstance(value, (list, tuple)):
        for inner in value:
            yield from _iter_strings(inner)
    elif isinstance(value, str):
        yield value


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _ensure_private_directory(path: Path, *, create: bool = False) -> None:
    if path.is_symlink():
        raise FormalCheckpointError(f"checkpoint directory must not be a symlink: {path.name}")
    existed = path.exists()
    if create:
        path.mkdir(parents=True, exist_ok=True, mode=0o700)
    if not path.is_dir():
        raise FormalCheckpointError(f"checkpoint directory is missing: {path.name}")
    if create and not existed:
        try:
            os.chmod(path, 0o700)
        except OSError as exc:
            raise FormalCheckpointError("cannot enforce checkpoint directory mode 0700") from exc
    try:
        mode = path.stat(follow_symlinks=False).st_mode
    except OSError as exc:
        raise FormalCheckpointError("cannot inspect checkpoint directory permissions") from exc
    if not stat.S_ISDIR(mode) or mode & 0o077:
        raise FormalCheckpointError(
            f"checkpoint directory is not owner-only: {path.name}"
        )


def _atomic_write_json(path: Path, value: Any) -> None:
    _ensure_private_directory(path.parent, create=True)
    if path.is_symlink():
        raise FormalCheckpointError(f"checkpoint file must not be a symlink: {path.name}")
    payload = _canonical_bytes(value) + b"\n"
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{path.name}.",
            dir=path.parent,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            os.chmod(temporary, 0o600)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        os.chmod(path, 0o600)
        _fsync_directory(path.parent)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _read_json(path: Path) -> dict[str, Any]:
    """Read one private regular file without following a raced-in symlink."""

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise FormalCheckpointError(
            f"checkpoint file is missing or unsafe: {path.name}"
        ) from exc
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise FormalCheckpointError(
                f"checkpoint file is missing or unsafe: {path.name}"
            )
        if metadata.st_mode & 0o077:
            raise FormalCheckpointError(
                f"checkpoint file is not owner-only: {path.name}"
            )
        with os.fdopen(descriptor, "r", encoding="utf-8") as handle:
            descriptor = -1
            text = handle.read()
        value = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite,
        )
    except FormalCheckpointError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise FormalCheckpointError(f"checkpoint file is malformed: {path.name}") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    if not isinstance(value, dict):
        raise FormalCheckpointError(f"checkpoint file is not an object: {path.name}")
    return value


def _checkpoint_files(
    directory: Path,
    *,
    final_name_re: re.Pattern[str],
    label: str,
    clean_orphans: bool = False,
) -> list[Path]:
    """Return checkpoint files, removing only our strictly named temp orphans.

    A SIGKILL can leave the private file created by ``NamedTemporaryFile``
    immediately before ``os.replace``.  Such a file is not checkpoint state and
    is safe to discard while holding the single-writer lock.  Every other
    unexpected directory entry remains a fail-closed condition.
    """

    _ensure_private_directory(directory)
    paths: list[Path] = []
    removed_orphan = False
    for path in sorted(directory.iterdir(), key=lambda item: item.name):
        name = path.name
        if final_name_re.fullmatch(name):
            try:
                metadata = path.lstat()
            except OSError as exc:
                raise FormalCheckpointError(
                    f"checkpoint {label} directory contains an unsafe entry"
                ) from exc
            if not stat.S_ISREG(metadata.st_mode):
                raise FormalCheckpointError(
                    f"checkpoint {label} directory contains an unsafe entry"
                )
            paths.append(path)
            continue
        orphan = re.fullmatch(r"\.(.+)\.([A-Za-z0-9_-]{6,32})", name)
        if orphan is None or not final_name_re.fullmatch(orphan.group(1)):
            raise FormalCheckpointError(
                f"checkpoint {label} directory contains an unexpected entry"
            )
        if _ATOMIC_TEMP_SUFFIX_RE.fullmatch(orphan.group(2)) is None:
            raise FormalCheckpointError(
                f"checkpoint {label} directory contains an unexpected entry"
            )
        try:
            metadata = path.lstat()
        except OSError as exc:
            raise FormalCheckpointError(
                f"checkpoint {label} directory contains an unsafe temp entry"
            ) from exc
        if not stat.S_ISREG(metadata.st_mode):
            raise FormalCheckpointError(
                f"checkpoint {label} directory contains an unsafe temp entry"
            )
        if clean_orphans:
            try:
                path.unlink()
            except OSError as exc:
                raise FormalCheckpointError(
                    f"cannot remove checkpoint {label} atomic temp orphan"
                ) from exc
            removed_orphan = True
    if removed_orphan:
        _fsync_directory(directory)
    return paths


def _reject_duplicate_keys(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise FormalCheckpointError(f"duplicate checkpoint JSON key: {key}")
        result[key] = value
    return result


def _reject_nonfinite(value: str) -> None:
    raise FormalCheckpointError(f"non-finite checkpoint JSON number: {value}")


def _atomic_write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    _ensure_private_directory(path.parent, create=True)
    if path.is_symlink():
        raise FormalCheckpointError(f"materialized file must not be a symlink: {path.name}")
    payload = b"".join(_canonical_bytes(row) + b"\n" for row in rows)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{path.name}.", dir=path.parent, delete=False
        ) as handle:
            temporary = Path(handle.name)
            os.chmod(temporary, 0o600)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        os.chmod(path, 0o600)
        _fsync_directory(path.parent)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _normalized_provider_caps(
    provider_caps: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    if set(provider_caps) != set(PROVIDER_SLOTS):
        raise FormalCheckpointError("checkpoint provider caps must cover Tavily and DeepSeek exactly")
    normalized: dict[str, dict[str, Any]] = {}
    for provider in sorted(PROVIDER_SLOTS):
        raw = provider_caps[provider]
        if isinstance(raw, Mapping):
            if set(raw) != {"cap", "scope_id"}:
                raise FormalCheckpointError("provider budget must contain only cap and scope_id")
            cap = raw.get("cap")
            scope_id = raw.get("scope_id")
        else:
            cap = raw
            scope_id = f"{provider}-budget/v1"
        if isinstance(cap, bool) or not isinstance(cap, int) or cap < 0:
            raise FormalCheckpointError("provider physical-attempt cap must be non-negative")
        if not isinstance(scope_id, str) or not scope_id or scope_id.strip() != scope_id:
            raise FormalCheckpointError("provider budget scope_id must be canonical")
        normalized[provider] = {"cap": cap, "scope_id": scope_id}
    return normalized


@dataclass(frozen=True)
class CheckpointSpec:
    checkpoint_id: str
    intent: Mapping[str, Any]
    provider_caps: Mapping[str, Mapping[str, Any]]
    candidate_frame: tuple[Mapping[str, Any], ...]
    max_slot_attempts: int
    active_provider_slots: Mapping[str, tuple[str, ...]]

    @classmethod
    def build(
        cls,
        *,
        intent: Mapping[str, Any],
        provider_caps: Mapping[str, Any],
        candidate_frame: Sequence[Mapping[str, Any]] | None = None,
        max_slot_attempts: int = 3,
        active_provider_slots: Mapping[str, Sequence[str]] | None = None,
    ) -> "CheckpointSpec":
        required_intent = {
            "authorization_sha256",
            "builder_code_sha256",
            "config_sha256",
            "data_build_id",
            "protocol_code_sha256s",
            "train_records_sha256",
        }
        if not isinstance(intent, Mapping) or not required_intent.issubset(intent):
            raise FormalCheckpointError("checkpoint intent lacks formal code/config/data bindings")
        _reject_secret_fields(intent, label="checkpoint intent")
        canonical_intent = _plain_copy(dict(intent))
        authorization_sha256 = canonical_intent.get("authorization_sha256")
        if not isinstance(authorization_sha256, str) or not _HASH_RE.fullmatch(
            authorization_sha256
        ):
            raise FormalCheckpointError("checkpoint intent has an invalid authorization hash")
        if (
            isinstance(max_slot_attempts, bool)
            or not isinstance(max_slot_attempts, int)
            or max_slot_attempts <= 0
        ):
            raise FormalCheckpointError("max_slot_attempts must be a positive integer")
        frame: list[Mapping[str, Any]] = []
        for expected_rank, raw in enumerate(candidate_frame or (), start=1):
            if not isinstance(raw, Mapping):
                raise FormalCheckpointError("candidate frame contains a non-object row")
            row = _plain_copy(dict(raw))
            if row.get("rank") != expected_rank:
                raise FormalCheckpointError("candidate frame ranks must be contiguous from one")
            term = row.get("term")
            if not isinstance(term, str) or not term or term.strip() != term:
                raise FormalCheckpointError("candidate frame contains an invalid term")
            frame.append(MappingProxyType(row))
        terms = [str(row["term"]) for row in frame]
        if len(terms) != len(set(terms)):
            raise FormalCheckpointError("candidate frame terms must be unique")
        if active_provider_slots is None:
            normalized_active_slots = {
                provider: tuple(slots) for provider, slots in PROVIDER_SLOTS.items()
            }
        else:
            if set(active_provider_slots) != set(PROVIDER_SLOTS):
                raise FormalCheckpointError(
                    "active provider slots must cover Tavily and DeepSeek exactly"
                )
            normalized_active_slots: dict[str, tuple[str, ...]] = {}
            for provider, known_slots in PROVIDER_SLOTS.items():
                raw_slots = active_provider_slots[provider]
                if isinstance(raw_slots, (str, bytes)) or not isinstance(
                    raw_slots, Sequence
                ):
                    raise FormalCheckpointError("active provider slots must be sequences")
                supplied = tuple(raw_slots)
                if (
                    len(supplied) != len(set(supplied))
                    or any(slot not in known_slots for slot in supplied)
                ):
                    raise FormalCheckpointError("active provider slots are invalid")
                normalized_active_slots[provider] = tuple(
                    slot for slot in known_slots if slot in supplied
                )
        checkpoint_id = "fchk-" + _sha256(
            {
                "schema_version": CHECKPOINT_SCHEMA_VERSION,
                "intent": canonical_intent,
            }
        )
        return cls(
            checkpoint_id=checkpoint_id,
            intent=MappingProxyType(canonical_intent),
            provider_caps=MappingProxyType(
                {
                    key: MappingProxyType(value)
                    for key, value in _normalized_provider_caps(provider_caps).items()
                }
            ),
            candidate_frame=tuple(frame),
            max_slot_attempts=max_slot_attempts,
            active_provider_slots=MappingProxyType(normalized_active_slots),
        )

    def to_payload(self) -> dict[str, Any]:
        payload = {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "checkpoint_id": self.checkpoint_id,
            "intent": _plain_copy(self.intent),
            "provider_caps": _plain_copy(self.provider_caps),
            "candidate_frame": _plain_copy(self.candidate_frame),
            "candidate_frame_sha256": _sha256(self.candidate_frame),
            "max_slot_attempts": self.max_slot_attempts,
            "active_provider_slots": {
                provider: list(self.active_provider_slots[provider])
                for provider in sorted(PROVIDER_SLOTS)
            },
        }
        payload["spec_sha256"] = _sha256(payload)
        return payload


@dataclass(frozen=True)
class AttemptReservation:
    reservation_id: str
    provider: str
    rank: int
    slot: str
    attempt: int
    request_sha256: str
    sequence: int


@dataclass(frozen=True)
class AttemptOutcome:
    reservation: AttemptReservation
    status: str


@dataclass(frozen=True)
class SlotSuccess:
    provider: str
    rank: int
    slot: str
    attempt: int
    request_sha256: str
    response: Any
    capture: Any


@dataclass(frozen=True)
class CandidateCommit:
    rank: int
    commit_sha256: str
    previous_commit_sha256: str
    row_bundle: Mapping[str, Any]


class FormalLexiconCheckpoint:
    """Single-writer durable checkpoint bound to one immutable formal intent."""

    def __init__(
        self,
        root: Path,
        spec: CheckpointSpec,
        *,
        forbidden_values: Sequence[str] = (),
    ) -> None:
        self.root = root
        self.spec = spec
        self._forbidden_values = tuple(
            value for value in forbidden_values if isinstance(value, str) and value
        )
        self._forbidden_hashes = tuple(
            hashlib.sha256(value.encode("utf-8")).hexdigest()
            for value in self._forbidden_values
        )
        self._lock_handle: Any | None = None
        self._closed = False
        self._poisoned = False
        self._owner_pid = os.getpid()
        self._mutex = threading.RLock()
        self._attempt_rows: list[dict[str, Any]] = []
        self._attempts_by_sequence: dict[int, dict[str, Any]] = {}
        self._attempts_by_slot: dict[
            tuple[str, int, str], list[dict[str, Any]]
        ] = {}
        self._provider_counts = {provider: 0 for provider in PROVIDER_SLOTS}
        self._ambiguous_attempt_count = 0
        self._slot_index: dict[tuple[str, int, str], dict[str, Any]] = {}
        self._candidate_commits: list[dict[str, Any]] = []

    @classmethod
    def create(
        cls,
        root: str | Path,
        spec: CheckpointSpec,
        *,
        forbidden_values: Sequence[str] = (),
    ) -> "FormalLexiconCheckpoint":
        target = Path(root)
        checkpoint = cls(target, spec, forbidden_values=forbidden_values)
        checkpoint._audit_forbidden_values(
            spec.to_payload(), label="checkpoint intent and candidate frame"
        )
        if target.is_symlink() or target.exists():
            raise FormalCheckpointError("refusing to replace an existing checkpoint")
        parent = target.parent
        _ensure_private_directory(parent, create=True)
        temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=parent))
        try:
            os.chmod(temporary, 0o700)
            _atomic_write_json(temporary / "manifest.json", spec.to_payload())
            for name in ("attempts", "slots", "candidates"):
                _ensure_private_directory(temporary / name, create=True)
            _atomic_write_json(
                temporary / "attempts" / "HEAD.json",
                cls._attempt_head_payload(
                    spec,
                    sequence=0,
                    reservation_head_sha256=_GENESIS_RESERVATION_SHA256,
                    provider_counts={provider: 0 for provider in PROVIDER_SLOTS},
                ),
            )
            lock_path = temporary / "writer.lock"
            descriptor = os.open(lock_path, os.O_CREAT | os.O_WRONLY, 0o600)
            os.close(descriptor)
            os.chmod(lock_path, 0o600)
            _fsync_directory(temporary)
            os.replace(temporary, target)
            _fsync_directory(parent)
        except Exception:
            if temporary.exists():
                shutil.rmtree(temporary)
            raise
        checkpoint._acquire_lock()
        checkpoint._validate_and_reconcile()
        return checkpoint

    @classmethod
    def resume(
        cls,
        root: str | Path,
        spec: CheckpointSpec,
        *,
        forbidden_values: Sequence[str] = (),
    ) -> "FormalLexiconCheckpoint":
        target = Path(root)
        if target.is_symlink() or not target.is_dir():
            raise FormalCheckpointError("checkpoint does not exist or is unsafe")
        checkpoint = cls(target, spec, forbidden_values=forbidden_values)
        checkpoint._audit_forbidden_values(
            spec.to_payload(), label="checkpoint intent and candidate frame"
        )
        checkpoint._acquire_lock()
        try:
            checkpoint._validate_and_reconcile()
        except Exception:
            checkpoint.close()
            raise
        return checkpoint

    @classmethod
    def open_or_create(
        cls,
        root: str | Path,
        spec: CheckpointSpec,
        *,
        forbidden_values: Sequence[str] = (),
    ) -> "FormalLexiconCheckpoint":
        target = Path(root)
        if target.exists() or target.is_symlink():
            return cls.resume(target, spec, forbidden_values=forbidden_values)
        try:
            return cls.create(target, spec, forbidden_values=forbidden_values)
        except FormalCheckpointError:
            if target.is_dir() and not target.is_symlink():
                return cls.resume(target, spec, forbidden_values=forbidden_values)
            raise

    def __enter__(self) -> "FormalLexiconCheckpoint":
        self._require_open()
        return self

    def __exit__(self, _type: Any, _value: Any, _traceback: Any) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def close(self) -> None:
        with self._mutex:
            if self._closed:
                return
            if self._lock_handle is not None:
                try:
                    # A forked child must not explicitly unlock the parent's
                    # shared open-file description.
                    if os.getpid() == self._owner_pid:
                        fcntl.flock(self._lock_handle.fileno(), fcntl.LOCK_UN)
                finally:
                    self._lock_handle.close()
            self._lock_handle = None
            self._closed = True

    def _acquire_lock(self) -> None:
        _ensure_private_directory(self.root)
        lock_path = self.root / "writer.lock"
        flags = (
            os.O_CREAT
            | os.O_RDWR
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        try:
            descriptor = os.open(lock_path, flags, 0o600)
        except OSError as exc:
            raise FormalCheckpointError(
                "checkpoint writer lock is missing or unsafe"
            ) from exc
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_mode & 0o077:
            os.close(descriptor)
            raise FormalCheckpointError(
                "checkpoint writer lock is not an owner-only regular file"
            )
        handle = os.fdopen(descriptor, "r+b", buffering=0)
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            handle.close()
            raise FormalCheckpointError("checkpoint already has an active writer") from exc
        self._lock_handle = handle

    def _require_open(self) -> None:
        if self._closed or self._lock_handle is None:
            raise FormalCheckpointError("checkpoint is closed")
        if os.getpid() != self._owner_pid:
            raise FormalCheckpointError(
                "checkpoint handle cannot be used after process fork"
            )
        if self._poisoned:
            raise FormalCheckpointError(
                "checkpoint requires close and resume after an interrupted durable write"
            )

    def _audit_forbidden_values(self, value: Any, *, label: str) -> None:
        if not self._forbidden_values:
            return
        for text in _iter_strings(value):
            if any(secret in text for secret in self._forbidden_values) or any(
                secret_hash in text for secret_hash in self._forbidden_hashes
            ):
                raise FormalCheckpointError(
                    f"{label} contains a preflight credential value"
                )

    def _validate_and_reconcile(self) -> None:
        self._require_open()
        manifest = _read_json(self.root / "manifest.json")
        self._audit_forbidden_values(manifest, label="checkpoint manifest")
        if manifest != self.spec.to_payload():
            raise FormalCheckpointError(
                "checkpoint intent, candidate frame, budget, or policy drifted"
            )
        # First pass is strictly read-only: no pending-attempt conversion, HEAD
        # advancement, slot repair, or orphan cleanup is allowed until every
        # persisted component and candidate binding has validated together.
        attempts = self._validated_attempt_rows()
        slots = self._validated_slot_rows(attempt_rows=attempts)
        self._validated_candidate_commits(slot_rows=slots)

        self._clean_atomic_temp_orphans()
        attempts = self._validated_attempt_rows(
            reconcile_head=True, reconcile_pending=True
        )
        slots = self._validated_slot_rows(
            attempt_rows=attempts, repair_missing=True
        )
        commits = self._validated_candidate_commits(slot_rows=slots)
        self._install_indexes(attempts=attempts, slots=slots, commits=commits)

    def _install_indexes(
        self,
        *,
        attempts: Sequence[Mapping[str, Any]],
        slots: Sequence[Mapping[str, Any]],
        commits: Sequence[Mapping[str, Any]],
    ) -> None:
        self._attempt_rows = [dict(row) for row in attempts]
        self._attempts_by_sequence = {
            int(row["sequence"]): row for row in self._attempt_rows
        }
        self._attempts_by_slot = {}
        for row in self._attempt_rows:
            self._attempts_by_slot.setdefault(
                (str(row["provider"]), int(row["rank"]), str(row["slot"])), []
            ).append(row)
        self._provider_counts = {
            provider: sum(
                1 for row in self._attempt_rows if row["provider"] == provider
            )
            for provider in PROVIDER_SLOTS
        }
        self._ambiguous_attempt_count = sum(
            1 for row in self._attempt_rows if row["status"] == "ambiguous"
        )
        self._slot_index = {
            (str(row["provider"]), int(row["rank"]), str(row["slot"])): dict(row)
            for row in slots
        }
        self._candidate_commits = [dict(row) for row in commits]

    def _clean_atomic_temp_orphans(self) -> None:
        _checkpoint_files(
            self.root / "attempts",
            final_name_re=re.compile(r"(?:HEAD\.json|[0-9]{9}\.json)\Z"),
            label="attempt",
            clean_orphans=True,
        )
        _checkpoint_files(
            self.root / "slots",
            final_name_re=re.compile(
                r"(?:tavily|deepseek)-[0-9]{6,}-[1-9][0-9]*\.json\Z"
            ),
            label="slot",
            clean_orphans=True,
        )
        _checkpoint_files(
            self.root / "candidates",
            final_name_re=re.compile(r"[0-9]{6,}\.json\Z"),
            label="candidate",
            clean_orphans=True,
        )

    def _attempt_paths(self, *, clean_orphans: bool = False) -> list[Path]:
        directory = self.root / "attempts"
        paths = _checkpoint_files(
            directory,
            final_name_re=re.compile(r"(?:HEAD\.json|[0-9]{9}\.json)\Z"),
            label="attempt",
            clean_orphans=clean_orphans,
        )
        head_paths = [path for path in paths if path.name == "HEAD.json"]
        if len(head_paths) != 1:
            raise FormalCheckpointError("checkpoint attempt HEAD is missing or duplicated")
        return [path for path in paths if path.name != "HEAD.json"]

    @staticmethod
    def _attempt_head_payload(
        spec: CheckpointSpec,
        *,
        sequence: int,
        reservation_head_sha256: str,
        provider_counts: Mapping[str, int],
    ) -> dict[str, Any]:
        payload = {
            "schema_version": ATTEMPT_HEAD_SCHEMA_VERSION,
            "checkpoint_id": spec.checkpoint_id,
            "sequence": sequence,
            "reservation_head_sha256": reservation_head_sha256,
            "provider_counts": {
                provider: provider_counts[provider]
                for provider in sorted(PROVIDER_SLOTS)
            },
        }
        payload["head_sha256"] = _sha256(payload)
        return payload

    def _write_attempt_head(
        self,
        *,
        sequence: int,
        reservation_head_sha256: str,
        provider_counts: Mapping[str, int],
    ) -> None:
        _atomic_write_json(
            self.root / "attempts" / "HEAD.json",
            self._attempt_head_payload(
                self.spec,
                sequence=sequence,
                reservation_head_sha256=reservation_head_sha256,
                provider_counts=provider_counts,
            ),
        )

    def _read_attempt_head(self) -> dict[str, Any]:
        row = _read_json(self.root / "attempts" / "HEAD.json")
        expected_fields = {
            "schema_version",
            "checkpoint_id",
            "sequence",
            "reservation_head_sha256",
            "provider_counts",
            "head_sha256",
        }
        unhashed = {key: value for key, value in row.items() if key != "head_sha256"}
        sequence = row.get("sequence")
        counts = row.get("provider_counts")
        if (
            set(row) != expected_fields
            or row.get("schema_version") != ATTEMPT_HEAD_SCHEMA_VERSION
            or row.get("checkpoint_id") != self.spec.checkpoint_id
            or isinstance(sequence, bool)
            or not isinstance(sequence, int)
            or sequence < 0
            or not isinstance(row.get("reservation_head_sha256"), str)
            or _HASH_RE.fullmatch(str(row.get("reservation_head_sha256"))) is None
            or not isinstance(counts, Mapping)
            or set(counts) != set(PROVIDER_SLOTS)
            or any(
                isinstance(counts.get(provider), bool)
                or not isinstance(counts.get(provider), int)
                or counts.get(provider) < 0
                or counts.get(provider) > self.spec.provider_caps[provider]["cap"]
                for provider in PROVIDER_SLOTS
            )
            or row.get("head_sha256") != _sha256(unhashed)
        ):
            raise FormalCheckpointError("provider attempt HEAD is invalid")
        return row

    def _validated_attempt_rows(
        self,
        *,
        reconcile_pending: bool = False,
        reconcile_head: bool = False,
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        seen_reservations: set[str] = set()
        seen_slots: set[tuple[str, int, str, int]] = set()
        previous_reservation_sha256 = _GENESIS_RESERVATION_SHA256
        provider_counts = {provider: 0 for provider in PROVIDER_SLOTS}
        for expected_sequence, path in enumerate(self._attempt_paths(), start=1):
            if path.name != f"{expected_sequence:09d}.json":
                raise FormalCheckpointError("provider attempt sequence is not contiguous")
            row = _read_json(path)
            self._audit_forbidden_values(row, label="durable provider attempt")
            if row.get("schema_version") != ATTEMPT_SCHEMA_VERSION:
                raise FormalCheckpointError("unsupported provider-attempt checkpoint schema")
            provider = row.get("provider")
            rank = row.get("rank")
            slot = row.get("slot")
            attempt = row.get("attempt")
            request_sha256 = row.get("request_sha256")
            request_payload = row.get("request_payload")
            if (
                row.get("checkpoint_id") != self.spec.checkpoint_id
                or provider not in PROVIDER_SLOTS
                or slot not in PROVIDER_SLOTS[str(provider)]
                or slot not in self.spec.active_provider_slots[str(provider)]
                or isinstance(rank, bool)
                or not isinstance(rank, int)
                or not 1 <= rank <= len(self.spec.candidate_frame)
                or isinstance(attempt, bool)
                or not isinstance(attempt, int)
                or not 1 <= attempt <= self.spec.max_slot_attempts
                or not isinstance(request_sha256, str)
                or not _HASH_RE.fullmatch(request_sha256)
                or request_sha256 != _sha256(request_payload)
                or row.get("sequence") != expected_sequence
            ):
                raise FormalCheckpointError("provider attempt coordinates are invalid")
            budget = self.spec.provider_caps[str(provider)]
            if row.get("budget_scope_id") != budget["scope_id"]:
                raise FormalCheckpointError("provider attempt budget scope drifted")
            reservation_id = row.get("reservation_id")
            provider_counts[str(provider)] += 1
            expected_counts = {
                name: provider_counts[name] for name in sorted(PROVIDER_SLOTS)
            }
            reservation_payload = {
                "checkpoint_id": self.spec.checkpoint_id,
                "sequence": expected_sequence,
                "previous_reservation_sha256": previous_reservation_sha256,
                "provider": provider,
                "budget_scope_id": budget["scope_id"],
                "provider_counts_after": expected_counts,
                "rank": rank,
                "slot": slot,
                "attempt": attempt,
                "request_sha256": request_sha256,
            }
            expected_reservation_sha256 = _sha256(reservation_payload)
            if (
                row.get("previous_reservation_sha256")
                != previous_reservation_sha256
                or row.get("provider_counts_after") != expected_counts
                or row.get("reservation_sha256") != expected_reservation_sha256
            ):
                raise FormalCheckpointError("provider reservation hash chain is invalid")
            unhashed = {
                key: value
                for key, value in row.items()
                if key not in {"reservation_id", "outcome_sha256"}
            }
            expected_reservation_id = "pat-" + expected_reservation_sha256
            if reservation_id != expected_reservation_id:
                raise FormalCheckpointError("provider attempt reservation identity is invalid")
            status = row.get("status")
            if status not in {
                "reserved",
                "ambiguous",
                "retryable_failure",
                "terminal_failure",
                "success",
            }:
                raise FormalCheckpointError("provider attempt has an invalid status")
            if status != "reserved":
                declared_outcome = row.get("outcome_sha256")
                if declared_outcome != _sha256(unhashed):
                    raise FormalCheckpointError("provider attempt outcome hash is invalid")
            key = (str(provider), rank, str(slot), attempt)
            if reservation_id in seen_reservations or key in seen_slots:
                raise FormalCheckpointError("provider attempt is duplicated")
            seen_reservations.add(str(reservation_id))
            seen_slots.add(key)
            rows.append(row)
            previous_reservation_sha256 = expected_reservation_sha256
        head = self._read_attempt_head()
        expected_head = self._attempt_head_payload(
            self.spec,
            sequence=len(rows),
            reservation_head_sha256=previous_reservation_sha256,
            provider_counts=provider_counts,
        )
        head_sequence = int(head["sequence"])
        if head_sequence > len(rows):
            raise FormalCheckpointError(
                "provider attempt history was rolled back behind its durable HEAD"
            )
        if head_sequence < len(rows):
            # The attempt file is persisted before HEAD.  At most the one
            # in-flight reservation can be ahead if SIGKILL lands between the
            # two fsync-backed writes.  Advancing HEAD counts it conservatively.
            if head_sequence != len(rows) - 1:
                raise FormalCheckpointError(
                    "provider attempt HEAD is too far behind its reservation chain"
                )
            prefix_counts = {provider: 0 for provider in PROVIDER_SLOTS}
            for prefix_row in rows[:head_sequence]:
                prefix_counts[prefix_row["provider"]] += 1
            prefix_hash = (
                rows[head_sequence - 1]["reservation_sha256"]
                if head_sequence
                else _GENESIS_RESERVATION_SHA256
            )
            expected_old_head = self._attempt_head_payload(
                self.spec,
                sequence=head_sequence,
                reservation_head_sha256=prefix_hash,
                provider_counts=prefix_counts,
            )
            if head != expected_old_head:
                raise FormalCheckpointError("provider attempt HEAD does not match its prefix")
            if reconcile_head:
                self._write_attempt_head(
                    sequence=len(rows),
                    reservation_head_sha256=previous_reservation_sha256,
                    provider_counts=provider_counts,
                )
                head = expected_head
        if head_sequence == len(rows) and head != expected_head:
            raise FormalCheckpointError(
                "provider attempt HEAD does not match its reservation chain"
            )
        for provider in PROVIDER_SLOTS:
            provider_rows = [row for row in rows if row["provider"] == provider]
            if len(provider_rows) > self.spec.provider_caps[provider]["cap"]:
                raise FormalCheckpointError("provider physical-attempt cap was exceeded")
        grouped: dict[tuple[str, int, str], list[dict[str, Any]]] = {}
        for row in rows:
            grouped.setdefault(
                (row["provider"], row["rank"], row["slot"]), []
            ).append(row)
        for slot_rows in grouped.values():
            attempts = [row["attempt"] for row in slot_rows]
            if attempts != list(range(1, len(slot_rows) + 1)):
                raise FormalCheckpointError("provider slot attempts are not contiguous")
            request_hashes = {row["request_sha256"] for row in slot_rows}
            if len(request_hashes) != 1:
                raise FormalCheckpointError("provider slot request payload drifted between attempts")
            terminal_indexes = [
                index
                for index, row in enumerate(slot_rows)
                if row["status"] in {"success", "terminal_failure"}
            ]
            if terminal_indexes and terminal_indexes != [len(slot_rows) - 1]:
                raise FormalCheckpointError("provider slot contains attempts after a terminal outcome")
        if reconcile_pending:
            for row in rows:
                if row["status"] != "reserved":
                    continue
                row.update(
                    {
                        "status": "ambiguous",
                        "response": None,
                        "capture": None,
                        "detail": {
                            "error_type": "ambiguous_process_interruption",
                            "retryable": True,
                        },
                    }
                )
                unhashed = {
                    key: value
                    for key, value in row.items()
                    if key not in {"reservation_id", "outcome_sha256"}
                }
                row["outcome_sha256"] = _sha256(unhashed)
                _atomic_write_json(
                    self.root / "attempts" / f"{row['sequence']:09d}.json", row
                )
        return rows

    def _slot_path(self, provider: str, rank: int, slot: str) -> Path:
        slot_index = PROVIDER_SLOTS[provider].index(slot) + 1
        return self.root / "slots" / f"{provider}-{rank:06d}-{slot_index}.json"

    def _validated_slot_rows(
        self,
        *,
        attempt_rows: Sequence[Mapping[str, Any]] | None = None,
        repair_missing: bool = False,
    ) -> list[dict[str, Any]]:
        directory = self.root / "slots"
        paths = _checkpoint_files(
            directory,
            final_name_re=re.compile(
                r"(?:tavily|deepseek)-[0-9]{6,}-[1-9][0-9]*\.json\Z"
            ),
            label="slot",
        )
        if attempt_rows is None:
            attempt_rows = self._validated_attempt_rows()
        successful_attempts = {
            (row["provider"], row["rank"], row["slot"]): row
            for row in attempt_rows
            if row["status"] == "success"
        }
        rows: list[dict[str, Any]] = []
        seen: set[tuple[str, int, str]] = set()
        for path in paths:
            row = _read_json(path)
            self._audit_forbidden_values(row, label="durable provider slot")
            if row.get("schema_version") != SLOT_SCHEMA_VERSION:
                raise FormalCheckpointError("unsupported provider-slot checkpoint schema")
            provider = row.get("provider")
            rank = row.get("rank")
            slot = row.get("slot")
            if (
                row.get("checkpoint_id") != self.spec.checkpoint_id
                or not isinstance(provider, str)
                or not isinstance(slot, str)
            ):
                raise FormalCheckpointError("provider success slot coordinates are invalid")
            self._validate_coordinates(provider, rank, slot)
            key = (provider, rank, slot)
            attempt = successful_attempts.get(key)
            if attempt is None:
                raise FormalCheckpointError("provider success slot lacks its attempt outcome")
            expected = self._slot_payload(attempt)
            if row != expected or path.name != self._slot_path(*key).name:
                raise FormalCheckpointError("provider success slot is corrupt")
            if key in seen:
                raise FormalCheckpointError("provider success slot is duplicated")
            seen.add(key)
            rows.append(row)
        for key, attempt in successful_attempts.items():
            if key in seen:
                continue
            expected = self._slot_payload(attempt)
            # Missing indexes are recoverable because the full response and
            # its outcome hash already live in the durable attempt record.
            if repair_missing:
                self._write_slot_success(self._slot_path(*key), attempt)
            rows.append(expected)
        rows.sort(
            key=lambda row: (
                row["rank"],
                sorted(PROVIDER_SLOTS).index(row["provider"]),
                PROVIDER_SLOTS[row["provider"]].index(row["slot"]),
            )
        )
        return rows

    def _slot_payload(self, attempt: Mapping[str, Any]) -> dict[str, Any]:
        payload = {
            "schema_version": SLOT_SCHEMA_VERSION,
            "checkpoint_id": self.spec.checkpoint_id,
            "provider": attempt["provider"],
            "rank": attempt["rank"],
            "slot": attempt["slot"],
            "attempt": attempt["attempt"],
            "request_sha256": attempt["request_sha256"],
            "response": attempt.get("response"),
            "capture": attempt.get("capture"),
            "attempt_outcome_sha256": attempt.get("outcome_sha256"),
        }
        payload["slot_sha256"] = _sha256(payload)
        return payload

    def _write_slot_success(self, path: Path, attempt: Mapping[str, Any]) -> None:
        _atomic_write_json(path, self._slot_payload(attempt))

    def get_slot_success(
        self,
        provider: str,
        rank: int,
        slot: str,
        request_payload: Any | None = None,
    ) -> SlotSuccess | None:
        with self._mutex:
            self._require_open()
            self._validate_coordinates(provider, rank, slot)
            row = self._slot_index.get((provider, rank, slot))
            if row is None:
                return None
            # One bounded direct read closes the cached-index TOCTOU gap without
            # rescanning thousands of unrelated attempts.
            if _read_json(self._slot_path(provider, rank, slot)) != row:
                raise FormalCheckpointError("provider success slot is corrupt")
            expected_request_sha256 = (
                _sha256(request_payload) if request_payload is not None else None
            )
            if (
                expected_request_sha256 is not None
                and row.get("request_sha256") != expected_request_sha256
            ):
                raise FormalCheckpointError(
                    "completed provider slot request payload drifted"
                )
            return SlotSuccess(
                provider=provider,
                rank=rank,
                slot=slot,
                attempt=int(row["attempt"]),
                request_sha256=str(row["request_sha256"]),
                response=_plain_copy(row.get("response")),
                capture=_plain_copy(row.get("capture")),
            )

    def _validate_coordinates(self, provider: str, rank: int, slot: str) -> None:
        if provider not in PROVIDER_SLOTS or slot not in PROVIDER_SLOTS[provider]:
            raise FormalCheckpointError("unknown provider slot")
        if slot not in self.spec.active_provider_slots[provider]:
            raise FormalCheckpointError("provider slot is inactive for this checkpoint")
        if isinstance(rank, bool) or not isinstance(rank, int) or not 1 <= rank <= len(
            self.spec.candidate_frame
        ):
            raise FormalCheckpointError("provider slot rank is outside the candidate frame")

    def reserve_attempt(
        self,
        provider: str,
        rank: int,
        slot: str,
        request_payload: Any,
    ) -> AttemptReservation:
        with self._mutex:
            self._require_open()
            self._validate_coordinates(provider, rank, slot)
            _reject_secret_fields(request_payload, label="provider request payload")
            self._audit_forbidden_values(
                request_payload, label="provider request payload"
            )
            request_value = _plain_copy(request_payload)
            request_sha256 = _sha256(request_value)
            key = (provider, rank, slot)
            slot_rows = self._attempts_by_slot.get(key, [])
            if any(row["request_sha256"] != request_sha256 for row in slot_rows):
                raise FormalCheckpointError(
                    "provider request payload drifted for an existing slot"
                )
            if slot_rows and slot_rows[-1]["status"] == "success":
                raise FormalCheckpointError("provider slot is already complete")
            if slot_rows and slot_rows[-1]["status"] == "terminal_failure":
                raise FormalCheckpointError("provider slot ended in a terminal failure")
            if slot_rows and slot_rows[-1]["status"] == "reserved":
                raise FormalCheckpointError("provider slot has an unresolved reservation")
            attempt = len(slot_rows) + 1
            if attempt > self.spec.max_slot_attempts:
                raise FormalCheckpointError("provider slot attempt budget is exhausted")
            if self._provider_counts[provider] >= self.spec.provider_caps[provider]["cap"]:
                raise FormalCheckpointError(
                    f"{provider} global physical-attempt budget is exhausted"
                )
            sequence = len(self._attempt_rows) + 1
            previous_reservation_sha256 = (
                self._attempt_rows[-1]["reservation_sha256"]
                if self._attempt_rows
                else _GENESIS_RESERVATION_SHA256
            )
            provider_counts_after = dict(self._provider_counts)
            provider_counts_after[provider] += 1
            provider_counts_after = {
                name: provider_counts_after[name] for name in sorted(PROVIDER_SLOTS)
            }
            reservation_payload = {
                "checkpoint_id": self.spec.checkpoint_id,
                "sequence": sequence,
                "previous_reservation_sha256": previous_reservation_sha256,
                "provider": provider,
                "budget_scope_id": self.spec.provider_caps[provider]["scope_id"],
                "provider_counts_after": provider_counts_after,
                "rank": rank,
                "slot": slot,
                "attempt": attempt,
                "request_sha256": request_sha256,
            }
            reservation_sha256 = _sha256(reservation_payload)
            reservation_id = "pat-" + reservation_sha256
            row = {
                "schema_version": ATTEMPT_SCHEMA_VERSION,
                "checkpoint_id": self.spec.checkpoint_id,
                "sequence": sequence,
                "reservation_id": reservation_id,
                "previous_reservation_sha256": previous_reservation_sha256,
                "reservation_sha256": reservation_sha256,
                "provider": provider,
                "budget_scope_id": self.spec.provider_caps[provider]["scope_id"],
                "provider_counts_after": provider_counts_after,
                "rank": rank,
                "slot": slot,
                "attempt": attempt,
                "request_sha256": request_sha256,
                "request_payload": request_value,
                "request_dispatched": True,
                "status": "reserved",
                "response": None,
                "capture": None,
                "detail": None,
            }
            try:
                # Both fsync-backed writes complete before this reservation is
                # returned to the provider caller for physical dispatch.
                _atomic_write_json(
                    self.root / "attempts" / f"{sequence:09d}.json", row
                )
                self._write_attempt_head(
                    sequence=sequence,
                    reservation_head_sha256=reservation_sha256,
                    provider_counts=provider_counts_after,
                )
            except Exception:
                self._poisoned = True
                raise
            self._attempt_rows.append(row)
            self._attempts_by_sequence[sequence] = row
            self._attempts_by_slot.setdefault(key, []).append(row)
            self._provider_counts[provider] += 1
            return AttemptReservation(
                reservation_id=reservation_id,
                provider=provider,
                rank=rank,
                slot=slot,
                attempt=attempt,
                request_sha256=request_sha256,
                sequence=sequence,
            )

    def finish_attempt(
        self,
        reservation: AttemptReservation,
        *,
        status: str,
        response: Any = None,
        capture: Any = None,
        detail: Any = None,
    ) -> AttemptOutcome:
        with self._mutex:
            self._require_open()
            if status not in {"success", "retryable_failure", "terminal_failure"}:
                raise FormalCheckpointError("attempt outcome status is invalid")
            path = self.root / "attempts" / f"{reservation.sequence:09d}.json"
            cached = self._attempts_by_sequence.get(reservation.sequence)
            expected = {
                "reservation_id": reservation.reservation_id,
                "provider": reservation.provider,
                "rank": reservation.rank,
                "slot": reservation.slot,
                "attempt": reservation.attempt,
                "request_sha256": reservation.request_sha256,
                "sequence": reservation.sequence,
            }
            if cached is None or any(
                cached.get(key) != value for key, value in expected.items()
            ):
                raise FormalCheckpointError(
                    "attempt reservation does not match durable state"
                )
            if _read_json(path) != cached:
                raise FormalCheckpointError("durable attempt changed after validation")
            if cached.get("status") != "reserved":
                if cached.get("status") == status:
                    return AttemptOutcome(reservation=reservation, status=status)
                raise FormalCheckpointError(
                    "attempt reservation already has a different outcome"
                )
            self._audit_forbidden_values(
                {"response": response, "capture": capture, "detail": detail},
                label="provider attempt outcome",
            )
            _reject_secret_fields(
                {"response": response, "capture": capture, "detail": detail},
                label="provider attempt outcome",
            )
            row = dict(cached)
            row.update(
                {
                    "status": status,
                    "response": _plain_copy(response),
                    "capture": _plain_copy(capture),
                    "detail": _plain_copy(detail),
                }
            )
            unhashed = {
                key: value
                for key, value in row.items()
                if key not in {"reservation_id", "outcome_sha256"}
            }
            row["outcome_sha256"] = _sha256(unhashed)
            try:
                _atomic_write_json(path, row)
            except Exception:
                self._poisoned = True
                raise
            cached.clear()
            cached.update(row)
            if status == "success":
                slot_path = self._slot_path(
                    reservation.provider, reservation.rank, reservation.slot
                )
                slot_key = (
                    reservation.provider,
                    reservation.rank,
                    reservation.slot,
                )
                expected_slot = self._slot_payload(row)
                try:
                    existing_slot = self._slot_index.get(slot_key)
                    if existing_slot is not None:
                        if (
                            _read_json(slot_path) != existing_slot
                            or existing_slot != expected_slot
                        ):
                            raise FormalCheckpointError(
                                "provider slot success conflicts with existing state"
                            )
                    elif slot_path.exists() or slot_path.is_symlink():
                        raise FormalCheckpointError(
                            "unexpected provider slot appeared during attempt"
                        )
                    else:
                        self._write_slot_success(slot_path, row)
                        self._slot_index[slot_key] = expected_slot
                except Exception:
                    self._poisoned = True
                    raise
            return AttemptOutcome(reservation=reservation, status=status)

    def can_retry(self, provider: str, rank: int, slot: str) -> bool:
        with self._mutex:
            self._require_open()
            self._validate_coordinates(provider, rank, slot)
            slot_rows = self._attempts_by_slot.get((provider, rank, slot), [])
            if slot_rows and slot_rows[-1]["status"] in {
                "success",
                "terminal_failure",
                "reserved",
            }:
                return False
            return (
                len(slot_rows) < self.spec.max_slot_attempts
                and self._provider_counts[provider]
                < self.spec.provider_caps[provider]["cap"]
            )

    @property
    def committed_prefix(self) -> int:
        with self._mutex:
            self._require_open()
            return len(self._candidate_commits)

    def _candidate_commit_path(self, rank: int) -> Path:
        return self.root / "candidates" / f"{rank:06d}.json"

    def _slot_bindings_for_rank(
        self,
        rank: int,
        *,
        slot_rows: Sequence[Mapping[str, Any]] | None = None,
    ) -> dict[str, dict[str, str]]:
        if slot_rows is None:
            indexed = self._slot_index
        else:
            indexed = {
                (row["provider"], row["rank"], row["slot"]): row
                for row in slot_rows
            }
        bindings: dict[str, dict[str, str]] = {}
        for provider in sorted(PROVIDER_SLOTS):
            provider_bindings: dict[str, str] = {}
            for slot in self.spec.active_provider_slots[provider]:
                row = indexed.get((provider, rank, slot))
                if row is None:
                    raise FormalCheckpointError(
                        "candidate commit requires every active provider slot to succeed"
                    )
                slot_sha256 = row.get("slot_sha256")
                if not isinstance(slot_sha256, str) or _HASH_RE.fullmatch(slot_sha256) is None:
                    raise FormalCheckpointError("candidate provider slot hash is invalid")
                provider_bindings[slot] = slot_sha256
            bindings[provider] = provider_bindings
        return bindings

    def _validated_candidate_commits(
        self,
        *,
        slot_rows: Sequence[Mapping[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        directory = self.root / "candidates"
        paths = _checkpoint_files(
            directory,
            final_name_re=re.compile(r"[0-9]{6,}\.json\Z"),
            label="candidate",
        )
        if slot_rows is None:
            slot_rows = self._validated_slot_rows()
        commits: list[dict[str, Any]] = []
        previous = "0" * 64
        for expected_rank, path in enumerate(paths, start=1):
            if path.is_symlink() or not path.is_file() or path.name != f"{expected_rank:06d}.json":
                raise FormalCheckpointError("candidate commits are not a contiguous prefix")
            row = _read_json(path)
            self._audit_forbidden_values(row, label="durable candidate commit")
            if (
                row.get("schema_version") != CANDIDATE_COMMIT_SCHEMA_VERSION
                or row.get("checkpoint_id") != self.spec.checkpoint_id
                or row.get("rank") != expected_rank
                or row.get("previous_commit_sha256") != previous
            ):
                raise FormalCheckpointError("candidate commit lineage is invalid")
            bundle = row.get("row_bundle")
            if not isinstance(bundle, Mapping) or bundle.get("candidate") != self.spec.candidate_frame[
                expected_rank - 1
            ]:
                raise FormalCheckpointError("candidate commit does not match the frozen frame")
            expected_slot_bindings = self._slot_bindings_for_rank(
                expected_rank, slot_rows=slot_rows
            )
            if row.get("provider_slot_sha256s") != expected_slot_bindings:
                raise FormalCheckpointError(
                    "candidate commit is not bound to its successful provider slots"
                )
            unhashed = {key: value for key, value in row.items() if key != "commit_sha256"}
            expected_hash = _sha256(unhashed)
            if row.get("commit_sha256") != expected_hash:
                raise FormalCheckpointError("candidate commit hash is invalid")
            previous = expected_hash
            commits.append(row)
        return commits

    def commit_candidate(
        self, rank: int, row_bundle: Mapping[str, Any]
    ) -> CandidateCommit:
        with self._mutex:
            self._require_open()
            commits = self._candidate_commits
            if (
                isinstance(rank, bool)
                or not isinstance(rank, int)
                or not 1 <= rank <= len(self.spec.candidate_frame)
            ):
                raise FormalCheckpointError("candidate commit rank is invalid")
            bundle = _plain_copy(dict(row_bundle))
            _reject_secret_fields(bundle, label="candidate checkpoint commit")
            self._audit_forbidden_values(bundle, label="candidate checkpoint commit")
            if bundle.get("candidate") != self.spec.candidate_frame[rank - 1]:
                raise FormalCheckpointError(
                    "candidate commit row does not match its frozen frame"
                )
            if rank <= len(commits):
                existing = commits[rank - 1]
                if _read_json(self._candidate_commit_path(rank)) != existing:
                    raise FormalCheckpointError(
                        "durable candidate commit changed after validation"
                    )
                if existing.get("row_bundle") != bundle:
                    raise FormalCheckpointError(
                        "candidate commit conflicts with durable state"
                    )
                return CandidateCommit(
                    rank=rank,
                    commit_sha256=str(existing["commit_sha256"]),
                    previous_commit_sha256=str(existing["previous_commit_sha256"]),
                    row_bundle=MappingProxyType(_plain_copy(bundle)),
                )
            if rank != len(commits) + 1:
                raise FormalCheckpointError(
                    "candidate commits must form a contiguous prefix"
                )
            provider_slot_sha256s = self._slot_bindings_for_rank(rank)
            previous = commits[-1]["commit_sha256"] if commits else "0" * 64
            row = {
                "schema_version": CANDIDATE_COMMIT_SCHEMA_VERSION,
                "checkpoint_id": self.spec.checkpoint_id,
                "rank": rank,
                "previous_commit_sha256": previous,
                "provider_slot_sha256s": provider_slot_sha256s,
                "row_bundle": bundle,
            }
            row["commit_sha256"] = _sha256(row)
            try:
                _atomic_write_json(self._candidate_commit_path(rank), row)
            except Exception:
                self._poisoned = True
                raise
            self._candidate_commits.append(row)
            return CandidateCommit(
                rank=rank,
                commit_sha256=str(row["commit_sha256"]),
                previous_commit_sha256=str(previous),
                row_bundle=MappingProxyType(_plain_copy(bundle)),
            )

    def completed_candidate_rows(self) -> tuple[Mapping[str, Any], ...]:
        with self._mutex:
            self._require_open()
            return tuple(
                MappingProxyType(_plain_copy(row["row_bundle"]))
                for row in self._candidate_commits
            )

    def _logical_search_captures(
        self, commits: Sequence[Mapping[str, Any]]
    ) -> list[dict[str, Any]]:
        slot_rows = {
            (row["rank"], row["slot"]): row
            for row in self._slot_index.values()
            if row["provider"] == "tavily"
        }
        captures: list[dict[str, Any]] = []
        for rank, commit in enumerate(commits, start=1):
            web_row = commit["row_bundle"].get("web_evidence")
            if not isinstance(web_row, Mapping):
                raise FormalCheckpointError("candidate commit lacks Web evidence")
            queries = web_row.get("queries")
            evidence = web_row.get("evidence")
            logical_slots = self.spec.active_provider_slots["tavily"]
            if not logical_slots and isinstance(queries, list) and len(queries) == 3:
                # Disabled Web still emits the three deterministic local query
                # coverage rows, but consumes no Tavily provider slot.
                logical_slots = PROVIDER_SLOTS["tavily"]
            expected_query_count = len(logical_slots)
            if (
                not isinstance(queries, list)
                or len(queries) != expected_query_count
                or not isinstance(evidence, list)
            ):
                raise FormalCheckpointError("candidate Web evidence is malformed")
            for slot, query in zip(
                logical_slots, queries
            ):
                success = slot_rows.get((rank, slot))
                capture = success.get("capture") if success is not None else None
                if isinstance(capture, Mapping):
                    row = {
                        "term": capture.get("term"),
                        "query": capture.get("query"),
                        "results": capture.get("results"),
                        "error": capture.get("error"),
                    }
                else:
                    row = {
                        "term": web_row.get("term"),
                        "query": query,
                        "results": [
                            item
                            for item in evidence
                            if isinstance(item, Mapping) and item.get("query") == query
                        ],
                        "error": None,
                    }
                captures.append(row)
        return captures

    def _llm_captures(self) -> list[dict[str, Any]]:
        rows = [
            row for row in self._attempt_rows if row["provider"] == "deepseek"
        ]
        rows.sort(
            key=lambda row: (
                row["rank"],
                PROVIDER_SLOTS["deepseek"].index(row["slot"]),
                row["attempt"],
            )
        )
        captures: list[dict[str, Any]] = []
        for row in rows:
            capture = row.get("capture")
            if isinstance(capture, Mapping):
                captures.append(
                    {
                        "stage": capture.get("stage", row["slot"]),
                        "term": capture.get("term"),
                        "attempt": row["attempt"],
                        "request_payload": capture.get(
                            "request_payload", row.get("request_payload")
                        ),
                        "raw_response": capture.get("raw_response"),
                        "parsed_response": capture.get("parsed_response"),
                        "error": capture.get("error"),
                    }
                )
                continue
            error_type = (
                row.get("detail", {}).get("error_type")
                if isinstance(row.get("detail"), Mapping)
                else row["status"]
            )
            captures.append(
                {
                    "stage": row["slot"],
                    "term": self.spec.candidate_frame[row["rank"] - 1].get("term"),
                    "attempt": row["attempt"],
                    "request_payload": row.get("request_payload"),
                    "raw_response": {"error": {"type": error_type}},
                    "parsed_response": None,
                    "error": str(error_type),
                }
            )
        return captures

    def _safe_tavily_attempts(self) -> list[dict[str, Any]]:
        rows = [
            row for row in self._attempt_rows if row["provider"] == "tavily"
        ]
        rows.sort(
            key=lambda row: (
                row["rank"],
                PROVIDER_SLOTS["tavily"].index(row["slot"]),
                row["attempt"],
            )
        )
        safe: list[dict[str, Any]] = []
        for row in rows:
            status = row["status"]
            detail = row.get("detail") if isinstance(row.get("detail"), Mapping) else {}
            capture = row.get("capture") if isinstance(row.get("capture"), Mapping) else {}
            results = capture.get("results")
            if not isinstance(results, list):
                response = row.get("response")
                results = response.get("results") if isinstance(response, Mapping) else None
            safe.append(
                {
                    "provider": "tavily",
                    "budget_scope_id": row["budget_scope_id"],
                    "rank": row["rank"],
                    "slot": PROVIDER_SLOTS["tavily"].index(row["slot"]) + 1,
                    "attempt": row["attempt"],
                    "status": (
                        "success"
                        if status == "success"
                        else "ambiguous"
                        if status == "ambiguous"
                        else "failure"
                    ),
                    "http_status": detail.get("http_status"),
                    "error_type": (
                        None if status == "success" else detail.get("error_type", status)
                    ),
                    "retryable": status in {"retryable_failure", "ambiguous"},
                    "result_count": len(results) if status == "success" and isinstance(results, list) else None,
                    "request_dispatched": True,
                }
            )
        return safe

    def materialize(self, output_dir: str | Path) -> dict[str, Any]:
        with self._mutex:
            self._require_open()
            attempts = self._validated_attempt_rows()
            slots = self._validated_slot_rows(attempt_rows=attempts)
            commits_on_disk = self._validated_candidate_commits(slot_rows=slots)
            if (
                attempts != self._attempt_rows
                or {
                    (row["provider"], row["rank"], row["slot"]): row
                    for row in slots
                }
                != self._slot_index
                or commits_on_disk != self._candidate_commits
            ):
                raise FormalCheckpointError(
                    "durable checkpoint changed after its in-memory index was built"
                )
            commits = self._candidate_commits
            if len(commits) != len(self.spec.candidate_frame):
                raise FormalCheckpointError(
                    "cannot materialize an incomplete candidate frame"
                )
            target = Path(output_dir)
            _ensure_private_directory(target, create=True)
            bundles = [row["row_bundle"] for row in commits]
            _atomic_write_jsonl(target / "candidates.jsonl", self.spec.candidate_frame)
            _atomic_write_jsonl(
                target / "web_evidence.jsonl",
                [bundle["web_evidence"] for bundle in bundles],
            )
            _atomic_write_jsonl(
                target / "llm_judgements.jsonl",
                [bundle["llm_judgement"] for bundle in bundles],
            )
            _atomic_write_jsonl(
                target / "rejected.jsonl",
                [
                    bundle["rejected"]
                    for bundle in bundles
                    if bundle.get("rejected") is not None
                ],
            )
            debug_dir = target / "debug"
            _ensure_private_directory(debug_dir, create=True)
            _atomic_write_jsonl(debug_dir / "llm_calls.jsonl", self._llm_captures())
            _atomic_write_jsonl(
                debug_dir / "search_calls.jsonl",
                self._logical_search_captures(commits),
            )
            _atomic_write_jsonl(
                debug_dir / "tavily_attempts.jsonl", self._safe_tavily_attempts()
            )
            return self.summary()

    def summary(self) -> dict[str, Any]:
        with self._mutex:
            self._require_open()
            reservation_head_sha256 = (
                self._attempt_rows[-1]["reservation_sha256"]
                if self._attempt_rows
                else _GENESIS_RESERVATION_SHA256
            )
            expected_head = self._attempt_head_payload(
                self.spec,
                sequence=len(self._attempt_rows),
                reservation_head_sha256=reservation_head_sha256,
                provider_counts=self._provider_counts,
            )
            if _read_json(self.root / "attempts" / "HEAD.json") != expected_head:
                raise FormalCheckpointError(
                    "provider attempt HEAD changed after validation"
                )
            final_commit_sha256 = (
                self._candidate_commits[-1]["commit_sha256"]
                if self._candidate_commits
                else "0" * 64
            )
            if self._attempt_rows and _read_json(
                self.root
                / "attempts"
                / f"{len(self._attempt_rows):09d}.json"
            ) != self._attempt_rows[-1]:
                raise FormalCheckpointError(
                    "provider attempt tail changed after validation"
                )
            if self._candidate_commits and _read_json(
                self._candidate_commit_path(len(self._candidate_commits))
            ) != self._candidate_commits[-1]:
                raise FormalCheckpointError(
                    "candidate commit head changed after validation"
                )
            return {
                "schema_version": "stage1-formal-lexicon-checkpoint-summary/v2",
                "checkpoint_id": self.spec.checkpoint_id,
                "candidate_count": len(self.spec.candidate_frame),
                "committed_prefix": len(self._candidate_commits),
                "provider_attempt_counts": {
                    provider: self._provider_counts[provider]
                    for provider in sorted(PROVIDER_SLOTS)
                },
                "provider_attempt_caps": {
                    provider: self.spec.provider_caps[provider]["cap"]
                    for provider in sorted(PROVIDER_SLOTS)
                },
                "provider_attempt_scopes": {
                    provider: self.spec.provider_caps[provider]["scope_id"]
                    for provider in sorted(PROVIDER_SLOTS)
                },
                "ambiguous_attempt_count": self._ambiguous_attempt_count,
                "attempt_head": {
                    "reservation_count": len(self._attempt_rows),
                    "provider_counts": {
                        provider: self._provider_counts[provider]
                        for provider in sorted(PROVIDER_SLOTS)
                    },
                    "reservation_head_sha256": reservation_head_sha256,
                    "head_sha256": expected_head["head_sha256"],
                },
                "candidate_commit_head": {
                    "committed_prefix": len(self._candidate_commits),
                    "commit_sha256": final_commit_sha256,
                },
            }


__all__ = [
    "AttemptOutcome",
    "AttemptReservation",
    "CandidateCommit",
    "CheckpointSpec",
    "FormalCheckpointError",
    "FormalLexiconCheckpoint",
    "PROVIDER_SLOTS",
    "SlotSuccess",
]
