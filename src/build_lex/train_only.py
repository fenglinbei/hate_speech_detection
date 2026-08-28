"""Immutable train-only Stage-1 terminology-library builder and validator.

The mining/judgement implementation is injected as a callable.  This wrapper
controls its input boundary, records every relevant hash, assigns stable term
evidence IDs, and publishes a category-free content-addressed target.  The
``lexicon`` artifact kind is retained only for pipeline compatibility.
"""

from __future__ import annotations

import copy
import hashlib
import inspect
import json
import os
import re
import shutil
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from rag.types import canonical_json, content_sha256, stable_term_evidence_id
from data.train_partition import (
    TrainPartitionError,
    load_train_partition,
    validate_train_partition_target,
)
from data.stage1_data import Stage1DataError, validate_data_target
from data.training_artifacts import (
    TrainingArtifactError,
    resolve_dependency_target,
    validate_dependency_ref,
)


LEXICON_SCHEMA_VERSION = "stage1-train-only-terminology-library/v1"
LEXICON_MANIFEST_VERSION = "stage1-terminology-library-manifest/v1"
LEXICON_PROVENANCE_VERSION = "stage1-terminology-library-provenance/v1"
LEXICON_BUILD_POLICY_VERSION = "train-only-terminology-library-build/v1"
PILOT_GATED_TERMINOLOGY_MANIFEST_VERSION = (
    "stage1-terminology-lifecycle-lexicon-manifest/v1"
)
PILOT_GATED_TERMINOLOGY_PUBLICATION_POLICY = (
    "pilot-gated-terminology-lifecycle/v1"
)
FORMAL_AUTHORIZATION_SCHEMA_VERSION = "stage1-formal-lexicon-authorization/v5"
TERMINOLOGY_LIBRARY_ROLE = "terminology-understanding-library/v1"
LOCATOR_REF_VERSION = "stage1-locator-ref/v1"
DEPENDENCY_REF_VERSION = "stage1-dependency-ref/v1"
PAYLOAD_MANIFEST_VERSION = "stage1-payload-manifest/v1"
ARTIFACT_KIND = "lexicon"
LEXICON_BUILD_ID_RE = re.compile(r"^lex-[0-9a-f]{64}$")
LEXICON_ENTRY_ID_RE = re.compile(r"^lex:v2:[0-9a-f]{64}$")
DATA_BUILD_ID_RE = re.compile(r"^data-[0-9a-f]{64}$")
AUTHORIZED_FORMAL_LLM_MODEL = "deepseek-v4-flash"
AUTHORIZED_PROVIDER_MODEL_RE = re.compile(
    r"^deepseek-v4-flash(?:-[0-9]{4}|-[0-9]{8})?\Z",
    re.IGNORECASE,
)
RAW_AUDIT_FILES = (
    "candidates.jsonl",
    "web_evidence.jsonl",
    "llm_judgements.jsonl",
    "rejected.jsonl",
)
FORMAL_PROTOCOL_SOURCE_FILES = (
    "formal_checkpoint.py",
    "train_only.py",
    "stage1_preflight.py",
    "web_search.py",
)
FORMAL_CAPTURE_FILES = {
    "llm_calls": "debug_llm_calls.jsonl",
    "search_calls": "debug_search_calls.jsonl",
    "tavily_attempts": "debug_tavily_attempts.jsonl",
}
FORMAL_LLM_STAGES = (
    "context_judge",
    "web_evidence_judge",
    "final_lexicon_judge",
)
CANONICAL_LEXICON_CATEGORIES = frozenset(
    {"Sexism", "Racism", "Region", "LGBTQ", "others"}
)
FORMAL_EXECUTION_POLICY = {
    "input": "preflight in-memory train snapshot",
    "resume": True,
    "checkpoint_policy": "provider-slot-checkpoint/v1",
    "ambiguous_attempt_policy": "count-and-retry-within-budget/v1",
    "single_writer": True,
    "atomic_fsync": True,
    "checkpoint_retained_after_publication": True,
    "debug_capture": True,
    "strict_provider_response_audit": True,
    "durable_provider_attempt_ledger": True,
    "injected_clients": False,
}
FORMAL_CHECKPOINT_SCOPE_ANCHOR_SCHEMA_VERSION = (
    "stage1-formal-lexicon-checkpoint-scope-anchor/v1"
)
FORMAL_CHECKPOINT_RUNTIME_LOGICAL_ROOT = (
    "exps/causal_context/stage1_p0/lexicons"
)
FORMAL_DEEPSEEK_ATTEMPT_SCOPE_ID = (
    "stage1-p0-wp3-formal-full-deepseek/v1"
)
_SECRET_KEYS = {
    "api_key",
    "apikey",
    "access_token",
    "auth_token",
    "token",
    "password",
    "secret",
    "authorization",
}
TERMINOLOGY_TASK_FIELD_KEYS = frozenset(
    {
        "annotation_count",
        "categories",
        "category",
        "category_counts",
        "category_purity",
        "hate_count",
        "hate_precision",
        "hateful",
        "label",
        "labels",
        "log_odds",
        "non_hate_count",
        "nonhate_penalty",
        "primary_category",
        "targeted_group",
    }
)
TERMINOLOGY_RESPONSE_KEYS = {
    "context_judge": frozenset({"supported", "confidence", "reason"}),
    "web_evidence_judge": frozenset(
        {"supported", "confidence", "reason", "evidence_ids"}
    ),
    "final_lexicon_judge": frozenset(
        {
            "include",
            "definition",
            "usage_notes",
            "ambiguity_notes",
            "variants",
            "confidence",
            "reason",
            "evidence_ids",
        }
    ),
}


class TrainOnlyLexiconError(ValueError):
    pass


@dataclass(frozen=True)
class FrozenTrainInput:
    records: tuple[Mapping[str, Any], ...]
    record_ids: tuple[str, ...]
    data_build_id: str
    train_data_sha256: str
    train_ids_sha256: str
    source_train_data_sha256: str
    source_train_ids_sha256: str
    source_mode: str
    train_path: Path | None
    data_ref: Mapping[str, Any]
    train_partition_ref: Mapping[str, Any] | None
    train_partition_dependency: Mapping[str, Any] | None
    forbidden_ids: frozenset[str]
    forbidden_hashes: frozenset[str]
    forbidden_contents: frozenset[str]


_FORMAL_AUTHORITY = object()


class _FormalBuildAuthorization:
    """Opaque, process-local capability minted only by the formal preflight."""

    __slots__ = (
        "_authority",
        "_binding",
        "_config",
        "_config_sha256",
        "_credential_values",
        "_data_ref_file_sha256",
        "_data_ref_path",
        "_train_partition_ref_file_sha256",
        "_train_partition_ref_path",
        "_workspace_root",
        "_frozen_train_input",
        "_records_sha256",
        "_used",
    )

    def __init__(
        self,
        authority: object,
        *,
        binding: Mapping[str, Any],
        config: Mapping[str, Any],
        config_sha256: str,
        credential_values: Mapping[str, str],
        data_ref_file_sha256: str,
        data_ref_path: Path,
        train_partition_ref_file_sha256: str,
        train_partition_ref_path: Path,
        workspace_root: Path,
        frozen_train_input: FrozenTrainInput,
        records_sha256: str,
    ) -> None:
        if authority is not _FORMAL_AUTHORITY:
            raise TrainOnlyLexiconError("formal build authorizations may only be minted by preflight")
        self._authority = authority
        self._binding = copy.deepcopy(dict(binding))
        self._config = copy.deepcopy(dict(config))
        self._config_sha256 = config_sha256
        raw_credential_values = tuple(
            value
            for value in credential_values.values()
            if isinstance(value, str) and value
        )
        self._credential_values = tuple(
            dict.fromkeys(
                (
                    *raw_credential_values,
                    *(
                        hashlib.sha256(value.encode("utf-8")).hexdigest()
                        for value in raw_credential_values
                    ),
                )
            )
        )
        self._data_ref_file_sha256 = data_ref_file_sha256
        self._data_ref_path = data_ref_path
        self._train_partition_ref_file_sha256 = (
            train_partition_ref_file_sha256
        )
        self._train_partition_ref_path = train_partition_ref_path
        self._workspace_root = workspace_root
        self._frozen_train_input = frozen_train_input
        self._records_sha256 = records_sha256
        self._used = False

    def __repr__(self) -> str:
        return "<formal Stage-1 lexicon build authorization>"


def _canonical_bytes(value: Any) -> bytes:
    return canonical_json(value).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _portable_ref_file_sha256(value: Mapping[str, Any]) -> str:
    """Hash the exact canonical JSON file wire written inside the target."""

    return _sha256_bytes(_canonical_bytes(value) + b"\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reject_duplicate_json_keys(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise TrainOnlyLexiconError(f"JSON object contains a duplicate key: {key}")
        value[key] = item
    return value


def _reject_nonfinite_json_number(value: str) -> Any:
    raise TrainOnlyLexiconError(f"JSON contains a non-finite number: {value}")


def _load_json(path: Path) -> Any:
    return json.loads(
        path.read_text(encoding="utf-8-sig"),
        object_pairs_hook=_reject_duplicate_json_keys,
        parse_constant=_reject_nonfinite_json_number,
    )


def _write_json(path: Path, value: Any) -> None:
    path.write_bytes(_canonical_bytes(value) + b"\n")


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _ensure_private_runtime_directory(path: Path) -> None:
    if path.is_symlink():
        raise TrainOnlyLexiconError(
            "formal checkpoint runtime directory must not be a symlink"
        )
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    if not path.is_dir() or path.is_symlink():
        raise TrainOnlyLexiconError(
            "formal checkpoint runtime directory is unsafe"
        )
    os.chmod(path, 0o700)


def _scope_anchor_hash_payload(value: Mapping[str, Any]) -> dict[str, Any]:
    return {key: copy.deepcopy(item) for key, item in value.items() if key != "anchor_sha256"}


def _finalize_scope_anchor(value: Mapping[str, Any]) -> dict[str, Any]:
    payload = _scope_anchor_hash_payload(value)
    payload["anchor_sha256"] = _sha256_bytes(_canonical_bytes(payload))
    return payload


def _read_scope_anchor(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise TrainOnlyLexiconError(
            "formal checkpoint scope anchor is missing or unsafe"
        )
    if path.stat().st_mode & 0o077:
        raise TrainOnlyLexiconError(
            "formal checkpoint scope anchor is not owner-only"
        )
    value = _load_json(path)
    if not isinstance(value, Mapping):
        raise TrainOnlyLexiconError(
            "formal checkpoint scope anchor is malformed"
        )
    document = dict(value)
    if document.get("anchor_sha256") != _sha256_bytes(
        _canonical_bytes(_scope_anchor_hash_payload(document))
    ):
        raise TrainOnlyLexiconError(
            "formal checkpoint scope anchor hash is invalid"
        )
    return document


def _write_scope_anchor(
    path: Path,
    value: Mapping[str, Any],
    *,
    exclusive: bool,
) -> None:
    _ensure_private_runtime_directory(path.parent)
    payload = _canonical_bytes(_finalize_scope_anchor(value)) + b"\n"
    if exclusive:
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
            os.link(temporary, path, follow_symlinks=False)
            os.chmod(path, 0o600)
            _fsync_directory(path.parent)
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()
        return
    if path.is_symlink():
        raise TrainOnlyLexiconError(
            "formal checkpoint scope anchor must not be a symlink"
        )
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


def _formal_checkpoint_scope_projection(
    config: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    web = config.get("web_settings")
    runtime = config.get("runtime_settings")
    web_budget = (
        web.get("physical_attempt_budget") if isinstance(web, Mapping) else None
    )
    llm_cap = (
        runtime.get("max_llm_http_attempts")
        if isinstance(runtime, Mapping)
        else None
    )
    if (
        not isinstance(web_budget, Mapping)
        or set(web_budget) != {"scope_id", "cap"}
        or not isinstance(web_budget.get("scope_id"), str)
        or not web_budget.get("scope_id")
        or isinstance(web_budget.get("cap"), bool)
        or not isinstance(web_budget.get("cap"), int)
        or web_budget.get("cap") <= 0
        or isinstance(llm_cap, bool)
        or not isinstance(llm_cap, int)
        or llm_cap <= 0
    ):
        raise TrainOnlyLexiconError(
            "formal checkpoint provider scopes or caps are invalid"
        )
    return {
        "deepseek": {
            "scope_id": FORMAL_DEEPSEEK_ATTEMPT_SCOPE_ID,
            "cap": llm_cap,
        },
        "tavily": {
            "scope_id": web_budget["scope_id"],
            "cap": web_budget["cap"],
        },
    }


def _expected_formal_checkpoint_scope_anchor(
    *,
    workspace_root: Path,
    config: Mapping[str, Any],
    binding: Mapping[str, Any],
    state: str,
) -> tuple[Path, Path, dict[str, Any]]:
    if state not in {"prepared", "active"}:
        raise TrainOnlyLexiconError("formal checkpoint scope state is invalid")
    provider_scopes = _formal_checkpoint_scope_projection(config)
    scope_identity = {
        provider: provider_scopes[provider]["scope_id"]
        for provider in sorted(provider_scopes)
    }
    scope_id = "fscope-" + _sha256_bytes(_canonical_bytes(scope_identity))
    runtime_root = (
        workspace_root / FORMAL_CHECKPOINT_RUNTIME_LOGICAL_ROOT
    ).resolve()
    try:
        runtime_root.relative_to(workspace_root.resolve())
    except ValueError as exc:
        raise TrainOnlyLexiconError(
            "formal checkpoint runtime root escapes the workspace"
        ) from exc
    checkpoint_parent = runtime_root / ".formal-checkpoints"
    anchor_parent = runtime_root / ".formal-checkpoint-scopes"
    checkpoint_root = checkpoint_parent / scope_id
    anchor_path = anchor_parent / f"{scope_id}.json"
    authorization_sha256 = binding.get("authorization_sha256")
    if (
        not isinstance(authorization_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", authorization_sha256) is None
    ):
        raise TrainOnlyLexiconError(
            "formal checkpoint scope lacks its authorization identity"
        )
    anchor = {
        "schema_version": FORMAL_CHECKPOINT_SCOPE_ANCHOR_SCHEMA_VERSION,
        "scope_id": scope_id,
        "scope_identity": scope_identity,
        "provider_scopes": provider_scopes,
        "authorization_sha256": authorization_sha256,
        "checkpoint_logical_path": checkpoint_root.relative_to(
            workspace_root.resolve()
        ).as_posix(),
        "state": state,
    }
    return checkpoint_root, anchor_path, _finalize_scope_anchor(anchor)


def _validate_formal_checkpoint_scope_anchor(
    *,
    checkpoint_root: Path,
    anchor_path: Path,
    expected_anchor: Mapping[str, Any],
) -> str:
    document = _read_scope_anchor(anchor_path)
    state = document.get("state")
    if state not in {"prepared", "active"}:
        raise TrainOnlyLexiconError(
            "formal checkpoint scope anchor has an invalid state"
        )
    expected = dict(expected_anchor)
    expected["state"] = state
    expected = _finalize_scope_anchor(expected)
    if document != expected:
        raise TrainOnlyLexiconError(
            "formal checkpoint scope, budget, or authorization drifted"
        )
    if checkpoint_root.is_symlink():
        raise TrainOnlyLexiconError(
            "formal checkpoint scope root must not be a symlink"
        )
    if state == "active" and not checkpoint_root.is_dir():
        raise TrainOnlyLexiconError(
            "active formal checkpoint scope lost its durable checkpoint"
        )
    return str(state)


def _prepare_formal_checkpoint_scope(
    *,
    workspace_root: Path,
    config: Mapping[str, Any],
    binding: Mapping[str, Any],
) -> tuple[Path, Path, dict[str, Any]]:
    checkpoint_root, anchor_path, expected = (
        _expected_formal_checkpoint_scope_anchor(
            workspace_root=workspace_root,
            config=config,
            binding=binding,
            state="prepared",
        )
    )
    _ensure_private_runtime_directory(checkpoint_root.parent)
    _ensure_private_runtime_directory(anchor_path.parent)
    if anchor_path.exists() or anchor_path.is_symlink():
        _validate_formal_checkpoint_scope_anchor(
            checkpoint_root=checkpoint_root,
            anchor_path=anchor_path,
            expected_anchor=expected,
        )
    else:
        if checkpoint_root.exists() or checkpoint_root.is_symlink():
            raise TrainOnlyLexiconError(
                "formal checkpoint exists without its scope anchor"
            )
        try:
            _write_scope_anchor(anchor_path, expected, exclusive=True)
        except FileExistsError:
            _validate_formal_checkpoint_scope_anchor(
                checkpoint_root=checkpoint_root,
                anchor_path=anchor_path,
                expected_anchor=expected,
            )
    return checkpoint_root, anchor_path, expected


def _activate_formal_checkpoint_scope(
    *,
    checkpoint_root: str | Path,
    anchor_path: str | Path,
    expected_anchor: Mapping[str, Any],
) -> None:
    checkpoint = Path(checkpoint_root)
    anchor = Path(anchor_path)
    if not checkpoint.is_dir() or checkpoint.is_symlink():
        raise TrainOnlyLexiconError(
            "cannot activate a missing or unsafe formal checkpoint"
        )
    _validate_formal_checkpoint_scope_anchor(
        checkpoint_root=checkpoint,
        anchor_path=anchor,
        expected_anchor=expected_anchor,
    )
    active = dict(expected_anchor)
    active["state"] = "active"
    _write_scope_anchor(anchor, active, exclusive=False)


def _ids_sha256(ids: Sequence[str]) -> str:
    return _sha256_bytes(_canonical_bytes(list(ids)))


def _formal_config_sha256(config: Mapping[str, Any]) -> str:
    """Hash the exact JSON-compatible config consumed by the formal wrapper."""

    try:
        return _sha256_bytes(_canonical_bytes(config))
    except (TypeError, ValueError) as exc:
        raise TrainOnlyLexiconError("formal lexicon config is not canonical JSON") from exc


def _snapshot_train_input(train_input: FrozenTrainInput) -> FrozenTrainInput:
    """Detach preflight input from mutable JSON objects owned by callers."""

    records = tuple(copy.deepcopy(dict(record)) for record in train_input.records)
    return FrozenTrainInput(
        records=records,
        record_ids=tuple(train_input.record_ids),
        data_build_id=train_input.data_build_id,
        train_data_sha256=train_input.train_data_sha256,
        train_ids_sha256=train_input.train_ids_sha256,
        source_train_data_sha256=train_input.source_train_data_sha256,
        source_train_ids_sha256=train_input.source_train_ids_sha256,
        source_mode=train_input.source_mode,
        # Formal builds must never consume this path after preflight.  It is
        # retained only as source metadata; the wrapper writes records from the
        # in-memory snapshot to its private temporary directory.
        train_path=train_input.train_path,
        data_ref=copy.deepcopy(dict(train_input.data_ref)),
        train_partition_ref=(
            copy.deepcopy(dict(train_input.train_partition_ref))
            if train_input.train_partition_ref is not None
            else None
        ),
        train_partition_dependency=(
            copy.deepcopy(dict(train_input.train_partition_dependency))
            if train_input.train_partition_dependency is not None
            else None
        ),
        forbidden_ids=frozenset(train_input.forbidden_ids),
        forbidden_hashes=frozenset(train_input.forbidden_hashes),
        forbidden_contents=frozenset(train_input.forbidden_contents),
    )


def _expected_formal_builder_hash() -> str:
    path = Path(__file__).with_name("llm_lexicon_builder.py")
    if not path.is_file():
        raise TrainOnlyLexiconError("frozen formal lexicon builder source is missing")
    return sha256_file(path)


def _expected_formal_protocol_hashes() -> dict[str, str]:
    source_root = Path(__file__).parent
    hashes: dict[str, str] = {}
    for filename in FORMAL_PROTOCOL_SOURCE_FILES:
        path = source_root / filename
        if not path.is_file():
            raise TrainOnlyLexiconError(
                f"frozen formal lexicon protocol source is missing: {filename}"
            )
        hashes[f"src/build_lex/{filename}"] = sha256_file(path)
    return dict(sorted(hashes.items()))


def _mint_formal_build_authorization(
    *,
    dataset: str,
    config: Mapping[str, Any],
    data_ref_path: str | Path,
    train_partition_ref_path: str | Path,
    workspace_root: str | Path,
    frozen_train_input: FrozenTrainInput,
    credential_values: Mapping[str, str],
) -> object:
    """Mint an opaque one-use authorization after a successful preflight.

    This private entry point is intentionally imported by ``stage1_preflight``;
    the opaque sentinel prevents a serialized/reconstructed object from being
    accepted by the build wrapper.
    """

    if dataset != "full":
        raise TrainOnlyLexiconError("formal Stage-1 lexicon authorization is restricted to dataset=full")
    snapshot = _snapshot_train_input(frozen_train_input)
    if (
        snapshot.train_partition_ref is None
        or snapshot.train_partition_dependency is None
        or snapshot.source_mode != "data_ref+train_partition"
    ):
        raise TrainOnlyLexiconError(
            "formal authorization requires a validated fit-only train partition"
        )
    try:
        frozen_data_dependency = validate_dependency_ref(
            snapshot.data_ref, expected_kind="data"
        )
        frozen_partition_dependency = validate_dependency_ref(
            snapshot.train_partition_ref, expected_kind="train-partition"
        )
    except TrainingArtifactError as exc:
        raise TrainOnlyLexiconError(
            "formal authorization requires portable data/partition dependencies"
        ) from exc
    if (
        frozen_partition_dependency != snapshot.train_partition_dependency
        or frozen_data_dependency.get("artifact_id") != snapshot.data_build_id
    ):
        raise TrainOnlyLexiconError(
            "formal authorization dependency identities are inconsistent"
        )
    config_snapshot = copy.deepcopy(dict(config))
    config_sha256 = _formal_config_sha256(config_snapshot)
    records_sha256 = _sha256_bytes(_canonical_bytes(list(snapshot.records)))
    resolved_data_ref_path = Path(data_ref_path).resolve()
    data_ref_file_sha256 = sha256_file(resolved_data_ref_path)
    resolved_partition_ref_path = Path(train_partition_ref_path).resolve()
    partition_ref_file_sha256 = sha256_file(resolved_partition_ref_path)
    binding = {
        "schema_version": FORMAL_AUTHORIZATION_SCHEMA_VERSION,
        "dataset": dataset,
        "builder_module": "build_lex.llm_lexicon_builder",
        "builder_name": "build_lexicon",
        "builder_code_sha256": _expected_formal_builder_hash(),
        "protocol_code_sha256s": _expected_formal_protocol_hashes(),
        "config_sha256": config_sha256,
        "data_build_id": snapshot.data_build_id,
        "data_dependency": snapshot.data_ref,
        "data_dependency_ref_sha256": _portable_ref_file_sha256(
            snapshot.data_ref
        ),
        "train_partition_dependency": snapshot.train_partition_dependency,
        "train_partition_dependency_ref_sha256": _portable_ref_file_sha256(
            snapshot.train_partition_ref
        ),
        "train_record_count": len(snapshot.records),
        "train_records_sha256": records_sha256,
        "train_data_sha256": snapshot.train_data_sha256,
        "train_ids_sha256": snapshot.train_ids_sha256,
        "source_train_data_sha256": snapshot.source_train_data_sha256,
        "source_train_ids_sha256": snapshot.source_train_ids_sha256,
        "build_policy_version": LEXICON_BUILD_POLICY_VERSION,
        "execution_policy": FORMAL_EXECUTION_POLICY,
    }
    binding["authorization_sha256"] = _sha256_bytes(_canonical_bytes(binding))
    return _FormalBuildAuthorization(
        _FORMAL_AUTHORITY,
        binding=binding,
        config=config_snapshot,
        config_sha256=config_sha256,
        credential_values=credential_values,
        data_ref_file_sha256=data_ref_file_sha256,
        data_ref_path=resolved_data_ref_path,
        train_partition_ref_file_sha256=partition_ref_file_sha256,
        train_partition_ref_path=resolved_partition_ref_path,
        workspace_root=Path(workspace_root).resolve(),
        frozen_train_input=snapshot,
        records_sha256=records_sha256,
    )


def _consume_formal_build_authorization(
    authorization: object,
    *,
    dataset: str,
    config: Mapping[str, Any],
    data_ref: str | Path | None,
    train_partition_ref: str | Path | None,
    train_records: Sequence[Mapping[str, Any]] | None,
    builder: Callable[..., Mapping[str, Any]],
    judge_client: Any,
    web_searcher: Any,
    workspace_root: str | Path | None,
) -> tuple[FrozenTrainInput, dict[str, Any], dict[str, Any], tuple[str, ...]]:
    """Validate and consume the capability before the build creates anything."""

    if not isinstance(authorization, _FormalBuildAuthorization) or authorization._authority is not _FORMAL_AUTHORITY:
        raise TrainOnlyLexiconError("formal lexicon build requires a successful same-process preflight authorization")
    if authorization._used:
        raise TrainOnlyLexiconError("formal lexicon build authorization has already been consumed")
    if dataset != "full" or dataset != authorization._binding.get("dataset"):
        raise TrainOnlyLexiconError("formal lexicon dataset does not match its preflight authorization")
    if train_records is not None or data_ref is None or train_partition_ref is None:
        raise TrainOnlyLexiconError(
            "formal lexicon authorization requires exactly its frozen data/partition refs"
        )
    if Path(data_ref).resolve() != authorization._data_ref_path:
        raise TrainOnlyLexiconError("formal lexicon data_ref path does not match its preflight authorization")
    if Path(train_partition_ref).resolve() != authorization._train_partition_ref_path:
        raise TrainOnlyLexiconError(
            "formal lexicon train_partition_ref path does not match preflight"
        )
    if workspace_root is None:
        raise TrainOnlyLexiconError(
            "formal lexicon build requires an explicit workspace_root"
        )
    if Path(workspace_root).resolve() != authorization._workspace_root:
        raise TrainOnlyLexiconError(
            "formal lexicon workspace_root differs from preflight"
        )
    try:
        current_data_ref_sha256 = sha256_file(authorization._data_ref_path)
        current_partition_ref_sha256 = sha256_file(
            authorization._train_partition_ref_path
        )
    except OSError as exc:
        raise TrainOnlyLexiconError("formal lexicon data_ref disappeared after preflight") from exc
    if current_data_ref_sha256 != authorization._data_ref_file_sha256:
        raise TrainOnlyLexiconError("formal lexicon data_ref drifted after preflight")
    if (
        current_partition_ref_sha256
        != authorization._train_partition_ref_file_sha256
    ):
        raise TrainOnlyLexiconError(
            "formal lexicon train_partition_ref drifted after preflight"
        )
    if _formal_config_sha256(config) != authorization._config_sha256:
        raise TrainOnlyLexiconError("formal lexicon config drifted after preflight")
    if _formal_config_sha256(authorization._config) != authorization._config_sha256:
        raise TrainOnlyLexiconError("formal lexicon authorized config snapshot is corrupt")
    if judge_client is not None or web_searcher is not None:
        raise TrainOnlyLexiconError("formal lexicon builds forbid injected judgement or web clients")

    # Import lazily to avoid a train_only <-> CLI import cycle.
    from build_lex import llm_lexicon_builder as frozen_builder_module

    if builder is not frozen_builder_module.build_lexicon:
        raise TrainOnlyLexiconError("formal lexicon builds require the frozen llm_lexicon_builder.build_lexicon")
    if (
        getattr(builder, "__module__", None) != authorization._binding.get("builder_module")
        or getattr(builder, "__name__", None) != authorization._binding.get("builder_name")
        or _builder_code_sha256(builder) != authorization._binding.get("builder_code_sha256")
        or _expected_formal_builder_hash() != authorization._binding.get("builder_code_sha256")
    ):
        raise TrainOnlyLexiconError("formal lexicon builder code drifted after preflight")

    train_input = _snapshot_train_input(authorization._frozen_train_input)
    if train_input.train_path is None:
        raise TrainOnlyLexiconError("formal lexicon authorization lacks a frozen train source")
    try:
        current_train_sha256 = sha256_file(train_input.train_path)
        payload_path = train_input.train_path.parent / "payload_manifest.json"
        current_payload_sha256 = sha256_file(payload_path)
    except OSError as exc:
        raise TrainOnlyLexiconError("formal lexicon data dependency disappeared after preflight") from exc
    if current_train_sha256 != train_input.source_train_data_sha256:
        raise TrainOnlyLexiconError("formal lexicon train data drifted after preflight")
    if current_payload_sha256 != train_input.data_ref.get("payload_manifest_sha256"):
        raise TrainOnlyLexiconError("formal lexicon data payload manifest drifted after preflight")
    records_sha256 = _sha256_bytes(_canonical_bytes(list(train_input.records)))
    reloaded = resolve_train_input(
        data_ref=authorization._data_ref_path,
        train_partition_ref=authorization._train_partition_ref_path,
        formal=True,
        workspace_root=authorization._workspace_root,
    )
    if reloaded != train_input:
        raise TrainOnlyLexiconError(
            "formal lexicon partitioned train snapshot drifted after preflight"
        )
    binding = authorization._binding
    expected_static = {
        "schema_version": FORMAL_AUTHORIZATION_SCHEMA_VERSION,
        "dataset": "full",
        "config_sha256": authorization._config_sha256,
        "build_policy_version": LEXICON_BUILD_POLICY_VERSION,
        "protocol_code_sha256s": _expected_formal_protocol_hashes(),
        "execution_policy": FORMAL_EXECUTION_POLICY,
    }
    if any(binding.get(key) != value for key, value in expected_static.items()):
        raise TrainOnlyLexiconError("formal lexicon authorization protocol binding is corrupt")
    expected_lineage = {
        "data_build_id": train_input.data_build_id,
        "data_dependency": train_input.data_ref,
        "data_dependency_ref_sha256": _portable_ref_file_sha256(
            train_input.data_ref
        ),
        "train_partition_dependency": train_input.train_partition_dependency,
        "train_partition_dependency_ref_sha256": _portable_ref_file_sha256(
            train_input.train_partition_ref
        ),
        "train_record_count": len(train_input.records),
        "train_records_sha256": records_sha256,
        "train_data_sha256": train_input.train_data_sha256,
        "train_ids_sha256": train_input.train_ids_sha256,
        "source_train_data_sha256": train_input.source_train_data_sha256,
        "source_train_ids_sha256": train_input.source_train_ids_sha256,
    }
    if records_sha256 != authorization._records_sha256 or any(
        binding.get(key) != value for key, value in expected_lineage.items()
    ):
        raise TrainOnlyLexiconError("formal lexicon train snapshot drifted after preflight")
    declared_hash = binding.get("authorization_sha256")
    unhashed = {key: value for key, value in binding.items() if key != "authorization_sha256"}
    if declared_hash != _sha256_bytes(_canonical_bytes(unhashed)):
        raise TrainOnlyLexiconError("formal lexicon authorization binding is corrupt")
    authorization._used = True
    return (
        train_input,
        copy.deepcopy(dict(binding)),
        copy.deepcopy(dict(authorization._config)),
        tuple(authorization._credential_values),
    )


def _record_id(record: Mapping[str, Any], index: int) -> str:
    value = record.get("id")
    if value is None or str(value) == "":
        raise TrainOnlyLexiconError(f"train record at index {index} has no stable id")
    return str(value)


def _record_hash(record: Mapping[str, Any]) -> str:
    return _sha256_bytes(_canonical_bytes(record))


def _validate_records(records: Sequence[Mapping[str, Any]], *, formal: bool) -> tuple[str, ...]:
    ids = tuple(_record_id(record, index) for index, record in enumerate(records))
    if len(ids) != len(set(ids)):
        raise TrainOnlyLexiconError("train record IDs are not unique")
    if formal and any(not value.isdecimal() or str(int(value)) != value for value in ids):
        raise TrainOnlyLexiconError("formal normalized train IDs must be canonical decimal strings")
    for index, record in enumerate(records):
        if not isinstance(record.get("content"), str):
            raise TrainOnlyLexiconError(f"train record {ids[index]!r} has invalid content")
        if not isinstance(record.get("quadruples"), list):
            raise TrainOnlyLexiconError(f"train record {ids[index]!r} has no normalized quadruple array")
    return ids


def _payload_entry(payload_manifest: Mapping[str, Any], relative_path: str) -> Mapping[str, Any] | None:
    for entry in payload_manifest.get("files", []):
        if isinstance(entry, Mapping) and entry.get("path") == relative_path:
            return entry
    return None


def _verify_external_payload_manifest(target_dir: Path, payload_manifest: Mapping[str, Any]) -> None:
    if payload_manifest.get("schema_version") != PAYLOAD_MANIFEST_VERSION:
        raise TrainOnlyLexiconError("unsupported data payload manifest")
    actual: list[dict[str, Any]] = []
    for path in sorted(target_dir.rglob("*")):
        if not path.is_file() or path.name == "payload_manifest.json":
            continue
        if path.is_symlink():
            raise TrainOnlyLexiconError("data target payload cannot contain symlinks")
        relative = path.relative_to(target_dir).as_posix()
        actual.append({"path": relative, "size": path.stat().st_size, "sha256": sha256_file(path)})
    if payload_manifest.get("files") != actual:
        raise TrainOnlyLexiconError("data target payload does not match its manifest")


def _resolve_data_target(
    *, data_identity: Mapping[str, Any], target_dir: Path
) -> FrozenTrainInput:
    data_build_id = data_identity.get("artifact_id")
    if not isinstance(data_build_id, str) or not DATA_BUILD_ID_RE.fullmatch(data_build_id):
        raise TrainOnlyLexiconError("data-ref has an invalid data artifact ID")
    if not target_dir.is_absolute() or not target_dir.is_dir() or target_dir.name != data_build_id:
        raise TrainOnlyLexiconError("data-ref target is missing or does not match its artifact ID")
    payload_path = target_dir / "payload_manifest.json"
    train_path = target_dir / "train.json"
    split_path = target_dir / "split_manifest.json"
    if not payload_path.is_file() or not train_path.is_file() or not split_path.is_file():
        raise TrainOnlyLexiconError("data target is missing train/split/payload inputs")
    if data_identity.get("payload_manifest_sha256") != sha256_file(payload_path):
        raise TrainOnlyLexiconError("data-ref payload manifest hash mismatch")
    payload_manifest = _load_json(payload_path)
    _verify_external_payload_manifest(target_dir, payload_manifest)
    train_entry = _payload_entry(payload_manifest, "train.json")
    split_entry = _payload_entry(payload_manifest, "split_manifest.json")
    if train_entry is None or split_entry is None:
        raise TrainOnlyLexiconError("data payload manifest does not anchor train and split files")
    if train_entry.get("sha256") != sha256_file(train_path) or split_entry.get("sha256") != sha256_file(split_path):
        raise TrainOnlyLexiconError("data payload file hash mismatch")

    records_value = _load_json(train_path)
    if not isinstance(records_value, list) or not all(isinstance(item, Mapping) for item in records_value):
        raise TrainOnlyLexiconError("normalized train payload must be an array of objects")
    records = tuple(records_value)
    record_ids = _validate_records(records, formal=True)
    split = _load_json(split_path)
    if split.get("schema_version") != "stage1-split/v1":
        raise TrainOnlyLexiconError("unsupported split manifest")
    declared_train_ids = tuple(str(value) for value in split.get("train_ids", []))
    if declared_train_ids != record_ids:
        raise TrainOnlyLexiconError("train.json IDs do not exactly match split_manifest.train_ids")
    train_ids_hash = _ids_sha256(record_ids)
    if split.get("train_ids_sha256") != train_ids_hash:
        raise TrainOnlyLexiconError("split_manifest.train_ids_sha256 mismatch")

    dev_ids = {str(value) for value in split.get("dev_ids", [])}
    test_ids = {str(value) for value in split.get("test_ids", [])}
    forbidden_ids = dev_ids | test_ids
    if forbidden_ids.intersection(record_ids):
        raise TrainOnlyLexiconError("train IDs overlap dev/test IDs")
    forbidden_hashes: set[str] = set()
    forbidden_contents: set[str] = set()
    source_hashes = split.get("source_sha256", {})
    if isinstance(source_hashes, Mapping):
        forbidden_hashes.update(
            str(value) for value in source_hashes.values() if isinstance(value, str)
        )
    for split_name in ("dev", "test"):
        split_file = target_dir / f"{split_name}.json"
        if not split_file.is_file():
            continue
        values = _load_json(split_file)
        for value in values if isinstance(values, list) else []:
            if not isinstance(value, Mapping):
                continue
            forbidden_hashes.add(_record_hash(value))
            content = value.get("content")
            if isinstance(content, str):
                forbidden_contents.add(content)
                forbidden_hashes.add(content_sha256(content))

    return FrozenTrainInput(
        records=records,
        record_ids=record_ids,
        data_build_id=data_build_id,
        train_data_sha256=sha256_file(train_path),
        train_ids_sha256=train_ids_hash,
        source_train_data_sha256=sha256_file(train_path),
        source_train_ids_sha256=train_ids_hash,
        source_mode="data_ref",
        train_path=train_path,
        data_ref=copy.deepcopy(dict(data_identity)),
        train_partition_ref=None,
        train_partition_dependency=None,
        forbidden_ids=frozenset(forbidden_ids),
        forbidden_hashes=frozenset(forbidden_hashes),
        forbidden_contents=frozenset(forbidden_contents),
    )


def _resolve_data_ref(data_ref_path: Path) -> FrozenTrainInput:
    locator = _load_json(data_ref_path)
    if (
        not isinstance(locator, Mapping)
        or locator.get("schema_version") != LOCATOR_REF_VERSION
        or locator.get("artifact_kind") != "data"
    ):
        raise TrainOnlyLexiconError("data-ref is not a Stage-1 data locator")
    target_dir = Path(str(locator.get("target_path", "")))
    return _resolve_data_target(data_identity=locator, target_dir=target_dir)


def _apply_train_partition(
    train_input: FrozenTrainInput,
    *,
    train_partition_ref: str | Path,
    workspace_root: str | Path,
) -> FrozenTrainInput:
    if train_input.source_mode != "data_ref":
        raise TrainOnlyLexiconError(
            "train partitions may only be applied to a frozen data_ref"
        )
    data_dependency = {
        "schema_version": "stage1-dependency-ref/v1",
        "artifact_kind": "data",
        "artifact_id": train_input.data_build_id,
        "payload_manifest_sha256": train_input.data_ref.get(
            "payload_manifest_sha256"
        ),
    }
    try:
        bundle = load_train_partition(
            train_partition_ref,
            workspace_root=workspace_root,
            expected_data_dependency=data_dependency,
        )
    except (TrainPartitionError, OSError, ValueError) as exc:
        raise TrainOnlyLexiconError(
            f"train-partition validation failed: {exc}"
        ) from exc
    return _materialize_partition_view(
        train_input,
        fit_ids=bundle.fit_ids,
        calibration_ids=bundle.calibration_ids,
        data_dependency=bundle.data_dependency,
        partition_dependency=bundle.partition_dependency,
    )


def _materialize_partition_view(
    train_input: FrozenTrainInput,
    *,
    fit_ids: Sequence[str],
    calibration_ids: Sequence[str],
    data_dependency: Mapping[str, Any],
    partition_dependency: Mapping[str, Any],
) -> FrozenTrainInput:
    by_id = {str(record["id"]): record for record in train_input.records}
    fit_identity = tuple(str(value) for value in fit_ids)
    calibration_identity = tuple(str(value) for value in calibration_ids)
    if (
        len(fit_identity) != len(set(fit_identity))
        or len(calibration_identity) != len(set(calibration_identity))
        or set(fit_identity).intersection(calibration_identity)
        or set(fit_identity).union(calibration_identity) != set(by_id)
    ):
        raise TrainOnlyLexiconError(
            "train-partition fit/calibration IDs are not disjoint and exhaustive"
        )
    fit_records = tuple(copy.deepcopy(dict(by_id[value])) for value in fit_identity)
    calibration_records = tuple(by_id[value] for value in calibration_identity)
    forbidden_ids = set(train_input.forbidden_ids)
    forbidden_hashes = set(train_input.forbidden_hashes)
    forbidden_contents = set(train_input.forbidden_contents)
    for record in calibration_records:
        forbidden_ids.add(str(record["id"]))
        forbidden_hashes.add(_record_hash(record))
        content = record.get("content")
        if isinstance(content, str):
            forbidden_contents.add(content)
            forbidden_hashes.add(content_sha256(content))
    fit_data_sha256 = _sha256_bytes(_canonical_bytes(list(fit_records)))
    return FrozenTrainInput(
        records=fit_records,
        record_ids=fit_identity,
        data_build_id=train_input.data_build_id,
        train_data_sha256=fit_data_sha256,
        train_ids_sha256=_ids_sha256(fit_identity),
        source_train_data_sha256=train_input.source_train_data_sha256,
        source_train_ids_sha256=train_input.source_train_ids_sha256,
        source_mode="data_ref+train_partition",
        train_path=train_input.train_path,
        data_ref=copy.deepcopy(dict(data_dependency)),
        train_partition_ref=copy.deepcopy(dict(partition_dependency)),
        train_partition_dependency=copy.deepcopy(dict(partition_dependency)),
        forbidden_ids=frozenset(forbidden_ids),
        forbidden_hashes=frozenset(forbidden_hashes),
        forbidden_contents=frozenset(forbidden_contents),
    )


def _resolve_portable_train_input(
    *,
    data_dependency: Mapping[str, Any],
    train_partition_dependency: Mapping[str, Any],
    workspace_root: str | Path,
) -> FrozenTrainInput:
    """Deep-resolve the two locator-free dependencies embedded in a formal target."""

    root = Path(workspace_root).resolve()
    try:
        frozen_data = validate_dependency_ref(data_dependency, expected_kind="data")
        frozen_partition = validate_dependency_ref(
            train_partition_dependency, expected_kind="train-partition"
        )
        data_target = resolve_dependency_target(frozen_data, root)
        partition_target = resolve_dependency_target(frozen_partition, root)
        data_report = validate_data_target(data_target)
        partition_report = validate_train_partition_target(
            partition_target,
            workspace_root=root,
            expected_data_dependency=frozen_data,
        )
    except (
        TrainingArtifactError,
        Stage1DataError,
        TrainPartitionError,
        OSError,
        ValueError,
    ) as exc:
        raise TrainOnlyLexiconError(
            f"portable formal dependency failed deep validation: {exc}"
        ) from exc
    if (
        data_report.get("data_build_id") != frozen_data["artifact_id"]
        or data_report.get("payload_manifest_sha256")
        != frozen_data["payload_manifest_sha256"]
        or partition_report.get("partition_dependency") != frozen_partition
    ):
        raise TrainOnlyLexiconError(
            "portable formal dependency identity differs from validated target"
        )
    full_train = _resolve_data_target(
        data_identity=frozen_data,
        target_dir=data_target,
    )
    return _materialize_partition_view(
        full_train,
        fit_ids=partition_report["fit_ids"],
        calibration_ids=partition_report["calibration_ids"],
        data_dependency=frozen_data,
        partition_dependency=frozen_partition,
    )


def resolve_train_input(
    *,
    data_ref: str | Path | None = None,
    train_partition_ref: str | Path | None = None,
    train_records: Sequence[Mapping[str, Any]] | None = None,
    formal: bool = True,
    workspace_root: str | Path | None = None,
    forbidden_dev_test_ids: Sequence[str] = (),
    forbidden_dev_test_hashes: Sequence[str] = (),
) -> FrozenTrainInput:
    """Resolve exactly one train input; formal mode requires the frozen locator."""

    if data_ref is not None and train_records is not None:
        raise TrainOnlyLexiconError("provide data_ref or train_records, not both")
    if formal and (data_ref is None or train_partition_ref is None):
        raise TrainOnlyLexiconError(
            "formal lexicon builds require data_ref and train_partition_ref"
        )
    if data_ref is not None:
        resolved = _resolve_data_ref(Path(data_ref))
        if train_partition_ref is not None:
            if workspace_root is None:
                raise TrainOnlyLexiconError(
                    "partitioned lexicon input requires workspace_root"
                )
            return _apply_train_partition(
                resolved,
                train_partition_ref=train_partition_ref,
                workspace_root=workspace_root,
            )
        return resolved
    if train_partition_ref is not None:
        raise TrainOnlyLexiconError(
            "train_partition_ref cannot be used without data_ref"
        )
    if train_records is None:
        raise TrainOnlyLexiconError("no train input was provided")
    records = tuple(train_records)
    if not all(isinstance(item, Mapping) for item in records):
        raise TrainOnlyLexiconError("train_records must contain only mappings")
    record_ids = _validate_records(records, formal=False)
    train_bytes = _canonical_bytes(list(records))
    train_hash = _sha256_bytes(train_bytes)
    descriptor = {
        "schema_version": "stage1-direct-train-input/v1",
        "artifact_kind": "engineering-train-records",
        "artifact_id": "data-direct-" + train_hash,
        "train_data_sha256": train_hash,
        "train_ids_sha256": _ids_sha256(record_ids),
    }
    return FrozenTrainInput(
        records=records,
        record_ids=record_ids,
        data_build_id=descriptor["artifact_id"],
        train_data_sha256=train_hash,
        train_ids_sha256=descriptor["train_ids_sha256"],
        source_train_data_sha256=train_hash,
        source_train_ids_sha256=descriptor["train_ids_sha256"],
        source_mode="direct_records",
        train_path=None,
        data_ref=descriptor,
        train_partition_ref=None,
        train_partition_dependency=None,
        forbidden_ids=frozenset(str(value) for value in forbidden_dev_test_ids),
        forbidden_hashes=frozenset(str(value) for value in forbidden_dev_test_hashes),
        forbidden_contents=frozenset(),
    )


def _sanitize(value: Any, key_path: tuple[str, ...] = ()) -> Any:
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, inner in value.items():
            normalized_key = str(key).lower()
            if normalized_key in _SECRET_KEYS or normalized_key.endswith(("_password", "_secret")):
                continue
            if key_path == () and key == "data_paths":
                continue
            result[str(key)] = _sanitize(inner, (*key_path, str(key)))
        return result
    if isinstance(value, (list, tuple)):
        return [_sanitize(item, key_path) for item in value]
    if isinstance(value, Path):
        return value.as_posix()
    return value


def _iter_scalar_strings(value: Any):
    if isinstance(value, Mapping):
        for inner in value.values():
            yield from _iter_scalar_strings(inner)
    elif isinstance(value, (list, tuple)):
        for inner in value:
            yield from _iter_scalar_strings(inner)
    elif isinstance(value, str):
        yield value


def _iter_all_strings(value: Any):
    if isinstance(value, Mapping):
        for key, inner in value.items():
            yield str(key)
            yield from _iter_all_strings(inner)
    elif isinstance(value, (list, tuple)):
        for inner in value:
            yield from _iter_all_strings(inner)
    elif isinstance(value, str):
        yield value


def _assert_no_locator_or_absolute_path(value: Any, *, label: str) -> None:
    """Reject workspace-specific locator material from formal identity metadata."""

    if isinstance(value, Mapping):
        for raw_key, inner in value.items():
            key = str(raw_key)
            normalized = key.lower()
            if (
                normalized == "target_path"
                or normalized.endswith("_locator")
                or "locator_sha256" in normalized
            ):
                raise TrainOnlyLexiconError(
                    f"{label} contains forbidden locator field {key!r}"
                )
            _assert_no_locator_or_absolute_path(inner, label=label)
        return
    if isinstance(value, (list, tuple)):
        for inner in value:
            _assert_no_locator_or_absolute_path(inner, label=label)
        return
    if isinstance(value, str) and (
        Path(value).is_absolute() or re.match(r"^[A-Za-z]:[\\/]", value)
    ):
        raise TrainOnlyLexiconError(
            f"{label} contains a workspace-specific absolute path"
        )


def _audit_no_credential_values(value: Any, credential_values: Sequence[str], label: str) -> None:
    secrets = tuple(secret for secret in credential_values if secret)
    if not secrets:
        return
    for text in _iter_all_strings(value):
        if any(secret in text for secret in secrets):
            raise TrainOnlyLexiconError(f"{label} contains a preflight credential value")


def _audit_no_forbidden_sources(value: Any, train_input: FrozenTrainInput, label: str) -> None:
    forbidden = set(train_input.forbidden_ids) | set(train_input.forbidden_hashes) | set(train_input.forbidden_contents)
    leaked = sorted(set(_iter_scalar_strings(value)).intersection(forbidden))
    if leaked:
        raise TrainOnlyLexiconError(f"{label} contains forbidden dev/test source values: {leaked[:5]}")


def _support_ids(term: Mapping[str, Any]) -> tuple[str, ...]:
    metadata = term.get("metadata")
    support = metadata.get("support") if isinstance(metadata, Mapping) else None
    values = support.get("sample_ids", []) if isinstance(support, Mapping) else []
    return tuple(str(value) for value in values) if isinstance(values, list) else ()


def _terminology_forbidden_key_paths(
    value: Any, *, path: str = "entry"
) -> list[str]:
    paths: list[str] = []
    if isinstance(value, Mapping):
        for key, nested in value.items():
            key_text = str(key)
            child_path = f"{path}.{key_text}"
            if key_text in TERMINOLOGY_TASK_FIELD_KEYS:
                paths.append(child_path)
            paths.extend(
                _terminology_forbidden_key_paths(nested, path=child_path)
            )
    elif isinstance(value, (list, tuple)):
        for index, nested in enumerate(value):
            paths.extend(
                _terminology_forbidden_key_paths(
                    nested, path=f"{path}[{index}]"
                )
            )
    return paths


def _normalize_terms(terms: Sequence[Mapping[str, Any]], train_input: FrozenTrainInput) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    train_ids = set(train_input.record_ids)
    for index, raw in enumerate(terms):
        if not isinstance(raw, Mapping):
            raise TrainOnlyLexiconError(f"lexicon term at index {index} is not an object")
        term = copy.deepcopy(dict(raw))
        term_text = term.get("term")
        definition = term.get("definition")
        forbidden_paths = _terminology_forbidden_key_paths(term)
        if forbidden_paths:
            raise TrainOnlyLexiconError(
                f"terminology entry {term_text!r} contains task fields: "
                f"{forbidden_paths}"
            )
        if not all(isinstance(value, str) and value.strip() for value in (term_text, definition)):
            raise TrainOnlyLexiconError(
                f"terminology entry at index {index} lacks term/definition"
            )
        term_text = term_text.strip()
        definition = definition.strip()
        term["term"] = term_text
        term["definition"] = definition
        variants = term.get("variants", [])
        if variants is None:
            variants = []
        if not isinstance(variants, list) or not all(isinstance(value, str) for value in variants):
            raise TrainOnlyLexiconError(f"lexicon term {term_text!r} has invalid variants")
        usage_notes = term.get("usage_notes", "") or ""
        ambiguity_notes = term.get("ambiguity_notes", "") or ""
        if not isinstance(usage_notes, str) or not isinstance(ambiguity_notes, str):
            raise TrainOnlyLexiconError(
                f"terminology entry {term_text!r} has invalid usage/ambiguity notes"
            )
        term["usage_notes"] = usage_notes.strip()
        term["ambiguity_notes"] = ambiguity_notes.strip()
        term["variants"] = [value.strip() for value in variants if value.strip()]
        entry_id = stable_term_evidence_id(
            term_text,
            definition,
            term["variants"],
            term["usage_notes"],
            term["ambiguity_notes"],
        )
        if term.get("lexicon_id") not in {None, entry_id}:
            raise TrainOnlyLexiconError(f"lexicon term {term_text!r} has a non-canonical lexicon_id")
        if entry_id in seen_ids:
            raise TrainOnlyLexiconError(f"duplicate stable lexicon ID {entry_id}")
        seen_ids.add(entry_id)
        support_ids = _support_ids(term)
        if any(value not in train_ids for value in support_ids):
            raise TrainOnlyLexiconError(f"lexicon term {term_text!r} references a non-train support ID")
        if set(support_ids).intersection(train_input.forbidden_ids):
            raise TrainOnlyLexiconError(f"lexicon term {term_text!r} references a dev/test support ID")
        term["lexicon_id"] = entry_id
        normalized.append(term)
    _audit_no_forbidden_sources(normalized, train_input, "lexicon terms")
    return normalized


def _read_jsonl_rows(
    path: Path,
    *,
    label: str,
    train_input: FrozenTrainInput,
    required: bool,
) -> list[dict[str, Any]]:
    if path.is_symlink():
        raise TrainOnlyLexiconError(f"{label} must not be a symlink")
    if not path.is_file():
        if required:
            raise TrainOnlyLexiconError(f"formal lexicon evidence is missing {label}")
        return []
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise TrainOnlyLexiconError(f"cannot read {label} as UTF-8") from exc
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            if required:
                raise TrainOnlyLexiconError(f"formal lexicon evidence has a blank row: {label}:{line_no}")
            continue
        try:
            row = json.loads(
                line,
                object_pairs_hook=_reject_duplicate_json_keys,
                parse_constant=_reject_nonfinite_json_number,
            )
        except json.JSONDecodeError as exc:
            raise TrainOnlyLexiconError(f"malformed JSONL evidence {label}:{line_no}") from exc
        if not isinstance(row, dict):
            raise TrainOnlyLexiconError(f"lexicon evidence row is not an object: {label}:{line_no}")
        _audit_no_forbidden_sources(row, train_input, f"{label}:{line_no}")
        rows.append(row)
    return rows


def _load_raw_artifacts(
    output_dir: Path,
    train_input: FrozenTrainInput,
    *,
    required: bool,
) -> tuple[dict[str, dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    hashes: dict[str, dict[str, Any]] = {}
    rows_by_file: dict[str, list[dict[str, Any]]] = {}
    for filename in RAW_AUDIT_FILES:
        path = output_dir / filename
        rows = _read_jsonl_rows(
            path,
            label=filename,
            train_input=train_input,
            required=required,
        )
        if path.is_file():
            hashes[filename] = {
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
                "row_count": len(rows),
            }
            rows_by_file[filename] = rows
    return hashes, rows_by_file


def _raw_artifact_hashes(output_dir: Path, train_input: FrozenTrainInput) -> dict[str, dict[str, Any]]:
    return _load_raw_artifacts(output_dir, train_input, required=False)[0]


def _load_debug_captures(
    *,
    llm_path: Path,
    search_path: Path,
    tavily_attempts_path: Path | None = None,
    train_input: FrozenTrainInput,
    required: bool,
) -> dict[str, list[dict[str, Any]]]:
    llm_source = _read_jsonl_rows(
        llm_path,
        label=llm_path.name,
        train_input=train_input,
        required=required,
    )
    search_source = _read_jsonl_rows(
        search_path,
        label=search_path.name,
        train_input=train_input,
        required=required,
    )
    tavily_attempts_source = (
        _read_jsonl_rows(
            tavily_attempts_path,
            label=tavily_attempts_path.name,
            train_input=train_input,
            required=required,
        )
        if tavily_attempts_path is not None
        else []
    )
    llm_rows = [
        {
            "stage": row.get("stage"),
            "term": row.get("term"),
            "attempt": row.get("attempt"),
            "request_payload": row.get("request_payload"),
            "raw_response": row.get("raw_response"),
            "parsed_response": row.get("parsed_response"),
            "error": row.get("error"),
        }
        for row in llm_source
    ]
    search_rows = [
        {
            "term": row.get("term"),
            "query": row.get("query"),
            "results": row.get("results"),
            "error": row.get("error"),
        }
        for row in search_source
    ]
    tavily_attempt_rows = [
        {
            "provider": row.get("provider"),
            "budget_scope_id": row.get("budget_scope_id"),
            "rank": row.get("rank"),
            "slot": row.get("slot"),
            "attempt": row.get("attempt"),
            "status": row.get("status"),
            "http_status": row.get("http_status"),
            "error_type": row.get("error_type"),
            "retryable": row.get("retryable"),
            "result_count": row.get("result_count"),
            "request_dispatched": row.get("request_dispatched"),
        }
        for row in tavily_attempts_source
    ]
    return {
        "llm_calls": llm_rows,
        "search_calls": search_rows,
        "tavily_attempts": tavily_attempt_rows,
    }


def _capture_hashes(captures: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, str | None]:
    llm_rows = list(captures.get("llm_calls", []))
    search_rows = list(captures.get("search_calls", []))
    tavily_attempt_rows = list(captures.get("tavily_attempts", []))
    return {
        "raw_response_sha256": _sha256_bytes(_canonical_bytes(llm_rows)) if llm_rows else None,
        "web_snapshot_sha256": _sha256_bytes(_canonical_bytes(search_rows)) if search_rows else None,
        "tavily_attempts_sha256": (
            _sha256_bytes(_canonical_bytes(tavily_attempt_rows))
            if tavily_attempt_rows
            else None
        ),
    }


def _debug_capture_hashes(output_dir: Path, train_input: FrozenTrainInput) -> dict[str, str | None]:
    captures = _load_debug_captures(
        llm_path=output_dir / "debug" / "llm_calls.jsonl",
        search_path=output_dir / "debug" / "search_calls.jsonl",
        tavily_attempts_path=output_dir / "debug" / "tavily_attempts.jsonl",
        train_input=train_input,
        required=False,
    )
    return _capture_hashes(captures)


def _positive_rank(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise TrainOnlyLexiconError(f"{label} must contain a positive integer rank")
    return value


def _nonempty_term(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise TrainOnlyLexiconError(f"{label} must contain a canonical non-empty term")
    return value


def _captured_raw_json(row: Mapping[str, Any]) -> Mapping[str, Any] | None:
    raw = row.get("raw_response")
    choices = raw.get("choices") if isinstance(raw, Mapping) else None
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], Mapping):
        return None
    choice = choices[0]
    if choice.get("finish_reason") in {"length", "content_filter", "insufficient_system_resource"}:
        return None
    message = choice.get("message")
    content = message.get("content") if isinstance(message, Mapping) else None
    if not isinstance(content, str) or not content.strip():
        return None
    try:
        parsed = json.loads(
            content,
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=_reject_nonfinite_json_number,
        )
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, Mapping) and parsed else None


def _valid_successful_llm_capture(row: Mapping[str, Any]) -> bool:
    request = row.get("request_payload")
    parsed = row.get("parsed_response")
    messages = request.get("messages") if isinstance(request, Mapping) else None
    error = row.get("error")
    raw_parsed = _captured_raw_json(row)
    return (
        (error is None or error == "")
        and isinstance(request, Mapping)
        and isinstance(request.get("model"), str)
        and bool(request.get("model").strip())
        and isinstance(messages, list)
        and bool(messages)
        and all(
            isinstance(message, Mapping)
            and isinstance(message.get("role"), str)
            and isinstance(message.get("content"), str)
            and bool(message.get("content"))
            for message in messages
        )
        and isinstance(parsed, Mapping)
        and bool(parsed)
        and raw_parsed == parsed
    )


def _validated_provider_usage(
    raw_response: Any,
    *,
    requested_model: Any,
) -> tuple[str, dict[str, int]]:
    if not isinstance(raw_response, Mapping):
        raise TrainOnlyLexiconError("successful LLM capture lacks its raw provider response")
    response_model = raw_response.get("model")
    if not isinstance(response_model, str) or not response_model.strip():
        raise TrainOnlyLexiconError("successful LLM capture lacks its provider-returned model")
    requested = str(requested_model or "").strip().casefold()
    returned = response_model.strip().casefold()
    if (
        requested != AUTHORIZED_FORMAL_LLM_MODEL
        or AUTHORIZED_PROVIDER_MODEL_RE.fullmatch(returned) is None
    ):
        raise TrainOnlyLexiconError(
            "provider-returned LLM model does not match the authorized model family"
        )
    usage = raw_response.get("usage")
    fields = (
        "prompt_tokens",
        "prompt_cache_hit_tokens",
        "prompt_cache_miss_tokens",
        "completion_tokens",
        "total_tokens",
    )
    if not isinstance(usage, Mapping):
        raise TrainOnlyLexiconError("successful LLM capture lacks provider token usage")
    normalized: dict[str, int] = {}
    for field in fields:
        value = usage.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise TrainOnlyLexiconError(
                f"successful LLM capture has invalid provider usage field {field}"
            )
        normalized[field] = value
    if (
        normalized["prompt_cache_hit_tokens"]
        + normalized["prompt_cache_miss_tokens"]
        != normalized["prompt_tokens"]
        or normalized["prompt_tokens"] + normalized["completion_tokens"]
        != normalized["total_tokens"]
    ):
        raise TrainOnlyLexiconError("provider token usage totals are inconsistent")
    return response_model.strip(), normalized


def _validate_stage_response(
    stage: str,
    response: Mapping[str, Any],
    *,
    allowed_evidence_ids: Sequence[str] | None = None,
    resource_role: str = "derogatory-lexicon/v1",
) -> None:
    terminology_mode = resource_role == TERMINOLOGY_LIBRARY_ROLE
    confidence = response.get("confidence")
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
        raise TrainOnlyLexiconError(f"{stage} response has an invalid confidence")
    if not isinstance(response.get("reason"), str):
        raise TrainOnlyLexiconError(f"{stage} response has no textual reason")
    if terminology_mode:
        _reject_terminology_task_fields(stage, response)
        expected_keys = TERMINOLOGY_RESPONSE_KEYS.get(stage)
        if expected_keys is None or set(response) != expected_keys:
            raise TrainOnlyLexiconError(
                f"{stage} terminology response keys must be exactly "
                f"{sorted(expected_keys or ())}; got {sorted(response)}"
            )
    if stage in {"context_judge", "web_evidence_judge"}:
        if not isinstance(response.get("supported"), bool):
            raise TrainOnlyLexiconError(f"{stage} response has no boolean supported decision")
    if stage == "context_judge":
        if not terminology_mode:
            categories = response.get("categories")
            if not isinstance(response.get("category"), str) or not isinstance(categories, list):
                raise TrainOnlyLexiconError("context_judge response has invalid category fields")
            _validate_stage_categories(
                stage,
                response,
                negative=not bool(response.get("supported")),
            )
    if stage == "web_evidence_judge":
        _validate_stage_evidence_ids(
            stage,
            response,
            allowed_evidence_ids=allowed_evidence_ids,
        )
        if response.get("supported") is True and not response.get("evidence_ids"):
            raise TrainOnlyLexiconError(
                "web_evidence_judge cannot return supported=true without citing supplied Web evidence"
            )
    if stage == "final_lexicon_judge":
        if not isinstance(response.get("include"), bool):
            raise TrainOnlyLexiconError("final_lexicon_judge response has no boolean include decision")
        scalar_fields = (
            ("definition", "usage_notes", "ambiguity_notes", "reason")
            if terminology_mode
            else ("category", "definition", "nonhateful_meaning", "reason")
        )
        for key in scalar_fields:
            if not isinstance(response.get(key), str):
                raise TrainOnlyLexiconError(f"final_lexicon_judge response has an invalid {key}")
        array_fields = (
            ("variants", "evidence_ids")
            if terminology_mode
            else ("categories", "variants", "evidence_ids")
        )
        for key in array_fields:
            values = response.get(key)
            if not isinstance(values, list) or not all(isinstance(value, str) for value in values):
                raise TrainOnlyLexiconError(f"final_lexicon_judge response has an invalid {key}")
        if terminology_mode:
            if response.get("include") is True and not response.get("definition"):
                raise TrainOnlyLexiconError(
                    "included terminology response has an empty definition"
                )
        else:
            _validate_stage_categories(
                stage,
                response,
                negative=not bool(response.get("include")),
            )
        _validate_stage_evidence_ids(
            stage,
            response,
            allowed_evidence_ids=allowed_evidence_ids,
        )


def _reject_terminology_task_fields(
    stage: str, response: Mapping[str, Any]
) -> None:
    forbidden = _terminology_forbidden_key_paths(
        response, path=f"{stage}_response"
    )
    if forbidden:
        raise TrainOnlyLexiconError(
            f"{stage} terminology response exposes task fields: {forbidden}"
        )


def _validate_stage_categories(
    stage: str,
    response: Mapping[str, Any],
    *,
    negative: bool,
) -> None:
    category = response.get("category")
    categories = response.get("categories")
    if category not in CANONICAL_LEXICON_CATEGORIES:
        raise TrainOnlyLexiconError(f"{stage} response has a non-canonical category")
    if (
        not isinstance(categories, list)
        or not categories
        or any(value not in CANONICAL_LEXICON_CATEGORIES for value in categories)
        or len(categories) != len(set(categories))
        or category not in categories
    ):
        raise TrainOnlyLexiconError(f"{stage} response has non-canonical categories")
    if negative and (category != "others" or categories != ["others"]):
        raise TrainOnlyLexiconError(
            f"{stage} negative response must use category/categories=others"
        )


def _validate_stage_evidence_ids(
    stage: str,
    response: Mapping[str, Any],
    *,
    allowed_evidence_ids: Sequence[str] | None,
) -> None:
    evidence_ids = response.get("evidence_ids")
    if (
        not isinstance(evidence_ids, list)
        or not all(
            isinstance(value, str) and value and value.strip() == value
            for value in evidence_ids
        )
        or len(evidence_ids) != len(set(evidence_ids))
    ):
        raise TrainOnlyLexiconError(f"{stage} response has invalid evidence_ids")
    if allowed_evidence_ids is None:
        return
    allowed = set(allowed_evidence_ids)
    unknown = sorted(set(evidence_ids) - allowed)
    if unknown:
        raise TrainOnlyLexiconError(
            f"{stage} response cites IDs outside the supplied Web evidence namespace"
        )


def _validate_formal_evidence(
    *,
    config: Mapping[str, Any],
    train_input: FrozenTrainInput,
    rows_by_file: Mapping[str, Sequence[Mapping[str, Any]]],
    captures: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    terminology_mode = config.get("resource_role") == TERMINOLOGY_LIBRARY_ROLE
    candidates = list(rows_by_file.get("candidates.jsonl", []))
    web_rows = list(rows_by_file.get("web_evidence.jsonl", []))
    judgement_rows = list(rows_by_file.get("llm_judgements.jsonl", []))
    rejected_rows = list(rows_by_file.get("rejected.jsonl", []))
    if not candidates:
        raise TrainOnlyLexiconError("formal lexicon build produced zero candidates")
    candidate_settings = config.get("candidate_settings")
    max_candidates = (
        candidate_settings.get("max_candidates")
        if isinstance(candidate_settings, Mapping)
        else None
    )
    if (
        isinstance(max_candidates, bool)
        or not isinstance(max_candidates, int)
        or len(candidates) > max_candidates
    ):
        raise TrainOnlyLexiconError("formal candidate count exceeds its authorized request budget")

    candidate_terms: list[str] = []
    train_ids = set(train_input.record_ids)
    for index, row in enumerate(candidates, start=1):
        if terminology_mode:
            exposed = _terminology_forbidden_key_paths(
                row, path=f"candidates[{index - 1}]"
            )
            if exposed:
                raise TrainOnlyLexiconError(
                    "category-free terminology candidate exposes task fields: "
                    f"{exposed}"
                )
            source_counts = row.get("source_counts")
            if (
                not isinstance(source_counts, Mapping)
                or not source_counts
                or set(source_counts) != {"content"}
                or isinstance(source_counts.get("content"), bool)
                or not isinstance(source_counts.get("content"), int)
                or source_counts["content"] <= 0
            ):
                raise TrainOnlyLexiconError(
                    "category-free terminology candidates must be mined from content only"
                )
        if _positive_rank(row.get("rank"), label="candidates.jsonl") != index:
            raise TrainOnlyLexiconError("candidate ranks must be contiguous and ordered from 1")
        candidate_terms.append(_nonempty_term(row.get("term"), label="candidates.jsonl"))
        support_ids = row.get("support_sample_ids")
        contexts = row.get("sample_contexts")
        if (
            not isinstance(support_ids, list)
            or not support_ids
            or not all(isinstance(value, str) and value in train_ids for value in support_ids)
            or len(support_ids) != len(set(support_ids))
        ):
            raise TrainOnlyLexiconError("formal candidate support IDs are not a unique train-only array")
        if (
            not isinstance(contexts, list)
            or not contexts
            or not all(
                isinstance(context, Mapping)
                and isinstance(context.get("id"), str)
                and context.get("id") in train_ids
                for context in contexts
            )
        ):
            raise TrainOnlyLexiconError("formal candidate contexts are not exclusively train-derived")
        if terminology_mode:
            for context in contexts:
                if context.get("source") != "content":
                    raise TrainOnlyLexiconError(
                        "category-free terminology contexts must come from content only"
                    )
                exposed = _terminology_forbidden_key_paths(
                    context, path="candidate_context"
                )
                if exposed:
                    raise TrainOnlyLexiconError(
                        "category-free terminology context exposes task fields: "
                        f"{exposed}"
                    )
    if len(candidate_terms) != len(set(candidate_terms)):
        raise TrainOnlyLexiconError("formal candidate terms must be unique")

    if len(web_rows) != len(candidates):
        raise TrainOnlyLexiconError("web_evidence.jsonl must contain exactly one row per candidate")
    expected_queries: dict[str, tuple[str, ...]] = {}
    evidence_ids_by_term: dict[str, tuple[str, ...]] = {}
    web_backend = str(
        (config.get("web_settings") if isinstance(config.get("web_settings"), Mapping) else {}).get(
            "backend", ""
        )
        or ""
    ).lower()
    for index, (term, row) in enumerate(zip(candidate_terms, web_rows), start=1):
        if _positive_rank(row.get("rank"), label="web_evidence.jsonl") != index or row.get("term") != term:
            raise TrainOnlyLexiconError("web evidence rank/term does not match its candidate")
        queries = row.get("queries")
        evidence = row.get("evidence")
        if (
            not isinstance(queries, list)
            or len(queries) != 3
            or not all(isinstance(query, str) and query.strip() == query and query for query in queries)
            or len(queries) != len(set(queries))
        ):
            raise TrainOnlyLexiconError("web evidence must contain exactly three unique canonical queries")
        if not isinstance(evidence, list) or not all(isinstance(item, Mapping) for item in evidence):
            raise TrainOnlyLexiconError("web evidence payload must be an array of objects")
        evidence_ids = [item.get("id") for item in evidence]
        if (
            not all(
                isinstance(value, str) and value and value.strip() == value
                for value in evidence_ids
            )
            or len(evidence_ids) != len(set(evidence_ids))
        ):
            raise TrainOnlyLexiconError(
                "web evidence must use unique non-empty evidence IDs"
            )
        if web_backend == "disabled" and evidence:
            raise TrainOnlyLexiconError("disabled web backend cannot publish non-empty evidence")
        expected_queries[term] = tuple(queries)
        evidence_ids_by_term[term] = tuple(evidence_ids)

    if len(judgement_rows) != len(candidates):
        raise TrainOnlyLexiconError("llm_judgements.jsonl must contain exactly one row per candidate")
    judgement_by_term: dict[str, Mapping[str, Any]] = {}
    for index, (term, row) in enumerate(zip(candidate_terms, judgement_rows), start=1):
        if _positive_rank(row.get("rank"), label="llm_judgements.jsonl") != index or row.get("term") != term:
            raise TrainOnlyLexiconError("LLM judgement rank/term does not match its candidate")
        llm_error = row.get("llm_error")
        if llm_error is not None and llm_error != "":
            raise TrainOnlyLexiconError("formal LLM judgement contains a failed candidate")
        for stage in FORMAL_LLM_STAGES:
            response = row.get(stage)
            if not isinstance(response, Mapping) or not response:
                raise TrainOnlyLexiconError(f"formal LLM judgement lacks a non-empty {stage} response")
            _validate_stage_response(
                stage,
                response,
                resource_role=str(
                    config.get("resource_role", "derogatory-lexicon/v1")
                ),
                allowed_evidence_ids=(
                    evidence_ids_by_term[term]
                    if stage in {"web_evidence_judge", "final_lexicon_judge"}
                    else None
                ),
            )
        context_judge = row["context_judge"]
        web_judge = row["web_evidence_judge"]
        final_judge = row["final_lexicon_judge"]
        if final_judge.get("include") is True:
            context_supported = context_judge.get("supported") is True
            web_supported = web_judge.get("supported") is True
            if not (context_supported or web_supported):
                raise TrainOnlyLexiconError(
                    "final_lexicon_judge cannot include a candidate rejected by both evidence judges"
                )
            if (
                config.get("resource_role") != TERMINOLOGY_LIBRARY_ROLE
                and context_supported
                and not web_supported
            ):
                context_categories = context_judge.get("categories")
                final_categories = final_judge.get("categories")
                if (
                    isinstance(context_categories, list)
                    and isinstance(final_categories, list)
                    and not set(context_categories).intersection(final_categories)
                ):
                    raise TrainOnlyLexiconError(
                        "final_lexicon_judge category conflicts with the only supporting evidence judge"
                    )
        judgement_by_term[term] = row

    for row in rejected_rows:
        term = _nonempty_term(row.get("term"), label="rejected.jsonl")
        rank = _positive_rank(row.get("rank"), label="rejected.jsonl")
        if rank > len(candidate_terms) or candidate_terms[rank - 1] != term:
            raise TrainOnlyLexiconError("rejected row does not match its candidate rank/term")
        if terminology_mode:
            candidate_payload = row.get("candidate")
            if isinstance(candidate_payload, Mapping):
                exposed = _terminology_forbidden_key_paths(
                    candidate_payload, path="rejected_candidate"
                )
                if exposed:
                    raise TrainOnlyLexiconError(
                        "category-free rejected candidate exposes task fields: "
                        f"{exposed}"
                    )

    search_rows = list(captures.get("search_calls", []))
    expected_query_pairs = {
        (term, query) for term, queries in expected_queries.items() for query in queries
    }
    actual_query_pairs: list[tuple[str, str]] = []
    search_by_pair: dict[tuple[str, str], Sequence[Mapping[str, Any]]] = {}
    web_settings = config.get("web_settings")
    if not isinstance(web_settings, Mapping):
        raise TrainOnlyLexiconError("formal evidence audit lacks authorized Web settings")
    evidence_text_limits = {
        "title": web_settings.get("max_title_chars"),
        "snippet": web_settings.get("max_snippet_chars"),
        "url": web_settings.get("max_url_chars"),
        "source": web_settings.get("max_source_chars"),
    }
    if web_backend != "disabled" and any(
        isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0
        for limit in evidence_text_limits.values()
    ):
        raise TrainOnlyLexiconError("formal Web evidence text limits are invalid")
    for row in search_rows:
        term = _nonempty_term(row.get("term"), label="debug search capture")
        query = row.get("query")
        if not isinstance(query, str) or not query:
            raise TrainOnlyLexiconError("debug search capture has an invalid query")
        search_error = row.get("error")
        results = row.get("results")
        if (
            (search_error is not None and search_error != "")
            or not isinstance(results, list)
            or not all(isinstance(item, Mapping) for item in results)
        ):
            raise TrainOnlyLexiconError("formal debug search capture contains an error or invalid results")
        if web_backend == "disabled" and results:
            raise TrainOnlyLexiconError("disabled web backend has a non-empty search capture")
        for item in results:
            if (
                not isinstance(item.get("id"), str)
                or not item.get("id")
                or item.get("query") != query
                or any(not isinstance(item.get(key, ""), str) for key in ("title", "snippet", "url", "source"))
                or any(
                    len(item.get(field, "")) > limit
                    for field, limit in evidence_text_limits.items()
                )
            ):
                raise TrainOnlyLexiconError("formal search result capture has an invalid evidence record")
            result_url = item.get("url", "")
            if result_url:
                try:
                    parsed_url = urlsplit(result_url)
                except ValueError as exc:
                    raise TrainOnlyLexiconError(
                        "formal search result capture has an invalid evidence URL"
                    ) from exc
                if parsed_url.scheme.lower() not in {"http", "https"} or not parsed_url.netloc:
                    raise TrainOnlyLexiconError(
                        "formal search result capture has an invalid evidence URL"
                    )
        max_results = web_settings.get("max_results")
        if web_backend != "disabled" and (
            isinstance(max_results, bool)
            or not isinstance(max_results, int)
            or len(results) > max_results
        ):
            raise TrainOnlyLexiconError("formal search capture exceeds its authorized result budget")
        pair = (term, query)
        actual_query_pairs.append(pair)
        search_by_pair[pair] = results
    if len(actual_query_pairs) != len(set(actual_query_pairs)) or set(actual_query_pairs) != expected_query_pairs:
        raise TrainOnlyLexiconError("debug search captures do not exactly cover every candidate query")
    for term, row in zip(candidate_terms, web_rows):
        collected: list[Mapping[str, Any]] = []
        seen_ids: set[str] = set()
        for query in expected_queries[term]:
            for item in search_by_pair[(term, query)]:
                evidence_id = str(item["id"])
                if evidence_id in seen_ids:
                    continue
                seen_ids.add(evidence_id)
                collected.append(item)
        if collected != row.get("evidence"):
            raise TrainOnlyLexiconError("web evidence does not match the normalized debug search captures")

    tavily_attempt_rows = list(captures.get("tavily_attempts", []))
    physical_budget = web_settings.get("physical_attempt_budget")
    retry_policy = web_settings.get("transport_retry_policy")
    web_attempt_cap = (
        physical_budget.get("cap")
        if isinstance(physical_budget, Mapping)
        else None
    )
    web_budget_scope = (
        physical_budget.get("scope_id")
        if isinstance(physical_budget, Mapping)
        else None
    )
    web_retries = (
        retry_policy.get("retries")
        if isinstance(retry_policy, Mapping)
        else None
    )
    if web_backend == "search_api":
        if (
            isinstance(web_attempt_cap, bool)
            or not isinstance(web_attempt_cap, int)
            or web_attempt_cap < len(expected_query_pairs)
            or not isinstance(web_budget_scope, str)
            or not web_budget_scope
            or isinstance(web_retries, bool)
            or not isinstance(web_retries, int)
            or web_retries < 0
        ):
            raise TrainOnlyLexiconError(
                "formal Web physical-attempt policy is invalid"
            )
        if not tavily_attempt_rows:
            raise TrainOnlyLexiconError(
                "formal search-api build lacks Tavily physical-attempt captures"
            )
    elif tavily_attempt_rows:
        raise TrainOnlyLexiconError(
            "non-search Web backend cannot publish Tavily attempt captures"
        )

    tavily_attempt_keys: list[tuple[int, int, int]] = []
    tavily_by_slot: dict[tuple[int, int], list[Mapping[str, Any]]] = {}
    expected_tavily_order: list[tuple[int, int, int]] = []
    for row in tavily_attempt_rows:
        rank = _positive_rank(row.get("rank"), label="Tavily attempt capture")
        slot = row.get("slot")
        attempt = row.get("attempt")
        if rank > len(candidate_terms):
            raise TrainOnlyLexiconError(
                "Tavily attempt capture references an unknown candidate rank"
            )
        if isinstance(slot, bool) or not isinstance(slot, int) or not 1 <= slot <= 3:
            raise TrainOnlyLexiconError("Tavily attempt capture has an invalid slot")
        if (
            isinstance(attempt, bool)
            or not isinstance(attempt, int)
            or attempt <= 0
            or not isinstance(web_retries, int)
            or attempt > web_retries + 1
        ):
            raise TrainOnlyLexiconError(
                "Tavily attempt capture exceeds its per-query retry budget"
            )
        if (
            row.get("provider") != "tavily"
            or row.get("budget_scope_id") != web_budget_scope
            or row.get("request_dispatched") is not True
        ):
            raise TrainOnlyLexiconError(
                "Tavily attempt capture disagrees with its authorized provider budget"
            )
        status = row.get("status")
        if status not in {"success", "failure", "ambiguous"}:
            raise TrainOnlyLexiconError("Tavily attempt capture has an invalid status")
        http_status = row.get("http_status")
        if http_status is not None and (
            isinstance(http_status, bool)
            or not isinstance(http_status, int)
            or not 100 <= http_status <= 599
        ):
            raise TrainOnlyLexiconError(
                "Tavily attempt capture has an invalid HTTP status"
            )
        result_count = row.get("result_count")
        error_type = row.get("error_type")
        retryable = row.get("retryable")
        if status == "success":
            if (
                isinstance(result_count, bool)
                or not isinstance(result_count, int)
                or result_count < 0
                or result_count > web_settings.get("max_results")
                or error_type is not None
                or retryable is not False
            ):
                raise TrainOnlyLexiconError(
                    "successful Tavily attempt capture is malformed"
                )
        elif (
            result_count is not None
            or not isinstance(error_type, str)
            or not error_type
            or retryable is not True
        ):
            raise TrainOnlyLexiconError(
                "failed or ambiguous Tavily attempt capture is malformed"
            )
        key = (rank, slot, attempt)
        tavily_attempt_keys.append(key)
        tavily_by_slot.setdefault((rank, slot), []).append(row)

    if (
        len(tavily_attempt_keys) != len(set(tavily_attempt_keys))
        or len(tavily_attempt_rows) > (web_attempt_cap or 0)
    ):
        raise TrainOnlyLexiconError(
            "Tavily physical-attempt captures are duplicate or over budget"
        )
    if web_backend == "search_api":
        for rank, term in enumerate(candidate_terms, start=1):
            for slot, query in enumerate(expected_queries[term], start=1):
                rows = tavily_by_slot.get((rank, slot), [])
                attempts = [row.get("attempt") for row in rows]
                if attempts != list(range(1, len(rows) + 1)):
                    raise TrainOnlyLexiconError(
                        "Tavily per-query attempts are not contiguous and ordered"
                    )
                success_indexes = [
                    index
                    for index, row in enumerate(rows)
                    if row.get("status") == "success"
                ]
                if success_indexes != [len(rows) - 1]:
                    raise TrainOnlyLexiconError(
                        "Tavily attempt history must end in exactly one success per query"
                    )
                result_count = rows[-1].get("result_count")
                if result_count != len(search_by_pair[(term, query)]):
                    raise TrainOnlyLexiconError(
                        "Tavily success result count disagrees with logical search capture"
                    )
                expected_tavily_order.extend(
                    (rank, slot, attempt_index)
                    for attempt_index in range(1, len(rows) + 1)
                )
    if tavily_attempt_keys != expected_tavily_order:
        raise TrainOnlyLexiconError(
            "Tavily attempt captures are not in canonical candidate/slot order"
        )

    llm_rows = list(captures.get("llm_calls", []))
    llm_settings = config.get("llm_settings")
    if not isinstance(llm_settings, Mapping):
        raise TrainOnlyLexiconError("formal evidence audit lacks authorized LLM settings")
    retries = llm_settings.get("retries")
    if isinstance(retries, bool) or not isinstance(retries, int) or retries < 0:
        raise TrainOnlyLexiconError("formal evidence audit has an invalid retry budget")
    expected_stage_pairs = {(term, stage) for term in candidate_terms for stage in FORMAL_LLM_STAGES}
    successful_pairs: list[tuple[str, str]] = []
    attempt_keys: list[tuple[str, str, int]] = []
    llm_attempts_by_pair: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    provider_response_models: dict[str, int] = {}
    provider_usage_totals = {
        "response_count": 0,
        "prompt_tokens": 0,
        "prompt_cache_hit_tokens": 0,
        "prompt_cache_miss_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }
    for row in llm_rows:
        term = _nonempty_term(row.get("term"), label="debug LLM capture")
        stage = row.get("stage")
        attempt = row.get("attempt")
        if term not in judgement_by_term or stage not in FORMAL_LLM_STAGES:
            raise TrainOnlyLexiconError("debug LLM capture references an unknown term or stage")
        if isinstance(attempt, bool) or not isinstance(attempt, int) or attempt <= 0:
            raise TrainOnlyLexiconError("debug LLM capture has an invalid attempt")
        if attempt > retries + 1:
            raise TrainOnlyLexiconError("debug LLM capture exceeds its authorized retry budget")
        attempt_keys.append((term, str(stage), attempt))
        llm_attempts_by_pair.setdefault((term, str(stage)), []).append(row)
        request = row.get("request_payload")
        if not isinstance(request, Mapping):
            raise TrainOnlyLexiconError("debug LLM capture lacks its request payload")
        expected_request_fields = {
            "model": llm_settings.get("model"),
            "max_tokens": llm_settings.get("max_tokens"),
            "stream": llm_settings.get("stream"),
            "response_format": {"type": "json_object"},
            "thinking": llm_settings.get("thinking"),
        }
        if llm_settings.get("send_temperature") is True:
            expected_request_fields["temperature"] = llm_settings.get("temperature")
        if llm_settings.get("reasoning_effort") is not None:
            expected_request_fields["reasoning_effort"] = llm_settings.get("reasoning_effort")
        if any(request.get(key) != value for key, value in expected_request_fields.items()):
            raise TrainOnlyLexiconError("debug LLM request does not match its authorized provider config")
        raw_response = row.get("raw_response")
        captured_provider_usage = None
        if isinstance(raw_response, Mapping) and (
            "choices" in raw_response
            or "model" in raw_response
            or "usage" in raw_response
        ):
            response_model, captured_provider_usage = _validated_provider_usage(
                raw_response,
                requested_model=llm_settings.get("model"),
            )
            authorized_max_tokens = llm_settings.get("max_tokens")
            if (
                isinstance(authorized_max_tokens, bool)
                or not isinstance(authorized_max_tokens, int)
                or captured_provider_usage["completion_tokens"]
                > authorized_max_tokens
            ):
                raise TrainOnlyLexiconError(
                    "provider completion usage exceeds the authorized output budget"
                )
            provider_response_models[response_model] = (
                provider_response_models.get(response_model, 0) + 1
            )
            provider_usage_totals["response_count"] += 1
            for field, value in captured_provider_usage.items():
                provider_usage_totals[field] += value
        if _valid_successful_llm_capture(row):
            if captured_provider_usage is None:
                raise TrainOnlyLexiconError(
                    "successful LLM capture lacks auditable provider model/usage"
                )
            successful_pairs.append((term, str(stage)))
            if row.get("parsed_response") != judgement_by_term[term].get(str(stage)):
                raise TrainOnlyLexiconError("successful LLM capture disagrees with its published stage judgement")
        elif not isinstance(row.get("error"), str) or not row.get("error"):
            raise TrainOnlyLexiconError("debug LLM capture is neither a successful response nor an explicit failed attempt")
    if len(attempt_keys) != len(set(attempt_keys)):
        raise TrainOnlyLexiconError("debug LLM captures contain duplicate stage attempts")
    runtime_settings = config.get("runtime_settings")
    llm_attempt_cap = (
        runtime_settings.get("max_llm_http_attempts")
        if isinstance(runtime_settings, Mapping)
        else None
    )
    if (
        isinstance(llm_attempt_cap, bool)
        or not isinstance(llm_attempt_cap, int)
        or llm_attempt_cap <= 0
        or len(llm_rows) > llm_attempt_cap
    ):
        raise TrainOnlyLexiconError(
            "debug LLM captures exceed the authorized global physical-attempt budget"
        )
    expected_llm_attempt_order: list[tuple[str, str, int]] = []
    for term in candidate_terms:
        for stage in FORMAL_LLM_STAGES:
            rows = llm_attempts_by_pair.get((term, stage), [])
            attempts = [row.get("attempt") for row in rows]
            if attempts != list(range(1, len(rows) + 1)):
                raise TrainOnlyLexiconError(
                    "debug LLM per-stage attempts are not contiguous and ordered"
                )
            success_indexes = [
                index
                for index, row in enumerate(rows)
                if _valid_successful_llm_capture(row)
            ]
            if success_indexes != [len(rows) - 1]:
                raise TrainOnlyLexiconError(
                    "debug LLM attempt history must end in exactly one success per stage"
                )
            expected_llm_attempt_order.extend(
                (term, stage, attempt_index)
                for attempt_index in range(1, len(rows) + 1)
            )
    if attempt_keys != expected_llm_attempt_order:
        raise TrainOnlyLexiconError(
            "debug LLM captures are not in canonical candidate/stage order"
        )
    if len(successful_pairs) != len(set(successful_pairs)) or set(successful_pairs) != expected_stage_pairs:
        raise TrainOnlyLexiconError("debug LLM captures do not contain exactly one success for each candidate and stage")

    success_by_stage = {
        stage: sum(1 for _term, item_stage in successful_pairs if item_stage == stage)
        for stage in FORMAL_LLM_STAGES
    }
    return {
        "schema_version": "stage1-formal-lexicon-evidence-audit/v3",
        "candidate_count": len(candidates),
        "web_evidence_count": len(web_rows),
        "llm_judgement_count": len(judgement_rows),
        "rejected_count": len(rejected_rows),
        "web_backend": web_backend,
        "query_count": len(expected_query_pairs),
        "search_capture_count": len(search_rows),
        "logical_web_request_count": len(search_rows) if web_backend == "search_api" else 0,
        "physical_web_attempt_count": len(tavily_attempt_rows),
        "web_retry_attempt_count": max(0, len(tavily_attempt_rows) - len(search_rows)),
        "ambiguous_web_attempt_count": sum(
            1 for row in tavily_attempt_rows if row.get("status") == "ambiguous"
        ),
        "authorized_web_physical_attempt_cap": web_attempt_cap if web_backend == "search_api" else 0,
        "llm_capture_count": len(llm_rows),
        "authorized_max_llm_capture_count": llm_attempt_cap,
        "successful_llm_capture_count": len(successful_pairs),
        "successful_llm_captures_by_stage": success_by_stage,
        "provider_response_models": dict(sorted(provider_response_models.items())),
        "provider_usage_totals": provider_usage_totals,
    }


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("wb") as handle:
        for row in rows:
            handle.write(_canonical_bytes(row) + b"\n")


def _capture_artifact_hashes(target_dir: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for filename in FORMAL_CAPTURE_FILES.values():
        path = target_dir / filename
        if path.is_file():
            rows = path.read_text(encoding="utf-8").splitlines()
            result[filename] = {
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
                "row_count": len(rows),
            }
    return result


def _prospective_capture_artifact_hashes(
    captures: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for capture_name, filename in FORMAL_CAPTURE_FILES.items():
        rows = list(captures.get(capture_name, []))
        payload = b"".join(_canonical_bytes(row) + b"\n" for row in rows)
        result[filename] = {
            "size": len(payload),
            "sha256": _sha256_bytes(payload),
            "row_count": len(rows),
        }
    return result


def _validated_formal_checkpoint_summary(
    value: Any,
    *,
    config: Mapping[str, Any],
    evidence_audit: Mapping[str, Any],
) -> dict[str, Any]:
    required = {
        "schema_version",
        "checkpoint_id",
        "candidate_count",
        "committed_prefix",
        "provider_attempt_counts",
        "provider_attempt_caps",
        "provider_attempt_scopes",
        "ambiguous_attempt_count",
        "attempt_head",
        "candidate_commit_head",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise TrainOnlyLexiconError(
            "formal builder lacks a canonical checkpoint completion summary"
        )
    summary = copy.deepcopy(dict(value))
    if summary.get("schema_version") != "stage1-formal-lexicon-checkpoint-summary/v2":
        raise TrainOnlyLexiconError("formal checkpoint summary schema mismatch")
    checkpoint_id = summary.get("checkpoint_id")
    if (
        not isinstance(checkpoint_id, str)
        or re.fullmatch(r"fchk-[0-9a-f]{64}", checkpoint_id) is None
    ):
        raise TrainOnlyLexiconError("formal checkpoint summary ID is invalid")
    candidate_count = evidence_audit.get("candidate_count")
    if (
        isinstance(candidate_count, bool)
        or not isinstance(candidate_count, int)
        or candidate_count <= 0
        or summary.get("candidate_count") != candidate_count
        or summary.get("committed_prefix") != candidate_count
    ):
        raise TrainOnlyLexiconError(
            "formal checkpoint summary does not cover the candidate frame"
        )
    expected_scopes = _formal_checkpoint_scope_projection(config)
    expected_caps = {
        provider: expected_scopes[provider]["cap"]
        for provider in sorted(expected_scopes)
    }
    expected_scope_ids = {
        provider: expected_scopes[provider]["scope_id"]
        for provider in sorted(expected_scopes)
    }
    expected_counts = {
        "deepseek": evidence_audit.get("llm_capture_count"),
        "tavily": evidence_audit.get("physical_web_attempt_count"),
    }
    if (
        summary.get("provider_attempt_caps") != expected_caps
        or summary.get("provider_attempt_scopes") != expected_scope_ids
        or summary.get("provider_attempt_counts") != expected_counts
    ):
        raise TrainOnlyLexiconError(
            "formal checkpoint provider scope, cap, or attempt counts drifted"
        )
    if any(
        isinstance(count, bool) or not isinstance(count, int) or count < 0
        for count in expected_counts.values()
    ):
        raise TrainOnlyLexiconError(
            "formal checkpoint attempt counts are invalid"
        )
    total_attempts = sum(expected_counts.values())
    ambiguous = summary.get("ambiguous_attempt_count")
    if (
        isinstance(ambiguous, bool)
        or not isinstance(ambiguous, int)
        or not 0 <= ambiguous <= total_attempts
    ):
        raise TrainOnlyLexiconError(
            "formal checkpoint ambiguous-attempt count is invalid"
        )
    attempt_head = summary.get("attempt_head")
    if (
        not isinstance(attempt_head, Mapping)
        or set(attempt_head)
        != {
            "reservation_count",
            "provider_counts",
            "reservation_head_sha256",
            "head_sha256",
        }
        or attempt_head.get("reservation_count") != total_attempts
        or attempt_head.get("provider_counts") != expected_counts
        or any(
            not isinstance(attempt_head.get(key), str)
            or re.fullmatch(r"[0-9a-f]{64}", attempt_head[key]) is None
            for key in ("reservation_head_sha256", "head_sha256")
        )
    ):
        raise TrainOnlyLexiconError("formal checkpoint attempt HEAD is invalid")
    commit_head = summary.get("candidate_commit_head")
    if (
        not isinstance(commit_head, Mapping)
        or set(commit_head) != {"committed_prefix", "commit_sha256"}
        or commit_head.get("committed_prefix") != candidate_count
        or not isinstance(commit_head.get("commit_sha256"), str)
        or re.fullmatch(r"[0-9a-f]{64}", commit_head["commit_sha256"])
        is None
        or commit_head["commit_sha256"] == "0" * 64
    ):
        raise TrainOnlyLexiconError(
            "formal checkpoint candidate commit HEAD is invalid"
        )
    return summary


def _call_builder(
    builder: Callable[..., Mapping[str, Any]],
    dataset: str,
    config: Mapping[str, Any],
    judge_client: Any,
    web_searcher: Any,
    formal_checkpoint_context: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    kwargs: dict[str, Any] = {}
    if judge_client is not None:
        kwargs["judge_client"] = judge_client
    if web_searcher is not None:
        kwargs["web_searcher"] = web_searcher
    if formal_checkpoint_context is not None:
        kwargs["formal_checkpoint_context"] = dict(formal_checkpoint_context)
    try:
        return builder(dataset, dict(config), **kwargs)
    except Exception as exc:
        if formal_checkpoint_context is not None:
            from build_lex.formal_checkpoint import FormalCheckpointError

            checkpoint_error: BaseException | None = exc
            while checkpoint_error is not None:
                if isinstance(checkpoint_error, FormalCheckpointError):
                    raise TrainOnlyLexiconError(str(checkpoint_error)) from exc
                checkpoint_error = checkpoint_error.__cause__
        raise


def _builder_code_sha256(builder: Callable[..., Any]) -> str:
    path_value = inspect.getsourcefile(builder)
    if path_value and Path(path_value).is_file():
        return sha256_file(Path(path_value))
    return _sha256_bytes(repr(builder).encode("utf-8"))


def _build_payload_manifest(target_dir: Path) -> dict[str, Any]:
    files = []
    for path in sorted(target_dir.iterdir(), key=lambda item: item.name):
        if not path.is_file() or path.name == "payload_manifest.json":
            continue
        if path.is_symlink():
            raise TrainOnlyLexiconError("lexicon payload cannot contain symlinks")
        files.append({"path": path.name, "size": path.stat().st_size, "sha256": sha256_file(path)})
    return {"schema_version": PAYLOAD_MANIFEST_VERSION, "files": files}


def _atomic_write_ref(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix=f".{path.name}.", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        os.chmod(temporary, 0o600)
        handle.write(_canonical_bytes(value) + b"\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    os.chmod(path, 0o600)
    _fsync_directory(path.parent)


def _fsync_regular_tree(root: Path) -> None:
    """Make an already validated payload durable before its final rename."""

    directories = [root]
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        if path.is_symlink():
            raise TrainOnlyLexiconError(
                "formal publication payload cannot contain symlinks"
            )
        if path.is_dir():
            directories.append(path)
            continue
        if not path.is_file():
            raise TrainOnlyLexiconError(
                "formal publication payload contains a non-regular entry"
            )
        flags = os.O_RDONLY
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(path, flags)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    for directory in sorted(
        directories,
        key=lambda item: len(item.parts),
        reverse=True,
    ):
        _fsync_directory(directory)


def build_train_only_lexicon(
    dataset: str,
    config: Mapping[str, Any],
    *,
    builder: Callable[..., Mapping[str, Any]],
    data_ref: str | Path | None = None,
    train_partition_ref: str | Path | None = None,
    train_records: Sequence[Mapping[str, Any]] | None = None,
    formal: bool = True,
    target_root: str | Path,
    write_ref: str | Path | None = None,
    judge_client: Any = None,
    web_searcher: Any = None,
    forbidden_dev_test_ids: Sequence[str] = (),
    forbidden_dev_test_hashes: Sequence[str] = (),
    build_authorization: object | None = None,
    workspace_root: str | Path | None = None,
) -> dict[str, Any]:
    """Run a legacy-compatible miner behind a strict train-only boundary."""

    formal_binding: dict[str, Any] | None = None
    formal_secret_values: tuple[str, ...] = ()
    if formal:
        # This is intentionally the first operation with any external side
        # effect.  It validates config/data/builder/client bindings entirely in
        # memory and consumes a one-use process-local capability.
        (
            train_input,
            formal_binding,
            authorized_config,
            formal_secret_values,
        ) = _consume_formal_build_authorization(
            build_authorization,
            dataset=str(dataset),
            config=config,
            data_ref=data_ref,
            train_partition_ref=train_partition_ref,
            train_records=train_records,
            builder=builder,
            judge_client=judge_client,
            web_searcher=web_searcher,
            workspace_root=workspace_root,
        )
        # Ignore the caller-owned mapping after the hash check.  All execution
        # and provenance below use the detached config snapshot minted by the
        # preflight, closing mutation/TOCTOU gaps within the same process.
        config = authorized_config
    else:
        if build_authorization is not None:
            raise TrainOnlyLexiconError("engineering builds must not receive a formal authorization")
        if train_partition_ref is not None:
            raise TrainOnlyLexiconError(
                "engineering lexicon builds cannot claim a formal train partition"
            )
        train_input = resolve_train_input(
            data_ref=data_ref,
            train_records=train_records,
            formal=False,
            forbidden_dev_test_ids=forbidden_dev_test_ids,
            forbidden_dev_test_hashes=forbidden_dev_test_hashes,
        )
    resource_role = str(
        config.get("resource_role", TERMINOLOGY_LIBRARY_ROLE)
    ).strip()
    if resource_role != TERMINOLOGY_LIBRARY_ROLE:
        raise TrainOnlyLexiconError(
            "Stage 1 main evidence must use the category-free terminology-library role"
        )
    root = Path(target_root).resolve()
    if formal:
        try:
            root.relative_to(Path(workspace_root).resolve())
        except ValueError as exc:
            raise TrainOnlyLexiconError(
                "formal lexicon target_root must live below workspace_root"
            ) from exc
    root.mkdir(parents=True, exist_ok=True)
    temporary_root = Path(tempfile.mkdtemp(prefix=".lexicon-build.", dir=root))
    formal_checkpoint_context = None
    if formal:
        if not isinstance(formal_binding, Mapping):
            raise TrainOnlyLexiconError(
                "formal lexicon build lacks its authorization binding"
            )
        authorization_sha256 = formal_binding.get("authorization_sha256")
        if (
            not isinstance(authorization_sha256, str)
            or re.fullmatch(r"[0-9a-f]{64}", authorization_sha256) is None
        ):
            raise TrainOnlyLexiconError(
                "formal lexicon authorization lacks a canonical checkpoint identity"
            )
        checkpoint_root, scope_anchor_path, scope_anchor = (
            _prepare_formal_checkpoint_scope(
                workspace_root=Path(workspace_root).resolve(),
                config=config,
                binding=formal_binding,
            )
        )
        formal_checkpoint_context = {
            "checkpoint_root": str(checkpoint_root),
            "scope_anchor_path": str(scope_anchor_path),
            "scope_anchor": scope_anchor,
            "intent": copy.deepcopy(dict(formal_binding)),
            # Runtime-only containment guard.  Values are never serialized into
            # the checkpoint manifest, summary, provenance, or public target.
            "forbidden_values": tuple(formal_secret_values),
        }
    try:
        train_path = train_input.train_path
        if formal or train_path is None:
            train_path = temporary_root / "train.json"
            _write_json(train_path, list(train_input.records))
        legacy_output_dir = temporary_root / "legacy"
        if formal:
            # The formal builder writes its candidate audit before checkpoint
            # materialisation.  Create the staging directory privately first
            # so no corpus-derived artifact is ever born under the process
            # umask's usual 0755 directory mode.
            _ensure_private_runtime_directory(legacy_output_dir)
            _ensure_private_runtime_directory(legacy_output_dir / "debug")
        build_config = copy.deepcopy(dict(config))
        # Engineering callers may omit the role, but the Stage-1 wrapper never
        # permits the injected builder to fall back to its legacy categorized
        # mode.
        build_config["resource_role"] = resource_role
        build_config["data_paths"] = {
            "input_paths": [str(train_path)],
            "output_dir": str(legacy_output_dir),
        }
        build_config.setdefault("runtime_settings", {})["resume"] = bool(formal)
        build_config["runtime_settings"]["debug"] = True
        build_config["runtime_settings"]["debug_dir"] = str(legacy_output_dir / "debug")
        if formal:
            build_config.setdefault("llm_settings", {})[
                "strict_provider_audit"
            ] = True
        result = _call_builder(
            builder,
            dataset,
            build_config,
            judge_client,
            web_searcher,
            formal_checkpoint_context=formal_checkpoint_context,
        )
        legacy_lexicon_path = legacy_output_dir / "lexicon.json"
        if legacy_lexicon_path.is_file():
            legacy_payload = _load_json(legacy_lexicon_path)
        else:
            legacy_payload = {"terms": result.get("terms", [])}
        forbidden_payload_paths = _terminology_forbidden_key_paths(
            legacy_payload, path="builder_output"
        )
        if forbidden_payload_paths:
            raise TrainOnlyLexiconError(
                "category-free terminology builder output contains task fields: "
                f"{forbidden_payload_paths}"
            )
        terms_value = legacy_payload.get("terms", result.get("terms", []))
        if not isinstance(terms_value, list):
            raise TrainOnlyLexiconError("legacy builder did not produce a term array")
        terms = _normalize_terms(terms_value, train_input)
        raw_artifacts, raw_rows = _load_raw_artifacts(
            legacy_output_dir,
            train_input,
            required=formal,
        )
        captures = _load_debug_captures(
            llm_path=legacy_output_dir / "debug" / "llm_calls.jsonl",
            search_path=legacy_output_dir / "debug" / "search_calls.jsonl",
            tavily_attempts_path=legacy_output_dir / "debug" / "tavily_attempts.jsonl",
            train_input=train_input,
            required=formal,
        )
        capture_hashes = _capture_hashes(captures)
        evidence_audit: dict[str, Any] | None = None
        checkpoint_completion: dict[str, Any] | None = None
        capture_artifacts: dict[str, dict[str, Any]] = {}
        if formal:
            _audit_no_credential_values(
                {
                    "legacy_payload": legacy_payload,
                    "builder_result": result,
                    "terms": terms,
                    "raw_evidence": raw_rows,
                    "debug_captures": captures,
                },
                formal_secret_values,
                "formal lexicon evidence",
            )
            evidence_audit = _validate_formal_evidence(
                config=config,
                train_input=train_input,
                rows_by_file=raw_rows,
                captures=captures,
            )
            checkpoint_completion = _validated_formal_checkpoint_summary(
                result.get("formal_checkpoint_summary"),
                config=config,
                evidence_audit=evidence_audit,
            )
            capture_artifacts = _prospective_capture_artifact_hashes(captures)
            required_capture_hashes = (
                "raw_response_sha256",
                "web_snapshot_sha256",
            )
            if any(capture_hashes[key] is None for key in required_capture_hashes):
                raise TrainOnlyLexiconError("formal lexicon builds require non-empty raw LLM and web captures")
            if (
                str(config.get("web_settings", {}).get("backend", "")).lower()
                == "search_api"
                and capture_hashes["tavily_attempts_sha256"] is None
            ):
                raise TrainOnlyLexiconError(
                    "formal search-api builds require non-empty Tavily attempt captures"
                )
        else:
            if capture_hashes["raw_response_sha256"] is None:
                capture_hashes["raw_response_sha256"] = raw_artifacts.get("llm_judgements.jsonl", {}).get("sha256")
            if capture_hashes["web_snapshot_sha256"] is None:
                capture_hashes["web_snapshot_sha256"] = raw_artifacts.get("web_evidence.jsonl", {}).get("sha256")
        sanitized_config = _sanitize(config)
        config_sha256 = _formal_config_sha256(config) if formal else _sha256_bytes(_canonical_bytes(sanitized_config))
        code_sha256 = _builder_code_sha256(builder)
        wrapper_code_sha256 = sha256_file(Path(__file__))
        protocol_code_sha256s = _expected_formal_protocol_hashes()
        terms_sha256 = _sha256_bytes(_canonical_bytes(terms))
        id_inputs = {
            "schema_version": LEXICON_SCHEMA_VERSION,
            "resource_role": resource_role,
            "dataset": str(dataset),
            "data_build_id": train_input.data_build_id,
            "train_data_sha256": train_input.train_data_sha256,
            "train_ids_sha256": train_input.train_ids_sha256,
            "source_train_data_sha256": train_input.source_train_data_sha256,
            "source_train_ids_sha256": train_input.source_train_ids_sha256,
            "train_partition_dependency": train_input.train_partition_dependency,
            "source_partition": (
                "fit"
                if train_input.train_partition_dependency is not None
                else "legacy-full-train"
            ),
            "calibration_contribution_count": 0,
            "builder_config_sha256": config_sha256,
            "builder_code_sha256": code_sha256,
            "wrapper_code_sha256": wrapper_code_sha256,
            "protocol_code_sha256s": protocol_code_sha256s,
            "build_policy_version": LEXICON_BUILD_POLICY_VERSION,
            "terms_sha256": terms_sha256,
            "raw_artifact_hashes": raw_artifacts,
            **capture_hashes,
        }
        if formal:
            id_inputs.update(
                {
                    "formal_authorization": formal_binding,
                    "formal_checkpoint_completion": checkpoint_completion,
                    "evidence_audit": evidence_audit,
                    "capture_artifacts": capture_artifacts,
                }
            )
        lexicon_build_id = "lex-" + _sha256_bytes(_canonical_bytes(id_inputs))
        # Publish only presentation metadata with an explicit allow-list.  A
        # builder's debug/report fields must never silently become verifier
        # evidence or a second task-label channel.
        public_fields = {
            key: copy.deepcopy(legacy_payload[key])
            for key in ("title", "language")
            if key in legacy_payload
        }
        lexicon_document = {
            "schema_version": LEXICON_SCHEMA_VERSION,
            "resource_role": resource_role,
            "lexicon_build_id": lexicon_build_id,
            "dataset": str(dataset),
            "source": "train-only-terminology-library-builder",
            "description": (
                "Derived exclusively from the frozen fit partition of the normalized train split."
                if formal
                else "Engineering-only legacy train-derived lexicon."
            ),
            **public_fields,
            "total_terms": len(terms),
            "terms": terms,
        }
        manifest = {
            "schema_version": LEXICON_MANIFEST_VERSION,
            "resource_role": resource_role,
            "lexicon_build_id": lexicon_build_id,
            "data_build_id": train_input.data_build_id,
            "source_split": "train",
            "source_partition": (
                "fit"
                if train_input.train_partition_dependency is not None
                else "legacy-full-train"
            ),
            "source_mode": train_input.source_mode,
            "scientific_eligible": bool(formal),
            "train_record_count": len(train_input.records),
            "train_data_sha256": train_input.train_data_sha256,
            "train_ids_sha256": train_input.train_ids_sha256,
            "source_train_data_sha256": train_input.source_train_data_sha256,
            "source_train_ids_sha256": train_input.source_train_ids_sha256,
            "train_partition_dependency": train_input.train_partition_dependency,
            "fit_only_verified": bool(
                train_input.train_partition_dependency is not None
            ),
            "calibration_contribution_count": 0,
            "builder_config_sha256": config_sha256,
            "builder_code_sha256": code_sha256,
            "wrapper_code_sha256": wrapper_code_sha256,
            "protocol_code_sha256s": protocol_code_sha256s,
            "build_policy_version": LEXICON_BUILD_POLICY_VERSION,
            "terms_sha256": terms_sha256,
            "raw_artifact_hashes": raw_artifacts,
            **capture_hashes,
            "lexicon_ids": [term["lexicon_id"] for term in terms],
            "train_only_verified": True,
            "id_inputs": id_inputs,
        }
        if formal:
            manifest.update(
                {
                    "formal_authorization": formal_binding,
                    "formal_checkpoint_completion": checkpoint_completion,
                    "evidence_audit": evidence_audit,
                    "capture_artifacts": capture_artifacts,
                }
            )
        provenance = {
            "schema_version": LEXICON_PROVENANCE_VERSION,
            "resource_role": resource_role,
            "lexicon_build_id": lexicon_build_id,
            "data_dependency": {
                "data_build_id": train_input.data_build_id,
                "split": "train",
                "train_data_sha256": train_input.train_data_sha256,
                "train_ids_sha256": train_input.train_ids_sha256,
            },
            "train_partition_dependency": train_input.train_partition_dependency,
            "lexicon_source_partition": (
                "fit"
                if train_input.train_partition_dependency is not None
                else "legacy-full-train"
            ),
            "calibration_contribution_count": 0,
            "builder": {
                "name": getattr(builder, "__name__", builder.__class__.__name__),
                "code_sha256": code_sha256,
                "wrapper_code_sha256": wrapper_code_sha256,
                "protocol_code_sha256s": protocol_code_sha256s,
                "policy_version": LEXICON_BUILD_POLICY_VERSION,
                "config_sha256": config_sha256,
                "resolved_config": sanitized_config,
            },
            "llm": _sanitize(config.get("llm_settings", {})),
            "web": _sanitize(config.get("web_settings", {})),
            "raw_artifact_hashes": raw_artifacts,
            **capture_hashes,
            "train_only_verified": True,
        }
        if formal:
            provenance.update(
                {
                    "formal_authorization": formal_binding,
                    "formal_checkpoint_completion": checkpoint_completion,
                    "evidence_audit": evidence_audit,
                    "capture_artifacts": capture_artifacts,
                    "formal_execution_policy": FORMAL_EXECUTION_POLICY,
                }
            )
            _assert_no_locator_or_absolute_path(
                formal_binding, label="formal authorization"
            )
            _assert_no_locator_or_absolute_path(
                id_inputs, label="formal lexicon ID inputs"
            )
            _assert_no_locator_or_absolute_path(
                provenance, label="formal lexicon provenance"
            )
        _audit_no_forbidden_sources(lexicon_document, train_input, "lexicon document")
        _audit_no_forbidden_sources(manifest, train_input, "lexicon manifest")
        _audit_no_forbidden_sources(provenance, train_input, "lexicon provenance")

        payload_dir = temporary_root / lexicon_build_id
        payload_dir.mkdir()
        _write_json(payload_dir / "data_ref.json", train_input.data_ref)
        if train_input.train_partition_ref is not None:
            _write_json(
                payload_dir / "train_partition_ref.json",
                train_input.train_partition_ref,
            )
        _write_json(payload_dir / "lexicon.json", lexicon_document)
        _write_json(payload_dir / "manifest.json", manifest)
        _write_json(payload_dir / "provenance.json", provenance)
        if formal:
            for filename in RAW_AUDIT_FILES:
                shutil.copyfile(legacy_output_dir / filename, payload_dir / filename)
            _write_jsonl(payload_dir / FORMAL_CAPTURE_FILES["llm_calls"], captures["llm_calls"])
            _write_jsonl(payload_dir / FORMAL_CAPTURE_FILES["search_calls"], captures["search_calls"])
            _write_jsonl(
                payload_dir / FORMAL_CAPTURE_FILES["tavily_attempts"],
                captures["tavily_attempts"],
            )
        _write_json(payload_dir / "payload_manifest.json", _build_payload_manifest(payload_dir))
        validate_lexicon_target(
            payload_dir,
            require_directory_name=False,
            workspace_root=workspace_root,
        )
        _fsync_regular_tree(payload_dir)

        final_dir = root / lexicon_build_id
        if final_dir.exists():
            validate_lexicon_target(final_dir, workspace_root=workspace_root)
            if _load_json(final_dir / "payload_manifest.json") != _load_json(payload_dir / "payload_manifest.json"):
                raise TrainOnlyLexiconError(f"existing target {lexicon_build_id} has different content")
        else:
            os.replace(payload_dir, final_dir)
            _fsync_directory(root)
        validate_lexicon_target(final_dir, workspace_root=workspace_root)
        locator = {
            "schema_version": LOCATOR_REF_VERSION,
            "artifact_kind": ARTIFACT_KIND,
            "artifact_id": lexicon_build_id,
            "target_path": str(final_dir.resolve()),
            "payload_manifest_sha256": sha256_file(final_dir / "payload_manifest.json"),
        }
        if write_ref is not None:
            _atomic_write_ref(Path(write_ref), locator)
        return locator
    finally:
        if temporary_root.exists():
            shutil.rmtree(temporary_root)


def _validate_entry_ids(lexicon: Mapping[str, Any], expected_ids: Sequence[str]) -> None:
    terms = lexicon.get("terms")
    if not isinstance(terms, list):
        raise TrainOnlyLexiconError("lexicon terms must be an array")
    actual_ids: list[str] = []
    for term in terms:
        if not isinstance(term, Mapping):
            raise TrainOnlyLexiconError("lexicon contains a non-object term")
        variants = term.get("variants", []) or []
        expected = stable_term_evidence_id(
            str(term.get("term", "")),
            str(term.get("definition", "")),
            list(variants),
            str(term.get("usage_notes", "")),
            str(term.get("ambiguity_notes", "")),
        )
        if term.get("lexicon_id") != expected or not LEXICON_ENTRY_ID_RE.fullmatch(expected):
            raise TrainOnlyLexiconError("lexicon entry ID cannot be recomputed")
        actual_ids.append(expected)
    if len(actual_ids) != len(set(actual_ids)) or list(expected_ids) != actual_ids:
        raise TrainOnlyLexiconError("lexicon entry IDs are duplicate or disagree with manifest")


def _validate_published_formal_authorization(
    *,
    binding: Any,
    manifest: Mapping[str, Any],
    provenance: Mapping[str, Any],
    id_inputs: Mapping[str, Any],
    train_input: FrozenTrainInput,
) -> None:
    if not isinstance(binding, Mapping):
        raise TrainOnlyLexiconError("formal lexicon target lacks its preflight authorization binding")
    if binding.get("schema_version") != FORMAL_AUTHORIZATION_SCHEMA_VERSION:
        raise TrainOnlyLexiconError("unsupported formal lexicon authorization schema")
    declared_hash = binding.get("authorization_sha256")
    unhashed = {key: value for key, value in binding.items() if key != "authorization_sha256"}
    if declared_hash != _sha256_bytes(_canonical_bytes(unhashed)):
        raise TrainOnlyLexiconError("formal lexicon authorization hash mismatch")
    if manifest.get("formal_authorization") != binding or provenance.get("formal_authorization") != binding:
        raise TrainOnlyLexiconError("formal lexicon authorization is inconsistent across payloads")
    if id_inputs.get("formal_authorization") != binding:
        raise TrainOnlyLexiconError("formal lexicon authorization disagrees with ID inputs")

    builder_provenance = provenance.get("builder")
    resolved_config = (
        builder_provenance.get("resolved_config")
        if isinstance(builder_provenance, Mapping)
        else None
    )
    if not isinstance(resolved_config, Mapping):
        raise TrainOnlyLexiconError("formal lexicon provenance lacks its exact resolved config")
    expected = {
        "dataset": "full",
        "builder_module": "build_lex.llm_lexicon_builder",
        "builder_name": "build_lexicon",
        "builder_code_sha256": manifest.get("builder_code_sha256"),
        "protocol_code_sha256s": manifest.get("protocol_code_sha256s"),
        "config_sha256": manifest.get("builder_config_sha256"),
        "data_build_id": train_input.data_build_id,
        "data_dependency": train_input.data_ref,
        "data_dependency_ref_sha256": _portable_ref_file_sha256(
            train_input.data_ref
        ),
        "train_partition_dependency": train_input.train_partition_dependency,
        "train_partition_dependency_ref_sha256": _portable_ref_file_sha256(
            train_input.train_partition_ref
        ),
        "train_record_count": len(train_input.records),
        "train_records_sha256": _sha256_bytes(_canonical_bytes(list(train_input.records))),
        "train_data_sha256": train_input.train_data_sha256,
        "train_ids_sha256": train_input.train_ids_sha256,
        "source_train_data_sha256": train_input.source_train_data_sha256,
        "source_train_ids_sha256": train_input.source_train_ids_sha256,
        "build_policy_version": LEXICON_BUILD_POLICY_VERSION,
        "execution_policy": FORMAL_EXECUTION_POLICY,
    }
    if set(binding) != {
        "schema_version",
        "authorization_sha256",
        *expected.keys(),
    }:
        raise TrainOnlyLexiconError(
            "formal lexicon authorization fields are not canonical"
        )
    if any(binding.get(key) != value for key, value in expected.items()):
        raise TrainOnlyLexiconError("formal lexicon authorization does not match published config/code/data lineage")
    if _formal_config_sha256(resolved_config) != binding.get("config_sha256"):
        raise TrainOnlyLexiconError("published formal config hash does not match its authorization")
    if provenance.get("formal_execution_policy") != FORMAL_EXECUTION_POLICY:
        raise TrainOnlyLexiconError("published formal execution policy is invalid")
    if provenance.get("llm") != _sanitize(resolved_config.get("llm_settings", {})):
        raise TrainOnlyLexiconError("formal LLM provenance disagrees with its authorized config")
    if provenance.get("web") != _sanitize(resolved_config.get("web_settings", {})):
        raise TrainOnlyLexiconError("formal web provenance disagrees with its authorized config")


def validate_lexicon_target(
    target_dir: str | Path,
    *,
    require_directory_name: bool = True,
    workspace_root: str | Path | None = None,
) -> dict[str, Any]:
    target = Path(target_dir)
    required = ("data_ref.json", "lexicon.json", "manifest.json", "provenance.json", "payload_manifest.json")
    if any(not (target / filename).is_file() for filename in required):
        raise TrainOnlyLexiconError("lexicon target is missing required payload files")
    lexicon = _load_json(target / "lexicon.json")
    manifest = _load_json(target / "manifest.json")
    provenance = _load_json(target / "provenance.json")
    if manifest.get("schema_version") == PILOT_GATED_TERMINOLOGY_MANIFEST_VERSION:
        if workspace_root is None:
            raise TrainOnlyLexiconError(
                "pilot-gated terminology publication requires workspace_root"
            )
        try:
            from build_lex.terminology_resolution import (
                validate_stage1_published_library,
            )

            return validate_stage1_published_library(
                target, workspace_root=workspace_root
            )
        except Exception as exc:
            if isinstance(exc, TrainOnlyLexiconError):
                raise
            raise TrainOnlyLexiconError(
                f"pilot-gated terminology publication is invalid: {exc}"
            ) from exc
    if lexicon.get("schema_version") != LEXICON_SCHEMA_VERSION:
        raise TrainOnlyLexiconError("unsupported lexicon schema")
    if manifest.get("schema_version") != LEXICON_MANIFEST_VERSION:
        raise TrainOnlyLexiconError("unsupported lexicon manifest schema")
    if provenance.get("schema_version") != LEXICON_PROVENANCE_VERSION:
        raise TrainOnlyLexiconError("unsupported lexicon provenance schema")
    if any(
        value.get("resource_role") != TERMINOLOGY_LIBRARY_ROLE
        for value in (lexicon, manifest, provenance)
    ):
        raise TrainOnlyLexiconError(
            "lexicon target is not a category-free terminology library"
        )
    forbidden_lexicon_paths = _terminology_forbidden_key_paths(
        lexicon, path="lexicon"
    )
    if forbidden_lexicon_paths:
        raise TrainOnlyLexiconError(
            "terminology-library target contains task fields: "
            f"{forbidden_lexicon_paths}"
        )
    build_id = manifest.get("lexicon_build_id")
    if not isinstance(build_id, str) or not LEXICON_BUILD_ID_RE.fullmatch(build_id):
        raise TrainOnlyLexiconError("invalid lexicon build ID")
    if lexicon.get("lexicon_build_id") != build_id or provenance.get("lexicon_build_id") != build_id:
        raise TrainOnlyLexiconError("lexicon build ID is inconsistent across payloads")
    expected_id = "lex-" + _sha256_bytes(_canonical_bytes(manifest.get("id_inputs")))
    if expected_id != build_id:
        raise TrainOnlyLexiconError("lexicon build ID does not match canonical ID inputs")
    id_inputs = manifest.get("id_inputs")
    if not isinstance(id_inputs, Mapping):
        raise TrainOnlyLexiconError("lexicon manifest has no canonical ID inputs")
    for key in (
        "resource_role",
        "data_build_id",
        "train_data_sha256",
        "train_ids_sha256",
        "source_train_data_sha256",
        "source_train_ids_sha256",
        "train_partition_dependency",
        "source_partition",
        "calibration_contribution_count",
        "builder_config_sha256",
        "builder_code_sha256",
        "wrapper_code_sha256",
        "protocol_code_sha256s",
        "build_policy_version",
        "terms_sha256",
        "raw_artifact_hashes",
        "raw_response_sha256",
        "web_snapshot_sha256",
        "tavily_attempts_sha256",
    ):
        if manifest.get(key) != id_inputs.get(key):
            raise TrainOnlyLexiconError(f"lexicon manifest field {key} disagrees with ID inputs")
    for key in (
        "raw_artifact_hashes",
        "raw_response_sha256",
        "web_snapshot_sha256",
        "tavily_attempts_sha256",
    ):
        if provenance.get(key) != manifest.get(key):
            raise TrainOnlyLexiconError(f"lexicon provenance field {key} disagrees with manifest")
    builder_provenance = provenance.get("builder")
    if (
        not isinstance(builder_provenance, Mapping)
        or builder_provenance.get("protocol_code_sha256s")
        != manifest.get("protocol_code_sha256s")
    ):
        raise TrainOnlyLexiconError(
            "lexicon provenance protocol source hashes disagree with manifest"
        )
    if require_directory_name and target.name != build_id:
        raise TrainOnlyLexiconError("lexicon target directory name does not match build ID")
    if manifest.get("source_split") != "train" or manifest.get("train_only_verified") is not True:
        raise TrainOnlyLexiconError("lexicon manifest is not train-only verified")
    if provenance.get("train_only_verified") is not True:
        raise TrainOnlyLexiconError("lexicon provenance is not train-only verified")
    is_formal = manifest.get("source_mode") == "data_ref+train_partition"
    if manifest.get("scientific_eligible") is not is_formal:
        raise TrainOnlyLexiconError(
            "lexicon scientific eligibility disagrees with partition lineage"
        )
    if is_formal:
        _assert_no_locator_or_absolute_path(
            manifest.get("formal_authorization"),
            label="published formal authorization",
        )
        _assert_no_locator_or_absolute_path(
            id_inputs, label="published formal lexicon ID inputs"
        )
        _assert_no_locator_or_absolute_path(
            provenance, label="published formal lexicon provenance"
        )
        if (
            manifest.get("source_partition") != "fit"
            or manifest.get("fit_only_verified") is not True
            or manifest.get("calibration_contribution_count") != 0
            or provenance.get("lexicon_source_partition") != "fit"
            or provenance.get("calibration_contribution_count") != 0
        ):
            raise TrainOnlyLexiconError(
                "formal lexicon source partition is not fit"
            )
        if not (target / "train_partition_ref.json").is_file():
            raise TrainOnlyLexiconError(
                "formal lexicon target lacks train_partition_ref.json"
            )
        for key in ("raw_response_sha256", "web_snapshot_sha256"):
            value = manifest.get(key)
            if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
                raise TrainOnlyLexiconError(f"formal lexicon manifest lacks {key}")
        resolved_web = provenance.get("web")
        if (
            isinstance(resolved_web, Mapping)
            and str(resolved_web.get("backend", "")).lower() == "search_api"
        ):
            value = manifest.get("tavily_attempts_sha256")
            if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
                raise TrainOnlyLexiconError(
                    "formal search-api lexicon manifest lacks tavily_attempts_sha256"
                )
    if manifest.get("terms_sha256") != _sha256_bytes(_canonical_bytes(lexicon.get("terms"))):
        raise TrainOnlyLexiconError("lexicon terms hash mismatch")
    _validate_entry_ids(lexicon, manifest.get("lexicon_ids", []))
    stored_payload = _load_json(target / "payload_manifest.json")
    if stored_payload != _build_payload_manifest(target):
        raise TrainOnlyLexiconError("lexicon payload manifest mismatch")

    data_ref = _load_json(target / "data_ref.json")
    if is_formal:
        if workspace_root is None:
            raise TrainOnlyLexiconError(
                "formal lexicon validation requires an explicit workspace_root"
            )
        partition_ref_value = _load_json(target / "train_partition_ref.json")
        try:
            frozen_data_ref = validate_dependency_ref(
                data_ref, expected_kind="data"
            )
            frozen_partition_ref = validate_dependency_ref(
                partition_ref_value, expected_kind="train-partition"
            )
        except TrainingArtifactError as exc:
            raise TrainOnlyLexiconError(
                "formal lexicon embeds a non-portable dependency ref"
            ) from exc
        binding = manifest.get("formal_authorization")
        if (
            not isinstance(binding, Mapping)
            or binding.get("data_dependency_ref_sha256")
            != sha256_file(target / "data_ref.json")
            or binding.get("train_partition_dependency_ref_sha256")
            != sha256_file(target / "train_partition_ref.json")
        ):
            raise TrainOnlyLexiconError(
                "formal portable dependency file content differs from authorization"
            )
        if frozen_partition_ref != manifest.get("train_partition_dependency"):
            raise TrainOnlyLexiconError(
                "formal embedded train-partition dependency differs from manifest"
            )
        train_input = _resolve_portable_train_input(
            data_dependency=frozen_data_ref,
            train_partition_dependency=frozen_partition_ref,
            workspace_root=workspace_root,
        )
    elif isinstance(data_ref, Mapping) and data_ref.get(
        "schema_version"
    ) == LOCATOR_REF_VERSION:
        train_input = _resolve_data_ref(target / "data_ref.json")
    else:
        train_input = None

    if train_input is not None:
        if train_input.data_build_id != manifest.get("data_build_id"):
            raise TrainOnlyLexiconError("lexicon data dependency ID mismatch")
        if train_input.train_data_sha256 != manifest.get("train_data_sha256"):
            raise TrainOnlyLexiconError("lexicon train data hash mismatch")
        if train_input.train_ids_sha256 != manifest.get("train_ids_sha256"):
            raise TrainOnlyLexiconError("lexicon train ID hash mismatch")
        if (
            train_input.source_train_data_sha256
            != manifest.get("source_train_data_sha256")
            or train_input.source_train_ids_sha256
            != manifest.get("source_train_ids_sha256")
            or train_input.train_partition_dependency
            != manifest.get("train_partition_dependency")
        ):
            raise TrainOnlyLexiconError(
                "lexicon fit/full-train partition lineage mismatch"
            )
        _normalize_terms(lexicon.get("terms", []), train_input)
        _audit_no_forbidden_sources(lexicon, train_input, "published lexicon")
        _audit_no_forbidden_sources(manifest, train_input, "published manifest")
        _audit_no_forbidden_sources(provenance, train_input, "published provenance")
        if is_formal:
            for key in (
                "formal_authorization",
                "formal_checkpoint_completion",
                "evidence_audit",
                "capture_artifacts",
            ):
                if manifest.get(key) != id_inputs.get(key):
                    raise TrainOnlyLexiconError(f"formal manifest field {key} disagrees with ID inputs")
                if provenance.get(key) != manifest.get(key):
                    raise TrainOnlyLexiconError(f"formal provenance field {key} disagrees with manifest")
            _validate_published_formal_authorization(
                binding=manifest.get("formal_authorization"),
                manifest=manifest,
                provenance=provenance,
                id_inputs=id_inputs,
                train_input=train_input,
            )
            raw_artifacts, raw_rows = _load_raw_artifacts(target, train_input, required=True)
            captures = _load_debug_captures(
                llm_path=target / FORMAL_CAPTURE_FILES["llm_calls"],
                search_path=target / FORMAL_CAPTURE_FILES["search_calls"],
                tavily_attempts_path=target
                / FORMAL_CAPTURE_FILES["tavily_attempts"],
                train_input=train_input,
                required=True,
            )
            capture_hashes = _capture_hashes(captures)
            capture_artifacts = _capture_artifact_hashes(target)
            evidence_audit = _validate_formal_evidence(
                config=provenance["builder"]["resolved_config"],
                train_input=train_input,
                rows_by_file=raw_rows,
                captures=captures,
            )
            if raw_artifacts != manifest.get("raw_artifact_hashes"):
                raise TrainOnlyLexiconError("published formal raw evidence hashes disagree with manifest")
            if capture_artifacts != manifest.get("capture_artifacts"):
                raise TrainOnlyLexiconError("published formal capture files disagree with manifest")
            if evidence_audit != manifest.get("evidence_audit"):
                raise TrainOnlyLexiconError("published formal evidence coverage disagrees with manifest")
            checkpoint_completion = _validated_formal_checkpoint_summary(
                manifest.get("formal_checkpoint_completion"),
                config=provenance["builder"]["resolved_config"],
                evidence_audit=evidence_audit,
            )
            if checkpoint_completion != manifest.get(
                "formal_checkpoint_completion"
            ):
                raise TrainOnlyLexiconError(
                    "published formal checkpoint completion is invalid"
                )
            for key, value in capture_hashes.items():
                if value != manifest.get(key):
                    raise TrainOnlyLexiconError(f"published formal capture hash {key} disagrees with manifest")
    elif manifest.get("source_mode") not in {"direct_records", "data_ref"}:
        raise TrainOnlyLexiconError("lexicon target has an unsupported source mode")
    return manifest


def validate_lexicon_ref(
    ref_path: str | Path,
    *,
    workspace_root: str | Path | None = None,
) -> dict[str, Any]:
    locator = _load_json(Path(ref_path))
    if locator.get("schema_version") != LOCATOR_REF_VERSION or locator.get("artifact_kind") != ARTIFACT_KIND:
        raise TrainOnlyLexiconError("locator is not a Stage-1 lexicon ref")
    build_id = locator.get("artifact_id")
    if not isinstance(build_id, str) or not LEXICON_BUILD_ID_RE.fullmatch(build_id):
        raise TrainOnlyLexiconError("invalid lexicon artifact ID in locator")
    target = Path(str(locator.get("target_path", "")))
    manifest = validate_lexicon_target(target, workspace_root=workspace_root)
    if manifest.get("lexicon_build_id") != build_id:
        raise TrainOnlyLexiconError("lexicon locator ID mismatch")
    if locator.get("payload_manifest_sha256") != sha256_file(target / "payload_manifest.json"):
        raise TrainOnlyLexiconError("lexicon locator payload hash mismatch")
    return manifest


__all__ = [
    "LEXICON_SCHEMA_VERSION",
    "LEXICON_MANIFEST_VERSION",
    "LEXICON_PROVENANCE_VERSION",
    "PILOT_GATED_TERMINOLOGY_MANIFEST_VERSION",
    "PILOT_GATED_TERMINOLOGY_PUBLICATION_POLICY",
    "TrainOnlyLexiconError",
    "FrozenTrainInput",
    "resolve_train_input",
    "build_train_only_lexicon",
    "validate_lexicon_target",
    "validate_lexicon_ref",
]
