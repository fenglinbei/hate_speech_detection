"""Stateless, content-addressed Stage 1 training schedules.

The training-evidence artifact freezes *sets* of train-only evidence.  This
module is the only place that turns those sets into per-seed, per-epoch demo
orders and source-level dropout masks.  All pseudo-random choices are SHA-256
functions of immutable coordinates; they do not depend on process state,
dataloader workers, or traversal order.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from data.context_manifest import chat_prompt_text, render_messages, text_sha256, token_count
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
    write_canonical_jsonl,
    write_locator_ref,
)


SCHEDULE_SCHEMA_VERSION = "stage1-training-schedule/v1"
SCHEDULE_RECORD_SCHEMA_VERSION = "stage1-training-schedule-record/v1"
SCHEDULE_SLOT_META_SCHEMA_VERSION = "stage1-training-schedule-slot/v1"
SCHEDULE_ARTIFACT_KIND = "training-schedule"
SCHEDULE_RNG_POLICY = "sha256-stateless-order-dropout/v1"
DEMO_ORDER_POLICY = "per-seed-epoch-query-sha256/v1"
DROP_RNG_POLICY = "sha256-model-seed-epoch-query-source/v1"
FIT_PRESENTATION_POLICY = "per-seed-epoch-query/v1"
CALIBRATION_PRESENTATION_POLICY = "calibration-epoch-1-wire/v1"
PARTITION_ARTIFACT_KIND = "train-partition"
PARTITION_CONFIG = {
    "artifact_kind": PARTITION_ARTIFACT_KIND,
    "policy": "immutable-explicit-ref/v1",
    "nominal_hash_policy": "assertion-only",
}
FIXED_PRESENTATION_CONFIG = {
    "policy": CALIBRATION_PRESENTATION_POLICY,
    "wire_epoch": 1,
    "demo_order_across_epochs": True,
    "source_mask_across_epochs": True,
}
RENDERER_REVISION = "stage1-training-evidence-renderer/v1"
OVERFLOW_POLICY = "prompt-plus-gold-plus-eos-hard-fail/v1"
SUPPORTED_ROLES = ("M_LD", "M_drop")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
MODEL_KEY_RE = re.compile(r"^(M_LD|M_drop)/seed-([1-9][0-9]*)$")


class TrainingScheduleError(TrainingArtifactError):
    """Raised when a Stage 1 schedule violates its frozen contract."""


def _ids(values: Any, *, where: str) -> list[str]:
    if not isinstance(values, list) or any(not isinstance(value, str) or not value for value in values):
        raise TrainingScheduleError(f"{where} must be an array of non-empty strings")
    if len(values) != len(set(values)):
        raise TrainingScheduleError(f"{where} contains duplicate IDs")
    return list(values)


def _query_sort_key(record: Mapping[str, Any]) -> tuple[int, Any]:
    query_id = str(record.get("query", {}).get("id", ""))
    return (0, int(query_id)) if query_id.isdigit() else (1, query_id)


def _token_ids(tokenizer: Any, text: str) -> list[int]:
    if hasattr(tokenizer, "encode"):
        values = tokenizer.encode(text, add_special_tokens=False)
    else:
        encoded = tokenizer(text, add_special_tokens=False)
        if isinstance(encoded, Mapping):
            values = encoded.get("input_ids")
        else:
            values = getattr(encoded, "input_ids", None)
    if values is None:
        raise TrainingScheduleError("tokenizer did not return input_ids")
    if values and isinstance(values[0], list):
        if len(values) != 1:
            raise TrainingScheduleError("tokenizer unexpectedly returned a batch")
        values = values[0]
    return [int(value) for value in values]


def _stateless_hex(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def stateless_demo_order(
    demo_ids: Sequence[str], *, seed: int, epoch: int, query_id: str
) -> list[str]:
    """Return a permutation shared by M_LD and M_drop for one coordinate."""

    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise TrainingScheduleError("schedule seed must be a non-negative integer")
    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch <= 0:
        raise TrainingScheduleError("schedule epochs are one-based positive integers")
    if not isinstance(query_id, str) or not query_id:
        raise TrainingScheduleError("query_id must be non-empty text")
    frozen = _ids(list(demo_ids), where="demo_ids")
    return sorted(
        frozen,
        key=lambda demo_id: (
            _stateless_hex(
                {
                    "policy": DEMO_ORDER_POLICY,
                    "seed": seed,
                    "epoch": epoch,
                    "query_id": query_id,
                    "namespace": "demo_order",
                    "demo_id": demo_id,
                }
            ),
            demo_id,
        ),
    )


def stateless_source_use(
    *, seed: int, epoch: int, query_id: str, source: str, probability: float = 0.5
) -> bool:
    """Return whether M_drop keeps one evidence source.

    Exactly p=.5 is intentionally the only supported probability.  The first
    bit of an independent namespace digest is an unbiased Bernoulli draw;
    ``True`` means keep and ``False`` means drop.
    """

    if source not in {"lexicon", "demo"}:
        raise TrainingScheduleError("dropout source must be lexicon or demo")
    if probability != 0.5:
        raise TrainingScheduleError("Stage 1 M_drop probability is frozen at exactly 0.5")
    digest = hashlib.sha256(
        canonical_json_bytes(
            {
                "policy": DROP_RNG_POLICY,
                "model_role": "M_drop",
                "seed": seed,
                "epoch": epoch,
                "query_id": query_id,
                "namespace": f"drop_{source}",
            }
        )
    ).digest()
    return bool(digest[0] & 0x80)


def _evidence_items(
    section: Mapping[str, Any], *, kind: str
) -> tuple[list[str], dict[str, dict[str, Any]]]:
    id_key = f"{kind}_id"
    ids = _ids(section.get("ids"), where=f"{kind}_evidence.ids")
    items = section.get("items")
    if not isinstance(items, list):
        raise TrainingScheduleError(f"{kind}_evidence.items must be an array")
    by_id: dict[str, dict[str, Any]] = {}
    for item in items:
        if not isinstance(item, Mapping):
            raise TrainingScheduleError(f"{kind} evidence item must be an object")
        item_id = item.get(id_key)
        block = item.get("rendered_block")
        block_hash = item.get("rendered_block_sha256")
        if not isinstance(item_id, str) or not item_id or item_id in by_id:
            raise TrainingScheduleError(f"invalid or duplicate {kind} evidence ID")
        if not isinstance(block, str) or not isinstance(block_hash, str):
            raise TrainingScheduleError(f"{kind} evidence {item_id} lacks a rendered block")
        if text_sha256(block) != block_hash:
            raise TrainingScheduleError(f"{kind} evidence block hash mismatch: {item_id}")
        by_id[item_id] = dict(item)
    if ids != sorted(ids) or list(by_id) != sorted(by_id) or ids != list(by_id):
        raise TrainingScheduleError(
            f"{kind} evidence must use matching canonical ID/item set encoding"
        )
    return ids, by_id


def validate_evidence_record_for_schedule(record: Mapping[str, Any]) -> None:
    if record.get("schema_version") != "stage1-training-evidence-record/v1":
        raise TrainingScheduleError("unsupported training evidence record schema")
    query = record.get("query")
    if not isinstance(query, Mapping):
        raise TrainingScheduleError("training evidence record lacks query")
    for key in ("id", "content", "gold_text"):
        if not isinstance(query.get(key), str) or (key != "gold_text" and not query[key]):
            raise TrainingScheduleError(f"training evidence query.{key} is invalid")
    if query.get("partition") not in {"fit", "calibration"}:
        raise TrainingScheduleError(
            "training evidence query.partition must be fit or calibration"
        )
    if not isinstance(query.get("actual_gold_tokens"), int) or isinstance(
        query.get("actual_gold_tokens"), bool
    ) or query["actual_gold_tokens"] < 0:
        raise TrainingScheduleError("query.actual_gold_tokens must be a non-negative integer")
    _evidence_items(record.get("lexicon_evidence", {}), kind="lexicon")
    _evidence_items(record.get("demo_evidence", {}), kind="demo")
    rendering = record.get("rendering")
    if not isinstance(rendering, Mapping):
        raise TrainingScheduleError("evidence record lacks rendering contract")
    if not isinstance(rendering.get("system_prompt"), str):
        raise TrainingScheduleError("rendering.system_prompt must be text")
    template = rendering.get("user_prompt_template")
    if not isinstance(template, str) or not all(
        marker in template for marker in ("{lexicons}", "{examples}", "{text}")
    ):
        raise TrainingScheduleError("rendering.user_prompt_template is invalid")
    if rendering.get("thinking_mode") is not False:
        raise TrainingScheduleError("Stage 1 training must disable thinking mode")
    budget = record.get("budget")
    if not isinstance(budget, Mapping):
        raise TrainingScheduleError("evidence record lacks budget")
    maximum = budget.get("max_sequence_tokens")
    reserve = budget.get("completion_reserve_tokens")
    eos_tokens = budget.get("eos_tokens")
    if (
        not isinstance(maximum, int)
        or isinstance(maximum, bool)
        or not isinstance(reserve, int)
        or isinstance(reserve, bool)
        or not 0 < reserve < maximum
        or eos_tokens != 1
    ):
        raise TrainingScheduleError("evidence record has an invalid sequence budget")
    if budget.get("overflow_policy") != OVERFLOW_POLICY:
        raise TrainingScheduleError("evidence overflow policy is not fail-closed")
    recorded_hash = record.get("record_sha256")
    if not isinstance(recorded_hash, str) or not SHA256_RE.fullmatch(recorded_hash):
        raise TrainingScheduleError("evidence record lacks record_sha256")
    # Evidence record hashes intentionally exclude the parent build ID so the
    # evidence artifact ID can bind semantic records without a hash cycle.
    actual_hash = canonical_sha256(
        {
            key: value
            for key, value in record.items()
            if key not in {"training_evidence_build_id", "record_sha256"}
        }
    )
    if actual_hash != recorded_hash:
        raise TrainingScheduleError("training evidence record hash mismatch")


def _slot_contract(slot: Mapping[str, Any]) -> dict[str, Any]:
    model_key = slot.get("model_key")
    match = MODEL_KEY_RE.fullmatch(str(model_key))
    if match is None:
        raise TrainingScheduleError(f"invalid formal model_key: {model_key!r}")
    role = slot.get("role")
    seed = slot.get("seed")
    if role != match.group(1) or seed != int(match.group(2)):
        raise TrainingScheduleError(f"model key/role/seed mismatch for {model_key}")
    if slot.get("training_required") is not True:
        raise TrainingScheduleError(f"formal slot {model_key} must require training")
    epochs = slot.get("epochs")
    if not isinstance(epochs, int) or isinstance(epochs, bool) or epochs <= 0:
        raise TrainingScheduleError(f"formal slot {model_key} has invalid epochs")
    config = slot.get("train_config_resolved")
    if not isinstance(config, Mapping):
        raise TrainingScheduleError(f"formal slot {model_key} lacks resolved train config")
    if config.get("model_role") != role:
        raise TrainingScheduleError(f"slot {model_key} train config role mismatch")
    policy = config.get("context_policy")
    if not isinstance(policy, Mapping):
        raise TrainingScheduleError(f"slot {model_key} lacks context policy")
    data = config.get("data")
    if not isinstance(data, Mapping):
        raise TrainingScheduleError(f"slot {model_key} lacks data policy")
    if data.get("partition") != PARTITION_CONFIG:
        raise TrainingScheduleError(
            f"slot {model_key} does not consume an immutable train partition"
        )
    if data.get("fixed_presentation") != FIXED_PRESENTATION_CONFIG:
        raise TrainingScheduleError(
            f"slot {model_key} does not freeze calibration presentation"
        )
    expected_probability = 0.0 if role == "M_LD" else 0.5
    if policy.get("lexicon_dropout_probability") != expected_probability:
        raise TrainingScheduleError(f"slot {model_key} lexicon dropout mismatch")
    if policy.get("demonstration_dropout_probability") != expected_probability:
        raise TrainingScheduleError(f"slot {model_key} demo dropout mismatch")
    if policy.get("dropout_independent") is not True:
        raise TrainingScheduleError(f"slot {model_key} requires independent source dropout")
    if policy.get("order_policy") != DEMO_ORDER_POLICY:
        raise TrainingScheduleError(f"slot {model_key} order policy mismatch")
    if policy.get("matched_permutation_across_roles") is not True:
        raise TrainingScheduleError(f"slot {model_key} must match orders across roles")
    if role == "M_drop" and policy.get("dropout_rng_policy") != DROP_RNG_POLICY:
        raise TrainingScheduleError(f"slot {model_key} dropout RNG policy mismatch")
    return dict(slot)


def validate_plan_slots_for_schedule(plan: Mapping[str, Any]) -> list[dict[str, Any]]:
    if plan.get("schema_version") != "stage1-training-plan/v1":
        raise TrainingScheduleError("unsupported training plan schema")
    if plan.get("scope") != "formal" or plan.get("scientific_eligible") is not True:
        raise TrainingScheduleError("formal training schedule requires a scientific formal plan")
    raw_slots = plan.get("ordered_model_slots")
    if not isinstance(raw_slots, list) or not raw_slots:
        raise TrainingScheduleError("training plan has no ordered formal slots")
    slots = [_slot_contract(slot) for slot in raw_slots]
    keys = [slot["model_key"] for slot in slots]
    if len(keys) != len(set(keys)):
        raise TrainingScheduleError("training plan contains duplicate model keys")
    by_seed: dict[int, dict[str, dict[str, Any]]] = {}
    for slot in slots:
        by_seed.setdefault(slot["seed"], {})[slot["role"]] = slot
    for seed, roles in by_seed.items():
        if set(roles) != set(SUPPORTED_ROLES):
            raise TrainingScheduleError(f"seed {seed} must contain exactly M_LD and M_drop")
        if roles["M_LD"]["epochs"] != roles["M_drop"]["epochs"]:
            raise TrainingScheduleError(f"seed {seed} roles must use the same epoch count")
    rng = plan.get("order_dropout_rng_policy")
    if not isinstance(rng, Mapping):
        raise TrainingScheduleError("training plan lacks RNG policy")
    if rng.get("order") != DEMO_ORDER_POLICY or rng.get("matched_permutation_across_roles") is not True:
        raise TrainingScheduleError("training plan demo-order policy is invalid")
    if rng.get("fit_presentation") != FIT_PRESENTATION_POLICY:
        raise TrainingScheduleError("training plan fit presentation policy is invalid")
    if rng.get("calibration_presentation") != FIXED_PRESENTATION_CONFIG:
        raise TrainingScheduleError(
            "training plan calibration presentation policy is invalid"
        )
    drop = rng.get("m_drop")
    if not isinstance(drop, Mapping) or drop.get("lexicon_probability") != 0.5 or drop.get(
        "demonstration_probability"
    ) != 0.5 or drop.get("independent_sources") is not True or drop.get("rng") != DROP_RNG_POLICY:
        raise TrainingScheduleError("training plan M_drop policy is invalid")
    return slots


def build_schedule_record(
    evidence_record: Mapping[str, Any],
    slot: Mapping[str, Any],
    *,
    epoch: int,
    query_ordinal: int,
    schedule_build_id: str,
    tokenizer: Any,
) -> dict[str, Any]:
    """Materialize and fully preflight one immutable schedule row."""

    validate_evidence_record_for_schedule(evidence_record)
    frozen_slot = _slot_contract(slot)
    query = evidence_record["query"]
    query_id = query["id"]
    partition = query["partition"]
    presentation_epoch = 1 if partition == "calibration" else epoch
    lexicon_ids, lexicon_items = _evidence_items(
        evidence_record["lexicon_evidence"], kind="lexicon"
    )
    demo_ids, demo_items = _evidence_items(evidence_record["demo_evidence"], kind="demo")
    ordered_demo_ids = stateless_demo_order(
        demo_ids,
        seed=frozen_slot["seed"],
        epoch=presentation_epoch,
        query_id=query_id,
    )
    if frozen_slot["role"] == "M_LD":
        use_lexicon = True
        use_demos = True
    else:
        use_lexicon = stateless_source_use(
            seed=frozen_slot["seed"],
            epoch=presentation_epoch,
            query_id=query_id,
            source="lexicon",
        )
        use_demos = stateless_source_use(
            seed=frozen_slot["seed"],
            epoch=presentation_epoch,
            query_id=query_id,
            source="demo",
        )

    active_lexicons = lexicon_ids if use_lexicon else []
    active_demos = ordered_demo_ids if use_demos else []
    rendering = evidence_record["rendering"]
    messages = render_messages(
        query_content=query["content"],
        lexicon_ids=active_lexicons,
        demo_ids=active_demos,
        lexicon_catalog=lexicon_items,
        demo_catalog=demo_items,
        system_prompt=rendering["system_prompt"],
        user_prompt_template=rendering["user_prompt_template"],
    )
    prompt_text = chat_prompt_text(messages, tokenizer)
    prompt_tokens = token_count(prompt_text, tokenizer)
    gold_ids = _token_ids(tokenizer, query["gold_text"])
    if len(gold_ids) != query["actual_gold_tokens"]:
        raise TrainingScheduleError(
            f"query {query_id} gold token count differs from frozen training evidence"
        )
    if getattr(tokenizer, "eos_token_id", None) is None:
        raise TrainingScheduleError("schedule tokenizer must define eos_token_id")
    sequence_tokens = prompt_tokens + len(gold_ids) + 1
    maximum = evidence_record["budget"]["max_sequence_tokens"]
    if sequence_tokens > maximum:
        raise TrainingScheduleError(
            f"schedule overflow for {frozen_slot['model_key']} epoch={epoch} "
            f"query={query_id}: {sequence_tokens}>{maximum}"
        )
    row = {
        "schema_version": SCHEDULE_RECORD_SCHEMA_VERSION,
        "schedule_build_id": schedule_build_id,
        "training_evidence_record_sha256": evidence_record["record_sha256"],
        "model_key": frozen_slot["model_key"],
        "role": frozen_slot["role"],
        "seed": frozen_slot["seed"],
        "epoch": epoch,
        "partition": partition,
        "presentation_epoch": presentation_epoch,
        "query_ordinal": query_ordinal,
        "query_id": query_id,
        "ordered_lexicon_ids": lexicon_ids,
        "ordered_demo_ids": ordered_demo_ids,
        "use_lexicon": use_lexicon,
        "use_demos": use_demos,
        "instruction": messages[0]["content"],
        "input": messages[1]["content"],
        "output": query["gold_text"],
        "content": query["content"],
        "rendered_prompt_sha256": text_sha256(prompt_text),
        "rendered_prompt_tokens": prompt_tokens,
        "actual_gold_tokens": len(gold_ids),
        "gold_plus_eos_tokens": len(gold_ids) + 1,
        "sequence_tokens": sequence_tokens,
        "max_sequence_tokens": maximum,
        "overflow_policy": OVERFLOW_POLICY,
    }
    row["record_sha256"] = canonical_sha256(row)
    return row


def _validate_schedule_record_wire(row: Mapping[str, Any]) -> None:
    if row.get("schema_version") != SCHEDULE_RECORD_SCHEMA_VERSION:
        raise TrainingScheduleError("unsupported schedule record schema")
    recorded = row.get("record_sha256")
    if not isinstance(recorded, str) or not SHA256_RE.fullmatch(recorded):
        raise TrainingScheduleError("schedule record lacks a valid record hash")
    actual = canonical_sha256({key: value for key, value in row.items() if key != "record_sha256"})
    if recorded != actual:
        raise TrainingScheduleError("schedule record hash mismatch")
    for key in ("rendered_prompt_tokens", "actual_gold_tokens", "gold_plus_eos_tokens", "sequence_tokens"):
        if not isinstance(row.get(key), int) or isinstance(row.get(key), bool) or row[key] < 0:
            raise TrainingScheduleError(f"schedule record {key} is invalid")
    if row.get("gold_plus_eos_tokens") != row.get("actual_gold_tokens") + 1:
        raise TrainingScheduleError("schedule gold+EOS count mismatch")
    if row.get("sequence_tokens") != row.get("rendered_prompt_tokens") + row.get(
        "gold_plus_eos_tokens"
    ):
        raise TrainingScheduleError("schedule sequence token arithmetic mismatch")
    if row.get("sequence_tokens") > row.get("max_sequence_tokens", -1):
        raise TrainingScheduleError("schedule record exceeds max_sequence_tokens")
    if row.get("overflow_policy") != OVERFLOW_POLICY:
        raise TrainingScheduleError("schedule record overflow policy mismatch")
    partition = row.get("partition")
    if partition not in {"fit", "calibration"}:
        raise TrainingScheduleError("schedule record partition is invalid")
    expected_presentation_epoch = 1 if partition == "calibration" else row.get("epoch")
    if row.get("presentation_epoch") != expected_presentation_epoch:
        raise TrainingScheduleError("schedule fixed-presentation epoch is invalid")


def _records_digest(rows: Sequence[Mapping[str, Any]]) -> str:
    return canonical_sha256([row["record_sha256"] for row in rows])


def _infer_workspace_root(container: Path, dependency: Mapping[str, Any]) -> Path:
    frozen = validate_dependency_ref(dependency)
    logical = Path(frozen["logical_repo_path"])
    for candidate in (container, *container.parents):
        if (candidate / logical).resolve().is_dir():
            try:
                resolve_dependency_target(frozen, candidate)
            except TrainingArtifactError:
                continue
            return candidate.resolve()
    raise TrainingScheduleError(
        f"cannot resolve portable dependency {frozen['logical_repo_path']} from {container}"
    )


def _load_evidence_target(
    target: Path,
    dependency: Mapping[str, Any],
    *,
    workspace_root: Path,
    tokenizer: Any | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    from data.training_evidence import validate_training_evidence_target

    if validate_payload_manifest(target) != dependency["payload_manifest_sha256"]:
        raise TrainingScheduleError("training evidence dependency payload mismatch")
    meta, records = validate_training_evidence_target(
        target,
        workspace_root=workspace_root,
        tokenizer=tokenizer,
    )
    if meta.get("training_evidence_build_id") != dependency["artifact_id"]:
        raise TrainingScheduleError("training evidence artifact ID mismatch")
    for record in records:
        validate_evidence_record_for_schedule(record)
        if record.get("training_evidence_build_id") != dependency["artifact_id"]:
            raise TrainingScheduleError("evidence record belongs to another artifact")
    records = sorted(records, key=_query_sort_key)
    if len(records) != meta.get("record_count"):
        raise TrainingScheduleError("training evidence record count mismatch")
    if len({record["query"]["id"] for record in records}) != len(records):
        raise TrainingScheduleError("training evidence contains duplicate query IDs")
    return meta, records


def tokenizer_revision_from_directory(tokenizer_root: str | Path) -> str:
    """Hash tokenizer/config files without reading multi-gigabyte model shards."""

    root = Path(tokenizer_root)
    if not root.is_dir() or root.is_symlink():
        raise TrainingScheduleError(f"invalid tokenizer root: {root}")
    names = {
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
        "special_tokens_map.json",
        "added_tokens.json",
        "chat_template.jinja",
        "sentencepiece.bpe.model",
        "spiece.model",
    }
    files = [path for path in sorted(root.iterdir()) if path.is_file() and path.name in names]
    if not any(path.name.startswith("tokenizer") for path in files):
        raise TrainingScheduleError("tokenizer root lacks tokenizer files")
    inventory = [
        {"path": path.name, "size": path.stat().st_size, "sha256": sha256_file(path)}
        for path in files
    ]
    return "tok-" + canonical_sha256({"schema_version": "stage1-tokenizer-revision/v1", "files": inventory})


def _schedule_id_inputs(
    *,
    training_plan_dependency: Mapping[str, Any],
    training_evidence_dependency: Mapping[str, Any],
    train_partition_dependency: Mapping[str, Any],
    renderer_revision: str,
    tokenizer_revision: str,
    schedule_builder_code_sha256: str,
) -> dict[str, Any]:
    return {
        "training_plan_dependency": dict(training_plan_dependency),
        "training_evidence_dependency": dict(training_evidence_dependency),
        "train_partition_dependency": dict(train_partition_dependency),
        "schedule_schema_version": SCHEDULE_SCHEMA_VERSION,
        "renderer_revision": renderer_revision,
        "tokenizer_revision": tokenizer_revision,
        "schedule_builder_code_sha256": schedule_builder_code_sha256,
    }


def _resolved_schedule_config(tokenizer_revision: str) -> dict[str, Any]:
    return {
        "schema_version": "stage1-training-schedule-config/v1",
        "rng_policy": SCHEDULE_RNG_POLICY,
        "demo_order_policy": DEMO_ORDER_POLICY,
        "dropout_rng_policy": DROP_RNG_POLICY,
        "fit_presentation_policy": FIT_PRESENTATION_POLICY,
        "calibration_presentation": copy.deepcopy(FIXED_PRESENTATION_CONFIG),
        "m_ld": {"use_lexicon": True, "use_demos": True},
        "m_drop": {
            "lexicon_probability": 0.5,
            "demonstration_probability": 0.5,
            "independent_sources": True,
            "budget_reallocation": False,
        },
        "overflow_policy": OVERFLOW_POLICY,
        "renderer_revision": RENDERER_REVISION,
        "tokenizer_revision": tokenizer_revision,
    }


def _build_training_schedule_with_tokenizer(
    *,
    training_plan_ref: str | Path,
    write_ref: str | Path,
    tokenizer: Any,
    tokenizer_revision: str | None = None,
    training_evidence_ref: str | Path | None = None,
    train_partition_ref: str | Path | None = None,
    artifact_root: str | Path | None = None,
    workspace_root: str | Path | None = None,
) -> dict[str, Any]:
    """Build and atomically publish the complete formal slot×epoch registry."""

    from data.training_plan import load_training_plan

    root_assertion = (
        Path(workspace_root).resolve()
        if workspace_root is not None
        else Path(__file__).resolve().parents[2]
    )
    plan_locator, plan_target, plan = load_training_plan(
        training_plan_ref,
        workspace_root=root_assertion,
        tokenizer=tokenizer,
    )
    slots = validate_plan_slots_for_schedule(plan)
    evidence_dependency = validate_dependency_ref(
        plan.get("training_evidence_dependency", {}), expected_kind="training-evidence"
    )
    partition_dependency = validate_dependency_ref(
        plan.get("train_partition_dependency", {}),
        expected_kind=PARTITION_ARTIFACT_KIND,
    )
    inferred_root = _infer_workspace_root(plan_target, evidence_dependency)
    if inferred_root != root_assertion:
        raise TrainingScheduleError("training plan dependency root differs from workspace_root")
    evidence_target = resolve_dependency_target(evidence_dependency, root_assertion)
    partition_target = resolve_dependency_target(
        partition_dependency, root_assertion
    )
    if training_evidence_ref is not None:
        evidence_locator, explicit_target = resolve_locator_ref(
            training_evidence_ref, "training-evidence"
        )
        explicit_dependency = portable_dependency(evidence_locator, explicit_target, root_assertion)
        if explicit_dependency != evidence_dependency or explicit_target.resolve() != evidence_target:
            raise TrainingScheduleError("explicit evidence ref does not match the training plan")
    if train_partition_ref is not None:
        partition_locator, explicit_partition_target = resolve_locator_ref(
            train_partition_ref, PARTITION_ARTIFACT_KIND
        )
        explicit_partition_dependency = portable_dependency(
            partition_locator, explicit_partition_target, root_assertion
        )
        if (
            explicit_partition_dependency != partition_dependency
            or explicit_partition_target.resolve() != partition_target
        ):
            raise TrainingScheduleError(
                "explicit partition ref does not match the training plan"
            )
    evidence_meta, evidence_records = _load_evidence_target(
        evidence_target,
        evidence_dependency,
        workspace_root=root_assertion,
        tokenizer=tokenizer,
    )
    if evidence_meta.get("train_partition_dependency") != partition_dependency:
        raise TrainingScheduleError(
            "training evidence and plan point to different train partitions"
        )
    if evidence_meta.get("base_model_dependency") != plan.get(
        "base_model_dependency"
    ):
        raise TrainingScheduleError(
            "training evidence and plan point to different base models"
        )
    if evidence_meta.get("renderer_revision") != RENDERER_REVISION:
        raise TrainingScheduleError("training evidence renderer revision is unsupported")
    frozen_tokenizer_revision = evidence_meta.get("tokenizer_revision")
    if not isinstance(frozen_tokenizer_revision, str) or not frozen_tokenizer_revision:
        raise TrainingScheduleError("training evidence lacks tokenizer revision")
    if tokenizer_revision is not None and frozen_tokenizer_revision != tokenizer_revision:
        raise TrainingScheduleError("tokenizer revision differs from training evidence")
    tokenizer_revision = frozen_tokenizer_revision
    plan_dependency = portable_dependency(plan_locator, plan_target, root_assertion)
    builder_hash = sha256_file(__file__)
    id_inputs = _schedule_id_inputs(
        training_plan_dependency=plan_dependency,
        training_evidence_dependency=evidence_dependency,
        train_partition_dependency=partition_dependency,
        renderer_revision=RENDERER_REVISION,
        tokenizer_revision=tokenizer_revision,
        schedule_builder_code_sha256=builder_hash,
    )
    schedule_build_id = "sch-" + canonical_sha256(id_inputs)
    resolved = _resolved_schedule_config(tokenizer_revision)

    root = Path(artifact_root).resolve() if artifact_root is not None else evidence_target.parent.parent
    target_parent = root / "training_schedules"
    target = target_parent / schedule_build_id

    # The lifecycle ID is fully determined before materialising 173,430
    # production rows.  For an idempotent rebuild, re-render and deeply
    # validate the existing immutable target with the caller's tokenizer
    # instead of first creating an equally large throw-away staging tree.
    # This still detects a tokenizer implementation drifting behind the same
    # declared revision because every stored row is reconstructed below.
    if target.exists():
        _validate_schedule_target(
            target,
            workspace_root=root_assertion,
            tokenizer=tokenizer,
            require_directory_name=True,
            _source_lease_held=True,
        )
        payload_hash = validate_payload_manifest(target)
        return write_locator_ref(
            write_ref,
            artifact_kind=SCHEDULE_ARTIFACT_KIND,
            artifact_id=schedule_build_id,
            target=target,
            payload_manifest_sha256=payload_hash,
        )

    staging = new_staging_directory(target_parent, schedule_build_id)
    slot_registry: list[dict[str, Any]] = []
    all_record_hashes: list[str] = []
    try:
        write_canonical_json(staging / "config.resolved.json", resolved)
        write_canonical_json(staging / "training_plan_ref.json", plan_dependency)
        write_canonical_json(staging / "training_evidence_ref.json", evidence_dependency)
        write_canonical_json(staging / "train_partition_ref.json", partition_dependency)
        for slot in slots:
            slot_directory = Path("schedules") / slot["role"] / f"seed-{slot['seed']}"
            epoch_registry: list[dict[str, Any]] = []
            for epoch in range(1, slot["epochs"] + 1):
                rows = [
                    build_schedule_record(
                        evidence_record,
                        slot,
                        epoch=epoch,
                        query_ordinal=query_ordinal,
                        schedule_build_id=schedule_build_id,
                        tokenizer=tokenizer,
                    )
                    for query_ordinal, evidence_record in enumerate(evidence_records)
                ]
                relative_path = slot_directory / f"epoch-{epoch}.jsonl"
                write_canonical_jsonl(
                    staging / relative_path, rows, key="query_ordinal", numeric_key=True
                )
                digest = _records_digest(rows)
                epoch_registry.append(
                    {
                        "epoch": epoch,
                        "relative_path": relative_path.as_posix(),
                        "record_count": len(rows),
                        "records_sha256": digest,
                    }
                )
                all_record_hashes.extend(row["record_sha256"] for row in rows)
            slot_meta = {
                "schema_version": SCHEDULE_SLOT_META_SCHEMA_VERSION,
                "schedule_build_id": schedule_build_id,
                "model_key": slot["model_key"],
                "role": slot["role"],
                "seed": slot["seed"],
                "epochs": slot["epochs"],
                "query_count_per_epoch": len(evidence_records),
                "epoch_registry": epoch_registry,
                "slot_records_sha256": canonical_sha256(
                    [entry["records_sha256"] for entry in epoch_registry]
                ),
            }
            write_canonical_json(staging / slot_directory / "schedule.meta.json", slot_meta)
            slot_registry.append(
                {
                    "model_key": slot["model_key"],
                    "role": slot["role"],
                    "seed": slot["seed"],
                    "epochs": slot["epochs"],
                    "relative_meta_path": (slot_directory / "schedule.meta.json").as_posix(),
                    "slot_records_sha256": slot_meta["slot_records_sha256"],
                }
            )
        meta = {
            "schema_version": SCHEDULE_SCHEMA_VERSION,
            "schedule_build_id": schedule_build_id,
            "training_plan_dependency": plan_dependency,
            "training_evidence_dependency": evidence_dependency,
            "train_partition_dependency": partition_dependency,
            "renderer_revision": RENDERER_REVISION,
            "tokenizer_revision": tokenizer_revision,
            "rng_policy": SCHEDULE_RNG_POLICY,
            "slot_registry": slot_registry,
            "total_record_count": len(all_record_hashes),
            "all_schedule_records_sha256": canonical_sha256(all_record_hashes),
            "preflight_status": "complete-pass",
            "id_inputs": id_inputs,
        }
        write_canonical_json(staging / "schedule.meta.json", meta)
        write_canonical_json(
            staging / "provenance.json",
            {
                "schema_version": "stage1-training-schedule-provenance/v1",
                "schedule_build_id": schedule_build_id,
                "schedule_builder_code_sha256": builder_hash,
                "id_inputs": id_inputs,
            },
        )
        payload_hash = finalize_target_atomic(
            staging,
            target,
            validate_staging=lambda directory: _validate_schedule_target(
                directory,
                workspace_root=root_assertion,
                tokenizer=tokenizer,
                require_directory_name=False,
                _source_lease_held=True,
            ),
        )
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return write_locator_ref(
        write_ref,
        artifact_kind=SCHEDULE_ARTIFACT_KIND,
        artifact_id=schedule_build_id,
        target=target,
        payload_manifest_sha256=payload_hash,
    )


def build_training_schedule(
    *,
    training_plan_ref: str | Path,
    write_ref: str | Path,
    tokenizer: Any | None = None,
    tokenizer_revision: str | None = None,
    training_evidence_ref: str | Path | None = None,
    train_partition_ref: str | Path | None = None,
    artifact_root: str | Path | None = None,
    workspace_root: str | Path | None = None,
) -> dict[str, Any]:
    """Build a schedule while holding the evidence tokenizer source lease."""

    root = (
        Path(workspace_root).resolve()
        if workspace_root is not None
        else Path(__file__).resolve().parents[2]
    )
    _plan_locator, plan_target = resolve_locator_ref(
        training_plan_ref, expected_kind="training-plan"
    )
    plan = load_json(plan_target / "plan.resolved.json")
    if not isinstance(plan, Mapping):
        raise TrainingScheduleError("training plan payload is not an object")
    evidence_dependency = validate_dependency_ref(
        plan.get("training_evidence_dependency", {}),
        expected_kind="training-evidence",
    )
    evidence_target = resolve_dependency_target(evidence_dependency, root)
    from data.training_evidence import frozen_evidence_tokenizer_lease

    with frozen_evidence_tokenizer_lease(
        evidence_target,
        workspace_root=root,
        tokenizer=tokenizer,
    ) as tokenizer_object:
        return _build_training_schedule_with_tokenizer(
            training_plan_ref=training_plan_ref,
            write_ref=write_ref,
            tokenizer=tokenizer_object,
            tokenizer_revision=tokenizer_revision,
            training_evidence_ref=training_evidence_ref,
            train_partition_ref=train_partition_ref,
            artifact_root=artifact_root,
            workspace_root=root,
        )


def _validate_schedule_target(
    target: Path,
    *,
    workspace_root: Path,
    tokenizer: Any | None,
    require_directory_name: bool,
    _source_lease_held: bool = False,
) -> dict[str, Any]:
    if not _source_lease_held:
        validate_payload_manifest(target)
        bootstrap_meta = load_json(target / "schedule.meta.json")
        if not isinstance(bootstrap_meta, Mapping):
            raise TrainingScheduleError("schedule.meta.json must be an object")
        bootstrap_dependency = validate_dependency_ref(
            bootstrap_meta.get("training_evidence_dependency", {}),
            expected_kind="training-evidence",
        )
        bootstrap_evidence_target = resolve_dependency_target(
            bootstrap_dependency, workspace_root
        )
        from data.training_evidence import frozen_evidence_tokenizer_lease

        with frozen_evidence_tokenizer_lease(
            bootstrap_evidence_target,
            workspace_root=workspace_root,
            tokenizer=tokenizer,
        ) as tokenizer_object:
            return _validate_schedule_target(
                target,
                workspace_root=workspace_root,
                tokenizer=tokenizer_object,
                require_directory_name=require_directory_name,
                _source_lease_held=True,
            )
    validate_payload_manifest(target)
    meta = load_json(target / "schedule.meta.json")
    if not isinstance(meta, dict):
        raise TrainingScheduleError("schedule.meta.json must be an object")
    schema_path = Path(__file__).resolve().parents[2] / "schemas" / "stage1_training_schedule_v1.schema.json"
    validate_json_schema(meta, schema_path)
    schedule_id = meta["schedule_build_id"]
    if require_directory_name and target.name != schedule_id:
        raise TrainingScheduleError("schedule ID does not match target directory")
    expected_id = "sch-" + canonical_sha256(meta["id_inputs"])
    if expected_id != schedule_id:
        raise TrainingScheduleError("schedule build ID cannot be recomputed")
    plan_dependency = validate_dependency_ref(
        meta["training_plan_dependency"], expected_kind="training-plan"
    )
    evidence_dependency = validate_dependency_ref(
        meta["training_evidence_dependency"], expected_kind="training-evidence"
    )
    partition_dependency = validate_dependency_ref(
        meta["train_partition_dependency"], expected_kind=PARTITION_ARTIFACT_KIND
    )
    if meta["id_inputs"] != _schedule_id_inputs(
        training_plan_dependency=plan_dependency,
        training_evidence_dependency=evidence_dependency,
        train_partition_dependency=partition_dependency,
        renderer_revision=meta["renderer_revision"],
        tokenizer_revision=meta["tokenizer_revision"],
        schedule_builder_code_sha256=meta["id_inputs"]["schedule_builder_code_sha256"],
    ):
        raise TrainingScheduleError("schedule ID inputs are not canonical")
    if load_json(target / "config.resolved.json") != _resolved_schedule_config(
        meta["tokenizer_revision"]
    ):
        raise TrainingScheduleError("resolved schedule config mismatch")
    if load_json(target / "provenance.json") != {
        "schema_version": "stage1-training-schedule-provenance/v1",
        "schedule_build_id": schedule_id,
        "schedule_builder_code_sha256": meta["id_inputs"][
            "schedule_builder_code_sha256"
        ],
        "id_inputs": meta["id_inputs"],
    }:
        raise TrainingScheduleError("schedule provenance mismatch")
    if load_json(target / "training_plan_ref.json") != plan_dependency:
        raise TrainingScheduleError("schedule training_plan_ref.json mismatch")
    if load_json(target / "training_evidence_ref.json") != evidence_dependency:
        raise TrainingScheduleError("schedule training_evidence_ref.json mismatch")
    if load_json(target / "train_partition_ref.json") != partition_dependency:
        raise TrainingScheduleError("schedule train_partition_ref.json mismatch")
    from data.training_plan import validate_training_plan_target

    plan_target = resolve_dependency_target(plan_dependency, workspace_root)
    evidence_target = resolve_dependency_target(evidence_dependency, workspace_root)
    partition_target = resolve_dependency_target(
        partition_dependency, workspace_root
    )
    tokenizer_object = tokenizer
    plan = validate_training_plan_target(
        plan_target,
        workspace_root=workspace_root,
        tokenizer=tokenizer_object,
    )
    slots = validate_plan_slots_for_schedule(plan)
    if plan.get("training_evidence_dependency") != evidence_dependency:
        raise TrainingScheduleError("schedule evidence does not match training plan")
    if plan.get("train_partition_dependency") != partition_dependency:
        raise TrainingScheduleError("schedule partition does not match training plan")
    evidence_meta, evidence_records = _load_evidence_target(
        evidence_target,
        evidence_dependency,
        workspace_root=workspace_root,
        tokenizer=tokenizer_object,
    )
    if evidence_meta.get("renderer_revision") != meta["renderer_revision"]:
        raise TrainingScheduleError("schedule/evidence renderer revision mismatch")
    if evidence_meta.get("tokenizer_revision") != meta["tokenizer_revision"]:
        raise TrainingScheduleError("schedule/evidence tokenizer revision mismatch")
    if evidence_meta.get("train_partition_dependency") != partition_dependency:
        raise TrainingScheduleError("schedule/evidence partition lineage mismatch")
    if evidence_meta.get("base_model_dependency") != plan.get(
        "base_model_dependency"
    ):
        raise TrainingScheduleError("schedule/evidence base-model lineage mismatch")
    from data.train_partition import validate_train_partition_target

    partition_report = validate_train_partition_target(
        partition_target, workspace_root=workspace_root
    )
    if partition_report.get("partition_dependency") != partition_dependency:
        raise TrainingScheduleError(
            "schedule partition dependency cannot be reproduced"
        )
    partition_rows = load_jsonl(partition_target / "partition.jsonl")
    partition_by_id = {row["query_id"]: row for row in partition_rows}
    evidence_ids = [record["query"]["id"] for record in evidence_records]
    if evidence_ids != list(partition_by_id):
        raise TrainingScheduleError(
            "schedule evidence registry differs from the train partition"
        )

    expected_registry: list[dict[str, Any]] = []
    expected_files = {
        "config.resolved.json",
        "training_plan_ref.json",
        "training_evidence_ref.json",
        "train_partition_ref.json",
        "schedule.meta.json",
        "provenance.json",
        "payload_manifest.json",
    }
    all_hashes: list[str] = []
    paired_orders: dict[tuple[int, int, str], list[str]] = {}
    for slot in slots:
        slot_directory = Path("schedules") / slot["role"] / f"seed-{slot['seed']}"
        slot_meta_path = target / slot_directory / "schedule.meta.json"
        expected_files.add((slot_directory / "schedule.meta.json").as_posix())
        slot_meta = load_json(slot_meta_path)
        if not isinstance(slot_meta, dict) or slot_meta.get("schema_version") != SCHEDULE_SLOT_META_SCHEMA_VERSION:
            raise TrainingScheduleError(f"invalid slot meta for {slot['model_key']}")
        if (
            slot_meta.get("schedule_build_id") != schedule_id
            or slot_meta.get("model_key") != slot["model_key"]
            or slot_meta.get("role") != slot["role"]
            or slot_meta.get("seed") != slot["seed"]
            or slot_meta.get("epochs") != slot["epochs"]
            or slot_meta.get("query_count_per_epoch") != len(evidence_records)
        ):
            raise TrainingScheduleError(f"slot meta disagrees with plan: {slot['model_key']}")
        epochs = slot_meta.get("epoch_registry")
        if not isinstance(epochs, list) or len(epochs) != slot["epochs"]:
            raise TrainingScheduleError(f"slot epoch registry is incomplete: {slot['model_key']}")
        epoch_digests: list[str] = []
        for epoch in range(1, slot["epochs"] + 1):
            entry = epochs[epoch - 1]
            relative = slot_directory / f"epoch-{epoch}.jsonl"
            expected_files.add(relative.as_posix())
            if entry.get("epoch") != epoch or entry.get("relative_path") != relative.as_posix():
                raise TrainingScheduleError("slot epoch registry path/order mismatch")
            rows = load_jsonl(target / relative)
            if len(rows) != len(evidence_records) or entry.get("record_count") != len(rows):
                raise TrainingScheduleError("schedule epoch record count mismatch")
            if [row.get("query_ordinal") for row in rows] != list(range(len(rows))):
                raise TrainingScheduleError("schedule query ordinals are not contiguous")
            for query_ordinal, (row, evidence_record) in enumerate(zip(rows, evidence_records)):
                _validate_schedule_record_wire(row)
                if (
                    row.get("schedule_build_id") != schedule_id
                    or row.get("model_key") != slot["model_key"]
                    or row.get("role") != slot["role"]
                    or row.get("seed") != slot["seed"]
                    or row.get("epoch") != epoch
                    or row.get("query_id") != evidence_record["query"]["id"]
                    or row.get("training_evidence_record_sha256") != evidence_record["record_sha256"]
                    or row.get("partition")
                    != partition_by_id[evidence_record["query"]["id"]]["partition"]
                ):
                    raise TrainingScheduleError("schedule record coordinate/dependency mismatch")
                presentation_epoch = 1 if row["partition"] == "calibration" else epoch
                if row.get("presentation_epoch") != presentation_epoch:
                    raise TrainingScheduleError(
                        "schedule presentation epoch differs from partition policy"
                    )
                expected_order = stateless_demo_order(
                    evidence_record["demo_evidence"]["ids"],
                    seed=slot["seed"],
                    epoch=presentation_epoch,
                    query_id=row["query_id"],
                )
                if row.get("ordered_demo_ids") != expected_order:
                    raise TrainingScheduleError("schedule demo order is not stateless/reproducible")
                expected_lexicon_ids = evidence_record["lexicon_evidence"]["ids"]
                if row.get("ordered_lexicon_ids") != expected_lexicon_ids:
                    raise TrainingScheduleError("schedule lexicon set/order differs from evidence")
                pair_key = (slot["seed"], epoch, row["query_id"])
                if pair_key in paired_orders and paired_orders[pair_key] != expected_order:
                    raise TrainingScheduleError("paired roles do not share demo order")
                paired_orders[pair_key] = expected_order
                expected_use = (
                    (True, True)
                    if slot["role"] == "M_LD"
                    else (
                        stateless_source_use(
                            seed=slot["seed"], epoch=presentation_epoch, query_id=row["query_id"], source="lexicon"
                        ),
                        stateless_source_use(
                            seed=slot["seed"], epoch=presentation_epoch, query_id=row["query_id"], source="demo"
                        ),
                    )
                )
                if (row.get("use_lexicon"), row.get("use_demos")) != expected_use:
                    raise TrainingScheduleError("schedule source mask is not stateless/reproducible")
                lexicon_ids, lexicon_items = _evidence_items(
                    evidence_record["lexicon_evidence"], kind="lexicon"
                )
                _, demo_items = _evidence_items(
                    evidence_record["demo_evidence"], kind="demo"
                )
                rendering = evidence_record["rendering"]
                expected_messages = render_messages(
                    query_content=evidence_record["query"]["content"],
                    lexicon_ids=lexicon_ids if expected_use[0] else [],
                    demo_ids=expected_order if expected_use[1] else [],
                    lexicon_catalog=lexicon_items,
                    demo_catalog=demo_items,
                    system_prompt=rendering["system_prompt"],
                    user_prompt_template=rendering["user_prompt_template"],
                )
                if (
                    row.get("instruction") != expected_messages[0]["content"]
                    or row.get("input") != expected_messages[1]["content"]
                    or row.get("output") != evidence_record["query"]["gold_text"]
                    or row.get("content") != evidence_record["query"]["content"]
                    or row.get("actual_gold_tokens")
                    != evidence_record["query"]["actual_gold_tokens"]
                    or row.get("max_sequence_tokens")
                    != evidence_record["budget"]["max_sequence_tokens"]
                ):
                    raise TrainingScheduleError(
                        "schedule rendered training item differs from frozen evidence"
                    )
                expected_row = build_schedule_record(
                    evidence_record,
                    slot,
                    epoch=epoch,
                    query_ordinal=query_ordinal,
                    schedule_build_id=schedule_id,
                    tokenizer=tokenizer_object,
                )
                if row != expected_row:
                    raise TrainingScheduleError("schedule row cannot be re-rendered exactly")
                all_hashes.append(row["record_sha256"])
            digest = _records_digest(rows)
            if entry.get("records_sha256") != digest:
                raise TrainingScheduleError("schedule epoch digest mismatch")
            epoch_digests.append(digest)
        slot_digest = canonical_sha256(epoch_digests)
        if slot_meta.get("slot_records_sha256") != slot_digest:
            raise TrainingScheduleError("schedule slot digest mismatch")
        expected_registry.append(
            {
                "model_key": slot["model_key"],
                "role": slot["role"],
                "seed": slot["seed"],
                "epochs": slot["epochs"],
                "relative_meta_path": (slot_directory / "schedule.meta.json").as_posix(),
                "slot_records_sha256": slot_digest,
            }
        )
    if meta.get("slot_registry") != expected_registry:
        raise TrainingScheduleError("schedule slot registry differs from training plan")
    if meta.get("total_record_count") != len(all_hashes):
        raise TrainingScheduleError("schedule total record count mismatch")
    if meta.get("all_schedule_records_sha256") != canonical_sha256(all_hashes):
        raise TrainingScheduleError("schedule global record digest mismatch")
    if meta.get("preflight_status") != "complete-pass":
        raise TrainingScheduleError("schedule preflight is not complete-pass")
    ensure_exact_file_set(target, expected_files)
    return {
        "valid": True,
        "schedule_build_id": schedule_id,
        "slot_count": len(slots),
        "record_count": len(all_hashes),
        "tokenizer_revalidated": True,
    }


def validate_training_schedule(
    *, schedule_ref: str | Path, tokenizer: Any | None = None
) -> dict[str, Any]:
    locator, target = resolve_locator_ref(schedule_ref, SCHEDULE_ARTIFACT_KIND)
    meta = load_json(target / "schedule.meta.json")
    if not isinstance(meta, Mapping):
        raise TrainingScheduleError("invalid schedule meta")
    plan_dependency = validate_dependency_ref(
        meta.get("training_plan_dependency", {}), expected_kind="training-plan"
    )
    workspace_root = _infer_workspace_root(target, plan_dependency)
    evidence_dependency = validate_dependency_ref(
        meta.get("training_evidence_dependency", {}),
        expected_kind="training-evidence",
    )
    evidence_target = resolve_dependency_target(
        evidence_dependency, workspace_root
    )
    from data.training_evidence import frozen_evidence_tokenizer_lease

    # Even an explicitly supplied tokenizer is only a test/engineering object;
    # all schedule row rendering and exact replay still execute while the
    # evidence-bound tokenizer/base source trees are freshly verified and held.
    with frozen_evidence_tokenizer_lease(
        evidence_target,
        workspace_root=workspace_root,
        tokenizer=tokenizer,
    ) as tokenizer_object:
        report = _validate_schedule_target(
            target,
            workspace_root=workspace_root,
            tokenizer=tokenizer_object,
            require_directory_name=True,
            _source_lease_held=True,
        )
    if locator["artifact_id"] != report["schedule_build_id"]:
        raise TrainingScheduleError("schedule locator artifact ID mismatch")
    return report


def load_training_schedule(
    schedule_ref: str | Path,
    *,
    tokenizer: Any | None = None,
) -> tuple[dict[str, Any], Path, dict[str, Any]]:
    locator, target = resolve_locator_ref(schedule_ref, SCHEDULE_ARTIFACT_KIND)
    validate_training_schedule(schedule_ref=schedule_ref, tokenizer=tokenizer)
    meta = load_json(target / "schedule.meta.json")
    return locator, target, meta


def load_model_epoch_records(
    schedule_target: str | Path, *, model_key: str, epoch: int
) -> list[dict[str, Any]]:
    match = MODEL_KEY_RE.fullmatch(model_key)
    if match is None or epoch <= 0:
        raise TrainingScheduleError("invalid model_key or epoch")
    target = Path(schedule_target)
    path = target / "schedules" / match.group(1) / f"seed-{match.group(2)}" / f"epoch-{epoch}.jsonl"
    rows = load_jsonl(path)
    for row in rows:
        _validate_schedule_record_wire(row)
        if row.get("model_key") != model_key or row.get("epoch") != epoch:
            raise TrainingScheduleError("schedule epoch file contains a foreign record")
    return rows


__all__ = [
    "DEMO_ORDER_POLICY",
    "DROP_RNG_POLICY",
    "OVERFLOW_POLICY",
    "RENDERER_REVISION",
    "SCHEDULE_ARTIFACT_KIND",
    "SCHEDULE_RECORD_SCHEMA_VERSION",
    "SCHEDULE_SCHEMA_VERSION",
    "TrainingScheduleError",
    "build_schedule_record",
    "build_training_schedule",
    "load_model_epoch_records",
    "load_training_schedule",
    "stateless_demo_order",
    "stateless_source_use",
    "tokenizer_revision_from_directory",
    "validate_evidence_record_for_schedule",
    "validate_plan_slots_for_schedule",
    "validate_training_schedule",
]
