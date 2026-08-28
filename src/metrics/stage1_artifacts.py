"""Immutable Stage-1 free-evaluation and factorial-analysis lifecycles.

This module never runs a model.  It consumes already-frozen generation and
margin targets, verifies their complete paired frames and portable lineage,
then publishes new content-addressed evaluation/analysis targets.  Direct
numeric/text fixtures remain separate engineering-only CLI commands.
"""

from __future__ import annotations

import copy
import hashlib
import math
import re
import shutil
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

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
    write_bytes_atomic,
    write_canonical_json,
    write_locator_ref,
)
from data.generation_lifecycle import (
    GENERATION_ARTIFACT_KIND,
    GenerationLifecycleError,
    validate_generation_target as validate_upstream_generation_target,
)
from data.counterfactual_lifecycle import (
    CounterfactualLifecycleError,
    validate_cf_target,
)
from metrics.stage1_metrics import (
    ASSIGNMENT_VERSION,
    DEFAULT_MAX_TUPLES,
    DEFAULT_SOFT_THRESHOLD,
    METRIC_SCHEMA_VERSION,
    SIMILARITY_VERSION,
    aggregate_query_metrics,
    evaluate_query,
    flip_table,
)
from metrics.stage1_statistics import (
    classify_behavior_test,
    classify_margin_test,
    factorial_effects,
    fixed_seed_margin_bootstrap,
    fixed_seed_paired_bootstrap,
    holm_adjust,
    placebo_effects,
)
from model.stage1_registry import (
    MODEL_ARTIFACT_KIND,
    REGISTRY_ARTIFACT_KIND,
    ModelRegistryError,
    ResolvedModelSourceContract,
    resolve_registered_model_dependency,
    validate_model_registry_target,
    verified_model_source_lease,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_ROOT = REPOSITORY_ROOT / "schemas"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
CONDITIONS = ("C0", "CL", "CD", "CLD", "PL", "PD")
FIELDS = ("target", "argument", "targeted_group", "hateful")
EVALUATION_KIND = "evaluation"
ANALYSIS_KIND = "analysis"
GENERATION_KIND = GENERATION_ARTIFACT_KIND
REGISTRY_KIND = REGISTRY_ARTIFACT_KIND
MARGIN_KIND = "margin"
EVALUATION_META_SCHEMA = "stage1-evaluation/v1"
ANALYSIS_META_SCHEMA = "stage1-factorial-analysis/v1"


class Stage1ArtifactError(ValueError):
    """Raised when a downstream Stage-1 artifact violates frozen protocol."""


def _artifact_error(exc: Exception) -> Stage1ArtifactError:
    return Stage1ArtifactError(str(exc))


def _object(source: str | Path | Mapping[str, Any], *, name: str) -> dict[str, Any]:
    if isinstance(source, Mapping):
        return copy.deepcopy(dict(source))
    try:
        value = load_json(source)
    except TrainingArtifactError as exc:
        raise _artifact_error(exc) from exc
    if not isinstance(value, dict):
        raise Stage1ArtifactError(f"{name} must be a JSON object")
    return value


def _hash_text(value: str) -> str:
    return hashlib.sha256(value.replace("\r\n", "\n").encode("utf-8")).hexdigest()


def _ordered_id_hash(ids: Sequence[str]) -> str:
    return canonical_sha256(list(ids))


def _strict_hash(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not SHA256_RE.fullmatch(value):
        raise Stage1ArtifactError(f"{label} must be a full SHA-256")
    return value


def _dependency_target(
    dependency: Mapping[str, Any],
    *,
    workspace_root: str | Path,
    kinds: str | Sequence[str],
) -> Path:
    try:
        validate_dependency_ref(dependency, expected_kind=kinds)
        return resolve_dependency_target(dependency, workspace_root)
    except TrainingArtifactError as exc:
        raise _artifact_error(exc) from exc


def _assert_target_unchanged(target: Path, expected_payload_hash: str, *, label: str) -> None:
    try:
        actual = validate_payload_manifest(target)
    except TrainingArtifactError as exc:
        raise _artifact_error(exc) from exc
    if actual != expected_payload_hash:
        raise Stage1ArtifactError(f"{label} changed while downstream target was built")


def _assert_canonical_json(path: Path, expected: Any, *, label: str) -> None:
    if path.read_bytes() != canonical_json_bytes(expected) + b"\n":
        raise Stage1ArtifactError(f"{label} is not canonical JSON")


def _assert_ordered_jsonl(
    path: Path, expected: Sequence[Mapping[str, Any]], *, label: str
) -> None:
    wire = b"".join(canonical_json_bytes(dict(row)) + b"\n" for row in expected)
    if path.read_bytes() != wire:
        raise Stage1ArtifactError(f"{label} is not canonical ordered JSONL")


def _scope_matrix(
    *,
    scope: str,
    split: str,
    sealing_status: str,
    scientific_eligible: bool,
    mode: str | None = None,
) -> None:
    if scope == "engineering-smoke":
        if split != "dev" or sealing_status != "unsealed-dev" or scientific_eligible:
            raise Stage1ArtifactError("engineering-smoke must be unsealed dev and ineligible")
        if mode is not None and mode != "engineering-smoke":
            raise Stage1ArtifactError("engineering-smoke mode mismatch")
    elif scope == "pilot":
        if split != "dev" or sealing_status != "unsealed-dev" or scientific_eligible:
            raise Stage1ArtifactError("pilot must be unsealed dev and ineligible")
        if mode is not None and mode != "pilot":
            raise Stage1ArtifactError("pilot analysis mode mismatch")
    elif scope == "formal":
        expected_sealing = "sealed-test" if split == "test" else "unsealed-dev"
        if split not in {"dev", "test"} or sealing_status != expected_sealing or not scientific_eligible:
            raise Stage1ArtifactError("formal scope/split/sealing/scientific matrix mismatch")
        if mode is not None and mode != "confirmatory":
            raise Stage1ArtifactError("formal analysis must use confirmatory mode")
    else:
        raise Stage1ArtifactError(f"unsupported registry scope: {scope!r}")


def _control_kind_for_split_sealing(*, split: Any, sealing_status: Any) -> str:
    """Return the only control artifact kind allowed by a split/sealing pair."""

    if split == "dev" and sealing_status == "unsealed-dev":
        return "control"
    if split == "test" and sealing_status == "sealed-test":
        return "test-control"
    raise Stage1ArtifactError(
        "control dependency requires dev/unsealed-dev or test/sealed-test lineage"
    )


def _assert_control_dependency_kind(
    dependency: Any,
    *,
    split: Any,
    sealing_status: Any,
    label: str,
) -> str:
    expected = _control_kind_for_split_sealing(
        split=split, sealing_status=sealing_status
    )
    if not isinstance(dependency, Mapping) or dependency.get("artifact_kind") != expected:
        raise Stage1ArtifactError(
            f"{label} control kind differs from split/sealing lineage"
        )
    return expected


def resolve_evaluation_profile(source: str | Path | Mapping[str, Any]) -> dict[str, Any]:
    profile = _object(source, name="evaluation profile")
    if profile.get("schema_version") != "stage1-evaluation-profile/v1":
        raise Stage1ArtifactError("unsupported evaluation profile schema")
    if profile.get("ordered_conditions") != list(CONDITIONS):
        raise Stage1ArtifactError("evaluation conditions must be exactly C0/CL/CD/CLD/PL/PD")
    parser = profile.get("parser")
    if not isinstance(parser, Mapping) or parser.get("strict_only") is not True:
        raise Stage1ArtifactError("formal evaluator requires strict-only parsing")
    for flag in ("allow_recovery_for_scoring", "allow_legacy_pipe", "allow_markdown_fence", "allow_trailing_text"):
        if parser.get(flag) is not False:
            raise Stage1ArtifactError(f"evaluation parser must freeze {flag}=false")
    denominator = profile.get("denominator")
    if not isinstance(denominator, Mapping) or denominator.get("policy") != "fixed-expected-query-frame/v1":
        raise Stage1ArtifactError("evaluation denominator policy mismatch")
    for flag in ("exclude_parse_failures", "exclude_invalid_outputs", "silent_id_intersection"):
        if denominator.get(flag) is not False:
            raise Stage1ArtifactError(f"evaluation denominator must freeze {flag}=false")
    if profile.get("invalid_output_correctness") is not False:
        raise Stage1ArtifactError("invalid outputs must remain wrong in the denominator")
    assignment = profile.get("assignment")
    if not isinstance(assignment, Mapping) or assignment.get("version") != ASSIGNMENT_VERSION:
        raise Stage1ArtifactError("evaluation assignment version mismatch")
    if assignment.get("tie_break") != "lexicographic-canonical-tuple":
        raise Stage1ArtifactError("evaluation assignment tie-break mismatch")
    if parser.get("wire_format") != "compact-json-array-v1" or parser.get(
        "quad_schema"
    ) != "canonical-quad-json/v1":
        raise Stage1ArtifactError("evaluation wire/schema contract mismatch")
    expected_metrics = {
        "tuple": ["precision", "recall", "f1_avg", "hard_exact", "soft_exact"],
        "target_argument": ["unbound_exact", "similarity"],
        "targeted_group_hateful": ["bound_precision", "bound_recall", "bound_f1"],
        "format": ["strict_format_valid", "canonical_wire_equal"],
        "flip": ["wrong_to_correct", "correct_to_wrong", "net_flip", "exact_mcnemar"],
    }
    if profile.get("metrics") != expected_metrics:
        raise Stage1ArtifactError("evaluation metric family/order mismatch")
    profile["soft_threshold"] = DEFAULT_SOFT_THRESHOLD
    profile["max_tuples"] = DEFAULT_MAX_TUPLES
    return profile


def resolve_analysis_profile(source: str | Path | Mapping[str, Any]) -> dict[str, Any]:
    profile = _object(source, name="analysis profile")
    if profile.get("schema_version") != "stage1-analysis-profile/v1":
        raise Stage1ArtifactError("unsupported analysis profile schema")
    mode = profile.get("mode")
    if mode not in {"engineering-smoke", "pilot", "confirmatory"}:
        raise Stage1ArtifactError("analysis mode is invalid")
    if profile.get("ordered_conditions") != list(CONDITIONS):
        raise Stage1ArtifactError("analysis conditions must be exactly C0/CL/CD/CLD/PL/PD")
    if profile.get("reducer_version") != "stage1-factorial-reducer/v1":
        raise Stage1ArtifactError("analysis reducer version mismatch")
    bootstrap = profile.get("bootstrap")
    if not isinstance(bootstrap, Mapping):
        raise Stage1ArtifactError("analysis profile lacks bootstrap settings")
    if bootstrap.get("unit") != "paired-query" or bootstrap.get("family_stream") != "sha256-seed-family-replicate/v1":
        raise Stage1ArtifactError("analysis bootstrap policy mismatch")
    if not isinstance(bootstrap.get("replicates"), int) or bootstrap["replicates"] <= 0:
        raise Stage1ArtifactError("analysis bootstrap replicate count is invalid")
    if not isinstance(bootstrap.get("seed"), int):
        raise Stage1ArtifactError("analysis bootstrap seed is invalid")
    expected_scientific = mode == "confirmatory"
    if profile.get("scientific_eligible") is not expected_scientific:
        raise Stage1ArtifactError("analysis profile mode/scientific flag mismatch")
    gate = profile.get("gate_policy")
    if not isinstance(gate, Mapping):
        raise Stage1ArtifactError("analysis profile lacks gate policy")
    if (mode == "confirmatory") is not bool(gate.get("emit_confirmatory_gate")):
        raise Stage1ArtifactError("analysis mode/gate emission mismatch")
    expected_splits = ["dev", "test"] if mode == "confirmatory" else ["dev"]
    if profile.get("allowed_splits") != expected_splits:
        raise Stage1ArtifactError("analysis mode has a non-canonical split allowlist")
    if bootstrap.get("interval") != "percentile-95":
        raise Stage1ArtifactError("analysis interval policy must be percentile-95")
    if mode == "pilot":
        if profile.get("required_model_keys") != ["M_LD/seed-42", "M_drop/seed-42"]:
            raise Stage1ArtifactError("pilot profile must freeze the two seed-42 slots")
    if mode == "confirmatory":
        if bootstrap.get("resample_training_seeds") is not False:
            raise Stage1ArtifactError("confirmatory bootstrap cannot resample training seeds")
        if profile.get("fixed_training_seeds") != [42, 43, 44]:
            raise Stage1ArtifactError("confirmatory profile must freeze seeds 42/43/44")
        if profile.get("primary_model_role") != "M_drop" or profile.get(
            "secondary_model_role"
        ) != "M_LD":
            raise Stage1ArtifactError("confirmatory primary/secondary roles are not frozen")
        if profile.get("estimand") != {
            "population": "frozen-eligible-query-frame",
            "seed_scope": "fixed-checkpoint-set",
            "seed_aggregation": "mean-of-within-seed-effects",
            "pool_seed_query_records": False,
            "policy": "intention-to-treat",
        }:
            raise Stage1ArtifactError("confirmatory estimand permits pooling or drift")
        if profile.get("directional_stability") != {
            "minimum_seeds_in_claim_direction": 2,
            "forbid_any_seed_at_opposite_sesoi": True,
        }:
            raise Stage1ArtifactError("confirmatory directional-stability policy mismatch")
        multiple = profile.get("multiple_testing")
        if not isinstance(multiple, Mapping) or multiple.get("method") != "holm" or float(
            multiple.get("alpha", -1)
        ) != 0.05:
            raise Stage1ArtifactError("confirmatory Holm policy mismatch")
        families = multiple.get("families")
        if not isinstance(families, Mapping) or list(families) != [
            "free_generation_corroboration",
            "gold_margin_total_effect",
            "gold_margin_relevance",
        ]:
            raise Stage1ArtifactError("confirmatory Holm families/order mismatch")
        behavior = families["free_generation_corroboration"]
        if behavior.get("contrasts") != ["CL-PL", "CD-PD"] or behavior.get(
            "endpoints"
        ) != list(BEHAVIOR_ENDPOINTS):
            raise Stage1ArtifactError("confirmatory behavior endpoints/contrasts mismatch")
        for family, contrasts in (
            ("gold_margin_total_effect", ["CL-C0", "CD-C0"]),
            ("gold_margin_relevance", ["CL-PL", "CD-PD"]),
        ):
            config = families[family]
            if config.get("contrasts") != contrasts or config.get("fields") != list(FIELDS):
                raise Stage1ArtifactError(f"confirmatory {family} frame mismatch")
        if gate != {
            "emit_confirmatory_gate": True,
            "h1_1_requires_margin_support_for_both_sources_in_primary_model": True,
            "free_generation_is_corroborating_not_substitutive": True,
            "report_harm_as_tradeoff_not_cancellation": True,
        }:
            raise Stage1ArtifactError("confirmatory gate policy mismatch")
    return profile


def resolve_margin_profile(source: str | Path | Mapping[str, Any]) -> dict[str, Any]:
    profile = _object(source, name="margin scorer profile")
    try:
        validate_json_schema(
            profile, SCHEMA_ROOT / "stage1_margin_scorer_profile_v1.schema.json"
        )
    except TrainingArtifactError as exc:
        raise _artifact_error(exc) from exc
    if profile.get("schema_version") != "stage1-margin-scorer-profile/v1":
        raise Stage1ArtifactError("margin scorer profile schema mismatch")
    if profile.get("backend") != "local-huggingface-causal-lm":
        raise Stage1ArtifactError("margin scorer must use the registered local HF backend")
    if profile.get("ordered_conditions") != list(CONDITIONS) or profile.get(
        "fields"
    ) != list(FIELDS):
        raise Stage1ArtifactError("margin scorer profile frame mismatch")
    runtime = profile.get("runtime")
    expected_runtime = {
        "dtype": "bfloat16",
        "device_map": "auto",
        "batch_unit": "gold-cf-pair",
        "batch_size": 1,
        "max_sequence_tokens": 2048,
        "truncation_allowed": False,
        "trust_remote_code": False,
        "right_padding": True,
        "position_ids": "attention-mask-cumsum-minus-one-zero-on-padding/v1",
        "log_softmax_dtype": "float32",
        "thinking_mode": False,
    }
    if runtime != expected_runtime:
        raise Stage1ArtifactError("margin scorer runtime contract mismatch")
    if profile.get("tokenization") != {
        "prompt_response_segmentation": "separate-no-special-tokens/v1",
        "span_mask": "minimal-overlap-cover/v1",
        "score_complete_json_value_literal": True,
        "causal_shift": "logits-minus-last-vs-input-minus-first",
        "record_boundary_crossing": True,
    }:
        raise Stage1ArtifactError("margin tokenization/span-mask contract mismatch")
    if profile.get("aggregation") != {
        "primary_token_aggregation": "mean",
        "sensitivity_token_aggregation": "sum",
        "tuple_aggregation": "equal-weight-within-query",
        "query_aggregation": "equal-weight",
        "hateful_interpretation": "gold-prefix-conditional",
    }:
        raise Stage1ArtifactError("margin aggregation contract mismatch")
    if profile.get("record_outputs") != [
        "token_ids",
        "token_count",
        "character_span",
        "token_span",
        "sum_logprob",
        "mean_logprob",
        "left_boundary_crossing",
        "right_boundary_crossing",
    ]:
        raise Stage1ArtifactError("margin scorer output audit fields/order mismatch")
    if profile.get("eligibility") != {
        "mask_timing": "frozen-before-scoring",
        "mask_shared_across_conditions_and_models": True,
        "overflow_policy": "hard-fail-and-common-complete-case-mask",
    }:
        raise Stage1ArtifactError("margin eligibility policy mismatch")
    return profile


def _margin_tokenizer_lease_sources(
    contract: ResolvedModelSourceContract,
    *,
    require_full_coverage: bool,
) -> tuple[str, ...]:
    """Select the tokenizer lease members without weakening old artifacts.

    Newly registered tokenizer inventories cover their complete regular-file
    trees.  Older non-scientific model artifacts may contain the historical
    tokenizer-only projection and remain replayable.  A scientific replay may
    use that projection only when the complete base inventory covers the same
    physical tree and contains every projected tokenizer row; leasing both
    inventories then protects the effective union used by the constructor.
    """

    if not isinstance(contract, ResolvedModelSourceContract):
        raise Stage1ArtifactError(
            "margin tokenizer replay requires a typed registered-model source contract"
        )
    tokenizer_inventory = contract.tokenizer_inventory
    base_inventory = contract.base_inventory
    if not isinstance(tokenizer_inventory, Mapping) or not isinstance(
        base_inventory, Mapping
    ):
        raise Stage1ArtifactError("margin tokenizer source inventories are malformed")
    policy = tokenizer_inventory.get("inventory_policy")
    if policy == "all-regular-files/v1":
        return ("tokenizer",)
    if policy != "tokenizer-files/v1":
        raise Stage1ArtifactError("margin tokenizer inventory policy is unsupported")
    if not require_full_coverage:
        return ("tokenizer",)
    if (
        base_inventory.get("inventory_policy") != "all-regular-files/v1"
        or base_inventory.get("logical_repo_path")
        != tokenizer_inventory.get("logical_repo_path")
    ):
        raise Stage1ArtifactError(
            "scientific margin tokenizer subset lacks same-tree full base coverage"
        )
    try:
        base_files = {
            str(row["path"]): (row["size"], row["sha256"])
            for row in base_inventory["files"]
        }
        tokenizer_files = {
            str(row["path"]): (row["size"], row["sha256"])
            for row in tokenizer_inventory["files"]
        }
    except (KeyError, TypeError) as exc:
        raise Stage1ArtifactError(
            "scientific margin tokenizer/base coverage rows are malformed"
        ) from exc
    if not tokenizer_files or any(
        base_files.get(path) != signature
        for path, signature in tokenizer_files.items()
    ):
        raise Stage1ArtifactError(
            "scientific margin full base inventory does not cover tokenizer subset"
        )
    return ("tokenizer", "base")


def resolve_decision_register(
    source: str | Path | Mapping[str, Any], *, profile: Mapping[str, Any]
) -> dict[str, Any]:
    decision = _object(source, name="decision register")
    if decision.get("schema_version") != "stage1-decision-register/v1" or decision.get(
        "status"
    ) != "frozen":
        raise Stage1ArtifactError("analysis requires the frozen Stage 1 decision register")
    decisions = decision.get("decisions")
    if not isinstance(decisions, Mapping):
        raise Stage1ArtifactError("decision register lacks frozen decisions")
    seeds = decisions.get("D7_training_seeds")
    if not isinstance(seeds, Mapping) or seeds.get("formal") != [42, 43, 44] or seeds.get(
        "automatic_seed_expansion"
    ) is not False:
        raise Stage1ArtifactError("decision register training-seed policy mismatch")
    if profile["mode"] == "confirmatory":
        gate = decisions.get("D9_dev_gate")
        if not isinstance(gate, Mapping) or gate.get("bootstrap_replicates") != profile[
            "bootstrap"
        ]["replicates"] or gate.get("bootstrap_seed") != profile["bootstrap"]["seed"]:
            raise Stage1ArtifactError("decision register/profile bootstrap mismatch")
        if gate.get("multiple_testing") != "holm-by-pre-registered-family":
            raise Stage1ArtifactError("decision register multiple-testing policy mismatch")
    return decision


def validate_registry_target(
    target_dir: str | Path,
    *,
    workspace_root: str | Path = REPOSITORY_ROOT,
    require_directory_name: bool = True,
) -> dict[str, Any]:
    try:
        registry = validate_model_registry_target(
            target_dir,
            workspace_root=workspace_root,
            require_name=require_directory_name,
        )
    except (TrainingArtifactError, ModelRegistryError) as exc:
        raise _artifact_error(exc) from exc
    target = Path(target_dir)
    normalized = []
    for entry in registry["entries"]:
        normalized.append(
            {
                **copy.deepcopy(dict(entry)),
                "scientific_eligible": bool(entry["model_scientific_eligible"]),
            }
        )
    return {
        "model_registry_id": registry["model_registry_id"],
        "registry_scope": registry["scope"],
        "scientific_eligible": registry["scientific_eligible"],
        "training_plan_dependency": registry["training_plan_dependency"],
        "models": normalized,
        "payload_manifest_sha256": sha256_file(target / "payload_manifest.json"),
    }


def validate_registry_ref(
    registry_ref: str | Path, *, workspace_root: str | Path = REPOSITORY_ROOT
) -> tuple[dict[str, Any], dict[str, Any], Path]:
    try:
        locator, target = resolve_locator_ref(registry_ref, REGISTRY_KIND)
    except TrainingArtifactError as exc:
        raise _artifact_error(exc) from exc
    report = validate_registry_target(target, workspace_root=workspace_root)
    if locator["artifact_id"] != report["model_registry_id"] or locator[
        "payload_manifest_sha256"
    ] != report["payload_manifest_sha256"]:
        raise Stage1ArtifactError("registry locator mismatch")
    return locator, report, target


def _context_frame(target: Path, *, split: str) -> list[dict[str, Any]]:
    records_path = target / f"context_manifest.{split}.jsonl"
    if not records_path.is_file():
        raise Stage1ArtifactError("context dependency lacks split manifest records")
    rows = load_jsonl(records_path)
    pool_path = target / "catalogs" / f"query_pool.{split}.jsonl"
    pool: dict[str, Mapping[str, Any]] = {}
    if pool_path.is_file():
        for raw in load_jsonl(pool_path):
            query = raw.get("query", raw)
            pool[str(query.get("id", ""))] = query
    frame = []
    for row in rows:
        query = row.get("query")
        if not isinstance(query, Mapping):
            raise Stage1ArtifactError("context record lacks query")
        query_id = str(query.get("id", ""))
        source = pool.get(query_id, query)
        content = source.get("content")
        gold = source.get("gold", source.get("quadruples"))
        if not isinstance(content, str) or gold is None:
            raise Stage1ArtifactError("context frame lacks frozen content/gold")
        content_hash = _hash_text(content)
        gold_hash = source.get("gold_sha256", query.get("gold_sha256"))
        _strict_hash(gold_hash, label="context gold hash")
        if query.get("content_sha256") not in {None, content_hash} or query.get(
            "gold_sha256"
        ) not in {None, gold_hash}:
            raise Stage1ArtifactError("context frame content/gold hash mismatch")
        record_hash = row.get("record_sha256")
        _strict_hash(record_hash, label="context record hash")
        frame.append(
            {
                "id": query_id,
                "content": content,
                "gold": gold,
                "content_sha256": content_hash,
                "gold_sha256": gold_hash,
                "context_record_sha256": record_hash,
            }
        )
    ids = [row["id"] for row in frame]
    if len(ids) != len(set(ids)) or not ids:
        raise Stage1ArtifactError("context query frame is empty or duplicated")
    return frame


def _validate_generation_profile(profile: Mapping[str, Any], conditions: Sequence[str]) -> None:
    if profile.get("schema_version") != "stage1-generation-profile/v1":
        raise Stage1ArtifactError("generation profile schema mismatch")
    if profile.get("ordered_conditions") != list(conditions):
        raise Stage1ArtifactError("generation profile condition order mismatch")
    sampling = profile.get("sampling")
    failure = profile.get("failure_policy")
    if not isinstance(sampling, Mapping) or sampling.get("do_sample") is not False:
        raise Stage1ArtifactError("generation profile must be deterministic")
    if not isinstance(failure, Mapping) or failure.get("require_complete_paired_block") is not True:
        raise Stage1ArtifactError("generation profile must require a complete paired block")


def _validated_generation_status(
    *, finish_reason: Any, runner_status: Any
) -> tuple[str, str]:
    """Cross-check the generation stop state before metric evaluation."""

    if finish_reason not in {"eos", "length", "fixture"}:
        raise Stage1ArtifactError("generation finish reason is unsupported")
    expected_status = "length" if finish_reason == "length" else "ok"
    if runner_status != expected_status:
        raise Stage1ArtifactError(
            "generation finish_reason/runner_status mismatch"
        )
    return str(finish_reason), expected_status


def validate_generation_target(
    target_dir: str | Path,
    *,
    workspace_root: str | Path = REPOSITORY_ROOT,
    require_directory_name: bool = True,
) -> dict[str, Any]:
    target = Path(target_dir)
    try:
        raw_meta = load_json(target / "generation.meta.json")
        if not isinstance(raw_meta, Mapping):
            raise Stage1ArtifactError("generation meta must be an object")
        upstream_report = validate_upstream_generation_target(
            target,
            workspace_root=workspace_root,
            require_scientific=raw_meta.get("scope") == "formal",
        )
    except (TrainingArtifactError, GenerationLifecycleError) as exc:
        raise _artifact_error(exc) from exc
    meta = dict(raw_meta)
    conditions = meta.get("ordered_conditions")
    if conditions != list(CONDITIONS):
        raise Stage1ArtifactError("generation condition frame is not frozen/exact")
    dependencies = meta.get("dependencies")
    if not isinstance(dependencies, Mapping):
        raise Stage1ArtifactError("generation lacks frozen dependencies")
    registry_dep = dependencies.get("model_registry")
    context_dep = dependencies.get("context")
    control_dep = dependencies.get("control")
    registry_target = _dependency_target(
        registry_dep, workspace_root=workspace_root, kinds=REGISTRY_KIND
    )
    registry = validate_registry_target(registry_target, workspace_root=workspace_root)
    model_key = meta.get("model_key")
    matched = [row for row in registry["models"] if row["model_key"] == model_key]
    if len(matched) != 1:
        raise Stage1ArtifactError("generation model binding differs from registry")
    model_dep = matched[0]["model_dependency"]
    plan_dep = registry["training_plan_dependency"]
    split = meta.get("split")
    if split not in {"dev", "test"}:
        raise Stage1ArtifactError("free evaluation only accepts dev/test generation")
    sealing = "sealed-test" if split == "test" else "unsealed-dev"
    generation_scope = meta.get("scope")
    expected_generation_scope = "formal" if registry["registry_scope"] == "formal" else "engineering"
    if generation_scope != expected_generation_scope:
        raise Stage1ArtifactError("generation scope differs from registry scope")
    _scope_matrix(
        scope=registry["registry_scope"],
        split=split,
        sealing_status=sealing,
        scientific_eligible=bool(meta.get("scientific_eligible")),
    )
    if split == "test" and context_dep["artifact_kind"] != "test-context":
        raise Stage1ArtifactError("sealed generation does not use test-context")
    if split == "dev" and context_dep["artifact_kind"] != "context":
        raise Stage1ArtifactError("dev generation does not use dev context")
    if control_dep is None:
        raise Stage1ArtifactError("full Stage 1 condition family requires a control dependency")
    expected_control_kind = _assert_control_dependency_kind(
        control_dep,
        split=split,
        sealing_status=sealing,
        label="generation",
    )
    _dependency_target(
        control_dep,
        workspace_root=workspace_root,
        kinds=expected_control_kind,
    )
    query_ids = list(meta.get("ordered_query_ids", []))
    if not query_ids or len(query_ids) != len(set(query_ids)):
        raise Stage1ArtifactError("generation ordered query frame is empty or duplicated")
    if meta.get("ordered_query_ids_sha256") != _ordered_id_hash(query_ids):
        raise Stage1ArtifactError("generation ordered query frame hash mismatch")
    if upstream_report.get("query_count") != len(query_ids) or upstream_report.get(
        "complete_paired_blocks"
    ) is not True:
        raise Stage1ArtifactError("generation paired block is incomplete")
    profile = load_json(target / "generation_profile.resolved.json")
    _validate_generation_profile(profile, conditions)
    generation_id = meta.get("generation_run_id")
    if require_directory_name and target.name != generation_id:
        raise Stage1ArtifactError("generation target directory/ID mismatch")
    records = load_jsonl(target / "generations.jsonl")
    by_condition: dict[str, list[dict[str, Any]]] = {condition: [] for condition in conditions}
    for record in records:
        condition = str(record["condition"])
        finish_reason, runner_status = _validated_generation_status(
            finish_reason=record.get("finish_reason"),
            runner_status=record.get("runner_status"),
        )
        by_condition[condition].append(
            {
                "id": str(record["query_id"]),
                "condition": condition,
                "model_key": str(record["model_key"]),
                "raw_output": record["raw_output"],
                "runner_status": runner_status,
                "finish_reason": finish_reason,
                "gold": copy.deepcopy(record["gold"]),
                "content_sha256": record["content_sha256"],
                "gold_sha256": record["gold_sha256"],
                "prompt_sha256": record["prompt_sha256"],
                "context_record_sha256": record["context_record_sha256"],
            }
        )
    for condition, rows in by_condition.items():
        if [row["id"] for row in rows] != query_ids:
            raise Stage1ArtifactError(f"generation {condition} silently changed query frame/order")
    if meta.get("model_role") != matched[0]["role"] or meta.get("model_seed") != matched[0]["seed"]:
        raise Stage1ArtifactError("generation model role/seed differs from registry slot")
    return {
        "generation_run_id": generation_id,
        "scope": registry["registry_scope"],
        "scientific_eligible": meta["scientific_eligible"],
        "split": split,
        "sealing_status": sealing,
        "model_key": model_key,
        "role": matched[0]["role"],
        "seed": matched[0]["seed"],
        "conditions": conditions,
        "query_ids": query_ids,
        "predictions": by_condition,
        "registry_dependency": registry_dep,
        "training_plan_dependency": plan_dep,
        "model_dependency": model_dep,
        "context_dependency": context_dep,
        "control_dependency": control_dep,
        "payload_manifest_sha256": upstream_report["payload_manifest_sha256"],
    }


def validate_generation_ref(
    generation_ref: str | Path,
    *,
    workspace_root: str | Path = REPOSITORY_ROOT,
) -> tuple[dict[str, Any], dict[str, Any], Path]:
    try:
        locator, target = resolve_locator_ref(generation_ref, GENERATION_KIND)
    except TrainingArtifactError as exc:
        raise _artifact_error(exc) from exc
    report = validate_generation_target(target, workspace_root=workspace_root)
    if locator["artifact_id"] != report["generation_run_id"] or locator[
        "payload_manifest_sha256"
    ] != report["payload_manifest_sha256"]:
        raise Stage1ArtifactError("generation locator mismatch")
    return locator, report, target


def _evaluation_code_sha256() -> str:
    return canonical_sha256(
        {
            "artifact_lifecycle_sha256": sha256_file(__file__),
            "metric_module_sha256": sha256_file(Path(__file__).with_name("stage1_metrics.py")),
        }
    )


def _evaluation_id_inputs(
    *,
    generation_dependency: Mapping[str, Any],
    generation: Mapping[str, Any],
    profile: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "generation_dependency": dict(generation_dependency),
        "context_dependency": generation["context_dependency"],
        "split": generation["split"],
        "sealing_status": generation["sealing_status"],
        "evaluation_profile_sha256": canonical_sha256(profile),
        "strict_parser_schema_version": "canonical-quad-json/v1",
        "normalizer_schema_version": "canonical-quad-json/v1",
        "tuple_assignment_version": ASSIGNMENT_VERSION,
        "metric_schema_version": METRIC_SCHEMA_VERSION,
        "similarity_version": SIMILARITY_VERSION,
        "flip_schema_version": "stage1-flip-table/v1",
        "sufficient_stat_schema_version": "stage1-evaluation-summary/v1",
        "eligibility_policy": "fixed-expected-query-frame/v1",
        "evaluator_code_sha256": _evaluation_code_sha256(),
    }


def _evaluate_generation(
    generation: Mapping[str, Any], profile: Mapping[str, Any]
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    evaluated: dict[str, list[dict[str, Any]]] = {}
    summaries: dict[str, Any] = {}
    for condition in generation["conditions"]:
        rows = []
        for prediction in generation["predictions"][condition]:
            finish_reason, runner_status = _validated_generation_status(
                finish_reason=prediction.get("finish_reason"),
                runner_status=prediction.get("runner_status"),
            )
            row = evaluate_query(
                query_id=prediction["id"],
                condition=condition,
                raw_output=prediction["raw_output"],
                gold=prediction["gold"],
                runner_status=runner_status,
                content_sha256=prediction["content_sha256"],
                gold_sha256=prediction["gold_sha256"],
                prompt_sha256=prediction["prompt_sha256"],
                context_record_sha256=prediction["context_record_sha256"],
                soft_threshold=float(profile["soft_threshold"]),
                max_tuples=int(profile["max_tuples"]),
            )
            row["finish_reason"] = finish_reason
            rows.append(row)
        evaluated[condition] = rows
        summaries[condition] = aggregate_query_metrics(rows)
    return evaluated, summaries


def _evaluation_meta(
    *,
    evaluation_id: str,
    id_inputs: Mapping[str, Any],
    generation: Mapping[str, Any],
    evaluated: Mapping[str, Sequence[Mapping[str, Any]]],
    profile: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": EVALUATION_META_SCHEMA,
        "evaluation_id": evaluation_id,
        "generation_run_id": generation["generation_run_id"],
        "scope": generation["scope"],
        "model_key": generation["model_key"],
        "role": generation["role"],
        "seed": generation["seed"],
        "split": generation["split"],
        "sealing_status": generation["sealing_status"],
        "scientific_eligible": generation["scientific_eligible"],
        "ordered_conditions": list(generation["conditions"]),
        "ordered_query_ids": list(generation["query_ids"]),
        "ordered_query_ids_sha256": _ordered_id_hash(generation["query_ids"]),
        "fixed_denominator_query_count": len(generation["query_ids"]),
        "infra_exclusion_ids": [],
        "invalid_output_policy": "empty-prediction-retain-denominator/v1",
        "resolved_profile_sha256": canonical_sha256(profile),
        "contract_versions": {
            "strict_parser": "canonical-quad-json/v1",
            "normalizer": "canonical-quad-json/v1",
            "tuple_assignment": ASSIGNMENT_VERSION,
            "metrics": METRIC_SCHEMA_VERSION,
            "similarity": SIMILARITY_VERSION,
            "flip": "stage1-flip-table/v1",
            "sufficient_stats": "stage1-evaluation-summary/v1",
        },
        "id_inputs": copy.deepcopy(dict(id_inputs)),
        "per_condition_sufficient_stats_sha256": {
            condition: canonical_sha256(list(rows))
            for condition, rows in evaluated.items()
        },
    }


def build_free_evaluation_artifact(
    *,
    generation_run_ref: str | Path,
    evaluation_profile: str | Path | Mapping[str, Any],
    write_ref: str | Path,
    split: str,
    sealed: bool | None = None,
    target_root: str | Path | None = None,
    workspace_root: str | Path = REPOSITORY_ROOT,
) -> dict[str, Any]:
    profile = resolve_evaluation_profile(evaluation_profile)
    locator, generation, generation_target = validate_generation_ref(
        generation_run_ref, workspace_root=workspace_root
    )
    if split != generation["split"]:
        raise Stage1ArtifactError("--split assertion differs from generation target")
    derived_sealed = generation["sealing_status"] == "sealed-test"
    if sealed is not None and sealed is not derived_sealed:
        raise Stage1ArtifactError("--sealed assertion differs from derived sealing status")
    if profile["ordered_conditions"] != generation["conditions"]:
        raise Stage1ArtifactError("evaluation/generation condition frame mismatch")
    generation_dependency = portable_dependency(locator, generation_target, workspace_root)
    id_inputs = _evaluation_id_inputs(
        generation_dependency=generation_dependency,
        generation=generation,
        profile=profile,
    )
    evaluation_id = "eval-" + canonical_sha256(id_inputs)
    evaluated, summaries = _evaluate_generation(generation, profile)
    meta = _evaluation_meta(
        evaluation_id=evaluation_id,
        id_inputs=id_inputs,
        generation=generation,
        evaluated=evaluated,
        profile=profile,
    )
    provenance = {
        "schema_version": "stage1-evaluation-provenance/v1",
        "evaluation_id": evaluation_id,
        "generation_dependency": generation_dependency,
        "context_dependency": generation["context_dependency"],
        "evaluation_profile_sha256": canonical_sha256(profile),
        "evaluator_code_sha256": _evaluation_code_sha256(),
        "observed_outputs_in_lifecycle_id": False,
        "upstream_mutation_allowed": False,
    }
    root = Path(target_root).resolve() if target_root is not None else generation_target.parent.parent
    parent = root / "evaluations"
    target = parent / evaluation_id
    staging = new_staging_directory(parent, evaluation_id)
    upstream_hash = locator["payload_manifest_sha256"]
    try:
        write_canonical_json(staging / "generation_run_ref.json", generation_dependency)
        write_canonical_json(staging / "context_ref.json", generation["context_dependency"])
        write_canonical_json(staging / "evaluation_profile.resolved.json", profile)
        write_canonical_json(staging / "evaluation.meta.json", meta)
        write_canonical_json(staging / "provenance.json", provenance)
        for condition, rows in evaluated.items():
            _write_ordered_jsonl(
                staging / "per_query_metrics" / f"{condition}.jsonl", rows
            )
        write_canonical_json(staging / "summary.json", summaries)
        payload_hash = finalize_target_atomic(
            staging,
            target,
            validate_staging=lambda candidate: validate_evaluation_target(
                candidate,
                workspace_root=workspace_root,
                generation_target=generation_target,
                require_directory_name=False,
            ),
        )
    except (TrainingArtifactError, OSError) as exc:
        if staging.exists():
            shutil.rmtree(staging)
        raise _artifact_error(exc) from exc
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    _assert_target_unchanged(generation_target, upstream_hash, label="generation target")
    try:
        return write_locator_ref(
            write_ref,
            artifact_kind=EVALUATION_KIND,
            artifact_id=evaluation_id,
            target=target,
            payload_manifest_sha256=payload_hash,
        )
    except TrainingArtifactError as exc:
        raise _artifact_error(exc) from exc


def validate_evaluation_target(
    target_dir: str | Path,
    *,
    workspace_root: str | Path = REPOSITORY_ROOT,
    generation_target: str | Path | None = None,
    require_directory_name: bool = True,
) -> dict[str, Any]:
    target = Path(target_dir)
    try:
        validate_payload_manifest(target)
    except TrainingArtifactError as exc:
        raise _artifact_error(exc) from exc
    meta = load_json(target / "evaluation.meta.json")
    if not isinstance(meta, Mapping) or meta.get("schema_version") != EVALUATION_META_SCHEMA:
        raise Stage1ArtifactError("evaluation meta has wrong schema")
    try:
        validate_json_schema(
            meta, SCHEMA_ROOT / "stage1_evaluation_artifact_v1.schema.json"
        )
    except TrainingArtifactError as exc:
        raise _artifact_error(exc) from exc
    conditions = meta.get("ordered_conditions")
    expected_files = {
        "generation_run_ref.json",
        "context_ref.json",
        "evaluation_profile.resolved.json",
        "evaluation.meta.json",
        "provenance.json",
        "summary.json",
        "payload_manifest.json",
        *(f"per_query_metrics/{condition}.jsonl" for condition in conditions or []),
    }
    try:
        ensure_exact_file_set(target, expected_files)
    except TrainingArtifactError as exc:
        raise _artifact_error(exc) from exc
    dependency = load_json(target / "generation_run_ref.json")
    try:
        validate_dependency_ref(dependency, expected_kind=GENERATION_KIND)
        upstream = Path(generation_target) if generation_target is not None else resolve_dependency_target(
            dependency, workspace_root
        )
    except TrainingArtifactError as exc:
        raise _artifact_error(exc) from exc
    if upstream.name != dependency["artifact_id"] or validate_payload_manifest(upstream) != dependency[
        "payload_manifest_sha256"
    ]:
        raise Stage1ArtifactError("evaluation generation dependency mismatch")
    generation = validate_generation_target(upstream, workspace_root=workspace_root)
    profile = resolve_evaluation_profile(load_json(target / "evaluation_profile.resolved.json"))
    if load_json(target / "context_ref.json") != generation["context_dependency"]:
        raise Stage1ArtifactError("evaluation context dependency mismatch")
    id_inputs = _evaluation_id_inputs(
        generation_dependency=dependency, generation=generation, profile=profile
    )
    evaluation_id = "eval-" + canonical_sha256(id_inputs)
    if require_directory_name and target.name != evaluation_id:
        raise Stage1ArtifactError("evaluation target directory/ID mismatch")
    evaluated, summaries = _evaluate_generation(generation, profile)
    for condition, expected in evaluated.items():
        metrics_path = target / "per_query_metrics" / f"{condition}.jsonl"
        actual = load_jsonl(metrics_path)
        if actual != expected:
            raise Stage1ArtifactError(f"evaluation {condition} cannot be replayed")
        _assert_ordered_jsonl(
            metrics_path, expected, label=f"evaluation {condition} metrics"
        )
        for row in actual:
            try:
                validate_json_schema(row, SCHEMA_ROOT / "stage1_evaluation_v1.schema.json")
            except TrainingArtifactError as exc:
                raise _artifact_error(exc) from exc
    if load_json(target / "summary.json") != summaries:
        raise Stage1ArtifactError("evaluation summary cannot be replayed")
    _assert_canonical_json(target / "summary.json", summaries, label="evaluation summary")
    expected_meta = _evaluation_meta(
        evaluation_id=evaluation_id,
        id_inputs=id_inputs,
        generation=generation,
        evaluated=evaluated,
        profile=profile,
    )
    if meta != expected_meta:
        raise Stage1ArtifactError("evaluation meta cannot be replayed")
    _assert_canonical_json(target / "evaluation.meta.json", expected_meta, label="evaluation meta")
    _assert_canonical_json(
        target / "evaluation_profile.resolved.json",
        profile,
        label="evaluation resolved profile",
    )
    _assert_canonical_json(
        target / "generation_run_ref.json", dependency, label="evaluation generation ref"
    )
    _assert_canonical_json(
        target / "context_ref.json",
        generation["context_dependency"],
        label="evaluation context ref",
    )
    expected_provenance = {
        "schema_version": "stage1-evaluation-provenance/v1",
        "evaluation_id": evaluation_id,
        "generation_dependency": dependency,
        "context_dependency": generation["context_dependency"],
        "evaluation_profile_sha256": canonical_sha256(profile),
        "evaluator_code_sha256": _evaluation_code_sha256(),
        "observed_outputs_in_lifecycle_id": False,
        "upstream_mutation_allowed": False,
    }
    if load_json(target / "provenance.json") != expected_provenance:
        raise Stage1ArtifactError("evaluation provenance cannot be replayed")
    _assert_canonical_json(
        target / "provenance.json", expected_provenance, label="evaluation provenance"
    )
    return {
        "evaluation_id": evaluation_id,
        "generation_run_id": generation["generation_run_id"],
        "scope": generation["scope"],
        "model_key": generation["model_key"],
        "role": generation["role"],
        "seed": generation["seed"],
        "split": generation["split"],
        "sealing_status": generation["sealing_status"],
        "scientific_eligible": generation["scientific_eligible"],
        "conditions": generation["conditions"],
        "query_ids": generation["query_ids"],
        "records": evaluated,
        "summaries": summaries,
        "registry_dependency": generation["registry_dependency"],
        "training_plan_dependency": generation["training_plan_dependency"],
        "context_dependency": generation["context_dependency"],
        "control_dependency": generation["control_dependency"],
        "payload_manifest_sha256": sha256_file(target / "payload_manifest.json"),
    }


def validate_evaluation_ref(
    evaluation_ref: str | Path,
    *,
    workspace_root: str | Path = REPOSITORY_ROOT,
) -> tuple[dict[str, Any], dict[str, Any], Path]:
    try:
        locator, target = resolve_locator_ref(evaluation_ref, EVALUATION_KIND)
    except TrainingArtifactError as exc:
        raise _artifact_error(exc) from exc
    report = validate_evaluation_target(target, workspace_root=workspace_root)
    if locator["artifact_id"] != report["evaluation_id"] or locator[
        "payload_manifest_sha256"
    ] != report["payload_manifest_sha256"]:
        raise Stage1ArtifactError("evaluation locator mismatch")
    return locator, report, target


def _write_ordered_jsonl(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    write_bytes_atomic(
        path,
        b"".join(canonical_json_bytes(dict(row)) + b"\n" for row in rows),
    )


def _margin_id_inputs_valid(
    id_inputs: Mapping[str, Any],
    *,
    dependencies: Mapping[str, Any],
    meta: Mapping[str, Any],
    profile: Mapping[str, Any],
) -> None:
    required = {
        "schema_version": "stage1-margin-id-inputs/v1",
        "training_plan_dependency": dependencies["training_plan"],
        "model_registry_dependency": dependencies["registry"],
        "model_key": meta["model_key"],
        "model_dependency": dependencies["model"],
        "context_dependency": dependencies["context"],
        "control_dependency": dependencies["control"],
        "cf_dependency": dependencies["cf"],
        "split": meta["split"],
        "sealing_status": meta["sealing_status"],
        "ordered_conditions": meta["ordered_conditions"],
        "expected_ordered_query_ids_sha256": meta["ordered_query_ids_sha256"],
        "scorer_profile_sha256": canonical_sha256(profile),
        "eligibility_mask_schema": "stage1-prefrozen-field-mask/v1",
        "eligibility_mask_sha256": canonical_sha256(meta["field_eligibility_masks"]),
    }
    expected_keys = set(required) | {
        "serializer_schema_sha256",
        "tokenizer_contract_sha256",
        "span_mask_contract_sha256",
        "aggregation_contract_sha256",
        "runtime_contract_sha256",
        "batch_calibration_contract_sha256",
        "scorer_code_sha256",
    }
    if set(id_inputs) != expected_keys:
        raise Stage1ArtifactError(
            "margin ID inputs are missing, extra, or contain observed outcomes"
        )
    for key, expected in required.items():
        if id_inputs.get(key) != expected:
            raise Stage1ArtifactError(f"margin ID input {key} mismatch")
    for key in (
        "serializer_schema_sha256", "tokenizer_contract_sha256",
        "span_mask_contract_sha256", "aggregation_contract_sha256",
        "runtime_contract_sha256", "batch_calibration_contract_sha256",
        "scorer_code_sha256",
    ):
        _strict_hash(id_inputs.get(key), label=f"margin {key}")


def validate_margin_target(
    target_dir: str | Path,
    *,
    workspace_root: str | Path = REPOSITORY_ROOT,
    require_directory_name: bool = True,
    model_dependency_resolver: Callable[..., Any] | None = None,
    tokenizer_loader: Callable[[Path], Any] | None = None,
) -> dict[str, Any]:
    """Deep, read-only replay validator for an immutable margin target."""

    # Imported lazily to avoid a module cycle: the publisher itself calls this
    # validator before atomic finalize.
    from data.build_context_manifest import validate_context_target
    from data.control_manifest import validate_control_target
    from metrics.stage1_margin_lifecycle import (
        CALIBRATION_PAIR_TARGET,
        CALIBRATION_RECORD_SCHEMA,
        CALIBRATION_SCHEMA,
        CALIBRATION_TOLERANCE_FLOOR,
        CALIBRATION_TOLERANCE_HARD_CAP,
        CALIBRATION_TOLERANCE_MULTIPLIER,
        CALIBRATION_TRAVERSAL,
        LOCAL_EXECUTOR_DESCRIPTOR,
        _batch_contract,
        _cf_frame,
        _contract_hashes,
        _contrast_rows,
        _coverage,
        _same_dependency_identity,
        _summary,
    )

    target = Path(target_dir)
    try:
        validate_payload_manifest(target)
    except TrainingArtifactError as exc:
        raise _artifact_error(exc) from exc
    meta = load_json(target / "margin.meta.json")
    if not isinstance(meta, Mapping) or meta.get("schema_version") != "stage1-margin-run/v1":
        raise Stage1ArtifactError("margin meta has wrong schema")
    try:
        validate_json_schema(meta, SCHEMA_ROOT / "stage1_margin_artifact_v1.schema.json")
    except TrainingArtifactError as exc:
        raise _artifact_error(exc) from exc
    conditions = meta.get("ordered_conditions")
    if conditions != list(CONDITIONS):
        raise Stage1ArtifactError("margin condition frame is not exact")
    expected_files = {
        "margin.meta.json", "scorer_profile.resolved.json", "runtime_contract.json",
        "model_registry_ref.json", "training_plan_ref.json", "model_ref.json",
        "context_ref.json", "control_ref.json", "cf_ref.json", "provenance.json",
        "summary.json", "query_contrasts.jsonl", "batch_calibration.jsonl",
        "payload_manifest.json",
        *(f"margins/{condition}.jsonl" for condition in conditions),
        *(f"tuple_scores/{condition}.jsonl" for condition in conditions),
    }
    try:
        ensure_exact_file_set(target, expected_files)
    except TrainingArtifactError as exc:
        raise _artifact_error(exc) from exc
    profile = resolve_margin_profile(load_json(target / "scorer_profile.resolved.json"))
    runtime_contract = load_json(target / "runtime_contract.json")
    if not isinstance(runtime_contract, Mapping):
        raise Stage1ArtifactError("margin runtime contract must be an object")
    try:
        validate_json_schema(
            runtime_contract, SCHEMA_ROOT / "stage1_margin_runtime_contract_v1.schema.json"
        )
    except TrainingArtifactError as exc:
        raise _artifact_error(exc) from exc
    if runtime_contract.get("executor") != meta.get("executor") or meta.get(
        "runtime_contract_sha256"
    ) != canonical_sha256(runtime_contract):
        raise Stage1ArtifactError("margin runtime/executor contract mismatch")
    expected_batch_contract = _batch_contract(profile)
    if runtime_contract.get("batch_contract") != expected_batch_contract or meta.get(
        "batch_contract"
    ) != expected_batch_contract:
        raise Stage1ArtifactError("margin batch-unit contract mismatch")
    dependencies = {
        "registry": load_json(target / "model_registry_ref.json"),
        "training_plan": load_json(target / "training_plan_ref.json"),
        "model": load_json(target / "model_ref.json"),
        "context": load_json(target / "context_ref.json"),
        "control": load_json(target / "control_ref.json"),
        "cf": load_json(target / "cf_ref.json"),
    }
    split = meta.get("split")
    sealing_status = meta.get("sealing_status")
    expected_control_kind = _assert_control_dependency_kind(
        dependencies["control"],
        split=split,
        sealing_status=sealing_status,
        label="margin",
    )
    registry_target = _dependency_target(
        dependencies["registry"], workspace_root=workspace_root, kinds=REGISTRY_KIND
    )
    registry = validate_registry_target(registry_target, workspace_root=workspace_root)
    formal_margin = (
        registry.get("registry_scope") == "formal"
        or meta.get("scientific_eligible") is True
    )
    if formal_margin and profile.get("runtime", {}).get("trust_remote_code") is not False:
        raise Stage1ArtifactError(
            "formal margin tokenizer/model loading requires trust_remote_code=false"
        )
    if formal_margin and (
        model_dependency_resolver is not None or tokenizer_loader is not None
    ):
        raise Stage1ArtifactError(
            "formal margin validation forbids injected model/tokenizer callbacks"
        )
    _dependency_target(
        dependencies["training_plan"], workspace_root=workspace_root, kinds="training-plan"
    )
    _dependency_target(
        dependencies["model"], workspace_root=workspace_root, kinds=MODEL_ARTIFACT_KIND
    )
    context_target = _dependency_target(
        dependencies["context"], workspace_root=workspace_root,
        kinds=("context", "test-context"),
    )
    control_target = _dependency_target(
        dependencies["control"],
        workspace_root=workspace_root,
        kinds=expected_control_kind,
    )
    cf_target = _dependency_target(
        dependencies["cf"], workspace_root=workspace_root, kinds="counterfactual"
    )
    if registry["training_plan_dependency"] != dependencies["training_plan"]:
        raise Stage1ArtifactError("margin registry/training-plan lineage mismatch")
    matched = [row for row in registry["models"] if row["model_key"] == meta.get("model_key")]
    if len(matched) != 1 or matched[0]["model_dependency"] != dependencies["model"]:
        raise Stage1ArtifactError("margin model binding differs from registry")
    resolver = model_dependency_resolver or resolve_registered_model_dependency
    try:
        resolved_model = resolver(
            registry_dependency=dependencies["registry"],
            model_key=str(meta.get("model_key", "")),
            workspace_root=workspace_root,
        )
    except Exception as exc:
        raise _artifact_error(exc) from exc
    if (
        resolved_model.registry_id != dependencies["registry"]["artifact_id"]
        or resolved_model.model_artifact_id != dependencies["model"]["artifact_id"]
        or resolved_model.model_key != meta.get("model_key")
        or resolved_model.role != matched[0]["role"]
        or resolved_model.seed != matched[0]["seed"]
        or bool(resolved_model.scientific_eligible)
        is not bool(matched[0]["scientific_eligible"])
    ):
        raise Stage1ArtifactError("margin registered-model resolver binding mismatch")

    def semantic_replay(
        tokenizer_path: Path,
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        if formal_margin:
            # Scientific context/control artifacts own their tokenizer source
            # contracts.  Replaying them with the registered scoring-model
            # tokenizer would reintroduce a caller-selected source.
            return (
                validate_context_target(
                    context_target, workspace_root=workspace_root
                ),
                validate_control_target(
                    control_target,
                    context_target=context_target,
                    workspace_root=workspace_root,
                ),
                validate_cf_target(cf_target, workspace_root=workspace_root),
            )
        if tokenizer_loader is not None:
            tokenizer = tokenizer_loader(tokenizer_path)
        else:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(
                str(tokenizer_path),
                local_files_only=True,
                trust_remote_code=False,
            )
        return (
            validate_context_target(
                context_target,
                tokenizer=tokenizer,
                workspace_root=workspace_root,
            ),
            validate_control_target(
                control_target,
                tokenizer=tokenizer,
                context_target=context_target,
                workspace_root=workspace_root,
            ),
            validate_cf_target(cf_target, workspace_root=workspace_root),
        )

    try:
        source_contract = getattr(resolved_model, "source_contract", None)
        if source_contract is None:
            if formal_margin:
                raise Stage1ArtifactError(
                    "formal margin resolved model lacks a typed source contract"
                )
            if tokenizer_loader is None:
                raise Stage1ArtifactError(
                    "unleased engineering tokenizer replay requires an explicit loader"
                )
            context_meta, control_report, cf_report = semantic_replay(
                Path(resolved_model.tokenizer_path).resolve()
            )
        else:
            if not isinstance(source_contract, ResolvedModelSourceContract):
                raise Stage1ArtifactError(
                    "margin resolved model source contract is not typed"
                )
            source_names = _margin_tokenizer_lease_sources(
                source_contract,
                require_full_coverage=formal_margin,
            )
            with verified_model_source_lease(
                source_contract, source_names=source_names
            ) as sources:
                tokenizer_path = sources.tokenizer_path.resolve()
                if tokenizer_path != Path(resolved_model.tokenizer_path).resolve():
                    raise Stage1ArtifactError(
                        "margin tokenizer path differs from its verified source contract"
                    )
                context_meta, control_report, cf_report = semantic_replay(
                    tokenizer_path
                )
    except Exception as exc:
        raise Stage1ArtifactError(f"margin upstream replay failed: {exc}") from exc
    if context_meta.get("split") != split or control_report.get("split") != split or cf_report.get(
        "split"
    ) != split:
        raise Stage1ArtifactError("margin context/control/CF split mismatch")
    if not _same_dependency_identity(
        load_json(control_target / "context_ref.json"), dependencies["context"]
    ) or not _same_dependency_identity(
        load_json(cf_target / "context_ref.json"), dependencies["context"]
    ):
        raise Stage1ArtifactError("margin context/control/CF lineage mismatch")
    if context_meta.get("budget", {}).get("tokenizer_revision") != resolved_model.tokenizer_revision:
        raise Stage1ArtifactError("margin context tokenizer revision mismatch")
    if meta.get("scientific_eligible") is True and (
        cf_report.get("scientific_eligible") is not True
        or context_meta.get("scientific_eligible") is not True
    ):
        raise Stage1ArtifactError("scientific margin run uses an ineligible upstream")
    _scope_matrix(
        scope=registry["registry_scope"], split=split,
        sealing_status=meta.get("sealing_status"),
        scientific_eligible=bool(meta.get("scientific_eligible")),
    )
    expected_context_kind = "test-context" if split == "test" else "context"
    if dependencies["context"]["artifact_kind"] != expected_context_kind:
        raise Stage1ArtifactError("margin context kind does not derive its sealing status")
    executor = meta.get("executor")
    if not isinstance(executor, Mapping) or runtime_contract.get("executor") != executor:
        raise Stage1ArtifactError("margin executor descriptor mismatch")
    if meta.get("scientific_eligible") is True and dict(executor) != LOCAL_EXECUTOR_DESCRIPTOR:
        raise Stage1ArtifactError("formal margin target lacks the frozen local-HF executor")

    frame = _context_frame(context_target, split=split)
    query_ids = [row["id"] for row in frame]
    if meta.get("ordered_query_ids") != query_ids or meta.get(
        "ordered_query_ids_sha256"
    ) != _ordered_id_hash(query_ids):
        raise Stage1ArtifactError("margin ordered query frame differs from context")
    cf_rows, derived_masks, tuple_counts = _cf_frame(
        cf_target=cf_target, split=split, context_frame=frame
    )
    masks = meta.get("field_eligibility_masks")
    if masks != derived_masks:
        raise Stage1ArtifactError("margin masks differ from the prediction-blind CF frame")
    id_inputs = meta.get("id_inputs")
    if not isinstance(id_inputs, Mapping):
        raise Stage1ArtifactError("margin meta lacks ID inputs")
    _margin_id_inputs_valid(
        id_inputs, dependencies=dependencies, meta=meta, profile=profile
    )
    expected_contracts = _contract_hashes(
        profile=profile, handle=resolved_model, runtime_contract=runtime_contract
    )
    if any(id_inputs.get(key) != value for key, value in expected_contracts.items()):
        raise Stage1ArtifactError("margin code/tokenizer/runtime contract hash mismatch")
    margin_id = "mgn-" + canonical_sha256(id_inputs)
    if meta.get("margin_run_id") != margin_id:
        raise Stage1ArtifactError("margin run ID cannot be recomputed")
    if require_directory_name and target.name != margin_id:
        raise Stage1ArtifactError("margin target directory/ID mismatch")

    calibration = meta.get("batch_calibration")
    calibration_rows = load_jsonl(target / "batch_calibration.jsonl")
    expected_calibration_identities = [
        {
            "id": query_id,
            "condition": condition,
            "tuple_index": tuple_index,
            "field": field,
        }
        for condition in CONDITIONS
        for query_index, query_id in enumerate(query_ids)
        for tuple_index in range(tuple_counts[query_id])
        for field in FIELDS
        if masks[field][query_index]
    ][:CALIBRATION_PAIR_TARGET]
    actual_calibration_identities = []
    for row in calibration_rows:
        try:
            validate_json_schema(
                row,
                SCHEMA_ROOT
                / "stage1_margin_batch_calibration_record_v1.schema.json",
            )
        except TrainingArtifactError as exc:
            raise _artifact_error(exc) from exc
        if row.get("schema_version") != CALIBRATION_RECORD_SCHEMA:
            raise Stage1ArtifactError("margin calibration row schema mismatch")
        actual_calibration_identities.append(
            {
                "id": row["id"],
                "condition": row["condition"],
                "tuple_index": row["tuple_index"],
                "field": row["field"],
            }
        )
        numeric_keys = (
            "batched_gold_mean_logprob",
            "batched_counterfactual_mean_logprob",
            "singleton_gold_mean_logprob",
            "singleton_counterfactual_mean_logprob",
            "batched_margin",
            "singleton_margin",
            "abs_margin_delta",
        )
        if any(not math.isfinite(float(row[key])) for key in numeric_keys):
            raise Stage1ArtifactError("margin calibration row is non-finite")
        batched_margin = row["batched_gold_mean_logprob"] - row[
            "batched_counterfactual_mean_logprob"
        ]
        singleton_margin = row["singleton_gold_mean_logprob"] - row[
            "singleton_counterfactual_mean_logprob"
        ]
        if (
            row["batched_margin"] != batched_margin
            or row["singleton_margin"] != singleton_margin
            or row["abs_margin_delta"]
            != abs(batched_margin - singleton_margin)
            or row["compared_token_count"]
            != len(row["gold_token_ids"]) + len(row["counterfactual_token_ids"])
        ):
            raise Stage1ArtifactError("margin calibration arithmetic cannot be replayed")
    if actual_calibration_identities != expected_calibration_identities:
        raise Stage1ArtifactError(
            "margin calibration did not use the deterministic first-100 frame"
        )
    if registry["registry_scope"] == "formal" and len(
        calibration_rows
    ) != CALIBRATION_PAIR_TARGET:
        raise Stage1ArtifactError(
            "formal margin target lacks 100 calibration pairs"
        )
    _assert_ordered_jsonl(
        target / "batch_calibration.jsonl",
        calibration_rows,
        label="margin batch calibration",
    )
    maximum_delta = max(float(row["abs_margin_delta"]) for row in calibration_rows)
    frozen_tolerance = max(
        CALIBRATION_TOLERANCE_FLOOR,
        CALIBRATION_TOLERANCE_MULTIPLIER * maximum_delta,
    )
    if frozen_tolerance > CALIBRATION_TOLERANCE_HARD_CAP:
        raise Stage1ArtifactError(
            "margin calibration tolerance exceeds the 5e-3 hard cap"
        )
    calibration_hash = canonical_sha256(calibration_rows)
    expected_calibration = {
        "schema_version": CALIBRATION_SCHEMA,
        "passed": True,
        "traversal_policy": CALIBRATION_TRAVERSAL,
        "requested_pair_count": CALIBRATION_PAIR_TARGET,
        "observed_pair_count": len(calibration_rows),
        "requested_pair_count_reached": len(calibration_rows)
        == CALIBRATION_PAIR_TARGET,
        "calibration_records_sha256": calibration_hash,
        "tolerance_floor": CALIBRATION_TOLERANCE_FLOOR,
        "tolerance_multiplier": CALIBRATION_TOLERANCE_MULTIPLIER,
        "tolerance_hard_cap": CALIBRATION_TOLERANCE_HARD_CAP,
        "observed_max_abs_margin_delta": maximum_delta,
        "frozen_tolerance": frozen_tolerance,
        "compared_token_count": sum(
            int(row["compared_token_count"]) for row in calibration_rows
        ),
    }
    if calibration != expected_calibration or meta.get(
        "batch_calibration_records_sha256"
    ) != calibration_hash:
        raise Stage1ArtifactError(
            "margin batch/unbatch calibration summary cannot be replayed"
        )

    frame_hashes = {row["id"]: row for row in frame}
    rows_by_condition: dict[str, list[dict[str, Any]]] = {}
    tuple_rows_by_condition: dict[str, list[dict[str, Any]]] = {}
    expected_margin_keys = [(query_id, field) for query_id in query_ids for field in FIELDS]
    expected_tuple_keys = [
        (query_id, tuple_index, field)
        for query_index, query_id in enumerate(query_ids)
        for tuple_index in range(tuple_counts[query_id])
        for field in FIELDS
        if masks[field][query_index]
    ]

    def validate_score(score: Any) -> None:
        if not isinstance(score, Mapping) or set(score) != {
            "token_ids", "token_count", "character_span", "token_span",
            "sum_logprob", "mean_logprob", "left_boundary_crossing",
            "right_boundary_crossing",
        }:
            raise Stage1ArtifactError("tuple score audit has non-canonical fields")
        if (
            not isinstance(score["token_ids"], list)
            or not score["token_ids"]
            or score["token_count"] != len(score["token_ids"])
            or any(not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in score["token_ids"])
            or not isinstance(score["character_span"], list)
            or len(score["character_span"]) != 2
            or score["character_span"][0] >= score["character_span"][1]
            or not isinstance(score["token_span"], list)
            or len(score["token_span"]) != 2
            or score["token_span"][0] >= score["token_span"][1]
            or score["token_span"][1] - score["token_span"][0] != score["token_count"]
            or any(not isinstance(score[key], bool) for key in ("left_boundary_crossing", "right_boundary_crossing"))
        ):
            raise Stage1ArtifactError("tuple score audit shape is invalid")
        total = float(score["sum_logprob"])
        mean = float(score["mean_logprob"])
        if not math.isfinite(total) or not math.isfinite(mean) or not math.isclose(
            mean, total / score["token_count"], rel_tol=1e-12, abs_tol=1e-12
        ):
            raise Stage1ArtifactError("tuple score audit arithmetic is invalid")

    for condition in conditions:
        tuple_path = target / "tuple_scores" / f"{condition}.jsonl"
        tuple_rows = load_jsonl(tuple_path)
        tuple_keys = [
            (str(row.get("id", "")), int(row.get("tuple_index", -1)), str(row.get("field", "")))
            for row in tuple_rows
        ]
        if tuple_keys != expected_tuple_keys:
            raise Stage1ArtifactError(f"margin {condition} changed the tuple/field score frame")
        for row in tuple_rows:
            try:
                validate_json_schema(
                    row, SCHEMA_ROOT / "stage1_margin_tuple_score_v1.schema.json"
                )
            except TrainingArtifactError as exc:
                raise _artifact_error(exc) from exc
            if row["condition"] != condition or row["model_key"] != meta["model_key"] or row[
                "status"
            ] != "ok":
                raise Stage1ArtifactError("tuple score identity mismatch")
            key = (row["id"], row["tuple_index"], row["field"])
            cf_row = cf_rows[key]
            if row["selected_cf_id"] != cf_row["selected_cf_id"] or row[
                "cf_record_sha256"
            ] != cf_row["record_sha256"]:
                raise Stage1ArtifactError("tuple score/CF identity mismatch")
            frozen = frame_hashes[row["id"]]
            for hash_key in ("content_sha256", "gold_sha256", "context_record_sha256"):
                if row[hash_key] != frozen[hash_key]:
                    raise Stage1ArtifactError("tuple score/context frame hash mismatch")
            validate_score(row["gold"])
            validate_score(row["counterfactual"])
            expected_mean = row["gold"]["mean_logprob"] - row["counterfactual"]["mean_logprob"]
            expected_sum = row["gold"]["sum_logprob"] - row["counterfactual"]["sum_logprob"]
            if row["mean_margin"] != expected_mean or row[
                "sum_margin_sensitivity"
            ] != expected_sum:
                raise Stage1ArtifactError("tuple margin cannot be replayed from field scores")
        _assert_ordered_jsonl(tuple_path, tuple_rows, label=f"margin {condition} tuple scores")
        tuple_rows_by_condition[condition] = tuple_rows

        margin_path = target / "margins" / f"{condition}.jsonl"
        rows = load_jsonl(margin_path)
        keys = [(str(row.get("id", "")), str(row.get("field", ""))) for row in rows]
        if keys != expected_margin_keys:
            raise Stage1ArtifactError(f"margin {condition} changed ordered query/field frame")
        grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
        for tuple_row in tuple_rows:
            grouped[(tuple_row["id"], tuple_row["field"])].append(tuple_row)
        for row in rows:
            try:
                validate_json_schema(row, SCHEMA_ROOT / "stage1_margin_record_v1.schema.json")
            except TrainingArtifactError as exc:
                raise _artifact_error(exc) from exc
            if row["condition"] != condition or row["model_key"] != meta["model_key"]:
                raise Stage1ArtifactError("margin row identity mismatch")
            index = query_ids.index(row["id"])
            eligible = bool(masks[row["field"]][index])
            details = grouped[(row["id"], row["field"])]
            if eligible:
                expected_margin = sum(float(item["mean_margin"]) for item in details) / len(details)
                expected_cf_hash = canonical_sha256([item["cf_record_sha256"] for item in details])
                if row["status"] != "ok" or row["margin_mean"] != expected_margin or row[
                    "cf_record_sha256"
                ] != expected_cf_hash:
                    raise Stage1ArtifactError("query margin cannot be replayed from tuple scores")
            elif details or row["status"] != "ineligible-pre-frozen" or row[
                "margin_mean"
            ] is not None or row["cf_record_sha256"] is not None:
                raise Stage1ArtifactError("ineligible margin row is not explicitly frozen")
            frozen = frame_hashes[row["id"]]
            for hash_key in ("content_sha256", "gold_sha256", "context_record_sha256"):
                if row[hash_key] != frozen[hash_key]:
                    raise Stage1ArtifactError("margin/context frame hashes differ")
        _assert_ordered_jsonl(margin_path, rows, label=f"margin {condition} sufficient statistics")
        rows_by_condition[condition] = rows

    expected_tuple_count = len(expected_tuple_keys)
    if meta.get("expected_row_count_per_condition") != len(expected_margin_keys) or meta.get(
        "observed_row_count_per_condition"
    ) != {condition: len(rows_by_condition[condition]) for condition in conditions}:
        raise Stage1ArtifactError("margin row-count meta mismatch")
    if meta.get("expected_tuple_score_count_per_condition") != expected_tuple_count or meta.get(
        "observed_tuple_score_count_per_condition"
    ) != {condition: len(tuple_rows_by_condition[condition]) for condition in conditions}:
        raise Stage1ArtifactError("margin tuple-score count meta mismatch")
    construction_coverage, scoring_coverage = _coverage(
        masks,
        context_frame=frame,
        tuple_counts=tuple_counts,
        cf_rows=cf_rows,
        rows_by_condition=rows_by_condition,
        tuple_rows_by_condition=tuple_rows_by_condition,
    )
    sufficient_hashes = {
        condition: canonical_sha256(rows_by_condition[condition]) for condition in conditions
    }
    tuple_hashes = {
        condition: canonical_sha256(tuple_rows_by_condition[condition]) for condition in conditions
    }
    if meta.get("construction_coverage") != construction_coverage:
        raise Stage1ArtifactError("margin construction coverage cannot be replayed")
    if meta.get("scoring_coverage") != scoring_coverage:
        raise Stage1ArtifactError("margin scoring coverage cannot be replayed")
    if meta.get("per_condition_sufficient_stats_sha256") != sufficient_hashes or meta.get(
        "per_condition_tuple_scores_sha256"
    ) != tuple_hashes:
        raise Stage1ArtifactError("margin sufficient-stat hashes cannot be replayed")

    contrast_rows = load_jsonl(target / "query_contrasts.jsonl")
    expected_contrasts = _contrast_rows(
        model_key=meta["model_key"], context_frame=frame, masks=masks,
        rows_by_condition=rows_by_condition,
    )
    if contrast_rows != expected_contrasts or meta.get(
        "query_contrasts_sha256"
    ) != canonical_sha256(expected_contrasts):
        raise Stage1ArtifactError("margin query contrasts cannot be replayed")
    for row in contrast_rows:
        try:
            validate_json_schema(
                row, SCHEMA_ROOT / "stage1_margin_query_contrast_v1.schema.json"
            )
        except TrainingArtifactError as exc:
            raise _artifact_error(exc) from exc
    _assert_ordered_jsonl(
        target / "query_contrasts.jsonl", expected_contrasts,
        label="margin query contrasts",
    )
    summary = _summary(
        model_key=meta["model_key"], context_frame=frame, masks=masks,
        rows_by_condition=rows_by_condition, contrast_rows=expected_contrasts,
        construction=construction_coverage, scoring=scoring_coverage,
    )
    if load_json(target / "summary.json") != summary:
        raise Stage1ArtifactError("margin summary cannot be replayed")
    _assert_canonical_json(target / "summary.json", summary, label="margin summary")
    _assert_canonical_json(target / "margin.meta.json", meta, label="margin meta")
    _assert_canonical_json(target / "scorer_profile.resolved.json", profile, label="margin scorer profile")
    _assert_canonical_json(target / "runtime_contract.json", runtime_contract, label="margin runtime contract")
    for filename, key in (
        ("model_registry_ref.json", "registry"), ("training_plan_ref.json", "training_plan"),
        ("model_ref.json", "model"), ("context_ref.json", "context"),
        ("control_ref.json", "control"), ("cf_ref.json", "cf"),
    ):
        _assert_canonical_json(target / filename, dependencies[key], label=f"margin {filename}")
    expected_provenance = {
        "schema_version": "stage1-margin-provenance/v1",
        "margin_run_id": margin_id,
        "dependencies": copy.deepcopy(dependencies),
        "model_key": meta["model_key"],
        "model_resolver": {
            "protocol": "model.stage1_registry.resolve_registered_model_dependency/v1",
            "accepted_source": "embedded-stage1-model-registry-dependency-only",
            "direct_model_path_allowed": False,
        },
        "executor": dict(executor),
        "scorer_profile_sha256": canonical_sha256(profile),
        "runtime_contract_sha256": canonical_sha256(runtime_contract),
        "batch_calibration_contract_sha256": id_inputs["batch_calibration_contract_sha256"],
        "scorer_code_sha256": id_inputs["scorer_code_sha256"],
        "observed_margins_in_lifecycle_id": False,
        "upstream_mutation_allowed": False,
    }
    if load_json(target / "provenance.json") != expected_provenance:
        raise Stage1ArtifactError("margin provenance mismatch")
    _assert_canonical_json(target / "provenance.json", expected_provenance, label="margin provenance")
    return {
        "margin_run_id": margin_id,
        "scope": registry["registry_scope"],
        "scientific_eligible": meta["scientific_eligible"],
        "split": split,
        "sealing_status": meta["sealing_status"],
        "model_key": meta["model_key"],
        "role": matched[0]["role"],
        "seed": matched[0]["seed"],
        "conditions": conditions,
        "query_ids": query_ids,
        "field_masks": {field: list(masks[field]) for field in FIELDS},
        "records": rows_by_condition,
        "registry_dependency": dependencies["registry"],
        "training_plan_dependency": dependencies["training_plan"],
        "context_dependency": dependencies["context"],
        "control_dependency": dependencies["control"],
        "cf_dependency": dependencies["cf"],
        "payload_manifest_sha256": sha256_file(target / "payload_manifest.json"),
    }


def validate_margin_ref(
    margin_ref: str | Path,
    *,
    workspace_root: str | Path = REPOSITORY_ROOT,
) -> tuple[dict[str, Any], dict[str, Any], Path]:
    try:
        locator, target = resolve_locator_ref(margin_ref, MARGIN_KIND)
    except TrainingArtifactError as exc:
        raise _artifact_error(exc) from exc
    report = validate_margin_target(target, workspace_root=workspace_root)
    if locator["artifact_id"] != report["margin_run_id"] or locator[
        "payload_manifest_sha256"
    ] != report["payload_manifest_sha256"]:
        raise Stage1ArtifactError("margin locator mismatch")
    return locator, report, target


BEHAVIOR_ENDPOINTS = (
    "tuple/f1_avg",
    "field_unbound/target/similarity",
    "field_unbound/argument/similarity",
    "field_bound/targeted_group/f1",
)
FLIP_ENDPOINTS = (
    "tuple/hard",
    "tuple/soft",
    "field_unbound/target",
    "field_unbound/argument",
    "field_bound/targeted_group",
    "field_bound/hateful",
    "format/strict",
    "tuple_count",
)
FLIP_CONTRASTS = (
    ("C0", "CL", "CL-C0"),
    ("C0", "CD", "CD-C0"),
    ("C0", "CLD", "CLD-C0"),
    ("PL", "CL", "CL-PL"),
    ("PD", "CD", "CD-PD"),
)


def _endpoint_values(
    records: Sequence[Mapping[str, Any]], endpoint: str
) -> list[float | tuple[int, ...]]:
    """Project frozen records to the minimal sufficient statistic per query."""

    if endpoint == "tuple/f1_avg":
        return [
            (
                int(row["tuple"]["hard"]["tp"]),
                int(row["tuple"]["hard"]["fp"]),
                int(row["tuple"]["hard"]["fn"]),
                int(row["tuple"]["soft"]["tp"]),
                int(row["tuple"]["soft"]["fp"]),
                int(row["tuple"]["soft"]["fn"]),
            )
            for row in records
        ]
    if endpoint == "field_unbound/target/similarity":
        return [float(row["field_unbound"]["target"]["similarity"]) for row in records]
    if endpoint == "field_unbound/argument/similarity":
        return [float(row["field_unbound"]["argument"]["similarity"]) for row in records]
    if endpoint == "field_bound/targeted_group/f1":
        return [
            (
                int(row["field_bound"]["targeted_group"]["tp"]),
                int(row["field_bound"]["targeted_group"]["fp"]),
                int(row["field_bound"]["targeted_group"]["fn"]),
            )
            for row in records
        ]
    raise Stage1ArtifactError(f"unsupported behavior endpoint {endpoint}")


def _reduce_endpoint_values(
    values: Sequence[float | tuple[int, ...]], endpoint: str
) -> float:
    if not values:
        raise Stage1ArtifactError("behavior endpoint frame is empty")
    if endpoint in {
        "field_unbound/target/similarity",
        "field_unbound/argument/similarity",
    }:
        result = sum(float(value) for value in values) / len(values)
    elif endpoint == "tuple/f1_avg":
        counts = [sum(int(value[index]) for value in values) for index in range(6)]
        hard_denominator = 2 * counts[0] + counts[1] + counts[2]
        soft_denominator = 2 * counts[3] + counts[4] + counts[5]
        hard_f1 = 2 * counts[0] / hard_denominator if hard_denominator else 0.0
        soft_f1 = 2 * counts[3] / soft_denominator if soft_denominator else 0.0
        result = (hard_f1 + soft_f1) / 2.0
    elif endpoint == "field_bound/targeted_group/f1":
        counts = [sum(int(value[index]) for value in values) for index in range(3)]
        denominator = 2 * counts[0] + counts[1] + counts[2]
        result = 2 * counts[0] / denominator if denominator else 0.0
    else:  # pragma: no cover - guarded by profile resolution and projection
        raise Stage1ArtifactError(f"unsupported behavior endpoint {endpoint}")
    if not math.isfinite(result):
        raise Stage1ArtifactError(f"non-finite behavior endpoint {endpoint}")
    return result


def _load_run_ref_map(
    source: str | Path | Mapping[str, Any], *, base: Path | None = None
) -> list[dict[str, Any]]:
    value = _object(source, name="analysis run-ref map")
    try:
        validate_json_schema(value, SCHEMA_ROOT / "stage1_run_ref_map_v1.schema.json")
    except TrainingArtifactError as exc:
        raise _artifact_error(exc) from exc
    if value.get("schema_version") != "stage1-run-ref-map/v1":
        raise Stage1ArtifactError("run-ref map has wrong schema")
    entries = value.get("entries")
    if not isinstance(entries, list) or not entries:
        raise Stage1ArtifactError("run-ref map entries must be non-empty")
    root = base or (Path(source).resolve().parent if not isinstance(source, Mapping) else REPOSITORY_ROOT)
    result = []
    for entry in entries:
        if not isinstance(entry, Mapping) or set(entry) != {
            "model_key", "evaluation_ref", "margin_ref"
        }:
            raise Stage1ArtifactError("run-ref entry has non-canonical fields")
        row = dict(entry)
        for key in ("evaluation_ref", "margin_ref"):
            raw = row[key]
            if not isinstance(raw, str) or not raw:
                raise Stage1ArtifactError(f"run-ref entry lacks {key}")
            path = Path(raw)
            row[key] = str(path if path.is_absolute() else (root / path).resolve())
        result.append(row)
    keys = [str(row["model_key"]) for row in result]
    if len(keys) != len(set(keys)):
        raise Stage1ArtifactError("run-ref map has duplicate model keys")
    return result


def _analysis_inputs(
    *,
    registry_ref: str | Path,
    run_ref_map: str | Path | Mapping[str, Any],
    profile: Mapping[str, Any],
    split: str,
    sealed: bool | None,
    workspace_root: str | Path,
) -> dict[str, Any]:
    registry_locator, registry, registry_target = validate_registry_ref(
        registry_ref, workspace_root=workspace_root
    )
    mode = profile["mode"]
    _scope_matrix(
        scope=registry["registry_scope"],
        split=split,
        sealing_status="sealed-test" if split == "test" else "unsealed-dev",
        scientific_eligible=bool(profile["scientific_eligible"]),
        mode=mode,
    )
    if split not in profile.get("allowed_splits", []):
        raise Stage1ArtifactError("analysis profile does not allow requested split")
    derived_sealed = split == "test"
    if sealed is not None and sealed is not derived_sealed:
        raise Stage1ArtifactError("--sealed assertion differs from analysis split")
    entries = _load_run_ref_map(run_ref_map)
    expected_keys = list(registry["models"])
    expected_model_keys = [row["model_key"] for row in expected_keys]
    if mode == "pilot":
        declared = profile.get("required_model_keys")
        if declared != expected_model_keys:
            raise Stage1ArtifactError("pilot profile/registry model keys are not exact")
    entry_keys = [row["model_key"] for row in entries]
    if entry_keys != expected_model_keys:
        raise Stage1ArtifactError("run refs must follow and exactly cover registry slot order")
    registry_dependency = portable_dependency(
        registry_locator, registry_target, workspace_root
    )
    evaluations: dict[str, dict[str, Any]] = {}
    margins: dict[str, dict[str, Any]] = {}
    evaluation_deps: list[dict[str, Any]] = []
    margin_deps: list[dict[str, Any]] = []
    targets: list[tuple[Path, str, str]] = [(registry_target, registry_locator["payload_manifest_sha256"], "registry")]
    lineage: dict[str, Any] | None = None
    query_ids: list[str] | None = None
    field_masks: Mapping[str, Sequence[bool]] | None = None
    for entry, model in zip(entries, registry["models"], strict=True):
        evaluation_locator, evaluation, evaluation_target = validate_evaluation_ref(
            entry["evaluation_ref"], workspace_root=workspace_root
        )
        margin_locator, margin, margin_target = validate_margin_ref(
            entry["margin_ref"], workspace_root=workspace_root
        )
        key = entry["model_key"]
        for report, label in ((evaluation, "evaluation"), (margin, "margin")):
            if report["model_key"] != key or report["role"] != model["role"] or report[
                "seed"
            ] != model["seed"]:
                raise Stage1ArtifactError(f"{label} does not bind registry slot {key}")
            if report["scope"] != registry["registry_scope"] or report["split"] != split:
                raise Stage1ArtifactError(f"{label} scope/split differs from analysis")
            expected_sealing = "sealed-test" if split == "test" else "unsealed-dev"
            if report["sealing_status"] != expected_sealing:
                raise Stage1ArtifactError(f"{label} sealing status differs from analysis")
            if report["scientific_eligible"] is not profile["scientific_eligible"]:
                raise Stage1ArtifactError(f"{label} scientific eligibility differs from profile")
            if report["conditions"] != list(CONDITIONS):
                raise Stage1ArtifactError(f"{label} condition block is incomplete")
            _assert_control_dependency_kind(
                report.get("control_dependency"),
                split=report.get("split"),
                sealing_status=report.get("sealing_status"),
                label=label,
            )
            if report["registry_dependency"] != registry_dependency or report[
                "training_plan_dependency"
            ] != registry["training_plan_dependency"]:
                raise Stage1ArtifactError(f"{label} registry/plan dependency mismatch")
        current_lineage = {
            "context_dependency": evaluation["context_dependency"],
            "control_dependency": evaluation["control_dependency"],
            "margin_context_dependency": margin["context_dependency"],
            "margin_control_dependency": margin["control_dependency"],
            "cf_dependency": margin["cf_dependency"],
        }
        if current_lineage["context_dependency"] != current_lineage["margin_context_dependency"] or current_lineage[
            "control_dependency"
        ] != current_lineage["margin_control_dependency"]:
            raise Stage1ArtifactError("evaluation/margin context-control lineage mismatch")
        if lineage is None:
            lineage = current_lineage
        elif lineage != current_lineage:
            raise Stage1ArtifactError("model runs do not share exact context/control/CF lineage")
        if query_ids is None:
            query_ids = list(evaluation["query_ids"])
        if evaluation["query_ids"] != query_ids or margin["query_ids"] != query_ids:
            raise Stage1ArtifactError("evaluation/margin master query frames differ")
        if field_masks is None:
            field_masks = margin["field_masks"]
        elif field_masks != margin["field_masks"]:
            raise Stage1ArtifactError("margin pre-frozen masks differ across model runs")
        evaluations[key] = evaluation
        margins[key] = margin
        evaluation_deps.append(
            {"model_key": key, "dependency": portable_dependency(evaluation_locator, evaluation_target, workspace_root)}
        )
        margin_deps.append(
            {"model_key": key, "dependency": portable_dependency(margin_locator, margin_target, workspace_root)}
        )
        targets.extend(
            [
                (evaluation_target, evaluation_locator["payload_manifest_sha256"], f"evaluation {key}"),
                (margin_target, margin_locator["payload_manifest_sha256"], f"margin {key}"),
            ]
        )
    assert lineage is not None and query_ids is not None and field_masks is not None
    if registry["registry_scope"] == "formal":
        fixed = profile.get("fixed_training_seeds")
        for role in {row["role"] for row in registry["models"]}:
            seeds = [row["seed"] for row in registry["models"] if row["role"] == role]
            if seeds != fixed:
                raise Stage1ArtifactError("formal role does not cover exact fixed seed order")
    return {
        "registry_locator": registry_locator,
        "registry": registry,
        "registry_target": registry_target,
        "registry_dependency": registry_dependency,
        "evaluations": evaluations,
        "margins": margins,
        "evaluation_dependencies": evaluation_deps,
        "margin_dependencies": margin_deps,
        "lineage": lineage,
        "query_ids": query_ids,
        "field_masks": field_masks,
        "targets": targets,
    }


def _role_seed_cells(
    reports: Mapping[str, Mapping[str, Any]],
    registry_models: Sequence[Mapping[str, Any]],
    *,
    role: str,
    values: Callable[[Mapping[str, Any], str], Sequence[Any]],
    conditions: Sequence[str],
) -> dict[int, dict[str, Sequence[Any]]]:
    result: dict[int, dict[str, Sequence[Any]]] = {}
    for model in registry_models:
        if model["role"] != role:
            continue
        seed = 0 if model["seed"] is None else int(model["seed"])
        if seed in result:
            raise Stage1ArtifactError(f"duplicate seed {seed} for role {role}")
        report = reports[model["model_key"]]
        result[seed] = {condition: values(report, condition) for condition in conditions}
    if not result:
        raise Stage1ArtifactError(f"analysis has no runs for role {role}")
    return result


def _margin_values(report: Mapping[str, Any], condition: str, field: str) -> list[float]:
    rows = report["records"][condition]
    by_id = {
        row["id"]: row for row in rows if row["field"] == field
    }
    values = []
    for index, query_id in enumerate(report["query_ids"]):
        row = by_id[query_id]
        values.append(float(row["margin_mean"]) if report["field_masks"][field][index] else 0.0)
    return values


def _run_reducers(
    inputs: Mapping[str, Any], profile: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    registry_models = inputs["registry"]["models"]
    roles = list(dict.fromkeys(row["role"] for row in registry_models))
    bootstrap = profile["bootstrap"]
    b = int(bootstrap["replicates"])
    seed = int(bootstrap["seed"])
    factorial_output: dict[str, Any] = {
        "schema_version": "stage1-factorial-results/v1",
        "roles": {},
        "flips": {},
    }
    bootstrap_output: dict[str, Any] = {
        "schema_version": "stage1-bootstrap-results/v1",
        "behavior": {},
        "margin": {},
    }
    behavior_results: dict[tuple[str, str], dict[str, Any]] = {}
    for role in roles:
        factorial_output["roles"][role] = {}
        factorial_output["flips"][role] = {}
        for model in registry_models:
            if model["role"] != role:
                continue
            model_key = model["model_key"]
            report = inputs["evaluations"][model_key]
            factorial_output["flips"][role][model_key] = {
                contrast_name: {
                    endpoint: flip_table(
                        report["records"][left],
                        report["records"][right],
                        endpoint,
                    )
                    for endpoint in FLIP_ENDPOINTS
                }
                for left, right, contrast_name in FLIP_CONTRASTS
            }
        bootstrap_output["behavior"][role] = {}
        for endpoint in BEHAVIOR_ENDPOINTS:
            factorial_cells = _role_seed_cells(
                inputs["evaluations"], registry_models, role=role,
                values=lambda report, condition, endpoint=endpoint: _endpoint_values(
                    report["records"][condition], endpoint
                ),
                conditions=("C0", "CL", "CD", "CLD"),
            )
            factorial_result = fixed_seed_paired_bootstrap(
                seed_cells=factorial_cells,
                metric=lambda values, endpoint=endpoint: _reduce_endpoint_values(
                    values, endpoint
                ),
                effect=factorial_effects,
                conditions=("C0", "CL", "CD", "CLD"),
                family="factorial-secondary",
                bootstrap_seed=seed,
                replicates=b,
            )
            placebo_cells = _role_seed_cells(
                inputs["evaluations"], registry_models, role=role,
                values=lambda report, condition, endpoint=endpoint: _endpoint_values(
                    report["records"][condition], endpoint
                ),
                conditions=("C0", "CL", "CD", "PL", "PD"),
            )
            placebo_result = fixed_seed_paired_bootstrap(
                seed_cells=placebo_cells,
                metric=lambda values, endpoint=endpoint: _reduce_endpoint_values(
                    values, endpoint
                ),
                effect=placebo_effects,
                conditions=("C0", "CL", "CD", "PL", "PD"),
                family="free_generation_corroboration",
                bootstrap_seed=seed,
                replicates=b,
            )
            factorial_output["roles"][role][endpoint] = {
                "factorial_points": {
                    key: value["point"] for key, value in factorial_result["effects"].items()
                },
                "placebo_points": {
                    key: value["point"] for key, value in placebo_result["effects"].items()
                },
            }
            bootstrap_output["behavior"][role][endpoint] = {
                "factorial": factorial_result,
                "placebo": placebo_result,
            }
            behavior_results[(role, endpoint)] = placebo_result

    primary_role = profile.get("primary_model_role", roles[0])
    bootstrap_output["margin"][primary_role] = {}
    margin_results: dict[tuple[str, str, str], dict[str, Any]] = {}
    for family_name, contrasts in {
        "gold_margin_total_effect": (("CL", "C0", "TE_L"), ("CD", "C0", "TE_D")),
        "gold_margin_relevance": (("CL", "PL", "REL_L"), ("CD", "PD", "REL_D")),
    }.items():
        bootstrap_output["margin"][primary_role][family_name] = {}
        for field in FIELDS:
            bootstrap_output["margin"][primary_role][family_name][field] = {}
            cells = _role_seed_cells(
                inputs["margins"], registry_models, role=primary_role,
                values=lambda report, condition, field=field: _margin_values(report, condition, field),
                conditions=tuple(dict.fromkeys(value for left, right, _ in contrasts for value in (left, right))),
            )
            for left, right, contrast_name in contrasts:
                paired = {
                    fixed_seed: {left: values[left], right: values[right]}
                    for fixed_seed, values in cells.items()
                }
                result = fixed_seed_margin_bootstrap(
                    seed_cells=paired,
                    left_condition=left,
                    right_condition=right,
                    eligibility_mask=inputs["field_masks"][field],
                    family=family_name,
                    bootstrap_seed=seed,
                    replicates=b,
                )
                bootstrap_output["margin"][primary_role][family_name][field][contrast_name] = result
                margin_results[(family_name, field, contrast_name)] = result

    gate = {
        "schema_version": "stage1-scientific-gate/v1",
        "mode": profile["mode"],
        "scientific_eligible": profile["scientific_eligible"],
        "confirmatory_gate_emitted": False,
        "status": "engineering-only" if profile["mode"] == "engineering-smoke" else "exploratory-only",
    }
    if profile["mode"] == "confirmatory":
        family_cfg = profile["multiple_testing"]["families"]
        behavior_tests = []
        contrast_keys = (("L_relevance_advantage", "CL-PL"), ("D_relevance_advantage", "CD-PD"))
        for contrast_order, (effect_key, contrast) in enumerate(contrast_keys):
            for endpoint_order, endpoint in enumerate(BEHAVIOR_ENDPOINTS):
                result = behavior_results[(primary_role, endpoint)]
                effect = result["effects"][effect_key]
                behavior_tests.append(
                    {
                        "contrast": contrast,
                        "endpoint": endpoint,
                        "p": effect["p_two_sided"],
                        "contrast_order": contrast_order,
                        "endpoint_order": endpoint_order,
                        "point": effect["point"],
                        "ci": effect["ci"],
                        "seed_effects": {
                            str(s): values[effect_key]
                            for s, values in result["seed_points"].items()
                        },
                    }
                )
        behavior_tests = holm_adjust(behavior_tests)
        sesoi = float(family_cfg["free_generation_corroboration"]["sesoi_absolute_delta"])
        for test in behavior_tests:
            test["classification"] = classify_behavior_test(
                point=test["point"], ci=test["ci"], holm_p=test["holm_p"],
                seed_effects=test["seed_effects"], sesoi=sesoi,
            )
        margin_gate: dict[str, Any] = {}
        for family_name in ("gold_margin_total_effect", "gold_margin_relevance"):
            tests = []
            contrast_order_map = {name: index for index, name in enumerate(
                ("TE_L", "TE_D") if family_name == "gold_margin_total_effect" else ("REL_L", "REL_D")
            )}
            for field_order, field in enumerate(FIELDS):
                for contrast_name in contrast_order_map:
                    result = margin_results[(family_name, field, contrast_name)]
                    tests.append(
                        {
                            "field": field,
                            "contrast": contrast_name,
                            "p": result["raw"]["p_two_sided"],
                            "contrast_order": contrast_order_map[contrast_name],
                            "endpoint_order": field_order,
                            "raw_point": result["raw"]["point"],
                            "raw_ci": result["raw"]["ci"],
                            "dz_point": result["d_z"]["point"],
                            "seed_dz": {
                                str(s): float(values["d_z"])
                                for s, values in result["seed_points"].items()
                            },
                        }
                    )
            tests = holm_adjust(tests)
            sesoi_dz = float(family_cfg[family_name]["sesoi_dz"])
            for test in tests:
                test["classification"] = classify_margin_test(
                    raw_ci=test["raw_ci"], raw_holm_p=test["holm_p"],
                    dz_point=test["dz_point"], seed_dz=test["seed_dz"], sesoi_dz=sesoi_dz,
                )
            margin_gate[family_name] = tests
        behavior_support = {
            source_name: any(
                test["classification"]["confirmatory_positive"]
                for test in behavior_tests if test["contrast"].startswith(condition)
            )
            for source_name, condition in (("L", "CL"), ("D", "CD"))
        }
        behavior_harm = {
            source_name: any(
                test["classification"]["confirmatory_harm"]
                for test in behavior_tests if test["contrast"].startswith(condition)
            )
            for source_name, condition in (("L", "CL"), ("D", "CD"))
        }
        total = margin_gate["gold_margin_total_effect"]
        margin_support = {
            source_name: any(
                test["classification"]["confirmatory_positive"]
                for test in total if test["contrast"] == contrast
            )
            for source_name, contrast in (("L", "TE_L"), ("D", "TE_D"))
        }
        margin_harm = {
            source_name: any(
                test["classification"]["confirmatory_harm"]
                for test in total if test["contrast"] == contrast
            )
            for source_name, contrast in (("L", "TE_L"), ("D", "TE_D"))
        }
        if margin_support["L"] and margin_support["D"]:
            h1 = "supported_with_tradeoff" if any(margin_harm.values()) else "supported"
        elif margin_support["L"] or margin_support["D"]:
            h1 = "partial"
        else:
            h1 = "not_supported_on_dev"
        gate = {
            "schema_version": "stage1-scientific-gate/v1",
            "mode": "confirmatory",
            "scientific_eligible": True,
            "confirmatory_gate_emitted": True,
            "behavior_tests": behavior_tests,
            "behavior_support": behavior_support,
            "behavior_harm": behavior_harm,
            "margin_tests": margin_gate,
            "margin_support": margin_support,
            "margin_harm": margin_harm,
            "h1_1_status": h1,
        }
    return factorial_output, bootstrap_output, gate


def _analysis_code_sha256() -> str:
    return canonical_sha256(
        {
            "artifact_lifecycle_sha256": sha256_file(__file__),
            "statistics_module_sha256": sha256_file(Path(__file__).with_name("stage1_statistics.py")),
            "metrics_module_sha256": sha256_file(Path(__file__).with_name("stage1_metrics.py")),
        }
    )


def _analysis_id_inputs(
    *,
    inputs: Mapping[str, Any],
    profile: Mapping[str, Any],
    decision_register: Mapping[str, Any],
    split: str,
) -> dict[str, Any]:
    multiple = profile.get("multiple_testing", {})
    endpoints = multiple.get("families", {}).get(
        "free_generation_corroboration", {}
    ).get("endpoints", list(BEHAVIOR_ENDPOINTS))
    return {
        "training_plan_dependency": inputs["registry"]["training_plan_dependency"],
        "model_registry_dependency": inputs["registry_dependency"],
        "analysis_profile_sha256": canonical_sha256(profile),
        "decision_register_sha256": canonical_sha256(decision_register),
        "evaluation_dependencies": inputs["evaluation_dependencies"],
        "margin_dependencies": inputs["margin_dependencies"],
        "split": split,
        "sealing_status": "sealed-test" if split == "test" else "unsealed-dev",
        "estimand": profile.get(
            "estimand",
            {
                "population": "frozen-eligible-query-frame",
                "policy": "intention-to-treat",
            },
        ),
        "master_frame": {
            "ordered_query_ids_sha256": _ordered_id_hash(inputs["query_ids"]),
            "query_count": len(inputs["query_ids"]),
            "field_eligibility_masks_sha256": canonical_sha256(inputs["field_masks"]),
        },
        "endpoints": endpoints,
        "flip_diagnostics": {
            "schema_version": "stage1-flip-table/v1",
            "ordered_endpoints": list(FLIP_ENDPOINTS),
            "ordered_contrasts": [
                {"left": left, "right": right, "name": name}
                for left, right, name in FLIP_CONTRASTS
            ],
        },
        "multiple_testing": multiple,
        "bootstrap_profile": profile["bootstrap"],
        "gate_policy_sha256": canonical_sha256(profile["gate_policy"]),
        "analyzer_code_sha256": _analysis_code_sha256(),
    }


def _analysis_meta(
    *,
    analysis_id: str,
    id_inputs: Mapping[str, Any],
    inputs: Mapping[str, Any],
    profile: Mapping[str, Any],
    factorial: Mapping[str, Any],
    bootstrap: Mapping[str, Any],
    gate: Mapping[str, Any],
    split: str,
) -> dict[str, Any]:
    return {
        "schema_version": ANALYSIS_META_SCHEMA,
        "analysis_id": analysis_id,
        "registry_scope": inputs["registry"]["registry_scope"],
        "mode": profile["mode"],
        "split": split,
        "sealing_status": "sealed-test" if split == "test" else "unsealed-dev",
        "scientific_eligible": profile["scientific_eligible"],
        "ordered_model_keys": [row["model_key"] for row in inputs["registry"]["models"]],
        "ordered_query_ids_sha256": _ordered_id_hash(inputs["query_ids"]),
        "query_count": len(inputs["query_ids"]),
        "ordered_conditions": list(CONDITIONS),
        "bootstrap_stream_derivation": profile["bootstrap"]["family_stream"],
        "bootstrap_replicates": profile["bootstrap"]["replicates"],
        "bootstrap_seed": profile["bootstrap"]["seed"],
        "holm_families": list(profile.get("multiple_testing", {}).get("families", {})),
        "gate_policy": profile["gate_policy"],
        "id_inputs": copy.deepcopy(dict(id_inputs)),
        "observed_payload_sha256": {
            "factorial": canonical_sha256(factorial),
            "bootstrap": canonical_sha256(bootstrap),
            "scientific_gate": canonical_sha256(gate),
        },
    }


def build_factorial_analysis_artifact(
    *,
    model_registry_ref: str | Path,
    run_ref_map: str | Path | Mapping[str, Any],
    analysis_profile: str | Path | Mapping[str, Any],
    decision_register: str | Path | Mapping[str, Any],
    write_ref: str | Path,
    split: str,
    mode: str | None = None,
    n_bootstrap: int | None = None,
    seed: int | None = None,
    sealed: bool | None = None,
    target_root: str | Path | None = None,
    workspace_root: str | Path = REPOSITORY_ROOT,
) -> dict[str, Any]:
    profile = resolve_analysis_profile(analysis_profile)
    if mode is not None and mode != profile["mode"]:
        raise Stage1ArtifactError("--mode assertion differs from resolved analysis profile")
    if n_bootstrap is not None and n_bootstrap != profile["bootstrap"]["replicates"]:
        raise Stage1ArtifactError("--n-bootstrap assertion differs from resolved profile")
    if seed is not None and seed != profile["bootstrap"]["seed"]:
        raise Stage1ArtifactError("--seed assertion differs from resolved profile")
    decision = resolve_decision_register(decision_register, profile=profile)
    inputs = _analysis_inputs(
        registry_ref=model_registry_ref,
        run_ref_map=run_ref_map,
        profile=profile,
        split=split,
        sealed=sealed,
        workspace_root=workspace_root,
    )
    id_inputs = _analysis_id_inputs(
        inputs=inputs, profile=profile, decision_register=decision, split=split
    )
    analysis_id = "ana-" + canonical_sha256(id_inputs)
    factorial, bootstrap, gate = _run_reducers(inputs, profile)
    meta = _analysis_meta(
        analysis_id=analysis_id,
        id_inputs=id_inputs,
        inputs=inputs,
        profile=profile,
        factorial=factorial,
        bootstrap=bootstrap,
        gate=gate,
        split=split,
    )
    provenance = {
        "schema_version": "stage1-analysis-provenance/v1",
        "analysis_id": analysis_id,
        "model_registry_dependency": inputs["registry_dependency"],
        "evaluation_dependencies": inputs["evaluation_dependencies"],
        "margin_dependencies": inputs["margin_dependencies"],
        "analysis_profile_sha256": canonical_sha256(profile),
        "decision_register_sha256": canonical_sha256(decision),
        "analyzer_code_sha256": _analysis_code_sha256(),
        "observed_results_in_lifecycle_id": False,
        "upstream_mutation_allowed": False,
    }
    root = Path(target_root).resolve() if target_root is not None else inputs[
        "registry_target"
    ].parent.parent
    parent = root / "analyses"
    target = parent / analysis_id
    staging = new_staging_directory(parent, analysis_id)
    try:
        write_canonical_json(
            staging / "training_plan_ref.json",
            inputs["registry"]["training_plan_dependency"],
        )
        write_canonical_json(staging / "model_registry_ref.json", inputs["registry_dependency"])
        write_canonical_json(staging / "evaluation_refs.json", inputs["evaluation_dependencies"])
        write_canonical_json(staging / "margin_refs.json", inputs["margin_dependencies"])
        write_canonical_json(staging / "decision_register.json", decision)
        write_canonical_json(staging / "analysis_profile.resolved.json", profile)
        write_canonical_json(staging / "analysis.meta.json", meta)
        write_canonical_json(staging / "provenance.json", provenance)
        write_canonical_json(staging / "factorial.json", factorial)
        write_canonical_json(staging / "bootstrap.json", bootstrap)
        write_canonical_json(staging / "scientific_gate.json", gate)
        payload_hash = finalize_target_atomic(
            staging,
            target,
            validate_staging=lambda candidate: validate_analysis_target(
                candidate,
                workspace_root=workspace_root,
                require_directory_name=False,
            ),
        )
    except (TrainingArtifactError, OSError) as exc:
        if staging.exists():
            shutil.rmtree(staging)
        raise _artifact_error(exc) from exc
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    for upstream, payload_hash_before, label in inputs["targets"]:
        _assert_target_unchanged(upstream, payload_hash_before, label=label)
    try:
        return write_locator_ref(
            write_ref,
            artifact_kind=ANALYSIS_KIND,
            artifact_id=analysis_id,
            target=target,
            payload_manifest_sha256=payload_hash,
        )
    except TrainingArtifactError as exc:
        raise _artifact_error(exc) from exc


def _analysis_inputs_from_target(
    target: Path, *, workspace_root: str | Path, profile: Mapping[str, Any]
) -> dict[str, Any]:
    frozen_meta = load_json(target / "analysis.meta.json")
    meta_split = frozen_meta.get("split") if isinstance(frozen_meta, Mapping) else None
    if meta_split not in {"dev", "test"}:
        raise Stage1ArtifactError("analysis target has an invalid split")
    registry_dep = load_json(target / "model_registry_ref.json")
    registry_target = _dependency_target(
        registry_dep, workspace_root=workspace_root, kinds=REGISTRY_KIND
    )
    registry = validate_registry_target(registry_target, workspace_root=workspace_root)
    if load_json(target / "training_plan_ref.json") != registry["training_plan_dependency"]:
        raise Stage1ArtifactError("analysis training-plan dependency differs from registry")
    evaluation_deps = load_json(target / "evaluation_refs.json")
    margin_deps = load_json(target / "margin_refs.json")
    if not isinstance(evaluation_deps, list) or not isinstance(margin_deps, list):
        raise Stage1ArtifactError("analysis run dependency lists must be arrays")
    model_keys = [row["model_key"] for row in registry["models"]]
    if [row.get("model_key") for row in evaluation_deps] != model_keys or [
        row.get("model_key") for row in margin_deps
    ] != model_keys:
        raise Stage1ArtifactError("analysis dependency lists do not exactly follow registry")
    evaluations: dict[str, dict[str, Any]] = {}
    margins: dict[str, dict[str, Any]] = {}
    targets = [(registry_target, registry_dep["payload_manifest_sha256"], "registry")]
    query_ids = None
    field_masks = None
    lineage = None
    for model, eval_entry, margin_entry in zip(
        registry["models"], evaluation_deps, margin_deps, strict=True
    ):
        eval_dep = eval_entry.get("dependency")
        margin_dep = margin_entry.get("dependency")
        eval_target = _dependency_target(
            eval_dep, workspace_root=workspace_root, kinds=EVALUATION_KIND
        )
        margin_target = _dependency_target(
            margin_dep, workspace_root=workspace_root, kinds=MARGIN_KIND
        )
        evaluation = validate_evaluation_target(eval_target, workspace_root=workspace_root)
        margin = validate_margin_target(margin_target, workspace_root=workspace_root)
        key = model["model_key"]
        for report in (evaluation, margin):
            if report["model_key"] != key or report["role"] != model["role"] or report[
                "seed"
            ] != model["seed"]:
                raise Stage1ArtifactError("analysis dependency/model slot mismatch")
            if report["registry_dependency"] != registry_dep or report[
                "training_plan_dependency"
            ] != registry["training_plan_dependency"]:
                raise Stage1ArtifactError("analysis dependency registry/plan mismatch")
            if report["scope"] != registry["registry_scope"] or report["split"] != meta_split:
                raise Stage1ArtifactError("analysis dependency scope/split mismatch")
            expected_sealing = "sealed-test" if meta_split == "test" else "unsealed-dev"
            if report["sealing_status"] != expected_sealing:
                raise Stage1ArtifactError("analysis dependency sealing status mismatch")
            if report["scientific_eligible"] is not profile["scientific_eligible"]:
                raise Stage1ArtifactError("analysis dependency scientific eligibility mismatch")
            if report["conditions"] != list(CONDITIONS):
                raise Stage1ArtifactError("analysis dependency condition frame is incomplete")
            _assert_control_dependency_kind(
                report.get("control_dependency"),
                split=report.get("split"),
                sealing_status=report.get("sealing_status"),
                label="analysis dependency",
            )
        current_lineage = {
            "context_dependency": evaluation["context_dependency"],
            "control_dependency": evaluation["control_dependency"],
            "margin_context_dependency": margin["context_dependency"],
            "margin_control_dependency": margin["control_dependency"],
            "cf_dependency": margin["cf_dependency"],
        }
        if current_lineage["context_dependency"] != current_lineage["margin_context_dependency"] or current_lineage[
            "control_dependency"
        ] != current_lineage["margin_control_dependency"]:
            raise Stage1ArtifactError("analysis evaluation/margin lineage mismatch")
        if lineage is None:
            lineage = current_lineage
        elif lineage != current_lineage:
            raise Stage1ArtifactError("analysis dependencies have different frozen lineage")
        if query_ids is None:
            query_ids = evaluation["query_ids"]
        if evaluation["query_ids"] != query_ids or margin["query_ids"] != query_ids:
            raise Stage1ArtifactError("analysis dependency query frames differ")
        if field_masks is None:
            field_masks = margin["field_masks"]
        elif field_masks != margin["field_masks"]:
            raise Stage1ArtifactError("analysis dependency margin masks differ")
        evaluations[key] = evaluation
        margins[key] = margin
        targets.extend(
            [
                (eval_target, eval_dep["payload_manifest_sha256"], f"evaluation {key}"),
                (margin_target, margin_dep["payload_manifest_sha256"], f"margin {key}"),
            ]
        )
    assert query_ids is not None and field_masks is not None and lineage is not None
    return {
        "registry": registry,
        "registry_target": registry_target,
        "registry_dependency": registry_dep,
        "evaluations": evaluations,
        "margins": margins,
        "evaluation_dependencies": evaluation_deps,
        "margin_dependencies": margin_deps,
        "lineage": lineage,
        "query_ids": query_ids,
        "field_masks": field_masks,
        "targets": targets,
    }


def validate_analysis_target(
    target_dir: str | Path,
    *,
    workspace_root: str | Path = REPOSITORY_ROOT,
    require_directory_name: bool = True,
) -> dict[str, Any]:
    target = Path(target_dir)
    try:
        validate_payload_manifest(target)
        ensure_exact_file_set(
            target,
            {
                "training_plan_ref.json", "model_registry_ref.json",
                "evaluation_refs.json", "margin_refs.json",
                "decision_register.json", "analysis_profile.resolved.json",
                "analysis.meta.json", "provenance.json", "factorial.json",
                "bootstrap.json", "scientific_gate.json", "payload_manifest.json",
            },
        )
    except TrainingArtifactError as exc:
        raise _artifact_error(exc) from exc
    profile = resolve_analysis_profile(load_json(target / "analysis_profile.resolved.json"))
    decision = resolve_decision_register(
        load_json(target / "decision_register.json"), profile=profile
    )
    meta = load_json(target / "analysis.meta.json")
    if not isinstance(meta, Mapping) or meta.get("schema_version") != ANALYSIS_META_SCHEMA:
        raise Stage1ArtifactError("analysis meta has wrong schema")
    try:
        validate_json_schema(
            meta, SCHEMA_ROOT / "stage1_analysis_artifact_v1.schema.json"
        )
    except TrainingArtifactError as exc:
        raise _artifact_error(exc) from exc
    split = meta.get("split")
    inputs = _analysis_inputs_from_target(
        target, workspace_root=workspace_root, profile=profile
    )
    model_keys = [row["model_key"] for row in inputs["registry"]["models"]]
    if profile["mode"] == "pilot" and model_keys != profile["required_model_keys"]:
        raise Stage1ArtifactError("pilot analysis dependency slots differ from profile")
    if inputs["registry"]["registry_scope"] == "formal":
        for role in {row["role"] for row in inputs["registry"]["models"]}:
            if [
                row["seed"]
                for row in inputs["registry"]["models"]
                if row["role"] == role
            ] != profile["fixed_training_seeds"]:
                raise Stage1ArtifactError("formal analysis dependency seed frame is incomplete")
    _scope_matrix(
        scope=inputs["registry"]["registry_scope"], split=split,
        sealing_status=meta.get("sealing_status"),
        scientific_eligible=bool(profile["scientific_eligible"]), mode=profile["mode"],
    )
    id_inputs = _analysis_id_inputs(
        inputs=inputs, profile=profile, decision_register=decision, split=split
    )
    analysis_id = "ana-" + canonical_sha256(id_inputs)
    if require_directory_name and target.name != analysis_id:
        raise Stage1ArtifactError("analysis target directory/ID mismatch")
    factorial, bootstrap, gate = _run_reducers(inputs, profile)
    if load_json(target / "factorial.json") != factorial:
        raise Stage1ArtifactError("factorial results cannot be replayed")
    _assert_canonical_json(target / "factorial.json", factorial, label="factorial results")
    if load_json(target / "bootstrap.json") != bootstrap:
        raise Stage1ArtifactError("bootstrap results cannot be replayed")
    _assert_canonical_json(target / "bootstrap.json", bootstrap, label="bootstrap results")
    if load_json(target / "scientific_gate.json") != gate:
        raise Stage1ArtifactError("scientific gate cannot be replayed")
    _assert_canonical_json(target / "scientific_gate.json", gate, label="scientific gate")
    expected_meta = _analysis_meta(
        analysis_id=analysis_id, id_inputs=id_inputs, inputs=inputs, profile=profile,
        factorial=factorial, bootstrap=bootstrap, gate=gate, split=split,
    )
    if meta != expected_meta:
        raise Stage1ArtifactError("analysis meta cannot be replayed")
    _assert_canonical_json(target / "analysis.meta.json", expected_meta, label="analysis meta")
    _assert_canonical_json(
        target / "analysis_profile.resolved.json", profile, label="analysis resolved profile"
    )
    _assert_canonical_json(
        target / "decision_register.json", decision, label="analysis decision register"
    )
    _assert_canonical_json(
        target / "training_plan_ref.json",
        inputs["registry"]["training_plan_dependency"],
        label="analysis training-plan ref",
    )
    _assert_canonical_json(
        target / "model_registry_ref.json",
        inputs["registry_dependency"],
        label="analysis registry ref",
    )
    _assert_canonical_json(
        target / "evaluation_refs.json",
        inputs["evaluation_dependencies"],
        label="analysis evaluation refs",
    )
    _assert_canonical_json(
        target / "margin_refs.json",
        inputs["margin_dependencies"],
        label="analysis margin refs",
    )
    expected_provenance = {
        "schema_version": "stage1-analysis-provenance/v1",
        "analysis_id": analysis_id,
        "model_registry_dependency": inputs["registry_dependency"],
        "evaluation_dependencies": inputs["evaluation_dependencies"],
        "margin_dependencies": inputs["margin_dependencies"],
        "analysis_profile_sha256": canonical_sha256(profile),
        "decision_register_sha256": canonical_sha256(decision),
        "analyzer_code_sha256": _analysis_code_sha256(),
        "observed_results_in_lifecycle_id": False,
        "upstream_mutation_allowed": False,
    }
    if load_json(target / "provenance.json") != expected_provenance:
        raise Stage1ArtifactError("analysis provenance cannot be replayed")
    _assert_canonical_json(
        target / "provenance.json", expected_provenance, label="analysis provenance"
    )
    return {
        "analysis_id": analysis_id,
        "mode": profile["mode"],
        "scope": inputs["registry"]["registry_scope"],
        "split": split,
        "scientific_eligible": profile["scientific_eligible"],
        "model_count": len(inputs["registry"]["models"]),
        "query_count": len(inputs["query_ids"]),
        "payload_manifest_sha256": sha256_file(target / "payload_manifest.json"),
    }


def validate_analysis_ref(
    analysis_ref: str | Path,
    *,
    workspace_root: str | Path = REPOSITORY_ROOT,
) -> dict[str, Any]:
    try:
        locator, target = resolve_locator_ref(analysis_ref, ANALYSIS_KIND)
    except TrainingArtifactError as exc:
        raise _artifact_error(exc) from exc
    report = validate_analysis_target(target, workspace_root=workspace_root)
    if locator["artifact_id"] != report["analysis_id"] or locator[
        "payload_manifest_sha256"
    ] != report["payload_manifest_sha256"]:
        raise Stage1ArtifactError("analysis locator mismatch")
    return report


def validate_evaluation_ref_report(
    evaluation_ref: str | Path,
    *,
    workspace_root: str | Path = REPOSITORY_ROOT,
) -> dict[str, Any]:
    return validate_evaluation_ref(evaluation_ref, workspace_root=workspace_root)[1]


__all__ = [
    "Stage1ArtifactError",
    "build_factorial_analysis_artifact",
    "build_free_evaluation_artifact",
    "resolve_analysis_profile",
    "resolve_evaluation_profile",
    "resolve_margin_profile",
    "validate_analysis_ref",
    "validate_analysis_target",
    "validate_evaluation_ref_report",
    "validate_evaluation_target",
    "validate_generation_target",
    "validate_margin_ref",
    "validate_margin_target",
    "validate_registry_target",
]
