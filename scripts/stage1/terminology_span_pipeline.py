#!/usr/bin/env python3
"""CLI for the WP3 category-free terminology span lifecycle."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPOSITORY_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from build_lex.terminology_span_pipeline import (  # noqa: E402
    TerminologySpanError,
    _canonical_sha,
    build_census_documents,
    build_full_audit_package,
    build_pilot_review_package,
    build_validation_extension_review_package,
    calibrate_pilot_gate,
    evaluate_full_audit,
    load_pipeline_config,
    materialize_full_span_frame,
    pilot_tasks,
    qwen_provider_config,
    qwen_contract,
    resolve_train_input,
    run_full_qwen_scan,
    run_qwen_tasks,
    validate_census_artifact,
    validate_full_span_frame,
    validate_inconclusive_pilot_gate,
    validate_pilot_decision,
    validate_tune_review_prerequisite,
    validation_extension_tasks,
    write_census_artifact,
    write_pilot_decision,
)
from data.training_artifacts import canonical_sha256, write_canonical_json  # noqa: E402
from build_lex.terminology_resolution import (  # noqa: E402
    BgeSimilarity,
    OpenAIJsonProvider,
    ProviderSettings,
    SafePageFetcher,
    build_resolution_execution_contract,
    calibrate_resolution_gate,
    finalize_library,
    merge_human_resolution,
    publish_stage1_library,
    run_resolution,
    validate_library,
    validate_stage1_published_library,
)
from build_lex.web_search import WebSearcher  # noqa: E402


DEFAULT_CONFIG = REPOSITORY_ROOT / "config/stage1/terminology_span_pipeline.json"
DEFAULT_LEXICON_CONFIG = REPOSITORY_ROOT / "config/stage1/lexicon_train_only.json"
DEFAULT_DATA_REF = REPOSITORY_ROOT / "exps/causal_context/stage1_p0/refs/data_ref.json"
DEFAULT_PARTITION_REF = REPOSITORY_ROOT / "exps/causal_context/stage1_p0/refs/train_partition_ref.json"
DEFAULT_ARTIFACT_ROOT = REPOSITORY_ROOT / "exps/causal_context/stage1_p0/terminology_spans"
DEFAULT_REVIEW_ROOT = REPOSITORY_ROOT / "exps/causal_context/stage1_p0/review_packages"
DEFAULT_TEMPLATES = REPOSITORY_ROOT / "tools/terminology_span_review"


def _print(value: object) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))


def _config(args: argparse.Namespace) -> dict:
    return load_pipeline_config(args.config)


def _frozen(args: argparse.Namespace):
    return resolve_train_input(
        data_ref=args.data_ref,
        train_partition_ref=args.train_partition_ref,
        workspace_root=REPOSITORY_ROOT,
        formal=True,
    )


def command_census(args: argparse.Namespace) -> int:
    config = _config(args)
    frozen = _frozen(args)
    target = write_census_artifact(
        build_census_documents(frozen, config), output_root=args.output_root
    )
    report = validate_census_artifact(target)
    _print(
        {
            "census_dir": str(target),
            "census_id": report["census_id"],
            "fit_record_count": report["metadata"]["fit_record_count"],
            "pilot_counts": report["metadata"]["pilot_counts"],
        }
    )
    return 0


def _run_pilot_phase(args: argparse.Namespace, phase: str) -> int:
    config = _config(args)
    census, tasks = pilot_tasks(args.census_dir, phase=phase)
    if canonical_sha256(config) != census["metadata"]["config_sha256"]:
        raise TerminologySpanError(
            "pilot config differs from census; rerun span-census before Qwen"
        )
    provider = qwen_provider_config(config)
    result = run_qwen_tasks(
        tasks=tasks,
        checkpoint_path=args.checkpoint,
        provider=provider,
        config=config,
        fit_data_sha256=str(census["metadata"]["fit_data_sha256"]),
        limit=args.limit,
    )
    result["phase"] = phase
    _print(result)
    return 2 if result["failures"] else 0


def command_pilot_tune(args: argparse.Namespace) -> int:
    return _run_pilot_phase(args, "tune")


def command_pilot_validate(args: argparse.Namespace) -> int:
    prerequisite = validate_tune_review_prerequisite(
        census_dir=args.census_dir,
        checkpoint_path=args.checkpoint,
        config=_config(args),
        review_package_dir=args.tune_review_package,
        annotation_path=args.tune_annotations,
    )
    print(
        "[qwen-span] A1 review prerequisite="
        + prerequisite["annotation_sha256"],
        flush=True,
    )
    return _run_pilot_phase(args, "validation")


def command_pilot_extend(args: argparse.Namespace) -> int:
    config = _config(args)
    frozen = _frozen(args)
    census, tasks, _ = validation_extension_tasks(
        frozen=frozen, census_dir=args.census_dir, block=args.block
    )
    if canonical_sha256(config) != census["metadata"]["config_sha256"]:
        raise TerminologySpanError("validation extension config differs from census")
    provider = qwen_provider_config(config)
    gate = validate_inconclusive_pilot_gate(
        args.pilot_gate,
        census=census,
        config=config,
        qwen_contract_sha256=_canonical_sha(
            qwen_contract(provider, config_sha256=_canonical_sha(config))
        ),
    )
    expected_block = len(gate["validation_extension_blocks"]) + 1
    if args.block != expected_block:
        raise TerminologySpanError(
            f"next validation extension block must be {expected_block}"
        )
    result = run_qwen_tasks(
        tasks=tasks,
        checkpoint_path=args.checkpoint,
        provider=provider,
        config=config,
        fit_data_sha256=str(census["metadata"]["fit_data_sha256"]),
        limit=args.limit,
    )
    result["phase"] = "validation-extension"
    result["extension_block"] = args.block
    _print(result)
    return 2 if result["failures"] else 0


def command_review_pilot(args: argparse.Namespace) -> int:
    _print(
        build_pilot_review_package(
            census_dir=args.census_dir,
            checkpoint_path=args.checkpoint,
            config=_config(args),
            templates_dir=args.templates_dir,
            output_root=args.output_root,
            phase=args.phase,
        )
    )
    return 0


def command_review_extension(args: argparse.Namespace) -> int:
    _print(
        build_validation_extension_review_package(
            frozen=_frozen(args),
            census_dir=args.census_dir,
            block=args.block,
            checkpoint_path=args.checkpoint,
            config=_config(args),
            templates_dir=args.templates_dir,
            output_root=args.output_root,
        )
    )
    return 0


def _extension_reviews(args: argparse.Namespace) -> list[tuple[Path, Path]]:
    return [tuple(pair) for pair in (args.extension_review or [])]


def command_calibrate(args: argparse.Namespace) -> int:
    gate = calibrate_pilot_gate(
        census_dir=args.census_dir,
        checkpoint_path=args.checkpoint,
        config=_config(args),
        review_package_dir=args.review_package,
        annotation_path=args.annotations,
        extension_reviews=_extension_reviews(args),
        gold_output_path=args.gold_output,
    )
    write_canonical_json(args.output, gate)
    _print(gate)
    return 0 if gate["status"] == "PASS" else 3


def command_approve(args: argparse.Namespace) -> int:
    gate = calibrate_pilot_gate(
        census_dir=args.census_dir,
        checkpoint_path=args.checkpoint,
        config=_config(args),
        review_package_dir=args.review_package,
        annotation_path=args.annotations,
        extension_reviews=_extension_reviews(args),
        gold_output_path=args.gold_output,
    )
    decision = write_pilot_decision(
        gate=gate,
        decision=args.decision,
        reviewer_id=args.reviewer_id,
        notes=args.notes,
        output_path=args.output,
    )
    _print(decision)
    return 0


def command_full(args: argparse.Namespace) -> int:
    config = _config(args)
    frozen = _frozen(args)
    result = run_full_qwen_scan(
        frozen=frozen,
        census_dir=args.census_dir,
        pilot_decision_path=args.pilot_decision,
        checkpoint_path=args.checkpoint,
        config=config,
        provider=qwen_provider_config(config),
        exception_resolution_path=args.exception_resolution,
        limit=args.limit,
    )
    if args.limit is None and not result["failures"]:
        target = materialize_full_span_frame(
            frozen=frozen,
            census_dir=args.census_dir,
            pilot_decision_path=args.pilot_decision,
            checkpoint_path=args.checkpoint,
            config=config,
            output_root=args.output_root,
        )
        result["span_frame_dir"] = str(target)
        result["span_frame_id"] = validate_full_span_frame(target)["frame_id"]
    _print(result)
    return 2 if result["failures"] else 0


def command_audit(args: argparse.Namespace) -> int:
    config = _config(args)
    if args.annotations is None:
        _print(
            build_full_audit_package(
                span_frame_dir=args.span_frame,
                config=config,
                templates_dir=args.templates_dir,
                output_root=args.output_root,
            )
        )
        return 0
    if args.review_package is None or args.output is None or args.decision is None:
        raise TerminologySpanError(
            "audit evaluation requires --review-package, --decision and --output"
        )
    result = evaluate_full_audit(
        review_package_dir=args.review_package,
        annotation_path=args.annotations,
        config=config,
        reviewer_decision=args.decision,
        notes=args.notes,
        output_path=args.output,
    )
    _print(result)
    return 0 if result["reviewer_decision"] == "PASS" else 3


def command_validate(args: argparse.Namespace) -> int:
    if args.kind == "census":
        result = validate_census_artifact(args.path)
        _print({"valid": True, "kind": args.kind, "id": result["census_id"]})
    elif args.kind == "span-frame":
        result = validate_full_span_frame(args.path)
        _print({"valid": True, "kind": args.kind, "id": result["frame_id"]})
    else:
        result = validate_pilot_decision(args.path, require_pass=False)
        _print(
            {
                "valid": True,
                "kind": args.kind,
                "decision": result["decision"],
                "id": result["decision_sha256"],
            }
        )
    return 0


def command_calibrate_resolution(args: argparse.Namespace) -> int:
    frame = validate_full_span_frame(args.span_frame)
    result = calibrate_resolution_gate(
        gold_path=args.gold,
        execution_contract_path=args.execution_contract,
        span_frame_id=frame["frame_id"],
        config=_config(args),
        output_path=args.output,
    )
    _print(result)
    return 0


def _resolution_providers(
    config: dict,
    lexicon_config_path: Path,
    *,
    require_credentials: bool = True,
) -> tuple[
    OpenAIJsonProvider,
    OpenAIJsonProvider,
    WebSearcher,
    dict,
    dict,
]:
    lexicon_config = json.loads(lexicon_config_path.read_text(encoding="utf-8"))
    qwen = config["qwen"]
    qwen_provider = OpenAIJsonProvider(
        ProviderSettings(
            provider="qwen",
            model=qwen["model"],
            api_base=qwen["api_base"],
            api_key=os.environ.get(qwen["api_key_env"]) or "EMPTY",
            timeout=qwen["timeout_seconds"],
            # The content-addressed ResolutionCheckpoint owns the three-attempt
            # budget; the HTTP adapter must perform exactly one physical call.
            max_attempts=1,
            max_tokens=max(1024, qwen["max_tokens"]),
        )
    )
    deepseek = lexicon_config["llm_settings"]
    deepseek_key = os.environ.get(deepseek["api_key_env"])
    if require_credentials and not deepseek_key:
        raise TerminologySpanError(
            f"missing resolution credential: {deepseek['api_key_env']}"
        )
    deepseek_provider = OpenAIJsonProvider(
        ProviderSettings(
            provider="deepseek",
            model=deepseek["model"],
            api_base=deepseek["api_base"],
            api_key=deepseek_key or "CONTRACT_ONLY",
            timeout=deepseek["timeout"],
            max_attempts=1,
            max_tokens=deepseek["max_tokens"],
        )
    )
    web_settings = json.loads(
        json.dumps(lexicon_config["web_settings"], ensure_ascii=False)
    )
    web_settings["transport_retry_policy"] = {
        "id": "resolution-checkpoint-owned/v1",
        "retries": 0,
        "base_sleep_seconds": 0.0,
    }
    searcher = WebSearcher(web_settings)
    attempt_budgets = {
        "search": int(web_settings["physical_attempt_budget"]["cap"]),
        "deepseek": int(
            lexicon_config["runtime_settings"]["max_llm_http_attempts"]
        ),
    }
    return qwen_provider, deepseek_provider, searcher, web_settings, attempt_budgets


def command_freeze_resolution_contract(args: argparse.Namespace) -> int:
    config = _config(args)
    qwen, deepseek, searcher, web_settings, attempt_budgets = _resolution_providers(
        config,
        args.lexicon_config,
        require_credentials=False,
    )
    try:
        from data.build_context_manifest import embedding_model_file_tree_sha256

        model_path = REPOSITORY_ROOT / config["resolution"]["bge_model_path"]
        observed = embedding_model_file_tree_sha256(model_path)
        if observed != config["resolution"]["bge_model_file_tree_sha256"]:
            raise TerminologySpanError("frozen BGE model file tree differs")
        contract = build_resolution_execution_contract(
            config=config,
            qwen=qwen,
            deepseek=deepseek,
            searcher=searcher,
            similarity=None,
            web_settings=web_settings,
            attempt_budgets=attempt_budgets,
        )
        write_canonical_json(args.output, contract)
    finally:
        searcher.close()
    _print(contract)
    return 0


def command_run_resolution(args: argparse.Namespace) -> int:
    config = _config(args)
    qwen, deepseek, searcher, web_settings, attempt_budgets = _resolution_providers(
        config, args.lexicon_config
    )
    try:
        model_path = REPOSITORY_ROOT / config["resolution"]["bge_model_path"]
        similarity = BgeSimilarity(
            model_path,
            expected_file_tree_sha256=config["resolution"][
                "bge_model_file_tree_sha256"
            ],
        )
        execution_contract = build_resolution_execution_contract(
            config=config,
            qwen=qwen,
            deepseek=deepseek,
            searcher=searcher,
            similarity=similarity,
            web_settings=web_settings,
            attempt_budgets=attempt_budgets,
        )
        target = run_resolution(
            span_frame_dir=args.span_frame,
            full_audit_decision_path=args.full_audit_decision,
            resolution_gate_path=args.resolution_gate,
            config=config,
            checkpoint_path=args.checkpoint,
            output_root=args.output_root,
            qwen=qwen,
            deepseek=deepseek,
            searcher=searcher,
            fetcher=SafePageFetcher(),
            similarity=similarity,
            execution_contract=execution_contract,
            limit=args.limit,
        )
    finally:
        searcher.close()
    _print({"resolution_dir": str(target), "resolution_id": target.name})
    return 0


def command_merge_human(args: argparse.Namespace) -> int:
    result = merge_human_resolution(
        resolution_dir=args.resolution_dir,
        human_path=args.human,
        output_path=args.output,
    )
    _print(result)
    return 0


def command_finalize_library(args: argparse.Namespace) -> int:
    target = finalize_library(
        resolution_dir=args.resolution_dir,
        merged_path=args.merged,
        output_root=args.output_root,
    )
    report = validate_library(target)
    _print(
        {
            "library_dir": str(target),
            "library_id": report["library_id"],
            "entry_count": report["entry_count"],
        }
    )
    return 0


def command_validate_library(args: argparse.Namespace) -> int:
    report = validate_library(args.library_dir)
    _print(
        {
            "valid": True,
            "library_id": report["library_id"],
            "entry_count": report["entry_count"],
        }
    )
    return 0


def command_publish_stage1(args: argparse.Namespace) -> int:
    locator = publish_stage1_library(
        library_dir=args.library_dir,
        span_frame_dir=args.span_frame,
        pilot_decision_path=args.pilot_decision,
        full_audit_decision_path=args.full_audit_decision,
        resolution_gate_path=args.resolution_gate,
        resolution_dir=args.resolution_dir,
        merged_path=args.merged,
        human_resolution_path=args.human,
        config=_config(args),
        data_ref=args.data_ref,
        train_partition_ref=args.train_partition_ref,
        workspace_root=REPOSITORY_ROOT,
        target_root=args.target_root,
        write_ref=args.write_ref,
    )
    report = validate_stage1_published_library(
        locator["target_path"], workspace_root=REPOSITORY_ROOT
    )
    _print(
        {
            "lexicon_ref": str(args.write_ref.resolve()),
            "lexicon_build_id": report["lexicon_build_id"],
            "term_count": len(report["lexicon_ids"]),
            "scientific_eligible": report["scientific_eligible"],
        }
    )
    return 0


def _common_config(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)


def _common_data(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-ref", type=Path, default=DEFAULT_DATA_REF)
    parser.add_argument(
        "--train-partition-ref", type=Path, default=DEFAULT_PARTITION_REF
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    census = sub.add_parser("span-census", help="run label-free A0 census")
    _common_config(census)
    _common_data(census)
    census.add_argument("--output-root", type=Path, default=DEFAULT_ARTIFACT_ROOT / "census")
    census.set_defaults(function=command_census)

    for name, function in (
        ("span-pilot-tune", command_pilot_tune),
        ("span-pilot-validate", command_pilot_validate),
    ):
        child = sub.add_parser(name)
        _common_config(child)
        child.add_argument("--census-dir", type=Path, required=True)
        child.add_argument("--checkpoint", type=Path, required=True)
        child.add_argument("--limit", type=int)
        if name == "span-pilot-validate":
            child.add_argument(
                "--tune-review-package", type=Path, required=True
            )
            child.add_argument("--tune-annotations", type=Path, required=True)
        child.set_defaults(function=function)

    extension = sub.add_parser(
        "span-pilot-extend",
        help="run the next locked 100-record A2 block after INCONCLUSIVE",
    )
    _common_config(extension)
    _common_data(extension)
    extension.add_argument("--census-dir", type=Path, required=True)
    extension.add_argument("--checkpoint", type=Path, required=True)
    extension.add_argument("--pilot-gate", type=Path, required=True)
    extension.add_argument("--block", type=int, required=True)
    extension.add_argument("--limit", type=int)
    extension.set_defaults(function=command_pilot_extend)

    review = sub.add_parser("review-span-pilot")
    _common_config(review)
    review.add_argument("--census-dir", type=Path, required=True)
    review.add_argument("--checkpoint", type=Path, required=True)
    review.add_argument("--templates-dir", type=Path, default=DEFAULT_TEMPLATES)
    review.add_argument("--output-root", type=Path, default=DEFAULT_REVIEW_ROOT)
    review.add_argument(
        "--phase",
        choices=("tune", "all"),
        default="all",
        help="use tune before A2 exposure; use all only after the configuration is frozen",
    )
    review.set_defaults(function=command_review_pilot)

    extension_review = sub.add_parser("review-span-extension")
    _common_config(extension_review)
    _common_data(extension_review)
    extension_review.add_argument("--census-dir", type=Path, required=True)
    extension_review.add_argument("--checkpoint", type=Path, required=True)
    extension_review.add_argument("--block", type=int, required=True)
    extension_review.add_argument("--templates-dir", type=Path, default=DEFAULT_TEMPLATES)
    extension_review.add_argument("--output-root", type=Path, default=DEFAULT_REVIEW_ROOT)
    extension_review.set_defaults(function=command_review_extension)

    calibrate = sub.add_parser("calibrate-gates")
    _common_config(calibrate)
    calibrate.add_argument("--census-dir", type=Path, required=True)
    calibrate.add_argument("--checkpoint", type=Path, required=True)
    calibrate.add_argument("--review-package", type=Path, required=True)
    calibrate.add_argument("--annotations", type=Path, required=True)
    calibrate.add_argument(
        "--extension-review",
        nargs=2,
        action="append",
        type=Path,
        metavar=("PACKAGE", "ANNOTATIONS"),
        help="append a contiguous 100-record locked validation review",
    )
    calibrate.add_argument("--output", type=Path, required=True)
    calibrate.add_argument("--gold-output", type=Path, required=True)
    calibrate.set_defaults(function=command_calibrate)

    approve = sub.add_parser("approve-span-pilot")
    _common_config(approve)
    approve.add_argument("--census-dir", type=Path, required=True)
    approve.add_argument("--checkpoint", type=Path, required=True)
    approve.add_argument("--review-package", type=Path, required=True)
    approve.add_argument("--annotations", type=Path, required=True)
    approve.add_argument(
        "--extension-review",
        nargs=2,
        action="append",
        type=Path,
        metavar=("PACKAGE", "ANNOTATIONS"),
    )
    approve.add_argument("--decision", choices=("PASS", "FAIL"), required=True)
    approve.add_argument("--reviewer-id", required=True)
    approve.add_argument("--notes", default="")
    approve.add_argument("--output", type=Path, required=True)
    approve.add_argument("--gold-output", type=Path, required=True)
    approve.set_defaults(function=command_approve)

    full = sub.add_parser("span-discover-full")
    _common_config(full)
    _common_data(full)
    full.add_argument("--census-dir", type=Path, required=True)
    full.add_argument("--pilot-decision", type=Path, required=True)
    full.add_argument("--checkpoint", type=Path, required=True)
    full.add_argument("--exception-resolution", type=Path)
    full.add_argument("--limit", type=int)
    full.add_argument("--output-root", type=Path, default=DEFAULT_ARTIFACT_ROOT / "frames")
    full.set_defaults(function=command_full)

    audit = sub.add_parser("audit-full-spans")
    _common_config(audit)
    audit.add_argument("--span-frame", type=Path, required=True)
    audit.add_argument("--templates-dir", type=Path, default=DEFAULT_TEMPLATES)
    audit.add_argument("--output-root", type=Path, default=DEFAULT_REVIEW_ROOT)
    audit.add_argument("--review-package", type=Path)
    audit.add_argument("--annotations", type=Path)
    audit.add_argument("--decision", choices=("PASS", "FAIL"))
    audit.add_argument("--notes", default="")
    audit.add_argument("--output", type=Path)
    audit.set_defaults(function=command_audit)

    resolution_gate = sub.add_parser("calibrate-resolution-gates")
    _common_config(resolution_gate)
    resolution_gate.add_argument("--span-frame", type=Path, required=True)
    resolution_gate.add_argument("--gold", type=Path, required=True)
    resolution_gate.add_argument(
        "--execution-contract", type=Path, required=True
    )
    resolution_gate.add_argument("--output", type=Path, required=True)
    resolution_gate.set_defaults(function=command_calibrate_resolution)

    freeze_resolution = sub.add_parser(
        "freeze-resolution-contract",
        help="freeze model/prompt/Web/BGE identities before resolution Gold review",
    )
    _common_config(freeze_resolution)
    freeze_resolution.add_argument(
        "--lexicon-config", type=Path, default=DEFAULT_LEXICON_CONFIG
    )
    freeze_resolution.add_argument("--output", type=Path, required=True)
    freeze_resolution.set_defaults(function=command_freeze_resolution_contract)

    resolution = sub.add_parser("run-resolution")
    _common_config(resolution)
    resolution.add_argument(
        "--lexicon-config", type=Path, default=DEFAULT_LEXICON_CONFIG
    )
    resolution.add_argument("--span-frame", type=Path, required=True)
    resolution.add_argument("--full-audit-decision", type=Path, required=True)
    resolution.add_argument("--resolution-gate", type=Path, required=True)
    resolution.add_argument("--checkpoint", type=Path, required=True)
    resolution.add_argument("--limit", type=int)
    resolution.add_argument(
        "--output-root", type=Path, default=DEFAULT_ARTIFACT_ROOT / "resolutions"
    )
    resolution.set_defaults(function=command_run_resolution)

    merge = sub.add_parser("merge-human")
    merge.add_argument("--resolution-dir", type=Path, required=True)
    merge.add_argument("--human", type=Path, required=True)
    merge.add_argument("--output", type=Path, required=True)
    merge.set_defaults(function=command_merge_human)

    finalize = sub.add_parser("finalize-library")
    finalize.add_argument("--resolution-dir", type=Path, required=True)
    finalize.add_argument("--merged", type=Path, required=True)
    finalize.add_argument(
        "--output-root", type=Path, default=DEFAULT_ARTIFACT_ROOT / "libraries"
    )
    finalize.set_defaults(function=command_finalize_library)

    library_validate = sub.add_parser("validate-library")
    library_validate.add_argument("--library-dir", type=Path, required=True)
    library_validate.set_defaults(function=command_validate_library)

    publish = sub.add_parser(
        "publish-stage1-library",
        help="publish the frozen lifecycle as the main experiment lexicon ref",
    )
    _common_config(publish)
    _common_data(publish)
    publish.add_argument("--library-dir", type=Path, required=True)
    publish.add_argument("--span-frame", type=Path, required=True)
    publish.add_argument("--pilot-decision", type=Path, required=True)
    publish.add_argument("--full-audit-decision", type=Path, required=True)
    publish.add_argument("--resolution-gate", type=Path, required=True)
    publish.add_argument("--resolution-dir", type=Path, required=True)
    publish.add_argument("--merged", type=Path, required=True)
    publish.add_argument("--human", type=Path, required=True)
    publish.add_argument(
        "--target-root",
        type=Path,
        default=REPOSITORY_ROOT / "exps/causal_context/stage1_p0/lexicons",
    )
    publish.add_argument("--write-ref", type=Path, required=True)
    publish.set_defaults(function=command_publish_stage1)

    validate = sub.add_parser("validate")
    validate.add_argument("kind", choices=("census", "span-frame", "pilot-decision"))
    validate.add_argument("path", type=Path)
    validate.set_defaults(function=command_validate)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        return int(args.function(args))
    except TerminologySpanError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
