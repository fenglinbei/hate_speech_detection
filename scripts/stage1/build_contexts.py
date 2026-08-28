#!/usr/bin/env python3
"""Stage 1 data lifecycle CLI.

The context-building subcommands will be added by later work packages.  WP1 owns
only the immutable data audit/adjudication/finalization commands below.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from data.stage1_data import (
    Stage1DataError,
    audit_data,
    export_data_adjudication,
    export_data_review_subset,
    finalize_data,
    merge_completed_adjudications,
    prepare_data_declaration,
    validate_data,
)
from data.build_context_manifest import (
    ContextBuildError,
    build_prepared_context_artifact,
    embedding_model_file_tree_sha256,
    resolve_embedding_model_path,
    seal_test_context_artifact,
    validate_context_ref,
)
from data.retrieval_bundle import (
    RetrievalBundleError,
    build_lexicon_catalog,
    cosine_score_matrices,
    prepare_context_bundle_from_scores,
)
from data.train_partition import (
    TrainPartitionError,
    load_train_partition,
    validate_train_partition,
)
from data.training_artifacts import portable_dependency
from build_lex.train_only import validate_lexicon_ref
from model.stage1_registry import (
    ModelRegistryError,
    ResolvedModelSourceContract,
    inventory_regular_file_tree,
    verified_model_source_lease,
)


def _tokenizer(path: Path):
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise ContextBuildError("transformers is required for context build/validation") from exc
    return AutoTokenizer.from_pretrained(
        path, trust_remote_code=False, local_files_only=True
    )


def _full_inventory(path: Path, *, label: str) -> dict[str, object]:
    try:
        return inventory_regular_file_tree(
            path,
            workspace_root=REPOSITORY_ROOT,
            label=label,
            inventory_policy="all-regular-files/v1",
        )
    except ModelRegistryError as exc:
        raise RetrievalBundleError(str(exc)) from exc


def _file_tree_sha256(path: Path) -> str:
    try:
        return embedding_model_file_tree_sha256(path)
    except ContextBuildError as exc:
        raise RetrievalBundleError(str(exc)) from exc


def _prepare_retrieval(args: argparse.Namespace) -> dict[str, object]:
    initial_data_report = validate_data(data_ref=args.data_ref)
    initial_lexicon_report = validate_lexicon_ref(
        args.lexicon_ref, workspace_root=REPOSITORY_ROOT
    )
    partition_bundle = None
    initial_partition_report = None
    if args.train_partition_ref is None:
        if not args.engineering:
            raise RetrievalBundleError(
                "formal retrieval requires --train-partition-ref"
            )
    else:
        if args.engineering:
            raise RetrievalBundleError(
                "--train-partition-ref is reserved for formal retrieval"
            )
        initial_partition_report = validate_train_partition(
            args.train_partition_ref, workspace_root=REPOSITORY_ROOT
        )
        partition_bundle = load_train_partition(
            args.train_partition_ref, workspace_root=REPOSITORY_ROOT
        )
    data_locator = json.loads(args.data_ref.read_text(encoding="utf-8"))
    lexicon_locator = json.loads(args.lexicon_ref.read_text(encoding="utf-8"))
    if partition_bundle is not None:
        if (
            initial_lexicon_report.get("source_mode")
            != "data_ref+train_partition"
            or initial_lexicon_report.get("scientific_eligible") is not True
            or initial_lexicon_report.get("train_partition_dependency")
            != partition_bundle.partition_dependency
        ):
            raise RetrievalBundleError(
                "formal lexicon is not bound to the selected fit partition"
            )
        selected_data_dependency = {
            "schema_version": "stage1-dependency-ref/v1",
            "artifact_kind": "data",
            "artifact_id": data_locator.get("artifact_id"),
            "payload_manifest_sha256": data_locator.get(
                "payload_manifest_sha256"
            ),
        }
        partition_data_dependency = {
            key: partition_bundle.data_dependency[key]
            for key in (
                "schema_version",
                "artifact_kind",
                "artifact_id",
                "payload_manifest_sha256",
            )
        }
        if partition_data_dependency != selected_data_dependency:
            raise RetrievalBundleError(
                "train partition is bound to a different data artifact"
            )
    data_target = Path(data_locator["target_path"])
    lexicon_target = Path(lexicon_locator["target_path"])
    data_dependency = portable_dependency(
        data_locator, data_target, REPOSITORY_ROOT
    )
    lexicon_dependency = portable_dependency(
        lexicon_locator, lexicon_target, REPOSITORY_ROOT
    )
    train_records = json.loads((data_target / "train.json").read_text(encoding="utf-8"))
    fit_records = (
        list(partition_bundle.fit_records)
        if partition_bundle is not None
        else train_records
    )
    query_records = json.loads((data_target / f"{args.split}.json").read_text(encoding="utf-8"))
    if args.limit is not None:
        if not args.engineering:
            raise RetrievalBundleError("--limit is engineering-only")
        query_records = query_records[: args.limit]
    terms = json.loads((lexicon_target / "lexicon.json").read_text(encoding="utf-8"))["terms"]
    config = json.loads(args.config.read_text(encoding="utf-8"))
    model_path = args.embedding_model.absolute()
    repository_root = REPOSITORY_ROOT.resolve()
    try:
        logical_model_path = model_path.relative_to(repository_root).as_posix()
    except ValueError as exc:
        raise RetrievalBundleError("embedding model must be inside the repository") from exc
    try:
        model_path = resolve_embedding_model_path(
            logical_model_path, repository_root
        )
    except ContextBuildError as exc:
        raise RetrievalBundleError(str(exc)) from exc
    model_tree_sha256 = _file_tree_sha256(model_path)
    model_inventory = _full_inventory(
        model_path, label="formal retrieval embedding scorer"
    )
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RetrievalBundleError("sentence-transformers is required") from exc
    lexicon_catalog = build_lexicon_catalog(terms)
    contract = ResolvedModelSourceContract(
        workspace_root=REPOSITORY_ROOT,
        checkpoint_inventory=model_inventory,
        tokenizer_inventory=model_inventory,
        base_inventory=model_inventory,
    )
    try:
        with verified_model_source_lease(
            contract, source_names=("checkpoint",)
        ) as paths:
            model = SentenceTransformer(
                str(paths.checkpoint_path),
                device=args.device,
                local_files_only=True,
                trust_remote_code=False,
            )
            demo_scores, lexicon_scores = cosine_score_matrices(
                model=model,
                train_texts=[record["content"] for record in fit_records],
                query_texts=[record["content"] for record in query_records],
                lexicon_texts=[row["rendered_block"] for row in lexicon_catalog],
                batch_size=args.batch_size,
            )
    except ModelRegistryError as exc:
        raise RetrievalBundleError(
            f"retrieval scorer source lease failed: {exc}"
        ) from exc
    bundle = prepare_context_bundle_from_scores(
        train_records=train_records,
        query_records=query_records,
        lexicon_terms=terms,
        demo_scores=demo_scores,
        lexicon_scores=lexicon_scores,
        split=args.split,
        retrieval_config=config["retrieval"],
        data_locator=data_locator,
        lexicon_locator=lexicon_locator,
        data_dependency=data_dependency,
        lexicon_dependency=lexicon_dependency,
        scorer_provenance={
            "backend": "sentence-transformers-cosine/v1",
            "logical_model_path": logical_model_path,
            "model_file_tree_sha256": model_tree_sha256,
            "device_class": "cuda" if str(args.device).startswith("cuda") else "cpu",
            "batch_size": args.batch_size,
        },
        fit_demo_records=(fit_records if partition_bundle is not None else None),
        calibration_ids=(
            list(partition_bundle.calibration_ids)
            if partition_bundle is not None
            else None
        ),
        train_partition_dependency=(
            partition_bundle.partition_dependency
            if partition_bundle is not None
            else None
        ),
        train_partition_locator=(
            partition_bundle.locator if partition_bundle is not None else None
        ),
    )
    if (
        validate_data(data_ref=args.data_ref) != initial_data_report
        or validate_lexicon_ref(
            args.lexicon_ref, workspace_root=REPOSITORY_ROOT
        )
        != initial_lexicon_report
        or (
            args.train_partition_ref is not None
            and validate_train_partition(
                args.train_partition_ref, workspace_root=REPOSITORY_ROOT
            )
            != initial_partition_report
        )
        or json.loads(args.data_ref.read_text(encoding="utf-8")) != data_locator
        or json.loads(args.lexicon_ref.read_text(encoding="utf-8")) != lexicon_locator
        or _file_tree_sha256(model_path) != model_tree_sha256
    ):
        raise RetrievalBundleError(
            "retrieval dependency changed while score matrices were computed"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(bundle, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return {
        "output": str(args.output.resolve()),
        "split": args.split,
        "query_count": len(query_records),
        "bundle_sha256": bundle["bundle_sha256"],
        "scientific_eligible": not args.engineering,
    }


def _seal_test(args: argparse.Namespace) -> dict[str, object]:
    """Compute the one permitted test retrieval pass from frozen dev policy."""

    workspace = args.workspace_root.resolve()
    frozen_locator = json.loads(args.frozen_context_ref.read_text(encoding="utf-8"))
    data_locator = json.loads(args.data_ref.read_text(encoding="utf-8"))
    partition_bundle = load_train_partition(
        args.train_partition_ref, workspace_root=workspace
    )
    frozen_target = Path(frozen_locator.get("target_path", ""))
    data_target = Path(data_locator.get("target_path", ""))
    if (
        frozen_locator.get("artifact_kind") != "context"
        or not frozen_target.is_dir()
        or data_locator.get("artifact_kind") != "data"
        or not data_target.is_dir()
    ):
        raise ContextBuildError("seal-test refs are not resolvable frozen context/data locators")
    compact_data = {
        "schema_version": "stage1-dependency-ref/v1",
        "artifact_kind": "data",
        "artifact_id": data_locator.get("artifact_id"),
        "payload_manifest_sha256": data_locator.get("payload_manifest_sha256"),
    }
    partition_data = {
        key: partition_bundle.data_dependency[key]
        for key in (
            "schema_version",
            "artifact_kind",
            "artifact_id",
            "payload_manifest_sha256",
        )
    }
    if partition_data != compact_data:
        raise ContextBuildError(
            "seal-test train partition is bound to a different data artifact"
        )
    for label, target in (("frozen context", frozen_target), ("data", data_target)):
        try:
            target.resolve().relative_to(workspace)
        except ValueError as exc:
            raise ContextBuildError(
                f"seal-test {label} target must live below workspace"
            ) from exc
    config = json.loads(
        (frozen_target / "config.resolved.json").read_text(encoding="utf-8")
    )
    prepared_meta = json.loads(
        (frozen_target / "prepared_bundle.meta.json").read_text(encoding="utf-8")
    )
    frozen_report = validate_context_ref(
        args.frozen_context_ref,
        workspace_root=workspace,
    )
    if (
        frozen_report.get("split") != "dev"
        or frozen_report.get("scientific_eligible") is not True
    ):
        raise ContextBuildError("seal-test requires a scientific frozen dev context")
    validate_data(data_ref=args.data_ref)
    retrieval_provenance = prepared_meta.get("retrieval_provenance", {})
    scorer = retrieval_provenance.get("scorer", {})
    logical_model_path = scorer.get("logical_model_path")
    batch_size = scorer.get("batch_size")
    if (
        scorer.get("backend") != "sentence-transformers-cosine/v1"
        or not isinstance(logical_model_path, str)
        or not logical_model_path
        or Path(logical_model_path).is_absolute()
        or ".." in Path(logical_model_path).parts
        or isinstance(batch_size, bool)
        or not isinstance(batch_size, int)
        or batch_size <= 0
    ):
        raise ContextBuildError("frozen dev scorer lacks logical model path/batch size")
    model_path = resolve_embedding_model_path(logical_model_path, workspace)
    if _file_tree_sha256(model_path) != scorer.get("model_file_tree_sha256"):
        raise ContextBuildError("frozen embedding model file tree changed")
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ContextBuildError("sentence-transformers is required for seal-test") from exc
    device_class = scorer.get("device_class")
    if device_class not in {"cpu", "cuda"}:
        raise ContextBuildError("frozen scorer device class is invalid")
    train_records = json.loads((data_target / "train.json").read_text(encoding="utf-8"))
    fit_records = list(partition_bundle.fit_records)
    test_records = json.loads((data_target / "test.json").read_text(encoding="utf-8"))
    lexicon_rows = [
        json.loads(line)
        for line in (frozen_target / "catalogs/lexicon_pool.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    lexicon_terms = [
        {
            key: row[key]
            for key in ("lexicon_id", "term", "category", "definition", "variants")
        }
        for row in lexicon_rows
    ]
    runtime_identity = frozen_report.get("id_inputs", {}).get(
        "runtime_source_identity", {}
    )
    scorer_inventory = runtime_identity.get("scorer", {}).get("inventory")
    if not isinstance(scorer_inventory, dict):
        raise ContextBuildError("frozen dev context lacks scorer inventory")
    contract = ResolvedModelSourceContract(
        workspace_root=workspace,
        checkpoint_inventory=scorer_inventory,
        tokenizer_inventory=scorer_inventory,
        base_inventory=scorer_inventory,
    )
    try:
        with verified_model_source_lease(
            contract, source_names=("checkpoint",)
        ) as paths:
            model = SentenceTransformer(
                str(paths.checkpoint_path),
                device="cuda:0" if device_class == "cuda" else "cpu",
                local_files_only=True,
                trust_remote_code=False,
            )
            demo_scores, lexicon_scores = cosine_score_matrices(
                model=model,
                train_texts=[row["content"] for row in fit_records],
                query_texts=[row["content"] for row in test_records],
                lexicon_texts=[row["rendered_block"] for row in lexicon_rows],
                batch_size=batch_size,
            )
    except ModelRegistryError as exc:
        raise ContextBuildError(
            f"sealed scorer source lease failed: {exc}"
        ) from exc
    bundle = prepare_context_bundle_from_scores(
        train_records=train_records,
        query_records=test_records,
        lexicon_terms=lexicon_terms,
        demo_scores=demo_scores,
        lexicon_scores=lexicon_scores,
        split="test",
        retrieval_config=config["retrieval"],
        data_locator=data_locator,
        data_dependency=partition_bundle.data_dependency,
        lexicon_dependency=frozen_report["id_inputs"][
            "source_dependency_refs"
        ]["lexicon"],
        scorer_provenance=scorer,
        fit_demo_records=fit_records,
        calibration_ids=list(partition_bundle.calibration_ids),
        train_partition_dependency=partition_bundle.partition_dependency,
        train_partition_locator=partition_bundle.locator,
    )
    return seal_test_context_artifact(
        frozen_context_ref=args.frozen_context_ref,
        data_ref=args.data_ref,
        train_partition_ref=args.train_partition_ref,
        prepared_bundle=bundle,
        write_ref=args.write_ref,
        target_root=args.target_root,
        workspace_root=workspace,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build and validate Stage 1 immutable data/context artifacts."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    audit = subparsers.add_parser(
        "audit-data", help="Audit the frozen std train/test sources."
    )
    audit.add_argument("--config", required=True, type=Path)
    audit.add_argument("--review-rubric", required=True, type=Path)
    audit.add_argument("--write-ref", required=True, type=Path)

    export = subparsers.add_parser(
        "export-data-adjudication",
        help="Copy the immutable empty adjudication template to a review workspace.",
    )
    export.add_argument("--audit-ref", required=True, type=Path)
    export.add_argument("--output", required=True, type=Path)

    export_subset = subparsers.add_parser(
        "export-data-review-subset",
        help="Export a model-output-free human template and context packet for selected issues.",
    )
    export_subset.add_argument("--audit-ref", required=True, type=Path)
    export_subset.add_argument("--issue-kind", action="append", dest="issue_kinds", required=True)
    export_subset.add_argument("--reviewer-id", required=True)
    export_subset.add_argument("--output", required=True, type=Path)
    export_subset.add_argument("--packet-output", required=True, type=Path)

    merge = subparsers.add_parser(
        "merge-data-adjudication",
        help="Merge disjoint completed review streams into the exact frozen issue frame.",
    )
    merge.add_argument("--audit-ref", required=True, type=Path)
    merge.add_argument("--input", action="append", dest="inputs", required=True, type=Path)
    merge.add_argument("--reviewer-id", required=True)
    merge.add_argument("--output", required=True, type=Path)

    declaration = subparsers.add_parser(
        "prepare-data-declaration",
        help="Hash completed adjudication rows and prepare an unsigned declaration.",
    )
    declaration.add_argument("--audit-ref", required=True, type=Path)
    declaration.add_argument("--adjudication-file", required=True, type=Path)
    declaration.add_argument("--reviewer-id", required=True)
    declaration.add_argument("--write-template", required=True, type=Path)

    finalize = subparsers.add_parser(
        "finalize-data", help="Apply signed adjudication and create normalized splits."
    )
    finalize.add_argument("--config", required=True, type=Path)
    finalize.add_argument("--audit-ref", required=True, type=Path)
    finalize.add_argument("--data-blind-review-ref", required=True, type=Path)
    finalize.add_argument("--adjudication-file", required=True, type=Path)
    finalize.add_argument("--reviewer-declaration", required=True, type=Path)
    finalize.add_argument("--write-ref", required=True, type=Path)

    validate = subparsers.add_parser(
        "validate-data", help="Read-only validation of a finalized data target."
    )
    validate.add_argument("--data-ref", required=True, type=Path)

    prepare = subparsers.add_parser(
        "prepare-retrieval",
        help="Compute content-only BGE scores and freeze a train-only prepared bundle.",
    )
    prepare.add_argument("--data-ref", required=True, type=Path)
    prepare.add_argument("--train-partition-ref", type=Path)
    prepare.add_argument("--lexicon-ref", required=True, type=Path)
    prepare.add_argument("--config", required=True, type=Path)
    prepare.add_argument("--split", required=True, choices=["train", "dev"])
    prepare.add_argument("--embedding-model", required=True, type=Path)
    prepare.add_argument("--device", default="cuda:0")
    prepare.add_argument("--batch-size", type=int, default=64)
    prepare.add_argument("--limit", type=int)
    prepare.add_argument("--engineering", action="store_true")
    prepare.add_argument("--output", required=True, type=Path)

    build = subparsers.add_parser(
        "build",
        help="Freeze a prepared train-only retrieval bundle into C0/CL/CD/CLD context artifacts.",
    )
    build.add_argument("--config", required=True, type=Path)
    build.add_argument("--prepared-bundle", required=True, type=Path)
    build.add_argument("--data-ref", type=Path)
    build.add_argument("--train-partition-ref", type=Path)
    build.add_argument("--lexicon-ref", type=Path)
    build.add_argument("--workspace-root", type=Path, default=REPOSITORY_ROOT)
    build.add_argument(
        "--tokenizer",
        type=Path,
        help="Engineering-only tokenizer; formal builds resolve config.budget.tokenizer_path.",
    )
    build.add_argument("--target-root", type=Path)
    build.add_argument("--write-ref", required=True, type=Path)
    build.add_argument(
        "--engineering",
        action="store_true",
        help="Allow a synthetic/precomputed bundle without formal runtime refs.",
    )

    render = subparsers.add_parser(
        "render", help="Validate and report the already-frozen runner adapters."
    )
    render.add_argument("--context-ref", required=True, type=Path)
    render.add_argument("--tokenizer", type=Path, help="Required only for engineering artifacts.")

    context_validate = subparsers.add_parser(
        "validate", help="Read-only full replay validation of a context target."
    )
    context_validate.add_argument("--context-ref", required=True, type=Path)
    context_validate.add_argument(
        "--tokenizer", type=Path, help="Required only for engineering artifacts."
    )
    seal = subparsers.add_parser(
        "seal-test",
        help="Build test context once from exact frozen dev config/code/retrieval policy.",
    )
    seal.add_argument("--frozen-context-ref", required=True, type=Path)
    seal.add_argument("--data-ref", required=True, type=Path)
    seal.add_argument("--train-partition-ref", required=True, type=Path)
    seal.add_argument("--write-ref", required=True, type=Path)
    seal.add_argument("--target-root", type=Path)
    seal.add_argument("--workspace-root", type=Path, default=REPOSITORY_ROOT)
    return parser


def _print_result(value: object) -> None:
    print(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2))


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "audit-data":
            result = audit_data(
                config_path=args.config,
                review_rubric_path=args.review_rubric,
                write_ref=args.write_ref,
            )
        elif args.command == "export-data-adjudication":
            output = export_data_adjudication(
                audit_ref=args.audit_ref, output_path=args.output
            )
            result = {"exported": True, "output": str(output.resolve())}
        elif args.command == "export-data-review-subset":
            result = export_data_review_subset(
                audit_ref=args.audit_ref,
                issue_kinds=args.issue_kinds,
                reviewer_id=args.reviewer_id,
                output_path=args.output,
                packet_output_path=args.packet_output,
            )
        elif args.command == "merge-data-adjudication":
            result = merge_completed_adjudications(
                audit_ref=args.audit_ref,
                adjudication_files=args.inputs,
                reviewer_id=args.reviewer_id,
                output_path=args.output,
            )
        elif args.command == "prepare-data-declaration":
            result = prepare_data_declaration(
                audit_ref=args.audit_ref,
                adjudication_file=args.adjudication_file,
                reviewer_id=args.reviewer_id,
                write_template=args.write_template,
            )
        elif args.command == "finalize-data":
            result = finalize_data(
                config_path=args.config,
                audit_ref=args.audit_ref,
                data_blind_review_ref=args.data_blind_review_ref,
                adjudication_file=args.adjudication_file,
                reviewer_declaration=args.reviewer_declaration,
                write_ref=args.write_ref,
            )
        elif args.command == "validate-data":
            result = validate_data(data_ref=args.data_ref)
        elif args.command == "prepare-retrieval":
            result = _prepare_retrieval(args)
        elif args.command == "build":
            if args.engineering and args.tokenizer is None:
                raise ContextBuildError("engineering build requires --tokenizer")
            if not args.engineering and args.tokenizer is not None:
                raise ContextBuildError(
                    "formal build resolves tokenizer only from config; remove --tokenizer"
                )
            result = build_prepared_context_artifact(
                prepared_bundle=args.prepared_bundle,
                config=args.config,
                tokenizer=(
                    _tokenizer(args.tokenizer) if args.tokenizer is not None else None
                ),
                write_ref=args.write_ref,
                formal=not args.engineering,
                data_ref=args.data_ref,
                train_partition_ref=args.train_partition_ref,
                lexicon_ref=args.lexicon_ref,
                target_root=args.target_root,
                workspace_root=args.workspace_root,
            )
        elif args.command == "seal-test":
            result = _seal_test(args)
        elif args.command in {"render", "validate"}:
            if args.tokenizer is not None:
                locator = json.loads(args.context_ref.read_text(encoding="utf-8"))
                target = Path(str(locator.get("target_path", "")))
                metas = list(target.glob("context_manifest.*.meta.json"))
                if len(metas) == 1:
                    preview = json.loads(metas[0].read_text(encoding="utf-8"))
                    if preview.get("scientific_eligible") is True:
                        raise ContextBuildError(
                            "scientific validation resolves tokenizer only from frozen config; remove --tokenizer"
                        )
            result = validate_context_ref(
                args.context_ref,
                tokenizer=(
                    _tokenizer(args.tokenizer) if args.tokenizer is not None else None
                ),
            )
        else:  # pragma: no cover - argparse prevents this branch
            raise Stage1DataError(f"Unsupported command: {args.command}")
    except (
        Stage1DataError,
        ContextBuildError,
        RetrievalBundleError,
        TrainPartitionError,
        ModelRegistryError,
        OSError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    _print_result(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
