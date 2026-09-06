#!/usr/bin/env python3
"""Build, preflight, execute, finalize, and evaluate the WP3 S2.1b dev run."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
for import_root in (REPOSITORY_ROOT, SOURCE_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from build_lex.terminology_candidate_successor_run import (  # noqa: E402
    PrivateResponseVault,
    ProviderCheckpoint,
    RequestsProviderTransport,
    SuccessorRunError,
    build_successor_plan_from_artifacts,
    evaluate_development_run,
    finalize_run_artifacts,
    load_provider_credentials,
    load_successor_public_tasks,
    preflight_offline,
    run_formal_slots,
    run_live_preflight,
    validate_development_evaluation,
    validate_run_artifact,
    validate_successor_plan,
    write_development_evaluation,
    write_successor_plan,
)
from data.training_artifacts import load_json, load_jsonl  # noqa: E402


def _path(root: Path, value: Path) -> Path:
    return value if value.is_absolute() else root / value


def _checkpoint(path: Path, plan: dict, mode: str) -> ProviderCheckpoint:
    if path.exists():
        return ProviderCheckpoint.resume(path, plan=plan, mode=mode)
    return ProviderCheckpoint.create(path, plan=plan, mode=mode)


def _vault(path: Path, plan: dict, credentials: dict[str, str]) -> PrivateResponseVault:
    if path.exists():
        return PrivateResponseVault(
            path,
            plan_id=plan["plan_id"],
            forbidden_values=tuple(credentials.values()),
        )
    return PrivateResponseVault.create(
        path,
        plan_id=plan["plan_id"],
        forbidden_values=tuple(credentials.values()),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace-root", type=Path, default=REPOSITORY_ROOT)
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build-run-plan")
    build.add_argument("--current-plan", type=Path, required=True)
    build.add_argument(
        "--current-plan-schema",
        type=Path,
        default=Path("schemas/wp3_candidate_current_run_plan_v1.schema.json"),
    )
    build.add_argument("--g3-run-ref", type=Path, required=True)
    build.add_argument("--pricing-evidence", type=Path, required=True)
    build.add_argument("--pricing-bundle", type=Path)
    build.add_argument("--returned-model-allowlists", type=Path)
    build.add_argument("--output", type=Path, required=True)

    offline = sub.add_parser("preflight-offline")
    offline.add_argument("--plan", type=Path, required=True)

    live = sub.add_parser("run-live-preflight")
    live.add_argument("--plan", type=Path, required=True)
    live.add_argument("--checkpoint", type=Path, required=True)
    live.add_argument("--vault", type=Path, required=True)
    live.add_argument("--g3-run-ref", type=Path, required=True)
    live.add_argument("--confirm-six-paid-calls", action="store_true")

    status = sub.add_parser("status")
    status.add_argument("--plan", type=Path, required=True)
    status.add_argument("--checkpoint", type=Path, required=True)
    status.add_argument(
        "--mode", choices=("formal", "synthetic-preflight"), required=True
    )

    run = sub.add_parser("run")
    run.add_argument("--plan", type=Path, required=True)
    run.add_argument("--authorization", type=Path, required=True)
    run.add_argument("--frame-ref", type=Path, required=True)
    run.add_argument("--checkpoint", type=Path, required=True)
    run.add_argument("--vault", type=Path, required=True)

    finalize = sub.add_parser("finalize-run")
    finalize.add_argument("--plan", type=Path, required=True)
    finalize.add_argument("--frame-ref", type=Path, required=True)
    finalize.add_argument("--checkpoint", type=Path, required=True)
    finalize.add_argument("--vault", type=Path, required=True)
    finalize.add_argument("--g3-run-ref", type=Path, required=True)
    finalize.add_argument("--public-output-parent", type=Path, required=True)
    finalize.add_argument("--private-output-parent", type=Path, required=True)
    finalize.add_argument("--public-ref", type=Path)
    finalize.add_argument("--private-ref", type=Path)
    finalize.add_argument("--expected-previous-public-id")
    finalize.add_argument("--expected-previous-private-id")

    validate = sub.add_parser("validate-run")
    validate.add_argument("--run-dir", type=Path, required=True)
    validate.add_argument("--frame-ref", type=Path, required=True)
    validate.add_argument("--private-vault-dir", type=Path, required=True)

    evaluate = sub.add_parser("evaluate-dev")
    evaluate.add_argument("--run-dir", type=Path, required=True)
    evaluate.add_argument("--private-vault-dir", type=Path, required=True)
    evaluate.add_argument("--raw-gold-ref", type=Path, required=True)
    evaluate.add_argument("--frame-dir", type=Path, required=True)
    evaluate.add_argument("--legacy-generator-config", type=Path, required=True)
    evaluate.add_argument("--protocol", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)

    validate_eval = sub.add_parser("validate-dev-evaluation")
    validate_eval.add_argument("--evaluation", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = args.workspace_root.resolve()
    exit_code = 0
    try:
        if args.command == "build-run-plan":
            aliases = (
                load_json(_path(root, args.returned_model_allowlists))
                if args.returned_model_allowlists
                else None
            )
            plan = build_successor_plan_from_artifacts(
                current_plan_path=_path(root, args.current_plan),
                current_plan_schema_path=_path(root, args.current_plan_schema),
                g3_run_ref=_path(root, args.g3_run_ref),
                pricing_evidence_path=_path(root, args.pricing_evidence),
                pricing_bundle_path=(
                    _path(root, args.pricing_bundle)
                    if args.pricing_bundle is not None
                    else None
                ),
                workspace_root=root,
                returned_model_allowlists=aliases,
            )
            write_successor_plan(_path(root, args.output), plan)
            output = {"plan_id": plan["plan_id"], "output": str(args.output)}
        elif args.command == "preflight-offline":
            plan = validate_successor_plan(_path(root, args.plan), workspace_root=root)
            output = preflight_offline(plan)
        elif args.command == "run-live-preflight":
            if not args.confirm_six_paid_calls:
                raise SuccessorRunError(
                    "live preflight requires --confirm-six-paid-calls"
                )
            plan = validate_successor_plan(_path(root, args.plan), workspace_root=root)
            credentials = load_provider_credentials(
                plan, workspace_root=root
            )
            checkpoint = _checkpoint(
                _path(root, args.checkpoint), plan, "synthetic-preflight"
            )
            try:
                vault = _vault(_path(root, args.vault), plan, credentials)
                transport = RequestsProviderTransport()
                try:
                    output = run_live_preflight(
                        plan=plan,
                        checkpoint=checkpoint,
                        vault=vault,
                        transport=transport,
                        credentials=credentials,
                        g3_run_ref=_path(root, args.g3_run_ref),
                    )
                    if output.get("protocol_gate_passed") is not True:
                        exit_code = 3
                finally:
                    transport.close()
            finally:
                checkpoint.close()
        elif args.command == "status":
            plan = validate_successor_plan(_path(root, args.plan), workspace_root=root)
            with ProviderCheckpoint.resume(
                _path(root, args.checkpoint), plan=plan, mode=args.mode
            ) as checkpoint:
                output = checkpoint.summary()
        elif args.command == "run":
            plan = validate_successor_plan(_path(root, args.plan), workspace_root=root)
            # run_formal_slots validates the independent receipt before invoking
            # transport_factory; loading credentials has no external effect.
            credentials = load_provider_credentials(plan, workspace_root=root)
            tasks = load_successor_public_tasks(
                plan, frame_ref=_path(root, args.frame_ref)
            )
            checkpoint = _checkpoint(_path(root, args.checkpoint), plan, "formal")
            try:
                vault = _vault(_path(root, args.vault), plan, credentials)
                output = run_formal_slots(
                    plan=plan,
                    authorization=_path(root, args.authorization),
                    workspace_root=root,
                    tasks=tasks,
                    checkpoint=checkpoint,
                    vault=vault,
                    transport_factory=RequestsProviderTransport,
                    credentials=credentials,
                )
            finally:
                checkpoint.close()
        elif args.command == "finalize-run":
            plan = validate_successor_plan(_path(root, args.plan), workspace_root=root)
            tasks = load_successor_public_tasks(
                plan, frame_ref=_path(root, args.frame_ref)
            )
            from build_lex.terminology_candidate_successor_run import (
                build_successor_slot_grid,
            )

            with ProviderCheckpoint.resume(
                _path(root, args.checkpoint), plan=plan, mode="formal"
            ) as checkpoint:
                vault = PrivateResponseVault(
                    _path(root, args.vault), plan_id=plan["plan_id"]
                )
                output = finalize_run_artifacts(
                    plan=plan,
                    checkpoint=checkpoint,
                    vault=vault,
                    slots=build_successor_slot_grid(plan, tasks),
                    public_tasks=tasks,
                    g3_run_ref=_path(root, args.g3_run_ref),
                    public_output_parent=_path(root, args.public_output_parent),
                    private_output_parent=_path(root, args.private_output_parent),
                    public_ref=(
                        _path(root, args.public_ref) if args.public_ref else None
                    ),
                    private_ref=(
                        _path(root, args.private_ref) if args.private_ref else None
                    ),
                    expected_previous_public_id=args.expected_previous_public_id,
                    expected_previous_private_id=args.expected_previous_private_id,
                )
        elif args.command == "validate-run":
            result = validate_run_artifact(
                _path(root, args.run_dir),
                frame_ref=_path(root, args.frame_ref),
                private_vault_dir=_path(root, args.private_vault_dir),
            )
            output = {
                "run_id": result["run_id"],
                "payload_manifest_sha256": result["payload_manifest_sha256"],
            }
        elif args.command == "evaluate-dev":
            report = evaluate_development_run(
                run_dir=_path(root, args.run_dir),
                private_vault_dir=_path(root, args.private_vault_dir),
                raw_gold_ref=_path(root, args.raw_gold_ref),
                frame_dir=_path(root, args.frame_dir),
                workspace_root=root,
                legacy_generator_config_path=_path(
                    root, args.legacy_generator_config
                ),
                protocol_path=_path(root, args.protocol),
            )
            write_development_evaluation(_path(root, args.output), report)
            output = report
        else:
            output = validate_development_evaluation(
                _path(root, args.evaluation)
            )
    except (SuccessorRunError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(output, ensure_ascii=False, sort_keys=True, indent=2))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
