from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

import data.stage1_p0_validation as validation_module
from data.stage1_p0_validation import (
    Stage1P0ValidationError,
    ValidationContext,
    _normalise_semantic_report,
    _requirements,
    _downstream_chain_check,
    _formal_exact_dependency_chain_check,
    _sealed_boundary_check,
    _smoke_determinism_check,
    validate_stage1_p0,
    write_report_sidecar,
)
from data.stage1_data_audit_validation import (
    Stage1DataAuditValidationError,
    validate_data_audit_target,
)
from data.training_artifacts import (
    build_payload_manifest,
    canonical_json_bytes,
    canonical_sha256,
    load_json,
    write_canonical_json,
    write_canonical_jsonl,
    write_locator_ref,
)
from model.stage1_registry import inventory_regular_file_tree


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
REPORT_SCHEMA = REPOSITORY_ROOT / "schemas/stage1_p0_validation_report_v1.schema.json"
ZERO_HASH = "0" * 64


def _write_decision_register(root: Path) -> None:
    destination = root / "config/stage1/decision_register.json"
    destination.parent.mkdir(parents=True)
    destination.write_bytes(
        (REPOSITORY_ROOT / "config/stage1/decision_register.json").read_bytes()
    )
    schema_root = root / "schemas"
    schema_root.mkdir(parents=True)
    for name in (
        "stage1_data_audit_v1.schema.json",
        "stage1_data_rubric_meta_v1.schema.json",
        "stage1_data_audit_issue_v1.schema.json",
    ):
        (schema_root / name).write_bytes((REPOSITORY_ROOT / "schemas" / name).read_bytes())


def _build_incomplete_audit(root: Path) -> tuple[Path, Path]:
    stage1_root = root / "exps/causal_context/stage1_p0"
    rubric = b"# Frozen fixture rubric\n"
    rubric_hash = hashlib.sha256(rubric).hexdigest()
    rubric_meta = {
        "schema_version": "stage1-data-adjudication-rubric-meta/v1",
        "rubric_version": "stage1-data-adjudication-rubric/v1",
        "rubric_body_sha256": rubric_hash,
        "declaration_schema_version": "stage1-data-reviewer-declaration/v1",
        "decision_codes": ["accepted", "corrected"],
        "reason_codes": {
            "group-hate-atypical": {
                "accepted": ["valid-independent-label-combination"],
                "corrected": ["correct-source-label"],
            }
        },
        "edit_contract": {},
    }
    rubric_meta_hash = canonical_sha256(rubric_meta)
    id_inputs = {
        "schema_version": "stage1-data-audit/v1",
        "source_inventory_sha256": ZERO_HASH,
        "split_policy_sha256": ZERO_HASH,
        "audit_config_sha256": ZERO_HASH,
        "audit_rule_version": "stage1-std-data-audit/v1",
        "normalization_schema_version": "stage1-source-normalization/v1",
        "rubric_body_sha256": rubric_hash,
        "rubric_meta_sha256": rubric_meta_hash,
        "audit_code_sha256": ZERO_HASH,
    }
    audit_id = "daudit-" + canonical_sha256(id_inputs)
    issue_id = "dissue:v1:" + "1" * 64
    issue = {
        "schema_version": "stage1-data-audit-issue/v1",
        "data_audit_id": audit_id,
        "issue_id": issue_id,
        "issue_kind": "group-hate",
        "issue_code": "group-hate-atypical",
        "issue_subtype": "fixture",
        "issue_rule_version": "stage1-data-issues/v1",
        "severity": "blocking",
        "locations": [
            {
                "source_key": "std-train",
                "source_file_sha256": ZERO_HASH,
                "source_ordinal": 0,
                "split": "train",
                "source_record_id": "1",
                "tuple_index": 0,
                "json_pointer": "/quadruples/0",
                "observed_value_sha256": ZERO_HASH,
            }
        ],
        "accept_allowed": True,
        "allowed_edit_paths": ["/quadruples/0/targeted_group"],
        "review_context": {},
    }
    template = {
        "schema_version": "stage1-data-adjudication-row/v1",
        "data_audit_id": audit_id,
        "issue_id": issue_id,
        "issue_kind": "group-hate",
        "decision": "",
        "edits": [],
        "reason_code": "",
        "reason": "",
        "reviewer_id": "fixture-reviewer",
        "reviewed_at": "",
    }
    meta = {
        "schema_version": "stage1-data-audit/v1",
        "data_audit_id": audit_id,
        "ordered_issue_ids_sha256": canonical_sha256([issue_id]),
        "blocking_issue_count": 1,
        "blocking_issue_counts_by_kind": {"group-hate": 1},
        "blocking_issue_counts_by_code": {"group-hate-atypical": 1},
        "blocking_issue_counts_by_split": {"train": 1},
        "warning_count": 0,
        "audit_id_inputs": id_inputs,
    }
    target = stage1_root / "data_audits" / audit_id
    target.mkdir(parents=True)
    (target / "adjudication_rubric.md").write_bytes(rubric)
    write_canonical_json(target / "adjudication_rubric.meta.json", rubric_meta)
    write_canonical_jsonl(target / "adjudication_template.jsonl", [template], key="issue_id")
    write_canonical_json(target / "audit.meta.json", meta)
    write_canonical_json(
        target / "audit_report.json",
        {
            "schema_version": "stage1-data-audit-report/v1",
            "data_audit_id": audit_id,
            "blocking": {"count": 1},
            "warnings": {"count": 0},
        },
    )
    write_canonical_json(target / "config.resolved.json", {"fixture": True})
    write_canonical_jsonl(target / "issues.jsonl", [issue], key="issue_id")
    write_canonical_json(
        target / "provenance.json",
        {
            "schema_version": "stage1-data-audit-provenance/v1",
            "data_audit_id": audit_id,
            "rubric_body_sha256": rubric_hash,
            "rubric_meta_sha256": rubric_meta_hash,
        },
    )
    write_canonical_json(target / "source_inventory.json", {"fixture": True})
    write_canonical_json(target / "split_manifest.proposed.json", {"fixture": True})
    write_canonical_json(target / "payload_manifest.json", build_payload_manifest(target))
    ref = stage1_root / "refs/data_audit_ref.json"
    write_locator_ref(
        ref,
        artifact_kind="data-audit",
        artifact_id=audit_id,
        target=target,
        payload_manifest_sha256=hashlib.sha256(
            (target / "payload_manifest.json").read_bytes()
        ).hexdigest(),
    )
    review_input = stage1_root / "review_inputs/data_adjudication.jsonl"
    write_canonical_jsonl(review_input, [template], key="issue_id")
    return target, ref


def _build_generation_ref(root: Path) -> None:
    stage1_root = root / "exps/causal_context/stage1_p0"
    artifact_id = "gen-" + "2" * 64
    target = stage1_root / "generation_runs" / artifact_id
    target.mkdir(parents=True)
    write_canonical_json(
        target / "generation.meta.json",
        {"schema_version": "fixture-generation/v1", "query_count": 20},
    )
    write_canonical_json(target / "payload_manifest.json", build_payload_manifest(target))
    write_locator_ref(
        stage1_root / "refs/generation_run_ref.json",
        artifact_kind="generation-run",
        artifact_id=artifact_id,
        target=target,
        payload_manifest_sha256=hashlib.sha256(
            (target / "payload_manifest.json").read_bytes()
        ).hexdigest(),
    )


def _generation_validator(
    _ref_path: Path,
    *,
    workspace_root: Path,
    require_scientific: bool = False,
) -> dict[str, Any]:
    assert workspace_root.is_absolute()
    assert require_scientific is False
    return {
        "scope": "engineering",
        "split": "dev",
        "sealing_status": "unsealed-dev",
        "query_count": 20,
        "complete_paired_blocks": True,
        "scientific_eligible": False,
    }


def _audit_validator(target: Path, *, workspace_root: Path) -> dict[str, int]:
    del workspace_root
    meta = load_json(target / "audit.meta.json")
    return {
        "blocking_issue_count": int(meta["blocking_issue_count"]),
        "warning_count": int(meta["warning_count"]),
    }


def _fixture_workspace(root: Path, *, generation: bool = True) -> tuple[Path, Path]:
    _write_decision_register(root)
    audit_target, audit_ref = _build_incomplete_audit(root)
    if generation:
        _build_generation_ref(root)
    (root / ".env").write_text("STAGE1_API_KEY=must-not-leak\n", encoding="utf-8")
    return audit_target, audit_ref


def test_structurally_self_consistent_fake_audit_fails_deep_replay(
    tmp_path: Path,
) -> None:
    root = tmp_path / "workspace"
    audit_target, _ = _fixture_workspace(root, generation=False)
    with pytest.raises(Stage1DataAuditValidationError):
        validate_data_audit_target(audit_target, workspace_root=root)


def test_repository_audit_replays_from_frozen_std_sources() -> None:
    root = Path(__file__).resolve().parents[2]
    locator = load_json(
        root / "exps/causal_context/stage1_p0/refs/data_audit_ref.json"
    )
    report = validate_data_audit_target(
        locator["target_path"], workspace_root=root
    )
    assert report["valid"] is True
    assert report["blocking_issue_count"] == 34
    assert report["warning_count"] == 99


def _validate(root: Path, *, mode: str = "engineering-smoke") -> dict[str, Any]:
    return validate_stage1_p0(
        workspace_root=root,
        mode=mode,
        validator_overrides={
            "audit": _audit_validator,
            "generation": _generation_validator,
        },
        schema_path=REPORT_SCHEMA,
    )


def _tree_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _portable_fixture_dependency(
    *, kind: str, artifact_id: str, logical_path: str, hash_digit: str
) -> dict[str, str]:
    return {
        "schema_version": "stage1-dependency-ref/v1",
        "artifact_kind": kind,
        "artifact_id": artifact_id,
        "payload_manifest_sha256": hash_digit * 64,
        "logical_repo_path": logical_path,
    }


def test_engineering_smoke_prefers_executed_legacy_refs() -> None:
    requirements = {item.check_id: item for item in _requirements("engineering-smoke")}

    assert requirements["context.dev"].ref_names[0] == "legacy_smoke_context_ref.json"
    assert "context.train" not in requirements
    assert (
        requirements["control.dev"].ref_names[0]
        == "legacy_smoke_control_diagnostic_ref.json"
    )
    assert (
        requirements["counterfactual.proposal"].ref_names[0]
        == "smoke_cf_proposal_ref.json"
    )
    assert (
        requirements["counterfactual.blind_review"].ref_names[0]
        == "smoke_cf_blind_review_ref.json"
    )
    assert requirements["model.registry"].expected_scope == "engineering-smoke"
    assert requirements["downstream.generation"].expected_scope == "engineering"
    assert requirements["downstream.evaluation"].expected_scope == "engineering-smoke"
    assert "data.train_partition" not in requirements


def test_formal_readiness_never_falls_back_to_smoke_refs() -> None:
    requirements = {item.check_id: item for item in _requirements("formal-readiness")}

    assert "model.legacy_smoke" not in requirements
    assert requirements["context.train"].ref_names == ("train_context_ref.json",)
    assert requirements["context.train"].expected_split == "train"
    assert requirements["context.dev"].ref_names == ("dev_context_ref.json",)
    assert requirements["context.dev"].expected_split == "dev"
    assert requirements["control.dev"].ref_names == ("control_ref.json",)
    assert requirements["counterfactual.proposal"].ref_names == (
        "cf_proposal_ref.json",
    )
    assert requirements["counterfactual.blind_review"].ref_names == (
        "cf_blind_review_ref.json",
    )
    assert requirements["data.train_partition"].ref_names == (
        "train_partition_ref.json",
    )
    assert requirements["data.train_partition"].expected_kinds == (
        "train-partition",
    )


def test_semantic_normalisation_preserves_registry_scope_fallback() -> None:
    report = _normalise_semantic_report(
        {"registry_scope": "engineering-smoke", "scientific_eligible": False}
    )

    assert report == {
        "registry_scope": "engineering-smoke",
        "scientific_eligible": False,
    }


def test_semantic_normalisation_preserves_cf_blind_review_identity() -> None:
    report = _normalise_semantic_report(
        {
            "cf_proposal_id": "cfp-" + "a" * 64,
            "cf_blind_review_id": "cfblind-" + "b" * 64,
            "review_id": "review:v1:" + "c" * 64,
        }
    )

    assert report == {
        "cf_blind_review_id": "cfblind-" + "b" * 64,
        "cf_proposal_id": "cfp-" + "a" * 64,
        "review_id": "review:v1:" + "c" * 64,
    }


def test_semantic_normalisation_preserves_safe_margin_cf_dependency() -> None:
    dependency = _portable_fixture_dependency(
        kind="counterfactual",
        artifact_id="cf-" + "a" * 64,
        logical_path="artifacts/counterfactuals/cf-" + "a" * 64,
        hash_digit="b",
    )

    report = _normalise_semantic_report({"cf_dependency": dependency})

    assert report["cf_dependency"] == dependency


def test_smoke_determinism_gate_rejects_fixture_and_accepts_two_real_passes(
    tmp_path: Path,
) -> None:
    context = ValidationContext(
        workspace_root=(tmp_path / "workspace").resolve(),
        mode="engineering-smoke",
    )
    context.semantic_reports["smoke.determinism"] = {
        "scope": "engineering",
        "split": "dev",
        "scientific_eligible": False,
        "complete_paired_blocks": True,
        "ordered_conditions": ["C0", "CL", "CD", "CLD", "PL", "PD"],
        "query_count": 20,
        "execution_repetitions": 1,
        "exact_rerun_match": True,
        "executor_backend": "fixture",
        "executor_id": "engineering-fixture/v1",
    }
    source = {"smoke.determinism": {"status": "PASS"}}

    rejected = _smoke_determinism_check(context, source)

    assert rejected["status"] == "FAIL"
    assert rejected["reason_code"] == "real-dev-two-pass-determinism-contract-mismatch"
    context.semantic_reports["smoke.determinism"].update(
        {
            "execution_repetitions": 2,
            "executor_backend": "transformers",
            "executor_id": "hf-local-transformers/v1",
        }
    )

    accepted = _smoke_determinism_check(context, source)

    assert accepted["status"] == "PASS"


def test_p0_tokenizer_load_is_full_tree_leased_and_remote_code_disabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = (tmp_path / "workspace").resolve()
    tokenizer_root = root / "models/tokenizer"
    tokenizer_root.mkdir(parents=True)
    (tokenizer_root / "tokenizer_config.json").write_text("{}\n", encoding="utf-8")
    (tokenizer_root / "tokenizer.json").write_text("{}\n", encoding="utf-8")
    inventory = inventory_regular_file_tree(
        tokenizer_root,
        workspace_root=root,
        inventory_policy="all-regular-files/v1",
    )
    artifact_id = "mdl-" + "d" * 64
    target = root / "artifacts/models" / artifact_id
    target.mkdir(parents=True)
    write_canonical_json(
        target / "model.json",
        {
            "checkpoint_inventory": inventory,
            "tokenizer_inventory": inventory,
            "base_inventory": inventory,
        },
    )
    write_canonical_json(target / "payload_manifest.json", build_payload_manifest(target))
    ref = root / "exps/causal_context/stage1_p0/refs/base_model_ref.json"
    write_locator_ref(
        ref,
        artifact_kind="stage1-model",
        artifact_id=artifact_id,
        target=target,
        payload_manifest_sha256=hashlib.sha256(
            (target / "payload_manifest.json").read_bytes()
        ).hexdigest(),
    )
    calls: list[tuple[Path, dict[str, Any]]] = []
    sentinel = object()

    def fake_from_pretrained(path: str | Path, **kwargs: Any) -> object:
        calls.append((Path(path).resolve(), kwargs))
        return sentinel

    import transformers

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", fake_from_pretrained)
    context = ValidationContext(workspace_root=root, mode="formal-readiness")

    loaded = validation_module._load_tokenizer(
        context,
        ref_path=root / "exps/causal_context/stage1_p0/refs/dev_context_ref.json",
    )

    assert loaded is sentinel
    assert calls == [
        (
            tokenizer_root.resolve(),
            {"trust_remote_code": False, "local_files_only": True},
        )
    ]


def test_engineering_generic_ref_uses_registered_legacy_tokenizer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = (tmp_path / "workspace").resolve()
    refs = root / "exps/causal_context/stage1_p0/refs"

    def register(name: str, marker: str) -> Path:
        tokenizer_root = root / "models" / marker
        tokenizer_root.mkdir(parents=True)
        (tokenizer_root / "tokenizer.json").write_text(
            '{"marker":"' + marker + '"}\n', encoding="utf-8"
        )
        inventory = inventory_regular_file_tree(
            tokenizer_root,
            workspace_root=root,
            inventory_policy="all-regular-files/v1",
        )
        artifact_id = "mdl-" + marker[0] * 64
        target = root / "artifacts/models" / artifact_id
        target.mkdir(parents=True)
        write_canonical_json(
            target / "model.json",
            {
                "checkpoint_inventory": inventory,
                "tokenizer_inventory": inventory,
                "base_inventory": inventory,
            },
        )
        write_canonical_json(
            target / "payload_manifest.json", build_payload_manifest(target)
        )
        write_locator_ref(
            refs / name,
            artifact_kind="stage1-model",
            artifact_id=artifact_id,
            target=target,
            payload_manifest_sha256=hashlib.sha256(
                (target / "payload_manifest.json").read_bytes()
            ).hexdigest(),
        )
        return tokenizer_root.resolve()

    register("base_model_ref.json", "base")
    legacy_root = register("legacy_model_ref.json", "legacy")
    loaded_paths: list[Path] = []
    sentinel = object()

    def fake_from_pretrained(path: str | Path, **kwargs: Any) -> object:
        assert kwargs == {"trust_remote_code": False, "local_files_only": True}
        loaded_paths.append(Path(path).resolve())
        return sentinel

    import transformers

    monkeypatch.setattr(
        transformers.AutoTokenizer, "from_pretrained", fake_from_pretrained
    )
    context = ValidationContext(workspace_root=root, mode="engineering-smoke")
    loaded = validation_module._load_tokenizer(
        context,
        ref_path=refs / "smoke_model_registry_ref.json",
    )

    assert loaded is sentinel
    assert loaded_paths == [legacy_root]


@pytest.mark.parametrize(
    ("field", "value", "reason_code"),
    (
        ("scope", "formal", "artifact-scope-mismatch"),
        ("split", "test", "artifact-split-mismatch"),
        (
            "scientific_eligible",
            True,
            "artifact-scientific-eligibility-mismatch",
        ),
        ("sealing_status", "sealed-test", "artifact-sealing-status-mismatch"),
    ),
)
def test_engineering_generation_contract_mismatch_is_fail_closed(
    tmp_path: Path, field: str, value: Any, reason_code: str
) -> None:
    root = tmp_path / "workspace"
    _fixture_workspace(root)

    def mismatched_validator(
        ref_path: Path,
        *,
        workspace_root: Path,
        require_scientific: bool = False,
    ) -> dict[str, Any]:
        report = _generation_validator(
            ref_path,
            workspace_root=workspace_root,
            require_scientific=require_scientific,
        )
        report[field] = value
        return report

    report = validate_stage1_p0(
        workspace_root=root,
        mode="engineering-smoke",
        validator_overrides={
            "audit": _audit_validator,
            "generation": mismatched_validator,
        },
        schema_path=REPORT_SCHEMA,
    )
    generation = next(
        check for check in report["checks"] if check["check_id"] == "downstream.generation"
    )

    assert generation["status"] == "FAIL"
    assert generation["reason_code"] == reason_code


def test_incomplete_human_review_is_blocked_and_validation_is_read_only(
    tmp_path: Path,
) -> None:
    root = tmp_path / "workspace"
    _fixture_workspace(root)
    before = _tree_hashes(root)

    first = _validate(root)
    second = _validate(root)

    assert first == second
    assert first["status"] == "BLOCKED"
    assert first["ready"] is False
    human = next(check for check in first["checks"] if check["check_id"] == "data.human_review")
    assert human["status"] == "BLOCKED"
    assert human["metrics"]["expected_issue_count"] == 1
    assert human["metrics"]["completed_review_count"] == 0
    assert _tree_hashes(root) == before
    rendered = json.dumps(first, ensure_ascii=False, sort_keys=True)
    assert str(root.resolve()) not in rendered
    assert "must-not-leak" not in rendered
    body = {key: value for key, value in first.items() if key != "report_sha256"}
    assert first["report_sha256"] == hashlib.sha256(canonical_json_bytes(body)).hexdigest()


def test_report_hash_is_independent_of_workspace_absolute_path(tmp_path: Path) -> None:
    left = tmp_path / "host-a/workspace"
    right = tmp_path / "host-b/workspace"
    _fixture_workspace(left)
    _fixture_workspace(right)

    left_report = _validate(left)
    right_report = _validate(right)

    assert left_report == right_report
    assert left_report["report_sha256"] == right_report["report_sha256"]
    generation = next(
        check for check in left_report["checks"] if check["check_id"] == "downstream.generation"
    )
    assert generation["status"] == "PASS"
    assert generation["artifact"]["logical_target_path"].startswith(
        "exps/causal_context/stage1_p0/generation_runs/"
    )


def test_formal_missing_artifacts_are_structured_gaps(tmp_path: Path) -> None:
    root = tmp_path / "workspace"
    _fixture_workspace(root, generation=False)

    report = _validate(root, mode="formal-readiness")
    gaps = {gap["check_id"]: gap for gap in report["gaps"]}

    assert report["status"] == "BLOCKED"
    assert gaps["data.human_review"]["status"] == "BLOCKED"
    assert gaps["training.plan"]["ref_path"].endswith("/training_plan_ref.json")
    assert gaps["context.train"]["ref_path"].endswith("/train_context_ref.json")
    assert gaps["context.dev"]["ref_path"].endswith("/dev_context_ref.json")
    assert gaps["model.registry"]["ref_path"].endswith("/model_registry_ref.json")
    assert gaps["model.registry"]["phase"] == "post-training-formal"
    assert gaps["downstream.analysis"]["status"] == "PENDING"


def test_formal_readiness_accepts_existing_engineering_smoke_proof(
    tmp_path: Path,
) -> None:
    root = tmp_path / "workspace"
    _fixture_workspace(root, generation=True)
    refs = root / "exps/causal_context/stage1_p0/refs"
    (refs / "legacy_smoke_generation_ref.json").write_bytes(
        (refs / "generation_run_ref.json").read_bytes()
    )
    observed_require_scientific: list[bool] = []

    def engineering_smoke_validator(
        _ref_path: Path,
        *,
        workspace_root: Path,
        require_scientific: bool = True,
    ) -> dict[str, Any]:
        assert workspace_root.is_absolute()
        observed_require_scientific.append(require_scientific)
        return {
            "scope": "engineering",
            "split": "dev",
            "sealing_status": "unsealed-dev",
            "query_count": 20,
            "complete_paired_blocks": True,
            "ordered_conditions": ["C0", "CL", "CD", "CLD", "PL", "PD"],
            "execution_repetitions": 2,
            "exact_rerun_match": True,
            "executor_backend": "transformers",
            "executor_id": "hf-local-transformers/v1",
            "scientific_eligible": False,
        }

    report = validate_stage1_p0(
        workspace_root=root,
        mode="formal-readiness",
        validator_overrides={
            "audit": _audit_validator,
            "generation": engineering_smoke_validator,
        },
        schema_path=REPORT_SCHEMA,
    )
    checks = {check["check_id"]: check for check in report["checks"]}

    assert observed_require_scientific == [False]
    assert checks["smoke.determinism"]["status"] == "PASS"
    assert checks["smoke.determinism_gate"]["status"] == "PASS"


def test_formal_downstream_is_not_opened_before_formal_registry_passes(
    tmp_path: Path,
) -> None:
    root = tmp_path / "workspace"
    _fixture_workspace(root, generation=True)

    def must_not_be_called(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        raise AssertionError("engineering downstream must not be opened by formal readiness")

    report = validate_stage1_p0(
        workspace_root=root,
        mode="formal-readiness",
        validator_overrides={
            "audit": _audit_validator,
            "generation": must_not_be_called,
        },
        schema_path=REPORT_SCHEMA,
    )
    generation = next(
        check for check in report["checks"] if check["check_id"] == "downstream.generation"
    )

    assert generation["status"] == "PENDING"
    assert generation["reason_code"] == "formal-upstream-prerequisite-not-ready"


def test_existing_invalid_artifact_is_fail_not_pending(tmp_path: Path) -> None:
    root = tmp_path / "workspace"
    target, _ref = _fixture_workspace(root, generation=False)
    (target / "adjudication_rubric.md").write_text("tampered\n", encoding="utf-8")

    report = _validate(root)
    audit = next(check for check in report["checks"] if check["check_id"] == "data.audit")

    assert report["status"] == "FAIL"
    assert audit["status"] == "FAIL"
    assert audit["reason_code"] == "data-audit-hash-chain-invalid"


def test_sealed_ref_is_detected_without_opening_its_content(tmp_path: Path) -> None:
    root = tmp_path / "workspace"
    _fixture_workspace(root, generation=False)
    sealed_ref = root / "exps/causal_context/stage1_p0/refs/test_context_ref.json"
    sealed_ref.write_text("not-json SEALED-CONTENT-MUST-NOT-BE-READ", encoding="utf-8")

    report = _validate(root)
    sealed = next(check for check in report["checks"] if check["check_id"] == "sealed.boundary")

    assert sealed["status"] == "FAIL"
    assert report["sealed_test"]["execution_performed"] is False
    assert report["sealed_test"]["target_content_read"] is False
    assert report["sealed_test"]["discovered_ref_names"] == ["test_context_ref.json"]
    assert "SEALED-CONTENT-MUST-NOT-BE-READ" not in json.dumps(report)


def test_generic_ref_to_sealed_target_aborts_before_artifact_validation(
    tmp_path: Path,
) -> None:
    root = tmp_path / "workspace"
    _fixture_workspace(root, generation=False)
    target = root / "exps/causal_context/stage1_p0/test_cf_proposals/SEALED"
    target.mkdir(parents=True)
    (target / "sentinel.txt").write_text(
        "GENERIC-SEALED-TARGET-MUST-NOT-BE-READ", encoding="utf-8"
    )
    write_canonical_json(
        root / "exps/causal_context/stage1_p0/refs/cf_proposal_ref.json",
        {
            "schema_version": "stage1-locator-ref/v1",
            "artifact_kind": "cf-proposal",
            "artifact_id": "cfp-" + "3" * 64,
            "target_path": str(target),
            "payload_manifest_sha256": "4" * 64,
        },
    )

    report = _validate(root)
    sealed = next(check for check in report["checks"] if check["check_id"] == "sealed.boundary")
    proposal = next(
        check for check in report["checks"] if check["check_id"] == "counterfactual.proposal"
    )

    assert sealed["status"] == "FAIL"
    assert "test/cf_proposal_ref.json" in report["sealed_test"][
        "discovered_ref_names"
    ]
    assert proposal["reason_code"] == "sealed-boundary-failed-validation-aborted"
    assert report["sealed_test"]["target_content_read"] is False
    assert "GENERIC-SEALED-TARGET-MUST-NOT-BE-READ" not in json.dumps(report)


def test_unreferenced_test_generation_target_is_detected_without_content_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "workspace"
    _fixture_workspace(root, generation=False)
    sensitive_root = (
        root
        / "exps/causal_context/stage1_p0/test_generations/stolen"
    )
    sensitive_root.mkdir(parents=True)
    sentinel = "RAW-TEST-GENERATION-MUST-NEVER-BE-READ-OR-REPORTED"
    (sensitive_root / "context_ref.json").write_text(sentinel, encoding="utf-8")
    original_reader = validation_module._read_boundary_metadata

    def guarded_reader(path: Path) -> Any:
        assert sensitive_root not in (path, *path.parents)
        return original_reader(path)

    monkeypatch.setattr(validation_module, "_read_boundary_metadata", guarded_reader)
    report = _validate(root)
    sealed = next(check for check in report["checks"] if check["check_id"] == "sealed.boundary")
    rendered = json.dumps(report, ensure_ascii=False, sort_keys=True)

    assert sealed["status"] == "FAIL"
    assert sealed["reason_code"] == "sealed-test-boundary-violation"
    assert any(
        label.startswith("test/path/generation/")
        for label in report["sealed_test"]["discovered_ref_names"]
    )
    assert "stolen" not in rendered
    assert sentinel not in rendered
    assert report["sealed_test"]["target_content_read"] is False


@pytest.mark.parametrize(
    ("relative_path", "expected_category", "as_directory"),
    (
        ("archive/.TeSt-EvAlUaTiOnS/stolen", "evaluation", True),
        ("archive/deeper/SealedTestMargins/stolen", "margin", True),
        ("archive/.sidecars/ANALYSIS_TEST_REF.JSON", "analysis", False),
    ),
)
def test_nested_hidden_and_case_aliases_cannot_bypass_sealed_scan(
    tmp_path: Path,
    relative_path: str,
    expected_category: str,
    as_directory: bool,
) -> None:
    root = tmp_path / "workspace"
    _fixture_workspace(root, generation=False)
    stage1_root = root / "exps/causal_context/stage1_p0"
    target = stage1_root / relative_path
    if as_directory:
        target.mkdir(parents=True)
        (target / "payload.bin").write_bytes(b"OPAQUE-TEST-PAYLOAD")
    else:
        target.parent.mkdir(parents=True)
        target.write_bytes(b"OPAQUE-TEST-SIDECAR")

    report = _validate(root)

    assert any(
        label.startswith(f"test/path/{expected_category}/")
        for label in report["sealed_test"]["discovered_ref_names"]
    )
    assert "OPAQUE-TEST" not in json.dumps(report, sort_keys=True)


def test_opaque_symlink_is_a_fail_closed_sealed_boundary_violation(
    tmp_path: Path,
) -> None:
    root = tmp_path / "workspace"
    _fixture_workspace(root, generation=False)
    external = tmp_path / "external-target"
    external.mkdir()
    sentinel = "SYMLINKED-RAW-TEST-CONTENT-MUST-NOT-BE-READ"
    (external / "raw.json").write_text(sentinel, encoding="utf-8")
    link = root / "exps/causal_context/stage1_p0/cache/.opaque"
    link.parent.mkdir(parents=True)
    link.symlink_to(external, target_is_directory=True)

    report = _validate(root)
    labels = report["sealed_test"]["discovered_ref_names"]
    rendered = json.dumps(report, ensure_ascii=False, sort_keys=True)

    assert any(label.startswith("test/symlink/artifact/") for label in labels)
    assert str(external) not in rendered
    assert sentinel not in rendered


def test_nested_dependency_ref_to_test_target_is_detected_without_target(
    tmp_path: Path,
) -> None:
    root = tmp_path / "workspace"
    _fixture_workspace(root, generation=False)
    stage1_root = root / "exps/causal_context/stage1_p0"
    nested_ref = stage1_root / "generation_runs/orphan/context_ref.json"
    write_canonical_json(
        nested_ref,
        {
            "schema_version": "stage1-locator-ref/v1",
            "artifact_kind": "test-context",
            "artifact_id": "tctx-" + "5" * 64,
            "target_path": str(stage1_root / "test_contexts/not-created"),
            "payload_manifest_sha256": "6" * 64,
        },
    )

    report = _validate(root)

    assert any(
        label.startswith("test/ref/context/")
        for label in report["sealed_test"]["discovered_ref_names"]
    )


def test_preseal_data_split_and_out_of_scope_synthetic_policy_are_allowed(
    tmp_path: Path,
) -> None:
    root = tmp_path / "workspace"
    _fixture_workspace(root, generation=False)
    stage1_root = root / "exps/causal_context/stage1_p0"
    data_target = stage1_root / ("data/data-" + "7" * 64)
    data_target.mkdir(parents=True)
    raw_sentinel = "ALLOWED-RAW-DATA-SPLIT-MUST-NOT-BE-READ"
    (data_target / "test.json").write_text(raw_sentinel, encoding="utf-8")
    synthetic = root / "artifacts/test_contexts/synthetic-fixture"
    synthetic.mkdir(parents=True)
    (synthetic / "query_pool.test.jsonl").write_text(
        "SYNTHETIC-ONLY", encoding="utf-8"
    )
    policy = root / "config/stage1/test_analysis_policy.json"
    policy.write_text('{"fixture":true}\n', encoding="utf-8")

    report = _validate(root)
    sealed = next(check for check in report["checks"] if check["check_id"] == "sealed.boundary")

    assert sealed["status"] == "PASS"
    assert report["sealed_test"]["discovered_ref_names"] == []
    assert raw_sentinel not in json.dumps(report, sort_keys=True)


def test_current_workspace_has_no_sealed_boundary_false_positive() -> None:
    check, summary = _sealed_boundary_check(
        ValidationContext(
            workspace_root=REPOSITORY_ROOT.resolve(),
            mode="engineering-smoke",
        )
    )

    assert check["status"] == "PASS"
    assert summary["discovered_ref_names"] == []
    assert summary["target_content_read"] is False


def test_formal_lineage_requires_selected_partition_through_every_consumer(
    tmp_path: Path,
) -> None:
    root = (tmp_path / "workspace").resolve()
    root.mkdir(parents=True)
    context = ValidationContext(workspace_root=root, mode="formal-readiness")
    specs = {
        "data.blind_review": ("data-blind-review", "dreview-" + "0" * 64, "0"),
        "data.normalized": ("data", "data-" + "a" * 64, "1"),
        "data.train_partition": ("train-partition", "tpart-" + "b" * 64, "2"),
        "data.lexicon": ("lexicon", "lex-" + "c" * 64, "3"),
        "infrastructure.environment": ("stage1-environment", "env-" + "d" * 64, "4"),
        "infrastructure.base_model": ("stage1-model", "mdl-" + "e" * 64, "5"),
        "context.train": ("context", "ctx-" + "0" * 64, "d"),
        "context.dev": ("context", "ctx-" + "f" * 64, "6"),
        "training.evidence": ("training-evidence", "tevd-" + "1" * 64, "7"),
        "training.plan": ("training-plan", "tpl-" + "2" * 64, "8"),
        "training.schedule": ("training-schedule", "sch-" + "3" * 64, "9"),
        "control.dev": ("control", "ctl-" + "4" * 64, "a"),
        "counterfactual.proposal": ("cf-proposal", "cfp-" + "5" * 64, "b"),
        "counterfactual.blind_review": (
            "cf-blind-review",
            "cfblind-" + "7" * 64,
            "d",
        ),
        "counterfactual.review": ("cf-review", "cfr-" + "8" * 64, "e"),
        "counterfactual.final": ("counterfactual", "cf-" + "9" * 64, "f"),
        "model.registry": ("stage1-model-registry", "mreg-" + "6" * 64, "c"),
    }
    for check_id, (kind, artifact_id, digit) in specs.items():
        short = check_id.replace(".", "-")
        target = root / "artifacts" / short
        target.mkdir(parents=True)
        context.artifact_targets[check_id] = target
        context.artifact_dependencies[check_id] = _portable_fixture_dependency(
            kind=kind,
            artifact_id=artifact_id,
            logical_path=f"artifacts/{short}",
            hash_digit=digit,
        )

    def bind(consumer: str, filename: str, producer: str) -> None:
        write_canonical_json(
            context.artifact_targets[consumer] / filename,
            context.artifact_dependencies[producer],
        )

    bind(
        "data.normalized",
        "data_blind_review_ref.json",
        "data.blind_review",
    )
    bind("data.train_partition", "data_ref.json", "data.normalized")
    bind("data.lexicon", "data_ref.json", "data.normalized")
    bind("data.lexicon", "train_partition_ref.json", "data.train_partition")
    bind("context.train", "data_ref.json", "data.normalized")
    bind("context.train", "train_partition_ref.json", "data.train_partition")
    bind("context.train", "lexicon_ref.json", "data.lexicon")
    bind("context.dev", "data_ref.json", "data.normalized")
    bind("context.dev", "train_partition_ref.json", "data.train_partition")
    bind("context.dev", "lexicon_ref.json", "data.lexicon")
    bind("training.evidence", "context_ref.json", "context.train")
    bind("training.evidence", "train_partition_ref.json", "data.train_partition")
    bind(
        "training.evidence",
        "base_model_ref.json",
        "infrastructure.base_model",
    )
    bind("training.plan", "training_evidence_ref.json", "training.evidence")
    bind("training.plan", "train_partition_ref.json", "data.train_partition")
    bind("training.plan", "base_model_ref.json", "infrastructure.base_model")
    bind("training.plan", "environment_ref.json", "infrastructure.environment")
    bind("training.schedule", "training_plan_ref.json", "training.plan")
    bind("training.schedule", "training_evidence_ref.json", "training.evidence")
    bind("training.schedule", "train_partition_ref.json", "data.train_partition")
    bind("control.dev", "context_ref.json", "context.dev")
    bind("counterfactual.proposal", "context_ref.json", "context.dev")
    bind(
        "counterfactual.blind_review",
        "proposal_ref.json",
        "counterfactual.proposal",
    )
    bind(
        "counterfactual.review",
        "proposal_ref.json",
        "counterfactual.proposal",
    )
    bind(
        "counterfactual.review",
        "cf_blind_review_ref.json",
        "counterfactual.blind_review",
    )
    bind("counterfactual.final", "context_ref.json", "context.dev")
    bind(
        "counterfactual.final",
        "proposal_ref.json",
        "counterfactual.proposal",
    )
    bind(
        "counterfactual.final",
        "review_ref.json",
        "counterfactual.review",
    )
    bind("model.registry", "training_plan_ref.json", "training.plan")

    shared_context_config = {
        "schema_version": "fixture-context-config/v1",
        "retrieval": {"policy": "same"},
        "rendering": {"policy": "same"},
        "budget": {"policy": "same"},
    }
    shared_retrieval = {
        "schema_version": "stage1-retrieval-provenance/v1",
        "policy_version": "stage1-fit-only-cosine-bundle/v1",
        "scorer": {"backend": "fixture", "model_sha256": "e" * 64},
    }
    for check_id, split in (("context.train", "train"), ("context.dev", "dev")):
        target = context.artifact_targets[check_id]
        write_canonical_json(target / "config.resolved.json", shared_context_config)
        write_canonical_json(
            target / "prepared_bundle.meta.json",
            {"retrieval_provenance": shared_retrieval},
        )
        write_canonical_json(
            target / f"context_manifest.{split}.meta.json",
            {"id_inputs": {"builder_code_sha256": "f" * 64}},
        )

    model_id = "mdl-" + "7" * 64
    model_target = root / "artifacts" / model_id
    model_target.mkdir(parents=True)
    for filename, producer in (
        ("training_plan_ref.json", "training.plan"),
        ("schedule_ref.json", "training.schedule"),
        ("base_model_ref.json", "infrastructure.base_model"),
        ("environment_ref.json", "infrastructure.environment"),
    ):
        write_canonical_json(model_target / filename, context.artifact_dependencies[producer])
    write_canonical_json(
        model_target / "payload_manifest.json", build_payload_manifest(model_target)
    )
    model_dependency = {
        "schema_version": "stage1-dependency-ref/v1",
        "artifact_kind": "stage1-model",
        "artifact_id": model_id,
        "payload_manifest_sha256": hashlib.sha256(
            (model_target / "payload_manifest.json").read_bytes()
        ).hexdigest(),
        "logical_repo_path": f"artifacts/{model_id}",
    }
    write_canonical_json(
        context.artifact_targets["model.registry"] / "registry.json",
        {"entries": [{"model_dependency": model_dependency}]},
    )

    checks = {check_id: {"status": "PASS"} for check_id in specs}
    assert _formal_exact_dependency_chain_check(context, checks)["status"] == "PASS"

    wrong_data_review = dict(context.artifact_dependencies["data.blind_review"])
    wrong_data_review["artifact_id"] = "dreview-" + "f" * 64
    write_canonical_json(
        context.artifact_targets["data.normalized"]
        / "data_blind_review_ref.json",
        wrong_data_review,
    )
    assert _formal_exact_dependency_chain_check(context, checks)["status"] == "FAIL"
    bind(
        "data.normalized",
        "data_blind_review_ref.json",
        "data.blind_review",
    )

    # The exact lineage must never confuse the selected dev context with the
    # train context consumed by training evidence.
    bind("training.evidence", "context_ref.json", "context.dev")
    assert _formal_exact_dependency_chain_check(context, checks)["status"] == "FAIL"
    bind("training.evidence", "context_ref.json", "context.train")

    wrong_review = dict(context.artifact_dependencies["counterfactual.review"])
    wrong_review["artifact_id"] = "cfr-" + "a" * 64
    write_canonical_json(
        context.artifact_targets["counterfactual.final"] / "review_ref.json",
        wrong_review,
    )
    assert _formal_exact_dependency_chain_check(context, checks)["status"] == "FAIL"
    bind(
        "counterfactual.final",
        "review_ref.json",
        "counterfactual.review",
    )

    wrong_blind_review = dict(
        context.artifact_dependencies["counterfactual.blind_review"]
    )
    wrong_blind_review["artifact_id"] = "cfblind-" + "a" * 64
    write_canonical_json(
        context.artifact_targets["counterfactual.review"]
        / "cf_blind_review_ref.json",
        wrong_blind_review,
    )
    assert _formal_exact_dependency_chain_check(context, checks)["status"] == "FAIL"
    bind(
        "counterfactual.review",
        "cf_blind_review_ref.json",
        "counterfactual.blind_review",
    )

    dev_config = dict(shared_context_config)
    dev_config["budget"] = {"policy": "different"}
    write_canonical_json(
        context.artifact_targets["context.dev"] / "config.resolved.json",
        dev_config,
    )
    assert _formal_exact_dependency_chain_check(context, checks)["status"] == "FAIL"
    write_canonical_json(
        context.artifact_targets["context.dev"] / "config.resolved.json",
        shared_context_config,
    )

    wrong_partition = dict(context.artifact_dependencies["data.train_partition"])
    wrong_partition["artifact_id"] = "tpart-" + "8" * 64
    write_canonical_json(
        context.artifact_targets["data.lexicon"] / "train_partition_ref.json",
        wrong_partition,
    )
    failed = _formal_exact_dependency_chain_check(context, checks)
    assert failed["status"] == "FAIL"
    assert failed["reason_code"] == "formal-exact-dependency-lineage-mismatch"


def test_downstream_chain_requires_selected_refs_in_exact_analysis_lineage(
    tmp_path: Path,
) -> None:
    root = (tmp_path / "workspace").resolve()
    context = ValidationContext(workspace_root=root, mode="engineering-smoke")
    targets = {
        key: root / "artifacts" / key
        for key in ("generation", "evaluation", "margin", "analysis")
    }
    for target in targets.values():
        target.mkdir(parents=True)

    def dependency(kind: str, atom: str) -> dict[str, Any]:
        return {
            "schema_version": "stage1-dependency-ref/v1",
            "artifact_kind": kind,
            "artifact_id": f"{atom}-" + atom[0] * 64,
            "payload_manifest_sha256": atom[-1] * 64,
            "logical_repo_path": f"artifacts/{atom}",
        }

    registry = dependency("stage1-model-registry", "registry")
    generation = dependency("generation-run", "generation")
    evaluation = dependency("evaluation", "evaluation")
    margin = dependency("margin", "margin")
    context_dep = dependency("context", "context")
    control_dep = dependency("control", "control")
    final_cf = dependency("counterfactual", "finalcf")
    for filename, value in (
        ("model_registry_ref.json", registry),
        ("context_ref.json", context_dep),
        ("control_ref.json", control_dep),
    ):
        write_canonical_json(targets["generation"] / filename, value)
        write_canonical_json(targets["margin"] / filename, value)
    write_canonical_json(targets["margin"] / "cf_ref.json", final_cf)
    write_canonical_json(
        targets["evaluation"] / "generation_run_ref.json", generation
    )
    write_canonical_json(targets["analysis"] / "model_registry_ref.json", registry)
    write_canonical_json(
        targets["analysis"] / "evaluation_refs.json",
        [{"model_key": "M_legacy/smoke", "dependency": evaluation}],
    )
    write_canonical_json(
        targets["analysis"] / "margin_refs.json",
        [{"model_key": "M_legacy/smoke", "dependency": margin}],
    )
    context.artifact_targets = {
        "downstream.generation": targets["generation"],
        "downstream.evaluation": targets["evaluation"],
        "downstream.margin": targets["margin"],
        "downstream.analysis": targets["analysis"],
    }
    context.artifact_dependencies = {
        "model.registry": registry,
        "context.dev": context_dep,
        "control.dev": control_dep,
        "counterfactual.final": final_cf,
        "downstream.generation": generation,
        "downstream.evaluation": evaluation,
        "downstream.margin": margin,
    }
    common = {
        "split": "dev",
        "scientific_eligible": False,
        "sealing_status": "unsealed-dev",
        "model_key": "M_legacy/smoke",
        "role": "legacy-smoke-only",
        "seed": None,
        "query_count": 20,
    }
    context.semantic_reports = {
        "downstream.generation": {
            **common,
            "scope": "engineering",
            "generation_run_id": generation["artifact_id"],
        },
        "downstream.evaluation": {
            **common,
            "scope": "engineering-smoke",
            "generation_run_id": generation["artifact_id"],
        },
        "downstream.margin": {
            **common,
            "scope": "engineering-smoke",
            "cf_dependency": final_cf,
        },
        "downstream.analysis": {**common, "scope": "engineering-smoke"},
    }
    checks = {
        key: {"status": "PASS"}
        for key in context.semantic_reports
    }
    checks["counterfactual.final"] = {"status": "PASS"}
    assert _downstream_chain_check(context, checks)["status"] == "PASS"

    checks["counterfactual.final"] = {"status": "PENDING"}
    assert _downstream_chain_check(context, checks)["status"] == "PENDING"
    checks["counterfactual.final"] = {"status": "PASS"}

    # A fully valid margin branch for CF-B must not satisfy a top-level CF-A
    # selection, even when both the file and semantic report agree on CF-B.
    swapped_cf = dependency("counterfactual", "swappedcf")
    write_canonical_json(targets["margin"] / "cf_ref.json", swapped_cf)
    context.semantic_reports["downstream.margin"]["cf_dependency"] = swapped_cf
    swapped = _downstream_chain_check(context, checks)
    assert swapped["status"] == "FAIL"
    assert swapped["reason_code"] == "downstream-exact-dependency-lineage-mismatch"
    write_canonical_json(targets["margin"] / "cf_ref.json", final_cf)
    context.semantic_reports["downstream.margin"]["cf_dependency"] = final_cf

    wrong = dependency("evaluation", "different")
    write_canonical_json(
        targets["analysis"] / "evaluation_refs.json",
        [{"model_key": "M_legacy/smoke", "dependency": wrong}],
    )
    failed = _downstream_chain_check(context, checks)
    assert failed["status"] == "FAIL"
    assert failed["reason_code"] == "downstream-exact-dependency-lineage-mismatch"


def test_sidecar_write_is_atomic_and_rejects_immutable_targets(tmp_path: Path) -> None:
    root = tmp_path / "workspace"
    audit_target, _audit_ref = _fixture_workspace(root, generation=False)
    report = _validate(root)
    destination = root / "reports/stage1-p0.json"

    write_report_sidecar(report, destination, workspace_root=root)

    assert destination.read_bytes() == canonical_json_bytes(report) + b"\n"
    with pytest.raises(Stage1P0ValidationError, match="immutable target"):
        write_report_sidecar(
            report,
            audit_target / "validation.json",
            workspace_root=root,
        )
    with pytest.raises(Stage1P0ValidationError, match="refs directory"):
        write_report_sidecar(
            report,
            root / "exps/causal_context/stage1_p0/refs/p0.json",
            workspace_root=root,
        )
