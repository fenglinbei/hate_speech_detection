from __future__ import annotations

import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest

from data.stage1_p0_validation import ValidationContext, _requirements, _sealed_boundary_check
from data.test_receipt import (
    ARTIFACT_KIND,
    TestReceiptError,
    build_test_receipt,
    validate_test_receipt,
)
from data.training_artifacts import load_json, resolve_locator_ref


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def _command(command_id: str, source: str) -> dict[str, object]:
    return {
        "command_id": command_id,
        "group": "fixture",
        "argv": ["{python}", "-c", source],
        "working_directory": ".",
        "timeout_seconds": 30,
    }


def _workspace(root: Path, *, failing: bool = False) -> Path:
    for directory in ("src", "scripts/stage1", "schemas", "config/stage1", "environment"):
        (root / directory).mkdir(parents=True, exist_ok=True)
    (root / "src/fixture.py").write_text("VALUE = 1\n", encoding="utf-8")
    (root / "scripts/stage1/fixture.py").write_text("VALUE = 2\n", encoding="utf-8")
    (root / "train.py").write_text("VALUE = 3\n", encoding="utf-8")
    (root / "run.py").write_text("VALUE = 4\n", encoding="utf-8")
    (root / "environment/stage1-p0.yml").write_text("name: fixture\n", encoding="utf-8")
    (root / "schemas/stage1_test_receipt_v1.schema.json").write_bytes(
        (REPOSITORY_ROOT / "schemas/stage1_test_receipt_v1.schema.json").read_bytes()
    )
    _write_json(
        root / "schemas/fixture.schema.json",
        {"$schema": "https://json-schema.org/draft/2020-12/schema", "type": "object"},
    )
    commands = [
        _command("verify.pass", "print('ok')"),
        _command(
            "verify.second",
            "import sys; print('warning', file=sys.stderr); "
            + ("raise SystemExit(7)" if failing else "print('done')"),
        ),
    ]
    _write_json(
        root / "config/stage1/p0_test_commands.json",
        {
            "schema_version": "stage1-test-command-manifest/v1",
            "execution_policy": "actual-subprocess-no-import/v1",
            "log_normalization_policy": "utf8-replace-crlf-lf-strip-ansi-portable-paths/v1",
            "commands": commands,
        },
    )
    _write_json(root / "config/stage1/fixture.json", {"frozen": True})
    return root


def test_receipt_executes_and_deep_validates_complete_all_pass_target(tmp_path: Path) -> None:
    root = _workspace(tmp_path / "workspace")
    locator, receipt = build_test_receipt(workspace_root=root, jobs=2)

    assert locator["artifact_kind"] == ARTIFACT_KIND
    assert receipt["all_passed"] is True
    assert receipt["command_count"] == 2
    report = validate_test_receipt(
        root / "exps/causal_context/stage1_p0/refs/verification_receipt_ref.json",
        workspace_root=root,
    )
    assert report["test_receipt_id"] == receipt["test_receipt_id"]
    assert report["all_passed"] is True
    _locator, target = resolve_locator_ref(
        root / "exps/causal_context/stage1_p0/refs/verification_receipt_ref.json",
        expected_kind=ARTIFACT_KIND,
    )
    assert (target / "logs/verify.pass.stdout.txt").read_text(encoding="utf-8") == "ok\n"
    assert load_json(target / "results.json")["ordered_command_ids"] == [
        "verify.pass",
        "verify.second",
    ]
    boundary, _summary = _sealed_boundary_check(
        ValidationContext(workspace_root=root.resolve(), mode="engineering-smoke")
    )
    assert boundary["status"] == "PASS"


def test_receipt_failure_publishes_neither_target_nor_locator(tmp_path: Path) -> None:
    root = _workspace(tmp_path / "workspace", failing=True)

    with pytest.raises(TestReceiptError, match="verify.second"):
        build_test_receipt(workspace_root=root, jobs=2)

    assert not (root / "exps/causal_context/stage1_p0/refs/verification_receipt_ref.json").exists()
    target_root = root / "exps/causal_context/stage1_p0/verification_receipts"
    assert not target_root.exists() or not list(target_root.iterdir())


def test_receipt_rejects_tampered_log(tmp_path: Path) -> None:
    root = _workspace(tmp_path / "workspace")
    build_test_receipt(workspace_root=root)
    ref = root / "exps/causal_context/stage1_p0/refs/verification_receipt_ref.json"
    _locator, target = resolve_locator_ref(ref, expected_kind=ARTIFACT_KIND)
    (target / "logs/verify.pass.stdout.txt").write_text("forged\n", encoding="utf-8")

    with pytest.raises(TestReceiptError):
        validate_test_receipt(ref, workspace_root=root)


def test_receipt_rejects_missing_result_log(tmp_path: Path) -> None:
    root = _workspace(tmp_path / "workspace")
    build_test_receipt(workspace_root=root)
    ref = root / "exps/causal_context/stage1_p0/refs/verification_receipt_ref.json"
    _locator, target = resolve_locator_ref(ref, expected_kind=ARTIFACT_KIND)
    (target / "logs/verify.second.stderr.txt").unlink()

    with pytest.raises(TestReceiptError):
        validate_test_receipt(ref, workspace_root=root)


def test_receipt_rejects_current_source_change_after_test(tmp_path: Path) -> None:
    root = _workspace(tmp_path / "workspace")
    build_test_receipt(workspace_root=root)
    ref = root / "exps/causal_context/stage1_p0/refs/verification_receipt_ref.json"
    (root / "src/fixture.py").write_text("VALUE = 99\n", encoding="utf-8")

    with pytest.raises(TestReceiptError, match="source/config/schema"):
        validate_test_receipt(ref, workspace_root=root)


def test_both_p0_modes_require_blocking_test_receipt() -> None:
    for mode in ("engineering-smoke", "formal-readiness"):
        requirements = {item.check_id: item for item in _requirements(mode)}
        receipt = requirements["tests.receipt"]
        assert receipt.expected_kinds == (ARTIFACT_KIND,)
        assert receipt.ref_names == ("verification_receipt_ref.json",)
        assert receipt.missing_status == "BLOCKED"


def test_frozen_manifest_covers_each_model_registry_node_exactly_once() -> None:
    manifest = load_json(REPOSITORY_ROOT / "config/stage1/p0_test_commands.json")
    prefix = "src/tests/test_stage1_model_registry.py::"
    frozen_nodes = [
        token
        for command in manifest["commands"]
        for token in command["argv"]
        if token.startswith(prefix)
    ]
    assert "src/tests/test_stage1_model_registry.py" not in {
        token
        for command in manifest["commands"]
        for token in command["argv"]
    }

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "--collect-only",
            "-q",
            "src/tests/test_stage1_model_registry.py",
        ],
        cwd=REPOSITORY_ROOT,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    collected_nodes = [
        line.strip()
        for line in completed.stdout.splitlines()
        if line.startswith(prefix)
    ]
    assert Counter(frozen_nodes) == Counter(collected_nodes)
    assert all(count == 1 for count in Counter(frozen_nodes).values())
