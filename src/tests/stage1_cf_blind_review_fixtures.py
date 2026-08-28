"""Deterministic test-only fixture for the mandatory CF blind-review lineage."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from data.training_artifacts import (
    load_json,
    load_jsonl,
    write_canonical_json,
    write_canonical_jsonl,
)
from review.cf_blind_review import (
    PANEL_REVIEWER_ID,
    export_cf_human_review,
    merge_cf_human_review,
    run_cf_blind_review,
)
from review.d14_contract import D14_SYNTHETIC_EXECUTION_MODE


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


class DeterministicCFReviewTransport:
    """Return one valid, high-confidence pass for every frozen reviewer call."""

    def __call__(
        self,
        url: str,
        request_payload: Mapping[str, Any],
        headers: Mapping[str, str],
        timeout: int,
    ) -> dict[str, Any]:
        del url, headers, timeout
        model = str(request_payload["model"])
        content = json.dumps(
            {
                "decision": "pass",
                "reason_code": "valid-local-foil",
                "note": "deterministic fixture pass",
                "confidence": 0.99,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return {
            "id": "fixture-cf-review-request",
            "model": model,
            "choices": [
                {"finish_reason": "stop", "message": {"content": content}}
            ],
        }


def build_completed_cf_blind_review_fixture(
    *,
    proposal_ref: str | Path,
    workspace_root: str | Path,
    stem: str,
) -> tuple[Path, Path, Path]:
    """Run two fake independent reviewers and explicitly complete any human queue."""

    root = Path(workspace_root).resolve()
    fixture_root = root / "cf_blind_review_fixtures" / stem
    fixture_root.mkdir(parents=True, exist_ok=True)
    env_file = fixture_root / ".env"
    env_file.write_text(
        "\n".join(
            (
                "GLM_API_KEY=fixture-glm-secret",
                "DEEPSEEK_API_KEY=fixture-deepseek-secret",
                "STAGE1_GLM_API_BASE=https://fixture-glm.invalid/v1",
                "STAGE1_DEEPSEEK_API_BASE=https://fixture-deepseek.invalid/v1",
                "STAGE1_GLM_MODEL=fixture-glm-model",
                "STAGE1_DEEPSEEK_MODEL=fixture-deepseek-model",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    policy = load_json(REPOSITORY_ROOT / "config/stage1/blind_review.json")
    policy_path = fixture_root / "blind_review.json"
    write_canonical_json(policy_path, policy)

    blind_ref = fixture_root / "cf_blind_review_ref.json"
    run_cf_blind_review(
        proposal_ref=proposal_ref,
        policy_path=policy_path,
        env_file=env_file,
        output_dir=fixture_root / "artifacts",
        write_ref=blind_ref,
        workspace_root=root,
        transport=DeterministicCFReviewTransport(),
        execution_mode=D14_SYNTHETIC_EXECUTION_MODE,
    )
    human_file = fixture_root / "human_completed.jsonl"
    packet_file = fixture_root / "human_packets.jsonl"
    export_cf_human_review(
        blind_review_ref=blind_ref,
        output=human_file,
        packet_output=packet_file,
        workspace_root=root,
    )
    human_rows = load_jsonl(human_file)
    for row in human_rows:
        row.update(
            {
                "decision": "pass",
                "reason_code": "valid-local-foil",
                "note": "explicit deterministic fixture human review",
                "reviewer_id": PANEL_REVIEWER_ID,
            }
        )
    write_canonical_jsonl(human_file, human_rows, key="candidate_id")

    merged = fixture_root / "merged_review.jsonl"
    declaration_path = fixture_root / "reviewer_declaration.json"
    merge_cf_human_review(
        blind_review_ref=blind_ref,
        human_completed=human_file,
        output=merged,
        declaration_output=declaration_path,
        workspace_root=root,
    )
    declaration = load_json(declaration_path)
    declaration["attestation_confirmed"] = True
    write_canonical_json(declaration_path, declaration)
    return blind_ref, merged, declaration_path


__all__ = [
    "DeterministicCFReviewTransport",
    "build_completed_cf_blind_review_fixture",
]
