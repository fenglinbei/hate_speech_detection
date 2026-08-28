"""Small-content fixtures that satisfy the real Stage-1 artifact contracts.

The finalized data contract intentionally fixes the production split frame, so
these fixtures keep that frame while using compact deterministic record text.
They do not patch or bypass production validators.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from data.training_artifacts import (
    build_payload_manifest,
    canonical_sha256,
    validate_payload_manifest,
    write_bytes_atomic,
    write_canonical_json,
    write_locator_ref,
)


SPLIT_COUNTS = {"train": 5781, "dev": 643, "test": 1605}


def make_context_tokenizer_source_fixture(root: Path) -> Path:
    """Create stable source bytes for tests that patch the backend constructor."""

    target = root / "models" / "base" / "Qwen3-8B"
    target.mkdir(parents=True, exist_ok=True)
    write_canonical_json(
        target / "tokenizer_config.json",
        {
            "fixture_only": True,
            "revision": "qwen3-8b-stage1-v1",
            "trust_remote_code": False,
        },
    )
    write_bytes_atomic(
        target / "tokenizer.json", b"stage1-context-tokenizer-source-fixture/v1\n"
    )
    return target


def _record(identifier: int, *, split: str) -> dict[str, Any]:
    content = "train text" if identifier == 1 else f"{split}-{identifier}"
    return {
        "id": str(identifier),
        "content": content,
        "quadruples": [
            {
                "target": None,
                "argument": content,
                "targeted_group": ["Racism"],
                "hateful": "hate",
            }
        ],
    }


def make_semantic_data_artifact(
    root: Path,
    *,
    collapse_calibration_cluster: bool = False,
) -> tuple[Path, dict[str, list[dict[str, Any]]]]:
    """Create a validator-clean, content-addressed finalized data fixture."""

    make_context_tokenizer_source_fixture(root)

    starts = {"train": 1, "dev": 5782, "test": 6425}
    splits = {
        split: [
            _record(identifier, split=split)
            for identifier in range(start, start + SPLIT_COUNTS[split])
        ]
        for split, start in starts.items()
    }
    if collapse_calibration_cluster:
        shared_content = "synthetic calibration content cluster"
        for record in splits["train"][5:]:
            record["content"] = shared_content
            record["quadruples"] = [
                {
                    "target": None,
                    "argument": shared_content,
                    "targeted_group": ["Racism"],
                    "hateful": "hate",
                }
            ]
    split_manifest = {
        "schema_version": "stage1-split/v1",
        "source_sha256": {"std-train": "1" * 64, "std-test": "2" * 64},
        "policy": "prefix-5781-643/v1",
        "id_policy": "canonical-decimal-string/v1",
    }
    for split, records in splits.items():
        identifiers = [record["id"] for record in records]
        split_manifest[f"{split}_ids"] = identifiers
        split_manifest[f"{split}_ids_sha256"] = canonical_sha256(identifiers)

    audit_dependency = {
        "schema_version": "stage1-dependency-ref/v1",
        "artifact_kind": "data-audit",
        "artifact_id": "daudit-" + "a" * 64,
        "payload_manifest_sha256": "b" * 64,
        "logical_repo_path": "artifacts/data_audits/daudit-" + "a" * 64,
    }
    blind_review_dependency = {
        "schema_version": "stage1-dependency-ref/v1",
        "artifact_kind": "data-blind-review",
        "artifact_id": "dreview-" + "b" * 64,
        "payload_manifest_sha256": "c" * 64,
        "logical_repo_path": "artifacts/data_blind_reviews/dreview-" + "b" * 64,
    }
    empty_jsonl_sha256 = hashlib.sha256(b"").hexdigest()
    declaration = {
        "schema_version": "stage1-data-reviewer-declaration/v1",
        "data_audit_id": audit_dependency["artifact_id"],
        "reviewer_id": "dual-blind-panel-v1",
        "rubric_body_sha256": "3" * 64,
        "rubric_meta_sha256": "4" * 64,
        "ordered_issue_ids_sha256": canonical_sha256([]),
        "completed_rows_sha256": empty_jsonl_sha256,
        "saw_condition_outputs": False,
        "saw_model_scores": False,
        "attestation_confirmed": True,
    }
    queue_rows: list[dict[str, Any]] = []
    adjudication_frame = {
        "schema_version": "stage1-data-adjudication-frame/v1",
        "data_audit_dependency": audit_dependency,
        "data_blind_review_dependency": blind_review_dependency,
        "blind_scope": {
            "issue_kinds": ["group-hate"],
            "issue_ids_sha256": canonical_sha256([]),
            "auto_issue_ids_sha256": canonical_sha256([]),
            "human_issue_ids_sha256": canonical_sha256([]),
        },
        "audit_human_only_scope": {
            "issue_kinds": ["field-type"],
            "issue_ids_sha256": canonical_sha256([]),
        },
        "blind_auto_adjudication_sha256": empty_jsonl_sha256,
        "blind_human_completed_sha256": empty_jsonl_sha256,
        "audit_human_only_completed_sha256": empty_jsonl_sha256,
        "human_adjudication_queue_sha256": empty_jsonl_sha256,
        "final_adjudication_rows_sha256": empty_jsonl_sha256,
    }
    adjudication_frame_sha256 = canonical_sha256(adjudication_frame)
    data_review_id = "dreview-" + canonical_sha256(
        {
            "data_audit_id": audit_dependency["artifact_id"],
            "data_blind_review_dependency": blind_review_dependency,
            "adjudication_rows_sha256": empty_jsonl_sha256,
            "human_adjudication_queue_sha256": empty_jsonl_sha256,
            "adjudication_frame_sha256": adjudication_frame_sha256,
            "declaration_sha256": canonical_sha256(declaration),
        }
    )
    finalizer_code_sha256 = "d" * 64
    split_payloads = {
        split: json.dumps(
            records,
            ensure_ascii=False,
            sort_keys=False,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
        for split, records in splits.items()
    }
    data_id_inputs = {
        "schema_version": "stage1-data-manifest/v1",
        "data_audit_dependency": audit_dependency,
        "data_blind_review_dependency": blind_review_dependency,
        "data_review_id": data_review_id,
        "adjudication_rows_sha256": empty_jsonl_sha256,
        "adjudication_log_sha256": empty_jsonl_sha256,
        "human_adjudication_queue_sha256": empty_jsonl_sha256,
        "adjudication_frame_sha256": adjudication_frame_sha256,
        "reviewer_declaration_sha256": canonical_sha256(declaration),
        "adjudication_input_sha256": "f" * 64,
        "declaration_input_sha256": "0" * 64,
        "split_manifest_sha256": canonical_sha256(split_manifest),
        "source_inventory_sha256": "1" * 64,
        "substring_warning_count": 0,
        "train_content_sha256": hashlib.sha256(split_payloads["train"]).hexdigest(),
        "dev_content_sha256": hashlib.sha256(split_payloads["dev"]).hexdigest(),
        "test_content_sha256": hashlib.sha256(split_payloads["test"]).hexdigest(),
        "split_policy_version": "prefix-5781-643/v1",
        "normalization_policy_version": "stage1-source-normalization/v1",
        "output_schema_version": "stage1-normalized-record/v1",
        "finalizer_code_sha256": finalizer_code_sha256,
    }
    data_id = "data-" + canonical_sha256(data_id_inputs)
    target = root / "artifacts" / "data" / data_id
    target.mkdir(parents=True)
    write_canonical_json(target / "audit_ref.json", audit_dependency)
    write_canonical_json(
        target / "data_blind_review_ref.json", blind_review_dependency
    )
    for split, payload in split_payloads.items():
        write_bytes_atomic(target / f"{split}.json", payload)
    write_canonical_json(target / "split_manifest.json", split_manifest)
    write_bytes_atomic(target / "adjudication_rows.jsonl", b"")
    write_bytes_atomic(target / "adjudication_log.jsonl", b"")
    write_bytes_atomic(target / "human_adjudication_queue.jsonl", b"")
    write_canonical_json(target / "adjudication_frame.json", adjudication_frame)
    write_canonical_json(target / "reviewer_declaration.json", declaration)
    write_canonical_json(
        target / "audit_report.json",
        {
            "schema_version": "stage1-data-finalization-report/v1",
            "data_build_id": data_id,
            "data_review_id": data_review_id,
            "blocking_issue_count": 0,
            "resolved_issue_count": 0,
            "decision_counts": {},
            "accepted_group_hate_warning_count": 0,
            "substring_warning_count": 0,
            "post_finalize_schema_valid": True,
        },
    )
    write_canonical_json(
        target / "provenance.json",
        {
            "schema_version": "stage1-data-provenance/v1",
            "data_build_id": data_id,
            "data_id_inputs": data_id_inputs,
            "source_inventory_sha256": "1" * 64,
            "split_manifest_sha256": canonical_sha256(split_manifest),
            "finalizer_code_sha256": finalizer_code_sha256,
        },
    )
    write_canonical_json(target / "payload_manifest.json", build_payload_manifest(target))
    ref = root / "refs" / "data_ref.json"
    write_locator_ref(
        ref,
        artifact_kind="data",
        artifact_id=data_id,
        target=target,
        payload_manifest_sha256=validate_payload_manifest(target),
    )
    return ref, splits


def readdress_semantic_data_artifact(data_ref: Path) -> str:
    """Legally rebuild a mutated semantic fixture under a fresh content ID."""

    locator = json.loads(data_ref.read_text(encoding="utf-8"))
    target = Path(locator["target_path"])
    split_manifest = json.loads(
        (target / "split_manifest.json").read_text(encoding="utf-8")
    )
    split_payloads: dict[str, bytes] = {}
    for split in ("train", "dev", "test"):
        records = json.loads((target / f"{split}.json").read_text(encoding="utf-8"))
        identifiers = [record["id"] for record in records]
        split_manifest[f"{split}_ids"] = identifiers
        split_manifest[f"{split}_ids_sha256"] = canonical_sha256(identifiers)
        split_payloads[split] = (
            json.dumps(
                records,
                ensure_ascii=False,
                sort_keys=False,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
            + b"\n"
        )
        write_bytes_atomic(target / f"{split}.json", split_payloads[split])
    write_canonical_json(target / "split_manifest.json", split_manifest)

    provenance = json.loads((target / "provenance.json").read_text(encoding="utf-8"))
    id_inputs = provenance["data_id_inputs"]
    id_inputs["split_manifest_sha256"] = canonical_sha256(split_manifest)
    for split, payload in split_payloads.items():
        id_inputs[f"{split}_content_sha256"] = hashlib.sha256(payload).hexdigest()
    data_id = "data-" + canonical_sha256(id_inputs)
    provenance["data_build_id"] = data_id
    provenance["split_manifest_sha256"] = canonical_sha256(split_manifest)
    write_canonical_json(target / "provenance.json", provenance)
    report = json.loads((target / "audit_report.json").read_text(encoding="utf-8"))
    report["data_build_id"] = data_id
    write_canonical_json(target / "audit_report.json", report)
    write_canonical_json(target / "payload_manifest.json", build_payload_manifest(target))

    destination = target.parent / data_id
    if destination != target:
        if destination.exists():
            raise AssertionError(f"semantic data fixture ID collision: {destination}")
        target.rename(destination)
    write_locator_ref(
        data_ref,
        artifact_kind="data",
        artifact_id=data_id,
        target=destination,
        payload_manifest_sha256=validate_payload_manifest(destination),
    )
    return data_id


def make_formal_lexicon_artifact(root: Path, *, data_ref: Path) -> Path:
    """Build a real-semantic formal lexicon using deterministic fake I/O clients."""

    # Imported lazily to keep ordinary fixture discovery independent of the
    # lexicon integration-test module and its heavier mocks.
    from collections import Counter

    from build_lex.llm_lexicon_builder import CandidateStats
    from tests.test_stage1_train_only_lexicon import formal_build

    build_root = root
    candidates = []
    for term in ("火星人", "金星人"):
        candidates.append(
            CandidateStats(
                term=term,
                dataset="full",
                language="zh",
                total_count=1,
                source_counts=Counter({"content": 1}),
                track_counts=Counter({"contrastive_phrase": 1}),
                support_sample_ids=["1"],
                sample_contexts=[
                    {
                        "id": "1",
                        "source": "content",
                        "content": "train text",
                    }
                ],
            )
        )
    formal_build(build_root, data_ref=data_ref, candidates=candidates)
    return build_root / "lexicon_ref.json"


def make_embedding_model_fixture(root: Path) -> tuple[Path, str]:
    from data.build_context_manifest import embedding_model_file_tree_sha256

    model = root / "models" / "semantic-embedding"
    model.mkdir(parents=True)
    write_bytes_atomic(model / "weights.bin", b"semantic-embedding-fixture/v1\n")
    return model, embedding_model_file_tree_sha256(model)
