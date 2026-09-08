#!/usr/bin/env python3
r"""Build a minimal paired-review release without sessions or training inputs.

Example:
    python3 -B deploy/general_model_paired_review/build_release.py \
        --output-dir /tmp/hsd-review-release-20260908

The output directory must be empty and inside /tmp. Its release/ directory and
release.tar.gz contain identical files. Original frozen manifest bytes are kept;
release_manifest.json inventories only the files actually needed by the service.
Archive ownership, permissions, ordering and timestamps are fixed for repeatable
builds. Verification creates and removes an isolated automated-test session.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
from pathlib import Path
import re
import subprocess
import sys
import tarfile
import tempfile
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = REPOSITORY_ROOT / (
    "exps/causal_context/general_model_ld_nolabel_paired_cases_v1/results/paired-cases-02"
)
CODE_FILES = (
    "scripts/stage1/general_model_paired_review.py",
    "tools/general_model_paired_review_ui/server.py",
    "tools/general_model_paired_review_ui/store.py",
    "tools/annotated_lexicon_operation_review_ui/server.py",
    "src/build_lex/annotated_lexicon_repair.py",
    "src/build_lex/annotated_lexicon_operation_review.py",
    "src/build_lex/repair_regex_validation.py",
    "src/rag/controlled_lexicon_matcher.py",
)
STATIC_FILES = (
    "tools/general_model_paired_review_ui/index.html",
    "tools/general_model_paired_review_ui/app.js",
    "tools/general_model_paired_review_ui/core.js",
    "tools/general_model_paired_review_ui/styles.css",
    "tools/wp3_candidate_review_ui/styles.css",
    "tools/wp3_candidate_review_ui/core.js",
)
_RESOURCE_NAME = re.compile(r"([A-Za-z0-9_-]+)-1-resources\.md\Z")
_SHA256 = re.compile(r"[a-f0-9]{64}\Z")


def sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")


def read_file(root: Path, relative: str) -> bytes:
    path = root / relative
    resolved = path.resolve(strict=True)
    if not resolved.is_relative_to(root) or path.is_symlink() or not resolved.is_file():
        raise ValueError(f"expected a regular file inside its source directory: {relative}")
    return resolved.read_bytes()


def collect_payload(data_dir: Path) -> tuple[dict[str, bytes], dict[str, Any]]:
    """Snapshot each included file once, validating those exact frozen bytes."""
    payload = {
        name: read_file(REPOSITORY_ROOT, name)
        for name in (*CODE_FILES, *STATIC_FILES)
    }
    manifest_bytes = read_file(data_dir, "manifest.json")
    source_manifest = json.loads(manifest_bytes)
    if not isinstance(source_manifest, dict) or source_manifest.get("status") != "complete":
        raise ValueError("source manifest must describe a complete paired analysis")
    artifacts = source_manifest.get("artifacts")
    identity = source_manifest.get("identity")
    if not isinstance(artifacts, dict) or not isinstance(identity, str) or not identity:
        raise ValueError("source manifest lacks its artifact hashes or identity")
    payload["data/manifest.json"] = manifest_bytes

    def include(relative: str) -> bytes:
        content = read_file(data_dir, relative)
        expected = artifacts.get(relative)
        if not isinstance(expected, str) or not _SHA256.fullmatch(expected) or sha256(content) != expected:
            raise ValueError(f"frozen artifact hash is missing or differs: {relative}")
        payload["data/" + relative] = content
        return content

    index = json.loads(include("cases/cards_index.json"))
    initial = json.loads(include("cases/initial_review_batch.json"))
    if not isinstance(index, list) or not index or not isinstance(initial, list):
        raise ValueError("case index must be nonempty and initial batch must be a list")
    case_ids: set[str] = set()
    card_paths: set[str] = set()
    for entry in index:
        if not isinstance(entry, dict):
            raise ValueError("case index contains a non-object entry")
        key = entry.get("query_id")
        resource = entry.get("resources_card")
        if not isinstance(key, str) or not key or key in case_ids or not isinstance(resource, str):
            raise ValueError("case index contains missing or duplicate query IDs")
        match = _RESOURCE_NAME.fullmatch(Path(resource).name)
        if match is None:
            raise ValueError("case index contains an invalid resource card name")
        relative = "cases/card_data/" + match.group(1) + ".json"
        if relative in card_paths:
            raise ValueError("case index refers to the same card more than once")
        card = json.loads(include(relative))
        selection = card.get("selection", {}) if isinstance(card, dict) else {}
        if (
            selection.get("query_id") != key
            or selection.get("split") != "discovery"
            or selection.get("focus_task") not in {"hate", "group"}
        ):
            raise ValueError(f"card identity differs or contains a non-discovery case: {relative}")
        case_ids.add(key)
        card_paths.add(relative)

    initial_ids = [row.get("query_id") if isinstance(row, dict) else None for row in initial]
    if (
        any(not isinstance(key, str) for key in initial_ids)
        or len(set(initial_ids)) != len(initial_ids)
        or not set(initial_ids) <= case_ids
    ):
        raise ValueError("initial batch contains missing, duplicate or unauthorized case IDs")
    if "cases/ai_review.csv" in artifacts:
        rows = list(csv.DictReader(io.StringIO(include("cases/ai_review.csv").decode("utf-8"))))
        ai_ids = [row.get("query_id") for row in rows]
        if len(set(ai_ids)) != len(ai_ids) or not set(ai_ids) <= case_ids:
            raise ValueError("AI review contains duplicate or unauthorized case IDs")

    metadata = {
        "schema_version": "general-model-paired-review-release/v1",
        "source_identity": identity,
        "source_manifest_sha256": sha256(manifest_bytes),
        "runtime": {"python": ">=3.10", "platform": "linux", "third_party_packages": []},
        "case_count": len(case_ids),
        "initial_case_count": len(initial_ids),
        "payload_file_count": len(payload),
        "payload_bytes": sum(map(len, payload.values())),
        "files": {
            name: {"sha256": sha256(content), "bytes": len(content)}
            for name, content in sorted(payload.items())
        },
    }
    return payload, metadata


_VERIFY_PROGRAM = r'''
import hashlib
import json
from pathlib import Path
import sys

release = Path(sys.argv[1]).resolve()
session = Path(sys.argv[2]).resolve()
sys.path[:0] = [str(release), str(release / "src")]
from tools.general_model_paired_review_ui.server import PairedWebService

manifest = json.loads((release / "release_manifest.json").read_bytes())
actual = {p.relative_to(release).as_posix() for p in release.rglob("*") if p.is_file()}
if actual != set(manifest["files"]) | {"release_manifest.json"}:
    raise RuntimeError("release inventory differs")
for name, expected in manifest["files"].items():
    content = (release / name).read_bytes()
    if len(content) != expected["bytes"] or hashlib.sha256(content).hexdigest() != expected["sha256"]:
        raise RuntimeError("release file hash differs: " + name)
service = PairedWebService(
    data_dir=release / "data", session_path=session,
    reviewer_id="automated-release-verification",
)
service.configure_network(8772, public_origin="https://hsd.fenglin.pro")
bootstrap = service.bootstrap()
if bootstrap["status"]["item_count"] != manifest["case_count"]:
    raise RuntimeError("case count differs")
if bootstrap["status"]["initial_count"] != manifest["initial_case_count"]:
    raise RuntimeError("initial count differs")
for row in bootstrap["items"]:
    item = service.store.item_state(row["item_id"])
    if item["review"]["status"] != "unreviewed" or item["trajectory"] is not None or item["ai_review"] is not None:
        raise RuntimeError("isolated material visibility differs")
reopened = PairedWebService(
    data_dir=release / "data", session_path=session,
    reviewer_id="automated-release-verification",
)
if reopened.store.bootstrap() != service.store.bootstrap():
    raise RuntimeError("isolated session did not resume")
for name, module in list(sys.modules.items()):
    if name.startswith(("tools.", "build_lex.", "rag.")) and getattr(module, "__file__", None):
        if not Path(module.__file__).resolve().is_relative_to(release):
            raise RuntimeError("module was imported outside the release: " + name)
print(json.dumps({"status": "ok", "python": sys.version.split()[0], "case_count": manifest["case_count"]}))
'''


def verify_release(release: Path) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="hsd-review-release-verify-", dir="/tmp") as temporary:
        result = subprocess.run(
            [sys.executable, "-I", "-B", "-S", "-c", _VERIFY_PROGRAM,
             str(release), str(Path(temporary) / "session.json")],
            cwd=temporary, capture_output=True, text=True, timeout=60, check=False,
        )
    if result.returncode:
        raise ValueError("isolated release verification failed:\n" + result.stderr.strip())
    return json.loads(result.stdout)


def write_archive(path: Path, payload: dict[str, bytes]) -> None:
    directories = {"release"}
    for name in payload:
        directories.update("release/" + parent.as_posix() for parent in Path(name).parents if parent != Path("."))
    with path.open("xb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0, compresslevel=9) as compressed:
            with tarfile.open(fileobj=compressed, mode="w|", format=tarfile.PAX_FORMAT) as archive:
                for name in sorted(directories):
                    info = tarfile.TarInfo(name)
                    info.type, info.mode, info.mtime = tarfile.DIRTYPE, 0o755, 0
                    archive.addfile(info)
                for name, content in sorted(payload.items()):
                    info = tarfile.TarInfo("release/" + name)
                    info.size, info.mode, info.mtime = len(content), 0o644, 0
                    archive.addfile(info, io.BytesIO(content))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", required=True, type=Path, help="new or empty output directory inside /tmp")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA, help="frozen paired-case source directory")
    args = parser.parse_args()
    try:
        output = args.output_dir.resolve()
        temporary_root = Path("/tmp").resolve()
        if output == temporary_root or not output.is_relative_to(temporary_root):
            raise ValueError("--output-dir must be a directory inside /tmp")
        if output.exists() and (not output.is_dir() or any(output.iterdir())):
            raise ValueError("--output-dir already exists and is not an empty directory")
        payload, metadata = collect_payload(args.data_dir.resolve(strict=True))
        payload["release_manifest.json"] = json_bytes(metadata)
        output.mkdir(parents=True, exist_ok=True)
        release = output / "release"
        release.mkdir()
        for name, content in sorted(payload.items()):
            path = release / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
        verification = verify_release(release)
        archive = output / "release.tar.gz"
        write_archive(archive, payload)
        print(json.dumps({
            "release_dir": str(release), "archive": str(archive),
            "file_count": len(payload), "payload_file_count": metadata["payload_file_count"],
            "case_count": metadata["case_count"], "initial_case_count": metadata["initial_case_count"],
            "source_manifest_sha256": metadata["source_manifest_sha256"],
            "release_manifest_sha256": sha256(payload["release_manifest.json"]),
            "archive_sha256": sha256(archive.read_bytes()),
            "archive_bytes": archive.stat().st_size, "verification": verification,
        }, ensure_ascii=False, indent=2))
    except (OSError, ValueError, TypeError, KeyError, subprocess.TimeoutExpired) as exc:
        parser.exit(1, f"release build failed: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
