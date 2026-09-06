#!/usr/bin/env python3
"""Fail-closed self-check for the isolated WP3 G3 runtime."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import stat
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from build_lex.terminology_candidate_generators_v2 import (  # noqa: E402
    build_pypinyin_romanizer,
    load_g3_profile,
)


EXPECTED = {
    "attrs": "26.1.0",
    "certifi": "2026.7.22",
    "charset-normalizer": "3.5.1",
    "idna": "3.19",
    "jsonschema": "4.26.0",
    "jsonschema-specifications": "2025.9.1",
    "pypinyin": "0.55.0",
    "pypdf": "6.0.0",
    "referencing": "0.37.0",
    "requests": "2.33.1",
    "rpds-py": "2026.6.3",
    "typing-extensions": "4.16.0",
    "urllib3": "2.7.0",
}
LOCK_PATH = Path("environment/wp3-g3-requirements.lock")
LOCK_SHA256 = "722aabfaa7abec9067c04e31e4620578d133c7cddc1c16c3d4390a282a598b2f"
SOURCE_INTAKE_LOCK_PATH = Path(
    "environment/wp3-g3-source-intake-requirements.lock"
)
SOURCE_INTAKE_LOCK_SHA256 = (
    "473914b646645ed0d6acf8215428df851e1930ae4672b2932dce642373f7a12c"
)


def check_environment() -> dict[str, object]:
    if (
        sys.implementation.name != "cpython"
        or sys.version_info[:2] != (3, 12)
        or sys.platform != "linux"
        or platform.machine() != "x86_64"
    ):
        raise RuntimeError("WP3 G3 Python ABI differs")
    lock = REPOSITORY_ROOT / LOCK_PATH
    metadata = lock.lstat()
    if lock.is_symlink() or not stat.S_ISREG(metadata.st_mode):
        raise RuntimeError("WP3 G3 requirements lock is not a regular file")
    if hashlib.sha256(lock.read_bytes()).hexdigest() != LOCK_SHA256:
        raise RuntimeError("WP3 G3 requirements lock hash differs")
    source_intake_lock = REPOSITORY_ROOT / SOURCE_INTAKE_LOCK_PATH
    source_intake_metadata = source_intake_lock.lstat()
    if source_intake_lock.is_symlink() or not stat.S_ISREG(
        source_intake_metadata.st_mode
    ):
        raise RuntimeError("WP3 G3 source-intake lock is not a regular file")
    if (
        hashlib.sha256(source_intake_lock.read_bytes()).hexdigest()
        != SOURCE_INTAKE_LOCK_SHA256
    ):
        raise RuntimeError("WP3 G3 source-intake requirements lock hash differs")
    installed = {
        package: importlib.metadata.version(package) for package in EXPECTED
    }
    if installed != EXPECTED:
        raise RuntimeError("WP3 G3 installed package versions differ")
    profile = load_g3_profile(
        "config/stage1/wp3_g3_profile_full_v2.json",
        workspace_root=REPOSITORY_ROOT,
    )
    romanizer = build_pypinyin_romanizer(
        profile, workspace_root=REPOSITORY_ROOT
    )
    vectors = {
        text: romanizer(text)
        for text in ("音乐", "银行", "重庆", "模样", "模型", "绿女略")
    }
    return {
        "scope": "wp3-g3-only",
        "python": ".".join(str(value) for value in sys.version_info[:3]),
        "requirements_lock_sha256": LOCK_SHA256,
        "source_intake_requirements_lock_sha256": SOURCE_INTAKE_LOCK_SHA256,
        "packages": installed,
        "romanizer_backend_id": profile["romanizer"]["backend_id"],
        "romanizer_vectors": vectors,
        "status": "passed",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("check",))
    parser.parse_args(argv)
    try:
        result = check_environment()
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
