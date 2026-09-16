#!/usr/bin/env python3
"""Validate, evaluate or reproduce frozen dual references on CPU only."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT)]
from diagnostics.general_model_evidence_evaluation import (
    EXPERIMENT, build, file_sha, json_bytes, load_inputs, local_path, read_json,
    require, sha, verify_output, write_output,
)

DEFAULT_CONFIG = ROOT / 'config/stage1/general_model_evidence_dual_reference_v1.json'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('validate', 'evaluate', 'check'))
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--allow-missing-scores', action='store_true',
                        help='Only if the raw file is absent: emit discrete results with null revised margins. Never accepts a corrupt/incomplete raw file.')
    args = parser.parse_args()
    root = args.root.resolve()
    require(root == ROOT, 'run this checkout\'s script in its own repository root; do not redirect source code')
    config = args.config.resolve()
    require(config.is_relative_to(root), 'config outside repository')
    if args.command == 'validate':
        data = load_inputs(root, config, args.allow_missing_scores)
        data[0].unchanged()
        print('Input/reference identities and discovery frame validated; no model or GPU loaded.')
        return
    require(args.output is not None, '--output is required')
    target = args.output.resolve()
    require(target.is_relative_to(root / EXPERIMENT / 'reviews') and target != root / EXPERIMENT / 'reviews',
            'output must be under the private experiment reviews tree')
    c = read_json(config)
    pointer = read_json(local_path(root, c['reference_pointer']['path']))
    freeze = local_path(root, str(EXPERIMENT / pointer['freeze_path']))
    require(not target.is_relative_to(freeze) and not freeze.is_relative_to(target), 'output overlaps frozen reference directory')
    if args.command == 'check':
        old = verify_output(target)
        for name, h in old['source_files'].items():
            require(file_sha(local_path(root, name)) == h, 'source changed since evaluation: ' + name)
        for name, h in old['code_sha256'].items():
            require(file_sha(root / name) == h, 'check requires the recorded implementation: ' + name)
        require(old['config_sha256'] == file_sha(config), 'check config mismatch')
    else:
        require(not target.exists() and not target.is_symlink(), 'refusing to overwrite evaluation')
    files, sources, audit = build(root, config, args.allow_missing_scores)
    if args.command == 'check':
        for name, raw in files.items():
            require((target / name).read_bytes() == raw, 'evaluation reconstruction mismatch: ' + name)
        print('Evaluation and all tables reproduce byte for byte; no model or GPU loaded.')
        return
    code_paths = [Path(__file__).resolve(), root / 'src/diagnostics/general_model_evidence_evaluation.py',
                  root / 'src/diagnostics/general_model_numeric_analysis.py']
    code = {str(path.relative_to(root)): {'sha256': file_sha(path), 'text': path.read_text()} for path in code_paths}
    files['execution_source.json'] = json_bytes(code)
    manifest = {
        'schema_version': 'evidence-dual-reference-run/v1', 'status': 'complete',
        'created_at': datetime.now(timezone.utc).isoformat(),
        'execution_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
        'tracked_worktree_modified': bool(subprocess.check_output(['git','status','--porcelain=v1','--untracked-files=no'],cwd=root,text=True).strip()),
        'python': platform.python_version(), 'config_sha256': file_sha(config),
        'source_files': sources, 'code_sha256': {name: record['sha256'] for name,record in code.items()},
        'reference_session_revision': audit['reference_session_revision'],
        'reference_manifest_sha256': audit['reference_manifest_sha256'],
        'raw_scores_available': audit['continuous_status'] == 'available',
        'allow_model_forward': False, 'allow_gpu': False, 'dual_reference_evaluation_completed': True,
        'artifacts': {name: sha(raw) for name,raw in files.items()},
    }
    files['manifest.json'] = json_bytes(manifest)
    write_output(target, files)
    print(str(target))


if __name__ == '__main__':
    try:
        main()
    except (ValueError, OSError, KeyError, TypeError) as exc:
        print('Evaluation failed: ' + str(exc), file=sys.stderr)
        raise SystemExit(1)
