#!/usr/bin/env python3
"""Create a private, digest-identical regular-file model copy; never replace shared paths."""
from __future__ import annotations
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import stat
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT)]
from diagnostics.general_model_evidence_evaluation import file_sha, json_bytes, read_json, require, write_output
from scripts.review.freeze_evidence_lexicon_scope import WORK


def copy_inventory(snapshot, source_root, target):
    """Stream each expected file, verify its bytes, and leave sources untouched."""
    require(not target.exists(), 'model copy already exists; choose another version')
    target.mkdir(parents=True)
    receipts = []
    for item in snapshot['files']:
        relative = Path(item['path'])
        require(not relative.is_absolute() and '..' not in relative.parts, 'unsafe model path')
        source = source_root / relative
        resolved = source.resolve(strict=True)
        require(resolved.is_file() and not resolved.is_symlink(), 'model source target is not regular')
        status = resolved.stat(); target_file = target / relative
        target_file.parent.mkdir(parents=True, exist_ok=True)
        hasher = hashlib.sha256(); size = 0
        with resolved.open('rb') as reader, target_file.open('xb') as writer:
            for chunk in iter(lambda: reader.read(8 * 1024 * 1024), b''):
                writer.write(chunk); hasher.update(chunk); size += len(chunk)
            writer.flush(); os.fsync(writer.fileno())
        after = resolved.stat()
        require((status.st_dev, status.st_ino, status.st_size, status.st_mtime_ns, status.st_ctime_ns)
                == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns), 'source changed while copying')
        require(source.resolve(strict=True) == resolved, 'source link changed while copying')
        require(size == item['size'] and hasher.hexdigest() == item['sha256'], 'model copy digest differs: ' + str(relative))
        require(stat.S_ISREG(target_file.lstat().st_mode) and not target_file.is_symlink(), 'copy is not regular')
        target_file.chmod(0o444)
        receipts.append({'path': relative.as_posix(), 'sha256': hasher.hexdigest(), 'size': size,
                         'source_resolved_path': str(resolved), 'source_was_symlink': source.is_symlink(),
                         'target_is_regular_file': True, 'source_mutated': False})
        print(json.dumps({'event': 'verified-model-copy-file', 'path': relative.as_posix(), 'bytes': size}), flush=True)
    return receipts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--target', type=Path, default=WORK / 'model-copy-01')
    parser.add_argument('--package', type=Path, default=WORK / 'model-load-package-01')
    args = parser.parse_args(); target, package = args.target.resolve(), args.package.resolve()
    require(target.parent == WORK and package.parent == WORK, 'use private experiment tree')
    parent_plan = read_json(WORK / 'frozen-01/plan.json')
    old_package = Path(parent_plan['runtime_parent_plan']['package_path'])
    models = read_json(old_package / 'models.json')
    require(models and models[0]['key'] == 'qwen3-8b', 'unexpected primary model package')
    source_root = ROOT / models[0]['inventory']['logical_repo_path']
    receipts = copy_inventory(models[0]['inventory'], source_root, target)
    changed = deepcopy(models); relative = target.relative_to(ROOT).as_posix()
    changed[0]['path'] = relative
    for name in ('inventory', 'tokenizer_inventory'): changed[0][name]['logical_repo_path'] = relative
    receipt = {'schema_version': 'evidence-model-regular-copy/v1', 'status': 'complete',
        'created_at': datetime.now(timezone.utc).isoformat(), 'old_package_path': str(old_package),
        'original_models_sha256': file_sha(old_package / 'models.json'), 'copy_path': relative,
        'file_tree_sha256': models[0]['inventory']['file_tree_sha256'],
        'tokenizer_tree_sha256': models[0]['tokenizer_inventory']['file_tree_sha256'],
        'files': receipts, 'all_content_hashes_match_original': True, 'shared_sources_mutated': False,
        'copy_method': 'independent regular files, streaming hash and fsync; no hardlinks or symlinks',
        'runtime_lease_still_required': 'unchanged registry performs fresh full hashes and regular-file leases before and after load',
        'scientific_inputs_scores_and_thresholds_changed': False}
    files = {'config.resolved.json': (old_package / 'config.resolved.json').read_bytes(),
             'models.json': json_bytes(changed), 'model-copy-receipt.json': json_bytes(receipt)}
    files['manifest.json'] = json_bytes({'schema_version': 'evidence-load-only-model-package/v1',
        'purpose': 'path-only adapter for existing verified LocalRunner, not a new scientific experiment package',
        'artifacts': {n: hashlib.sha256(b).hexdigest() for n, b in files.items()}})
    write_output(package, files)
    print(json.dumps({'status': 'complete', 'model_copy': relative, 'package': str(package),
                      'bytes': sum(r['size'] for r in receipts), 'files': len(receipts)}), flush=True)


if __name__ == '__main__': main()
