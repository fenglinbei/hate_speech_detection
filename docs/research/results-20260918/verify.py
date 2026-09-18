#!/usr/bin/env python3
"""Read-only verification of the result archive's byte inventory."""
from pathlib import Path
import hashlib
import json
import sys


def main():
    directory = Path(__file__).resolve().parent
    root = directory.parents[2]
    manifest = json.loads((directory / 'manifest.json').read_text(encoding='utf-8'))
    errors = []
    paths = set()
    total_bytes = 0
    for record in manifest['files']:
        relative = Path(record['path'])
        if relative.is_absolute() or '..' in relative.parts or record['path'] in paths:
            errors.append({'path': record['path'], 'error': 'invalid or duplicate path'})
            continue
        paths.add(record['path'])
        path = root / relative
        if not path.is_file() or path.is_symlink():
            errors.append({'path': record['path'], 'error': 'missing or non-regular file'})
            continue
        digest = hashlib.sha256()
        size = 0
        with path.open('rb') as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b''):
                digest.update(chunk)
                size += len(chunk)
        total_bytes += size
        if size != record['bytes'] or digest.hexdigest() != record['sha256']:
            errors.append({'path': record['path'], 'error': 'byte count or SHA256 differs'})
    print(json.dumps({'status': 'failed' if errors else 'passed',
                      'files': len(paths), 'bytes': total_bytes, 'errors': errors},
                     ensure_ascii=False, indent=2))
    return bool(errors)


if __name__ == '__main__':
    sys.exit(main())
