#!/usr/bin/env python3
"""Check publication artifacts only; never import or execute research runners."""
import argparse
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
TEXT = {'.json', '.jsonl', '.txt', '.py', '.md', '.tsv', '.svg', '.html',
        '.cjs', '.conf', '.js', '.css', ''}
SECRET_PATTERNS = {
    'private_key': re.compile(r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----'),
    'github_token': re.compile(r'\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,})'),
    'aws_access_key': re.compile(r'\b(?:AKIA|ASIA)[A-Z0-9]{16}\b'),
    'api_secret': re.compile(r'\bsk-(?:proj-|ant-api\d+-)?[A-Za-z0-9_-]{32,}'),
    'basic_auth_literal': re.compile(r'(?i)\b(?:Authorization\s*[:=]\s*[\"\x27]?|[\"\x27]Authorization[\"\x27]\s*:\s*[\"\x27])Basic\s+[A-Za-z0-9+/]{16,}={0,2}'),
    'password_literal': re.compile(r'(?i)[\"\x27](?:password|passwd|api_key|access_token|refresh_token)[\"\x27]\s*:\s*[\"\x27]([^\"\x27\n]{12,})[\"\x27]'),
}


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def objects(value):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            if isinstance(child, (dict, list)):
                yield from objects(child)
    elif isinstance(value, list):
        for child in value:
            if isinstance(child, (dict, list)):
                yield from objects(child)


def local_reference(value, source, known):
    path = Path(value)
    if path.is_absolute():
        try:
            return str(path.relative_to(ROOT))
        except ValueError:
            return None
    if source == 'deploy/case_attention/digitalocean-sgp/deployments/20260918-01/closeout.json':
        return str(Path('deploy/case_attention/digitalocean-sgp') / path)
    if value in known or path.parts[:1] in [(x,) for x in
            ('src', 'scripts', 'reviews', 'docs', 'deploy', 'tools', 'exps', 'data', 'models', '.conda')]:
        return value
    if str(Path('reviews') / path) in known:
        return str(Path('reviews') / path)
    if value.startswith('sentence-completion-v1/'):
        # Material provenance is relative to the prior human review workbench.
        return str(Path('exps/causal_context/general_model_evidence_applicability_v1/ai_reviews/v2-reannotation-20260911') / path)
    return str((ROOT / source).parent.joinpath(path).resolve().relative_to(ROOT))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--staged', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((HERE / 'manifest.json').read_text())
    records = {r['path']: r for r in manifest['files']}
    tracked = set(subprocess.check_output(['git', 'ls-files', '-z'], cwd=ROOT).decode().split('\0'))
    known = set(records) | tracked
    errors, mismatches, secret_findings = [], [], []
    omitted_other = set()
    counts, omitted = Counter(), Counter()
    pins = set()
    cache = {}

    def check_pin(obj, source):
        if not (isinstance(obj.get('path'), str) and
                isinstance(obj.get('sha256'), str) and
                re.fullmatch(r'[a-f0-9]{64}', obj['sha256'])):
            return
        try:
            ref = local_reference(obj['path'], source, known)
        except ValueError:
            ref = None
        if ref in known and (ROOT / ref).is_file():
            if ref not in cache:
                raw = (ROOT / ref).read_bytes()
                cache[ref] = (len(raw), digest(raw))
            actual_size, actual_sha = cache[ref]
            counts['included_reference_occurrences'] += 1
            pins.add((ref, obj['sha256']))
            size = obj.get('bytes')
            if actual_sha != obj['sha256'] or (isinstance(size, int) and size != actual_size):
                mismatches.append({'source': source, 'path': ref,
                                   'expected_sha256': obj['sha256'], 'actual_sha256': actual_sha,
                                   'expected_bytes': size, 'actual_bytes': actual_size})
        elif ref is None:
            omitted['external_machine_dependency'] += 1
        elif ref.startswith(('models/', '.conda/', 'checkpoints/')):
            omitted['model_or_environment'] += 1
        elif '.view.json' in ref or '-aggregates.json' in ref:
            omitted['dense_attention_display'] += 1
        elif '/synthetic-' in ref:
            omitted['synthetic_fixture'] += 1
        elif '/hosting-' in ref and ('/prepared/' in ref or '/runtime/' in ref or '/build-attempt-' in ref):
            omitted['hosting_bundle_or_private_runtime'] += 1
        elif re.search(r'/run-\d+/(records|arrays|format|format-arrays|invocations)/', ref) or Path(ref).suffix in {'.npy', '.npz'}:
            omitted['raw_run_record_or_array'] += 1
        elif Path(ref).suffix in {'.log', '.lock', '.pid'} or Path(ref).name in {'STOP', 'CANCEL'}:
            omitted['runtime_log_or_control'] += 1
        else:
            omitted['other_local_dependency_outside_archive'] += 1
            omitted_other.add(ref)

    archive_files = sorted(p for p in HERE.iterdir() if p.is_file()
                           and p.name != 'verification.json')
    paths = sorted(set(records) | {str(p.relative_to(ROOT)) for p in archive_files})
    for rel in paths:
        path = ROOT / rel
        if path.is_symlink() or not path.is_file():
            errors.append({'path': rel, 'error': 'missing or non-regular file'})
            continue
        raw = path.read_bytes()
        if rel in records:
            record = records[rel]
            if len(raw) != record['bytes'] or digest(raw) != record['sha256']:
                errors.append({'path': rel, 'error': 'archive bytes differ'})
        if path.suffix not in TEXT:
            counts['binary_files'] += 1
            continue
        try:
            text = raw.decode('utf-8')
            counts['utf8_files'] += 1
            for kind, pattern in SECRET_PATTERNS.items():
                for match in pattern.finditer(text):
                    # Do not print or save candidate values.
                    secret_findings.append({'path': rel, 'line': text.count('\n', 0, match.start()) + 1,
                                            'kind': kind})
            if path.suffix == '.py':
                ast.parse(text, filename=rel)
                counts['python_sources'] += 1
            documents = []
            if path.suffix == '.json':
                documents = [json.loads(text)]
                counts['json_files'] += 1
            elif path.suffix == '.jsonl':
                documents = [json.loads(line) for line in text.splitlines() if line.strip()]
                counts['jsonl_files'] += 1
                counts['jsonl_records'] += len(documents)
            for document in documents:
                for obj in objects(document):
                    check_pin(obj, rel)
        except (ValueError, SyntaxError, UnicodeDecodeError) as exc:
            errors.append({'path': rel, 'error': type(exc).__name__})

    # This new index supplies repository-relative links without editing sealed prose.
    for target in re.findall(r'\]\(([^\s)]+)\)', (HERE / 'README.md').read_text()):
        if '://' in target or target.startswith('#'):
            continue
        dest = (HERE / target.split('#', 1)[0]).resolve()
        rel = str(dest.relative_to(ROOT))
        if dest.name == 'verification.json' and dest.parent == HERE:
            continue  # Written by this command after validation.
        if not dest.exists() or not (rel in paths or rel in known or
                any(x.startswith(rel.rstrip('/') + '/') for x in paths + list(known))):
            errors.append({'path': target, 'error': 'new archive navigation target not delivered'})
        counts['navigation_links'] += 1

    for name in manifest['completed_experiments']:
        selector = json.loads((ROOT / 'docs/research/experiment-plans' / name / 'results-current.json').read_text())
        if selector.get('status') != 'complete':
            errors.append({'path': name, 'error': 'scientific result not complete'})
        counts['complete_selectors'] += 1

    # Historical receipts are preserved as history, never treated as final pins.
    # Every exception binds both old and current bytes and cites existing evidence.
    history_path = HERE / 'historical-references.json'
    disclosed = []
    if history_path.exists():
        history = json.loads(history_path.read_text())
        for entry in history['references']:
            expected = {k: entry[k] for k in ('source', 'path', 'expected_sha256',
                        'actual_sha256', 'expected_bytes', 'actual_bytes')}
            if expected not in mismatches:
                errors.append({'path': entry['path'], 'error': 'historical disclosure does not match observed pin'})
            elif any(p not in known for p in entry['evidence']):
                errors.append({'path': entry['path'], 'error': 'historical evidence not delivered'})
            else:
                mismatches.remove(expected)
                disclosed.append(entry)
        for entry in history['credential_scan_false_positives']:
            finding = {k: entry[k] for k in ('path', 'line', 'kind')}
            raw = (ROOT / entry['path']).read_bytes()
            line = raw.decode().splitlines()[entry['line'] - 1]
            match = SECRET_PATTERNS[entry['kind']].search(line)
            if (finding not in secret_findings or digest(raw) != entry['file_sha256'] or
                    match is None or digest(match.group(1).encode()) != entry['literal_sha256']):
                errors.append({'path': entry['path'], 'error': 'negative-auth-test disclosure does not match'})
            else:
                secret_findings.remove(finding)

    staged = {'status': 'not_requested'}
    if args.staged:
        staged_records = subprocess.check_output(['git', 'ls-files', '--stage', '-z'], cwd=ROOT).split(b'\0')
        index = {}
        for line in staged_records:
            if line:
                metadata, name = line.split(b'\t', 1)
                _, oid, stage = metadata.split()
                if stage != b'0':
                    errors.append({'error': 'unmerged index entry'})
                index[name.decode()] = oid.decode()
        expected_paths = set(records) | {str(p.relative_to(ROOT)) for p in HERE.iterdir() if p.is_file()}
        changed = set(subprocess.check_output(['git', 'diff', '--cached', '--name-only', '-z'], cwd=ROOT).decode().rstrip('\0').split('\0'))
        if changed != expected_paths:
            errors.append({'error': 'staged path set differs', 'missing': sorted(expected_paths - changed),
                           'extra': sorted(changed - expected_paths)})
        git = subprocess.Popen(['git', 'cat-file', '--batch'], cwd=ROOT,
                               stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        try:
            for rel, record in records.items():
                if rel not in index:
                    errors.append({'path': rel, 'error': 'not staged'})
                    continue
                git.stdin.write((index[rel] + '\n').encode())
                git.stdin.flush()
                header = git.stdout.readline().split()
                n = int(header[2])
                raw = git.stdout.read(n)
                assert git.stdout.read(1) == b'\n'
                if len(raw) != record['bytes'] or digest(raw) != record['sha256']:
                    errors.append({'path': rel, 'error': 'index blob differs from inventory'})
        finally:
            git.stdin.close()
            git.stdout.close()
            git.wait()
        whitespace = subprocess.run(['git', 'diff', '--cached', '--check'], cwd=ROOT, capture_output=True, text=True)
        if whitespace.returncode:
            errors.append({'error': 'git diff --cached --check failed', 'details': whitespace.stdout})
        staged = {'status': 'passed' if not errors else 'failed', 'inventory_files': len(records),
                  'method': 'Git index blob byte counts and SHA256 equal every inventory record.',
                  'whitespace_check_exit_code': whitespace.returncode,
                  'archive_documents_bound_by_commit': True}

    result = {'schema_version': 'research-result-archive-verification/v1',
              'status': 'failed' if errors or mismatches or secret_findings else 'passed',
              'scope': 'Publication bytes, parsing, frozen references, new navigation and optional index checks; no research computation.',
              'base_commit': manifest['base_commit'], 'archive_inventory_files': len(records),
              'archive_inventory_bytes': sum(r['bytes'] for r in records.values()),
              'worktree_checks': dict(counts), 'included_reference_unique_pins': len(pins),
              'unarchived_reference_occurrences': dict(omitted),
              'other_unarchived_local_paths': sorted(omitted_other),
              'historical_references_disclosed': disclosed,
              'credential_scan_false_positives': history['credential_scan_false_positives'] if history_path.exists() else [],
              'reference_mismatches': mismatches, 'credential_pattern_findings': secret_findings,
              'errors': errors, 'new_GPU_forwards': 0, 'scientific_or_human_fields_modified': False,
              'scientific_tests': 'Not rerun: frozen implementation and prior numerical/test receipts retained.',
              'staged_bytes_verification': staged}
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'status': result['status'], 'files': len(records), 'checks': dict(counts),
                      'errors': len(errors), 'reference_mismatches': len(mismatches),
                      'credential_findings': len(secret_findings), 'output': str(args.output)}, ensure_ascii=False))
    return result['status'] != 'passed'


if __name__ == '__main__':
    sys.exit(main())
