#!/usr/bin/env python3
"""Path-only execution amendment for digest-identical private regular model files."""
from copy import deepcopy
from datetime import datetime, timezone
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT)]
from diagnostics.general_model_evidence_evaluation import canonical, file_sha, json_bytes, read_json, require, sha, write_output
from scripts.review.freeze_evidence_lexicon_scope import WORK, load_frozen

PARENT = WORK / 'frozen-01'
PARENT_HASH = 'd6b1145d15624d92e9e767821e8dffa94c3fa6003a2fe6d3a4d932ac028cc9ba'
PACKAGE = WORK / 'model-load-package-01'
CODE = tuple('scripts/review/' + name + '.py' for name in (
    'prepare_evidence_model_copy', 'freeze_evidence_lexicon_execution_v2', 'test_evidence_lexicon_execution_v2'))
CHANGED = {'plan.json', 'audit.json', 'execution_source.json'}


def verify_adapter(parent):
    old_package = Path(parent['runtime_parent_plan']['package_path'])
    old = read_json(old_package / 'models.json'); models = read_json(PACKAGE / 'models.json')
    receipt = read_json(PACKAGE / 'model-copy-receipt.json')
    require(receipt['status'] == 'complete' and receipt['all_content_hashes_match_original']
            and not receipt['shared_sources_mutated'] and not receipt['scientific_inputs_scores_and_thresholds_changed'], 'invalid model copy receipt')
    require(receipt['original_models_sha256'] == file_sha(old_package / 'models.json'), 'wrong original model contract')
    expected = deepcopy(old)
    expected[0]['path'] = receipt['copy_path']
    for name in ('inventory', 'tokenizer_inventory'): expected[0][name]['logical_repo_path'] = receipt['copy_path']
    require(models == expected, 'model adapter changed more than source paths')
    require((PACKAGE / 'config.resolved.json').read_bytes() == (old_package / 'config.resolved.json').read_bytes(), 'runtime config changed')
    require([{k: row[k] for k in ('path', 'sha256', 'size')} for row in receipt['files']] == old[0]['inventory']['files'], 'copy inventory differs from historical bytes')
    copy_path = ROOT / receipt['copy_path']
    require(copy_path.parent == WORK and not copy_path.is_symlink(), 'invalid private copy location')
    require({p.relative_to(copy_path).as_posix() for p in copy_path.rglob('*') if p.is_file()} == {r['path'] for r in receipt['files']}, 'private copy inventory differs')
    for row in receipt['files']:
        p = copy_path / row['path']
        require(p.is_file() and not p.is_symlink() and p.stat().st_size == row['size'], 'private model copy is not a regular file')
    m = read_json(PACKAGE / 'manifest.json')
    for name, h in m['artifacts'].items(): require(file_sha(PACKAGE / name) == h, 'model adapter artifact changed')
    return receipt


def build():
    require(file_sha(PARENT / 'manifest.json') == PARENT_HASH, 'original lexicon preparation changed')
    parent, contexts, history = load_frozen(PARENT)
    receipt = verify_adapter(parent)
    failed = read_json(WORK / 'run-01/run_manifest.json')
    require(failed['status'] == 'failed' and not failed['model_forward_executed'] and not failed['checks'], 'unexpected first run state')
    manifest = read_json(PARENT / 'manifest.json')
    sources = dict(manifest['source_files']); code = dict(parent['code_sha256'])
    for name, h in {**manifest['artifacts'], 'manifest.json': PARENT_HASH}.items(): sources[str((PARENT / name).relative_to(ROOT))] = h
    for p in [*(PACKAGE / name for name in ('config.resolved.json', 'models.json', 'model-copy-receipt.json', 'manifest.json')),
              WORK / 'run-01/run_manifest.json']:
        sources[str(p.relative_to(ROOT))] = file_sha(p)
    for name in CODE: sources[name] = code[name] = file_sha(ROOT / name)
    amendment = {'schema_version': 'evidence-lexicon-model-path-amendment/v2', 'author': 'assistant',
        'reason': 'Existing shared model shards became symlinks; unchanged registered source lease rejects them before loading.',
        'old_package_path': parent['runtime_parent_plan']['package_path'], 'new_package_path': str(PACKAGE),
        'model_copy_receipt_sha256': file_sha(PACKAGE / 'model-copy-receipt.json'),
        'model_file_tree_sha256': receipt['file_tree_sha256'], 'tokenizer_file_tree_sha256': receipt['tokenizer_tree_sha256'],
        'full_model_and_tokenizer_bytes_match': True, 'regular_source_lease_and_fresh_hash_checks_retained': True,
        'first_failed_run_preserved': True, 'scientific_artifacts_unchanged': True,
        'numeric_kernels_runtime_identity_and_thresholds_unchanged': True, 'human_fields_changed': 0}
    plan = deepcopy(parent); plan.pop('plan_id')
    plan['runtime_parent_plan']['package_path'] = str(PACKAGE)
    plan.update(source_files=sources, code_sha256=code, model_path_amendment=amendment,
                preparation_plan_id=parent['plan_id'], preparation_manifest_sha256=PARENT_HASH)
    plan['plan_id'] = 'evidence-lexicon-scope-' + sha(canonical(plan).encode())
    files = {name: (PARENT / name).read_bytes() for name in manifest['artifacts'] if name not in CHANGED}
    files.update({'plan.json': json_bytes(plan), 'model-path-amendment.json': json_bytes(amendment),
        'audit.json': json_bytes({**read_json(PARENT / 'audit.json'), 'model_path_only_amendment': True,
                                'parent_manifest_sha256': PARENT_HASH, 'scientific_artifacts_unchanged': sorted(files)}),
        'execution_source.json': json_bytes({p: {'sha256': h, 'text': (ROOT / p).read_text()} for p, h in code.items()})})
    for p, h in sources.items(): require(file_sha(ROOT / p) == h, 'source changed: ' + p)
    return files, sources


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=WORK / 'frozen-02')
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args(); target = args.output.resolve()
    require(target.parent == WORK and target.name == 'frozen-02', 'use the separate path-amendment freeze')
    files, sources = build()
    if args.check:
        manifest = read_json(target / 'manifest.json')
        require(manifest['source_files'] == sources and set(manifest['artifacts']) == set(files), 'reconstruction frame differs')
        for name, raw in files.items(): require((target / name).read_bytes() == raw and manifest['artifacts'][name] == sha(raw), 'reconstruction differs: ' + name)
        load_frozen(target)
    else:
        files['manifest.json'] = json_bytes({'schema_version': 'evidence-lexicon-scope-freeze/v1', 'status': 'frozen',
            'created_at': datetime.now(timezone.utc).isoformat(), 'source_files': sources, 'artifacts': {k: sha(v) for k, v in files.items()}})
        write_output(target, files)
    print(canonical({'status': 'verified' if args.check else 'frozen', 'model_path_only_amendment': True,
                     'contexts': 320, 'candidates': 640, 'model_forward_executed': False}))


if __name__ == '__main__': main()
