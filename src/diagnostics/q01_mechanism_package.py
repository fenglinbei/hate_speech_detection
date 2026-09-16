"""Immutable preparation and run-independent integrity checks for Q01 hooks."""
from __future__ import annotations

import json
from pathlib import Path

from diagnostics.general_model_evidence_evaluation import require, file_sha, json_bytes
from diagnostics.q01_mechanism_inputs import digest, DESCRIPTOR

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'exps/causal_context/general_model_evidence_applicability_v1'
WORK = BASE / 'reviews/q01-local-mechanism-v1'
FREEZE = WORK / 'frozen-01'
PUBLIC = BASE / 'q01-local-mechanism-v1'
PARENT = BASE / 'reviews/functional-query-diagnostics-v1/execution-01/frozen-stage-1-01'
PARENT_HASH = '31ef96979d17c7bda5343ec7e3691b61ce9817b30abccc07c36db91e77df2f5f'


def read_json(path): return json.loads(Path(path).read_text())


def read_lines(path):
    with Path(path).open() as f: return [json.loads(line) for line in f if line.strip()]


def verify_sources(sources):
    for name, expected in sources.items():
        require(file_sha(ROOT / name) == expected, 'source changed: ' + name)


def load_frozen(directory=FREEZE):
    directory = Path(directory).resolve()
    manifest = read_json(directory / 'manifest.json')
    require(manifest['schema_version'] == 'q01-mechanism-freeze/v1' and manifest['status'] == 'frozen', 'not a Q01 freeze')
    verify_sources(manifest['source_files'])
    require({p.name for p in directory.iterdir()} == {*manifest['artifacts'], 'manifest.json'}, 'freeze inventory changed')
    for name, h in manifest['artifacts'].items():
        require(file_sha(directory / name) == h, 'frozen artifact changed: ' + name)
    plan = read_json(directory / 'plan.json'); body = dict(plan); pid = body.pop('plan_id')
    require(pid == 'q01-mechanism-' + digest(body), 'plan ID differs')
    require(plan['source_files'] == manifest['source_files'], 'source closure differs')
    for name, h in plan['data_sha256'].items():
        require(manifest['artifacts'][name] == h, 'plan data binding differs: ' + name)
    contexts = read_lines(directory / 'contexts.jsonl')
    positions = {r['record_id']: r for r in read_lines(directory / 'positions.jsonl')}
    requests = read_lines(directory / 'requests.jsonl')
    require(len(contexts) == len(positions) == 96, 'source frame coverage differs')
    for c in contexts:
        require(c['context_sha256'] == digest({k: v for k, v in c.items() if k != 'context_sha256'}), 'source context identity changed')
        require({k: c[k] for k in DESCRIPTOR} == {k: positions[c['record_id']][k] for k in DESCRIPTOR}, 'position source differs')
    require(len(requests) == len({r['request_id'] for r in requests}) == plan['budget']['unique_requests'], 'request count differs')
    for r in requests:
        require(r['request_id'] == 'QMP-' + digest({k: v for k, v in r.items() if k != 'request_id'}), 'request identity differs')
    require(sum(p['candidate_evaluations'] for p in plan['schedule']) == plan['budget']['scheduled_candidate_evaluations'], 'schedule budget differs')
    # analysis-reference.json is hashed above, never parsed by this scoring loader.
    return plan, contexts, positions, requests


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('wb') as f:
        f.write(json_bytes(value)); f.flush()
        import os
        os.fsync(f.fileno())
    temporary.replace(path)
