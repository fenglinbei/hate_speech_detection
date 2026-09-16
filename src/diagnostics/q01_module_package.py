"""Independent module-refinement freeze; all first-round sources remain immutable."""
from pathlib import Path
from diagnostics.general_model_evidence_evaluation import require, file_sha
from diagnostics.q01_mechanism_package import ROOT, BASE, read_json, read_lines, verify_sources, atomic_json
from diagnostics.q01_mechanism_inputs import digest, DESCRIPTOR

PARENT_WORK = BASE / 'reviews/q01-local-mechanism-v1'
PARENT = PARENT_WORK / 'frozen-01'
PARENT_HASH = 'e9391ef24ff6cf136840040d21430890098dcf266a98c66e899cc111bb9fc3c0'
WORK = BASE / 'reviews/q01-module-refinement-v1'
FREEZE = WORK / 'frozen-01'
PUBLIC = BASE / 'q01-module-refinement-v1'


def load_frozen(directory=FREEZE):
    directory = Path(directory).resolve()
    manifest = read_json(directory / 'manifest.json')
    require(manifest['schema_version'] == 'q01-module-freeze/v1' and manifest['status'] == 'frozen', 'not a module freeze')
    verify_sources(manifest['source_files'])
    require({p.name for p in directory.iterdir()} == {*manifest['artifacts'], 'manifest.json'}, 'freeze inventory changed')
    for name, h in manifest['artifacts'].items():
        require(file_sha(directory / name) == h, 'frozen artifact changed: ' + name)
    plan = read_json(directory / 'plan.json')
    body = dict(plan)
    pid = body.pop('plan_id')
    require(pid == 'q01-module-' + digest(body), 'plan ID differs')
    require(plan['source_files'] == manifest['source_files'], 'source closure differs')
    for name, h in plan['data_sha256'].items():
        require(manifest['artifacts'][name] == h, 'plan data binding differs: ' + name)
    contexts = read_lines(directory / 'contexts.jsonl')
    positions = {r['record_id']: r for r in read_lines(directory / 'positions.jsonl')}
    requests = read_lines(directory / 'requests.jsonl')
    require(len(contexts) == len(positions) == 96, 'source frame differs')
    for c in contexts:
        require(c['context_sha256'] == digest({k: v for k, v in c.items() if k != 'context_sha256'}), 'source context changed')
        require({k: c[k] for k in DESCRIPTOR} == {k: positions[c['record_id']][k] for k in DESCRIPTOR}, 'position source differs')
    require(len(requests) == len({r['request_id'] for r in requests}) == plan['budget']['unique_requests'], 'request frame differs')
    for r in requests:
        require(r['request_id'] == 'QMP-' + digest({k: v for k, v in r.items() if k != 'request_id'}), 'request identity differs')
    require(sum(p['candidate_evaluations'] for p in plan['schedule']) == plan['budget']['scheduled_candidate_evaluations'], 'schedule differs')
    require([p['phase'] for p in plan['schedule']] == ['engineering'] * 6 + ['science'] * 6, 'engineering must precede science')
    # analysis-reference.json is hashed but never parsed here or during GPU scoring.
    return plan, contexts, positions, requests
