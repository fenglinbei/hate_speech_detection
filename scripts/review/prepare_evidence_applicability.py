#!/usr/bin/env python3
"""Prepare discovery-only inputs, then join immutable AI notes into a review bundle.

No model is run here. The input stage excludes query Gold/predictions from AI
inputs. Demo labels must exist before original-answer comparison is materialized.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
EXP = ROOT / 'exps/causal_context/general_model_evidence_applicability_v1'
PREP = EXP / 'preparation'
DATA = ROOT / 'exps/causal_context/general_model_ld_nolabel_paired_cases_v1/results/paired-cases-02'
POLICY = ROOT / 'docs/research/annotation-guidelines/evidence-applicability-annotation-policy-v1.md'
LEXICON = ROOT / 'data/lexicon/annotated_lexicon_mechanism_frozen_v1.json'

def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()

def encoded(value) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + '\n').encode()

def identity(value) -> str:
    return digest(encoded(value))

def write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_bytes() != content:
        raise ValueError('Refusing to overwrite different preparation artifact: ' + str(path))
    if not path.exists():
        path.write_bytes(content)

def rows(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

def jsonl(path: Path, values) -> None:
    write(path, ''.join(json.dumps(v, ensure_ascii=False) + '\n' for v in values).encode())

def prepare() -> dict:
    from rag.controlled_lexicon_matcher import ControlledLexiconMatcher, _normalized_text_with_boundaries
    import regex
    assert regex.__version__ == '2026.4.4'
    manifest_raw = (DATA / 'manifest.json').read_bytes()
    manifest = json.loads(manifest_raw)
    baseline = json.loads((EXP / 'audits/baseline-audit-20260908.json').read_bytes())
    assert digest(manifest_raw) == baseline['paired_manifest_sha256']
    policy_freeze = json.loads((EXP / 'policy_freeze.json').read_bytes())
    assert digest(POLICY.read_bytes()) == '660d973c6248b17068ce44f68ebc4680515e54fd17354bd51650ace7c436557a'
    lexraw = LEXICON.read_bytes()
    assert digest(lexraw) == '31240193eeba29f712560e2e80e89ee9dd5c3c969aa180d10b95bf2451882385'
    lexicon = json.loads(lexraw)
    matcher_code = (ROOT / 'src/rag/controlled_lexicon_matcher.py').read_bytes()
    assert matcher_code == subprocess.check_output(['git', 'show', baseline['scientific_source_commit'] + ':src/rag/controlled_lexicon_matcher.py'], cwd=ROOT)
    matcher = ControlledLexiconMatcher(lexicon['terms'], lexicon_sha256=digest(lexraw), policy_sha256=lexicon['matcher_policy_sha256'])
    def frozen(name):
        raw = (DATA / name).read_bytes()
        assert digest(raw) == manifest['artifacts'][name], name
        return json.loads(raw)
    index = frozen('cases/cards_index.json')
    assert len(index) == 32
    objects, cases, matches = {}, {}, {}
    original_demos = {}
    def obj(key, kind, source):
        value = {'id': key, 'kind': kind, 'version': identity(source), 'source': source}
        assert key not in objects or objects[key] == value, key
        objects[key] = value
        return key
    def replay(text, expected):
        if text not in matches:
            result = matcher.match(text)
            normalized, boundaries = _normalized_text_with_boundaries(text)
            matches[text] = {'selected': [c for c in result['candidates'] if c['selection'] == 'selected'], 'normalized_text': normalized, 'normalized_to_raw_boundaries': boundaries}
        result = matches[text]
        assert {c['lexicon_id'] for c in result['selected']} == set(expected), 'Matcher aggregate differs; provenance_mismatch'
        return result
    for row in index:
        path = row['resources_card'].replace('/cards/', '/card_data/').replace('-1-resources.md', '.json')
        card = frozen(path)
        assert card['selection']['split'] == 'discovery'
        qid, query = row['query_id'], card['query']['content']
        resources = card['profile']['resources']
        ids = [obj('query:' + qid, 'query', {'text': query, 'query_id': qid, 'text_sha256': digest(query.encode())})]
        for demo in card['demonstrations']:
            did, text = demo['id'], demo['content']
            original = {task: demo['projection'][task] for task in ('hate', 'group')}
            if did in original_demos:
                assert original_demos[did] == {'id': 'demo:' + did, 'original_answer': original, 'text': text}
            original_demos[did] = {'id': 'demo:' + did, 'original_answer': original, 'text': text}
            ids.append(obj('demo:' + did, 'demo', {'text': text, 'demo_id': did, 'text_sha256': digest(text.encode()), 'dataset_id': baseline['source_dataset_id'], 'original_answer': original, 'original_quadruples': demo['quadruples']}))
            ids.append(obj('relation:' + qid + ':' + did, 'relation', {'text': text, 'texts': {'query': query, 'demo': text}, 'query_id': qid, 'demo_id': did, 'introduced_entry_ids': resources['demo_match_ids'][did]}))
        sources = [('query', qid, query, replay(query, resources['lq_ids']))]
        sources += [('demo', d['id'], d['content'], replay(d['content'], resources['demo_match_ids'][d['id']])) for d in card['demonstrations']]
        for entry in card['lexicon_entries']:
            eid = entry['lexicon_id']
            rendered = [{'condition': c['condition'], 'task': c['task'], 'prompt_sha256': c['prompt_sha256']} for c in card['contexts'] if eid in c['trace']['injected_lexicon_ids']]
            assert rendered
            for sense in entry['senses']:
                sid = sense['sense_id']
                ids.append(obj('definition:' + sid, 'definition', {'text': sense['definition'], 'term': entry['term'], 'entry_id': eid, 'sense_id': sid, 'categories': sense['categories'], 'lexicon_sha256': digest(lexraw)}))
                for source_kind, source_id, text, match in sources:
                    for hit in match['selected']:
                        if hit['lexicon_id'] != eid:
                            continue
                        start, end = hit['span']
                        assert text[start:end] == hit['raw_surface']
                        key = f'hit:{qid}:{source_kind}:{source_id}:{sid}:{start}:{end}'
                        ids.append(obj(key, 'hit', {'text': text, 'texts': {'query': query, 'demo': text if source_kind == 'demo' else '', 'definition': sense['definition']}, 'query_id': qid, 'source_kind': source_kind, 'source_id': source_id, 'term': entry['term'], 'entry_id': eid, 'sense_id': sid, 'raw_span': [start, end], 'raw_surface': hit['raw_surface'], 'normalized_span': hit['normalized_span'], 'normalized_text': match['normalized_text'], 'normalized_to_raw_boundaries': match['normalized_to_raw_boundaries'], 'rendered_in': rendered, 'provenance_status': 'verified_frozen_matcher_replay'}))
        cases[qid] = {'query_id': qid, 'query_object_id': ids[0], 'object_ids': ids, 'comparison': {'gold': {task: card['query']['projection'][task] for task in ('hate', 'group')}, 'original_quadruples': card['query']['quadruples'], 'trajectories': card['profile']['conditions'], 'selection': card['selection'], 'contexts': card['contexts'], 'ai_explanation': {'status': 'awaiting_material_review', 'review_kind': 'ai_note', 'note': '材料尚未人工确认；暂缓机制解释，不影响提交本例裁决。'}}}
    result = {'schema_version': 'general-model-evidence-bundle/v1', 'policy': {'version': 'evidence-applicability-annotation-policy/v1', 'sha256': digest(POLICY.read_bytes()), 'text': POLICY.read_text()}, 'source_identity': {'paired_identity': manifest['identity'], 'paired_manifest_sha256': digest(manifest_raw), 'dataset_id': baseline['source_dataset_id'], 'partition_id': baseline['source_partition_id'], 'lexicon_sha256': digest(lexraw), 'matcher_sha256': digest(matcher_code), 'matcher_source_commit': baseline['scientific_source_commit'], 'matcher_policy_sha256': matcher.policy_sha256, 'regex_version': regex.__version__, 'reserve_body_read': False}, 'order': list(cases), 'cases': cases, 'objects': objects}
    write(PREP / 'base_bundle.json', encoded(result))
    # Inputs for labels intentionally contain just opaque ID and text.
    for kind, filename in [('query', 'queries'), ('demo', 'demos')]:
        values = [{'id': o['id'], 'text': o['source']['text']} for o in objects.values() if o['kind'] == kind]
        # Existing independently prepared inputs may have another stable order.
        target = PREP / 'ai_inputs' / (filename + '.jsonl')
        if target.exists():
            assert {x['id']: x for x in rows(target)} == {x['id']: x for x in values}
        else:
            jsonl(target, values)
    for kind in ('relation', 'definition', 'hit'):
        safe = [{'id': o['id'], 'kind': kind, 'source': o['source']} for o in objects.values() if o['kind'] == kind]
        jsonl(PREP / 'ai_inputs' / (kind + 's.jsonl'), safe)
    jsonl(PREP / 'original_demo_answers.jsonl', original_demos.values())
    counts = Counter(o['kind'] for o in objects.values())
    write(PREP / 'preparation_receipt.json', encoded({'source_identity': result['source_identity'], 'counts': counts, 'query_count': len(cases), 'ai_input_sha256': {p.name: digest(p.read_bytes()) for p in (PREP / 'ai_inputs').glob('*.jsonl') if p.stem in {'queries', 'demos', 'relations', 'definitions', 'hits'}}, 'human_confirmations_created': 0}))
    partition_inputs(result)
    print(json.dumps(counts))
    return result

def partition_inputs(bundle):
    bundle = json.loads(encoded(bundle))
    demo_rows = rows(PREP / 'ai_inputs/demos.jsonl')
    for index in range(4):
        jsonl(PREP / f'ai_inputs/demos_part{index+1}.jsonl', demo_rows[index*70:(index+1)*70])
    used, parts = set(), [[], []]
    for index, key in enumerate(bundle['order']):
        case = bundle['cases'][key]
        objects = [bundle['objects'][oid] for oid in case['object_ids']]
        safe = {'query_id': key, 'query': bundle['objects'][case['query_object_id']]['source']['text'], 'demos': {}, 'definitions': {}, 'required_drafts': []}
        for obj in objects:
            source = obj['source']
            if obj['kind'] == 'demo':
                safe['demos'][source['demo_id']] = source['text']
            if obj['kind'] == 'definition':
                safe['definitions'][source['sense_id']] = {'term': source['term'], 'text': source['text'], 'categories': source['categories']}
            if obj['kind'] in {'relation', 'hit', 'definition'} and obj['id'] not in used:
                used.add(obj['id'])
                safe['required_drafts'].append({'id': obj['id'], 'kind': obj['kind'], **{k: v for k, v in source.items() if k in {'demo_id', 'introduced_entry_ids', 'source_kind', 'source_id', 'sense_id', 'raw_span', 'raw_surface', 'entry_id'}}})
        parts[index // 16].append(safe)
    for index, part in enumerate(parts):
        jsonl(PREP / f'ai_inputs/resource_cases_part{index+1}.jsonl', part)

def compare_demos():
    from tools.general_model_paired_review_ui.evidence_schema import normalize_values
    originals = {r['id']: r for r in rows(PREP / 'original_demo_answers.jsonl')}
    drafts = [r for p in sorted((PREP / 'ai_outputs').glob('demo_labels*.jsonl')) for r in rows(p)]
    assert len(drafts) == len(originals) and {r['id'] for r in drafts} == set(originals)
    compared = []
    for row in drafts:
        value = normalize_values('query', row['values'], {'text': originals[row['id']]['text']}, required=True)
        status = {}
        for task in ('hate', 'group'):
            reason = value[task + '_reason']
            status[task + '_original_status'] = ('policy_ambiguous' if reason in {'individual_scope', 'institution_scope', 'category_boundary'} else 'context_insufficient') if value[task] is None else ('accepted' if value[task] == originals[row['id']]['original_answer'][task] else 'suspected_error')
        compared.append({'id': row['id'], 'status': status, 'label_draft_sha256': identity(row), 'provenance': {'review_kind': 'ai_note', 'comparison_method': 'deterministic_comparison_after_independent_label_draft', 'scope': 'demo_labels_and_original_demo_answers_only', 'reliability_mapping': {'accepted': 'accepted', 'suspected_error': 'disputed', 'policy_ambiguous': 'unknown', 'context_insufficient': 'unknown'}}})
    jsonl(PREP / 'ai_outputs/demo_answer_comparison.jsonl', compared)

def assemble():
    from tools.general_model_paired_review_ui.evidence_schema import normalize_values
    bundle = json.loads((PREP / 'base_bundle.json').read_bytes())
    # Read-only source record reference, never imported as new human decisions.
    parent_path = ROOT / 'exps/causal_context/general_model_ld_nolabel_paired_cases_v1/reviews/paired-cases-02/runtime/evidence-parent-session-20260908.json'
    parent_raw = parent_path.read_bytes()
    parent = json.loads(parent_raw)
    assert parent['source_identity'] == bundle['source_identity']['paired_identity']
    bundle['source_identity']['parent_review'] = {'authority': 'digitalocean-sgp', 'session_path': '/var/lib/hsd-general-model-paired-review/session.json', 'sha256': digest(parent_raw), 'revision': parent['revision'], 'reviewer_id': parent['reviewer_id'], 'confirmed_count_at_snapshot': sum(r['status'] == 'confirmed' for r in parent['records'].values()), 'reference_only': True}
    for key, case in bundle['cases'].items():
        prior = parent['records'][key]
        case['prior_review_reference'] = {'source_session_sha256': digest(parent_raw), 'status': prior['status'], 'record_sha256': identity(prior), 'known_exposure': {'resources': 'seen' if prior['status'] != 'unreviewed' else 'unsure', 'gold': 'seen' if prior.get('resources_locked_at') else 'unsure', 'results': 'seen' if prior.get('resources_locked_at') else 'unsure', 'ai': 'seen' if prior.get('ai_revealed_at') else 'unsure'}}
        if key == '3169':
            case['prior_review_reference']['known_exposure'] = dict.fromkeys(('gold', 'resources', 'results', 'ai'), 'seen')
            case['prior_review_reference']['additional_exposure_source'] = '2026-09-08 用户已查看本任务中 case 3169 的完整流程演示，包含材料、原 Gold、预测及 AI 解释；演示不是人工裁决。'
    compare_demos()
    comparisons = {r['id']: r for r in rows(PREP / 'ai_outputs/demo_answer_comparison.jsonl')}
    drafts = {}
    for pattern in ('query_labels.jsonl', 'demo_labels*.jsonl', 'resource_drafts*.jsonl'):
        for path in sorted((PREP / 'ai_outputs').glob(pattern)):
            for row in rows(path):
                assert row['id'] not in drafts, row['id']
                row['provenance']['output_file_sha256'] = digest(path.read_bytes())
                drafts[row['id']] = row
    assert set(drafts) == set(bundle['objects']), (set(bundle['objects']) - set(drafts), set(drafts) - set(bundle['objects']))
    for key, obj in bundle['objects'].items():
        src = obj['source']
        if obj['kind'] == 'relation':
            obj['depends_on'] = ['query:' + src['query_id'], 'demo:' + src['demo_id']]
        elif obj['kind'] == 'hit':
            obj['depends_on'] = list(dict.fromkeys(['query:' + src['query_id'], 'definition:' + src['sense_id'], src['source_kind'] + ':' + src['source_id']]))
        row = drafts[key]
        if obj['kind'] == 'demo':
            row['answer_comparison'] = comparisons[key]
            row['values'] = {**row['values'], **comparisons[key]['status']}
        row['values'] = normalize_values(obj['kind'], row['values'], obj['source'], required=True)
        row['provenance'].update(review_kind='ai_note', policy_sha256=bundle['policy']['sha256'], human_confirmed=False)
        obj['ai_draft'] = {**row, 'version': identity(row)}
    bundle['preparation'] = {'ai_note_count': len(drafts), 'human_confirmed': 0, 'input_files': {p.name: digest(p.read_bytes()) for p in (PREP / 'ai_inputs').glob('*.jsonl')}, 'output_files': {p.name: digest(p.read_bytes()) for p in (PREP / 'ai_outputs').glob('*.jsonl')}}
    write(EXP / 'bundle/evidence_bundle.json', encoded(bundle))
    print(json.dumps({'objects': len(drafts), 'cases': len(bundle['cases']), 'bundle_sha256': identity(bundle)}))

if __name__ == '__main__':
    if sys.flags.optimize:
        raise RuntimeError('Run without -O/PYTHONOPTIMIZE; verification assertions are required')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['inputs', 'assemble'])
    args = parser.parse_args()
    prepare() if args.stage == 'inputs' else assemble()
