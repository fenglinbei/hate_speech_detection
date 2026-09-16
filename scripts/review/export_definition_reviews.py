#!/usr/bin/env python3
"""Export local definition reviews; explicit replies never confirm AI fields.

This is a local record layer. It does not write to a server session, edit the
frozen lexicon, or substitute corrected definitions in experiment inputs.
"""
import argparse
import copy
import csv
import fcntl
import hashlib
import io
import json
import tempfile
from collections import Counter
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN = ROOT / 'exps/causal_context/general_model_evidence_applicability_v1/ai_reviews/v2-reannotation-20260911'
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'tools/general_model_paired_review_ui')]
from evidence_schema import CHOICES, normalize_values

HUMAN_FIELDS = {'definition_verdict', 'adopted_definition', 'rewrite_instruction'}
VERDICTS = CHOICES['definition_verdict']


def read(path):
    return json.loads(path.read_bytes())


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, indent=2) + '\n').encode()


def digest(data):
    return hashlib.sha256(data).hexdigest()


def ref(path, run):
    return {'path': str(path.relative_to(run)), 'sha256': digest(path.read_bytes())}


def verify(source, run):
    path = run / source['path']
    assert path.resolve().is_relative_to(run.resolve()), source
    if digest(path.read_bytes()) != source['sha256'] and source['path'] == 'sentence_context.json':
        # Old definition batches bound this mutable pointer. Later explicit
        # sentence rechecks archive its exact bytes; never rebind old batches
        # to newer sentence decisions or relax verification for other files.
        archived = run / 'sentence-completion-v1/context-history' / (source['sha256'] + '.json')
        assert archived.exists() and digest(archived.read_bytes()) == source['sha256'], f'Historical context missing: {path}'
        return archived
    assert digest(path.read_bytes()) == source['sha256'], f'Source changed: {path}'
    return path


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temp = Path(stream.name)
        stream.write(data)
        stream.flush()
    temp.replace(path)


def apply_event(active, history, event, origin):
    """Repeated decisions require a field-specific, reasoned amendment."""
    scope = event['scope']
    assert len(scope) == len(set(scope)) and set(scope) == set(event['values']) <= HUMAN_FIELDS
    assert scope and event['review_kind'] == 'human_with_ai'
    assert event['online_material_confirmation_added'] is False
    amendments = event.get('amends', {})
    assert set(amendments) <= set(scope)
    for field in scope:
        value = event['values'][field]
        if field == 'definition_verdict':
            assert value in VERDICTS, value
        else:
            assert isinstance(value, str) and value.strip(), (field, value)
        previous = active.get(field)
        if previous:
            assert amendments.get(field) == previous['event_id'], f'Explicit amendment required: {field}'
            assert event.get('amendment_reason', '').strip(), 'Amendment reason required'
        else:
            assert field not in amendments, f'Amendment refers to missing field: {field}'
        record = {'value': value, 'event_id': event['event_id'], 'source': origin,
                  'rationale_verbatim': event['rationale'], 'review_kind': 'human_with_ai'}
        history.setdefault(field, []).append(record)
        active[field] = record


def bound_question(bindings, question_id):
    assert len(question_id) == 3 and question_id[0] == 'request_user_input_async'
    assert isinstance(question_id[1], str) and type(question_id[2]) is int
    key = (question_id[1], question_id[2])
    # The first batch predates saving call IDs in its binding table.
    legacy_key = (None, question_id[2])
    assert key in bindings or legacy_key in bindings, 'Unknown question request'
    return bindings[key] if key in bindings else bindings[legacy_key]


def build(run):
    base = run / 'resource-reviews-v1'
    manifest = read(run / 'input_manifest.json')
    input_path = run / 'inputs/definitions.jsonl'
    assert digest(input_path.read_bytes()) == manifest['inputs']['definitions.jsonl']
    originals = [json.loads(line) for line in input_path.read_text().splitlines() if line.strip()]
    all_inputs = {r['id']: r for r in originals}
    assert len(all_inputs) == len(originals) == manifest['counts']['definition']
    assert set(all_inputs) <= set(manifest['pending_ids'])
    source_refs = [ref(run / 'input_manifest.json', run), ref(input_path, run)]
    batches, results, all_event_ids = [], {}, set()
    notes = {'case_summaries': [], 'lexical_supplements': [], 'followup_checks': []}
    prior_references = {}
    for folder in sorted((base / 'definitions').glob('batch-*')):
        if not (folder / 'selection.json').exists():
            continue
        selection = read(folder / 'selection.json')
        selected = selection['selected_ids']
        assert len(set(selected)) == len(selected) == selection['object_count']
        assert set(selected) <= set(all_inputs) and not (set(selected) & set(results))
        for key in ('inputs', 'definition_input', 'manifest', 'sentence_context',
                    'prior_definition_reference', 'initial_ai_drafts'):
            verify(selection[key], run)
        for source in selection['calibration_snapshot']['artifacts'].values():
            verify(source, run)
        if 'previous_definition_current' in selection:
            previous_path = verify(selection['previous_definition_current'], run)
            previous = read(previous_path)
            calibration = selection['resource_calibration_snapshot']
            assert previous['snapshot_id'] == calibration['snapshot_id']
            assert previous['artifacts'] == calibration['artifacts']
            for source in calibration['artifacts'].values():
                verify(source, run)
        for source in selection['reuse_metadata_sources']:
            verify(source, run)
        local_inputs = [json.loads(line) for line in (folder / 'inputs.jsonl').read_text().splitlines()]
        assert local_inputs == [all_inputs[oid] for oid in selected], 'Frozen definition changed'
        for prior in read(folder / 'prior-human-definition-reference.json')['records']:
            oid = prior['object_id']
            assert oid not in all_inputs, 'Prior confirmations cannot count as new work'
            assert oid not in prior_references or prior_references[oid] == prior
            prior_references[oid] = prior
        draft_doc = read(folder / 'ai-drafts.json')
        drafts = {r['object_id']: r for r in draft_doc['records']}
        assert set(drafts) == set(selected) and len(drafts) == len(draft_doc['records'])
        questions = {}
        question_paths = [folder / 'questions.json', *sorted(folder.glob('additional-questions-*.json'))]
        for question_path in question_paths:
            question_doc = read(question_path)
            assert question_doc['selection_sha256'] == digest((folder / 'selection.json').read_bytes())
            rows = question_doc['records']
            assert len(rows) == question_doc['question_count']
            assert {r['question_index'] for r in rows} == set(range(len(rows)))
            assert len({q['object_id'] for q in rows}) == len(rows)
            source_refs.append(ref(question_path, run))
            for q in rows:
                assert q['object_id'] in selected
                assert q['original_definition'] == all_inputs[q['object_id']]['source']['text']
                key = (q.get('request_call_id'), q['question_index'])
                assert key not in questions, 'Duplicate question request binding'
                questions[key] = {**q, 'binding_source': ref(question_path, run)}
        for name in ('selection.json', 'ai-drafts.json', 'questions.json', 'web-evidence.json',
                     'prior-human-definition-reference.json'):
            source_refs.append(ref(folder / name, run))
        active, history = {}, {}
        for path in sorted((folder / 'discussions').glob('user-decisions-*.json')):
            doc = read(path)
            assert doc['selection_sha256'] == digest((folder / 'selection.json').read_bytes())
            source_refs.append(ref(path, run))
            for index, event in enumerate(doc['records']):
                oid = event['object_id']
                assert oid in selected and event['event_id'] not in all_event_ids
                all_event_ids.add(event['event_id'])
                assert event['source_sha256'] == digest(encoded(all_inputs[oid]['source']))
                reply_path = verify(event['source_reply_reference'], run)
                source_refs.append(ref(reply_path, run))
                reply = read(reply_path)['replies'][event['source_reply_index']]
                question_id = json.loads(reply['questionItemId'])
                assert question_id == event['question_item_id']
                question = bound_question(questions, question_id)
                assert question['object_id'] == oid
                assert set(event['scope']) <= set(question['allowed_user_fields'])
                if 'question_text_verbatim' in question:
                    assert reply['question'] == question['question_text_verbatim']
                if 'source_question_reference' in event:
                    verify(event['source_question_reference'], run)
                    assert event['source_question_reference'] == question['binding_source']
                assert all_inputs[oid]['source']['entry_id'] in reply['question']
                assert reply['answer'] == event['rationale']
                apply_event(active.setdefault(oid, {}), history.setdefault(oid, {}), event,
                            {**ref(path, run), 'record_index': index})
        for oid in selected:
            original = all_inputs[oid]
            draft = drafts[oid]
            source = original['source']
            assert draft['source_sha256'] == digest(encoded(source))
            assert draft['definition_text_sha256'] == digest(source['text'].encode())
            assert draft['base_version'] == manifest['pending_versions'][oid]
            assert draft['review_kind'] == 'ai_note' and not draft['new_human_fields']
            assert draft['online_material_confirmed'] is False
            assert draft['assessment_status'] in ('ai_reviewed', 'needs_discussion')
            values = normalize_values('definition', draft['values'], source,
                                      required=draft['assessment_status'] == 'ai_reviewed')
            assert bool(values['definition_verdict']) == (draft['assessment_status'] == 'ai_reviewed')
            assert draft['proposal']['definition_verdict'] in VERDICTS
            human = active.get(oid, {})
            verdict = human.get('definition_verdict', {}).get('value', values['definition_verdict'])
            provenance = 'human_with_ai' if 'definition_verdict' in human else 'ai_note' if verdict else 'pending'
            status = 'needs_discussion' if not verdict else 'human_answered' if human else 'ai_reviewed'
            results[oid] = {'object_id': oid, 'kind': 'definition', 'batch': folder.name,
                            'original_source': copy.deepcopy(source), 'source_sha256': draft['source_sha256'],
                            'base_version': draft['base_version'], 'definition_verdict': verdict,
                            'verdict_provenance': provenance, 'assessment_status': status,
                            'ai_draft': copy.deepcopy(draft), 'human_fields': copy.deepcopy(human),
                            'human_field_history': copy.deepcopy(history.get(oid, {})),
                            'adopted_definition': human.get('adopted_definition', {}).get('value'),
                            'rewrite_instruction': human.get('rewrite_instruction', {}).get('value'),
                            'meaning_unresolved': verdict == 'uncertain', 'online_material_confirmed': False}
        notes_path = folder / 'post-decision-notes.json'
        if notes_path.exists():
            post = read(notes_path)
            assert post['review_kind'] == 'ai_note'
            assert post['new_general_human_rules'] == post['online_material_confirmations_added'] == 0
            source_refs.append(ref(notes_path, run))
            for source in post['sources']:
                verify(source, run)
            for application in post['instruction_applications']:
                oid = application['object_id']
                assert oid in selected
                assert application['verbatim_user_approval'] is False
                assert application['source_event_id'] == results[oid]['human_fields']['rewrite_instruction']['event_id']
                results[oid]['instruction_applied_definition_ai'] = copy.deepcopy(application)
            for key in notes:
                for item in post[key]:
                    basis = item.get('basis_event_ids', [item.get('source_event_id')])
                    assert set(basis) <= all_event_ids
                    if key == 'case_summaries':
                        assert item['is_explicit_general_user_rule'] is False
                    if key == 'followup_checks':
                        assert item['existing_queue_items_only'] and not item['sentence_decision_changed']
                        assert set(item['target_hit_ids']) <= set(manifest['pending_ids'])
                    notes[key].append(copy.deepcopy(item))
        batches.append({'batch': folder.name, 'selection': ref(folder / 'selection.json', run),
                        'question_count': len(questions), 'selected_ids': selected})
    records = list(results.values())
    counts = Counter(r['assessment_status'] for r in records)
    status = {'definitions_total': len(all_inputs), 'definitions_selected': len(records),
              'definitions_not_started': len(all_inputs) - len(records),
              'definitions_ai_reviewed': counts['ai_reviewed'],
              'definitions_user_answered': counts['human_answered'],
              'definition_quality_user_fields': sum('definition_verdict' in r['human_fields'] for r in records),
              'definition_quality_ai_fields': sum(r['verdict_provenance'] == 'ai_note' for r in records),
              'definitions_user_rewrite_only': sum(bool(r['human_fields']) and 'definition_verdict' not in r['human_fields'] for r in records),
              'definitions_awaiting_reply': counts['needs_discussion'],
              'definitions_meaning_unresolved': sum(r['meaning_unresolved'] for r in records),
              'explicitly_adopted_rewrites': sum(r['adopted_definition'] is not None for r in records),
              'missing_original_definitions': sum(not r['original_source']['text'] for r in records),
              'missing_definitions_without_adopted_rewrite': sum(not r['original_source']['text'] and not r['adopted_definition'] for r in records),
              'human_field_events': sum(len(h) for r in records for h in r['human_field_history'].values()),
              'instruction_applied_definitions_ai': sum('instruction_applied_definition_ai' in r for r in records),
              'case_preference_summaries_ai': len(notes['case_summaries']),
              'occurrence_followups_queued': len(notes['followup_checks']),
              'prior_definition_confirmations_referenced': len(prior_references),
              'online_material_confirmations_added': 0}
    return {'schema_version': 'local-evidence-definition-reviews/v1', 'status': status,
            'batches': batches, 'sources': list({r['path']: r for r in source_refs}.values()),
            'records': records, 'prior_human_references': list(prior_references.values()),
            'preference_notes': notes,
            'limits': ['AI判定不计人工确认；用户只裁定实际回复的字段。',
                       '改写建议与明确采纳的改写分列；均不覆盖原词库及实验输入。',
                       '未决是一项已作出的判断，仍需另行核验词义；不算合理定义。']}


def markdown(doc):
    c = doc['status']
    source_labels = {'human_with_ai': '用户裁决', 'ai_note': 'AI 判定', 'pending': '待回复'}
    lines = ['# 定义复核与裁决偏好', '',
             f"已整理 {c['definitions_selected']}/{c['definitions_total']} 项；仅 AI 复核 {c['definitions_ai_reviewed']} 项，"
             f"用户参与 {c['definitions_user_answered']} 项，待回复 {c['definitions_awaiting_reply']} 项，"
             f"尚未开始 {c['definitions_not_started']} 项。", '',
             f"按字段计，原定义质量由用户直接裁定 {c['definition_quality_user_fields']} 项、AI 判定 {c['definition_quality_ai_fields']} 项；"
             f"{c['explicitly_adopted_rewrites']} 份完整改写获明确采纳。只回答改写不会自动计作四选一质量裁决。", '',
             f"当前另有 {c['definitions_meaning_unresolved']} 项词义未决。旧人工定义 {c['prior_definition_confirmations_referenced']} 项仅作参考，不计本轮新增。", '',
             '这里只评定义本身；命中是否适用、句子标签和严重度另判。AI 归纳不自动成为已确认规则。', '',
             '| 词语 | 原定义判定 | 来源 | 已明确采纳改写 |', '| --- | --- | --- | --- |']
    for r in doc['records']:
        term = r['original_source']['term'].replace('|', '\\|')
        lines.append(f"| {term} | {VERDICTS.get(r['definition_verdict'], '待回复')} | "
                     f"{source_labels[r['verdict_provenance']]} | {'是' if r['adopted_definition'] else '按用户要求补写' if r.get('instruction_applied_definition_ai') else '否'} |")
    if doc['preference_notes']['case_summaries']:
        lines.extend(['', '## 本轮偏好速查', '', '以下是对明确个案裁决的 AI 归纳，不是额外获得批准的通用规则。', '',
                      '| 编号／情形 | 个案归纳 | 复用边界 |', '| --- | --- | --- |'])
        for note in doc['preference_notes']['case_summaries']:
            lines.append(f"| {note['id']} · {note['topic']} | {note['case_summary_ai']} | {note['boundary_ai']} |")
    for note in doc['preference_notes']['followup_checks']:
        lines.extend(['', f"后续复查 {note['linked_sentence_id']}：{note['reason_ai']}", '',
                      note.get('consistency_note_ai', '这属于新增词义依据后的具体用法复查，尚不能认定前后裁决冲突；旧句子裁决保持。')])
    for r in doc['records']:
        s = r['original_source']
        lines.extend(['', f"## {s['term']} · {s['sense_id']}", '', f"> {s['text']}", '',
                      f"原定义：{VERDICTS.get(r['definition_verdict'], '待回复')}（{source_labels[r['verdict_provenance']]}）。"])
        for field, h in r['human_fields'].items():
            lines.append(f"\n用户字段 {field}：{h['value']}。原答：{h['rationale_verbatim']}")
        lines.append('\nAI 复核依据：' + r['ai_draft']['values']['note'])
        proposed = r['ai_draft']['proposal']['suggested_definition']
        if proposed:
            lines.append('\n原 AI 改写建议：' + proposed)
        application = r.get('instruction_applied_definition_ai')
        if application:
            lines.append('\n按用户修改要求补写的当前文本：' + application['definition_text'])
            lines.append('\n此文本由 AI 按明确要求拟写；修改要求已获确认，新措辞不计逐字人工作答。')
        if any(len(h) > 1 for h in r['human_field_history'].values()):
            lines.append('\n此项有显式修订；旧字段值及修订来源保存在 JSON 的 human_field_history。')
    return ('\n'.join(lines) + '\n').encode()


def export(run, *, check=False):
    doc = build(run)
    data = encoded(doc)
    snapshot_id = digest(data)
    out = run / 'resource-reviews-v1/definitions'
    folder = out / 'snapshots' / snapshot_id
    stream = io.StringIO(newline='')
    columns = ['object_id', 'term', 'original_definition', 'definition_verdict', 'verdict_provenance',
               'assessment_status', 'adopted_definition', 'rewrite_instruction', 'ai_suggested_definition',
               'instruction_applied_definition_ai']
    writer = csv.DictWriter(stream, fieldnames=columns)
    writer.writeheader()
    for r in doc['records']:
        writer.writerow({k: r.get(k, '') for k in columns} | {
            'term': r['original_source']['term'], 'original_definition': r['original_source']['text'],
            'ai_suggested_definition': r['ai_draft']['proposal']['suggested_definition'] or '',
            'instruction_applied_definition_ai': r.get('instruction_applied_definition_ai', {}).get('definition_text', '')})
    files = {'reviews.json': data, 'reviews.csv': stream.getvalue().encode(), 'preferences.md': markdown(doc)}
    current = {'schema_version': 'local-evidence-definition-current/v1', 'snapshot_id': snapshot_id,
               'status': doc['status'], 'artifacts': {name: {'path': str((folder / name).relative_to(run)),
                                                          'sha256': digest(content)} for name, content in files.items()}}
    artifacts = {folder / name: content for name, content in files.items()}
    artifacts[out / 'current.json'] = encoded(current)
    artifacts[out.parent / 'preferences.md'] = files['preferences.md']
    artifacts[run / 'definition_context.json'] = encoded({
        'schema_version': 'local-evidence-definition-context/v1',
        'current': {'path': str((out / 'current.json').relative_to(run)), 'sha256': digest(encoded(current))},
        'snapshot_id': snapshot_id, 'online_session_updated': False})
    for path, content in artifacts.items():
        if check:
            assert path.read_bytes() == content, f'Stale export: {path}'
        elif path.is_relative_to(folder) and path.exists():
            assert path.read_bytes() == content, f'Immutable snapshot conflict: {path}'
        else:
            write(path, content)
    return current


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=DEFAULT_RUN)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    out = args.run / 'resource-reviews-v1/definitions'
    if args.check:
        result = export(args.run, check=True)
    else:
        out.mkdir(parents=True, exist_ok=True)
        with (out / '.export.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            result = export(args.run)
    print(json.dumps({'snapshot_id': result['snapshot_id'], **result['status']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
