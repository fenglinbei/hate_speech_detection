#!/usr/bin/env python3
"""Export local occurrence reviews without modifying sentences or human sessions."""
import argparse
import copy
import csv
import fcntl
import io
import json
from pathlib import Path
import sys

from export_definition_reviews import ROOT, DEFAULT_RUN, read, encoded, digest, ref, verify, write

sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'tools/general_model_paired_review_ui')]
from evidence_schema import CHOICES, normalize_values

FIELDS = {'source_fit', 'query_fit'}


def apply_event(active, history, event, origin):
    scope = event['scope']
    assert scope and len(scope) == len(set(scope))
    assert set(scope) == set(event['values']) <= FIELDS
    assert event['review_kind'] == 'human_with_ai'
    assert event['online_material_confirmation_added'] is False
    amendments = event.get('amends', {})
    assert set(amendments) <= set(scope)
    for field in scope:
        value = event['values'][field]
        assert value in CHOICES[field]
        old = active.get(field)
        if old:
            assert amendments.get(field) == old['event_id'], 'Explicit amendment required'
            assert event.get('amendment_reason', '').strip(), 'Amendment reason required'
        else:
            assert field not in amendments
        row = {'value': value, 'event_id': event['event_id'], 'source': origin,
               'rationale_verbatim': event['rationale'], 'review_kind': 'human_with_ai'}
        active[field] = row
        history.setdefault(field, []).append(row)


def effective_issues(values, original_issues):
    """Issue flags are AI-derived, not silently promoted human fields."""
    issues = [values['source_fit']]
    if values['query_fit'] == 'uncertain' and 'uncertain' not in issues:
        issues.append('uncertain')
    for flag in ('overly_narrow_definition', 'provenance_mismatch'):
        if flag in original_issues:
            issues.append(flag)
    return issues


def check_reply(event, questions, folder, run):
    reply_path = verify(event['source_reply_reference'], run)
    reply = read(reply_path)['replies'][event['source_reply_index']]
    qid = json.loads(reply['questionItemId'])
    assert len(qid) == 3 and qid[0] == 'request_user_input_async'
    assert isinstance(qid[1], str) and type(qid[2]) is int
    assert qid == event['question_item_id']
    if event.get('source_question_reference'):
        question_path = verify(event['source_question_reference'], run)
        assert question_path.parent == folder / 'consistency-rechecks'
        question = read(question_path)
        receipt_path = question_path.with_name(question_path.stem + '-request-receipt.json')
        receipt = read(receipt_path)
        assert receipt['questions_sha256'] == digest(question_path.read_bytes())
        assert question['question_index'] == qid[2]
        primary_index = question['resolves_primary_question_index']
        assert primary_index in questions
        assert set(question['object_ids']) <= set(questions[primary_index]['object_ids'])
        assert set(question['allowed_user_fields']) <= set(questions[primary_index]['allowed_user_fields'])
    else:
        receipt = read(folder / 'request-receipt.json')
        assert receipt['questions_sha256'] == digest((folder / 'questions.json').read_bytes())
        assert qid[2] in questions
        question = questions[qid[2]]
        primary_index = qid[2]
    assert qid[1] == receipt['request_call_id'], 'Reply belongs to a different request'
    assert reply['question'] == question['question_text_verbatim']
    assert reply['answer'] == event['rationale']
    assert event['object_id'] in question['object_ids']
    assert set(event['scope']) <= set(question['allowed_user_fields'])
    return primary_index, ref(reply_path, run)


def resolved_questions(questions, active, related):
    """Related instructions may close discussion without inventing fit fields."""
    field_answered = {i for i, q in questions.items()
                      if all(set(q['allowed_user_fields']) <= set(active.get(oid, {}))
                             for oid in q['object_ids'])}
    related_answered = {i for i, q in questions.items()
                        if all(any(e['primary_question_index'] == i for e in related.get(oid, []))
                               for oid in q['object_ids'])}
    return field_answered, related_answered - field_answered


def build(run):
    base = run / 'resource-reviews-v1/hits'
    manifest = read(run / 'input_manifest.json')
    all_inputs = {}
    sources = [ref(run / 'input_manifest.json', run)]
    for path in sorted((run / 'inputs').glob('resources-*.json')):
        assert digest(path.read_bytes()) == manifest['inputs'][path.name]
        package = read(path)
        for hit in package['required']:
            if hit['kind'] != 'hit':
                continue
            source = hit['source']
            text = package['query'] if source['source_kind'] == 'query' else package['demos'][source['source_id']]
            a, z = source['raw_span']
            assert 0 <= a < z <= len(text) and text[a:z] == source['raw_surface']
            assert hit['id'] not in all_inputs
            all_inputs[hit['id']] = {'object_id': hit['id'], 'query_id': package['query_id'],
                'original_hit': hit, 'source_text': text, 'query_text': package['query'],
                'original_definition': package['definitions'][source['sense_id']], 'package': ref(path, run)}
        sources.append(ref(path, run))
    assert len(all_inputs) == manifest['counts']['hit']
    assert set(all_inputs) == {oid for oid in manifest['pending_ids'] if oid.startswith('hit:')}
    records, all_events, batches, preferences = {}, set(), [], []
    for folder in sorted(base.glob('batch-*')):
        if not (folder / 'selection.json').exists():
            continue
        selection = read(folder / 'selection.json')
        selected = selection['selected_ids']
        assert len(set(selected)) == len(selected) == selection['object_count']
        assert set(selected) <= set(all_inputs) and not set(selected) & set(records)
        for name in ('manifest', 'inputs', 'initial_ai_drafts', 'prior_human_reference'):
            verify(selection[name], run)
        for source in [*selection['packages'], *selection['starting_contexts']]:
            verify(source, run)
        for source in selection['definition_snapshot']['artifacts'].values():
            verify(source, run)
        original_rows = read(folder / 'inputs.json')['records']
        assert original_rows == [all_inputs[oid] for oid in selected], 'Frozen source changed'
        drafts_doc = read(folder / 'ai-drafts.json')
        drafts = {r['object_id']: r for r in drafts_doc['records']}
        assert set(drafts) == set(selected) and len(drafts) == len(drafts_doc['records'])
        qdoc = read(folder / 'questions.json')
        questions = {q['question_index']: q for q in qdoc['records']}
        assert set(questions) == set(range(qdoc['question_count']))
        assert len(questions) == len(qdoc['records'])
        for question in questions.values():
            assert set(question['object_ids']) <= set(selected)
            assert set(question['allowed_user_fields']) <= FIELDS
        active, history, related = {}, {}, {}
        for path in sorted((folder / 'discussions').glob('user-decisions-*.json')):
            doc = read(path)
            assert doc['selection_sha256'] == digest((folder / 'selection.json').read_bytes())
            sources.append(ref(path, run))
            for index, event in enumerate(doc['records']):
                oid = event['object_id']
                assert oid in selected and event['event_id'] not in all_events
                all_events.add(event['event_id'])
                assert event['source_sha256'] == digest(encoded(all_inputs[oid]['original_hit']['source']))
                _, origin = check_reply(event, questions, folder, run)
                sources.append(origin)
                if event.get('source_question_reference'):
                    question_path = verify(event['source_question_reference'], run)
                    sources.append(ref(question_path, run))
                    sources.append(ref(question_path.with_name(question_path.stem + '-request-receipt.json'), run))
                apply_event(active.setdefault(oid, {}), history.setdefault(oid, {}), event,
                            {**ref(path, run), 'record_index': index})
        for path in sorted((folder / 'discussions').glob('question-outcomes-*.json')):
            doc = read(path)
            assert doc['selection_sha256'] == digest((folder / 'selection.json').read_bytes())
            sources.append(ref(path, run))
            for index, event in enumerate(doc['records']):
                oid = event['object_id']
                assert oid in selected and event['event_id'] not in all_events
                all_events.add(event['event_id'])
                assert event['resolution_kind'] == 'separate_lexicon_entry_requested'
                assert event['scope'] == event['new_human_fields'] == []
                assert event['review_kind'] == 'human_with_ai'
                assert event['online_material_confirmation_added'] is False
                primary, origin = check_reply(event, questions, folder, run)
                addition_path = verify(event['lexicon_addition_reference'], run)
                addition = read(addition_path)
                assert addition['source_reply_reference'] == event['source_reply_reference']
                assert addition['source_reply_index'] == event['source_reply_index']
                assert addition['user_instruction_verbatim'] == event['rationale']
                assert addition['human_confirmed_content']['separate_entry'] is True
                assert addition['frozen_resources_modified'] is False
                assert addition['online_material_confirmations_added'] == 0
                question_path = verify(event['source_question_reference'], run)
                sources.extend([origin, ref(addition_path, run), ref(question_path, run),
                                ref(question_path.with_name(question_path.stem + '-request-receipt.json'), run)])
                related.setdefault(oid, []).append({**event, 'primary_question_index': primary,
                    'source': {**ref(path, run), 'record_index': index}})
        # A multi-position question stays pending until every displayed field is
        # answered, or every position has an explicit alternative instruction.
        # Such an instruction never confirms fit fields.
        field_answered, related_answered = resolved_questions(questions, active, related)
        replied = field_answered | related_answered
        pending_ids = {oid for i, q in questions.items() if i not in replied for oid in q['object_ids']}
        for oid in selected:
            item = all_inputs[oid]
            source = item['original_hit']['source']
            draft = drafts[oid]
            assert draft['review_kind'] == 'ai_note' and draft['new_human_fields'] == []
            assert draft['online_material_confirmed'] is False
            assert draft['base_version'] == manifest['pending_versions'][oid]
            assert draft['source_sha256'] == digest(encoded(source))
            assert draft['source_text_sha256'] == digest(item['source_text'].encode())
            assert draft['query_text_sha256'] == digest(item['query_text'].encode())
            assert draft['definition_text_sha256'] == digest(item['original_definition']['text'].encode())
            evidence_source = {'texts': {'query': item['query_text'], 'demo': item['source_text'],
                                         'definition': item['original_definition']['text']}}
            normalize_values('hit', draft['values'], evidence_source, required=True)
            values = copy.deepcopy(draft['values'])
            human = active.get(oid, {})
            for field, decision in human.items():
                values[field] = decision['value']
            values['issues'] = effective_issues(values, draft['values']['issues'])
            normalize_values('hit', values, evidence_source, required=True)
            records[oid] = {**item, 'batch': folder.name, 'base_version': draft['base_version'],
                'values': values, 'field_provenance': {f: 'user_discussion' if f in human else 'ai_note' for f in sorted(FIELDS)},
                'issue_provenance': 'ai_derived', 'ai_draft': draft,
                'human_fields': human, 'human_field_history': history.get(oid, {}),
                'related_user_outcomes': related.get(oid, []),
                'assessment_status': 'awaiting_reply' if oid in pending_ids else 'user_participated' if human else 'user_related_instruction' if oid in related else 'ai_reviewed',
                'meaning_unresolved': 'uncertain' in (values['source_fit'], values['query_fit']),
                'online_material_confirmed': False}
        batch_info = {'batch': folder.name, 'selected': len(selected), 'question_count': len(questions),
                      'questions_answered': len(replied), 'questions_pending': len(questions) - len(replied),
                      'questions_answered_by_fit_fields': len(field_answered),
                      'questions_resolved_by_related_instruction': len(related_answered)}
        batches.append(batch_info)
        for name in ('selection.json', 'ai-drafts.json', 'questions.json'):
            sources.append(ref(folder / name, run))
        for name in ('request-receipt.json', 'post-decision-notes.json', 'web-evidence.json',
                     'query-literal-form-application.json', 'source-rule-application.json'):
            path = folder / name
            if path.exists():
                sources.append(ref(path, run))
                if name == 'post-decision-notes.json':
                    preferences.extend(read(path).get('case_summaries_ai', []))
    rows = list(records.values())
    status = {'hits_total': len(all_inputs), 'hits_selected': len(rows),
        'hits_not_started': len(all_inputs) - len(rows),
        'hits_ai_only': sum(not r['human_fields'] and not r['related_user_outcomes'] for r in rows),
        'hits_user_participated': sum(bool(r['human_fields'] or r['related_user_outcomes']) for r in rows),
        'hits_with_explicit_fit_fields': sum(bool(r['human_fields']) for r in rows),
        'hits_with_related_instruction_only': sum(bool(r['related_user_outcomes']) and not r['human_fields'] for r in rows),
        'source_fit_user_fields': sum('source_fit' in r['human_fields'] for r in rows),
        'query_fit_user_fields': sum('query_fit' in r['human_fields'] for r in rows),
        'explicit_human_field_events': sum(len(h) for r in rows for h in r['human_field_history'].values()),
        'questions_pending': sum(b['questions_pending'] for b in batches),
        'hits_awaiting_reply': sum(r['assessment_status'] == 'awaiting_reply' for r in rows),
        'source_fit_unresolved': sum(r['values']['source_fit'] == 'uncertain' for r in rows),
        'query_fit_unresolved': sum(r['values']['query_fit'] == 'uncertain' for r in rows),
        'online_material_confirmations_added': 0}
    return {'schema_version': 'local-hit-reviews/v1', 'status': status, 'batches': batches,
        'sources': list({(s['path'], s['sha256']): s for s in sources}.values()), 'records': rows,
        'case_summaries_ai': preferences,
        'limits': ['本地 AI 协同记录不等同在线人工确认。',
                   'source_fit 与 query_fit 分开；定义质量与原句标签另存。',
                   '采纳改写的比较不替换原定义或冻结提示。',
                   '问题标记与解释仍为 AI，只有实际回复字段计入人工作答。']}


def render(doc, run):
    s = doc['status']
    lines = ['# 命中适配复核', '',
        f"已整理 {s['hits_selected']}/{s['hits_total']} 处；用户参与 {s['hits_user_participated']} 处，待回复 {s['questions_pending']} 个问题。", '',
        f"原义项来源适配不明 {s['source_fit_unresolved']} 处，查询适配不明 {s['query_fit_unresolved']} 处。不明与待回复分开计数。", '',
        '原文、原定义与当前结果逐位置保留。词义适配不等同句子仇恨；AI 初稿理由不自动成为用户理由。', '']
    for batch in doc['batches']:
        folder = run / 'resource-reviews-v1/hits' / batch['batch']
        questions = read(folder / 'questions.json')['records']
        lines += [f"## {batch['batch']} 问题与来源", '']
        by_id = {r['object_id']: r for r in doc['records']}
        for q in questions:
            lines += [f"### {q['question_index'] + 1} · {q['question_key']}", '', q['question_text_verbatim'], '']
            for oid in q['object_ids']:
                r = by_id[oid]
                for field in q['allowed_user_fields']:
                    h = r['human_fields'].get(field)
                    text = '待回复' if not h else f"{CHOICES[field][h['value']]}；原答：{h['rationale_verbatim']}"
                    if not h and r['related_user_outcomes']:
                        text = (f"{CHOICES[field][r['values'][field]]}（AI）；用户另行要求新增词条，"
                                f"不计为本字段人工裁决。原答：{r['related_user_outcomes'][-1]['rationale']}")
                    lines.append(f"- `{oid}` / {field}：{text}")
            lines.append('')
    lines += ['## 逐位置结果', '', '| 查询 | 来源 | 词形与位置 | 来源适配 | 查询适配 | 人工字段 |',
              '| --- | --- | --- | --- | --- | --- |']
    for r in doc['records']:
        h = r['original_hit']['source']
        cells = [r['query_id'], h['source_kind'] + ':' + h['source_id'],
                 f"{h['raw_surface']} {h['raw_span']}", CHOICES['source_fit'][r['values']['source_fit']],
                 CHOICES['query_fit'][r['values']['query_fit']], '、'.join(sorted(r['human_fields'])) or '—']
        lines.append('| ' + ' | '.join(str(c).replace('|', '\\|').replace('\n', ' ') for c in cells) + ' |')
    lines += ['', '## 裁决复用提示（AI 归纳）', '']
    for note in doc['case_summaries_ai']:
        lines.append(f"- {note['id']}：{note['summary']} 适用边界：{note['boundary']}")
    return ('\n'.join(lines) + '\n').encode()


def artifacts(doc, run):
    output = io.StringIO()
    fields = ['object_id', 'query_id', 'source_kind', 'source_id', 'surface', 'start', 'end',
              'original_definition', 'source_text', 'query_text', 'source_fit', 'query_fit',
              'source_fit_provenance', 'query_fit_provenance', 'assessment_status']
    writer = csv.DictWriter(output, fieldnames=fields)
    writer.writeheader()
    for r in doc['records']:
        h = r['original_hit']['source']
        writer.writerow({'object_id': r['object_id'], 'query_id': r['query_id'],
            'source_kind': h['source_kind'], 'source_id': h['source_id'], 'surface': h['raw_surface'],
            'start': h['raw_span'][0], 'end': h['raw_span'][1], 'original_definition': r['original_definition']['text'],
            'source_text': r['source_text'], 'query_text': r['query_text'],
            **{f: r['values'][f] for f in sorted(FIELDS)},
            **{f + '_provenance': r['field_provenance'][f] for f in sorted(FIELDS)},
            'assessment_status': r['assessment_status']})
    return {'reviews.json': encoded(doc), 'reviews.csv': output.getvalue().encode(),
            'preferences.md': render(doc, run)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, default=DEFAULT_RUN)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    base = args.run / 'resource-reviews-v1/hits'
    base.mkdir(parents=True, exist_ok=True)
    with (base / '.export.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        doc = build(args.run)
        data = artifacts(doc, args.run)
        snapshot_id = digest(data['reviews.json'])
        snapshot = base / 'snapshots' / snapshot_id
        pointer = {'schema_version': 'local-hit-current/v1', 'snapshot_id': snapshot_id,
                   'status': doc['status'], 'artifacts': {
                       name: {'path': str((snapshot / name).relative_to(args.run)), 'sha256': digest(raw)}
                       for name, raw in data.items()}}
        if args.check:
            assert read(base / 'current.json') == pointer
            for name, raw in data.items():
                assert (snapshot / name).read_bytes() == raw
            assert (base / 'preferences.md').read_bytes() == data['preferences.md']
        else:
            for name, raw in data.items():
                target = snapshot / name
                if target.exists():
                    assert target.read_bytes() == raw, 'Immutable snapshot differs'
                else:
                    write(target, raw)
            write(base / 'preferences.md', data['preferences.md'])
            write(base / 'current.json', encoded(pointer))
            write(args.run / 'hit_context.json', encoded({'schema_version': 'local-hit-context/v1',
                'current': ref(base / 'current.json', args.run), 'snapshot_id': snapshot_id,
                'online_confirmations_added': 0}))
        print(json.dumps({'snapshot_id': snapshot_id, 'status': doc['status']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
