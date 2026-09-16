#!/usr/bin/env python3
"""Export local paired-text relationship reviews with field-scoped provenance.

This writer has no session or network API. Existing online confirmations and
sentence/lexicon reviews are references, not new relationship adjudications.
"""
import argparse
import copy
import csv
import fcntl
import io
import json
from collections import Counter
from pathlib import Path

from export_definition_reviews import (
    ROOT, DEFAULT_RUN, CHOICES, normalize_values, read, encoded, digest, ref, verify, write,
)

FIELDS = ('topic_hate', 'topic_group', 'rule_hate', 'rule_group', 'lexicon_risk')
LABELS = dict(zip(FIELDS, ('hate 话题', 'group 话题', 'hate 规则', 'group 规则', '词典风险')))
FIELD_CHOICES = {f: CHOICES['lexicon_risk' if f == 'lexicon_risk' else 'relation'] for f in FIELDS}


def apply_event(active, history, event, origin):
    """Validate the entire event before applying any field; retain amendments."""
    scope = event['scope']
    assert scope and len(scope) == len(set(scope))
    assert set(scope) == set(event['values']) <= set(FIELDS)
    assert event['review_kind'] == 'human_with_ai'
    assert event['online_material_confirmation_added'] is False
    assert isinstance(event['rationale'], str) and event['rationale'].strip()
    amendments = event.get('amends', {})
    assert set(amendments) <= set(scope)
    updates = {}
    for field in scope:
        value = event['values'][field]
        assert value in FIELD_CHOICES[field], (field, value)
        previous = active.get(field)
        if previous:
            assert amendments.get(field) == previous['event_id'], 'Explicit amendment required'
            assert event.get('amendment_reason', '').strip(), 'Amendment reason required'
        else:
            assert field not in amendments
        updates[field] = {
            'value': value, 'event_id': event['event_id'], 'source': origin,
            'rationale_verbatim': event['rationale'], 'review_kind': 'human_with_ai',
        }
    for field, row in updates.items():
        active[field] = row
        history.setdefault(field, []).append(row)


def check_reply(event, questions, folder, run):
    reply_path = verify(event['source_reply_reference'], run)
    reply = read(reply_path)['replies'][event['source_reply_index']]
    qid = json.loads(reply['questionItemId'])
    assert len(qid) == 3 and qid[0] == 'request_user_input_async'
    assert isinstance(qid[1], str) and type(qid[2]) is int
    assert qid == event['question_item_id']
    if event.get('source_question_reference'):
        path = verify(event['source_question_reference'], run)
        assert path.parent == folder / 'consistency-rechecks'
        question = read(path)
        receipt_path = path.with_name(path.stem + '-request-receipt.json')
        primary = question['resolves_primary_question_index']
        assert primary in questions
        assert question['question_index'] == qid[2]
        assert set(question['object_ids']) <= set(questions[primary]['object_ids'])
        assert set(question['allowed_user_fields']) <= set(questions[primary]['allowed_user_fields'])
    else:
        path = folder / 'questions.json'
        receipt_path = folder / 'request-receipt.json'
        assert qid[2] in questions
        primary = qid[2]
        question = questions[primary]
    receipt = read(receipt_path)
    assert receipt['questions_sha256'] == digest(path.read_bytes())
    assert qid[1] == receipt['request_call_id'], 'Reply belongs to a different request'
    assert reply['question'] == question['question_text_verbatim']
    assert reply['answer'] == event['rationale']
    assert event['object_id'] in question['object_ids']
    assert set(event['scope']) <= set(question['allowed_user_fields'])
    return primary, [ref(p, run) for p in (reply_path, path, receipt_path)]


def resolved_questions(questions, active):
    # One answer about one field or one pair cannot close other fields/pairs.
    return {i for i, q in questions.items()
            if all(set(q['allowed_user_fields']) <= set(active.get(oid, {}))
                   for oid in q['object_ids'])}


def build(run):
    base = run / 'resource-reviews-v1/relations'
    manifest = read(run / 'input_manifest.json')
    sources = [ref(run / 'input_manifest.json', run)]
    all_inputs = {}
    for path in sorted((run / 'inputs').glob('resources-*.json')):
        assert digest(path.read_bytes()) == manifest['inputs'][path.name]
        package = read(path)
        for material in package['required']:
            if material['kind'] != 'relation':
                continue
            oid = material['id']
            did = material['source']['demo_id']
            assert oid not in all_inputs
            all_inputs[oid] = {
                'object_id': oid, 'query_id': package['query_id'], 'demo_id': did,
                'query_text': package['query'], 'demo_text': package['demos'][did],
                'original_relation': material, 'package': ref(path, run),
            }
        sources.append(ref(path, run))
    assert len(all_inputs) == manifest['counts']['relation']
    assert set(all_inputs) == {oid for oid in manifest['pending_ids'] if oid.startswith('relation:')}
    records, batches, all_events, prior_ids, preferences = {}, [], set(), set(), []
    for folder in sorted(base.glob('batch-*')):
        if not (folder / 'selection.json').exists():
            continue
        selection = read(folder / 'selection.json')
        selected = selection['selected_ids']
        assert len(selected) == len(set(selected)) == selection['object_count']
        assert set(selected) <= set(all_inputs) and not set(selected) & set(records)
        for name in ('preparation', 'inputs', 'initial_ai_drafts', 'questions',
                     'prior_human_reference', 'authorization'):
            sources.append(ref(verify(selection[name], run), run))
        preparation = read(verify(selection['preparation'], run))
        assert preparation['selected_ids'] == selected
        assert preparation['object_count'] == len(selected)
        for name in ('manifest', 'inputs', 'authorization', 'prior_human_reference'):
            sources.append(ref(verify(preparation[name], run), run))
        for reference in [*preparation['packages'], *preparation['starting_contexts']]:
            sources.append(ref(verify(reference, run), run))
        for name in ('definition_snapshot', 'hit_snapshot', 'sentence_snapshot'):
            for reference in preparation[name]['artifacts'].values():
                sources.append(ref(verify(reference, run), run))
        policy = preparation['policy_source']
        assert digest((ROOT / policy['path']).read_bytes()) == policy['sha256']
        assert read(folder / 'inputs.json')['records'] == [all_inputs[oid] for oid in selected]
        for prior in read(verify(selection['prior_human_reference'], run))['records']:
            if prior['kind'] == 'relation':
                assert prior['object_id'] not in all_inputs
                prior_ids.add(prior['object_id'])
        draft_doc = read(folder / 'ai-drafts.json')
        drafts = {r['object_id']: r for r in draft_doc['records']}
        assert len(drafts) == len(draft_doc['records']) and set(drafts) == set(selected)
        qdoc = read(folder / 'questions.json')
        questions = {q['question_index']: q for q in qdoc['records']}
        assert len(questions) == len(qdoc['records']) == selection['question_count']
        assert set(questions) == set(range(qdoc['question_count']))
        for q in questions.values():
            assert q['object_ids'] and set(q['object_ids']) <= set(selected)
            assert q['allowed_user_fields'] and set(q['allowed_user_fields']) <= set(FIELDS)
        active, history, followup_answers = {}, {}, {}
        for path in sorted((folder / 'discussions').glob('user-decisions-*.json')):
            doc = read(path)
            assert doc['selection_sha256'] == digest((folder / 'selection.json').read_bytes())
            sources.append(ref(path, run))
            for index, event in enumerate(doc['records']):
                oid = event['object_id']
                assert oid in selected and event['event_id'] not in all_events
                all_events.add(event['event_id'])
                assert event['source_sha256'] == digest(encoded(all_inputs[oid]['original_relation']['source']))
                _, refs = check_reply(event, questions, folder, run)
                sources.extend(refs)
                apply_event(active.setdefault(oid, {}), history.setdefault(oid, {}), event,
                            {**ref(path, run), 'record_index': index})
                if event.get('source_question_reference'):
                    key = event['source_question_reference']['path']
                    followup_answers.setdefault(key, {}).setdefault(oid, {}).update(event['values'])
        replied = resolved_questions(questions, active)
        pending_ids = {oid for i, q in questions.items() if i not in replied for oid in q['object_ids']}
        followups = []
        for path in sorted((folder / 'consistency-rechecks').glob('*-question.json')):
            q = read(path)
            primary = questions[q['resolves_primary_question_index']]
            assert set(q['object_ids']) <= set(primary['object_ids'])
            assert set(q['allowed_user_fields']) <= set(primary['allowed_user_fields'])
            key = str(path.relative_to(run))
            answered = bool(resolved_questions({0: q}, followup_answers.get(key, {})))
            if not answered:
                pending_ids.update(q['object_ids'])
            followups.append({'question': q, 'source': ref(path, run), 'answered': answered})
            sources.append(ref(path, run))
            receipt_path = path.with_name(path.stem + '-request-receipt.json')
            if receipt_path.exists():
                sources.append(ref(receipt_path, run))
        hit_doc = read(verify(preparation['hit_snapshot']['artifacts']['reviews.json'], run))
        hits = {r['object_id']: r for r in hit_doc['records']}
        for oid in selected:
            item = all_inputs[oid]
            source = item['original_relation']['source']
            draft = drafts[oid]
            assert draft['review_kind'] == 'ai_note' and draft['new_human_fields'] == []
            assert draft['online_material_confirmed'] is False
            assert draft['base_version'] == manifest['pending_versions'][oid]
            assert draft['source_sha256'] == digest(encoded(source))
            assert draft['query_id'] == item['query_id'] and draft['demo_id'] == item['demo_id']
            for field in ('query_text', 'demo_text'):
                assert draft[field + '_sha256'] == digest(item[field].encode())
            assert draft['hit_snapshot'] == preparation['hit_snapshot']['snapshot_id']
            expected_entries = set(source['introduced_entry_ids'])
            entries = draft['lexicon_assessment_ai']
            assert len(entries) == len(expected_entries)
            assert {e['entry_id'] for e in entries} == expected_entries
            for entry in entries:
                assert entry['source_edges'], 'Missing occurrence support'
                for edge in entry['source_edges']:
                    hit = hits[edge['hit_object_id']]
                    hit_source = hit['original_hit']['source']
                    assert hit['query_id'] == item['query_id']
                    assert hit_source['source_kind'] == 'demo' and hit_source['source_id'] == item['demo_id']
                    assert hit_source['entry_id'] == entry['entry_id']
                    assert edge['sense_id'] == hit_source['sense_id']
                    assert edge['raw_surface'] == hit_source['raw_surface']
                    assert edge['original_definition'] == hit['original_definition']['text']
                    for field in ('source_fit', 'query_fit'):
                        assert edge[field] == hit['values'][field]
                    assert edge['fit_field_provenance'] == hit['field_provenance']
            evidence_source = {'texts': {'query': item['query_text'], 'demo': item['demo_text']}}
            normalize_values('relation', draft['values'], evidence_source, required=True)
            values = copy.deepcopy(draft['values'])
            human = active.get(oid, {})
            for field, decision in human.items():
                values[field] = decision['value']
            normalize_values('relation', values, evidence_source, required=True)
            records[oid] = {
                **item, 'batch': folder.name, 'base_version': draft['base_version'], 'values': values,
                'field_provenance': {f: 'user_discussion' if f in human else 'ai_note' for f in FIELDS},
                'evidence_and_note_provenance': 'ai_note', 'ai_draft': draft,
                'human_fields': human, 'human_field_history': history.get(oid, {}),
                'assessment_status': 'awaiting_reply' if oid in pending_ids else 'user_participated' if human else 'ai_reviewed',
                'meaning_unresolved_fields': [f for f in FIELDS if values[f] == 'unclear'],
                'online_material_confirmed': False,
            }
        batches.append({
            'batch': folder.name, 'selected': len(selected), 'question_count': len(questions),
            'questions_answered': len(replied),
            'primary_questions_pending': len(questions) - len(replied),
            'followup_questions_pending': sum(not q['answered'] for q in followups),
            'questions_pending': len(questions) - len(replied) + sum(not q['answered'] for q in followups),
            'pending_question_indices': sorted(set(questions) - replied),
            'followup_questions': followups,
        })
        for name in ('selection.json', 'ai-drafts.json', 'questions.json', 'request-sent.json',
                     'request-receipt.json', 'post-decision-notes.json', 'calibration-application.json'):
            path = folder / name
            if path.exists():
                sources.append(ref(path, run))
                if name == 'calibration-application.json':
                    application = read(path)
                    verify(application['before_ai_drafts'], run)
                    assert application['after_ai_drafts_sha256'] == digest((folder / 'ai-drafts.json').read_bytes())
                    assert application['reviewed_object_ids'] == selected
                    assert application['reviewed_count'] == len(selected)
                    for change in application['records']:
                        assert change['object_id'] in selected
                        assert change['generalized_fields_remain_ai'] is True
                if name == 'post-decision-notes.json':
                    for note in read(path).get('case_summaries_ai', []):
                        assert note['summary_provenance'] == 'ai_summary_of_explicit_user_answers'
                        assert set(note['source_event_ids']) <= all_events
                        for source in note['source_event_references']:
                            event_doc = read(verify(source['source'], run))
                            assert source['event_id'] in {e['event_id'] for e in event_doc['records']}
                        preferences.append(note)
    rows = list(records.values())
    status = {
        'relations_total': len(all_inputs), 'relations_selected': len(rows),
        'relations_not_started': len(all_inputs) - len(rows),
        'relations_ai_only': sum(not r['human_fields'] for r in rows),
        'relations_user_participated': sum(bool(r['human_fields']) for r in rows),
        'explicit_human_fields': {f: sum(f in r['human_fields'] for r in rows) for f in FIELDS},
        'explicit_human_field_events': sum(len(h) for r in rows for h in r['human_field_history'].values()),
        'questions_pending': sum(b['questions_pending'] for b in batches),
        'relations_awaiting_reply': sum(r['assessment_status'] == 'awaiting_reply' for r in rows),
        'unresolved_fields': {f: sum(r['values'][f] == 'unclear' for r in rows) for f in FIELDS},
        'value_counts': {f: dict(sorted(Counter(r['values'][f] for r in rows).items())) for f in FIELDS},
        'prior_relationship_confirmations_referenced': len(prior_ids),
        'online_material_confirmations_added': 0,
    }
    return {
        'schema_version': 'local-relation-reviews/v1', 'status': status, 'batches': batches,
        'sources': list({(s['path'], s['sha256']): s for s in sources}.values()),
        'records': rows, 'case_summaries_ai': preferences,
        'limits': [
            '本地 AI 协同记录不等同在线人工确认；此前确认只作参照。',
            '每个实际回复仅确认所提取字段；AI 理由、证据及推广仍单列。',
            '词典风险是当前查询适配的判断，不证明模型已受影响或该示例独占词条来源。',
            '词典适配、定义质量、示例答案可靠性和关系字段分别记录。',
        ],
    }


def cell(value):
    return str(value).replace('|', '\\|').replace('\n', ' ')


def render(doc, run):
    s = doc['status']
    lines = ['# 示例与查询关系复核', '',
        f"已整理 {s['relations_selected']}/{s['relations_total']} 组；用户参与 {s['relations_user_participated']} 组，待回复 {s['questions_pending']} 个问题。", '',
        '“不明”是判断结果，与待回复分开计数。每列独立记录；同一标签不自动代表相关。', '',
        '## 可复查的裁决偏好', '',
        '| 个案／规则 | 当前归纳（AI） | 适用边界 | 来源 |', '| --- | --- | --- | --- |']
    for note in doc['case_summaries_ai']:
        lines.append('| ' + ' | '.join(cell(note.get(k, '')) for k in ('id', 'summary', 'boundary', 'source')) + ' |')
    if not doc['case_summaries_ai']:
        lines += ['| — | 等待首批校准回复 | 暂不将初稿推广为用户偏好 | — |']
    lines += ['', '## 逐组结果', '',
              '| 查询 | 示例 | hate 话题 | group 话题 | hate 规则 | group 规则 | 词典风险 | 人工字段 |',
              '| --- | --- | --- | --- | --- | --- | --- | --- |']
    for r in doc['records']:
        cells = [r['query_id'], r['demo_id'], *[FIELD_CHOICES[f][r['values'][f]] for f in FIELDS],
                 '、'.join(LABELS[f] for f in FIELDS if f in r['human_fields']) or '—']
        lines.append('| ' + ' | '.join(map(cell, cells)) + ' |')
    by_id = {r['object_id']: r for r in doc['records']}
    for batch in doc['batches']:
        folder = run / 'resource-reviews-v1/relations' / batch['batch']
        lines += ['', f"## {batch['batch']} 校准原问与实际回复", '']
        for q in read(folder / 'questions.json')['records']:
            lines += [f"### {q['question_index'] + 1} · {q['question_key']}", '', q['question_text_verbatim'], '']
            for oid in q['object_ids']:
                r = by_id[oid]
                for f in q['allowed_user_fields']:
                    human = r['human_fields'].get(f)
                    text = '待回复' if not human else f"{FIELD_CHOICES[f][human['value']]}；原答：{human['rationale_verbatim']}"
                    lines.append(f"- `{oid}` / {LABELS[f]}：{text}")
            lines.append('')
        for followup in batch['followup_questions']:
            q = followup['question']
            lines += ['### 一致性补充', '', q['question_text_verbatim'], '',
                      '状态：' + ('已回复' if followup['answered'] else '待回复'), '']
            for oid in q['object_ids']:
                for f in q['allowed_user_fields']:
                    for h in by_id[oid]['human_field_history'].get(f, []):
                        lines.append(f"- `{oid}` / {LABELS[f]}：{FIELD_CHOICES[f][h['value']]}；原答：{h['rationale_verbatim']}")
            lines.append('')
    return ('\n'.join(lines) + '\n').encode()


def artifacts(doc, run):
    output = io.StringIO()
    fields = ['object_id', 'query_id', 'demo_id', 'query_text', 'demo_text', *FIELDS,
              *[f + '_provenance' for f in FIELDS], 'assessment_status', 'note_ai']
    writer = csv.DictWriter(output, fieldnames=fields)
    writer.writeheader()
    for r in doc['records']:
        writer.writerow({
            **{f: r[f] for f in ('object_id', 'query_id', 'demo_id', 'query_text', 'demo_text', 'assessment_status')},
            **{f: r['values'][f] for f in FIELDS},
            **{f + '_provenance': r['field_provenance'][f] for f in FIELDS},
            'note_ai': r['values']['note'],
        })
    return {'reviews.json': encoded(doc), 'reviews.csv': output.getvalue().encode(),
            'preferences.md': render(doc, run)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, default=DEFAULT_RUN)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    base = args.run / 'resource-reviews-v1/relations'
    base.mkdir(parents=True, exist_ok=True)
    with (base / '.export.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        doc = build(args.run)
        data = artifacts(doc, args.run)
        snapshot_id = digest(data['reviews.json'])
        snapshot = base / 'snapshots' / snapshot_id
        pointer = {
            'schema_version': 'local-relation-current/v1', 'snapshot_id': snapshot_id,
            'status': doc['status'], 'artifacts': {
                name: {'path': str((snapshot / name).relative_to(args.run)), 'sha256': digest(raw)}
                for name, raw in data.items()},
        }
        context = {'schema_version': 'local-relation-context/v1',
                   'current': {'path': str((base / 'current.json').relative_to(args.run)),
                               'sha256': digest(encoded(pointer))},
                   'snapshot_id': snapshot_id, 'online_confirmations_added': 0}
        if args.check:
            assert read(base / 'current.json') == pointer
            for name, raw in data.items():
                assert (snapshot / name).read_bytes() == raw
            assert (base / 'preferences.md').read_bytes() == data['preferences.md']
            assert read(args.run / 'relation_context.json') == context
        else:
            for name, raw in data.items():
                target = snapshot / name
                if target.exists():
                    assert target.read_bytes() == raw, 'Immutable snapshot differs'
                else:
                    write(target, raw)
            write(base / 'preferences.md', data['preferences.md'])
            write(base / 'current.json', encoded(pointer))
            write(args.run / 'relation_context.json', encoded(context))
        print(json.dumps({'snapshot_id': snapshot_id, 'status': doc['status']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
