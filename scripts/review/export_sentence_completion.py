#!/usr/bin/env python3
"""Export the local sentence closeout without changing frozen prior decisions.

Only field-scoped reply records count as human decisions. AI assessments and
rule-derived values remain separate, including explicitly reopened old cases.
"""
import argparse
import copy
import csv
import hashlib
import io
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN = ROOT / 'exps/causal_context/general_model_evidence_applicability_v1/ai_reviews/v2-reannotation-20260911'
FIELDS = ('hate', 'group', 'attack_severity')
GROUPS = {'Racism', 'Region', 'Sexism', 'LGBTQ', 'others'}
FORMS = {'derogation', 'personal_insult', 'group_stigmatization', 'dehumanization',
         'exclusion_or_rights_denial', 'violence_endorsement', 'violence_threat_or_call'}


def read(path):
    return json.loads(path.read_bytes())


def digest(data):
    return hashlib.sha256(data).hexdigest()


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, indent=2) + '\n').encode()


def ref(path, run):
    return {'path': str(path.relative_to(run)), 'sha256': digest(path.read_bytes())}


def verify(source, run):
    path = run / source['path']
    assert path.resolve().is_relative_to(run.resolve()), source
    assert digest(path.read_bytes()) == source['sha256'], f'Source changed: {path}'


def validate_value(field, value):
    if field == 'attack_severity':
        assert value is None or type(value) is int and 0 <= value <= 4, value
    elif field == 'group':
        assert value is None or isinstance(value, list) and len(value) == len(set(value)) and set(value) <= GROUPS, value
    else:
        assert value in ('hate', 'non-hate', None), value


def mapped_hate(score):
    return None if score is None else 'hate' if score > 0 else 'non-hate'


def display(v):
    return '待定' if v is None else '、'.join(v) or '[]' if isinstance(v, list) else str(v)


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_bytes(data)
    tmp.replace(path)


def build(run):
    out = run / 'sentence-completion-v1'
    selection = read(out / 'selection.json')
    for key in ('baseline', 'initial_ai_drafts', 'input_manifest', 'prior_group_impact', 'previous_round'):
        verify(selection[key], run)
    for source in selection['policy_sources']:
        verify(source, run)
    for source in selection['calibration_snapshot']['artifacts'].values():
        verify(source, run)
    baseline_doc = read(run / selection['baseline']['path'])
    for source in baseline_doc['sources']:
        verify(source, run)
    baseline = {r['object_id']: r for r in baseline_doc['records']}
    assert len(baseline) == len(baseline_doc['records']) == 270
    selected = selection['selected_ids']
    assert len(set(selected)) == len(selected) == selection['object_count'] == 71
    assert set(selected) == set(selection['severity_missing_ids'] + selection['severity_reopened_ids'] + selection['group_priority_recheck_ids'])
    assert len(selection['severity_missing_ids']) == 60
    assert set(selection['severity_reopened_ids']) == {'demo:1509', 'demo:2004'}
    task_scopes = copy.deepcopy(selection['task_scopes'])
    group_policy = selection['task_policies']['group']
    followup = None
    followup_path = out / 'policy-followup.json'
    if followup_path.exists():
        followup = read(followup_path)
        for key in ('context', 'confirmation', 'initial_selection'):
            verify(followup[key], run)
        assert followup['initial_selection'] == ref(out/'selection.json', run)
        context = read(run/followup['context']['path'])
        document = context['document']
        assert digest((ROOT/document['path']).read_bytes()) == document['sha256']
        assert context['policy'] == followup['group_policy']
        extra = followup['additional_selected_ids']
        assert not (set(extra) & set(selected)) and len(set(extra)) == len(extra)
        assert set(extra) <= set(baseline)
        assert set(extra) == set(followup['additional_task_scopes'])
        assert all(scope == ['group'] for scope in followup['additional_task_scopes'].values())
        selected = selected + extra
        task_scopes.update(followup['additional_task_scopes'])
        group_policy = followup['group_policy']
    # Explicit later case rechecks extend the old selection without rewriting
    # its frozen bytes or the policy-followup receipt.
    case_followups = []
    extra_drafts = []
    for path in sorted((out / 'explicit-followups').glob('*/selection.json')):
        extension = read(path)
        for key in ('authorization_reply', 'question', 'prior_current', 'prior_result', 'ai_drafts'):
            verify(extension[key], run)
        prior = read(run / extension['prior_current']['path'])
        assert extension['prior_result'] == prior['artifacts']['current-sentences.json']
        extra = extension['selected_ids']
        assert len(set(extra)) == len(extra) and not set(extra) & set(selected)
        assert set(extra) <= set(baseline)
        assert set(extension['task_scopes']) == set(extra)
        assert all(scope and len(scope) == len(set(scope)) and set(scope) <= {'group', 'attack_severity'}
                   for scope in extension['task_scopes'].values())
        for scope in extension['task_scopes'].values():
            for field in scope:
                expected_policy = group_policy if field == 'group' else selection['task_policies'][field]
                assert extension['task_policies'][field] == expected_policy
        selected = selected + extra
        task_scopes.update(extension['task_scopes'])
        extra_drafts.extend(read(run / extension['ai_drafts']['path'])['records'])
        case_followups.append((path, extension))
    drafts_doc = read(out / 'ai-drafts.json')
    drafts_doc['records'] = drafts_doc['records'] + extra_drafts
    drafts = {r['object_id']: r for r in drafts_doc['records']}
    assert len(drafts) == len(drafts_doc['records']) and set(drafts) == set(selected)
    source_refs = [ref(out / name, run) for name in ('selection.json', 'baseline.json', 'ai-drafts.json')]
    if followup:
        source_refs.append(ref(followup_path, run))
    for path, extension in case_followups:
        source_refs.append(ref(path, run))
        source_refs.extend(extension[key] for key in ('authorization_reply', 'question', 'prior_current', 'prior_result', 'ai_drafts'))
    decisions = {}
    for path in sorted((out / 'discussions').glob('user-decisions-[0-9][0-9].json')):
        doc = read(path)
        assert doc['selection_sha256'] == digest((out / 'selection.json').read_bytes())
        source_refs.append(ref(path, run))
        for index, event in enumerate(doc['records']):
            oid = event['object_id']
            assert oid in selected
            assert event['text_sha256'] == baseline[oid]['text_sha256']
            assert set(event['scope']) == set(event['values']) <= set(FIELDS)
            assert event['review_kind'] == 'human_with_ai'
            assert event['online_material_confirmation_added'] is False
            assert set(event['scope']) <= set(task_scopes[oid])
            verify(event['source_reply_reference'], run)
            replies = read(run / event['source_reply_reference']['path'])['replies']
            reply = replies[event['source_reply_index']]
            assert reply['answer'] == event['rationale'], oid
            assert reply['questionItemId'] == json.dumps(event['question_item_id'], ensure_ascii=False, separators=(',', ':')), oid
            for field, value in event['values'].items():
                validate_value(field, value)
                assert (oid, field) not in decisions, f'Explicit amendment required for duplicate: {oid}/{field}'
                policy = event.get('task_policies', {}).get(field) or doc['task_policies'][field]
                allowed = [selection['task_policies'][field]]
                if field == 'group':
                    allowed.append(group_policy)
                assert policy in allowed
                decisions[oid, field] = {'value': value, 'source': ref(path, run),
                                         'source_record_index': index, 'event': event, 'task_policy': policy}

    records = []
    for oid, original in baseline.items():
        r = copy.deepcopy(original)
        original_sources = r.pop('source_records', {})
        r['historical_source_records'] = original_sources
        r['current_field_sources'] = {
            field: {'source': original_sources.get('severity' if field == 'attack_severity' else 'label', selection['baseline']),
                    'object_id': oid} for field in FIELDS}
        r['field_history'] = {}
        r['field_status'] = {f: 'retained' for f in FIELDS}
        r['human_decisions_in_completion'] = {}
        if oid in drafts:
            draft = drafts[oid]
            assert draft['text_sha256'] == r['text_sha256'] == digest(r['text'].encode())
            assert set(draft['scope']) == set(draft['values']) == set(draft['status'])
            assert set(draft['scope']) == set(task_scopes[oid])
            assert draft['assessment_provenance'] == 'ai_note'
            assert all(q in r['text'] for q in draft['quotes']), oid
            if 'attack_severity' in draft['scope']:
                score = draft['values']['attack_severity']
                assert set(draft['attack_forms']) <= FORMS
                if draft['status']['attack_severity'] == 'ai_scored':
                    assert score is not None
                    assert bool(draft['attack_forms']) == (score > 0)
                elif draft['status']['attack_severity'] == 'needs_discussion':
                    assert score is None and len(draft['candidate_scores']) >= 2
                    assert draft['proposal']['attack_severity'] in draft['candidate_scores']
                else:
                    raise AssertionError(draft['status'])
            r['completion_ai_assessment'] = copy.deepcopy(draft)
            for field, value in draft['values'].items():
                validate_value(field, value)
                r['field_history'][field] = {'value': original['values'].get(field),
                                            'field_provenance': original['field_provenance'].get(field),
                                            'task_policy': original['task_policies'].get(field),
                                            'source': selection['baseline']}
                pending = draft['status'][field] == 'needs_discussion'
                r['values'][field] = None if pending else value
                r['field_provenance'][field] = 'pending_discussion' if pending else 'ai_note'
                r['current_field_sources'][field] = {'source': ref(out/'ai-drafts.json', run), 'object_id': oid, 'field': field}
                r['field_status'][field] = draft['status'][field]
                r['task_policies'][field] = group_policy if field == 'group' else selection['task_policies'][field]
                if field == 'group' and draft.get('reopen_reason'):
                    r['group_reopen'] = {'reason': draft['reopen_reason'], 'authorization': draft.get('reopen_authorization', selection['authorization_text']),
                                          'previous_decision_retained': True, 'human_reconfirmation_inferred': False}
            for field in FIELDS:
                decision = decisions.get((oid, field))
                if decision is None:
                    continue
                r['field_history'].setdefault(field, {'value': original['values'].get(field),
                    'field_provenance': original['field_provenance'].get(field),
                    'task_policy': original['task_policies'].get(field), 'source': selection['baseline']})
                r['values'][field] = decision['value']
                r['field_provenance'][field] = 'user_discussion'
                r['field_status'][field] = 'human_adjudicated'
                r['task_policies'][field] = decision['task_policy']
                r['human_decisions_in_completion'][field] = decision
                r['current_field_sources'][field] = {'source': decision['source'], 'source_record_index': decision['source_record_index'], 'field': field}
            if draft.get('retained_task_acknowledgment'):
                r['retained_task_acknowledgment'] = draft['retained_task_acknowledgment']
                r['severity_rationale_context'] = 'Historical scoring rationale; lexical interpretation was explicitly rechecked later.'
            if 'attack_severity' in draft['scope']:
                r['historical_severity_assessment'] = r.get('severity_assessment')
                r['severity_assessment'] = {'values': {'attack_severity': r['values']['attack_severity'],
                    'attack_forms': draft['attack_forms'], 'rationale': draft['rationale'],
                    'evidence': [{'source': 'text', 'start': r['text'].index(q), 'end': r['text'].index(q)+len(q), 'text': q} for q in draft['quotes']]},
                    'assessment_status': 'scored' if r['values']['attack_severity'] is not None else 'needs_discussion',
                    'supporting_fields_provenance': 'ai_note', 'human_score_decision': r['human_decisions_in_completion'].get('attack_severity')}
        score = r['values']['attack_severity']
        expected = mapped_hate(score)
        if r['values'].get('hate') != expected:
            r['field_history'].setdefault('hate', {'value': r['values'].get('hate'),
                'field_provenance': r['field_provenance'].get('hate'),
                'task_policy': r['task_policies'].get('hate'), 'source': selection['baseline']})
        if (oid, 'hate') in decisions:
            assert r['values']['hate'] == expected, f'Explicit hate conflicts with active mapping: {oid}'
        else:
            r['values']['hate'] = expected
            score_source = 'human' if r['field_provenance']['attack_severity'] == 'user_discussion' else 'ai'
            r['field_provenance']['hate'] = 'pending_severity' if score is None else f'rule_mapping_from_{score_source}_score'
            r['field_status']['hate'] = 'pending_severity' if score is None else 'rule_mapped'
            r['current_field_sources']['hate'] = {'mapping_context': next(s for s in selection['policy_sources'] if s['path']=='hate_mapping_context.json'),
                                                 'severity_source': r['current_field_sources']['attack_severity']}
        r['task_policies']['hate'] = selection['task_policies']['hate']
        # Do not retain baseline agreement flags after a later field-scoped amendment.
        historical = r.get('historical_hate', {})
        r['comparison_with_historical_hate'] = 'unmapped' if expected is None else 'agrees' if historical.get('value') == expected else 'differs'
        r['compatible_explicit_human_hate'] = bool(historical.get('explicit_human_decision') and historical.get('value') == expected)
        r['online_material_confirmed'] = False
        for field in FIELDS:
            validate_value(field, r['values'][field])
        if oid not in selected:
            assert r['values'] == original['values'], f'Unselected case changed: {oid}'
        records.append(r)

    by_id = {r['object_id']: r for r in records}
    pending = {f: [r['object_id'] for r in records if r['values'][f] is None] for f in FIELDS}
    # Includes pending task status even if the user expressly returns an unresolved value.
    open_scopes = [{'object_id': oid, 'field': f} for oid in selected for f in drafts[oid]['scope']
                   if by_id[oid]['values'][f] is None or by_id[oid]['field_status'][f] == 'needs_discussion']
    changed = {f: [r['object_id'] for r in records if r['values'][f] != baseline[r['object_id']]['values'][f]] for f in FIELDS}
    summary = {'object_count': 270, 'initial_completion_object_count': 71, 'completion_object_count': len(selected),
               'severity_numeric_count': sum(r['values']['attack_severity'] is not None for r in records),
               'severity_source_counts': dict(Counter(r['field_provenance']['attack_severity'] for r in records)),
               'severity_counts': dict(Counter(str(r['values']['attack_severity']) for r in records)),
               'hate_counts': dict(Counter(str(r['values']['hate']) for r in records)),
               'pending_ids': pending, 'open_scopes': open_scopes, 'complete': not open_scopes,
               'changed_from_baseline_ids': changed,
               'new_human_field_event_counts': dict(Counter(f for oid, f in decisions)),
               'group_priority_rechecks_completed': sum(by_id[oid]['values']['group'] is not None for oid in selection['group_priority_recheck_ids']),
               'additional_policy_rechecks_completed': sum(by_id[oid]['values']['group'] is not None for oid in (followup or {}).get('additional_group_recheck_ids', [])),
               'explicit_case_rechecks_completed': sum(all(by_id[oid]['field_status'][f] == 'human_adjudicated' for f in e['task_scopes'][oid]) for _, e in case_followups for oid in e['selected_ids']),
               'human_decisions_added_by_ai_or_mapping': 0, 'online_material_confirmations_added': 0}
    result = {'schema_version': 'evidence-sentence-completion-current/v1',
              'scope': 'Local current sentence view; preserves prior layers. Not an online human confirmation or independent annotation.',
              'sources': source_refs, 'calibration_snapshot': selection['calibration_snapshot'],
              'summary': summary, 'records': records}
    round_view = {'schema_version': 'evidence-sentence-completion-round/v1', 'review_round': 9,
                  'summary': summary, 'records': [by_id[oid] for oid in selected]}
    artifacts = {'current-sentences.json': encoded(result), 'review.json': encoded(round_view)}
    for name, rows in [('current-sentences.csv', records), ('review.csv', round_view['records'])]:
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(['object_id', 'text', 'hate', 'group', 'attack_severity', 'field_provenance', 'field_status', 'history_json'])
        for r in rows:
            w.writerow([r['object_id'], r['text'], r['values']['hate'], json.dumps(r['values']['group'], ensure_ascii=False),
                        r['values']['attack_severity'], json.dumps(r['field_provenance'], ensure_ascii=False),
                        json.dumps(r['field_status'], ensure_ascii=False), json.dumps(r['field_history'], ensure_ascii=False)])
        artifacts[name] = buf.getvalue().encode()
    lines = ['# 句子部分收尾复核', '',
             f"共270句；本轮原涉及71句，政策及个案补充后共{len(selected)}句。已有数值严重度 {summary['severity_numeric_count']}/270；原9条历史类别复核已处理 {summary['group_priority_rechecks_completed']}/9，政策补充复核已处理 {summary['additional_policy_rechecks_completed']} 条，后续个案复核已处理 {summary['explicit_case_rechecks_completed']} 条。", '',
             '当前表按严重度映射 hate。字段分别标明用户裁决、AI补标和规则映射；未回答的建议不视为确认。旧人工裁决及其政策保留在历史层。', '',
             f"[完整270句 JSON]({out / 'current-sentences.json'}) · [完整270句 CSV]({out / 'current-sentences.csv'}) · [本轮 JSON]({out / 'review.json'})", '',
             '## 待讨论', '']
    for item in open_scopes:
        oid, field = item['object_id'], item['field']
        lines += [f"- #{oid.split(':')[1]} · {field}：{drafts[oid]['rationale']}"]
    if not open_scopes:
        lines += ['本轮疑问项已处理完毕。']
    lines += ['', '## 本轮记录', '']
    for r in round_view['records']:
        lines += [f"### #{r['object_id'].split(':')[1]}", '', '> ' + r['text'].replace('\n', '\n> '), '',
                  ' / '.join(display(r['values'][f]) for f in ('hate','attack_severity','group')), '',
                  '字段来源：' + '；'.join(f"{f}={r['field_provenance'][f]}" for f in FIELDS), '',
                  'AI依据：' + r['completion_ai_assessment']['rationale'], '']
        for field, event in r['human_decisions_in_completion'].items():
            lines += [f"- 用户 {field}：{display(event['value'])}；原答：{event['event']['rationale']}"]
        if r['field_history']:
            lines += ['', '前值（历史保留）：' + '；'.join(f"{f}={display(h['value'])}" for f,h in r['field_history'].items()), '']
    artifacts['review.md'] = ('\n'.join(lines) + '\n').encode()
    version = digest(encoded({name: digest(data) for name, data in sorted(artifacts.items())}))
    return version, artifacts, summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, default=DEFAULT_RUN)
    p.add_argument('--check', action='store_true')
    args = p.parse_args()
    version, artifacts, summary = build(args.run)
    out = args.run / 'sentence-completion-v1'
    for name, data in artifacts.items():
        frozen = out / 'snapshots' / version / name
        if args.check or frozen.exists():
            assert frozen.read_bytes() == data, f'Frozen output differs: {frozen}'
        else:
            write(frozen, data)
        current = out / name
        if args.check:
            assert current.read_bytes() == data, f'Refresh required: {current}'
        else:
            write(current, data)
    pointer = encoded({'snapshot_id': version, 'complete': summary['complete'],
        'artifacts': {name: {'path': str((out/'snapshots'/version/name).relative_to(args.run)), 'sha256': digest(data)} for name,data in artifacts.items()}})
    if args.check:
        assert (out/'current.json').read_bytes() == pointer
    else:
        write(out/'current.json', pointer)
    print(json.dumps({'snapshot_id': version, **summary}, ensure_ascii=False))


if __name__ == '__main__':
    main()
