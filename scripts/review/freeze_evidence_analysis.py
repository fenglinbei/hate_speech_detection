#!/usr/bin/env python3
"""Freeze read-only evidence references; never change a human decision or run a model."""
from __future__ import annotations
import argparse
from collections import Counter
import csv
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
from tools.general_model_paired_review_ui.evidence_store import EvidenceReviewStore
from tools.general_model_paired_review_ui.evidence_finalization import current_final

BASE = ROOT / 'exps/causal_context/general_model_evidence_applicability_v1'

def sha(raw):
    return hashlib.sha256(raw).hexdigest()

def encoded(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + '\n').encode()

def require(condition, message):
    if not condition:
        raise ValueError(message)

def reference_gate(native, case_label, material_label, reconciled=False):
    reasons = []
    if not native['reference_eligible']:
        reasons.append('native_reference_unavailable')
    if case_label != material_label and not reconciled:
        reasons.append('case_material_difference_pending_confirmation')
    return not reasons, reasons

def build(session_path, expected_sha, reconciliation_path=None):
    raw = session_path.read_bytes()
    require(sha(raw) == expected_sha, 'session byte identity mismatch')
    s = json.loads(raw)
    reconciliation = json.loads(reconciliation_path.read_bytes()) if reconciliation_path else {}
    if reconciliation:
        require(reconciliation.get('session_sha256') == expected_sha and
                reconciliation.get('reviewer_id') == s['reviewer_id'] and
                reconciliation.get('authorization_text') and
                reconciliation.get('source') == 'explicit_user_reply', 'unbound reconciliation')
    approved = set(reconciliation.get('case_task_keys', []))
    bundle_path = BASE / 'bundle/evidence_bundle.json'
    policy_path = BASE / 'policies/group-scope-v2/policy_amendment.json'
    # The production reader validates only an isolated temporary copy. No writer is started.
    with tempfile.TemporaryDirectory(prefix='evidence-reference-reader-') as tmp:
        copy = Path(tmp) / 'session.json'
        copy.write_bytes(raw)
        store = EvidenceReviewStore(bundle_path=bundle_path, policy_path=policy_path,
                                    session_path=copy, reviewer_id=s['reviewer_id'])
        export = store.snapshot(s['revision'])
        require(copy.read_bytes() == raw, 'reader changed session')
        require(len(store.order) == 32 and len(store.objects) == 1072, 'unexpected review scope')
        require(all(store.cases[k]['comparison']['selection']['split'] == 'discovery' for k in store.order), 'non-discovery input')
        require(all(r['status'] == 'confirmed' for r in s['records'].values()), 'case review incomplete')
        require(all(r['status'] == 'confirmed' for r in s['objects'].values()), 'material review incomplete')
        require(not any(export['object_staleness'].values()), 'stale material dependencies')
        require(not any(v['stale'] for v in export['case_eligibility'].values()), 'stale case snapshots')
        materials = []
        for oid, obj in store.objects.items():
            row = s['objects'][oid]
            materials.append({'record_id': oid, 'kind': obj['kind'], 'source': obj['source'],
                              'source_version': obj['version'], 'review': row,
                              'native_task_eligibility': export['object_eligibility'].get(oid),
                              'stale': export['object_staleness'][oid]})
        overlay, adjudications, eligibility, differences = [], [], [], []
        for key in store.order:
            case, r = store.cases[key], s['records'][key]
            oid = case['query_object_id']
            obj, q = store.objects[oid], s['objects'][oid]
            text = obj['source']['text']
            require(sha(text.encode()) == obj['source']['text_sha256'], 'query text hash mismatch')
            for task in ('hate', 'group'):
                rid = key + ':' + task
                label, material = r['assessment'][task], q['values'][task]
                native = export['case_eligibility'][key]['tasks'][task]
                usable, reasons = reference_gate(native, label, material, rid in approved)
                final = current_final(q)
                material_policy = (final or {}).get('task_policies', {}).get(task) or q.get('task_reviews', {}).get(task, {}).get('policy')
                original = store._gold(key)[task]
                if label != material:
                    differences.append({'query_id': key, 'task': task, 'text': text,
                                        'material_label': material, 'case_label': label,
                                        'material_confirmed_at': q['confirmed_at'], 'case_confirmed_at': r['confirmed_at'],
                                        'resolution': 'explicit_case_precedence' if rid in approved else 'pending_confirmation'})
                common = {'record_id': rid, 'query_id': key, 'task': task,
                          'text_sha256': sha(text.encode()), 'source_manifest_sha256': store.bundle['source_identity']['paired_manifest_sha256'],
                          'session_sha256': expected_sha, 'session_revision': s['revision'],
                          'reviewer_id': s['reviewer_id'], 'review_kind': r.get('review_kind'),
                          'adjudication_mode': r.get('adjudication_mode', 'single_review'),
                          'confirmed_at': native['confirmed_at'], 'native_case_policy': native['policy'],
                          'query_material_policy': material_policy,
                          'policy_provenance_note': 'Case policy and material policy are distinct source receipts, not silently harmonized.',
                          'material_snapshot_sha256': r['material_snapshots'][-1]['sha256'],
                          'prior_exposure': r['prior_exposure']}
                overlay.append({**common, 'original_label': original, 'adjudicated_label': label,
                                'query_material_label': material, 'original_status': r['assessment'][task + '_original_status'],
                                'label_changed': original != label, 'analysis_reference_eligible': usable})
                adjudications.append({**common, 'label': label, 'status': r['status'],
                                      'case_assessment': r['assessment'], 'query_review_record_id': oid,
                                      'query_reason': q['values'].get(task + '_reason'),
                                      'query_reason_scope': 'material judgment only; not an invented case revision rationale',
                                      'query_evidence': q['values'].get('evidence'),
                                      'case_task_review': r.get('task_reviews', {}).get(task),
                                      'reconciliation': reconciliation if rid in approved else None})
                eligibility.append({**common, 'native_eligibility': native,
                                    'reference_analysis_eligible': usable, 'unavailable_reasons': reasons,
                                    'user_selected_use': r['assessment'][task + '_use'],
                                    'original_bucket': case['comparison']['selection']['primary_bucket'],
                                    'reviewed_behavior_type': None, 'reviewed_behavior_type_status': 'awaiting_dual_reference_evaluation',
                                    'input_control_eligible': False, 'input_control_pending': ['specific_intervention_and_hypothesis_not_frozen', 'alternative_explanation_not_frozen'],
                                    'error_attribution_certified': False,
                                    'error_attribution_note': 'Preserve human original_status; label differences alone do not separate source errors from policy changes.',
                                    'explanation_choice': r['assessment']['explanation_choice']})
        require(approved <= {d['query_id'] + ':' + d['task'] for d in differences}, 'reconciliation outside difference scope')
        docs = [{'policy': {'version': store.bundle_policy['version'], 'sha256': store.bundle_policy['sha256']}, 'text': store.bundle_policy['text']},
                {'active_policy': store.policy}]
        docs += [doc for batch in s.get('finalizations', {}).values() for doc in batch.get('policy_documents', [])]
        summary = {'cases_confirmed': 32, 'material_objects_confirmed': len(materials),
                   'material_counts': dict(Counter(m['kind'] for m in materials)),
                   'native_reference_tasks': sum(e['native_eligibility']['reference_eligible'] for e in eligibility),
                   'analysis_reference_tasks': sum(e['reference_analysis_eligible'] for e in eligibility),
                   'unresolved_differences': sum(d['resolution'] == 'pending_confirmation' for d in differences),
                   'case_material_differences': differences,
                   'original_status_counts': {t: dict(Counter(o['original_status'] for o in overlay if o['task'] == t)) for t in ('hate','group')},
                   'label_change_counts': {t: sum(o['label_changed'] for o in overlay if o['task'] == t) for t in ('hate','group')},
                   'all_case_notes_empty': all(not r['assessment']['note'] for r in s['records'].values()),
                   'explanations': dict(Counter(r['assessment']['explanation_choice'] for r in s['records'].values())),
                   'input_control_tasks_ready': 0, 'stale_objects': 0, 'stale_cases': 0,
                   'model_forward_executed': False, 'dual_reference_evaluation_executed': False,
                   'authoritative_records_modified': False}
    require(session_path.read_bytes() == raw, 'source changed during freeze')
    def lines(rows):
        return ''.join(json.dumps(r,ensure_ascii=False,sort_keys=True) + '\n' for r in rows).encode()
    files = {'gold_overlay.jsonl': lines(overlay), 'adjudications.jsonl': lines(adjudications),
             'eligibility.jsonl': lines(eligibility), 'material_reviews.jsonl': lines(materials),
             'policy_provenance.json': encoded(docs), 'audit.json': encoded(summary),
             'finalizations.json': encoded(s.get('finalizations', {}))}
    out=io.StringIO(); writer=csv.writer(out)
    writer.writerow(['query_id','task','original_label','reviewed_label','material_label','original_status','reference_eligible','pending'])
    for o,e in zip(overlay,eligibility):
        writer.writerow([o['query_id'],o['task'],json.dumps(o['original_label'],ensure_ascii=False),json.dumps(o['adjudicated_label'],ensure_ascii=False),json.dumps(o['query_material_label'],ensure_ascii=False),o['original_status'],e['reference_analysis_eligible'],';'.join(e['unavailable_reasons'])])
    files['reference_summary.csv']=out.getvalue().encode('utf-8-sig')
    code_paths=[Path(__file__),ROOT/'tools/general_model_paired_review_ui/evidence_store.py',ROOT/'tools/general_model_paired_review_ui/evidence_schema.py',ROOT/'tools/general_model_paired_review_ui/evidence_policy.py',ROOT/'tools/general_model_paired_review_ui/evidence_finalization.py',ROOT/'src/build_lex/annotated_lexicon_repair.py']
    manifest={'schema_version':'evidence-analysis-reference-freeze/v1', 'status':'frozen',
              'session_sha256':expected_sha, 'session_revision':s['revision'],
              'authoritative_source':'digitalocean-sgp:/var/lib/hsd-general-model-paired-review/evidence-applicability-v1/session.json',
              'bundle_sha256':sha(bundle_path.read_bytes()),'active_policy_sha256':sha(policy_path.read_bytes()),
              'reconciliation_sha256':sha(reconciliation_path.read_bytes()) if reconciliation_path else None,
              'execution_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
              'code_sha256':{str(p.relative_to(ROOT)):sha(p.read_bytes()) for p in code_paths},
              'artifacts':{k:sha(v) for k,v in files.items()}, 'counts':{k:v for k,v in summary.items() if isinstance(v,int)},
              'limits':['Single reviewer, AI-assisted; prior exposure retained.', 'Original status retained verbatim, not a certified source-error rate.', 'No input-control or mechanism gate granted.', 'Private source and review details stay in ignored reviews tree.']}
    files['manifest.json']=encoded(manifest)
    return files

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--session',type=Path,required=True);p.add_argument('--expected-sha256',required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--reconciliation',type=Path)
    p.add_argument('--check',action='store_true')
    a=p.parse_args();target=a.output.resolve()
    require(target.is_relative_to(BASE/'reviews') and target != BASE/'reviews','output must be under private reviews tree')
    files=build(a.session,a.expected_sha256,a.reconciliation)
    if a.check:
        require(set(x.name for x in target.iterdir()) == set(files),'freeze inventory mismatch')
        for name,payload in files.items():require((target/name).read_bytes()==payload,'rebuild mismatch: '+name)
        print('Frozen references reproduce exactly.')
    else:
        require(not target.exists(),'refusing to overwrite a freeze')
        target.parent.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='.freeze-',dir=target.parent) as tmp:
            for name,payload in files.items():
                f=Path(tmp)/name;f.write_bytes(payload);f.chmod(0o600)
            Path(tmp).rename(target)
        print(str(target))
if __name__=='__main__':main()
