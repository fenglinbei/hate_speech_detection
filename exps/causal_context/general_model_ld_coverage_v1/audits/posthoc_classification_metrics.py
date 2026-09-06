"""CPU-only post-hoc classification report; requires preserved local run inputs."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / 'src'))
from diagnostics.general_model_tasks import evaluate_predictions, GROUP_LABELS
from sklearn.metrics import accuracy_score, f1_score

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', type=Path, required=True)
OUT = parser.parse_args().output
OUT.mkdir(parents=True, exist_ok=True)
BASE = ROOT / 'exps/causal_context/general_model_ld_coverage_v1'
def read(p):
    return json.loads(p.read_text())
def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''):
            h.update(b)
    return h.hexdigest()

ref = read(BASE / 'plan_ref.json')
plan_path = BASE / 'plans' / ref['plan_id'] / 'plan.json'
assert sha(plan_path) == ref['plan_sha256']
plan = read(plan_path)
package = ROOT / 'exps/causal_context/general_model_ld_v2/packages' / plan['package_id']
assert sha(package / 'manifest.json') == plan['package_manifest_sha256']
entry = next(r for r in read(package / 'manifest.json')['files'] if r['path'] == 'queries.dev.jsonl')
assert sha(package / 'queries.dev.jsonl') == entry['sha256']
gold = {str(r['id']): r['projection'] for r in map(json.loads, (package / 'queries.dev.jsonl').read_text().splitlines())}
conditions = ['C0', 'CLq', 'CLnew', 'CD', 'CLqD', 'CLDnew', 'PLnew', 'PD']
rows = {(c,t): [] for c in conditions for t in ('hate','group')}
predictions = {}
gold_values = {key: [] for key in rows}
ties = 0
raw = BASE / 'runs/coverage-01/dev-b1/scores.jsonl'
assert sha(raw) == read(raw.parent / 'manifest.json')['scores_sha256']
for line in raw.open():
    r = json.loads(line)
    assert r['plan_id'] == ref['plan_id']
    c,t,q = r['condition'], r['task'], r['query_id']
    scores = [x['scores']['answer_sum'] for x in r['candidates']]
    ties += sum(s == max(scores) for s in scores) > 1
    best = max(r['candidates'], key=lambda x: x['scores']['answer_sum'])
    gold_labels = [gold[q][t]] if t == 'hate' else gold[q][t]
    gold_candidate = next(x for x in r['candidates'] if set(x['labels']) == set(gold_labels))
    gold_score = gold_candidate['scores']['answer_sum']
    nll = max(scores) + math.log(sum(math.exp(s-max(scores)) for s in scores)) - gold_score
    gold_values[c,t].append((gold_score == max(scores), nll, math.exp(-nll)))
    assert (c,t,q) not in predictions
    predictions[c,t,q] = best['labels'][0] if t == 'hate' else best['labels']
    rows[c,t].append({'query_id':q, 'gold':gold[q][t], 'prediction':best['canonical_answer']})
assert ties == 0
assert len(predictions) == 10288
summary, details = [], {}
for c in conditions:
    result = {}
    for t in ('hate','group'):
        assert {r['query_id'] for r in rows[c,t]} == set(gold)
        ev = evaluate_predictions(t, rows[c,t])
        m = ev['metrics']
        assert m['invalid_count'] == 0
        yy = [r['gold'] for r in rows[c,t]]
        pp = [predictions[c,t,r['query_id']] for r in rows[c,t]]
        if t == 'group':
            yy = [[int(l in x) for l in GROUP_LABELS] for x in yy]
            pp = [[int(l in x) for l in GROUP_LABELS] for x in pp]
        assert abs(accuracy_score(yy,pp)-m['accuracy' if t=='hate' else 'exact_match']) < 1e-12
        for average in ('micro','macro'):
            assert abs(f1_score(yy,pp,average=average,zero_division=0)-m[average+'_f1']) < 1e-12
        result[t] = m
    details[c] = result
    h,g = result['hate'],result['group']
    joint = sum(predictions[c,'hate',q] == gold[q]['hate'] and set(predictions[c,'group',q]) == set(gold[q]['group']) for q in gold)
    summary.append({'condition':c, 'n':len(gold), 'hate_accuracy':h['accuracy'],
        'hate_precision':h['per_label']['hate']['precision'], 'hate_recall':h['per_label']['hate']['recall'],
        'hate_f1':h['per_label']['hate']['f1'], 'hate_macro_f1':h['macro_f1'],
        'group_micro_f1':g['micro_f1'], 'group_macro_f1':g['macro_f1'],
        'group_sample_f1':g['sample_f1'], 'group_exact_match':g['exact_match'],
        'joint_exact_match':joint/len(gold), 'joint_correct':joint})
payload = {'run':'coverage-01', 'score':'answer_sum', 'decision_rule':'argmax canonical candidate; tasks decoded independently; no gold tuning',
           'scope':'post-hoc descriptive dev classification; not free generation or tuple extraction',
           'top_score_ties':ties, 'gold_source_sha256':entry['sha256'], 'raw_scores_sha256':sha(raw),
           'sklearn_crosscheck':'passed', 'summary':summary, 'metrics':details}
(OUT / 'metrics.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n')
with (OUT / 'summary.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=summary[0],lineterminator="\n");w.writeheader();w.writerows(summary)
gold_summary = []
for c in conditions:
    row = {'condition': c, 'n': len(gold)}
    for t in ('hate', 'group'):
        values = gold_values[c,t]
        correct = sum(v[0] for v in values)
        row.update({t+'_correct': correct, t+'_top1_pct': 100*correct/len(values),
                    t+'_gold_nll': sum(v[1] for v in values)/len(values),
                    t+'_gold_mass': sum(v[2] for v in values)/len(values)})
    gold_summary.append(row)
with (OUT / 'gold_summary.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=gold_summary[0],lineterminator="\n");w.writeheader();w.writerows(gold_summary)
for r in summary:
    print(r['condition'], ' '.join(f'{k}={v*100:.2f}' for k,v in r.items() if isinstance(v,float)))
print('Gold hate counts:', {l:sum(g['hate']==l for g in gold.values()) for l in ('hate','non-hate')})
print('Per label group F1:', {c:{l:round(m['f1']*100,2) for l,m in details[c]['group']['per_label'].items()} for c in conditions})
