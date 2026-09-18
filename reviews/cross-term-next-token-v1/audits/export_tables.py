import csv
import io
import json
from pathlib import Path

ROOT = Path('/data/liaozijie/hate_speech_detection')
OUT = ROOT / 'reviews/cross-term-next-token-v1/results-01'
data = json.loads((OUT / 'results.json').read_text())
material = json.loads((ROOT / 'docs/research/experiment-plans/cross-term-materials-v1/frozen-01/materials.json').read_text())
queries = {q['query']['material_id']: q['query']['raw_text'] for q in material['queries']}

def write_table(name, rows, columns):
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=columns, delimiter='\t', lineterminator='\n', extrasaction='ignore')
    writer.writeheader()
    writer.writerows(rows)
    text = stream.getvalue()
    path = OUT / name
    if path.exists():
        assert path.read_text() == text
    else:
        path.write_text(text)

score_columns = ['condition_id','query_id','family_id','lexicon_arm','demo_arm','adopted_reference','adopted_severity',
    'm','margin_error_bound','reference_aligned_margin','raw_prediction','resolution','raw_correct','conservative_correct',
    'pair_support_no','legal_mass','log_legal_mass','z_yes','z_no','log_p_yes','log_p_no','physical_score_id','prompt_sha256']
write_table('scores.tsv', data['scores'], score_columns)
expressions = []
for row in data['expressions']:
    r = dict(row)
    r.update(effect=row['effect']['value'], bound=row['effect']['bound'], raw_direction=row['effect']['resolution'])
    r['terms'] = json.dumps(row['terms'], ensure_ascii=False, separators=(',', ':'))
    expressions.append(r)
write_table('comparisons.tsv', expressions, ['comparison_id','query_id','family_id','scope','contrast_type','effect','bound',
    'raw_direction','reference_aligned_change','reference_aligned_resolution','verified_classification_transition','terms'])
write_table('family-summaries.tsv', data['family_equal_summaries'], ['scope','contrast_type','family_count','comparison_count',
    'family_equal_mean_effect','family_equal_mean_reference_aligned_change','family_equal_bound',
    'aligned_resolved_positive','aligned_resolved_negative','numerical_unresolved'])
core = [('absent','none'),('L','none'),('N','none'),('absent','same_A'),('L','same_A'),('N','same_A'),
        ('absent','same_B'),('L','same_B'),('N','same_B'),('absent','other_A'),('absent','other_B')]
wide = []
for qid,text in queries.items():
    scores = [r for r in data['scores'] if r['query_id']==qid]
    lookup = {(r['lexicon_arm'],r['demo_arm']):r for r in scores}
    row = {'query_id':qid,'query_text':text,'reference':scores[0]['adopted_reference'],'severity':scores[0]['adopted_severity'],
           'conditions':len(scores),'correct_conditions':sum(r['conservative_correct'] for r in scores),
           'error_or_unresolved_conditions':','.join(r['condition_id'] for r in scores if not r['conservative_correct'])}
    for key in core:
        name = f'm_{key[0]}_{key[1]}'
        row[name] = lookup[key]['m'] if key in lookup else ''
    wide.append(row)
write_table('query-overview.tsv', wide, ['query_id','query_text','reference','severity','conditions','correct_conditions']+
    [f'm_{k[0]}_{k[1]}' for k in core]+['error_or_unresolved_conditions'])
print(json.dumps({'score_rows':len(data['scores']),'expression_rows':len(expressions),
    'query_rows':len(wide),'summary_rows':len(data['family_equal_summaries'])},ensure_ascii=False))
