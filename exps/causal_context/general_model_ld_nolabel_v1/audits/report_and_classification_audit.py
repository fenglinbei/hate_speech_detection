"""Recompute classification from sealed raw candidates and export the completed study."""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT/'exps/causal_context/general_model_ld_nolabel_v1'
RUN = BASE/'runs/nolabel-01'
OUT = BASE/'results/nolabel-01'
CONDITIONS = ('C0','CLnew','CD','CLDnew','CLnewNoCat','CLDnewNoCat')
LABELS = ('Racism','Region','LGBTQ','Sexism','others')
PAIRS = {'NoCat_without_D':('CLnewNoCat','CLnew'), 'NoCat_with_D':('CLDnewNoCat','CLDnew')}


def read(path):
    return json.loads(path.read_text())


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def write_csv(path, rows):
    with path.open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def close(a,b):
    if not math.isclose(float(a),float(b),rel_tol=1e-12,abs_tol=1e-12):
        raise ValueError(f'classification recomputation differs: {a} vs {b}')


def direct_f1(g,p,task):
    if task=='hate':
        # Two fixed classes, even when a bootstrap resample misses one class.
        g=np.column_stack((g,~g))
        p=np.column_stack((p,~p))
    tp=np.logical_and(g,p).sum(axis=0)
    fp=np.logical_and(~g,p).sum(axis=0)
    fn=np.logical_and(g,~p).sum(axis=0)
    if task=='group':
        tp,fp,fn=tp.sum(),fp.sum(),fn.sum()
        return float(2*tp/(2*tp+fp+fn)) if 2*tp+fp+fn else 0.0
    den=2*tp+fp+fn
    return float(np.divide(2*tp,den,out=np.zeros(2,dtype=float),where=den!=0).mean())


def main():
    run=read(RUN/'run_manifest.json')
    if run['status']!='complete':
        raise ValueError('Gold/report access requires completed sealed scientific run')
    ref=read(BASE/'plan_ref.json')
    plan_path=Path(ref['target_path'])/'plan.json'
    assert sha(plan_path)==ref['plan_sha256']
    plan=read(plan_path)
    assert plan['plan_id']==run['plan_id']
    assert sha(RUN/'analysis/manifest.json')==run['analysis_manifest_sha256']
    assert sha(RUN/'dev-b1/manifest.json')==run['raw_manifest_sha256']
    raw_manifest=read(RUN/'dev-b1/manifest.json')
    raw_hash=sha(RUN/'dev-b1/scores.jsonl')
    assert raw_hash==raw_manifest['scores_sha256']
    analysis_manifest=read(RUN/'analysis/manifest.json')
    assert sha(RUN/'analysis/analysis.json')==analysis_manifest['analysis_sha256']
    analysis=read(RUN/'analysis/analysis.json')
    package=Path(plan['package_path'])
    assert sha(package/'manifest.json')==plan['package_manifest_sha256']
    entry=next(r for r in read(package/'manifest.json')['files'] if r['path']=='queries.dev.jsonl')
    assert sha(package/'queries.dev.jsonl')==entry['sha256']
    gold={str(r['id']):r['projection'] for r in map(json.loads,(package/'queries.dev.jsonl').read_text().splitlines())}
    qids=[q['query_id'] for q in plan['frame']]
    assert len(qids)==643 and set(qids)==set(gold)
    predictions={}
    for line in (RUN/'dev-b1/scores.jsonl').open():
        block=json.loads(line)
        q,t,c=block['query_id'],block['task'],block['condition']
        assert block['plan_id']==plan['plan_id'] and (q,t,c) not in predictions
        best=min(block['candidates'],key=lambda x:(-x['scores']['answer_sum'],x['ordinal']))
        predictions[q,t,c]=best['labels']
    assert set(predictions)=={(q,t,c) for q in qids for t in ('hate','group') for c in CONDITIONS}
    truth={'hate':np.array([gold[q]['hate']=='hate' for q in qids]),
           'group':np.array([[l in gold[q]['group'] for l in LABELS] for q in qids])}
    pred={c:{'hate':np.array(['hate' in predictions[q,'hate',c] for q in qids]),
             'group':np.array([[l in predictions[q,'group',c] for l in LABELS] for q in qids])} for c in CONDITIONS}
    for q in analysis['per_query']:
        for c in CONDITIONS:
            for t in ('hate','group'):
                assert q['conditions'][c][t]['prediction']['labels']==predictions[q['query_id'],t,c]
    summary=[]
    for c in CONDITIONS:
        h,g=pred[c]['hate'],pred[c]['group']
        joint=int(np.logical_and(h==truth['hate'],np.all(g==truth['group'],axis=1)).sum())
        row={'condition':c,'n':len(qids),'hate_accuracy':accuracy_score(truth['hate'],h),
            'hate_precision':precision_score(truth['hate'],h,zero_division=0),
            'hate_recall':recall_score(truth['hate'],h,zero_division=0),
            'hate_f1':f1_score(truth['hate'],h,zero_division=0),
            'hate_macro_f1':f1_score(truth['hate'],h,labels=[True,False],average='macro',zero_division=0),
            'group_micro_f1':f1_score(truth['group'],g,average='micro',zero_division=0),
            'group_macro_f1':f1_score(truth['group'],g,average='macro',zero_division=0),
            'group_exact_match':accuracy_score(truth['group'],g),
            'group_sample_f1':f1_score(truth['group'],g,average='samples',zero_division=1),
            'joint_exact_match':joint/len(qids),'joint_correct':joint}
        saved=next(r for r in analysis['classification'] if r['stratum']=='all' and r['condition']==c)
        for t,names in (('hate',('accuracy','macro_f1')),('group',('micro_f1','macro_f1','exact_match','sample_f1'))):
            for name in names:
                close(row[t+'_'+name],saved['metrics'][t][name])
        for name in ('precision','recall','f1'):
            close(row['hate_'+name],saved['metrics']['hate']['per_label']['hate'][name])
        close(row['joint_exact_match'],saved['joint_exact_match'])
        for t in ('hate','group'):
            close(direct_f1(truth[t],pred[c][t],t),row['hate_macro_f1' if t=='hate' else 'group_micro_f1'])
        summary.append(row)
    # Independent direct index resampling, rather than the producer's count-weight matrix.
    rng=np.random.Generator(np.random.PCG64(42))
    draws={(t,name):[] for t in ('hate','group') for name in PAIRS}
    for _ in range(10000):
        ix=rng.integers(0,len(qids),size=len(qids))
        scores={(t,c):direct_f1(truth[t][ix],pred[c][t][ix],t)
                for t in ('hate','group') for c in ('CLnew','CLDnew','CLnewNoCat','CLDnewNoCat')}
        for t in ('hate','group'):
            for name,(a,b) in PAIRS.items():
                draws[t,name].append(scores[t,a]-scores[t,b])
    primary=[]
    for r in analysis['primary_classification_differences']:
        ci=np.quantile(draws[r['task'],r['contrast']],[.025,.975])
        np.testing.assert_allclose(ci,r['descriptive_ci95'],rtol=1e-12,atol=1e-12)
        primary.append({k:v for k,v in r.items() if k!='descriptive_ci95'}|{'ci95_low':float(ci[0]),'ci95_high':float(ci[1])})
    OUT.mkdir(parents=True,exist_ok=True)
    write_csv(OUT/'classification.csv',summary)
    write_csv(OUT/'primary_differences.csv',primary)
    gold_rows=[]
    for c in CONDITIONS:
        row={'condition':c}
        for t in ('hate','group'):
            for metric in ('mass','nll','rank'):
                cell=next(r for r in analysis['condition_summaries'] if r['condition']==c and r['stratum']=='all'
                          and r['task']==t and r['metric']=='answer_sum/gold/'+metric)
                row[t+'_gold_'+metric]=cell['mean']
        gold_rows.append(row)
    write_csv(OUT/'gold.csv',gold_rows)
    per_label=[]
    for r in analysis['classification']:
        if 'metrics' not in r:
            continue
        for t,m in r['metrics'].items():
            for label,values in m['per_label'].items():
                per_label.append({'condition':r['condition'],'stratum':r['stratum'],'n':r['n'],'task':t,'label':label,**values})
    write_csv(OUT/'per_label.csv',per_label)
    (OUT/'classification_all_strata.json').write_text(json.dumps(analysis['classification'],ensure_ascii=False,indent=2)+'\n')
    # Auxiliary raw-score summaries stay separate from the four primary intervals.
    for family in ('condition','contrast'):
        rows=analysis[family+'_summaries']
        write_csv(OUT/(family+'_auxiliary.csv'),rows)
    text=['# Qwen3-8B 去词典类别字段：完整 dev 结果','',
          '本轮全部 643 条 dev、六条件、7,716 blocks 已完成。分类取不含 EOS 的答案总 logprob 最高分候选。',
          '四个配对 F1 区间已从封存 raw 重新构建预测并直接重抽样复算；分类点值与 sklearn 一致。','',
          '## 分类指标','',
          '| 条件 | Hate Accuracy | Hate F1 | Hate Macro-F1 | Group Micro-F1 | Group Macro-F1 | 两项同时正确 |',
          '|---|---:|---:|---:|---:|---:|---:|']
    for r in summary:
        text.append('| '+r['condition']+' | '+' | '.join(f'{100*r[k]:.2f}%' for k in
            ('hate_accuracy','hate_f1','hate_macro_f1','group_micro_f1','group_macro_f1','joint_exact_match'))+' |')
    text+=['','## 预注册主要差值','',
           '均为“去类别字段 − 保留类别字段”，单位为百分点。95% 区间为逐点描述性配对 bootstrap 区间。','',
           '| 对比 | 指标 | 差值 | 95% 区间 |','|---|---|---:|---|']
    for r in primary:
        text.append(f'| {r["contrast"]} | {r["task"]} {r["metric"]} | {100*r["difference"]:+.2f} | [{100*r["ci95_low"]:+.2f}, {100*r["ci95_high"]:+.2f}] |')
    text+=['','## 解释边界','',
           '删除仅作用于词典的显式类别字段，定义及示例答案保持；输入因此缩短，定义仍可能透露类别。',
           '这是已暴露 dev 上的固定候选分类结果，不是自由生成、test 泛化或内部机制识别。辅助读数不替代四个主要分类差值。','',
           '逐类与分层结果见 per_label.csv 和 classification_all_strata.json；Gold 见 gold.csv；所有原始辅助计分见两个 auxiliary.csv。']
    (OUT/'REPORT.md').write_text('\n'.join(text)+'\n')
    receipt={'passed':True,'plan_id':plan['plan_id'],'raw_scores_sha256':raw_hash,
        'analysis_sha256':analysis_manifest['analysis_sha256'],'checker_sha256':sha(Path(__file__)),
        'classified_blocks':len(predictions),'sklearn_points_verified':True,'paired_ci_targets_recomputed':4,
        'bootstrap_repetitions':10000,'bootstrap_seed':42,'test_content_read':False,
        'files':{p.name:sha(p) for p in sorted(OUT.iterdir()) if p.is_file() and p.name!='report_manifest.json'}}
    (OUT/'report_manifest.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':
    main()
