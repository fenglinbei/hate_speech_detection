#!/usr/bin/env python3
"""CPU analysis after normal release; development lock required for confirmation."""
import argparse
from collections import defaultdict
import json
from fractions import Fraction as F
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from diagnostics import reference_routing_content_inputs_v1 as c
from diagnostics import reference_routing_content_runtime_v1 as rt
from diagnostics import reference_routing_content_analysis_v1 as a


def load(prepared,run,audit,stage):
    run=Path(run);check=c.read(audit);state=c.read(run/'state.json')
    c.require(check['status']=='PASS' and not check['synthetic'] and check['stage']==stage and check.get('independent_from_runtime_readout'),'Independent scientific audit missing')
    c.require(check['raw_seal']==c.info(run/'raw-seal.json'),'Audit data identity differs')
    c.require(state['status']=='COMPLETE' and state['owned_worker_absent'] and state['worker_exit_code']==0,'GPU must be normally released')
    seal=c.read(run/'raw-seal.json');c.require(seal['prepared_manifest']==c.info(Path(prepared)/'manifest.json'),'Wrong preparation')
    c.verify(seal['qualification']);bounds=c.read(seal['qualification']['path'])['per_job_margin_bounds']
    records={}
    for item in seal['records']:
        c.verify(item);r=c.read(item['path'])
        if r['pass']=='production':records[r['job_id']]={'m':r['readout']['m'],'bound':bounds[r['job_id']],'record':item,
                'elapsed_seconds':r['elapsed_seconds'],'valid_tokens':r['valid_tokens'],'tensor_tokens':r['tensor_tokens']}
    c.require(len(records)==check['physical_score_records'],'Incomplete science scores')
    return records


def lock(prepared,run,audit,output):
    output=Path(output);c.require(not output.exists(),'Calibration lock is immutable')
    r=load(prepared,run,audit,'development')
    qs=[q for q in c.read(Path(prepared)/'analysis-references.json')['queries'] if q['split']=='development']
    parameters,tables=a.fit(qs,r)
    # No legitimate confirmation worker can pass its gate before this lock exists.
    # Scan this experiment namespace as an additional exposure guard.
    for statepath in c.WORK.glob('run-*/state.json'):
        s=c.read(statepath)
        c.require(s.get('stage')!='confirmation','Existing confirmation exposure requires a new exploratory protocol')
    output.mkdir(parents=True)
    value=dict(parameters,prepared_manifest=c.info(Path(prepared)/'manifest.json'),split='development',
        confirmation_outputs_seen=False,development_audit=c.info(audit),development_state=c.info(Path(run)/'state.json'),
        development_raw_seal=c.info(Path(run)/'raw-seal.json'),task_candidate='U_CAL',no_future_refit=True)
    c.write(output/'parameters.json',value);c.write(output/'candidate-tables.json',tables)
    c.write(output/'manifest.json',{'schema_version':'reference-routing-content-calibration/v1',
        'artifacts':[c.info(p) for p in sorted(output.iterdir())],
        'sources':[c.info(Path(prepared)/'manifest.json'),c.info(audit),c.info(Path(run)/'state.json'),c.info(Path(run)/'raw-seal.json'),c.info(Path(__file__))],
        'immutable_after_seal':True})
    rt.calibration_gate(prepared,output/'manifest.json')
    return c.info(output/'manifest.json')


def build_report(records,qs,parameters,conditions,cost):
    primary={method:a.aggregate(a.primary_rows(qs,records,parameters,method),records) for method in a.METHODS}
    endpoint_rows=[];groups={};subgroups={}
    for method in a.METHODS:
        allrows=[]
        for condition in conditions:
            rows=[]
            for q in qs:
                ms=a.methods(q['query_id'],condition,parameters);s=ms[method];y=1 if q['reference']=='无' else -1
                row=dict(q,condition=condition,score=s,native=ms['N'],gain=y*(s-ms['N']));rows.append(row)
                endpoint_rows.append({**q,'condition':condition,'method':method,
                    'margin':s.summary(records),'native_margin':ms['N'].summary(records),'G':row['gain'].summary(records),
                    'prediction':a.prediction(s,records),'native_prediction':a.prediction(ms['N'],records)})
            groups[method+'/'+condition]=a.aggregate(rows,records);allrows+=rows
        main=[r for r in allrows if r['condition'] in ('MPS','MSP')]
        partitions={}
        for field in ('term_family_id','new_term','stratum'):
            for value in sorted({str(r[field]) for r in main}):
                part=[r for r in main if str(r[field])==value]
                partitions[field+'/'+value]=a.aggregate(part,records)
        subgroups[method]=partitions
    effects=a.bilateral(qs,records,parameters)
    relations=c.read(c.ADOPTED/'relations.json')['records'];relby={(r['target']['material_id'],r['source']['material_id']):r for r in relations}
    previews={r['request_id']:r for r in c.lines(c.DRAFT/'model-inputs-preview.jsonl')}
    for e in effects['rows']:
        source=previews[f'rrc-{e["query_id"]}-{e["order"]}']['demo_ids'][e['slot']-1]
        rel=relby[e['query_id'],source]
        e.update(source_demo_id=source,relation_id=rel['relation_id'],semantic_reference_fit=rel['semantic_reference_fit']['value'],rule_fit=rel['rule_fit']['value'])
    effects['by_human_relation']={}
    def relation_means(items):
        output={}
        for method in a.METHODS:
            output[method]={}
            for field in ('E','D'):
                byquery=defaultdict(list);terms={}
                for r in items:
                    s=r[field][method];v=a.Score({k:F(v) for k,v in s['coefficients'].items()},F(s['constant']))
                    byquery[r['query_id']].append(v);terms[r['query_id']]=r['term_family_id']
                byterm=defaultdict(list)
                for q,values in byquery.items():byterm[terms[q]].append(a.mean(values))
                output[method][field]=a.mean(a.mean(v) for v in byterm.values()).summary(records)
        return output
    for field in ('semantic_reference_fit','rule_fit'):
        effects['by_human_relation'][field]={level:{'endpoints':sum(r[field]==level for r in effects['rows']),
            'queries':len({r['query_id'] for r in effects['rows'] if r[field]==level}),
            'query_then_term_mean':relation_means([r for r in effects['rows'] if r[field]==level])}
            for level in sorted({r[field] for r in effects['rows']})}
    method_cost={}
    for method in a.METHODS:
        needed=set()
        for q in qs:
            for condition in conditions:
                needed.update(a.methods(q['query_id'],condition,parameters)[method].coeff)
            if method in ('U','U_CAL'):needed.add(f'rrc-{q["query_id"]}-M00/N')
        method_cost[method]={'distinct_production_forwards_with_shared_M00':len(needed),
            'sum_measured_forward_seconds':sum(records[k].get('elapsed_seconds',0) for k in needed),
            'valid_tokens':sum(records[k].get('valid_tokens',0) for k in needed),
            'meaning':'sum of measured production forward costs with same-query M00 reuse; not a separate end-to-end service benchmark',
            'scalar_CPU_calibration_or_scoring_only':method in ('OFFSET7','CAD05','N_CAL','U_CAL')}
    return {'primary':primary,'condition_tables':groups,'subgroups':subgroups,'per_endpoint':endpoint_rows,
        'bilateral':effects,'task_decision':a.task_decision(primary,parameters,cost_complete=bool(cost)),
        'parameters':parameters,'cost':cost,'method_cost':method_cost,'no_statistical_independence_claim':True}


def report(prepared,run,audit,stage,output,calibration=None):
    output=Path(output);c.require(not output.exists(),'New report version required')
    records=load(prepared,run,audit,stage)
    if stage=='stage-a':
        result=a.mechanism(records,c.read(Path(prepared)/'stage-a-references.json')['references'])
        attention=[]
        indices={r['request_id']:r for r in c.read(Path(prepared)/'stage-a-index.json')['inputs']}
        for q in c.read(Path(prepared)/'stage-a-references.json')['references']:
            for order in ('MPS','MSP'):
                rid=f'jmix-{q["query_id"]}-{order}';partition=indices[rid]['key_partition']
                for kind in ('N','U'):
                    av=rt.load_record(run,'production',rid+'/'+kind)[4]
                    for name,keys in partition.items():
                        mass=av['A'][:,keys].sum(-1,dtype='float64')
                        attention.append({'query_id':q['query_id'],'order':order,'kind':kind,'key_group':name,
                            'token_count':len(keys),'per_head_mass':mass.tolist(),
                            'per_head_density':(mass/len(keys)).tolist() if keys else None})
        result['attention_partition']=attention
        result['all_36_layer_trajectories_source']=c.info(Path(run)/'raw-seal.json')
    else:
        parameters=rt.calibration_gate(prepared,calibration)
        qs=[q for q in c.read(Path(prepared)/'analysis-references.json')['queries'] if q['split']==stage]
        conditions=['BG','BA','BI','MPS','MSP']+[f'{o}_replace_{k}' for o in ('MPS','MSP') for k in range(1,5)]
        result=build_report(records,qs,parameters,conditions,c.read(Path(run)/'cost.json'))
        if stage=='development':result['task_decision']={'status':'development_only_not_confirmation','automatic_next_run':False}
    output.mkdir(parents=True);c.write(output/'results.json',result)
    body=['# 冻结方案结果','',f'阶段：{stage}。完整逐例值、工程界、损害与未决均保存在 results.json。','']
    if stage=='stage-a':
        body+=['优先细化候选：'+str(result['priority_candidates'])+'。仅为本批六查询两次序的机制分流。']
    else:
        body+=['| 方法 | balanced accuracy | balanced G | 修复 | 损害 |','|---|---:|---:|---:|---:|']
        for method,s in result['primary'].items():body.append(f'| {method} | {s["balanced_accuracy"]:.6f} | {s["balanced_G"]["value"]:+.6f} | {s["transitions"]["counts"]["repair"]} | {s["transitions"]["counts"]["damage"]} |')
        body+=['',f'任务分流：{result["task_decision"]["status"]}。双侧分流：{result["bilateral"]["status"]}。']
    body+=['','这些工程误差界和资源门槛不等于统计显著性；后续实验没有被本结果自动授权。','']
    (output/'REPORT.md').write_text('\n'.join(body),encoding='utf-8')
    c.write(output/'manifest.json',{'schema_version':'reference-routing-content-results/v1',
        'artifacts':[c.info(p) for p in sorted(output.iterdir())],
        'sources':[c.info(Path(prepared)/'manifest.json'),c.info(audit),c.info(Path(run)/'raw-seal.json')]+([c.info(calibration)] if calibration else []),
        'immutable_after_seal':True})
    return c.info(output/'manifest.json')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['lock','report']);p.add_argument('--prepared',type=Path,default=c.PREPARED)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--audit',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--stage',choices=['stage-a','development','confirmation'],default='stage-a');p.add_argument('--calibration',type=Path)
    x=p.parse_args();r=lock(x.prepared,x.run,x.audit,x.output) if x.command=='lock' else report(x.prepared,x.run,x.audit,x.stage,x.output,x.calibration)
    print(json.dumps(r,indent=2))
