"""Post-release analysis; four native prompts, all interventions and controls."""
from __future__ import annotations
import csv
import io
import itertools
from pathlib import Path
import numpy as np
from diagnostics import hehe_focal_patch_inputs_v1 as c

NOTES=[
    '仅四个既有输入：Q01/Q02 × D01/D02；144个焦点干预和144个前置位置对照不是独立样本。',
    '层号从0开始。在单层decoder块输出处，用同一查询另一释义条件的完整原生向量替换所选位置。每次仅替换一层。',
    'Q01焦点联合替换两个“嘿嘿”token，Q02替换一个；前置对照分别为“回／个”和“被”。前置位置可以读取词典，不假定其效应为零。',
    'm=z(无)−z(有)，正值偏无。Q01参考无、Q02参考有，所以Q02的正Δm可能造成损害。',
    'Δm=替换后−接收方；donor-gap比例=Δm/(供体−接收方)，不截断，分母接近数值界为NA。范围是工程误差传播，不是统计置信区间。',
    '焦点减前置=两种替换后的m之差，共享的接收方抵消。所有层、方向和不利结果均保留。',
    '完整向量替换可造成混合上下文状态；位置、token数及扰动范数均影响结果。不能将一个有效位置解释为唯一语义路径。',
    '末层所选位置之后没有跨token层，预期零效果；自替换也应为零。零结果不排除词典直接作用于后续位置的路径。'
]

def ratio_interval(p,d,r,b):
    gap=d-r
    if abs(gap)<=2*b:return None,None
    values=[(pv-rv)/(dv-rv) for pv,dv,rv in itertools.product((p-b,p+b),(d-b,d+b),(r-b,r+b))]
    return (p-r)/gap,[min(values),max(values)]

def effect(job,patched,recipient,donor,bound,reference):
    p,r,d=patched['m'],recipient['m'],donor['m'];delta=p-r;gap=d-r
    fraction,interval=ratio_interval(p,d,r,bound)
    resolved=all(abs(x)>bound for x in (p,r))
    correct=lambda s:s['raw_prediction']==reference
    transition=('repair' if correct(patched) and not correct(recipient) else 'damage' if correct(recipient) and not correct(patched) else 'unchanged') if resolved else 'unresolved'
    return dict(job,m=p,recipient_m=r,donor_m=d,delta_m=delta,delta_bound=2*bound,
        donor_gap=gap,donor_gap_bound=2*bound,donor_gap_fraction=fraction,donor_gap_fraction_interval=interval,
        reference=reference,reference_aligned_delta=delta if reference=='无' else -delta,
        prediction=patched['raw_prediction'],recipient_prediction=recipient['raw_prediction'],
        flip=patched['raw_prediction']!=recipient['raw_prediction'] if resolved else None,
        transition=transition,direction='positive' if delta>2*bound else 'negative' if delta< -2*bound else 'unresolved')

def write_viewer(path,data):
    template=(c.ROOT/'tools/hehe_focal_patch_viewer_v1/viewer.html').read_text()
    encoded=c.canonical(data).decode().replace('<','\\u003c').replace('\u2028','\\u2028').replace('\u2029','\\u2029')
    c.require(template.count('__DATA_JSON__')==1,'Viewer placeholder differs')
    Path(path).write_text(template.replace('__DATA_JSON__',encoded),encoding='utf-8')

def preview(prepared):
    p=Path(prepared);plan,profile,reqs,jobs,selfs=c.validate(p,sealed=False)
    payload={'schema':'hehe-focal-patching-view/v1','status':'GPU未运行；已采用的真实输入和位置，无测量值',
             'layers':profile['layers'],'requests':reqs,'baselines':[],'effects':[], 'position_differences':[],
             'self_controls':[],'notes':NOTES,'margin_error_bound':None,'jobs':jobs}
    write_viewer(p/'viewer.html',payload);return c.info(p/'viewer.html')

def analyze(prepared,run,output):
    from diagnostics.hehe_focal_patch_runtime_v1 import check_run,load_record,record_path
    from diagnostics.cross_model_applicability_execution_v1 import readout
    p,run,out=map(Path,(prepared,run,output));checked=check_run(p,run)
    c.require(checked['status']=='complete','Full run must be normally released before analysis')
    c.require(not out.exists(),'New analysis output required')
    plan,profile,reqs,jobs,selfs=c.validate(p)
    refs={x['query_id']:x['reference'] for x in c.read(p/'analysis-references.json')['references']}
    q=checked['qualification'];bound=q['margin_error_bound'];binding=c.sha(run/'binding.json')
    byid={r['request_id']:r for r in reqs};banks={};scores={}
    for req in reqs:
        rec,v,s=load_record(run,'native-production',req,binding,profile)
        score=readout(v,profile['candidate_tokens'],bound);reference=refs[req['query_id']]
        score.update(request_id=req['request_id'],query_id=req['query_id'],dictionary_id=req['dictionary_id'],reference=reference,
                     reference_aligned_margin=score['m'] if reference=='无' else -score['m'],raw_reference_correct=score['raw_prediction']==reference)
        scores[req['request_id']]=score;banks[req['request_id']]=(rec,s)
    effects=[]
    for job in jobs:
        req=byid[job['recipient']];rec,v,_=load_record(run,'production',req,binding,profile,job)
        row=effect(job,readout(v,profile['candidate_tokens'],bound),scores[job['recipient']],scores[job['donor']],bound,refs[job['query_id']])
        rbank=banks[job['recipient']][1];dbank=banks[job['donor']][1]
        inds=[req['capture_positions'].index(i) for i in job['positions']]
        dr=dbank[job['layer'],inds].astype(np.float64);rr=rbank[job['layer'],inds].astype(np.float64)
        row.update(donor_state_l2=float(np.linalg.norm(dr-rr)),donor_state_relative_l2=float(np.linalg.norm(dr-rr)/max(1.,float(np.linalg.norm(rr)))),
                   donor_state_per_token_l2=np.linalg.norm(dr-rr,axis=-1).tolist(),source_record=c.info(record_path(run,'production',job['job_id'])))
        effects.append(row)
    keyed={(e['recipient'],e['layer'],e['group']):e for e in effects};pairs=[]
    for req in reqs:
        for layer in range(profile['layers']):
            a=keyed[req['request_id'],layer,'focal'];b=keyed[req['request_id'],layer,'pre'];value=a['m']-b['m']
            pairs.append({'recipient':req['request_id'],'query_id':req['query_id'],'donor':a['donor'],'layer':layer,
                          'focal_job':a['job_id'],'pre_job':b['job_id'],'delta_m':value,'bound':2*bound,
                          'reference_aligned_delta':value if refs[req['query_id']]=='无' else -value,
                          'direction':'positive' if value>2*bound else 'negative' if value< -2*bound else 'unresolved'})
    self_rows=[]
    for job in selfs:
        rec,v,_=load_record(run,'self',byid[job['recipient']],binding,profile,job)
        self_rows.append(dict(job,delta_m=readout(v,profile['candidate_tokens'])['m']-scores[job['recipient']]['m'],
                              changed_rows=rec['patch_proof']['changed_rows_in_unpadded_coordinates'],source_record=c.info(record_path(run,'self',job['job_id']))))
    payload={'schema':'hehe-focal-patching-view/v1','status':'GPU采集完成并正常释放；全部层和对照保留，CPU分析',
             'layers':profile['layers'],'requests':reqs,'baselines':list(scores.values()),'effects':effects,
             'position_differences':pairs,'self_controls':self_rows,'notes':NOTES,'margin_error_bound':bound,'jobs':jobs}
    out.mkdir(parents=True);c.write(out/'results.json',payload);c.write(out/'qualification.json',q)
    fields=['job_id','query_id','recipient','donor','group','layer','m','recipient_m','donor_m','delta_m','delta_bound','donor_gap','donor_gap_fraction','reference_aligned_delta','prediction','flip','transition','donor_state_l2','donor_state_relative_l2']
    stream=io.StringIO();w=csv.DictWriter(stream,fieldnames=fields,delimiter='\t',extrasaction='ignore');w.writeheader();w.writerows(effects)
    (out/'all-interventions.tsv').write_text(stream.getvalue(),encoding='utf-8');write_viewer(out/'index.html',payload)
    write_report(out,payload,q,checked)
    c.write(out/'audit.json',{'status':'pass','CPU_reconstructed':True,'query_references_joined_after_release':True,
                            'prepared_manifest':c.info(p/'manifest.json'),'raw_seal':c.info(run/'raw-seal.json'),
                            'release':c.read(run/'state.json')['resource_release'],'independent_samples_claimed':False})
    c.write(out/'manifest.json',{'artifacts':[c.info(x) for x in sorted(out.iterdir()) if x.is_file()]})
    return {'output':str(out),'native_inputs':len(scores),'interventions':len(effects),'self_controls':len(self_rows),'position_differences':len(pairs)}

def write_report(out,data,qualification,checked):
    rows=['# 查询焦点表示替换：第二阶段','','[交互逐层图](index.html) · [完整JSON](results.json) · [全部干预TSV](all-interventions.tsv)','',
          '仅使用四份既有prompt，在D01原释义与D02普通义之间双向替换查询位置的单层块输出。所有层和位置对照均保留。','',
          '## 新测量的基线','','|查询|词典|参考|输出|m|','|---|---|---|---|---:|']
    for s in data['baselines']:rows.append(f'|{s["query_id"]}|{s["dictionary_id"]}|{s["reference"]}|{s["raw_prediction"]}|{s["m"]:+.6f}|')
    rows+=['','## 全层结果概览','','最大绝对变化的层是事后描述，不是预先选定的因果层；完整曲线和全部端点见交互图/下载。','',
           '|接收方 ← 供体|位置|翻转层（0起）|最大绝对Δm所在层|Δm|donor-gap比例|','|---|---|---|---:|---:|---:|']
    for recipient in [s['request_id'] for s in data['baselines']]:
        for group in ('focal','pre'):
            es=[e for e in data['effects'] if e['recipient']==recipient and e['group']==group]
            largest=max(es,key=lambda e:abs(e['delta_m']));flips=[str(e['layer']) for e in es if e['flip']]
            ratio=largest['donor_gap_fraction'];rat='NA' if ratio is None else f'{ratio:+.6f}'
            rows.append(f'|{recipient} ← {largest["donor"]}|{c.LABELS[group]}|{", ".join(flips) or "无"}|{largest["layer"]}|{largest["delta_m"]:+.6f}|{rat}|')
    rows+=['','## 解释边界','']+['- '+x for x in NOTES]
    rows+=['','本次干预若改变输出，支持“这个固定位置集合在该层的状态替换足以影响此prompt的输出”。它不自动证明自然运行中仅靠该位置，也不把完整状态差异等同于单一词义。',
           '前置对照并非中性基线：其状态已经能读取词典；焦点与前置不同还包含位置和扰动范数差异。Q01为两token联合替换，不能定位到其中某一个。',
           '', '## 工程与数值验证','',
           f'新margin工程界为 {qualification["margin_error_bound"]:.12g}；差值界为 {2*qualification["margin_error_bound"]:.12g}。',
           '只读捕获、自替换、重复、逆序、左右padding、真实前缀、独立生产重放和所有替换端点的“单标签后EOS”均在本次运行内验证。',
           f'自替换 {len(data["self_controls"])} 个；末层结构零对照 {sum(e["last_layer_expected_zero"] for e in data["effects"])} 个。数值界不是统计置信区间。',
           '新donor和基线来自同一run，完整FP32词表向量与供体行均有哈希绑定；参考标签在raw seal及GPU进程正常退出后才加入。',
           '没有重新采集注意力；本轮检验状态替换的输出效应，不能直接归因到某个注意头。','']
    (out/'REPORT.md').write_text('\n'.join(rows),encoding='utf-8')
