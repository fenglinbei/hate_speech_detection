"""CPU-only conditional-restoration contrasts, trajectories, and readable report."""
from __future__ import annotations
import csv,itertools
from pathlib import Path
import numpy as np
from diagnostics import jingba_demo_donor_inputs_v1 as c
from diagnostics.hehe_bridge_report_v1 import describe
from diagnostics.hehe_focal_patch_report_v1 import effect

FIELDS=['probe_pre_mid_post','increment_attention_mlp','branch_projection_at_destination_scale',
        'existing_residual_rescaling','floating_remainder','rms_pre_mid_post','state_l2_pre_mid_post']


def contrast(n,u,r,bound):
    original=u-n;remaining=r-n;removed=u-r
    safe=abs(original)>2*bound
    ratio=removed/original if safe else None
    # Shared upstream score in numerator and denominator; enumerate its same
    # perturbation in both expressions. These are engineering envelopes, not CIs.
    corners=[(u+eu-r-er)/(u+eu-n-en) for en,eu,er in itertools.product([-bound,bound],repeat=3)] if safe else []
    return {'native_m':n,'upstream_m':u,'restored_m':r,'upstream_delta_m':original,
        'remaining_delta_m':remaining,'removed_delta_m':removed,'difference_bound':2*bound,
        'removed_fraction':ratio,'removed_fraction_interval':[min(corners),max(corners)] if safe else None,
        'denominator_safe':safe,'absolute_effect_reduced':abs(remaining)<abs(original)-4*bound,
        'absolute_effect_increased':abs(remaining)>abs(original)+4*bound,
        'interpretation':'conditional intervention contrast; not additive mediation shares'}


def difference(a,b):
    return {name:(np.asarray(a[name])-np.asarray(b[name])).tolist() for name in FIELDS}


def joint_contrast(n,u,a,b,ab,bound):
    gap=u-n;safe=abs(gap)>2*bound
    interaction=ab-a-b+u
    def envelope(fn):
        if not safe:return None
        values=[fn(nn,uu,aa,bb,jj) for nn,uu,aa,bb,jj in itertools.product(
            [n-bound,n+bound],[u-bound,u+bound],[a-bound,a+bound],
            [b-bound,b+bound],[ab-bound,ab+bound])]
        return [min(values),max(values)]
    return {'native_m':n,'upstream_m':u,'A_m':a,'B_m':b,'AB_m':ab,
        'AB_minus_A':ab-a,'AB_minus_B':ab-b,'pair_difference_bound':2*bound,
        'interaction_m':interaction,'interaction_bound':4*bound,
        'interaction_resolved':abs(interaction)>4*bound,
        'interaction_fraction':interaction/gap if safe else None,
        'interaction_fraction_interval':envelope(lambda nn,uu,aa,bb,jj:(jj-aa-bb+uu)/(uu-nn)),
        'additional_removed_after_A':a-ab,'additional_removed_after_B':b-ab,
        'additional_fraction_after_A':(a-ab)/gap if safe else None,
        'additional_fraction_after_B':(b-ab)/gap if safe else None,
        'additional_fraction_after_A_interval':envelope(lambda nn,uu,aa,bb,jj:(aa-jj)/(uu-nn)),
        'additional_fraction_after_B_interval':envelope(lambda nn,uu,aa,bb,jj:(bb-jj)/(uu-nn)),
        'absolute_reduced_vs_A':abs(ab-n)<abs(a-n)-4*bound,
        'absolute_reduced_vs_B':abs(ab-n)<abs(b-n)-4*bound,
        'denominator_safe':safe,
        'meaning':'Finite conditional interaction on m scale; no independent mediation shares or statistical CI.'}


def analyze(prepared,run,output):
    from diagnostics import jingba_demo_donor_runtime_v1 as rt
    from diagnostics.cross_model_applicability_execution_v1 import readout
    p,run,out=map(Path,(prepared,run,output));checked=rt.check_run(p,run)
    c.require(checked['status']=='complete','Full run must be normally released before analysis')
    c.require(not out.exists(),'Use a new analysis output')
    plan,profile,reqs,jobs,selfs=c.validate(p);byid={r['request_id']:r for r in reqs}
    refs={r['query_id']:r['reference'] for r in c.read(p/'analysis-references.json')['references']}
    q=checked['qualification'];bound=q['margin_error_bound'];probe_bound=q['trajectory']['probe_absolute_error_bound']
    binding=c.sha(run/'binding.json');scores={};banks={};native={};raw={}
    for req in reqs:
        rec,v,s=rt.load_record(run,'native-production',req,binding,profile)
        score=readout(v,profile['candidate_tokens'],bound);reference=refs[req['query_id']]
        score.update(request_id=req['request_id'],query_id=req['query_id'],dictionary_id=req['dictionary_id'],condition=req['condition'],reference=reference,
            reference_aligned_margin=score['m'] if reference=='无' else -score['m'],raw_reference_correct=score['raw_prediction']==reference)
        rid=req['request_id'];scores[rid]=score;banks[rid]=s
        raw[rid]=rt.load_trajectory(rec,req,profile,v)[0]
        native[rid]=dict(describe(raw[rid]),source=rec['trajectory'])
    effects=[];trajectories={}
    for job in jobs:
        req=byid[job['recipient']];rec,v,_=rt.load_record(run,'production',req,binding,profile,job)
        rid,did=job['recipient'],job['donor']
        row=effect(job,readout(v,profile['candidate_tokens'],bound),scores[rid],scores[did],bound,refs[job['query_id']])
        row.update(condition=job['condition'],restoration=job['restoration'],upstream_job_id=job['upstream_job_id'],
            restoration_proof=rec['patch_proof']['restoration'],source_record=c.info(rt.record_path(run,'production',job['job_id'])))
        inds=[req['capture_positions'].index(pos) for pos in job['positions']]
        dinds=[byid[did]['capture_positions'].index(pos) for pos in job['donor_positions']]
        delta=banks[did][job['layer'],dinds].astype(np.float64)-banks[rid][job['layer'],inds].astype(np.float64)
        row['donor_state_l2']=float(np.linalg.norm(delta))
        row['replacement_token_count']=len(inds)
        row['donor_state_l2_per_token']=np.linalg.norm(delta,axis=-1).tolist()
        row['donor_state_rms_per_coordinate']=float(np.sqrt(np.mean(delta*delta)))
        t=rt.load_trajectory(rec,req,profile,v)[0];description=describe(t)
        diffs=difference(description,native[rid])
        diffs['state_difference_l2_pre_mid_post']=np.linalg.norm(t['states'].astype(np.float64)-raw[rid]['states'],axis=-1).tolist()
        effects.append(row)
        trajectories[job['job_id']]={'values':description,'minus_recipient':diffs,'source':rec['trajectory']}
    es={e['job_id']:e for e in effects};contrasts=[]
    for e in effects:
        if e['restoration'] is None:continue
        u=es[e['upstream_job_id']];n=scores[e['recipient']]
        row=dict(job_id=e['job_id'],query_id=e['query_id'],donor=e['donor'],recipient=e['recipient'],
            condition=e['condition'],upstream_job_id=u['job_id'],
            native_prediction=n['raw_prediction'],upstream_prediction=u['prediction'],restored_prediction=e['prediction'],
            reference=refs[e['query_id']],**contrast(n['m'],u['m'],e['m'],bound))
        sign=1 if refs[e['query_id']]=='无' else -1
        row['reference_aligned_restore_minus_upstream']=(e['m']-u['m'])*sign
        row['trajectory_difference']=difference(trajectories[e['job_id']]['values'],trajectories[u['job_id']]['values'])
        contrasts.append(row)
    joint=[]
    for e in effects:
        if e['condition']!='restore-joint':continue
        u=es[e['upstream_job_id']]
        a=next(x for x in effects if x['recipient']==e['recipient'] and x['condition']=='restore-L26-attention')
        b=next(x for x in effects if x['recipient']==e['recipient'] and x['condition']=='restore-L28-mlp')
        jt,at,bt,ut=[trajectories[x['job_id']]['values'] for x in [e,a,b,u]]
        row=dict(job_id=e['job_id'],query_id=e['query_id'],recipient=e['recipient'],donor=e['donor'],
            upstream_job_id=u['job_id'],A_job_id=a['job_id'],B_job_id=b['job_id'],
            **joint_contrast(scores[e['recipient']]['m'],u['m'],a['m'],b['m'],e['m'],bound))
        row['trajectory_AB_minus_A']=difference(jt,at)
        row['trajectory_AB_minus_B']=difference(jt,bt)
        row['trajectory_interaction']=difference(difference(jt,at),difference(bt,ut))
        joint.append(row)
    controls=[]
    for job in selfs:
        rec,v,_=rt.load_record(run,'self',byid[job['recipient']],binding,profile,job)
        baseline=es[job['upstream_job_id']]['m'] if job['kind']=='conditional_self_control' else scores[job['recipient']]['m']
        controls.append(dict(job,delta_m=readout(v,profile['candidate_tokens'])['m']-baseline,
            source_record=c.info(rt.record_path(run,'self',job['job_id']))))
    pairs=[]
    for req in reqs:
        if req['condition']=='M00':continue
        u=next(e for e in effects if e['recipient']==req['request_id'] and e['condition']=='upstream')
        pre=next(e for e in effects if e['recipient']==req['request_id'] and e['condition']=='preceding')
        pairs.append({'query_id':req['query_id'],'recipient':req['request_id'],'focal_job_id':u['job_id'],'pre_job_id':pre['job_id'],'delta_m':u['m']-pre['m'],'bound':2*bound,'reference_aligned_delta':(u['m']-pre['m'])*(1 if refs[req['query_id']]=='无' else -1),'trajectory_difference':difference(trajectories[u['job_id']]['values'],trajectories[pre['job_id']]['values'])})
    data={'schema':'jingba-demo-donor-results/v1','status':'complete','layers':36,'patch_layer':17,'requests':reqs,'baselines':list(scores.values()),'effects':effects,'restoration_contrasts':[],'joint_contrasts':[],'self_controls':controls,'native_trajectories':native,'patched_trajectories':trajectories,'margin_error_bound':bound,'probe_logit_error_bound':probe_bound,'jobs':jobs,'position_differences':pairs,'condition_gaps':[],'dictionary_addition_gaps':[],'reference_provenance':c.read(p/'analysis-references.json')}
    out.mkdir(parents=True);c.write(out/'results.json',data);c.write(out/'qualification.json',q)
    for filename,rows in [('baselines.tsv',data['baselines']),('all-interventions.tsv',effects),('position-differences.tsv',pairs)]:
        fields=[k for k,v in rows[0].items() if not isinstance(v,(dict,list))]
        with (out/filename).open('w') as f:w=csv.DictWriter(f,fieldnames=fields,delimiter='\t',extrasaction='ignore');w.writeheader();w.writerows(rows)
    with (out/'all-trajectories.tsv').open('w') as f:
        w=csv.writer(f,delimiter='\t');w.writerow(['job_id','layer0','pre','mid','post','attention_increment','mlp_increment','attention_branch','mlp_branch','attention_scale','mlp_scale','attention_rounding','mlp_rounding'])
        for e in effects:
            t=trajectories[e['job_id']]['minus_recipient']
            for l in range(36):w.writerow([e['job_id'],l,*t['probe_pre_mid_post'][l],*t['increment_attention_mlp'][l],*t['branch_projection_at_destination_scale'][l],*t['existing_residual_rescaling'][l],*t['floating_remainder'][l]])
    render(out,data)
    c.write(out/'audit.json',{'status':'pass','query_references_joined_after_release':True,'prepared_manifest':c.info(p/'manifest.json'),'raw_seal':c.info(run/'raw-seal.json'),'release':c.read(run/'state.json')['resource_release']})
    c.write(out/'manifest.json',{'artifacts':[c.info(x) for x in sorted(out.rglob('*')) if x.is_file()]})
    return {'inputs':18,'interventions':24,'self_controls':24,'restoration_contrasts':0,'joint_contrasts':0,'all36layers':True}


def render(out,data):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    n={(r['query_id'],r['condition']):r for r in data['baselines']};e={r['job_id']:r for r in data['effects']}
    text=['# 无参考局部供体：示例帮助保留检验','','固定第17层（0起算）全部京巴两token完整block输出，供体为同句无参考M00，接收MPS/MSP保留四条正确示例。U换京巴；P换紧邻等token前置。所有条件均无词典；没有新文本、关系标签、层或头搜索。','','J05/J06宠物、J07/J08反对辱称，参考无；J09/J10实施或赞同攻击，参考有。m=无logit−有logit，正值判无。下表逐条区分参考帮助是否保留；两种混合次序不能当作两份独立查询。','','| 查询 | 参考 | 接收组 | 无参考M00 | 混合原生N | U最终 | U−N | P最终 | P−N | U变化 |','|---|---|---|---:|---:|---:|---:|---:|---:|---|']
    for q in c.IDS:
        for k in ['MPS','MSP']:
            nn=n[q,k];u=e[q+'-'+k+'-upstream'];p=e[q+'-'+k+'-preceding'];text.append('| '+q+' | '+nn['reference']+' | '+k+' | '+' | '.join(f'{v:+.6f}' for v in [n[q,'M00']['m'],nn['m'],u['m'],u['delta_m'],p['m'],p['delta_m']])+' | '+u['transition']+' |')
    text+=['','| 条件 | 原生正确 | U正确 | P正确 | U修复 | U损害 | 固定+7正确 |','|---|---:|---:|---:|---|---|---:|']
    for k in ['MPS','MSP']:
        us=[e[q+'-'+k+'-upstream'] for q in c.IDS];ps=[e[q+'-'+k+'-preceding'] for q in c.IDS]
        counts=[sum(n[q,k]['raw_reference_correct'] for q in c.IDS),sum(r['prediction']==n[r['query_id'],k]['reference'] for r in us),sum(r['prediction']==n[r['query_id'],k]['reference'] for r in ps)]
        shift=sum(('无' if n[q,k]['m']+7>0 else '有')==n[q,k]['reference'] for q in c.IDS)
        text.append('| '+k+' | '+' | '.join(str(x)+'/6' for x in counts)+' | '+(', '.join(r['query_id'] for r in us if r['transition']=='repair') or '无')+' | '+(', '.join(r['query_id'] for r in us if r['transition']=='damage') or '无')+' | '+str(shift)+'/6 |')
    folder=out/'figures';folder.mkdir()
    fig,axs=plt.subplots(2,1,figsize=(11,8))
    for ax,k in zip(axs,['MPS','MSP']):
        x=np.arange(6)
        for cond,off in [('upstream',-.16),('preceding',.16)]:ax.bar(x+off,[e[q+'-'+k+'-'+cond]['delta_m'] for q in c.IDS],.3,label=cond)
        ax.axhline(0,color='grey',lw=.7);ax.set_xticks(x,c.IDS);ax.set_title(k+' individual y scale');ax.set_ylabel('Final margin change');ax.legend()
    fig.tight_layout();fig.savefig(folder/'effects.png',dpi=145);fig.savefig(folder/'effects.svg');plt.close(fig)
    fig,axs=plt.subplots(3,2,figsize=(12,11))
    for ax,q in zip(axs.flat,c.IDS):
        for k in ['MPS','MSP']:
            for cond in c.CONDITIONS:
                t=data['patched_trajectories'][q+'-'+k+'-'+cond]['minus_recipient'];ax.plot(range(36),np.asarray(t['probe_pre_mid_post'])[:,2],label=k+' '+cond)
        ax.axhline(0,color='grey',lw=.6);ax.set_xlim(0,35);ax.set_title(q+' individual y scale');ax.legend(fontsize=7)
    fig.supxlabel('Layer, zero based');fig.supylabel('Pre-answer answer-direction projection change');fig.tight_layout();fig.savefig(folder/'all36.png',dpi=145);fig.savefig(folder/'all36.svg');plt.close(fig)
    text+=['','![最终效应](figures/effects.png)','','![全部36层](figures/all36.png)','','全层图为答案前中间状态的最终输出头读数，不是注意力权重或后续层的独立因果干预；各子图纵轴独立。attention/MLP新增及RMS分解保存在完整表。','','本轮只检验既有示例帮助是否保留，没有建立有害示例条件。即使改善，也不能直接称正确适用性选择；失败则是此固定规则的边界。位置控制未范数/词性匹配，移除示例改变长度/位置；所有材料已暴露且构造相关，没有独立泛化证据。固定+7沿用早期诊断，未在此次拟合，不是参考选择方法。','','378次前向；24自身、42精确标签后EOS端点；旧18原生完整向量、状态、轨迹精确重放。独立审计另列。没有网站发布。','','[执行协议](../prepared-01/PROTOCOL.md) · [全部输入](../prepared-01/ALL-PROMPTS.md)']
    (out/'REPORT.md').write_text('\n'.join(text))
