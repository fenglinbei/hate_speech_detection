"""CPU-only conditional-restoration contrasts, trajectories, and readable report."""
from __future__ import annotations
import csv,itertools
from pathlib import Path
import numpy as np
from diagnostics import dictionary_free_donor_inputs_v1 as c
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
    from diagnostics import dictionary_free_donor_runtime_v1 as rt
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
        score.update(request_id=req['request_id'],query_id=req['query_id'],dictionary_id=req['dictionary_id'],reference=reference,
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
    position_pairs=[];gaps=[]
    for req in reqs:
        if req['dictionary_id']!='D01':continue
        rid=req['request_id'];u=next(e for e in effects if e['recipient']==rid and e['condition']=='upstream')
        pre=next(e for e in effects if e['recipient']==rid and e['condition']=='preceding')
        delta=u['m']-pre['m'];sign=1 if refs[req['query_id']]=='无' else -1
        position_pairs.append({'query_id':req['query_id'],'recipient':rid,'donor':u['donor'],
            'focal_job_id':u['job_id'],'pre_job_id':pre['job_id'],'delta_m':delta,'bound':2*bound,
            'reference_aligned_delta':sign*delta,
            'trajectory_difference':difference(trajectories[u['job_id']]['values'],trajectories[pre['job_id']]['values'])})
    for qid in dict.fromkeys(r['query_id'] for r in reqs):
        a=next(r['request_id'] for r in reqs if r['query_id']==qid and r['dictionary_id']=='D01')
        b=next(r['request_id'] for r in reqs if r['query_id']==qid and r['dictionary_id']=='D02')
        delta=scores[b]['m']-scores[a]['m'];sign=1 if refs[qid]=='无' else -1
        gaps.append({'query_id':qid,'D01_request':a,'D02_request':b,'reference':refs[qid],
            'delta_m_D02_minus_D01':delta,'reference_aligned_delta':sign*delta,'bound':2*bound,
            'trajectory_difference':difference(native[b],native[a])})
    additions=[]
    for qid in dict.fromkeys(r['query_id'] for r in reqs):
        a=next(r['request_id'] for r in reqs if r['query_id']==qid and r['dictionary_id']=='D00')
        for did in ['D01','D02']:
            b=next(r['request_id'] for r in reqs if r['query_id']==qid and r['dictionary_id']==did)
            delta=scores[b]['m']-scores[a]['m'];sign=1 if refs[qid]=='无' else -1
            additions.append({'query_id':qid,'comparison':did+'-D00','base_request':a,'target_request':b,
                'reference':refs[qid],'delta_m':delta,'reference_aligned_delta':sign*delta,'bound':2*bound,
                'trajectory_difference':difference(native[b],native[a])})
    data={'schema':'dictionary-free-donor-results/v1','status':'complete','layers':36,'patch_layer':17,
        'requests':reqs,'baselines':list(scores.values()),'effects':effects,'restoration_contrasts':contrasts,'joint_contrasts':joint,
        'self_controls':controls,'native_trajectories':native,'patched_trajectories':trajectories,
        'margin_error_bound':bound,'probe_logit_error_bound':probe_bound,'jobs':jobs,
        'position_differences':position_pairs,'condition_gaps':gaps,'dictionary_addition_gaps':additions,
        'reference_provenance':c.read(p/'analysis-references.json'),
        'notes':NOTES}
    data['donor_comparisons']=donor_comparisons(data,plan)
    out.mkdir(parents=True);c.write(out/'results.json',data);c.write(out/'qualification.json',q)
    with (out/'all-interventions.tsv').open('w') as f:
        fields=['job_id','query_id','donor','recipient','condition','m','delta_m','reference_aligned_delta','prediction','transition','donor_state_l2']
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore',delimiter='\t');w.writeheader();w.writerows(effects)
    with (out/'restoration-contrasts.tsv').open('w') as f:
        fields=['job_id','query_id','condition','native_m','upstream_m','restored_m','upstream_delta_m','remaining_delta_m','removed_delta_m',
            'removed_fraction','absolute_effect_reduced','absolute_effect_increased','upstream_prediction','restored_prediction']
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore',delimiter='\t');w.writeheader();w.writerows(contrasts)
    with (out/'joint-contrasts.tsv').open('w') as f:
        fields=[k for k,v in joint[0].items() if not k.startswith('trajectory') and not isinstance(v,(list,dict))] if joint else ['job_id']
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore',delimiter='\t');w.writeheader();w.writerows(joint)
    with (out/'all-trajectories.tsv').open('w') as f:
        w=csv.writer(f,delimiter='\t');w.writerow(['job_id','comparison','layer0','probe_pre','probe_mid','probe_post','increment_attention','increment_mlp',
            'branch_attention','branch_mlp','scale_attention','scale_mlp','rounding_attention','rounding_mlp'])
        rows=[(e['job_id'],'minus-native',trajectories[e['job_id']]['minus_recipient']) for e in effects]
        rows += [(r['job_id'],'minus-upstream',r['trajectory_difference']) for r in contrasts]
        rows += [(r['job_id'],label,r[field]) for r in joint for label,field in [
            ('joint-minus-A','trajectory_AB_minus_A'),('joint-minus-B','trajectory_AB_minus_B'),('joint-interaction','trajectory_interaction')]]
        for jid,kind,d in rows:
            for li in range(36):w.writerow([jid,kind,li,*d['probe_pre_mid_post'][li],*d['increment_attention_mlp'][li],
                *d['branch_projection_at_destination_scale'][li],*d['existing_residual_rescaling'][li],*d['floating_remainder'][li]])
    for filename,items in [('baselines.tsv',data['baselines']),('position-differences.tsv',position_pairs),('donor-comparisons.tsv',data['donor_comparisons'])]:
        fields=[k for k,v in items[0].items() if not isinstance(v,(list,dict))]
        with (out/filename).open('w') as f:
            w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore',delimiter='\t');w.writeheader();w.writerows(items)
    with (out/'condition-gap-trajectories.tsv').open('w') as f:
        w=csv.writer(f,delimiter='\t');w.writerow(['query_id','layer0','pre','mid','post','increment_attention','increment_mlp','branch_attention','branch_mlp','scale_attention','scale_mlp','remainder_attention','remainder_mlp'])
        for gap in gaps:
            d=gap['trajectory_difference']
            for li in range(36):w.writerow([gap['query_id'],li,*d['probe_pre_mid_post'][li],*d['increment_attention_mlp'][li],*d['branch_projection_at_destination_scale'][li],*d['existing_residual_rescaling'][li],*d['floating_remainder'][li]])
    with (out/'dictionary-addition-trajectories.tsv').open('w') as f:
        w=csv.writer(f,delimiter='\t');w.writerow(['query_id','comparison','layer0','pre','mid','post','increment_attention','increment_mlp','branch_attention','branch_mlp','scale_attention','scale_mlp','remainder_attention','remainder_mlp'])
        for gap in additions:
            d=gap['trajectory_difference']
            for li in range(36):w.writerow([gap['query_id'],gap['comparison'],li,*d['probe_pre_mid_post'][li],*d['increment_attention_mlp'][li],*d['branch_projection_at_destination_scale'][li],*d['existing_residual_rescaling'][li],*d['floating_remainder'][li]])
    figures(out,data);write_report(out,data,run)
    c.write(out/'audit.json',{'status':'pass','CPU_reconstructed':True,'query_references_joined_after_release':True,
        'prepared_manifest':c.info(p/'manifest.json'),'raw_seal':c.info(run/'raw-seal.json'),
        'release':c.read(run/'state.json')['resource_release'],'independent_samples_claimed':False})
    c.write(out/'manifest.json',{'artifacts':[c.info(x) for x in sorted(out.rglob('*')) if x.is_file()]})
    return {'output':str(out),'native_inputs':len(reqs),'interventions':len(jobs),'restoration_contrasts':len(contrasts),'joint_contrasts':len(joint),'self_controls':len(selfs),'position_differences':len(position_pairs),'condition_gaps':len(gaps),'dictionary_addition_gaps':len(additions),'trajectory_layers':36}



NOTES=['12条已暴露开发查询，原文及参考不变；5条已审核AI构造、7条语料。','无词典供体是固定推理规则候选，不是正确答案状态；全数据保留。','位置/定义长度与扰动范数未匹配。']

def donor_comparisons(data,plan):
    out=[]
    for q in c.IDS:
        ns={r['dictionary_id']:r for r in data['baselines'] if r['query_id']==q}
        es={r['condition']:r for r in data['effects'] if r['query_id']==q};n=ns['D01'];u=es['upstream'];p=es['preceding'];u0=es['no-dictionary-focal'];p0=es['no-dictionary-preceding'];sign=1 if n['reference']=='无' else -1
        row={'query_id':q,'reference':n['reference'],'recipient':n['request_id'],
             'D00_m':ns['D00']['m'],'D01_m':n['m'],'D02_m':ns['D02']['m'],'U2_m':u['m'],'P2_m':p['m'],'U0_m':u0['m'],'P0_m':p0['m'],
             'U0_job_id':u0['job_id'],'P0_job_id':p0['job_id'],'U2_job_id':u['job_id'],'P2_job_id':p['job_id'],
             'U0_effect':u0['m']-n['m'],'P0_effect':p0['m']-n['m'],'U0_minus_P0':u0['m']-p0['m'],
             'U0_minus_U2':u0['m']-u['m'],'U0_minus_D00':u0['m']-ns['D00']['m'],
             'reference_aligned_U0_effect':sign*(u0['m']-n['m']),'pair_bound':2*data['margin_error_bound'],
             'U0_transition':u0['transition'],'U2_transition':u['transition'],'P0_transition':p0['transition'],
             'CPU_offset':plan['CPU_score_shift']['offset'],'CPU_shifted_m':n['m']+plan['CPU_score_shift']['offset']}
        for k in ['D00','D01','D02','U2','P2','U0','P0','CPU_shifted']:
            value=row[k+'_m'];row[k+'_prediction']='无' if value>0 else '有' if value<0 else None
        out.append(row)
    return out


def figures(out,data):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    folder=Path(out)/'figures';folder.mkdir();x=np.arange(12)
    fig,ax=plt.subplots(figsize=(14,5))
    for kind,offset in [('D00',-.3),('U2',-.1),('U0',.1),('P0',.3)]:
        rs=data['donor_comparisons'];ax.bar(x+offset,[r[kind+'_m']-r['D01_m'] for r in rs],width=.18,label=kind+' - D01')
    ax.axhline(0,color='grey',lw=.7);ax.set_xticks(x,c.IDS);ax.set_ylabel('Final margin change; m = logit(no) - logit(yes)');ax.legend();fig.tight_layout()
    fig.savefig(folder/'fixed-rule-effects.png',dpi=145);fig.savefig(folder/'fixed-rule-effects.svg');plt.close(fig)
    for group in 'JGB':
        fig,axes=plt.subplots(4,2,figsize=(13,12));layer=np.arange(36)
        for i,q in enumerate([q for q in c.IDS if q.startswith(group)]):
            row=next(r for r in data['donor_comparisons'] if r['query_id']==q)
            for k in ['U0','U2','P0']:
                t=data['patched_trajectories'][row[k+'_job_id']]['minus_recipient'];axes[i,0].plot(layer,np.asarray(t['probe_pre_mid_post'])[:,2],label=k+' - native')
            t=data['patched_trajectories'][row['U0_job_id']]['minus_recipient']
            for j,label in [(0,'attention'),(1,'MLP')]:axes[i,1].plot(layer,np.asarray(t['increment_attention_mlp'])[:,j],label=label)
            for j in range(2):
                ax=axes[i,j];ax.axhline(0,color='grey',lw=.6);ax.axvline(17,color='grey',ls=':');ax.set_title(q+' individual y scale');ax.set_xlim(0,35);ax.legend(fontsize=8)
        fig.supxlabel('Layer, zero based');fig.supylabel('Answer-direction projection / per-layer increment');fig.tight_layout()
        fig.savefig(folder/('all36-'+group+'.png'),dpi=145);fig.savefig(folder/('all36-'+group+'.svg'));plt.close(fig)


def write_report(out,data,run):
    rs=data['donor_comparisons'];correct=lambda k:sum(r[k+'_prediction']==r['reference'] for r in rs)
    repaired=[r['query_id'] for r in rs if r['D01_prediction']!=r['reference'] and r['U0_prediction']==r['reference']]
    damaged=[r['query_id'] for r in rs if r['D01_prediction']==r['reference'] and r['U0_prediction']!=r['reference']]
    text=['# 无词典供体的固定局部干预','',f'固定第17层、目标词位置的无词典状态替换，正确数从原词典条件的{correct("D01")}/12变为{correct("U0")}/12。修复：{repaired or "无"}；损坏：{damaged or "无"}。这是已暴露开发材料，不能视为独立效果确认。','',
      'D01提供已审核贬损义；D00完全不提供词典，D02提供普通义。U0把同一查询D00运行中目标词的第17层完整状态，放入保留D01词典的运行；U2使用D02供体。P0/P2改换相邻等token前置位置。主规则固定用于全部12条，不读取参考答案、不按输出选方向，也不需要普通义释义。规则理论需要两次前向，本轮额外前向用于诊断和工程复核。','',
      '三类词条各四条：普通语境、直接攻击、反对贬损、普通义但全文另有攻击。当前人工参考依次无/有/无/有；G02/G03保留源标签与当前审核的差异。m=无logit−有logit，正值判无。增加m是否改善取决于参考答案。','',
      '| 查询 | 参考 | D01原生 | 直接D00 | 普通义U2 | 无词典U0 | 前置P0 | U0变化 |','|---|---|---:|---:|---:|---:|---:|---:|']
    for r in rs:text.append('| '+r['query_id']+' | '+r['reference']+' | '+' | '.join(f'{r[k+"_m"]:+.6f}（{r[k+"_prediction"]}）' for k in ['D01','D00','U2','U0','P0'])+f' | {r["U0_effect"]:+.6f} |')
    text+=['','| 条件 | 正确数 |','|---|---:|']
    for k in ['D01','D00','D02','U2','P2','U0','P0','CPU_shifted']:text.append('| '+k+' | '+str(correct(k))+'/12 |')
    text+=['','CPU_shifted固定为D01的m+7，来自先前讨论，不在本轮拟合；对照简单统一偏移，不称独立基线优化。对照直接D00是必要的：若局部操作并无更好收益/损害组合，就不能仅因内部状态被改变而称方法有效。','',
      '![固定规则分数效应](figures/fixed-rule-effects.png)','',
      '## 全层传播与限制','',
      '答案方向投影用最终输出头读取中间状态，不是注意力权重或该层已作决定。全部36层保留；下图各查询/读数纵轴独立。分支投影、残差RMS缩放、余差详见all-trajectories.tsv。这里只在17层目标词处干预，后续曲线不等于逐层因果定位。','',
      '![京巴全层](figures/all36-J.png)','', '![垃圾全层](figures/all36-G.png)','', '![公交车全层](figures/all36-B.png)','',
      f'本轮756次前向、48自身控制、84格式端点。工程分数界{data["margin_error_bound"]:.9g}，两项差界2ε；不是统计显著性或实际效应阈值。独立核对新鲜36原生+24旧U2/P2端点和36状态银行精确重放，旧分数不代替新数据。','',
      '无词典供体没有“正确答案来源”保证。去掉词典同时改变长度和位置；P0未按范数/语法匹配。即使多个标签变正确，也可能是一般分数移动和距离零点不同，尚不能称自动识别适用性。本批12条已暴露材料包含5条人工接受的AI构造及7条真实语料，样本不足以主张通用修复。','',
      '若此固定规则失败，不据新结果改层号或强度再称预注册成功。下一步回到正确参考同时包含适用与不适用示例的材料，冻结规则后另做未参与开发的验证。']
    (Path(out)/'REPORT.md').write_text('\n'.join(text))
