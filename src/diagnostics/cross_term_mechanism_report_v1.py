"""CPU-only conditional-restoration contrasts, trajectories, and readable report."""
from __future__ import annotations
import csv,itertools
from pathlib import Path
import numpy as np
from diagnostics import cross_term_mechanism_inputs_v1 as c
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
    from diagnostics import cross_term_mechanism_runtime_v1 as rt
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
        if req['dictionary_id']=='D00':continue
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
    data={'schema':'cross-term-mechanism-results/v1','status':'complete','layers':36,'patch_layer':17,
        'requests':reqs,'baselines':list(scores.values()),'effects':effects,'restoration_contrasts':contrasts,'joint_contrasts':joint,
        'self_controls':controls,'native_trajectories':native,'patched_trajectories':trajectories,
        'margin_error_bound':bound,'probe_logit_error_bound':probe_bound,'jobs':jobs,
        'position_differences':position_pairs,'condition_gaps':gaps,'dictionary_addition_gaps':additions,
        'reference_provenance':c.read(p/'analysis-references.json'),
        'notes':NOTES}
    out.mkdir(parents=True);c.write(out/'results.json',data);c.write(out/'qualification.json',q)
    with (out/'all-interventions.tsv').open('w') as f:
        fields=['job_id','query_id','donor','recipient','condition','m','delta_m','reference_aligned_delta','prediction','transition','donor_state_l2']
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore',delimiter='\t');w.writeheader();w.writerows(effects)
    with (out/'restoration-contrasts.tsv').open('w') as f:
        fields=['job_id','query_id','condition','native_m','upstream_m','restored_m','upstream_delta_m','remaining_delta_m','removed_delta_m',
            'removed_fraction','absolute_effect_reduced','absolute_effect_increased','upstream_prediction','restored_prediction']
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore',delimiter='\t');w.writeheader();w.writerows(contrasts)
    with (out/'joint-contrasts.tsv').open('w') as f:
        fields=[k for k,v in joint[0].items() if not k.startswith('trajectory') and not isinstance(v,(list,dict))]
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
    for filename,items in [('baselines.tsv',data['baselines']),('position-differences.tsv',position_pairs)]:
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


def direction(e):return '普通义→贬损义' if e['donor'].endswith('D02') else '贬损义→普通义'


NOTES=[
    '12条开发查询，36个原生输入；5条已审核AI构造与7条真实语料，多个条件不是独立样本。',
    '每组01/03参考无，02/04参考有；G03作者反对概括贬损，原文保持无引号。',
    'm=无logit−有logit；正向是否改善取决于参考答案。',
    'D00仅原生基线；D01贬损义和D02普通义双向干预。',
    '固定17层焦点/前置，26注意力和28MLP单独/联合恢复；不重新挑层或头。',
    '供体/接收方绝对位置分开映射；释义长度不同，保留位置和前缀长度混杂。',
    '注意力分支为o_proj后的完整输出，不是注意力权重；投影受RMS归一化影响。',
    '前置位置不是词性/向量大小匹配，不预设零效应。',
    '全部原生库来自本次运行；不代入历史分数。',
    '工程界不是统计置信区间；全向量恢复比例不是独立中介份额。']


def family_figures(out,data):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import fontManager,FontProperties
    font='/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc';fontManager.addfont(font)
    plt.rcParams.update({'font.family':[FontProperties(fname=font).get_name(),'DejaVu Sans'],
        'axes.unicode_minus':False,'font.size':9,'svg.fonttype':'none'})
    x=np.arange(36);effects=data['effects'];tr=data['patched_trajectories']
    upstream=[e for e in effects if e['condition']=='upstream'];nrows=len(upstream)//2
    styles=[('upstream','#147d91','-','焦点U'),('preceding','#949b9f',':','前置P'),
            ('restore-L26-attention','#cc7041','--','26注意力A'),('restore-L28-mlp','#8058a1','-.','28MLP B'),
            ('restore-joint','#263f50','-','联合AB')]
    def save(fig,name):
        for ext in ['png','pdf','svg']:fig.savefig(out/f'{name}.{ext}',dpi=160,bbox_inches='tight')
        plt.close(fig)
    def axes_style(ax):
        ax.axhline(0,color='#89979c',lw=.7)
        for li in [17,26,28]:ax.axvline(li,color='#888',ls=':',lw=.8,alpha=.6)
        ax.set_xlim(-.5,35.5);ax.set_xticks([0,8,17,23,26,28,32,35]);ax.grid(axis='y',alpha=.15)
        ax.spines[['top','right']].set_visible(False)
    fig,axes=plt.subplots(nrows,2,figsize=(13,3.3*nrows),squeeze=False)
    for ax,u in zip(axes.flat,upstream):
        for i,(cond,color,ls,label) in enumerate(styles):
            e=next(e for e in effects if e['recipient']==u['recipient'] and e['condition']==cond)
            ax.bar(i,e['delta_m'],color=color,width=.6)
            ax.annotate(f'{e["delta_m"]:+.2f}\n{e["prediction"]}',(i,e['delta_m']),xytext=(0,4 if e['delta_m']>=0 else -4),
                textcoords='offset points',ha='center',va='bottom' if e['delta_m']>=0 else 'top',fontsize=8)
        ax.axhline(0,color='#89979c',lw=.7);ax.set_xticks(range(5),[s[3] for s in styles],fontsize=8)
        ax.set_title(f'{u["query_id"]} / {direction(u)}');ax.set_ylabel('最终m − 原生接收方m');ax.margins(y=.3)
        ax.spines[['top','right']].set_visible(False)
    fig.suptitle('新案例：固定层干预的最终效应（全部方向）',fontsize=14)
    fig.text(.5,.01,'m=无logit−有logit；每组01/03参考无，02/04参考有。零线为原生接收方；各面板独立纵轴。',ha='center')
    fig.tight_layout(rect=[0,.03,1,.97]);save(fig,'endpoint-effects')
    fig,axes=plt.subplots(nrows,2,figsize=(13,3.3*nrows),squeeze=False,sharex=True)
    for ax,u in zip(axes.flat,upstream):
        for cond,color,ls,label in styles:
            e=next(e for e in effects if e['recipient']==u['recipient'] and e['condition']==cond)
            ax.plot(x,np.asarray(tr[e['job_id']]['minus_recipient']['probe_pre_mid_post'])[:,2],color=color,ls=ls,label=label)
        axes_style(ax);ax.set_title(f'{u["query_id"]}/{direction(u)}');ax.legend(frameon=False,fontsize=8)
        ax.set_xlabel('层号（0起）');ax.set_ylabel('答案前层末投影 − 原生')
    fig.suptitle('全部36层：干预效应如何到达最终输出',fontsize=14)
    fig.text(.5,.01,'使用最终RMSNorm/输出头读取中间状态；读数受归一化影响，不表示该层已经作决定。',ha='center')
    fig.tight_layout(rect=[0,.03,1,.97]);save(fig,'remaining-trajectories')
    gaps=data['condition_gaps'];fig,axes=plt.subplots(len(gaps),3,figsize=(16,3*len(gaps)),sharex=True,squeeze=False)
    for row,g in enumerate(gaps):
        d=g['trajectory_difference'];axes[row,0].plot(x,np.asarray(d['probe_pre_mid_post'])[:,2],color='#263f50')
        axes[row,0].set_title(g['query_id']+' / 条件层末差距')
        for bi,label in enumerate(['注意力','MLP']):
            ax=axes[row,bi+1]
            for field,color,ls,caption in [('increment_attention_mlp','#173e58','-','每层新增差距'),
                ('branch_projection_at_destination_scale','#258b8a','-','分支投影'),('existing_residual_rescaling','#bd6245','--','RMS重缩放')]:
                ax.plot(x,np.asarray(d[field])[:,bi],color=color,ls=ls,label=caption)
            ax.set_title(g['query_id']+' / '+label);ax.legend(frameon=False,fontsize=8)
        for ax in axes[row]:axes_style(ax);ax.set_xlabel('层号（0起）');ax.set_ylabel('D02普通义 − D01贬损义')
    fig.suptitle('原生条件差距及逐层新增差距：同一查询的两种释义',fontsize=14)
    fig.text(.5,.01,'全部来自答案前位置；代数分支/RMS分解不是独立因果份额，完整浮点余项见数据。',ha='center')
    fig.tight_layout(rect=[0,.03,1,.97]);save(fig,'condition-gaps')
    fig,axes=plt.subplots(nrows,2,figsize=(13,3.3*nrows),sharex=True,squeeze=False)
    for ax,j in zip(axes.flat,data['joint_contrasts']):
        for field,color,ls,label in [('trajectory_AB_minus_A','#cc7041','-','AB−A'),
                ('trajectory_AB_minus_B','#8058a1','--','AB−B'),('trajectory_interaction','#263f50','-','AB−A−B＋U')]:
            ax.plot(x,np.asarray(j[field]['probe_pre_mid_post'])[:,2],color=color,ls=ls,label=label)
        axes_style(ax);ax.set_title(f'{j["query_id"]}/{direction(j)}');ax.legend(frameon=False,fontsize=8)
        ax.set_xlabel('层号（0起）');ax.set_ylabel('答案前层末投影差')
    fig.suptitle('联合恢复与两处单独恢复的比较',fontsize=14)
    fig.tight_layout(rect=[0,0,1,.97]);save(fig,'joint-comparisons')


def figures(out,data):
    for family in dict.fromkeys(r['query_id'][0] for r in data['requests']):
        folder=out/('figures-'+family);folder.mkdir()
        subset=dict(data)
        for key in ['effects','condition_gaps','joint_contrasts']:
            subset[key]=[r for r in data[key] if r['query_id'].startswith(family)]
        family_figures(folder,subset)
        # The same complete 36-layer decomposition, now against no-dictionary.
        addition=dict(subset)
        addition['condition_gaps']=[dict(r,query_id=r['query_id']+' / '+r['comparison']) for r in data['dictionary_addition_gaps'] if r['query_id'].startswith(family)]
        addition_figure(folder,addition['condition_gaps'])


def addition_figure(out,gaps):
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(len(gaps),3,figsize=(16,2.8*len(gaps)),squeeze=False,sharex=True)
    for row,g in enumerate(gaps):
        d=g['trajectory_difference']
        for col,(field,index,label) in enumerate([('probe_pre_mid_post',2,'层末差距'),('increment_attention_mlp',0,'注意力新增差距'),('increment_attention_mlp',1,'MLP新增差距')]):
            ax=axes[row,col];ax.plot(range(36),np.asarray(d[field])[:,index]);ax.axhline(0,color='gray',lw=.6)
            ax.axvspan(20,28,color='#dde9ed',alpha=.4);ax.set_title(g['query_id']+' / '+label)
            ax.set_xlabel('层号（0起）');ax.set_ylabel('条件 − D00无词典')
    fig.suptitle('加入词典相对无词典：答案前位置，全部36层；各面板独立纵轴')
    fig.tight_layout(rect=[0,0,1,.98])
    for ext in ['png','pdf','svg']:fig.savefig(out/('dictionary-addition.'+ext),dpi=150,bbox_inches='tight')
    plt.close(fig)


def write_report(out,data,run):
    scores={(s['query_id'],s['dictionary_id']):s for s in data['baselines']}
    refs={r['query_id']:r['reference'] for r in data['reference_provenance']['references']}
    ordinary=[q for q in refs if q.endswith('01')]
    flips=[q for q in ordinary if scores[q,'D00']['raw_prediction']=='无' and scores[q,'D01']['raw_prediction']=='有']
    effects={e['job_id']:e for e in data['effects']}
    repair=sum(e['transition']=='repair' for e in data['effects']);damage=sum(e['transition']=='damage' for e in data['effects'])
    text=['# 京巴、垃圾、公交车：固定层机制扩展结果','',
        f'本轮3条普通义无攻击查询中，加入贬损义后由无变为有的有{len(flips)}条：'+('、'.join(flips) or '无')+'。这描述当前固定材料中的行为，其他查询及未翻转情况见下表。',
        f'全部120个干预端点中，修复{repair}个、损害{damage}个；这是同一批12条查询的相关条件计数，不能作为独立样本准确率或通用修复率。','',
        '每个词条各有普通用法、直接辱人、反对辱称、普通用法且全文另有攻击四种语境。D00无词典；D01为已审核贬损义；D02为普通义，只提供该词条、无示例。G03按用户审核理解为反对概括贬损，保留原文无引号。所有原生与供体状态均在本次新运行产生。','',
        'm=z(无)−z(有)：正值更偏无，不是概率；参考无时增大有利，参考有时减小有利。','',
        '| 查询 | 原文 | 参考 | D00 输出 / m | D01 输出 / m | D02 输出 / m |','|---|---|---|---:|---:|---:|']
    for q in refs:
        r=next(r for r in data['requests'] if r['query_id']==q);sp=next(s for s in r['spans'] if s['id']=='query')
        query=r['prompt_text'][sp['char_start']:sp['char_end']].replace('|','\|')
        vals=[f"{scores[q,d]['raw_prediction']} / {scores[q,d]['m']:+.4f}" for d in ['D00','D01','D02']]
        text.append(f'| {q} | {query} | {refs[q]} | '+' | '.join(vals)+' |')
    text+=['','答案方向投影用最终输出头读取中间状态，观察偏向有还是无，不表示该层已作决定。条件差距和每层新增差距均来自答案前位置，保留全部36层；0起第20–28层只是预先关注区间。注意力和MLP的代数分支投影、RMS重缩放及浮点余项完整保留，不当成独立因果份额。','',
        'U把另一释义条件下第17层目标词完整向量放入接收方；P替换同层等数量前置token。A、B分别在U上恢复第26层注意力、第28层MLP的原生接收方输出；AB恢复两处。注意力输出指o_proj后整个分支向量，不是注意力权重。供体与接收方按相同查询token映射，绝对位置不同；前置位置不作向量范数匹配。','',
        '| 查询 / 供体→接收方 | U Δm | P Δm | A Δm | B Δm | AB Δm |','|---|---:|---:|---:|---:|---:|']
    for joint in data['joint_contrasts']:
        es=[next(e for e in data['effects'] if e['recipient']==joint['recipient'] and e['condition']==cond) for cond in c.CONDITIONS]
        bycond={e['condition']:e for e in es}
        vals=[bycond[k]['delta_m'] for k in ['upstream','preceding','restore-L26-attention','restore-L28-mlp','restore-joint']]
        text.append(f"| {joint['query_id']} / {direction(joint)} | "+' | '.join(f'{v:+.4f}' for v in vals)+' |')
    text+=['','恢复结果回答的是：在这项上游替换已经发生时，恢复分支能改变多少效应。它不能直接识别唯一自然路径；联合比例也不是独立贡献之和。分母接近工程误差界时比例记NA，所有反向、未翻转和负结果保留。','']
    for family,term in [('J','京巴'),('G','垃圾'),('B','公交车')]:
        if not any(q.startswith(family) for q in refs):continue
        text += [f'## {term}图表','',f'![最终效应](figures-{family}/endpoint-effects.png)','',
                 f'![D02与D01条件差距及分支/RMS分解](figures-{family}/condition-gaps.png)','',
                 f'![加入词典相对D00](figures-{family}/dictionary-addition.png)','',
                 f'![干预后全部层轨迹](figures-{family}/remaining-trajectories.png)','',
                 f'![联合与单独恢复](figures-{family}/joint-comparisons.png)','',
                 '图中层号0起，纵轴为原始分数差或投影差；各面板独立纵轴，不能凭线条高度比较不同查询效应。PDF/SVG同名文件可导出。','']
    text+=['## 解释边界与核对','']+['- '+n for n in NOTES]+['',
        '本轮检验固定方案能否跨词条重现。若有行为变化而固定层干预不对应，需区分词典作用和路径迁移；若没有标签翻转，仍检查分数、对照和恢复效应。不据结果重新筛案例或挑层。','',
        '[完整数值与来源](results.json)；[全部干预](all-interventions.tsv)；[恢复对比](restoration-contrasts.tsv)；[联合对比](joint-contrasts.tsv)；[原生分数](baselines.tsv)；[无词典对照轨迹](dictionary-addition-trajectories.tsv)；[资格门槛](qualification.json)。',
        '本文件是CPU报告，须结合独立数值复核及科学收尾记录使用；旧实验和网站未由此改动。','']
    (out/'REPORT.md').write_text('\n'.join(text),encoding='utf-8')
