"""CPU-only conditional-restoration contrasts, trajectories, and readable report."""
from __future__ import annotations
import csv,itertools
from pathlib import Path
import numpy as np
from diagnostics import jingba_context_inputs_v1 as c
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
    from diagnostics import jingba_context_runtime_v1 as rt
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
    data={'schema':'jingba-context-results/v1','status':'complete','layers':36,'patch_layer':17,
        'requests':reqs,'baselines':list(scores.values()),'effects':effects,'restoration_contrasts':contrasts,'joint_contrasts':joint,
        'self_controls':controls,'native_trajectories':native,'patched_trajectories':trajectories,
        'margin_error_bound':bound,'probe_logit_error_bound':probe_bound,'jobs':jobs,
        'position_differences':position_pairs,'condition_gaps':gaps,'dictionary_addition_gaps':additions,
        'reference_provenance':c.read(p/'analysis-references.json'),
        'notes':NOTES,'shared_prefix_comparison':shared_prefix_comparison(run,reqs,banks)}
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
    '6条本次新构造开发查询，18个原生输入；不是语料原句、完全独立样本或总体确认。',
    'J05/J06普通犬种，J07/J08反对辱称，参考无；J09/J10实施或认可攻击，参考有。',
    'm=无logit−有logit；增加是否改善取决于参考答案，非概率或准确率。',
    '固定主规则D02→D01第17层焦点；六条都保留。反向与前置位置作对照。',
    '无26/28恢复、无新层或头扫描。D00仅原生基线。',
    '供体/接收方绝对位置分别映射，释义长度相差18token，保留位置/长度混杂。',
    '前置对照只匹配token数量，不匹配词性或向量范数，不假定为零。',
    '注意力分支投影不是注意力权重；投影用最终输出头读取中间状态，受RMS归一化影响。',
    'J08/J10词前前缀相同，后文立场、措辞及长度不同；对比不是纯立场单因素实验。',
    '工程误差界不是统计置信区间；没有按结果重新选择材料、层或供体方向。']


def shared_prefix_comparison(run,reqs,banks):
    items=[]
    for did in ['D00','D01','D02']:
        a,b=[next(r for r in reqs if r['query_id']==q and r['dictionary_id']==did) for q in ['J08','J10']]
        item={'dictionary_id':did,'queries':['J08','J10'],
              'token_prefix_equal':a['input_ids'][:a['capture_prefix_length']]==b['input_ids'][:b['capture_prefix_length']]}
        for mode in ['true_prefix','full_prompt']:
            if mode=='full_prompt':aa,bb=[banks[r['request_id']] for r in [a,b]]
            else:
                aa,bb=[np.load(c.read(Path(run)/'records/native-prefix'/(r['request_id']+'.json'))['states']['path'],allow_pickle=False) for r in [a,b]]
            values={}
            for name,positions in [('focal',a['patch_position_sets']['focal']),('preceding',a['patch_position_sets']['pre'])]:
                inds=[a['capture_positions'].index(i) for i in positions]
                x,y=aa[:,inds].astype(np.float64),bb[:,inds].astype(np.float64)
                diff=abs(x-y).max(axis=(1,2));scale=np.maximum(1.,np.maximum(abs(x).max(axis=(1,2)),abs(y).max(axis=(1,2))))
                values[name]={'byte_equal':np.array_equal(x,y),'max_abs_by_layer':diff.tolist(),'scaled_by_layer':(diff/scale).tolist()}
            item[mode]=values
        items.append(item)
    return items


def figures(out,data):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import fontManager,FontProperties
    font='/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc';fontManager.addfont(font)
    plt.rcParams.update({'font.family':[FontProperties(fname=font).get_name(),'DejaVu Sans'],
                        'axes.unicode_minus':False,'font.size':9,'svg.fonttype':'none'})
    out=Path(out);folder=out/'figures';folder.mkdir();x=np.arange(36)
    scores=data['baselines'];effects=data['effects'];tr=data['patched_trajectories']
    def score(q,d):return next(r for r in scores if r['query_id']==q and r['dictionary_id']==d)
    def eff(q,d,k):return next(e for e in effects if e['query_id']==q and e['recipient'].endswith(d) and e['condition']==k)
    def finish(fig,name):
        fig.tight_layout(rect=(0,0,1,.96));fig.savefig(folder/(name+'.png'),dpi=150);fig.savefig(folder/(name+'.svg'));plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(14,7),sharey=True)
    for ax,qid in zip(axes.flat,c.IDS):
        s=[score(qid,d) for d in ['D00','D01','D02']];u,p=eff(qid,'D01','upstream'),eff(qid,'D01','preceding')
        ys=[r['m'] for r in s]+[u['m'],p['m']]
        ax.bar(range(5),ys,color=['#b1b1b1','#d66b50','#3b98a0','#376ca8','#baa36b'])
        ax.axhline(0,color='black',lw=.7);ax.set_xticks(range(5),['D00','D01','D02','U→D01','P→D01'],rotation=20)
        ax.set_title(qid+' 参考'+s[0]['reference']);ax.set_ylabel('m：正偏无，负偏有')
        for i,v in enumerate(ys):ax.annotate(f'{v:.2f}',(i,v),xytext=(0,3 if v>=0 else -12),textcoords='offset points',ha='center',fontsize=8)
    fig.suptitle('同一规则：D02普通义状态 → D01贬损义运行；共享纵轴，层17（0起）')
    finish(fig,'fixed-rule-endpoints')
    fig,axes=plt.subplots(6,3,figsize=(15,18))
    for row,qid in zip(axes,c.IDS):
        gap=next(g for g in data['condition_gaps'] if g['query_id']==qid)['trajectory_difference']
        for j,comparison in enumerate([gap,tr[eff(qid,'D01','upstream')['job_id']]['minus_recipient'],tr[eff(qid,'D02','upstream')['job_id']]['minus_recipient']]):
            a=row[j];a.plot(x,np.asarray(comparison['probe_pre_mid_post'])[:,2],label='累计：层末投影差',color='#222222',lw=1.5)
            inc=np.asarray(comparison['increment_attention_mlp']);a.plot(x,inc[:,0],label='本层注意力新增差',color='#157b89');a.plot(x,inc[:,1],label='本层MLP新增差',color='#c87828')
            if j:
                did='D01' if j==1 else 'D02';p=tr[eff(qid,did,'preceding')['job_id']]['minus_recipient']
                a.plot(x,np.asarray(p['probe_pre_mid_post'])[:,2],label='前置对照累计差',color='#8d68b5',ls='--')
                a.axvline(17,color='gray',ls=':',lw=.8)
            a.axhline(0,color='gray',lw=.5);a.axvspan(20,28,color='#e1d7ad',alpha=.15)
            a.set_xlim(0,35);a.set_xticks([0,10,17,20,28,35]);a.set_xlabel('层号（0起，全部36层）')
            a.set_title(qid+'：'+['D02−D01原生条件差','D02→D01 U−原生','D01→D02 U−原生'][j])
            a.set_ylabel('答案方向投影差')
    axes[0,0].legend(fontsize=7);axes[0,1].legend(fontsize=7)
    fig.suptitle('各面板独立纵轴；新增差包含分支输出与既有残差的RMS缩放变化')
    finish(fig,'gaps-and-intervention-trajectories')
    fig,axes=plt.subplots(2,3,figsize=(14,7))
    for ax,qid in zip(axes.flat,c.IDS):
        gap=next(g for g in data['condition_gaps'] if g['query_id']==qid)['trajectory_difference']
        for bi,branch in enumerate(['注意力','MLP']):
            ax.plot(x,np.asarray(gap['branch_projection_at_destination_scale'])[:,bi],label=branch+'分支投影差')
            ax.plot(x,np.asarray(gap['existing_residual_rescaling'])[:,bi],ls='--',label=branch+'既有残差缩放差')
        ax.axhline(0,color='gray',lw=.5);ax.set_title(qid+' D02−D01');ax.set_xlabel('层号（0起）');ax.set_xlim(0,35)
    axes[0,0].legend(fontsize=7);fig.suptitle('分支与归一化的分解：答案前读数，面板独立纵轴；不是注意力权重')
    finish(fig,'branch-and-rms')


def write_report(out,data,run):
    scores=data['baselines'];effects=data['effects']
    def score(q,d):return next(r for r in scores if r['query_id']==q and r['dictionary_id']==d)
    def eff(q,d,k):return next(e for e in effects if e['query_id']==q and e['recipient'].endswith(d) and e['condition']==k)
    n=sum(score(q,'D01')['raw_reference_correct'] for q in c.IDS)
    u=sum(eff(q,'D01','upstream')['prediction']==score(q,'D01')['reference'] for q in c.IDS)
    repairs=[q for q in c.IDS if eff(q,'D01','upstream')['transition']=='repair']
    damages=[q for q in c.IDS if eff(q,'D01','upstream')['transition']=='damage']
    doc=['# 六条新增京巴查询：固定规则的语境复现','',
         f'固定使用“普通义→贬损义、第17层京巴位置替换”后，这六条材料中正确判断由{n}/6变为{u}/6。修复：{",".join(repairs) or "无"}；新增误判：{",".join(damages) or "无"}。这是固定规则在六条已审核构造材料上的结果，尚不能作为通用修复的证明。','',
         'J05/J06中的京巴是宠物；J07/J08反对地域辱称，四条参考均为无。J09自己使用辱称，J10认可他人辱称，参考均为有。六条均为本次新构造，原文与答案经用户一次性全部通过；没有根据新模型输出筛选材料。','',
         'D00不提供词典；D01提供已审核的地域贬损义；D02提供家犬的普通义。所有条件使用相同任务指令，不提供示例，只提供京巴词条。U把D02运行中第17层“京巴”两个token的完整内部状态放入同一句D01运行；P以同样方式替换紧邻前方的两个token，作为位置对照。层号均从0开始。','',
         'm是模型对“无”的logit减去“有”的logit；正数偏无、负数偏有，它不是概率。正向变化对参考无的材料有利，对参考有的材料可能有害。','',
         '| 查询 | 参考 | 无词典D00 | 贬损义D01 | 普通义D02 | 普通义→贬损义 U | 同方向P |','|---|---|---|---|---|---|---|']
    for q in c.IDS:
        vals=[score(q,d) for d in ['D00','D01','D02']];a,b=eff(q,'D01','upstream'),eff(q,'D01','preceding')
        cells=[f'{v["raw_prediction"]}（{v["m"]:+.3f}）' for v in vals]+[f'{v["prediction"]}（{v["m"]:+.3f}）' for v in [a,b]]
        doc.append('| '+q+' | '+vals[0]['reference']+' | '+' | '.join(cells)+' |')
    doc+=['','括号为m，所有六条保留。相同替换规则没有借助参考答案选择供体、层或方向。',
          '', '![固定规则端点](figures/fixed-rule-endpoints.png)','',
          '反向替换也全部保留，用来检查方向不对称；它不与正向规则混在一起挑选最佳答案。','',
          '| 查询 | 方向 | U的分数变化 | P的分数变化 | U结果 | 参考方向上的结果变化 |','|---|---|---:|---:|---|---|']
    for q in c.IDS:
        for d in ['D01','D02']:
            a,b=eff(q,d,'upstream'),eff(q,d,'preceding')
            doc.append(f'| {q} | {direction(a)} | {a["delta_m"]:+.6f} | {b["delta_m"]:+.6f} | {a["prediction"]} | '+{'repair':'修复','damage':'新增误判','unchanged':'标签不变','unresolved':'误差界内未决'}[a['transition']]+' |')
    doc+=['','答案方向投影用最终输出头读取答案前的中间状态，观察其偏向有还是无；不表示这一层已经完成最终决定。下面同时展示层末累计差和注意力、MLP两个子步骤带来的新增差，保留全部36层。20—28层的底色沿用已有观察窗口，不重新选择有效层。','',
          '![条件差与干预轨迹](figures/gaps-and-intervention-trajectories.png)','',
          '上图各面板纵轴独立，比较高度前应读取刻度。注意力分支指o_proj后的完整输出，不能理解为“关注量”。新增投影差还受RMS归一化对既有残差的缩放影响，分解如下；它不是可相加的因果中介份额。','',
          '![分支与RMS](figures/branch-and-rms.png)','',
          'J08/J10直到京巴的完整token前缀相同，后面的文字才分别表达反对与认可。以下核对局部状态是否已不同：','',
          '| 条件 | 真正截到目标词的前缀：36层状态相同 | 完整prompt：焦点最大绝对差 | 完整prompt：第17层焦点差 |','|---|---|---:|---:|']
    for r in data['shared_prefix_comparison']:
        v=r['full_prompt']['focal']['max_abs_by_layer'];doc.append(f'| {r["dictionary_id"]} | {r["true_prefix"]["focal"]["byte_equal"]} | {max(v):.9g} | {v[17]:.9g} |')
    doc+=['','该比较区分相同因果前缀与不同后文的处理。完整前向的细微差别可能来自不同长度的浮点计算；是否改变最终判断须对照干预效应，不能仅凭曲线认定某个注意力头负责识别立场。','',
          f'工程检查：24个自替换控制全部保留；42个原生/跨条件端点验证单token有或无后EOS。分数工程误差界为{data["margin_error_bound"]:.9g}，中间单logit投影误差界为{data["probe_logit_error_bound"]:.9g}。独立高精度复核记录由科学关闭步骤另行链接，不将本报告自检冒充独立复核。','',
          '限制：D01/D02长度相差18token，词义、位置与前缀长度共同变化；P不是词性或范数匹配。J08/J10也同时改变后文长度与措辞。这六条包含成对构造，不是六个独立抽样，更不能把多个条件和方向算成新增样本。所有来源与完整数值保留，没有新增头/层搜索。','',
          '[全部分数](baselines.tsv) · [全部干预](all-interventions.tsv) · [36层干预差](all-trajectories.tsv) · [条件差距](condition-gap-trajectories.tsv) · [词典加入差距](dictionary-addition-trajectories.tsv) · [结构化结果](results.json)','',
          '原文如下，便于不查前文即可理解：','']
    for q in c.IDS:
        r=next(r for r in data['requests'] if r['query_id']==q);span=next(s for s in r['spans'] if s['id']=='query')
        doc.append(q+'（参考'+score(q,'D00')['reference']+'）：'+r['prompt_text'][span['char_start']:span['char_end']]+'\n')
    (Path(out)/'REPORT.md').write_text('\n'.join(doc)+'\n',encoding='utf-8')
