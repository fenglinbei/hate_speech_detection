"""CPU-only conditional-restoration contrasts, trajectories, and readable report."""
from __future__ import annotations
import csv,itertools
from pathlib import Path
import numpy as np
from diagnostics import jingba_query_scope_inputs_v1 as c
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
    from diagnostics import jingba_query_scope_runtime_v1 as rt
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
    data={'schema':'jingba-query-scope-results/v1','status':'complete','layers':36,'patch_layer':17,
        'requests':reqs,'baselines':list(scores.values()),'effects':effects,'restoration_contrasts':contrasts,'joint_contrasts':joint,
        'self_controls':controls,'native_trajectories':native,'patched_trajectories':trajectories,
        'margin_error_bound':bound,'probe_logit_error_bound':probe_bound,'jobs':jobs,
        'position_differences':position_pairs,'condition_gaps':gaps,'dictionary_addition_gaps':additions,
        'reference_provenance':c.read(p/'analysis-references.json'),
        'notes':NOTES,'shared_prefix_comparison':shared_prefix_comparison(run,reqs,banks)}
    data['scope_comparisons']=scope_comparisons(data)
    data['CPU_score_shift']=score_shift(data,plan)
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
        rows += [(r['whole_job_id'],'whole-minus-focal',r['trajectory_difference']) for r in data['scope_comparisons']]
        rows += [(r['job_id'],'minus-upstream',r['trajectory_difference']) for r in contrasts]
        rows += [(r['job_id'],label,r[field]) for r in joint for label,field in [
            ('joint-minus-A','trajectory_AB_minus_A'),('joint-minus-B','trajectory_AB_minus_B'),('joint-interaction','trajectory_interaction')]]
        for jid,kind,d in rows:
            for li in range(36):w.writerow([jid,kind,li,*d['probe_pre_mid_post'][li],*d['increment_attention_mlp'][li],
                *d['branch_projection_at_destination_scale'][li],*d['existing_residual_rescaling'][li],*d['floating_remainder'][li]])
    for filename,items in [('baselines.tsv',data['baselines']),('position-differences.tsv',position_pairs),('scope-comparisons.tsv',data['scope_comparisons']),('CPU-score-shift.tsv',data['CPU_score_shift']['records'])]:
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
    '6条已审核且多次观测的构造开发查询，18个原生输入；不是新材料、语料原句或独立确认。',
    'J05/J06普通犬种，J07/J08反对辱称，参考无；J09/J10实施或认可攻击，参考有。',
    'm=无logit−有logit；增加是否改善取决于参考答案，非概率或准确率。',
    '固定主方向D02→D01第17层整段查询W与焦点U比较；六条和双向都保留。W与U的token数及扰动范数不同。',
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
              'token_prefix_equal':a['input_ids'][:a['focal_prefix_length']]==b['input_ids'][:b['focal_prefix_length']]}
        for mode in ['query_prefix','full_prompt']:
            if mode=='full_prompt':aa,bb=[banks[r['request_id']] for r in [a,b]]
            else:
                aa,bb=[np.load(c.read(Path(run)/'records/native-prefix'/(r['request_id']+'.json'))['states']['path'],allow_pickle=False) for r in [a,b]]
            values={}
            for name,positions in [('focal',a['patch_position_sets']['focal']),('preceding',a['patch_position_sets']['pre'])]:
                inds=[a['capture_positions'].index(i) for i in positions]
                binds=[b['capture_positions'].index(i) for i in b['patch_position_sets']['focal' if name=='focal' else 'pre']]
                x,y=aa[:,inds].astype(np.float64),bb[:,binds].astype(np.float64)
                diff=abs(x-y).max(axis=(1,2));scale=np.maximum(1.,np.maximum(abs(x).max(axis=(1,2)),abs(y).max(axis=(1,2))))
                values[name]={'byte_equal':np.array_equal(x,y),'max_abs_by_layer':diff.tolist(),'scaled_by_layer':(diff/scale).tolist()}
            item[mode]=values
        items.append(item)
    return items


def scope_comparisons(data):
    scores={s['request_id']:s for s in data['baselines']};effects=data['effects'];tr=data['patched_trajectories'];bound=data['margin_error_bound'];rows=[]
    for w in effects:
        if w['condition']!='whole-query':continue
        u=next(e for e in effects if e['recipient']==w['recipient'] and e['condition']=='upstream')
        n,d=scores[w['recipient']]['m'],scores[w['donor']]['m'];gain=abs(d-u['m'])-abs(d-w['m'])
        rows.append(dict(query_id=w['query_id'],recipient=w['recipient'],donor=w['donor'],focal_job_id=u['job_id'],whole_job_id=w['job_id'],
            native_m=n,donor_m=d,focal_m=u['m'],whole_m=w['m'],focal_delta_m=u['m']-n,whole_delta_m=w['m']-n,
            whole_minus_focal=w['m']-u['m'],pair_difference_bound=2*bound,
            donor_minus_focal=d-u['m'],donor_minus_whole=d-w['m'],donor_distance_improvement=gain,distance_improvement_bound=4*bound,
            closer_to_donor=gain>4*bound,farther_from_donor=gain< -4*bound,
            focal_tokens=u['replacement_token_count'],whole_tokens=w['replacement_token_count'],
            focal_l2=u['donor_state_l2'],whole_l2=w['donor_state_l2'],focal_prediction=u['prediction'],whole_prediction=w['prediction'],whole_transition=w['transition'],
            trajectory_difference=difference(tr[w['job_id']]['values'],tr[u['job_id']]['values'])))
    return rows


def score_shift(data,plan):
    offset=plan['CPU_score_shift']['offset'];rows=[]
    for n in data['baselines']:
        if n['dictionary_id']!='D01':continue
        value=n['m']+offset;pred='无' if value>0 else '有' if value<0 else None
        rows.append(dict(query_id=n['query_id'],request_id=n['request_id'],native_m=n['m'],shifted_m=value,reference=n['reference'],
            native_prediction=n['raw_prediction'],shifted_prediction=pred,correct=pred==n['reference']))
    return dict(offset=offset,fit_on_current_results=False,new_model_forwards=0,independent_validation=False,records=rows,correct=sum(r['correct'] for r in rows),total=len(rows))


def figures(out,data):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import fontManager,FontProperties
    font='/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc';fontManager.addfont(font)
    plt.rcParams.update({'font.family':[FontProperties(fname=font).get_name(),'DejaVu Sans'],'axes.unicode_minus':False,'font.size':9,'svg.fonttype':'none'})
    folder=Path(out)/'figures';folder.mkdir();x=np.arange(36)
    def score(q,d):return next(r for r in data['baselines'] if r['query_id']==q and r['dictionary_id']==d)
    def eff(q,d,k):return next(e for e in data['effects'] if e['query_id']==q and e['recipient'].endswith(d) and e['condition']==k)
    def finish(fig,name):
        fig.savefig(folder/(name+'.png'),dpi=150,bbox_inches='tight');fig.savefig(folder/(name+'.svg'),bbox_inches='tight');plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(15,8),sharey=True,layout='constrained')
    for ax,q in zip(axes.flat,c.IDS):
        ys=[score(q,d)['m'] for d in ['D00','D01','D02']]+[eff(q,'D01',k)['m'] for k in ['upstream','whole-query','preceding']]
        ax.bar(range(6),ys,color=['#aaa','#cf7258','#389da3','#4477aa','#8055a3','#b9a15f'])
        ax.axhline(0,color='black',lw=.7);ax.set_xticks(range(6),['D00','D01','D02','目标词U','全查询W','前置P'],rotation=25)
        ax.set_title(q+' 参考'+score(q,'D01')['reference']);ax.set_ylabel('m：正偏无，负偏有');ax.margins(y=.2)
        for i,v in enumerate(ys):ax.annotate(f'{v:.2f}',(i,v),xytext=(0,3 if v>=0 else -12),textcoords='offset points',ha='center',fontsize=8)
    fig.suptitle('固定主方向：普通义D02 → 贬损义D01；所有六条，统一纵轴');finish(fig,'fixed-rule-endpoints')
    fig,axes=plt.subplots(6,2,figsize=(15,20),layout='constrained')
    for i,q in enumerate(c.IDS):
        for j,d in enumerate(['D01','D02']):
            ax=axes[i,j]
            for k,label,color in [('upstream','目标词U','#4477aa'),('whole-query','全查询W','#8055a3'),('preceding','前置P','#b9a15f')]:
                e=eff(q,d,k);t=data['patched_trajectories'][e['job_id']]['minus_recipient'];ax.plot(x,np.asarray(t['probe_pre_mid_post'])[:,2],label=label,color=color)
            gap=next(t for t in data['condition_gaps'] if t['query_id']==q)['trajectory_difference']
            ax.plot(x,np.asarray(gap['probe_pre_mid_post'])[:,2]*(1 if d=='D01' else -1),label='供体原生−接收原生',color='#555',ls='--')
            ax.axhline(0,color='gray',lw=.5);ax.axvspan(20,28,alpha=.07);ax.set_xlim(0,35);ax.set_xlabel('层号（0起）');ax.set_ylabel('层末答案方向投影差')
            ax.set_title(q+' '+('普通义→贬损义' if d=='D01' else '贬损义→普通义'));ax.legend(fontsize=8)
    fig.suptitle('相对接收方原生的累计变化：各面板独立纵轴，保留全部36层');finish(fig,'all-layer-scope-trajectories')
    fig,axes=plt.subplots(6,2,figsize=(15,20),layout='constrained')
    for i,q in enumerate(c.IDS):
        for j,d in enumerate(['D01','D02']):
            ax=axes[i,j];r=next(r for r in data['scope_comparisons'] if r['query_id']==q and r['recipient'].endswith(d));t=r['trajectory_difference']
            for bi,b in enumerate(['注意力','MLP']):
                ax.plot(x,np.asarray(t['increment_attention_mlp'])[:,bi],label=b+'新增差')
                ax.plot(x,np.asarray(t['existing_residual_rescaling'])[:,bi],ls=':',alpha=.7,label=b+'中RMS缩放项')
            ax.axhline(0,color='gray',lw=.5);ax.axvspan(20,28,alpha=.07);ax.set_xlim(0,35);ax.set_xlabel('层号（0起）');ax.set_ylabel('W−U的每层新增差');ax.set_title(q+' '+('普通义→贬损义' if d=='D01' else '贬损义→普通义'));ax.legend(fontsize=7)
    fig.suptitle('全查询相对目标词的差异在哪里扩大：各面板独立纵轴；不是注意力权重');finish(fig,'scope-increments-and-rms')


def write_report(out,data,run):
    def score(q,d):return next(r for r in data['baselines'] if r['query_id']==q and r['dictionary_id']==d)
    def eff(q,d,k):return next(e for e in data['effects'] if e['query_id']==q and e['recipient'].endswith(d) and e['condition']==k)
    counts={k:sum(eff(q,'D01',k)['prediction']==score(q,'D01')['reference'] for q in c.IDS) for k in ['upstream','whole-query','preceding']}
    n=sum(score(q,'D01')['raw_reference_correct'] for q in c.IDS);repairs=[q for q in c.IDS if eff(q,'D01','whole-query')['transition']=='repair'];damages=[q for q in c.IDS if eff(q,'D01','whole-query')['transition']=='damage']
    doc=['# 京巴第17层：目标词与整段查询替换','',
        f'在六条已审核且已观测的开发材料上，固定普通义→贬损义方向，原生正确{n}/6；替换京巴两个token后正确{counts["upstream"]}/6；替换整段查询后正确{counts["whole-query"]}/6。全查询替换修复：{",".join(repairs) or "无"}；新增误判：{",".join(damages) or "无"}。','',
        '本轮问的是：只改目标词时传递不充分的释义影响，是否分布在查询的其他位置？W把同一查询在另一释义条件下第17层的整段查询状态搬入，U只搬入京巴两个token；P搬入紧邻前方的两个token。层号从0开始。词典、任务指令、聊天模板和答案前位置不在替换范围。','',
        'D00无词典；D01为已审核的地域贬损义；D02为宠物家犬义。普通义→贬损义表示接收方仍看到贬损义词典，只在内部注入普通义条件的查询状态。m=无logit−有logit，正偏无，负偏有。分数不是概率；方向是否有利要结合参考答案。','',
        '| 查询 | 参考 | D00 | D01 | D02 | U→D01 | W→D01 | P→D01 |','|---|---|---:|---:|---:|---:|---:|---:|']
    for q in c.IDS:
        vals=[score(q,d)['m'] for d in ['D00','D01','D02']]+[eff(q,'D01',k)['m'] for k in ['upstream','whole-query','preceding']]
        doc.append('| '+q+' | '+score(q,'D01')['reference']+' | '+' | '.join(f'{v:+.6f}' for v in vals)+' |')
    doc+=['','![主方向结果](figures/fixed-rule-endpoints.png)','',
        '下面同时保留双向绝对变化。最后一列为“U到供体原生的距离−W到供体原生的距离”，正数表示W更接近供体结果。它不等于准确率、独立路径贡献或中介份额。','',
        '| 查询 | 方向 | U−原生 | W−原生 | W−U | 距供体缩短 | W标签 |','|---|---|---:|---:|---:|---:|---|']
    for r in data['scope_comparisons']:
        doc.append(f'| {r["query_id"]} | {direction(r)} | {r["focal_delta_m"]:+.6f} | {r["whole_delta_m"]:+.6f} | {r["whole_minus_focal"]:+.6f} | {r["donor_distance_improvement"]:+.6f} | {r["whole_prediction"]} |')
    doc+=['',f'W在{sum(r["closer_to_donor"] for r in data["scope_comparisons"])}/12个方向比U更接近供体原生分数，{sum(r["farther_from_donor"] for r in data["scope_comparisons"])}/12个方向更远。两个方向不是两份独立样本。','',
        '![全部36层累计差](figures/all-layer-scope-trajectories.png)','',
        '曲线用最终输出头读取答案前的中间状态，不能当作该层已经作出的最终决定。各面板纵轴独立。每层新增差可能来自分支输出，也可能来自RMS归一化对已有残差的缩放；两部分都保留。','',
        '![每层新增差和RMS](figures/scope-increments-and-rms.png)','',
        '为了判断标签表现是否仅相当于整体更偏向无，另做了纯CPU的m+7对照。常数7来自之前讨论对旧12条跨词条结果的事后检查，本轮在GPU执行前固定，未对新结果调参。它没有新的模型前向，也不是模型实际输出。','',
        '| 查询 | 原生D01 m | 加7后m | 加7标签 | 参考 | W标签 |','|---|---:|---:|---|---|---|']
    for r in data['CPU_score_shift']['records']:
        doc.append(f'| {r["query_id"]} | {r["native_m"]:+.6f} | {r["shifted_m"]:+.6f} | {r["shifted_prediction"]} | {r["reference"]} | {eff(r["query_id"],"D01","whole-query")["prediction"]} |')
    doc+=['',f'统一加7正确{data["CPU_score_shift"]["correct"]}/6。相同正确数或标签并不证明内部效应就是常数偏移；它只限制能从标签收益得出的结论。材料均已暴露，不能称为独立方法验证。','',
        '| 查询 | 接收条件 | U token数 | W token数 | U扰动L2 | W扰动L2 |','|---|---|---:|---:|---:|---:|']
    for r in data['scope_comparisons']:
        doc.append(f'| {r["query_id"]} | {r["recipient"].split("-")[-1]} | {r["focal_tokens"]} | {r["whole_tokens"]} | {r["focal_l2"]:.6f} | {r["whole_l2"]:.6f} |')
    doc+=['','W改变17—25token，U只改变2token，扰动范数也不同。W与U的差不能直接归给某一个后文位置；本轮未做“其余位置单独替换”，也没有范数匹配，因此不能分解非加性交互。W后续仍可读取接收方原词典，和从头使用另一种词典也不是同一运行。','',
        'J08/J10直到京巴的token前缀相同，但整段查询不同。捕获银行扩展到查询全文，本轮真正截断前向到查询末尾；焦点前缀相等与焦点状态比较另行保留，不能把它说成再次运行了仅到京巴的前缀。','',
        f'工程分数界{data["margin_error_bound"]:.9g}，单logit投影界{data["probe_logit_error_bound"]:.9g}；W−U为2倍分数界，与供体距离改善保守用4倍。486次前向的正常预算含36自身控制和54标签/EOS端点。高精度独立审计、旧42端点重放和进程释放凭据在科学关闭时另外保存。','',
        '所有六条和双向保留；18份prompt与已审核版本逐字/token相同。没有新层、头或下游分支恢复，没有网站发布。相关构造材料、释义长度相差18token、后文措辞和长度同时变化等限制仍存在。研究最终仍须走向不依赖测试金标签的选择性利用方法及独立任务验证，不能用本轮局部机制结果代替。','',
        '[绝对分数](baselines.tsv) · [所有干预及范数](all-interventions.tsv) · [W与U比较](scope-comparisons.tsv) · [全部36层](all-trajectories.tsv) · [CPU偏移对照](CPU-score-shift.tsv) · [结构化结果](results.json)','',
        '本轮原文：','']
    for q in c.IDS:
        r=next(r for r in data['requests'] if r['query_id']==q);s=next(s for s in r['spans'] if s['id']=='query')
        doc.append(q+'（参考'+score(q,'D01')['reference']+'）：'+r['prompt_text'][s['char_start']:s['char_end']]+'\n')
    (Path(out)/'REPORT.md').write_text('\n'.join(doc)+'\n')
