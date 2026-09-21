"""CPU-only conditional-restoration contrasts, trajectories, and readable report."""
from __future__ import annotations
import csv,itertools
from pathlib import Path
import numpy as np
from diagnostics import jingba_query_complement_inputs_v1 as c
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
    from diagnostics import jingba_query_complement_runtime_v1 as rt
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
    data={'schema':'jingba-query-complement-results/v1','status':'complete','layers':36,'patch_layer':17,
        'requests':reqs,'baselines':list(scores.values()),'effects':effects,'restoration_contrasts':contrasts,'joint_contrasts':joint,
        'self_controls':controls,'native_trajectories':native,'patched_trajectories':trajectories,
        'margin_error_bound':bound,'probe_logit_error_bound':probe_bound,'jobs':jobs,
        'position_differences':position_pairs,'condition_gaps':gaps,'dictionary_addition_gaps':additions,
        'reference_provenance':c.read(p/'analysis-references.json'),
        'notes':NOTES,'shared_prefix_comparison':shared_prefix_comparison(run,reqs,banks)}
    data['scope_comparisons']=scope_comparisons(data)
    data['CPU_score_shift']=score_shift(data,plan)
    data['query_interactions']=query_interactions(data)
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
        rows += [(r['complement_job_id'],'I=W-U-C+N',r['trajectory_interaction']) for r in data['query_interactions']]
        rows += [(r['job_id'],'minus-upstream',r['trajectory_difference']) for r in contrasts]
        rows += [(r['job_id'],label,r[field]) for r in joint for label,field in [
            ('joint-minus-A','trajectory_AB_minus_A'),('joint-minus-B','trajectory_AB_minus_B'),('joint-interaction','trajectory_interaction')]]
        for jid,kind,d in rows:
            for li in range(36):w.writerow([jid,kind,li,*d['probe_pre_mid_post'][li],*d['increment_attention_mlp'][li],
                *d['branch_projection_at_destination_scale'][li],*d['existing_residual_rescaling'][li],*d['floating_remainder'][li]])
    for filename,items in [('baselines.tsv',data['baselines']),('position-differences.tsv',position_pairs),('scope-comparisons.tsv',data['scope_comparisons']),('CPU-score-shift.tsv',data['CPU_score_shift']['records']),('query-interactions.tsv',data['query_interactions'])]:
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
    '固定主方向D02→D01第17层新增词外C，与N/U/W/P比较；六条双向保留。U/C分区，范数与词性不匹配。',
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


def scope_figures(out,data):
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


def query_interactions(data):
    ns={s['request_id']:s for s in data['baselines']};es=data['effects'];tr=data['patched_trajectories'];bound=data['margin_error_bound'];rows=[]
    for cc in es:
        if cc['condition']!='query-complement':continue
        u,w=[next(e for e in es if e['recipient']==cc['recipient'] and e['condition']==k) for k in ['upstream','whole-query']]
        n=ns[cc['recipient']]['m'];uu,cv,ww=u['m'],cc['m'],w['m'];interaction=ww-uu-cv+n
        # (W-N)-(U-N)-(C-N) with the shared N retained exactly once.
        ti=difference(difference(tr[w['job_id']]['values'],tr[u['job_id']]['values']),tr[cc['job_id']]['minus_recipient'])
        rows.append(dict(query_id=cc['query_id'],recipient=cc['recipient'],donor=cc['donor'],focal_job_id=u['job_id'],whole_job_id=w['job_id'],complement_job_id=cc['job_id'],
            native_m=n,focal_m=uu,complement_m=cv,whole_m=ww,
            focal_delta_m=uu-n,complement_delta_m=cv-n,whole_delta_m=ww-n,
            conditional_complement_delta_m=ww-uu,conditional_focal_delta_m=ww-cv,
            additive_prediction_delta_m=uu+cv-2*n,interaction_m=interaction,interaction_bound=4*bound,pair_difference_bound=2*bound,
            interaction_resolved=abs(interaction)>4*bound,
            interaction_direction='positive' if interaction>4*bound else 'negative' if interaction< -4*bound else 'unresolved',
            focal_tokens=u['replacement_token_count'],complement_tokens=cc['replacement_token_count'],whole_tokens=w['replacement_token_count'],
            focal_l2=u['donor_state_l2'],complement_l2=cc['donor_state_l2'],whole_l2=w['donor_state_l2'],
            complement_prediction=cc['prediction'],complement_transition=cc['transition'],whole_prediction=w['prediction'],trajectory_interaction=ti))
    return rows


def figures(out,data):
    scope_figures(out,data)
    import matplotlib.pyplot as plt
    folder=Path(out)/'figures';x=np.arange(36)
    def finish(fig,name):
        fig.savefig(folder/(name+'.png'),dpi=150,bbox_inches='tight');fig.savefig(folder/(name+'.svg'),bbox_inches='tight');plt.close(fig)
    fig,axes=plt.subplots(2,1,figsize=(14,10),layout='constrained')
    for ax,did in zip(axes,['D01','D02']):
        rows=[next(r for r in data['query_interactions'] if r['query_id']==q and r['recipient'].endswith(did)) for q in c.IDS]
        for index,(field,label,color) in enumerate([('focal_delta_m','目标词U','#4477aa'),('complement_delta_m','词外C','#c88b28'),('whole_delta_m','整段W','#8055a3'),('interaction_m','组合差 I=W−U−C+N','#29896a')]):
            xx=np.arange(6)+(index-1.5)*.2;yy=[r[field] for r in rows]
            ax.bar(xx,yy,width=.19,color=color,label=label)
            for pos,v in zip(xx,yy):ax.annotate(f'{v:+.2f}',(pos,v),xytext=(0,3 if v>=0 else -10),textcoords='offset points',ha='center',fontsize=7)
        ax.set_xticks(np.arange(6),c.IDS);ax.set_title('普通义→贬损义' if did=='D01' else '贬损义→普通义');ax.axhline(0,color='black',lw=.7);ax.set_ylabel('分数变化（不是概率）');ax.margins(y=.25);ax.legend(fontsize=8)
    fig.suptitle('同一第17层：目标词、词外和整段替换；两个方向独立纵轴');finish(fig,'complement-effects-and-interaction')
    fig,axes=plt.subplots(6,2,figsize=(15,20),layout='constrained')
    for i,q in enumerate(c.IDS):
        for j,did in enumerate(['D01','D02']):
            ax=axes[i,j];r=next(r for r in data['query_interactions'] if r['query_id']==q and r['recipient'].endswith(did))
            tt=data['patched_trajectories'];c_t=tt[r['complement_job_id']]['minus_recipient'];conditional=difference(tt[r['whole_job_id']]['values'],tt[r['focal_job_id']]['values'])
            for v,label,color in [(c_t,'词外单独 C−N','#c88b28'),(conditional,'已换目标词后加词外 W−U','#8055a3'),(r['trajectory_interaction'],'两条曲线之差 I','#29896a')]:ax.plot(x,np.asarray(v['probe_pre_mid_post'])[:,2],label=label,color=color)
            ax.axhline(0,color='gray',lw=.5);ax.axvspan(20,28,alpha=.07);ax.set_xlim(0,35);ax.set_xlabel('层号（0起）');ax.set_ylabel('层末答案方向投影差');ax.set_title(q+' '+('普通义→贬损义' if did=='D01' else '贬损义→普通义'));ax.legend(fontsize=7)
    fig.suptitle('词外单独作用与条件作用：全部36层，各面板独立纵轴');finish(fig,'complement-conditional-trajectories')
    fig,axes=plt.subplots(6,2,figsize=(15,20),layout='constrained')
    for i,q in enumerate(c.IDS):
        for j,did in enumerate(['D01','D02']):
            ax=axes[i,j];t=next(r for r in data['query_interactions'] if r['query_id']==q and r['recipient'].endswith(did))['trajectory_interaction']
            for bi,b in enumerate(['注意力','MLP']):
                ax.plot(x,np.asarray(t['increment_attention_mlp'])[:,bi],label=b+'新增I')
                ax.plot(x,np.asarray(t['existing_residual_rescaling'])[:,bi],ls=':',alpha=.7,label=b+'中RMS缩放项')
            ax.axhline(0,color='gray',lw=.5);ax.axvspan(20,28,alpha=.07);ax.set_xlim(0,35);ax.set_xlabel('层号（0起）');ax.set_ylabel('每层新增组合差');ax.set_title(q+' '+('普通义→贬损义' if did=='D01' else '贬损义→普通义'));ax.legend(fontsize=7)
    fig.suptitle('组合差的层内变化与RMS分解；各面板独立纵轴；不是分支干预证据');finish(fig,'interaction-increments-and-rms')


def write_report(out,data,run):
    score=lambda q,d:next(r for r in data['baselines'] if r['query_id']==q and r['dictionary_id']==d)
    eff=lambda q,d,k:next(r for r in data['effects'] if r['query_id']==q and r['recipient'].endswith(d) and r['condition']==k)
    counts={k:sum(eff(q,'D01',k)['prediction']==score(q,'D01')['reference'] for q in c.IDS) for k in ['upstream','query-complement','whole-query','preceding']}
    n=sum(score(q,'D01')['raw_reference_correct'] for q in c.IDS)
    repairs=[q for q in c.IDS if eff(q,'D01','query-complement')['transition']=='repair'];damages=[q for q in c.IDS if eff(q,'D01','query-complement')['transition']=='damage']
    doc=['# 京巴第17层：词外查询位置单独替换','',
        f'固定普通义→贬损义方向，原生正确{n}/6；目标词U为{counts["upstream"]}/6、词外C为{counts["query-complement"]}/6、整段W为{counts["whole-query"]}/6。新增C修复：{",".join(repairs) or "无"}；新增误判：{",".join(damages) or "无"}。','',
        '本轮补的是：不改京巴自身，只换待判断文本中其他位置的第17层完整内部状态。C包含词前、词后以及查询内标点，共15—23token；U只换京巴2token；W把U和C一起换；N不干预。层号从0开始。供体为同一句在另一词典条件下的本轮新鲜状态；词典、任务、模板及答案前位置不换。P仍是前置2token对照，而且属于C，不能作为独立区域与C相加。','',
        'D00无词典，D01已审核地域贬损义，D02宠物家犬义。J05/J06宠物、J07/J08反对辱称，参考无；J09/J10实施或认可攻击，参考有。全部是原先已审核且多次观测的开发材料，18份模型prompt保持逐字/token一致。','',
        'm=无logit−有logit，正偏无负偏有。我们关注两个量：C−N是词外单独作用；W−U是已经换过目标词后再加入词外的作用。它们之差I=W−U−C+N，表示这两个背景下词外作用的差异。只看到I不为零不构成完整机制解释，需同时看绝对效应、方向、案例差异和标签收益。','',
        '| 查询 | 方向 | 目标词 U−N | 词外 C−N | 整段 W−N | 已换目标词后加词外 W−U | 组合差I | C标签 |','|---|---|---:|---:|---:|---:|---:|---|']
    for r in data['query_interactions']:
        doc.append(f'| {r["query_id"]} | {direction(r)} | {r["focal_delta_m"]:+.6f} | {r["complement_delta_m"]:+.6f} | {r["whole_delta_m"]:+.6f} | {r["conditional_complement_delta_m"]:+.6f} | {r["interaction_m"]:+.6f} | {r["complement_prediction"]} |')
    doc+=['','I也等于“已有词外时目标词的作用(W−C)−目标词单独作用(U−N)”。正I仅表示分数尺度上的额外正向差，不能自动称为协同或改善；负I也不能自动称为相互抑制。完整四格分数见[交互数据表](query-interactions.tsv)。','',
        '![两方向效应与组合差](figures/complement-effects-and-interaction.png)','',
        '| 查询 | 参考 | D00 | D01 | D02 | U→D01 | C→D01 | W→D01 | P→D01 |','|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for q in c.IDS:
        vals=[score(q,d)['m'] for d in ['D00','D01','D02']]+[eff(q,'D01',k)['m'] for k in ['upstream','query-complement','whole-query','preceding']]
        doc.append('| '+q+' | '+score(q,'D01')['reference']+' | '+' | '.join(f'{v:+.6f}' for v in vals)+' |')
    doc+=['','J07/J08在三个原生词典条件下均误判；这里没有一个已判断正确的原生供体。因此没有修复不能直接等同于没有传递，分数传递也不能直接等同于选择性正确利用。','',
        '![全36层条件与单独作用](figures/complement-conditional-trajectories.png)','',
        '图中读数在答案前位置，最终输出头对中间状态的投影并非每层已完成的最终决定。保留所有36层，20—28底色只沿用旧观察窗口。所有轨迹面板独立纵轴。','',
        '![组合差新增与RMS](figures/interaction-increments-and-rms.png)','',
        '新增差同时受注意力/MLP分支和已有残差的RMS归一化缩放影响；注意力分支不是注意力关注量。这些曲线本身没有新增分支、头或层干预，不能据此命名因果模块。','',
        '| 查询 | 接收条件 | U/C/W token数 | U扰动L2 | C扰动L2 | W扰动L2 |','|---|---|---|---:|---:|---:|']
    for r in data['query_interactions']:
        doc.append(f'| {r["query_id"]} | {r["recipient"].split("-")[-1]} | {r["focal_tokens"]}/{r["complement_tokens"]}/{r["whole_tokens"]} | {r["focal_l2"]:.6f} | {r["complement_l2"]:.6f} | {r["whole_l2"]:.6f} |')
    doc+=['','U/C位置不重叠且恰好覆盖W，扰动平方范数可相加是几何事实，不要求最终分数可相加。两组并未匹配token数、范数或词性；C混合多个位置，不能直接归因于某个后文词或立场表示。两定义长度相差18token，位置/长度混杂仍保留。注入组合状态可能偏离自然前向，I是本干预条件下的有限交互，不能当作自然通路的分摊比例。','',
        f'工程分数界为{data["margin_error_bound"]:.9g}；两个分数差界2ε，I的四项界4ε={4*data["margin_error_bound"]:.9g}。I有{sum(r["interaction_resolved"] for r in data["query_interactions"])}/12项可在该工程界外分辨，但这不是统计显著性或实际重要性的阈值。中间单logit投影界{data["probe_logit_error_bound"]:.9g}。','',
        f'沿用纯CPU固定m+7对照，主方向正确{data["CPU_score_shift"]["correct"]}/6；没有新拟合或模型前向，来源仍是旧暴露开发结果上的事后诊断，不是独立方法验证。','',
        '本轮通常594次前向，48自身控制与66格式端点；正常释放后独立核查全部分数/轨迹/交互，18原生+36旧U/P/W完整向量与轨迹和18完整状态银行必须精确重放上一轮。旧数据不代替新结果。所有六条、双向、例外及无收益结果都保留，没有按新结果筛层/词/条件。','',
        '[全部干预](all-interventions.tsv) · [交互和四格分数](query-interactions.tsv) · [全部36层轨迹](all-trajectories.tsv) · [原生分数](baselines.tsv) · [结构化结果](results.json) · [协议](../prepared-01/PROTOCOL.md)','',
        '本轮原文：','']
    for q in c.IDS:
        r=next(r for r in data['requests'] if r['query_id']==q);s=next(s for s in r['spans'] if s['id']=='query')
        doc.append(q+'（参考'+score(q,'D01')['reference']+'）：'+r['prompt_text'][s['char_start']:s['char_end']]+'\n')
    (Path(out)/'REPORT.md').write_text('\n'.join(doc)+'\n')
