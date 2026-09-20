"""CPU-only conditional-restoration contrasts, trajectories, and readable report."""
from __future__ import annotations
import csv,itertools
from pathlib import Path
import numpy as np
from diagnostics import jingba_attn_restore_inputs_v1 as c
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
    from diagnostics import jingba_attn_restore_runtime_v1 as rt
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
    data={'schema':'jingba-attn-restore-results/v1','status':'complete','layers':36,'patch_layer':17,
        'requests':reqs,'baselines':list(scores.values()),'effects':effects,'restoration_contrasts':contrasts,'joint_contrasts':joint,
        'self_controls':controls,'native_trajectories':native,'patched_trajectories':trajectories,
        'margin_error_bound':bound,'probe_logit_error_bound':probe_bound,'jobs':jobs,
        'position_differences':position_pairs,'condition_gaps':gaps,'dictionary_addition_gaps':additions,
        'reference_provenance':c.read(p/'analysis-references.json'),
        'synthetic':bool(c.read(p/'manifest.json').get('synthetic',False)),'notes':NOTES,'shared_prefix_comparison':shared_prefix_comparison(run,reqs,banks)}
    data['context_restoration']=context_restoration(data)
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
    '六条既有审核构造材料原样复用，不是新语料、独立抽样或总体确认。',
    'J05/J06宠物、J07/J08反对辱称，参考无；J09/J10实施或认可攻击，参考有。',
    'm=无logit−有logit；正向变化是否有益取决于参考，不是概率。',
    'U是第17层目标词替换；R仅进一步恢复答案前第26层注意力分支为新计算的接收条件原生输出。',
    '恢复原生分支不等同于修复错误。D00仅原生基线；没有层/头扫描或28层干预。',
    '供体/接收方分别映射；释义长度差18token；前置对照不匹配词性/范数。',
    '投影不是注意力权重，包含RMS归一化影响；全36层和两方向全部保留。',
    'J08/J10后文立场、措辞、长度同时变化；不是纯立场单因素对照。',
    '消减比例描述条件性分数效应，不是可加中介份额、准确率或通用修复能力。',
    '工程误差界不是统计置信区间；旧结果仅在正常释放后比对，绝不代入本轮计算。']


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



def context_restoration(data):
    effects=data['effects'];tr=data['patched_trajectories'];bound=data['margin_error_bound'];rows=[]
    for d in ['D01','D02']:
        def e(q,k):return next(e for e in effects if e['query_id']==q and e['recipient'].endswith(d) and e['condition']==k)
        a,b,ra,rb=[e(q,k) for q,k in [('J08','upstream'),('J10','upstream'),('J08','restore-L26-attention'),('J10','restore-L26-attention')]]
        cu=a['delta_m']-b['delta_m'];cr=ra['delta_m']-rb['delta_m']
        changed=(ra['m']-a['m'])-(rb['m']-b['m'])
        ts=[tr[z['job_id']]['minus_recipient'] for z in [a,b,ra,rb]]
        rows.append(dict(recipient_dictionary=d,direction=direction(a),J08_upstream=a['job_id'],J10_upstream=b['job_id'],
            J08_restored=ra['job_id'],J10_restored=rb['job_id'],upstream_context_difference=cu,
            remaining_context_difference=cr,change_due_to_restoration=changed,
            context_difference_bound=4*bound,change_bound=4*bound,change_resolved=abs(changed)>4*bound,
            absolute_context_difference_reduced=abs(cr)<abs(cu)-8*bound,
            absolute_context_difference_increased=abs(cr)>abs(cu)+8*bound,
            upstream_trajectory_difference=difference(ts[0],ts[1]),remaining_trajectory_difference=difference(ts[2],ts[3]),
            change_trajectory_difference=difference(difference(ts[2],ts[0]),difference(ts[3],ts[1]))))
    return rows


def figures(out,data):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import fontManager,FontProperties
    font='/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc';fontManager.addfont(font)
    plt.rcParams.update({'font.family':[FontProperties(fname=font).get_name(),'DejaVu Sans'],
        'axes.unicode_minus':False,'font.size':9,'svg.fonttype':'none'})
    folder=Path(out)/'figures';folder.mkdir();x=np.arange(36)
    effects=data['effects'];tr=data['patched_trajectories'];prefix='CPU SYNTHETIC / 非科研结果：' if data['synthetic'] else ''
    def eff(q,d,k):return next(e for e in effects if e['query_id']==q and e['recipient'].endswith(d) and e['condition']==k)
    def finish(fig,name):
        fig.tight_layout(rect=(0,0,1,.96));fig.savefig(folder/(name+'.png'),dpi=140);fig.savefig(folder/(name+'.svg'));plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(14,7))
    for ax,q in zip(axes.flat,c.IDS):
        vals=[eff(q,d,k)['delta_m'] for d in ['D01','D02'] for k in ['upstream','restore-L26-attention']]
        ax.bar(range(4),vals,color=['#376ca8','#e39335']*2);ax.axhline(0,color='black',lw=.6)
        ax.set_xticks(range(4),['普通→贬损 U','普通→贬损 R','贬损→普通 U','贬损→普通 R'],rotation=20,ha='right')
        ax.set_title(q);ax.set_ylabel('相对接收原生的分数变化 m−N')
        for i,v in enumerate(vals):ax.annotate(f'{v:+.3f}',(i,v),xytext=(0,4 if v>=0 else -13),textcoords='offset points',ha='center',fontsize=8)
    fig.suptitle(prefix+'恢复第26层注意力分支后，目标词替换效应如何变化（各面板独立纵轴）');finish(fig,'restoration-effects')
    fig,axes=plt.subplots(6,2,figsize=(14,17))
    for row,q in enumerate(c.IDS):
        for col,d in enumerate(['D01','D02']):
            ax=axes[row,col]
            for k,label,color in [('upstream','U−N','#376ca8'),('restore-L26-attention','R−N','#e39335'),('preceding','P−N','#999999')]:
                t=tr[eff(q,d,k)['job_id']]['minus_recipient'];ax.plot(x,np.asarray(t['probe_pre_mid_post'])[:,2],label=label,color=color)
            ax.axhline(0,color='black',lw=.5);ax.axvspan(20,28,color='#eef1f3',zorder=0)
            ax.axvline(17,color='gray',ls=':');ax.axvline(26,color='#c08025',ls=':');ax.set_xlim(0,35)
            ax.set_title(q+' '+direction(eff(q,d,'upstream')));ax.set_ylabel('层末答案方向投影差');ax.set_xlabel('层号（0起）')
    axes[0,0].legend();fig.suptitle(prefix+'答案前完整36层：各面板独立纵轴；投影含RMS归一化影响');finish(fig,'restoration-trajectories')
    fig,axes=plt.subplots(2,2,figsize=(14,8))
    for i,r in enumerate(data['context_restoration']):
        for key,label,color in [('upstream_trajectory_difference','替换后 CU','#376ca8'),('remaining_trajectory_difference','恢复后 CR','#e39335')]:
            axes[i,0].plot(x,np.asarray(r[key]['probe_pre_mid_post'])[:,2],label=label,color=color)
        t=r['change_trajectory_difference']
        for bi,label in enumerate(['注意力','MLP']):axes[i,1].plot(x,np.asarray(t['increment_attention_mlp'])[:,bi],label=label+'步骤新增差')
        for ax in axes[i]:
            ax.axhline(0,color='black',lw=.5);ax.axvline(26,color='#c08025',ls=':');ax.set_xlim(0,35);ax.set_xlabel('层号（0起）');ax.legend(fontsize=8)
        axes[i,0].set_title(r['direction']+'：J08效应−J10效应');axes[i,1].set_title('恢复造成的差异变化 CR−CU，按步骤分解')
    fig.suptitle(prefix+'共享目标词状态，不同后文：响应差距是否由该分支承接（独立纵轴）');finish(fig,'paired-context-restoration')
    fig,axes=plt.subplots(6,2,figsize=(14,17))
    for row,q in enumerate(c.IDS):
        for col,d in enumerate(['D01','D02']):
            ax=axes[row,col];r=next(r for r in data['restoration_contrasts'] if r['query_id']==q and r['recipient'].endswith(d));t=r['trajectory_difference']
            for bi,branch in enumerate(['注意力','MLP']):
                ax.plot(x,np.asarray(t['branch_projection_at_destination_scale'])[:,bi],label=branch+'分支')
                ax.plot(x,np.asarray(t['existing_residual_rescaling'])[:,bi],ls='--',label=branch+'残差缩放')
            ax.axhline(0,color='black',lw=.5);ax.axvline(26,color='gray',ls=':');ax.set_xlim(0,35)
            ax.set_title(q+' '+direction(eff(q,d,'upstream'))+' R−U');ax.set_xlabel('层号（0起）')
    axes[0,0].legend(fontsize=7);fig.suptitle(prefix+'恢复造成的投影变化：分支与RMS缩放分解（独立纵轴，不是因果份额）');finish(fig,'restoration-rms')


def write_report(out,data,run):
    cs=data['restoration_contrasts'];effects=data['effects'];scores=data['baselines']
    reduced=sum(r['absolute_effect_reduced'] for r in cs);increased=sum(r['absolute_effect_increased'] for r in cs)
    ordinary=[r for r in cs if r['query_id'] in ['J05','J06']];oreduced=sum(r['absolute_effect_reduced'] for r in ordinary)
    flips=[r for r in cs if r['restored_prediction']!=r['upstream_prediction']]
    label='CPU SYNTHETIC：以下全部为合成测试，禁止解释为科研结果。\n\n' if data['synthetic'] else ''
    doc=['# 京巴：第26层注意力分支恢复','',label+f'在第17层“京巴”状态替换后，恢复答案前第26层注意力分支，使12个方向中的{reduced}个效应绝对值减小、{increased}个增大。两条普通犬种查询J05/J06的四个方向中有{oreduced}个减小；恢复前后有{len(flips)}个标签改变。这里只判断该分支是否参与传递此前干预的影响，不能直接等同于修复误判。','',
        'J05/J06谈宠物，J07/J08反对地域辱称，参考答案都是“无”；J09自己实施辱称，J10认可他人辱称，参考为“有”。沿用六条已经审核的构造材料，任务、标签、释义和全文不变。D00无词典，D01提供地域贬损义，D02提供普通犬种义；不提供示例。','',
        'N是当前接收条件原生运行。U把另一释义条件第17层“京巴”两个token的完整状态放进接收条件。R在U基础上，将答案前第26层注意力合并后的完整输出恢复为N在同处的输出，然后继续计算。P是第17层紧邻前置位置的等token数替换。全部状态在本轮重新计算，两种方向和全部36层均保留，层号从0开始。','',
        'm是“无”与“有”的logit之差：正数偏向无，负数偏向有，并不是概率。比较U−N与R−N是在问：恢复该分支后，原先替换造成的影响还剩多少。恢复为原生值未必更正确，因为原生判断本身可能有误。','',
        '| 查询 | 参考 | 替换方向 | N分数 | U分数 | R分数 | U−N | R−N | 有向效应消减 | N/U/R标签 |',
        '|---|---|---|---:|---:|---:|---:|---:|---:|---|']
    for r in cs:
        e=next(e for e in effects if e['job_id']==r['job_id']);fraction='NA' if r['removed_fraction'] is None else f'{100*r["removed_fraction"]:.1f}%'
        doc.append(f'| {r["query_id"]} | {r["reference"]} | {direction(e)} | {r["native_m"]:+.3f} | {r["upstream_m"]:+.3f} | {r["restored_m"]:+.3f} | {r["upstream_delta_m"]:+.6f} | {r["remaining_delta_m"]:+.6f} | {fraction} | {r["native_prediction"]}/{r["upstream_prediction"]}/{r["restored_prediction"]} |')
    doc+=['','消减比例=(U−R)/(U−N)，保留符号，不裁剪为0—100%。负数表示相对原效应进一步扩大，超过100%表示跨过原生分数；不是准确率或可相加的中介份额。小效应对应的大比例仍须结合绝对分数阅读。','',
        '![恢复后的干预效应](figures/restoration-effects.png)','',
        'J08和J10在“京巴”之前共享完整前缀，后文分别反对、认可辱称。下表比较两条的替换效应差：先计算各自相对N的变化，再做J08减J10。若恢复后差距缩小，支持这处注意力输出参与承接不同后文下的响应差异；若仍有差距或扩大，说明该处不能独自解释这种差异。','',
        '| 方向 | 恢复前差距 CU | 恢复后差距 CR | 恢复引起的变化 | 绝对差距减小 |','|---|---:|---:|---:|---|']
    for r in data['context_restoration']:doc.append(f'| {r["direction"]} | {r["upstream_context_difference"]:+.6f} | {r["remaining_context_difference"]:+.6f} | {r["change_due_to_restoration"]:+.6f} | {"是" if r["absolute_context_difference_reduced"] else "否"} |')
    doc+=['','二者后文的措辞、长度和答案位置也不同，不能将差距直接归结为“正确识别作者立场”。共享前缀与全层局部状态的重新核验如下。','',
        '| 词典条件 | 真正截到目标词的前缀状态36层相同 | 完整prompt焦点状态36层相同 |','|---|---|---|']
    for r in data['shared_prefix_comparison']:doc.append(f'| {r["dictionary_id"]} | {r["true_prefix"]["focal"]["byte_equal"]} | {r["full_prompt"]["focal"]["byte_equal"]} |')
    doc+=['','![配对语境响应差](figures/paired-context-restoration.png)','',
        '答案方向投影是用最终输出头读取答案前中间状态，观察偏向有或无；它不表示该层已经完成判断。完整轨迹如下，展示层末相对N的投影差。26层之前R与U应一致，这是干预实施位置决定的结构性质，不能据此说差异天然“起源于26层”。','',
        '![36层轨迹](figures/restoration-trajectories.png)','',
        '所有图的面板使用独立纵轴，横向比较前请看刻度。后续层可能补偿、扩大或反转前面的变化；RMS归一化对已有残差的缩放也会改变投影，分解如下。注意力分支输出与注意力权重是不同读数。','',
        '![分支与归一化](figures/restoration-rms.png)','',
        f'工程检查保留48个自身控制和54个单token有/无后EOS端点；m的工程误差界为{data["margin_error_bound"]:.9g}，单logit中间投影界为{data["probe_logit_error_bound"]:.9g}。差值与比例按共享分数依赖计算边界，不是统计置信区间。独立高精度复核、上轮42端点完整向量与轨迹一致性由关闭记录另行证明，不能用报告本身的自检代替。','',
        '结论范围：这是一项在固定第17层替换之上的条件性恢复实验。它能检验第26层注意力完整输出在此干预效应中的作用，不能证明唯一或必要的自然语义通路，也未识别具体注意力头。D01/D02相差18token，前置对照不是范数或词性匹配；六条是相关构造材料。所有反例和两方向保留，没有新增头/层扫描。','',
        '[全部干预](all-interventions.tsv) · [恢复对比](restoration-contrasts.tsv) · [36层差值](all-trajectories.tsv) · [原生分数](baselines.tsv) · [完整结构化数据](results.json)','',
        '全文材料：','']
    for q in c.IDS:
        r=next(r for r in data['requests'] if r['query_id']==q);span=next(s for s in r['spans'] if s['id']=='query')
        ref=next(s['reference'] for s in scores if s['query_id']==q)
        doc.append(q+'（参考'+ref+'）：'+r['prompt_text'][span['char_start']:span['char_end']]+'\n')
    (Path(out)/'REPORT.md').write_text('\n'.join(doc)+'\n',encoding='utf-8')
