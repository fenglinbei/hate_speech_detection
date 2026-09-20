"""CPU-only conditional-restoration contrasts, trajectories, and readable report."""
from __future__ import annotations
import csv,itertools
from pathlib import Path
import numpy as np
from diagnostics import hehe_transfer_inputs_v1 as c
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
    from diagnostics import hehe_transfer_runtime_v1 as rt
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
        delta=banks[did][job['layer'],inds].astype(np.float64)-banks[rid][job['layer'],inds].astype(np.float64)
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
    data={'schema':'hehe-transfer-results/v1','status':'complete','layers':36,'patch_layer':17,
        'requests':reqs,'baselines':list(scores.values()),'effects':effects,'restoration_contrasts':contrasts,'joint_contrasts':joint,
        'self_controls':controls,'native_trajectories':native,'patched_trajectories':trajectories,
        'margin_error_bound':bound,'probe_logit_error_bound':probe_bound,'jobs':jobs,
        'position_differences':position_pairs,'condition_gaps':gaps,
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
    figures(out,data);write_report(out,data,run)
    c.write(out/'audit.json',{'status':'pass','CPU_reconstructed':True,'query_references_joined_after_release':True,
        'prepared_manifest':c.info(p/'manifest.json'),'raw_seal':c.info(run/'raw-seal.json'),
        'release':c.read(run/'state.json')['resource_release'],'independent_samples_claimed':False})
    c.write(out/'manifest.json',{'artifacts':[c.info(x) for x in sorted(out.iterdir()) if x.is_file()]})
    return {'output':str(out),'native_inputs':len(reqs),'interventions':len(jobs),'restoration_contrasts':len(contrasts),'joint_contrasts':len(joint),'self_controls':len(selfs),'position_differences':len(position_pairs),'condition_gaps':len(gaps),'trajectory_layers':36}


def direction(e):return '普通义→原义' if e['donor'].endswith('D02') else '原义→普通义'


NOTES=[
    '四条已采用新原文、八个输入；多个干预方向不是独立样本。',
    'T01/T02参考无；T03按用户要求继承原标签有；T04参考有，单列全文其他攻击的边界。',
    'm=无logit−有logit；正向对T01/T02有利、对T03/T04不利。',
    '固定17层焦点及前置位置，26注意力/28MLP单独与联合恢复；不重新挑层。',
    '注意力恢复指o_proj后整个子层向量，不是注意力热图权重。',
    '全部来源来自同次新运行，未代入旧实验分数或向量。',
    '前置对照并非词性或向量大小匹配，也不预设为零。',
    '工程界不是统计置信区间；全向量恢复比例不是独立中介份额。']


def figures(out,data):
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
    fig.text(.5,.01,'m=无logit−有logit；T01/T02参考无，T03/T04参考有。零线为原生接收方；各面板独立纵轴。',ha='center')
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
        for ax in axes[row]:axes_style(ax);ax.set_xlabel('层号（0起）');ax.set_ylabel('D02普通义 − D01原义')
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


def write_report(out,data,run):
    es=data['effects'];cs=data['restoration_contrasts'];js=data['joint_contrasts']
    scores={(s['query_id'],s['dictionary_id']):s for s in data['baselines']}
    ordinary=[q for q in ['T01','T02'] if (q,'D01') in scores]
    reproduced=sum(scores[q,'D01']['raw_prediction']=='有' and scores[q,'D02']['raw_prediction']=='无' for q in ordinary)
    stronger=sum(abs(e['delta_m'])>abs(next(p['delta_m'] for p in es if p['recipient']==e['recipient'] and p['condition']=='preceding'))+4*data['margin_error_bound'] for e in es if e['condition']=='upstream')
    reduced=sum(j['absolute_reduced_vs_A'] and j['absolute_reduced_vs_B'] for j in js)
    state=c.read(run/'state.json');elapsed=sum(i['ended_at_unix']-i['started_at_unix'] for i in state['invocations'])
    def fmt(x):return 'NA' if x is None else f'{x:+.4f}'
    def pct(x):return 'NA' if x is None else f'{100*x:.1f}%'
    lines=['# 四条新案例：固定层机制能否延伸？','',
        f'两条普通笑声案例中，{reproduced}/2条重现“原释义答有、普通义答无”的行为模式。'
        f'全部8个替换方向中，焦点效应绝对值比前置对照大且超出工程界的有{stronger}个；联合恢复比两种单独恢复都进一步减小绝对效应的有{reduced}个。'
        '这些是四条材料的具体结果，不是总体成功率，也不能仅凭这些计数确认同一条自然词义路径。','',
        '## 比较对象','',
        'T01/T02为普通笑声（参考无）；T03是#7050，按用户明确要求继承原标签有；T04是笑声之外另有攻击（参考有）的边界。'
        'T03的二元参考已经采用，但这不是新增族群身份、严重度或纯词义裁决。T04不算第二条旧Q02同类复现。','',
        '| 查询 | 完整原文 | 参考 |','|---|---|---|']
    for q in dict.fromkeys(r['query_id'] for r in data['requests']):
        req=next(r for r in data['requests'] if r['query_id']==q);span=next(s for s in req['spans'] if s['id']=='query')
        text=req['prompt_text'][span['char_start']:span['char_end']]
        lines.append(f'| {q} | {text} | {scores[q,"D01"]["reference"]} |')
    lines += ['', '同一查询只替换嘿嘿的词典释义：D01为原侮辱义，D02为普通笑声义。其余五词、当前任务、Qwen3-8B、无示例设置保持一致。'
        '四条真实语料未改写，没有依据模型输出筛掉案例。', '',
        '## 原生输出与条件差距','',
        'm=z(无)−z(有)，衡量输出更偏哪一个字；正向对T01/T02有利、对T03/T04不利。下表分数是原始logit差，不是概率。','',
        '| 查询 | D01输出 / m | D02输出 / m | 普通义−原义 | 参考对齐变化 |','|---|---:|---:|---:|---:|']
    for g in data['condition_gaps']:
        q=g['query_id'];a,b=scores[q,'D01'],scores[q,'D02']
        lines.append(f'| {q} | {a["raw_prediction"]} / {a["m"]:+.4f} | {b["raw_prediction"]} / {b["m"]:+.4f} | {g["delta_m_D02_minus_D01"]:+.4f} | {g["reference_aligned_delta"]:+.4f} |')
    lines += ['', '![原生条件差距与新增差距](condition-gaps.png)','',
        '答案方向投影使用最终RMSNorm和输出头读取中间状态，测量它当时偏向有还是无，不表示该层已经作决定。'
        '左列是答案前位置的层末条件差；中、右列分别展示注意力与MLP的每层新增差距，以及分支投影和残差重缩放分解。'
        '层号从0开始，保留全部36层；不从新结果里另挑干预层。','',
        '## 固定第17层：焦点与前置对照','',
        'U把另一释义条件下第17层嘿嘿的完整向量放进当前运行，P只替换紧邻前置位置。T01/T02/T04前置为逗号，T03为你的；均一个token。'
        '对照可以受词典影响，并非词性或向量大小匹配，不能预设为零。方向普通义→原义表示供体D02、接收方D01。','',
        '| 查询/方向 | U相对原生效应 / 输出 | P相对原生效应 / 输出 | U−P |','|---|---:|---:|---:|']
    for pair in data['position_differences']:
        u=next(e for e in es if e['job_id']==pair['focal_job_id']);p=next(e for e in es if e['job_id']==pair['pre_job_id'])
        lines.append(f'| {pair["query_id"]}/{direction(pair)} | {u["delta_m"]:+.4f} / {u["prediction"]} | {p["delta_m"]:+.4f} / {p["prediction"]} | {pair["delta_m"]:+.4f} |')
    lines += ['', '![全部最终干预效应](endpoint-effects.png)','',
        'U/P/A/B/AB柱高都是相对各自原生接收方N的差值。正值不统一代表修复；标签是否翻转还取决于距离决策零点有多远。', '',
        '## 下游单独与联合恢复','',
        'A在U基础上把答案前26层注意力输出恢复成N的向量；B独立恢复28层MLP；AB同时依次恢复两处。'
        '这里的注意力输出是o_proj后、残差相加前的整个4096维向量，与注意力热图权重不同。两处来源始终是同次运行的N。','',
        '| 查询/方向 | U效应 | A剩余 | B剩余 | AB剩余 | A/B/AB移除比例 |','|---|---:|---:|---:|---:|---|']
    for j in js:
        u,a,b,ab=[next(e for e in es if e['job_id']==j[k]) for k in ['upstream_job_id','A_job_id','B_job_id','job_id']]
        rr=[next(r for r in cs if r['job_id']==e['job_id']) for e in [a,b,ab]]
        lines.append(f'| {j["query_id"]}/{direction(j)} | {u["delta_m"]:+.4f} | {a["delta_m"]:+.4f} | {b["delta_m"]:+.4f} | {ab["delta_m"]:+.4f} | '+ ' / '.join(pct(r['removed_fraction']) for r in rr)+' |')
    lines += ['', '移除比例=(U−恢复后)/(U−N)。负值、超过100%和无效结果均保留；分母靠近工程界时记NA。'
        '尤其小U效应下，比例很大不代表绝对作用很大，应先看分数差。移除比例不能相加成独立中介贡献。','',
        '| 查询/方向 | AB−A | AB−B | 交互I=AB−A−B＋U | 比两处单独都进一步减弱 |','|---|---:|---:|---:|---|']
    for j in js:
        lines.append(f'| {j["query_id"]}/{direction(j)} | {j["AB_minus_A"]:+.4f} | {j["AB_minus_B"]:+.4f} | {j["interaction_m"]:+.4f} | {"是" if j["absolute_reduced_vs_A"] and j["absolute_reduced_vs_B"] else "否"} |')
    lines += ['', '![全部层的干预轨迹](remaining-trajectories.png)','',
        '![联合与单独恢复差距](joint-comparisons.png)','',
        '交互量衡量当前m尺度上偏离简单相加的部分；非零不能单独确定串行中介、冗余或后续补偿。'
        'AB在26注意力前与U相同、28MLP前与A相同，是操作边界核查，不作为额外的机制发现。全部RMS分解和浮点余项保留在JSON/TSV。','',
        '## 验证与结论范围','',
        '64个自身控制、采集开关、重复/逆序、左右padding、真实前缀、正式重放，以及48个单标签后EOS端点通过原门槛。'
        '所有原生、干预与恢复来源均新计算，未代入旧分数；工作进程正常退出并核实释放后才合并参考标签。','',
        f'通常且本轮完整协议要求456次前向；两阶段工作耗时合计{elapsed:.2f}秒。m工程界{data["margin_error_bound"]:.12g}，候选投影界{data["probe_logit_error_bound"]:.12g}。'
        '这些误差界反映工程精度，不是统计置信区间。独立120位分数与扩展精度轨迹审计通过后才科学封存。','',
        '固定层不出现旧案例的效应，也是本轮需要保留的结果。全向量替换可能混合不同上下文状态，未隔离纯词义变量；'
        '新案例的反例不能自动归为层号不对，也不能为追求复现而更换层或材料。当前四条内容筛选案例不支持总体频率估计。','',
        '[完整输入](../prepared-01/ALL-PROMPTS.md) · [预先协议](../prepared-01/PROTOCOL.md) · [原生分数](baselines.tsv) · [全部干预](all-interventions.tsv) · '
        '[前置比较](position-differences.tsv) · [恢复比较](restoration-contrasts.tsv) · [联合交互](joint-contrasts.tsv) · '
        '[原生条件全层差距](condition-gap-trajectories.tsv) · [干预全层TSV](all-trajectories.tsv) · [完整JSON](results.json) · [独立审计](../result-audit-01.json)','']
    (out/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
