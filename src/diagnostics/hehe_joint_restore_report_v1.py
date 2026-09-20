"""CPU-only conditional-restoration contrasts, trajectories, and readable report."""
from __future__ import annotations
import csv,itertools
from pathlib import Path
import numpy as np
from diagnostics import hehe_joint_restore_inputs_v1 as c
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
    from diagnostics import hehe_joint_restore_runtime_v1 as rt
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
    data={'schema':'hehe-joint-restoration-results/v1','status':'complete','layers':36,'patch_layer':17,
        'requests':reqs,'baselines':list(scores.values()),'effects':effects,'restoration_contrasts':contrasts,'joint_contrasts':joint,
        'self_controls':controls,'native_trajectories':native,'patched_trajectories':trajectories,
        'margin_error_bound':bound,'probe_logit_error_bound':probe_bound,'jobs':jobs,
        'notes':['四条曝光材料，十六配置不是独立样本。','注意力指o_proj后整个子层输出，不是注意力权重。',
            '固定第17层跨条件替换后，新增加26注意力与28MLP联合恢复；重测两处单独恢复比较。',
            '恢复来源均为同次运行原生接收方；条件自身控制取同次上游运行。',
            'm=无logit−有logit；正向对Q01有利、对Q02不利。',
            '分数区间为工程界传播，不是统计置信区间；恢复比例不可相加。',
            '低效应可以伴随冗余、绕行或补偿，不证明自然计算中该组件无作用。']}
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
    figures(out,data);write_report(out,data,run)
    c.write(out/'audit.json',{'status':'pass','CPU_reconstructed':True,'query_references_joined_after_release':True,
        'prepared_manifest':c.info(p/'manifest.json'),'raw_seal':c.info(run/'raw-seal.json'),
        'release':c.read(run/'state.json')['resource_release'],'independent_samples_claimed':False})
    c.write(out/'manifest.json',{'artifacts':[c.info(x) for x in sorted(out.iterdir()) if x.is_file()]})
    return {'output':str(out),'native_inputs':4,'interventions':16,'restoration_contrasts':12,'joint_contrasts':4,'self_controls':28,'trajectory_layers':36}


def direction(e):return '普通义→原义' if e['donor'].endswith('D02') else '原义→普通义'


def figures(out,data):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import fontManager,FontProperties
    font='/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc';fontManager.addfont(font)
    plt.rcParams.update({'font.family':[FontProperties(fname=font).get_name(),'DejaVu Sans'],
        'axes.unicode_minus':False,'font.size':10,'svg.fonttype':'none'})
    x=np.arange(36);effects=data['effects'];tr=data['patched_trajectories'];contrasts=data['restoration_contrasts']
    upstream=[e for e in effects if e['restoration'] is None]
    styles=[('upstream','#147d91','-','仅17层替换U'),('restore-L26-attention','#cc7041','--','恢复26注意力A'),('restore-L28-mlp','#8058a1','-.','恢复28MLP B'),('restore-joint','#263f50','-','两处联合恢复AB')]
    def save(fig,name):
        for ext in ['png','pdf','svg']:fig.savefig(out/f'{name}.{ext}',dpi=175,bbox_inches='tight')
        plt.close(fig)
    def base(ax):
        ax.axhline(0,color='#85939b',lw=.7);ax.axvline(17,color='#999',ls=':',lw=1)
        ax.axvline(26,color='#cc7041',alpha=.5,ls=':',lw=1);ax.axvline(28,color='#8058a1',alpha=.5,ls=':',lw=1)
        ax.set_xlim(-.5,35.5);ax.set_xticks([0,8,17,20,23,26,28,32,35]);ax.grid(axis='y',alpha=.15)
        ax.spines[['top','right']].set_visible(False)
    fig,axes=plt.subplots(2,2,figsize=(13,8))
    for ax,u in zip(axes.flat,upstream):
        for i,(condition,color,ls,label) in enumerate(styles):
            e=next(e for e in effects if e['recipient']==u['recipient'] and e['condition']==condition)
            ax.bar(i,e['delta_m'],color=color,width=.6)
            ax.annotate(f'{e["delta_m"]:+.3f}\n输出{e["prediction"]}',(i,e['delta_m']),xytext=(0,5 if e['delta_m']>=0 else -5),
                textcoords='offset points',ha='center',va='bottom' if e['delta_m']>=0 else 'top',fontsize=9)
        ax.axhline(0,color='#78828b',lw=.7);ax.set_xticks(range(4),['仅17层替换U','恢复26注意力A','恢复28MLP B','两处联合AB'],fontsize=9)
        ax.set_title(f'{u["query_id"]} / {direction(u)}');ax.set_ylabel('最终m − 原生接收方m');ax.margins(y=.25)
        ax.spines[['top','right']].set_visible(False)
    fig.suptitle('两处联合恢复，比单独恢复多削弱多少上游效应？',fontsize=14)
    fig.text(.5,.01,'m=无logit−有logit；零线为原生接收方。参考Q01无、Q02有；正值不统一代表修复。各面板独立纵轴。',ha='center',fontsize=9)
    fig.tight_layout(rect=[0,.035,1,.95]);save(fig,'endpoint-effects')
    for mode in ['remaining','restore-minus-upstream']:
        fig,axes=plt.subplots(2,2,figsize=(13,8),sharex=True)
        for ax,u in zip(axes.flat,upstream):
            for condition,color,ls,label in styles:
                if mode!='remaining' and condition=='upstream':continue
                e=next(e for e in effects if e['recipient']==u['recipient'] and e['condition']==condition)
                d=tr[e['job_id']]['minus_recipient'] if mode=='remaining' else next(r['trajectory_difference'] for r in contrasts if r['job_id']==e['job_id'])
                ax.plot(x,np.asarray(d['probe_pre_mid_post'])[:,2],color=color,ls=ls,lw=1.8,label=label)
            base(ax);ax.set_title(f'{u["query_id"]} / {direction(u)}');ax.legend(frameon=False,fontsize=8)
            ax.set_ylabel('答案前层末投影差');ax.set_xlabel('层索引（0起，全部36层）')
        comparison='各干预 − 原生接收方' if mode=='remaining' else '分支恢复后 − 仅17层替换'
        fig.suptitle(f'逐层答案方向投影：{comparison}',fontsize=14)
        fig.text(.5,.01,'使用最终RMSNorm与输出头，读数受归一化影响，不表示该层已作出最终决定；各面板独立纵轴。',ha='center',fontsize=9)
        fig.tight_layout(rect=[0,.035,1,.95]);save(fig,mode+'-trajectories')
    for condition in ['restore-L26-attention','restore-L28-mlp','restore-joint']:
        fig,axes=plt.subplots(4,2,figsize=(13,13),sharex=True)
        for row,u in enumerate(upstream):
            r=next(r for r in contrasts if r['recipient']==u['recipient'] and r['condition']==condition)
            for bi,label in enumerate(['注意力','MLP']):
                ax=axes[row,bi];d=r['trajectory_difference']
                for key,color,ls,title in [('increment_attention_mlp','#173e58','-','观测投影增量'),('branch_projection_at_destination_scale','#258b8a','-','新增分支投影'),('existing_residual_rescaling','#bd6245','--','已有残差重缩放')]:
                    ax.plot(x,np.asarray(d[key])[:,bi],color=color,ls=ls,label=title)
                base(ax);ax.set_title(f'{u["query_id"]}/{direction(u)} / {label}');ax.legend(frameon=False,fontsize=8)
                ax.set_ylabel('恢复后 − 仅上游替换');ax.set_xlabel('层索引（0起）')
        fig.suptitle(condition+'：全部36层增量及RMS分解（非独立因果份额）',fontsize=13)
        fig.tight_layout(rect=[0,0,1,.97]);save(fig,condition+'-normalization')
    for field,name,title in [('trajectory_AB_minus_A','joint-minus-A','已有26注意力恢复后，再恢复28MLP：AB−A'),
        ('trajectory_AB_minus_B','joint-minus-B','联合恢复与只恢复28MLP：AB−B'),
        ('trajectory_interaction','joint-interaction','偏离简单相加：AB−A−B＋U')]:
        fig,axes=plt.subplots(4,3,figsize=(16,13),sharex=True)
        for row,j in enumerate(data['joint_contrasts']):
            d=j[field]
            ax=axes[row,0];ax.plot(x,np.asarray(d['probe_pre_mid_post'])[:,2],color='#263f50')
            ax.set_title(f'{j["query_id"]}/{direction(j)} / 层末投影差');ax.set_ylabel('最终RMSNorm/输出头诊断')
            for bi,label in enumerate(['注意力','MLP']):
                ax=axes[row,bi+1]
                for key,color,ls,caption in [('increment_attention_mlp','#173e58','-','观测增量'),('branch_projection_at_destination_scale','#258b8a','-','分支投影'),('existing_residual_rescaling','#bd6245','--','RMS重缩放')]:
                    ax.plot(x,np.asarray(d[key])[:,bi],color=color,ls=ls,label=caption)
                ax.set_title(f'{j["query_id"]}/{direction(j)} / {label}');ax.legend(frameon=False,fontsize=8)
            for ax in axes[row]:base(ax);ax.set_xlabel('层号（0起，全部36层）')
        fig.suptitle(title+'；答案前位置，原始投影的差值，非独立因果份额',fontsize=14)
        fig.tight_layout(rect=[0,0,1,.97]);save(fig,name)


def write_report(out,data,run):
    es=data['effects'];cs=data['restoration_contrasts'];js=data['joint_contrasts']
    baselines={r['request_id']:r for r in data['baselines']}
    elapsed=sum(i['ended_at_unix']-i['started_at_unix'] for i in c.read(run/'state.json')['invocations'])
    reduced=sum(j['absolute_reduced_vs_A'] and j['absolute_reduced_vs_B'] for j in js)
    resolved=sum(j['interaction_resolved'] for j in js)
    lines=['# 两处联合恢复：是否比单独恢复进一步削弱上游效应？','',
        f'四个方向中，联合恢复有{reduced}个方向比两种单独恢复都进一步减小最终效应的绝对值；有{resolved}个方向在最终分数上明确偏离简单相加。'
        '这检验了两处在固定上游干预下的联合效应，尚不能单独判断是否存在唯一的串行词义路径。','',
        '## 比较对象与操作','',
        '| 查询 | 原文 | 参考答案 |','|---|---|---|',
        '| Q01（#3169） | 我想回个嘿嘿嘿嘿。。。感觉好押韵 | 无 |',
        '| Q02（已审核#3660） | 主要是被嘿嘿玩过的，那不是一般的思想，那得多么的。。。 | 有 |','',
        'D01提供原侮辱义，D02提供普通笑声义；四份完整prompt、任务、单token有/无输出均与前轮相同。Q02还有其他贬损线索，两个查询不是纯词义最小对。方向“普通义→原义”表示从普通义运行取内部向量，放入原义运行；反向亦然。','',
        'N是没有干预的原生接收方。U首先在第17层，将查询“嘿嘿”的完整内部状态替换为另一释义条件的状态（Q01两个token、Q02一个）。'
        'A在U基础上，把答案前第26层注意力输出恢复成N的值；B独立恢复第28层MLP输出。新增AB在同一运行依次恢复这两处，来源始终是N。全部层号从0开始。','',
        '这里的注意力输出是经o_proj后、残差相加前的整个4096维向量，与热图注意力权重不同。MLP亦在残差相加前恢复。恢复发生在原始prompt最后位置，不是生成答案token的位置。'
        '两处恢复会影响其后的计算；尤其A改变了28层收到的状态，因此不能简单把单独恢复的比例相加。N/U/A/B/AB都在本轮重新计算，旧分数未代入。','',
        '## 最终效应与标签','',
        'm=无logit−有logit，越正越偏“无”；正向对Q01有利，对Q02不利。下表列相对N的分数差，括号是最终输出。零效应表示回到N的分数，并不统一表示正确。','',
        '| 查询/供体→接收方 | 原生N的m | U效应 | A效应 | B效应 | 联合AB效应 |',
        '|---|---:|---:|---:|---:|---:|']
    for j in js:
        group=[next(e for e in es if e['job_id']==j[key]) for key in ['upstream_job_id','A_job_id','B_job_id','job_id']]
        cells=[f'{e["delta_m"]:+.4f}（{e["prediction"]}）' for e in group]
        lines.append(f'| {j["query_id"]}/{direction(j)} | {j["native_m"]:+.4f} | '+' | '.join(cells)+' |')
    lines += ['', '![相对原生的最终效应](endpoint-effects.png)','',
        '移除比例=(U分数−恢复后分数)/(U分数−N分数)。0表示没有移除原干预效应，1表示回到N；负值或超过1也保留。'
        '比例不是概率，不能作为独立中介份额；同时检查相对N的绝对效应，以免越过N后反而离得更远。','',
        '| 查询/方向 | A移除比例 | B移除比例 | AB移除比例 | 比A/B都进一步减弱 |',
        '|---|---:|---:|---:|---|']
    def pct(x):return 'NA' if x is None else f'{100*x:.1f}%'
    for j in js:
        rr=[next(r for r in cs if r['job_id']==j[key]) for key in ['A_job_id','B_job_id','job_id']]
        lines.append(f'| {j["query_id"]}/{direction(j)} | '+' | '.join(pct(r['removed_fraction']) for r in rr)+f' | {"是" if j["absolute_reduced_vs_A"] and j["absolute_reduced_vs_B"] else "否"} |')
    lines += ['', '标签是否翻转还取决于恢复前离零点有多远；只看有/无，可能漏掉已经明确发生的分数变化。完整m、参考对齐变化及未翻转方向见附表。','',
        '## 联合效果是否可以简单相加','',
        '先比较AB−A：已有26层注意力恢复之后，再恢复28层MLP，最终分数还改变多少。AB−B则比较联合恢复与仅恢复28层。'
        '交互量I=mAB−mA−mB+mU衡量联合变化偏离两种单独变化之和的程度；它不是额外的一份独立因果贡献。','',
        '| 查询/方向 | AB−A | AB−B | 交互I | I/原U效应 |', '|---|---:|---:|---:|---:|']
    for j in js:
        lines.append(f'| {j["query_id"]}/{direction(j)} | {j["AB_minus_A"]:+.4f} | {j["AB_minus_B"]:+.4f} | {j["interaction_m"]:+.4f} | {pct(j["interaction_fraction"])} |')
    lines += ['', 'I/原U效应为正，表示联合移除量小于两个单独移除量之和；为负表示超过该和。这里仅描述最终m这个尺度上的交互。'
        '非零可以来自前后依赖、非线性或后续补偿，不能据此确认冗余或串行中介；即使接近零也不代表两处独立。','',
        '## 全部36层发生了什么','',
        '答案方向投影使用最终RMSNorm和输出头读取中间状态，衡量当时偏向“有/无”的程度，不表示该层已经作出决定。所有曲线来自答案前位置，层号0起。'
        '主图画差值以区分新增变化，保留全部36层和晚层补偿；没有仅选看起来有效的层。','',
        '![已有A后再恢复B的变化](joint-minus-A.png)','',
        '左列为AB−A的层末投影差；中、右列为注意力与MLP部分的增量差。新增分支投影和已有残差RMS重缩放是采用更新后尺度的代数分解，浮点余项也保留在完整数据中。'
        '这几项不能解释成独立因果份额。AB与A在28层MLP恢复前必须逐元素相同，这是操作正确性的核查；恢复后的变化才是需要解释的结果。','',
        '[联合与仅B的差值](joint-minus-B.pdf) · [逐层交互](joint-interaction.pdf) · [所有配置相对N](remaining-trajectories.pdf) · [所有恢复相对U](restore-minus-upstream-trajectories.pdf) · [联合恢复RMS分解](restore-joint-normalization.pdf)','',
        '## 核查、边界与后续问题','',
        '16个原生自身控制和12个条件自身控制要求完整输出向量及全层轨迹精确不变。条件自身控制保留跨释义上游替换，但把两处分支恢复成U自身的值，以检查双重回填操作。'
        '采集器开关、重复、逆序、左右padding、真实供体前缀、正式重放及全部20个单标签后EOS端点，均按原门槛通过后才分析。'
        'AB的26层恢复前与U相同、28层MLP恢复前与A相同；安装的两个向量均来自本轮N。','',
        f'预期且核验192次前向；GPU两阶段工作耗时合计{elapsed:.2f}秒，工作进程退出并验证释放后才合并人工参考。'
        f'm工程界为{data["margin_error_bound"]:.12g}，单候选投影界为{data["probe_logit_error_bound"]:.12g}；I的保守工程界为4倍m界。'
        '归一化交互的区间枚举五个分数的端点并保留共享U依赖；这些不是统计置信区间。独立审计另行对全部原始向量、轨迹与派生数值核验后才完成科学封存。','',
        '四份材料已在前轮观察，四个方向不是独立确认样本。恢复整个向量混合了不同运行状态，并没有单独删去词义信息。联合恢复后即使仍有剩余，也不能把剩余归于某个未测试的唯一模块。','',
        '本轮回答的最小问题是“两处一起恢复是否比单独恢复更充分”。如果更充分但仍有剩余，应先将该结论作为这四份输入的机制证据保留；若继续追问串行联系，需要明确区分26层改变经28层传递的部分与其他后续影响。'
        '不据当前结果自动扩展头扫描、材料或新GPU配置。','',
        '[完整输入](../prepared-01/ALL-PROMPTS.md) · [预先协议](../prepared-01/PROTOCOL.md) · [所有干预](all-interventions.tsv) · [恢复比较](restoration-contrasts.tsv) · [联合交互](joint-contrasts.tsv) · [全层TSV](all-trajectories.tsv) · [JSON](results.json) · [独立审计](../result-audit-01.json)','']
    (out/'REPORT.md').write_text('\n'.join(lines))
