"""CPU-only conditional-restoration contrasts, trajectories, and readable report."""
from __future__ import annotations
import csv,itertools
from pathlib import Path
import numpy as np
from diagnostics import hehe_branch_restore_inputs_v1 as c
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


def analyze(prepared,run,output):
    from diagnostics import hehe_branch_restore_runtime_v1 as rt
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
    controls=[]
    for job in selfs:
        rec,v,_=rt.load_record(run,'self',byid[job['recipient']],binding,profile,job)
        baseline=es[job['upstream_job_id']]['m'] if job['kind']=='conditional_self_control' else scores[job['recipient']]['m']
        controls.append(dict(job,delta_m=readout(v,profile['candidate_tokens'])['m']-baseline,
            source_record=c.info(rt.record_path(run,'self',job['job_id']))))
    data={'schema':'hehe-branch-restoration-results/v1','status':'complete','layers':36,'patch_layer':17,
        'requests':reqs,'baselines':list(scores.values()),'effects':effects,'restoration_contrasts':contrasts,
        'self_controls':controls,'native_trajectories':native,'patched_trajectories':trajectories,
        'margin_error_bound':bound,'probe_logit_error_bound':probe_bound,'jobs':jobs,
        'notes':['四条曝光材料，十二配置不是独立样本。','注意力指o_proj后整个子层输出，不是注意力权重。',
            '固定第17层跨条件替换后，单独恢复答案前26注意力或28MLP，不合并两处。',
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
    with (out/'all-trajectories.tsv').open('w') as f:
        w=csv.writer(f,delimiter='\t');w.writerow(['job_id','comparison','layer0','probe_pre','probe_mid','probe_post','increment_attention','increment_mlp',
            'branch_attention','branch_mlp','scale_attention','scale_mlp','rounding_attention','rounding_mlp'])
        rows=[(e['job_id'],'minus-native',trajectories[e['job_id']]['minus_recipient']) for e in effects]
        rows += [(r['job_id'],'minus-upstream',r['trajectory_difference']) for r in contrasts]
        for jid,kind,d in rows:
            for li in range(36):w.writerow([jid,kind,li,*d['probe_pre_mid_post'][li],*d['increment_attention_mlp'][li],
                *d['branch_projection_at_destination_scale'][li],*d['existing_residual_rescaling'][li],*d['floating_remainder'][li]])
    figures(out,data);write_report(out,data,run)
    c.write(out/'audit.json',{'status':'pass','CPU_reconstructed':True,'query_references_joined_after_release':True,
        'prepared_manifest':c.info(p/'manifest.json'),'raw_seal':c.info(run/'raw-seal.json'),
        'release':c.read(run/'state.json')['resource_release'],'independent_samples_claimed':False})
    c.write(out/'manifest.json',{'artifacts':[c.info(x) for x in sorted(out.iterdir()) if x.is_file()]})
    return {'output':str(out),'native_inputs':4,'interventions':12,'restoration_contrasts':8,'self_controls':20,'trajectory_layers':36}


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
    styles=[('upstream','#147d91','-','仅17层替换'),('restore-L26-attention','#cc7041','--','加26层注意力恢复'),('restore-L28-mlp','#8058a1','-.','加28层MLP恢复')]
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
        ax.axhline(0,color='#78828b',lw=.7);ax.set_xticks(range(3),['仅17层替换','加26层注意力恢复','加28层MLP恢复'],fontsize=9)
        ax.set_title(f'{u["query_id"]} / {direction(u)}');ax.set_ylabel('最终m − 原生接收方m');ax.margins(y=.25)
        ax.spines[['top','right']].set_visible(False)
    fig.suptitle('单独恢复一个后续分支，原有最终输出效应剩下多少？',fontsize=14)
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
    for condition in ['restore-L26-attention','restore-L28-mlp']:
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


def write_report(out,data,run):
    cs=data['restoration_contrasts'];es=data['effects'];reduced=sum(r['absolute_effect_reduced'] for r in cs)
    elapsed=sum(i['ended_at_unix']-i['started_at_unix'] for i in c.read(run/'state.json')['invocations'])
    lines=['# 固定第17层替换后的单分支恢复', '',
        f'本轮将答案前第26层注意力输出、第28层MLP输出分别恢复为原生接收方的值。8个方向×组件比较中，{reduced}个的最终效应绝对值减小。'
        '这里检验的是这两个组件在已施加第17层干预后的作用，尚不能据此确认唯一的自然词义路径。', '',
        'Q01是原#3169“我想回个嘿嘿嘿嘿。。。感觉好押韵”，参考“无”；Q02是原已审核#3660“主要是被嘿嘿玩过的，那不是一般的思想，那得多么的。。。”，参考“有”。'
        'D01只提供原侮辱义，D02只提供普通笑声义。四份prompt、任务和标签保持原样。Q02还有其他贬损线索，两个查询不是纯词义最小对。', '',
        '激活替换是把另一释义条件下的内部向量放进当前运行。首先固定替换第17层查询“嘿嘿”位置的完整状态（Q01两个token、Q02一个），称为上游替换U。'
        '再分别把答案前最后一个prompt位置的第26层注意力输出，或第28层MLP输出，改回当前接收条件未受干预时的原生值，称为恢复R。两处独立测试；全部层号0起。', '',
        '这里“注意力输出”是整个注意力子层经输出投影后的向量，不是热图中的注意力权重。恢复发生在残差相加之前；其余位置保留U的计算，后续层重新计算。', '',
        '## 最终输出效应', '',
        'm=无logit−有logit，数值越大越偏向“无”。表中的效应均减去原生接收方m；因此正数对Q01有利、对Q02不利。'
        '“移除比例”比较恢复前后少了多少原干预效应，0表示保留、1表示回到原生分数，负值表示沿原方向增强；超过1表示越过原生值，需同时看剩余效应。它不是概率，也不能把两组件比例相加。', '',
        '| 查询/供体→接收方 | 恢复位置 | 上游效应 | 恢复后效应 | 移除比例 | 输出 U→R |', '|---|---|---:|---:|---:|---|']
    for r in cs:
        site='26注意力' if r['condition']=='restore-L26-attention' else '28MLP'
        fraction='NA' if r['removed_fraction'] is None else f'{100*r["removed_fraction"]:.1f}%'
        lines.append(f'| {r["query_id"]}/{direction(r)} | {site} | {r["upstream_delta_m"]:+.4f} | {r["remaining_delta_m"]:+.4f} | {fraction} | {r["upstream_prediction"]}→{r["restored_prediction"]} |')
    lines += ['', '![最终效应](endpoint-effects.png)', '',
        '若恢复使效应减弱，支持该处改变参与了承接这次上游干预的结果。效应仍保留，意味着这一个位置和子层未承接全部变化；不能直接认定剩余量都来自某条特定路径。'
        '若效应未减弱或增强，则需结合后续补偿判断，不能仅因原来的轨迹峰值较大就把该组件视为必要中介。', '',
        '## 变化从哪里出现，后来如何发展', '',
        '答案方向投影用最终RMSNorm与输出头读取中间状态，测量其偏向“有/无”，不表示该层已作出最终决定。第一张轨迹图比较每个干预与原生接收方；第二张只显示分支恢复对U的新增变化。'
        '读数均来自答案前位置，保留全部36层；恢复之前的数组须逐元素相同，这与图上看起来接近不同。', '',
        '![相对原生的完整轨迹](remaining-trajectories.png)', '',
        '![恢复新增的变化](restore-minus-upstream-trajectories.png)', '',
        '注意力/MLP前后的投影增量同时受新增向量和RMS尺度变化影响。附图采用更新后尺度分解为新增分支投影、已有残差重缩放，并保存浮点余项；这是代数分解，不是两份独立因果贡献。'
        '晚层的补偿、放大与反方向变化保留在完整图和TSV中。', '',
        '[26注意力恢复：全部增量与归一化分解](restore-L26-attention-normalization.pdf) · [28MLP恢复：全部增量与归一化分解](restore-L28-mlp-normalization.pdf)', '',
        '## 核查与解释边界', '',
        '12个原生自身控制及8个条件自身控制均要求完整向量和全部轨迹精确不变。后者保留跨条件上游替换，但回填U自己的分支输出，检查回填操作本身是否引入额外变化。'
        '完整采集器开关、重复、逆序、左右padding、供体真实前缀、正式重放及16个有/无后EOS端点均通过后才生成本报告。', '',
        f'正常预算152次前向，两阶段工作耗时合计{elapsed:.2f}秒。GPU工作进程正常退出并验证释放后才合并参考标签。'
        f'本轮m工程界为{data["margin_error_bound"]:.12g}，单候选投影工程界为{data["probe_logit_error_bound"]:.12g}。'
        '比值区间同时扰动三个分数并保留共享上游分数依赖，不是统计置信区间。', '',
        '四份材料已在前几轮反复观察；本轮8个比较不构成8个独立样本。恢复整个向量也不等同于只删除“词义信息”，与原生值混合后可能出现交互。'
        '只测试了两处单独恢复，未测试联合恢复或整个层/所有位置，不能宣布组件全局必要、无效或得到完整中介比例。', '',
        '[完整JSON](results.json) · [12个干预端点](all-interventions.tsv) · [8个恢复比较](restoration-contrasts.tsv) · [全部36层数值](all-trajectories.tsv) · [完整原文](../prepared-01/ALL-PROMPTS.md) · [协议](../prepared-01/PROTOCOL.md)', '']
    (out/'REPORT.md').write_text('\n'.join(lines))
