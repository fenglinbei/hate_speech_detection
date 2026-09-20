"""Offline viewer generation and post-release CPU analysis. No GPU imports."""
from __future__ import annotations

import base64
import csv
import io
from pathlib import Path

import numpy as np

from diagnostics import hehe_presentation_inputs_v1 as c
from diagnostics.case_attention_capture_v1 import aggregate


def component_requests(prepared, requests):
    """Add reviewed semantic subdivisions for analysis only; native inputs unchanged."""
    from copy import deepcopy
    records = {r['request_id']: r for r in c.read(Path(prepared) / 'definition-components.json')['records']}
    display = deepcopy(requests)
    for req in display:
        item = records[req['request_id']]
        for part in item['components']:
            req['spans'].append({'id': 'lex-0419:component:' + part['component'],
                'kind': 'lexicon_definition_component', 'parent': item['parent_span_id'],
                'label': '嘿嘿·' + {'original': '原释义', 'ordinary': '普通义'}[part['component']],
                'char_start': part['char_start'], 'char_end': part['char_end'],
                'token_positions': part['owned_token_positions']})
        if item['separator_or_cross_boundary_tokens']:
            req['spans'].append({'id': 'lex-0419:component:boundary',
                'kind': 'lexicon_definition_component', 'parent': item['parent_span_id'],
                'label': '嘿嘿·连接／边界', 'char_start': None, 'char_end': None,
                'token_positions': item['separator_or_cross_boundary_tokens']})
    return display


def viewer_data(requests, status, scores=None):
    return {'schema_version': 'case-attention-view/v1', 'status': status, 'roles': dict(zip(c.ROLES, c.ROLE_LABELS)), 'role_order': list(c.ROLES),
            'layers': 36, 'heads': 32, 'scores': scores or {}, 'requests': requests,
            'query_labels': {'Q01':'Q01 · #3169 普通聊天','Q02':'Q02 · #3660 攻击','Q03':'Q03 · 引用并反对'},
            'notes': ['三条依赖查询、十八个输入；探索性案例，非独立确认集。',
                      'D03/D04交换分行顺序；D05/D06交换整句顺序。整句相对分行改变连接词、重复、标点和长度。',
                      'Q02另含性经历贬损；Q03为已采用的AI反驳改写。',
                      '灰色表示未提供或不可见；无示例时示例结束为结构性NA。',
                      '嘿嘿完整释义与两段子片段重叠，不能相加；段间边界另列。',
                      '注意力没有正负标签意义，也不证明因果贡献。']}


def write_viewer(path, data):
    template = (c.ROOT / 'tools/hehe_presentation_viewer_v1/viewer.html').read_text(encoding='utf-8')
    # Prevent a real source string from terminating the non-executable JSON tag.
    payload = c.canonical(data).decode('utf-8').replace('<', '\\u003c').replace('\u2028', '\\u2028').replace('\u2029', '\\u2029')
    c.require(template.count('__DATA_JSON__') == 1, 'Viewer template placeholder differs')
    Path(path).write_text(template.replace('__DATA_JSON__', payload), encoding='utf-8')


def preview(prepared):
    prepared = Path(prepared)
    path = prepared / 'viewer.html'
    c.require(not path.exists(), 'Do not overwrite a delivered viewer')
    write_viewer(path, viewer_data(component_requests(prepared, c.lines(prepared / 'inputs.jsonl')), 'GPU未运行：这里只展示真实材料和已冻结位置，不含注意力测量'))
    return c.info(path)


def analyze(prepared, run, output):
    from diagnostics.hehe_presentation_runtime_v1 import check_run, load_record
    from diagnostics.cross_model_applicability_execution_v1 import readout
    prepared, run, out = map(Path, (prepared, run, output))
    checked = check_run(prepared, run)
    c.require(checked['status'] == 'complete', 'Full run must be normally released before analysis')
    c.require(not out.exists(), 'New analysis directory required')
    _, profile, requests = c.validate(prepared)
    requests = component_requests(prepared, requests)
    refs = {r['query_id']: r for r in c.read(prepared / 'analysis-references.json')['references']}
    q = checked['qualification']
    scores, summaries, compact = {}, [], []
    out.mkdir(parents=True)
    for req in requests:
        rec, vector, attention, mechanism = load_record(run, 'production', req, c.sha(run / 'binding.json'), profile)
        score = readout(vector, profile['candidate_tokens'], q['margin_error_bound'])
        score.update(request_id=req['request_id'], query_id=req['query_id'], condition=req['condition'],
                     reference=refs[req['query_id']]['reference'], raw_reference_correct=score['raw_prediction'] == refs[req['query_id']]['reference'])
        score['dictionary_id'] = req['dictionary_id']
        score['reference_aligned_margin'] = score['m'] if score['reference'] == '无' else -score['m']
        score['resolved_reference_correct'] = score['raw_reference_correct'] if score['resolution'] != 'numerical_unresolved' else None
        scores[req['request_id']] = score
        aggregated = aggregate(attention, req)
        for row in aggregated:
            row['mass_error_bound'] = q['attention_mass_bound'] if row['visible'] else None
            row['density_error_bound'] = q['attention_mass_bound'] / row['mean_visible_tokens'] if row['visible'] else None
        c.write(out / (req['request_id'] + '-aggregates.json'), {'request_id': req['request_id'], 'aggregates': aggregated,
                 'attention_mass_bound': q['attention_mass_bound'], 'source': rec['attention']})
        # A visualization copy only. Native maps and FP64 aggregation remain the
        # authoritative numeric artifacts; float32 display is explicitly labeled.
        little = np.asarray(attention, dtype='<f4')
        display = {'request_id': req['request_id'], 'prompt_sha256': req['prompt_sha256'],
                   'shape': list(little.shape), 'encoding': 'little-endian-float32-base64-visualization-only',
                   'data': base64.b64encode(little.tobytes()).decode('ascii'), 'score': score,
                   'source': rec['attention'], 'aggregates': aggregated}
        c.write(out / (req['request_id'] + '.view.json'), display)
        for a in aggregated:
            if a['visible']:
                mass=np.asarray(a['mass'],dtype=np.float64);density=np.asarray(a['density'],dtype=np.float64)
                compact.append({'request_id':req['request_id'],'role':a['role'],'span_id':a['span_id'],
                                'all_layer_head_mean_mass':float(mass.mean()),'all_layer_head_mean_density':float(density.mean()),
                                'late_18_35_mean_mass':float(mass[18:36].mean()) if mass.shape[0]>=36 else None,
                                'late_18_35_mean_density':float(density[18:36].mean()) if density.shape[0]>=36 else None,
                                'mean_visible_tokens':a['mean_visible_tokens']})
                for layer in range(profile['layers']):
                    summaries.append({'request_id': req['request_id'], 'role': a['role'], 'span_id': a['span_id'], 'layer': layer,
                                      'mean_head_mass': float(np.mean(a['mass'][layer])),
                                      'mean_head_density': float(np.mean(a['density'][layer])),
                                      'mean_visible_tokens': a['mean_visible_tokens']})
    comparisons = []
    for expression in c.read(prepared/'comparisons.json')['comparisons']:
        coefficients={term['request_id']:term['coefficient'] for term in expression['terms']}
        value=sum(coefficient*scores[rid]['m'] for rid,coefficient in coefficients.items())
        bound=sum(abs(coefficient)*scores[rid]['margin_error_bound'] for rid,coefficient in coefficients.items())
        transition=None
        if len(coefficients)==2 and sorted(coefficients.values())==[-1,1]:
            left=scores[next(k for k,v in coefficients.items() if v==1)]
            right=scores[next(k for k,v in coefficients.items() if v==-1)]
            transition='unresolved' if 'numerical_unresolved' in (left['resolution'],right['resolution']) else (
                'repair' if left['raw_reference_correct'] and not right['raw_reference_correct'] else
                'damage' if right['raw_reference_correct'] and not left['raw_reference_correct'] else 'unchanged')
        reference = refs[expression['query_id']]['reference']
        comparisons.append(dict(expression,reference_aligned_delta=value if reference == '无' else -value,delta_m=value,bound=bound,direction_resolved=abs(value)>bound,
                                direction='positive' if value>bound else 'negative' if value < -bound else 'unresolved',
                                transition=transition,value=value,numerical_bound=bound))
    c.write(out/'attention-summary.json',compact)
    c.write(out / 'scores.json', list(scores.values()))
    c.write(out / 'comparisons.json', comparisons)
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=list(summaries[0]), delimiter='\t')
    writer.writeheader()
    writer.writerows(summaries)
    (out / 'layer-summary.tsv').write_text(stream.getvalue(), encoding='utf-8')
    write_viewer(out / 'index.html', viewer_data(requests, 'GPU原始结果已封存，工作进程已正常退出；本页为CPU分析', scores))
    mechanistic = summarize_mechanism(prepared, run, out, requests, profile, scores, q)
    write_report(out, scores, comparisons, mechanistic, q)
    c.write(out / 'audit.json', {'CPU_reconstruction_passed': True, 'query_references_joined_after_release': True,
        'release': c.read(run / 'state.json')['resource_release'], 'raw_seal': c.info(run / 'raw-seal.json'),
        'prepared_manifest': c.info(prepared / 'manifest.json'), 'independent_cases_claimed': False, 'query_count': 3, 'construction_dependency': 'Q03 derives from Q02; all exposed exploration',
        'attention_is_causal_explanation': False, 'display_copy_precision': 'float32', 'analysis_precision': 'float64'})
    c.write(out / 'manifest.json', {'artifacts': [c.info(p) for p in sorted(out.iterdir()) if p.is_file()]})
    return {'output': str(out), 'conditions': len(scores), 'comparisons': len(comparisons)}


def summarize_mechanism(prepared, run, out, requests, profile, scores, qualification):
    from diagnostics.hehe_presentation_runtime_v1 import load_record
    records, cache = [], {}
    for req in requests:
        rec, _, _, m = load_record(run, 'production', req, c.sha(run/'binding.json'), profile)
        cache[req['request_id']] = (req, m)
        pos = m['positions'].tolist()
        answer = pos.index(req['roles']['pre_answer'][0])
        probe = (m['lens_logits'][:,:,:,1].astype(np.float64)-m['lens_logits'][:,:,:,0])[:,:,answer]
        states = m['states'].astype(np.float64)
        norm = np.linalg.norm(states,axis=-1)
        # The output direction uses each target's native post-block RMS scale.
        # This local projection is additive at a fixed scale, not propagated
        # causal contribution to the final answer or a probability.
        direction = (m['label_weights'][1].astype(np.float64)-m['label_weights'][0])*m['norm_weight']
        source_rows = []
        for ri, target in enumerate(m['av_positions'].tolist()):
            state = states[:,2,pos.index(target)]
            rms = np.sqrt(np.mean(state*state,axis=-1)+m['norm_eps'][0])
            output = m['source_outputs'][:,ri].astype(np.float64)
            local = np.einsum('lgd,d->lg',output,direction)/rms[:,None]
            for gi,name in enumerate(c.GROUPS):
                visible = bool(m['source_counts'][ri,gi])
                mass = m['source_mass'][:,ri,gi].mean(axis=-1)
                source_rows.append({'target_position':target,'target_labels':req['mechanism']['position_labels'][str(target)],
                    'group':name,'label':c.GROUP_LABELS[gi],'visible':visible,'visible_tokens':int(m['source_counts'][ri,gi]),
                    'mean_head_mass':mass.tolist() if visible else None,
                    'mean_head_density':(mass/m['source_counts'][ri,gi]).tolist() if visible else None,
                    'mean_head_AV_norm':np.linalg.norm(m['av_heads'][:,ri,gi].astype(np.float64),axis=-1).mean(axis=-1).tolist() if visible else None,
                    'source_output_norm':np.linalg.norm(output[:,gi],axis=-1).tolist() if visible else None,
                    'local_direction_projection':local[:,gi].tolist() if visible else None})
        record={'request_id':req['request_id'],'score':scores[req['request_id']],
            'positions':pos,'position_labels':req['mechanism']['position_labels'],
            'pre_answer_probe_margin_by_site':probe.tolist(), 'sites':['pre','mid','post'],
            'pre_answer_attention_probe_change':(probe[:,1]-probe[:,0]).tolist(),
            'pre_answer_mlp_probe_change':(probe[:,2]-probe[:,1]).tolist(),
            'state_norm':norm.tolist(),'source_rows':source_rows,'source':rec['mechanism']}
        records.append(record)
    pairs=[]
    endpoints=[]
    for q in ('Q01','Q02','Q03'):
        for a,b in [('D02','D01'),('D04','D03'),('D06','D05'),('D05','D03'),('D06','D04')]:
            endpoints.append((f'hpm-{q}-{a}',f'hpm-{q}-{b}','definition_or_presentation'))
    for d in ('D01','D02','D03','D04','D05','D06'):
        endpoints.append((f'hpm-Q03-{d}',f'hpm-Q02-{d}','quotation_rejection_and_length'))
    # Synthetic lifecycle tests use their own IDs and intentionally skip real
    # scientific endpoint pairs; they exercise all single-record summaries.
    for left,right,kind in endpoints:
        if left not in cache or right not in cache:
            continue
        lr,lm=cache[left];rr,rm=cache[right]
        for role in ('lexicon_end','query_focal','query_end','pre_answer'):
            lp,rp=lr['roles'][role],rr['roles'][role]
            c.require(len(lp)==len(rp),'Pair role token count differs')
            for token_index,(lpos,rpos) in enumerate(zip(lp,rp)):
                a=lm['states'][:,2,lm['positions'].tolist().index(lpos)].astype(np.float64)
                b=rm['states'][:,2,rm['positions'].tolist().index(rpos)].astype(np.float64)
                an,bn=np.linalg.norm(a,axis=-1),np.linalg.norm(b,axis=-1)
                cosine=np.divide(np.sum(a*b,axis=-1),an*bn,out=np.full_like(an,np.nan),where=an*bn>0)
                pairs.append({'left':left,'right':right,'kind':kind,'role':role,'token_index':token_index,
                    'l2':np.linalg.norm(a-b,axis=-1).tolist(),
                    'relative_l2':(np.linalg.norm(a-b,axis=-1)/np.maximum(1.0,bn)).tolist(),
                    'cosine':[None if not np.isfinite(x) else float(x) for x in cosine]})
    payload={'schema':'hehe-presentation-mechanism-view/v1','groups':c.GROUPS,'group_labels':c.GROUP_LABELS,
        'layers':profile['layers'],'records':records,'representation_pairs':pairs,
        'qualifications':qualification['mechanism'],
        'notes':['层号从0开始；所有层与来源均保留。','句中位置的标签投影不作为分类；图只展示答案前标签投影。',
            '固定末层归一化／输出头的早层投影是诊断，不能当作模型在该层已做出的决策。',
            '来源输出按key位置分组，value已含此前上下文；投影大小不是最终因果贡献。',
            'Q03焦点处读不到后面的反对句；Q02/Q03还同时改变引述、立场与长度。']}
    c.write(out/'mechanism-summary.json',payload)
    template=(c.ROOT/'tools/hehe_presentation_viewer_v1/mechanism.html').read_text(encoding='utf-8')
    encoded=c.canonical(payload).decode().replace('<','\\u003c').replace('\u2028','\\u2028').replace('\u2029','\\u2029')
    c.require(template.count('__DATA_JSON__')==1,'Mechanism template placeholder')
    (out/'mechanism.html').write_text(template.replace('__DATA_JSON__',encoded),encoding='utf-8')
    return payload


def write_report(out, scores, comparisons, mechanism, qualification):
    rows=['# 呈现对照与机制读数','','18个依赖条件的探索性实验；Q03是已采用的AI派生文本。',
        '任务、查询、参考标签与其他词条保持冻结。新增9个呈现输入，原9个输入重新采集中间表示；旧实验未重启。','',
        '[逐层注意力](index.html) · [机制读数](mechanism.html) · [全部比较](comparisons.json) · [原始汇总](mechanism-summary.json)','',
        'm = z(无) − z(有)。Q02参考为有，因此参考方向与原始m相反。','',
        '|输入|输出|参考|m|','|---|---|---|---:|']
    for rid,s in scores.items():rows.append(f'|{rid}|{s["raw_prediction"]}|{s["reference"]}|{s["m"]:+.6f}|')
    rows += ['', '|比较|Δm|参考方向Δ|转变|','|---|---:|---:|---|']
    for x in comparisons:rows.append(f'|{x["comparison_id"]}|{x["delta_m"]:+.6f}|{x["reference_aligned_delta"]:+.6f}|{x["transition"] or "交互量"}|')
    rows += ['', '## 读数解释','',
        'D04−D03与D06−D05分别检验分行和整句版本的顺序敏感性；交互量为(D06−D05)−(D04−D03)。',
        '整句版比对应分行版多1 token，还改变连接词、词形重复及标点，不能把差异全部归给换行。',
        '整句原义/普通义分别20/18个归属token，连接与边界6个；注意力同时保留质量和密度。','',
        '机制页面给出答案前pre/mid/post投影、注意力与MLP两次残差更新对应的投影变化、各来源AV范数、输出范数及局部方向投影。',
        '“局部方向投影”用该层原生post状态的RMS尺度固定归一化，衡量输出向量相对有/无输出方向；没有传播后续层，不能解释为最终分数的因果贡献。',
        '所有角色的逐token状态、分头AV及分组输出保存在production/*.mechanism.npz；不得以平均值代替原始token身份。',
        '反对句在Q03焦点token之后，故焦点状态不能读取它。后续变化可在引文结束、反对段开始、查询结束和答案前检查。',
        '表示距离与固定输出头投影用于提出候选层，不单独确立内部语义或因果路径。','',
        '## 后续激活替换','',
        '方法已采用，未自动运行。计划在Q01/Q02的D01与D02之间双向替换焦点token的decoder块输出，保留36层、self-patch和等数量其他查询位置控制。',
        '末层焦点替换应为结构性零效果，因为之后没有跨token注意力层。报告原始Δm、预测是否改变与不截断的donor-gap比例；分母接近数值界时为NA。',
        'Q01焦点为两个token，作为显式联合集合；Q02为一个。阴性结果只限制该位置和替换方式，不能排除词典直接传到答案的路径。','',
        '## 验证','',
        f'新测量数值界：margin {qualification["margin_error_bound"]:.9g}；attention mass {qualification["attention_mass_bound"]:.9g}。',
        f'机制重建最大缩放误差 {qualification["mechanism"]["reconstruction_max_scaled"]:.9g}；投影绝对误差 {qualification["mechanism"]["projection_max_absolute"]:.9g}。',
        '完整prompt日志向量、注意力、状态的只读hook、repeat/reverse/padding/prefix/replay验证与单token标签后EOS格式门均在新run内验证；参考在GPU正常释放和raw seal之后加入。','']
    (out/'REPORT.md').write_text('\n'.join(rows),encoding='utf-8')
