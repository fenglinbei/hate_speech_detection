"""Adopted order/expression controls and reference-free mechanism geometry."""
from __future__ import annotations

from copy import deepcopy
import hashlib
from pathlib import Path
import sys

from diagnostics.case_attention_inputs_v1 import (
    ROOT, TASK, ROLES, ROLE_LABELS, require, read, lines, canonical, digest, sha,
    info, verify, write, atomic, tokenizer, build_input)

WORK = ROOT / 'reviews/hehe-presentation-mechanism-v1'
PUBLIC = ROOT / 'docs/research/experiment-plans/hehe-presentation-mechanism-v1'
PREPARED = WORK / 'prepared-01'
OLD = ROOT / 'reviews/hehe-sense-context-v1/prepared-01'
CONDITIONS = [('L', False, 'definition')]
ACCEPTANCE = '没问题，可以按照这一轮实现，草案/D03~D06/推荐的机制探索方式都没问题'
FOCAL = 'lex-0419:definition'
GROUPS = ['definition_original', 'definition_ordinary', 'definition_scaffold',
          'query_focal', 'query_quote_other', 'query_rejection', 'query_other', 'remainder']
GROUP_LABELS = ['原释义子句', '普通义子句', '释义连接／边界', '查询词形',
                '引文其他token', '反对段', '查询其他／边界', '其余上下文']
MECHANISM_RULES = {
    'exact_repeat_reverse_replay': True,
    'padding_prefix_scaled_max_cap': 0.0001,
    'projection_absolute_cap': 0.001,
    'reconstruction_scaled_max_cap': 0.0001,
    'floor': 0.000001,
    'meaning': 'Predeclared FP32 numerical engineering limits, not statistical or causal uncertainty.',
}
CODE_NAMES = ['inputs', 'capture', 'runtime', 'report']
CODE = [ROOT / 'src/diagnostics' / f'hehe_presentation_{part}_v1.py' for part in CODE_NAMES]
CODE += [ROOT / 'scripts/review' / name for name in (
    'run_hehe_presentation_v1.py', 'test_hehe_presentation_v1.py',
    'audit_hehe_presentation_v1.py', 'audit_hehe_presentation_results_v1.py',
    'test_hehe_presentation_viewer_v1.cjs')]
CODE += [ROOT/'tools/hehe_presentation_viewer_v1'/name for name in ('viewer.html','mechanism.html')]


def dictionaries(materials):
    old = {d['dictionary_id']: deepcopy(d) for d in materials['dictionaries']}
    original, ordinary = old['D01']['definition'], old['D02']['definition']
    a = "是'黑黑'的中文谐音，用作对黑人群体的侮辱性称呼"
    b = '是模拟笑声的中文写法，用于轻松聊天、调侃或打趣等表达'
    old['D04'] = {'dictionary_id': 'D04', 'name': '分行：普通义→原义',
                  'definition': ordinary + '\n' + original, 'component_order': ['ordinary', 'original']}
    old['D05'] = {'dictionary_id': 'D05', 'name': '整句：原义→普通义',
                  'definition': "'嘿嘿'既可以" + a + '，也可以' + b + '。', 'component_order': ['original', 'ordinary']}
    old['D06'] = {'dictionary_id': 'D06', 'name': '整句：普通义→原义',
                  'definition': "'嘿嘿'既可以" + b + '，也可以' + a + '。', 'component_order': ['ordinary', 'original']}
    for did, item in old.items():
        item['components_text'] = {'original': a, 'ordinary': b} if did in ('D05', 'D06') else {'original': original, 'ordinary': ordinary}
        item['composition_authorship'] = 'assistant' if did in ('D04','D05','D06') else 'inherited; see original provenance'
        item['decision'] = 'accept' if did in ('D04', 'D05', 'D06') else 'inherited_acceptance'
    return list(old.values())


def geometry(row, dictionary, query):
    """Disjoint source partition, separate from overlapping viewer spans."""
    definition = next(s for s in row['spans'] if s['id'] == FOCAL)
    qspan = next(s for s in row['spans'] if s['id'] == 'query')
    groups = {name: [] for name in GROUPS}
    components = []
    for name in dictionary['component_order']:
        text = dictionary['components_text'][name]
        require(dictionary['definition'].count(text) == 1, 'Ambiguous definition component')
        start = definition['char_start'] + dictionary['definition'].index(text)
        end = start + len(text)
        owned = [i for i in definition['token_positions'] if start <= row['token_offsets'][i][0] < row['token_offsets'][i][1] <= end]
        groups['definition_' + name] = owned
        components.append({'component': name, 'text': text, 'char_start': start, 'char_end': end,
                           'owned_token_positions': owned,
                           'touching_token_positions': [i for i, (a, b) in enumerate(row['token_offsets']) if a < end and b > start]})
    assigned = set(groups['definition_original'] + groups['definition_ordinary'])
    groups['definition_scaffold'] = sorted(set(definition['token_positions']) - assigned)
    groups['query_focal'] = list(row['roles']['query_focal'])
    extra_rows = {}
    if query['query_id'] == 'Q03':
        text = query['text']
        quoted_start, quoted_end = text.index('“') + 1, text.index('”')
        reject_start = quoted_end + 1
        require(text[reject_start:].startswith('我不赞同'), 'Rejection boundary changed')
        for name, start, end in [('query_quote_other', quoted_start, quoted_end),
                                 ('query_rejection', reject_start, len(text))]:
            start += qspan['char_start']; end += qspan['char_start']
            groups[name] = [i for i in qspan['token_positions']
                            if start <= row['token_offsets'][i][0] < row['token_offsets'][i][1] <= end
                            and i not in groups['query_focal']]
        # Closing quotation is retained as a real token even if its punctuation
        # crosses the quoted-content boundary; it belongs to query_other.
        cut = qspan['char_start'] + reject_start
        extra_rows['quoted_end'] = max(i for i in qspan['token_positions'] if row['token_offsets'][i][1] <= cut)
        extra_rows['rejection_start'] = groups['query_rejection'][0]
    used = set(groups['query_focal'] + groups['query_quote_other'] + groups['query_rejection'])
    groups['query_other'] = sorted(set(qspan['token_positions']) - used)
    used = [i for name, ps in groups.items() if name != 'remainder' for i in ps]
    require(len(set(used)) == len(used), 'Mechanism source groups overlap')
    groups['remainder'] = sorted(set(range(row['prompt_tokens'])) - set(used))
    labels = {}
    for role in ('lexicon_end', 'query_focal', 'query_end', 'pre_answer'):
        for n, pos in enumerate(row['roles'][role]):
            labels.setdefault(pos, []).append(role + (f':{n}' if role == 'query_focal' else ''))
    for name, pos in extra_rows.items():
        labels.setdefault(pos, []).append(name)
    row['mechanism'] = {
        'candidate_ids': [row['candidate_tokens']['有'], row['candidate_tokens']['无']],
        'positions': sorted(labels), 'position_labels': {str(k): v for k, v in labels.items()},
        'av_positions': sorted(set(row['roles']['query_focal'] + row['roles']['pre_answer'])),
        'source_group_order': GROUPS, 'source_group_labels': GROUP_LABELS,
        'source_groups': groups, 'disjoint_exhaustive_partition': True,
        'definition_components': components,
        'scaffold_is_not_a_sense': True,
        'causal_future_keys_unavailable': True,
    }
    return {'request_id': row['request_id'], 'parent_span_id': FOCAL, 'components': components,
            'separator_or_cross_boundary_tokens': groups['definition_scaffold'],
            'overlap_policy': 'Definition parent overlaps components; scaffold kept separate.'}


def build_requests():
    materials = read(OLD / 'materials.json')
    ds = dictionaries(materials)
    tok = tokenizer()
    system = (OLD / 'model-task.txt').read_text(encoding='utf-8')
    require(system == (TASK / 'model-task.txt').read_text(encoding='utf-8'), 'Latest task changed')
    historical = {r['request_id']: r for r in lines(OLD / 'scoring-inputs.jsonl')}
    requests, parts, preservation = [], [], []
    for q in materials['queries']:
        for d in ds:
            case = {'query_id': q['query_id'], 'query_text': q['text'], 'lexicon': deepcopy(materials['base_lexicon']), 'demos': []}
            next(e for e in case['lexicon'] if e['lexicon_id'] == 'lex-0419')['senses'][0]['definition'] = d['definition']
            r = read_json(canonical(build_input(case, CONDITIONS[0], tok, system)))
            r.update(request_id=f'hpm-{q["query_id"]}-{d["dictionary_id"]}', dictionary_id=d['dictionary_id'])
            if d['dictionary_id'] in ('D01', 'D02', 'D03'):
                old = historical[f'hsc-{q["query_id"]}-{d["dictionary_id"]}']
                require({k:v for k,v in r.items() if k != 'request_id'} == {k:v for k,v in old.items() if k != 'request_id'}, 'Historical prompt changed')
                preservation.append({'request_id': r['request_id'], 'old_request_id': old['request_id'],
                                     'prompt_sha256': r['prompt_sha256'], 'input_ids_sha256': r['input_ids_sha256'],
                                     'new_measurement_only': True, 'old_scores_substituted': False})
            parts.append(geometry(r, d, q))
            requests.append(r)
    require('torch' not in sys.modules, 'Material construction imported torch')
    return requests, parts, preservation, materials, ds


def read_json(data):
    import json
    return json.loads(data)


def comparisons():
    output = []
    for q in ('Q01', 'Q02', 'Q03'):
        for a,b,kind in [('D02','D01','single_sense'),('D03','D01','dual_vs_original'),('D03','D02','dual_vs_ordinary'),
                          ('D04','D03','order_lines'),('D06','D05','order_sentence'),
                          ('D05','D03','expression_original_first'),('D06','D04','expression_ordinary_first')]:
            output.append({'comparison_id': f'{q}-{a}-minus-{b}', 'query_id':q, 'kind':kind,
                'terms':[{'request_id':f'hpm-{q}-{a}','coefficient':1},{'request_id':f'hpm-{q}-{b}','coefficient':-1}]})
        output.append({'comparison_id':q+'-order-by-expression','query_id':q,'kind':'interaction',
            'terms':[{'request_id':f'hpm-{q}-{d}','coefficient':coef} for d,coef in [('D06',1),('D05',-1),('D04',-1),('D03',1)]]})
    return output


def prepare(output=PREPARED):
    from diagnostics.cross_model_applicability_execution_v1 import runtime_versions, framework_sources
    out = Path(output).absolute()
    require(not out.exists(), 'Use a new preparation directory')
    old_manifest = read(OLD / 'manifest.json')
    for item in old_manifest['artifacts'] + old_manifest['sources']:
        verify(item)
    reqs, parts, preserved, mats, ds = build_requests()
    out.mkdir(parents=True)
    PUBLIC.mkdir(parents=True, exist_ok=True)
    adoption = {'status':'accepted','user_message':ACCEPTANCE,'accepted_items':['D04','D05','D06','mechanism_stage_1','mechanism_stage_2_method'],
                'D03_reaffirmed':True,'prior_query_and_definition_adoptions':info(OLD/'adoption.json'),
                'AI_authorship_retained':True,'new_individual_labels_created':False,
                'stage_2_execution':'Only after stage-1 readouts; no automatic activation intervention in the acquisition run.'}
    write(out/'adoption.json',adoption)
    write(out/'materials.json',dict(mats, schema='hehe-presentation-materials/v1', dictionaries=ds, status='adopted'))
    write(out/'definition-components.json',{'records':parts})
    write(out/'historical-input-matches.json',{'matches':preserved})
    write(out/'comparisons.json',{'comparisons':comparisons()})
    (out/'analysis-references.json').write_bytes((OLD/'analysis-references.json').read_bytes())
    (out/'model-task.txt').write_bytes((OLD/'model-task.txt').read_bytes())
    for name in ('inputs.jsonl','scoring-inputs.jsonl'):
        with (out/name).open('xb') as f:
            for r in reqs: f.write(canonical(r)+b'\n')
    profile = deepcopy(read(OLD/'model-profile.json'))
    cfg = read(ROOT/'models/base/Qwen3-8B/config.json')
    profile.update(hidden_size=cfg['hidden_size'], head_dim=cfg['head_dim'], kv_heads=cfg['num_key_value_heads'], rms_norm_eps=cfg['rms_norm_eps'])
    write(out/'model-profile.json',profile)
    plan = deepcopy(read(OLD/'execution-plan.json'))
    plan.update(schema_version='hehe-presentation-execution/v1', status='adopted_cpu_preparation_gpu_pending',
        runtime_versions=runtime_versions(),input_count=18,comparison_count=24,dictionary_versions=[d['dictionary_id'] for d in ds],
        accepted_review_items=adoption['accepted_items'],new_prompts=9,remeasured_old_prompts=9,
        mechanism_acceptance=MECHANISM_RULES,automatic_patch_phase=False,
        mechanism_axes={'state_sites':['pre','mid','post'],'branches':['attention','mlp'],
                        'source_groups':GROUPS,'targets':'individual focal tokens and pre_answer; state-only boundary positions also retained',
                        'AV_group_mass_dtype':'float64','native_states_AV_output_dtype':'float32'},
        budget={'unique_prompts':18,'engineering_full_forwards':108,'prefix_forwards':18,'production_forwards':18,
                'format_extra_forward_max':126,'usual_if_label_then_eos':162,'total_forward_max':270})
    write(out/'execution-plan.json',plan)
    patches=[]
    byid={r['request_id']:r for r in reqs}
    for q in ('Q01','Q02'):
        for recipient,donor in [('D01','D02'),('D02','D01')]:
            r=byid[f'hpm-{q}-{recipient}'];d=byid[f'hpm-{q}-{donor}']
            require(r['prompt_tokens']==d['prompt_tokens'] and r['roles']==d['roles'],'Patch alignment')
            for layer in range(36):
                patches.append({'query_id':q,'recipient':r['request_id'],'donor':d['request_id'],'layer':layer,
                    'site':'decoder_block_output','positions':r['roles']['query_focal'],
                    'last_layer_expected_zero_by_causal_geometry':layer==35})
    write(out/'patch-plan.json',{'status':'method_adopted_deferred_until_stage_1','automatic_execution':False,
        'primary_interventions':patches,'primary_forward_count':144,'controls':['self_patch','same_size_other_query_positions'],
        'metrics':['raw_delta_m','prediction_change','donor_gap_fraction_unclipped'],
        'reference_alignment':'Q02 sign is opposite to raw m; no universal repair claim.',
        'null_result_scope':'This position/site alone may be insufficient; dictionary routes can bypass it.'})
    estimate={'estimated_GPU_and_checks_minutes':[10,20],'recommended_reserved_minutes':30,
        'recommended_free_disk_bytes':20_000_000_000,'future_single_device_min_mib':44000,
        'basis':'162 usual forwards plus sparse state/AV telemetry; prior same-model81-forward run used114s in two loaded phases. New hook cost is unmeasured.',
        'guaranteed_completion_time':False,'GPU_inventory_checked_during_preparation':False}
    write(out/'resource-estimate.json',estimate)
    for r in reqs:
        p=out/'prompts'/(r['request_id']+'.txt');p.parent.mkdir(exist_ok=True);p.write_text(r['prompt_text'],encoding='utf-8')
    doc=['# 已采用的呈现与机制采集材料','','9个新输入＋9个旧输入的新测量；旧实验不重启，参考标签不进入模型输入。','']
    for d in ds: doc += ['## '+d['dictionary_id']+' '+d['name'],'','```text',d['definition'],'```','']
    for r in reqs: doc += ['## '+r['request_id'],'',str(r['prompt_tokens'])+' tokens','','```text',r['prompt_text'],'```','']
    (out/'ALL-PROMPTS.md').write_text('\n'.join(doc)+'\n',encoding='utf-8')
    sources = [OLD/'manifest.json', OLD/'materials.json', OLD/'scoring-inputs.jsonl', OLD/'analysis-references.json', OLD/'adoption.json',
        OLD/'execution-plan.json', OLD/'model-profile.json', ROOT/'src/diagnostics/case_attention_inputs_v1.py',
        ROOT/'src/diagnostics/case_attention_capture_v1.py', ROOT/'src/diagnostics/cross_model_applicability_models_v1.py',
        ROOT/'src/diagnostics/cross_model_applicability_execution_v1.py'] + CODE + framework_sources()
    write(out/'source-ledger.json',{'files':[info(p) for p in sorted(set(sources))], 'historical_weight_inventory_reused':True})
    (out/'README.md').write_text('# 呈现对照与机制采集\n\n材料已通过，18个输入、24个比较。仅第一阶段只读采集；激活替换计划独立且不会自动启动。\n\n'
        'D03/D04等长；D05/D06等长，比对应分行版多1 token。整句比较同时改变连接词、重复和标点。\n\n'
        '状态、注意力及MLP输出按token保存；来源分组AV保留全部分母且分组不重叠。原始注意力仍用原6角色。'
        '答案前层间标签投影仅为诊断；句中投影不作为分类。来源向量也携带更早上下文，不是独立语义因果贡献。\n\n'
        '通常162次前向，格式最坏270次；新采集开销未实测，预计10–20分钟、建议预留30分钟。未来GPU需新绑定与不变的数值/格式资格验证。\n',encoding='utf-8')
    return {'inputs':18,'new_prompts':9,'old_prompt_measurements':9,'comparisons':24,'GPU_started':False}


def validate(prepared, sealed=True, weights=False):
    from diagnostics.cross_model_applicability_execution_v1 import runtime_versions
    p=Path(prepared)
    if sealed:
        m=read(p/'manifest.json');require(m['schema_version']=='hehe-presentation-prepared/v1','Wrong freeze')
        for x in m['artifacts']+m['sources']:verify(x)
    for x in read(p/'source-ledger.json')['files']:verify(x)
    plan,profile=read(p/'execution-plan.json'),read(p/'model-profile.json')
    require(plan['runtime_versions']==runtime_versions(),'Runtime changed')
    require(plan['acceptance']==read(OLD/'execution-plan.json')['acceptance'],'Original numerical gates changed')
    require(plan['mechanism_acceptance']==MECHANISM_RULES,'Mechanism gates changed')
    for x in profile['metadata_sources']:verify(x)
    for x in profile['weight_sources']:
        if weights:verify(x)
        else:
            s=Path(x['path']).stat();require((s.st_size,s.st_mtime_ns)==(x['bytes'],x['mtime_ns']),'Weights changed')
    reqs=lines(p/'scoring-inputs.jsonl')
    require(len(reqs)==18 and len({r['request_id'] for r in reqs})==18,'Wrong inventory')
    require({(r['query_id'],r['dictionary_id']) for r in reqs}=={(q,f'D0{d}') for q in ('Q01','Q02','Q03') for d in range(1,7)},'Missing condition')
    for r in reqs:
        require(not {'reference','gold','human_decision'}&set(r),'Reference in worker input')
        require(digest(r['input_ids'])==r['input_ids_sha256'] and hashlib.sha256(r['prompt_text'].encode()).hexdigest()==r['prompt_sha256'],'Input binding')
        require(r['roles']['pre_answer']==[r['prompt_tokens']-1] and r['roles']['demos_end']==[],'Readout geometry')
        groups=r['mechanism']['source_groups'];flat=[i for name in GROUPS for i in groups[name]]
        require(sorted(flat)==list(range(r['prompt_tokens'])),'Source partition')
    return plan,profile,reqs


def seal(prepared):
    p=Path(prepared).absolute();require(not (p/'manifest.json').exists(),'Already sealed')
    validate(p,sealed=False)
    for name in ('cpu-tests.json','cpu-audit.json','viewer-cpu-test.json'):
        rec=read(p/name);require(rec['status']=='pass' and not rec.get('CUDA_initialized',False),'CPU checks failed')
        for item in rec.get('implementation_snapshot',[]):verify(item)
    write(p/'manifest.json',{'schema_version':'hehe-presentation-prepared/v1','status':'cpu_complete_gpu_pending',
        'artifacts':[info(x) for x in sorted(p.rglob('*')) if x.is_file()],
        'sources':read(p/'source-ledger.json')['files'],'immutable_after_seal':True})
    atomic(PUBLIC/'current.json',{'status':'adopted_prepared_gpu_pending','directory':str(p.relative_to(ROOT)),
        'manifest':info(p/'manifest.json'),'new_inputs':9,'total_inputs':18,'GPU_started':False})
    return info(p/'manifest.json')
