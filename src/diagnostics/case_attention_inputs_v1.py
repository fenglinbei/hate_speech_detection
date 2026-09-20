"""Versioned CPU-only input preparation for the two exposed attention cases."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PUBLIC = ROOT / 'docs/research/experiment-plans/case-attention-v1'
WORK = ROOT / 'reviews/case-attention-v1'
PREPARED = WORK / 'prepared-01'
TASK = ROOT / 'docs/research/experiment-plans/task-applicability-scoring-v1/frozen-01'
CARDS = ROOT / 'exps/causal_context/general_model_ld_nolabel_paired_cases_v1/results/paired-cases-02/cases/card_data'
REFERENCES = ROOT / 'exps/causal_context/general_model_evidence_applicability_v1/reviews/analysis-freeze-20260912/frozen-v1/material_reviews.jsonl'
PROFILE = ROOT / 'reviews/cross-model-applicability-execution-v1/prepared-01/model-profiles.json'
CONDITIONS = [('C0', False, 'none'), ('D', True, 'none'), ('L', False, 'definition'),
              ('LD', True, 'definition'), ('LC', False, 'category'), ('LDC', True, 'category')]
ROLES = ('pre_answer', 'lexicon_end', 'demos_end', 'query_end', 'query_all', 'query_focal')
ROLE_LABELS = ('答案前', '词典结束／示例前', '示例结束／查询前', '查询正文末尾', '查询正文各位置平均', '查询词形位置平均')
CODE = [ROOT / 'src/diagnostics' / f'case_attention_{n}_v1.py' for n in ('inputs', 'capture', 'runtime', 'report')]
CODE += [ROOT / 'scripts/review' / n for n in ('run_case_attention_v1.py', 'test_case_attention_v1.py', 'audit_case_attention_v1.py')]
CODE += [ROOT / 'scripts/review/test_case_attention_viewer_v1.cjs']
CODE += [ROOT / 'tools/case_attention_viewer_v1/viewer.html', PUBLIC / 'PROTOCOL.md']


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def lines(path):
    return [json.loads(s) for s in Path(path).read_text(encoding='utf-8').splitlines()]


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        while chunk := f.read(4 * 1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def info(path):
    p = Path(path).absolute()
    return {'path': str(p), 'bytes': p.stat().st_size, 'sha256': sha(p)}


def verify(item):
    p = Path(item['path'])
    require(p.is_file() and p.stat().st_size == item['bytes'] and sha(p) == item['sha256'], f'Pinned bytes changed: {p}')


def write(path, value):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('xb') as f:
        f.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False).encode('utf-8') + b'\n')


def atomic(path, value, replace=False):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + f'.{os.getpid()}.tmp')
    with tmp.open('xb') as f:
        f.write(canonical(value) + b'\n')
        f.flush()
        os.fsync(f.fileno())
    try:
        if replace:
            os.replace(tmp, p)
        else:
            os.link(tmp, p)
    finally:
        tmp.unlink(missing_ok=True)


def tokenizer():
    # These environment flags apply only to CPU preparation/audit processes.
    for name in ('USE_TORCH', 'USE_TF', 'USE_FLAX', 'USE_TORCH_XLA'):
        os.environ[name] = '0'
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(str(ROOT / 'models/base/Qwen3-8B'), local_files_only=True,
                                        trust_remote_code=False, use_fast=True)


def load_materials():
    reviews = {r['record_id']: r for r in lines(REFERENCES)}
    cases, refs, changes = [], [], []
    for file in ('01-541.json', '07-3169.json'):
        c = read(CARDS / file)
        q = c['query']
        qr = reviews['query:' + q['id']]
        require(qr['review']['status'] == 'confirmed' and not qr['stale'], 'Unconfirmed query reference')
        require(qr['source']['text'] == q['content'], 'Query reference text mismatch')
        case = {'query_id': q['id'], 'query_text': q['content'], 'lexicon': c['lexicon_entries'], 'demos': []}
        for n, d in enumerate(c['demonstrations'], 1):
            r = reviews['demo:' + d['id']]
            require(r['source']['text'] == d['content'] and r['review']['status'] == 'confirmed' and not r['stale'], 'Demo reference mismatch')
            old, reviewed = d['projection']['hate'], r['review']['values']['hate']
            require(reviewed in ('hate', 'non-hate'), 'Unresolved demo label')
            item = {'id': d['id'], 'slot': n, 'text': d['content'], 'answer': '有' if reviewed == 'hate' else '无',
                    'original_answer': old, 'reviewed_answer': reviewed, 'reference_record_id': r['record_id'],
                    'review_sha256': digest(r), 'review_provenance': r['review'],
                    'new_individual_adjudication': False}
            case['demos'].append(item)
            if old != reviewed:
                changes.append({'query_id': q['id'], 'demo_id': d['id'], 'slot': n,
                                'original_answer': '有' if old == 'hate' else '无', 'display_answer': item['answer']})
        refs.append({'query_id': q['id'], 'reference': '有' if qr['review']['values']['hate'] == 'hate' else '无',
                     'original_gold': q['projection']['hate'], 'source': info(REFERENCES), 'review': qr['review'],
                     'new_individual_adjudication': False, 'scope': 'Existing reviewed binary label transferred to this new task; no new severity invented.'})
        cases.append(case)
    require(len(changes) == 4, 'Reviewed-vs-original answer inventory changed')
    return cases, refs, changes


def render(case, with_demos, lexicon_mode, system):
    """Same outer renderer as the current task; character spans do not enter prompts."""
    pieces, spans = [], []
    length = 0

    def add(text, sid=None, label=None, kind='structure', parent=None):
        nonlocal length
        start = length
        pieces.append(text)
        length += len(text)
        if sid:
            spans.append({'id': sid, 'label': label or sid, 'kind': kind, 'parent': parent,
                          'char_start': start, 'char_end': length})

    add('【参考词典】\n')
    if lexicon_mode == 'none':
        add('（无）', 'lexicon_empty', '词典为空', 'placeholder')
    else:
        for n, entry in enumerate(case['lexicon'], 1):
            require(len(entry['senses']) == 1, 'This case frame expects one sense per entry')
            sense = entry['senses'][0]
            eid = entry['lexicon_id']
            if n > 1:
                add('\n\n')
            start = length
            add(f'词条{n}：\n词形：')
            add(entry['term'], eid + ':term', entry['term'] + '·词形', 'lexicon_term', eid)
            add('\n释义：')
            add(sense['definition'], eid + ':definition', entry['term'] + '·释义', 'lexicon_definition', eid)
            if lexicon_mode == 'category':
                add('\n类别：')
                add(json.dumps(sense['categories'], ensure_ascii=False, separators=(',', ':')), eid + ':category',
                    entry['term'] + '·类别', 'lexicon_category', eid)
            spans.append({'id': eid, 'label': entry['term'], 'kind': 'lexicon', 'parent': None,
                          'char_start': start, 'char_end': length})
    add('\n\n')
    lexicon_boundary = length
    add('【参考示例】\n')
    if not with_demos:
        add('（无）', 'demos_empty', '示例为空', 'placeholder')
    else:
        for n, demo in enumerate(case['demos'], 1):
            if n > 1:
                add('\n\n')
            start, did = length, 'demo-' + demo['id']
            add(f'示例{n}：\n文本：')
            add(demo['text'], did + ':text', f'示例{n} #{demo["id"]}·正文', 'demo_text', did)
            add('\n答案：')
            add(demo['answer'], did + ':answer', f'示例{n} #{demo["id"]}·答案', 'demo_answer', did)
            spans.append({'id': did, 'label': f'示例{n} #{demo["id"]}', 'kind': 'demo', 'parent': None,
                          'char_start': start, 'char_end': length})
    add('\n\n')
    demo_boundary = length
    add('【待判断文本】\n')
    add(case['query_text'], 'query', '待判断文本', 'query')
    return ''.join(pieces), spans, {'lexicon_end': lexicon_boundary, 'demos_end': demo_boundary}


def build_input(case, condition, tok, system):
    cid, demos, mode = condition
    user, spans, boundaries = render(case, demos, mode, system)
    messages = [{'role': 'system', 'content': system}, {'role': 'user', 'content': user}]
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    require(prompt.count(user) == 1 and prompt.count(system) == 1, 'Ambiguous rendered message')
    shift, system_start = prompt.index(user), prompt.index(system)
    for s in spans:
        s['char_start'] += shift
        s['char_end'] += shift
    spans.append({'id': 'system', 'label': '任务指令', 'kind': 'system', 'parent': None,
                  'char_start': system_start, 'char_end': system_start + len(system)})
    encoded = tok(prompt, add_special_tokens=False, return_offsets_mapping=True)
    ids, offsets = encoded['input_ids'], encoded['offset_mapping']
    require(0 < len(ids) <= 8192, 'Input must not be truncated')
    for answer, token in [('有', 18830), ('无', 42192)]:
        require(tok.encode(answer, add_special_tokens=False) == [token] and
                tok.encode(prompt + answer, add_special_tokens=False) == ids + [token], 'Single-token answer boundary failed')

    # A token touching two semantic leaves is assigned to the boundary bucket,
    # never counted twice or split into fictional fractional tokens.
    leaves = [s for s in spans if s['kind'] not in ('demo', 'lexicon')]
    owners, crossing = [], []
    for i, (a, b) in enumerate(offsets):
        hits = [s for s in leaves if a < s['char_end'] and b > s['char_start']]
        full = [s for s in hits if s['char_start'] <= a < b <= s['char_end']]
        if len(full) == 1 and len(hits) == 1:
            owner = full[0]['id']
        else:
            owner = 'boundary' if hits else 'structure'
            if hits:
                crossing.append({'token': i, 'offset': [a, b], 'touching': [s['id'] for s in hits]})
        owners.append(owner)
    for s in spans:
        if s['kind'] in ('demo', 'lexicon'):
            s['token_positions'] = [i for i, (a, b) in enumerate(offsets)
                                    if s['char_start'] <= a < b <= s['char_end']]
        else:
            s['token_positions'] = [i for i, owner in enumerate(owners) if owner == s['id']]
    for sid, label in [('structure', '模板／分隔符'), ('boundary', '跨片段边界 token')]:
        spans.append({'id': sid, 'label': label, 'kind': sid, 'parent': None, 'char_start': None, 'char_end': None,
                      'token_positions': [i for i, owner in enumerate(owners) if owner == sid]})

    roles = {}
    prefix_proofs = []
    for role, present in [('lexicon_end', mode != 'none'), ('demos_end', demos)]:
        cut = shift + boundaries[role]
        prefix = tok.encode(prompt[:cut], add_special_tokens=False)
        require(prefix == ids[:len(prefix)] and offsets[len(prefix) - 1][1] <= cut, 'Boundary cuts a retokenized token')
        roles[role] = [len(prefix) - 1] if present else []
        if present:
            prefix_proofs.append({'role': role, 'char_cut': cut, 'tokens': len(prefix),
                                  'prefix_ids_sha256': digest(prefix), 'prefix_encoding_exact': True})
    q = next(s for s in spans if s['id'] == 'query')
    qpos = [i for i, (a, b) in enumerate(offsets) if a < q['char_end'] and b > q['char_start']]
    focal = '男同' if case['query_id'] == '541' else '嘿嘿'
    fp = set()
    start = 0
    while (at := case['query_text'].find(focal, start)) >= 0:
        a, b = q['char_start'] + at, q['char_start'] + at + len(focal)
        fp.update(i for i, (x, y) in enumerate(offsets) if x < b and y > a)
        start = at + len(focal)
    require(fp and qpos, 'Missing query/focal roles')
    roles.update(pre_answer=[len(ids) - 1], query_end=[qpos[-1]], query_all=qpos, query_focal=sorted(fp))
    return {'request_id': f'case-{case["query_id"]}-{cid}', 'query_id': case['query_id'], 'condition': cid,
            'with_demos': demos, 'lexicon_mode': mode, 'messages': messages, 'prompt_text': prompt,
            'prompt_sha256': hashlib.sha256(prompt.encode('utf-8')).hexdigest(), 'input_ids': ids,
            'input_ids_sha256': digest(ids), 'prompt_tokens': len(ids), 'token_offsets': offsets,
            'token_text': [tok.decode([i]) for i in ids], 'token_owners': owners, 'spans': spans,
            'roles': {name: roles[name] for name in ROLES}, 'prefix_proofs': prefix_proofs,
            'boundary_tokens': crossing, 'candidate_tokens': {'有': 18830, '无': 42192}}


def prepare(output=PREPARED):
    from diagnostics.cross_model_applicability_execution_v1 import runtime_versions, framework_sources
    out = Path(output).absolute()
    require(not out.exists(), 'Preparation refuses an existing directory')
    tok, system = tokenizer(), (TASK / 'model-task.txt').read_text(encoding='utf-8')
    cases, refs, changes = load_materials()
    inputs = [build_input(case, arm, tok, system) for case in cases for arm in CONDITIONS]
    profile = read(PROFILE)['models'][0]
    profile = {k: v for k, v in profile.items() if k not in ('format_probe_ids', 'production_ids')}
    profile['heads'] = read(Path(profile['local_directory']) / 'config.json')['num_attention_heads']
    require(profile['model_key'] == 'qwen3-8b' and (profile['layers'], profile['heads']) == (36, 32), 'Unexpected model')
    for item in profile['metadata_sources']:
        verify(item)
    for item in profile['weight_sources']:
        st = Path(item['path']).stat()
        require((st.st_size, st.st_mtime_ns) == (item['bytes'], item['mtime_ns']), 'Weight stat mismatch')
    out.mkdir(parents=True)
    (out / 'inputs.jsonl').write_bytes(b''.join(canonical(r) + b'\n' for r in inputs))
    write(out / 'materials.json', {'cases': cases, 'display_answer_changes': changes,
                                  'dictionary_policy': 'Exact original case senses/categories; no AI rewrites or new entries.',
                                  'answer_policy': 'Existing human-reviewed binary labels, mapped to 有/无; user selected this policy.'})
    write(out / 'analysis-references.json', {'scorer_must_not_parse': True, 'references': refs})
    write(out / 'model-profile.json', profile)
    prefix_count = sum(len(r['prefix_proofs']) for r in inputs)
    plan = {'schema_version': 'case-attention-plan/v1', 'status': 'cpu_preparation_gpu_pending',
            'runtime_versions': runtime_versions(), 'conditions': [r[0] for r in CONDITIONS], 'query_ids': ['541', '3169'],
            'roles': dict(zip(ROLES, ROLE_LABELS)), 'layers': 36, 'heads': 32, 'scope': 'two exposed descriptive development cases',
            'attention_capture': 'Native eager probabilities, read-only hook, per-head FP64 mean over registered target rows; no renormalization over materials.',
            'attention_storage_dtype': 'float64', 'compute_dtype': 'float32', 'query_reference_join_during_worker': False,
            'acceptance': {'margin_repeat_order_hook_cap': 0.0, 'margin_padding_prefix_cap': 0.001,
                          'attention_repeat_order_cap': 0.0, 'attention_element_cap': 0.0001,
                          'attention_row_l1_cap': 0.001, 'attention_row_sum_cap': 0.000002,
                          'margin_bound_floor': 0.000001, 'attention_mass_bound_floor': 0.0000001},
            'passes': ['baseline', 'capture', 'repeat', 'reverse', 'left', 'right', 'prefix'],
            'format': {'all_conditions': True, 'exact_label_then_eos': True, 'max_new_tokens': 8},
            'budget': {'unique_prompts': 12, 'engineering_full_forwards': 72, 'prefix_forwards': prefix_count,
                       'production_forwards': 12, 'format_extra_forward_max': 84,
                       'total_forward_max': 72 + prefix_count + 12 + 84,
                       'usual_if_label_then_eos': 72 + prefix_count + 12 + 12},
            'allocation': None, 'GPU_qualified': False, 'auto_launch_or_poll': False,
            'terminal_restart': False, 'references_are_newly_adjudicated': False,
            'historical_input_replay_claimed': False, 'human_choice': '使用已审核标签（推荐）'}
    write(out / 'execution-plan.json', plan)
    src = [*CODE, REFERENCES, PROFILE, TASK / 'model-task.txt', TASK / 'scoring-spec.json', TASK / 'validate_contract.py',
           TASK / 'manifest.json', CARDS / '01-541.json', CARDS / '07-3169.json',
           ROOT / 'src/diagnostics/cross_model_applicability_models_v1.py',
           ROOT / 'src/diagnostics/cross_model_applicability_execution_v1.py', *framework_sources()]
    write(out / 'source-ledger.json', {'files': [info(p) for p in sorted(set(src))],
                                     'weight_hashes_reused_from_pinned_inventory': True,
                                     'full_weight_rehash_required_at_future_load': True})
    write(out / 'preparation-summary.json', {'prompt_tokens': {r['request_id']: r['prompt_tokens'] for r in inputs},
        'candidate_boundary_checks': 24, 'prefix_encoding_checks': prefix_count,
        'attention_bytes_one_full_pass': sum(36 * 32 * 6 * r['prompt_tokens'] * 8 for r in inputs),
        'boundary_token_count': sum(len(r['boundary_tokens']) for r in inputs),
        'research_model_weights_loaded': False, 'GPU_forward_executed': False, 'display_answer_changes': changes})
    documents(out)
    return {'directory': str(out), 'prompts': len(inputs), 'budget': plan['budget']}


def documents(prepared):
    """Human-readable complete material and position exports, without inference."""
    import csv
    import io
    out = Path(prepared)
    require(not (out / 'manifest.json').exists(), 'Cannot edit a sealed preparation')
    material, inputs = read(out / 'materials.json'), lines(out / 'inputs.jsonl')
    text = ['# 两个真实案例的完整输入材料', '',
            '使用已审核二元答案映射为有／无。原文、原数据答案和此前人审来源保留；本次不是新的逐条严重度裁决。', '']
    for case in material['cases']:
        text.extend([f'## 查询 #{case["query_id"]}', '', case['query_text'], '', '### 完整词典', ''])
        for entry in case['lexicon']:
            sense = entry['senses'][0]
            text.extend([f'#### {entry["term"]} / {entry["lexicon_id"]}', '', sense['definition'], '',
                         '原类别：' + json.dumps(sense['categories'], ensure_ascii=False), ''])
        text.extend(['### 完整真实示例（原顺序）', ''])
        for d in case['demos']:
            text.extend([f'#### 示例 {d["slot"]} / #{d["id"]}', '', d['text'], '',
                         f'本次展示答案：**{d["answer"]}**；原数据答案：{d["original_answer"]}；已审核答案：{d["reviewed_answer"]}。', ''])
    with (out / 'MATERIALS.md').open('x', encoding='utf-8') as f:
        f.write('\n'.join(text))
    s = io.StringIO()
    writer = csv.writer(s, delimiter='\t')
    writer.writerow(['request_id', 'prompt_tokens', 'role', 'applicable', 'token_positions', 'last_token_ids', 'prefix_verified'])
    for r in inputs:
        for role in ROLES:
            positions = r['roles'][role]
            writer.writerow([r['request_id'], r['prompt_tokens'], role, bool(positions), json.dumps(positions),
                             json.dumps([r['input_ids'][i] for i in positions]), role in [p['role'] for p in r['prefix_proofs']]])
    with (out / 'positions.tsv').open('x', encoding='utf-8') as f:
        f.write(s.getvalue())
    raw_bytes = 6 * sum(36 * 32 * 6 * r['prompt_tokens'] * 8 for r in inputs)
    raw_bytes += sum(36 * 32 * 6 * p['tokens'] * 8 for r in inputs for p in r['prefix_proofs'])
    estimate = {'status': 'estimate_not_measurement', 'hardware_assumption': 'one idle NVIDIA L20 48GB, pinned FP32 Qwen3-8B',
                'expected_total_minutes': [10, 20], 'recommended_window_minutes': 30,
                'includes': ['two model loads/weight checks', 'engineering + format', 'explicit full phase', 'CPU reconstruction and report'],
                'excludes': ['waiting for idle GPU', 'failure investigation', 'authorization wait'],
                'scientific_run_forward_count_usual': 110, 'forward_count_max': 182,
                'raw_attention_bytes_estimate': raw_bytes, 'recommended_free_disk_bytes': 8_000_000_000,
                'FP32_weight_bytes': read(out / 'model-profile.json')['fp32_weight_bytes'],
                'basis': 'Earlier completed 8B run: 2716 forwards/1289 s engineering, 384 forwards/291 s full, including phase overhead. New prompts/attention transfer and disk checks differ; no new GPU benchmark.',
                'first_real_engineering_phase_updates_estimate': True}
    write(out / 'time-estimate.json', estimate)
    readme = '''# #541 / #3169 注意力实验：GPU 前交付

完整输入、采集、工程检查、未来 GPU 执行器、CPU 分析和交互查看器已实现。本目录不包含真实模型注意力结果。实际设备 allocation=null，GPU 数值资格待取得；未创建绑定、GPU run、队列或空闲轮询。

先打开 [完整材料](MATERIALS.md)、[位置表](positions.tsv)、[查看器](viewer.html) 和 [协议](../../../docs/research/experiment-plans/case-attention-v1/PROTOCOL.md)。查看器在 GPU 未运行时显示真实输入和灰色待测区域，不显示伪造热图。

本次使用当前完整任务指令与单 token“有／无”。两个真实查询、各自完整的 10 个真实示例和 2／6 个原词条；原顺序；六条件各一条，共 12 个 prompt。用户已选择按既有已审核答案展示，4 条示例答案变化可见 MATERIALS.md 和 materials.json；原始标签未覆盖。

主位置：答案前。辅助：词典结束／示例前、示例结束／查询前、查询末尾、查询全句平均、词形位置平均。后两种平均保留全部头，使用 FP64 归约。读取阶段的对比只作描述；未读到的片段为 NA。

工程：72 个完整前向 + 14 个原生截断前缀 + 所有 12 条 greedy 单字/EOS 检查；正常输出时多 12 次前向。通过并正常退出后，显式 full 再做 12 条。常规合计 110 次，上限 182 次。预先固定数值门槛、结果复建、STOP 暂停／显式恢复、损坏检测、终态保护和只读 hook 检查均已实现。

时间估计：在一张空闲 L20 48GB 上，GPU 工程＋正式阶段及 CPU 复核/报告约 **10–20 分钟**；建议留 **30 分钟**窗口。估计包含两阶段加载／权重核验，不含等待空闲或故障排查，不能当成新环境实测吞吐。原始注意力预计约 4–5 GB，建议至少留 8 GB 磁盘；FP32 权重约 32.76 GB，另需运行缓冲。当前没有查询实时 GPU 占用。

## CPU 入口

在仓库根目录，以固定 `.conda/stage1-p0/bin/python` 执行。所有命令加 `PYTHONUTF8=1`。

```bash
PYTHONUTF8=1 .conda/stage1-p0/bin/python scripts/review/run_case_attention_v1.py validate
PYTHONUTF8=1 .conda/stage1-p0/bin/python scripts/review/audit_case_attention_v1.py
PYTHONUTF8=1 CUDA_VISIBLE_DEVICES='' .conda/stage1-p0/bin/python scripts/review/test_case_attention_v1.py
```

## 下一次明确 GPU 窗口中的入口（本次未执行）

先 bind：指定届时核实空闲的 `--gpu INDEX`、新的 `--output` 绑定文件与具体 `--authorization-note`。bind 会核验权重、显存及占用，不能从旧空闲回执推定现在空闲。

```text
run_case_attention_v1.py bind --gpu INDEX --output NEW_BOUND --authorization-note ACTUAL_FUTURE_AUTHORIZATION
run_case_attention_v1.py run --bound NEW_BOUND --run reviews/case-attention-v1/run-01
run_case_attention_v1.py check --run reviews/case-attention-v1/run-01
run_case_attention_v1.py run --bound NEW_BOUND --run reviews/case-attention-v1/run-01 --phase full
run_case_attention_v1.py check --run reviews/case-attention-v1/run-01
run_case_attention_v1.py analyze --run reviews/case-attention-v1/run-01 --output reviews/case-attention-v1/results-01
```

上例的 run 首次只能 engineering，只有通过所有资格、格式检查且正常释放的 qualified 状态才可 full。暂停时保留 STOP，明确决定继续后移走 STOP 并用相同 phase 与 `--resume`；failed/complete 不可重启。任何异常保留现场，不自动降低精度、改阈值或重试。

最终查看器可离线选择 `*.view.json`，或在结果目录用 `python -m http.server --bind 127.0.0.1 PORT` 启动本地读取。查询、条件、角色、层、头、片段层级、密度／总质量和阶段／条件差图均可切换，统一色标可手动指定；可导出 SVG。图形副本 FP32，精确数值使用原始 FP64 注意力数组与 aggregates.json。

研究范围仅为两个已暴露案例。最新任务下是否重现历史错误尚未知；注意力热图不能独立证明适用性机制、词典损害、示例修复或跨样本规律。参考答案只在 raw seal 和正常 worker release 后加入 CPU 分析。
'''
    with (out / 'README.md').open('x', encoding='utf-8') as f:
        f.write(readme)


def seal(prepared):
    out = Path(prepared)
    require(not (out / 'manifest.json').exists(), 'Manifest already sealed; create a new version')
    for name in ('cpu-tests.json', 'cpu-audit.json', 'viewer-cpu-test.json', 'README.md', 'viewer.html'):
        require((out / name).is_file(), f'Missing delivery artifact {name}')
    for name in ('cpu-tests.json', 'cpu-audit.json', 'viewer-cpu-test.json'):
        receipt = read(out / name)
        require(receipt.get('status') == 'pass' and not receipt.get('CUDA_initialized', False), f'Failed CPU receipt: {name}')
        for source in receipt.get('implementation_snapshot', []):
            verify(source)
    validate(out, sealed=False)
    for item in read(out / 'source-ledger.json')['files']:
        verify(item)
    write(out / 'manifest.json', {'schema_version': 'case-attention-prepared/v1', 'status': 'cpu_complete_gpu_pending',
          'artifacts': [info(p) for p in sorted(out.iterdir()) if p.is_file()],
          'sources': read(out / 'source-ledger.json')['files'], 'GPU_allocation': None, 'GPU_qualified': False,
          'immutable_after_seal': True})
    return info(out / 'manifest.json')


def validate(prepared, sealed=True, weights=False):
    from diagnostics.cross_model_applicability_execution_v1 import runtime_versions
    p = Path(prepared)
    if sealed:
        m = read(p / 'manifest.json')
        for item in m['artifacts'] + m['sources']:
            verify(item)
    for item in read(p / 'source-ledger.json')['files']:
        verify(item)
    plan, profile = read(p / 'execution-plan.json'), read(p / 'model-profile.json')
    require(plan['runtime_versions'] == runtime_versions(), 'Runtime versions differ')
    for item in profile['metadata_sources']:
        verify(item)
    for item in profile['weight_sources']:
        if weights:
            verify(item)
        else:
            st = Path(item['path']).stat()
            require((st.st_size, st.st_mtime_ns) == (item['bytes'], item['mtime_ns']), 'Weight stat changed')
    requests = lines(p / 'inputs.jsonl')
    require(len(requests) == 12 and len({r['request_id'] for r in requests}) == 12, 'Input inventory differs')
    for r in requests:
        require(r['prompt_tokens'] == len(r['input_ids']) and digest(r['input_ids']) == r['input_ids_sha256'], 'Input token binding differs')
        require(hashlib.sha256(r['prompt_text'].encode('utf-8')).hexdigest() == r['prompt_sha256'], 'Prompt binding differs')
        require(r['roles']['pre_answer'] == [len(r['input_ids']) - 1], 'Answer position differs')
    return plan, profile, requests
