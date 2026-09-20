#!/usr/bin/env python3
"""Prepare nine CPU-only, reference-free prompts and a separate human review draft.

No model weights, torch, CUDA, GPU allocation, or executor are used here.
This is a new draft; historical experiment inputs and human decisions are read only.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import csv
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from diagnostics import case_attention_inputs_v1 as c

WORK = ROOT / 'reviews/hehe-sense-context-v1'
DEFAULT = WORK / 'draft-01'
OLD = ROOT / 'reviews/case-content-replacement-v1/prepared-01'
AUDITOR = ROOT / 'scripts/review/audit_hehe_sense_context_v1.py'
FOCAL = 'lex-0419'
QUERY3 = '有人说：“主要是被嘿嘿玩过的，那不是一般的思想，那得多么的。。。”我不赞同这种说法，不能因为一个人的交往经历就贬低对方。'
REQUEST = '增加一个条件，同时提供原释义以及普通义，然后给我一下完整的新材料组合给我审核一下'


def write_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as f:
        f.write(text)


def write_lines(path, records):
    with Path(path).open('xb') as f:
        for record in records:
            f.write(c.canonical(record) + b'\n')


def table(path, fields, rows):
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fields, delimiter='\t', lineterminator='\n')
    writer.writeheader()
    writer.writerows({key: '' if row.get(key) is None else row[key] for key in fields} for row in rows)
    write_text(path, buffer.getvalue())


def link(path, label=None):
    return f'[{label or Path(path).name}]({Path(path).absolute()})'


def source_pins(paths):
    pins = {}
    for manifest in (c.PREPARED / 'manifest.json', OLD / 'manifest.json'):
        data = c.read(manifest)
        for record in data['artifacts'] + data['sources']:
            pins[record['path']] = record
    for path in paths:
        key = str(Path(path).absolute())
        c.require(key in pins, f'Historical source is not pinned: {key}')
        c.verify(pins[key])


def build(output):
    out = Path(output).absolute()
    c.require(not out.exists(), 'Existing drafts are immutable; select a new directory')
    selector = c.read(c.TASK.parent / 'current.json')
    c.require(selector['frozen_directory'] == str(c.TASK.relative_to(ROOT)), 'Latest task changed')
    used = [c.PREPARED / 'materials.json', c.REFERENCES, OLD / 'rendered-cases.jsonl',
            OLD / 'scoring-inputs.jsonl', OLD / 'adoption.json', c.TASK / 'model-task.txt',
            ROOT / 'src/diagnostics/case_attention_inputs_v1.py']
    source_pins(used)
    originals = c.read(c.PREPARED / 'materials.json')
    base = next(case for case in originals['cases'] if case['query_id'] == '3169')
    selected = {record['record_id']: record for record in c.lines(c.REFERENCES)
                if record['record_id'] in ('query:3169', 'demo:3660')}
    c.require(len(selected) == 2, 'Missing inherited human records')
    for rid, expected in [('query:3169', 'non-hate'), ('demo:3660', 'hate')]:
        record = selected[rid]
        c.require(record['review']['status'] == 'confirmed' and not record['stale'], 'Stale reference')
        c.require(record['review']['values']['hate'] == expected, 'Inherited reference changed')
    query2 = selected['demo:3660']['source']['text']
    c.require(f'“{query2}”' in QUERY3, 'Quoted attack is not byte-identical')
    prior = {row['request_id']: row for row in c.lines(OLD / 'rendered-cases.jsonl')
             if row['request_id'] in ('ccr-3169-L-base', 'ccr-3169-L-L05')}
    prior_inputs = {row['request_id']: row for row in c.lines(OLD / 'scoring-inputs.jsonl')
                    if row['request_id'] in prior}
    adoption = next(item for item in c.read(OLD / 'adoption.json')['items'] if item['item_id'] == 'L05')
    c.require(adoption['decision'] == 'accept', 'L05 not adopted')
    original = next(e for e in base['lexicon'] if e['lexicon_id'] == FOCAL)['senses'][0]['definition']
    ordinary = next(e for e in prior['ccr-3169-L-L05']['case']['lexicon']
                    if e['lexicon_id'] == FOCAL)['senses'][0]['definition']
    dictionaries = [
        {'dictionary_id': 'D01', 'name': '原释义', 'definition': original,
         'component_order': ['original'], 'provenance': 'Exact previous original definition',
         'new_human_decision': None},
        {'dictionary_id': 'D02', 'name': '普通义', 'definition': ordinary,
         'component_order': ['ordinary'], 'provenance': 'Exact accepted L05; AI authorship retained',
         'inherited_adoption': adoption, 'new_human_decision': None},
        {'dictionary_id': 'D03', 'name': '原释义＋普通义', 'definition': original + '\n' + ordinary,
         'component_order': ['original', 'ordinary'], 'separator': '\n',
         'provenance': 'Assistant composition of the two unchanged paragraphs; new combination pending',
         'new_human_decision': None},
    ]
    queries = [
        {'query_id': 'Q01', 'source_id': '3169', 'kind': 'real_unchanged', 'name': '普通聊天',
         'text': base['query_text'], 'reference': '无', 'reference_status': 'inherited_human_confirmed',
         'reference_record_id': 'query:3169', 'new_human_decision': None,
         'exposure': 'Previously exposed focal query and outcomes',
         'interpretation': '既有普通聊天原文及无标签；本轮没有新裁决。'},
        {'query_id': 'Q02', 'source_id': '3660', 'kind': 'real_unchanged', 'name': '作者实施攻击',
         'text': query2, 'reference': '有', 'reference_status': 'inherited_human_confirmed',
         'reference_record_id': 'demo:3660', 'new_human_decision': None,
         'new_role_selection_status': 'pending', 'exposure': 'Previously exposed as a demonstration',
         'interpretation': '原文及严重度2／有已有审核；改作查询的选用待审。还含性经历贬损线索，不能把有标签完全归于嘿嘿。'},
        {'query_id': 'Q03', 'source_id': None, 'kind': 'assistant_derived', 'name': '引用并反对攻击',
         'text': QUERY3, 'reference': None, 'proposed_reference': '无', 'proposed_severity': 0,
         'reference_status': 'pending', 'reference_record_id': None, 'new_human_decision': None,
         'parent_query_id': 'Q02', 'authorship': 'assistant',
         'exposure': 'New assistant rewrite of an exposed source; not independent confirmation',
         'interpretation': '保留整段Q02引文，批评这种说法并反对按交往经历贬低人；拟判无。正文未解释嘿嘿的义项，未另行辱骂发言者。'},
    ]
    review_items = [
        {'item_id': 'Q02', 'title': '选用真实 #3660 作为攻击查询',
         'review_scope': '同意该查询用途，并保留已有有标签及性经历贬损的限制；不重新裁定旧标签。'},
        {'item_id': 'Q03', 'title': '反驳改写及参考答案',
         'review_scope': '审核完整新正文和拟议无／0级，并确认引文与作者反对立场的区分。'},
        {'item_id': 'D03', 'title': '双义同时提供的具体写法',
         'review_scope': '同一嘿嘿词条内，原释义在前、普通义在后，以一个换行分隔；两段原文不变。'},
    ]
    for item in review_items:
        item.update(status='pending', human_decision=None, human_reason=None,
                    proposal_author='assistant', GPU_eligible=False)
    material = {'schema': 'hehe-sense-context-materials/v1', 'status': 'review_draft',
                'queries': queries, 'dictionaries': dictionaries,
                'base_lexicon': deepcopy(base['lexicon']), 'focal_lexicon_id': FOCAL,
                'focal_term': '嘿嘿', 'with_demos': False,
                'dictionary_category_fields_displayed': False, 'review_items': review_items}
    system = (c.TASK / 'model-task.txt').read_text(encoding='utf-8')
    tok = c.tokenizer()
    inputs, ledger, components, historical_matches, comparisons = [], [], [], [], []
    for query in queries:
        for dictionary in dictionaries:
            case = {'query_id': query['query_id'], 'query_text': query['text'],
                    'lexicon': deepcopy(base['lexicon']), 'demos': []}
            entry = next(e for e in case['lexicon'] if e['lexicon_id'] == FOCAL)
            entry['senses'][0]['definition'] = dictionary['definition']
            row = json.loads(c.canonical(c.build_input(case, ('L', False, 'definition'), tok, system)))
            row['request_id'] = f'hsc-{query["query_id"]}-{dictionary["dictionary_id"]}'
            row['dictionary_id'] = dictionary['dictionary_id']
            inputs.append(row)
            definition_span = next(s for s in row['spans'] if s['id'] == FOCAL + ':definition')
            cursor = definition_span['char_start']
            component_rows = []
            for name in dictionary['component_order']:
                content = {'original': original, 'ordinary': ordinary}[name]
                end = cursor + len(content)
                owned = [i for i, (a, b) in enumerate(row['token_offsets'])
                         if cursor <= a < b <= end and row['token_owners'][i] == FOCAL + ':definition']
                touched = [i for i, (a, b) in enumerate(row['token_offsets']) if a < end and b > cursor]
                component_rows.append({'component': name, 'text': content, 'char_start': cursor,
                                       'char_end': end, 'owned_token_positions': owned,
                                       'touching_token_positions': touched})
                cursor = end + 1
            assigned = {i for part in component_rows for i in part['owned_token_positions']}
            components.append({'request_id': row['request_id'], 'parent_span_id': definition_span['id'],
                               'components': component_rows,
                               'separator_or_cross_boundary_tokens': sorted(set(definition_span['token_positions']) - assigned),
                               'overlap_policy': 'Parent overlaps components; never add parent and children. Touching tokens can overlap; not additive.'})
            previous_id = None
            if query['query_id'] == 'Q01' and dictionary['dictionary_id'] != 'D03':
                previous_id = 'ccr-3169-L-base' if dictionary['dictionary_id'] == 'D01' else 'ccr-3169-L-L05'
                old = prior_inputs[previous_id]
                for field in ('prompt_text', 'messages', 'input_ids', 'roles', 'spans', 'token_offsets', 'token_owners'):
                    c.require(row[field] == old[field], f'Historical input mismatch: {field}')
                historical_matches.append({'request_id': row['request_id'], 'old_request_id': previous_id,
                                           'prompt_sha256': row['prompt_sha256'], 'input_ids_sha256': row['input_ids_sha256'],
                                           'prompt_and_tokens_exact': True, 'old_scores_used': False})
            ledger.append({'request_id': row['request_id'], 'query_id': query['query_id'],
                           'dictionary_id': dictionary['dictionary_id'], 'prompt_tokens': row['prompt_tokens'],
                           'pre_answer': row['roles']['pre_answer'][0],
                           'lexicon_end': row['roles']['lexicon_end'][0],
                           'definition_owned_tokens': len(definition_span['token_positions']),
                           'historical_input': previous_id})
        for left, right in [('D02', 'D01'), ('D03', 'D01'), ('D03', 'D02')]:
            comparisons.append({'comparison_id': f'{query["query_id"]}-{left}-minus-{right}',
                                'terms': [{'request_id': f'hsc-{query["query_id"]}-{left}', 'coefficient': 1},
                                          {'request_id': f'hsc-{query["query_id"]}-{right}', 'coefficient': -1}],
                                'metric': 'z[42192]-z[18830]', 'value': None, 'bound': None,
                                'independent_sample': False})
    c.require(len({row['prompt_sha256'] for row in inputs}) == 9, 'Duplicate prompts')
    c.require('torch' not in sys.modules, 'CPU preparation unexpectedly imported torch')
    out.mkdir(parents=True)
    c.write(out / 'materials.json', material)
    c.write(out / 'source-records.json', {'records': selected, 'L05_adoption': adoption,
                                        'new_human_records_created': 0})
    c.write(out / 'review-reference-proposals.json', {'excluded_from_model_inputs': True, 'queries': queries})
    c.write(out / 'feedback-template.json', {'status': 'blank_not_submitted', 'items': review_items,
                                            'silence_is_acceptance': False})
    c.write(out / 'definition-components.json', {'records': components})
    c.write(out / 'comparisons.json', {'comparisons': comparisons})
    c.write(out / 'historical-input-matches.json', {'matches': historical_matches})
    c.write(out / 'authorization-context.json', {'user_request': REQUEST, 'scope': 'data and human review draft',
                                               'GPU_started': False, 'current_GPU_window': None,
                                               'previous_completed_runs_are_terminal': True})
    c.write(out / 'design.json', {
        'status': 'review_draft_not_execution_freeze', 'model_key': 'qwen3-8b',
        'unique_scientific_inputs': 9, 'queries': ['Q01', 'Q02', 'Q03'],
        'dictionaries': ['D01', 'D02', 'D03'], 'with_demos': False,
        'task_source': c.info(c.TASK / 'model-task.txt'), 'candidate_tokens': {'有': 18830, '无': 42192},
        'primary_metric': 'm=z[42192]-z[18830]', 'paired_comparisons': 9,
        'reference_aligned_margin': {'Q01': 'm', 'Q02': '-m', 'Q03': 'm only if proposed 无 is adopted'},
        'attention_plan': {'layers': 36, 'heads': 32, 'primary_role': 'pre_answer',
                           'auxiliary_roles': ['lexicon_end', 'query_focal', 'query_end', 'query_all'],
                           'demos_end': 'structural NA: no demonstrations',
                           'aggregates': ['mass', 'density'], 'future_capture_qualification_pending': True},
        'limitations': [
            'D03 changes information, length, positions, term repetition, and ordering together; no equal-length or reverse-order dual control.',
            'D01 and D02 preserve within-query length; contexts are not length-matched minimal pairs.',
            'Q02 contains additional sexual-history denigration; correct 有 does not establish correct 嘿嘿 sense selection.',
            'Q03 changes quotation and stance wording and length; it derives from Q02 and is not an independent sample.',
            'Two previously exposed real sources and one assistant rewrite; no independent confirmation, population accuracy or statistical significance claim.',
            'No no-dictionary baseline in these nine cells; estimate dictionary-version effects within the registered prompts.',
            'Attention is descriptive; no internal causal mechanism intervention is proposed in this draft.',
        ], 'future_execution': 'After review, separate adoption and runtime preparation with fresh numerical/format qualification and a valid GPU window; never restart old runs.'})
    write_lines(out / 'model-inputs.jsonl', inputs)
    table(out / 'input-ledger.tsv', list(ledger[0]), ledger)
    table(out / 'review-checklist.tsv', ['item_id', 'title', 'review_scope', 'status', 'human_decision', 'human_reason'], review_items)
    write_text(out / 'model-task.txt', system)
    for row in inputs:
        write_text(out / 'prompts' / (row['request_id'] + '.txt'), row['prompt_text'])
    all_prompts = ['# 嘿嘿：全部9个原生输入', '', '均在答案生成之前结束；审核标签不在模型输入中。', '']
    for row in inputs:
        all_prompts += [f'## {row["request_id"]}', '', f'{row["prompt_tokens"]} tokens；完整原生文本：', '',
                        '```text', row['prompt_text'], '```', '']
    write_text(out / 'ALL-PROMPTS.md', '\n'.join(all_prompts) + '\n')
    doc = [
        '# 嘿嘿释义与语境：9个组合审核稿', '',
        '**状态：待审核的数据草稿，尚未运行。** 3条查询×3种词典版本；Qwen3-8B，最新任务指令，仅词典L。', '',
        '新增条件D03同时给出原释义和普通义。查询Q01复用#3169；Q02复用已审核真实例#3660；Q03是基于Q02的新反驳改写。只有一条新改写，没有新增prompt内示例。', '',
        f'以下展示所有查询、三种目标释义、另外五个完整词条及完整任务。9个完整原生输入见 {link(out / "ALL-PROMPTS.md")}，也可逐格打开独立txt。', '',
        '**本轮需回复3项：Q02选用、Q03正文及“无”标签、D03双义排法。** Q01、Q02的旧人审标签和D02的L05采用记录保留，不要求重新确认旧决定。可回复“Q02通过；Q03通过／改为……；D03通过”。本文件和空白反馈模板本身不代表已经通过。', '',
        '## 查询原文与参考状态', '',
    ]
    for query in queries:
        label = query.get('reference') or f'拟为{query["proposed_reference"]}，待审核'
        doc += [f'### {query["query_id"]} · {query["name"]}', '',
                '```text', query['text'], '```', '', f'参考答案：**{label}**。{query["interpretation"]}', '']
        if query['query_id'] == 'Q02':
            doc += ['来源为已有人工确认的demo:3660，原字节不变；这次把它用作查询。Q02的“有”可以同时受到性经历贬损措辞影响，因此它是攻击识别的整体对照，不能单独证明模型理解了种族义。这个例子此前已在示例中暴露。', '']
        if query['query_id'] == 'Q03':
            doc += ['AI拟议依据：当前作者明确反对引文的贬损，只批评“这种说法”，未另外辱骂说话者。引文内的Q02完整原文未改；新增上下文是AI创作，不能称为真实原始语料或继承Q02的人审标签。', '']
    doc += ['## 三种嘿嘿释义', '', 'D01–D03是本轮词典版本编号，9格均使用仅词典L条件。三种版本均位于词条3；词形“嘿嘿”不变。下面代码块中的半角单引号和换行就是实际输入字符。', '']
    for dictionary in dictionaries:
        doc += [f'### {dictionary["dictionary_id"]} · {dictionary["name"]}', '',
                '```text', dictionary['definition'], '```', '']
    doc += ['D03在同一“释义”字段中直接分两行，原释义在前、普通义在后；没有额外添加“本句适用哪个义项”、输出标签、择义规则或优先级指令。两段单独内容原样复用，新的并列方式仍需审核。', '',
            '**长度与顺序限制：** D03更长，重复一次词形，并把普通义放在后面。D03效果同时包含这些变化；本轮没有长度匹配双义控制和逆序版本，不能据此排除位置、长度或近因效应。', '',
            '## 其余五个完整词条', '', '下面文字与词条顺序在9个输入中均不变；类别字段不展示。D03变长后，后续词条和查询的token位置相应后移。', '']
    for i, entry in enumerate(base['lexicon'], 1):
        if entry['lexicon_id'] != FOCAL:
            doc += [f'### 词条{i} · {entry["term"]}', '', '```text', entry['senses'][0]['definition'], '```', '']
    doc += ['## 9个完整组合', '',
            '| 组合 | 查询 | 嘿嘿释义 | tokens | 答案前位置（0起始） | 释义内token数（排除边界） |',
            '|---|---|---|---:|---:|---:|']
    for row in ledger:
        doc.append(f'| {link(out / "prompts" / (row["request_id"] + ".txt"), row["request_id"])} | {row["query_id"]} | {row["dictionary_id"]} | {row["prompt_tokens"]} | {row["pre_answer"]} | {row["definition_owned_tokens"]} |')
    doc += ['', 'Q01-D01和Q01-D02与上一轮对应L输入的prompt、token及角色位置完全一致；这只复用输入，不把旧分数当作将来新运行的测量。9个组合不是9个独立样本。', '',
            '完整用户消息按以下顺序组织：参考词典（6条）→参考示例（无）→待判断文本。三条查询均使用同一套背景词典，未重新检索词典，也未给新增查询加入示例。', '',
            '## 完整任务指令', '', '以下system消息连同末尾换行原样来自当前frozen-01。', '', '```text', system.rstrip('\n'), '```', '',
            '## 计划读数与解释范围', '',
            '主读数沿用m=logit(无)−logit(有)，同时保留两个候选logit、原始预测和数值未决状态。每条查询分别比较D02−D01、D03−D01、D03−D02，共9个依赖比较。Q02参考为有，参考对齐分数为−m；不能把所有正Δm都解释成改善。', '',
            '注意力作为辅助，计划保留36层／32头、注意力总量和每可见token密度，查看答案前、词典结束、查询词形及查询正文读数。无示例，所以“示例结束”记结构性NA。双义两段另有字符和token定位，但父释义与子段重叠，不能相加；跨边界token单列。', '',
            'Q01与Q02不是词义和立场正交的最小对；Q03改动了引号、立场表达和长度。Q02/Q03共享原文，所有材料是探索性案例，不作独立确认集或总体准确率估计。D03下出现无／有／无，只支持这些具体输入的行为符合参考，仍需保留分数变化和其他可能解释。', '',
            '本稿没有加入无词典基线、新模型、新词项、示例标签翻转或内部激活干预。审核后再形成独立采用记录及运行准备；旧实验均已完成，不能重启。', '',
            '## 审核清单', '',
            '| 编号 | 审核内容 | 当前状态 |', '|---|---|---|',
    ]
    doc += [f'| {item["item_id"]} | {item["review_scope"]} | 待审核 |' for item in review_items]
    doc += ['', f'机器可读材料：{link(out / "materials.json")}；原人审来源：{link(out / "source-records.json")}；空白反馈：{link(out / "feedback-template.json")}。', '',
            'CPU复核单独保存；参考标签及AI解释未拼入模型输入。GPU数值、格式、注意力采集与时间窗口仍属于后续运行准备，本稿没有启动这些操作。', '']
    write_text(out / 'REVIEW.md', '\n'.join(doc) + '\n')
    write_text(out / 'README.md', '# 嘿嘿：小规模语境实验审核草稿\n\n'
               + f'入口：{link(out / "REVIEW.md")}。3条查询、3种释义、9个完整输入。\n\n'
               + '这是待审核材料，不是已经采用的执行冻结；反馈另存新版本，不修改本稿或旧实验。\n')
    tokenizer_files = [ROOT / 'models/base/Qwen3-8B' / name for name in
                       ['config.json', 'tokenizer.json', 'tokenizer_config.json', 'vocab.json', 'merges.txt']]
    source_pins(tokenizer_files)
    sources = sorted(set(used + tokenizer_files + [Path(__file__), AUDITOR,
                     c.PREPARED / 'manifest.json', OLD / 'manifest.json',
                     c.TASK / 'manifest.json', c.TASK.parent / 'current.json']))
    c.write(out / 'source-ledger.json', {'sources': [c.info(path) for path in sources],
                                        'human_sources_read_only': True, 'weights_read': False,
                                        'torch_imported': False, 'GPU_touched': False})
    c.write(out / 'manifest.json', {'schema': 'hehe-sense-context-review-manifest/v1',
                                   'created_at_utc': datetime.now(timezone.utc).isoformat(),
                                   'status': 'review_draft_not_adopted',
                                   'artifacts': [c.info(path) for path in sorted(out.rglob('*')) if path.is_file()],
                                   'sources': [c.info(path) for path in sources],
                                   'pending_items': [item['item_id'] for item in review_items],
                                   'GPU_started': False, 'immutable_after_delivery': True})
    print(json.dumps({'directory': str(out), 'inputs': len(inputs), 'manifest': c.info(out / 'manifest.json')}, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT)
    build(parser.parse_args().output)
