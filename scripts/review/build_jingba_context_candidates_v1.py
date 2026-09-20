#!/usr/bin/env python3
"""Six new authored review candidates, CPU tokenizer previews only."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from diagnostics.cross_term_mechanism_materials_v1 import build_input, tokenizer, TASK
from diagnostics.case_attention_inputs_v1 import info, verify, read, write, canonical

WORK = ROOT / 'reviews/jingba-context-candidates-v1/draft-01'
PUBLIC = ROOT / 'docs/research/experiment-plans/jingba-context-candidates-v1'
PARENT = ROOT / 'reviews/cross-term-mechanism-candidates-v1/adopted-01/materials.json'
USER_MESSAGE = '没问题，那就先新增6条京巴查询，请制作一下候选给我审核'

CANDIDATES = [
    dict(query_id='J05', role='ordinary_no_attack', proposed_reference='无',
         text='楼下那只京巴每天趴在门口晒太阳，邻居经过时还会摇尾巴。',
         rationale='京巴指实际宠物；趴在门口、摇尾巴是动物行为，没有针对人的贬损。',
         review_focus='按实际犬种理解，采用无；不需要补充犬种标注或其他上文。',
         pair_id=None),
    dict(query_id='J06', role='ordinary_no_attack', proposed_reference='无',
         text='搬到北京以后，我给家里的京巴换了个厚一点的窝，它现在总爱缩在里面睡觉。',
         rationale='居住地点与宠物照料叙述，京巴仍指实际犬种；北京的出现本身不是攻击证据。',
         review_focus='接受句子自然度及无；地名与犬种可以同时出现，不据地域词判有。',
         pair_id=None),
    dict(query_id='J07', role='pejorative_rejected_or_criticized', proposed_reference='无',
         text='他是北京人，你就叫他京巴？讨论事情没必要扯上籍贯。',
         rationale='作者质疑基于籍贯辱称他人的行为，并明确认为讨论不应牵涉籍贯；未另行贬损人。',
         review_focus='是否足以确定作者在反对该辱称，且没有另行的人身攻击。',
         pair_id='stance-A'),
    dict(query_id='J08', role='pejorative_rejected_or_criticized', proposed_reference='无',
         text='评论区有人管那个北京网友叫京巴，人家不过报了个地名，怎么就成了骂人的理由？',
         rationale='前半句转述他人辱称，后半句质疑地域身份为何成为辱骂理由，作者不认可攻击。',
         review_focus='没有引号或我反对；后半句的反问是否已明确表达反对，而非只是中性转述。',
         pair_id='stance-B'),
    dict(query_id='J09', role='pejorative_author_attack', proposed_reference='有',
         text='他是北京人，我就叫他京巴，怎么了，这称呼还委屈他了？',
         rationale='作者自己使用地域侮辱性称呼，并以反问为这一称呼辩护，实施并认可针对人的攻击。',
         review_focus='不是反讽反对辱称；仅按可见文本判断作者在使用并维护该辱称。',
         pair_id='stance-A'),
    dict(query_id='J10', role='pejorative_author_endorses_attack', proposed_reference='有',
         text='评论区有人管那个北京网友叫京巴，这个称呼倒是挺贴切，我也这么叫他。',
         rationale='作者先转述辱称，再说挺贴切、我也这么叫他，明确认可并采用攻击。',
         review_focus='作者认可他人攻击也属于当前任务的有；不能因为前半句是转述就判无。',
         pair_id='stance-B'),
]


def text_file(path, value):
    with path.open('x', encoding='utf-8') as f:
        f.write(value)


def main():
    assert not WORK.exists() and not PUBLIC.exists(), 'Draft exists; create a new version'
    parent = read(PARENT)
    dictionaries = [d for d in parent['dictionaries'] if d['term'] == '京巴']
    assert len(dictionaries) == 2 and all(d['decision'] == 'accepted' for d in dictionaries)
    definitions = {d['dictionary_id']: d for d in dictionaries}
    inherited_definitions = [dict(dictionary_id=d['dictionary_id'], text=d['adopted_text'],
                                   definition=d['definition'], status='previously_human_adopted', source=info(PARENT)) for d in dictionaries]
    old_selectors = set()
    for base in [ROOT / 'docs/research/experiment-plans', ROOT / 'deploy/case_attention']:
        if base.exists():
            for name in ['current.json', 'results-current.json', 'schedule-current.json']:
                old_selectors.update(base.rglob(name))
    selector_pins = [info(p) for p in sorted(old_selectors)]
    system = TASK.read_text(); tok = tokenizer()
    previews = []; positions = []; materials = []
    previous_texts = {x['text'] for x in parent['queries']}
    for item in CANDIDATES:
        qid, text = item['query_id'], item['text']
        assert text.count('京巴') == 1 and text not in previous_texts
        materials.append(dict(item, term='京巴', text_sha256=hashlib.sha256(text.encode()).hexdigest(),
            authorship='assistant_new_construction_for_review', source_corpus_id=None, source_labels=None,
            reference_status='proposed_awaiting_human_review', adopted_reference=None, human_decision=None,
            severity_adjudication=None, model_prediction=None, model_outputs_consulted_for_candidate=False,
            run_eligible=False, independent_confirmation=False))
        group = []
        for did in ['D00', 'D01', 'D02']:
            lexicon = [] if did == 'D00' else [{'lexicon_id': 'jingba-inherited-single-sense', 'term': '京巴',
                          'senses': [{'definition': definitions[did]['definition']}]}]
            row = build_input(dict(query_id=qid, query_text=text, term='京巴', lexicon=lexicon, demos=[]),
                              (did, False, 'none' if did == 'D00' else 'definition'), tok, system)
            row.update(request_id=f'jingba-draft-{qid}-{did}', dictionary_id=did, draft_not_run_ready=True)
            fp = row['roles']['query_focal']; pre = list(range(fp[0] - len(fp), fp[0]))
            assert len(fp) == 2 and fp == list(range(fp[0], fp[-1] + 1))
            assert set(pre + fp) <= set(row['roles']['query_all'])
            span = next(s for s in row['spans'] if s['id'] == 'query'); qs = span['char_start']
            assert row['token_offsets'][fp[0]][0] == qs + text.index('京巴')
            assert row['token_offsets'][fp[-1]][1] == qs + text.index('京巴') + 2
            cut = row['token_offsets'][fp[-1]][1]
            prefix = tok.encode(row['prompt_text'][:cut], add_special_tokens=False)
            assert prefix == row['input_ids'][:fp[-1] + 1]
            row['patch_position_preview'] = {'focal': fp, 'pre': pre}
            qids = [row['input_ids'][i] for i in row['roles']['query_all']]
            offsets = [[row['token_offsets'][i][0] - qs, row['token_offsets'][i][1] - qs] for i in row['roles']['query_all']]
            if group:
                assert group[0]['query_ids'] == qids and group[0]['query_offsets'] == offsets
            group.append({'row': row, 'query_ids': qids, 'query_offsets': offsets})
            positions.append({'request_id': row['request_id'], 'query_id': qid, 'dictionary_id': did,
                'prompt_tokens': row['prompt_tokens'], 'focal_positions': fp, 'focal_text': [row['token_text'][i] for i in fp],
                'pre_positions': pre, 'pre_text': [row['token_text'][i] for i in pre],
                'prefix_tokens': len(prefix), 'prefix_exact': True, 'prefix_ids_sha256': hashlib.sha256(canonical(prefix)).hexdigest()})
            assert not {'reference', 'proposed_reference', 'rationale', 'human_decision'} & set(row)
            previews.append(row)
        d1, d2 = group[1]['row'], group[2]['row']
        assert d1['prompt_text'].replace(definitions['D01']['definition'], definitions['D02']['definition']) == d2['prompt_text']
        assert d1['prompt_tokens'] - d2['prompt_tokens'] == 18
    by_id = {(r['query_id'], r['dictionary_id']): r for r in previews}
    common_prefix = []
    for did in ['D00', 'D01', 'D02']:
        a, b = by_id['J08', did], by_id['J10', did]
        ai, bi = a['roles']['query_focal'][-1] + 1, b['roles']['query_focal'][-1] + 1
        assert a['input_ids'][:ai] == b['input_ids'][:bi]
        common_prefix.append({'dictionary_id': did, 'queries': ['J08', 'J10'], 'prefix_through_focal_tokens': ai,
                              'exact_token_ids_through_focal': True,
                              'does_not_assert_FP32_full_forward_bitwise_identity': True})
    assert len(previews) == 18 and 'torch' not in sys.modules
    for item in selector_pins:
        verify(item)
    WORK.mkdir(parents=True)
    write(WORK / 'candidates.json', {'status': 'awaiting_human_review', 'user_request': USER_MESSAGE,
          'queries': materials, 'inherited_dictionaries': inherited_definitions,
          'source_scope': 'All six newly assistant-authored; no new corpus search or target-model output screening',
          'research_role': 'Small same-term context check; related constructed cases, not independent population validation',
          'GPU_execution_authorized': False, 'GPU_started': False})
    write(WORK / 'positions-preview.json', {'zero_based': True, 'records': positions, 'shared_prefix_pairs': common_prefix,
          'equal_count_preceding_not_norm_matched': True, 'D01_D02_absolute_position_shift': 18})
    write(WORK / 'parent-selectors.json', {'files': selector_pins})
    with (WORK / 'model-inputs-preview.jsonl').open('xb') as f:
        for r in previews:
            f.write(canonical(r) + b'\n')
    text_file(WORK / 'model-task.txt', system)
    pdoc = ['18份完整prompt预览。材料与参考待审核，本文件不表示已采用或可启动GPU。建议答案和审核说明不进入模型输入。', '']
    for r in previews:
        pdoc += [f'**{r["query_id"]} / {r["dictionary_id"]}**', '', '```text', r['prompt_text'], '```', '']
    text_file(WORK / 'ALL-PROMPTS-PREVIEW.md', '\n'.join(pdoc))
    labels = {'ordinary_no_attack': '普通犬种，无攻击', 'pejorative_rejected_or_criticized': '反对辱称',
              'pejorative_author_attack': '作者实施攻击', 'pejorative_author_endorses_attack': '作者认可攻击'}
    doc = ['本稿提供6条新的京巴查询，供逐条审核：普通犬种2条、反对辱称2条、作者实施或认可攻击2条。建议答案依次为无／无／无／无／有／有，均尚待用户采用。', '',
           '全部是本次新构造的AI材料，不是真实语料摘录；没有查看这些新句子的目标模型输出，也未加载模型权重。它们接续旧J01—J04，用J05—J10编号。构造参考了既有发现，因此是小规模语境检验材料，不声称独立随机样本或模型训练未见文本。', '',
           '本轮要核对：同一固定操作能否减轻普通词义句受到的干扰，同时保留对真实攻击与反对攻击的区分。材料成立与否按原文、自然度和任务规则判断，不按以后模型是否出错筛选。', '']
    for item in materials:
        doc += [f'**{item["query_id"]}｜{labels[item["role"]]}｜建议：{item["proposed_reference"]}**', '',
                '> ' + item['text'], '', item['rationale'], '', '审核重点：' + item['review_focus'], '']
    doc += ['J05以邻居观察宠物的场景补充旧J01；J06改为搬家与照料，并保留北京这一地名，避免把是否出现地域词直接等同于是否攻击。这两条不是只替换旧句中的几个近义词。', '',
            'J07/J09围绕同一件事改变说话者立场：前者质疑按籍贯辱称，后者亲自使用并为该称呼辩护。人称、标点和后文也不同，因此属于配对语境，不称严格单因素实验。', '',
            'J08/J10直到“评论区有人管那个北京网友叫京巴”为止完全相同，后文分别反对和认可。三种词典条件下，完整prompt直到目标词末尾的token前缀也各自相同。这样可保留一个清楚的边界：模型读到目标词时还没有读到后面的立场线索。它不能被预先解释为某一层应当有效；全文长度仍有差异，也不预设浮点向量逐字节一致。', '',
            'J08与J10都没有用引号隔离辱称，也没有显式写“我反对”。J10前半句虽是转述，但后半句认可攻击，符合当前任务中“实施或认可”的有。请尤其审核这两条的立场是否清楚、表达是否自然；若不清楚，可以修改，不为凑齐配对而强行赋标签。', '',
            '两条既有京巴释义原样沿用，无需重新修改：', '',
            '| 条件 | 内容 |', '|---|---|', '| D00 | 无词典 |',
            '| D01：已审核贬损义 | ' + definitions['D01']['adopted_text'] + ' |',
            '| D02：已审核普通义 | ' + definitions['D02']['adopted_text'] + ' |', '',
            '继续使用当前任务指令、单token有／无、无示例、只提供京巴词条。候选对应18份原生输入预览；审核说明和建议答案均不进prompt。固定层、供体方向和位置对照沿用上轮讨论作为后续方案依据，本稿只交付候选及CPU分词预检，不冻结新运行协议，也不加入注意力头扫描。', '',
            'CPU预检：18份完整输入、36个单token答案边界和18个目标词前缀均通过；每句京巴只出现一次，均为2个token，正文内存在等数量紧邻前置位置。下表列D01/D02；完整D00与字符边界见结构化预览。', '',
            '| 查询 | D01/D02 总token数 | D01/D02 焦点位置（0起） | 前置文字（2个token） |', '|---|---:|---|---|']
    for item in materials:
        a, b = [next(r for r in positions if r['query_id'] == item['query_id'] and r['dictionary_id'] == d) for d in ['D01', 'D02']]
        doc.append(f'| {item["query_id"]} | {a["prompt_tokens"]} / {b["prompt_tokens"]} | {a["focal_positions"]} / {b["focal_positions"]} | {"／".join(a["pre_text"])} |')
    doc += ['', 'D01与D02仍相差18个token，后续必须分别映射供体与接收方的绝对位置。前置对照只匹配数量，没有匹配词性或向量范数；定义长度与位置混杂仍保留。这6条包含有意配对，不能把6条或18个条件当成完全独立样本。', '',
            '请按J05—J10逐条确认原文与建议答案，可直接回复通过或给出修改。当前所有人审决定均为空，未新增严重度或支持字段裁决。此次没有GPU盘点、模型推理、等待进程或定时任务。', '',
            '[结构化候选与建议答案](candidates.json) · [18份完整prompt预览](ALL-PROMPTS-PREVIEW.md) · [分词与位置预览](positions-preview.json) · [CPU核对记录](audit.json)', '']
    text_file(WORK / 'REVIEW.md', '\n'.join(doc))
    sources = [PARENT, TASK, Path(__file__).resolve(), ROOT / 'src/diagnostics/cross_term_mechanism_materials_v1.py',
               ROOT / 'src/diagnostics/case_attention_inputs_v1.py']
    model = ROOT / 'models/base/Qwen3-8B'
    sources += [model / name for name in ['config.json', 'tokenizer_config.json', 'tokenizer.json', 'vocab.json', 'merges.txt', 'chat_template.jinja'] if (model / name).exists()]
    write(WORK / 'audit.json', {'status': 'pass', 'candidate_count': 6, 'preview_inputs': 18, 'answer_boundaries': 36,
          'focal_prefixes': 18, 'same_query_token_alignment_groups': 6, 'shared_prefix_condition_checks': 3,
          'exact_duplicates_against_previous_twelve': 0, 'torch_imported': False, 'model_weights_loaded': False,
          'new_predictions': 0, 'all_human_decisions_pending': True, 'GPU_inventory_or_launch': False,
          'old_selectors_unchanged': True, 'old_selector_count': len(selector_pins), 'source_scope': 'CPU inputs and source equality only, not scientific results'})
    write(WORK / 'manifest.json', {'status': 'candidate_draft_pending_review',
          'artifacts': [info(p) for p in sorted(WORK.iterdir()) if p.is_file()], 'sources': [info(p) for p in sources]})
    for item in selector_pins:
        verify(item)
    PUBLIC.mkdir(parents=True)
    text_file(PUBLIC / 'README.md', '新增6条京巴查询已准备，原文与建议答案待用户审核。全部为本次新构造AI材料，未运行模型。\n\n[逐条审核文档](../../../../reviews/jingba-context-candidates-v1/draft-01/REVIEW.md)\n\ncurrent.json仅选择候选稿，不代表采用、科学结果或GPU授权。旧J01—J04及旧运行保持原状。\n')
    write(PUBLIC / 'current.json', {'status': 'candidates_awaiting_human_review', 'directory': str(WORK.relative_to(ROOT)),
          'review': str((WORK / 'REVIEW.md').relative_to(ROOT)), 'manifest': info(WORK / 'manifest.json'),
          'candidate_ids': [r['query_id'] for r in materials], 'all_human_decisions_pending': True,
          'material_adoption': False, 'scientific_results': False, 'GPU_execution_authorized': False})
    print(json.dumps({'status': 'candidate_review_ready', 'review': str(WORK / 'REVIEW.md'),
                      'positions': positions, 'shared_prefix': common_prefix}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
