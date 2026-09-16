#!/usr/bin/env python3
"""Build an AI-proposed case-use table from frozen reviews and evaluated behavior."""
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from diagnostics.general_model_evidence_evaluation import (
    EXPERIMENT, CORE, CONDITIONS, TASKS, Sources, canonical, csv_bytes, file_sha,
    json_bytes, jsonl, local_path, read_json, read_lines, require, sha, unique,
    verify_output, write_output,
)

BASE = ROOT / EXPERIMENT
PLAN = ROOT / 'docs/research/experiment-plans/general-model-evidence-applicability-audit-controls-v1.md'
PREPARATION_ORDER = ('3169', '5086', '541', '6037', '1128', '4026')
CORRECT = ('C0', 'S', 'D', 'SD')
NOTES = {
    ('3169', 'hate'): '优先核查 lex-0419 共享义项与示例 3660 的定向对照；3660 的规则关联为部分，不能仅凭话题为无放入严格 U。',
    ('3169', 'group'): '比较 C0/D 与 S/SD 的目标类别变化；保留查询对 lex-0419 的共享来源。',
    ('5086', 'hate'): '保留单独正确、组合错误的重点现象；严格 U 筛查为空时，另定义定向对照，不放宽关联条件凑组。',
    ('5086', 'group'): '在当前审核参考下核查含示例条件多出的 others；示例答案差异与词典来源分别研究。',
    ('541', 'hate'): '四个 NoCat 核心条件均正确；保留立场对照与单独的类别字段对比，不能用核心位型概括 G 的作用。',
    ('541', 'group'): '保留含示例条件补全类别的正向候选；核查示例 826 等的原答案差异，分开定义规则关联和答案修订。',
    ('6037', 'hate'): '四个核心条件均错误，保留连续分数与失败对照用途，不按 group 的组合成功解释 hate。',
    ('6037', 'group'): '保留组合才正确的候选；核查示例 5230 的国女命中边，其 group 答案差异须单独登记。',
    ('1128', 'hate'): '稳定正确对照候选；仍需匹配实际干预的资源、长度和操作类型。',
    ('1128', 'group'): '稳定空集合对照候选；仍需匹配实际干预，不能将本例视为无信息的中性输入。',
    ('4026', 'hate'): '审核参考下转为四条件正确，列为参考敏感与稳定正确对照候选；保持原案例历史。',
    ('4026', 'group'): '稳定正确对照候选；与重点案例匹配操作后再确定用途。',
    ('61', 'group'): '审核参考为 Sexism，四条件均错误；保留持续目标类别错误诊断，原稳定正确用途失效。',
    ('2297', 'group'): '审核后组合也不正确；保留目标类别与义项诊断，撤下组合成功的描述。',
    ('1900', 'hate'): '原仅组合正确变为仅组合错误；重新确定效应方向，保留参考敏感案例属性。',
    ('7285', 'hate'): '原仅组合正确变为仅组合错误；重新确定效应方向，保留参考敏感案例属性。',
}


def accepted_field(row, field):
    """Do not promote adapter/supporting fields outside an explicit finalization."""
    review = row['review']
    final = review.get('final_annotation')
    values = final['values'] if final else review['values']
    present = field in values and review['status'] == 'confirmed' and not row['stale']
    return {
        'value': values.get(field) if present else None, 'available': present,
        'adoption': review.get('adoption', 'native_confirmation'),
        'pre_acceptance_provenance': final.get('pre_acceptance_field_provenance', {}).get(field) if final else None,
        'record_id': row['record_id'], 'record_version': review['version'],
    }


def screen_demo(demo, relation, task):
    label = accepted_field(demo, task)
    topic, rule = (accepted_field(relation, f'{field}_{task}') for field in ('topic', 'rule'))
    native = (demo['native_task_eligibility'] or {}).get(task, {})
    resolved = label['available'] and label['value'] is not None and native.get('reference_eligible') is True
    original = demo['source']['original_answer'][task]
    same = original == label['value'] if resolved else None
    # This is a descriptive candidate screen, never a finalized U or causal claim.
    reasons = []
    if not resolved:
        reasons.append('demo_task_reference_unavailable')
    elif not same:
        reasons.append('original_answer_differs_from_reviewed_reference')
    if not topic['available'] or not rule['available']:
        reasons.append('relation_field_not_adopted_or_unavailable')
    elif topic['value'] != 'none' or rule['value'] != 'none':
        reasons.append('topic_and_rule_are_not_both_none')
    return {'demo_id': demo['source']['demo_id'], 'original_answer': original,
            'reviewed_answer': label, 'topic': topic, 'rule': rule,
            'original_answer_matches_reviewed': same,
            'strict_u_candidate': not reasons, 'screen_reasons': reasons,
            'relation_record_id': relation['record_id'],
            'introduced_entry_ids': relation['source']['introduced_entry_ids'],
            'native_demo_eligibility': native}


def behavior(mask):
    correct = [c for c, bit in zip(CORRECT, mask) if bit == '1']
    if not correct:
        return '四个核心条件均错误'
    if len(correct) == 4:
        return '四个核心条件均正确'
    return '仅 ' + '、'.join(correct) + ' 正确'


def roles(mask, changed):
    result = []
    if mask == '1111':
        result.append('稳定正确对照候选')
    if mask == '0000':
        result.append('持续错误诊断候选')
    if mask == '0001':
        result.append('组合才正确的对照候选')
    if mask[1:3] == '11' and mask[3] == '0':
        result.append('单独正确、组合错误的对照候选')
    if mask[2] != mask[3]:
        result.append('词典增量对照候选')
    if mask[1] != mask[3]:
        result.append('示例增量对照候选')
    if not result:
        result.append('单资源差异诊断候选')
    if changed:
        result.append('参考切换敏感案例')
    return result


def build():
    src = Sources(ROOT)
    def take(path, expected=None):
        return src.verify(path, expected or file_sha(path))
    ep = read_json(take(BASE / 'dual-reference-v1/current.json'))
    result = local_path(ROOT, str(EXPERIMENT / ep['result_path']))
    em = read_json(take(result / 'manifest.json', ep['manifest_sha256']))
    verify_output(result)
    for name, h in em['artifacts'].items():
        take(result / name, h)
    rp = read_json(take(BASE / ep['reference_pointer_path'], ep['reference_pointer_sha256']))
    freeze = local_path(ROOT, str(EXPERIMENT / rp['freeze_path']))
    fm = read_json(take(freeze / 'manifest.json', rp['manifest_sha256']))
    require(ep['reference_manifest_sha256'] == rp['manifest_sha256'] == em['reference_manifest_sha256'], 'mixed reference versions')
    for name, h in fm['artifacts'].items():
        take(freeze / name, h)
    bundle = read_json(take(BASE / 'bundle/evidence_bundle.json', fm['bundle_sha256']))
    take(PLAN)
    take(Path(__file__).resolve())
    take(ROOT / 'src/diagnostics/general_model_evidence_evaluation.py', em['code_sha256']['src/diagnostics/general_model_evidence_evaluation.py'])
    take(ROOT / 'src/diagnostics/general_model_numeric_analysis.py', em['code_sha256']['src/diagnostics/general_model_numeric_analysis.py'])
    masks = unique(read_lines(result / 'case_masks.jsonl'), lambda r: (r['query_id'], r['task']))
    blocks = unique(read_lines(result / 'blocks.jsonl'), lambda r: (r['query_id'], r['task'], r['condition']))
    eligibility = unique(read_lines(freeze / 'eligibility.jsonl'), lambda r: (r['query_id'], r['task']))
    materials = unique(read_lines(freeze / 'material_reviews.jsonl'), lambda r: r['record_id'])
    interactions = unique(read_lines(result / 'margin_interactions.jsonl'), lambda r: (r['query_id'], r['task'], r['reference']))
    order = read_json(result / 'summary.json')['coverage'][0]['eligible_query_ids']
    require(len(order) == 32 and set(order) == set(bundle['cases']), 'discovery frame mismatch')
    require(set(masks) == set(eligibility) == {(q,t) for q in order for t in TASKS}, 'task frame mismatch')
    rows, screening = [], []
    for q in order:
        ids = bundle['cases'][q]['object_ids']
        require(len(set(ids)) == len(ids) and all(oid in materials for oid in ids), 'missing or repeated material')
        linked = [materials[oid] for oid in ids]
        require(all(r['source_version'] == bundle['objects'][r['record_id']]['version'] for r in linked), 'material source drift')
        require(all(r['review']['status'] == 'confirmed' and not r['stale'] for r in linked), 'material confirmation is stale')
        relations = [r for r in linked if r['kind'] == 'relation']
        require(len(relations) == 10, 'expected ten frozen relations')
        for task in TASKS:
            m, e = masks[q, task], eligibility[q, task]
            bs = [blocks[q, task, c] for c in CONDITIONS]
            ds = [screen_demo(materials['demo:' + r['source']['demo_id']], r, task) for r in relations]
            u = [d['demo_id'] for d in ds if d['strict_u_candidate']]
            diff = [d['demo_id'] for d in ds if d['original_answer_matches_reviewed'] is False]
            pending = ['specific_material_operation_not_selected', 'hypothesis_and_alternative_not_frozen',
                       'retained_inputs_and_shared_sources_not_checked_for_operation',
                       'new_prompt_replay_and_position_controls_not_prepared']
            if not u:
                pending.append('no_demo_passes_strict_U_screen; define_another_intervention_if_needed')
            if diff:
                pending.append('keep_demo_answer_differences_separate_from_applicability_contrast')
            if task == 'group':
                pending.append('fixed_group_foil_not_selected')
            proposed = roles(m['reviewed_mask'], m['mask_changed'])
            require(e['input_control_eligible'] is False and e['explanation_choice'] == 'defer', 'new human use state requires a new table contract')
            row = {
                'query_id': q, 'task': task, 'text_sha256': bs[0]['text_sha256'],
                'original_bucket': m['original_bucket'], 'original_label': m['original_label'],
                'reviewed_label': m['reviewed_label'], 'query_material_label': m['query_material_label'],
                'original_mask': m['original_mask'], 'reviewed_mask': m['reviewed_mask'],
                'mask_order': list(CORE), 'observed_behavior': behavior(m['reviewed_mask']),
                'behavior_scope': 'four_core_conditions_only', 'mask_changed': m['mask_changed'],
                'six_condition_readouts': {b['condition']: {k:b[k] for k in ('prediction','reviewed_correct','revised_margin','top_gap','score_mode_sensitive','within_two_epsilon')} for b in bs},
                'reviewed_margin_interaction': interactions[q, task, 'reviewed'],
                'reference_analysis_eligible': e['reference_analysis_eligible'],
                'native_case_policy': e['native_case_policy'], 'query_material_policy': e['query_material_policy'],
                'user_selected_use': e['user_selected_use'], 'explanation_choice': e['explanation_choice'],
                'proposed_roles': proposed, 'proposal_kind': 'ai_proposal', 'proposal_human_adopted': False,
                'preparation_rank_proposed': PREPARATION_ORDER.index(q) + 1 if q in PREPARATION_ORDER else None,
                'proposal_note': NOTES.get((q,task), '先按审核材料确定具体操作，再决定是否列入运行队列。'),
                'material_record_ids': ids, 'material_counts': dict(Counter(r['kind'] for r in linked)),
                'strict_u_candidate_demo_ids': u, 'demo_answer_difference_ids': diff,
                'input_control_eligible': False, 'mechanism_ready': False, 'pending_work': pending,
            }
            screening.append({'query_id':q, 'task':task, 'demos':ds,
                              'definitions': [{'record_id':r['record_id'], 'verdict':accepted_field(r,'definition_verdict'),
                                               'adopted_definition':r['review'].get('final_annotation',{}).get('adopted_definition')} for r in linked if r['kind']=='definition'],
                              'hits': [{'record_id':r['record_id'], 'source_fit':accepted_field(r,'source_fit'),
                                       'query_fit':accepted_field(r,'query_fit'), 'source_kind':r['source']['source_kind'],
                                       'source_id':r['source']['source_id'], 'entry_id':r['source']['entry_id'],
                                       'raw_span':r['source']['raw_span']} for r in linked if r['kind']=='hit']})
            rows.append(row)
    audit = {'schema_version':'evidence-case-uses-audit/v1', 'status':'passed', 'queries':32, 'task_rows':len(rows),
             'linked_unique_materials':len({oid for row in rows for oid in row['material_record_ids']}),
             'proposed_preparation_order':list(PREPARATION_ORDER), 'preparation_order_is_run_selection':False,
             'tasks_without_strict_u_candidates':sum(not r['strict_u_candidate_demo_ids'] for r in rows),
             'source_reference_manifest_sha256':rp['manifest_sha256'], 'source_evaluation_manifest_sha256':ep['manifest_sha256'],
             'human_decisions_created':0,'input_controls_enabled':0,'model_forward_executed':False,'production_writes':0}
    flat = [{k:r[k] for k in ('query_id','task','original_bucket','original_label','reviewed_label','original_mask','reviewed_mask',
                             'observed_behavior','proposed_roles','preparation_rank_proposed','proposal_note',
                             'strict_u_candidate_demo_ids','demo_answer_difference_ids','reference_analysis_eligible',
                             'proposal_human_adopted','input_control_eligible','pending_work')} for r in rows]
    doc = ['# 案例用途表 v1：分析事实与用途建议', '',
           '覆盖原 32 条 discovery 的 64 个任务。标签与数值为已冻结事实；用途、准备顺序为本次 AI 建议，尚未登记为人工采纳。', '',
           '位型次序为 C0、S、D、SD；S 是去显式类别字段的词典。此处“稳定”仅描述四个核心条件；六条件读数另存 JSONL。', '',
           'U 候选要求示例任务参考可用、原答案与审核一致，且已采纳的话题与规则关联均为 none；候选列表不是已冻结 U。答案差异是参考比较，不是源标注错误裁决。', '',
           '建议准备顺序：3169 → 5086 → 541 → 6037 → 1128 → 4026。先核查具体操作与对照匹配；最终运行名单另定，全部 32 条始终保留。', '',
           '| 案例/任务 | 审核参考 | 原→新位型 | 用途建议 | 严格 U 候选数 | 答案差异数 |',
           '| --- | --- | --- | --- | ---: | ---: |']
    for r in rows:
        doc.append(f"| {r['query_id']}/{r['task']} | {canonical(r['reviewed_label'])} | `{r['original_mask']} → {r['reviewed_mask']}` | {'；'.join(r['proposed_roles'])} | {len(r['strict_u_candidate_demo_ids'])} | {len(r['demo_answer_difference_ids'])} |")
    doc += ['', '## 优先核查说明', '']
    for q in PREPARATION_ORDER:
        for r in rows:
            if r['query_id']==q:
                doc.append(f"- **{q}/{r['task']}**：{r['proposal_note']} 严格 U 候选：{canonical(r['strict_u_candidate_demo_ids'])}；答案差异示例：{canonical(r['demo_answer_difference_ids'])}。")
    doc += ['', '输入实验与机制就绪状态均保持 false；已完成参考分析不代表具体干预或内部解释已成立。每条具体待办见 case_uses.csv / case_uses.jsonl，资源字段与采纳来源见 resource_screening.jsonl。', '']
    files = {'case_uses.jsonl':jsonl(rows), 'case_uses.csv':csv_bytes(flat),
             'resource_screening.jsonl':jsonl(screening), 'CASE_USES.md':'\n'.join(doc).encode(), 'audit.json':json_bytes(audit),
             'execution_source.py':Path(__file__).read_bytes()}
    src.unchanged()
    return files, src.hashes, audit


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--check', action='store_true')
    args=parser.parse_args()
    target=args.output.resolve()
    require(target.is_relative_to(BASE/'reviews') and target!=BASE/'reviews', 'output must be in private reviews')
    files, sources, audit=build()
    if args.check:
        manifest=read_json(target/'manifest.json')
        require(manifest['schema_version']=='evidence-case-uses-proposal/v1' and manifest['status']=='proposal', 'not a case-use proposal')
        require(set(files)==set(manifest['artifacts']) and {p.name for p in target.iterdir()}=={*files,'manifest.json'}, 'output inventory mismatch')
        require(manifest['source_files']==sources, 'proposal source versions changed')
        for name, raw in files.items():
            require((target/name).read_bytes()==raw and sha(raw)==manifest['artifacts'][name], 'case-use reproduction mismatch: '+name)
        print('Case-use proposal reproduces byte for byte; human records unchanged.')
    else:
        files['manifest.json']=json_bytes({'schema_version':'evidence-case-uses-proposal/v1','status':'proposal',
                                         'created_at':datetime.now(timezone.utc).isoformat(),'source_files':sources,
                                         'source_reference_manifest_sha256':audit['source_reference_manifest_sha256'],
                                         'source_evaluation_manifest_sha256':audit['source_evaluation_manifest_sha256'],
                                         'artifacts':{name:sha(raw) for name,raw in files.items()}})
        write_output(target,files)
        print(target)


if __name__=='__main__':
    try:
        main()
    except (ValueError, OSError, KeyError, TypeError) as exc:
        print('Case-use update failed: '+str(exc),file=sys.stderr)
        raise SystemExit(1)
