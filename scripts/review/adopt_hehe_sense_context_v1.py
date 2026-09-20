#!/usr/bin/env python3
"""Record the user's explicit acceptance without rewriting the delivered draft."""
from __future__ import annotations
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from diagnostics import case_attention_inputs_v1 as c

WORK = ROOT / 'reviews/hehe-sense-context-v1'
PUBLIC = ROOT / 'docs/research/experiment-plans/hehe-sense-context-v1'
USER_MESSAGE = 'Q02、Q03、D03通过'
ACCEPTED = ['Q02', 'Q03', 'D03']


def adopt(output):
    draft, out = WORK / 'draft-01', Path(output).absolute()
    c.require(not out.exists() and not (WORK / 'feedback-01.json').exists(), 'Adoption already recorded')
    manifest = c.read(draft / 'manifest.json')
    for item in manifest['artifacts'] + manifest['sources']:
        c.verify(item)
    old_selector = (PUBLIC / 'current.json').read_bytes()
    c.require(c.read(PUBLIC / 'current.json')['manifest'] == c.info(draft / 'manifest.json'), 'Unexpected reviewed draft')
    material = c.read(draft / 'materials.json')
    items = {item['item_id']: item for item in material['review_items']}
    c.require(list(items) == ACCEPTED, 'Accepted scope differs from displayed review queue')
    feedback = {'schema': 'hehe-sense-context-feedback/v1', 'user_message': USER_MESSAGE,
                'recorded_at_utc': datetime.now(timezone.utc).isoformat(),
                'accepted_item_ids': ACCEPTED, 'review_kind': 'explicit_bulk_acceptance_of_named_items',
                'reviewed_manifest': c.info(draft / 'manifest.json'),
                'individual_interview_invented': False, 'original_AI_authorship_retained': True,
                'current_GPU_window': None}
    c.write(WORK / 'feedback-01.json', feedback)
    out.mkdir(parents=True)
    (out / 'selector-before-adoption.json').write_bytes(old_selector)
    copied = ['model-inputs.jsonl', 'model-task.txt', 'ALL-PROMPTS.md', 'input-ledger.tsv',
              'definition-components.json', 'comparisons.json', 'historical-input-matches.json', 'source-records.json']
    for name in copied:
        (out / name).write_bytes((draft / name).read_bytes())
    shutil.copytree(draft / 'prompts', out / 'prompts')
    adopted = []
    for iid in ACCEPTED:
        adopted.append({'item_id': iid, 'decision': 'accept', 'proposal_sha256': c.digest(items[iid]),
                        'review_scope': items[iid]['review_scope'], 'feedback': c.info(WORK / 'feedback-01.json'),
                        'review_kind': feedback['review_kind'], 'AI_authorship_retained': True,
                        'new_individual_interview': False})
    adoption = {'status': 'all_3_accepted', 'items': adopted, 'pending_items': [],
                'old_human_records_overwritten': False, 'scientific_prompt_changes': 0,
                'Q03_reference': '无', 'Q03_severity': 0,
                'Q03_authorship': 'assistant_derived_from_real_3660',
                'Q03_reference_provenance': 'explicit_bulk_adoption_of_displayed_AI_proposal',
                'Q02_scope': 'Selection as query; original reviewed 有/2 inherited, not re-adjudicated',
                'D03_scope': 'Exact two-paragraph composition and order, not universal sense applicability'}
    c.write(out / 'adoption.json', adoption)
    current = deepcopy(material)
    current.update(status='adopted', draft_manifest=c.info(draft / 'manifest.json'))
    current['review_items'] = adopted
    for query in current['queries']:
        if query['query_id'] == 'Q02':
            query['new_role_selection_status'] = 'accepted'
        elif query['query_id'] == 'Q03':
            query.update(reference='无', reference_status='human_adopted_AI_proposal',
                         adopted_severity=0, adoption=c.info(out / 'adoption.json'))
        if query['query_id'] in ('Q02', 'Q03'):
            query['new_human_decision'] = 'accept'
    current['dictionaries'][2]['new_human_decision'] = 'accept'
    c.write(out / 'materials.json', current)
    refs = []
    for query in current['queries']:
        refs.append({'query_id': query['query_id'], 'reference': query['reference'],
                     'reference_status': query['reference_status'], 'text_sha256': hashlib.sha256(query['text'].encode()).hexdigest(),
                     'reference_record_id': query['reference_record_id'],
                     'source': c.info(draft / 'source-records.json') if query['query_id'] != 'Q03' else c.info(out / 'adoption.json'),
                     'new_individual_adjudication': False, 'authorship': query['kind']})
    c.write(out / 'analysis-references.json', {'scorer_must_not_parse': True, 'references': refs})
    design = c.read(draft / 'design.json')
    design['status'] = 'adopted_materials_runtime_preparation_pending'
    design['reference_aligned_margin']['Q03'] = 'm'
    c.write(out / 'design.json', design)
    c.write(out / 'input-preservation.json', {'draft_manifest': c.info(draft / 'manifest.json'),
            'model_input_records': 9, 'all_model_input_bytes_identical': True,
            'files': [{'name': name, 'source': c.info(draft / name), 'adopted': c.info(out / name)} for name in copied],
            'query_references_never_inserted_into_prompts': True})
    (out / 'ADOPTION.md').write_text(
        '# 嘿嘿释义与语境：材料已通过\n\n用户原话：“Q02、Q03、D03通过”。\n\n'
        '- Q02：接受将真实#3660用作查询；沿用其已审核有／2级。\n'
        '- Q03：接受原稿中的完整反驳文本及无／0级；保留AI改写及批量采用来源。\n'
        '- D03：接受同一词条中原释义在前、普通义在后，以一个换行分隔。\n\n'
        '9份prompt、token和组件定位原样保留；原草稿和空白模板仍表示构建时状态，当前采用状态以adoption.json为准。\n\n'
        f'完整原审核稿：[REVIEW.md]({draft}/REVIEW.md)。\n\n'
        '本记录完成材料采用；GPU运行准备、数值资格和时间窗口单独记录。\n', encoding='utf-8')
    sources = [Path(__file__), draft / 'manifest.json', WORK / 'feedback-01.json']
    sources += [Path(item['path']) for item in manifest['artifacts'] + manifest['sources']]
    c.write(out / 'manifest.json', {'schema': 'hehe-sense-context-adoption/v1', 'status': 'adopted',
            'accepted_items': ACCEPTED, 'artifacts': [c.info(p) for p in sorted(out.rglob('*')) if p.is_file()],
            'sources': [c.info(p) for p in sorted(set(sources))], 'GPU_started': False, 'immutable_after_seal': True})
    c.atomic(PUBLIC / 'current.json', {'schema': 'hehe-sense-context-selector/v1', 'status': 'materials_adopted',
             'directory': str(out.relative_to(ROOT)), 'manifest': c.info(out / 'manifest.json'),
             'draft': str(draft.relative_to(ROOT)), 'adoption': str((out / 'adoption.json').relative_to(ROOT)),
             'accepted_item_ids': ACCEPTED, 'pending_item_ids': [], 'scientific_input_count': 9,
             'execution_ready': False, 'GPU_started': False}, replace=True)
    return c.info(out / 'manifest.json')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=WORK / 'adopted-01')
    print(adopt(parser.parse_args().output))
