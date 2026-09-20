"""Nine adopted inputs; versioned CPU preparation and immutable execution contract."""
from __future__ import annotations
from copy import deepcopy
import hashlib
from pathlib import Path
import shutil

from diagnostics.case_attention_inputs_v1 import (
    ROOT, TASK, ROLES, ROLE_LABELS, require, read, lines, canonical, digest, sha,
    info, verify, write, atomic)

WORK = ROOT / 'reviews/hehe-sense-context-v1'
PUBLIC = ROOT / 'docs/research/experiment-plans/hehe-sense-context-v1'
ADOPTED = WORK / 'adopted-01'
PREPARED = WORK / 'prepared-01'
OLD = ROOT / 'reviews/case-content-replacement-v1/prepared-01'
CONDITIONS = [('L', False, 'definition')]
CODE = [ROOT / 'src/diagnostics' / f'hehe_sense_context_{part}_v1.py' for part in ('inputs', 'runtime', 'report')]
CODE += [ROOT / 'scripts/review' / name for name in (
    'run_hehe_sense_context_v1.py', 'run_hehe_sense_context_window_v1.py',
    'audit_hehe_sense_context_execution_v1.py', 'test_hehe_sense_context_v1.py',
    'test_hehe_sense_context_viewer_v1.cjs')]
CODE += [ROOT / 'tools/hehe_sense_context_viewer_v1/viewer.html']


def prepare(output=PREPARED):
    from diagnostics.cross_model_applicability_execution_v1 import runtime_versions, framework_sources
    out = Path(output).absolute()
    require(not out.exists(), 'Use a new preparation directory')
    adoption_manifest = read(ADOPTED / 'manifest.json')
    for item in adoption_manifest['artifacts'] + adoption_manifest['sources']:
        verify(item)
    adopted = read(ADOPTED / 'adoption.json')
    require(adopted['status'] == 'all_3_accepted' and adopted['pending_items'] == [], 'Materials not adopted')
    out.mkdir(parents=True)
    names = ['model-inputs.jsonl', 'model-task.txt', 'ALL-PROMPTS.md', 'input-ledger.tsv',
             'definition-components.json', 'comparisons.json', 'historical-input-matches.json',
             'source-records.json', 'adoption.json', 'materials.json', 'analysis-references.json',
             'design.json', 'ADOPTION.md', 'input-preservation.json']
    for name in names:
        (out / name).write_bytes((ADOPTED / name).read_bytes())
    shutil.copytree(ADOPTED / 'prompts', out / 'prompts')
    for name in ('inputs.jsonl', 'scoring-inputs.jsonl'):
        (out / name).write_bytes((ADOPTED / 'model-inputs.jsonl').read_bytes())
    profile = read(OLD / 'model-profile.json')
    for item in profile['metadata_sources']:
        verify(item)
    for item in profile['weight_sources']:
        stat = Path(item['path']).stat()
        require((stat.st_size, stat.st_mtime_ns) == (item['bytes'], item['mtime_ns']), 'Weight metadata changed')
    write(out / 'model-profile.json', profile)
    requests = lines(out / 'scoring-inputs.jsonl')
    n = len(requests)
    prefixes = sum(len(row['prefix_proofs']) for row in requests)
    plan = deepcopy(read(OLD / 'execution-plan.json'))
    for key in ('max_GPU_window_minutes', 'absolute_GPU_deadline', 'human_choice'):
        plan.pop(key, None)
    plan.update(schema_version='hehe-sense-context-execution/v1',
        status='adopted_cpu_preparation_gpu_pending', runtime_versions=runtime_versions(),
        conditions=['L'], dictionary_versions=['D01', 'D02', 'D03'], query_ids=['Q01', 'Q02', 'Q03'],
        input_count=n, comparison_count=9, all_materials_adopted=True,
        accepted_review_items=['Q02', 'Q03', 'D03'],
        scope='Two exposed real source texts and one adopted AI rejection; exploratory and dependent',
        current_GPU_window=None, allocation=None, GPU_qualified=False,
        historical_input_replay_claimed=True, historical_scores_used=False,
        references_are_newly_adjudicated=False,
        reference_provenance='Q01/Q02 inherited; Q03 named bulk adoption of AI draft',
        budget={'unique_prompts': n, 'engineering_full_forwards': 6*n, 'prefix_forwards': prefixes,
                'production_forwards': n, 'format_extra_forward_max': 7*n,
                'usual_if_label_then_eos': 8*n+prefixes, 'total_forward_max': 14*n+prefixes})
    write(out / 'execution-plan.json', plan)
    full_bytes = sum(profile['layers']*profile['heads']*len(ROLES)*row['prompt_tokens']*8 for row in requests)
    prefix_bytes = sum(profile['layers']*profile['heads']*len(ROLES)*(row['roles'][p['role']][0]+1)*8
                       for row in requests for p in row['prefix_proofs'])
    write(out / 'resource-estimate.json', {'estimated_GPU_and_checks_minutes': [5, 10],
          'basis': 'Previous same-model L20 case run completed 110 forwards in about 187s including load/release; this plan usually has81 forwards. CPU report and fresh qualification add variability.',
          'guaranteed_completion_time': False, 'recommended_reserved_minutes': 15,
          'full_attention_bytes_per_pass': full_bytes, 'retained_raw_attention_bytes': 6*full_bytes+prefix_bytes,
          'recommended_free_disk_bytes': 8_000_000_000, 'future_single_device_min_mib': 44000,
          'GPU_inventory_checked_during_preparation': False})
    sources = [Path(item['path']) for item in adoption_manifest['artifacts'] + adoption_manifest['sources']]
    sources += CODE + framework_sources() + [ADOPTED / 'manifest.json', OLD / 'manifest.json',
        OLD / 'model-profile.json', OLD / 'execution-plan.json',
        ROOT / 'src/diagnostics/case_attention_capture_v1.py',
        ROOT / 'src/diagnostics/cross_model_applicability_models_v1.py',
        ROOT / 'src/diagnostics/cross_model_applicability_execution_v1.py',
        ROOT / 'src/diagnostics/case_content_replacement_runtime_v1.py',
        ROOT / 'scripts/review/run_case_content_replacement_window_v1.py']
    write(out / 'source-ledger.json', {'files': [info(p) for p in sorted(set(sources))],
          'historical_weight_inventory_reused': True, 'new_weight_content_read': False,
          'weight_stat_checks': len(profile['weight_sources']), 'GPU_allocation': None})
    (out / 'README.md').write_text(
        '# 嘿嘿释义与语境：已采用，CPU运行准备\n\n'
        'Q02、Q03、D03均通过。Q03正式采用无／0级，保留AI改写及明确批量采用来源；Q01/Q02沿用原人审。\n\n'
        '9个科学输入与审核稿逐字节一致；9个配对比较，参考标签仅在GPU释放后的分析中读取。'
        '双义两段可分别查看注意力；父段与子段重叠，不能相加。\n\n'
        '通常81次前向：54次完整输入工程验证、9次前缀、9次格式续写、9次正式采集；格式最坏135次。'
        '一个空闲L20上估计5–10分钟，建议预留15分钟和8GB磁盘；实际完成时间不保证。\n\n'
        '本准备没有绑定GPU或启动执行。未来时间窗口单独记录；守护在截止前120秒请求STOP，'
        '60秒对核验过的本任务进程TERM，20秒仍存活才KILL。旧实验均保持终态。\n', encoding='utf-8')
    from diagnostics.hehe_sense_context_report_v1 import preview
    preview(out)
    return {'directory': str(out), 'inputs': n, 'prefix_records': prefixes, 'budget': plan['budget'], 'GPU_started': False}


def validate(prepared, sealed=True, weights=False):
    from diagnostics.cross_model_applicability_execution_v1 import runtime_versions
    out = Path(prepared)
    if sealed:
        manifest = read(out / 'manifest.json')
        require(manifest['schema_version'] == 'hehe-sense-context-prepared/v1'
                and manifest['accepted_review_items'] == ['Q02', 'Q03', 'D03'], 'Wrong prepared experiment')
        for item in manifest['artifacts'] + manifest['sources']:
            verify(item)
    for item in read(out / 'source-ledger.json')['files']:
        verify(item)
    plan, profile = read(out / 'execution-plan.json'), read(out / 'model-profile.json')
    require(plan['runtime_versions'] == runtime_versions() and plan['all_materials_adopted'], 'Runtime/review mismatch')
    require(plan['accepted_review_items'] == ['Q02', 'Q03', 'D03'], 'Adoption scope changed')
    require(plan['acceptance'] == read(OLD / 'execution-plan.json')['acceptance'], 'Qualification gates changed')
    for item in profile['metadata_sources']:
        verify(item)
    for item in profile['weight_sources']:
        if weights:
            verify(item)
        else:
            stat = Path(item['path']).stat()
            require((stat.st_size, stat.st_mtime_ns) == (item['bytes'], item['mtime_ns']), 'Weight metadata changed')
    requests = lines(out / 'scoring-inputs.jsonl')
    require(len(requests) == len({row['request_id'] for row in requests}) == 9, 'Input inventory changed')
    require({(r['query_id'], r['dictionary_id']) for r in requests}
            == {(q,d) for q in ('Q01','Q02','Q03') for d in ('D01','D02','D03')}, 'Incomplete input cross')
    require(sum(len(r['prefix_proofs']) for r in requests) == 9, 'Prefix inventory changed')
    require((out / 'scoring-inputs.jsonl').read_bytes() == (ADOPTED / 'model-inputs.jsonl').read_bytes(), 'Approved inputs changed')
    for row in requests:
        require(row['condition'] == 'L' and not row['with_demos'] and row['roles']['demos_end'] == [], 'Condition changed')
        require(len(row['input_ids']) == row['prompt_tokens'] and digest(row['input_ids']) == row['input_ids_sha256'], 'Token binding')
        require(hashlib.sha256(row['prompt_text'].encode()).hexdigest() == row['prompt_sha256'], 'Prompt binding')
        require(row['roles']['pre_answer'] == [row['prompt_tokens']-1], 'Answer boundary')
        require(not {'reference','gold','human_decision','review_status','eligible_for_GPU'} & set(row), 'Reference in scorer input')
    return plan, profile, requests


def seal(prepared):
    out = Path(prepared).absolute()
    require(not (out / 'manifest.json').exists(), 'Preparation already sealed')
    validate(out, sealed=False)
    for name in ('cpu-tests.json', 'cpu-audit.json', 'watchdog-cpu-test.json', 'viewer-cpu-test.json'):
        receipt = read(out / name)
        require(receipt['status'] in ('pass','passed') and not receipt.get('CUDA_initialized',False), f'Failed CPU check: {name}')
        for item in receipt.get('implementation_snapshot', []):
            verify(item)
    write(out / 'manifest.json', {'schema_version': 'hehe-sense-context-prepared/v1',
          'status': 'cpu_complete_gpu_pending', 'accepted_review_items': ['Q02','Q03','D03'],
          'artifacts': [info(p) for p in sorted(out.rglob('*')) if p.is_file()],
          'sources': read(out / 'source-ledger.json')['files'],
          'GPU_allocation': None, 'GPU_qualified': False, 'immutable_after_seal': True})
    write(WORK / 'selector-before-preparation-01.json', read(PUBLIC / 'current.json'))
    atomic(PUBLIC / 'current.json', {'schema': 'hehe-sense-context-selector/v1',
           'status': 'adopted_prepared_gpu_pending', 'directory': str(out.relative_to(ROOT)),
           'manifest': info(out / 'manifest.json'), 'adopted': str(ADOPTED.relative_to(ROOT)),
           'accepted_item_ids': ['Q02','Q03','D03'], 'pending_item_ids': [],
           'scientific_input_count': 9, 'CPU_execution_ready': True, 'GPU_started': False,
           'current_GPU_window': None}, replace=True)
    return info(out / 'manifest.json')
