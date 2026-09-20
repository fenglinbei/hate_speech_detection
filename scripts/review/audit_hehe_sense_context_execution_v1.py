#!/usr/bin/env python3
"""Read-only independent check of adoption, unchanged prompts and new runtime binding."""
from __future__ import annotations
import argparse
import ast
from fractions import Fraction
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT / 'reviews/hehe-sense-context-v1'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def verify(record):
    data = Path(record['path']).read_bytes()
    assert len(data) == record['bytes'] and hashlib.sha256(data).hexdigest() == record['sha256'], record['path']


def module(path):
    spec = importlib.util.spec_from_file_location('hsc_independent_original_draft_audit', path)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def audit(prepared):
    prepared = Path(prepared).absolute()
    original = module(ROOT / 'scripts/review/audit_hehe_sense_context_v1.py').audit(WORK / 'draft-01')
    for directory in (WORK / 'adopted-01', prepared):
        if (directory / 'manifest.json').exists():
            manifest = read(directory / 'manifest.json')
            for item in manifest['artifacts'] + manifest['sources']:
                verify(item)
    for item in read(prepared / 'source-ledger.json')['files']:
        verify(item)
    feedback = read(WORK / 'feedback-01.json')
    assert feedback['user_message'] == 'Q02、Q03、D03通过'
    assert feedback['accepted_item_ids'] == ['Q02', 'Q03', 'D03']
    adoption = read(prepared / 'adoption.json')
    assert adoption['pending_items'] == [] and adoption['status'] == 'all_3_accepted'
    assert all(x['decision'] == 'accept' and x['AI_authorship_retained'] for x in adoption['items'])
    draft_bytes = (WORK / 'draft-01/model-inputs.jsonl').read_bytes()
    for directory in (WORK / 'adopted-01', prepared):
        assert (directory / 'model-inputs.jsonl').read_bytes() == draft_bytes
    assert (prepared / 'scoring-inputs.jsonl').read_bytes() == draft_bytes
    refs = read(prepared / 'analysis-references.json')['references']
    assert {r['query_id']:r['reference'] for r in refs} == {'Q01':'无','Q02':'有','Q03':'无'}
    assert next(r for r in refs if r['query_id']=='Q03')['reference_status'] == 'human_adopted_AI_proposal'
    mats = read(prepared / 'materials.json')
    q3 = next(q for q in mats['queries'] if q['query_id']=='Q03')
    assert q3['authorship'] == 'assistant' and q3['adopted_severity'] == 0
    assert q3['text'] == next(q for q in read(WORK / 'draft-01/materials.json')['queries'] if q['query_id']=='Q03')['text']
    plan = read(prepared / 'execution-plan.json')
    prior_plan = read(ROOT / 'reviews/case-content-replacement-v1/prepared-01/execution-plan.json')
    assert plan['acceptance'] == prior_plan['acceptance']
    assert plan['budget'] == {'unique_prompts':9,'engineering_full_forwards':54,'prefix_forwards':9,
                              'production_forwards':9,'format_extra_forward_max':63,
                              'usual_if_label_then_eos':81,'total_forward_max':135}
    assert plan['current_GPU_window'] is None and plan['allocation'] is None and not plan['GPU_qualified']
    old_runtime = (ROOT / 'src/diagnostics/case_content_replacement_runtime_v1.py').read_text()
    expected_runtime = old_runtime.replace('case_content_replacement_inputs_v1', 'hehe_sense_context_inputs_v1').replace('run_case_content_replacement_v1.py','run_hehe_sense_context_v1.py')
    assert (ROOT / 'src/diagnostics/hehe_sense_context_runtime_v1.py').read_text() == expected_runtime
    old_window = ast.parse((ROOT / 'scripts/review/run_case_content_replacement_window_v1.py').read_text())
    new_window = ast.parse((ROOT / 'scripts/review/run_hehe_sense_context_window_v1.py').read_text())
    nodes = lambda tree: {n.name:ast.dump(n,include_attributes=False) for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
    a,b = nodes(old_window),nodes(new_window)
    assert all(a[name] == b[name] for name in ('identity','same_process','send_owned','Window','self_test'))
    expressions = read(prepared / 'comparisons.json')['comparisons']
    request_ids = [json.loads(line)['request_id'] for line in draft_bytes.splitlines()]
    arithmetic_checks = 0
    for case in range(6):
        scores = {rid:Fraction((i+1)*(case+2)-17,13) for i,rid in enumerate(request_ids)}
        for e in expressions:
            plus = next(t['request_id'] for t in e['terms'] if t['coefficient']==1)
            minus = next(t['request_id'] for t in e['terms'] if t['coefficient']==-1)
            exact = sum(t['coefficient']*scores[t['request_id']] for t in e['terms'])
            assert exact == scores[plus]-scores[minus]
            sign = -1 if plus.startswith('hsc-Q02-') else 1
            assert sign*exact == sign*scores[plus]-sign*scores[minus]
            arithmetic_checks += 1
    assert 'torch' not in sys.modules
    return {'status':'pass','review_draft_reconstruction':original,
            'accepted_items':['Q02','Q03','D03'],'adopted_references':{'Q01':'无','Q02':'有','Q03':'无'},
            'unchanged_model_input_records':9,'new_runtime_equals_prior_after_name_redirection':True,
            'watchdog_ownership_and_stop_logic_unchanged':True,'qualification_gates_unchanged':True,
            'exact_fraction_checks':arithmetic_checks,'GPU_qualification_completed':False,
            'weights_loaded':False,'CUDA_initialized':False,'GPU_touched':False}


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared',type=Path,default=WORK/'prepared-01')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args(); result=audit(args.prepared)
    if args.output:
        with args.output.open('x',encoding='utf-8') as f: json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps(result,ensure_ascii=False))
