"""Independent CPU inventory audit and final sealing of the new execution preparation."""
import argparse
from collections import Counter
import json
from pathlib import Path
import runpy
import sys

ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'src'),str(ROOT)]
from diagnostics import cross_model_applicability_execution_v1 as c


def audit(prepared):
    prepared=Path(prepared)
    freeze_check=runpy.run_path(str(ROOT/'scripts/review/freeze_cross_model_applicability_v1.py'))['check']()
    assert freeze_check['adopted_original_materials']==48 and freeze_check['adopted_relations']==240
    plan,profiles,requests=c.check_prepared(prepared,sealed=(prepared/'manifest.json').exists())
    ids=set()
    raw_source={}
    for profile in profiles:
        key=profile['model_key']
        native=[]
        for frame in ['new-model-inputs','legacy-model-inputs']:
            native+=c.jsonl(c.SCIENCE/f'tokenized/{key}-{frame}.jsonl')
        assert len(native)==len(requests[key])==540
        assert profile['allocation'] is None and profile['GPU_qualified'] is False
        assert profile['compute_dtype']=='float32' and profile['readout_dtype']=='float64'
        assert profile['attention']=='eager' and profile['batch_size']==1 and profile['use_cache'] is False
        counts=Counter(r['source_frame'] for r in requests[key])
        assert counts=={'naturalistic_primary_development':192,'explanatory_supplement':192,'legacy_core':120,'legacy_N_diagnostic':36}
        for request,original in zip(requests[key],native):
            assert request['condition_id']==original['condition_id']
            assert request['input_ids']==original['input_ids'] and request['prompt_sha256']==original['prompt_sha256']
            assert request['candidate_tokens']==original['candidate_tokens']
            assert not any(field in request for field in ['human_label','adopted_label','original_gold','rule_fit','sense_fit'])
            assert request['prompt_tokens']<=8192
            pair=(key,request['request_id']);assert pair not in ids;ids.add(pair)
        lookup={r['request_id']:r for r in requests[key]}
        if key!='qwen3-8b':
            assert all(lookup[r]['query_id'] is None for r in profile['production_ids'][:156])
            assert all(lookup[r]['query_id'] is not None for r in profile['production_ids'][156:])
        else:assert len(profile['production_ids'])==384 and all(lookup[r]['query_id'] for r in profile['production_ids'])
        for qid in {r['query_id'] for r in requests[key] if r['query_id']}:
            selected=max([r for r in requests[key] if r['query_id']==qid],key=lambda r:(r['prompt_tokens'],r['request_id']))
            assert selected['request_id'] in profile['format_probe_ids']
        raw_source[key]={'requests':540,'production':len(profile['production_ids']),'format_probes':16,
                         'candidate_tokens':profile['candidate_tokens'],'minimum_tokens':min(r['prompt_tokens'] for r in requests[key]),
                         'maximum_tokens':max(r['prompt_tokens'] for r in requests[key])}
    actual_new=c.read(c.SCIENCE/'analysis-plan.json')
    assert len(actual_new['comparisons'])==1152
    assert Counter(r['analysis_role'] for r in actual_new['comparisons'])=={'naturalistic_primary_development':576,'explanatory_supplement':576}
    assert sum(r['priority']=='primary' for r in actual_new['comparisons'])==208
    assert actual_new['aggregation']['pool_primary_and_explanatory_supplement'] is False
    legacy=c.read(prepared/'legacy-analysis.json')
    assert len(legacy['comparisons'])==168 and len(legacy['references'])==12
    original=c.read(c.LEGACY_SOURCES[1])
    assert [r['terms'] for r in legacy['comparisons']]==[r['terms'] for r in original['comparisons']]
    assert Counter(legacy['condition_frames'].values())=={'legacy_core':120,'legacy_N_diagnostic':36}
    assert set(legacy['condition_query_ids'])=={r['condition_id'] for r in c.jsonl(c.SCIENCE/'legacy-model-inputs.jsonl')}
    tests=c.read(prepared/'cpu-tests.json')
    assert tests['status']=='pass' and tests['tests']==9 and tests['failures']==tests['errors']==0
    assert tests['research_pretrained_model_forward_calls']==tests['GPU_forward_calls']==0
    assert plan['budget']['engineering_forwards']==5*540*3==8100
    assert plan['budget']['new_scientific_forwards']==384*3+156*2==1464
    return {'status':'pass','science_freeze':c.info(c.SCIENCE/'manifest.json'),
        'adopted_original_materials':48,'adopted_relations':240,
        'primary_naturalistic_families':2,'explanatory_supplement_families':2,'conservative_dependency_clusters':1,
        'model_specific_input_records_checked':len(ids),'models':raw_source,
        'new_expressions':1152,'naturalistic_primary_priority_readouts':208,
        'legacy_core_expressions':168,'legacy_query_references':12,'legacy_external_N':36,
        'CPU_tests_passed':9,'GPU_qualification':False,'model_pretrained_forward_calls':0,
        'meaning':'CPU structural and implementation evidence only; model-specific GPU acceptance remains pending.'}


def seal(prepared):
    prepared=Path(prepared)
    assert not (prepared/'manifest.json').exists(),'Immutable preparation already sealed'
    assert not list(c.WORK.glob('bound-*.json')) and not list(c.WORK.glob('run-*')),'Unexpected GPU binding/run before CPU closeout'
    report=audit(prepared);c.dump(prepared/'independent-cpu-audit.json',report)
    scientific_selector=ROOT/'docs/research/experiment-plans/cross-model-applicability-v1/current.json'
    (prepared/'scientific-selector-at-preparation.json').write_bytes(scientific_selector.read_bytes())
    c.dump(prepared/'manifest.json',{'schema_version':'cross-model-applicability-execution-preparation/v1',
        'created_at_utc':c.now(),'status':'CPU_complete_GPU_qualification_pending','scientific_input_freeze':True,
        'runtime_implementation_frozen':True,'GPU_allocation':None,'GPU_qualified':False,
        'artifacts':[c.info(p) for p in sorted(prepared.rglob('*')) if p.is_file()],
        'sources':[c.info(p) for p in [*c.IMPLEMENTATION,Path(__file__).resolve(),c.SCIENCE/'manifest.json']]})
    base=ROOT/'docs/research/experiment-plans/cross-model-applicability-execution-v1';base.mkdir(exist_ok=True)
    c.dump(base/'current.json',{'schema_version':'cross-model-applicability-execution-selector/v1',
        'status':'CPU_prepared_stop_before_GPU','prepared_directory':str(prepared.relative_to(ROOT)),
        'manifest':c.info(prepared/'manifest.json'),'scientific_manifest':c.info(c.SCIENCE/'manifest.json'),
        'delivery':str((prepared/'README.md').relative_to(ROOT)),
        'allocation':None,'GPU_run_started':False,'GPU_qualified':False,'model_pretrained_forward_calls':0,
        'next_step':'Bind an authorized GPU window; qualify each model, then execute the frozen legacy and development/supplement matrices.'})
    selected=c.read(scientific_selector)
    selected.update(status='all_reviewed_scientific_inputs_frozen_CPU_runtime_prepared',
                    execution_preparation=c.info(prepared/'manifest.json'),runtime_implementation_frozen=True,
                    next_step='Model-specific GPU allocation and numerical/format qualification; no bound file or run exists.')
    c.dump(scientific_selector,selected)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--prepared',type=Path,default=c.PREPARED);p.add_argument('--seal',action='store_true')
    a=p.parse_args();result=seal(a.prepared) if a.seal else audit(a.prepared)
    if not a.seal and (a.prepared/'manifest.json').exists():assert result==c.read(a.prepared/'independent-cpu-audit.json')
    print(json.dumps(result,ensure_ascii=False,indent=2))
