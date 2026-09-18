#!/usr/bin/env python3
"""Freeze a nonnumerical UTF-8 runtime amendment and audited engineering import."""
import ast
import difflib
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT)]
from diagnostics import cross_model_applicability_execution_v1 as c

WORK=c.WORK
OUT=WORK/'utf8-io-amendment-01'
SOURCE=WORK/'run-glm4-9b-chat-02'
PREP=WORK/'prepared-glm-content-boundary-01'
ENTRY=ROOT/'scripts/review/run_cross_model_applicability_utf8_v1.py'
ORIGINAL=ROOT/'scripts/review/run_cross_model_applicability_v1.py'
BASE=WORK/'bound-glm4-9b-chat-03-base.json'
BOUND=WORK/'bound-glm4-9b-chat-03.json'


def prepare():
    assert sys.flags.utf8_mode==1 and not OUT.exists() and not BOUND.exists()
    _,profiles,_=c.check_prepared(PREP)
    state=c.read(SOURCE/'state.json')
    assert state['status']=='failed' and state['error_type']=='UnicodeDecodeError'
    assert state['owned_worker_absent'] and state['worker_exit_code']==1 and state['new_forwards']==2700
    assert not Path(f'/proc/{state["worker_pid"]}').exists()
    closeout=WORK/'audits/glm-utf8-failure-numerical-closeout-01.json';checked=c.read(closeout)
    assert checked['failed_state']==c.info(SOURCE/'state.json')
    assert checked['qualification_arithmetic']['status']=='pass' and len(checked['sources'])==2700
    entries=[]
    for info in checked['sources']:
        c.verify(info);p=Path(info['path']);r=c.read(p);c.verify(r['raw_logits'])
        entries.append({'stage':r['pass_id'],'request_id':r['request_id'],'receipt':info,'vector':r['raw_logits']})
    assert len({(r['stage'],r['request_id']) for r in entries})==2700
    original_ast={n.name:ast.dump(n) for n in ast.parse(ORIGINAL.read_text(encoding='utf8')).body if isinstance(n,ast.FunctionDef)}
    amended_ast={n.name:ast.dump(n) for n in ast.parse(ENTRY.read_text(encoding='utf8')).body if isinstance(n,ast.FunctionDef)}
    unchanged=['save_vector','seal_or_check','engineering_values','validate_format_record','analyze','main']
    assert all(original_ast[name]==amended_ast[name] for name in unchanged)
    probe=Path(entries[0]['receipt']['path'])
    code="import locale,sys,json;from pathlib import Path;locale.setlocale(locale.LC_CTYPE,'C');json.loads(Path(sys.argv[1]).read_text())"
    baseline=subprocess.run([sys.executable,'-c',code,str(probe)],env=dict(os.environ,PYTHONUTF8='0'),capture_output=True,text=True)
    fixed=subprocess.run([sys.executable,'-c',code,str(probe)],env=dict(os.environ,PYTHONUTF8='1'),capture_output=True,text=True)
    assert baseline.returncode!=0 and 'UnicodeDecodeError' in baseline.stderr
    assert fixed.returncode==0,fixed.stderr
    OUT.mkdir()
    c.dump(OUT/'encoding-test.json',{'status':'pass','forced_LC_CTYPE':'C','UTF8_mode_0_reproduces_UnicodeDecodeError':True,
        'UTF8_mode_1_reads_same_bytes':True,'source_receipt':entries[0]['receipt'],'GPU_forwards':0,
        'unchanged_runner_functions_AST_checked':unchanged})
    (OUT/'runner.diff').write_text(''.join(difflib.unified_diff(ORIGINAL.read_text(encoding='utf8').splitlines(True),
        ENTRY.read_text(encoding='utf8').splitlines(True),fromfile=str(ORIGINAL),tofile=str(ENTRY))),encoding='utf8')
    c.dump(OUT/'engineering-import.json',{'schema_version':'verified-engineering-import/v1',
        'model_key':'glm4-9b-chat','prepared_manifest':c.info(PREP/'manifest.json'),
        'source_directory':str(SOURCE),'source_run_id':state['run_id'],
        'source_state':c.info(SOURCE/'state.json'),'source_binding':c.info(SOURCE/'binding.json'),
        'source_identity':c.info(SOURCE/'runtime-identity.json'),'numerical_closeout':c.info(closeout),
        'source_release':state['resource_release'],'records':entries,
        'physical_score_ids_and_original_producer_files_preserved':True,
        'production_or_format_records_imported':False,'source_run_restart_allowed':False})
    sources=[ENTRY,Path(__file__).resolve(),ORIGINAL,*c.IMPLEMENTATION,PREP/'manifest.json',BASE]
    manifest={'schema_version':'cross-model-UTF8-runtime-amendment/v1','created_at':c.now(),
        'required_python_utf8_mode':1,'entry_point':c.info(ENTRY),
        'purpose':'Protect JSON/text reads against C locale changes; numerical code and frozen input bytes remain unchanged. New run imports all audited engineering receipts from a terminal I/O failure.',
        'original_run_kept_failed_and_unmodified':True,'numerical_tolerance_or_scientific_input_changes':False,
        'artifacts':[c.info(p) for p in sorted(OUT.iterdir()) if p.is_file()],
        'sources':[c.info(p) for p in sorted(set(sources))]}
    c.dump(OUT/'manifest.json',manifest)
    b=c.read(BASE)
    assert b['prepared_manifest']==c.info(PREP/'manifest.json') and b['allocation']==c.read(SOURCE/'binding.json')['allocation']
    b.update(runtime_amendment=c.info(OUT/'manifest.json'),engineering_import=c.info(OUT/'engineering-import.json'),
             python_UTF8_mode_required=1,old_failed_run_must_not_restart=True)
    c.atomic(BOUND,b)
    return test()


def test():
    assert sys.flags.utf8_mode==1
    module=runpy.run_path(str(ENTRY));module['validate_amendment'](BOUND)
    fixture=WORK/'audits/utf8-import-CPU-fixture-01'
    assert not fixture.exists()
    fixture.mkdir();c.atomic(fixture/'binding.json',c.read(BOUND))
    c.atomic(fixture/'state.json',{'run_id':'SYNTHETIC_CPU_IMPORT_AUDIT_ONLY','GPU_forward_calls':0})
    _,profiles,rs=c.check_prepared(PREP);profile=profiles[0];requests=rs['glm4-9b-chat']
    identity=c.read(SOURCE/'runtime-identity.json')
    values,sources=module['engineering_values'](fixture,requests,c.sha(fixture/'binding.json'),profile,identity)
    q=c.qualify(values,c.read(PREP/'execution-plan.json')['acceptance'])
    assert q==c.read(WORK/'audits/glm-utf8-failure-numerical-closeout-01.json')['qualification_arithmetic']
    assert len(sources)==2700 and all(Path(r['path']).is_relative_to(SOURCE) for r in sources)
    request=dict(requests[0],input_ids=requests[0]['input_ids']+[198])
    try:module['check_record'](fixture,'reference',request,c.sha(fixture/'binding.json'),profile,identity=identity)
    except ValueError as e:assert 'Padding/answer-position' in str(e)
    else:raise AssertionError('Changed input prefix was accepted')
    rp,vp=module['paths'](fixture,'production',requests[0]['request_id'])
    assert rp.is_relative_to(fixture) and vp.is_relative_to(fixture)
    receipt={'status':'pass','runtime_amendment':c.info(OUT/'manifest.json'),'binding':c.info(BOUND),
        'engineering_records_and_vectors_reconstructed':2700,'qualification_exactly_matches_original_CPU_reconstruction':True,
        'changed_prefix_rejected':True,'production_paths_use_new_run':True,'fixture_is_explicitly_CPU_only':True,
        'extra_GPU_forwards':0,'query_reference_join_performed':False}
    c.atomic(WORK/'audits/utf8-import-CPU-test-01.json',receipt)
    return receipt


if __name__=='__main__':
    print(json.dumps(prepare(),ensure_ascii=False,indent=2))
