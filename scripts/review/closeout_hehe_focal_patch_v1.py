#!/usr/bin/env python3
"""Normal-release, audit and report gates for an immutable stage2 completion."""
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import json,sys,time
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from diagnostics import hehe_focal_patch_inputs_v1 as c
from diagnostics.cross_model_applicability_execution_v1 import gpu_inventory

def main():
    work=c.WORK;run=work/'run-01';launch=work/'launch-01';out=work/'closeout-01'
    assert not out.exists();state=c.read(run/'state.json')
    assert state['status']=='complete' and state['worker_exit_code']==0 and state['owned_worker_absent']
    assert c.read(launch/'state.json')['event']=='complete_GPU_released'
    owners={i[k] for i in state['invocations'] for k in ('worker_pid','controller_pid')}
    assert all(not Path(f'/proc/{pid}').exists() for pid in owners),'Owned processes still present'
    inventory=gpu_inventory();assert all(not any(row.split(',')[-1].strip()==str(pid) for row in inventory['compute_processes']) for pid in owners)
    for i in state['invocations']:
        release=c.read(run/('release-'+i['id']+'.json'));assert release['worker_exit_code']==0 and release['owned_worker_absent']
    checked=c.read(launch/'complete-check.json');assert checked['status']=='complete' and checked['CPU_reconstructed']
    audit=c.read(work/'audits/results-01.json');browser=c.read(work/'audits/browser-results-01.json')
    assert audit['status']==browser['status']=='pass' and audit['absolute_vectors']==2340 and browser['measured']
    for p in [c.PREPARED/'manifest.json',work/'results-01/manifest.json',work/'report-01/manifest.json']:
        m=c.read(p)
        for r in m['artifacts']+m.get('sources',[]):c.verify(r)
    counts={i['id']:0 for i in state['invocations']}
    for p in (run/'records').glob('*/*.json'):
        rec=c.read(p);inv=c.read(rec['producer']['path'])['invocation'];counts[inv]+=1
    for p in (run/'format').glob('*.json'):
        rec=c.read(p);inv=c.read(rec['producer']['path'])['invocation'];counts[inv]+=len(rec['steps'])
    assert sum(counts.values())==2340 and sorted(counts.values())==[292,2048]
    release=c.read(state['resource_release']['path']);finished=release['checked_at_unix']
    out.mkdir();c.write(out/'final-inventory.json',dict(inventory,checked_at_unix=time.time(),owned_processes_absent=sorted(owners)))
    data={'status':'complete','terminal_do_not_restart':True,'prepared_manifest':c.info(c.PREPARED/'manifest.json'),
          'run_state':c.info(run/'state.json'),'raw_seal':c.info(run/'raw-seal.json'),'engineering_seal':c.info(run/'engineering-seal.json'),
          'release':state['resource_release'],'forwards':2340,'forwards_by_invocation':counts,'native_inputs':4,'focal_interventions':144,
          'preceding_position_interventions':144,'self_controls':288,'last_layer_controls':8,'fresh_margin_bound':checked['qualification']['margin_error_bound'],
          'owned_processes_absent':sorted(owners),'deadline':None,'GPU_released_at':datetime.fromtimestamp(finished,ZoneInfo('Asia/Shanghai')).isoformat(),
          'wall_seconds_first_start_to_release':finished-state['invocations'][0]['started_at_unix'],
          'phase_seconds':[i['ended_at_unix']-i['started_at_unix'] for i in state['invocations']],
          'audit':c.info(work/'audits/results-01.json'),'browser':c.info(work/'audits/browser-results-01.json'),
          'results_manifest':c.info(work/'results-01/manifest.json'),'report_manifest':c.info(work/'report-01/manifest.json'),
          'scientific_scope':'two exposed texts, four unchanged prompts; all layers/both directions, preceding controls, no independent confirmation',
          'old_runs_or_references_modified':False,'new_reference_join_after_release':True}
    c.write(out/'closeout.json',data)
    (out/'README.md').write_text('# 第二阶段完成\n\n单张GPU0完成2340次前向并正常退出；本次run终态禁止重启。\n\n[报告](../report-01/REPORT.md) · [全部层交互图](../results-01/index.html) · [完整读数](../results-01/results.json)。四个既有输入、144个焦点干预、144个前置位置干预及288个自替换全部保留。独立数值审计、实际浏览器检查和正常释放证明见closeout.json。\n\n旧结果和参考标签保持原样；本次结果独立采集，不能将依赖条件视为新样本。前置位置不假定零效应，向量移植不证明唯一词义或注意头路径。网站发布另有部署selector及收据。\n',encoding='utf-8')
    sources=[c.info(Path(__file__)),c.info(ROOT/'scripts/review/write_hehe_focal_patch_report_v1.py'),c.info(launch/'source-pin.json'),
             c.info(work/'audits/results-01.json'),c.info(work/'audits/browser-results-01.json'),data['results_manifest'],data['report_manifest']]
    c.write(out/'manifest.json',{'status':'complete','sources':sources,'artifacts':[c.info(p) for p in sorted(out.iterdir()) if p.is_file()]})
    c.atomic(c.PUBLIC/'results-current.json',{'status':'complete','terminal_do_not_restart':True,'prepared':str(c.PREPARED.relative_to(ROOT)),
        'run':str(run.relative_to(ROOT)),'results':str((work/'results-01').relative_to(ROOT)), 'report':str((work/'report-01/REPORT.md').relative_to(ROOT)),
        'viewer':str((work/'results-01/index.html').relative_to(ROOT)),'closeout_manifest':c.info(out/'manifest.json'),
        'GPU_released_at':data['GPU_released_at'],'deadline':None})
    print(json.dumps(data,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
