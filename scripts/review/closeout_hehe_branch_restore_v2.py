#!/usr/bin/env python3
"""Version2: close a released/audited run after the isolated report-copy repair."""
import json,sys
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from diagnostics import hehe_branch_restore_inputs_v1 as c


def main():
    work=c.WORK;run=work/'run-01';out=work/'closeout-01'
    c.require(not out.exists(),'Closeout already exists')
    c.validate(c.PREPARED)
    state=c.read(run/'state.json')
    c.require(state['status']=='complete' and state['owned_worker_absent'] and state['worker_exit_code']==0,'Normal release required')
    c.verify(state['resource_release']);release=c.read(state['resource_release']['path'])
    c.require(release['owned_worker_absent'] and release['worker_exit_code']==0,'Release proof incomplete')
    pids=sorted({i['controller_pid'] for i in state['invocations']}|{i['worker_pid'] for i in state['invocations']})
    c.require(all(not Path(f'/proc/{pid}').exists() for pid in pids),'Owned process remains')
    audit=c.read(work/'result-audit-01.json')
    c.require(audit['status']=='pass' and audit['absolute_vectors']==152 and audit['self_controls']==20 and
        audit['restoration_contrasts']==8 and audit['historical_replay']['full_vectors_exact']==8,'Independent audit incomplete')
    for folder in ['prepared-01','results-01','report-02']:
        m=c.read(work/folder/'manifest.json')
        for row in m['artifacts']+m.get('sources',[])+m.get('inputs',[]):c.verify(row)
        if 'source' in m:c.verify(m['source'])
    for row in c.read(work/'launch-02/parent-selectors.json')['files']:c.verify(row)
    launch=c.read(work/'launch-02/state.json')
    c.require(launch['event']=='complete_GPU_released','Controller not complete')
    recovery=work/'recovery-01'
    c.require(c.read(recovery/'report-regression-tests.json')['status']=='pass','Report regression check incomplete')
    c.require(c.read(recovery/'process-release-check.json')['status']=='pass','Process release not verified')
    for row in c.read(recovery/'diagnosis.json')['preserved_inputs']:c.verify(row)
    for row in c.read(recovery/'source-pins.json')['files']:c.verify(row)
    for row in c.read(work/'schedule-01/source-pins.json')['files']:c.verify(row)
    c.require(c.read(work/'report-02/copy-audit.json')['all_copied_bytes_identical'],'Copied numerical artifacts changed')
    out.mkdir()
    receipt={'status':'complete','terminal_do_not_restart':True,'closed_at_UTC':datetime.now(timezone.utc).isoformat(),
        'owned_processes_absent':pids,'GPU_release':release,'forwards':152,
        'phase_seconds':sum(i['ended_at_unix']-i['started_at_unix'] for i in state['invocations']),
        'first_start_to_last_release_seconds':state['invocations'][-1]['ended_at_unix']-state['invocations'][0]['started_at_unix'],
        'input_count':4,'upstream_configurations':4,'single_branch_restorations':8,'self_controls':20,'format_endpoints':16,
        'result_audit':c.info(work/'result-audit-01.json'),
        'old_scientific_and_website_selectors_unchanged':c.info(work/'launch-02/parent-selectors.json'),
        'research_scope':'same four exposed prompts; conditional component roles, not a unique natural path',
        'joint_restoration_or_head_interventions_executed':False,
        'report_recovery':c.info(recovery/'diagnosis.json'),
        'report_regression_tests':c.info(recovery/'report-regression-tests.json'),
        'all_owned_processes_checked':c.info(recovery/'process-release-check.json'),
        'original_scheduler_failure_preserved':c.info(work/'schedule-01/state.json'),
        'report_version':'report-02','no_GPU_retry_or_numeric_gate_change':True}
    c.write(out/'closeout.json',receipt)
    (out/'README.md').write_text('# 两处单分支恢复：已完成\n\n本实验终态COMPLETE，不可重启。完整报告位于../report-02/REPORT.md；原始结果和协议各自封存。\n\n四个原生端点及四个上游替换端点与上一轮完整向量一致；本轮新增8个分别恢复26层注意力或28层MLP的条件。读数衡量固定第17层干预下的组件作用，不能相加为中介份额或推广为唯一自然路径。\n')
    paths=[work/'prepared-01/manifest.json',work/'results-01/manifest.json',work/'report-02/manifest.json',
        work/'result-audit-01.json',run/'state.json',run/'raw-seal.json',
        work/'launch-02/complete-check.json',work/'launch-02/authorization.json',work/'launch-02/source-pin.json',Path(__file__),
        recovery/'diagnosis.json',recovery/'report-regression-tests.json',recovery/'process-release-check.json',recovery/'source-pins.json']
    c.write(out/'manifest.json',{'artifacts':[c.info(p) for p in sorted(out.iterdir()) if p.is_file()],
        'sources':[c.info(p) for p in paths]})
    selector={'status':'complete','terminal_do_not_restart':True,'prepared':str(c.PREPARED.relative_to(ROOT)),
        'run':str(run.relative_to(ROOT)),'results':str((work/'results-01').relative_to(ROOT)),
        'report':str((work/'report-02/REPORT.md').relative_to(ROOT)),
        'closeout_manifest':c.info(out/'manifest.json'),
        'GPU_released_at':datetime.fromtimestamp(release['checked_at_unix'],timezone.utc).isoformat()}
    c.atomic(c.PUBLIC/'results-current.json',selector)
    scheduled=c.read(c.PUBLIC/'schedule-current.json')
    scheduled.update(status='complete_after_CPU_report_recovery',GPU_started=True,GPU_released=True,
        original_scheduler_terminal=c.info(work/'schedule-01/state.json'),
        results_selector=c.info(c.PUBLIC/'results-current.json'),report=selector['report'])
    c.atomic(c.PUBLIC/'schedule-current.json',scheduled,replace=True)
    print(json.dumps(selector,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
