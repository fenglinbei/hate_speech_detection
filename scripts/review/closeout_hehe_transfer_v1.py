#!/usr/bin/env python3
"""Seal the new-case experiment after release and independent numerical audit."""
import json,sys
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from diagnostics import hehe_transfer_inputs_v1 as c


def main():
    work=c.WORK;run=work/'run-01';launch=work/'launch-01';out=work/'closeout-01'
    c.require(not out.exists(),'Closeout exists; do not repeat')
    plan,_,rows,jobs,selfs=c.validate(c.PREPARED)
    state=c.read(run/'state.json');c.require(state['status']=='complete' and state['owned_worker_absent'] and state['worker_exit_code']==0,'Normal release required')
    c.verify(state['resource_release']);release=c.read(state['resource_release']['path'])
    c.require(release['owned_worker_absent'] and release['worker_exit_code']==0,'Release proof missing')
    pids=sorted({i['controller_pid'] for i in state['invocations']}|{i['worker_pid'] for i in state['invocations']})
    c.require(all(not Path(f'/proc/{pid}').exists() for pid in pids),'Owned process remains')
    audit=c.read(work/'result-audit-01.json')
    c.require(audit['status']=='pass' and audit['absolute_vectors']==456 and audit['effects']==40 and audit['self_controls']==64
        and audit['restoration_contrasts']==24 and audit['joint_contrasts']==8 and audit['branch_proofs']==320,'Independent audit incomplete')
    c.require(all(audit['new_case_comparisons'].values()) and not audit['historical_replay']['applicable'],'New-case audit scope mismatch')
    for folder in ['prepared-01','results-01']:
        manifest=c.read(work/folder/'manifest.json')
        for item in manifest['artifacts']+manifest.get('sources',[]):c.verify(item)
    for item in c.read(launch/'parent-selectors.json')['files']:c.verify(item)
    for item in c.read(launch/'source-pin.json').values():c.verify(item)
    c.require(c.read(launch/'state.json')['event']=='complete_GPU_released','Controller incomplete')
    c.require(c.read(work/'process-release-check.json')['status']=='pass','Host release check missing')
    out.mkdir()
    receipt={'status':'complete','terminal_do_not_restart':True,'closed_at_UTC':datetime.now(timezone.utc).isoformat(),
        'owned_processes_absent':pids,'GPU_release':release,'forwards':456,'input_count':8,'new_case_count':4,
        'upstream_configurations':8,'preceding_controls':8,'single_restorations':16,'joint_restorations':8,
        'self_controls':64,'format_endpoints':48,'result_audit':c.info(work/'result-audit-01.json'),
        'phase_seconds':sum(i['ended_at_unix']-i['started_at_unix'] for i in state['invocations']),
        'first_start_to_last_release_seconds':state['invocations'][-1]['ended_at_unix']-state['invocations'][0]['started_at_unix'],
        'old_scientific_and_website_selectors_unchanged':c.info(launch/'parent-selectors.json'),
        'research_scope':'Four adopted new cases; fixed17/26/28 interventions, not independent population confirmation or a unique path',
        'head_or_layer_search':False,'all_owned_processes_checked':c.info(work/'process-release-check.json')}
    c.write(out/'closeout.json',receipt)
    (out/'README.md').write_text('# 新案例固定层复现：已完成\n\n终态COMPLETE，不可重启。报告见../results-01/REPORT.md。四条新案例、八份输入全部保留；T03按用户要求继承有标签，T04单列边界。固定17/26/28层，未搜索新层/头，未替代旧分数。\n')
    paths=[work/'prepared-01/manifest.json',work/'results-01/manifest.json',work/'result-audit-01.json',
           run/'state.json',run/'raw-seal.json',launch/'complete-check.json',launch/'authorization.json',
           launch/'source-pin.json',work/'process-release-check.json',Path(__file__)]
    c.write(out/'manifest.json',{'artifacts':[c.info(p) for p in sorted(out.iterdir()) if p.is_file()],
                               'sources':[c.info(p) for p in paths]})
    selector={'status':'complete','terminal_do_not_restart':True,'prepared':str(c.PREPARED.relative_to(ROOT)),
        'run':str(run.relative_to(ROOT)),'results':str((work/'results-01').relative_to(ROOT)),
        'report':str((work/'results-01/REPORT.md').relative_to(ROOT)),'closeout_manifest':c.info(out/'manifest.json'),
        'GPU_released_at':datetime.fromtimestamp(release['checked_at_unix'],timezone.utc).isoformat()}
    c.atomic(c.PUBLIC/'results-current.json',selector)
    print(json.dumps(selector,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
