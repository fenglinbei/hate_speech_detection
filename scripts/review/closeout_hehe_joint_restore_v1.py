#!/usr/bin/env python3
"""Seal this new joint experiment only, after normal release and independent audit."""
import json,sys
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from diagnostics import hehe_joint_restore_inputs_v1 as c


def main():
    work=c.WORK;run=work/'run-01';out=work/'closeout-01';launch=work/'launch-01'
    c.require(not out.exists(),'Closeout already exists')
    c.validate(c.PREPARED)
    state=c.read(run/'state.json')
    c.require(state['status']=='complete' and state['owned_worker_absent'] and state['worker_exit_code']==0,'Normal release required')
    c.verify(state['resource_release']);release=c.read(state['resource_release']['path'])
    c.require(release['owned_worker_absent'] and release['worker_exit_code']==0,'Release proof incomplete')
    pids=sorted({i['controller_pid'] for i in state['invocations']}|{i['worker_pid'] for i in state['invocations']})
    c.require(all(not Path(f'/proc/{pid}').exists() for pid in pids),'Owned process remains')
    audit=c.read(work/'result-audit-01.json')
    c.require(audit['status']=='pass' and audit['absolute_vectors']==192 and audit['self_controls']==28 and
        audit['restoration_contrasts']==12 and audit['joint_contrasts']==4 and
        audit['historical_replay']['full_vectors_exact']==16 and audit['historical_replay']['comparator_trajectories_exact']==16,
        'Independent audit incomplete')
    for folder in ['prepared-01','results-01']:
        m=c.read(work/folder/'manifest.json')
        for row in m['artifacts']+m.get('sources',[]):c.verify(row)
    for row in c.read(launch/'parent-selectors.json')['files']:c.verify(row)
    c.verify(c.read(launch/'source-pin.json')['launcher'])
    c.require(c.read(launch/'state.json')['event']=='complete_GPU_released','Controller not complete')
    c.require(c.read(work/'process-release-check.json')['status']=='pass','Host process check missing')
    out.mkdir()
    receipt={'status':'complete','terminal_do_not_restart':True,'closed_at_UTC':datetime.now(timezone.utc).isoformat(),
        'owned_processes_absent':pids,'GPU_release':release,'forwards':192,
        'phase_seconds':sum(i['ended_at_unix']-i['started_at_unix'] for i in state['invocations']),
        'first_start_to_last_release_seconds':state['invocations'][-1]['ended_at_unix']-state['invocations'][0]['started_at_unix'],
        'input_count':4,'upstream_configurations':4,'single_branch_comparators':8,'joint_restorations':4,
        'self_controls':28,'format_endpoints':20,'result_audit':c.info(work/'result-audit-01.json'),
        'old_scientific_and_website_selectors_unchanged':c.info(launch/'parent-selectors.json'),
        'research_scope':'same four exposed prompts; conditional joint effects, not a unique natural path',
        'new_text_labels_or_head_interventions':False,'all_owned_processes_checked':c.info(work/'process-release-check.json')}
    c.write(out/'closeout.json',receipt)
    (out/'README.md').write_text('# 两处联合恢复：已完成\n\n本实验终态COMPLETE，不可重启。报告位于../results-01/REPORT.md。四份输入和双向上游替换不变，只新增26注意力＋28MLP联合恢复一种配置；原生、上游与单独恢复比较均重新生成。交互为当前分数尺度的有限条件差，不能解释为独立因果份额。\n')
    paths=[work/'prepared-01/manifest.json',work/'results-01/manifest.json',work/'result-audit-01.json',
        run/'state.json',run/'raw-seal.json',launch/'complete-check.json',launch/'authorization.json',
        launch/'source-pin.json',work/'process-release-check.json',Path(__file__)]
    c.write(out/'manifest.json',{'artifacts':[c.info(p) for p in sorted(out.iterdir()) if p.is_file()],
        'sources':[c.info(p) for p in paths]})
    selector={'status':'complete','terminal_do_not_restart':True,'prepared':str(c.PREPARED.relative_to(ROOT)),
        'run':str(run.relative_to(ROOT)),'results':str((work/'results-01').relative_to(ROOT)),
        'report':str((work/'results-01/REPORT.md').relative_to(ROOT)),
        'closeout_manifest':c.info(out/'manifest.json'),
        'GPU_released_at':datetime.fromtimestamp(release['checked_at_unix'],timezone.utc).isoformat()}
    c.atomic(c.PUBLIC/'results-current.json',selector)
    print(json.dumps(selector,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
