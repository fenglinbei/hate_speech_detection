#!/usr/bin/env python3
"""Seal a normally released bridge run and publish its isolated result selector."""
import json,sys,time
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from diagnostics import hehe_bridge_inputs_v1 as c


def main():
    work=c.WORK;run=work/'run-01';out=work/'closeout-01'
    c.require(not out.exists(),'Closeout already exists')
    state=c.read(run/'state.json')
    c.require(state['status']=='complete' and state['owned_worker_absent'] and state['worker_exit_code']==0,'Normal release required')
    c.verify(state['resource_release']);release=c.read(state['resource_release']['path'])
    c.require(release['owned_worker_absent'] and release['worker_exit_code']==0,'Release proof incomplete')
    pids=sorted({i['controller_pid'] for i in state['invocations']}|{i['worker_pid'] for i in state['invocations']})
    c.require(all(not Path(f'/proc/{pid}').exists() for pid in pids),'Owned process remains')
    for name in ['result-audit-01.json','historical-replay-audit-01.json']:
        c.require(c.read(work/name)['status']=='pass','Independent audit missing')
    for folder in ['prepared-01','results-01','report-01']:
        m=c.read(work/folder/'manifest.json')
        for row in m['artifacts']+m.get('sources',[])+m.get('inputs',[]):c.verify(row)
        if 'source' in m:c.verify(m['source'])
    for row in c.read(work/'launch-01/parent-selectors.json')['files']:c.verify(row)
    out.mkdir()
    receipt={'status':'complete','terminal_do_not_restart':True,'closed_at_UTC':datetime.now(timezone.utc).isoformat(),
        'owned_processes_absent':pids,'GPU_release':release,'forwards':108,
        'phase_seconds':sum(i['ended_at_unix']-i['started_at_unix'] for i in state['invocations']),
        'first_start_to_last_release_seconds':state['invocations'][-1]['ended_at_unix']-state['invocations'][0]['started_at_unix'],
        'input_count':4,'cross_configurations':8,'self_controls':8,'format_endpoints':12,
        'result_audit':c.info(work/'result-audit-01.json'),'historical_replay':c.info(work/'historical-replay-audit-01.json'),
        'old_scientific_and_website_selectors_unchanged':c.info(work/'launch-01/parent-selectors.json'),
        'research_scope':'same four exposed prompts; no unique path or necessary component established',
        'further_component_interventions_executed':False,
        'CPU_test_harness_failure_preserved':c.info(work/'cpu-development-01/runtime-test-02.json')}
    c.write(out/'closeout.json',receipt)
    (out/'README.md').write_text('# 第17层替换轨迹：已完成\n\n本实验终态COMPLETE，不可重启。四个原生端点及八个替换端点均与历史完整词表向量一致；新增的是同一次替换之后的全部36层答案前轨迹。\n\n阅读../report-01/REPORT.md；数据在../results-01，输入和协议在../prepared-01。独立数值审核、历史复核和GPU正常释放见closeout.json。\n\n第17层嘿嘿替换后，第18层注意力后首次出现超出工程界的投影变化；23层MLP、26层注意力、28层MLP在四个方向同向响应最终delta。仍保留早期反方向、晚期补偿和RMS缩放。具体组件的必要性尚未检验。\n')
    paths=[work/'prepared-01/manifest.json',work/'results-01/manifest.json',work/'report-01/manifest.json',
        work/'result-audit-01.json',work/'historical-replay-audit-01.json',run/'state.json',run/'raw-seal.json',
        work/'launch-01/complete-check.json',work/'launch-01/authorization.json',work/'launch-01/source-pin.json',
        work/'cpu-development-01/test-harness-fix.json',Path(__file__)]
    c.write(out/'manifest.json',{'artifacts':[c.info(p) for p in sorted(out.iterdir()) if p.is_file()],
        'sources':[c.info(p) for p in paths]})
    selector={'status':'complete','terminal_do_not_restart':True,'prepared':str(c.PREPARED.relative_to(ROOT)),
        'run':str(run.relative_to(ROOT)),'results':str((work/'results-01').relative_to(ROOT)),
        'report':str((work/'report-01/REPORT.md').relative_to(ROOT)),
        'closeout_manifest':c.info(out/'manifest.json'),
        'GPU_released_at':datetime.fromtimestamp(release['checked_at_unix'],timezone.utc).isoformat()}
    c.atomic(c.PUBLIC/'results-current.json',selector)
    print(json.dumps(selector,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
