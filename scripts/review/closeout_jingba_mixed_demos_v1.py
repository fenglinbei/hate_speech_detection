#!/usr/bin/env python3
import json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from diagnostics import jingba_mixed_demos_inputs_v1 as c
w=c.WORK;run=w/'run-01';launch=w/'launch-01';out=w/'closeout-01'
c.validate(c.PREPARED);s=c.read(run/'state.json');a=c.read(w/'result-audit-01.json');host=c.read(w/'process-release-check.json')
c.require(s['status']=='complete' and s['owned_worker_absent'] and s['worker_exit_code']==0,'Normal release first')
pids=sorted({i['controller_pid'] for i in s['invocations']}|{i['worker_pid'] for i in s['invocations']})
c.require(host['status']=='pass' and host['host_pid_context'] and host['all_owned_processes_absent'] and sorted(r['pid'] for r in host['processes'])==pids and all(not r['proc_exists'] for r in host['processes']),'Host proof missing')
c.require(a['status']=='pass' and a['absolute_vectors']==270 and a['native_contrasts']==36 and a['historical_replay']['all_exact_equal'],'Independent audit required')
for p in [c.PREPARED/'manifest.json',w/'results-01/manifest.json']:
 m=c.read(p)
 for x in m['artifacts']+m.get('sources',[]):c.verify(x)
for x in c.read(launch/'parent-selectors.json')['files']:c.verify(x)
for key in ['source','raw_seal','results']:c.verify(a[key])
c.require(c.read(launch/'state.json')['event']=='complete_GPU_released','Controller incomplete');c.require(not out.exists(),'No repeated closeout');out.mkdir()
release=c.read(s['resource_release']['path']);c.verify(s['resource_release'])
c.write(out/'closeout.json',{'status':'complete','terminal_do_not_restart':True,'forwards':270,'native_inputs':30,'format_endpoints':30,'native_contrasts':36,'all36layers':True,'new_internal_interventions':0,'owned_processes_absent':pids,'GPU_release':release,'first_start_to_last_release_seconds':release['checked_at_unix']-s['invocations'][0]['started_at_unix'],'phase_seconds':sum(i['ended_at_unix']-i['started_at_unix'] for i in s['invocations']),'GPU_retry':False,'website_publication':False,'closed_at_unix':time.time()})
(out/'README.md').write_text('# 正确示例混合比较完成\n\n[结果报告](../results-01/REPORT.md) · [独立审计](../result-audit-01.json)\n\n270次前向正常完成，旧六条无示例输入精确重放。本轮无内部干预及网站发布。')
c.write(out/'manifest.json',{'artifacts':[c.info(p) for p in sorted(out.iterdir()) if p.is_file()],'sources':[c.info(p) for p in [c.PREPARED/'manifest.json',w/'results-01/manifest.json',w/'result-audit-01.json',w/'process-release-check.json',run/'state.json',run/'raw-seal.json',Path(__file__)]]})
selector={'status':'complete','terminal_do_not_restart':True,'prepared':str(c.PREPARED.relative_to(ROOT)),'run':str(run.relative_to(ROOT)),'results':str((w/'results-01').relative_to(ROOT)),'report':str((w/'results-01/REPORT.md').relative_to(ROOT)),'closeout_manifest':c.info(out/'manifest.json'),'website_publication':False}
c.atomic(c.PUBLIC/'results-current.json',selector);(c.PUBLIC/'README.md').write_text('正确示例分组与混合比较已完成。\n\n[结果报告](../../../../reviews/jingba-mixed-demos-v1/results-01/REPORT.md) · [执行协议](../../../../reviews/jingba-mixed-demos-v1/prepared-01/PROTOCOL.md)\n')
print(json.dumps(selector,ensure_ascii=False))
