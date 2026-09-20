#!/usr/bin/env python3
"""Close the completed six-query run after independent audit and host release."""
import argparse,json,sys
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from diagnostics import jingba_attn_restore_inputs_v1 as c

def main(check_only=False):
    work=c.WORK;run=work/'run-01';launch=work/'launch-01';out=work/'closeout-01'
    c.validate(c.PREPARED)
    state=c.read(run/'state.json');c.require(state['status']=='complete' and state['owned_worker_absent'] and state['worker_exit_code']==0,'Normal release required')
    c.verify(state['resource_release']);release=c.read(state['resource_release']['path'])
    pids=sorted({i['controller_pid'] for i in state['invocations']}|{i['worker_pid'] for i in state['invocations']})
    host=c.read(work/'process-release-check.json')
    c.require(host['status']=='pass' and host['host_pid_context'] and host['all_owned_processes_absent'],'Host release proof required')
    c.require(sorted(r['pid'] for r in host['processes'])==pids and all(not r['proc_exists'] for r in host['processes']),'PID ownership mismatch')
    c.require(all(not Path(f'/proc/{pid}').exists() for pid in pids),'Owned process present')
    audit=c.read(work/'result-audit-01.json')
    c.require(audit['status']=='pass' and audit['absolute_vectors']==498 and audit['effects']==36 and audit['self_controls']==48 and audit['restoration_contrasts']==12 and audit['joint_contrasts']==0 and audit['branch_proofs']==120,'Independent inventory mismatch')
    c.require(audit['shared_prefix_checks']==3 and audit['all_original_gates_retained'] and all(audit['new_case_comparisons'].values()),'Independent checks incomplete')
    c.require(audit['context_restoration_checks']==2 and audit['historical_replay']['all_exact_equal'],'Paired context / historical replay missing')
    c.require(audit['historical_replay']['native_vectors']==18 and audit['historical_replay']['upstream_and_preceding_vectors']==24,'Historical replay inventory')
    for key in ['source','raw_seal','results']:c.verify(audit[key])
    for folder in ['prepared-01','results-01']:
        m=c.read(work/folder/'manifest.json')
        for item in m['artifacts']+m.get('sources',[]):c.verify(item)
    for item in c.read(launch/'parent-selectors.json')['files']:c.verify(item)
    for item in c.read(launch/'source-pin.json').values():c.verify(item)
    c.require(c.read(launch/'state.json')['event']=='complete_GPU_released','Controller incomplete')
    if check_only:return {'status':'pass','GPU_forwards':0}
    c.require(not out.exists() and not (c.PUBLIC/'results-current.json').exists(),'Never repeat scientific closeout')
    out.mkdir()
    receipt=dict(status='complete',terminal_do_not_restart=True,closed_at_UTC=datetime.now(timezone.utc).isoformat(),
        GPU_release=release,owned_processes_absent=pids,forwards=498,input_count=18,new_case_count=0,reused_case_count=6,
        upstream_configurations=12,preceding_controls=12,self_controls=48,format_endpoints=54,
        single_restorations=12,joint_restorations=0,head_or_layer_search=False,
        result_audit=c.info(work/'result-audit-01.json'),all_owned_processes_checked=c.info(work/'process-release-check.json'),
        phase_seconds=sum(i['ended_at_unix']-i['started_at_unix'] for i in state['invocations']),
        first_start_to_last_release_seconds=release['checked_at_unix']-state['invocations'][0]['started_at_unix'],
        old_scientific_and_website_selectors_unchanged=c.info(launch/'parent-selectors.json'),
        scientific_data_changed=False,GPU_retry=False,website_publication=False,
        research_scope='Six human-adopted related assistant-authored same-term queries; fixed layer17 U/P plus layer26 attention restoration from recipient-native; no independent population confirmation')
    c.write(out/'closeout.json',receipt)
    (out/'README.md').write_text('# 京巴第26层注意力分支恢复已完成\n\n498次前向正常完成并释放GPU。48自身控制、54格式端点和独立高精度数值复核通过。全部六条/双向保留；没有网站发布或旧实验重启。\n\n[完整结果报告](../results-01/REPORT.md) · [独立审计](../result-audit-01.json)\n')
    sources=[c.PREPARED/'manifest.json',work/'results-01/manifest.json',work/'result-audit-01.json',work/'process-release-check.json',run/'state.json',run/'raw-seal.json',launch/'source-pin.json',launch/'complete-check.json',Path(__file__)]
    c.write(out/'manifest.json',dict(artifacts=[c.info(p) for p in sorted(out.iterdir()) if p.is_file()],sources=[c.info(p) for p in sources]))
    selector=dict(status='complete',terminal_do_not_restart=True,prepared=str(c.PREPARED.relative_to(ROOT)),run=str(run.relative_to(ROOT)),results=str((work/'results-01').relative_to(ROOT)),report=str((work/'results-01/REPORT.md').relative_to(ROOT)),closeout_manifest=c.info(out/'manifest.json'),GPU_released_at=datetime.fromtimestamp(release['checked_at_unix'],timezone.utc).isoformat(),website_publication=False)
    c.atomic(c.PUBLIC/'results-current.json',selector)
    (c.PUBLIC/'README.md').write_text('六条既有京巴查询的第26层注意力分支恢复实验已完成，GPU已释放。\n\n[结果报告](../../../../reviews/jingba-attn-restore-v1/results-01/REPORT.md) · [独立复核](../../../../reviews/jingba-attn-restore-v1/result-audit-01.json) · [执行协议](../../../../reviews/jingba-attn-restore-v1/prepared-01/PROTOCOL.md)\n\n全部材料与两方向保留；固定第17层目标词替换后恢复第26层注意力分支，不扫描层/头。网站未发布本轮结果。\n')
    return selector
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--check',action='store_true');a=parser.parse_args()
    print(json.dumps(main(a.check),ensure_ascii=False,indent=2))
