#!/usr/bin/env python3
"""CPU-only scientific closeout with disclosed continuation audit recovery."""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from diagnostics import cross_term_mechanism_inputs_v1 as c


def preflight():
    work = c.WORK; run = work / 'run-01'; launch = work / 'launch-01'
    c.validate(c.PREPARED)
    state = c.read(run / 'state.json')
    c.require(state['status'] == 'complete' and state['owned_worker_absent'] and state['worker_exit_code'] == 0, 'Normal GPU release required')
    c.verify(state['resource_release']); release = c.read(state['resource_release']['path'])
    c.require(release['owned_worker_absent'] and release['worker_exit_code'] == 0, 'Release proof missing')
    pids = sorted({i['controller_pid'] for i in state['invocations']} | {i['worker_pid'] for i in state['invocations']})
    host = c.read(work / 'process-release-check.json')
    c.require(host['status'] == 'pass' and host['host_pid_context'] and host['all_owned_processes_absent'], 'Host release proof missing')
    c.require(sorted(x['pid'] for x in host['processes']) == pids and all(not x['proc_exists'] for x in host['processes']), 'Host PID scope mismatch')
    c.require(all(not Path(f'/proc/{pid}').exists() for pid in pids), 'Owned PID present')
    audit = c.read(work / 'result-audit-02.json')
    c.require(audit['status'] == 'pass' and audit['absolute_vectors'] == 1476 and audit['effects'] == 120
              and audit['self_controls'] == 192 and audit['restoration_contrasts'] == 72
              and audit['joint_contrasts'] == 24 and audit['branch_proofs'] == 960, 'Independent audit incomplete')
    c.require(audit['all_original_execution_and_numerical_gates_retained'], 'Execution gates changed')
    recovery = audit['continuation_comparator_recovery']
    c.require(recovery['format_branch_proofs'] == 96 and recovery['all_before_vectors_exactly_reconstructed']
              and recovery['all_installed_vectors_exactly_reconstructed'] and recovery['nonformat_comparator_policy_unchanged']
              and recovery['numerical_execution_gates_unchanged'] and not recovery['GPU_rerun'], 'Recovery scope mismatch')
    c.require(recovery['exact_unappended_comparators'] + recovery['exact_right_padding_comparators'] == 96, 'Missing format comparator')
    c.require(all(audit['new_case_comparisons'].values()) and not audit['historical_replay']['applicable'], 'New-case audit mismatch')
    for key in ['source', 'original_auditor_source', 'raw_seal', 'results']:
        c.verify(audit[key])
    for folder in ['prepared-01', 'results-01', 'interpretation-01', 'recovery-01']:
        manifest = c.read(work / folder / ('manifest-02.json' if folder == 'recovery-01' else 'manifest.json'))
        for item in manifest['artifacts'] + manifest.get('sources', []):
            c.verify(item)
    for item in c.read(launch / 'parent-selectors.json')['files']:
        c.verify(item)
    for item in c.read(launch / 'source-pin.json').values():
        c.verify(item)
    c.require(c.read(launch / 'state.json')['event'] == 'complete_GPU_released', 'Controller incomplete')
    c.require(c.read(work / 'recovery-01/test-02.json')['status'] == 'pass', 'Recovery regression tests failed')
    c.require(c.read(work / 'recovery-01/failure-01.json')['exit_code'] == 1, 'Original failure not retained')
    return state, release, pids


def main(check_only=False):
    state, release, pids = preflight()
    if check_only:
        print(json.dumps({'status': 'pass', 'mode': 'CPU_closeout_preflight', 'GPU_forwards': 0}))
        return
    work = c.WORK; run = work / 'run-01'; launch = work / 'launch-01'; out = work / 'closeout-03'
    c.require(not out.exists(), 'Closeout exists; do not repeat')
    c.require(not (c.PUBLIC / 'results-current.json').exists(), 'Scientific selector already exists')
    out.mkdir()
    receipt = {
        'status': 'complete', 'terminal_do_not_restart': True, 'closed_at_UTC': datetime.now(timezone.utc).isoformat(),
        'owned_processes_absent': pids, 'GPU_release': release, 'forwards': 1476, 'input_count': 36, 'new_case_count': 12,
        'upstream_configurations': 24, 'preceding_controls': 24, 'single_restorations': 48, 'joint_restorations': 24,
        'self_controls': 192, 'format_endpoints': 156, 'result_audit': c.info(work / 'result-audit-02.json'),
        'phase_seconds': sum(i['ended_at_unix'] - i['started_at_unix'] for i in state['invocations']),
        'first_start_to_last_release_seconds': release['checked_at_unix'] - state['invocations'][0]['started_at_unix'],
        'old_scientific_and_website_selectors_unchanged': c.info(launch / 'parent-selectors.json'),
        'research_scope': 'Twelve adopted development cases; fixed17/26/28, not independent confirmation or a unique path',
        'head_or_layer_search': False, 'all_owned_processes_checked': c.info(work / 'process-release-check.json'),
        'CPU_audit_recovery': c.info(work / 'recovery-01/manifest-02.json'),
        'original_audit_failure_preserved': c.info(work / 'recovery-01/failure-01.json'),
        'GPU_retry': False, 'scientific_data_changed': False, 'website_publication': False,
        'interpretation': c.info(work / 'interpretation-01/manifest.json'),
    }
    c.write(out / 'closeout.json', receipt)
    (out / 'README.md').write_text('# 三词条机制扩展：已完成\n\n1476次前向、全部12条查询与36份输入保留。GPU终态COMPLETE，不可重启。\n\n[结果解读](../interpretation-01/REPORT.md)、[完整报告](../results-01/REPORT.md)、[独立审计恢复说明](../recovery-01/README.md)。\n\n原审计错误地要求追加答案后的废弃分支向量必须匹配未追加条件；新审计对所有960处分支证明保持精确重建，其中8处续算采用同一因果前缀的既有右填充对照。运行与数值门槛、原始数据均未改动，没有GPU重跑或网站发布。\n')
    paths = [work / p for p in ['prepared-01/manifest.json', 'results-01/manifest.json', 'result-audit-02.json',
             'recovery-01/manifest-02.json', 'interpretation-01/manifest.json', 'process-release-check.json']]
    paths += [run / 'state.json', run / 'raw-seal.json', launch / 'complete-check.json',
              launch / 'authorization.json', launch / 'source-pin.json', Path(__file__)]
    c.write(out / 'manifest.json', {'artifacts': [c.info(p) for p in sorted(out.iterdir()) if p.is_file()],
                                   'sources': [c.info(p) for p in paths]})
    selector = {'status': 'complete', 'terminal_do_not_restart': True,
                'prepared': str(c.PREPARED.relative_to(ROOT)), 'run': str(run.relative_to(ROOT)),
                'results': str((work / 'results-01').relative_to(ROOT)),
                'report': str((work / 'interpretation-01/REPORT.md').relative_to(ROOT)),
                'full_report': str((work / 'results-01/REPORT.md').relative_to(ROOT)),
                'closeout_manifest': c.info(out / 'manifest.json'),
                'GPU_released_at': datetime.fromtimestamp(release['checked_at_unix'], timezone.utc).isoformat(),
                'CPU_audit_recovery_disclosed': True, 'website_publication': False}
    c.atomic(c.PUBLIC / 'results-current.json', selector)
    print(json.dumps(selector, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--check', action='store_true')
    main(parser.parse_args().check)
