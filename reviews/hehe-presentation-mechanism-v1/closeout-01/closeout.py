from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import hashlib, json, shutil

ROOT = Path(__file__).resolve().parents[3]
WORK = ROOT / 'reviews/hehe-presentation-mechanism-v1'
PUBLIC = ROOT / 'docs/research/experiment-plans/hehe-presentation-mechanism-v1'
OUT = Path(__file__).resolve().parent

def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4194304),b''): h.update(b)
    return h.hexdigest()
def info(p):
    p=Path(p); return {'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p)}
def verify(x):
    assert Path(x['path']).stat().st_size == x['bytes'] and sha(x['path']) == x['sha256'], x['path']
def write(p,x):
    with Path(p).open('x',encoding='utf-8') as f: f.write(json.dumps(x,ensure_ascii=False,indent=2)+'\n')

assert read(WORK/'run-01/state.json')['status']=='complete'
assert read(WORK/'launch-02/complete-check.json')['status']=='complete'
release=read(WORK/'launch-02/final-release.json')
assert release['status']=='normally_released' and release['new_forwards']==162
assert all(x['absent'] for x in release['owned_processes'])
assert release['inventory']['compute_processes']==[]
assert all(x['used_mib']==0 and x['utilization']==0 for x in release['inventory']['devices'])
verify(release['run_release'])
for x in release['source_pins'].values(): verify(x)
assert not (WORK/'run-01/STOP').exists()
audit=read(WORK/'audits/results-01.json'); browser=read(WORK/'audits/viewer-results-01/audit.json')
assert audit['status']=='pass' and browser['status']=='pass'
assert audit['absolute_logit_vectors']==162 and audit['registered_expressions']==24
assert browser['real_conditions']==18 and browser['page_errors']==[]
for name in ['prepared-01','results-01','report-01']:
    m=read(WORK/name/'manifest.json')
    for x in m['artifacts']+m.get('sources',[]): verify(x)
for x in audit['sources']+browser['sources']: verify(x)
timing=read(WORK/'report-01/metrics.json')['timing']
closeout={'status':'complete','terminal_do_not_restart':True,'GPU_device':0,
    'counts':{'engineering_full':108,'prefix':18,'format':18,'production':18,'total':162},
    'timing':timing,'normal_worker_releases':2,'final_release':info(WORK/'launch-02/final-release.json'),
    'audits':[info(WORK/'audits/results-01.json'),info(WORK/'audits/viewer-results-01/audit.json')],
    'report':info(WORK/'report-01/manifest.json'),'results':info(WORK/'results-01/manifest.json'),
    'prepared':info(WORK/'prepared-01/manifest.json'),'raw_seal':info(WORK/'run-01/raw-seal.json'),
    'query_references_joined_after_raw_seal_and_normal_release':True,
    'activation_patching_executed':False,'stage2':'deferred until stage1 interpretation',
    'independent_confirmation':False,'prior_experiments_restarted':False,
    'new_website_release':'separate authorized incremental deployment pending',
    'at':datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()}
write(OUT/'closeout.json',closeout)
(OUT/'README.md').write_text('''# 本轮第一阶段完成

全部18个输入、24个注册比较和逐层机制读数已完成。GPU0共162次前向，两个工作进程正常退出，所有持有进程均已消失；最终四卡显存/利用率均为0，无计算进程。不得重启本次运行。

报告：[REPORT.md](../report-01/REPORT.md)。交互注意力：[index.html](../results-01/index.html)。机制页面：[mechanism.html](../results-01/mechanism.html)。独立数值复核与真实浏览器核验均通过，见closeout.json与manifest.json。

四种双义呈现均未修复Q01；分行换序与整句换序的分数变化方向相反。注意力、来源向量大小与局部方向分开呈现；这些中间读数不建立最终分类的因果路径。第二阶段激活替换仍待第一阶段解释后另行执行。

旧实验、已审核标签与历史结果保持原样。本轮结果及报告将按用户授权另行增量部署；部署状态由网站发布selector与发布收据决定。
''',encoding='utf-8')
shutil.copyfile(PUBLIC/'availability-current.json',OUT/'previous-availability-current.json')
files=[OUT/'closeout.py',OUT/'closeout.json',OUT/'README.md',OUT/'previous-availability-current.json',
       WORK/'launch-02/authorization.json',WORK/'launch-02/source-pin.json',WORK/'launch-02/state.json',
       WORK/'launch-02/engineering-check.json',WORK/'launch-02/complete-check.json',WORK/'run-01/qualification.json']
manifest={'status':'complete','immutable_after_seal':True,'artifacts':[info(p) for p in files],
          'scientific_chain':[closeout[k] for k in ['prepared','raw_seal','results','report','final_release']]+closeout['audits']}
write(OUT/'manifest.json',manifest)
selector={'status':'complete','terminal_do_not_restart':True,'prepared':str((WORK/'prepared-01').relative_to(ROOT)),
          'run':str((WORK/'run-01').relative_to(ROOT)),'results':str((WORK/'results-01').relative_to(ROOT)),
          'report':str((WORK/'report-01/REPORT.md').relative_to(ROOT)),
          'viewer':str((WORK/'results-01/index.html').relative_to(ROOT)),
          'mechanism_viewer':str((WORK/'results-01/mechanism.html').relative_to(ROOT)),
          'closeout_manifest':info(OUT/'manifest.json'),'GPU_released_at':timing['released_at'],
          'deadline':None,'stage2':'deferred'}
write(PUBLIC/'results-current.json',selector)
temp=PUBLIC/'availability-current.json.presentation.tmp'
write(temp,{'status':'complete_GPU_released','terminal_do_not_restart':True,
           'supersedes_closed_attempt':'launch-01','launch':'launch-02',
           'release':info(WORK/'launch-02/final-release.json'),'results_selector':info(PUBLIC/'results-current.json')})
temp.replace(PUBLIC/'availability-current.json')
print(json.dumps({'status':'complete','closeout_manifest':info(OUT/'manifest.json')}))
