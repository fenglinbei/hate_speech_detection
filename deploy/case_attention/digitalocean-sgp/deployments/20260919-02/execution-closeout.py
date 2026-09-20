from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import hashlib,json,shutil,time
root=Path(__file__).resolve().parents[4]
w=Path(__file__).resolve().parents[1]
base=root/'deploy/case_attention/digitalocean-sgp'
out=base/'deployments/20260919-02'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4194304),b''):h.update(b)
 return h.hexdigest()
def info(p):return {'path':str(p),'bytes':Path(p).stat().st_size,'sha256':sha(p)}
def write(p,x):
 with Path(p).open('x',encoding='utf-8') as f:f.write(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
assert read(w/'browser-live-complete-01/browser.json')['status']=='pass'
assert read(w/'external-https.json')['status']=='pass'
assert read(w/'credential-cleanup.json')['temporary_plaintext_removed']
assert read(w/'preview-cleanup.json')['owned_preview_absent']
assert read(w/'archive-cleanup.json')['archive_removed']
assert not (w/'runtime/credential.json').exists()
for item in read(w/'deployment-source-pins.json')['files']+[read(w/'browser-continuation-v2-source-pins.json')['file']]:
 assert sha(item['path'])==item['sha256']
receipt=read(w/'activation-receipt.json');assert receipt['status']=='deployed'
before=read(w/'remote-receipts/before.json');after=read(w/'remote-receipts/after.json');assert before==after
for p in (w/'remote-receipts').iterdir():
 if p.is_file():shutil.copyfile(p,out/p.name)
for name in ['staging-receipt.json','external-https.json','credential-cleanup.json','preview-cleanup.json','archive-cleanup.json','browser-continuation-source-pins.json','browser-continuation-v2-source-pins.json']:
 shutil.copyfile(w/name,out/name)
shutil.copyfile(w/'browser-live-01/failure.json',out/'live-browser-attempt-01-timeout.json')
shutil.copyfile(w/'browser-continuation-live-02/browser.json',out/'live-browser-continuation.json')
shutil.copyfile(w/'browser-continuation-local-02/browser.json',out/'local-browser-continuation.json')
shutil.copyfile(w/'browser-live-complete-01/browser.json',out/'live-browser.json')
for name in ['mean.svg','layers.svg','desktop-mean.png','mobile.png','hehe-mean.svg','hehe-desktop-mean.png','hehe-mobile.png']:
 shutil.copyfile(w/'browser-live-complete-01'/name,out/('live-'+name))
readme=f'''# 第三轮结果增量部署完成

入口：[第三轮 · 嘿嘿释义与语境](https://hsd.fenglin.pro/#round=hehe&request=hsc-Q01-D03&mode=mean&parts=parts&ceiling=0.01)。当前发布 `incremental-20260919-02`。

新增Q01、Q02、Q03与原释义、普通义、双义并列共9个条件；原12+88条件保留，总计109条件。双义的原释义、普通义和分段边界分别显示，支持全部36层、32头、36层等权平均、质量/密度、条件与位置差分、SVG导出以及完整报告和数据下载。继续使用既有登录。

发布清单SHA256：`{receipt['manifest_sha256']}`。仅替换Nginx的hsd静态root并平滑reload，当前配置SHA256：`{receipt['new_config_sha256']}`。249个受保护文件、两份审核会话、原认证、审核/PDF进程启动时间和重启次数以及PDF HTTPS响应在前后严格一致。

7,680个旧文件通过硬链接原样继承，363项新增资产及gzip通过校验，上传仅191,016,960字节。独立CPU核对109份完整输入/分数、1,289,088个层平均聚合值；新增1,435,392个token平均值核对通过。本地和真实HTTPS浏览器各检查13,042个显示值，并通过跨轮次切换、全部9条件、组件NA、差分、查询深链接、下载与手机布局。

科学输入、参考标签、封存FP64注意力、原评分与全部旧实验均未改动，没有新GPU运行。层平均不重新归一化；NA不是零；注意力不表示标签方向或因果贡献。新一轮是两个暴露来源和一条AI衍生文本，不能当作独立确认集。

线上浏览器首段完成10,648个数值后，在旧条件冷启动深链接遇到180秒网络超时，无页面异常或数值失败。原记录完整保留；独立续验脚本将网络等待上限设为600秒，保持数值断言不变，补齐深链接/下载和余下2,394个数值。完整线上记录合并两段通过的检查，并与本地完整检查序列逐项核对。

上线后临时明文凭据已删除，本机预览已停止；只清理本次已展开且已验证的重复上传tar包。旧发布、旧文件及所有本地归档保留。

回退目标为 `incremental-20260919-01`。先核对当前配置摘要匹配本次记录，再恢复本目录或远端 `/opt/hsd-case-attention/deployments/incremental-20260919-02/previous-nginx.conf`，执行 `nginx -t` 并reload Nginx。只回退页面，不回滚审核记录、不重启审核/PDF服务，不重跑实验。

实现见 `../../incremental-hehe-v1/README.md`，所有执行/审计/浏览器来源由本目录的closeout.json与deployment-source-pins.json固定。
'''
(out/'README.md').write_text(readme,encoding='utf-8')
closeout={'status':'complete','url':'https://hsd.fenglin.pro/','release':receipt['release'],
 'release_manifest_sha256':receipt['manifest_sha256'],'config_sha256':receipt['new_config_sha256'],
 'rounds':{'original':12,'content_replacement':88,'hehe_sense_context':9},'conditions':109,
 'GPU_used':False,'scientific_inputs_scores_and_human_records_changed':False,'protected_services_and_sessions_unchanged':True,
 'temporary_plaintext_credentials_deleted':True,'owned_preview_stopped':True,'uploaded_duplicate_archive_cleaned':True,
 'browser_checked_values_each':read(w/'browser-live-complete-01/browser.json')['checked_values'],
 'continuation_initial_live_seconds':read(w/'browser-live-complete-01/browser.json')['timings']['continuation_initial_seconds'],
 'server_free_bytes_after_cleanup':read(w/'archive-cleanup.json')['disk_free'],
 'source_files':read(w/'deployment-source-pins.json')['files']+[read(w/'browser-continuation-v2-source-pins.json')['file']],
 'artifacts':[info(p) for p in sorted(out.iterdir()) if p.is_file()],
 'at':datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),'at_unix':time.time()}
write(out/'closeout.json',closeout)
selector={'status':'deployed','url':'https://hsd.fenglin.pro/','release':receipt['release'],
 'release_manifest_sha256':receipt['manifest_sha256'],'closeout':'deployments/20260919-02/closeout.json',
 'closeout_sha256':sha(out/'closeout.json'),'previous_selector':'deployments/20260919-02/previous-current.json'}
temp=base/'current.json.hehe.tmp';write(temp,selector);temp.replace(base/'current.json')
print(json.dumps({'status':'deployed_and_verified','selector':selector,'checked_values':closeout['browser_checked_values_each']}))
