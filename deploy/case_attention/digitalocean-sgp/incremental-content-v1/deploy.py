#!/usr/bin/env python3
"""Stage an additive static release, then switch only the existing hsd root.

Credentials are accepted on stdin for activation, never persisted. Old releases,
authentication, both review sessions and the PDF application are preserved.
"""
import argparse
import fcntl
import gzip
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
import tarfile
import time

BASE=Path('/opt/hsd-case-attention')
SITE=Path('/etc/nginx/sites-available/hsd.fenglin.pro')
AUTH=Path('/etc/nginx/.htpasswd-hsd-case-attention')
LOCK=Path('/var/lib/hsd-general-model-paired-review/.static-release.lock')

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4194304),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def write(p,data,mode=0o600):
    with Path(p).open('xb') as f:os.fchmod(f.fileno(),mode);f.write(data)
def save(p,data):write(p,(json.dumps(data,ensure_ascii=False,indent=2)+'\n').encode())
def verify(p,record,compressed=False):
    assert p.stat().st_size==record['gzip_bytes' if compressed else 'bytes'] and sha(p)==record['gzip_sha256' if compressed else 'sha256'],str(p)
def relative(name):
    p=PurePosixPath(name)
    assert not p.is_absolute() and all(x not in ('..','.') for x in p.parts) and str(p)==name
    assert re.fullmatch(r'[A-Za-z0-9_./-]+',name)
    return name
def service(name):
    keys=['MainPID','ExecMainStartTimestamp','NRestarts','ActiveState','SubState']
    t=subprocess.check_output(['systemctl','show',name,*['--property='+k for k in keys]],text=True)
    return dict(x.split('=',1) for x in t.splitlines())
def pdf_http():
    result=subprocess.run(['curl','--silent','--show-error','--noproxy','*','--max-time','30','--resolve','pdf.fenglin.pro:443:127.0.0.1','--write-out','\n%{http_code}','https://pdf.fenglin.pro/'],capture_output=True,check=True)
    body,status=result.stdout.rsplit(b'\n',1)
    return {'status':int(status),'body_sha256':hashlib.sha256(body).hexdigest(),'body_bytes':len(body)}
def protected():
    files=[Path('/etc/nginx/conf.d/pdf-translate-reader.conf'),Path('/etc/systemd/system/pdf-translate-reader.service'),
        Path('/etc/systemd/system/hsd-general-model-paired-review.service'),Path('/etc/nginx/.htpasswd-hsd-review'),AUTH,
        Path('/var/lib/hsd-general-model-paired-review/session.json'),Path('/var/lib/hsd-general-model-paired-review/evidence-applicability-v1/session.json')]
    files += [p for p in sorted(Path('/var/www/pdf-translate-reader').rglob('*')) if p.is_file()]
    return {'files':{str(p):sha(p) for p in files},'pdf':service('pdf-translate-reader.service'),'review':service('hsd-general-model-paired-review.service'),
        'review_release':str(Path('/opt/hsd-general-model-paired-review/current').resolve()),'pdf_https':pdf_http()}
def request(path,credential=None,gzip_ok=False):
    config=''
    if credential:
        assert re.fullmatch(r'[A-Za-z0-9_-]+',credential['username']) and re.fullmatch(r'[A-Za-z0-9]+',credential['password'])
        config+='user = "'+credential['username']+':'+credential['password']+'"\n'
    if gzip_ok:config+='header = "Accept-Encoding: gzip"\n'
    result=subprocess.run(['curl','--silent','--show-error','--noproxy','*','--max-time','60','--resolve','hsd.fenglin.pro:443:127.0.0.1',
        '--config','-','--write-out','\n%{http_code}','https://hsd.fenglin.pro'+path],input=config.encode(),capture_output=True,check=True)
    body,status=result.stdout.rsplit(b'\n',1);return int(status),body
def atomic_config(data):
    temp=SITE.with_name(SITE.name+'.incremental.tmp');write(temp,data,0o644);os.replace(temp,SITE)

def stage(args):
    assert sha(args.archive)==args.archive_sha256
    release=BASE/'releases'/args.name
    assert not release.exists(), 'An existing release must not be overwritten.'
    with tarfile.open(args.archive,'r:') as archive:
        members=archive.getmembers();byname={x.name:x for x in members}
        assert len(byname)==len(members) and all(x.isfile() and not x.issym() and not x.islnk() for x in members)
        payload_manifest_bytes=archive.extractfile(byname['release-manifest.json']).read()
        manifest_bytes=payload_manifest_bytes;manifest=json.loads(manifest_bytes);replacements={}
        if args.ui_patch:
            assert sha(args.ui_patch)==args.ui_patch_sha256
            with tarfile.open(args.ui_patch,'r:') as patch:
                entries=patch.getmembers();assert all(x.isfile() and not x.issym() and not x.islnk() for x in entries)
                replacements={x.name:patch.extractfile(x).read() for x in entries};assert len(entries)==len(replacements)
            manifest_bytes=replacements.pop('release-manifest.json');revised=json.loads(manifest_bytes)
            revision=revised['ui_revision'];assert revision['payload_manifest_sha256']==hashlib.sha256(payload_manifest_bytes).hexdigest()
            names=revision['changed_assets'];assert names and set(names)<=set(['index.html','app.js','style.css'])
            assert set(replacements)=={'public/'+name+'.gz' for name in names}
            assert {k:v for k,v in revised.items() if k not in ['ui_revision','files','sources']}=={k:v for k,v in manifest.items() if k not in ['files','sources']}
            assert len(manifest['files'])==len(revised['files']) and len(manifest['sources'])==len(revised['sources'])
            for old,new in zip(manifest['files'],revised['files']):
                assert old['name']==new['name']
                if old['name'] not in names:assert old==new
            for old,new in zip(manifest['sources'],revised['sources']):
                assert old['path']==new['path']
                if Path(old['path']).name not in names:assert old==new
            manifest=revised
        assert hashlib.sha256(manifest_bytes).hexdigest()==args.manifest_sha256
        assert manifest['schema']=='hsd-incremental-attention/v1' and manifest['existing_scientific_values_changed'] is False
        assert [(r['key'],r['conditions']) for r in manifest['rounds']]==[('original',12),('replacement',88)]
        assert sha(SITE)==manifest['expected_config_sha256']
        parent=BASE/'releases'/manifest['parent_release']
        assert parent.parent==BASE/'releases' and sha(parent/'release-manifest.json')==manifest['parent_manifest_sha256']
        assert ('root '+str(parent/'public')+';') in SITE.read_text()
        inherited={x['name']:x for x in manifest['inherited_files']}
        old={x['name']:x for x in read(parent/'release-manifest.json')['files'] if x['name'] not in ('index.html','index.html.gz')}
        assert inherited==old
        allowed={'release-manifest.json','mean-audit.json'}|{'public/'+relative(x['name'])+'.gz' for x in manifest['files']}
        assert set(byname)==allowed and len(allowed)==len(manifest['files'])+2
        newnames={x['name'] for x in manifest['files']}
        assert not newnames.intersection(inherited)
        required=sum(x['bytes']+x['gzip_bytes'] for x in manifest['files'])
        assert shutil.disk_usage(BASE).free>required+1073741824,'Insufficient staging space with safety reserve.'
        release.mkdir(mode=0o755);(release/'public').mkdir(mode=0o755)
        for member in members:
            path=release/relative(member.name);path.parent.mkdir(parents=True,exist_ok=True,mode=0o755)
            if member.name=='release-manifest.json':write(path,manifest_bytes,0o644)
            elif member.name in replacements:write(path,replacements[member.name],0o644)
            else:
                with archive.extractfile(member) as src,path.open('xb') as dst:shutil.copyfileobj(src,dst,4194304)
            os.chmod(path,0o644)
        if args.ui_patch:write(release/'payload-release-manifest.json',payload_manifest_bytes,0o644)
        verify(release/'mean-audit.json',manifest['mean_audit'])
        assert read(release/'mean-audit.json')['status']=='pass'
        for item in manifest['inherited_files']:
            original=parent/'public'/relative(item['name']);verify(original,item)
            dest=release/'public'/item['name'];dest.parent.mkdir(parents=True,exist_ok=True,mode=0o755);os.link(original,dest)
        for i,item in enumerate(manifest['files']):
            compressed=release/'public'/(item['name']+'.gz');verify(compressed,item,True)
            target=release/'public'/item['name']
            with gzip.open(compressed,'rb') as src,target.open('xb') as dst:shutil.copyfileobj(src,dst,4194304)
            os.chmod(target,0o644);verify(target,item)
            if i%250==0:print(json.dumps({'stage':'verified','new_files':i}),flush=True)
        receipt={'status':'staged','release':str(release),'archive_sha256':args.archive_sha256,'manifest_sha256':args.manifest_sha256,
            'inherited_files':len(inherited),'new_files':len(manifest['files']),'hardlinked_parent':str(parent),'disk_free':shutil.disk_usage(BASE).free,'at_unix':time.time()}
        save(release/'staging-receipt.json',receipt);print(json.dumps(receipt),flush=True)

def activate(args):
    credential=json.load(sys.stdin);assert credential['username']=='liaozijie'
    assert Path('/etc/nginx/sites-enabled/hsd.fenglin.pro').resolve()==SITE
    release=BASE/'releases'/args.name;manifest=read(release/'release-manifest.json')
    assert sha(release/'release-manifest.json')==args.manifest_sha256
    assert read(release/'staging-receipt.json')['manifest_sha256']==args.manifest_sha256
    assert sha(SITE)==manifest['expected_config_sha256']
    # Verify staged bytes again immediately before the switch, including inherited assets.
    for x in manifest['files']:
        verify(release/'public'/x['name'],x);verify(release/'public'/(x['name']+'.gz'),x,True)
    for x in manifest['inherited_files']:verify(release/'public'/x['name'],x)
    before=protected();assert before['pdf']['ActiveState']==before['review']['ActiveState']=='active'
    assert before['pdf_https']['status'] in (200,401)
    original=SITE.read_bytes();expected='root '+str(BASE/'releases'/manifest['parent_release']/'public')+';'
    assert original.decode().count(expected)==1
    revised=original.replace(expected.encode(),('root '+str(release/'public')+';').encode())
    receipt_dir=BASE/'deployments'/args.name;receipt_dir.mkdir(mode=0o700)
    write(receipt_dir/'previous-nginx.conf',original);write(receipt_dir/'new-nginx.conf',revised);save(receipt_dir/'before.json',before)
    switched=False
    try:
        atomic_config(revised);switched=True
        subprocess.run(['nginx','-t'],check=True,capture_output=True)
        subprocess.run(['systemctl','reload','nginx'],check=True,capture_output=True);time.sleep(1)
        files={x['name']:x for x in manifest['files']};checks=[]
        selected=['index.html','app.js','style.css','catalog.json','rounds/original/catalog.json','rounds/replacement/catalog.json','reports/index.html','reports/comparisons.json']
        for key,rid in [('original','case-541-LD'),('original','case-3169-LD'),('replacement','ccr-541-LD-base'),('replacement','ccr-3169-LD-L05'),('replacement','ccr-541-LD-O01')]:
            selected += [f'rounds/{key}/{rid}.meta.json',f'rounds/{key}/{rid}.mean.f64']
        selected += ['rounds/replacement/ccr-3169-LD-L05.layer-29.f32']
        for name in selected:
            path='/'+name;code,_=request(path);assert code==401,(path,code)
            code,data=request(path,credential,True);assert code==200 and hashlib.sha256(data).hexdigest()==files[name]['gzip_sha256'],(path,code)
            code,_=request(path+'.gz');assert code==401
            checks.append({'file':name,'unauthenticated':401,'authenticated':200,'gzip_hash_matches':True})
        code,body=request('/',credential);assert code==200 and hashlib.sha256(body).hexdigest()==files['index.html']['sha256']
        code,_=request('/',{'username':credential['username'],'password':'deliberatelywrong'});assert code==401
        old=next(x for x in manifest['inherited_files'] if x['name']=='case-541-LD.layer-29.f32.gz')
        code,data=request('/case-541-LD.layer-29.f32',credential,True);assert code==200 and hashlib.sha256(data).hexdigest()==old['sha256']
        after=protected();save(receipt_dir/'after.json',after);assert after==before,'Protected service, session or PDF response changed.'
        receipt={'status':'deployed','url':'https://hsd.fenglin.pro/','release':str(release),'manifest_sha256':args.manifest_sha256,
            'new_config_sha256':sha(SITE),'previous_config_sha256':hashlib.sha256(original).hexdigest(),'checks':checks,
            'preserved_auth_and_both_review_sessions':True,'preserved_PDF_config_static_service_HTTPS':True,'at_unix':time.time()}
        save(receipt_dir/'receipt.json',receipt);print(json.dumps(receipt),flush=True)
    except BaseException:
        if switched:
            atomic_config(original);subprocess.run(['nginx','-t'],check=True,capture_output=True);subprocess.run(['systemctl','reload','nginx'],check=True,capture_output=True)
        save(receipt_dir/'failed.json',{'restored_previous_nginx':switched,'at_unix':time.time()});raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['stage','activate']);p.add_argument('--name',required=True);p.add_argument('--manifest-sha256',required=True)
    p.add_argument('--archive',type=Path);p.add_argument('--archive-sha256');p.add_argument('--ui-patch',type=Path);p.add_argument('--ui-patch-sha256');args=p.parse_args()
    assert os.getuid()==0 and re.fullmatch(r'incremental-[0-9]{8}-[0-9]{2}',args.name)
    with LOCK.open('a+b') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        (stage if args.action=='stage' else activate)(args)
