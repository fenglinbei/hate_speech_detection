#!/usr/bin/env python3
"""Activate one verified static release; preserve the live review state/services."""
import argparse
import fcntl
import gzip
import grp
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tarfile
import time


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4194304), b''):
            h.update(block)
    return h.hexdigest()


def write(path, data, mode=0o600):
    path = Path(path)
    with path.open('xb') as f:
        os.fchmod(f.fileno(), mode)
        f.write(data)


def atomic_config(path, data):
    temp = path.with_name(path.name + '.case-attention.tmp')
    write(temp, data, 0o644)
    os.replace(temp, path)


def service(name):
    props = ['MainPID', 'ExecMainStartTimestamp', 'NRestarts', 'ActiveState', 'SubState']
    text = subprocess.check_output(['systemctl','show',name,*['--property='+p for p in props]], text=True)
    return dict(line.split('=',1) for line in text.splitlines())


def protected():
    paths = [Path('/etc/nginx/conf.d/pdf-translate-reader.conf'), Path('/etc/systemd/system/pdf-translate-reader.service')]
    paths += [p for p in sorted(Path('/var/www/pdf-translate-reader').rglob('*')) if p.is_file()]
    paths += [Path('/etc/systemd/system/hsd-general-model-paired-review.service'), Path('/etc/nginx/.htpasswd-hsd-review')]
    return {'files':{str(p):sha(p) for p in paths},'pdf':service('pdf-translate-reader.service'),
            'review':service('hsd-general-model-paired-review.service'),
            'review_release':str(Path('/opt/hsd-general-model-paired-review/current').resolve())}


def request(path, credential=None, gzip_ok=False, method='GET'):
    command=['curl','--silent','--show-error','--noproxy','*','--max-time','60',
             '--resolve','hsd.fenglin.pro:443:127.0.0.1','--write-out','\n%{http_code}',
             '--request',method,'--config','-','https://hsd.fenglin.pro'+path]
    config = ''
    if credential:
        assert re.fullmatch(r'[A-Za-z0-9_-]+',credential['username'])
        assert re.fullmatch(r'[A-Za-z0-9]+',credential['password'])
        config += 'user = "'+credential['username']+':'+credential['password']+'"\n'
    if gzip_ok:
        config += 'header = "Accept-Encoding: gzip"\n'
    result=subprocess.run(command,input=config.encode(),capture_output=True,check=True)
    body, status=result.stdout.rsplit(b'\n',1)
    return int(status),body


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--archive',type=Path,required=True)
    parser.add_argument('--sha256',required=True)
    parser.add_argument('--expected-config-sha256',required=True)
    args=parser.parse_args()
    assert os.getuid()==0
    assert sha(args.archive)==args.sha256
    credential=json.load(sys.stdin)  # Never passed in argv, URLs, manifests or logs.
    assert credential['username']=='liaozijie'
    base=Path('/opt/hsd-case-attention')
    release=base/'releases'/args.sha256
    receipt_dir=base/'deployments'/args.sha256
    site=Path('/etc/nginx/sites-available/hsd.fenglin.pro')
    auth=Path('/etc/nginx/.htpasswd-hsd-case-attention')
    with Path('/var/lib/hsd-general-model-paired-review/.static-release.lock').open('a+b') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert sha(site)==args.expected_config_sha256,'Site changed since inspection'
        assert not release.exists() and not receipt_dir.exists() and not auth.exists()
        assert Path('/etc/nginx/sites-enabled/hsd.fenglin.pro').resolve()==site
        before=protected()
        assert before['review']['ActiveState']==before['pdf']['ActiveState']=='active'
        release.mkdir(parents=True,mode=0o755)
        receipt_dir.mkdir(parents=True,mode=0o700)
        for parent in (base, base/'releases'):
            os.chmod(parent,0o755)
        os.chmod(base/'deployments',0o700)
        with tarfile.open(args.archive,'r:') as archive:
            for member in archive.getmembers():
                assert member.isfile() and not member.issym() and not member.islnk()
                assert member.name=='release-manifest.json' or re.fullmatch(r'public/(?:index\.html|case-(?:541|3169)-(?:C0|D|L|LC|LD|LDC)\.view\.json)\.gz',member.name)
                target=release/member.name
                target.parent.mkdir(exist_ok=True,mode=0o755)
                with archive.extractfile(member) as src,target.open('xb') as dst:
                    shutil.copyfileobj(src,dst,4194304)
                os.chmod(target,0o644)
        manifest=json.loads((release/'release-manifest.json').read_text())
        assert len(manifest['files'])==13 and manifest['scientific_payloads_changed'] is False
        for item in manifest['files']:
            compressed=release/'public'/(item['name']+'.gz')
            target=release/'public'/item['name']
            assert sha(compressed)==item['gzip_sha256']
            with gzip.open(compressed,'rb') as src,target.open('xb') as dst:
                shutil.copyfileobj(src,dst,4194304)
            os.chmod(target,0o644)
            assert target.stat().st_size==item['bytes'] and sha(target)==item['sha256']
            if item['name'].endswith('.view.json'):
                assert item['sha256']==item['source_sha256']
        original=site.read_bytes()
        write(receipt_dir/'previous-nginx.conf',original)
        write(receipt_dir/'before.json',(json.dumps(before,indent=2)+'\n').encode())
        hashed=subprocess.run(['openssl','passwd','-6','-stdin'],input=(credential['password']+'\n').encode(),capture_output=True,check=True).stdout.strip()
        assert hashed.startswith(b'$6$') and b'\n' not in hashed
        write(auth,credential['username'].encode()+b':'+hashed+b'\n',0o640)
        os.chown(auth,0,grp.getgrnam('www-data').gr_gid)
        old=original.decode()
        pattern=r'    location / \{\n        proxy_pass http://127\.0\.0\.1:8772;[\s\S]*?\n    \}'
        replacement='    root '+str(release/'public')+';\n    index index.html;\n    charset utf-8;\n    gzip_static on;\n    gzip_vary on;\n    add_header Cache-Control "private, no-cache" always;\n\n    location / {\n        limit_except GET HEAD { deny all; }\n        try_files $uri $uri/ =404;\n    }'
        new,count=re.subn(pattern,lambda _:replacement,old)
        assert count==1
        assert new.count('auth_basic_user_file /etc/nginx/.htpasswd-hsd-review;')==1
        new=new.replace('auth_basic_user_file /etc/nginx/.htpasswd-hsd-review;','auth_basic_user_file '+str(auth)+';')
        new=new.replace('auth_basic "HSD human review";','auth_basic "HSD attention visualization";')
        write(receipt_dir/'new-nginx.conf',new.encode())
        switched=False
        try:
            atomic_config(site,new.encode()); switched=True
            subprocess.run(['nginx','-t'],check=True,capture_output=True)
            subprocess.run(['systemctl','reload','nginx'],check=True,capture_output=True)
            time.sleep(1)
            tests=[]
            for path in ('/','/case-541-LD.view.json','/case-3169-LDC.view.json.gz'):
                status,_=request(path)
                assert status==401,(path,status)
                tests.append({'path':path,'without_login':status})
            status,_=request('/',{'username':credential['username'],'password':'deliberatelywrong'})
            assert status==401
            for item in manifest['files']:
                path='/' if item['name']=='index.html' else '/'+item['name']
                status,body=request(path,credential,gzip_ok=True)
                assert status==200 and hashlib.sha256(body).hexdigest()==item['gzip_sha256'],(path,status)
                tests.append({'path':path,'authenticated_status':status,'compressed_payload_hash_matches':True})
            status,body=request('/',credential)
            assert status==200 and hashlib.sha256(body).hexdigest()==manifest['files'][0]['sha256']
            assert protected()==before,'Unrelated service or historical deployment changed'
            receipt={'status':'deployed','url':'https://hsd.fenglin.pro/','at_unix':time.time(),'archive_sha256':args.sha256,
                     'source_manifest_sha256':manifest['source_results_manifest_sha256'],'release':str(release),
                     'previous_config':str(receipt_dir/'previous-nginx.conf'),'new_config_sha256':sha(site),
                     'login_username':credential['username'],'scientific_payloads_changed':False,
                     'historical_review_and_pdf_services_unchanged':True,'checks':tests}
            write(receipt_dir/'receipt.json',(json.dumps(receipt,indent=2)+'\n').encode())
            print(json.dumps(receipt,indent=2))
        except BaseException:
            if switched:
                atomic_config(site,original)
                subprocess.run(['nginx','-t'],check=True,capture_output=True)
                subprocess.run(['systemctl','reload','nginx'],check=True,capture_output=True)
            write(receipt_dir/'failed.json',json.dumps({'status':'failed','restored_previous_nginx':switched,'at_unix':time.time()}).encode())
            raise


if __name__=='__main__':
    main()
