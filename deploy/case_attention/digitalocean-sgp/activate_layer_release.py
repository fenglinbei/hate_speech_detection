#!/usr/bin/env python3
"""Switch only the static root after validating lossless layer transport."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import time

from activate_release import sha, protected, request, atomic_config, write


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--release',type=Path,required=True);parser.add_argument('--expected-config-sha256',required=True)
    args=parser.parse_args();credential=json.load(sys.stdin)
    assert args.release.parent==Path('/opt/hsd-case-attention/releases')
    receipt_dir=Path('/opt/hsd-case-attention/deployments')/args.release.name
    site=Path('/etc/nginx/sites-available/hsd.fenglin.pro')
    with Path('/var/lib/hsd-general-model-paired-review/.static-release.lock').open('a+b') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert sha(site)==args.expected_config_sha256
        assert not receipt_dir.exists()
        manifest=json.loads((args.release/'release-manifest.json').read_text())
        assert manifest['scientific_values_changed'] is False and manifest['layer_bytes_exact']
        for item in manifest['files']:
            p=args.release/'public'/item['name'];assert p.stat().st_size==item['bytes'] and sha(p)==item['sha256'],p.name
        before=protected();original=site.read_bytes()
        replacement='    root '+str(args.release/'public')+';'
        revised,count=re.subn(r'    root /opt/hsd-case-attention/releases/[^/]+/public;',lambda _:replacement,original.decode())
        assert count==1
        receipt_dir.mkdir(mode=0o700)
        write(receipt_dir/'previous-nginx.conf',original);write(receipt_dir/'new-nginx.conf',revised.encode())
        atomic_config(site,revised.encode())
        try:
            subprocess.run(['nginx','-t'],check=True,capture_output=True)
            subprocess.run(['systemctl','reload','nginx'],check=True,capture_output=True);time.sleep(1)
            files={x['name']:x for x in manifest['files']}
            checks=[]
            for name in ['index.html']+[x['request_id']+'.meta.json' for x in manifest['conditions']]+['case-541-LD.layer-35.f32','case-3169-LD.layer-05.f32','case-3169-LDC.layer-35.f32']:
                path='/' if name=='index.html' else '/'+name
                status,_=request(path);assert status==401
                status,body=request(path,credential,gzip_ok=True)
                assert status==200 and hashlib.sha256(body).hexdigest()==files[name+'.gz']['sha256'],name
                checks.append({'name':name,'authenticated':200,'unauthenticated':401,'payload_matches':True})
            assert protected()==before
            receipt={'status':'deployed','url':'https://hsd.fenglin.pro/','release':str(args.release),
                     'manifest_sha256':sha(args.release/'release-manifest.json'),'new_config_sha256':sha(site),
                     'previous_config':str(receipt_dir/'previous-nginx.conf'),'science_unchanged':True,
                     'other_services_unchanged':True,'checks':checks,'at_unix':time.time()}
            write(receipt_dir/'receipt.json',(json.dumps(receipt,indent=2)+'\n').encode());print(json.dumps(receipt,indent=2))
        except BaseException:
            atomic_config(site,original);subprocess.run(['nginx','-t'],check=True,capture_output=True);subprocess.run(['systemctl','reload','nginx'],check=True,capture_output=True)
            raise


if __name__=='__main__':main()
