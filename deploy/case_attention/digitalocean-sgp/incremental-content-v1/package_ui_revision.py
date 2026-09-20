#!/usr/bin/env python3
"""Package a verified UI-only refinement while the large immutable payload uploads."""
import argparse
import importlib.util
from pathlib import Path
import tarfile

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('builder',HERE/'build.py');b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)
p=argparse.ArgumentParser();p.add_argument('--prepared',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
assert not a.output.exists();a.output.mkdir(parents=True)
m=b.read(a.prepared/'release-manifest.json');changed=[]
for name in ['index.html','app.js','style.css']:
    record=next(x for x in m['files'] if x['name']==name);source=HERE/name;target=a.prepared/'public'/name
    if b.sha(source)==record['sha256']:continue
    assert source.read_bytes()==target.read_bytes(),'The refined UI must already be browser-checked.'
    changed.append(name);b.compress(target);gz=target.with_name(name+'.gz')
    record.update(bytes=target.stat().st_size,sha256=b.sha(target),gzip_bytes=gz.stat().st_size,gzip_sha256=b.sha(gz))
    s=next(x for x in m['sources'] if x['path']==str(source));s.update(b.info(source))
m['ui_revision']={'payload_manifest_sha256':b.sha(a.prepared/'release-manifest.json'),'changed_assets':changed,
    'reason':'Browser-verified responsive overflow fix and native hidden-attribute handling; scientific assets and values unchanged.'}
assert changed
b.write(a.output/'release-manifest.json',m)
with tarfile.open(a.output/'ui-patch.tar','w') as tar:
    tar.add(a.output/'release-manifest.json',arcname='release-manifest.json')
    for name in changed:tar.add(a.prepared/'public'/(name+'.gz'),arcname='public/'+name+'.gz')
b.write(a.output/'package.json',{'archive':b.info(a.output/'ui-patch.tar'),'manifest':b.info(a.output/'release-manifest.json'),'payload_manifest_sha256':m['ui_revision']['payload_manifest_sha256']})
print((a.output/'package.json').read_text())
