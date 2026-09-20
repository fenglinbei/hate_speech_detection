#!/usr/bin/env python3
"""Independently compare the hosted catalogs/aggregates to sealed scientific files."""
import argparse
import hashlib
import json
from pathlib import Path
import re

ROOT=Path(__file__).resolve().parents[4]
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4194304),b''):h.update(b)
    return h.hexdigest()
p=argparse.ArgumentParser();p.add_argument('--public',type=Path,required=True);p.add_argument('--manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
assert not a.output.exists()
manifest=read(a.manifest);prompts=aggregates=0;checks=[]
for key,experiment in [('original','case-attention-v1'),('replacement','case-content-replacement-v1'),('hehe','hehe-sense-context-v1')]:
    result=ROOT/'reviews'/experiment/'results-01'
    original=json.loads(re.search(r'<script id="data" type="application/json">([\s\S]*?)</script>',(result/'index.html').read_text(encoding='utf-8')).group(1))
    hosted=read(a.public/'rounds'/key/'catalog.json')
    for k,v in original.items():
        if k!='requests':assert hosted[k]==v,(key,k)
    assert len(original['requests'])==len(hosted['requests'])
    for old,r in zip(original['requests'],hosted['requests']):
        assert {k:v for k,v in r.items() if k!='hosted'}==old,r['request_id']
        meta=read(a.public/r['hosted']['meta']);raw=read(result/(r['request_id']+'-aggregates.json'))['aggregates']
        assert meta['aggregates']==raw and meta['score']==hosted['scores'][r['request_id']]
        assert len(meta['mean_aggregates'])==len(raw)
        for mean,source in zip(meta['mean_aggregates'],raw):
            assert (mean['role'],mean['span_id'],mean['visible'])==(source['role'],source['span_id'],source['visible'])
            if not source['visible']:assert mean['mass'] is None and mean['density'] is None;continue
            for metric in ['mass','density']:
                for h in range(32):
                    value=sum(source[metric][l][h] for l in range(36))/36
                    assert abs(value-mean[metric][h])<2e-15,(r['request_id'],metric,h)
                    aggregates+=1
        prompts+=1
    checks.append({'round':key,'all_original_catalog_fields_exact':True,'conditions':len(hosted['requests'])})
inherited_checked=0
for f in manifest['inherited_files']:
    p=a.public/f['name'];assert p.stat().st_size==f['bytes'] and sha(p)==f['sha256'],p
    inherited_checked+=1
for i,f in enumerate(manifest['files']):
    p=a.public/f['name'];g=p.with_name(p.name+'.gz')
    assert p.stat().st_size==f['bytes'] and sha(p)==f['sha256'],p
    assert g.stat().st_size==f['gzip_bytes'] and sha(g)==f['gzip_sha256'],g
    if i%500==0:print(json.dumps({'file_checks':i}),flush=True)
receipt={'status':'pass','GPU_used':False,'manifest_sha256':sha(a.manifest),'all_new_hosted_files_checked':len(manifest['files']),'inherited_files_checked':inherited_checked,
    'full_prompt_and_score_records_exact':prompts,'independently_recomputed_layer_mean_aggregate_values':aggregates,'checks':checks}
a.output.write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print(json.dumps(receipt))
