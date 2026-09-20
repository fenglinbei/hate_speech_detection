#!/usr/bin/env python3
"""Lossless hosted transport: unchanged aggregates, one FP32 layer per request."""
import argparse
import base64
import gzip
import hashlib
import json
import os
from pathlib import Path
import re


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(4194304),b''):h.update(b)
    return h.hexdigest()


def compressed(path):
    with (path.with_name(path.name+'.gz')).open('wb') as f:
        with gzip.GzipFile(filename='',fileobj=f,mode='wb',compresslevel=6,mtime=0) as z:
            z.write(path.read_bytes())


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--source',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();assert not args.output.exists()
    args.output.mkdir(parents=True,mode=0o755);public=args.output/'public';public.mkdir()
    original=json.loads((args.source/'release-manifest.json').read_text())
    for item in original['files']:
        p=args.source/'public'/item['name'];assert sha(p)==item['sha256']
        if p.name!='index.html':
            os.link(p,public/p.name);os.link(p.with_name(p.name+'.gz'),public/(p.name+'.gz'))
    rows=[]
    for item in original['files']:
        if not item['name'].endswith('.view.json'):continue
        p=json.loads((public/item['name']).read_text());raw=base64.b64decode(p.pop('data'))
        shape=p['shape'];assert shape[:3]==[36,32,6];stride=32*6*shape[3]*4;assert len(raw)==36*stride
        p['encoding']='little-endian-float32-layer-files-visualization-only';p['layer_files']=[]
        combined=hashlib.sha256()
        for layer in range(36):
            name=p['request_id']+f'.layer-{layer:02d}.f32';chunk=raw[layer*stride:(layer+1)*stride]
            target=public/name;target.write_bytes(chunk);combined.update(chunk);compressed(target)
            p['layer_files'].append({'file':name,'bytes':len(chunk),'sha256':sha(target)})
        assert combined.hexdigest()==hashlib.sha256(raw).hexdigest()
        p['transport_source']={'file':item['name'],'sha256':item['sha256'],'display_bytes_sha256':combined.hexdigest()}
        meta=public/(p['request_id']+'.meta.json');meta.write_text(json.dumps(p,ensure_ascii=False,separators=(',',':'))+'\n');compressed(meta)
        # JSON roundtrip must keep every original FP64 aggregate and score exact.
        check=json.loads(meta.read_text())
        for key in ('aggregates','score','shape','source','request_id','prompt_sha256'):
            assert check[key]==p[key],key
        rows.append({'request_id':p['request_id'],'source_sha256':item['sha256'],'display_bytes_sha256':combined.hexdigest(),
                     'metadata_sha256':sha(meta),'metadata_gzip_bytes':meta.with_name(meta.name+'.gz').stat().st_size})
        print(json.dumps({'prepared':p['request_id'],'metadata_gzip_bytes':rows[-1]['metadata_gzip_bytes']}),flush=True)
        del p,check,raw
    html=(args.source/'public/index.html').read_text()
    start=html.index('async function loadUncached(r){');end=html.index('\nconst pendingLoads=',start)
    loader="""async function loadUncached(r){if(cache.has(r.request_id))return;const response=await fetch(r.request_id+'.meta.json');if(!response.ok)throw Error('该条件尚无结果文件');const p=await response.json();if(p.request_id!==r.request_id||p.prompt_sha256!==r.prompt_sha256||p.shape.join(',')!==[36,32,6,r.prompt_tokens].join(',')||p.encoding!=='little-endian-float32-layer-files-visualization-only')throw Error('结果与冻结输入不一致');p.values=new Float32Array(p.shape.reduce((a,b)=>a*b,1));p.loadedLayers=new Set();cache.set(p.request_id,p);}
const pendingLayers=new Map();async function loadLayer(r,layer){await load(r);const p=cache.get(r.request_id);if(p.loadedLayers.has(layer))return;const key=r.request_id+':'+layer;if(pendingLayers.has(key))return pendingLayers.get(key);const promise=(async()=>{const f=p.layer_files[layer],response=await fetch(f.file);if(!response.ok)throw Error('当前层数据加载失败');const bytes=await response.arrayBuffer();if(bytes.byteLength!==f.bytes)throw Error('当前层数据长度不一致');const digest=await crypto.subtle.digest('SHA-256',bytes),hex=Array.from(new Uint8Array(digest),x=>x.toString(16).padStart(2,'0')).join('');if(hex!==f.sha256)throw Error('当前层数据校验失败');p.values.set(new Float32Array(bytes),layer*32*6*r.prompt_tokens);p.loadedLayers.add(layer)})();pendingLayers.set(key,promise);try{await promise}finally{pendingLayers.delete(key)}}"""
    html=html[:start]+loader+html[end:]
    old="await Promise.all([load(current),...(comparison?[load(comparison)]:[])]);"
    assert html.count(old)==1
    html=html.replace(old,"await Promise.all([loadLayer(current,+$('layer').value),...(comparison?[load(comparison)]:[])]);")
    old="weights=new Float64Array(len);if(p){";assert html.count(old)==1
    html=html.replace(old,"weights=new Float64Array(len);if(p&&p.loadedLayers.has(layer)){")
    old="else if(p&&r.roles[role].length)";assert html.count(old)==1
    html=html.replace(old,"else if(p&&p.loadedLayers.has(layer)&&r.roles[role].length)")
    old="loaded:!!p,role,layer,head";assert html.count(old)==1
    html=html.replace(old,"loaded:!!p&&p.loadedLayers.has(layer),role,layer,head")
    (public/'index.html').write_text(html);compressed(public/'index.html')
    # Pin every derived asset; no model outputs or statistical values recomputed.
    manifest={'schema':'hsd-attention-layer-transport/v1','original_manifest_sha256':sha(args.source/'release-manifest.json'),
              'script_sha256':sha(Path(__file__)),'scientific_values_changed':False,'layer_bytes_exact':True,'conditions':rows,
              'files':[{'name':p.name,'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(public.iterdir())]}
    (args.output/'release-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'status':'prepared','manifest_sha256':sha(args.output/'release-manifest.json'),'files':len(manifest['files'])}),flush=True)


if __name__=='__main__':main()
