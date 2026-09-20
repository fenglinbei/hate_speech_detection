#!/usr/bin/env python3
"""Build an additive hosted release and exact FP64 layer means, without GPU use."""
import argparse
import base64
import gzip
import hashlib
import json
from pathlib import Path
import re
import shutil
import tarfile
import numpy as np
from markdown_it import MarkdownIt

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
PARENT=HERE.parent/'deployments/20260918-01/layer-release-manifest-v2.json'

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4194304),b''):h.update(b)
    return h.hexdigest()

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(v,ensure_ascii=False,separators=(',',':'))+'\n',encoding='utf-8')
def verify(x):assert Path(x['path']).stat().st_size==x['bytes'] and sha(x['path'])==x['sha256'],x['path']
def info(p):return {'path':str(p),'bytes':Path(p).stat().st_size,'sha256':sha(p)}
def compress(p):
    with p.open('rb') as src,p.with_name(p.name+'.gz').open('wb') as f:
        with gzip.GzipFile(filename='',fileobj=f,mode='wb',compresslevel=6,mtime=0) as z:shutil.copyfileobj(src,z,4194304)

def build(output):
    assert not output.exists();public=output/'public';public.mkdir(parents=True)
    parent=read(PARENT);oldfiles={f['name']:f for f in parent['files']}
    assets=[];proofs=[];sources=[];rounds=[]
    def asset(p):
        compress(p);g=p.with_name(p.name+'.gz')
        record={'name':str(p.relative_to(public)),'bytes':p.stat().st_size,'sha256':sha(p),'gzip_bytes':g.stat().st_size,'gzip_sha256':sha(g)}
        assets.append(record);return record
    for key,experiment,title in [('original','case-attention-v1','首轮 · 原始注意力（12 条件）'),('replacement','case-content-replacement-v1','本轮 · 内容替换（88 条件）')]:
        result=ROOT/'reviews'/experiment/'results-01';selector=read(ROOT/'docs/research/experiment-plans'/experiment/'results-current.json')
        verify(selector['closeout_manifest']);manifest=read(result/'manifest.json');expected={Path(x['path']).name:x for x in manifest['artifacts']}
        verify(expected['index.html']);html=(result/'index.html').read_text(encoding='utf-8')
        data=json.loads(re.search(r'<script id="data" type="application/json">([\s\S]*?)</script>',html).group(1))
        sources.extend([info(result/'manifest.json'),selector['closeout_manifest']])
        ledger={};proposals={}
        if key=='replacement':
            prepared=ROOT/'reviews'/experiment/'prepared-01'
            ledger={r['request_id']:r for r in read(prepared/'input-ledger.json')['inputs']}
            proposals={p['item_id']:p for p in read(prepared/'material-proposals.json')['items']}
        for r in data['requests']:
            rid=r['request_id'];viewfile=result/(rid+'.view.json');verify(expected[viewfile.name]);view=read(viewfile)
            raw=base64.b64decode(view.pop('data'));T=r['prompt_tokens'];stride=32*6*T*4
            assert view['shape']==[36,32,6,T] and len(raw)==36*stride
            folder=public/'rounds'/key;folder.mkdir(parents=True,exist_ok=True)
            layer_files=[]
            for l in range(36):
                chunk=raw[l*stride:(l+1)*stride];name=rid+f'.layer-{l:02d}.f32';digest=hashlib.sha256(chunk).hexdigest()
                if key=='original':
                    assert digest==oldfiles[name]['sha256'] and len(chunk)==oldfiles[name]['bytes']
                    # Local preview copies only; deployment inherits these exact files.
                    path=public/name;path.write_bytes(chunk);url='../../'+name
                else:
                    path=folder/name;path.write_bytes(chunk);asset(path);url=name
                layer_files.append({'file':url,'bytes':len(chunk),'sha256':digest})
            source=view['source'];verify(source);attention=np.load(source['path'],allow_pickle=False)
            assert attention.dtype==np.float64 and attention.shape==tuple(view['shape'])
            assert np.asarray(attention,dtype='<f4').tobytes()==raw,'Published layer bytes differ from sealed display'
            mean=attention.mean(axis=0,dtype=np.float64)
            meanpath=folder/(rid+'.mean.f64');meanpath.write_bytes(np.asarray(mean,dtype='<f8').tobytes());asset(meanpath)
            # Independently stream the 36 source layers; check all derived entries.
            independent=np.zeros_like(mean)
            for layer in attention:independent+=layer/36
            error=float(np.max(np.abs(mean-independent)));assert error<2e-15
            mean_aggregates=[];aggregate_error=0.0
            spans={s['id']:s for s in r['spans']}
            for a in view['aggregates']:
                if not a['visible']:
                    mean_aggregates.append({'role':a['role'],'span_id':a['span_id'],'visible':False,'mass':None,'density':None});continue
                ri=data['role_order'].index(a['role']);keys=spans[a['span_id']]['token_positions']
                mass=np.asarray(a['mass'],dtype=np.float64).mean(axis=0)
                density=np.asarray(a['density'],dtype=np.float64).mean(axis=0)
                check=np.take(mean[:,ri,:],keys,axis=1).sum(axis=1,dtype=np.float64)
                e=max(float(np.max(np.abs(check-mass))),float(np.max(np.abs(check/a['mean_visible_tokens']-density))))
                aggregate_error=max(aggregate_error,e);assert e<2e-12
                mean_aggregates.append({'role':a['role'],'span_id':a['span_id'],'visible':True,'mass':mass.tolist(),'density':density.tolist(),'mean_visible_tokens':a['mean_visible_tokens'],'mass_error_bound':a['mass_error_bound'],'density_error_bound':a['density_error_bound']})
            view.update(encoding='little-endian-float32-layer-files-visualization-only',layer_files=layer_files,
                mean_file={'file':meanpath.name,'bytes':meanpath.stat().st_size,'sha256':sha(meanpath),'encoding':'little-endian-float64','shape':[32,6,T]},
                mean_aggregates=mean_aggregates,mean_definition='Equal arithmetic mean of all 36 layers; heads and readout roles remain separate.',
                transport_source={'file':viewfile.name,'sha256':expected[viewfile.name]['sha256'],'display_bytes_sha256':hashlib.sha256(raw).hexdigest()})
            meta=folder/(rid+'.meta.json');write(meta,view);asset(meta)
            roundtrip=read(meta)
            for field in ('aggregates','score','shape','source','request_id','prompt_sha256'):assert roundtrip[field]==view[field]
            d=ledger.get(rid,{});item=proposals.get(d.get('item_id'),{})
            r['hosted']={'meta':str(meta.relative_to(public)),'module':d.get('module','baseline'),'item_id':d.get('item_id'),'title':item.get('title','原始条件'), 'baseline':d.get('baseline_request_id'), 'variant':d.get('variant')}
            proofs.append({'request_id':rid,'source':expected[viewfile.name], 'raw_attention':source,'layer_bytes_exact':True,'aggregates_and_scores_exact':True,
                'mean_sha256':sha(meanpath),'mean_elements':int(mean.size),'independent_mean_max_error':error,'mean_aggregate_max_error':aggregate_error})
            print(json.dumps({'round':key,'completed':rid,'mean_error':error}),flush=True)
            del attention,raw,view,roundtrip,independent,mean
        data['round_key']=key;data['title']=title
        catalog=public/'rounds'/key/'catalog.json';write(catalog,data);asset(catalog)
        rounds.append({'key':key,'label':title,'catalog':str(catalog.relative_to(public)),'conditions':len(data['requests'])})
    write(public/'catalog.json',{'rounds':rounds,'default_round':'original','version':'incremental-content-v1','science_scope':'两个已暴露案例；比较共享端点，非独立确认。'});asset(public/'catalog.json')
    for name in ('index.html','app.js','style.css'):
        shutil.copyfile(HERE/name,public/name);asset(public/name)
    report=ROOT/'reviews/case-content-replacement-v1/report-02'
    report_manifest=read(report/'manifest.json');sources.append(info(report/'manifest.json'))
    for record in report_manifest['artifacts']:verify(record)
    figures=['definition-replacements','answer-replacements','body-answer-crosses','order-attention']
    for name in ['REPORT.md','all-inputs.tsv']+[f+ext for f in figures for ext in ['.png','.pdf']]:
        target=public/'reports'/name;target.parent.mkdir(exist_ok=True);shutil.copyfile(report/name,target);asset(target)
    for original,name in [(ROOT/'reviews/case-content-replacement-v1/results-01/comparisons.json','comparisons.json'),(ROOT/'reviews/case-content-replacement-v1/audits/results-01.json','audit.json')]:
        target=public/'reports'/name;shutil.copyfile(original,target);asset(target);sources.append(info(original))
    # Render a hosted copy with working links; preserve the original Markdown bytes.
    md=(report/'REPORT.md').read_text(encoding='utf-8').replace('../results-01/index.html','../#round=replacement').replace('../results-01/comparisons.json','comparisons.json').replace('../audits/results-01.json','audit.json')
    body=MarkdownIt('commonmark').enable('table').render(md)
    report_html='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>内容替换实验 · 完整报告</title><link rel="stylesheet" href="../style.css"><style>article{max-width:1150px;margin:25px auto;padding:25px;background:white}article img{max-width:100%;height:auto}article table{border-collapse:collapse;display:block;overflow:auto}article td,article th{border:1px solid #d8e3e7;padding:8px}article h2{margin-top:35px}article h3{margin-top:25px}</style><article><a href="../#round=replacement">← 返回交互页面</a>'+body+'<p>PDF：'+' · '.join('<a href="'+f+'.pdf">'+f+'</a>' for f in figures)+'</p></article></html>'
    (public/'reports/index.html').write_text(report_html,encoding='utf-8');asset(public/'reports/index.html')
    write(output/'mean-audit.json',{'status':'pass','GPU_used':False,'conditions':100,'mean_axes':'layer only; 36 equal weights','proofs':proofs})
    manifest={'schema':'hsd-incremental-attention/v1','parent_release':'layers-20260918-03','parent_manifest_sha256':sha(PARENT),
        'expected_config_sha256':'33f1293645c4ac2fc7d9f1ed2a2fab0befe69effc490740405b69db48fef7b20','rounds':rounds,
        'inherited_files':[x for x in parent['files'] if x['name'] not in ('index.html','index.html.gz')], 'files':assets,
        'existing_scientific_values_changed':False,'new_statistics':'FP64 equal arithmetic mean of 36 sealed attention layers',
        'sources':sources+[info(HERE/n) for n in ('build.py','index.html','app.js','style.css')], 'mean_audit':info(output/'mean-audit.json')}
    write(output/'release-manifest.json',manifest)
    with tarfile.open(output/'delta.tar','w') as tar:
        tar.add(output/'release-manifest.json',arcname='release-manifest.json')
        tar.add(output/'mean-audit.json',arcname='mean-audit.json')
        for f in assets:tar.add(public/(f['name']+'.gz'),arcname='public/'+f['name']+'.gz')
    package={'archive':info(output/'delta.tar'),'manifest':info(output/'release-manifest.json'),'new_public_bytes':sum(f['bytes']+f['gzip_bytes'] for f in assets),'inherited_data_reuploaded':False}
    write(output/'package.json',package);print(json.dumps(package),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();build(a.output.resolve())
