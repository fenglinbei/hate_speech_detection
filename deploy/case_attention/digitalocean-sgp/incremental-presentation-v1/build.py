#!/usr/bin/env python3
"""Add only the 18 sealed presentation/mechanism inputs to the existing hosted release."""
import argparse,base64,errno,gzip,hashlib,json,os,re,shutil,tarfile
from pathlib import Path
import numpy as np
from markdown_it import MarkdownIt
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
PARENT=HERE.parent/'deployments/20260919-02/release-manifest.json'
OLD_PUBLIC=ROOT/'reviews/hehe-sense-context-v1/hosting-01/prepared/public'
REPLACED=('index.html','app.js','catalog.json')
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4194304),b''):h.update(b)
 return h.hexdigest()
def info(p):return {'path':str(p),'bytes':Path(p).stat().st_size,'sha256':sha(p)}
def verify(x):assert Path(x['path']).stat().st_size==x['bytes'] and sha(x['path'])==x['sha256'],x['path']
def write(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x',encoding='utf-8') as f:f.write(json.dumps(v,ensure_ascii=False,separators=(',',':'))+'\n')
def compress(p):
 with p.open('rb') as src,p.with_name(p.name+'.gz').open('xb') as dst:
  with gzip.GzipFile(filename='',fileobj=dst,mode='wb',compresslevel=6,mtime=0) as z:shutil.copyfileobj(src,z,4194304)
def physical(manifest):
 files={x['name']:x for x in manifest.get('inherited_files',[])}
 for x in manifest['files']:
  files[x['name']]={k:x[k] for k in ('name','bytes','sha256')}
  if 'gzip_bytes' in x:files[x['name']+'.gz']={'name':x['name']+'.gz','bytes':x['gzip_bytes'],'sha256':x['gzip_sha256']}
 return files

def build(output):
 assert not output.exists();public=output/'public';public.mkdir(parents=True)
 parent=read(PARENT);assert sha(PARENT)=='f441b364ec767a5aabf8ea2924b50bf84656891c2b0e9e655b664a0dcc679976'
 removed={n+s for n in REPLACED for s in ('','.gz')}
 inherited=[v for k,v in sorted(physical(parent).items()) if k not in removed]
 assets=[];proofs=[];sources=[info(PARENT)]
 def asset(p):
  compress(p);g=p.with_name(p.name+'.gz')
  x={'name':str(p.relative_to(public)),'bytes':p.stat().st_size,'sha256':sha(p),'gzip_bytes':g.stat().st_size,'gzip_sha256':sha(g)}
  assets.append(x);return x
 # Local preview inheritance only. The upload does NOT include these files.
 for item in inherited:
  name=item['name'];target=public/name;target.parent.mkdir(parents=True,exist_ok=True)
  candidates=[OLD_PUBLIC/name,Path('/tmp/hsd-case-attention-publish-20260918-01/public')/name]
  if name=='style.css':candidates.insert(0,HERE/name)
  source=next((p for p in candidates if p.exists() and p.stat().st_size==item['bytes'] and sha(p)==item['sha256']),None)
  if source is not None:
   try:os.link(source,target)
   except OSError as error:
    if error.errno!=errno.EXDEV:raise
    shutil.copyfile(source,target)
  elif name.endswith('.gz') and (public/name[:-3]).exists():compress(public/name[:-3])
  elif name.endswith('.meta.json') and '/' not in name:
   rid=name[:-10];v=read(ROOT/'reviews/case-attention-v1/results-01'/(rid+'.view.json'));raw=base64.b64decode(v.pop('data'));T=v['shape'][-1];stride=32*6*T*4
   v['encoding']='little-endian-float32-layer-files-visualization-only'
   v['layer_files']=[{'file':rid+f'.layer-{i:02d}.f32','bytes':stride,'sha256':hashlib.sha256(raw[i*stride:(i+1)*stride]).hexdigest()} for i in range(36)]
   source_view=ROOT/'reviews/case-attention-v1/results-01'/(rid+'.view.json')
   v['transport_source']={'file':source_view.name,'sha256':sha(source_view),'display_bytes_sha256':hashlib.sha256(raw).hexdigest()}
   write(target,v)
  else:raise AssertionError('Cannot reconstruct exact inherited local asset: '+name)
  assert target.stat().st_size==item['bytes'] and sha(target)==item['sha256'],name
 print(json.dumps({'inherited_local_files_verified':len(inherited)}),flush=True)
 result=ROOT/'reviews/hehe-presentation-mechanism-v1/results-01';prepared=result.parent/'prepared-01';report=result.parent/'report-01'
 selector=read(ROOT/'docs/research/experiment-plans/hehe-presentation-mechanism-v1/results-current.json');verify(selector['closeout_manifest'])
 manifest=read(result/'manifest.json');expected={Path(x['path']).name:x for x in manifest['artifacts']};verify(expected['index.html'])
 data=json.loads(re.search(r'<script id="data" type="application/json">([\s\S]*?)</script>',(result/'index.html').read_text()).group(1))
 sources.extend([info(result/'manifest.json'),selector['closeout_manifest'],info(report/'manifest.json')])
 title='第四轮 · 释义顺序与机制（18 条件）';key='presentation';folder=public/'rounds'/key;folder.mkdir(parents=True)
 names={'D01':'原释义','D02':'普通义','D03':'分行原义→普通义','D04':'分行普通义→原义','D05':'整句原义→普通义','D06':'整句普通义→原义'}
 for r in data['requests']:
  rid=r['request_id'];viewfile=result/(rid+'.view.json');verify(expected[viewfile.name]);view=read(viewfile)
  raw=base64.b64decode(view.pop('data'));T=r['prompt_tokens'];stride=32*6*T*4
  assert view['shape']==[36,32,6,T] and len(raw)==36*stride
  layer_files=[]
  for layer in range(36):
   p=folder/(rid+f'.layer-{layer:02d}.f32');p.write_bytes(raw[layer*stride:(layer+1)*stride]);asset(p)
   layer_files.append({'file':p.name,'bytes':p.stat().st_size,'sha256':sha(p)})
  source=view['source'];verify(source);attention=np.load(source['path'],allow_pickle=False)
  assert attention.dtype==np.float64 and attention.shape==tuple(view['shape']) and attention.astype('<f4').tobytes()==raw
  mean=attention.mean(axis=0,dtype=np.float64);independent=np.zeros_like(mean)
  for layer in attention:independent+=layer/36
  error=float(np.max(np.abs(mean-independent)));assert error<2e-15
  p=folder/(rid+'.mean.f64');p.write_bytes(mean.astype('<f8').tobytes());asset(p)
  mean_file={'file':p.name,'bytes':p.stat().st_size,'sha256':sha(p),'encoding':'little-endian-float64','shape':[32,6,T]}
  ma=[];maximum=0.;spans={s['id']:s for s in r['spans']}
  for a in view['aggregates']:
   if not a['visible']:ma.append({'role':a['role'],'span_id':a['span_id'],'visible':False,'mass':None,'density':None});continue
   mass=np.array(a['mass']).mean(axis=0);density=np.array(a['density']).mean(axis=0);ri=data['role_order'].index(a['role'])
   check=mean[:,ri,spans[a['span_id']]['token_positions']].sum(axis=1)
   e=max(float(abs(check-mass).max()),float(abs(check/a['mean_visible_tokens']-density).max()));maximum=max(maximum,e);assert e<2e-12
   ma.append(dict(a,mass=mass.tolist(),density=density.tolist()))
  view.update(encoding='little-endian-float32-layer-files-visualization-only',layer_files=layer_files,mean_file=mean_file,mean_aggregates=ma,
       mean_definition='Equal arithmetic mean of all36 layers; heads and readout roles remain separate.',
       transport_source={'file':viewfile.name,'sha256':expected[viewfile.name]['sha256'],'display_bytes_sha256':hashlib.sha256(raw).hexdigest()})
  p=folder/(rid+'.meta.json');write(p,view);asset(p)
  r['hosted']={'meta':str(p.relative_to(public)),'module':'dictionary','item_id':r['dictionary_id'],'title':names[r['dictionary_id']],
       'display_label':r['dictionary_id']+' · '+names[r['dictionary_id']],'baseline':f"hpm-{r['query_id']}-D01",'variant':None}
  proofs.append({'request_id':rid,'source':expected[viewfile.name],'raw_attention':source,'layer_bytes_exact':True,'scores_and_aggregates_exact':True,
       'mean_elements':int(mean.size),'independent_mean_max_error':error,'mean_aggregate_max_error':maximum})
  print(json.dumps({'completed':rid,'mean_error':error}),flush=True)
  del raw,attention,view,mean,independent
 data.update(round_key=key,title=title,report_url='reports/presentation/',mechanism_url='reports/presentation/mechanism.html',scope_note='3条查询 × 6种释义，共18个输入、24个注册比较；新增9个呈现输入，原9输入补采机制。',
       science_scope='两个已暴露真实来源和一条已采用的AI衍生文本；条件相互依赖，非独立确认。')
 write(folder/'catalog.json',data);asset(folder/'catalog.json')
 directory=read(OLD_PUBLIC/'catalog.json');assert [(x['key'],x['conditions']) for x in directory['rounds']]==[('original',12),('replacement',88),('hehe',9)]
 rounds=directory['rounds']+[{'key':key,'label':title,'catalog':'rounds/presentation/catalog.json','conditions':18}]
 directory.update(rounds=rounds,version='incremental-presentation-v1',science_scope='四轮探索性实验，来源与条件相互依赖；各轮参考与分数独立保留。')
 write(public/'catalog.json',directory);asset(public/'catalog.json')
 for name in ('index.html','app.js'):shutil.copyfile(HERE/name,public/name);asset(public/name)
 destination=public/'reports/presentation';destination.mkdir(parents=True)
 for item in read(report/'manifest.json')['artifacts']:verify(item)
 figures=['all-outputs','layer-probes','source-outputs']
 copies=[(report/name,name) for name in ['REPORT.md','all-outputs.tsv','all-comparisons.tsv','source-outputs.tsv','metrics.json']+[f+ext for f in figures for ext in ('.png','.pdf','.svg')]]
 copies += [(prepared/'ALL-PROMPTS.md','ALL-PROMPTS.md'),(result/'scores.json','scores.json'),(result/'comparisons.json','comparisons.json'),(result/'mechanism-summary.json','mechanism-summary.json'),
       (result.parent/'audits/results-01.json','audit.json'),(result.parent/'audits/viewer-results-01/audit.json','browser-audit.json')]
 for source,name in copies:shutil.copyfile(source,destination/name);asset(destination/name);sources.append(info(source))
 mechanism_source=result/'mechanism.html';mechanism=mechanism_source.read_text().replace('href="index.html"','href="../../#round=presentation&amp;request=hpm-Q01-D04&amp;mode=mean&amp;parts=parts"').replace('href="REPORT.md"','href="./"')
 (destination/'mechanism.html').write_text(mechanism);asset(destination/'mechanism.html');sources.append(info(mechanism_source))
 md=(report/'REPORT.md').read_text().replace(str(report)+'/','').replace(str(prepared/'ALL-PROMPTS.md'),'ALL-PROMPTS.md')
 for source,target in [(result/'index.html','../../#round=presentation&request=hpm-Q01-D04&mode=mean&parts=parts'),(result/'scores.json','scores.json'),(result/'comparisons.json','comparisons.json'),(result/'mechanism-summary.json','mechanism-summary.json'),(result.parent/'audits/results-01.json','audit.json'),(result.parent/'audits/viewer-results-01/audit.json','browser-audit.json')]:md=md.replace(str(source),target)
 md=md.replace('](index.html)','](../../#round=presentation&request=hpm-Q01-D04&mode=mean&parts=parts)')
 body=MarkdownIt('commonmark').enable('table').render(md);assert not re.search(r'(?:href|src)="/data/',body)
 header='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>释义顺序与机制 · 完整报告</title><link rel="stylesheet" href="../../style.css"><style>article{max-width:1150px;margin:25px auto;padding:25px;background:white}article img{max-width:100%;height:auto}article table{border-collapse:collapse;display:block;overflow:auto}article td,article th{border:1px solid #d8e3e7;padding:8px}</style><article><a href="../../#round=presentation">← 返回本轮交互页面</a>'
 tail='<p>PDF：'+' · '.join('<a href="'+n+'.pdf">'+n+'</a>' for n in figures)+'</p></article></html>'
 (destination/'index.html').write_text(header+body+tail);asset(destination/'index.html')
 write(output/'mean-audit.json',{'status':'pass','GPU_used':False,'new_conditions':18,'retained_conditions':109,'mean_axes':'layer only;36 equal weights','proofs':proofs})
 manifest={'schema':'hsd-incremental-attention/v2','parent_release':'incremental-20260919-02','parent_manifest_sha256':sha(PARENT),
    'expected_config_sha256':'09f4b0e53933031752badeca3774c5c7a166fb03e97009320b4260f57f60af40','rounds':rounds,
    'replaced_assets':list(REPLACED),'inherited_files':inherited,'files':assets,'existing_scientific_values_changed':False,
    'new_statistics':'FP64 equal arithmetic mean of36 sealed attention layers for18 new inputs only',
    'sources':sources+[info(HERE/n) for n in ('build.py','index.html','app.js','style.css')],'mean_audit':info(output/'mean-audit.json')}
 write(output/'release-manifest.json',manifest)
 with tarfile.open(output/'delta.tar','w') as tar:
  for name in ('release-manifest.json','mean-audit.json'):tar.add(output/name,arcname=name)
  for f in assets:tar.add(public/(f['name']+'.gz'),arcname='public/'+f['name']+'.gz')
 package={'archive':info(output/'delta.tar'),'manifest':info(output/'release-manifest.json'),'new_public_bytes':sum(x['bytes']+x['gzip_bytes'] for x in assets),'inherited_data_reuploaded':False,'inherited_files':len(inherited)}
 write(output/'package.json',package);print(json.dumps(package),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();build(a.output.resolve())
