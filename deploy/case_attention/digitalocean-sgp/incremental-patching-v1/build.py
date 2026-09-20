#!/usr/bin/env python3
"""Inherit every old scientific asset and add the sealed stage2 report."""
import argparse,errno,gzip,hashlib,json,os,re,shutil,tarfile
from pathlib import Path
from markdown_it import MarkdownIt
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
WORK=ROOT/'reviews/hehe-focal-patching-v1';PARENT=HERE.parent/'deployments/20260919-03/release-manifest.json'
OLD_PUBLIC=ROOT/'reviews/hehe-presentation-mechanism-v1/hosting-01/prepared/public'
def read(p):return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4194304),b''):h.update(b)
    return h.hexdigest()
def info(p):return dict(path=str(Path(p).absolute()),bytes=Path(p).stat().st_size,sha256=sha(p))
def verify(r):assert Path(r['path']).stat().st_size==r['bytes'] and sha(r['path'])==r['sha256'],r['path']
def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:json.dump(v,f,ensure_ascii=False,separators=(',',':'),allow_nan=False);f.write('\n')
def physical(manifest):
    out={r['name']:r for r in manifest.get('inherited_files',[])}
    for r in manifest['files']:
        out[r['name']]={k:r[k] for k in ('name','bytes','sha256')}
        out[r['name']+'.gz']={'name':r['name']+'.gz','bytes':r['gzip_bytes'],'sha256':r['gzip_sha256']}
    return out
def build(output):
    output=Path(output).absolute();assert not output.exists();public=output/'public';public.mkdir(parents=True)
    assert sha(PARENT)=='acfe6e0c51bfa7ea8e5629143ed68e9e224dc3318f78fea8396e47e7725a5a7b';parent=read(PARENT)
    selector=read(ROOT/'docs/research/experiment-plans/hehe-focal-patching-v1/results-current.json');verify(selector['closeout_manifest'])
    result=WORK/'results-01';report=WORK/'report-01';prepared=WORK/'prepared-01'
    for name in [result/'manifest.json',report/'manifest.json']:
        for r in read(name)['artifacts']:verify(r)
    assert read(WORK/'audits/results-01.json')['status']=='pass' and read(WORK/'audits/browser-results-01.json')['status']=='pass'
    inherited=[v for k,v in sorted(physical(parent).items()) if k not in ['index.html','index.html.gz']]
    for r in inherited:
        src=OLD_PUBLIC/r['name'];assert src.stat().st_size==r['bytes'] and sha(src)==r['sha256']
        target=public/r['name'];target.parent.mkdir(parents=True,exist_ok=True)
        try:os.link(src,target)
        except OSError as e:
            if e.errno!=errno.EXDEV:raise
            shutil.copyfile(src,target)
    print(json.dumps({'inherited_files':len(inherited)}),flush=True)
    files=[];sources=[info(PARENT),selector['closeout_manifest'],info(result/'manifest.json'),info(report/'manifest.json')]
    def asset(p):
        g=p.with_name(p.name+'.gz')
        with p.open('rb') as src,g.open('xb') as dst:
            with gzip.GzipFile(filename='',fileobj=dst,mode='wb',compresslevel=6,mtime=0) as z:shutil.copyfileobj(src,z,4194304)
        files.append(dict(name=str(p.relative_to(public)),bytes=p.stat().st_size,sha256=sha(p),gzip_bytes=g.stat().st_size,gzip_sha256=sha(g)))
    shutil.copyfile(HERE/'index.html',public/'index.html');asset(public/'index.html')
    dest=public/'reports/patching';dest.mkdir(parents=True)
    copies=[(result/'index.html','index.html'),(result/'results.json','results.json'),(result/'all-interventions.tsv','all-interventions.tsv'),
            (report/'REPORT.md','REPORT.md'),(report/'metrics.json','metrics.json'),(prepared/'ALL-PROMPTS.md','ALL-PROMPTS.md'),
            (prepared/'PROTOCOL.md','PROTOCOL.md'),(prepared/'interventions.json','interventions.json'),
            (WORK/'audits/results-01.json','audit.json'),(WORK/'audits/browser-results-01.json','browser-audit.json')]
    copies += [(report/('patch-effects'+ext),'patch-effects'+ext) for ext in ('.png','.svg','.pdf')]
    for src,name in copies:shutil.copyfile(src,dest/name);asset(dest/name);sources.append(info(src))
    md=(report/'REPORT.md').read_text();html=MarkdownIt('commonmark').enable('table').render(md)
    assert not re.search(r'(?:href|src)="/data/',html)
    header='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>查询表示替换 · 报告</title><style>body{font:16px/1.7 system-ui;background:#f3f6f9;color:#15364c}article{max-width:1100px;margin:auto;padding:25px;background:white}img{max-width:100%;height:auto}table{display:block;overflow:auto;border-collapse:collapse}td,th{border:1px solid #d8e3e7;padding:8px}a{color:#007c85}</style><article><a href="./">← 逐层交互图</a> · <a href="/">注意力实验首页</a>'
    (dest/'REPORT.html').write_text(header+html+'</article></html>');asset(dest/'REPORT.html')
    write(output/'asset-audit.json',{'status':'pass','GPU_used':False,'old_conditions':127,'old_scientific_files_byte_identical':True,
                                  'new_native_inputs':4,'new_cross_patches':288,'new_self_controls':288,'old_means_changed':False,
                                  'scientific_audit':info(WORK/'audits/results-01.json'),'browser_audit':info(WORK/'audits/browser-results-01.json')})
    sources += [info(HERE/x) for x in ('build.py','deploy.py','index.html','audit_release.py','check_browser.cjs')]
    manifest={'schema':'hsd-incremental-attention/v2','parent_release':'incremental-20260919-03','parent_manifest_sha256':sha(PARENT),
              'expected_config_sha256':'3148d7abd1ba3a3ec395e2fe300293b3436cb6a47cc9fb2c5b48f1d4f3ee5cc5',
              'rounds':parent['rounds'],'supplemental_analysis':{'key':'patching','native_prompts':4,'cross_patches':288,'self_controls':288},
              'replaced_assets':['index.html'],'inherited_files':inherited,'files':files,'existing_scientific_values_changed':False,
              'new_statistics':'Sealed complete-vector activation patching results; no new attention arrays or re-normalization',
              'sources':sources,'asset_audit':info(output/'asset-audit.json')}
    write(output/'release-manifest.json',manifest)
    with tarfile.open(output/'delta.tar','w') as tar:
        for name in ('release-manifest.json','asset-audit.json'):tar.add(output/name,arcname=name)
        for f in files:tar.add(public/(f['name']+'.gz'),arcname='public/'+f['name']+'.gz')
    package={'archive':info(output/'delta.tar'),'manifest':info(output/'release-manifest.json'),'new_public_bytes':sum(r['bytes']+r['gzip_bytes'] for r in files),'inherited_data_reuploaded':False,'inherited_files':len(inherited)}
    write(output/'package.json',package);print(json.dumps(package),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();build(a.output)
