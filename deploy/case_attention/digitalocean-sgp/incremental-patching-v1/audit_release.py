#!/usr/bin/env python3
"""Read-only inherited hashes, exact scientific payload and hosted asset audit."""
import argparse,gzip,hashlib,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4]
def read(p):return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4194304),b''):h.update(b)
    return h.hexdigest()
def main(output):
    output=Path(output);m=read(output/'release-manifest.json');public=output/'public'
    for r in m['sources']:assert Path(r['path']).stat().st_size==r['bytes'] and sha(r['path'])==r['sha256']
    for r in m['inherited_files']:
        p=public/r['name'];assert p.stat().st_size==r['bytes'] and sha(p)==r['sha256']
    for r in m['files']:
        p=public/r['name'];g=public/(r['name']+'.gz')
        assert p.stat().st_size==r['bytes'] and sha(p)==r['sha256'] and g.stat().st_size==r['gzip_bytes'] and sha(g)==r['gzip_sha256']
        assert gzip.decompress(g.read_bytes())==p.read_bytes()
    work=ROOT/'reviews/hehe-focal-patching-v1';science=read(work/'results-01/results.json');hosted=read(public/'reports/patching/results.json')
    assert hosted==science
    embedded=json.loads(re.search(r'<script id="data" type="application/json">([\s\S]*?)</script>',(public/'reports/patching/index.html').read_text()).group(1))
    assert embedded==science and len(science['effects'])==288 and len(science['self_controls'])==288 and len(science['position_differences'])==144
    for name,source in [('index.html',work/'results-01/index.html'),('all-interventions.tsv',work/'results-01/all-interventions.tsv'),('REPORT.md',work/'report-01/REPORT.md')]:
        assert (public/'reports/patching'/name).read_bytes()==source.read_bytes()
    old=ROOT/'reviews/hehe-presentation-mechanism-v1/hosting-01/prepared/public';new=(public/'index.html').read_text()
    addition='<p class="patchingEntry" style="margin:0;padding:12px 24px;background:#e6f1f2"><a id="patching" href="reports/patching/">新结果：查询“嘿嘿”表示替换 · 全36层与前置位置对照 →</a></p>\n'
    assert new.replace(addition,'')==(old/'index.html').read_text() and new.count(addition)==1
    assert read(public/'catalog.json')==read(old/'catalog.json') and sum(r['conditions'] for r in m['rounds'])==127
    result={'status':'pass','inherited_files_checked':len(m['inherited_files']),'new_assets_and_gzip_checked':len(m['files']),
            'old_conditions_exact':127,'new_payload_exact':True,'native_inputs':4,'cross_patches':288,'self_controls':288,'position_differences':144,
            'old_js_catalog_css_unchanged':True,'root_change':'one additive navigation link only','GPU_used':False}
    with (output.parent/'release-audit.json').open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepared',type=Path,required=True);a=p.parse_args();main(a.prepared)
