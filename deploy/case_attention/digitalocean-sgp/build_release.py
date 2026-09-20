#!/usr/bin/env python3
"""Package the immutable results with a small, separately versioned hosted UI."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import tarfile

ROOT = Path(__file__).resolve().parents[3]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4194304), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(mode=0o700, parents=True, exist_ok=False)
    public = args.output / 'public'
    public.mkdir()
    source = ROOT / 'reviews/case-attention-v1/results-01'
    selector = json.loads((ROOT / 'docs/research/experiment-plans/case-attention-v1/results-current.json').read_text())
    assert sha(selector['closeout_manifest']['path']) == selector['closeout_manifest']['sha256']
    manifest = json.loads((source / 'manifest.json').read_text())
    expected = {Path(x['path']).name: x for x in manifest['artifacts']}
    files = [source / 'index.html'] + sorted(source.glob('*.view.json'))
    assert len(files) == 13
    records = []
    for path in files:
        assert sha(path) == expected[path.name]['sha256'], path.name
        raw = path.read_bytes() if path.name == 'index.html' else None
        target = public / path.name
        if raw is not None:
            html = raw.decode('utf-8')
            old = '<label>加载结果文件（可多选）<input id="files" type="file" accept=".json" multiple></label>'
            assert html.count(old) == 1
            html = html.replace(old, old.replace('<label>', '<label style="display:none">'))
            old = '选择运行后生成的 *.view.json。也可用本地 HTTP 服务打开结果目录，按条件自动加载。原始 FP64 文件和汇总表用于精确复核。'
            assert html.count(old) == 1
            html = html.replace(old, '结果会随条件自动加载，无需选择本地文件。首次加载请稍候；切换层、注意力头和读取位置可查看不同读数。')
            assert html.count('async function load(r){') == 1
            html = html.replace('async function load(r){', 'async function loadUncached(r){')
            marker = "$('files').addEventListener"
            assert html.count(marker) == 1
            loader = "const pendingLoads=new Map();async function load(r){if(cache.has(r.request_id))return;if(pendingLoads.has(r.request_id))return pendingLoads.get(r.request_id);const promise=loadUncached(r).finally(()=>pendingLoads.delete(r.request_id));pendingLoads.set(r.request_id,promise);return promise;}\n"
            html = html.replace(marker, loader + marker)
            start = html.index("$('load').onclick=async()=>")
            end = html.index('\nfunction row(', start)
            html = html[:start] + "let loadSequence=0;$('load').onclick=async()=>{const sequence=++loadSequence,current=req(),comparison=$('compare').value?req($('compare').value):null;$('message').textContent='正在加载注意力数据，请稍候…';try{await Promise.all([load(current),...(comparison?[load(comparison)]:[])]);if(sequence===loadSequence){$('message').textContent='结果已加载';render()}}catch(e){if(sequence===loadSequence)$('message').textContent='加载失败，请点击加载按钮重试：'+e.message}};" + html[end:]
            ending = 'render();\n</script></body></html>'
            assert html.count(ending) == 1
            html = html.replace(ending, "render();\nif(location.protocol!=='file:')$('load').click();\n</script></body></html>")
            target.write_text(html, encoding='utf-8')
        else:
            shutil.copyfile(path, target)
        with target.open('rb') as src, (public / (path.name + '.gz')).open('wb') as out:
            with gzip.GzipFile(filename='', fileobj=out, mode='wb', compresslevel=6, mtime=0) as dst:
                shutil.copyfileobj(src, dst, 4194304)
        records.append({'name':path.name, 'bytes':target.stat().st_size, 'sha256':sha(target),
                        'source_sha256':expected[path.name]['sha256'],
                        'gzip_bytes':(public / (path.name+'.gz')).stat().st_size,
                        'gzip_sha256':sha(public / (path.name+'.gz'))})
        print(json.dumps({'file':path.name,'compressed_bytes':records[-1]['gzip_bytes']}), flush=True)
    release = {'schema':'hsd-case-attention-static-deployment/v1','source_results_manifest_sha256':sha(source/'manifest.json'),
               'source_closeout_manifest_sha256':selector['closeout_manifest']['sha256'],
               'source_builder_sha256':sha(Path(__file__)), 'files':records,
               'hosted_changes':['automatic initial load','online help text','loading feedback','deduplicated concurrent fetches'],
               'scientific_payloads_changed':False}
    (args.output / 'release-manifest.json').write_text(json.dumps(release,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    # Transfer only compressed payloads. Activation verifies and streams them
    # into exact uncompressed files, preserving gzip companions for nginx.
    with tarfile.open(args.output/'release.tar', 'w') as archive:
        archive.add(args.output/'release-manifest.json', arcname='release-manifest.json')
        for path in sorted(public.glob('*.gz')):
            archive.add(path, arcname='public/'+path.name)
    result={'archive':str(args.output/'release.tar'),'sha256':sha(args.output/'release.tar'),
            'bytes':(args.output/'release.tar').stat().st_size,'manifest_sha256':sha(args.output/'release-manifest.json')}
    (args.output/'package.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__ == '__main__':
    main()
