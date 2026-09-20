#!/usr/bin/env python3
"""Keep matrix-click navigation consistent with lazy layer loading."""
import argparse
import json
import os
from pathlib import Path
from prepare_layer_release import sha,compressed


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--source',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();assert not args.output.exists()
    manifest=json.loads((args.source/'release-manifest.json').read_text())
    args.output.mkdir(mode=0o755);public=args.output/'public';public.mkdir()
    for item in manifest['files']:
        p=args.source/'public'/item['name'];assert sha(p)==item['sha256']
        if p.name not in ('index.html','index.html.gz'):os.link(p,public/p.name)
    html=(args.source/'public/index.html').read_text()
    old="$('layer').value=l;render()";assert html.count(old)==2
    html=html.replace(old,"$('layer').value=l;render();$('load').click()")
    (public/'index.html').write_text(html);compressed(public/'index.html')
    manifest['parent_manifest_sha256']=sha(args.source/'release-manifest.json')
    manifest['data_preparer_sha256']=manifest['script_sha256'];manifest['script_sha256']=sha(Path(__file__))
    manifest['additional_hosted_change']='Matrix cell clicks load the selected layer before token coloring; all data unchanged.'
    manifest['files']=[{'name':p.name,'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(public.iterdir())]
    (args.output/'release-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'status':'prepared','manifest_sha256':sha(args.output/'release-manifest.json')}))


if __name__=='__main__':main()
