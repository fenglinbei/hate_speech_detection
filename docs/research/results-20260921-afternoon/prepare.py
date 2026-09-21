#!/usr/bin/env python3
"""Inventory this authorized result archive without modifying frozen artifacts."""
from pathlib import Path
import hashlib,json,os,re,subprocess,collections
ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent
BASE='4c02aa79e884f22b0f2c4536084f830b37541436'
SCIENTIFIC_NAMES=['jingba-query-scope-v1','jingba-query-complement-v1','jingba-query-regions-v1','dictionary-free-donor-v1','jingba-mixed-demos-v1','jingba-demo-donor-v1']
REVIEW_NAMES=SCIENTIFIC_NAMES+['autonomous-reference-progress-20260921']

def reason(path):
    parts=path.parts;name=path.name
    if path.is_symlink():return 'symlink'
    if path.suffix in ['.log','.lock','.npy','.npz','.tar','.gz','.tgz','.sqlite3','.pyc','.pid'] or name in ['STOP','CANCEL'] or any(x in parts for x in ['.locks','__pycache__','node_modules','.git']):return 'raw_binary_log_control'
    if name.endswith('.view.json') or name.endswith('-aggregates.json'):return 'dense_attention_display'
    if any(re.fullmatch('run-[0-9]+',x) for x in parts) and any(x in parts for x in ['records','arrays','format','format-arrays','invocations']):return 'per_forward_raw'
    if any(x.startswith('hosting-') for x in parts) and any(x in ['prepared','runtime'] or x.startswith('build-attempt-') for x in parts):return 'hosting_bundle_or_private_runtime'
    if any(x.startswith('synthetic-') for x in parts) and name not in ['README.md','synthetic-audit.json']:return 'synthetic_fixture'
    if any(x in parts for x in ['runtime','private']):return 'private_runtime'
    if re.search(r'(?:^|[-_.])(session|login|credentials|htpasswd|cookies)(?:[-_.]|$)',name,re.I):return 'private_state_or_credentials'
    return None

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def inventory():
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()==BASE,'Archive base changed'
    roots=[ROOT/'reviews'/n for n in REVIEW_NAMES]
    roots += [ROOT/'docs/research/experiment-plans'/n for n in REVIEW_NAMES if (ROOT/'docs/research/experiment-plans'/n).exists()]
    candidates={};excluded=collections.defaultdict(lambda:{'files':0,'bytes':0})
    def add(p):
        rel=p.relative_to(ROOT);why=reason(rel)
        if why:excluded[why]['files']+=1;excluded[why]['bytes']+=p.stat().st_size
        else:candidates[str(rel)]=p
    for folder in roots:
        for p in folder.rglob('*'):
            if p.is_file():add(p)
    # All new implementation files are in these two directories; no old runtime
    # leftovers under exps/ are pulled into the result archive.
    for folder in ['src/diagnostics','scripts/review']:
        files=subprocess.check_output(['git','ls-files','--others','--exclude-standard','-z','--',folder],cwd=ROOT).split(b'\0')
        for raw in files:
            if raw:add(ROOT/os.fsdecode(raw))
    for name in ['AGENTS.md','.gitattributes','docs/research/README.md']:candidates[name]=ROOT/name
    records=[]
    for rel,p in sorted(candidates.items()):
        assert p.is_file() and not p.is_symlink(),rel
        n=p.stat().st_size;assert n<100*1024*1024,(rel,'GitHub individual-file limit')
        records.append(dict(path=rel,bytes=n,sha256=sha(p)))
    return records,dict(excluded)

if __name__=='__main__':
    files,excluded=inventory()
    result={'schema_version':'research-result-archive/v1','base_commit':BASE,'archive_date':'2026-09-21',
        'scope':'Six completed experiments (3510 real forwards), immutable inputs/results/audits, fixed-rule and mixed-demo CPU interpretations and implementations; not a complete runtime backup.',
        'completed_experiments':SCIENTIFIC_NAMES,'files':files,'excluded_by_policy':excluded,
        'excluded_also':'Untracked historical exps/ runtimes already outside prior archive policy; model/environment/cache/private review sessions; not enumerated here.',
        'frozen_bytes_modified':False,'new_GPU_forwards':0}
    with (OUT/'manifest.json').open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps({'files':len(files),'bytes':sum(r['bytes'] for r in files),'excluded':excluded},ensure_ascii=False))
