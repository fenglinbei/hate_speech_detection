#!/usr/bin/env python3
"""Inventory this authorized result archive without modifying frozen artifacts."""
from pathlib import Path
import hashlib,json,os,re,subprocess,collections
ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent
BASE='04c77ccf0cc378c6c7ae34a3b1816faa0a662c20'
REVIEW_NAMES=['case-attention-v1','case-content-replacement-v1','hehe-sense-context-v1',
 'hehe-presentation-mechanism-v1','hehe-focal-patching-v1','hehe-gap-patch-interpretation-v1',
 'hehe-bridge-v1','hehe-branch-restore-v1','hehe-joint-restore-v1','hehe-transfer-candidates-v1',
 'hehe-transfer-v1','cross-term-mechanism-candidates-v1','cross-term-mechanism-v1',
 'jingba-context-candidates-v1','jingba-context-v1','jingba-attn-restore-v1']
SCIENTIFIC_NAMES=[n for n in REVIEW_NAMES if 'candidates' not in n and n!='hehe-gap-patch-interpretation-v1']
TOOL_NAMES=['case_attention_viewer_v1','case_content_replacement_viewer_v1','hehe_focal_patch_viewer_v1','hehe_presentation_viewer_v1','hehe_sense_context_viewer_v1']

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
    roots += [ROOT/'deploy/case_attention']+[ROOT/'tools'/n for n in TOOL_NAMES]
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
        'scope':'12 completed experiments, adopted/draft provenance, CPU interpretations, implementations, viewer/deployment evidence; not a complete runtime backup.',
        'completed_experiments':SCIENTIFIC_NAMES,'files':files,'excluded_by_policy':excluded,
        'excluded_also':'Untracked historical exps/ runtimes already outside prior archive policy; model/environment/cache/private review sessions; not enumerated here.',
        'frozen_bytes_modified':False,'new_GPU_forwards':0}
    with (OUT/'manifest.json').open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps({'files':len(files),'bytes':sum(r['bytes'] for r in files),'excluded':excluded},ensure_ascii=False))
