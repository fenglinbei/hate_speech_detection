"""Compare existing local hashes to public Hub metadata; uploads no local data."""
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'docs/research/experiment-plans/cross-model-applicability-v1/draft-01'
if (OUT/'manifest.json').exists():
    raise SystemExit('Delivered review draft is immutable; use a separately versioned revision.')
inventory = json.loads((OUT/'cpu-model-inventory.json').read_text())
records = []
for model in inventory['models']:
    repo = model['official_model_id']
    url = 'https://huggingface.co/api/models/'+repo+'?blobs=true'
    request = urllib.request.Request(url, headers={'User-Agent':'CPU-model-identity-audit/1'})
    with urllib.request.urlopen(request, timeout=15) as response:
        raw = response.read(2_000_001)
    assert len(raw) <= 2_000_000
    data = json.loads(raw)
    assert isinstance(data.get('sha'), str) and len(data['sha']) == 40
    siblings = {s['rfilename']:s for s in data['siblings']}
    checks=[]
    for local in model['weight_sources']+model['metadata_sources']:
        name = Path(local['path']).name
        remote = siblings.get(name)
        if remote is None:
            checks.append({'filename':name,'status':'not_in_current_upstream','local_sha256':local['sha256']})
            continue
        if remote.get('lfs'):
            expected = remote['lfs']['sha256']
            actual = local['sha256']
            hash_type = 'sha256'
        else:
            content = Path(local['path']).read_bytes()
            actual = hashlib.sha1(b'blob '+str(len(content)).encode()+b'\0'+content).hexdigest()
            expected = remote.get('blobId')
            hash_type = 'git_blob_sha1'
        checks.append({'filename':name,'hash_type':hash_type,'local_digest':actual,'upstream_digest':expected,
                       'status':'match' if actual==expected else 'different'})
    weight_names={Path(x['path']).name for x in model['weight_sources']}
    weight_match=all(x['status']=='match' for x in checks if x['filename'] in weight_names)
    record={'model_key':model['model_key'],'repository_id':data['id'],'commit':data['sha'],
            'public_metadata_url':url,'metadata_response_sha256':hashlib.sha256(raw).hexdigest(),
            'all_weights_match':weight_match,'checks':checks}
    records.append(record)
    print(json.dumps({'model_key':model['model_key'],'commit':data['sha'],'all_weights_match':weight_match,
                      'other_differences':[x['filename'] for x in checks if x['status']!='match']},ensure_ascii=False),flush=True)
result={'status':'all_weight_sources_match' if all(x['all_weights_match'] for x in records) else 'requires_identity_review',
        'models':records,'model_forward_calls':0,'local_data_uploaded':False}
(OUT/'upstream-verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
