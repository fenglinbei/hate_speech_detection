"""Independent read-only CPU audit of the candidate package and token pairs."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import runpy
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'src'))
from diagnostics.case_attention_inputs_v1 import tokenizer

def need(ok,msg):
    if not ok:raise ValueError(msg)
def read(p):return json.loads(p.read_text())
def sha(raw):return hashlib.sha256(raw).hexdigest()

def audit(folder):
    tok=tokenizer()
    materials=read(folder/'materials.json');queries=materials['queries'];demos=materials['demos']
    mats={m['material_id']:m for m in queries+demos}
    need(len(mats)==156,'material identity count')
    for m in mats.values():
        need(sha(m['raw_text'].encode())==m['text_sha256'],'material digest')
        need(m['human_reference'] is None and not m['accepted_fields'] and not m['adopted_fields'] and m['decision_ref'] is None,'invented human adoption')
        need(not m['eligible_for_model_execution'],'draft must not be execution-ready')
        for field in ('focal_occurrence_spans','author_stance_spans','quotation_spans','independent_attack_spans'):
            for span in m[field]:need(m['raw_text'][span['start']:span['end']]==span['text'],'raw span mismatch')
    for split in ('development','confirmation'):
        q=[x for x in queries if x['split']==split]
        need(len(q)==24 and Counter(x['proposed_answer'] for x in q)=={'有':12,'无':12},'split and intended label balance')
        need(Counter(x['stratum'] for x in q)=={'ordinary_clean':6,'ordinary_other_attack':6,'quoted_opposition':6,'endorsed_attack':6},'stratum balance')
    need(len({q['slot_id'] for q in queries})==48,'unique frozen slots')
    system=(ROOT/'docs/research/experiment-plans/task-applicability-scoring-v1/frozen-01/model-task.txt').read_text()
    records=[json.loads(line) for line in (folder/'model-inputs-preview.jsonl').read_text().splitlines()]
    by={(r['query_id'],r['condition']):r for r in records}
    need(len(by)==len(records)==672,'complete prompt identity matrix')
    boundary_checks=0;pairchecks=0;querymaps=0
    for r in records:
        q=mats[r['query_id']];ds=[mats[x] for x in r['demo_ids']]
        # Render independently of prepare.py and parent renderer.
        demo='\n\n'.join(f'示例{i}：\n文本：{d["raw_text"]}\n答案：{d["proposed_answer"]}' for i,d in enumerate(ds,1)) or '（无）'
        user=f'【参考词典】\n（无）\n\n【参考示例】\n{demo}\n\n【待判断文本】\n{q["raw_text"]}'
        text=tok.apply_chat_template([{'role':'system','content':system},{'role':'user','content':user}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
        ids=tok.encode(text,add_special_tokens=False)
        need(text==r['prompt_text'] and sha(text.encode())==r['prompt_sha256'],'exact prompt reconstruction')
        need(ids==r['input_ids'] and len(ids)==r['prompt_tokens'],'exact token reconstruction')
        need(sha(json.dumps(ids,separators=(',',':')).encode())==r['input_ids_sha256'],'token digest')
        need(r['pre_answer']==len(ids)-1,'answer position')
        for label,tid in [('有',18830),('无',42192)]:
            need(tok.encode(text+label,add_special_tokens=False)==ids+[tid],'answer token boundary');boundary_checks+=1
        need('human_reference' not in r and 'proposed_answer' not in r and not r['eligible_for_model_execution'],'preview leaks query reference or claims ready')
        need(tok.decode([ids[i] for i in r['query_focal_positions']])==q['focal_form'],'focal token identity')
        donor=by[q['material_id'],'M00']
        need([ids[i] for i in r['query_positions']]==[donor['input_ids'][i] for i in donor['query_positions']],'same-query token alignment');querymaps+=1
        if ds:need([d['proposed_answer'] for d in ds]==['无','有','无','有'],'label slot invariant')
    for q in queries:
        qid=q['material_id'];local=[r for r in records if r['query_id']==qid and r['condition']!='M00']
        need(len(local)==13 and len({len(r['input_ids']) for r in local})==1,'reference lengths')
        for order in ('MPS','MSP'):
            a=by[qid,order]
            for k in range(4):
                b=by[qid,f'{order}_replace_{k+1}']
                need([i for i,(x,y) in enumerate(zip(a['demo_ids'],b['demo_ids'])) if x!=y]==[k],'only one demo identity changes')
                x=a['demo_body_spans'][k];y=b['demo_body_spans'][k]
                lo,hi=x['token_start'],x['token_end']
                need((lo,hi)==(y['token_start'],y['token_end']),'equal body token interval')
                need(a['input_ids'][:lo]==b['input_ids'][:lo] and a['input_ids'][hi:]==b['input_ids'][hi:],'unchanged positions/ids')
                pairchecks+=1
    from jsonschema import Draft202012Validator,FormatChecker
    parent=ROOT/'docs/research/experiment-plans/task-applicability-scoring-v1/frozen-01'
    schema=Draft202012Validator(read(parent/'relation-record.schema.json'),format_checker=FormatChecker())
    validator=runpy.run_path(str(parent/'validate_contract.py'))['validate_relation']
    ai=read(folder/'relations-ai.json')['records'];blank=read(folder/'relations-unreviewed.json')['records']
    need(len(ai)==len(blank)==576,'relation count')
    seen=set()
    for r,b in zip(ai,blank):
        validator(r,schema);validator(b,schema)
        need(r['relation_id']==b['relation_id'] and r['relation_id'] not in seen,'relation identity');seen.add(r['relation_id'])
        for item in (r,b):
            need(item['provenance']['adoption']=='none' and not item['provenance']['accepted_fields'],'relation falsely adopted')
            for side in ('source','target'):
                source=mats[item[side]['material_id']]
                need(item[side]['raw_text']==source['raw_text'] and item[side]['text_sha256']==source['text_sha256'],'relation text identity')
        need(b['source_quality']['value'] is None and b['semantic_reference_fit']['value'] is None and b['rule_fit']['value'] is None,'blank human relation filled')
        for ref in r['presentation_refs']:
            rr=next(x for x in records if x['request_id']==ref['condition_id'])
            need(ref['prompt_sha256']==rr['prompt_sha256'] and ref['source_present']==(r['source']['material_id'] in rr['demo_ids']),'presentation identity')
    review=read(folder/'review-decisions-blank.json')
    need(len(review['records'])==732 and all(x['status']=='unreviewed' and not x['accepted_fields'] and x['decision_ref'] is None for x in review['records']),'human queue is not empty')
    need('torch' not in sys.modules,'audit imported real model backend')
    result={'status':'PASS','queries':48,'unique_demos':108,'relation_AI_and_blank_schema_checks':1152,'exact_prompt_reconstructions':672,
      'candidate_boundary_checks':boundary_checks,'query_donor_token_alignments':querymaps,'single_slot_pairs':pairchecks,
      'human_adoptions':0,'real_model_forwards':0,'CPU_only':True,'model_backend_imported':False}
    manifest=folder/'manifest.json'
    if manifest.exists():
        m=read(manifest);count=0
        for section in ('artifacts','sources','protected_selectors'):
            for item in m[section]:
                p=ROOT/item['path'];need(p.stat().st_size==item['bytes'] and sha(p.read_bytes())==item['sha256'],'manifest changed '+item['path']);count+=1
        result['pins_checked']=count
        expected={x['path'] for x in m['artifacts']}
        actual={str(x.relative_to(ROOT)) for x in folder.iterdir() if x.is_file() and x.name!='manifest.json'}
        need(actual==expected,'closed draft artifact inventory')
    return result

if __name__=='__main__':print(json.dumps(audit(HERE/sys.argv[1]),ensure_ascii=False,indent=2))
