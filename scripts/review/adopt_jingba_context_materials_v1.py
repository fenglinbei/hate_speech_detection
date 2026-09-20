#!/usr/bin/env python3
"""Adopt approved exact draft bytes; no tensor or GPU access."""
from copy import deepcopy
from pathlib import Path
import hashlib,json,sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from diagnostics.case_attention_inputs_v1 import read,write,canonical,info,verify,require,atomic,lines
from diagnostics.cross_term_mechanism_materials_v1 import TASK,text_file
DRAFT=ROOT/'reviews/jingba-context-candidates-v1/draft-01'
OUT=DRAFT.parent/'adopted-01'
QUOTE='全部通过，可以开始实验，当前GPU空闲'
def main():
    require(not OUT.exists(),'New adoption only')
    for x in read(DRAFT/'manifest.json')['artifacts']+read(DRAFT/'manifest.json')['sources']:verify(x)
    draft=read(DRAFT/'candidates.json');ids=[f'J{i:02}' for i in range(5,11)]
    require([q['query_id'] for q in draft['queries']]==ids,'Six approved candidates')
    refs=[];materials=[]
    for q in draft['queries']:
        require(hashlib.sha256(q['text'].encode()).hexdigest()==q['text_sha256'],'Draft text hash')
        require(q['proposed_reference']==('无' if int(q['query_id'][1:])<=8 else '有'),'Reference inventory')
        materials.append(dict(query_id=q['query_id'],term='京巴',text=q['text'],role=q['role'],pair_id=q['pair_id'],
            decision='accept',source_id=None,provenance=q,independent_confirmation=False))
        refs.append(dict(query_id=q['query_id'],reference=q['proposed_reference'],source_labels=None,source_id=None,
            reference_basis='user_bulk_adoption_of_proposed_current_task_reference',new_user_message=QUOTE,
            severity=None,new_severity_adjudication=False,reference_is_prompt_content=False,interpretation=q['rationale']))
    dictionaries=[dict(term='京巴',dictionary_id=d['dictionary_id'],adopted_text=d['text'],definition=d['definition'],
        decision='accepted',source_record=d,formatting='Lossless split into 词形/释义 fields.') for d in draft['inherited_dictionaries']]
    rows=[];positions=[]
    for r0 in lines(DRAFT/'model-inputs-preview.jsonl'):
        r=deepcopy(r0);r['request_id']=f'jctx-{r["query_id"]}-{r["dictionary_id"]}'
        require(r.pop('draft_not_run_ready') is True,'Draft origin required')
        r['patch_position_sets']=r.pop('patch_position_preview')
        fp=r['patch_position_sets']['focal'];pre=r['patch_position_sets']['pre'];r['term']='京巴'
        r['capture_positions']=sorted(pre+fp);r['capture_prefix_length']=max(fp)+1
        qs=next(s for s in r['spans'] if s['id']=='query')['char_start']
        r['query_relative_token_offsets']=[[a-qs,b-qs] for a,b in (r['token_offsets'][i] for i in r['roles']['query_all'])]
        require(r['prompt_text']==r0['prompt_text'] and r['input_ids']==r0['input_ids'],'Preview input changed')
        require(not {'reference','gold','hate','human_decision'}&set(r),'No reference in worker inputs')
        rows.append(r);positions.append(dict(request_id=r['request_id'],tokens=r['prompt_tokens'],focal_positions=fp,
            focal_text=[r['token_text'][i] for i in fp],pre_positions=pre,pre_text=[r['token_text'][i] for i in pre],prefix_token_count=max(fp)+1))
    for did in ['D00','D01','D02']:
        a,b=[next(r for r in rows if r['query_id']==q and r['dictionary_id']==did) for q in ['J08','J10']]
        require(a['input_ids'][:a['capture_prefix_length']]==b['input_ids'][:b['capture_prefix_length']],'Paired prefix changed')
    OUT.mkdir(parents=True)
    adoption=dict(schema='jingba-context-adoption/v1',status='all_six_materials_accepted',user_message=QUOTE,
        accepted_items=ids,accepted_terms=['京巴'],pending_material_items=[],GPU_execution_authorized=True,GPU_started=False,
        independent_confirmation=False,source_scope='New assistant-authored texts, not corpus examples',
        position_policy='Equal-count immediately preceding query tokens; not norm/wordclass matched or presumed zero.')
    write(OUT/'adoption.json',adoption)
    write(OUT/'analysis-references.json',dict(references=refs,worker_must_not_read=True))
    write(OUT/'materials.json',dict(queries=materials,dictionaries=dictionaries,with_demos=False,all_cases_retained=True,independent_confirmation=False))
    write(OUT/'positions.json',dict(records=positions,zero_based=True,same_count_not_norm_matched=True,donor_recipient_positions_must_be_mapped_separately=True))
    (OUT/'model-inputs.jsonl').write_bytes(b''.join(canonical(r)+b'\n' for r in rows))
    text_file(OUT/'model-task.txt',TASK.read_text())
    doc=['# 18份已采用输入','','与已审核预览逐字一致。D00无词典，D01贬损义，D02普通义；无示例。参考答案不进模型。','']
    ledger=['request_id\tquery_id\tdictionary_id\ttokens\tfocal\tpre\tprompt_sha256']
    for r in rows:
        text_file(OUT/'prompts'/(r['request_id']+'.txt'),r['prompt_text'])
        doc+=['## '+r['request_id'],'','```text',r['prompt_text'],'```','']
        ledger.append('\t'.join(map(str,[r['request_id'],r['query_id'],r['dictionary_id'],r['prompt_tokens'],r['patch_position_sets']['focal'],r['patch_position_sets']['pre'],r['prompt_sha256']])))
    text_file(OUT/'ALL-PROMPTS.md','\n'.join(doc)+'\n');text_file(OUT/'input-ledger.tsv','\n'.join(ledger)+'\n')
    text_file(OUT/'REVIEW.md','# 已全部采用\n\n用户确认：“'+QUOTE+'”。J05/J06/J07/J08参考无，J09/J10参考有。原文、建议答案和既有释义均原样采用，六条仍标记为AI构造。\n\n[原始逐条审核稿](../draft-01/REVIEW.md) · [完整模型输入](ALL-PROMPTS.md)\n')
    sources=sorted({p for p in DRAFT.rglob('*') if p.is_file()}|{TASK,Path(__file__).resolve(),ROOT/'src/diagnostics/cross_term_mechanism_materials_v1.py'})
    write(OUT/'manifest.json',dict(artifacts=[info(p) for p in sorted(OUT.rglob('*')) if p.is_file()],sources=[info(p) for p in sources],immutable_after_seal=True))
    work=ROOT/'reviews/jingba-context-v1';work.mkdir(parents=True,exist_ok=True)
    write(work/'accepted-decision.json',dict(status='accepted_materials_and_GPU_execution',user_message=QUOTE,GPU_execution_authorized=True,
        adoption=info(OUT/'manifest.json'),allowed_gpu_indices=[0,1,2,3],GPU_time_constraint=dict(confirmed=True,deadline_unix=None)))
    selector=ROOT/'docs/research/experiment-plans/jingba-context-candidates-v1/current.json'
    write(work/'candidate-selector-before-adoption.json',read(selector))
    atomic(selector,dict(status='materials_adopted',directory=str(OUT.relative_to(ROOT)),manifest=info(OUT/'manifest.json'),
        accepted_items=ids,scientific_results=False,GPU_execution_authorized=True,GPU_started=False),replace=True)
    print(json.dumps(dict(status='adopted',inputs=len(rows),references={r['query_id']:r['reference'] for r in refs}),ensure_ascii=False))
if __name__=='__main__':main()
