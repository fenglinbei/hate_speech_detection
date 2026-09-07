"""Separately reconstruct dictionary blocks from frozen fields; no Gold access."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / 'exps/causal_context/general_model_ld_nolabel_v1'
read = lambda p: json.loads(p.read_text())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
ref = read(BASE / 'plan_ref.json')
directory = Path(ref['target_path'])
plan = read(directory / 'plan.json')
assert sha(directory / 'plan.json') == ref['plan_sha256']
prior_ref = read(Path(plan['parent_plan_ref']))
prior = Path(prior_ref['target_path'])
old = {r['record_id']:r for r in map(json.loads,(prior/'contexts.dev.jsonl').read_text().splitlines())}
new = {r['record_id']:r for r in map(json.loads,(directory/'contexts.dev.jsonl').read_text().splitlines())}
lexicon = ROOT / plan['registration_config']['resource']['lexicon_path']
assert sha(lexicon) == plan['registration_config']['resource']['lexicon_sha256']
entries = {r['lexicon_id']:r for r in read(lexicon)['terms']}
labels = ('Racism','Region','LGBTQ','Sexism','others')
baseline_count = removal_count = empty_count = fields_removed = 0
token_reductions = []
for row in new.values():
    q,t,c = row['query_id'],row['task'],row['condition']
    if not c.endswith('NoCat'):
        assert row == old[row['record_id']]
        baseline_count += 1
        continue
    source = old[f'{q}:{t}:{c.removesuffix("NoCat")}']
    ids = source['trace']['injected_lexicon_ids']
    assert ids == sorted(set(ids))
    complete_parts,definition_parts = [],[]
    for identifier in ids:
        entry = entries[identifier]
        full = [f'词条：{entry["term"]}']
        definition = full.copy()
        for ordinal,sense in enumerate(entry['senses'],1):
            heading = f'义项 {ordinal}：'
            text = '定义：'+(sense['definition'] or '未提供释义（冻结时为空；未补写）')
            full += [heading,'类别：'+json.dumps([l for l in labels if l in sense['categories']],ensure_ascii=False,separators=(',',':')),text]
            definition += [heading,text]
            fields_removed += 1
        complete_parts.append('\n'.join(full))
        definition_parts.append('\n'.join(definition))
    full = '词典参考：\n'+'\n\n'.join(complete_parts) if ids else ''
    definition = '词典参考：\n'+'\n\n'.join(definition_parts) if ids else ''
    assert source['messages'][1]['content'].startswith(full)
    assert row['messages'][0] == source['messages'][0]
    assert row['messages'][1]['content'] == definition+source['messages'][1]['content'][len(full):]
    assert row['trace']['injected_demo_ids'] == source['trace']['injected_demo_ids']
    assert row['trace']['injected_blocks']['demo_sha256'] == source['trace']['injected_blocks']['demo_sha256']
    token_reductions.append(source['prompt_tokens']-row['prompt_tokens'])
    if not ids:
        assert row['prompt_text'] == source['prompt_text']
        empty_count += 1
    removal_count += 1
assert baseline_count == 5144 and removal_count == 2572 and empty_count == 12
receipt = {'plan_id':ref['plan_id'],'plan_sha256':ref['plan_sha256'],'passed':True,
           'checker_sha256':sha(Path(__file__)),'baseline_contexts_exact':baseline_count,
           'category_removal_contexts_exact':removal_count,'empty_resource_identity_contexts':empty_count,
           'removed_explicit_fields_across_tasks_and_conditions':fields_removed,
           'prompt_token_reduction_min':min(token_reductions),'prompt_token_reduction_max':max(token_reductions),
           'prompt_token_reduction_mean':sum(token_reductions)/len(token_reductions),
           'query_gold_loaded':False,'test_content_read':False,'model_forward_executed':False,
           'method':'manual-field-reconstruction-without-production-renderer'}
(BASE/'audits/input-audit.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(receipt,ensure_ascii=False,indent=2))
