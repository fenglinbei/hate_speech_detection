"""Meaningful CPU tests: tiny native models, exact arithmetic and staged lifecycle."""
import argparse
import copy
from decimal import Decimal, localcontext
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

os.environ.update(CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1')
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'src'),str(ROOT)]
from diagnostics import cross_model_applicability_execution_v1 as c
from diagnostics import cross_model_applicability_models_v1 as m
spec=importlib.util.spec_from_file_location('cmad_runner_test',ROOT/'scripts/review/run_cross_model_applicability_v1.py')
runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
DETAILS={}


class CPUChecks(unittest.TestCase):
    def test_prepared_input_isolation_and_native_frames(self):
        plan,profiles,requests=c.check_prepared(PREPARED,sealed=SEALED)
        for p in profiles:
            original=[]
            for name in ['new-model-inputs','legacy-model-inputs']:
                original+=c.jsonl(c.SCIENCE/f"tokenized/{p['model_key']}-{name}.jsonl")
            rs=requests[p['model_key']]
            self.assertEqual(len(rs),540)
            for r,t in zip(rs,original):
                self.assertEqual(r['input_ids'],t['input_ids']);self.assertEqual(r['prompt_sha256'],t['prompt_sha256'])
                self.assertEqual(r['candidate_tokens'],t['candidate_tokens'])
            self.assertEqual(len({r['query_id'] for r in rs if r['request_id'] in p['format_probe_ids']}),16)
            self.assertEqual(set(p['production_ids']),{r['request_id'] for r in rs if p['model_key']!='qwen3-8b' or r['query_id']})
        self.assertEqual(plan['budget']['prompt_forward_total_without_generation'],9564)
        DETAILS['CPU_frozen_token_records']=1620

    def test_answer_position_and_bad_input_guards(self):
        for n in [1,15,16,17,531,574,666]:
            ids=list(range(1,n+1))
            for side in ['none','left','right']:
                p=m.prepared_input(ids,0,side)
                self.assertEqual([x for x,v in zip(p['input_ids'],p['attention_mask']) if v],ids)
                self.assertEqual(p['position_ids'][p['last_valid_index']],n-1)
                self.assertEqual(p['input_ids'][p['last_valid_index']],n)
                self.assertEqual(sum(p['attention_mask']),n)
                if side=='right':self.assertLess(p['last_valid_index'],len(p['input_ids'])-1)
        for ids in [[],[-1],[0]*8193]:
            with self.assertRaises(ValueError):m.prepared_input(ids,0)

    def test_native_qwen_and_glm_selected_projection_and_reload(self):
        import torch
        # Some optional Transformers imports probe CUDA. CPU test explicitly
        # stubs availability; no model checkpoint or GPU context is required.
        with patch.object(torch.cuda,'is_available',return_value=False):
            m.configure_torch(cpu=True)
            from transformers import Qwen3Config,Qwen3ForCausalLM
            qcfg=Qwen3Config(vocab_size=97,hidden_size=32,intermediate_size=64,num_hidden_layers=2,
                            num_attention_heads=4,num_key_value_heads=2,head_dim=8,max_position_embeddings=128)
            qcfg._attn_implementation='eager'
            qc=Qwen3ForCausalLM(qcfg).float().eval()
            config_class,model_class=m.load_glm_classes(ROOT/'models/base/GLM-4-9B-Chat')
            gcfg=config_class(num_layers=2,padded_vocab_size=97,hidden_size=32,ffn_hidden_size=64,
                kv_channels=8,num_attention_heads=4,seq_length=128,multi_query_attention=True,multi_query_group_num=2,
                original_rope=True,torch_dtype='float32',num_hidden_layers=2)
            gcfg._attn_implementation='eager'
            gc=model_class(gcfg,empty_init=False,device='cpu').float().eval()
            for name,p in gc.named_parameters():torch.nn.init.uniform_(p,-.1,.1)
            results=[]
            for architecture,model in [('qwen3',qc),('chatglm',gc)]:
                mapping=m.device_map({'architecture':architecture,'layers':2},2)
                for name,_ in list(model.named_parameters())+list(model.named_buffers()):
                    self.assertTrue(any(name==prefix or name.startswith(prefix+'.') for prefix in mapping),name)
                max_native_error=0.;max_padding_error=0.
                for ids in [[3,5,7],[3,5,7,9,11,13,15,19,21,23,25,27,31,33,35,37,39]]:
                    _,baseline=m.forward_vector(model,architecture,ids,0)
                    for padding in ['none','left','right']:
                        prepared,selected=m.forward_vector(model,architecture,ids,0,padding)
                        tensors={k:torch.tensor([prepared[k]],dtype=torch.long) for k in ['input_ids','attention_mask','position_ids']}
                        kwargs={'return_last_logit':False} if architecture=='chatglm' else {'logits_to_keep':0}
                        with torch.inference_mode():native=model(**tensors,use_cache=False,return_dict=True,**kwargs).logits[0,prepared['last_valid_index']].numpy()
                        err=float(abs(native-selected).max());max_native_error=max(max_native_error,err)
                        max_padding_error=max(max_padding_error,float(abs(selected-baseline).max()))
                        self.assertLessEqual(err,1e-6)
                        self.assertLessEqual(float(abs(selected-baseline).max()),1e-6)
                # Same from_pretrained keyword path on a tiny synthetic safe-tensor
                # checkpoint verifies the local GLM loader with this HF version.
                with tempfile.TemporaryDirectory(prefix='cmad-tiny-reload-') as temp:
                    model.save_pretrained(temp,safe_serialization=True)
                    loaded,loading=type(model).from_pretrained(temp,local_files_only=True,use_safetensors=True,
                        torch_dtype=torch.float32,attn_implementation='eager',device_map={'':'cpu'},
                        low_cpu_mem_usage=True,output_loading_info=True)
                    loaded.eval()
                    self.assertFalse(any(loading.get(k) for k in ['missing_keys','unexpected_keys','mismatched_keys','error_msgs']))
                    _,before=m.forward_vector(model,architecture,[3,5,7],0)
                    _,after=m.forward_vector(loaded,architecture,[3,5,7],0)
                    self.assertTrue((before==after).all())
                results.append({'architecture':architecture,'native_selected_max_error':max_native_error,
                                'padding_max_error':max_padding_error,'tiny_checkpoint_reload_exact':True})
            self.assertFalse(torch.cuda.is_initialized())
            DETAILS['tiny_random_CPU_models']=results
            DETAILS['pretrained_research_model_weights_loaded']=False

    def test_decimal_readouts_and_candidate_identity(self):
        import numpy as np
        rng=np.random.default_rng(37);maximum=0.
        with localcontext() as ctx:
            ctx.prec=60
            for n in range(12):
                vector=rng.normal(0,9,size=97).astype(np.float32);a={'有':17,'无':29}
                r=c.readout(vector,a,1e-6)
                x=[Decimal.from_float(float(v)) for v in vector]
                lnz=sum(v.exp() for v in x).ln()
                expected={'m':x[29]-x[17],'log_p_no':x[29]-lnz,'log_p_yes':x[17]-lnz,
                          'legal_mass':(x[29].exp()+x[17].exp())/lnz.exp()}
                for key,value in expected.items():
                    error=abs(r[key]-float(value));maximum=max(maximum,error);self.assertLessEqual(error,1e-12)
                swapped=c.readout(vector,{'有':29,'无':17},1e-6)
                self.assertEqual(swapped['m'],-r['m'])
        DETAILS['independent_Decimal_precision']=60;DETAILS['readout_max_absolute_error']=maximum
        for vector in [np.array([0,np.nan],dtype=np.float32),np.array([0,1],dtype=np.float16)]:
            with self.assertRaises(ValueError):c.readout(vector,{'有':0,'无':1})

    def test_physical_alias_cancellation_and_unresolved_bounds(self):
        scores={'a':{'m':5.,'margin_error_bound':.01,'physical_score_id':'same'},
                'b':{'m':5.,'margin_error_bound':.01,'physical_score_id':'same'}}
        effect=c.linear_effect([{'condition_id':'a','coefficient':1},{'condition_id':'b','coefficient':-1}],scores)
        self.assertEqual((effect['value'],effect['bound'],effect['physical_terms']),(0.,0.,[]))
        self.assertEqual(c.resolution(.001,.001),'numerical_unresolved')
        scores['b']['m']=4.
        with self.assertRaises(ValueError):c.linear_effect([{'condition_id':'a','coefficient':1},{'condition_id':'b','coefficient':-1}],scores)

    def test_qualification_rejects_changed_runs(self):
        import numpy as np
        r=c.readout(np.array([2.,3.,5.],dtype=np.float32),{'有':0,'无':1})
        vals={name:{'x':copy.deepcopy(r)} for name,_,_ in c.PASSES}
        rules={'repeat_and_order_margin_cap':0.,'padding_margin_cap':.001,'logprob_identity_cap':1e-10,
               'margin_bound_floor':1e-6,'margin_bound_multiplier':2}
        self.assertEqual(c.qualify(vals,rules)['margin_error_bound'],1e-6)
        for stage in ['repeat','left_padding','reverse_order']:
            bad=copy.deepcopy(vals);bad[stage]['x']['m']+=.01
            with self.assertRaises(ValueError):c.qualify(bad,rules)
        for status in ['complete','failed','running_production','created']:
            with self.assertRaises(ValueError):c.validate_resume({'status':status,'binding':{}},{},'full',True)
        c.validate_resume({'status':'paused','binding':{}},{},'full',True)
        c.validate_resume({'status':'qualified','binding':{}},{},'full',False)
        with self.assertRaises(ValueError):c.validate_resume({'status':'paused','binding':{}},{'different':1},'full',True)

    def test_synthetic_worker_pause_resume_seals_and_tamper(self):
        import numpy as np
        import torch
        with tempfile.TemporaryDirectory(prefix='cmad-synthetic-lifecycle-') as temp:
            root=Path(temp);prep=root/'prepared';prep.mkdir();run=root/'run';run.mkdir()
            c.dump(prep/'manifest.json',{'synthetic_fixture_only':True})
            binding={'model_key':'toy','allocation':[],'prepared_manifest':c.info(prep/'manifest.json')}
            bound=root/'bound.json';c.dump(bound,binding);c.dump(run/'binding.json',binding)
            state={'run_id':'synthetic-only','binding':c.info(bound),'status':'launching','invocations':[{'id':'first'}]}
            c.dump(run/'state.json',state)
            req=[]
            for i in range(16):
                ids=[3,5,40+i];rid=f'toy-{i:02}'
                req.append({'request_id':rid,'condition_id':rid,'prompt_sha256':str(i),'input_ids_sha256':c.hashlib.sha256(c.canonical(ids)).hexdigest(),'input_ids':ids})
            profile={'model_key':'toy','architecture':'qwen3','vocab_size':97,'candidate_tokens':{'有':17,'无':29},
                'pad_token_id':0,'eos_token_ids':[2],'format_probe_ids':[r['request_id'] for r in req],
                'production_ids':[r['request_id'] for r in req],'metadata_sources':[],'weight_sources':[]}
            rules={'repeat_and_order_margin_cap':0.,'padding_margin_cap':.001,'logprob_identity_cap':1e-10,
                   'margin_bound_floor':1e-6,'margin_bound_multiplier':2}
            plan={'acceptance':rules,'format':{'max_new_tokens':8}}
            identity={'synthetic_fixture_only':True};calls=0;pause_at=3
            def forward(model,architecture,ids,pad,padding='none'):
                nonlocal calls
                calls+=1;v=np.zeros(97,dtype=np.float32);v[17]=1.;v[29]=2.+sum(ids)%5/10
                if ids[-1] in [17,29]:v[2]=20.
                if calls==pause_at:(run/'STOP').write_text('synthetic fixture pause')
                return m.prepared_input(ids,pad,padding),v
            with patch.object(c,'check_prepared',return_value=(plan,[profile],{'toy':req})),patch.object(m,'load_checkpoint',return_value=(object(),identity)),patch.object(m,'forward_vector',side_effect=forward):
                self.assertEqual(runner.worker(prep,bound,run,'first','full'),'paused')
                committed=sorted((run/'scores/reference').glob('*.json'))
                self.assertEqual(len(committed),3)
                originals={p:c.sha(p) for p in committed}
                (run/'STOP').unlink();s=c.read(run/'state.json');s['status']='launching';s['invocations'].append({'id':'second'});c.dump(run/'state.json',s)
                self.assertEqual(runner.worker(prep,bound,run,'second','full'),'complete_releasing')
                self.assertEqual(calls,112)  # 80 engineering +16 production +16 extra generation.
                self.assertTrue(all(c.sha(p)==h for p,h in originals.items()))
                s=c.read(run/'state.json');s.update(status='complete',owned_worker_absent=True,worker_exit_code=0)
                release=root/'synthetic-release.json';c.dump(release,{'synthetic_fixture_only':True,'owned_worker_absent':True,'worker_exit_code':0,'worker_pid':s['worker_pid']})
                s['resource_release']=c.info(release);c.dump(run/'state.json',s)
                result=runner.check_run(prep,run);self.assertEqual(len(result['production_records']),16)
                path=run/'scores/production/toy-00.json';bad=c.read(path);bad['prepared_input']['last_valid_index']=0;c.dump(path,bad)
                with self.assertRaises(ValueError):runner.check_run(prep,run)
            self.assertFalse(torch.cuda.is_initialized())
            DETAILS['synthetic_lifecycle']={'engineering_requests':80,'production_requests':16,'extra_generation_forwards':16,
                'checkpoint_requests_reused':3,'prefix_receipts_unchanged':True,'tampered_position_rejected':True,
                'resource_release_was_a_synthetic_fixture':True}

    def test_analysis_separates_primary_and_supplement(self):
        plan=c.read(c.SCIENCE/'analysis-plan.json');conditions=c.read(c.SCIENCE/'conditions.json')['conditions']
        records=[]
        for i,condition in enumerate(conditions):
            records.append({'condition_id':condition['condition_id'],'physical_score_id':f'synthetic:{i}',
                            'readout':{'m':float(i%13),'margin_error_bound':1e-6}})
        result=c.production_analysis(records,plan)
        self.assertEqual(len(result['comparisons']),1152)
        self.assertEqual(len(result['scores']),384)
        for row in result['stratified_descriptive_summaries']:
            self.assertEqual(len(row['families']),2)
            self.assertTrue(set(row['families'])<=({'CMAD-HP','CMAD-XC'} if row['analysis_role']=='naturalistic_primary_development' else {'CMAD-JSC','CMAD-MT'}))
        with patch.object(runner,'check_run',return_value={'status':'qualified'}):
            with self.assertRaises(ValueError):runner.analyze(None,None,None)
        for r in records:
            if r['condition_id']=='CMAD-HP-Q1-L0-D0':r['readout']['m']=-2.
            if r['condition_id']=='CMAD-HP-Q1-L1-D0':r['readout']['m']=2.
        controlled=c.production_analysis(records,plan)
        effect=next(r for r in controlled['comparisons'] if r['comparison_id']=='CMAD-HP-Q1-E-L1-D0')
        self.assertEqual(effect['verified_classification_transition'],'repair')
        self.assertEqual(effect['effect']['value'],4.)
        DETAILS['synthetic_analysis_expressions']=1152

    def test_all_legacy_queries_and_N_retained(self):
        plan=c.read(PREPARED/'legacy-analysis.json')
        records=[{'condition_id':cid,'physical_score_id':f'legacy-toy:{i}',
                  'readout':{'m':float(i%7-3),'margin_error_bound':1e-6}} for i,cid in enumerate(plan['condition_query_ids'])]
        result=c.production_analysis(records,plan)
        self.assertEqual(len(result['scores']),156);self.assertEqual(len(result['comparisons']),168)
        self.assertEqual(sum(r['source_frame']=='legacy_N_diagnostic' for r in result['scores']),36)
        self.assertEqual(len({r['query_id'] for r in result['scores']}),12)
        self.assertTrue(all(r['original_gold'] is None for r in result['scores']))
        DETAILS['synthetic_legacy_analysis']={'conditions':156,'queries':12,'core_comparisons':168,'external_N_conditions':36}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--prepared',type=Path,default=c.PREPARED)
    parser.add_argument('--sealed',action='store_true');parser.add_argument('--output',type=Path)
    args=parser.parse_args();PREPARED=args.prepared;SEALED=args.sealed
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(CPUChecks))
    report={'status':'pass' if result.wasSuccessful() else 'fail','tests':result.testsRun,
            'failures':len(result.failures),'errors':len(result.errors),'details':DETAILS,
            'research_pretrained_model_forward_calls':0,'GPU_forward_calls':0,
            'scope':'CPU tiny random models and synthetic fixtures only; not GPU numerical qualification.'}
    if args.output:c.dump(args.output,report)
    print(json.dumps(report,ensure_ascii=False,indent=2))
    sys.exit(0 if result.wasSuccessful() else 1)
