#!/usr/bin/env python3
"""Actual new controller/worker lifecycle on reference-free synthetic telemetry."""
import argparse,json,sys,tempfile,unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from diagnostics import jingba_context_inputs_v1 as c
from diagnostics import jingba_context_runtime_v1 as rt
from diagnostics import hehe_joint_restore_capture_v1 as cap
from diagnostics.cross_model_applicability_models_v1 import prepared_input
from diagnostics.cross_model_applicability_execution_v1 import runtime_versions

ARTIFACTS=None
RULES={'margin_repeat_order_hook_cap':0.,'margin_padding_prefix_cap':.001,'attention_repeat_order_cap':0.,
 'attention_element_cap':.0001,'attention_row_l1_cap':.001,'attention_row_sum_cap':.000002,
 'margin_bound_floor':.000001,'attention_mass_bound_floor':.0000001}


def fixture():
    rows=[]
    for q in c.IDS:
        for d,length in [('D00',3),('D01',7),('D02',5)]:
            rid=f'synthetic-{q}-{d}';ids=[6 if d=='D01' else 7]*length+[8,9,10,11,12,13]
            focal=list(range(length+2,length+4)) if q.startswith('J') else [length+3]
            pre=list(range(focal[0]-len(focal),focal[0]))
            rows.append({'request_id':rid,'query_id':q,'dictionary_id':d,'input_ids':ids,'prompt_tokens':len(ids),
                'input_ids_sha256':c.digest(ids),'prompt_sha256':c.digest(rid),'prompt_text':'synthetic-text-only','spans':[{'id':'query','char_start':0,'char_end':9}],
                'roles':{'query_all':list(range(length,length+5)),'pre_answer':[len(ids)-1]},'capture_positions':pre+focal,
                'capture_prefix_length':max(focal)+1,'candidate_tokens':{'有':3,'无':4},
                'patch_position_sets':{'focal':focal,'pre':pre}})
    jobs,selfs=c.make_jobs(rows)
    return rows,jobs,selfs


def synthetic(model,req,pad,*,positions=(),observe=False,layer=None,replacement=None,padding='none',prefix_length=None,append_tokens=(),trajectory=False,restoration=None):
    value={'D00':2.,'D01':-4.,'D02':4.}[req['dictionary_id']]
    def native_at(pos):return np.full(8,value+.125*(pos-req['roles']['query_all'][0]),np.float32)
    bank=np.repeat(np.stack([native_at(i) for i in positions])[None],36,axis=0) if observe else None
    h=np.full(8,value,np.float32);hs=[];bs=[];restore_proof=[] if restoration is not None else None
    p=prepared_input(req['input_ids']+list(append_tokens),pad,padding);valid=[i for i,x in enumerate(p['attention_mask']) if x]
    for li in range(36):
        pre=h.copy();a=np.zeros(8,np.float32)
        if layer is not None and li==layer+1:
            a[0]=(.75 if list(positions)==req['patch_position_sets']['focal'] else .1)*(float(replacement[0,0])-float(native_at(positions[0])[0]))
        if li==26:a[0]=np.float32(.4)*pre[0]
        mid=pre+a;mlp=np.zeros(8,np.float32)
        if li==28:mlp[0]=np.float32(.3)*mid[0]
        spec=next((ss for ss in (restoration or []) if ss['layer']==li),None)
        if spec is not None:
            branch=spec['branch'];before=(a if branch=='attention' else mlp).copy();replacement_branch=spec['value']
            if branch=='attention':a=replacement_branch.copy();mid=pre+a
            else:mlp=replacement_branch.copy()
            pos=spec['position']
            restore_proof.append({'layer':li,'branch':branch,'position':pos,'padded_position':valid[pos],
                'replacement_sha256':cap.array_sha(replacement_branch),'before_sha256':cap.array_sha(before),
                'replacement_l2':float(np.linalg.norm(replacement_branch.astype(np.float64))),
                'before_l2':float(np.linalg.norm(before.astype(np.float64))),
                'difference_l2':float(np.linalg.norm(replacement_branch.astype(np.float64)-before)),
                'changed_rows_in_unpadded_coordinates':[] if np.array_equal(before,replacement_branch) else [pos],
                'outside_rows_exact':True,'replacement_exact':True,'native_output_unmodified':True,
                'other_attention_outputs_preserved':branch=='attention'})
        h=mid+mlp
        hs.append(np.stack([pre,mid,h]));bs.append(np.stack([a,mlp]))
    states=np.stack(hs);normalized=states/np.sqrt(np.mean(states*states,axis=-1,keepdims=True)+np.float32(1e-6))
    weights=np.zeros((2,8),np.float32);weights[0,0]=-1;weights[1,0]=1
    lens=normalized@weights.T
    t={'states':states,'normalized':normalized,'lens_logits':lens,'branches':np.stack(bs),
        'label_weights':weights,'norm_weight':np.ones(8,np.float32),'norm_eps':np.asarray([1e-6],np.float64),
        'position':np.asarray(req['roles']['pre_answer'],np.int64),'candidate_ids':np.asarray([3,4],np.int64)}
    v=np.full(16,-20,np.float32);v[[3,4]]=lens[-1,2]
    if append_tokens:v[2]=20
    proof=None
    if layer is not None:
        p=prepared_input(req['input_ids']+list(append_tokens),pad,padding);valid=[i for i,x in enumerate(p['attention_mask']) if x]
        before=np.stack([native_at(i) for i in positions])
        proof={'layer':layer,'positions':list(positions),'padded_positions':[valid[i] for i in positions],
            'donor_sha256':cap.array_sha(replacement),'recipient_before_sha256':cap.array_sha(before),
            'changed_rows_in_unpadded_coordinates':[] if np.array_equal(before,replacement) else list(positions),
            'outside_rows_exact':True,'donor_rows_exact':True,'native_output_unmodified':True,'restoration':restore_proof}
    return v,bank,proof,t if trajectory else None


class Checks(unittest.TestCase):
    def test_lifecycle_and_corruption(self):
        rows,jobs,selfs=fixture()
        real_popen=rt.subprocess.Popen
        profile={'layers':36,'hidden_size':8,'vocab_size':16,'candidate_tokens':{'有':3,'无':4},
            'eos_token_ids':[2],'pad_token_id':0,'rms_norm_eps':1e-6,'metadata_sources':[],'weight_sources':[]}
        plan={'acceptance':RULES,'state_scaled_cap':.0001,'format':{'max_new_tokens':8},'budget':{'usual_total':378}}
        with tempfile.TemporaryDirectory(prefix='hehe-joint-restore-runtime-') as td, ExitStack() as stack:
            tmp=Path(td);prepared=tmp/'prepared';prepared.mkdir();c.write(prepared/'manifest.json',{'synthetic':True,'artifacts':[]})
            references=[{'query_id':q,'reference':'无' if int(q[1:])<=8 else '有',
                'reference_basis':'user_bulk_adoption_of_proposed_current_task_reference','source_labels':None} for q in c.IDS]
            c.write(prepared/'analysis-references.json',{'references':references})
            c.write(prepared/'execution-plan.json',plan);c.write(prepared/'model-profile.json',profile)
            c.write(prepared/'interventions.json',{'cross':jobs,'self':selfs})
            (prepared/'scoring-inputs.jsonl').write_bytes(b''.join(c.canonical(r)+b'\n' for r in rows))
            decision=tmp/'decision.json';constraint={'confirmed':True,'deadline_unix':None}
            c.write(decision,{'GPU_execution_authorized':True,'user_message':'SYNTHETIC CPU FIXTURE ONLY',
                'prepared_manifest':c.info(prepared/'manifest.json'),'GPU_time_constraint':constraint})
            allocation=[{'index':0,'uuid':'synthetic-only','name':'synthetic','total_mib':48000}]
            inventory={'devices':[dict(allocation[0],used_mib=0,utilization=0)],'compute_processes':[],'synthetic':True}
            bound=tmp/'bound.json';c.write(bound,{'prepared_manifest':c.info(prepared/'manifest.json'),'allocation':allocation,
                'runtime_versions':runtime_versions(),'authorization_note':'CPU SYNTHETIC ONLY','GPU_time_constraint':constraint,'execution_decision':c.info(decision)})
            run=tmp/'run-synthetic';calls=[]
            stack.enter_context(patch.object(c,'WORK',tmp));stack.enter_context(patch.object(c,'validate',return_value=(plan,profile,rows,jobs,selfs)))
            stack.enter_context(patch.object(rt,'gpu_inventory',return_value=inventory));stack.enter_context(patch.object(rt.signal,'signal'))
            stack.enter_context(patch('diagnostics.cross_model_applicability_models_v1.load_checkpoint',return_value=(object(),{'synthetic':True})))
            def counted(*a,**kw):
                v=synthetic(*a,**kw);calls.append(kw)
                if len(calls)==2:(run/'STOP').touch()
                return v
            stack.enter_context(patch.object(cap,'forward',side_effect=counted))
            class Process:
                pid=987654321
                def __init__(self,cmd,**kw):
                    get=lambda name:cmd[cmd.index(name)+1]
                    rt.worker(get('--prepared'),get('--bound'),get('--run'),get('--invocation'),get('--phase'))
                def wait(self):return 0
            stack.enter_context(patch.object(rt.subprocess,'Popen',Process))
            self.assertEqual(rt.supervise(prepared,bound,run)['status'],'paused');self.assertEqual(len(calls),2)
            path=rt.record_path(run,'native-baseline',rows[0]['request_id']);original=path.read_bytes();(run/'STOP').unlink()
            self.assertEqual(rt.supervise(prepared,bound,run,resume=True)['status'],'qualified')
            self.assertEqual(path.read_bytes(),original);self.assertEqual(len(calls),336)
            self.assertFalse(rt.check_run(prepared,run)['query_reference_join_performed'])
            from diagnostics.jingba_context_report_v1 import analyze
            with self.assertRaisesRegex(ValueError,'Full run'):analyze(prepared,run,tmp/'premature')
            self.assertFalse((tmp/'premature').exists())
            self.assertEqual(rt.supervise(prepared,bound,run,phase='full')['status'],'complete')
            self.assertEqual(len(calls),378);checked=rt.check_run(prepared,run)
            self.assertEqual(len(checked['qualification']['trajectory']['structural_guards']),144)
            self.assertEqual(checked['qualification']['margin_error_bound'],1e-6)
            from audit_jingba_context_results_v1 import audit_branch_sources
            records={}
            for file in (run/'records').glob('*/*.json'):
                rec=c.read(file);records[rec['stage'],rec['job']['job_id'] if rec['job'] else rec['request_id']]=rec
            request_map={r['request_id']:r for r in rows}
            self.assertEqual(audit_branch_sources(run,records,request_map,profile),0)
            target=records['production',jobs[0]['job_id']]['patch_proof']
            correct=target['restoration'];target['restoration']=[]
            with self.assertRaises(AssertionError):audit_branch_sources(run,records,request_map,profile)
            target['restoration']=correct
            # Rendering may invoke fc-list. Only worker launches are simulated;
            # do not route font discovery through the fake GPU worker process.
            with patch.object(rt.subprocess,'Popen',real_popen):
                result=analyze(prepared,run,tmp/'analysis')
            self.assertEqual(result['interventions'],24)
            self.assertEqual(result['restoration_contrasts'],0)
            self.assertEqual(len(checked['qualification']['trajectory']['restoration_guards']),0)
            results=c.read(tmp/'analysis/results.json')
            self.assertTrue(all(r['removed_fraction']>0 for r in results['restoration_contrasts']))
            self.assertTrue(all(s['delta_m']==0 for s in results['self_controls']))
            self.assertEqual(len(results['condition_gaps']),6)
            self.assertEqual(len(results['position_differences']),12)
            self.assertEqual({s['query_id']:s['reference'] for s in results['baselines']},{r['query_id']:r['reference'] for r in references})
            self.assertEqual(len(results['dictionary_addition_gaps']),12)
            from audit_jingba_context_results_v1 import audit
            from decimal import localcontext
            with localcontext() as ctx:
                ctx.prec=120;audited=audit(prepared,run,tmp/'analysis')
            self.assertEqual(audited['absolute_vectors'],378)
            if ARTIFACTS is not None:
                import shutil
                ARTIFACTS.mkdir(parents=True,exist_ok=False)
                shutil.copytree(tmp/'analysis',ARTIFACTS/'synthetic-analysis')
                c.write(ARTIFACTS/'synthetic-audit.json',{k:v for k,v in audited.items() if k not in ['raw_seal','results']})
                (ARTIFACTS/'README.md').write_text('# CPU合成数据检查\n\n全部为合成向量，不是本轮查询的模型结果。仅供检查图表、流水线与独立数值复核；临时运行数据已正常清理，来源路径仅为本次测试记录。\n')
            with self.assertRaisesRegex(ValueError,'terminal'):rt.supervise(prepared,bound,run,phase='full')
            rec=c.read(rt.record_path(run,'production',jobs[0]['job_id']));p=Path(rec['trajectory']['path'])
            original=p.read_bytes();bad=bytearray(original);bad[-1]^=1;p.write_bytes(bad)
            with self.assertRaisesRegex(ValueError,'Pinned bytes'):rt.check_run(prepared,run)
            p.write_bytes(original)
            p=run/'format'/(jobs[0]['job_id']+'.json');bad=c.read(p);bad['steps'][0]['patch_proof']['donor_sha256']='wrong';p.write_bytes(c.canonical(bad))
            with self.assertRaisesRegex(ValueError,'exact patch'):rt.check_run(prepared,run)

    def test_CPU_preparation_refuses_GPU_before_inventory_and_weights(self):
        with patch.object(rt,'gpu_inventory') as inv,patch.object(c,'validate') as validate:
            with self.assertRaisesRegex(ValueError,'not authorized'):rt.bind('/missing',0,'/unused','CPU only')
            inv.assert_not_called();validate.assert_not_called()
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);c.write(p/'manifest.json',{})
            denied=p/'denied.json';c.write(denied,{'GPU_execution_authorized':False,'user_message':'暂不启动GPU任务'})
            with patch.object(rt,'gpu_inventory') as inv,patch.object(c,'validate') as validate:
                with self.assertRaisesRegex(ValueError,'authorization'):rt.bind(p,0,p/'bound.json','CPU only',denied)
                inv.assert_not_called();validate.assert_not_called()
            self.assertFalse((p/'bound.json').exists())
            c.write(p/'unauthorized-bound.json',{})
            with patch.object(c,'validate') as validate,patch('diagnostics.cross_model_applicability_models_v1.load_checkpoint') as load:
                with self.assertRaisesRegex(ValueError,'decision missing'):rt.worker(p,p/'unauthorized-bound.json',p/'no-run','unregistered','engineering')
                validate.assert_not_called();load.assert_not_called()

    def test_mapped_positions_and_D00_native_only(self):
        rows,jobs,selfs=fixture();byid={r['request_id']:r for r in rows}
        self.assertEqual((len(rows),len(jobs),len(selfs)),(18,24,24))
        for j in jobs+selfs:
            r,d=byid[j['recipient']],byid[j['donor']]
            self.assertNotEqual(r['dictionary_id'],'D00');self.assertNotEqual(d['dictionary_id'],'D00')
            self.assertEqual([r['input_ids'][i] for i in j['positions']],[d['input_ids'][i] for i in j['donor_positions']])
            self.assertEqual(j['positions']==j['donor_positions'],j['kind']=='self_control')

    def test_original_score_gates(self):
        self.assertEqual(rt.qualify([{'kind':'repeat','margin_difference':0.}],RULES)['margin_error_bound'],1e-6)
        for v in [{'kind':'repeat','margin_difference':1e-12},{'kind':'left','margin_difference':.001001}]:
            with self.assertRaises(ValueError):rt.qualify([v],RULES)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path);p.add_argument('--artifacts',type=Path);a=p.parse_args()
    ARTIFACTS=a.artifacts
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
    receipt={'status':'pass' if result.wasSuccessful() else 'fail','tests':result.testsRun,
        'CUDA_initialized':False,'research_checkpoint_loaded':False,'GPU_inventory':'synthetic mock only',
        'synthetic_forwards':378,'failures':[str(e) for _,e in result.failures+result.errors]}
    if a.output:
        a.output.parent.mkdir(parents=True,exist_ok=True);c.write(a.output,receipt)
    print(json.dumps(receipt));sys.exit(0 if result.wasSuccessful() else 1)
