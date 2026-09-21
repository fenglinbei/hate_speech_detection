#!/usr/bin/env python3
import argparse,copy,json,sys,unittest
from decimal import Decimal,localcontext
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from diagnostics import dictionary_free_donor_inputs_v1 as c
from diagnostics.dictionary_free_donor_report_v1 import donor_comparisons
from audit_dictionary_free_donor_results_v1 import audit_donors
class Checks(unittest.TestCase):
    def test_fixed_rule_ledger_donor_mapping_and_no_gold(self):
        from test_dictionary_free_donor_runtime_v1 import fixture
        rows,jobs,selfs=fixture();by={r['request_id']:r for r in rows}
        self.assertEqual(len(jobs),48)
        for j in jobs:
            self.assertEqual(by[j['recipient']]['dictionary_id'],'D01')
            self.assertEqual(by[j['donor']]['dictionary_id'],'D00' if j['condition'].startswith('no-dictionary') else 'D02')
            self.assertEqual(j['layer'],17);self.assertIsNone(j['restoration'])
            self.assertFalse({'gold','reference','label'}&set(j))
    def test_reference_alignment_direction_and_no_refitting(self):
        data={'baselines':[],'effects':[],'margin_error_bound':1e-6};ms={};refs={q:'无' if q.endswith(('01','03')) else '有' for q in c.IDS}
        for q in c.IDS:
            for d,m in [('D00',-2.),('D01',-4.),('D02',3.)]:
                rid=q+'-'+d;data['baselines'].append({'query_id':q,'dictionary_id':d,'request_id':rid,'m':m,'reference':refs[q]});ms[rid]=Decimal(m)
            for cond,m,donor in [('upstream',1.,'D02'),('preceding',-3.9,'D02'),('no-dictionary-focal',-.5,'D00'),('no-dictionary-preceding',-3.75,'D00')]:
                data['effects'].append({'query_id':q,'recipient':q+'-D01','donor':q+'-'+donor,'job_id':q+'-'+cond,'condition':cond,'m':m,'transition':'unchanged'})
        data['donor_comparisons']=donor_comparisons(data,{'CPU_score_shift':{'offset':7.0}});es={e['job_id']:e for e in data['effects']}
        with localcontext() as ctx:
            ctx.prec=120;self.assertEqual(audit_donors(data,ms,es,refs,Decimal.from_float(1e-6)),12)
            for kind in ['score','direction','bound','offset','donor','duplicate']:
                bad=copy.deepcopy(data);r=bad['donor_comparisons'][0]
                if kind=='score':r['U0_effect']+=.5
                elif kind=='direction':r['reference_aligned_U0_effect']=-r['reference_aligned_U0_effect']
                elif kind=='bound':r['pair_bound']=4e-6
                elif kind=='offset':r['CPU_offset']=8.
                elif kind=='donor':r['U0_job_id']=r['U2_job_id']
                else:bad['donor_comparisons'][1]=r
                with self.assertRaises(AssertionError):audit_donors(bad,ms,es,refs,Decimal.from_float(1e-6))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
    c.write(a.output,{'status':'pass' if result.wasSuccessful() else 'fail','tests':result.testsRun,'CUDA_initialized':False,'failures':[str(e) for _,e in result.errors+result.failures]})
    sys.exit(0 if result.wasSuccessful() else 1)
