#!/usr/bin/env python3
"""Meaningful numerical, data-split, damage and bilateral regression fixtures."""
import argparse
from copy import deepcopy
from fractions import Fraction as F
import json
from pathlib import Path
import sys
import unittest
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from diagnostics import reference_routing_content_analysis_v1 as a
from diagnostics import reference_routing_content_inputs_v1 as c


def refs(n=24):
    return [{'query_id':f'D{i+1:02}','reference':'无' if i%2==0 else '有','term_family_id':f'T{i//8+1:02}',
             'split':'development','stratum':'CPU_fixture','new_term':False} for i in range(n)]


def records(qs):
    result={}
    for q in qs:
        y=1 if q['reference']=='无' else -1
        for condition in ['M00','BG','BA','BI','MPS','MSP']+[f'{o}_replace_{k}' for o in ('MPS','MSP') for k in range(1,5)]:
            for method in (['N'] if condition=='M00' else ['N','U','Q2','Q4']):
                result[f'rrc-{q["query_id"]}-{condition}/{method}']={'m':y*(2 if method=='U' else 1),'bound':1e-6}
    return result


class Checks(unittest.TestCase):
    def test_shared_record_bound_cancellation_and_CAD(self):
        r={'x':{'m':3,'bound':.1},'z':{'m':-1,'bound':.2}}
        n,z=a.physical('x'),a.physical('z')
        self.assertEqual((n+7-n).value(r),7);self.assertEqual((n+7-n).bound(r),0)
        g=F(3,2)*n-F(1,2)*z-n
        self.assertEqual(g.value(r),2);self.assertEqual(g.bound(r),(F(.1)+F(.2))/2)

    def test_unequal_class_counts_constant_offset_balanced_G_zero(self):
        qs=refs(3);r=records(qs);p={'bN':'0','bU':'0','QAS_factor':1}
        rows=a.primary_rows(qs,r,p,'OFFSET7');s=a.aggregate(rows,r)
        self.assertEqual(s['balanced_G']['exact_value'],'0');self.assertEqual(s['balanced_G']['exact_bound'],'0')
        self.assertNotEqual(s['all_endpoint_mean_G']['value'],0)
        self.assertEqual(s['per_class_G']['无']['value'],7);self.assertEqual(s['per_class_G']['有']['value'],-7)

    def test_fit_ties_dev_only_and_nonprimary_outputs_excluded(self):
        qs=refs();r=records(qs);parameters,tables=a.fit(qs,r)
        self.assertEqual(parameters,{'bN':'0','bU':'0','QAS_factor':1,'Bstar':'N'})
        changed=deepcopy(r)
        for key in changed:
            if '-MPS/' not in key and '-MSP/' not in key and '-M00/' not in key:changed[key]['m']=-999
        self.assertEqual(a.fit(qs,changed),(parameters,tables))
        bad=deepcopy(qs);bad[0]['split']='confirmation'
        with self.assertRaises(ValueError):a.fit(bad,r)

    def test_offset_grid_uses_negative_midpoints_and_all_extremes(self):
        qs=refs();r=records(qs)
        for q in qs:
            for order in ('MPS','MSP'):
                r[f'rrc-{q["query_id"]}-{order}/N']['m']=2 if q['reference']=='无' else -4
        _,t=a.fit(qs,r);values={F(x['offset']) for x in t['bN']['candidates']}
        self.assertEqual(values,{F(0),F(1),F(-3),F(5)})

    def test_unresolved_not_correct_damage_and_union_denominators(self):
        qs=refs(2);r=records(qs);p={'bN':'0','bU':'0','QAS_factor':1}
        r['rrc-D01-MPS/U']['m']=-1;r['rrc-D01-MSP/U']['m']=-1
        r['rrc-D02-MSP/U']['m']=0
        rows=a.primary_rows(qs,r,p,'U');s=a.aggregate(rows,r);t=s['transitions']
        self.assertEqual(t['counts']['damage'],2);self.assertEqual(t['any_order_damage_queries'],['D01'])
        self.assertEqual(t['counts']['correct_to_unresolved'],1);self.assertEqual(t['eligible_native_correct'],4)
        self.assertEqual(t['damage_rate'],.5);self.assertEqual(s['balanced_accuracy'],.25)

    def test_auc_order_mean_not_auc_of_mean_margin_and_near_ties(self):
        qs=refs(2);r=records(qs);p={'bN':'0','bU':'0','QAS_factor':1}
        for key,value in [('D01-MPS',2),('D02-MPS',1),('D01-MSP',-100),('D02-MSP',0)]:r['rrc-'+key+'/N']['m']=value
        s=a.auc(a.primary_rows(qs,r,p,'N'),r)
        self.assertEqual(s['order_mean']['value'],.5)
        r['rrc-D01-MPS/N']['m']=1
        s=a.auc(a.primary_rows(qs,r,p,'N'),r)['by_condition']['MPS']
        self.assertEqual(s['value'],.5);self.assertEqual(s['engineering_lower'],0);self.assertEqual(s['engineering_upper'],1)

    def test_bilateral_controls_and_both_sides_with_no_class_damage(self):
        qs=refs(4)
        for i,q in enumerate(qs):q['term_family_id']=f'T{i%2+1:02}'
        r=records(qs);p={'bN':'7','bU':'-3','QAS_factor':1}
        for i,q in enumerate(qs):
            y=1 if q['reference']=='无' else -1
            for order in ('MPS','MSP'):
                r[f'rrc-{q["query_id"]}-{order}/N']['m']=y*(3 if i<2 else 1)
                r[f'rrc-{q["query_id"]}-{order}/U']['m']=y*3
                for k in range(1,5):
                    for method in ('N','U'):r[f'rrc-{q["query_id"]}-{order}_replace_{k}/{method}']['m']=y*(1 if i<2 else 3)
                e,d=a.effects(q,order,1,p)
                self.assertEqual((e['CAD05']-F(3,2)*e['N']).bound(r),0)
                self.assertEqual((e['CAD05']-F(3,2)*e['N']).value(r),0)
                self.assertEqual(e['NOREF'].bound(r),0);self.assertEqual(e['NOREF'].value(r),0)
                self.assertEqual((e['U_CAL']-e['U']).value(r),0)
                self.assertEqual((e['OFFSET7']-e['N']).bound(r),0)
        result=a.bilateral(qs,r,p)
        self.assertEqual(result['status'],'limited_bilateral_evidence')
        self.assertEqual(result['groups']['helpful']['eligible_queries'],2)
        self.assertEqual(result['groups']['harmful']['eligible_terms'],2)
        self.assertEqual(a.bilateral(qs[:2],r,p)['status'],'one_sided_or_insufficient')
        r['rrc-D01-MPS/U']['m']=-10
        self.assertEqual(a.bilateral(qs,r,p)['status'],'mixed_or_negative_bilateral_result')

    def test_AV_exact_identity_route_gate_and_order_conflicts(self):
        qs=[{'query_id':f'J{i:02}','reference':'无'} for i in range(5,11)];r={}
        for q in qs:
            for order in ('MPS','MSP'):
                for key,value in {'N':0,'U':2,'P':0,'AV00':0,'AV01':0,'AV10':2,'AV11':2}.items():
                    r[f'jmix-{q["query_id"]}-{order}/{key}']={'m':value,'bound':1e-6}
        result=a.mechanism(r,qs);self.assertEqual(result['priority_candidates'],['reading_candidate'])
        for q in qs[:3]:r[f'jmix-{q["query_id"]}-MSP/AV01']['m']=2;r[f'jmix-{q["query_id"]}-MSP/AV10']['m']=0
        result=a.mechanism(r,qs);self.assertEqual(result['priority_candidates'],[])
        self.assertEqual(result['counts']['coupled_or_order_conflicting'],3)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    r=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
    out={'status':'PASS' if r.wasSuccessful() else 'FAIL','tests':r.testsRun,'synthetic_only':True,'research_forwards':0,
         'failures':[str(e) for _,e in r.errors+r.failures]}
    c.write(args.output,out);print(json.dumps(out));sys.exit(not r.wasSuccessful())
