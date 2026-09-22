"""Exact coefficient bookkeeping, frozen development selection and bilateral rules.

Fractions preserve the exact binary floats in recorded margins. Shared physical
scores cancel before engineering bounds are propagated. No inferential p values.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from fractions import Fraction as F
from collections import Counter, defaultdict
import math
import statistics
from diagnostics.case_attention_inputs_v1 import require

METHODS=('N','NOREF','U','OFFSET7','CAD05','QAS','N_CAL','U_CAL')
BASELINES=('N','NOREF','OFFSET7','CAD05','QAS','N_CAL')


def fraction(x):return x if isinstance(x,F) else F(x)


@dataclass
class Score:
    coeff:dict=field(default_factory=dict)
    constant:F=F(0)
    def __add__(self,other):
        if not isinstance(other,Score):other=Score(constant=fraction(other))
        d=dict(self.coeff)
        for k,v in other.coeff.items():d[k]=d.get(k,F(0))+v
        return Score({k:v for k,v in d.items() if v},self.constant+other.constant)
    __radd__=__add__
    def __mul__(self,amount):
        amount=fraction(amount)
        return Score({k:v*amount for k,v in self.coeff.items() if v*amount},self.constant*amount)
    __rmul__=__mul__
    def __sub__(self,other):return self+(-1*other if isinstance(other,Score) else -fraction(other))
    def value(self,records):return self.constant+sum((v*fraction(records[k]['m']) for k,v in self.coeff.items()),F(0))
    def bound(self,records):return sum((abs(v)*fraction(records[k]['bound']) for k,v in self.coeff.items()),F(0))
    def summary(self,records):
        v,b=self.value(records),self.bound(records)
        return {'value':float(v),'bound':float(b),'lower':float(v-b),'upper':float(v+b),
                'exact_value':str(v),'exact_bound':str(b),'coefficients':{k:str(v) for k,v in self.coeff.items()},
                'constant':str(self.constant),'direction':'positive' if v>b else 'negative' if v<-b else 'exact_zero' if v==b==0 else 'unresolved'}


def mean(scores):
    scores=list(scores);require(scores,'Empty score aggregation')
    return sum(scores,Score())*F(1,len(scores))


def physical(key):return Score({key:F(1)})


def lower(summary):return F(summary['exact_value'])-F(summary['exact_bound'])


def prediction(score,records):
    v,b=score.value(records),score.bound(records)
    return '无' if v>b else '有' if v<-b else None


def methods(qid,condition,parameters):
    prefix=f'rrc-{qid}-{condition}/';zero=physical(f'rrc-{qid}-M00/N')
    n,u=physical(prefix+'N'),physical(prefix+'U');f=parameters['QAS_factor']
    return {'N':n,'NOREF':zero,'U':u,'OFFSET7':n+7,'CAD05':F(3,2)*n-F(1,2)*zero,
            'QAS':n if f==1 else physical(prefix+f'Q{f}'),
            'N_CAL':n+F(parameters['bN']),'U_CAL':u+F(parameters['bU'])}


def weights(rows):
    """Order mean inside query, class mean inside term, then terms equally."""
    cells=defaultdict(lambda:defaultdict(lambda:defaultdict(list)))
    for row in rows:cells[row['term_family_id']][row['reference']][row['query_id']].append(row)
    result=[]
    for term,classes in sorted(cells.items()):
        for label,queries in sorted(classes.items()):
            for qid,items in sorted(queries.items()):
                w=F(1,len(cells)*len(classes)*len(queries)*len(items))
                result.extend((r,w) for r in items)
    require(sum(w for _,w in result)==1,'Weights do not sum to one')
    return result


def accuracy(rows,records):
    return sum((w for r,w in weights(rows) if prediction(r['score'],records)==r['reference']),F(0))


def transitions(rows,records):
    counts=Counter();wrong=correct=0;damaged=set();adverse=set()
    for r in rows:
        n=prediction(r['native'],records);p=prediction(r['score'],records);gold=r['reference']
        wrong+=n is not None and n!=gold;correct+=n==gold
        if n is None or p is None:kind='unresolved'
        elif n!=gold and p==gold:kind='repair'
        elif n==gold and p!=gold:kind='damage';damaged.add(r['query_id'])
        elif n==gold:kind='keep_correct'
        else:kind='still_wrong'
        counts[kind]+=1
        if n==gold and p!=gold:adverse.add(r['query_id'])
        if n==gold and p is None:counts['correct_to_unresolved']+=1
    return {'counts':{k:counts[k] for k in ('repair','damage','keep_correct','still_wrong','unresolved','correct_to_unresolved')},
            'eligible_native_wrong':wrong,'eligible_native_correct':correct,
            'repair_rate':counts['repair']/wrong if wrong else None,'damage_rate':counts['damage']/correct if correct else None,
            'any_order_damage_queries':sorted(damaged),'any_order_correct_to_wrong_or_unresolved_queries':sorted(adverse)}


def auc(rows,records):
    by=defaultdict(list)
    for r in rows:by[r['condition']].append(r)
    result={}
    for condition,items in by.items():
        yes=[r for r in items if r['reference']=='有'];no=[r for r in items if r['reference']=='无']
        if not yes or not no:continue
        estimates=[];lower=[];upper=[]
        for a in no:
            for b in yes:
                d=a['score']-b['score'];v,e=d.value(records),d.bound(records)
                estimates.append(F(1) if v>0 else F(0) if v<0 else F(1,2))
                if v>e:lo=hi=F(1)
                elif v<-e:lo=hi=F(0)
                elif v==e==0:lo=hi=F(1,2)
                else:lo,hi=F(0),F(1)
                lower.append(lo);upper.append(hi)
        result[condition]={'value':float(sum(estimates)/len(estimates)),
            'engineering_lower':float(sum(lower)/len(lower)),'engineering_upper':float(sum(upper)/len(upper)),
            'pairs':len(estimates)}
    return {'by_condition':result,'order_mean':{k:sum(x[k] for x in result.values())/len(result) for k in ('value','engineering_lower','engineering_upper')} if result else None}


def aggregate(rows,records):
    require(rows,'No rows to aggregate')
    gains=[r['gain'] for r in rows];values=[g.value(records) for g in gains]
    weighted=sum((r['gain']*w for r,w in weights(rows)),Score())
    labelmeans={}
    for label in ('无','有'):
        subset=[r for r in rows if r['reference']==label]
        if subset:labelmeans[label]=sum((r['gain']*w for r,w in weights(subset)),Score()).summary(records)
    return {'endpoints':len(rows),'queries':len({r['query_id'] for r in rows}),
        'balanced_accuracy':float(accuracy(rows,records)),'balanced_accuracy_exact':str(accuracy(rows,records)),
        'accuracy':sum(prediction(r['score'],records)==r['reference'] for r in rows)/len(rows),
        'class_accuracy':{label:float(accuracy([r for r in rows if r['reference']==label],records)) for label in ('无','有') if any(r['reference']==label for r in rows)},
        'class_accuracy_exact':{label:str(accuracy([r for r in rows if r['reference']==label],records)) for label in ('无','有') if any(r['reference']==label for r in rows)},
        'balanced_G':weighted.summary(records),'per_class_G':labelmeans,
        'all_endpoint_mean_G':mean(gains).summary(records),'median_G':float(statistics.median(values)),
        'min_G':float(min(values)),'max_G':float(max(values)),
        'G_directions':dict(Counter(g.summary(records)['direction'] for g in gains)),
        'transitions':transitions(rows,records),'AUC':auc(rows,records)}


def primary_rows(references,records,parameters,method):
    result=[]
    for q in references:
        for order in ('MPS','MSP'):
            ms=methods(q['query_id'],order,parameters);s=ms[method];y=1 if q['reference']=='无' else -1
            result.append(dict(q,condition=order,score=s,native=ms['N'],gain=y*(s-ms['N'])))
    return result


def fit(references,records):
    require(len(references)==24 and all(q['split']=='development' for q in references),'Fit uses exactly development queries')
    parameters={'bN':'0','bU':'0','QAS_factor':1};tables={}
    for method,field in (('N','bN'),('U','bU')):
        rows=primary_rows(references,records,parameters,method)
        margins=sorted({r['score'].value(records) for r in rows})
        candidates={F(0),-max(margins)-1,-min(margins)+1}
        candidates|={-(a+b)/2 for a,b in zip(margins,margins[1:])}
        table=[]
        for offset in sorted(candidates):
            shifted=[dict(r,score=r['score']+offset) for r in rows]
            table.append({'offset':str(offset),'BA':str(accuracy(shifted,records))})
        selected=min(table,key=lambda x:(-F(x['BA']),abs(F(x['offset'])),F(x['offset'])))
        parameters[field]=selected['offset'];tables[field]={'candidates':table,'selected':selected}
    qs=[]
    for f in (1,2,4):
        p=dict(parameters,QAS_factor=f);rows=primary_rows(references,records,p,'QAS')
        qs.append({'factor':f,'BA':str(accuracy(rows,records))})
    parameters['QAS_factor']=min(qs,key=lambda x:(-F(x['BA']),x['factor']))['factor'];tables['QAS']=qs
    baselines=[]
    for i,method in enumerate(BASELINES):
        rows=primary_rows(references,records,parameters,method)
        baselines.append({'method':method,'BA':str(accuracy(rows,records)),
                          'damage':transitions(rows,records)['counts']['damage'],'tie_order':i})
    parameters['Bstar']=min(baselines,key=lambda x:(-F(x['BA']),x['damage'],x['tie_order']))['method']
    tables['Bstar']=baselines
    return parameters,tables


def effects(q,order,slot,parameters):
    full=methods(q['query_id'],order,parameters);replacement=methods(q['query_id'],f'{order}_replace_{slot}',parameters)
    y=1 if q['reference']=='无' else -1
    e={method:y*(full[method]-replacement[method]) for method in METHODS}
    d={method:e[method]-e['N'] for method in METHODS}
    return e,d


def bilateral(references,records,parameters):
    rows=[]
    for q in references:
        for order in ('MPS','MSP'):
            for slot in range(1,5):
                e,d=effects(q,order,slot,parameters);n=e['N'];v,b=n.value(records),n.bound(records)
                group='helpful' if v-b>F(1,2) else 'harmful' if v+b<-F(1,2) else 'small_or_unresolved'
                uv,ub=e['U'].value(records),e['U'].bound(records);dv,db=d['U'].value(records),d['U'].bound(records)
                ms=methods(q['query_id'],order,parameters);npred,upred=prediction(ms['N'],records),prediction(ms['U'],records)
                rows.append(dict(q,order=order,slot=slot,behavioral_group=group,e=e,d=d,
                    preserved=uv-ub>0 and dv-db>=-F(1,2),attenuated=dv-db>F(1,2),
                    harmful_residual_band='within_small_harm_band' if uv-ub>=-F(1,2) else 'still_negative_or_unresolved',
                    worsened=dv+db<0,whole_prompt_new_adverse=npred==q['reference'] and upred!=q['reference']))
    summaries={};querygroups={}
    for group in ('helpful','harmful','small_or_unresolved'):
        subset=[r for r in rows if r['behavioral_group']==group];by=defaultdict(list)
        for r in subset:by[r['query_id']].append(r)
        qr=[]
        for qid,items in by.items():
            qr.append({'query_id':qid,'term':items[0]['term_family_id'],'D':mean(r['d']['U'] for r in items),
                       'all_preserved':all(r['preserved'] for r in items),'new_adverse':any(r['whole_prompt_new_adverse'] for r in items),
                       'eligible_slots_and_orders':len(items)})
        terms=defaultdict(list)
        for r in qr:terms[r['term']].append(r['D'])
        effect=mean(mean(v) for v in terms.values()) if terms else None
        querygroups[group]=qr
        summaries[group]={'eligible_endpoints':len(subset),'eligible_queries':len(qr),'eligible_terms':len(terms),
            'mean_D_U':effect.summary(records) if effect else None,
            'query_results':[{**{k:v for k,v in r.items() if k!='D'},'D':r['D'].summary(records)} for r in qr]}
    enough=all(summaries[g]['eligible_queries']>=2 and summaries[g]['eligible_terms']>=2 for g in ('helpful','harmful'))
    gates={}
    if enough:
        h=querygroups['harmful'];p=querygroups['helpful']
        gates={'harm_mean_D_lower_above_delta':lower(summaries['harmful']['mean_D_U'])>F(1,2),
               'harm_positive_query_fraction_above_half':sum(r['D'].value(records)>r['D'].bound(records) for r in h)*2>len(h),
               'help_mean_D_lower_at_least_negative_delta':lower(summaries['helpful']['mean_D_U'])>=-F(1,2),
               'help_all_slot_preservation_queries_at_least_80pct':sum(r['all_preserved'] for r in p)*5>=4*len(p),
               'help_no_new_wrong_or_unresolved':not any(r['new_adverse'] for r in p)}
    status='one_sided_or_insufficient' if not enough else 'limited_bilateral_evidence' if all(gates.values()) else 'mixed_or_negative_bilateral_result'
    exported=[]
    for r in rows:
        exported.append({**{k:v for k,v in r.items() if k not in ('e','d')},
            'E':{k:v.summary(records) for k,v in r['e'].items()},'D':{k:v.summary(records) for k,v in r['d'].items()}})
    return {'status':status,'gates':gates,'groups':summaries,'rows':exported,'delta':.5,'automatic_next_run':False}


def task_decision(primary,parameters,cost_complete):
    a,b=primary['U_CAL'],primary[parameters['Bstar']]
    gates={'BA_gain_at_least_1_over_24':F(a['balanced_accuracy_exact'])-F(b['balanced_accuracy_exact'])>=F(1,24),
        'both_classes_noninferior':all(F(a['class_accuracy_exact'][k])>=F(b['class_accuracy_exact'][k]) for k in ('无','有')),
        'damage_endpoints_noninferior':a['transitions']['counts']['damage']<=b['transitions']['counts']['damage'],
        'damage_query_union_noninferior':len(a['transitions']['any_order_damage_queries'])<=len(b['transitions']['any_order_damage_queries']),
        'balanced_G_lower_above_zero':lower(a['balanced_G'])>0,'cost_and_subgroups_complete':cost_complete}
    passed=all(gates.values())
    status='candidate_for_larger_validation' if passed else 'margin_gain_without_task_advantage' if gates['balanced_G_lower_above_zero'] else 'no_preregistered_task_advantage'
    return {'status':status,'gates':gates,'candidate':'U_CAL','Bstar':parameters['Bstar'],
        'calibration_assisted_accuracy_gain':F(a['balanced_accuracy_exact'])>F(primary['U']['balanced_accuracy_exact']),
        'automatic_next_run':False,'resource_rule_not_significance_test':True}


def mechanism(records,references):
    rows=[];by=defaultdict(list)
    for q in references:
        qid=q['query_id'];y=1 if q['reference']=='无' else -1
        for order in ('MPS','MSP'):
            prefix=f'jmix-{qid}-{order}/';cells={k:physical(prefix+k) for k in ('N','U','P','AV00','AV01','AV10','AV11')}
            ea=cells['AV11']-cells['AV01'];ev=cells['AV11']-cells['AV10'];joint=cells['AV11']-cells['AV00']
            interaction=cells['AV11']-cells['AV10']-cells['AV01']+cells['AV00']
            require((joint-ea-ev+interaction).coeff=={} and (joint-ea-ev+interaction).constant==0,'Factorial identity')
            def large(x):return abs(x.value(records))-x.bound(records)>F(1,2)
            def small(x):return abs(x.value(records))+x.bound(records)<=F(1,2)
            reading=large(ea) and small(ev) and small(interaction) and ea.value(records)*joint.value(records)>0
            content=large(ev) and small(ea) and small(interaction) and ev.value(records)*joint.value(records)>0
            state='reading_candidate' if reading else 'content_candidate' if content else 'coupled_or_background_dependent' if (large(ea) and large(ev)) or large(interaction) else 'limited_evidence_at_this_site'
            quantities={'E_A':ea,'E_V':ev,'E_joint':joint,'I_AV':interaction,'A_in_N_V':cells['AV10']-cells['AV00'],'V_in_N_A':cells['AV01']-cells['AV00']}
            rows.append({'query_id':qid,'order':order,'state':state,
                'contrasts':{k:{'raw':v.summary(records),'gold_aligned':(y*v).summary(records)} for k,v in quantities.items()},
                'margins':{k:v.summary(records) for k,v in cells.items()},'G':{k:(y*(v-cells['N'])).summary(records) for k,v in cells.items()}})
            by[qid].append(state)
    queries={qid:states[0] if len(set(states))==1 else 'coupled_or_order_conflicting' for qid,states in by.items()}
    counts=Counter(queries.values());priority=[k for k in ('reading_candidate','content_candidate') if counts[k]>=4]
    return {'rows':rows,'query_decisions':queries,'counts':dict(counts),'priority_candidates':priority,
            'all_six_queries_retained':len(by)==6,'automatic_next_run':False}
