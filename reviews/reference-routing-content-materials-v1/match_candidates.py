"""CPU tokenizer-only wording selection; does not evaluate the research model."""
import json
from pathlib import Path
from itertools import combinations
import runpy
import sys

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))
from diagnostics.case_attention_inputs_v1 import tokenizer

# Authored equivalent wording options. No meaningless padding or labels inserted.
# Each replacement can be applied once, only to a phrase in the original text.
OPTIONS=[
 ('已经',['已','现在已经']),('随后',['然后','随后又']),('接着',['随后','接下来']),
 ('最后',['最后又']),('先把',['首先把','先将']),('再把',['然后把','然后再把']),
 ('再记录',['随后记录','接下来记录']),('再由',['随后由']),('那个',['那一个']),
 ('这种',['这样的']),('是个',['是一个']),('一点也不过分',['并不过分','一点都不过分']),
 ('真是',['实在是','简直是']),('就是',['分明就是','根本就是']),
 ('根本不',['根本就不']),('我反对',['我明确反对','我坚决反对']),
 ('我不同意',['我不能同意','我明确不同意']),('不该',['不应该']),
 ('不值得',['根本不值得']),('没有必要',['完全没有必要']),('不能',['不可以']),
 ('请停止',['请立刻停止']),('请把',['请马上把']),('放在',['放置在']),
 ('核对',['仔细核对']),('检查',['仔细检查']),('清理',['仔细清理']),
 ('记录它',['仔细记录它']),('记录每日',['逐日记录每天']),
 ('新的',['新换的']),('刚洗完',['刚刚洗完']),('把水盆',['将水盆']),
 ('一点',['一些']),('并不合适',['很不合适']),('并不可取',['很不妥当']),
 ('我支持',['我明确支持']),('我要求',['我明确要求']),('我选',['我特意选']),
 ('就是为了',['就是特意为了']),('毫无',['完全没有']),
 ('并提醒',['还提醒']),('还提醒',['还特意提醒']),('提醒我',['提醒我要']),
 ('逐个',['一个一个']),('逐一',['一个一个']),('按顺序',['按照顺序']),
 ('按体型',['按照体型']),('按重量',['按照重量']),('按日期',['按照日期']),
 ('供参观',['供游客参观']),('每日',['每天']),('每月',['每个月']),
 ('并检查',['并且检查']),('并留出',['并且留出']),('并记录',['并且记录']),
 ('介绍',['详细介绍']),('展示',['展示出']),('用粉笔',['拿粉笔']),
 ('用软刷',['拿软刷']),('用薄纸',['用一层薄纸']),('调整',['重新调整']),
 ('放回',['放回到']),('摆在',['摆放在']),('装进',['装入']),('收在',['收进']),
 ('用于',['用来']),('统一',['统一安排']),('确认',['再次确认']),
 ('不是正常讨论',['并不属于正常讨论']),('并不公正',['很不公正']),
 ('很不妥',['很不妥当']),('不能赞成',['无法赞成']),
]

OPTIONS += runpy.run_path(str(HERE/'wording-options-02.py'))['EXTRA_OPTIONS']
OPTIONS += runpy.run_path(str(HERE/'wording-options-03.py'))['EXTRA_OPTIONS']
OPTIONS += runpy.run_path(str(HERE/'wording-options-04.py'))['EXTRA_OPTIONS']
OPTIONS += runpy.run_path(str(HERE/'wording-options-05.py'))['EXTRA_OPTIONS']

# Specific editorial correction; kept separate from the original authoring bytes.
EDITORIAL={'别把犯过错的工人骂成垃圾，指出操作问题就好，不能把整个人人格都否定掉。':
           '别把犯过错的工人骂成垃圾，指出操作问题就好，不能把整个人的人格都否定掉。'}

def variants(original):
    base=EDITORIAL.get(original,original)
    # Disjoint source spans avoid cascading edits and malformed duplicate modifiers.
    changes=[]
    for old,news in OPTIONS:
        at=base.find(old)
        if at>=0:
            for new in news: changes.append((at,at+len(old),old,new))
    choices=[([],base)]
    for a in range(len(changes)):
        combos=[(a,)]+[(a,b) for b in range(a+1,len(changes))]+[(a,b,c) for b,c in combinations(range(a+1,len(changes)),2)]
        for indices in combos:
            edits=sorted([changes[k] for k in indices])
            if any(x[1]>y[0] for x,y in zip(edits,edits[1:])): continue
            result=base
            for lo,hi,old,new in reversed(edits): result=result[:lo]+new+result[hi:]
            choices.append((edits,result))
    seen={}
    for edits,s in choices:
        if any(bad in s for bad in ['存放置在','认真仔细','毫无任何','为了看不起','本人分明就是','完全没有底线']): continue
        if s not in seen: seen[s]=edits
    return [{'text':s,'edits':[{'start':e[0],'end':e[1],'old':e[2],'new':e[3]} for e in es],
             'editorial_correction':base!=original} for s,es in seen.items()]

def main():
    source=runpy.run_path(str(HERE/'authoring-01.py'))
    tok=tokenizer()
    pools={}; misses=[]
    for name,texts in source['DEMOS'].items():
        all_options=[]
        for i,original in enumerate(texts):
            opts=variants(original)
            encoded=tok(['文本：'+o['text']+'\n答案：'+('无' if i%2==0 else '有') for o in opts],add_special_tokens=False)
            buckets={}
            for o,ids in zip(opts,encoded['input_ids']):
                count=len(ids); o['segment_tokens']=count
                rank=(len(o['edits']),sum(abs(len(e['old'])-len(e['new'])) for e in o['edits']),o['text'])
                if count not in buckets or rank < buckets[count][0]: buckets[count]=(rank,o)
            all_options.append(buckets)
        selected={}; counts={}
        for parity in (0,1):
            indices=list(range(parity,12,2))
            common=set.intersection(*(set(all_options[i]) for i in indices))
            if not common:
                misses.append({'pool':name,'parity':parity,'ranges':{str(i+1):sorted(all_options[i]) for i in indices}})
                continue
            n=min(common,key=lambda n:(sum(all_options[i][n][0][0] for i in indices),sum(all_options[i][n][0][1] for i in indices),n))
            counts[str(parity)]=n
            for i in indices:
                selected[str(i+1)]={'original_text':texts[i],**all_options[i][n][1]}
        pools[name]={'common_segment_tokens':counts,'selected':selected}
    out={'kind':'AI_wording_candidates_tokenizer_only_not_human_adoption','pools':pools,'unmatched':misses,'model_forwards':0,'torch_imported':'torch' in sys.modules}
    path=HERE/'wording-selection-05.json'
    with path.open('x',encoding='utf-8') as f: f.write(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'pools':len(pools),'unmatched':[{'pool':x['pool'],'parity':x['parity'],'ranges':{k:[min(v),max(v)] for k,v in x['ranges'].items()}} for x in misses],'selected_demos':sum(len(p['selected']) for p in pools.values()),'torch_imported':'torch' in sys.modules},ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()
