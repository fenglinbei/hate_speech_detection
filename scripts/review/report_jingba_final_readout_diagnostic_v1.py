#!/usr/bin/env python3
"""CPU-only final-readout decomposition; independent Decimal/longdouble verification."""
from pathlib import Path
from decimal import Decimal,localcontext
import sys,numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from diagnostics import jingba_query_complement_inputs_v1 as c
OUT=ROOT/'reviews/autonomous-reference-progress-20260921/readout-01'
def dec(x):return Decimal.from_float(float(x))
def main():
    assert not OUT.exists();OUT.mkdir(parents=True)
    data=c.read(c.WORK/'results-01/results.json');rows=[];sources=[c.info(c.WORK/'results-01/manifest.json'),c.info(c.WORK/'results-01/results.json'),c.info(Path(__file__))]
    for r in data['query_interactions']:
        ts={};ds={}
        for name,stage,key in [('N','native-production',r['recipient']),('U','production',r['focal_job_id']),('C','production',r['complement_job_id']),('W','production',r['whole_job_id'])]:
            rec=c.read(c.WORK/'run-01/records'/stage/(key+'.json'));c.verify(rec['trajectory']);sources.append(rec['trajectory'])
            with np.load(rec['trajectory']['path'],allow_pickle=False) as z:
                h=z['states'][-1,2].astype(np.longdouble);nw=z['norm_weight'].astype(np.longdouble);lw=z['label_weights'].astype(np.longdouble);eps=np.longdouble(z['norm_eps'][0]);weight=(lw[1]-lw[0])*nw
                num=np.sum(h*weight);rms=np.sqrt(np.mean(h*h)+eps);ts[name]=(num,rms)
                with localcontext() as ctx:
                    ctx.prec=120
                    nums=sum(dec(hh)*(dec(wn)-dec(wy))*dec(nn) for hh,wn,wy,nn in zip(h,lw[1],lw[0],nw))
                    rmss=(sum(dec(hh)**2 for hh in h)/len(h)+dec(eps)).sqrt();ds[name]=(nums,rmss)
                    assert abs(nums-dec(num))<Decimal('1e-10') and abs(rmss-dec(rms))<Decimal('1e-10')
        signs={'W':1,'U':-1,'C':-1,'N':1};ideal=sum(signs[k]*ts[k][0]/ts[k][1] for k in ts);common=sum(signs[k]*ts[k][0]/ts['N'][1] for k in ts)
        with localcontext() as ctx:
            ctx.prec=120;dc=sum(signs[k]*ds[k][0]/ds['N'][1] for k in ts);di=sum(signs[k]*ds[k][0]/ds[k][1] for k in ts)
            assert abs(dc-dec(common))<Decimal('1e-10') and abs(di-dec(ideal))<Decimal('1e-10')
        scaling=ideal-common;rest=np.longdouble(r['interaction_m'])-ideal
        assert abs(common+scaling+rest-r['interaction_m'])<1e-12
        rows.append({'query_id':r['query_id'],'recipient':r['recipient'],'I_m':r['interaction_m'],'common_native_RMS':float(common),'final_RMS_rescaling':float(scaling),'floating_remainder':float(rest),'RMS':{k:float(v[1]) for k,v in ts.items()}})
    c.write(OUT/'diagnostic.json',{'status':'pass','CPU_only':True,'GPU_forwards':0,'Decimal_precision':120,'records':rows,'sources':sources})
    text=['# 最终答案读出的缩放检查','',
      'J06普通义→贬损义方向的组合差为1.721081；把四种运行统一用接收条件原生的最终RMS缩放，差值仍约1.695546。因此，该差异不能完全归因于最后一步RMS归一化。此结论只排除一个读数解释，不指认哪一段文字、哪一层或哪个头负责。','',
      '对N/U/C/W的最后答案前状态h，先计算未除RMS的答案方向分子，再分别比较各自RMS与共同原生RMS。共同尺度项+缩放变化项+实际FP32读出余差=原I。120位十进制独立计算分子与RMS，与扩展精度实现逐项核对。它不是对归一化的模型干预。','',
      '此前J06第35层MLP新增差的RMS分解比较层内前后状态；这里比较最终四种运行，问题与分母不同，不能据此否定此前结果。','',
      '| 查询 | 接收 | 原I | 共同原生RMS | 最终缩放差 | 数值余差 |','|---|---|---:|---:|---:|---:|']
    for r in rows:text.append('| '+r['query_id']+' | '+r['recipient'][-3:]+' | '+' | '.join(f'{r[k]:+.6f}' for k in ['I_m','common_native_RMS','final_RMS_rescaling','floating_remainder'])+' |')
    text+=['','下一步只补词前B、词后A及各自与目标词联合UB/UA。保留六条、双向和全36层。这项选择在新模型输出前记录，不按新结果选层、方向或案例。']
    (OUT/'REPORT.md').write_text('\n'.join(text))
    c.write(OUT/'manifest.json',{'sources':sources,'artifacts':[c.info(p) for p in sorted(OUT.iterdir()) if p.is_file()]})
    print({'status':'pass','records':len(rows),'output':str(OUT)})
if __name__=='__main__':main()
