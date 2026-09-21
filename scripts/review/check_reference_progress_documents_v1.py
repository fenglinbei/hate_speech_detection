#!/usr/bin/env python3
"""Read-only check of scientific pins, public report links, and figure readability."""
from pathlib import Path
import argparse,re,sys,json
from PIL import Image
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from diagnostics.case_attention_inputs_v1 import read,write,info,verify,require
SEEN={
 'jingba-query-regions-v1':{'regions-effects.png','regions-factorial-all36.png'},
 'dictionary-free-donor-v1':{'fixed-rule-effects.png','all36-J.png'},
 'jingba-mixed-demos-v1':{'native-margins.png','all36.png'},
 'jingba-demo-donor-v1':{'effects.png','all36.png'},
}

def main(name):
 w=ROOT/'reviews'/name;p=ROOT/'docs/research/experiment-plans'/name;require(not (w/'document-check-01.json').exists(),'New receipt required');pins=0
 for folder in ['prepared-01','results-01','interpretation-01','closeout-01']:
  m=read(w/folder/'manifest.json')
  for x in m['artifacts']+m.get('sources',[]):verify(x);pins+=1
 for x in read(w/'prepared-01/source-ledger.json')['files']:verify(x);pins+=1
 old=read(w/'launch-01/parent-selectors.json')['files']
 for x in old:verify(x)
 selector=read(p/'results-current.json');require(selector['status']=='complete' and selector['terminal_do_not_restart'],'Wrong result selector');verify(selector['closeout_manifest'])
 paths=[w/'results-01/REPORT.md',w/'interpretation-01/REPORT.md',w/'prepared-01/PROTOCOL.md',p/'README.md'];links=0
 for path in paths:
  for match in re.finditer(r'\]\(([^)]+)\)',path.read_text()):
   ref=match.group(1).strip('<>')
   if ref.startswith(('http:','https:','#')):continue
   dest=ref.split('#')[0];target=Path(dest) if dest.startswith('/') else path.parent/dest
   require(target.exists(),str(path)+' missing '+ref);links+=1
 figures=[]
 for path in sorted((w/'results-01/figures').glob('*.png')):
  with Image.open(path) as im:size=list(im.size);im.verify()
  require(min(size)>500,'Unreadable figure dimensions');figures.append({'file':info(path),'size':size,'visually_inspected':path.name in SEEN[name]})
 data=read(w/'results-01/results.json');audit=read(w/'result-audit-01.json');require(audit['status']=='pass','Audit failure')
 require(len(data['baselines']) in [18,30,36] and data['layers']==36,'Results inventory');refs={r['query_id']:r['reference'] for r in data['baselines']}
 require(all(r['raw_reference_correct']==(r['raw_prediction']==refs[r['query_id']]) for r in data['baselines']),'Counts mismatch')
 if name=='dictionary-free-donor-v1':
  rs=data['donor_comparisons'];require(sum(r['U0_effect']>0 for r in rs)==9 and sum(r['U0_effect']<0 for r in rs)==3,'Signed effect count')
  require(all(r['U0_prediction']==r['U2_prediction']==r['CPU_shifted_prediction'] for r in rs),'Prediction claim')
 if name=='jingba-mixed-demos-v1':
  by={(r['query_id'],r['condition']):r for r in data['baselines']};require(.50<by['J08','MSP']['pair_support_no']<.51,'Near-boundary probability')
 if name=='jingba-query-regions-v1':require(all(e['transition']=='unchanged' for e in data['effects']),'Region label claim')
 receipt={'status':'pass','artifact_and_source_pins_verified':pins,'old_scientific_and_website_selectors_unchanged':len(old),'links_checked':links,'figures':figures,'derived_counts_checked':True,'readable_report':info(w/'interpretation-01/REPORT.md'),'public_entry':info(p/'README.md'),'GPU_forwards':0,'checker':info(Path(__file__))};write(w/'document-check-01.json',receipt);print(json.dumps({k:v for k,v in receipt.items() if k!='figures'},ensure_ascii=False))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('name',choices=list(SEEN));a=p.parse_args();main(a.name)
