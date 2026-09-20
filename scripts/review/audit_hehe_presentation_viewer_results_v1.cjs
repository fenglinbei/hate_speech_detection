// Post-release browser verification. Requires real sealed result files; no GPU.
'use strict';
const fs=require('fs'),path=require('path'),assert=require('assert'),crypto=require('crypto');
const {chromium}=require('playwright');
const root=path.resolve(__dirname,'../..');
const results=path.resolve(process.argv[2]||path.join(root,'reviews/hehe-presentation-mechanism-v1/results-01'));
const output=process.argv[3];
if(!output)throw Error('Provide a new receipt output path');
const summary=JSON.parse(fs.readFileSync(path.join(results,'mechanism-summary.json'),'utf8'));
assert.equal(summary.records.length,18);
const attentionHTML=fs.readFileSync(path.join(results,'index.html'),'utf8');
const data=JSON.parse(attentionHTML.match(/<script id="data" type="application\/json">([\s\S]*?)<\/script>/)[1]);
assert.equal(data.requests.length,18);
const pin=p=>({path:p,bytes:fs.statSync(p).size,sha256:crypto.createHash('sha256').update(fs.readFileSync(p)).digest('hex')});
(async()=>{
  let browser;
  const errors=[];let attentionValues=0,mechanismValues=0,NA=0;
  try{
    browser=await chromium.launch({executablePath:'/tmp/paired-review-playwright/chromium_headless_shell-1234/chrome-headless-shell-linux64/chrome-headless-shell',headless:true,args:['--no-sandbox','--disable-gpu']});
    // One page per condition bounds cache memory; one additional pair checks
    // difference signs/components. Each file already has an independent CPU audit.
    for(const req of data.requests){
      const page=await browser.newPage({viewport:{width:1440,height:1000}});page.on('pageerror',e=>errors.push(e.message));
      await page.goto('file://'+path.join(results,'index.html'));await page.waitForFunction(()=>window.viewerState);
      await page.selectOption('#query',req.query_id);await page.selectOption('#condition',req.request_id);
      assert.equal(await page.locator('#prompt').textContent(),req.prompt_text);
      const artifact=JSON.parse(fs.readFileSync(path.join(results,req.request_id+'.view.json'),'utf8'));
      await page.setInputFiles('#files',path.join(results,req.request_id+'.view.json'));
      await page.waitForFunction(rid=>window.viewerState.loaded&&window.viewerState.request===rid,req.request_id);
      for(const role of ['pre_answer','query_focal','lexicon_end','demos_end'])for(const metric of ['mass','density']){
        await page.selectOption('#role',role);await page.selectOption('#metric',metric);await page.selectOption('#parts','parts');
        for(const head of ['-1','0','31']){
          await page.selectOption('#head',head);
          const visibleSpans=req.spans.filter(s=>['demo_text','demo_answer','lexicon_term','lexicon_definition','lexicon_definition_component','lexicon_category','system','query','structure','boundary'].includes(s.kind));
          const actual=await page.locator('#mainheat tbody tr td').evaluateAll(xs=>xs.map(x=>x.dataset.value===''?null:Number(x.dataset.value)));
          const expected=[];
          for(let l=0;l<36;l++)for(const s of visibleSpans){const a=artifact.aggregates.find(x=>x.role===role&&x.span_id===s.id);expected.push(!a?.visible?null:head==='-1'?a[metric][l].reduce((x,y)=>x+y,0)/32:a[metric][l][Number(head)])}
          assert.deepEqual(actual,expected,req.request_id+' '+role+' '+metric+' head '+head);attentionValues+=actual.filter(x=>x!==null).length;NA+=actual.filter(x=>x===null).length;
        }
      }
      if(req.request_id==='hpm-Q01-D04'){
        const compare=JSON.parse(fs.readFileSync(path.join(results,'hpm-Q01-D03.view.json'),'utf8'));
        await page.setInputFiles('#files',[path.join(results,req.request_id+'.view.json'),path.join(results,'hpm-Q01-D03.view.json')]);
        await page.selectOption('#role','pre_answer');await page.selectOption('#metric','density');await page.selectOption('#head','-1');await page.selectOption('#compare','hpm-Q01-D03');
        await page.waitForFunction(()=>document.getElementById('mainheat')._export.rows.some(row=>row.some(Number.isFinite)));
        const spans=req.spans.filter(s=>['lexicon_term','lexicon_definition','lexicon_definition_component','system','query','structure','boundary'].includes(s.kind));
        const actual=await page.locator('#mainheat tbody tr td').evaluateAll(xs=>xs.map(x=>x.dataset.value===''?null:Number(x.dataset.value))),expected=[];
        for(let l=0;l<36;l++)for(const s of spans){const a=artifact.aggregates.find(x=>x.role==='pre_answer'&&x.span_id===s.id),b=compare.aggregates.find(x=>x.role==='pre_answer'&&x.span_id===s.id);expected.push(a?.visible&&b?.visible?a.density[l].reduce((x,y)=>x+y,0)/32-b.density[l].reduce((x,y)=>x+y,0)/32:null)}
        assert.deepEqual(actual,expected,'Order comparison sign or density differs');attentionValues+=actual.filter(x=>x!==null).length;
      }
      await page.close();
    }
    const page=await browser.newPage({viewport:{width:1440,height:1000}});page.on('pageerror',e=>errors.push(e.message));
    await page.goto('file://'+path.join(results,'mechanism.html'));await page.waitForFunction(()=>window.HPM);
    assert.deepEqual(await page.evaluate(()=>HPM.DATA),summary,'Embedded mechanism data differs from audited JSON');
    mechanismValues=await page.evaluate(()=>{
      let checked=0;const $=id=>document.getElementById(id),ok=(x,m)=>{if(!x)throw Error(m)},dots=id=>[...document.querySelectorAll('#'+id+' circle')];
      function values(id,series){const all=dots(id);let n=0;for(const s of series){const found=all.filter(x=>x.dataset.series===s.label);if(s.values===null){ok(!found.length,'NA is plotted');continue}const wanted=s.values.filter(Number.isFinite).length;ok(found.length===wanted,'Layer omitted or invented');n+=wanted;for(const d of found){ok(Number(d.dataset.value)===s.values[Number(d.dataset.layer)],id+' value mismatch');checked++}}ok(all.length===n,'Unexpected plotted series')}
      for(const r of HPM.DATA.records){$('request').value=r.request_id;HPM.changeRequest();values('lens',r.sites.map((label,i)=>({label,values:r.pre_answer_probe_margin_by_site.map(row=>row[i])})));values('branches',[{label:'注意力更新对应变化',values:r.pre_answer_attention_probe_change},{label:'MLP更新对应变化',values:r.pre_answer_mlp_probe_change}]);
        for(const p of [...new Set(r.source_rows.map(x=>x.target_position))])for(const metric of ['mean_head_mass','mean_head_density','mean_head_AV_norm','source_output_norm','local_direction_projection']){$('target').value=String(p);$('metric').value=metric;HPM.changeSource();values('source',r.source_rows.filter(x=>x.target_position===p).map(x=>({label:x.label,values:x[metric]})))}}
      for(const pair of HPM.DATA.representation_pairs){$('pair').value=pair.left+'|'+pair.right;HPM.changePair();$('role').value=pair.role+':'+pair.token_index;for(const metric of ['l2','relative_l2','cosine']){$('distance').value=metric;HPM.changeDistance();values('representations',[{label:$('distance').selectedOptions[0].textContent,values:pair[metric]}])}}
      return checked;
    });
    const download=page.waitForEvent('download');await page.locator('#lens button').click();const item=await download;const exported=output+'.svg';await item.saveAs(exported);assert(fs.readFileSync(exported,'utf8').startsWith('<svg'));
    await page.screenshot({path:output+'.png',fullPage:false});await page.setViewportSize({width:390,height:844});assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+2),'Mobile overflow');
    assert.deepEqual(errors,[]);
    const receipt={status:'pass',real_conditions:18,attention_values_checked:attentionValues,mechanism_values_checked:mechanismValues,NA_cells_checked:NA,
      raw_JSON_equals_embedded_data:true,order_difference_density_sign:true,SVG_and_mobile:true,page_errors:errors,browser_version:browser.version(),node_version:process.version,
      sources:[pin(__filename),pin(path.join(results,'manifest.json')),pin(path.join(results,'mechanism-summary.json')),pin(exported)]};
    fs.writeFileSync(output,JSON.stringify(receipt,null,2)+'\n',{flag:'wx'});console.log(JSON.stringify(receipt));
  }finally{if(browser)await browser.close()}
})().catch(e=>{console.error(e);process.exit(1)});
