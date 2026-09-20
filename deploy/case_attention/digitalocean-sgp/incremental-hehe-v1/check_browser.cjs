// Read-only checks against original scientific aggregates, locally or over HTTPS.
const fs=require('fs'),path=require('path'),assert=require('assert');
const {chromium}=require('playwright');
const root=path.resolve(__dirname,'../../../..');
const [base,output,credentialsFile]=process.argv.slice(2);
assert(base&&output);fs.mkdirSync(output,{recursive:true});
const read=p=>JSON.parse(fs.readFileSync(p,'utf8'));
const originals={};for(const [key,exp] of [['original','case-attention-v1'],['replacement','case-content-replacement-v1'],['hehe','hehe-sense-context-v1']]) {
  const folder=path.join(root,'reviews',exp,'results-01');
  originals[key]={folder,data:JSON.parse(fs.readFileSync(path.join(folder,'index.html'),'utf8').match(/<script id="data" type="application\/json">([\s\S]*?)<\/script>/)[1])};
}
const aggCache=new Map();
function aggregate(key,id){if(!aggCache.has(id)){aggCache.set(id,read(path.join(originals[key].folder,id+'-aggregates.json')).aggregates);if(aggCache.size>4)aggCache.delete(aggCache.keys().next().value)}return aggCache.get(id)}
const avg=x=>x.reduce((a,b)=>a+b,0)/x.length;
function expected(snapshot,span,layer,head) {
  function get(id,sid,role){const a=aggregate(snapshot.round,id).find(x=>x.role===role&&x.span_id===sid);if(!a?.visible)return null;
    const rows=a[snapshot.metric];return snapshot.mode==='mean'?avg(rows.map(xs=>head<0?avg(xs):xs[head])):head<0?avg(rows[layer]):rows[layer][head];}
  const current=get(snapshot.request,span.id,snapshot.role);if(!snapshot.compare)return current;
  let sid=span.id;
  if(snapshot.alignment==='slot'&&span.kind.startsWith('demo')){const m=span.label.match(/^示例(\d+) /);if(m){const cr=originals[snapshot.round].data.requests.find(x=>x.request_id===snapshot.compare);sid=cr.spans.find(x=>x.kind===span.kind&&x.label.startsWith('示例'+m[1]+' '))?.id||'missing'}}
  const other=get(snapshot.compare,sid,snapshot.compareRole);return current===null||other===null?null:current-other;
}
(async()=>{
  let browser;const errors=[],payloads=[],checks=[],timings={};let checkedValues=0;
  try {
    browser=await chromium.launch({executablePath:'/tmp/paired-review-playwright/chromium_headless_shell-1234/chrome-headless-shell-linux64/chrome-headless-shell',headless:true,
      args:['--no-sandbox','--disable-gpu',...(credentialsFile?['--proxy-server=direct://','--host-resolver-rules=MAP hsd.fenglin.pro 165.22.48.237']:[])]});
    const context=await browser.newContext({viewport:{width:1600,height:1150},...(credentialsFile?{httpCredentials:read(credentialsFile)}:{})});
    const page=await context.newPage();page.setDefaultTimeout(180000);page.on('pageerror',e=>errors.push(e.message));
    page.on('response',r=>{if(r.status()===200&&/\.(f32|f64|meta\.json)$/.test(r.url()))payloads.push(new URL(r.url()).pathname)});
    const wait=async match=>page.waitForFunction(match=>window.viewerState?.loaded&&Object.entries(match||{}).every(([k,v])=>window.viewerState[k]===v),match);
    const change=async(id,v,match={})=>{await page.selectOption('#'+id,String(v));await wait(match)};
    const click=async(id,match={})=>{await page.click('#'+id);await wait(match)};
    const validate=async tag=>{
      const s=await page.evaluate(()=>window.viewerState),r=originals[s.round].data.requests.find(x=>x.request_id===s.request);assert(r);
      assert.equal(await page.locator('#prompt').textContent(),r.prompt_text,tag+' Unicode prompt');
      const cells=await page.locator(s.mode==='mean'?'#chart .barRow':'#chart tbody td').evaluateAll(nodes=>nodes.map(x=>({span:x.dataset.span,value:x.dataset.value,layer:x.parentElement.rowIndex-1})));
      for(const x of cells){const span=r.spans.find(a=>a.id===x.span);assert(span);const e=expected(s,span,s.mode==='mean'?0:x.layer,s.head);
        if(e===null)assert.equal(x.value,'',tag+' missing');else assert(Math.abs(Number(x.value)-e)<2e-14,`${tag} ${x.span}: ${x.value} vs ${e}`);checkedValues++;}
      const headCells=await page.locator(s.mode==='mean'?'#headchart .barRow':'#headchart tbody td').evaluateAll(nodes=>nodes.map((x,i)=>({value:x.dataset.value,head:x.dataset.span?Number(x.dataset.span):i%32,layer:Math.floor(i/32)})));
      const span=r.spans.find(a=>a.id===s.selectedSpan);for(const x of headCells){const e=expected(s,span,x.layer,x.head);if(e===null)assert.equal(x.value,'');else assert(Math.abs(Number(x.value)-e)<2e-14,tag+' head distribution');checkedValues++}
      assert(s.cache.conditions<=4&&s.cache.layerFiles<=8&&s.cache.meanFiles<=4);
      checks.push({tag,request:s.request,mode:s.mode,role:s.role,head:s.head,metric:s.metric,compare:s.compare,cells:cells.length+headCells.length,cache:s.cache});console.log(JSON.stringify({stage:tag,values:checkedValues}));
    };
    const started=Date.now();assert.equal((await page.goto(base,{waitUntil:'domcontentloaded'})).status(),200);await wait({request:'case-541-LD'});
    timings.initial_seconds=(Date.now()-started)/1000;await validate('original-default');
    await page.locator('#chart tbody tr').nth(5).locator('td').first().click();await wait({layer:5});
    await page.locator('#headchart tbody tr').nth(7).locator('td').nth(3).click();await wait({layer:7,head:3});await validate('cell-click-layer-head');
    await click('tabMean',{mode:'mean'});await validate('original-mean-head3');
    const before=payloads.length;await change('head','-1',{head:-1});await validate('original-mean-all-heads');assert.equal(payloads.length,before,'head change must reuse binary');
    await page.locator('#comparisonOptions').evaluate(x=>x.open=true);
    await change('parts','parts',{parts:'parts'});await change('metric','density',{metric:'density'});await change('role','lexicon_end',{role:'lexicon_end'});await validate('original-prefix-density-NA');
    await change('round','replacement',{round:'replacement',request:'ccr-541-LD-base'});assert.equal(await page.locator('#condition option').count(),38);
    await change('role','pre_answer',{role:'pre_answer'});await change('metric','mass',{metric:'mass'});await validate('replacement-baseline');
    await change('condition','ccr-541-LD-O01',{request:'ccr-541-LD-O01'});await click('baseline',{compare:'ccr-541-LD-base'});await validate('order-material');
    await change('alignment','slot',{alignment:'slot'});await validate('order-slot');
    await change('query','3169',{request:'ccr-3169-LD-base'});assert.equal(await page.locator('#condition option').count(),50);
    await change('group','definition');assert.equal(await page.locator('#condition option').count(),8);
    await change('condition','ccr-3169-LD-L05',{request:'ccr-3169-LD-L05'});await change('alignment','material',{alignment:'material'});await click('baseline',{compare:'ccr-3169-LD-base'});await validate('definition-mean-effect');
    assert((await page.locator('#score').textContent()).includes('无'));assert((await page.locator('#delta').textContent()).includes('+21.661602'));
    await change('role','query_all',{role:'query_all'});await change('head','13',{head:13});await change('metric','density',{metric:'density'});await validate('query-all-density-head13');
    await change('compareRole','demos_end',{compareRole:'demos_end'});await validate('readout-difference');
    await click('resetCompare',{compare:''});await change('role','pre_answer',{role:'pre_answer'});await change('head','-1',{head:-1});await change('metric','mass',{metric:'mass'});await change('sort','attention',{sort:'attention'});
    await change('ceiling','0.01',{ceiling:.01});await validate('sorted-mean');
    const dl=page.waitForEvent('download');await page.click('#export');const download=await dl;await download.saveAs(path.join(output,'mean.svg'));
    const svg=fs.readFileSync(path.join(output,'mean.svg'),'utf8');assert(svg.includes('36 层等权平均')&&svg.includes('ccr-3169-LD-L05'));
    await page.screenshot({path:path.join(output,'desktop-mean.png'),fullPage:true});
    await click('tabLayers',{mode:'layers'});await change('layer','29',{layer:29});await validate('replacement-layer');
    for(const l of [6,7,8])await change('layer',l,{layer:l});assert((await page.evaluate(()=>window.viewerState)).cache.layerFiles<=8);
    const dl2=page.waitForEvent('download');await page.click('#export');await (await dl2).saveAs(path.join(output,'layers.svg'));
    await change('group','body');await change('condition','ccr-3169-LD-B04-same',{request:'ccr-3169-LD-B04-same'});await validate('body-boundary');
    await change('group','answer');
    await page.selectOption('#condition','ccr-3169-LD-A12');await page.selectOption('#condition','ccr-3169-LD-A13');await wait({request:'ccr-3169-LD-A13'});await validate('latest-selection-wins');
    await page.setViewportSize({width:390,height:844});assert(await page.evaluate(()=>document.documentElement.scrollWidth<=window.innerWidth+1),'Mobile document must not overflow');
    await page.screenshot({path:path.join(output,'mobile.png'),fullPage:true});
    const deep=await context.newPage();deep.setDefaultTimeout(180000);const deepPayloads=[];deep.on('response',r=>{if(r.status()===200&&/\.(f32|f64)$/.test(r.url()))deepPayloads.push(r.url())});
    await deep.goto(base+'#round=replacement&request=ccr-541-LD-O01&mode=mean&parts=parts&compare=ccr-541-LD-base&alignment=slot');
    await deep.waitForFunction(()=>window.viewerState?.loaded&&window.viewerState.mode==='mean');assert.equal(deepPayloads.filter(x=>x.endsWith('.f32')).length,0);assert.equal(deepPayloads.filter(x=>x.endsWith('.f64')).length,1);
    const report=await context.request.get(base+'reports/');assert.equal(report.status(),200);assert((await report.text()).includes('内容替换实验'));
    for(const file of ['comparisons.json','all-inputs.tsv','answer-replacements.pdf'])assert.equal((await context.request.get(base+'reports/'+file)).status(),200);
    await change('round','hehe',{round:'hehe',request:'hsc-Q01-D01'});
    assert.equal(await page.locator('#query option').count(),3);
    await page.setViewportSize({width:1600,height:1150});
    await click('tabMean',{mode:'mean'});await change('head','-1',{head:-1});await change('metric','mass',{metric:'mass'});
    await change('parts','parts',{parts:'parts'});await change('role','pre_answer',{role:'pre_answer'});
    for(const q of ['Q01','Q02','Q03']){
      await change('query',q,{request:`hsc-${q}-D01`});assert.equal(await page.locator('#condition option').count(),3);
      for(const d of ['D01','D02','D03']){
        await change('condition',`hsc-${q}-${d}`,{request:`hsc-${q}-${d}`});await validate(`hehe-${q}-${d}-mean`);
        const score=originals.hehe.data.scores[`hsc-${q}-${d}`];assert((await page.locator('#score').textContent()).includes('参考：'+score.reference));
      }
    }
    await click('baseline',{compare:'hsc-Q03-D01'});await validate('hehe-dual-original-difference-with-component-NA');
    await click('resetCompare',{compare:''});await change('role','query_focal',{role:'query_focal'});await change('head','7',{head:7});
    await change('metric','density',{metric:'density'});await validate('hehe-focal-density-head7');
    await change('compareRole','pre_answer',{compareRole:'pre_answer'});await validate('hehe-readout-difference');
    await click('resetCompare',{compare:''});await change('role','demos_end',{role:'demos_end'});await validate('hehe-no-demos-NA');
    await click('tabLayers',{mode:'layers'});await change('role','pre_answer',{role:'pre_answer'});await change('head','-1',{head:-1});await validate('hehe-layers-all-heads');
    await change('query','Q01',{request:'hsc-Q01-D01'});await change('condition','hsc-Q01-D03',{request:'hsc-Q01-D03'});
    await click('tabMean',{mode:'mean'});await validate('hehe-Q01-dual-mean');
    const texts=await page.locator('#chart .barLabel').allTextContents();assert(texts.includes('嘿嘿·原释义')&&texts.includes('嘿嘿·普通义'));
    const newdl=page.waitForEvent('download');await page.click('#export');await (await newdl).saveAs(path.join(output,'hehe-mean.svg'));
    assert(fs.readFileSync(path.join(output,'hehe-mean.svg'),'utf8').includes('hsc-Q01-D03'));
    await page.screenshot({path:path.join(output,'hehe-desktop-mean.png'),fullPage:true});
    await page.setViewportSize({width:390,height:844});assert(await page.evaluate(()=>document.documentElement.scrollWidth<=window.innerWidth+1));
    await page.screenshot({path:path.join(output,'hehe-mobile.png'),fullPage:true});
    const newest=await context.newPage();newest.setDefaultTimeout(180000);const newPayloads=[];
    newest.on('response',r=>{if(r.status()===200&&/\.(f32|f64)$/.test(r.url()))newPayloads.push(r.url())});
    await newest.goto(base+'#round=hehe&request=hsc-Q03-D03&mode=mean&parts=parts&compare=hsc-Q03-D01');
    await newest.waitForFunction(()=>window.viewerState?.loaded&&window.viewerState.request==='hsc-Q03-D03');
    assert.equal(await newest.locator('#query').inputValue(),'Q03');assert.equal(newPayloads.filter(x=>x.endsWith('.f32')).length,0);assert.equal(newPayloads.filter(x=>x.endsWith('.f64')).length,1);
    const newReport=await context.request.get(base+'reports/hehe/');assert.equal(newReport.status(),200);const reportText=await newReport.text();
    assert(reportText.includes('嘿嘿释义与语境')&&!reportText.includes('href="/data/')&&!reportText.includes('src="/data/'));
    for(const file of ['scores.json','comparisons.json','all-outputs.pdf','dual-definition-attention.pdf','ALL-PROMPTS.md'])assert.equal((await context.request.get(base+'reports/hehe/'+file)).status(),200);
    // Returning to a legacy round must repopulate queries instead of keeping Q IDs.
    await change('round','original',{round:'original',request:'case-541-LD'});assert.equal(await page.locator('#query option').count(),2);await validate('return-to-original-after-hehe');
    assert.deepEqual(errors,[]);
    const receipt={status:'pass',base,checks,checked_values:checkedValues,errors,timings,binary_requests:payloads.filter(x=>/\.(f32|f64)$/.test(x)),mean_deeplink_layer_downloads:0,mean_deeplink_mean_downloads:1,mobile_no_page_overflow:true,old_and_new_full_prompts_exact:true,third_round_queries_checked:3,third_round_conditions_checked:9,third_round_deep_link_and_reports:true};
    fs.writeFileSync(path.join(output,'browser.json'),JSON.stringify(receipt,null,2)+'\n');console.log(JSON.stringify({status:'pass',values:checkedValues,timings}));
  }catch(e){fs.writeFileSync(path.join(output,'failure.json'),JSON.stringify({error:e.stack,errors,checks,payloads},null,2));throw e}
  finally{await browser?.close()}
})().catch(e=>{console.error(e);process.exitCode=1});
