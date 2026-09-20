// Authenticated, read-only browser checks against the deployed site.
const fs=require('fs'),path=require('path'),assert=require('assert'),crypto=require('crypto');
const {chromium}=require('playwright');
(async()=>{
  const root=path.resolve(__dirname,'../../..');
  const credentials=JSON.parse(fs.readFileSync(process.argv[2],'utf8'));
  const output=path.resolve(process.argv[3]);
  assert(!fs.existsSync(path.join(output,'browser.json')));
  const source=path.join(root,'reviews/case-attention-v1/results-01');
  const data=JSON.parse(fs.readFileSync(path.join(source,'index.html'),'utf8').match(/<script id="data" type="application\/json">([\s\S]*?)<\/script>/)[1]);
  const aggregate=id=>JSON.parse(fs.readFileSync(path.join(source,id+'-aggregates.json'),'utf8')).aggregates;
  const expected=(id,role,span)=>aggregate(id).find(x=>x.role===role&&x.span_id===span).mass[5][3];
  let browser,page;const errors=[],payloads=[],timings={};const started=Date.now();
  try{
    browser=await chromium.launch({executablePath:'/tmp/paired-review-playwright/chromium_headless_shell-1234/chrome-headless-shell-linux64/chrome-headless-shell',headless:true,args:['--no-sandbox','--disable-gpu','--proxy-server=direct://','--host-resolver-rules=MAP hsd.fenglin.pro 165.22.48.237']});
    const context=await browser.newContext({httpCredentials:credentials,viewport:{width:1600,height:1150}});
    page=await context.newPage();
    page.setDefaultTimeout(180000);
    page.on('pageerror',e=>errors.push(e.message));
    page.on('response',r=>{if((r.url().endsWith('.meta.json')||r.url().endsWith('.f32'))&&r.status()===200)payloads.push(r.url().split('/').pop())});
    const response=await page.goto('https://hsd.fenglin.pro/',{waitUntil:'domcontentloaded',timeout:90000});
    assert.equal(response.status(),200);
    assert.equal(await page.title(),'真实案例 · 逐层注意力');
    console.log(JSON.stringify({stage:'authenticated_page_loaded'}));
    await page.waitForFunction(()=>window.viewerState?.loaded);
    timings.initial_load_seconds=(Date.now()-started)/1000;
    console.log(JSON.stringify({stage:'default_attention_loaded'}));
    assert.equal(await page.title(),'真实案例 · 逐层注意力');
    assert.equal(await page.locator('#mainheat tbody tr').count(),36);
    assert.equal(await page.locator('#message').textContent(),'结果已加载');
    assert.equal(await page.locator('#prompt').textContent(),data.requests.find(r=>r.request_id==='case-541-LD').prompt_text);
    await page.selectOption('#head','3');await page.selectOption('#layer','5');
    await page.waitForFunction(()=>window.viewerState?.loaded&&window.viewerState.layer===5);
    let actual=Number(await page.locator('#mainheat tbody tr').nth(5).locator('td').first().getAttribute('data-value'));
    const first=data.requests.find(r=>r.request_id==='case-541-LD').spans.find(s=>s.kind==='lexicon').id;
    assert(Math.abs(actual-expected('case-541-LD','pre_answer',first))<1e-14);
    await page.selectOption('#role','demos_end');await page.selectOption('#compareRole','lexicon_end');
    actual=Number(await page.locator('#mainheat tbody tr').nth(5).locator('td').first().getAttribute('data-value'));
    assert(Math.abs(actual-(expected('case-541-LD','demos_end',first)-expected('case-541-LD','lexicon_end',first)))<1e-14);
    await page.selectOption('#compareRole','');await page.selectOption('#role','lexicon_end');
    assert(await page.locator('#prompt .unseen').count()>0);
    await page.selectOption('#role','pre_answer');const switching=Date.now();await page.selectOption('#query','3169');
    await page.waitForFunction(()=>window.viewerState?.request==='case-3169-LD'&&window.viewerState.loaded);
    timings.switch_case_seconds=(Date.now()-switching)/1000;
    console.log(JSON.stringify({stage:'second_case_loaded'}));
    assert.equal(await page.locator('#prompt').textContent(),data.requests.find(r=>r.request_id==='case-3169-LD').prompt_text);
    const comparing=Date.now();await page.selectOption('#compare','LDC');
    await page.waitForFunction(()=>document.getElementById('message').textContent==='结果已加载');
    timings.compare_condition_seconds=(Date.now()-comparing)/1000;
    const second=data.requests.find(r=>r.request_id==='case-3169-LD').spans.find(s=>s.kind==='lexicon').id;
    actual=Number(await page.locator('#mainheat tbody tr').nth(5).locator('td').first().getAttribute('data-value'));
    assert(Math.abs(actual-(expected('case-3169-LD','pre_answer',second)-expected('case-3169-LDC','pre_answer',second)))<1e-14);
    const download=page.waitForEvent('download');await page.click('#export');const item=await download;
    await item.saveAs(path.join(output,'live-export.svg'));
    await page.selectOption('#compare','');await page.selectOption('#head','-1');await page.selectOption('#ceiling','0.1');
    await page.screenshot({path:path.join(output,'live-preview.png'),fullPage:false});
    assert.deepEqual(errors,[]);
    assert.equal(payloads.filter(x=>x==='case-541-LD.meta.json').length,1);
    assert(!payloads.some(x=>x.startsWith('case-3169-LDC.layer-')),'Comparisons should not fetch unused token rows');
    const receipt={status:'pass',url:'https://hsd.fenglin.pro/',https_certificate_verified:true,initial_data_loaded_without_file_selection:true,
      layer_rows:36,head_layer_value_matches:true,stage_comparison_matches:true,condition_comparison_matches:true,
      full_prompts_checked:2,future_text_unseen:true,svg_export_checked:true,payloads_loaded:payloads,page_errors:errors,timings,
      direct_origin_ip:'165.22.48.237',at:new Date().toISOString(),source_sha256:crypto.createHash('sha256').update(fs.readFileSync(__filename)).digest('hex')};
    fs.writeFileSync(path.join(output,'browser.json'),JSON.stringify(receipt,null,2)+'\n');console.log(JSON.stringify(receipt));
  }catch(error){
    const diagnostic={error:String(error),page_errors:errors,payloads,timings};
    if(page){try{diagnostic.page=await page.evaluate(()=>({url:location.href,title:document.title,message:document.getElementById('message')?.textContent,viewer:window.viewerState}));}catch{}}
    fs.writeFileSync(path.join(output,'browser-layer-failure.json'),JSON.stringify(diagnostic,null,2)+'\n');
    throw error;
  }finally{if(browser)await browser.close();}
})().catch(e=>{console.error(e);process.exit(1)});
