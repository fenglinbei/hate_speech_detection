// Read-only browser verification of real, normally released measurements.
const fs=require('fs'),path=require('path'),assert=require('assert'),crypto=require('crypto');
const {chromium}=require('playwright');
(async()=>{
  const root=path.resolve(__dirname,'../..'),results=path.join(root,'reviews/case-attention-v1/results-01');
  const output=path.join(root,'reviews/case-attention-v1/report-01');
  assert(fs.existsSync(path.join(output,'independent-audit.json')));
  assert(!fs.existsSync(path.join(output,'browser-audit.json')));
  const html=fs.readFileSync(path.join(results,'index.html'),'utf8');
  const data=JSON.parse(html.match(/<script id="data" type="application\/json">([\s\S]*?)<\/script>/)[1]);
  let browser;
  try{
    browser=await chromium.launch({executablePath:'/tmp/paired-review-playwright/chromium_headless_shell-1234/chrome-headless-shell-linux64/chrome-headless-shell',headless:true,args:['--no-sandbox','--disable-gpu']});
    const page=await browser.newPage({viewport:{width:1600,height:1150}}),errors=[];
    page.on('pageerror',e=>errors.push(e.message));
    await page.goto('file://'+path.join(results,'index.html'));
    await page.waitForFunction(()=>window.viewerState);
    await page.setInputFiles('#files',data.requests.map(r=>path.join(results,r.request_id+'.view.json')));
    for(const r of data.requests){
      await page.selectOption('#query',r.query_id);await page.selectOption('#condition',r.condition);
      await page.waitForFunction(id=>window.viewerState.request===id&&window.viewerState.loaded,r.request_id);
      assert.equal(await page.locator('#prompt').textContent(),r.prompt_text);
      assert.equal(await page.locator('#mainheat tbody tr').count(),36);
      const values=await page.locator('#mainheat td').evaluateAll(cells=>cells.map(x=>x.dataset.value));
      assert(values.some(x=>x!==''&&Number(x)>0));
    }
    await page.selectOption('#query','541');await page.selectOption('#condition','LD');
    await page.selectOption('#head','3');await page.selectOption('#layer','5');
    const r=data.requests.find(x=>x.request_id==='case-541-LD');
    const ag=JSON.parse(fs.readFileSync(path.join(results,r.request_id+'-aggregates.json'),'utf8')).aggregates;
    const span=r.spans.find(s=>s.kind==='lexicon');
    const value=(role,metric='mass')=>ag.find(a=>a.role===role&&a.span_id===span.id)[metric][5][3];
    let actual=Number(await page.locator('#mainheat tbody tr').nth(5).locator('td').first().getAttribute('data-value'));
    assert(Math.abs(actual-value('pre_answer'))<1e-14);
    await page.selectOption('#role','demos_end');await page.selectOption('#compareRole','lexicon_end');
    actual=Number(await page.locator('#mainheat tbody tr').nth(5).locator('td').first().getAttribute('data-value'));
    assert(Math.abs(actual-(value('demos_end')-value('lexicon_end')))<1e-14);
    await page.selectOption('#compareRole','');await page.selectOption('#role','lexicon_end');
    assert(await page.locator('#prompt .unseen').count()>0);
    const demoIndex=r.spans.filter(s=>['demo','lexicon','system','query','structure','boundary'].includes(s.kind)).findIndex(s=>s.kind==='demo');
    assert.equal(await page.locator('#mainheat tbody tr').first().locator('td').nth(demoIndex).getAttribute('data-value'),'');
    await page.selectOption('#role','pre_answer');await page.selectOption('#head','-1');await page.selectOption('#ceiling','0.1');
    await page.screenshot({path:path.join(output,'real-viewer-preview.png'),fullPage:false});
    const download=page.waitForEvent('download');await page.click('#export');const item=await download;
    const svg=path.join(output,'case-541-LD-pre-answer.svg');await item.saveAs(svg);
    assert(fs.readFileSync(svg,'utf8').startsWith('<svg'));
    assert.deepEqual(errors,[]);
    const receipt={status:'pass',research_measurements_loaded:12,full_unicode_prompts_checked:12,layer_rows:36,selected_head_axes_checked:true,
      stages_subtraction_matches_FP64_aggregates:true,future_demo_cells_are_NA:true,svg_export_checked:true,page_errors:errors,
      browser_version:browser.version(),node_version:process.version,
      source:{path:__filename,sha256:crypto.createHash('sha256').update(fs.readFileSync(__filename)).digest('hex')},
      viewer:{path:path.join(results,'index.html'),sha256:crypto.createHash('sha256').update(html).digest('hex')}};
    fs.writeFileSync(path.join(output,'browser-audit.json'),JSON.stringify(receipt,null,2)+'\n');console.log(JSON.stringify(receipt));
  }finally{if(browser)await browser.close();}
})().catch(e=>{console.error(e);process.exit(1)});
