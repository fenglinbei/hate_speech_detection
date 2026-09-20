const fs=require('fs'),path=require('path'),os=require('os'),assert=require('assert'),crypto=require('crypto');
const {chromium}=require('playwright');
const root=path.resolve(__dirname,'../..');
const arg=k=>{const i=process.argv.indexOf(k);return i<0?null:process.argv[i+1]};
const info=p=>({path:path.resolve(p),bytes:fs.statSync(p).size,sha256:crypto.createHash('sha256').update(fs.readFileSync(p)).digest('hex')});
(async()=>{
  const prepared=arg('--prepared')||path.join(root,'reviews/hehe-focal-patching-v1/prepared-01'),result=arg('--results'),url=arg('--url');
  const output=arg('--output')||path.join(prepared,'viewer-cpu-test.json'),template=path.join(root,'tools/hehe_focal_patch_viewer_v1/viewer.html');
  const tmp=fs.mkdtempSync(path.join(os.tmpdir(),'focal-patch-browser-'));let browser;
  try{
    browser=await chromium.launch({executablePath:'/tmp/paired-review-playwright/chromium_headless_shell-1234/chrome-headless-shell-linux64/chrome-headless-shell',headless:true,args:['--no-sandbox','--disable-gpu']});
    const credentials=arg('--credential')?JSON.parse(fs.readFileSync(arg('--credential'),'utf8')):null;
    const context=await browser.newContext({viewport:{width:1350,height:980},httpCredentials:credentials?{username:credentials.username,password:credentials.password}:undefined});
    const page=await context.newPage(),errors=[];page.on('pageerror',e=>errors.push(e.message));
    let data,target;
    if(result){data=JSON.parse(fs.readFileSync(path.join(result,'results.json'),'utf8'));target=url||'file://'+path.join(result,'index.html')}
    else{
      await page.goto('file://'+path.join(prepared,'viewer.html'));await page.waitForFunction(()=>window.HFP&&HFP.current);
      assert((await page.locator('#status').textContent()).includes('GPU未运行'));assert.equal(await page.locator('#plot circle').count(),0);assert.equal(await page.locator('#values tbody tr').count(),72);
      const raw=await page.evaluate(()=>HFP.DATA);data=JSON.parse(JSON.stringify(raw));data.status='明确CPU合成读数，仅用于展示验证';data.margin_error_bound=1e-6;
      data.baselines=data.requests.map(r=>({request_id:r.request_id,query_id:r.query_id,dictionary_id:r.dictionary_id,raw_prediction:r.dictionary_id==='D01'?'有':'无',m:r.dictionary_id==='D01'?-10:10,reference:r.query_id==='Q01'?'无':'有'}));
      data.effects=data.jobs.map((j,i)=>({...j,m:(j.layer-17)*(j.group==='focal'?1:.1),delta_m:(i-144)/10,donor_gap_fraction:i===0?null:(i-20)/33,
        reference_aligned_delta:(j.query_id==='Q01'?1:-1)*(i-144)/10,donor_state_l2:i/11,prediction:i%2?'有':'无',flip:i%3===0?null:i%3===1,transition:i%3===0?'unresolved':i%3===1?'repair':'unchanged'}));
      data.position_differences=data.requests.flatMap(r=>Array.from({length:36},(_,l)=>({recipient:r.request_id,layer:l,delta_m:(l-18)/5})));
      target='file://'+path.join(tmp,'fixture.html');fs.writeFileSync(path.join(tmp,'fixture.html'),fs.readFileSync(template,'utf8').replace('__DATA_JSON__',JSON.stringify(data).replace(/</g,'\\u003c')));
    }
    await page.goto(target,{timeout:600000});await page.waitForFunction(()=>window.HFP&&HFP.current,{},{timeout:600000});
    let checked=0,promptCount=0;
    for(const req of data.requests){
      await page.selectOption('#query',req.query_id);await page.selectOption('#recipient',req.dictionary_id);
      assert.equal(await page.locator('#prompt').textContent(),req.prompt_text);promptCount++;
      for(const [g,ps]of Object.entries(req.patch_position_sets))for(const p of ps)assert((await page.locator('#geometry').textContent()).includes(p+'「'+req.token_text[p]+'」'));
      for(const metric of ['delta_m','m','donor_gap_fraction','reference_aligned_delta','donor_state_l2']){
        await page.selectOption('#metric',metric);
        const dots=await page.locator('#plot circle').evaluateAll(es=>es.map(e=>({job:e.dataset.job,value:Number(e.dataset.value),layer:Number(e.dataset.layer),group:e.dataset.group})));
        const wanted=data.effects.filter(e=>e.recipient===req.request_id&&e[metric]!==null);assert.equal(dots.length,wanted.length);
        for(const e of wanted){const dot=dots.find(x=>x.job===e.job_id);assert(dot);assert.equal(dot.value,e[metric]);assert.equal(dot.layer,e.layer);assert.equal(dot.group,e.group);checked++}
      }
      const dots=await page.locator('#contrast circle').evaluateAll(es=>es.map(e=>({layer:Number(e.dataset.layer),value:Number(e.dataset.value)})));
      for(const e of data.position_differences.filter(e=>e.recipient===req.request_id)){assert.equal(dots.find(d=>d.layer===e.layer).value,e.delta_m);checked++}
      const table=await page.locator('#values tbody tr').evaluateAll(es=>es.map(e=>({layer:Number(e.dataset.layer),group:e.dataset.group,values:[...e.cells].slice(2,5).map(c=>c.dataset.value===''?null:Number(c.dataset.value)),texts:[...e.cells].map(c=>c.textContent)})));
      for(const e of data.effects.filter(e=>e.recipient===req.request_id)){const row=table.find(t=>t.layer===e.layer&&t.group===e.group);assert.deepEqual(row.values,[e.m,e.delta_m,e.donor_gap_fraction]);assert.equal(row.texts[5],e.prediction===null?'':e.prediction);assert.equal(row.texts[6],e.flip===null?'NA':e.flip?'是':'否');assert.equal(row.texts[7],e.transition);checked+=6}
    }
    const deep=target.split('#')[0]+'#query=Q02&recipient=D02&metric=donor_gap_fraction';await page.goto(deep,{timeout:600000});await page.waitForFunction(()=>HFP.current.request_id.includes('Q02-D02')&&HFP.current.metric==='donor_gap_fraction');
    await page.reload({timeout:600000});await page.waitForFunction(()=>window.HFP&&HFP.current.metric==='donor_gap_fraction');assert.equal(await page.locator('#query').inputValue(),'Q02');
    const download=page.waitForEvent('download');await page.click('#export');await (await download).saveAs(path.join(tmp,'plot.svg'));assert(fs.readFileSync(path.join(tmp,'plot.svg'),'utf8').includes('<svg'));
    await page.setViewportSize({width:390,height:844});assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+2),'Mobile page overflow');
    await page.screenshot({path:output.replace(/\.json$/,'.png'),fullPage:false});assert.deepEqual(errors,[]);
    const receipt={status:'pass',measured:Boolean(result),location:url?'actual HTTPS':'local file',values_checked:checked,prompt_displays:promptCount,all_layers_and_directions:true,
      NA_not_zero:true,Unicode_exact:true,SVG_export:true,cold_deep_link:true,mobile:true,page_errors:errors,CUDA_initialized:false,
      browser_version:browser.version(),implementation_snapshot:[__filename,template].map(info)};
    fs.writeFileSync(output,JSON.stringify(receipt,null,2)+'\n',{flag:'wx'});console.log(JSON.stringify(receipt));
  }finally{if(browser)await browser.close();fs.rmSync(tmp,{recursive:true,force:true})}
})().catch(e=>{console.error(e);process.exit(1)});
