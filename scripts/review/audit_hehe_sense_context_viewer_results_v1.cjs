// Read-only browser verification of completed research results. The server is
// loopback-only and exists only for this check; it never starts a GPU worker.
const fs = require('fs');
const path = require('path');
const http = require('http');
const assert = require('assert');
const crypto = require('crypto');
const {chromium} = require('playwright');

(async()=>{
  const root=path.resolve(__dirname,'../..'),work=path.join(root,'reviews/hehe-sense-context-v1');
  const results=path.join(work,'results-01'),out=path.join(work,'audits/viewer-results-01');
  assert(!fs.existsSync(out),'New audit directory required');
  const state=JSON.parse(fs.readFileSync(path.join(work,'run-01/state.json'),'utf8'));
  assert(state.status==='complete'&&state.owned_worker_absent&&state.worker_exit_code===0,'Normal release required');
  const html=fs.readFileSync(path.join(results,'index.html'),'utf8');
  const data=JSON.parse(html.match(/<script id="data" type="application\/json">([\s\S]*?)<\/script>/)[1]);
  const requests=new Map(data.requests.map(r=>[r.request_id,r]));
  fs.mkdirSync(out,{recursive:true});
  const server=http.createServer((req,res)=>{
    const name=decodeURIComponent(new URL(req.url,'http://localhost').pathname).slice(1)||'index.html';
    if(name!==path.basename(name)){res.writeHead(403);res.end();return;}
    const p=path.join(results,name);
    if(!fs.existsSync(p)||!fs.statSync(p).isFile()){res.writeHead(404);res.end();return;}
    res.setHeader('Content-Type',name.endsWith('.html')?'text/html; charset=utf-8':'application/json; charset=utf-8');
    fs.createReadStream(p).pipe(res);
  });
  await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));
  const url='http://127.0.0.1:'+server.address().port+'/index.html';
  const aggregates=rid=>JSON.parse(fs.readFileSync(path.join(results,rid+'-aggregates.json'),'utf8')).aggregates;
  const sources=new Set(),errors=[];let browser,cells=0;
  const get=(rows,role,sid,metric,layer,head)=>{
    const a=rows.find(x=>x.role===role&&x.span_id===sid);
    if(!a||!a.visible)return null;
    const v=a[metric][layer];return head<0?v.reduce((a,b)=>a+b,0)/v.length:v[head];
  };
  try{
    browser=await chromium.launch({executablePath:'/tmp/paired-review-playwright/chromium_headless_shell-1234/chrome-headless-shell-linux64/chrome-headless-shell',headless:true,args:['--no-sandbox','--disable-gpu']});
    const plans=[
      ...data.requests.map(r=>({rid:r.request_id})),
      ...['Q01','Q02','Q03'].map(q=>({rid:`hsc-${q}-D03`,other:`hsc-${q}-D01`})),
      {rid:'hsc-Q01-D03',compareRole:'lexicon_end'}
    ];
    for(const plan of plans){
      const r=requests.get(plan.rid);assert(r,'Unknown planned request '+plan.rid);
      const rows=aggregates(plan.rid),cr=plan.other?requests.get(plan.other):(plan.compareRole?r:null),otherRows=cr?aggregates(cr.request_id):null;
      sources.add(plan.rid);if(cr)sources.add(cr.request_id);
      const page=await browser.newPage({viewport:{width:1650,height:1050}});
      page.on('pageerror',e=>errors.push(e.message));
      await page.goto(url);await page.waitForFunction(()=>window.viewerState);
      await page.selectOption('#query',r.query_id);await page.selectOption('#condition',r.request_id);
      await page.waitForFunction(rid=>window.viewerState?.request===rid&&window.viewerState.loaded,r.request_id);
      if(plan.other){await page.selectOption('#compare',cr.request_id);await page.waitForFunction(rid=>cache.has(rid),cr.request_id);}
      if(plan.compareRole)await page.selectOption('#compareRole',plan.compareRole);
      await page.selectOption('#alignment',plan.alignment||'material');
      await page.selectOption('#parts','parts');
      const spans=r.spans.filter(s=>['demo_text','demo_answer','lexicon_term','lexicon_definition','lexicon_definition_component','lexicon_category','system','query','structure','boundary'].includes(s.kind));
      for(const [role,metric,head] of [['pre_answer','mass',-1],['query_focal','density',7],['demos_end','mass',3],['lexicon_end','density',-1]]){
        await page.selectOption('#role',role);await page.selectOption('#metric',metric);await page.selectOption('#head',String(head));
        const actual=await page.locator('#mainheat tbody tr').evaluateAll(trs=>trs.map(tr=>Array.from(tr.querySelectorAll('td')).map(td=>td.dataset.value===''?null:Number(td.dataset.value))));
        assert.equal(actual.length,36);
        for(let l=0;l<36;l++)for(let i=0;i<spans.length;i++){
          let expected=get(rows,role,spans[i].id,metric,l,head);
          if(cr){
            let sid=spans[i].id;
            if(plan.alignment==='slot'&&spans[i].kind.startsWith('demo')){
              const slot=spans[i].label.match(/^示例(\d+) /)[1];
              sid=cr.spans.find(s=>s.kind===spans[i].kind&&s.label.startsWith('示例'+slot+' '))?.id||'__missing__';
            }
            const value=get(otherRows,plan.compareRole||role,sid,metric,l,head);expected=expected===null||value===null?null:expected-value;
          }
          if(expected===null)assert.strictEqual(actual[l][i],null,'NA displayed as measurement');
          else assert(Math.abs(actual[l][i]-expected)<=1e-12,'Actual matrix differs from independent file aggregation');
          cells++;
        }
        assert.equal(await page.locator('#prompt').textContent(),r.prompt_text,'Unicode source differs');
      }
      if(r.request_id==='hsc-Q01-D03'&&!cr){
        await page.selectOption('#compare','');await page.selectOption('#role','pre_answer');
        await page.selectOption('#metric','density');await page.selectOption('#head','-1');await page.selectOption('#ceiling','0.01');
        await page.screenshot({path:path.join(out,'Q01-D03-results.png'),fullPage:false});
        const pending=page.waitForEvent('download');await page.click('#export');
        await(await pending).saveAs(path.join(out,'Q01-D03-heatmap.svg'));
        assert(fs.readFileSync(path.join(out,'Q01-D03-heatmap.svg'),'utf8').startsWith('<svg'));
      }
      await page.close();
    }
    assert.deepEqual(errors,[]);
    const fileInfo=p=>({path:p,bytes:fs.statSync(p).size,sha256:crypto.createHash('sha256').update(fs.readFileSync(p)).digest('hex')});
    const receipt={status:'pass',CPU_only:true,new_GPU_forwards:0,real_input_files_checked:[...sources],
      configurations_checked:plans.length,matrix_cells_checked:cells,roles_checked:['pre_answer','query_focal','demos_end','lexicon_end'],
      head_average_and_individual_heads:true,mass_and_visible_density:true,definition_component_alignment:true,condition_and_readout_differences:true,
      Unicode_source_preserved:true,unavailable_values_preserved:true,real_svg_export:true,page_errors:errors,
      browser_version:browser.version(),node_version:process.version,sources:[__filename,path.join(results,'manifest.json'),path.join(results,'index.html')].map(fileInfo),
      artifacts:['Q01-D03-results.png','Q01-D03-heatmap.svg'].map(n=>fileInfo(path.join(out,n)))};
    fs.writeFileSync(path.join(out,'audit.json'),JSON.stringify(receipt,null,2)+'\n');
    console.log(JSON.stringify(receipt));
  }finally{if(browser)await browser.close();await new Promise(resolve=>server.close(resolve));}
})().catch(e=>{console.error(e);process.exit(1)});
