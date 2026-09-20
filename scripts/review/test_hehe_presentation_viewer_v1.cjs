// Browser checks for the actual CPU preview, plus an explicitly synthetic /tmp
// result file. The synthetic visualization never enters scientific artifacts.
const fs = require('fs');
const os = require('os');
const path = require('path');
const assert = require('assert');
const crypto = require('crypto');
const {chromium} = require('playwright');

(async()=>{
  const root=path.resolve(__dirname,'../..'), prepared=path.join(root,'reviews/hehe-presentation-mechanism-v1/prepared-01');
  const html=fs.readFileSync(path.join(prepared,'viewer.html'),'utf8');
  const data=JSON.parse(html.match(/<script id="data" type="application\/json">([\s\S]*?)<\/script>/)[1]);
  const tmp=fs.mkdtempSync(path.join(os.tmpdir(),'attention-view-synthetic-'));
  let browser;
  try {
    browser=await chromium.launch({executablePath:'/tmp/paired-review-playwright/chromium_headless_shell-1234/chrome-headless-shell-linux64/chrome-headless-shell',headless:true,args:['--no-sandbox','--disable-gpu']});
    const page=await browser.newPage({viewport:{width:1440,height:1100}}),errors=[];
    page.on('pageerror',e=>errors.push(e.message));
    await page.goto('file://'+path.join(prepared,'viewer.html'));
    await page.waitForFunction(()=>window.viewerState);
    assert((await page.locator('#status').textContent()).includes('GPU未运行'));
    assert.equal(await page.locator('#mainheat tbody tr').count(),36);
    for (const r of data.requests) {
      await page.selectOption('#query',r.query_id);
      await page.selectOption('#condition',r.request_id);
      assert.equal(await page.locator('#prompt').textContent(),r.prompt_text,'Unicode prompt changed in display');
      const visible=await page.locator('#materials').textContent();
      for(const s of r.spans.filter(s=>['demo','lexicon','query'].includes(s.kind)))
        assert(visible.includes(Array.from(r.prompt_text).slice(s.char_start,s.char_end).join('')),'Full source span missing');
    }
    await page.selectOption('#alignment','slot');assert.equal(await page.locator('#alignment').inputValue(),'slot');await page.selectOption('#alignment','material');
    await page.selectOption('#query','Q01');await page.selectOption('#condition','hpm-Q01-D03');
    await page.selectOption('#role','pre_answer');await page.selectOption('#compareRole','lexicon_end');
    assert((await page.locator('#readout').textContent()).includes('词典结束'));
    await page.screenshot({path:path.join(prepared,'viewer-preview.png'),fullPage:false});
    // All numerical fixture files are kept outside the real preparation.
    const r=data.requests.find(r=>r.request_id==='hpm-Q01-D03'),T=r.prompt_tokens,shape=[36,32,6,T];
    const bytes=Buffer.alloc(shape.reduce((a,b)=>a*b,1)*4);
    const val=(l,h,ri,j)=>bytes.readFloatLE((((l*32+h)*6+ri)*T+j)*4);
    for(let l=0;l<36;l++)for(let h=0;h<32;h++)for(let ri=0;ri<6;ri++){
      const rows=r.roles[data.role_order[ri]],alpha=(l+h)/68*.3;
      if(!rows.length)continue;
      for(let j=0;j<T;j++){
        let v=0;for(const q of rows)if(j<=q)v+=(alpha*(j===0)+(1-alpha)/(q+1))/rows.length;
        bytes.writeFloatLE(v,(((l*32+h)*6+ri)*T+j)*4);
      }
    }
    const aggregates=[];
    data.role_order.forEach((role,ri)=>{for(const s of r.spans){const rows=r.roles[role],keys=s.token_positions;
      const visible=rows.length?rows.reduce((sum,q)=>sum+keys.filter(k=>k<=q).length,0)/rows.length:0;
      const mass=Array.from({length:36},(_,l)=>Array.from({length:32},(_,h)=>keys.reduce((sum,k)=>sum+val(l,h,ri,k),0)));
      aggregates.push({role,span_id:s.id,visible:visible>0,mass:visible?mass:null,density:visible?mass.map(xs=>xs.map(x=>x/visible)):null});
    }});
    const fixture={request_id:r.request_id,prompt_sha256:r.prompt_sha256,shape,encoding:'little-endian-float32-base64-visualization-only',
      data:bytes.toString('base64'),aggregates,score:{raw_prediction:'无',reference:'无',m:2,resolution:'EXPLICIT_SYNTHETIC_FIXTURE'},synthetic:true};
    const fixturePath=path.join(tmp,r.request_id+'.view.json');fs.writeFileSync(fixturePath,JSON.stringify(fixture));
    await page.setInputFiles('#files',fixturePath);await page.waitForFunction(()=>window.viewerState.loaded);
    await page.selectOption('#compareRole','');await page.selectOption('#role','pre_answer');
    await page.selectOption('#head','3');await page.selectOption('#layer','5');
    const sourceSpan=r.spans.find(s=>s.kind==='lexicon');
    await page.selectOption('#parts','parts');
    const labels=await page.locator('#mainheat thead th').allTextContents();
    assert(labels.includes('嘿嘿·原释义') && labels.includes('嘿嘿·普通义'),'Dual definition components missing');
    await page.selectOption('#parts','whole');
    const expected=aggregates.find(a=>a.role==='pre_answer'&&a.span_id===sourceSpan.id).mass[5][3];
    const actual=await page.locator('#mainheat tbody tr').nth(5).locator('td').first().getAttribute('data-value');
    assert(Math.abs(Number(actual)-expected)<1e-12,'Head/layer/role axes mismatch');
    await page.selectOption('#role','lexicon_end');
    assert(await page.locator('#prompt .unseen').count()>0,'Future text must be marked unseen');
    const download=page.waitForEvent('download');await page.click('#export');const item=await download;
    const exportPath=path.join(tmp,'synthetic.svg');await item.saveAs(exportPath);assert(fs.readFileSync(exportPath,'utf8').startsWith('<svg'));
    await page.selectOption('#role','demos_end');
    assert((await page.locator('#geometry').textContent()).includes('此条件未提供该材料'),'Absent demos must be NA');
    // A distinct synthetic mechanism payload exercises every input/position/
    // metric in the new native-state viewer, without suggesting real inference.
    const groups=['definition_original','definition_ordinary','definition_scaffold','query_focal','query_quote_other','query_rejection','query_other','remainder'];
    const records=data.requests.map((r,ri)=>({request_id:r.request_id,score:{raw_prediction:'无',reference:'无',m:ri+1},sites:['pre','mid','post'],position_labels:r.mechanism.position_labels,
      pre_answer_probe_margin_by_site:Array.from({length:36},(_,l)=>[l-ri,l-ri+.2,l-ri-.1]),
      pre_answer_attention_probe_change:Array(36).fill(.2),pre_answer_mlp_probe_change:Array(36).fill(-.3),
      source_rows:r.mechanism.av_positions.flatMap(p=>groups.map((g,gi)=>{const visible=r.mechanism.source_groups[g].some(k=>k<=p);const values=visible?Array.from({length:36},(_,l)=>(ri+1)*(gi+1)*(l+1)/100000):null;return {target_position:p,target_labels:r.mechanism.position_labels[String(p)],group:g,label:g,visible,mean_head_mass:values,mean_head_density:values,mean_head_AV_norm:values,source_output_norm:values,local_direction_projection:values}}))}));
    const payload={records,layers:36,notes:['明确标注的CPU合成读数'],representation_pairs:[{left:records[0].request_id,right:records[1].request_id,role:'pre_answer',token_index:0,relative_l2:Array(36).fill(.1),l2:Array(36).fill(2),cosine:Array(36).fill(.8)}]};
    const mt=fs.readFileSync(path.join(root,'tools/hehe_presentation_viewer_v1/mechanism.html'),'utf8').replace('__DATA_JSON__',JSON.stringify(payload).replace(/</g,'\\u003c'));
    const mp=path.join(tmp,'mechanism.html');fs.writeFileSync(mp,mt);await page.goto('file://'+mp);await page.waitForFunction(()=>window.HPM);
    const mechanismValues=await page.evaluate(()=>{let n=0;const $=id=>document.getElementById(id),check=(ok,why)=>{if(!ok)throw Error(why)};for(const r of HPM.DATA.records){$('request').value=r.request_id;HPM.changeRequest();for(const p of [...new Set(r.source_rows.map(x=>x.target_position))])for(const metric of ['mean_head_mass','mean_head_density','mean_head_AV_norm','source_output_norm','local_direction_projection']){$('target').value=String(p);$('metric').value=metric;HPM.changeSource();const expected=r.source_rows.filter(x=>x.target_position===p);const dots=[...document.querySelectorAll('#source circle')];let wanted=0;for(const row of expected){const found=dots.filter(x=>x.dataset.series===row.label);if(row[metric]===null){check(found.length===0,'NA has measured dots');continue}wanted+=36;check(found.length===36,'Missing layer');for(const dot of found){check(Number(dot.dataset.value)===row[metric][Number(dot.dataset.layer)],'Source value differs');n++}}check(dots.length===wanted,'Extra dots')}}return n});
    assert(mechanismValues>0);const svgDownload=page.waitForEvent('download');await page.locator('#lens button').click();await (await svgDownload).saveAs(path.join(tmp,'mechanism.svg'));assert(fs.readFileSync(path.join(tmp,'mechanism.svg'),'utf8').includes('<svg'));
    await page.setViewportSize({width:390,height:844});assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+2),'Mobile overflow');
    assert.deepEqual(errors,[]);
    const receipt={status:'pass',real_prompt_displays_checked:18,unicode_and_emoji_preserved:true,layer_rows:36,
      boundary_role_comparison:true,synthetic_import_axes_checked:true,svg_export_checked:true,page_errors:errors,
      synthetic_data_location:'/tmp only; deleted after test',research_attention_displayed:false,mechanism_values_checked:mechanismValues,mechanism_NA_and_SVG:true,
      browser_version:browser.version(),node_version:process.version,
      implementation_snapshot:[__filename,path.join(root,'tools/hehe_presentation_viewer_v1/viewer.html'),path.join(prepared,'viewer.html')].map(p=>({path:p,bytes:fs.statSync(p).size,sha256:crypto.createHash('sha256').update(fs.readFileSync(p)).digest('hex')}))};
    fs.writeFileSync(path.join(prepared,'viewer-cpu-test.json'),JSON.stringify(receipt,null,2)+'\n');
    console.log(JSON.stringify(receipt));
  } finally {if(browser)await browser.close();fs.rmSync(tmp,{recursive:true,force:true});}
})().catch(e=>{console.error(e);process.exit(1)});
