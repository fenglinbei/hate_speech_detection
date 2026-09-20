// Full pinned legacy suite, all new numerical values, navigation and downloads.
const fs=require('fs'),path=require('path'),assert=require('assert'),cp=require('child_process');
const {chromium}=require('playwright');
const root=path.resolve(__dirname,'../../../..'),[base,output,credentials]=process.argv.slice(2);assert(base&&output);
fs.mkdirSync(output,{recursive:true});const read=p=>JSON.parse(fs.readFileSync(p,'utf8'));
function run(args){const r=cp.spawnSync(process.execPath,args,{stdio:'inherit',env:process.env});assert.equal(r.status,0,'Browser child must exit normally')}
(async()=>{let browser;try{
  run([path.join(root,'deploy/case_attention/digitalocean-sgp/incremental-presentation-v1/check_browser.cjs'),base,path.join(output,'legacy'),...(credentials?[credentials]:[])]);
  run([path.join(root,'scripts/review/test_hehe_focal_patch_viewer_v1.cjs'),'--results',path.join(root,'reviews/hehe-focal-patching-v1/results-01'),'--url',base+'reports/patching/','--output',path.join(output,'patching.json'),...(credentials?['--credential',credentials]:[])]);
  browser=await chromium.launch({executablePath:'/tmp/paired-review-playwright/chromium_headless_shell-1234/chrome-headless-shell-linux64/chrome-headless-shell',headless:true,args:['--no-sandbox','--disable-gpu']});
  const context=await browser.newContext({...(credentials?{httpCredentials:read(credentials)}:{}),viewport:{width:390,height:844}});context.setDefaultTimeout(600000);
  const page=await context.newPage(),errors=[];page.on('pageerror',e=>errors.push(e.message));await page.goto(base,{timeout:600000});
  assert.equal(await page.locator('#patching').getAttribute('href'),'reports/patching/');await page.locator('#patching').click();await page.waitForFunction(()=>window.HFP&&HFP.current);
  assert((await page.locator('h1').textContent()).includes('表示替换'));assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+2));
  const downloads=[];
  for(const name of ['REPORT.html','REPORT.md','results.json','metrics.json','all-interventions.tsv','ALL-PROMPTS.md','PROTOCOL.md','interventions.json','patch-effects.pdf','patch-effects.svg','patch-effects.png','audit.json','browser-audit.json']){
    const r=await context.request.get(base+'reports/patching/'+name,{timeout:600000});assert.equal(r.status(),200);const body=await r.body();assert(body.length>0);if(name==='REPORT.html'){const text=body.toString();assert(text.includes('实验结果')&&!text.includes('href="/data/'))}downloads.push(name)
  }
  await page.click('#reportlink');assert((await page.locator('h1').first().textContent()).includes('实验结果'));assert.deepEqual(errors,[]);
  const legacy=read(path.join(output,'legacy/browser.json')),patching=read(path.join(output,'patching.json'));
  const receipt={status:'pass',base,legacy_values:legacy.checked_values,patching_values:patching.values_checked,checked_values:legacy.checked_values+patching.values_checked,
    legacy_checks:legacy.checks.map(({cache,...rest})=>rest),legacy_cache_limits_pass:true,new_all_layers_both_directions:true,exact_four_prompts:true,
    additive_navigation:true,downloads,mobile:true,errors};fs.writeFileSync(path.join(output,'browser.json'),JSON.stringify(receipt,null,2)+'\n',{flag:'wx'});console.log(JSON.stringify({status:'pass',values:receipt.checked_values}));
}finally{if(browser)await browser.close()}})().catch(e=>{fs.writeFileSync(path.join(output,'failure.json'),JSON.stringify({error:e.stack}));console.error(e);process.exit(1)});
