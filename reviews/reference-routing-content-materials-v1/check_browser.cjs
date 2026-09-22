"use strict";
// All human-looking test decisions stay in an ephemeral browser context and /tmp.
const assert=require('node:assert/strict'), fs=require('node:fs'), path=require('node:path'), os=require('node:os');
const {chromium}=require('playwright');
async function main(){
 const root=path.resolve(__dirname,process.argv[2]||'build-01');
 const output=path.resolve(__dirname,process.argv[3]||'checks/browser-01');fs.mkdirSync(output,{recursive:true});
 const executable=process.env.RRC_BROWSER_EXECUTABLE||'/tmp/paired-review-playwright/chromium_headless_shell-1234/chrome-headless-shell-linux64/chrome-headless-shell';
 const browser=await chromium.launch({headless:true,executablePath:executable,args:['--disable-gpu']});
 try{
  const context=await browser.newContext({viewport:{width:1440,height:1000},acceptDownloads:true});
  const page=await context.newPage(),errors=[];page.on('pageerror',e=>errors.push(String(e)));
  await page.goto('file://'+path.join(root,'REVIEW.html'));
  assert.equal(await page.locator('#progress').textContent(),'0 / 732 已确认');
  assert.equal(await page.locator('#item-id').textContent(),'D01');
  assert.equal(await page.locator('#answer').inputValue(),'');
  assert.equal(await page.locator('#queue .case-item').count(),48);
  await page.screenshot({path:path.join(output,'desktop.png'),fullPage:true});
  await page.locator('#copy-ai').click();
  assert.equal(await page.locator('#answer').inputValue(),'无');
  assert.equal(await page.locator('#naturalness').inputValue(),'');
  assert.equal(await page.locator('#progress').textContent(),'0 / 732 已确认');
  await page.locator('#confirm').click();
  assert.match(await page.locator('#notice').textContent(),/审核人/);
  await page.locator('#reviewer').fill('automated-browser-test-NOT-HUMAN');
  await page.locator('#naturalness').selectOption('usable');
  await page.locator('#confirm').click();
  assert.equal(await page.locator('#progress').textContent(),'1 / 732 已确认');
  await page.reload();
  assert.equal(await page.locator('#progress').textContent(),'1 / 732 已确认');
  assert.equal(await page.locator('#answer').inputValue(),'无');
  assert(await page.locator('#answer').isDisabled());
  page.once('dialog',d=>d.accept('synthetic reopen test'));
  await page.locator('#reopen').click();
  assert.equal(await page.locator('#progress').textContent(),'0 / 732 已确认');
  assert(!(await page.locator('#answer').isDisabled()));
  await page.locator('#kind').selectOption('relation');
  assert.equal(await page.locator('#queue .case-item').count(),576);
  assert(await page.locator('#source-card').isVisible());
  assert.equal(await page.locator('#semantic').inputValue(),'');
  await page.locator('#copy-ai').click();
  assert.equal(await page.locator('#semantic').inputValue(),'direct');
  await page.locator('#confirm').click();
  assert.equal(await page.locator('#progress').textContent(),'1 / 732 已确认');
  const downloaded=page.waitForEvent('download');await page.locator('#export').click();
  const download=await downloaded, exportPath=await download.path(), payload=JSON.parse(fs.readFileSync(exportPath,'utf8'));
  assert.equal(payload.package_id,'reference-routing-content-materials-v1/draft-01');
  assert.equal(Object.values(payload.records).filter(r=>r.status==='confirmed').length,1);
  assert(payload.history.some(r=>r.event==='reopen'));
  assert(payload.records.D01.status==='draft');
  // Wrong source version must be rejected without deleting local work.
  const bad=path.join(os.tmpdir(),'rrc-synthetic-wrong-version-'+process.pid+'.json');
  fs.writeFileSync(bad,JSON.stringify({...payload,data_sha256:'incorrect'}));
  await page.locator('#import-file').setInputFiles(bad);
  await page.waitForFunction(()=>document.getElementById('notice').textContent.includes('其他材料版本'));
  assert.equal(await page.locator('#progress').textContent(),'1 / 732 已确认');
  fs.unlinkSync(bad);
  const mobile=await browser.newContext({viewport:{width:390,height:844}});
  const mp=await mobile.newPage();await mp.goto('file://'+path.join(root,'REVIEW.html'));
  assert.equal(await mp.locator('#progress').textContent(),'0 / 732 已确认');
  const width=await mp.evaluate(()=>({page:document.documentElement.scrollWidth,viewport:innerWidth}));
  assert(width.page<=width.viewport+2,JSON.stringify(width));
  const queueBox=await mp.locator('#queue').boundingBox();assert(queueBox.height>=120,'mobile queue collapsed');
  const sidebarBox=await mp.locator('.case-sidebar').boundingBox();assert(sidebarBox.height<700,'mobile sidebar contains inherited full-height blank space');
  await mp.screenshot({path:path.join(output,'mobile.png'),fullPage:true});
  assert.deepEqual(errors,[]);
  const receipt={status:'PASS',scope:'offline_workbench_ephemeral_synthetic_browser_checks',
   checks:['initial_human_fields_empty','AI_copy_does_not_confirm','reviewer_required','explicit_confirmation','reload_persistence','reopen_preserves_history','separate_relation_fields','version_bound_export','reject_wrong_version_import','fresh_mobile_context_empty','mobile_no_horizontal_overflow','no_browser_errors'],
   real_human_decisions_written:0,real_model_forwards:0,old_workbenches_mutated:false};
  fs.writeFileSync(path.join(output,'receipt.json'),JSON.stringify(receipt,null,2)+'\n');
  console.log(JSON.stringify(receipt));
 }finally{await browser.close();}
}
main().catch(e=>{console.error(e);process.exitCode=1;});
