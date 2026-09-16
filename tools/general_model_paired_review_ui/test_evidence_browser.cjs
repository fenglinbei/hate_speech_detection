"use strict";
// Real browser + HTTP integration, using fictional materials and isolated sessions.
const assert = require("node:assert/strict");
const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");
const {spawn} = require("node:child_process");
const {chromium} = require("playwright");
const {expect} = require("@playwright/test");
const ROOT = path.resolve(__dirname, "../..");
const temp = fs.mkdtempSync(path.join(os.tmpdir(), "evidence-browser-"));
const bundleDir = path.join(temp, "bundle");
const bundleFile = path.join(bundleDir, "fixture.json");
const sessionFile = path.join(temp, "human-session.json");
const screenshots = path.join(temp, "screenshots");
fs.mkdirSync(bundleDir); fs.mkdirSync(screenshots);
let service, browser, baseUrl, serviceLog = "";
const errors = [];
function object(id, kind, source, values, depends_on=[]) {
  return {id,kind,source,depends_on,version:"fictional-source-v1",ai_draft:{review_kind:"ai_note",version:"fictional-ai-v1",values,provenance:{model:"automated-fixture",input_scope:"fictional materials only"}}};
}
const labelValues = {hate:"non-hate",group:[],hate_reason:"no_attack",group_reason:"no_group_target",stance:"neutral",target_types:["none"],expression_types:["neutral_or_laughter"],evidence:[],note:""};
const longDemoText = "今天适合散步。\n" + Array.from({length:28},(_,index)=>`虚构长文第 ${index+1} 段：沿着公园的小路慢慢行走，看树叶和天空，记录今天的天气。这些重复段落用于检查审核弹窗在完整正文较长时的滚动、勾选和退出行为。`).join("\n");
const objects = {
  q1:object("q1","query",{text:"🌿 今天很晴朗，出去散步。"},labelValues),
  q2:object("q2","query",{text:"明天也去散步。"},labelValues),
  demo:object("demo","demo",{text:longDemoText,demo_id:"fictional-demo",original_answer:{hate:"non-hate",group:[]}}, {...labelValues,hate_original_status:"accepted",group_original_status:"accepted"}),
  definition:object("definition","definition",{text:"散步指缓慢步行。",term:"散步"},{definition_verdict:"reasonable",issues:["valid_sense"],evidence:[],note:""}),
};
for(const i of [1,2]) {
  objects["r"+i]=object("r"+i,"relation",{texts:{query:objects["q"+i].source.text,demo:objects.demo.source.text}},{topic_hate:"partial",topic_group:"none",rule_hate:"direct",rule_group:"none",lexicon_risk:"no",evidence:[],note:""},["q"+i,"demo"]);
  objects["h"+i]=object("h"+i,"hit",{texts:{query:objects["q"+i].source.text,demo:objects.demo.source.text,definition:objects.definition.source.text},term:"散步",provenance:{status:"verified"}},{source_fit:"valid_sense",query_fit:"applicable",issues:["valid_sense"],evidence:[],note:""},["q"+i,"definition"]);
}
const cases={};
for(const i of [1,2])cases["c"+i]={split:"discovery",query_object_id:"q"+i,object_ids:["q"+i,"demo","definition","r"+i,"h"+i],comparison:{gold:{hate:"non-hate",group:[]},trajectories:{"0":{hate:{prediction:{labels:"non-hate"}},group:{prediction:{labels:[]}}}},ai_proposal:{candidate_explanation:"这是虚构浏览器测试材料。",alternative_explanation:"不产生真实审核结论。"}}};
cases.c1.comparison.original_quadruples=[{target:"测试原目标",argument:"测试原论述",targeted_group:["non-hate"],hateful:"non-hate"}];
cases.c1.comparison.contexts=[{condition:"C0",task:"hate",prompt_text:"隔离测试完整提示"}];
fs.writeFileSync(bundleFile,JSON.stringify({schema_version:"general-model-evidence-bundle/v1",policy:{version:"fixture-policy",sha256:"f".repeat(64)},source_identity:"fictional-browser-fixture",order:["c1","c2"],cases,objects}));
async function startService(){
  serviceLog="";
  service=spawn(process.env.PAIRED_REVIEW_PYTHON||"/usr/bin/python3",["scripts/stage1/general_model_paired_review.py","--session-file",path.join(temp,"old-isolated.json"),"--reviewer-id","automated-browser-test","--port","0","--evidence-bundle",bundleFile,"--evidence-session",sessionFile],{cwd:ROOT,env:{...process.env,PYTHONDONTWRITEBYTECODE:"1"},stdio:["ignore","pipe","pipe"]});
  service.stderr.on("data",data=>serviceLog+=data.toString());
  return new Promise((resolve,reject)=>{let stdout="";const timeout=setTimeout(()=>reject(new Error("Server startup timeout: "+serviceLog)),10000);service.once("exit",code=>{clearTimeout(timeout);reject(new Error("Server exited "+code+": "+serviceLog));});service.stdout.on("data",data=>{stdout+=data.toString();try{const info=JSON.parse(stdout.split("\n")[0]);if(info.url){clearTimeout(timeout);resolve(info.url);}}catch(_){}});});
}
async function stopService(){if(service&&service.exitCode===null)await new Promise(resolve=>{service.once("exit",resolve);service.kill("SIGTERM");});service=null;}
async function assertDialogControls(page, dialogId, controls) {
  // Inspect the current layout before click/scrollIntoView can rescue clipped UI.
  const geometry=await page.locator(dialogId).evaluate((dialog,ids)=>{
    const d=dialog.getBoundingClientRect();
    const visible={left:Math.max(d.left,0),top:Math.max(d.top,0),right:Math.min(d.right,innerWidth),bottom:Math.min(d.bottom,innerHeight)};
    return {dialogOverflow:dialog.scrollWidth-dialog.clientWidth,controls:ids.map(id=>{
      const element=document.querySelector(id),r=element.getBoundingClientRect();
      const hit=document.elementFromPoint(r.left+r.width/2,r.top+r.height/2);
      return {id,width:r.width,height:r.height,inside:r.left>=visible.left-1&&r.top>=visible.top-1&&r.right<=visible.right+1&&r.bottom<=visible.bottom+1,hit:hit===element||element.contains(hit)};
    })};
  },controls);
  assert.ok(geometry.dialogOverflow<=1,`${dialogId} must not overflow horizontally`);
  for(const control of geometry.controls){
    assert.ok(control.width>0&&control.height>0&&control.inside,`${control.id} must be fully inside dialog and viewport: ${JSON.stringify(control)}`);
    assert.ok(control.hit,`${control.id} must be directly reachable, not covered or clipped`);
    await expect(page.locator(control.id)).toBeInViewport({ratio:1});
  }
}
async function testBatchLayouts(url) {
  const revisions=async page=>(await(await page.request.get(baseUrl+"api/evidence/bootstrap")).json()).revision;
  for(const viewport of [{width:390,height:844},{width:1024,height:768},{width:1440,height:1000},{width:844,height:390}]){
    const context=await browser.newContext({viewport});
    const page=await context.newPage();page.on("pageerror",error=>errors.push(String(error)));
    await page.goto(url);await expect(page.locator("#item-id")).toHaveText("案例 #c1");
    for(const id of ["demo","definition","q1"]){await page.locator("#object-select").selectOption(id);await expect(page.locator("#object-select")).toHaveValue(id);}
    const revision=await revisions(page);
    await page.locator("#open-batch").click();await expect(page.locator("#batch-dialog")).toBeVisible();
    await expect(page.locator("#dismiss-batch")).toBeFocused();
    await expect(page.locator("#batch-items .batch-item")).toHaveCount(3);
    await expect(page.locator("#batch-items input:checked")).toHaveCount(0);
    await expect(page.locator("#confirm-batch")).toBeDisabled();
    await expect(page.locator("#batch-items")).toContainText("虚构长文第 28 段");
    await assertDialogControls(page,"#batch-dialog",["#dismiss-batch","#close-batch"]);
    const dimensions=await page.locator("#batch-items .batch-check").evaluateAll(labels=>labels.map(label=>{
      const input=label.querySelector("input"),text=label.querySelector("span"),box=input.getBoundingClientRect(),textBox=text.getBoundingClientRect();
      return {checkboxWidth:box.width,checkboxHeight:box.height,textWidth:textBox.width,labelOverflow:label.scrollWidth-label.clientWidth,textOverflow:text.scrollWidth-text.clientWidth};
    }));
    for(const row of dimensions){
      assert.ok(row.checkboxWidth>=8&&row.checkboxWidth<=24&&row.checkboxHeight>=8&&row.checkboxHeight<=24,`checkbox dimensions at ${viewport.width}×${viewport.height}: ${JSON.stringify(row)}`);
      assert.ok(row.textWidth>=Math.min(180,viewport.width*.4),`checkbox label needs readable line width: ${JSON.stringify(row)}`);
      assert.ok(row.labelOverflow<=1&&row.textOverflow<=1,`checkbox and text must fit their row: ${JSON.stringify(row)}`);
    }
    const body=page.locator("#batch-dialog .guide-body");
    assert.ok(await body.evaluate(element=>element.scrollHeight>element.clientHeight*2),"Long multi-item fixture must require substantial body scrolling");
    await page.screenshot({path:path.join(screenshots,`batch-open-${viewport.width}x${viewport.height}.png`)});
    const bounds=await body.boundingBox();assert.ok(bounds&&bounds.height>20,"dialog body must retain usable height");
    const beforeScroll=await body.evaluate(element=>element.scrollTop);
    await page.mouse.move(bounds.x+bounds.width/2,bounds.y+bounds.height/2);await page.mouse.wheel(0,Math.max(200,bounds.height*.8));
    await expect.poll(()=>body.evaluate(element=>element.scrollTop)).toBeGreaterThan(beforeScroll);
    await assertDialogControls(page,"#batch-dialog",["#dismiss-batch","#close-batch"]);
    await page.mouse.wheel(0,100000);
    await expect.poll(()=>body.evaluate(element=>element.scrollHeight-element.clientHeight-element.scrollTop)).toBeLessThanOrEqual(2);
    await assertDialogControls(page,"#batch-dialog",["#dismiss-batch","#close-batch"]);
    await page.screenshot({path:path.join(screenshots,`batch-scrolled-${viewport.width}x${viewport.height}.png`)});
    await page.locator("#dismiss-batch").click();await expect(page.locator("#batch-dialog")).not.toBeVisible();await expect(page.locator("#open-batch")).toBeFocused();
    assert.equal(await revisions(page),revision,"Closing an unchecked batch with × must not save or confirm");
    for(const action of ["#close-batch","Escape"]){
      await page.locator("#open-batch").click();await expect(page.locator("#batch-items input:checked")).toHaveCount(0);
      await assertDialogControls(page,"#batch-dialog",["#dismiss-batch","#close-batch"]);
      if(action==="Escape")await page.keyboard.press("Escape");else await page.locator(action).click();
      await expect(page.locator("#batch-dialog")).not.toBeVisible();await expect(page.locator("#open-batch")).toBeFocused();
      assert.equal(await revisions(page),revision,`${action} must close without changing the server revision`);
    }
    for(const action of ["#dismiss-guidelines","#close-guidelines","Escape"]){
      await page.locator("#guidelines").click();await expect(page.locator("#dismiss-guidelines")).toBeFocused();
      await assertDialogControls(page,"#guideline-dialog",["#dismiss-guidelines","#close-guidelines"]);
      if(action==="Escape")await page.keyboard.press("Escape");else await page.locator(action).click();
      await expect(page.locator("#guideline-dialog")).not.toBeVisible();await expect(page.locator("#guidelines")).toBeFocused();
      assert.equal(await revisions(page),revision,"Guidance interactions must not mutate the review");
    }
    await context.close();
  }
  console.log("PASS long multi-item batch at 390/1024/1440 and short landscape: bounded checkbox/text, real wheel scrolling, always-visible dismiss/return, Escape, focus restoration, unchanged revision");
}
async function main(){
  baseUrl=await startService(); browser=await chromium.launch({headless:true});
  const context=await browser.newContext({viewport:{width:1440,height:1000},acceptDownloads:true});
  const page=await context.newPage();page.on("pageerror",error=>errors.push(String(error)));
  const url=baseUrl+"evidence/";
  const state=async(key="c1")=>(await page.request.get(baseUrl+"api/evidence/items/"+key)).json();
  const boot=async()=>(await page.request.get(baseUrl+"api/evidence/bootstrap")).json();
  const choose=async(field,value)=>page.locator(`[data-field="${field}"] [data-value="${value}"]`).click();
  await testBatchLayouts(url);
  await page.goto(url);
  await expect(page.locator("#item-id")).toHaveText("案例 #c1");
  await expect(page.locator("#item-list .case-item")).toHaveCount(2);
  await expect(page.locator("#tab-comparison")).toBeDisabled();
  await expect(page.locator("#reveal-comparison")).toBeDisabled();
  assert.equal((await state()).comparison,null);assert.equal((await boot()).status.confirmed_object_count,0);
  await choose("hate","null");
  await expect(page.locator('[data-field="group_mode"] [data-value="empty"]')).toHaveAttribute("aria-pressed","true");
  await choose("hate","non-hate");await choose("group_mode","unknown");
  await expect(page.locator('[data-field="hate"] [data-value="non-hate"]')).toHaveAttribute("aria-pressed","true");
  await choose("group_mode","empty");await page.locator("#check-reasons").click();
  await page.locator("#review-note").fill("隔离测试：五秒自动保存");
  await page.waitForTimeout(1200);assert.equal((await state()).objects[0].review.status,"unreviewed");
  await expect(page.locator("#save-state")).toHaveAttribute("data-state","saved",{timeout:8000});
  assert.equal((await state()).objects[0].review.values.note,"隔离测试：五秒自动保存");
  assert.equal((await boot()).status.confirmed_object_count,0);
  console.log("PASS independent null/[] labels, delayed autosave, no automated adoption");

  await choose("group_mode","unknown");await page.locator("#check-reasons").click();await page.locator("#save-draft").click();
  await expect(page.locator("#save-state")).toHaveAttribute("data-state","saved");assert.equal((await state()).objects[0].review.values.group,null);
  const beforeIncompleteGroup=(await boot()).revision;
  await choose("group_mode","classes");await page.locator("#check-reasons").click();await page.waitForTimeout(5200);
  assert.equal((await state()).objects[0].review.values.group,null);assert.equal((await boot()).revision,beforeIncompleteGroup);
  await expect(page.locator('[data-field="group_mode"] [data-value="classes"]')).toHaveAttribute("aria-pressed","true");
  await expect(page.locator("#save-state")).toHaveAttribute("data-state","dirty");
  await expect(page.locator("#request-error-message")).toHaveText("请选择至少一个类别，或改选无类别 / 未决。");
  await page.locator("#save-draft").click();await page.locator("#primary-action").click();await page.locator("#next-item").click();
  await expect(page.locator("#item-id")).toHaveText("案例 #c1");assert.equal((await boot()).revision,beforeIncompleteGroup);assert.equal((await boot()).status.confirmed_object_count,0);
  const incompleteBackup=await page.evaluate(()=>Object.values(localStorage).map(value=>{try{return JSON.parse(value);}catch(_){return null;}}).find(value=>value?.item_id==="c1"));
  assert.equal(incompleteBackup.values._group_mode,"classes");assert.deepEqual(incompleteBackup.values.group,[]);
  await choose("group_mode","empty");await page.locator("#check-reasons").click();await page.locator("#save-draft").click();await expect(page.locator("#save-state")).toHaveAttribute("data-state","saved");
  console.log("PASS unfinished group selection remains local; autosave, manual save, confirmation and navigation cannot turn it into []");

  await page.locator("#object-material .source-text").evaluate(element=>{const selection=window.getSelection();const range=document.createRange();range.setStart(element.firstChild,3);range.setEnd(element.firstChild,5);selection.removeAllRanges();selection.addRange(range);element.dispatchEvent(new MouseEvent("mouseup",{bubbles:true}));});
  await page.locator("#add-evidence").click();await page.locator("#save-draft").click();
  await expect(page.locator("#save-state")).toHaveAttribute("data-state","saved");
  assert.deepEqual((await state()).objects[0].review.values.evidence,[{source:"text",start:2,end:4,text:"今天"}]);
  await page.locator("#review-note").fill("导航前必须保存");
  await page.route("**/api/evidence/save-object",route=>route.abort("failed"));
  await page.locator("#next-item").click();await expect(page.locator("#request-error")).toBeVisible();
  await expect(page.locator("#item-id")).toHaveText("案例 #c1");await expect(page.locator("#review-note")).toHaveValue("导航前必须保存");
  await page.unroute("**/api/evidence/save-object");await page.locator("#save-draft").click();
  await expect(page.locator("#save-state")).toHaveAttribute("data-state","saved");
  await page.locator("#next-item").click();await expect(page.locator("#item-id")).toHaveText("案例 #c2");
  await page.locator("#previous-item").click();await expect(page.locator("#review-note")).toHaveValue("导航前必须保存");
  console.log("PASS code-point evidence selection and failure-safe navigation/resume");

  const second=await browser.newContext({viewport:{width:1440,height:1000}});const other=await second.newPage();other.on("pageerror",error=>errors.push(String(error)));await other.goto(url);await expect(other.locator("#item-id")).toHaveText("案例 #c1");
  await page.locator("#review-note").fill("第一页保留的草稿");await other.locator("#review-note").fill("另一页面先保存");await other.locator("#save-draft").click();await expect(other.locator("#save-state")).toHaveAttribute("data-state","saved");
  await page.locator("#save-draft").click();await expect(page.locator("#recover-local")).toBeVisible();await expect(page.locator("#review-note")).toHaveValue("第一页保留的草稿");
  await page.locator("#recover-local").click();await expect(page.locator("#resume-notice")).toBeVisible();await expect(page.locator("#review-note")).toHaveValue("第一页保留的草稿");
  await page.waitForTimeout(5200);assert.equal((await state()).objects[0].review.values.note,"另一页面先保存");
  await page.locator("#save-draft").click();await expect(page.locator("#save-state")).toHaveAttribute("data-state","saved");assert.equal((await state()).objects[0].review.values.note,"第一页保留的草稿");
  console.log("PASS revision conflicts retain local draft, explicit recovery, explicit save");

  await other.reload();await expect(other.locator("#object-select")).toHaveValue("q1");
  await page.locator("#review-note").fill("远端确认后仍须保留的最初本地草稿");
  await other.locator("#review-note").fill("远端先确认的独立记录");await other.locator("#primary-action").click();await expect(other.locator("#save-state")).toHaveAttribute("data-state","saved");
  assert.equal((await state()).objects[0].review.status,"confirmed");
  await page.locator("#save-draft").click();await expect(page.locator("#recover-local")).toBeVisible();await page.locator("#recover-local").click();
  await expect(page.locator("#review-note")).toHaveValue("远端先确认的独立记录");await expect(page.locator("#locked-card")).toBeVisible();
  await page.route("**/api/evidence/reopen-object",route=>route.abort("failed"));
  await page.locator("#reopen").click();await page.locator("#confirm-reopen").click();await expect(page.locator("#save-state")).toHaveAttribute("data-state","error");
  const conflictBackup=await page.evaluate(()=>Object.values(localStorage).map(value=>{try{return JSON.parse(value);}catch(_){return null;}}).find(value=>value?.item_id==="c1"));
  assert.equal(conflictBackup.values.note,"远端确认后仍须保留的最初本地草稿");
  await page.locator("#cancel-reopen").click();await page.unroute("**/api/evidence/reopen-object");
  page.once("dialog",dialog=>dialog.accept());await page.reload();await expect(page.locator("#recover-local")).toBeVisible();
  await page.locator("#recover-local").click();await expect(page.locator("#review-note")).toHaveValue("远端先确认的独立记录");
  await page.locator("#reopen").click();await page.locator("#confirm-reopen").click();await expect(page.locator("#reopen-dialog")).not.toBeVisible();
  await page.locator("#recover-local").click();await expect(page.locator("#review-note")).toHaveValue("远端确认后仍须保留的最初本地草稿");await expect(page.locator("#resume-notice")).toBeVisible();
  await page.locator("#save-draft").click();await expect(page.locator("#save-state")).toHaveAttribute("data-state","saved");
  assert.equal((await state()).objects[0].review.values.note,"远端确认后仍须保留的最初本地草稿");assert.equal((await state()).objects[0].review.status,"draft");await second.close();
  console.log("PASS confirmed-server recovery and failed reopen preserve the original local draft across refresh");

  await page.locator("#object-select").selectOption("demo");await page.locator("#object-select").selectOption("definition");await page.locator("#object-select").selectOption("q1");
  await page.locator("#open-batch").click();await expect(page.locator("#batch-items input")).toHaveCount(3);await expect(page.locator("#batch-items input:checked")).toHaveCount(0);await expect(page.locator("#confirm-batch")).toBeDisabled();
  await page.locator('#batch-items input[value="q1"]').check();await page.locator('#batch-items input[value="demo"]').check();await expect(page.locator("#confirm-batch")).toHaveText("采纳所选 2 项");await page.locator("#confirm-batch").click();await expect(page.locator("#batch-dialog")).not.toBeVisible();assert.equal((await boot()).status.confirmed_object_count,2);assert.equal((await state()).objects.find(o=>o.id==="definition").review.status,"unreviewed");
  for(const id of ["definition","r1","h1"]){await page.locator("#object-select").selectOption(id);await expect(page.locator("#object-select")).toHaveValue(id);if(id==="definition"){await page.locator("#object-material .source-text").evaluate(element=>{const selection=window.getSelection();const range=document.createRange();range.setStart(element.firstChild,0);range.setEnd(element.firstChild,2);selection.removeAllRanges();selection.addRange(range);element.dispatchEvent(new MouseEvent("mouseup",{bubbles:true}));});await page.locator("#add-evidence").click();}await page.locator("#primary-action").click();await expect(page.locator("#save-state")).toHaveAttribute("data-state","saved");assert.equal((await state()).objects.find(o=>o.id===id).review.status,"confirmed");}
  assert.equal((await boot()).status.confirmed_count,0);assert.equal((await boot()).status.confirmed_object_count,5);
  await expect(page.locator("#reveal-comparison")).toBeEnabled();await page.locator("#reveal-comparison").click();await expect(page.locator("#panel-comparison")).toBeVisible();
  await page.getByText("原始四元组（1 条）",{exact:true}).click();await expect(page.getByText("测试原目标",{exact:true})).toBeVisible();
  await page.getByText("完整条件提示（1 份）",{exact:true}).click();await page.getByText("C0 · hate",{exact:true}).click();await expect(page.getByText("隔离测试完整提示",{exact:true})).toBeVisible();
  assert.equal((await state()).review.material_snapshots.length,1);assert.ok((await state()).comparison.gold);
  await expect(page.locator("#panel-comparison")).toContainText("hate 0");await expect(page.locator("#panel-comparison")).not.toContainText("[object Object]");
  await choose("hate_original_status","accepted");await choose("group_original_status","accepted");await choose("hate_use","reference_analysis");await choose("group_use","reference_analysis");await page.locator("#primary-action").click();await expect(page.locator("#item-id")).toHaveText("案例 #c2");
  assert.equal((await boot()).status.confirmed_count,1);assert.equal((await state()).review.review_kind,"human_with_ai");
  await page.locator("#object-select").selectOption("demo");await expect(page.locator("#reference-notice")).toContainText("引用既有人工审核");await expect(page.locator("#locked-card")).toBeVisible();
  await page.locator("#reopen").click();await page.locator("#confirm-reopen").click();await expect(page.locator("#reopen-dialog")).not.toBeVisible();await page.locator("#primary-action").click();await expect(page.locator("#save-state")).toHaveAttribute("data-state","saved");
  await page.locator("#previous-item").click();await expect(page.locator("#item-id")).toHaveText("案例 #c1");await page.locator("#object-select").selectOption("r1");await expect(page.locator("#reference-notice")).toContainText("引用版本已过期");await expect(page.locator("#primary-action")).toBeEnabled();await page.locator("#primary-action").click();await expect(page.locator("#save-state")).toHaveAttribute("data-state","saved");assert.equal((await state()).objects.find(o=>o.id==="r1").stale,false);
  console.log("PASS item confirmations, gated snapshot, human case confirmation, shared-version reuse and recheck");

  for(const viewport of [{width:1024,height:768},{width:390,height:844}]){const responsive=await browser.newContext({viewport});const small=await responsive.newPage();small.on("pageerror",error=>errors.push(String(error)));await small.goto(url);await expect(small.locator("#item-id")).toHaveText("案例 #c1");assert.ok(await small.evaluate(()=>document.documentElement.scrollWidth<=innerWidth));await small.locator("#sidebar-toggle").click();await expect(small.locator("#item-list .case-item").first()).toBeInViewport();await small.locator("#sidebar-close").click();await small.locator("#guidelines").click();await expect(small.locator("#close-guidelines")).toBeInViewport();await small.locator("#close-guidelines").click();await small.screenshot({path:path.join(screenshots,"review-"+viewport.width+".png"),fullPage:true});await responsive.close();}
  await page.screenshot({path:path.join(screenshots,"review-desktop.png")});
  await page.locator(".top-action-menu summary").click();const downloaded=page.waitForEvent("download");await page.locator("#export-json").click();const result=await downloaded;assert.ok(result.suggestedFilename().endsWith(".json"));
  await stopService();baseUrl=await startService();const fresh=await browser.newContext();const resumed=await fresh.newPage();await resumed.goto(baseUrl+"evidence/");await expect(resumed.locator("#item-id")).toHaveText("案例 #c1");const restarted=await(await resumed.request.get(baseUrl+"api/evidence/items/c1")).json();assert.equal(restarted.review.material_snapshots.length,1);assert.ok(restarted.objects.every(o=>o.review.status==="confirmed"));await fresh.close();
  assert.deepEqual(errors,[]);console.log("PASS responsive layouts, JSON download, server restart persistence; screenshots: "+screenshots);
}
main().catch(error=>{console.error(error);console.error(serviceLog);process.exitCode=1;}).finally(async()=>{if(browser)await browser.close();await stopService();});
