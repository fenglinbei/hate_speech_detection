"use strict";
// Real HTTP/browser integration. All mutations use an isolated temporary session.
const assert = require("node:assert/strict");
const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");
const {spawn} = require("node:child_process");
const {chromium} = require("playwright");
const {expect} = require("@playwright/test");
const ROOT = path.resolve(__dirname, "../..");
const temp = fs.mkdtempSync(path.join(os.tmpdir(), "paired-human-browser-"));
const sessionFile = path.join(temp, "session.json");
const artifacts = process.env.PAIRED_REVIEW_SCREENSHOTS || path.join(temp, "screenshots");
fs.mkdirSync(artifacts, {recursive: true});
let service = null, browser = null, serviceLog = "";
const errors = [];

async function startService() {
  serviceLog = "";
  service = spawn(process.env.PAIRED_REVIEW_PYTHON || "/usr/bin/python3", [
    "scripts/stage1/general_model_paired_review.py", "--session-file", sessionFile,
    "--reviewer-id", "automated-browser-test", "--port", "0",
  ], {cwd: ROOT, env: {...process.env, PYTHONDONTWRITEBYTECODE: "1"}, stdio: ["ignore", "pipe", "pipe"]});
  service.stderr.on("data", data => { serviceLog += data.toString(); });
  return new Promise((resolve, reject) => {
    let text = "";
    const timeout = setTimeout(() => reject(new Error("Review server did not start: " + serviceLog)), 10_000);
    service.once("exit", code => { clearTimeout(timeout); reject(new Error("Review server exited " + code + ": " + serviceLog)); });
    service.stdout.on("data", data => {
      text += data.toString();
      const line = text.split("\n")[0];
      try { const info = JSON.parse(line); if (info.url) { clearTimeout(timeout); resolve(info.url); } } catch (_) { /* Wait for the complete startup line. */ }
    });
  });
}
async function stopService() {
  if (!service || service.exitCode !== null) return;
  const child = service;
  await new Promise(resolve => { child.once("exit", resolve); child.kill("SIGTERM"); });
  service = null;
}
async function main() {
  const url = await startService();
  browser = await chromium.launch({headless: true});
  const context = await browser.newContext({viewport: {width: 1440, height: 1000}, acceptDownloads: true});
  const page = await context.newPage();
  page.on("pageerror", error => errors.push(String(error)));
  await page.goto(url);
  await expect(page.locator("#item-id")).toHaveText("案例 #3683");
  await expect(page.locator("#item-list .case-item")).toHaveCount(12);
  await expect(page.locator("#top-progress-text")).toHaveText("首批 0 / 12 已确认");
  await expect(page.locator("#tab-trajectory")).toBeDisabled();
  await expect(page.locator("#tab-ai")).toBeDisabled();
  const state = async (key = "3683") => (await page.request.get(url + "api/items/" + key)).json();
  const boot = async () => (await page.request.get(url + "api/bootstrap")).json();
  let initial = await state();
  assert.equal(initial.trajectory, null);
  assert.equal(initial.ai_review, null);
  assert.equal(typeof initial.query, "string");
  assert.ok(!JSON.stringify(initial).includes("H_rescue"));
  assert.equal((await page.request.get(url + "api/prompt/3683?condition=CD&task=hate")).status(), 404);
  assert.equal((await page.request.get(url + "api/items/not-discovery")).status(), 404);
  assert.equal((await page.request.get(url + "results/paired-cases-02/cases/ai_review.csv")).status(), 404);
  const b = await boot();
  assert.equal((await page.request.post(url + "api/save", {data: {session_token: "bad", expected_revision: b.revision}})).status(), 403);
  assert.equal((await page.request.post(url + "api/save", {headers: {Origin: "https://unrelated.invalid"}, data: {session_token: b.session_token, expected_revision: b.revision}})).status(), 403);
  await page.screenshot({path: path.join(artifacts, "resources-desktop.png")});
  console.log("PASS real API resource-only phase, discovery scope, and existing origin/token protections");

  for (const viewport of [{width: 1024, height: 768}, {width: 390, height: 844}]) {
    const responsive = await browser.newContext({viewport});
    const small = await responsive.newPage();
    small.on("pageerror", error => errors.push(String(error)));
    await small.goto(url);
    await expect(small.locator("#item-id")).toHaveText("案例 #3683");
    assert.ok(await small.evaluate(() => document.documentElement.scrollWidth <= innerWidth));
    await small.locator("#sidebar-toggle").click();
    await expect(small.locator("#item-list .case-item").first()).toBeInViewport();
    await small.locator("#sidebar-close").click();
    await small.locator("#guidelines").click();
    await expect(small.getByRole("button", {name: "知道了，开始复核"})).toBeInViewport();
    await small.locator("#guide-step-1 > summary").click();
    await expect(small.locator("#guide-resource-examples .guide-example")).toHaveCount(4);
    await small.locator("#guide-resource-examples .guide-example").last().scrollIntoViewIfNeeded();
    await expect(small.locator("#guide-resource-examples .guide-example").last()).toBeInViewport();
    assert.ok(await small.locator("#guideline-dialog").evaluate(dialog => dialog.scrollWidth <= dialog.clientWidth));
    await expect(small.getByRole("button", {name: "知道了，开始复核"})).toBeInViewport();
    await small.screenshot({path: path.join(artifacts, "guidance-" + viewport.width + ".png")});
    await small.keyboard.press("Escape");
    if (viewport.width < 760) {
      await small.locator("#mobile-editor").click();
      await expect(small.locator("#ambiguity_stance")).toBeInViewport();
      await small.locator("#mobile-materials").click();
      await expect(small.locator("#query-content")).toBeInViewport();
    }
    await small.screenshot({path: path.join(artifacts, "resources-" + viewport.width + ".png")});
    await responsive.close();
  }
  console.log("PASS tablet/mobile layout, sidebar, guidance, and reading/editing navigation");

  await page.locator('[data-guide-section="guide-step-1"]').click();
  await expect(page.locator("#guide-step-1")).toHaveAttribute("open", "");
  await page.getByRole("button", {name: "知道了，开始复核"}).click();
  await expect(page.locator("#ambiguity_stance")).toHaveAccessibleDescription(/只读查询/);
  assert.equal((await state()).review.status, "unreviewed");

  const fields = ["ambiguity_stance", "definition_fit", "category_relation", "demo_correspondence"];
  for (const field of fields) await page.locator("#" + field).fill("自动化隔离测试观察：" + field);
  let release, signalHeld;
  const pending = new Promise(resolve => { release = resolve; });
  const held = new Promise(resolve => { signalHeld = resolve; });
  let saves = 0;
  await page.route("**/api/save", async route => {
    saves++;
    if (saves === 1) { signalHeld(); await pending; }
    await route.continue();
  });
  await page.locator("#save-draft").click();
  await held;
  await expect(page.locator("#definition_fit")).toBeEnabled();
  await page.locator("#definition_fit").fill("在保存请求进行时继续写下的新观察。");
  release();
  await expect.poll(async () => (await state()).review.resources.definition_fit, {timeout: 12_000}).toBe("在保存请求进行时继续写下的新观察。");
  await expect(page.locator("#save-state")).toHaveAttribute("data-state", "saved");
  assert.ok(saves >= 2);
  await page.unroute("**/api/save");
  assert.equal((await state()).trajectory, null);
  await page.reload();
  await expect(page.locator("#definition_fit")).toHaveValue("在保存请求进行时继续写下的新观察。");
  console.log("PASS in-flight edits survive autosave and reload resumes the saved draft");

  await page.locator("#ambiguity_stance").fill("断网时应保留的观察 <img id=unexpected-injection src=x>");
  await page.route("**/api/save", route => route.fulfill({status: 500, json: {error: "isolated test failure"}}));
  await page.locator("#next-item").click();
  await expect(page.locator("#request-error")).toBeVisible();
  await expect(page.locator("#item-id")).toHaveText("案例 #3683");
  await expect(page.locator("#ambiguity_stance")).toHaveValue("断网时应保留的观察 <img id=unexpected-injection src=x>");
  await page.unroute("**/api/save");
  await page.locator("#save-draft").click();
  await expect(page.locator("#save-state")).toHaveAttribute("data-state", "saved");
  await page.locator("#primary-action").click();
  await expect(page.locator("#tab-trajectory")).toBeEnabled();
  await expect(page.locator("#trajectory-rows tr")).toHaveCount(6);
  await expect(page.locator("#definition_fit")).toBeDisabled();
  assert.equal(await page.locator("#unexpected-injection").count(), 0);
  assert.ok((await state()).review.resources_locked_at);
  const originalResourceHash = (await state()).review.resources_sha256;
  await page.locator("#open-prompt").click();
  await expect(page.locator("#prompt-text")).not.toHaveText("正在读取…");
  assert.ok((await page.locator("#prompt-text").textContent()).length > 100);
  await page.locator("#close-prompt").click();
  await page.screenshot({path: path.join(artifacts, "trajectory-desktop.png")});
  console.log("PASS failed save blocks navigation, frozen initial notes, literal text, and full prompt display");

  await page.locator("#reveal-ai").click();
  await expect(page.locator('[data-error="gold_verdict"]')).toBeVisible();
  await expect(page.locator("#tab-ai")).toBeDisabled();
  await page.locator('[data-gold="agree"]').click();
  await expect(page.locator("#gold_dispute-requirement")).toHaveText("认可时可选");
  await page.locator('[data-gold="dispute"]').click();
  await expect(page.locator("#gold_dispute-requirement")).toHaveText("当前必填");
  await page.locator('[data-gold="agree"]').click();
  await page.locator("#stage2_candidate_explanation").fill("自动化测试的人工初判候选解释。");
  await page.locator("#alternative_explanation").fill("自动化测试的替代解释。");
  await page.locator("#falsifiable_followup").fill("测试等长和同位置的输入对照。");
  await page.locator('[data-disposition="input_control"]').click();
  await expect(page.locator("#falsifiable_followup-requirement")).toHaveText("当前必填");
  await page.locator("#reveal-ai").click();
  await expect(page.locator("#tab-ai")).toBeEnabled();
  await expect(page.locator("#ai-content .ai-section").first()).toBeVisible();
  assert.equal((await state()).review.pre_ai_assessment.stage2_candidate_explanation, "自动化测试的人工初判候选解释。");
  assert.equal((await boot()).status.confirmed_count, 0);
  await page.locator("#ai_comparison").fill("这是隔离测试，不构成真实人工审阅。");
  await page.locator("#primary-action").click();
  await expect(page.locator("#item-id")).toHaveText("案例 #5086");
  await expect(page.locator("#top-progress-text")).toHaveText("首批 1 / 12 已确认");
  await page.locator("#previous-item").click();
  await expect(page.locator("#item-id")).toHaveText("案例 #3683");
  await expect(page.locator("#locked-card")).toBeVisible();
  await expect(page.locator("#save-draft")).toBeDisabled();
  console.log("PASS AI disclosure follows the saved human draft, explicit confirmation, and fixed next-case order");

  await page.locator("#reopen").click();
  await page.locator("#reopen-reason").fill("隔离测试：补充核验意见");
  await page.locator("#confirm-reopen").click();
  await expect(page.locator("#stage2_candidate_explanation")).toBeEnabled();
  const secondContext = await browser.newContext({viewport: {width: 1440, height: 1000}});
  const other = await secondContext.newPage();
  other.on("pageerror", error => errors.push(String(error)));
  await other.goto(url);
  await expect(other.locator("#item-id")).toHaveText("案例 #3683");
  await page.locator("#stage2_candidate_explanation").fill("第一个页面尚未保存的意见。");
  await other.locator("#stage2_candidate_explanation").fill("第二个页面先保存的意见。");
  await other.locator("#save-draft").click();
  await expect(other.locator("#save-state")).toHaveAttribute("data-state", "saved");
  await page.locator("#save-draft").click();
  await expect(page.locator("#recover-local")).toBeVisible();
  await expect(page.locator("#stage2_candidate_explanation")).toHaveValue("第一个页面尚未保存的意见。");
  assert.equal((await state()).review.assessment.stage2_candidate_explanation, "第二个页面先保存的意见。");
  await page.locator("#recover-local").click();
  await expect(page.locator("#resume-notice")).toBeVisible();
  await expect(page.locator("#stage2_candidate_explanation")).toHaveValue("第一个页面尚未保存的意见。");
  assert.equal((await state()).review.assessment.stage2_candidate_explanation, "第二个页面先保存的意见。");
  await page.locator("#save-draft").click();
  await expect(page.locator("#save-state")).toHaveAttribute("data-state", "saved");
  assert.equal((await state()).review.assessment.stage2_candidate_explanation, "第一个页面尚未保存的意见。");
  assert.equal((await state()).review.resources_sha256, originalResourceHash);
  await secondContext.close();
  console.log("PASS reopen retains history; revision conflicts preserve local edits and require explicit recovery/save");

  await page.locator('[data-scope="all"]').click();
  await expect(page.locator("#item-list .case-item")).toHaveCount(32);
  await page.locator('[data-scope="initial"]').click();
  await expect(page.locator("#item-list .case-item")).toHaveCount(12);
  await page.locator("#item-search").fill("3683");
  await expect(page.locator("#item-list .case-item")).toHaveCount(1);
  await page.locator("#item-search").fill("");
  await page.locator(".top-action-menu summary").click();
  const downloadPromise = page.waitForEvent("download");
  await page.locator("#export-csv").click();
  const exported = await downloadPromise;
  const csv = fs.readFileSync(await exported.path(), "utf8");
  assert.ok(csv.includes("human_with_ai_nonblind") && csv.includes("第一个页面尚未保存的意见。"));
  assert.equal((await boot()).status.confirmed_count, 0);
  const persisted = JSON.parse(fs.readFileSync(sessionFile, "utf8"));
  assert.ok(persisted.events.some(event => event.action === "reopen" && event.previous_record.status === "confirmed"));
  const revision = persisted.revision;
  await context.close();
  await stopService();
  const resumedUrl = await startService();
  const resumed = await browser.newPage();
  await resumed.goto(resumedUrl);
  await expect(resumed.locator("#stage2_candidate_explanation")).toHaveValue("第一个页面尚未保存的意见。");
  assert.equal(JSON.parse(fs.readFileSync(sessionFile, "utf8")).revision, revision);
  await resumed.close();
  assert.deepEqual(errors, []);
  console.log("PASS 12/32 scope, search, usable CSV export, amendments, and process restart resume");
  console.log("Screenshots: " + artifacts);
}
main().catch(error => {
  console.error(error);
  console.error(serviceLog.slice(-4000));
  process.exitCode = 1;
}).finally(async () => {
  if (browser) await browser.close();
  await stopService();
  console.log("Isolated test session: " + sessionFile);
});
