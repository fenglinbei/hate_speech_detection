"use strict";
// Authenticated HTTPS smoke test. Refuses to edit a real reviewer's session.
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const {chromium} = require("playwright");
const {expect} = require("@playwright/test");

const credentialsPath = process.env.HSD_REVIEW_CREDENTIALS_FILE;
if (!credentialsPath) throw new Error("Set HSD_REVIEW_CREDENTIALS_FILE to the private login JSON file.");
const credentials = JSON.parse(fs.readFileSync(credentialsPath, "utf8"));
const origin = "https://hsd.fenglin.pro";
const screenshots = process.env.HSD_REVIEW_SCREENSHOTS;
const credentialsOptions = {username: credentials.username, password: credentials.password};
let browser;

async function main() {
  browser = await chromium.launch({headless: true, args: [
    "--proxy-server=direct://",
    "--host-resolver-rules=MAP hsd.fenglin.pro 165.22.48.237",
  ]});
  const context = await browser.newContext({httpCredentials: credentialsOptions, viewport: {width: 1440, height: 1000}});
  const page = await context.newPage();
  const errors = [];
  page.on("pageerror", error => errors.push(String(error)));
  const response = await page.goto(origin, {waitUntil: "networkidle"});
  assert.equal(response.status(), 200);
  const bootstrap = await page.evaluate(async () => (await fetch("/api/bootstrap")).json());
  assert.equal(bootstrap.reviewer_id, "automated-deployment-test", "Refusing to edit a real reviewer's session.");
  assert.equal(bootstrap.status.confirmed_count, 0, "Use a fresh isolated deployment session.");
  assert.equal(bootstrap.status.item_count, 32);
  await expect(page.locator("#guidelines")).toHaveText("怎么标注");
  await expect(page.locator("#item-id")).toHaveText("案例 #3683");
  await expect(page.locator("#tab-trajectory")).toBeDisabled();
  for (const field of ["ambiguity_stance", "definition_fit", "category_relation", "demo_correspondence"]) {
    await page.locator("#" + field).fill("自动化部署验证（独立会话，不是人工结论）：" + field);
  }
  await page.locator("#save-draft").click();
  await expect(page.locator("#save-state")).toHaveAttribute("data-state", "saved");
  await page.reload({waitUntil: "networkidle"});
  await expect(page.locator("#definition_fit")).toHaveValue("自动化部署验证（独立会话，不是人工结论）：definition_fit");
  await page.locator("#primary-action").click();
  await expect(page.locator("#tab-trajectory")).toBeEnabled();
  await expect(page.locator("#trajectory-rows tr")).toHaveCount(6);
  await page.locator('[data-gold="uncertain"]').click();
  await expect(page.locator("#gold_dispute-requirement")).toHaveText("当前必填");
  await page.locator("#gold_dispute").fill("独立部署测试不判断本例 Gold，只验证表单保存。");
  await page.locator("#stage2_candidate_explanation").fill("自动化测试记录，不构成人工候选解释。");
  await page.locator("#alternative_explanation").fill("自动化测试记录，不构成人工替代解释。");
  await page.locator('[data-disposition="verify_first"]').click();
  await page.locator("#patching_defer_reason").fill("仅用于独立部署验证，不用于正式实验。");
  await page.locator("#primary-action").click();
  await expect(page.locator("#top-progress-text")).toHaveText("首批 1 / 12 已确认");
  await expect(page.locator("#item-id")).toHaveText("案例 #5086");
  await page.reload({waitUntil: "networkidle"});
  await expect(page.locator("#top-progress-text")).toHaveText("首批 1 / 12 已确认");
  if (screenshots) {
    fs.mkdirSync(screenshots, {recursive: true});
    await page.screenshot({path: path.join(screenshots, "https-desktop.png")});
  }
  const mobile = await browser.newContext({httpCredentials: credentialsOptions, viewport: {width: 390, height: 844}});
  const small = await mobile.newPage();
  small.on("pageerror", error => errors.push(String(error)));
  await small.goto(origin, {waitUntil: "networkidle"});
  await small.locator("#guidelines").click();
  await small.locator("#guide-step-1 > summary").click();
  await expect(small.locator("#guide-resource-examples .guide-example")).toHaveCount(4);
  await expect(small.getByRole("button", {name: "知道了，开始复核"})).toBeInViewport();
  assert.ok(await small.evaluate(() => document.documentElement.scrollWidth <= innerWidth));
  if (screenshots) await small.screenshot({path: path.join(screenshots, "https-guidance-mobile.png")});
  assert.deepEqual(errors, []);
  console.log("PASS trusted HTTPS login, resource notes/save/reload, phase gate, confirm/next, and mobile guidance in an isolated session.");
}

main().catch(error => { console.error(error.message); process.exitCode = 1; }).finally(async () => {
  if (browser) await browser.close();
});
