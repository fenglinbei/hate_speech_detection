"use strict";
// Production verification: no POSTs, adjudication edits, saves or confirmations.
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const https = require("node:https");
const {chromium} = require("playwright");
const {expect} = require("@playwright/test");
const credentials = JSON.parse(fs.readFileSync(process.env.HSD_REVIEW_CREDENTIALS_FILE, "utf8"));
const origin = "https://hsd.fenglin.pro";
let browser;
async function main() {
  browser = await chromium.launch({headless: true, args: ["--proxy-server=direct://", "--host-resolver-rules=MAP hsd.fenglin.pro 165.22.48.237"]});
  const unauth = {};
  for (const url of ["/evidence/", "/api/evidence/bootstrap", "/evidence/evidence.js", "/api/health"]) {
    const status = await new Promise((resolve, reject) => {
      const req = https.get({hostname: "165.22.48.237", servername: "hsd.fenglin.pro", path: url, headers: {Host: "hsd.fenglin.pro"}, timeout: 15000}, response => { response.resume(); resolve(response.statusCode); });
      req.on("error", reject); req.on("timeout", () => req.destroy(new Error("HTTPS timeout")));
    });
    unauth[url] = status;
    assert.equal(status, 401);
  }
  const context = await browser.newContext({httpCredentials: {username: credentials.username, password: credentials.password}, viewport: {width: 1440, height: 1000}});
  const page = await context.newPage();
  const errors = [], mutations = [];
  await page.route("**/api/**", route => ["GET", "HEAD"].includes(route.request().method()) ? route.continue() : route.abort());
  page.on("pageerror", e => errors.push(String(e)));
  page.on("request", r => { if (!["GET", "HEAD"].includes(r.method())) mutations.push(r.method()); });
  assert.equal((await page.goto(origin + "/evidence/", {waitUntil: "networkidle"})).status(), 200);
  await expect(page.locator("#item-list .case-item")).toHaveCount(32);
  const boot = await page.evaluate(async () => (await fetch("/api/evidence/bootstrap")).json());
  assert.equal(boot.reviewer_id, "liaozijie");
  assert.equal(boot.status.object_count, 1072);
  let policyCheck = null;
  if (process.env.HSD_EXPECT_POLICY_VERSION) {
    assert.equal(boot.policy.version, process.env.HSD_EXPECT_POLICY_VERSION);
    assert.equal(boot.bundle_policy.version, "evidence-applicability-annotation-policy/v1");
    assert.equal(boot.policy.bundle_sha256, boot.bundle_sha256);
    assert.deepEqual(boot.policy.impact.tasks, ["group"]);
    assert.equal(boot.policy.impact.object_ids.length, 632);
    assert.equal(boot.policy.impact.case_ids.length, 32);
    assert.equal(boot.policy_transition.human_confirmations_added, 0);
    assert.ok(boot.policy.document_text.includes("异性恋"));
    assert.ok(boot.choices.original_status.policy_changed);
    await expect(page.locator("#policy-card")).toBeVisible();
    await expect(page.locator("#policy-version")).toContainText("v2");
    await page.locator("#policy-details summary").click();
    await expect(page.locator("#policy-document")).toContainText("异性恋");
    await page.locator("#policy-details summary").click();
    policyCheck = {version: boot.policy.version, sha256: boot.policy.sha256,
      original_ai_policy: boot.bundle_policy.version, affected_group_objects: 632,
      confirmations_added_by_migration: 0};
  }
  const first = await page.evaluate(async () => (await fetch("/api/evidence/items/3683")).json());
  if (first.review.material_snapshots.length === 0) assert.equal(first.comparison, null);
  await expect(page.locator("#query-content")).not.toHaveText("等待载入…");
  await page.locator("#tab-materials").click();
  await expect(page.locator("#editor-fields .ai-notice")).toBeVisible();
  const shots = process.env.HSD_REVIEW_SCREENSHOTS;
  if (shots) { fs.mkdirSync(shots, {recursive: true}); await page.screenshot({path: path.join(shots, "evidence-production-desktop.png")}); }
  let materialNavigation = null;
  if (process.env.HSD_CHECK_MATERIAL_NAVIGATION === "1") {
    const selector = page.locator("#object-select");
    const previous = page.locator("#previous-object"), next = page.locator("#next-object");
    const ids = await selector.locator("option").evaluateAll(options => options.map(o => o.value));
    assert.ok(ids.length > 1);
    await expect(previous).toHaveAttribute("aria-keyshortcuts", "Alt+ArrowUp");
    await expect(next).toHaveAttribute("aria-keyshortcuts", "Alt+ArrowDown");
    await selector.selectOption(ids[0]);
    await expect(previous).toBeDisabled();
    await next.click(); await expect(selector).toHaveValue(ids[1]);
    await previous.click(); await expect(selector).toHaveValue(ids[0]);
    await page.evaluate(() => document.activeElement.blur());
    await page.keyboard.press("Alt+ArrowDown"); await expect(selector).toHaveValue(ids[1]);
    await page.keyboard.press("Alt+ArrowUp"); await expect(selector).toHaveValue(ids[0]);
    await selector.selectOption(ids.at(-1)); await expect(next).toBeDisabled();
    await page.locator('#object-kinds [data-kind="query"]').click();
    await expect(previous).toBeDisabled(); await expect(next).toBeDisabled();
    await page.locator('#object-kinds [data-kind="demo"]').click();
    const demoIds = await selector.locator("option").evaluateAll(options => options.map(o => o.value));
    assert.ok(demoIds.length > 1);
    await next.click(); await expect(selector).toHaveValue(demoIds[1]);
    const layouts = [];
    for (const viewport of [{width: 1440, height: 1000}, {width: 390, height: 844}, {width: 844, height: 390}]) {
      await page.setViewportSize(viewport);
      await previous.scrollIntoViewIfNeeded();
      await expect(previous).toBeInViewport({ratio: 1});
      await expect(next).toBeInViewport({ratio: 1});
      // Wait for the responsive sidebar transition to finish before assessing
      // reachability or capturing a viewport immediately after resizing.
      await expect.poll(() => page.locator(".object-navigation button").evaluateAll(buttons => buttons.map(button => {
        const r = button.getBoundingClientRect(), hit = document.elementFromPoint(r.left + r.width / 2, r.top + r.height / 2);
        return hit === button || button.contains(hit);
      }))).toEqual([true, true]);
      assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1));
      if (shots) await page.screenshot({path: path.join(shots, `evidence-material-navigation-${viewport.width}x${viewport.height}.png`), animations: "disabled"});
      layouts.push(viewport);
    }
    await page.locator('#object-kinds [data-kind="all"]').click();
    await selector.selectOption(ids[0]);
    materialNavigation = {buttons: true, shortcuts: ["Alt+ArrowUp", "Alt+ArrowDown"], boundaries: true, filtered_order: true, layouts};
  }
  const batchLayouts = [];
  for (const viewport of [{width: 1440, height: 1000}, {width: 390, height: 844}, {width: 844, height: 390}]) {
    await page.setViewportSize(viewport);
    await page.locator("#open-batch").click();
    await expect(page.locator("#dismiss-batch")).toBeFocused();
    const layout = await page.evaluate(() => {
      const dialog = document.querySelector("#batch-dialog"), bounds = dialog.getBoundingClientRect();
      const inside = selector => { const r = document.querySelector(selector).getBoundingClientRect(); return r.top >= bounds.top && r.bottom <= bounds.bottom && r.left >= bounds.left && r.right <= bounds.right && r.top >= 0 && r.bottom <= innerHeight; };
      return {header_visible: inside("#dismiss-batch"), footer_visible: inside("#close-batch"), overflow: dialog.scrollWidth > dialog.clientWidth + 1, checkbox_widths: [...dialog.querySelectorAll('input[type="checkbox"]')].map(input => input.getBoundingClientRect().width)};
    });
    assert.ok(layout.header_visible && layout.footer_visible && !layout.overflow);
    assert.ok(layout.checkbox_widths.every(width => width >= 16 && width <= 24));
    await expect(page.locator("#batch-items input:checked")).toHaveCount(0);
    const body = page.locator("#batch-dialog .guide-body");
    await body.hover(); await page.mouse.wheel(0, 1600); await page.waitForTimeout(150);
    await expect(page.locator("#dismiss-batch")).toBeInViewport({ratio: 1});
    await expect(page.locator("#close-batch")).toBeInViewport({ratio: 1});
    if (shots) await page.screenshot({path: path.join(shots, `evidence-batch-${viewport.width}x${viewport.height}.png`)});
    await page.locator("#close-batch").click();
    await expect(page.locator("#batch-dialog")).not.toBeVisible();
    await expect(page.locator("#open-batch")).toBeFocused();
    await page.locator("#open-batch").click(); await page.locator("#dismiss-batch").click();
    await expect(page.locator("#batch-dialog")).not.toBeVisible();
    await page.locator("#open-batch").click(); await page.keyboard.press("Escape");
    await expect(page.locator("#batch-dialog")).not.toBeVisible();
    batchLayouts.push({viewport, ...layout});
  }
  await page.setViewportSize({width: 390, height: 844});
  assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth + 1));
  if (policyCheck) {
    await page.locator("#policy-details summary").click();
    await expect(page.locator("#policy-document")).toBeVisible();
    assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth + 1));
    if (shots) await page.screenshot({path: path.join(shots, "evidence-policy-v2-mobile.png")});
    await page.locator("#policy-details summary").click();
  }
  if (shots) await page.screenshot({path: path.join(shots, "evidence-production-mobile.png")});
  await page.waitForTimeout(5500);
  const after = await page.evaluate(async () => (await fetch("/api/evidence/bootstrap")).json());
  assert.equal(after.revision, boot.revision, "Read-only browser load must not write evidence records");
  assert.deepEqual(mutations, []);
  assert.deepEqual(errors, []);
  const old = await page.evaluate(async () => (await fetch("/api/bootstrap")).json());
  console.log(JSON.stringify({status: "passed", auth_required: unauth, reviewer_id: boot.reviewer_id, evidence: boot.status, policy: policyCheck, material_navigation: materialNavigation, old_confirmed: old.status.confirmed_count, revision_unchanged: true, post_requests: 0, javascript_errors: 0, batch_layouts: batchLayouts, screenshots: shots || null}, null, 2));
}
main().catch(error => { console.error(error.message); process.exitCode = 1; }).finally(async () => { if (browser) await browser.close(); });
