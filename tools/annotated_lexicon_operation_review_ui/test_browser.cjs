"use strict";
// Fully intercepted in-memory API: no connection to any review session.
// Browser runtime: /tmp/node-v22.19.0-linux-x64/bin/node (Playwright requires >=20).
const assert = require("node:assert/strict");
const path = require("node:path");
const {chromium} = require("playwright");
const {expect} = require("@playwright/test");
const Ops = require("./core.js");
const clone = value => JSON.parse(JSON.stringify(value));
const entry = (term, id, definition) => ({lexicon_id: `lex-${id}`, term, variants: [], senses: [{sense_id: `sense-${id}`, definition, categories: ["others"]}], match_policy: {require_any: [], exclude_any: []}});
const items = [
  {item_id: "one", term: "它", operation_kind: "existing", flags: [], source_entries: [{source_row_index: 1, term: "它", category: "others", definition: "原始 <img src=x onerror=alert(1)>"}], proposed_entry: entry("它", "one", "依赖语境的贬损指代"), rationale: "核对上下文", open_questions: [], evidence: [{source_item_id: "blind-one", query_content: "😀它和它<script>alert(1)</script>", expected_spans: [{surface: "它", span: [1, 2]}, {surface: "它", span: [3, 4]}], dropped_candidates: [{surface: "它", span: [1, 2]}], review_notes: "<iframe>未被执行</iframe>"}], external_evidence: [{title: "不安全链接", url: "javascript:alert(1)", note: "应仅呈现文本"}, {title: "资料", url: "https://example.com/evidence", note: "仅作参考"}]},
  {item_id: "two", term: "yp", operation_kind: "new_entry", flags: [], source_entries: [], proposed_entry: entry("yp", "two", ""), rationale: "须核实", open_questions: ["当前义项不确定"], evidence: [], external_evidence: []},
];
async function fixture(browser, viewport = {width: 1440, height: 1000}) {
  const context = await browser.newContext({viewport});
  const page = await context.newPage();
  const decisions = Object.fromEntries(items.map(item => [item.item_id, {...Ops.defaultDecision(item), status: "draft"}]));
  const state = {revision: 1, saves: [], errors: [], conflictOnce: false, decisions, delayedSave: null};
  page.on("pageerror", error => state.errors.push(String(error)));
  const summary = item => ({item_id: item.item_id, term: item.term, operation_kind: item.operation_kind, flags: item.flags, query_preview: item.rationale, status: decisions[item.item_id].status, resolution: decisions[item.item_id].resolution});
  const status = () => ({item_count: items.length, confirmed_count: Object.values(decisions).filter(row => row.status === "confirmed").length, deferred_count: Object.values(decisions).filter(row => row.status === "deferred").length, open_count: Object.values(decisions).filter(row => row.status !== "confirmed").length, amendment_count: 0, finalized_bundle_id: null});
  const mutation = item => ({revision: String(state.revision), status: status(), item_summary: summary(item), decision: decisions[item.item_id]});
  await page.route("**/*", async route => {
    const url = new URL(route.request().url());
    assert.equal(url.origin, "http://operation-review.test");
    const json = data => route.fulfill({json: data});
    if (url.pathname === "/api/bootstrap") return json({frame_id: "test-only", revision: String(state.revision), status: status(), session_token: "mock", warnings: [], items: items.map(summary)});
    if (url.pathname.startsWith("/api/items/")) { const item = items.find(row => row.item_id === url.pathname.split("/").pop()); return json({revision: String(state.revision), item, decision: decisions[item.item_id], item_summary: summary(item)}); }
    if (url.pathname === "/api/save") {
      const payload = route.request().postDataJSON();
      if (state.delayedSave) await state.delayedSave;
      if (state.conflictOnce) { state.conflictOnce = false; state.revision += 1; return route.fulfill({status: 409, json: {error: "revision conflict"}}); }
      assert.equal(payload.expected_revision, String(state.revision));
      const item = items.find(row => row.item_id === payload.item_id);
      assert.deepEqual(Ops.validateDecision(item, payload.decision, {confirm: payload.confirm}), {});
      state.saves.push(clone(payload)); state.revision += 1;
      decisions[item.item_id] = {...payload.decision, status: payload.confirm ? payload.decision.resolution === "defer" ? "deferred" : "confirmed" : "draft"};
      return json(mutation(item));
    }
    if (url.pathname === "/api/reopen") {
      const payload = route.request().postDataJSON();
      assert.equal(payload.expected_revision, String(state.revision)); assert.ok(payload.reason);
      decisions[payload.item_id].status = "draft"; state.revision += 1;
      return json(mutation(items.find(row => row.item_id === payload.item_id)));
    }
    if (url.pathname === "/api/export") return json({revision: String(state.revision), decisions, checksum: "fixture-checksum"});
    const assets = {"/": "index.html", "/index.html": "index.html", "/app.js": "app.js", "/core.js": "core.js", "/styles.css": "styles.css", "/review-core.js": "../wp3_candidate_review_ui/core.js", "/review-base.css": "../wp3_candidate_review_ui/styles.css"};
    return assets[url.pathname] ? route.fulfill({path: path.resolve(__dirname, assets[url.pathname])}) : route.fulfill({status: 404});
  });
  await page.goto("http://operation-review.test/");
  await expect(page.locator("#item-id")).toHaveText("它");
  await expect(page.locator("#save-draft")).toBeEnabled();
  return {context, page, state};
}
async function main() {
  const browser = await chromium.launch();
  async function run(name, fn, viewport) { const f = await fixture(browser, viewport); try { await fn(f); assert.deepEqual(f.state.errors, []); console.log(`PASS ${name}`); } finally { await f.context.close(); } }
  try {
    if (process.env.OPERATION_REVIEW_LIVE_URL) {
      const origin = new URL(process.env.OPERATION_REVIEW_LIVE_URL);
      assert.ok(["127.0.0.1", "localhost"].includes(origin.hostname), "live inspection is local-only");
      for (const viewport of [{width: 1440, height: 1000}, {width: 390, height: 844}]) {
        const context = await browser.newContext({viewport});
        const page = await context.newPage();
        const errors = [];
        page.on("pageerror", error => errors.push(String(error)));
        await page.goto(origin.href);
        await expect(page.locator("#save-draft")).toBeEnabled();
        await expect(page.locator("#item-id")).not.toHaveText("正在读取审核项");
        assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth));
        if (process.env.OPERATION_REVIEW_SCREENSHOT_DIR) await page.screenshot({path: path.join(process.env.OPERATION_REVIEW_SCREENSHOT_DIR, `operation-real-workbench-${viewport.width}.png`)});
        await page.locator("#guidelines").click();
        await expect(page.getByRole("button", {name: "知道了，开始审核"})).toBeInViewport();
        if (process.env.OPERATION_REVIEW_SCREENSHOT_DIR) await page.screenshot({path: path.join(process.env.OPERATION_REVIEW_SCREENSHOT_DIR, `operation-real-guide-${viewport.width}.png`)});
        await page.keyboard.press("Escape");
        assert.deepEqual(errors, []);
        await context.close();
      }
      console.log("PASS read-only real-frame local API visual inspection");
      return;
    }
    await run("resolution shortcuts, input/IME/dialog/busy and locked guardrails", async ({page, state}) => {
      await page.keyboard.press("1");
      await expect(page.locator('[data-resolution="approve"]')).toHaveAttribute("aria-pressed", "true");
      assert.equal(state.saves.length, 0);
      await page.locator("#notes").fill("123[]");
      await page.keyboard.press("3");
      await expect(page.locator('[data-resolution="approve"]')).toHaveAttribute("aria-pressed", "true");
      await page.locator("#item-id").click();
      await page.evaluate(() => window.dispatchEvent(new KeyboardEvent("keydown", {key: "4", isComposing: true, bubbles: true})));
      await expect(page.locator('[data-resolution="approve"]')).toHaveAttribute("aria-pressed", "true");
      await page.locator("#guidelines").click();
      await page.keyboard.press("4"); await page.keyboard.press("Control+Enter");
      assert.equal(state.saves.length, 0);
      await page.keyboard.press("Escape");
      let release; state.delayedSave = new Promise(resolve => { release = resolve; });
      await page.keyboard.press("Control+s");
      await expect(page.locator("#save-draft")).toBeDisabled();
      await page.keyboard.press("4");
      release(); state.delayedSave = null;
      await expect(page.locator("#save-state")).toHaveAttribute("data-state", "saved");
      await expect(page.locator('[data-resolution="approve"]')).toHaveAttribute("aria-pressed", "true");
      await page.keyboard.press("Control+Enter");
      await expect(page.locator("#item-id")).toHaveText("yp");
      await page.keyboard.press("[");
      await expect(page.locator("#item-id")).toHaveText("它");
      await expect(page.locator("#save-draft")).toBeDisabled();
      await page.keyboard.press("4");
      await expect(page.locator('[data-resolution="approve"]')).toHaveAttribute("aria-pressed", "true");
      page.once("dialog", dialog => dialog.accept("修订理由")); await page.locator("#reopen").click();
      await expect(page.locator("#save-draft")).toBeEnabled();
    });
    await run("structured edits preserve IDs, empty senses block, deferral requires notes", async ({page, state}) => {
      const definition = page.locator(".sense-definition").first();
      await definition.fill("修订用途");
      await expect(page.locator('[data-resolution="revise"]')).toHaveAttribute("aria-pressed", "true");
      await page.locator("#save-draft").click();
      await expect(page.locator("#save-state")).toHaveAttribute("data-state", "saved");
      assert.equal(state.saves.at(-1).decision.entry.senses[0].sense_id, "sense-one");
      await page.locator("#add-sense").click();
      await page.locator("#confirm-next").click();
      await expect(page.locator("#senses-error")).toBeVisible();
      page.once("dialog", dialog => dialog.accept());
      await page.locator(".sense-card").nth(1).getByRole("button", {name: "移除义项"}).click();
      await page.locator("#confirm-next").click();
      await expect(page.locator("#item-id")).toHaveText("yp");
      await page.keyboard.press("1"); await page.keyboard.press("Control+Enter");
      await expect(page.locator("#senses-error")).toContainText("释义为空");
      await page.keyboard.press("3"); await page.keyboard.press("Control+Enter");
      await expect(page.locator("#notes-error")).toContainText("说明理由");
      await page.locator("#notes").fill("当前义项尚缺上下文依据");
      await page.keyboard.press("Control+Enter");
      await expect(page.locator("#decision-status")).toHaveAttribute("data-state", "deferred");
      await expect(page.locator("#top-progress-text")).toContainText("1 / 2 已确认 · 1 暂缓");
      await expect(page.locator("#save-draft")).toBeDisabled();
    });
    await run("safe rendering, Unicode evidence and bounded search", async ({page}) => {
      assert.deepEqual(await page.locator(".query-content mark").allTextContents(), ["它", "它"]);
      await expect(page.locator(".query-content")).toHaveText("😀它和它<script>alert(1)</script>");
      assert.equal(await page.locator(".source-pane script, .source-pane img, .source-pane iframe").count(), 0);
      assert.equal(await page.locator('a[href^="javascript:"]').count(), 0);
      await expect(page.locator('#external-evidence a')).toHaveAttribute("rel", "noopener noreferrer");
      await page.locator("#item-search").fill("yp");
      await expect(page.locator("#item-list .case-item")).toHaveCount(1);
      await expect(page.locator("#item-position")).toContainText("不在筛选");
      await page.locator("#item-list .case-item").click();
      await expect(page.locator("#item-id")).toHaveText("yp");
      await expect(page.locator("#previous-item")).toBeDisabled();
    });
    await run("CAS conflict preserves local draft without auto-overwrite", async ({page, state}) => {
      await page.locator(".sense-definition").fill("待保留的本地修改");
      state.conflictOnce = true;
      await page.locator("#save-draft").click();
      await expect(page.locator("#request-error")).toBeVisible();
      await expect(page.locator(".sense-definition")).toHaveValue("依赖语境的贬损指代");
      assert.equal(state.saves.length, 0);
      page.once("dialog", dialog => dialog.accept());
      await page.locator("#recover-local").click();
      await expect(page.locator(".sense-definition")).toHaveValue("待保留的本地修改");
      assert.equal(state.saves.length, 0);
      await page.locator("#save-draft").click();
      await expect(page.locator("#save-state")).toHaveAttribute("data-state", "saved");
      assert.equal(state.saves[0].expected_revision, "2");
    });
    await run("autosave stays draft and snapshot flushes pending edits", async ({page, state}) => {
      await page.clock.install();
      await page.locator("#notes").fill("自动保存联调");
      await page.clock.fastForward(5100);
      await expect(page.locator("#save-state")).toHaveAttribute("data-state", "saved");
      assert.equal(state.saves.length, 1);
      assert.equal(state.saves[0].confirm, false);
      assert.equal(state.decisions.one.status, "draft");
      await page.locator("#notes").fill("下载前保存当前草稿");
      const download = page.waitForEvent("download");
      await page.locator("#export-snapshot").click();
      const snapshot = await download;
      assert.match(snapshot.suggestedFilename(), /^repair-operation-review-/);
      assert.equal(state.saves.length, 2);
      assert.equal(state.saves[1].decision.notes, "下载前保存当前草稿");
      assert.equal(state.saves[1].confirm, false);
    });
    for (const viewport of [{width: 1440, height: 1000}, {width: 1024, height: 768}, {width: 390, height: 844}]) {
      await run(`workbench/help layout ${viewport.width}px`, async ({page}) => {
        assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth));
        if (viewport.width >= 768) {
          const source = await page.locator(".source-pane").boundingBox(), decision = await page.locator(".decision-pane").boundingBox();
          assert.ok(source.width >= 360); assert.ok(source.x + source.width <= decision.x + 1);
        }
        if (process.env.OPERATION_REVIEW_SCREENSHOT_DIR) await page.screenshot({path: path.join(process.env.OPERATION_REVIEW_SCREENSHOT_DIR, `operation-workbench-${viewport.width}.png`)});
        await page.locator("#guidelines").click();
        await expect(page.locator("#guideline-title")).toContainText("先解释用法");
        await expect(page.locator(".guide-shortcuts")).toContainText("Ctrl/⌘ Enter");
        const bounds = await page.locator("#guideline-dialog").boundingBox();
        assert.ok(bounds.x >= 0 && bounds.y >= 0 && bounds.x + bounds.width <= viewport.width + 1 && bounds.y + bounds.height <= viewport.height + 1);
        assert.ok(await page.locator(".guide-body").evaluate(el => el.scrollWidth <= el.clientWidth + 1));
        await expect(page.getByRole("button", {name: "知道了，开始审核"})).toBeInViewport();
        if (process.env.OPERATION_REVIEW_SCREENSHOT_DIR) await page.screenshot({path: path.join(process.env.OPERATION_REVIEW_SCREENSHOT_DIR, `operation-guide-${viewport.width}.png`)});
        await page.getByRole("button", {name: "知道了，开始审核"}).click();
      }, viewport);
    }
  } finally { await browser.close(); }
}
main().catch(error => { console.error(error); process.exitCode = 1; });
