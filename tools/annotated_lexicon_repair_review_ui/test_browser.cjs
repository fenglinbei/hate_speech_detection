"use strict";

// Fully intercepted, in-memory API: never contacts or mutates a review session.
// Run with Node >=20: node tools/annotated_lexicon_repair_review_ui/test_browser.cjs
const assert = require("node:assert/strict");
const path = require("node:path");
const {chromium} = require("playwright");
const {expect} = require("@playwright/test");
const Gold = require("./core.js");

const query = "😀基和基，遗漏畒勾，还有畒勾。";
const items = [
  {item_id: "one", source_item_id: "blind-one", cohort: "repair_target", query_content: query,
    prior_audit: {disposition: "reject", notes: "边界与词义测试"}, candidates: [
      {candidate_id: "first", surface: "基", span: [1, 2], source_types: ["legacy_hit"], source_hits: []},
      {candidate_id: "second", surface: "基", span: [3, 4], source_types: ["legacy_hit"], source_hits: []},
    ]},
  {item_id: "empty", source_item_id: "blind-empty", cohort: "repair_target", query_content: "😀空白候选",
    prior_audit: {disposition: "accept", notes: "无候选测试"}, candidates: []},
];

async function fixture(browser, viewport = {width: 1440, height: 980}) {
  const context = await browser.newContext({viewport});
  const page = await context.newPage();
  const decisions = Object.fromEntries(items.map(item => [item.item_id, {...Gold.defaultDecision(item), status: "draft"}]));
  const state = {saves: [], errors: [], decisions, revision: 1};
  page.on("pageerror", error => state.errors.push(String(error)));
  const summary = item => ({item_id: item.item_id, source_item_id: item.source_item_id, cohort: item.cohort,
    query_preview: item.query_content, candidate_count: item.candidates.length,
    surfaces: item.candidates.map(row => row.surface), status: decisions[item.item_id].status});
  const status = () => ({item_count: 2, confirmed_count: Object.values(decisions).filter(row => row.status === "confirmed").length});
  await page.route("**/*", async route => {
    const url = new URL(route.request().url());
    assert.equal(url.origin, "http://span-review.test");
    const json = data => route.fulfill({json: data});
    if (url.pathname === "/api/bootstrap") return json({frame_id: "test-only", session_token: "mock", revision: String(state.revision), status: status(), items: items.map(summary)});
    if (url.pathname.startsWith("/api/items/")) {
      const item = items.find(row => row.item_id === url.pathname.split("/").pop());
      return json({item, revision: String(state.revision), decision: decisions[item.item_id]});
    }
    if (url.pathname === "/api/save") {
      const payload = route.request().postDataJSON();
      const item = items.find(row => row.item_id === payload.item_id);
      assert.equal(payload.expected_revision, String(state.revision));
      assert.deepEqual(Gold.validateDecision(item, payload.decision, {confirm: payload.confirm}), {});
      state.saves.push(payload);
      state.revision += 1;
      decisions[item.item_id] = {...payload.decision, status: payload.confirm ? "confirmed" : "draft"};
      return json({revision: String(state.revision), status: status(), item_summary: summary(item), decision: decisions[item.item_id]});
    }
    const assets = {"/": "index.html", "/index.html": "index.html", "/app.js": "app.js", "/core.js": "core.js", "/styles.css": "styles.css",
      "/review-core.js": "../wp3_candidate_review_ui/core.js", "/review-base.css": "../wp3_candidate_review_ui/styles.css"};
    if (assets[url.pathname]) return route.fulfill({path: path.resolve(__dirname, assets[url.pathname])});
    return route.fulfill({status: 404});
  });
  await page.goto("http://span-review.test/");
  await expect(page.locator("#item-id")).toHaveText("blind-one");
  await expect(page.locator("#keep-all")).toBeEnabled();
  return {page, context, state};
}

// Select by codepoint positions using DOM text nodes, including across <mark>.
async function select(page, start, end, backward = false) {
  await page.locator("#query-content").scrollIntoViewIfNeeded();
  await page.evaluate(({start, end, backward}) => {
    const root = document.querySelector("#query-content");
    const characters = Array.from(root.textContent);
    const boundary = position => {
      let remaining = characters.slice(0, position).join("").length;
      const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
      let node;
      while ((node = walker.nextNode())) {
        if (remaining <= node.length) return {node, offset: remaining};
        remaining -= node.length;
      }
      throw new Error("invalid selection fixture");
    };
    const a = boundary(start), b = boundary(end);
    const selection = window.getSelection();
    selection.removeAllRanges();
    selection.setBaseAndExtent(backward ? b.node : a.node, backward ? b.offset : a.offset,
      backward ? a.node : b.node, backward ? a.offset : b.offset);
  }, {start, end, backward});
  await expect(page.locator("#selection-toolbar")).toBeVisible();
}

async function main() {
  const browser = await chromium.launch();
  async function run(name, fn, viewport) {
    const f = await fixture(browser, viewport);
    try { await fn(f); assert.deepEqual(f.state.errors, []); console.log(`PASS ${name}`); }
    finally { await f.context.close(); }
  }
  try {
    await run("bulk + occurrence shortcuts and guardrails", async ({page, state}) => {
      const cards = page.locator(".candidate-card");
      await page.keyboard.press("1");
      await expect(cards.nth(0)).toHaveAttribute("data-action", "keep");
      await expect(cards.nth(1)).toHaveAttribute("data-action", "keep");
      await page.keyboard.press("2");
      await expect(cards.nth(1)).toHaveAttribute("data-action", "drop");
      await page.keyboard.press("3");
      await expect(cards.nth(0)).toHaveAttribute("data-action", "");
      await page.keyboard.press("q");
      await page.keyboard.press("ArrowDown");
      await page.keyboard.press("a");
      await expect(cards.nth(0)).toHaveAttribute("data-action", "keep");
      await expect(cards.nth(1)).toHaveAttribute("data-action", "drop");
      await expect(cards.nth(1)).toHaveAttribute("aria-current", "true");
      await page.keyboard.press("ArrowDown");
      await expect(cards.nth(1)).toHaveAttribute("aria-current", "true");
      await expect(page.locator("#query-content mark")).toHaveText("基");
      await page.locator("#notes").fill("测试");
      await page.keyboard.type("123qa[]");
      await page.keyboard.press("ArrowUp");
      await expect(cards.nth(0)).toHaveAttribute("data-action", "keep");
      await expect(cards.nth(1)).toHaveAttribute("aria-current", "true");
      await page.keyboard.press("Control+s");
      await expect(page.locator("#save-state")).toHaveAttribute("data-state", "saved");
      assert.ok(state.saves.length);
      await page.locator("#guidelines").click();
      const count = state.saves.length;
      await page.keyboard.press("2");
      await page.keyboard.press("Control+Enter");
      await expect(page.locator("#guideline-dialog")).toBeVisible();
      assert.equal(state.saves.length, count);
      await page.keyboard.press("Escape");
      await expect(cards.nth(0)).toHaveAttribute("data-action", "keep");
      await page.keyboard.press("Control+Enter");
      await expect(page.locator("#item-id")).toHaveText("blind-empty");
      await page.keyboard.press("[");
      await expect(page.locator("#item-id")).toHaveText("blind-one");
      await expect(page.locator("#keep-all")).toBeDisabled();
      await page.keyboard.press("2");
      await page.keyboard.press("a");
      await expect(cards.nth(0)).toHaveAttribute("data-action", "keep");
    });

    await run("selection, repeated occurrences, reason editing and roundtrip", async ({page, state}) => {
      await select(page, 12, 14, true);
      await expect(page.locator("#selection-surface")).toHaveText("“畒勾”");
      await page.locator("#confirm-selection").click();
      await expect(page.locator("#additional-dialog")).toBeVisible();
      await expect(page.locator("#additional-position")).toContainText("[12, 14)");
      if (process.env.SPAN_GOLD_SCREENSHOT_DIR) await page.screenshot({path: path.join(process.env.SPAN_GOLD_SCREENSHOT_DIR, "span-selection.png")});
      await page.locator("#additional-reason").fill("   ");
      await page.locator("#submit-additional").click();
      await expect(page.locator("#additional-dialog-error")).toContainText("请填写");
      await page.locator("#additional-reason").fill("需要完整解释 | 不是第一个 occurrence");
      await page.keyboard.press("Control+s");
      assert.equal(state.saves.length, 0);
      await page.locator("#submit-additional").click();
      await expect(page.locator("#additional-dialog")).not.toBeVisible();
      await expect(page.locator("#additional-count")).toHaveText("1");
      await page.locator("#additional-list").getByRole("button", {name: "修改理由"}).click();
      await page.locator("#additional-reason").fill("修改后理由");
      await page.locator("#submit-additional").click();
      await page.locator("#clear-all").click();
      await expect(page.locator("#additional-count")).toHaveText("1");
      await page.locator("#save-draft").click();
      await expect(page.locator("#save-state")).toHaveAttribute("data-state", "saved");
      assert.deepEqual(state.saves.at(-1).decision.additional_spans, [{start: 12, end: 14, surface: "畒勾", reason: "修改后理由"}]);
      await page.reload();
      await expect(page.locator("#additional-list")).toContainText("修改后理由");
      await select(page, 12, 14);
      await expect(page.locator("#confirm-selection")).toBeDisabled();
      await expect(page.locator("#selection-error")).toContainText("已经补充");
      await select(page, 7, 9);
      await expect(page.locator("#confirm-selection")).toBeEnabled();
      await page.locator("#confirm-selection").click();
      await page.keyboard.press("Escape");
      await expect(page.locator("#additional-count")).toHaveText("1");
      await page.locator("#additional-list").getByRole("button", {name: "移除", exact: true}).click();
      await expect(page.locator("#additional-count")).toHaveText("0");
    });

    await run("cross-node Unicode selection, duplicate/overlap and outside selection", async ({page}) => {
      await select(page, 0, 3);
      await expect(page.locator("#selection-surface")).toHaveText("“😀基和”");
      await page.locator("#confirm-selection").click();
      await expect(page.locator("#additional-position")).toContainText("[0, 3)");
      await page.locator("#additional-reason").fill("跨节点选区测试");
      await page.locator("#submit-additional").click();
      await page.locator("#keep-all").click();
      await page.locator("#confirm-next").click();
      await expect(page.locator("#candidate-error")).toContainText("不能重叠");
      await expect(page.locator("#item-id")).toHaveText("blind-one");
      await page.locator("#additional-list").getByRole("button", {name: "移除", exact: true}).click();
      await select(page, 0, 3);
      await expect(page.locator("#selection-error")).toContainText("已保留候选重叠");
      await expect(page.locator("#confirm-selection")).toBeDisabled();
      await select(page, 1, 2);
      await expect(page.locator("#selection-error")).toContainText("已列为候选");
      await page.evaluate(() => {
        const range = document.createRange();
        range.selectNodeContents(document.querySelector("#prior-audit"));
        const selection = window.getSelection(); selection.removeAllRanges(); selection.addRange(range);
      });
      await expect(page.locator("#selection-toolbar")).not.toBeVisible();
      await page.locator("#drop-all").click();
      await page.locator("#save-draft").click();
      await expect(page.locator("#save-state")).toHaveAttribute("data-state", "saved");
      await page.keyboard.press("]");
      await expect(page.locator("#zero-candidate")).toBeVisible();
      await page.keyboard.press("ArrowDown");
      await page.keyboard.press("q");
      await select(page, 1, 3);
      await expect(page.locator("#selection-surface")).toHaveText("“空白”");
    });

    for (const viewport of [{width: 1440, height: 980}, {width: 1024, height: 768}, {width: 390, height: 844}]) {
      await run(`guidelines and layout ${viewport.width}px`, async ({page}) => {
        assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth));
        if (viewport.width >= 768) {
          const source = await page.locator(".source-pane").boundingBox();
          const decision = await page.locator(".decision-pane").boundingBox();
          assert.ok(source.width >= 360, "query pane must remain readable");
          assert.ok(source.x + source.width <= decision.x + 1, "workbench panes must not overlap");
          assert.ok(Math.abs(decision.x + decision.width - viewport.width) <= 1, "decision pane must align with viewport edge");
        }
        if (process.env.SPAN_GOLD_SCREENSHOT_DIR) await page.screenshot({path: path.join(process.env.SPAN_GOLD_SCREENSHOT_DIR, `span-workbench-${viewport.width}.png`)});
        await page.locator("#guidelines").click();
        await expect(page.locator("#guideline-title")).toContainText("先判断是否需要解释");
        await expect(page.locator(".rule-card")).toHaveCount(3);
        await expect(page.locator(".guide-shortcuts")).toContainText("Ctrl/⌘ Enter");
        const bounds = await page.locator("#guideline-dialog").boundingBox();
        assert.ok(bounds.x >= 0 && bounds.y >= 0 && bounds.x + bounds.width <= viewport.width + 1 && bounds.y + bounds.height <= viewport.height + 1);
        assert.ok(await page.locator(".guide-body").evaluate(el => el.scrollWidth <= el.clientWidth + 1));
        await expect(page.getByRole("button", {name: "知道了，开始审核"})).toBeInViewport();
        if (process.env.SPAN_GOLD_SCREENSHOT_DIR) await page.screenshot({path: path.join(process.env.SPAN_GOLD_SCREENSHOT_DIR, `span-guide-${viewport.width}.png`)});
        await page.getByRole("button", {name: "知道了，开始审核"}).click();
      }, viewport);
    }
  } finally { await browser.close(); }
}

main().catch(error => { console.error(error); process.exitCode = 1; });
