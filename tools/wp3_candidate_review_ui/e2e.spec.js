"use strict";

const {test, expect} = require("@playwright/test");

const REASONS = [
  "stable_core_candidate", "context_required", "evidence_required", "transparent",
  "ordinary_identity_or_name", "generic_insult", "fragment", "function_word_attached",
  "sentence_level", "wrong_boundary", "substring_projection", "unsupported_sense",
  "one_off_creation", "ambiguous_surface", "context_polysemy", "quoted_or_reclaimed",
  "evidence_conflict", "variant_unresolved", "non_contiguous_unresolved", "label_derived",
  "not_fit_attested", "duplicate", "no_neutral_gloss", "other",
];

const CASE_IDS = ["S21-001", "S21-002", "S21-003"];
const ACTION_NAMES = {
  accept: "接受",
  trim: "缩短边界",
  expand: "扩大边界",
  split: "拆分",
  reject: "驳回",
  defer: "暂缓",
};

function fixture(phase = "raw", delayMs = 0, options = {}) {
  let currentPhase = phase;
  let revision = "revision-1";
  let revisionNumber = 1;
  let conflictRemaining = options.conflictOnce ? 1 : 0;
  const writes = [];
  const searchRequests = [];
  const caseRequests = [];
  const raw = Object.fromEntries(CASE_IDS.map(caseId => [
    caseId,
    {status: "draft", needs_explanation: false, mentions: [], notes: ""},
  ]));
  const decisions = {};
  const proposals = {
    "S21-001": [
      {proposal_id: "p1", surface: "女拳不", occurrence_ordinal: 1, start: 0, end: 3},
      {proposal_id: "p2", surface: "女拳", occurrence_ordinal: 2, start: 6, end: 8},
    ],
    "S21-002": [
      {proposal_id: "p3", surface: "普通", occurrence_ordinal: 1, start: 0, end: 2},
    ],
    "S21-003": [
      {proposal_id: "p4", surface: "女拳表达", occurrence_ordinal: 1, start: 6, end: 10},
    ],
  };
  if (currentPhase === "diagnostic" || options.rawConfirmed) {
    for (const caseId of CASE_IDS) raw[caseId].status = "confirmed";
  }
  const contents = {
    "S21-001": "女拳不行，女拳也不行，这也是女拳表述",
    "S21-002": "普通文本",
    "S21-003": "冻结原文包含女拳表达，方便同类检索",
  };
  const aliases = {
    "S21-001": "术语开发-001",
    "S21-002": "术语开发-002",
    "S21-003": "术语开发-003",
  };

  function status() {
    const rawConfirmed = Object.values(raw).filter(row => row.status === "confirmed").length;
    const allProposals = Object.values(proposals).flat();
    const diagnosticConfirmed = Object.values(decisions).filter(row => row.status === "confirmed").length;
    return {
      frame_id: "frame-test",
      reviewer_id: "reviewer-test",
      phase: currentPhase,
      revision,
      raw: {confirmed: rawConfirmed, total: CASE_IDS.length},
      diagnostic: {
        confirmed: diagnosticConfirmed,
        total: currentPhase === "diagnostic" ? allProposals.length : 0,
      },
      amendment_count: 0,
      finalized_gold_id: null,
    };
  }

  function summary(caseId) {
    const rows = proposals[caseId];
    const confirmed = rows.filter(row => decisions[row.proposal_id] && decisions[row.proposal_id].status === "confirmed").length;
    const draft = rows.filter(row => decisions[row.proposal_id] && decisions[row.proposal_id].status === "draft").length;
    return {
      case_id: caseId,
      blind_alias: aliases[caseId],
      raw_status: raw[caseId].status,
      diagnostic: {confirmed, draft, total: currentPhase === "diagnostic" ? rows.length : 0},
      complete: currentPhase === "raw" ? raw[caseId].status === "confirmed" : confirmed === rows.length,
    };
  }

  function bootstrap() {
    return {
      schema_version: "wp3-s21-review-bootstrap/v1",
      session_token: "test-token",
      frame_id: "frame-test",
      reviewer_id: "reviewer-test",
      phase: currentPhase,
      revision,
      status: status(),
      case_summaries: CASE_IDS.map(summary),
      reason_codes: REASONS,
      provisional_routes: ["A_candidate", "B_candidate", "C_candidate"],
      proposal_actions: ["accept", "trim", "expand", "split", "reject", "defer"],
      warnings: ["DEVELOPMENT ONLY / NON-SEALED / NON-SCIENTIFIC"],
    };
  }

  function casePayload(caseId) {
    const row = {
      case_id: caseId,
      blind_alias: summary(caseId).blind_alias,
      content: contents[caseId],
      raw_annotation: raw[caseId],
    };
    if (currentPhase === "diagnostic") {
      row.proposals = proposals[caseId];
      row.diagnostic_decisions = Object.fromEntries(
        proposals[caseId].map(proposal => [proposal.proposal_id, decisions[proposal.proposal_id] || null]),
      );
    }
    return {
      schema_version: "wp3-s21-review-case/v1",
      revision,
      phase: currentPhase,
      case: row,
      case_summary: summary(caseId),
    };
  }

  function editDistance(left, right) {
    const a = Array.from(left);
    const b = Array.from(right);
    let previous = Array.from({length: b.length + 1}, (_, index) => index);
    for (let row = 1; row <= a.length; row += 1) {
      const current = [row];
      for (let column = 1; column <= b.length; column += 1) {
        current[column] = Math.min(
          current[column - 1] + 1,
          previous[column] + 1,
          previous[column - 1] + (a[row - 1] === b[column - 1] ? 0 : 1),
        );
      }
      previous = current;
    }
    return previous[b.length];
  }

  function fuzzyWindow(content, query) {
    const source = Array.from(content.toLowerCase());
    const needle = Array.from(query.toLowerCase());
    if (needle.length < 3) {
      const start = content.toLowerCase().indexOf(query.toLowerCase());
      return start < 0 ? null : {start, end: start + needle.length, distance: 0};
    }
    const limit = Math.min(4, Math.max(1, Math.floor(needle.length * 0.2)));
    let best = null;
    const minimumLength = Math.max(1, needle.length - limit);
    const maximumLength = needle.length + limit;
    for (let start = 0; start < source.length; start += 1) {
      for (let length = minimumLength; length <= maximumLength && start + length <= source.length; length += 1) {
        const distance = editDistance(source.slice(start, start + length).join(""), needle.join(""));
        if (distance > limit) continue;
        if (!best || distance < best.distance || (distance === best.distance && length < best.end - best.start)) {
          best = {start, end: start + length, distance};
        }
      }
    }
    return best;
  }

  function searchPayload(query, mode) {
    const normalized = query.toLowerCase();
    const terms = [...new Set(normalized.split(/\s+/u).filter(Boolean))];
    const matches = [];
    for (const caseId of CASE_IDS) {
      const fields = [
        ["case_id", caseId],
        ["alias", aliases[caseId]],
        ["content", contents[caseId]],
      ];
      if (mode === "all_terms") {
        if (!terms.every(term => fields.some(([, value]) => value.toLowerCase().includes(term)))) continue;
        const highlighted = fields.find(([, value]) => value.toLowerCase().includes(terms[0])) || fields[2];
        const start = highlighted[1].toLowerCase().indexOf(terms[0]);
        matches.push({
          case_id: caseId,
          matched_field: highlighted[0],
          snippet: highlighted[1],
          match_start: start,
          match_end: start < 0 ? -1 : start + terms[0].length,
          distance: 0,
        });
        continue;
      }
      if (mode === "fuzzy") {
        const exact = fields.slice(0, 2).find(([, value]) => value.toLowerCase().includes(normalized));
        if (exact) {
          const start = exact[1].toLowerCase().indexOf(normalized);
          matches.push({case_id: caseId, matched_field: exact[0], snippet: exact[1], match_start: start, match_end: start + query.length, distance: 0});
          continue;
        }
        const window = fuzzyWindow(contents[caseId], query);
        if (!window) continue;
        matches.push({case_id: caseId, matched_field: "content", snippet: contents[caseId], match_start: window.start, match_end: window.end, distance: window.distance});
        continue;
      }
      const exact = fields.find(([, value]) => value.toLowerCase().includes(normalized));
      if (!exact) continue;
      const start = exact[1].toLowerCase().indexOf(normalized);
      matches.push({case_id: caseId, matched_field: exact[0], snippet: exact[1], match_start: start, match_end: start + query.length, distance: 0});
    }
    if (mode === "fuzzy") {
      matches.sort((left, right) => left.distance - right.distance || CASE_IDS.indexOf(left.case_id) - CASE_IDS.indexOf(right.case_id));
    }
    return {
      schema_version: "wp3-s21-review-search/v1",
      frame_id: "frame-test",
      query,
      mode,
      matches,
    };
  }

  async function install(page) {
    await page.route("**/api/bootstrap", route => route.fulfill({json: bootstrap()}));
    await page.route("**/api/cases/search?*", route => {
      const url = new URL(route.request().url());
      const query = url.searchParams.get("q") || "";
      const mode = url.searchParams.get("mode") || "literal";
      searchRequests.push({query, mode});
      const searchDelay = (options.searchDelays && options.searchDelays[query]) || 0;
      return new Promise(resolve => setTimeout(resolve, searchDelay))
        .then(() => route.fulfill({json: searchPayload(query, mode)}))
        .catch(() => undefined);
    });
    await page.route(/\/api\/cases\/S21-\d{3}$/, async route => {
      const caseId = route.request().url().match(/(S21-\d{3})$/)[1];
      caseRequests.push(caseId);
      const caseDelay = (options.caseDelays && options.caseDelays[caseId]) || 0;
      if (caseDelay) await new Promise(resolve => setTimeout(resolve, caseDelay));
      return route.fulfill({json: casePayload(caseId)});
    });
    await page.route("**/api/raw", async route => {
      const body = route.request().postDataJSON();
      if (conflictRemaining > 0) {
        conflictRemaining -= 1;
        await route.fulfill({
          status: 409,
          contentType: "application/json",
          body: JSON.stringify({error: "review session changed concurrently"}),
        });
        return;
      }
      writes.push(body);
      if (delayMs) await new Promise(resolve => setTimeout(resolve, delayMs));
      revisionNumber += 1;
      revision = `revision-${revisionNumber}`;
      raw[body.case_id] = {
        status: body.confirm ? "confirmed" : "draft",
        needs_explanation: body.annotation.needs_explanation,
        mentions: body.annotation.mentions,
        notes: body.annotation.notes,
      };
      await route.fulfill({
        json: {
          schema_version: "wp3-s21-review-mutation/v1",
          revision,
          phase: currentPhase,
          status: status(),
          case_id: body.case_id,
          case_summary: summary(body.case_id),
          raw_annotation: raw[body.case_id],
        },
      });
    });
    await page.route("**/api/diagnostic", async route => {
      const body = route.request().postDataJSON();
      writes.push(body);
      revisionNumber += 1;
      revision = `revision-${revisionNumber}`;
      decisions[body.proposal_id] = {
        case_id: body.case_id,
        proposal_id: body.proposal_id,
        status: body.confirm ? "confirmed" : "draft",
        ...body.decision,
      };
      await route.fulfill({
        json: {
          schema_version: "wp3-s21-review-mutation/v1",
          revision,
          phase: currentPhase,
          status: status(),
          case_id: body.case_id,
          case_summary: summary(body.case_id),
          proposal_id: body.proposal_id,
          diagnostic_decision: decisions[body.proposal_id],
        },
      });
    });
    await page.route("**/api/reopen", async route => {
      const body = route.request().postDataJSON();
      revisionNumber += 1;
      revision = `revision-${revisionNumber}`;
      if (body.scope === "raw") {
        raw[body.case_id].status = "draft";
        for (const proposal of proposals[body.case_id]) delete decisions[proposal.proposal_id];
      } else if (decisions[body.proposal_id]) {
        decisions[body.proposal_id].status = "draft";
      }
      await route.fulfill({
        json: {
          schema_version: "wp3-s21-review-mutation/v1",
          revision,
          phase: currentPhase,
          status: status(),
          case_id: body.case_id,
          case_summary: summary(body.case_id),
          case: casePayload(body.case_id).case,
        },
      });
    });
    await page.route("**/api/lock-raw", async route => {
      currentPhase = "diagnostic";
      revisionNumber += 1;
      revision = `revision-${revisionNumber}`;
      await route.fulfill({json: bootstrap()});
    });
    await page.route("**/api/export", route => route.fulfill({
      status: 200,
      contentType: "application/zip",
      body: "PK-test",
    }));
  }

  return {install, caseRequests, searchRequests, writes};
}

async function selectText(page, start, end) {
  await page.locator("#content").evaluate((root, offsets) => {
    const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
    const nodes = [];
    let total = 0;
    while (walker.nextNode()) {
      const node = walker.currentNode;
      nodes.push({node, start: total, end: total + node.data.length});
      total += node.data.length;
    }
    const startRow = nodes.find(row => offsets.start >= row.start && offsets.start <= row.end);
    const endRow = nodes.find(row => offsets.end >= row.start && offsets.end <= row.end);
    const range = document.createRange();
    range.setStart(startRow.node, offsets.start - startRow.start);
    range.setEnd(endRow.node, offsets.end - endRow.start);
    const selection = window.getSelection();
    selection.removeAllRanges();
    selection.addRange(range);
    root.dispatchEvent(new MouseEvent("mouseup", {bubbles: true}));
  }, {start, end});
}

async function openCaseSidebarIfCollapsed(page) {
  const collapsed = await page.evaluate(() => matchMedia("(max-width: 1179px)").matches);
  if (!collapsed) return;
  const sidebar = page.locator("#case-sidebar");
  if (!(await sidebar.evaluate(element => element.classList.contains("open")))) {
    await page.locator("#sidebar-toggle").click();
    await expect(sidebar).toHaveClass(/open/);
  }
}

async function closeCaseSidebarIfCollapsed(page) {
  const collapsed = await page.evaluate(() => matchMedia("(max-width: 1179px)").matches);
  if (!collapsed) return;
  const sidebar = page.locator("#case-sidebar");
  if (await sidebar.evaluate(element => element.classList.contains("open"))) {
    await page.locator("#sidebar-close").click();
    await expect(sidebar).not.toHaveClass(/open/);
  }
}

test("phase A selection creates a mention and navigation flushes the original case", async ({page}) => {
  const data = fixture("raw", 120);
  await data.install(page);
  await page.goto("/index.html");
  await expect(page.locator("#case-position")).toContainText("S21-001");
  await selectText(page, 0, 2);
  await expect(page.locator("#selection-bar")).toBeVisible();
  await expect(page.locator("#quick-route-actions .quick-route-button")).toHaveCount(3);
  await page.locator('[data-route="A_candidate"]').click();
  await expect(page.locator("#decision-title")).toHaveText("Mention 1");
  await expect(page.locator(".decision-editor input").first()).toHaveValue("女拳");
  await page.getByLabel("Case 备注").fill("导航前必须保存");
  await page.locator("#next-case").click();
  await expect(page.locator("#case-position")).toContainText("S21-002");
  expect(data.writes[0].case_id).toBe("S21-001");
  expect(data.writes[0].annotation.notes).toBe("导航前必须保存");
});

test("phase A click and 1-3 keys create A/B/C mentions with route defaults", async ({page}) => {
  const data = fixture("raw");
  await data.install(page);
  await page.goto("/index.html");

  await selectText(page, 0, 2);
  await expect(page.locator("#quick-route-actions")).toBeVisible();
  await expect(page.locator('[data-route="A_candidate"]')).toContainText("稳定核心");
  await expect(page.locator('[data-route="B_candidate"]')).toContainText("需上下文");
  await expect(page.locator('[data-route="C_candidate"]')).toContainText("待核实");
  await page.locator('[data-route="A_candidate"]').click();
  await expect(page.locator("#decision-title")).toHaveText("Mention 1");

  await selectText(page, 5, 7);
  await page.keyboard.press("2");
  await expect(page.locator("#decision-title")).toHaveText("Mention 2");

  await selectText(page, 14, 18);
  await page.keyboard.press("3");
  await expect(page.locator("#decision-title")).toHaveText("Mention 3");

  await page.locator("#save-draft").click();
  await expect.poll(() => data.writes.length).toBeGreaterThan(0);
  const mentions = data.writes.at(-1).annotation.mentions;
  expect(mentions.map(row => row.provisional_route)).toEqual([
    "A_candidate",
    "B_candidate",
    "C_candidate",
  ]);
  expect(mentions.map(row => row.reason_codes)).toEqual([
    ["stable_core_candidate"],
    ["context_required"],
    ["evidence_required"],
  ]);
});

test("ABC collection criteria are available from the workbench", async ({page}) => {
  const data = fixture("raw");
  await data.install(page);
  await page.goto("/index.html");
  await page.getByRole("button", {name: "ABC 口径"}).click();
  await expect(page.locator("#guideline-dialog")).toBeVisible();
  await expect(page.locator("#guideline-dialog .guide-tier")).toHaveCount(3);
  await expect(page.locator("#guideline-dialog")).toContainText("A 可以稳定讲清；B 必须结合上下文；C 目前还不能可靠讲清");
  await expect(page.locator("#guideline-dialog")).toContainText("不要把明确噪声塞进 C");
  await expect(page.locator("#guideline-dialog")).toContainText("不是资源对象 → R；有价值但证据未决 → C；含义成立但需上下文 → B；全部稳定门槛通过 → A");
  const dimensions = await page.evaluate(() => ({
    scrollWidth: document.documentElement.scrollWidth,
    clientWidth: document.documentElement.clientWidth,
  }));
  expect(dimensions.scrollWidth).toBeLessThanOrEqual(dimensions.clientWidth);
});

test("case search mode cycles and frozen content results include safe highlights", async ({page}) => {
  const data = fixture("raw");
  await data.install(page);
  await page.goto("/index.html");
  await openCaseSidebarIfCollapsed(page);

  const search = page.locator("#case-search");
  const mode = page.locator("#search-mode");
  await expect(mode).toHaveText("连续 ↻");
  await expect(mode).toHaveAttribute("title", "点击切换为多词模式");
  await search.fill("女拳");
  await expect(page.locator("#search-status")).toHaveText("2 条搜索结果");
  await expect(page.locator("#case-list .case-item")).toHaveCount(2);
  await expect(page.locator("#case-list .case-match-snippet mark")).toHaveCount(2);
  await expect.poll(() => data.searchRequests.at(-1)).toEqual({query: "女拳", mode: "literal"});

  await mode.click();
  await expect(mode).toHaveText("多词 ↻");
  await expect.poll(() => data.searchRequests.at(-1)).toEqual({query: "女拳", mode: "all_terms"});
  await search.fill("冻结 同类");
  await expect(page.locator("#search-status")).toHaveText("1 条搜索结果");
  await expect(page.locator("#case-list .case-match-snippet mark")).toHaveText("冻结");
  await expect.poll(() => data.searchRequests.at(-1)).toEqual({query: "冻结 同类", mode: "all_terms"});

  await openCaseSidebarIfCollapsed(page);
  await mode.click();
  await expect(mode).toHaveText("模糊 ↻");
  await search.fill("女拳表达方");
  await expect(page.locator("#search-status")).toHaveText("1 条搜索结果");
  await expect(page.locator("#case-list .case-item strong")).toHaveText("S21-003");
  await expect(page.locator("#case-list .case-match-snippet mark")).toContainText("女拳表达");
  await expect.poll(() => data.searchRequests.at(-1)).toEqual({query: "女拳表达方", mode: "fuzzy"});

  await mode.click();
  await expect(mode).toHaveText("连续 ↻");
  await expect.poll(() => data.searchRequests.at(-1)).toEqual({query: "女拳表达方", mode: "literal"});

  await search.fill("冻结原文");
  await expect(page.locator("#search-status")).toHaveText("1 条搜索结果");
  await expect(page.locator("#case-position")).toContainText("S21-003");
  await expect(page.locator("#case-list .case-match-snippet mark")).toHaveText("冻结原文");
});

test("case navigation follows visible search order and reveals the active sidebar row", async ({page}, testInfo) => {
  test.skip(testInfo.project.name.includes("mobile"), "desktop/tablet sidebar positioning only");
  const data = fixture("raw");
  await data.install(page);
  await page.goto("/index.html");
  await openCaseSidebarIfCollapsed(page);

  await page.locator("#case-search").fill("女拳");
  await expect(page.locator("#search-status")).toHaveText("2 条搜索结果");
  await expect(page.locator("#case-position")).toContainText("S21-001");
  await closeCaseSidebarIfCollapsed(page);
  await page.locator("#next-case").click();
  await expect(page.locator("#case-position")).toContainText("结果 2 / 2");
  await expect(page.locator("#case-position")).toContainText("S21-003");
  await page.keyboard.press("[");
  await expect(page.locator("#case-position")).toContainText("结果 1 / 2");
  await expect(page.locator("#case-position")).toContainText("S21-001");
  await page.keyboard.press("]");
  await expect(page.locator("#case-position")).toContainText("S21-003");
  await openCaseSidebarIfCollapsed(page);
  await expect(page.locator('#case-list .case-item.active[data-case-id="S21-003"]')).toBeInViewport();
});

test("stale search responses cannot replace a newer frozen-content result", async ({page}) => {
  const data = fixture("raw", 0, {caseDelays: {"S21-003": 650}});
  await data.install(page);
  await page.goto("/index.html");

  const search = page.locator("#case-search");
  await search.fill("冻结原文");
  await expect.poll(() => data.caseRequests.includes("S21-003")).toBe(true);
  await search.fill("不存在的冻结表达");
  await expect(page.locator("#search-status")).toHaveText("无匹配结果");
  await expect(page.locator("#case-list .case-item")).toHaveCount(0);
  await page.waitForTimeout(700);
  await expect(page.locator("#case-position")).toContainText("S21-001");
  await expect(page.locator("#search-status")).toHaveText("无匹配结果");
});

test("open filter confirmation follows the saved queue and keeps a completed search", async ({page}) => {
  const data = fixture("raw");
  await data.install(page);
  await page.goto("/index.html");

  await openCaseSidebarIfCollapsed(page);
  await page.locator('#case-filters [data-filter="open"]').click();
  await closeCaseSidebarIfCollapsed(page);
  await page.getByRole("button", {name: "本条无可标注表达"}).click();
  await page.locator("#confirm-next").click();
  await expect(page.locator("#case-position")).toContainText("S21-002");
  await openCaseSidebarIfCollapsed(page);
  await expect(page.locator('#case-list [data-case-id="S21-001"]')).toHaveCount(0);

  await page.locator("#case-search").fill("普通");
  await expect(page.locator("#search-status")).toHaveText("显示 1 / 命中 1 条");
  await closeCaseSidebarIfCollapsed(page);
  await page.getByRole("button", {name: "本条无可标注表达"}).click();
  await page.locator("#confirm-next").click();
  await expect(page.locator("#case-position")).toContainText("S21-002");
  await openCaseSidebarIfCollapsed(page);
  await expect(page.locator("#case-search")).toHaveValue("普通");
  await expect(page.locator("#search-status")).toHaveText("当前结果已全部确认");
  await expect(page.locator("#case-list .case-item")).toHaveCount(0);
  await expect(page.locator("#previous-case")).toBeDisabled();
  await expect(page.locator("#next-case")).toBeDisabled();
  expect(data.writes.filter(row => row.confirm).map(row => row.case_id)).toEqual([
    "S21-001",
    "S21-002",
  ]);
});

test("confirmation locks editors and cannot enqueue a post-confirm draft", async ({page}) => {
  const data = fixture("raw", 500);
  await data.install(page);
  await page.goto("/index.html");

  await page.getByLabel("Case 备注").fill("确认前内容");
  await page.getByRole("button", {name: "本条无可标注表达"}).click();
  await page.locator("#confirm-next").click();
  await expect.poll(() => data.writes.length).toBe(1);
  await expect(page.locator("#case-search")).toBeDisabled();
  await expect.poll(() => page.locator("#decision-editor").evaluate(node => node.inert)).toBe(true);
  await expect(page.locator("#case-position")).toContainText("S21-002");
  await expect(page.locator("#case-search")).toBeEnabled();
  await expect.poll(() => page.locator("#decision-editor").evaluate(node => node.inert)).toBe(false);
  await page.waitForTimeout(800);
  expect(data.writes).toHaveLength(1);
  expect(data.writes[0].confirm).toBe(true);
  expect(data.writes[0].annotation.notes).toBe("确认前内容");
});

test("decision actions keep normal height and use the intended scroll layer", async ({page}, testInfo) => {
  const data = fixture("raw");
  await data.install(page);
  await page.goto("/index.html");
  const metrics = await page.evaluate(() => {
    const pane = document.querySelector(".decision-pane");
    const editor = document.querySelector("#decision-editor");
    const actions = document.querySelector(".decision-actions");
    const buttons = [...actions.querySelectorAll("button")];
    return {
      bodyOverflowY: getComputedStyle(document.body).overflowY,
      paneOverflowY: getComputedStyle(pane).overflowY,
      editorOverflowY: getComputedStyle(editor).overflowY,
      actionsPosition: getComputedStyle(actions).position,
      actionsHeight: actions.getBoundingClientRect().height,
      buttonHeights: buttons.map(button => button.getBoundingClientRect().height),
    };
  });
  expect(metrics.actionsHeight).toBeLessThan(80);
  expect(Math.max(...metrics.buttonHeights)).toBeLessThanOrEqual(48);
  if (testInfo.project.name.includes("mobile")) {
    expect(metrics.bodyOverflowY).toBe("auto");
    expect(metrics.actionsPosition).toBe("sticky");
  } else {
    expect(metrics.bodyOverflowY).toBe("hidden");
    expect(metrics.paneOverflowY).toBe("hidden");
    expect(metrics.editorOverflowY).toBe("auto");
  }
});

test("phase B renders one decision editor and action changes reset fields", async ({page}) => {
  const data = fixture("diagnostic");
  await data.install(page);
  await page.goto("/index.html");
  await expect(page.locator(".action-grid")).toHaveCount(1);
  await expect(page.locator(".action-choice")).toHaveCount(6);
  for (const button of await page.locator(".action-choice").all()) {
    await expect(button).not.toHaveText(/^\s*[1-6]\./);
  }
  await expect(page.locator(".result-row")).toHaveCount(1);
  await page.keyboard.press("k");
  await expect(page.locator("#proposal-position")).toHaveText("2 / 2");
  await page.keyboard.press("j");
  await expect(page.locator("#proposal-position")).toHaveText("1 / 2");
  const initiallyActive = await page.locator(".action-choice.active").textContent();
  await page.keyboard.press("6");
  await expect(page.locator(".action-choice.active")).toHaveText(initiallyActive);
  await page.locator(".action-choice").filter({hasText: "驳回"}).click();
  await expect(page.locator(".result-row")).toHaveCount(0);
  await expect(page.getByText("至少选择一个判定原因")).toBeVisible();
  await page.getByText("非自足片段", {exact: true}).click();
  await page.locator(".action-choice").filter({hasText: "拆分"}).click();
  await expect(page.locator(".result-row")).toHaveCount(2);
  await page.locator(".action-choice").filter({hasText: "暂缓"}).click();
  await expect(page.getByPlaceholder("请说明仍需解决的问题（必填）")).toBeVisible();
});

test("mention notes and destructive removal are separate visual blocks", async ({page}) => {
  const data = fixture("raw");
  await data.install(page);
  await page.goto("/index.html");
  await selectText(page, 0, 2);
  await page.locator('[data-route="A_candidate"]').click();

  const spacing = await page.evaluate(() => {
    const danger = document.querySelector(".mention-danger");
    const notes = document.querySelector('textarea[placeholder*="其他原因"]');
    const remove = danger && danger.querySelector("button");
    return {
      sibling: Boolean(danger && notes && danger.previousElementSibling && danger.previousElementSibling.contains(notes)),
      border: danger ? getComputedStyle(danger).borderTopWidth : "0px",
      gap: danger && notes && remove
        ? remove.getBoundingClientRect().top - notes.getBoundingClientRect().bottom
        : 0,
    };
  });
  expect(spacing.sibling).toBe(true);
  expect(spacing.border).not.toBe("0px");
  expect(spacing.gap).toBeGreaterThan(8);
});

for (const action of ["accept", "trim", "expand", "split", "reject", "defer"]) {
  test(`phase B ${action} path confirms one audited proposal`, async ({page}) => {
    const data = fixture("diagnostic");
    await data.install(page);
    await page.goto("/index.html");

    if (action !== "accept") {
      await page.locator(".action-choice").filter({hasText: ACTION_NAMES[action]}).click();
    }
    if (action === "trim") {
      await page.locator(".result-row input").nth(0).fill("女拳");
    } else if (action === "expand") {
      await page.locator(".result-row input").nth(0).fill("女拳不行");
    } else if (action === "split") {
      await page.locator(".result-row input").nth(0).fill("女");
      await page.locator(".result-row input").nth(2).fill("拳");
    } else if (action === "reject") {
      await page.getByText("非自足片段", {exact: true}).click();
    } else if (action === "defer") {
      await page.getByPlaceholder("请说明仍需解决的问题（必填）").fill("等待独立语料证据");
    }

    await page.locator("#confirm-next").click();
    await expect.poll(() => data.writes.filter(row => row.confirm).length).toBe(1);
    const confirmed = data.writes.find(row => row.confirm);
    expect(confirmed.case_id).toBe("S21-001");
    expect(confirmed.proposal_id).toBe("p1");
    expect(confirmed.decision.action).toBe(action);
  });
}

test("confirmed raw item can be reopened only with an amendment reason", async ({page}) => {
  const data = fixture("raw");
  await data.install(page);
  await page.goto("/index.html");
  await page.getByRole("button", {name: "本条无可标注表达"}).click();
  await page.locator("#confirm-next").click();
  await expect(page.locator("#case-position")).toContainText("S21-002");
  await page.locator("#previous-case").click();
  await expect(page.locator("#case-position")).toContainText("S21-001");
  await page.getByRole("button", {name: "重新打开原文标注"}).click();
  await expect(page.locator("#action-dialog")).toBeVisible();
  await page.getByRole("button", {name: "重新打开", exact: true}).click();
  await expect(page.locator("#dialog-input-error")).toBeVisible();
  await page.getByLabel("修改原因").fill("修正边界判断");
  await page.getByRole("button", {name: "重新打开", exact: true}).click();
  await expect(page.getByRole("button", {name: "本条无可标注表达"})).toBeVisible();
});

test("CAS conflict offers server version and local reapply choices", async ({page}) => {
  const data = fixture("raw", 0, {conflictOnce: true});
  await data.install(page);
  await page.goto("/index.html");
  await selectText(page, 0, 2);
  await page.locator('[data-route="A_candidate"]').click();
  await expect(page.locator("#conflict-dialog")).toBeVisible({timeout: 3000});
  await expect(page.locator("#conflict-details")).toContainText("changed concurrently");
  await page.getByRole("button", {name: "采用服务器版本"}).click();
  await expect(page.locator("#conflict-dialog")).toBeHidden();
  await expect(page.getByRole("button", {name: "本条无可标注表达"})).toBeVisible();
});

test("locking a complete raw phase returns the diagnostic bootstrap", async ({page}) => {
  const data = fixture("raw", 0, {rawConfirmed: true});
  await data.install(page);
  await page.goto("/index.html");
  await page.getByText("操作", {exact: true}).click();
  await page.getByRole("button", {name: "锁定阶段 A"}).click();
  await page.getByRole("button", {name: "锁定并揭示提案"}).click();
  await expect(page.locator("#phase-title")).toContainText("阶段 B");
  await expect(page.locator(".action-grid")).toHaveCount(1);
});

test("mobile layout exposes full-screen case drawer without horizontal overflow", async ({page}, testInfo) => {
  test.skip(!testInfo.project.name.includes("mobile"), "mobile project only");
  const data = fixture("raw");
  await data.install(page);
  await page.goto("/index.html");
  await page.locator("#sidebar-toggle").click();
  await expect(page.locator("#case-sidebar")).toHaveClass(/open/);
  const dimensions = await page.evaluate(() => ({
    scrollWidth: document.documentElement.scrollWidth,
    clientWidth: document.documentElement.clientWidth,
  }));
  expect(dimensions.scrollWidth).toBeLessThanOrEqual(dimensions.clientWidth);
  await expect(page.locator(".decision-actions")).toBeVisible();
});

test("mobile top action sheet keeps export available", async ({page}, testInfo) => {
  test.skip(!testInfo.project.name.includes("mobile"), "mobile project only");
  const data = fixture("raw");
  await data.install(page);
  await page.goto("/index.html");
  await page.getByText("操作", {exact: true}).click();
  const download = page.waitForEvent("download");
  await page.getByRole("button", {name: "导出审核包"}).click();
  expect((await download).suggestedFilename()).toBe("wp3-s21-development-review.zip");
});

test("mobile proposal sequence opens as a full-width bottom sheet", async ({page}, testInfo) => {
  test.skip(!testInfo.project.name.includes("mobile"), "mobile project only");
  const data = fixture("diagnostic");
  await data.install(page);
  await page.goto("/index.html");
  await page.getByRole("button", {name: "查看提案序列"}).click();
  await expect(page.locator("#item-strip-card")).toHaveClass(/sheet-open/);
  await expect(page.locator("#item-strip .item-chip")).toHaveCount(2);
  await page.locator("#item-strip .item-chip").nth(1).click();
  await expect(page.locator("#proposal-position")).toHaveText("2 / 2");
  await expect(page.locator("#item-strip-card")).not.toHaveClass(/sheet-open/);
});
