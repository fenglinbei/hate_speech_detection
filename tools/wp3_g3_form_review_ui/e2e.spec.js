"use strict";

const fs = require("fs");
const path = require("path");
const {test, expect} = require("@playwright/test");

const ROOT = __dirname;
const SHARED_ROOT = path.resolve(ROOT, "../wp3_candidate_review_ui");
const ITEM_IDS = ["g3form-one", "g3form-two", "g3form-three"];

function fixture(options = {}) {
  let revisionNumber = 1;
  let revision = `revision-${revisionNumber}`;
  let conflictRemaining = options.conflictOnce ? 1 : 0;
  const writes = [];
  const saveAttempts = [];
  const reopenWrites = [];
  const clientInstances = [];
  const items = {
    "g3form-one": {
      item_id: "g3form-one",
      surface: "永远滴神",
      canonical: "永远的神",
      proposed_family: "phonetic_variant",
      phonetic_scan_enabled: true,
      evidence_ids: ["ev-one"],
    },
    "g3form-two": {
      item_id: "g3form-two",
      surface: "神马",
      canonical: "什么",
      proposed_family: "phonetic_variant",
      phonetic_scan_enabled: false,
      evidence_ids: ["ev-two"],
    },
    "g3form-three": {
      item_id: "g3form-three",
      surface: "河蟹",
      canonical: "和谐",
      proposed_family: "orthographic_variant",
      phonetic_scan_enabled: false,
      evidence_ids: ["ev-three"],
    },
  };
  const evidence = {
    "g3form-one": [{
      evidence_id: "ev-one",
      source_id: "source-one",
      publisher: "公开材料甲",
      source_role: "primary",
      acquisition_mode: "manual",
      component_id: "component-one",
      relation_contract: "single-quote-surface-and-canonical/v2",
      quote: "“永远滴神”是“永远的神”的谐音写法。",
      relation_note: "材料明确给出两种形式。",
    }],
    "g3form-two": [{
      evidence_id: "ev-two",
      source_id: "source-two",
      publisher: "公开材料乙",
      source_role: "primary",
      acquisition_mode: "manual",
      component_id: "component-two",
      relation_contract: "single-quote-surface-and-canonical/v2",
      quote: "“神马”常被用作“什么”的谐音形式。",
      relation_note: "材料明确给出两种形式。",
    }],
    "g3form-three": [{
      evidence_id: "ev-three",
      source_id: "source-three",
      publisher: "公开材料丙",
      source_role: "primary",
      acquisition_mode: "manual",
      component_id: "component-three",
      relation_contract: "single-quote-surface-and-canonical/v2",
      quote: "“河蟹”是“和谐”的替代写法。",
      relation_note: "材料明确给出两种形式。",
    }],
  };
  const decisions = Object.fromEntries(ITEM_IDS.map(itemId => {
    const item = items[itemId];
    return [itemId, {
      status: "draft",
      action: "defer",
      surface: item.surface,
      canonical: item.canonical,
      family: item.proposed_family,
      phonetic_scan_enabled: item.phonetic_scan_enabled,
      evidence_ids: [...item.evidence_ids],
      notes: "",
    }];
  }));

  function status() {
    const confirmed = Object.values(decisions).filter(row => row.status === "confirmed");
    return {
      frame_id: "frame-test",
      reviewer_id: "reviewer-test",
      revision,
      finalized_reference_id: null,
      item_count: ITEM_IDS.length,
      confirmed_count: confirmed.length,
      deferred_count: confirmed.filter(row => row.action === "defer").length,
      amendment_count: reopenWrites.length,
      action_counts: {},
    };
  }

  function summary(itemId) {
    const item = items[itemId];
    return {
      item_id: itemId,
      surface: item.surface,
      canonical: item.canonical,
      proposed_family: item.proposed_family,
      publishers: [evidence[itemId][0].publisher],
      source_roles: [evidence[itemId][0].source_role],
      evidence_count: evidence[itemId].length,
      status: decisions[itemId].status,
      action: decisions[itemId].action,
    };
  }

  function bootstrap() {
    return {
      schema_version: "wp3-g3-form-review-bootstrap/v1",
      session_token: "test-token",
      frame_id: "frame-test",
      reviewer_id: "reviewer-test",
      revision,
      status: status(),
      actions: ["accept", "reject", "edit", "defer"],
      families: ["known_variant", "phonetic_variant", "orthographic_variant"],
      items: ITEM_IDS.map(summary),
      warnings: ["DEVELOPMENT ONLY / NON-SEALED / NON-SCIENTIFIC"],
    };
  }

  function itemState(itemId) {
    return {
      schema_version: "wp3-g3-form-review-item/v1",
      revision,
      item: items[itemId],
      evidence: evidence[itemId],
      decision: decisions[itemId],
      item_summary: summary(itemId),
    };
  }

  function mutation(itemId) {
    return {
      schema_version: "wp3-g3-form-review-mutation/v1",
      revision,
      status: status(),
      decision: decisions[itemId],
      item_summary: summary(itemId),
    };
  }

  async function install(page) {
    function recordClient(route) {
      clientInstances.push({
        path: new URL(route.request().url()).pathname,
        id: route.request().headers()["x-review-client-instance"] || "",
      });
    }
    await page.route("**/review-base.css", route => route.fulfill({
      contentType: "text/css",
      body: fs.readFileSync(path.join(SHARED_ROOT, "styles.css")),
    }));
    await page.route("**/review-core.js", route => route.fulfill({
      contentType: "text/javascript",
      body: fs.readFileSync(path.join(SHARED_ROOT, "core.js")),
    }));
    await page.route("**/api/bootstrap", route => {
      recordClient(route);
      return route.fulfill({json: bootstrap()});
    });
    await page.route(/\/api\/items\/g3form-(one|two|three)$/, route => {
      recordClient(route);
      const itemId = route.request().url().match(/(g3form-(?:one|two|three))$/)[1];
      return route.fulfill({json: itemState(itemId)});
    });
    await page.route("**/api/save", async route => {
      recordClient(route);
      const body = route.request().postDataJSON();
      saveAttempts.push(body);
      if (conflictRemaining > 0) {
        conflictRemaining -= 1;
        await route.fulfill({
          status: 409,
          contentType: "application/json",
          body: JSON.stringify({error: "review session changed concurrently"}),
        });
        return;
      }
      if (options.saveDelayMs) {
        await new Promise(resolve => setTimeout(resolve, options.saveDelayMs));
      }
      if (body.expected_revision !== revision) {
        await route.fulfill({
          status: 409,
          contentType: "application/json",
          body: JSON.stringify({error: "review session changed concurrently"}),
        });
        return;
      }
      writes.push(body);
      revisionNumber += 1;
      revision = `revision-${revisionNumber}`;
      decisions[body.item_id] = {
        status: body.confirm ? "confirmed" : "draft",
        ...body.decision,
      };
      await route.fulfill({json: mutation(body.item_id)});
    });
    await page.route("**/api/reopen", async route => {
      recordClient(route);
      const body = route.request().postDataJSON();
      reopenWrites.push(body);
      revisionNumber += 1;
      revision = `revision-${revisionNumber}`;
      decisions[body.item_id].status = "draft";
      await route.fulfill({json: mutation(body.item_id)});
    });
  }

  return {install, writes, saveAttempts, reopenWrites, decisions, clientInstances};
}

test("1-4 decisions, Ctrl+S and bracket navigation follow the S2.1 keyboard contract", async ({page}) => {
  const data = fixture();
  await data.install(page);
  await page.goto("/index.html");
  await expect(page.locator("#item-id")).toHaveText("g3form-one");
  await expect(page.locator('.action-choice[data-action="accept"]')).toBeEnabled();

  const actions = ["accept", "edit", "reject", "defer"];
  for (let index = 0; index < actions.length; index += 1) {
    await page.keyboard.press(String(index + 1));
    await expect(page.locator(`.action-choice[data-action="${actions[index]}"]`)).toHaveClass(/active/);
  }

  await page.keyboard.press("3");
  await page.keyboard.press("Control+s");
  await expect.poll(() => data.writes.length).toBe(1);
  expect(data.writes[0].confirm).toBe(false);
  expect(data.writes[0].decision.action).toBe("reject");
  await expect(page.locator("#save-state")).toHaveAttribute("data-state", "saved");
  await expect(page.locator("#next-item")).toBeEnabled();

  await page.keyboard.press("]");
  await expect(page.locator("#item-id")).toHaveText("g3form-two");
  await page.keyboard.press("[");
  await expect(page.locator("#item-id")).toHaveText("g3form-one");
});

test("single-key decisions are suppressed while editing text", async ({page}) => {
  const data = fixture();
  await data.install(page);
  await page.goto("/index.html");
  await expect(page.locator('.action-choice[data-action="accept"]')).toBeEnabled();
  await page.keyboard.press("2");
  await page.locator("#surface").focus();
  await page.keyboard.press("1");
  await expect(page.locator('.action-choice[data-action="edit"]')).toHaveClass(/active/);
});

test("confirm locks the decision and advances to the next unfinished item", async ({page}) => {
  const data = fixture();
  await data.install(page);
  await page.goto("/index.html");
  await expect(page.locator('.action-choice[data-action="accept"]')).toBeEnabled();
  await page.keyboard.press("1");
  await page.keyboard.press("Control+Enter");
  await expect.poll(() => data.writes.filter(row => row.confirm).length).toBe(1);
  expect(data.writes.find(row => row.confirm).decision.action).toBe("accept");
  await expect(page.locator("#item-id")).toHaveText("g3form-two");
});

test("annotation help is G3-specific and contains no ABC routing rubric", async ({page}) => {
  const data = fixture();
  await data.install(page);
  await page.goto("/index.html");
  await page.getByRole("button", {name: "标注说明"}).click();
  await expect(page.locator("#guideline-dialog")).toBeVisible();
  await expect(page.locator("#guideline-dialog")).toContainText("只审核冻结证据支持的形式关系");
  await expect(page.locator("#guideline-dialog")).toContainText("接受");
  await expect(page.locator("#guideline-dialog")).toContainText("修订");
  await expect(page.locator("#guideline-dialog")).toContainText("驳回");
  await expect(page.locator("#guideline-dialog")).toContainText("暂缓");
  await expect(page.locator("#guideline-dialog")).not.toContainText("A_candidate");
});

test("confirmed decisions reopen only after an amendment reason", async ({page}) => {
  const data = fixture();
  await data.install(page);
  await page.goto("/index.html");
  await expect(page.locator('.action-choice[data-action="accept"]')).toBeEnabled();
  await page.keyboard.press("1");
  await page.keyboard.press("Control+Enter");
  await expect(page.locator("#item-id")).toHaveText("g3form-two");
  await page.keyboard.press("[");
  await page.getByRole("button", {name: "重新打开"}).click();
  await page.getByRole("button", {name: "重新打开", exact: true}).last().click();
  await expect(page.locator("#dialog-input-error")).toBeVisible();
  await page.locator("#dialog-input").fill("修正关系 family");
  await page.getByRole("button", {name: "重新打开", exact: true}).last().click();
  await expect.poll(() => data.reopenWrites.length).toBe(1);
  expect(data.reopenWrites[0].reason).toBe("修正关系 family");
});

test("CAS conflict requires an explicit server-version or local-reapply choice", async ({page}) => {
  const data = fixture({conflictOnce: true});
  await data.install(page);
  await page.goto("/index.html");
  await expect(page.locator('.action-choice[data-action="accept"]')).toBeEnabled();
  await page.keyboard.press("3");
  await page.keyboard.press("Control+s");
  await expect(page.locator("#conflict-dialog")).toBeVisible();
  await expect(page.locator("#conflict-details")).toContainText("changed concurrently");
  await page.getByRole("button", {name: "重新应用本地决定"}).click();
  await expect.poll(() => data.writes.length).toBe(1);
  expect(data.writes[0].decision.action).toBe("reject");
});

test("in-flight autosave suppresses duplicate shortcuts and tags one page consistently", async ({page}) => {
  const data = fixture({saveDelayMs: 700});
  await data.install(page);
  await page.goto("/index.html");
  await expect(page.locator('.action-choice[data-action="accept"]')).toBeEnabled();
  await page.keyboard.press("3");
  await expect.poll(() => data.saveAttempts.length).toBe(1);
  await expect(page.locator("#save-state")).toHaveAttribute("data-state", "saving");
  await page.keyboard.press("Control+s");
  await page.keyboard.press("Control+Enter");
  await expect.poll(() => data.writes.length).toBe(1);
  await page.waitForTimeout(150);
  expect(data.saveAttempts).toHaveLength(1);
  await expect(page.locator("#conflict-dialog")).toBeHidden();
  const ids = new Set(data.clientInstances.map(row => row.id));
  expect(ids.size).toBe(1);
  expect([...ids][0]).toMatch(/^[a-zA-Z0-9-]{16,64}$/);
});

test("separate pages use different anonymous client-instance IDs", async ({page, context}) => {
  const data = fixture();
  await data.install(page);
  await page.goto("/index.html");
  await expect(page.locator("#item-id")).toHaveText("g3form-one");
  const secondPage = await context.newPage();
  await data.install(secondPage);
  await secondPage.goto("/index.html");
  await expect(secondPage.locator("#item-id")).toHaveText("g3form-one");
  const bootstrapIds = data.clientInstances
    .filter(row => row.path === "/api/bootstrap")
    .map(row => row.id);
  expect(new Set(bootstrapIds).size).toBe(2);
  await secondPage.close();
});

test("mobile workbench keeps the full-screen queue and has no horizontal overflow", async ({page}, testInfo) => {
  test.skip(!testInfo.project.name.includes("mobile"), "mobile project only");
  const data = fixture();
  await data.install(page);
  await page.goto("/index.html");
  await page.locator("#sidebar-toggle").click();
  await expect(page.locator("#item-sidebar")).toHaveClass(/open/);
  const dimensions = await page.evaluate(() => ({
    scrollWidth: document.documentElement.scrollWidth,
    clientWidth: document.documentElement.clientWidth,
  }));
  expect(dimensions.scrollWidth).toBeLessThanOrEqual(dimensions.clientWidth);
  await expect(page.locator("#confirm-next")).toBeVisible();
});
