"use strict";

const {chromium} = require("playwright");

function invariant(value, message) {
  if (!value) throw new Error(message);
}

async function waitForLoaded(page) {
  await page.waitForSelector("#content span", {state: "attached"});
  await page.waitForFunction(() => !document.querySelector("#case-position").textContent.includes("—"));
}

async function checkMobile(browser, baseURL, viewport) {
  const page = await browser.newPage({viewport});
  await page.goto(baseURL, {waitUntil: "domcontentloaded"});
  await waitForLoaded(page);
  await page.locator("#sidebar-toggle").click();
  invariant(await page.locator("#case-sidebar").evaluate(node => node.classList.contains("open")), "mobile case drawer did not open");
  const dimensions = await page.evaluate(() => ({
    clientWidth: document.documentElement.clientWidth,
    scrollWidth: document.documentElement.scrollWidth,
  }));
  invariant(dimensions.scrollWidth <= dimensions.clientWidth, "mobile page has horizontal overflow");
  invariant(await page.locator(".decision-actions").isVisible(), "mobile decision bar is not visible");
  await page.close();
  return dimensions;
}

(async () => {
  const baseURL = process.env.HSD_SMOKE_BASE;
  invariant(/^http:\/\/127\.0\.0\.1:\d+\/?$/.test(baseURL || ""), "HSD_SMOKE_BASE must be an explicit loopback URL");
  const browser = await chromium.launch({headless: true});
  try {
    const page = await browser.newPage({viewport: {width: 1440, height: 900}, acceptDownloads: true});
    await page.goto(baseURL, {waitUntil: "domcontentloaded"});
    await waitForLoaded(page);
    invariant((await page.locator("#phase-title").textContent()).includes("阶段 A"), "live session is not in Phase A");
    const firstCase = await page.locator("#case-position").textContent();

    await page.locator("#content").evaluate(root => {
      const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
      let node = walker.nextNode();
      while (node && !/\S/u.test(node.data)) node = walker.nextNode();
      if (!node) throw new Error("source text is empty");
      const start = node.data.search(/\S/u);
      const selected = Array.from(node.data.slice(start))[0];
      const range = document.createRange();
      range.setStart(node, start);
      range.setEnd(node, start + selected.length);
      const selection = window.getSelection();
      selection.removeAllRanges();
      selection.addRange(range);
      root.dispatchEvent(new MouseEvent("mouseup", {bubbles: true}));
    });
    await page.locator("#selection-bar").waitFor({state: "visible"});
    await page.locator("#use-selection").click();
    const surfaceLength = (await page.locator("#decision-editor input").first().inputValue()).length;
    invariant(surfaceLength > 0, "selected mention was not created");
    await page.locator("#save-draft").click();
    await page.waitForFunction(() => document.querySelector("#save-state").textContent.includes("已保存"));

    await page.reload({waitUntil: "domcontentloaded"});
    await waitForLoaded(page);
    invariant((await page.locator("#case-position").textContent()) === firstCase, "draft did not reopen on the same case");
    invariant(await page.locator("#item-strip .item-chip").count() === 1, "saved mention did not survive refresh");
    await page.locator("#confirm-next").click();
    await page.waitForFunction(previous => document.querySelector("#case-position").textContent !== previous, firstCase);

    await page.locator("#top-action-menu summary").click();
    const downloadPromise = page.waitForEvent("download");
    await page.locator("#export").click();
    const download = await downloadPromise;
    invariant(download.suggestedFilename() === "wp3-s21-development-review.zip", "export filename is unexpected");

    const mobileLarge = await checkMobile(browser, baseURL, {width: 412, height: 915});
    const mobileSmall = await checkMobile(browser, baseURL, {width: 375, height: 667});
    process.stdout.write(JSON.stringify({
      confirmed_navigation: true,
      desktop_case: firstCase.split(" · ").pop(),
      export_filename: download.suggestedFilename(),
      mobile_large: mobileLarge,
      mobile_small: mobileSmall,
      refresh_recovery: true,
      selected_surface_length: surfaceLength,
    }) + "\n");
  } finally {
    await browser.close();
  }
})().catch(error => {
  process.stderr.write(`${error.stack || error.message}\n`);
  process.exitCode = 1;
});
