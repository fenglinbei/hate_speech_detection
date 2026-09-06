"use strict";

const {defineConfig, devices} = require("@playwright/test");

module.exports = defineConfig({
  testDir: ".",
  timeout: 30_000,
  fullyParallel: false,
  workers: 1,
  use: {
    baseURL: "http://127.0.0.1:4173",
    trace: "retain-on-failure",
  },
  webServer: {
    command: "/usr/bin/python3 -m http.server 4173 --bind 127.0.0.1 --directory tools/wp3_candidate_review_ui",
    url: "http://127.0.0.1:4173/index.html",
    reuseExistingServer: true,
  },
  projects: [
    {name: "chromium-desktop", use: {...devices["Desktop Chrome"], viewport: {width: 1440, height: 900}}},
    {name: "chromium-compact", use: {...devices["Desktop Chrome"], viewport: {width: 1280, height: 720}}},
    {name: "chromium-tablet", use: {...devices["Desktop Chrome"], viewport: {width: 1024, height: 768}}},
    {name: "webkit-mobile", use: {...devices["iPhone 13"], viewport: {width: 412, height: 915}}},
    {name: "webkit-small-mobile", use: {...devices["iPhone SE"], viewport: {width: 375, height: 667}}},
  ],
});
