"use strict";

const path = require("path");
const {defineConfig, devices} = require("@playwright/test");

module.exports = defineConfig({
  testDir: ".",
  testMatch: "e2e.spec.js",
  timeout: 30_000,
  fullyParallel: false,
  workers: 1,
  use: {
    baseURL: "http://127.0.0.1:4174",
    trace: "retain-on-failure",
  },
  webServer: {
    command: "/usr/bin/python3 -m http.server 4174 --bind 127.0.0.1 --directory tools/wp3_g3_form_review_ui",
    cwd: path.resolve(__dirname, "../.."),
    url: "http://127.0.0.1:4174/index.html",
    reuseExistingServer: true,
  },
  projects: [
    {
      name: "chromium-desktop",
      use: {...devices["Desktop Chrome"], viewport: {width: 1440, height: 900}},
    },
    {
      name: "webkit-mobile",
      use: {...devices["iPhone 13"], viewport: {width: 412, height: 915}},
    },
  ],
});
