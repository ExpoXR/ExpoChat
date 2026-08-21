const { defineConfig } = require("@playwright/test");

const viewports = [
  ["phone-375", 375, 667],
  ["phone-390", 390, 844],
  ["phone-412", 412, 915],
  ["tablet-768", 768, 1024],
  ["tablet-820", 820, 1180],
  ["tablet-landscape", 1180, 820],
  ["desktop", 1440, 900]
];

module.exports = defineConfig({
  testDir: "tests/browser",
  timeout: 30_000,
  use: {
    baseURL: process.env.E2E_BASE_URL || "http://127.0.0.1:31001",
    trace: "retain-on-failure"
  },
  projects: viewports.map(([name, width, height], index) => ({
    name,
    use: {
      viewport: { width, height },
      extraHTTPHeaders: { "X-Forwarded-For": `10.20.0.${index + 1}` }
    }
  }))
});
