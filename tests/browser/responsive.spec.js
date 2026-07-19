const { test, expect } = require("@playwright/test");

test("page never overflows horizontally", async ({ page }) => {
  await page.goto("/");
  await expect(page.locator("body")).toBeVisible();
  const overflow = await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth);
  expect(overflow).toBeLessThanOrEqual(1);
  await expect(page.locator("#username")).toHaveCSS("font-size", /16px|13px/);
});

test("authenticated drawers and mobile navigation", async ({ page }, testInfo) => {
  test.skip(!process.env.E2E_USER || !process.env.E2E_PASSWORD, "E2E credentials not supplied");
  await page.goto("/");
  await page.locator("#username").fill(process.env.E2E_USER);
  await page.locator("#password").fill(process.env.E2E_PASSWORD);
  await page.locator("#loginForm button[type=submit]").click();
  await expect(page.locator("#app")).toBeVisible();
  if (testInfo.project.use.viewport.width <= 767) {
    const visiblePrimary = await page.locator(".activitybar .activity:visible").count();
    expect(visiblePrimary).toBe(4);
    await page.locator("#activityRuns").click();
    await expect(page.locator(".sidebar")).toHaveClass(/open/);
    await page.locator("#closeSidebarBtn").click();
  } else if (testInfo.project.use.viewport.width <= 1180) {
    await page.locator("#activityPlan").click();
    await expect(page.locator(".panel-area")).toHaveClass(/open/);
  }
  const overflow = await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth);
  expect(overflow).toBeLessThanOrEqual(1);
});

test("authenticated workspace chat", async ({ page }, testInfo) => {
  test.skip(!process.env.E2E_USER || !process.env.E2E_PASSWORD, "E2E credentials not supplied");
  await page.goto("/");
  await page.locator("#username").fill(process.env.E2E_USER);
  await page.locator("#password").fill(process.env.E2E_PASSWORD);
  await page.locator("#loginForm button[type=submit]").click();
  await expect(page.locator("#app")).toBeVisible();
  const config = await page.evaluate(() => fetch("/api/config").then((response) => response.json()));
  const workspace = `${config.allowed_roots[0]}/fixture`;
  await page.locator("#activityChat").click();
  await page.locator("#targetPath").fill(workspace);
  await page.locator("#activityFiles").click();
  await page.locator("#filePath").fill(workspace);
  await page.locator("#openPathBtn").click();
  await page.locator("#fileList .item", { hasText: "sample.py" }).click();
  await expect(page.locator("#pinFileBtn")).toHaveText("Unpin from Chat");
  await expect(page.locator("#contextTag")).toContainText("1");
  await page.locator("#activityChat").click();
  await page.locator("#newChatBtn").click();
  if (testInfo.project.use.viewport.width <= 767) {
    await page.locator("#closeSidebarBtn").click();
  }
  await page.locator('.editor-tab[data-editor="chatEditor"]').click();
  await page.locator("#prompt").fill("Read workspace fixture");
  await page.locator("#sendBtn").click();
  await expect(page.locator(".msg.assistant .msg-body").last()).toContainText("Workspace fixture answer");
});
