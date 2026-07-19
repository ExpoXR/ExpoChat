const test = require("node:test");
const assert = require("node:assert/strict");

test("markdown escapes raw HTML and preserves escaped code", async () => {
  const { renderMarkdown } = await import("../../public/js/render.mjs");
  const rendered = renderMarkdown("<img src=x onerror=alert(1)>\n\n```js\n<a>\n```");
  assert.ok(!rendered.includes("<img"));
  assert.ok(rendered.includes("&lt;a&gt;"));
});

test("byte formatter uses binary units", async () => {
  const { formatBytes } = await import("../../public/js/render.mjs");
  assert.equal(formatBytes(1024), "1.00 KiB");
});
