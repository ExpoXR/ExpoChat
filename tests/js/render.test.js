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

test("timestamps render in the browser's local timezone", async () => {
  const previousTimezone = process.env.TZ;
  process.env.TZ = "Europe/Berlin";
  try {
    const { formatLocalDateTime, formatLocalTime } = await import("../../public/js/render.mjs");
    assert.equal(formatLocalTime("2026-07-19T16:09:00+00:00"), "18:09:00");
    assert.equal(formatLocalDateTime("2026-07-19T16:09:00+00:00"), "2026-07-19 18:09:00");
  } finally {
    if (previousTimezone === undefined) delete process.env.TZ;
    else process.env.TZ = previousTimezone;
  }
});
