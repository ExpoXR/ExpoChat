const test = require("node:test");
const assert = require("node:assert/strict");

test("SSE parser retains partial frames", async () => {
  const { consumeSse } = await import("../../public/js/sse.mjs");
  const first = consumeSse("", "event: tool\ndata: {\"tool\":\"read_file\"}\n\ndata: {\"tok");
  assert.deepEqual(first.events[0], { event: "tool", data: { tool: "read_file" } });
  const second = consumeSse(first.remainder, "en\":\"ok\"}\n\n");
  assert.deepEqual(second.events[0].data, { token: "ok" });
});

test("SSE parser tolerates missing prefix space and ignores id/retry", async () => {
  const { consumeSse } = await import("../../public/js/sse.mjs");
  // No space after colon (spec-legal) plus id:/retry: fields that must be ignored.
  const out = consumeSse("", "id: 7\nretry: 3000\nevent:ping\ndata:{\"n\":1}\n\n");
  assert.equal(out.events.length, 1);
  assert.deepEqual(out.events[0], { event: "ping", data: { n: 1 } });
});
