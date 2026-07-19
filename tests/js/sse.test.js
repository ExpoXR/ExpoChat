const test = require("node:test");
const assert = require("node:assert/strict");

test("SSE parser retains partial frames", async () => {
  const { consumeSse } = await import("../../public/js/sse.mjs");
  const first = consumeSse("", "event: tool\ndata: {\"tool\":\"read_file\"}\n\ndata: {\"tok");
  assert.deepEqual(first.events[0], { event: "tool", data: { tool: "read_file" } });
  const second = consumeSse(first.remainder, "en\":\"ok\"}\n\n");
  assert.deepEqual(second.events[0].data, { token: "ok" });
});
