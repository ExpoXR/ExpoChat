const test = require("node:test");
const assert = require("node:assert/strict");

test("API helper adds CSRF only to mutations", async () => {
  const { createApi } = await import("../../public/js/api.mjs");
  const calls = [];
  const fetchImpl = async (path, options) => {
    calls.push({ path, options });
    return { ok: true, json: async () => ({ ok: true }) };
  };
  const api = createApi(() => "csrf", fetchImpl);
  await api("/read");
  await api("/write", { method: "POST" });
  assert.equal(calls[0].options.headers["X-CSRF-Token"], undefined);
  assert.equal(calls[1].options.headers["X-CSRF-Token"], "csrf");
});
