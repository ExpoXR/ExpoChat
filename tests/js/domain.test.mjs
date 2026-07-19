import assert from "node:assert/strict";
import test from "node:test";

import { buildChatPayload, chatEventStatus } from "../../public/js/chat.mjs";
import { artifactPresentation, runStatusLabel } from "../../public/js/runs.mjs";
import { BRAIN_MODELS, modelLabel, modelOptions, providerOptions } from "../../public/js/settings.mjs";
import { readPreferences, writePreferences } from "../../public/js/state.mjs";
import { splitCommand, updatePinnedPaths } from "../../public/js/workspace.mjs";

test("chat payload and phase status remain backward compatible", () => {
  const payload = buildChatPayload("inspect", "model", "/work", ["/work/app.py"]);
  assert.deepEqual(payload.context_paths, ["/work/app.py"]);
  assert.equal(chatEventStatus({ tool: "read_file" }), "Reading workspace: read_file");
});

test("workspace helpers bound pins and preserve quoted command arguments", () => {
  assert.deepEqual(updatePinnedPaths(["a", "b"], "a"), ["b", "a"]);
  assert.deepEqual(updatePinnedPaths(["a", "b"], "a", false), ["b"]);
  assert.deepEqual(splitCommand('pytest "tests/test core.py" -q'), {
    command: "pytest",
    args: ["tests/test core.py", "-q"],
  });
});

test("run and artifact helpers select explicit render types", () => {
  assert.equal(runStatusLabel({ status: "awaiting_approval" }), "awaiting approval");
  assert.equal(artifactPresentation({ kind: "changes" }, '{"changed":["a.py"]}').type, "json");
  assert.equal(artifactPresentation({ kind: "command" }, "exit=0").type, "command");
  assert.equal(artifactPresentation({ kind: "plan" }, "# Plan").type, "markdown");
  assert.deepEqual(
    artifactPresentation({ kind: "research" }, JSON.stringify({ summary: "# Human summary", events: [{ tool: "read" }] })),
    { type: "markdown", content: "# Human summary" },
  );
});

test("settings and preferences are pure and bounded", () => {
  const values = new Map();
  const storage = {
    getItem: (key) => values.get(key) ?? null,
    setItem: (key, value) => values.set(key, value),
  };
  writePreferences(storage, { model: "m", context: ["a"] });
  assert.deepEqual(readPreferences(storage).context, ["a"]);
  assert.ok(BRAIN_MODELS.codex.length > 2);
  assert.ok(BRAIN_MODELS.claude.length > 2);
  assert.equal(modelLabel("codex", "gpt-5.6-terra"), "GPT-5.6 Terra");
  assert.equal(modelOptions("claude", "custom-model")[0].value, "custom-model");
  assert.deepEqual(providerOptions([
    { provider: "codex", model: "gpt-5.6-sol", enabled: true, linked: true },
    { provider: "claude", model: "claude-sonnet-5", enabled: false, linked: false },
  ]), [
    { value: "codex", label: "ChatGPT · GPT-5.6 Sol" },
  ]);
});
