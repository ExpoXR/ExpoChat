import assert from "node:assert/strict";
import test from "node:test";

import { buildChatPayload, chatEventStatus } from "../../public/js/chat.mjs";
import { artifactPresentation, explorerActivityState, fileActivity, runStatusLabel, tokenCounts } from "../../public/js/runs.mjs";
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

test("token counters split Brain and Ollama usage", () => {
  assert.deepEqual(tokenCounts({
    brain: { input_tokens: 100, output_tokens: 25, total_tokens: 125 },
    ollama: { prompt_eval_count: 80, eval_count: 20 },
  }), {
    brain: { input: 100, output: 25, total: 125 },
    ollama: { input: 80, output: 20, total: 100 },
  });
  assert.equal(tokenCounts({ prompt_eval_count: 7, eval_count: 3 }).ollama.total, 10);
  assert.equal(tokenCounts('{"brain":{"total_tokens":12}}').brain.total, 12);
});

test("run activity tracks working and changed files for Explorer", () => {
  const events = [
    { event_type: "agent.activity", data_json: JSON.stringify({ phase: "implementation", state: "working", tool: "replace_text", path: "src/app.js" }) },
    { event_type: "agent.activity", data_json: JSON.stringify({ phase: "implementation", state: "changed", tool: "replace_text", path: "src/app.js" }) },
    { event_type: "agent.activity", data_json: JSON.stringify({ phase: "implementation", state: "working", tool: "write_file", path: "tests/app.test.js" }) },
  ];
  const activity = fileActivity(events, "/project", "implementing");
  assert.equal(activity.get("/project/src/app.js"), "changed");
  assert.equal(activity.get("/project/tests/app.test.js"), "working");
  assert.equal(explorerActivityState(activity, "/project/src", true), "changed");
  assert.equal(explorerActivityState(activity, "/project/tests", true), "working");
  assert.equal(fileActivity(events, "/project", "failed").has("/project/tests/app.test.js"), false);
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
