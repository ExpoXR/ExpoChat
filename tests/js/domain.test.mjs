import assert from "node:assert/strict";
import test from "node:test";

import { buildChatPayload, chatEventStatus } from "../../public/js/chat.mjs";
import { artifactPresentation, explorerActivityState, fileActivity, runStatusLabel, tokenCounts } from "../../public/js/runs.mjs";
import { BRAIN_MODELS, modelLabel, modelOptions, providerOptions } from "../../public/js/settings.mjs";
import { buildSettingsPayload, usageMeter } from "../../public/js/settings_api.mjs";
import { readPreferences, writePreferences } from "../../public/js/state.mjs";
import { splitCommand, updatePinnedPaths } from "../../public/js/workspace.mjs";
import { baseName, duplicateName, joinPath, parentDir, pasteTarget } from "../../public/js/fsops.mjs";

test("fs path helpers resolve names, parents, paste targets, and duplicates", () => {
  assert.equal(baseName("/work/dir/file.txt"), "file.txt");
  assert.equal(baseName("/work/dir/"), "dir");
  assert.equal(parentDir("/work/dir/file.txt"), "/work/dir");
  assert.equal(joinPath("/work/dir/", "x.py"), "/work/dir/x.py");
  assert.equal(pasteTarget("/dest", "/work/a/b.txt"), "/dest/b.txt");
  assert.equal(duplicateName("/work/a/b.txt"), "/work/a/b copy.txt");
  assert.equal(duplicateName("/work/a/folder"), "/work/a/folder copy");
});

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
  assert.ok(BRAIN_MODELS.gemini.length >= 2);
  assert.equal(modelLabel("codex", "gpt-5.6-terra"), "GPT-5.6 Terra");
  assert.equal(modelLabel("gemini", "gemini-2.5-pro"), "Gemini 2.5 Pro");
  assert.equal(modelOptions("claude", "custom-model")[0].value, "custom-model");
  assert.deepEqual(providerOptions([
    { provider: "codex", model: "gpt-5.6-sol", enabled: true, linked: true },
    { provider: "claude", model: "claude-sonnet-5", enabled: false, linked: false },
    { provider: "gemini", model: "gemini-2.5-pro", enabled: true, linked: true },
  ]), [
    { value: "codex", label: "ChatGPT · GPT-5.6 Sol" },
    { value: "gemini", label: "Gemini · Gemini 2.5 Pro" },
  ]);
});

test("settings payload building and usage meter are pure and clamped", () => {
  // clamps negatives/garbage to 0, drops blanks, validates theme, coerces bool
  assert.deepEqual(
    buildSettingsPayload({ token_budget_daily: "-5", token_budget_run: "", max_output_tokens: "2048", theme: "neon", agent_mode_default: 1 }),
    { token_budget_daily: 0, max_output_tokens: 2048, agent_mode_default: true },
  );
  assert.deepEqual(buildSettingsPayload({ theme: "light" }), { theme: "light" });

  const unlimited = usageMeter({ paid_today: { total: 1200 }, budgets: { daily: 0 } });
  assert.equal(unlimited.unlimited, true);
  assert.equal(unlimited.pct, 0);

  const capped = usageMeter({ paid_today: { total: 800 }, budgets: { daily: 1000 } });
  assert.equal(capped.pct, 80);
  assert.equal(capped.over, false);

  const over = usageMeter({ paid_today: { total: 1200 }, budgets: { daily: 1000 } });
  assert.equal(over.over, true);
  assert.equal(over.pct, 100);
});
