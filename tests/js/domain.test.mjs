import assert from "node:assert/strict";
import test from "node:test";

import { buildChatPayload, chatEventStatus } from "../../public/js/chat.mjs";
import { artifactPresentation, runStatusLabel } from "../../public/js/runs.mjs";
import { providerOptions } from "../../public/js/settings.mjs";
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
});

test("settings and preferences are pure and bounded", () => {
  const values = new Map();
  const storage = {
    getItem: (key) => values.get(key) ?? null,
    setItem: (key, value) => values.set(key, value),
  };
  writePreferences(storage, { model: "m", context: ["a"] });
  assert.deepEqual(readPreferences(storage).context, ["a"]);
  assert.deepEqual(providerOptions([{ provider: "codex", enabled: true }, { provider: "claude", enabled: false }]), [
    { value: "codex", label: "Codex (OpenAI)" },
  ]);
});
