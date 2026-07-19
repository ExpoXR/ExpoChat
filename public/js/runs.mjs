export function runStatusLabel(run) {
  return (run.status || "unknown").replaceAll("_", " ");
}

export function runEventData(event) {
  if (event?.data && typeof event.data === "object") return event.data;
  if (!event?.data_json) return {};
  try { return JSON.parse(event.data_json); } catch (_) { return {}; }
}

function activityPath(targetPath, path) {
  const value = String(path || "").replace(/^\.\//, "");
  if (!value || value === ".") return "";
  if (value.startsWith("/")) return value.replace(/\/$/, "");
  return `${String(targetPath || "").replace(/\/$/, "")}/${value}`;
}

export function fileActivity(events = [], targetPath = "", status = "") {
  const activity = new Map();
  for (const event of events) {
    if (event.event_type !== "agent.activity") continue;
    const data = runEventData(event);
    if (data.phase !== "implementation" || !["write_file", "replace_text", "delete_file"].includes(data.tool)) continue;
    const path = activityPath(targetPath, data.path);
    if (!path) continue;
    if (data.state === "working" || data.state === "changed") activity.set(path, data.state);
    else if (activity.get(path) === "working") activity.delete(path);
  }
  if (["completed", "failed", "cancelled", "rolled_back"].includes(status)) {
    for (const [path, state] of activity) {
      if (state === "working") activity.delete(path);
    }
  }
  return activity;
}

export function explorerActivityState(activity, path, isDirectory = false) {
  const exact = activity.get(path);
  if (exact) return exact;
  if (!isDirectory) return "";
  const prefix = `${String(path).replace(/\/$/, "")}/`;
  let changed = false;
  for (const [file, state] of activity) {
    if (!file.startsWith(prefix)) continue;
    if (state === "working") return "working";
    if (state === "changed") changed = true;
  }
  return changed ? "changed" : "";
}

function tokenBucket(bucket = {}) {
  const input = Number(bucket.input_tokens ?? bucket.prompt_eval_count ?? 0) || 0;
  const output = Number(bucket.output_tokens ?? bucket.eval_count ?? 0) || 0;
  const total = Number(bucket.total_tokens ?? (input + output)) || 0;
  return { input, output, total };
}

export function tokenCounts(usage = {}) {
  if (typeof usage === "string") {
    try { usage = JSON.parse(usage); } catch (_) { usage = {}; }
  }
  return {
    brain: tokenBucket(usage.brain || {}),
    ollama: tokenBucket(usage.ollama || usage),
  };
}

export function artifactPresentation(artifact, content) {
  if (artifact.kind === "diff" || content.startsWith("diff --git ") || content.startsWith("--- ")) {
    return { type: "diff", content };
  }
  try {
    const parsed = JSON.parse(content);
    if (artifact.kind !== "changes" && (parsed.summary || parsed.content)) {
      return { type: "markdown", content: parsed.summary || parsed.content };
    }
    return { type: "json", content: JSON.stringify(parsed, null, 2) };
  } catch (_) {
    if (["command", "console", "check"].includes(artifact.kind)) return { type: "command", content };
    return { type: "markdown", content };
  }
}
