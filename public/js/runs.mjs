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

// Derive a Brain → Worker → Verifier choreography lane from a run's status and
// its emitted events. Pure so it can be unit-tested; the UI renders each entry
// as an .agent-card. state ∈ idle | working | done | error.
export function runChoreography(run = {}, events = []) {
  const types = new Set((events || []).map((event) => event.event_type));
  const status = run.status || "";
  const agents = run.selected_agents || [];
  const worker = agents.find((agent) => (agent.roles || []).includes("implementation")) || agents[0] || {};
  const bad = status === "failed";

  const step = (active, done, failed, activeLabel, doneLabel) => ({
    state: failed ? "error" : done ? "done" : active ? "working" : "idle",
    label: failed ? "failed" : done ? doneLabel : active ? activeLabel : "idle",
  });

  const brainDone = types.has("plan.ready") ||
    ["awaiting_approval", "implementing", "verifying", "applying", "post_check", "completed"].includes(status);
  const implDone = types.has("apply.completed") || types.has("verification.completed") ||
    ["applying", "post_check", "completed"].includes(status);
  const verifyDone = status === "completed";

  // Subtask progress label
  const subs = run.subtasks || [];
  const waiting = status === "waiting_for_ollama";
  const brainWaiting = waiting && (run.plan_state === "provisional" || run.resume_status === "researching");
  const workerWaiting = waiting && (subs.length > 0 || run.resume_status === "implementing");
  const verifierWaiting = waiting && ["verifying", "post_check"].includes(run.resume_status);
  let workerLabel = "implementing…";
  if (subs.length > 0 && ["implementing", "waiting_for_ollama"].includes(status)) {
    const done = subs.filter((s) => s.status === "done").length;
    workerLabel = status === "waiting_for_ollama" ? `saved ${done}/${subs.length} · offline` : `subtasks ${done}/${subs.length}`;
  }

  return [
    { role: "brain", title: `Brain · ${run.brain_provider || "supervisor"}`,
      ...step(
        ["planning_provisional", "researching", "decomposing"].includes(status) || brainWaiting,
        brainDone, bad && !brainDone, brainWaiting ? "saved · Ollama offline" : "planning…", "planned",
      ) },
    { role: "worker", title: `Worker · ${worker.name || "Ollama"}`,
      ...step(status === "implementing" || workerWaiting, implDone, false, workerLabel, "implemented") },
    { role: "verifier", title: "Verifier",
      ...step(
        ["verifying", "post_check"].includes(status) || verifierWaiting,
        verifyDone, status === "rolled_back", verifierWaiting ? "saved · Ollama offline" : "verifying…", "passed",
      ) },
  ];
}

export function subtaskCards(subtasks = []) {
  if (!subtasks.length) return [];
  return subtasks.map((node) => {
    const deps = node.depends_on || [];
    return {
      node_id: node.node_id,
      title: node.title,
      status: node.status || "pending",
      role: node.role || "implementation",
      agent_name: node.agent_name || "",
      deps,
    };
  });
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
