// Cloud engine values in the model picker are prefixed so we can tell them
// apart from local Ollama model names.
const CLOUD_PREFIX = "cloud:";

export function isCloudEngine(engineValue) {
  return String(engineValue || "").startsWith(CLOUD_PREFIX);
}

export function cloudEngineValue(provider) {
  return `${CLOUD_PREFIX}${provider}`;
}

export function buildChatPayload(content, engineValue, targetPath, contextPaths, options = {}) {
  const cloud = isCloudEngine(engineValue);
  const provider = cloud ? String(engineValue).slice(CLOUD_PREFIX.length) : "ollama";
  const agentMode = Boolean(options.agentMode);
  return {
    content,
    // model must be non-empty; for cloud we pass the provider id (server resolves the model)
    model: (cloud ? provider : engineValue) || "default",
    target_path: targetPath,
    context_paths: [...contextPaths],
    provider,
    agent_mode: agentMode,
    brain_provider: agentMode ? (options.brainProvider || "") : "",
  };
}

export function chatEventStatus(event) {
  if (event.phase === "context") return "Inspecting workspace…";
  if (event.tool) return `Reading workspace: ${event.tool}`;
  return null;
}

// Map an `agent` SSE frame to a renderable choreography step, or null if not one.
export function chatAgentStep(event) {
  if (!event || !event.actor) return null;
  const title = event.actor === "brain"
    ? `Brain · ${event.provider || "supervisor"}`
    : `Worker · ${event.engine === "cloud" ? (event.provider || "cloud") : (event.model || "Ollama")}`;
  const label = {
    planning: "planning…",
    working: "executing…",
    done: "done",
  }[event.state] || event.state || "";
  return { actor: event.actor, state: event.state, title, label, text: event.text || "" };
}
