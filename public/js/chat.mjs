export function buildChatPayload(content, model, targetPath, contextPaths) {
  return {
    content,
    model,
    target_path: targetPath,
    context_paths: [...contextPaths],
  };
}

export function chatEventStatus(event) {
  if (event.phase === "context") return "Inspecting workspace…";
  if (event.tool) return `Reading workspace: ${event.tool}`;
  return null;
}
