export function runStatusLabel(run) {
  return (run.status || "unknown").replaceAll("_", " ");
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
