export function runStatusLabel(run) {
  return (run.status || "unknown").replaceAll("_", " ");
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
