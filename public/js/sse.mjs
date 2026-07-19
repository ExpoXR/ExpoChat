export function consumeSse(buffer, chunk) {
  const frames = (buffer + chunk).split(/\r?\n\r?\n/);
  const remainder = frames.pop() || "";
  const events = [];
  for (const frame of frames) {
    const event = frame.split(/\r?\n/).find((line) => line.startsWith("event: "))?.slice(7) || "message";
    const data = frame.split(/\r?\n/).filter((line) => line.startsWith("data: ")).map((line) => line.slice(6)).join("\n");
    if (!data) continue;
    try { events.push({ event, data: JSON.parse(data) }); } catch (_) { /* malformed frame */ }
  }
  return { events, remainder };
}
