// Tolerant SSE field parser: the spec allows an optional single space after the
// colon ("field:value" or "field: value") and defines id:/retry: fields we ignore.
function fieldValue(line, field) {
  if (!line.startsWith(`${field}:`)) return null;
  const rest = line.slice(field.length + 1);
  return rest.startsWith(" ") ? rest.slice(1) : rest;
}

export function consumeSse(buffer, chunk) {
  const frames = (buffer + chunk).split(/\r?\n\r?\n/);
  const remainder = frames.pop() || "";
  const events = [];
  for (const frame of frames) {
    const lines = frame.split(/\r?\n/);
    let event = "message";
    const dataLines = [];
    for (const line of lines) {
      const eventValue = fieldValue(line, "event");
      if (eventValue !== null) { event = eventValue || "message"; continue; }
      const dataValue = fieldValue(line, "data");
      if (dataValue !== null) dataLines.push(dataValue);
      // id:/retry:/comment lines are intentionally ignored
    }
    const data = dataLines.join("\n");
    if (!data) continue;
    try { events.push({ event, data: JSON.parse(data) }); } catch (_) { /* malformed frame */ }
  }
  return { events, remainder };
}
