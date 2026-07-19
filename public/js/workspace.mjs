export function updatePinnedPaths(paths, path, pinned = true, maxPaths = 20) {
  const next = paths.filter((item) => item !== path);
  if (pinned && path) next.push(path);
  return next.slice(-maxPaths);
}

export function splitCommand(value) {
  const parts = value.match(/(?:[^\s"']+|"[^"]*"|'[^']*')+/g) || [];
  const unquote = (item) => item.replace(/^['"]|['"]$/g, "");
  return { command: unquote(parts.shift() || ""), args: parts.map(unquote) };
}
