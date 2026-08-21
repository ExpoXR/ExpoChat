// Pure path helpers for Explorer file operations (unit-testable, no DOM).

export function baseName(path) {
  return String(path).split("/").filter(Boolean).pop() || "";
}

export function parentDir(path) {
  const parts = String(path).replace(/\/+$/, "").split("/");
  parts.pop();
  return parts.join("/") || "/";
}

export function joinPath(dir, name) {
  return `${String(dir).replace(/\/+$/, "")}/${name}`;
}

// Destination path when pasting clipboard entry into a directory.
export function pasteTarget(destDir, clipboardPath) {
  return joinPath(destDir, baseName(clipboardPath));
}

// "<name> copy<.ext>" sibling path for Duplicate.
export function duplicateName(path) {
  const base = baseName(path);
  const dir = parentDir(path);
  const dot = base.lastIndexOf(".");
  const stem = dot > 0 ? base.slice(0, dot) : base;
  const ext = dot > 0 ? base.slice(dot) : "";
  return joinPath(dir, `${stem} copy${ext}`);
}
