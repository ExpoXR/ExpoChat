export function escapeHtml(value) {
  return String(value)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

export function formatBytes(value) {
  const bytes = Number(value) || 0;
  if (bytes < 1024) return `${bytes} B`;
  const units = ["KiB", "MiB", "GiB", "TiB"];
  let size = bytes;
  let unit = -1;
  do { size /= 1024; unit += 1; } while (size >= 1024 && unit < units.length - 1);
  return `${size.toFixed(size >= 10 ? 1 : 2)} ${units[unit]}`;
}

export function renderMarkdown(value) {
  let text = String(value);
  const codeBlocks = [];
  text = text.replace(/```([\w.-]*)\r?\n?([\s\S]*?)```/g, (_, lang, code) => {
    const idx = codeBlocks.length;
    codeBlocks.push({ lang: lang.trim() || "text", code: escapeHtml(code.replace(/\r\n/g, "\n").trimEnd()) });
    return `\x00CODE${idx}\x00`;
  });
  text = escapeHtml(text);
  text = text.replace(/`([^`\n]+)`/g, "<code>$1</code>");
  text = text.replace(/^###\s+(.+)$/gm, "<h3>$1</h3>");
  text = text.replace(/^##\s+(.+)$/gm, "<h2>$1</h2>");
  text = text.replace(/^#\s+(.+)$/gm, "<h1>$1</h1>");
  text = text.replace(/\*\*\*(.+?)\*\*\*/g, "<strong><em>$1</em></strong>");
  text = text.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
  text = text.replace(/\*(.+?)\*/g, "<em>$1</em>");
  text = text.replace(/((?:^- .+\n?)+)/gm, (block) => `<ul>${block.trim().split("\n").map((line) => `<li>${line.slice(2)}</li>`).join("")}</ul>`);
  text = text.replace(/((?:^\d+\. .+\n?)+)/gm, (block) => `<ol>${block.trim().split("\n").map((line) => `<li>${line.replace(/^\d+\.\s+/, "")}</li>`).join("")}</ol>`);
  text = text.replace(/^---+$/gm, "<hr>");
  text = text.split(/\n{2,}/).map((part) => {
    const trimmed = part.trim();
    if (!trimmed) return "";
    if (/^<(h[1-6]|ul|ol|hr|pre|div)/.test(trimmed)) return trimmed;
    return `<p>${trimmed.replace(/\n/g, "<br>")}</p>`;
  }).join("\n");
  return text.replace(/\x00CODE(\d+)\x00/g, (_, index) => {
    const { lang, code } = codeBlocks[Number(index)];
    return `<div class="code-block"><div class="code-lang">${lang}</div><pre><code>${code}</code></pre></div>`;
  });
}
