import { createApi } from "/js/api.mjs";
import { buildChatPayload, chatEventStatus } from "/js/chat.mjs";
import { escapeHtml, formatBytes, formatLocalDateTime, formatLocalTime, renderMarkdown } from "/js/render.mjs";
import { artifactPresentation, explorerActivityState, fileActivity, runEventData, runStatusLabel, tokenCounts } from "/js/runs.mjs";
import { modelLabel, modelOptions, providerOptions } from "/js/settings.mjs";
import { consumeSse } from "/js/sse.mjs";
import { readPreferences, writePreferences } from "/js/state.mjs";
import { splitCommand, updatePinnedPaths } from "/js/workspace.mjs";

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

const $ = (id) => document.getElementById(id);

let models = [];
let chats = [];
let currentChat = null;
let currentFile = null;
let pinnedContextPaths = [];
let claudeEnabled = false;
let openaiEnabled = false;
let allowedRoots = [];
let csrfToken = "";
let runs = [];
let currentRun = null;
let currentRunData = null;
let runEventSource = null;
let brains = [];
let agents = [];
let drawerReturnFocus = null;
let searchCursor = null;
let activeSearch = "";
let chatNext = null;
let runNext = null;
let snapshotNext = null;
let timelineNext = null;
let planEditing = false;
let snapshotRetentionDays = 30;
let runFileActivity = new Map();

// ---------------------------------------------------------------------------
// Core API helper
// ---------------------------------------------------------------------------

const api = createApi(() => csrfToken);

// ---------------------------------------------------------------------------
// UI helpers
// ---------------------------------------------------------------------------

function show(id) { $(id).classList.remove("hidden"); }
function hide(id) { $(id).classList.add("hidden"); }

function setStatus(text) {
  $("status").textContent = text;
}

function setBusy(ids, busy) {
  ids.forEach((id) => { const el = $(id); if (el) el.disabled = busy; })
}

function setWorkspaceTag(path) {
  const tag = $("workspaceTag");
  if (!tag) return;
  if (path) {
    const label = path.split("/").filter(Boolean).pop() || path;
    tag.textContent = `📂 ${label}`;
    tag.title = `Workspace: ${path}`;
    tag.classList.remove("hidden");
  } else {
    tag.classList.add("hidden");
  };
}

function syncPinnedContext() {
  const tag = $("contextTag");
  const button = $("pinFileBtn");
  const currentPinned = Boolean(currentFile && pinnedContextPaths.includes(currentFile));
  button.disabled = !currentFile;
  button.textContent = currentPinned ? "Unpin from Chat" : "Pin to Chat";
  if (pinnedContextPaths.length) {
    tag.textContent = `📌 ${pinnedContextPaths.length}`;
    tag.title = `Pinned context:\n${pinnedContextPaths.join("\n")}`;
    tag.classList.remove("hidden");
  } else {
    tag.classList.add("hidden");
  }
}

function pinContextPath(path) {
  if (!path) return;
  pinnedContextPaths = updatePinnedPaths(pinnedContextPaths, path);
  syncPinnedContext();
  savePrefs();
}

function toggleCurrentFilePin() {
  if (!currentFile) return;
  if (pinnedContextPaths.includes(currentFile)) {
    pinnedContextPaths = updatePinnedPaths(pinnedContextPaths, currentFile, false);
    syncPinnedContext();
    savePrefs();
  } else {
    pinContextPath(currentFile);
  }
}

function resetPinnedContext() {
  pinnedContextPaths = [];
  syncPinnedContext();
  savePrefs();
}

function showToast(message, type = "info") {
  const t = $("toast");
  t.textContent = message;
  t.className = `toast toast-${type}`;
  clearTimeout(t._tid);
  t._tid = setTimeout(() => t.classList.add("hidden"), 3500);
}

// ---------------------------------------------------------------------------
// Layout switches
// ---------------------------------------------------------------------------

function switchSidePane(name) {
  ["chat", "runs", "files", "tools", "snaps", "timeline", "settings", "more"].forEach((pane) => {
    const panEl = $(pane + "Pane");
    if (panEl) panEl.classList.toggle("hidden", pane !== name);
    const btn = $("activity" + pane[0].toUpperCase() + pane.slice(1));
    if (btn) btn.classList.toggle("active", pane === name);
  });
  drawerReturnFocus = document.activeElement;
  document.querySelector(".sidebar").classList.add("open");
  syncDrawerState();
}

function syncDrawerState() {
  const mobile = window.matchMedia("(max-width: 767px)").matches;
  const open = document.querySelector(".sidebar").classList.contains("open") ||
               document.querySelector(".panel-area").classList.contains("open");
  $("drawerBackdrop").classList.toggle("hidden", !(mobile && open));
  $("activityPlan").setAttribute("aria-expanded", String(document.querySelector(".panel-area").classList.contains("open")));
}

function closeDrawers() {
  document.querySelector(".sidebar").classList.remove("open");
  document.querySelector(".panel-area").classList.remove("open");
  syncDrawerState();
  if (drawerReturnFocus && drawerReturnFocus.isConnected) drawerReturnFocus.focus();
  drawerReturnFocus = null;
}

function switchEditor(id) {
  ["chatEditor", "fileEditor", "diffEditor", "artifactEditor"].forEach((pane) =>
    $(pane).classList.toggle("hidden", pane !== id)
  );
  document.querySelectorAll(".editor-tab").forEach((tab) =>
    tab.classList.toggle("active", tab.dataset.editor === id)
  );
}

// ---------------------------------------------------------------------------
// Message rendering
// ---------------------------------------------------------------------------

function addMessageNode(role) {
  // Remove stream cursor if it was floating
  const cursor = $("streamCursor");
  if (cursor && cursor.parentElement === $("messages")) {
    $("messages").removeChild(cursor);
  }

  const node = document.createElement("article");
  node.className = `msg ${role}`;
  const label = document.createElement("div");
  label.className = "role";
  label.textContent = role === "assistant" ? "ollama" : role;
  const body = document.createElement("div");
  body.className = "msg-body";
  node.append(label, body);
  $("messages").appendChild(node);

  // Re-append stream cursor after the new node
  if (cursor) $("messages").appendChild(cursor);

  $("messages").scrollTop = $("messages").scrollHeight;
  return node;
}

function renderMessageContent(node, text) {
  const body = node.querySelector(".msg-body");
  body.innerHTML = renderMarkdown(text);
  $("messages").scrollTop = $("messages").scrollHeight;
}

function addMessage(role, content) {
  const node = addMessageNode(role);
  renderMessageContent(node, content);
  return node;
}

// ---------------------------------------------------------------------------
// Chat — model list + history
// ---------------------------------------------------------------------------

function renderChats() {
  $("chatList").innerHTML = "";
  chats.forEach((chat) => {
    const btn = document.createElement("button");
    btn.className = "item" + (chat.id === currentChat ? " active" : "");
    btn.textContent = chat.title || "New chat";
    btn.onclick = () => loadChat(chat.id);
    $("chatList").appendChild(btn);
  });
}

async function loadModels() {
  const data = await api("/api/models");
  models = data.models || [];
  const sel = $("modelSelect");
  const prev = sel.value || loadPrefs().model;
  sel.innerHTML = "";
  models.forEach((m) => {
    const opt = document.createElement("option");
    opt.value = m.name || m.model;
    opt.textContent = m.name || m.model;
    sel.appendChild(opt);
  });
  if (prev && models.find((m) => (m.name || m.model) === prev)) sel.value = prev;
  $("ollamaStatus").textContent = sel.value || "no model";
}

async function loadChats(append = false) {
  const cursor = append ? chatNext : 0;
  const data = await api(`/api/chats?cursor=${cursor || 0}&limit=50`);
  chats = append ? [...chats, ...(data.chats || [])] : (data.chats || []);
  chatNext = data.next_cursor;
  $("loadMoreChatsBtn").classList.toggle("hidden", chatNext === null || chatNext === undefined);
  renderChats();
}

async function loadChat(id) {
  currentChat = id;
  $("messages").innerHTML = '<span id="streamCursor" class="stream-cursor hidden">▋</span>';
  const data = await api(`/api/chats/${id}/messages`);
  const chat = data.chat || chats.find((item) => item.id === id);
  if (chat) {
    const previousTarget = $("targetPath").value.trim();
    $("targetPath").value = chat.target_path || "";
    $("planPath").value = chat.target_path || "";
    setWorkspaceTag(chat.target_path || "");
    if (previousTarget !== (chat.target_path || "")) resetPinnedContext();
    if (chat.model && [...$("modelSelect").options].some((option) => option.value === chat.model)) {
      $("modelSelect").value = chat.model;
      $("ollamaStatus").textContent = chat.model;
    }
  }
  (data.messages || []).forEach((m) => addMessage(m.role, m.content));
  renderChats();
  savePrefs();
  document.querySelector(".sidebar").classList.remove("open");
  syncDrawerState();
  switchEditor("chatEditor");
}

async function createChat(withSnapshot) {
  const model = $("modelSelect").value;
  const target = $("targetPath").value.trim();
  if (!model) {
    showToast("Select an available Ollama model.", "error");
    return;
  }
  if (!target) {
    show("targetHint");
    showToast("Set a Target Folder so Ollama can see your files.", "error");
    $("targetPath").focus();
    return;
  }
  hide("targetHint");
  setStatus(withSnapshot ? "Creating snapshot…" : "Creating chat…");
  setBusy(["newChatBtn", "saveNewChatBtn"], true);
  try {
    const data = await api("/api/chats", {
      method: "POST",
      body: JSON.stringify({ model, target_path: target, title: "New chat", snapshot: withSnapshot }),
    });
    currentChat = data.chat.id;
    await loadChats();
    await loadChat(currentChat);
    setStatus(data.snapshot ? `Snapshot: ${data.snapshot.kind}` : "Ready");
  } catch (err) {
    showToast("Chat error: " + err.message, "error");
    setStatus("Error");
  } finally {
    setBusy(["newChatBtn", "saveNewChatBtn"], false);
  }
}

async function newChat() { return createChat(false); }
async function saveAndNewChat() { return createChat(true); }

// ---------------------------------------------------------------------------
// Chat — streaming send
// ---------------------------------------------------------------------------

async function sendPrompt(event) {
  event.preventDefault();
  const content = $("prompt").value.trim();
  if (!content) return;
  if (!$("modelSelect").value) {
    showToast("Select an available Ollama model.", "error");
    return;
  }
  if (!$("targetPath").value.trim()) {
    show("targetHint");
    showToast("Set a Target Folder so Ollama can see your files.", "error");
    $("targetPath").focus();
    setBusy(["sendBtn", "prompt"], false);
    return;
  }
  hide("targetHint");
  if (!currentChat) {
    await createChat(false);
    if (!currentChat) return;
  }
  $("prompt").value = "";
  addMessage("user", content);
  setStatus("Thinking…");
  setBusy(["sendBtn", "prompt"], true);

  const node = addMessageNode("assistant");
  show("streamCursor");

  let fullText = "";
  try {
    const res = await fetch(`/api/chats/${currentChat}/message`, {
      method: "POST",
      credentials: "same-origin",
      headers: {
        "Content-Type": "application/json",
        "X-CSRF-Token": csrfToken,
      },
      body: JSON.stringify(buildChatPayload(
        content,
        $("modelSelect").value,
        $("targetPath").value.trim(),
        pinnedContextPaths,
      )),
    });

    if (!res.ok) {
      const err = await res.text().catch(() => res.statusText);
      throw new Error(err);
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buf = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      const parsedFrames = consumeSse(buf, decoder.decode(value, { stream: true }));
      buf = parsedFrames.remainder;
      for (const frame of parsedFrames.events) {
        const parsed = frame.data;
        if (parsed.done) {
          if (parsed.workspace) setWorkspaceTag(parsed.workspace);
          continue;
        }
        if (parsed.error) throw new Error(parsed.error);
        const eventStatus = chatEventStatus(parsed);
        if (eventStatus) setStatus(eventStatus);
        if (parsed.token) {
          fullText += parsed.token;
          renderMessageContent(node, fullText);
        }
      }
    }

    await loadChats();
    setStatus("Ready");
    savePrefs();
  } catch (err) {
    showToast(err.message, "error");
    setStatus("Error");
    if (!fullText) node.querySelector(".msg-body").textContent = "Error: " + err.message;
  } finally {
    hide("streamCursor");
    setBusy(["sendBtn", "prompt"], false);
  }
}

// ---------------------------------------------------------------------------
// File explorer
// ---------------------------------------------------------------------------

async function openPath(path) {
  const data = await api(`/api/files?path=${encodeURIComponent(path || "/")}`);
  if (data.is_dir === false) {
    await openFile(data.path);
    return;
  }
  $("filePath").value = data.path === "/" ? "" : data.path;
  $("fileList").innerHTML = "";

  // Sync the chat target folder and plan path to wherever the Explorer is pointing
  if (data.path && data.path !== "/") {
    const previousTarget = $("targetPath").value.trim();
    $("targetPath").value = data.path;
    $("planPath").value = data.path;
    if (previousTarget && previousTarget !== data.path) resetPinnedContext();
    setWorkspaceTag(data.path);
    hide("targetHint");
    savePrefs();
    const ctx = $("explorerContext");
    ctx.textContent = `📂 Ollama context: ${data.path}`;
    show("explorerContext");
  } else {
    hide("explorerContext");
  }

  if (data.path !== "/") {
    const up = document.createElement("button");
    up.className = "item";
    up.innerHTML = '<span class="file-icon">↑</span>..';
    const atAllowedRoot = allowedRoots.includes(data.path);
    const parent = atAllowedRoot
      ? ""
      : data.path.split("/").slice(0, -1).join("/") || "";
    up.onclick = () => openPath(parent);
    $("fileList").appendChild(up);
  }

  (data.items || []).forEach((item) => {
    const btn = document.createElement("button");
    btn.className = "item";
    btn.innerHTML = `<span class="file-icon">${item.is_dir ? "📁" : "📄"}</span>${escapeHtml(item.name)}`;
    btn.dataset.path = item.path;
    btn.dataset.directory = String(Boolean(item.is_dir));
    btn.onclick = () => (item.is_dir ? openPath(item.path) : openFile(item.path));
    $("fileList").appendChild(btn);
  });
  renderExplorerActivity();
}

function renderExplorerActivity() {
  document.querySelectorAll("#fileList .item[data-path]").forEach((item) => {
    item.querySelector(".file-status-dot")?.remove();
    const state = explorerActivityState(runFileActivity, item.dataset.path, item.dataset.directory === "true");
    if (!state) return;
    const dot = document.createElement("span");
    dot.className = `file-status-dot ${state}`;
    dot.title = state === "working" ? "Agent editing now" : "Changed by current run";
    dot.setAttribute("aria-label", dot.title);
    item.appendChild(dot);
  });
}

async function openFile(path) {
  const data = await api(`/api/file?path=${encodeURIComponent(path)}`);
  currentFile = data.path;
  pinContextPath(currentFile);
  $("fileTitle").textContent = data.path;
  document.querySelector('.editor-tab[data-editor="fileEditor"]').textContent =
    data.path.split("/").pop() || "file";
  $("editor").value = data.content;
  switchEditor("fileEditor");
  document.querySelector(".sidebar").classList.remove("open");
  savePrefs();
}

async function saveFile() {
  if (!currentFile) return;
  setStatus("Saving…");
  setBusy(["saveFileBtn"], true);
  try {
    const data = await api("/api/file", {
      method: "PUT",
      body: JSON.stringify({ path: currentFile, content: $("editor").value, chat_id: currentChat }),
    });
    renderDiff(data.diff || "");
    switchEditor("diffEditor");
    setStatus(data.snapshot ? `Saved — snapshot: ${data.snapshot.kind}` : "Saved");
  } catch (err) {
    showToast(err.message, "error");
    setStatus("Error");
  } finally {
    setBusy(["saveFileBtn"], false);
  }
}

// ---------------------------------------------------------------------------
// Diff rendering
// ---------------------------------------------------------------------------

function renderDiff(raw) {
  $("diffView").innerHTML = "";
  if (!raw) {
    $("diffView").textContent = "No text changes.";
    return;
  }
  raw.split("\n").forEach((line) => {
    const span = document.createElement("span");
    span.textContent = line + "\n";
    if      (line.startsWith("+") && !line.startsWith("+++")) span.className = "diff-add";
    else if (line.startsWith("-") && !line.startsWith("---")) span.className = "diff-del";
    else if (line.startsWith("@@"))                           span.className = "diff-hunk";
    $("diffView").appendChild(span);
  });
}

// ---------------------------------------------------------------------------
// Apply last code block to active file
// ---------------------------------------------------------------------------

async function applyLastCodeBlock() {
  if (!currentFile) {
    showToast("No file open — select a file in Explorer first.", "error");
    return;
  }
  const bodies = $("messages").querySelectorAll(".msg.assistant .msg-body");
  if (!bodies.length) {
    showToast("No assistant message found.", "error");
    return;
  }
  const lastBody = bodies[bodies.length - 1];
  const codeEl = [...lastBody.querySelectorAll(".code-block code")].pop();
  if (!codeEl) {
    showToast("No code block in last response.", "error");
    return;
  }
  const tmp = document.createElement("textarea");
  tmp.innerHTML = codeEl.innerHTML;
  $("editor").value = tmp.value;
  await saveFile();
}

// ---------------------------------------------------------------------------
// Search
// ---------------------------------------------------------------------------

async function searchFiles(append = false) {
  const root = $("filePath").value || $("targetPath").value;
  const q = $("searchTerm").value.trim();
  if (!q) return;
  if (!append || q !== activeSearch) {
    searchCursor = 0;
    activeSearch = q;
    $("searchResults").innerHTML = "";
  }
  setBusy(["searchBtn"], true);
  try {
    const data = await api(`/api/search?root=${encodeURIComponent(root)}&q=${encodeURIComponent(q)}&cursor=${searchCursor || 0}&limit=50`);
    (data.results || []).forEach((r) => {
      const btn = document.createElement("button");
      btn.className = "item";
      btn.textContent = `${r.file.split("/").pop()}:${r.line}  ${r.text.trim()}`;
      btn.title = r.file;
      btn.onclick = () => openFile(r.file);
      $("searchResults").appendChild(btn);
    });
    searchCursor = data.next_cursor;
    $("searchMoreBtn").classList.toggle("hidden", searchCursor === null || searchCursor === undefined);
    if (!data.results?.length && !append) {
      const p = document.createElement("p");
      p.className = "empty-hint";
      p.textContent = "No results.";
      $("searchResults").appendChild(p);
    }
  } catch (err) {
    showToast(err.message, "error");
  } finally {
    setBusy(["searchBtn"], false);
  }
}

// ---------------------------------------------------------------------------
// Terminal — under the chat
// ---------------------------------------------------------------------------

function toggleTerm() {
  const body = $("termBody");
  const btn  = $("termToggleBtn");
  const open = body.classList.toggle("hidden");
  btn.textContent = open ? "▶  Sandbox Console" : "▼  Sandbox Console";
  btn.setAttribute("aria-expanded", String(!open));
}

async function runTermCommand() {
  const cmd = $("termCommand").value.trim();
  if (!cmd) return;
  if (!currentRun) {
    showToast("Start or open a supervisor run first.", "error");
    return;
  }
  setBusy(["termRunBtn"], true);
  $("termOutput").textContent = "Running…";
  // Open the terminal section if it's collapsed
  const body = $("termBody");
  if (body.classList.contains("hidden")) {
    toggleTerm();
  }
  try {
    const { command, args } = splitCommand(cmd);
    const data = await api(`/api/runs/${currentRun}/console`, {
      method: "POST",
      body: JSON.stringify({ command, args }),
    });
    $("termOutput").textContent = `$ ${cmd}\n\n${data.content || data.error || JSON.stringify(data, null, 2)}`;
  } catch (err) {
    $("termOutput").textContent = "Error: " + err.message;
    showToast(err.message, "error");
  } finally {
    setBusy(["termRunBtn"], false);
  }
}

// ---------------------------------------------------------------------------
// Snapshots & Timeline
// ---------------------------------------------------------------------------

async function loadSnaps(append = false) {
  const cursor = append ? snapshotNext : 0;
  const [data, storage] = await Promise.all([api(`/api/snapshots?cursor=${cursor || 0}&limit=50`), api("/api/maintenance/storage")]);
  snapshotRetentionDays = Number(storage.limits?.snapshot_retention_days) || 30;
  $("cleanSnapsBtn").textContent = `Expire Old (>${snapshotRetentionDays}d)`;
  $("cleanSnapsBtn").title = `Expire snapshot archives older than ${snapshotRetentionDays} days`;
  $("storageSummary").textContent = `Snapshots ${formatBytes(storage.tracked.bytes)} · Orphans ${formatBytes(storage.orphan_bytes)} · Free ${formatBytes(storage.filesystem.free_bytes)}`;
  $("orphanList").innerHTML = "";
  (storage.orphans || []).forEach((orphan) => {
    const row = document.createElement("div");
    row.className = "item snap-item";
    const label = document.createElement("span");
    label.textContent = `Orphan · ${formatBytes(orphan.bytes)} · ${orphan.ref}`;
    const remove = document.createElement("button");
    remove.textContent = "Delete";
    remove.disabled = !orphan.cleanup_eligible;
    remove.onclick = async () => {
      if (!confirm(`Permanently delete orphan archive ${orphan.ref}? This cannot be undone.`)) return;
      await api("/api/snapshots/cleanup", {
        method: "POST",
        body: JSON.stringify({ dry_run: false, orphan_refs: [orphan.ref], cleanup_tracked: false }),
      });
      await loadSnaps();
    };
    row.append(label, remove);
    $("orphanList").appendChild(row);
  });
  if (!append) $("snapList").innerHTML = "";
  (data.snapshots || []).forEach((snap) => {
    const row = document.createElement("div");
    row.className = "item snap-item";
    const label = document.createElement("span");
    label.className = "snap-label";
    label.textContent = `${formatLocalDateTime(snap.created_at)}  [${snap.kind}]  ${snap.path.split("/").pop()}`;
    if (snap.archive_deleted_at) label.textContent += " · archive expired";
    label.title = `${snap.path} → ${snap.ref}`;
    const del = document.createElement("button");
    del.className = "snap-del-btn";
    del.textContent = "✕";
    del.title = "Delete this snapshot";
    del.disabled = Boolean(snap.archive_deleted_at || snap.protected);
    del.onclick = async (e) => {
      e.stopPropagation();
      if (!confirm(`Permanently delete snapshot from ${formatLocalDateTime(snap.created_at).slice(0, 10)}? Rollback will become unavailable.`)) return;
      try {
        await api(`/api/snapshots/${snap.id}`, { method: "DELETE" });
        await loadSnaps();
      } catch (err) {
        showToast("Delete failed: " + err.message, "error");
      }
    };
    const restore = document.createElement("button");
    restore.textContent = "Restore";
    restore.disabled = Boolean(snap.archive_deleted_at || snap.protected);
    restore.title = snap.protected ? "Protected by active run" : "Restore this snapshot";
    restore.onclick = async (e) => {
      e.stopPropagation();
      if (!confirm(`Restore ${snap.path} from this snapshot? Current files will be replaced.`)) return;
      try {
        await api(`/api/snapshots/${snap.id}/restore`, { method: "POST", body: "{}" });
        showToast("Snapshot restored.", "success");
        await Promise.all([loadSnaps(), loadTimeline(false)]);
      } catch (err) {
        showToast("Restore failed: " + err.message, "error");
      }
    };
    row.appendChild(label);
    row.appendChild(restore);
    row.appendChild(del);
    $("snapList").appendChild(row);
  });
  snapshotNext = data.next_cursor;
  $("loadMoreSnapsBtn").classList.toggle("hidden", snapshotNext === null || snapshotNext === undefined);
}

async function cleanOldSnaps() {
  setBusy(["cleanSnapsBtn"], true);
  try {
    const preview = await api("/api/snapshots/cleanup", { method: "POST", body: JSON.stringify({ days: snapshotRetentionDays, dry_run: true }) });
    if (!preview.tracked) {
      showToast("No tracked snapshot archives eligible for expiry.", "info");
      return;
    }
    if (!confirm(`Permanently expire ${preview.tracked} tracked snapshot archive(s)?`)) return;
    const data = await api("/api/snapshots/cleanup", { method: "POST", body: JSON.stringify({ days: snapshotRetentionDays, dry_run: false }) });
    showToast(`Expired ${data.deleted} snapshot archive(s) older than ${snapshotRetentionDays} days.`, data.deleted > 0 ? "success" : "info");
    await loadSnaps();
  } catch (err) {
    showToast("Cleanup failed: " + err.message, "error");
  } finally {
    setBusy(["cleanSnapsBtn"], false);
  }
}

async function loadTimeline(append = false) {
  const cursor = append ? timelineNext : 0;
  const data = await api(`/api/timeline?cursor=${cursor || 0}&limit=50`);
  if (!append) $("timelineList").innerHTML = "";
  (data.timeline || []).forEach((event) => {
    const item = document.createElement("button");
    item.className = "item";
    item.textContent = `${formatLocalDateTime(event.created_at)}  [${event.event_type}]  ${event.summary.slice(0, 60)}`;
    item.onclick = () => {
      renderDiff(event.diff || event.summary);
      switchEditor("diffEditor");
    };
    $("timelineList").appendChild(item);
  });
  timelineNext = data.next_cursor;
  $("loadMoreTimelineBtn").classList.toggle("hidden", timelineNext === null || timelineNext === undefined);
}

// ---------------------------------------------------------------------------
// AI Plan — right panel
// ---------------------------------------------------------------------------

async function generatePlan() {
  planEditing = false;
  const path = $("planPath").value.trim()
            || $("filePath").value.trim()
            || $("targetPath").value.trim();
  if (path) $("planPath").value = path;
  const task = $("planTask").value.trim();
  const provider = $("planProvider").value;

  if (!task) {
    showToast("Please enter a task description.", "error");
    return;
  }
  if (!path) {
    showToast("Select a target workspace folder.", "error");
    return;
  }
  if (!provider) {
    showToast("Link Codex or Claude in Brains & Agents.", "error");
    return;
  }
  setBusy(["generatePlanBtn"], true);
  $("planStatus").textContent = "Creating staged research run…";
  hide("planResultArea");
  show("runProgress");
  $("runEventList").innerHTML = "";

  try {
    const data = await api("/api/runs", {
      method: "POST",
      body: JSON.stringify({
        target_path: path,
        task,
        brain_provider: provider,
        web_research: $("webResearch").checked,
      }),
    });
    currentRun = data.id;
    renderCurrentRun(data);
    await loadRuns();
    subscribeRun(data.id);
  } catch (err) {
    $("planStatus").textContent = "";
    showToast("Run error: " + err.message, "error");
  } finally {
    setBusy(["generatePlanBtn"], false);
  }
}

async function implementPlan() {
  const plan = $("planEditorContent").value.trim();
  if (!plan) {
    showToast("No plan to send — generate one first.", "error");
    return;
  }
  if (!currentRun) {
    showToast("No active supervisor run.", "error");
    return;
  }
  setBusy(["approvePlanBtn"], true);
  setStatus("Creating History and snapshot…");
  try {
    const data = await api(`/api/runs/${currentRun}/approve`, {
      method: "POST",
      body: JSON.stringify({ plan }),
    });
    hide("planResultArea");
    renderCurrentRun(data);
    subscribeRun(currentRun);
    await loadRuns();
    setStatus("Implementation running");
    switchEditor("chatEditor");
  } catch (err) {
    showToast(err.message, "error");
    if (err.message.includes("research restarted")) subscribeRun(currentRun);
    setStatus("Run waiting");
  } finally {
    setBusy(["approvePlanBtn"], false);
  }
}

function setPlanEditing(enabled) {
  planEditing = enabled;
  $("planProof").classList.toggle("hidden", enabled);
  $("planEditLabel").classList.toggle("hidden", !enabled);
  $("planEditorContent").classList.toggle("hidden", !enabled);
  $("planEditorContent").readOnly = !enabled;
  $("editPlanBtn").classList.toggle("hidden", enabled);
  $("savePlanBtn").classList.toggle("hidden", !enabled);
  $("cancelPlanEditBtn").classList.toggle("hidden", !enabled);
  if (enabled) $("planEditorContent").focus();
}

async function savePlanChanges() {
  if (!currentRun) return;
  const plan = $("planEditorContent").value.trim();
  if (!plan) {
    showToast("Plan cannot be empty.", "error");
    return;
  }
  setBusy(["savePlanBtn"], true);
  try {
    const run = await api(`/api/runs/${currentRun}/plan`, {
      method: "PUT",
      body: JSON.stringify({ plan }),
    });
    planEditing = false;
    renderCurrentRun(run);
    await loadRun(currentRun);
    showToast("Plan changes saved.", "success");
  } catch (err) {
    showToast(err.message, "error");
  } finally {
    setBusy(["savePlanBtn"], false);
  }
}

async function redoPlan() {
  if (!currentRun || !confirm("Discard current draft and research a new plan?")) return;
  setBusy(["redoPlanBtn"], true);
  try {
    const run = await api(`/api/runs/${currentRun}/redo`, { method: "POST", body: "{}" });
    planEditing = false;
    hide("planResultArea");
    renderCurrentRun(run);
    subscribeRun(currentRun);
    await loadRuns();
  } catch (err) {
    showToast(err.message, "error");
  } finally {
    setBusy(["redoPlanBtn"], false);
  }
}

async function rejectPlan() {
  if (!currentRun) return;
  try {
    await api(`/api/runs/${currentRun}/reject`, { method: "POST", body: "{}" });
    await loadRun(currentRun);
  } catch (err) {
    showToast(err.message, "error");
  }
}

function renderCurrentRun(run) {
  if (currentRun && currentRun !== run.id) planEditing = false;
  currentRun = run.id;
  currentRunData = run;
  const queue = run.queue_position ? ` · queue #${run.queue_position}` : "";
  const chosen = (run.selected_agents || []).map((agent) => agent.name).join(" → ");
  $("currentRunBadge").textContent = `${run.id.slice(0, 8)} · ${runStatusLabel(run)}${queue}${chosen ? ` · ${chosen}` : ""}`;
  show("currentRunBadge");
  show("runProgress");
  $("planStatus").textContent = run.error ? `${runStatusLabel(run)} — ${run.error}` : runStatusLabel(run);
  const awaitingApproval = run.status === "awaiting_approval";
  if (!awaitingApproval) planEditing = false;
  const visiblePlan = awaitingApproval
    ? (run.draft_plan || run.approved_plan)
    : (run.approved_plan || run.draft_plan);
  if (visiblePlan) {
    $("planProof").innerHTML = renderMarkdown(visiblePlan);
    if (!planEditing) $("planEditorContent").value = visiblePlan;
    $("approvePlanBtn").classList.toggle("hidden", !awaitingApproval);
    $("editPlanBtn").classList.toggle("hidden", !awaitingApproval || planEditing);
    $("savePlanBtn").classList.toggle("hidden", !awaitingApproval || !planEditing);
    $("cancelPlanEditBtn").classList.toggle("hidden", !awaitingApproval || !planEditing);
    $("redoPlanBtn").classList.toggle("hidden", !awaitingApproval);
    $("rejectPlanBtn").classList.toggle("hidden", !awaitingApproval);
    $("planProof").classList.toggle("hidden", planEditing);
    $("planEditLabel").classList.toggle("hidden", !planEditing);
    $("planEditorContent").classList.toggle("hidden", !planEditing);
    $("planEditorContent").readOnly = !planEditing;
    show("planResultArea");
  } else {
    hide("planResultArea");
  }
  const usage = tokenCounts(run.usage_json || {});
  const formatTokens = (value) => new Intl.NumberFormat().format(value);
  $("brainTokenCount").textContent = formatTokens(usage.brain.total);
  $("brainTokenCount").title = `${formatTokens(usage.brain.input)} input · ${formatTokens(usage.brain.output)} output`;
  $("ollamaTokenCount").textContent = formatTokens(usage.ollama.total);
  $("ollamaTokenCount").title = `${formatTokens(usage.ollama.input)} input · ${formatTokens(usage.ollama.output)} output`;
  $("cancelRunBtn").classList.toggle("hidden", ["applying", "post_check", "completed", "failed", "cancelled", "rolled_back"].includes(run.status));
  $("resumeRunBtn").classList.toggle("hidden", run.status !== "failed");
  $("rollbackRunBtn").classList.toggle("hidden", !run.snapshot_id || !["completed", "failed"].includes(run.status));
  $("cloneRunBtn").classList.toggle("hidden", !["completed", "failed", "cancelled", "rolled_back"].includes(run.status));
  setStatus(runStatusLabel(run));
}

function renderRunEvents(events) {
  const list = $("runEventList");
  list.innerHTML = "";
  (events || []).forEach((event) => {
    const row = document.createElement("div");
    row.className = "run-event";
    row.innerHTML = `<span>${escapeHtml(formatLocalTime(event.created_at))}</span><strong>${escapeHtml(event.event_type)}</strong><p>${escapeHtml(event.message)}</p>`;
    list.appendChild(row);
  });
  list.scrollTop = list.scrollHeight;
  runFileActivity = fileActivity(events, currentRunData?.target_path || "", currentRunData?.status || "");
  renderExplorerActivity();
  renderRunActivity(events);
}

function renderRunActivity(events) {
  let card = $("runActivityCard");
  const visible = (events || []).filter((event) =>
    event.event_type === "agent.activity" ||
    ["implementation.started", "verification.completed", "apply.completed", "run.completed", "run.failed", "rollback.completed"].includes(event.event_type)
  ).slice(-100);
  if (!visible.length) {
    card?.remove();
    return;
  }
  if (!card) {
    card = document.createElement("article");
    card.id = "runActivityCard";
    card.className = "msg run-activity";
    card.innerHTML = '<div class="role"></div><div class="run-activity-body"></div>';
    const cursor = $("streamCursor");
    $("messages").insertBefore(card, cursor?.parentElement === $("messages") ? cursor : null);
  }
  card.querySelector(".role").textContent = `plan run · ${runStatusLabel(currentRunData || {})}`;
  const body = card.querySelector(".run-activity-body");
  body.innerHTML = "";
  visible.forEach((event) => {
    const data = runEventData(event);
    const row = document.createElement("div");
    row.className = "run-activity-row";
    const dot = document.createElement("span");
    dot.className = `run-activity-dot ${data.state || (event.event_type === "run.failed" ? "failed" : "done")}`;
    const text = document.createElement("span");
    text.textContent = `${formatLocalTime(event.created_at)}  ${event.message}`;
    row.append(dot, text);
    body.appendChild(row);
  });
  $("messages").scrollTop = $("messages").scrollHeight;
}

function renderRunArtifacts(artifacts) {
  const list = $("runArtifactList");
  list.innerHTML = "";
  (artifacts || []).forEach((artifact) => {
    const button = document.createElement("button");
    button.className = "item";
    button.textContent = `${formatLocalDateTime(artifact.created_at)} · ${artifact.kind} · ${artifact.name}`;
    button.onclick = async () => {
      try {
        const data = await api(`/api/runs/${currentRun}/artifacts/${artifact.id}`);
        const presentation = artifactPresentation(artifact, data.content);
        if (presentation.type === "diff") {
          renderDiff(presentation.content);
          switchEditor("diffEditor");
        } else {
          $("artifactTitle").textContent = `${formatLocalDateTime(artifact.created_at)} · ${artifact.kind} · ${artifact.name}`;
          if (presentation.type === "markdown") {
            $("artifactView").innerHTML = renderMarkdown(presentation.content);
          } else {
            $("artifactView").innerHTML = "";
            const pre = document.createElement("pre");
            pre.className = presentation.type === "command" ? "term-output" : "artifact-json";
            pre.textContent = presentation.content;
            $("artifactView").appendChild(pre);
          }
          switchEditor("artifactEditor");
        }
      } catch (err) { showToast(err.message, "error"); }
    };
    list.appendChild(button);
  });
}

async function loadRun(id) {
  const run = await api(`/api/runs/${id}`);
  renderCurrentRun(run);
  renderRunEvents(run.events);
  renderRunArtifacts(run.artifacts);
  if (run.target_path) {
    $("targetPath").value = run.target_path;
    $("planPath").value = run.target_path;
  }
  return run;
}

async function loadRuns(append = false) {
  const cursor = append ? runNext : 0;
  const data = await api(`/api/runs?cursor=${cursor || 0}&limit=50`);
  runs = append ? [...runs, ...(data.runs || [])] : (data.runs || []);
  runNext = data.next_cursor;
  $("runList").innerHTML = "";
  runs.forEach((run) => {
    const btn = document.createElement("button");
    btn.className = "item" + (run.id === currentRun ? " active" : "");
    btn.innerHTML = `<strong>${escapeHtml(run.task.slice(0, 55))}</strong><small>${escapeHtml(runStatusLabel(run))} · ${escapeHtml(run.brain_provider)}</small>`;
    btn.onclick = async () => {
      await loadRun(run.id);
      closeDrawers();
      drawerReturnFocus = btn;
      document.querySelector(".panel-area").classList.add("open");
      syncDrawerState();
      subscribeRun(run.id);
    };
    $("runList").appendChild(btn);
  });
  $("loadMoreRunsBtn").classList.toggle("hidden", runNext === null || runNext === undefined);
}

function subscribeRun(id) {
  if (runEventSource) runEventSource.close();
  runEventSource = new EventSource(`/api/runs/${id}/events`);
  runEventSource.onmessage = () => loadRun(id).catch(() => {});
  ["run.created", "research.started", "research.completed", "plan.ready", "plan.edited", "plan.redo", "plan.approved", "scope.approved", "scope.approval_required", "implementation.started", "agent.activity", "verification.completed", "apply.completed", "rollback.completed", "run.completed", "run.failed", "run.cancelled", "plan.stale"].forEach((name) => {
    runEventSource.addEventListener(name, () => {
      loadRun(id).then((run) => {
        loadRuns().catch(() => {});
        if (["completed", "failed", "cancelled", "rolled_back"].includes(run.status)) runEventSource.close();
      }).catch(() => {});
    });
  });
  runEventSource.onerror = () => loadRun(id).then((run) => {
    if (["completed", "failed", "cancelled", "rolled_back"].includes(run.status)) runEventSource.close();
  }).catch(() => {});
}

async function runAction(action) {
  if (!currentRun) return;
  try {
    await api(`/api/runs/${currentRun}/${action}`, { method: "POST", body: "{}" });
    await loadRun(currentRun);
    subscribeRun(currentRun);
  } catch (err) {
    showToast(err.message, "error");
  }
}

async function loadBrains() {
  const data = await api("/api/brains");
  brains = data.brains || [];
  brains.forEach((brain) => {
    const prefix = brain.provider === "codex" ? "codex" : "claude";
    populateModelSelect(brain.provider, brain.model);
    $(`${prefix}Model`).value = brain.model;
    const linked = Boolean(brain.enabled && brain.linked && !brain.last_error);
    $(`${prefix}State`).textContent = linked
      ? `Linked via ${brain.source}${brain.validated_at ? " · validated" : ""}${brain.last_error ? " · " + brain.last_error : ""}`
      : brain.last_error || "Not linked";
    $(`${prefix}State`).classList.toggle("linked", linked);
    const indicator = brain.provider === "codex" ? $("openaiIndicator") : $("claudeIndicator");
    indicator.textContent = `${modelLabel(brain.provider, brain.model)} ${linked ? "✓" : "!"}`;
    indicator.classList.toggle("linked", linked);
    indicator.classList.toggle("hidden", !brain.enabled && !brain.linked);
  });
  syncProviderOptions();
}

function populateModelSelect(provider, current = "") {
  const select = $(provider === "codex" ? "codexModel" : "claudeModel");
  const selected = current || select.value;
  select.innerHTML = "";
  modelOptions(provider, selected).forEach((model) => {
    const option = document.createElement("option");
    option.value = model.value;
    option.textContent = model.label;
    select.appendChild(option);
  });
  if (selected) select.value = selected;
}

function syncProviderOptions() {
  const enabled = providerOptions(brains);
  const select = $("planProvider");
  const selected = select.value;
  select.innerHTML = "";
  enabled.forEach((provider) => {
    const option = document.createElement("option");
    option.value = provider.value;
    option.textContent = provider.label;
    select.appendChild(option);
  });
  if (!enabled.length) {
    const option = document.createElement("option");
    option.value = "";
    option.textContent = "No AI provider configured";
    select.appendChild(option);
  } else if (enabled.some((provider) => provider.value === selected)) {
    select.value = selected;
  }
}

async function saveBrain(provider) {
  const prefix = provider === "codex" ? "codex" : "claude";
  try {
    await api("/api/brains", {
      method: "PUT",
      body: JSON.stringify({ provider, model: $(`${prefix}Model`).value.trim(), api_key: $(`${prefix}Key`).value.trim(), enabled: true }),
    });
    $(`${prefix}Key`).value = "";
    await loadBrains();
    showToast(`${provider} linked`, "success");
  } catch (err) { showToast(err.message, "error"); }
}

async function disconnectBrain(provider) {
  const prefix = provider === "codex" ? "codex" : "claude";
  try {
    await api("/api/brains", {
      method: "PUT",
      body: JSON.stringify({ provider, model: $(`${prefix}Model`).value.trim(), enabled: false }),
    });
    await loadBrains();
    showToast(`${provider} disconnected`, "success");
  } catch (err) { showToast(err.message, "error"); }
}

async function testBrain(provider) {
  try {
    await api(`/api/brains/${provider}/validate`, { method: "POST", body: "{}" });
    await loadBrains();
    showToast(`${provider} validation passed`, "success");
  } catch (err) { showToast(err.message, "error"); }
}

async function loadAgents() {
  const data = await api("/api/agents");
  agents = data.agents || [];
  $("agentList").innerHTML = "";
  agents.forEach((agent) => {
    const row = document.createElement("div");
    row.className = "agent-row";
    row.innerHTML = `<strong>${escapeHtml(agent.name)}</strong><span>${escapeHtml((agent.roles || []).join(" · "))}</span><small>${escapeHtml((agent.capabilities || []).join(", "))} · priority ${agent.priority}</small>`;
    const roles = document.createElement("input");
    roles.value = (agent.roles || []).join(", ");
    roles.setAttribute("aria-label", `${agent.name} roles`);
    const priority = document.createElement("input");
    priority.type = "number";
    priority.min = "0";
    priority.max = "1000";
    priority.value = agent.priority;
    priority.setAttribute("aria-label", `${agent.name} priority`);
    const prompt = document.createElement("textarea");
    prompt.rows = 2;
    prompt.value = agent.system_prompt || "";
    prompt.placeholder = "Optional agent system prompt";
    prompt.setAttribute("aria-label", `${agent.name} system prompt`);
    const save = document.createElement("button");
    save.textContent = "Save profile";
    save.onclick = async () => {
      try {
        await api(`/api/agents/${agent.id}`, {
          method: "PATCH",
          body: JSON.stringify({
            roles: roles.value.split(",").map((role) => role.trim()).filter(Boolean),
            priority: Number(priority.value),
            system_prompt: prompt.value,
          }),
        });
        await loadAgents();
      } catch (err) { showToast(err.message, "error"); }
    };
    const toggle = document.createElement("button");
    toggle.textContent = agent.enabled ? "Enabled" : "Disabled";
    toggle.onclick = async () => {
      try {
        await api(`/api/agents/${agent.id}`, { method: "PATCH", body: JSON.stringify({ enabled: !agent.enabled }) });
        await loadAgents();
      } catch (err) { showToast(err.message, "error"); }
    };
    row.append(roles, priority, prompt, save, toggle);
    $("agentList").appendChild(row);
  });
}

async function discoverAgentModels() {
  setBusy(["discoverAgentsBtn"], true);
  try {
    await api("/api/agents/discover", { method: "POST", body: "{}" });
    await loadAgents();
    showToast("Ollama agents discovered", "success");
  } catch (err) { showToast(err.message, "error"); }
  finally { setBusy(["discoverAgentsBtn"], false); }
}

// ---------------------------------------------------------------------------
// Preferences (localStorage)
// ---------------------------------------------------------------------------

function savePrefs() {
  try {
    writePreferences(localStorage, {
      model: $("modelSelect").value,
      target: $("targetPath").value,
      file: currentFile,
      chat: currentChat,
      context: pinnedContextPaths,
    });
  } catch (_) {}
}

function loadPrefs() {
  return readPreferences(localStorage);
}

// ---------------------------------------------------------------------------
// Auth & boot
// ---------------------------------------------------------------------------

async function login(event) {
  event.preventDefault();
  $("loginError").textContent = "";
  try {
    await api("/api/auth/login", {
      method: "POST",
      body: JSON.stringify({ username: $("username").value, password: $("password").value }),
    });
    await boot();
  } catch (err) {
    $("loginError").textContent = err.message;
  }
}

async function boot() {
  const me = await api("/api/auth/me");
  if (!me.user) {
    hide("app");
    show("login");
    return;
  }
  csrfToken = me.csrf || "";
  hide("login");
  show("app");
  setStatus("Loading…");

  // Config — know which AI providers are available
  try {
    const cfg = await api("/api/config");
    claudeEnabled = cfg.claude_enabled;
    openaiEnabled = cfg.openai_enabled;
    allowedRoots  = cfg.allowed_roots || [];

    if (claudeEnabled) show("claudeIndicator");
    if (openaiEnabled) show("openaiIndicator");

    // Populate plan provider selector based on what's enabled
    const provSel = $("planProvider");
    provSel.innerHTML = "";
    if (claudeEnabled) {
      const o = document.createElement("option");
      o.value = "claude"; o.textContent = "Claude (Anthropic)";
      provSel.appendChild(o);
    }
    if (openaiEnabled) {
      const o = document.createElement("option");
      o.value = "codex"; o.textContent = "Codex (OpenAI)";
      provSel.appendChild(o);
    }
    if (!claudeEnabled && !openaiEnabled) {
      const o = document.createElement("option");
      o.value = ""; o.textContent = "No AI provider configured";
      provSel.appendChild(o);
    }
  } catch (_) {}

  const prefs = loadPrefs();
  pinnedContextPaths = prefs.context || [];
  syncPinnedContext();

  if (prefs.target) {
    $("targetPath").value = prefs.target;
    $("planPath").value   = prefs.target;
    setWorkspaceTag(prefs.target);
  }

  try {
    await loadModels();
  } catch (err) {
    $("ollamaStatus").textContent = "offline";
    showToast("Ollama models unavailable: " + err.message, "error");
  }
  await loadChats();
  await Promise.allSettled([loadRuns(), loadBrains(), loadAgents()]);

  if (prefs.chat && chats.find((c) => c.id === prefs.chat)) {
    await loadChat(prefs.chat);
  }

  try { await openPath(prefs.target || ""); } catch (err) { showToast(err.message, "error"); }

  if (prefs.file) {
    try { await openFile(prefs.file); } catch (_) {}
  }

  setStatus("Ready");
}

// ---------------------------------------------------------------------------
// DOMContentLoaded — wire up all events
// ---------------------------------------------------------------------------

document.addEventListener("DOMContentLoaded", () => {
  populateModelSelect("codex", "gpt-5.6-sol");
  populateModelSelect("claude", "claude-sonnet-5");
  // Auth
  $("loginForm").onsubmit = login;
  $("logoutBtn").onclick  = async () => {
    await api("/api/auth/logout", { method: "POST", body: "{}" });
    location.reload();
  };

  // Chat
  $("chatForm").onsubmit = sendPrompt;
  $("newChatBtn").onclick      = newChat;
  $("saveNewChatBtn").onclick  = saveAndNewChat;
  $("applyCodeBtn").onclick = applyLastCodeBlock;
  $("modelSelect").onchange = () => {
    $("ollamaStatus").textContent = $("modelSelect").value || "no model";
    savePrefs();
  };
  $("targetPath").onchange = () => {
    hide("targetHint");
    setWorkspaceTag($("targetPath").value.trim());
    resetPinnedContext();
  };
  $("targetPath").oninput  = () => { if ($("targetPath").value.trim()) { hide("targetHint"); setWorkspaceTag($("targetPath").value.trim()); } };

  // Explorer
  $("openPathBtn").onclick = () => openPath($("filePath").value);
  $("pinFileBtn").onclick = toggleCurrentFilePin;
  $("saveFileBtn").onclick = saveFile;

  // Search
  $("searchBtn").onclick  = () => searchFiles(false);
  $("searchMoreBtn").onclick = () => searchFiles(true);
  $("searchTerm").addEventListener("keydown", (e) => {
    if (e.key === "Enter") searchFiles(false);
  });

  // Terminal (under chat)
  $("termToggleBtn").onclick = toggleTerm;
  $("termRunBtn").onclick    = runTermCommand;
  $("termCommand").addEventListener("keydown", (e) => {
    if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
      e.preventDefault();
      runTermCommand();
    }
  });

  // Snapshots / Timeline
  $("refreshSnapsBtn").onclick    = () => loadSnaps(false);
  $("cleanSnapsBtn").onclick      = cleanOldSnaps;
  $("refreshTimelineBtn").onclick = () => loadTimeline(false);
  $("loadMoreChatsBtn").onclick = () => loadChats(true);
  $("loadMoreRunsBtn").onclick = () => loadRuns(true);
  $("loadMoreSnapsBtn").onclick = () => loadSnaps(true);
  $("loadMoreTimelineBtn").onclick = () => loadTimeline(true);

  // AI Plan (right panel)
  $("generatePlanBtn").onclick = generatePlan;
  $("approvePlanBtn").onclick  = implementPlan;
  $("editPlanBtn").onclick = () => setPlanEditing(true);
  $("savePlanBtn").onclick = savePlanChanges;
  $("cancelPlanEditBtn").onclick = () => {
    planEditing = false;
    if (currentRunData) renderCurrentRun(currentRunData);
  };
  $("redoPlanBtn").onclick = redoPlan;
  $("rejectPlanBtn").onclick = rejectPlan;
  $("cancelRunBtn").onclick = () => runAction("cancel");
  $("resumeRunBtn").onclick = () => runAction("resume");
  $("rollbackRunBtn").onclick = () => {
    if (confirm("Restore acceptance snapshot and replace current workspace state?")) runAction("rollback");
  };
  $("cloneRunBtn").onclick = () => {
    if (!currentRunData) return;
    $("planTask").value = currentRunData.task || "";
    $("planPath").value = currentRunData.target_path || "";
    if (currentRunData.brain_provider) $("planProvider").value = currentRunData.brain_provider;
    hide("planResultArea");
    hide("runProgress");
    planEditing = false;
    currentRun = null;
    $("planTask").focus();
  };

  $("newRunBtn").onclick = () => {
    currentRun = null;
    planEditing = false;
    $("planTask").value = "";
    $("planEditorContent").value = "";
    hide("planResultArea");
    hide("runProgress");
    document.querySelector(".panel-area").classList.add("open");
    syncDrawerState();
    $("planTask").focus();
  };
  $("refreshRunsBtn").onclick = () => loadRuns(false);

  $("saveCodexBtn").onclick = () => saveBrain("codex");
  $("testCodexBtn").onclick = () => testBrain("codex");
  $("disconnectCodexBtn").onclick = () => disconnectBrain("codex");
  $("saveClaudeBtn").onclick = () => saveBrain("claude");
  $("testClaudeBtn").onclick = () => testBrain("claude");
  $("disconnectClaudeBtn").onclick = () => disconnectBrain("claude");
  $("discoverAgentsBtn").onclick = discoverAgentModels;

  // Activity bar — switches sidebar pane (plan button toggles the plan panel on mobile)
  $("activityChat").onclick     = () => switchSidePane("chat");
  $("activityRuns").onclick     = () => { switchSidePane("runs"); loadRuns(); };
  $("activityFiles").onclick    = () => switchSidePane("files");
  $("activityTools").onclick    = () => switchSidePane("tools");
  $("activitySnaps").onclick    = () => { switchSidePane("snaps");    loadSnaps();    };
  $("activityTimeline").onclick = () => { switchSidePane("timeline"); loadTimeline(); };
  $("activityPlan").onclick     = () => { document.querySelector(".panel-area").classList.toggle("open"); syncDrawerState(); };
  $("activitySettings").onclick = () => { switchSidePane("settings"); loadBrains(); loadAgents(); };
  $("activityMore").onclick = () => switchSidePane("more");
  $("moreToolsBtn").onclick = () => switchSidePane("tools");
  $("moreSnapsBtn").onclick = () => { switchSidePane("snaps"); loadSnaps(); };
  $("moreTimelineBtn").onclick = () => { switchSidePane("timeline"); loadTimeline(); };
  $("moreSettingsBtn").onclick = () => { switchSidePane("settings"); loadBrains(); loadAgents(); };

  // Editor tabs
  document.querySelectorAll(".editor-tab").forEach((tab) => {
    tab.onclick = () => switchEditor(tab.dataset.editor);
  });

  // Status bar toggles
  $("menuBtn").onclick      = () => { document.querySelector(".sidebar").classList.toggle("open"); syncDrawerState(); };
  $("workspaceBtn").onclick = () => { document.querySelector(".panel-area").classList.toggle("open"); syncDrawerState(); };
  $("drawerBackdrop").onclick = closeDrawers;
  $("closeSidebarBtn").onclick = closeDrawers;
  $("closePlanBtn").onclick = closeDrawers;

  // Keyboard shortcuts
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") {
      closeDrawers();
    }
    if (e.key === "Tab" && window.matchMedia("(max-width: 767px)").matches) {
      const sheet = document.querySelector(".panel-area.open") || document.querySelector(".sidebar.open");
      if (!sheet) return;
      const focusable = [...sheet.querySelectorAll("button:not([disabled]), input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex='-1'])")]
        .filter((element) => element.offsetParent !== null);
      if (!focusable.length) return;
      const first = focusable[0];
      const last = focusable[focusable.length - 1];
      if (e.shiftKey && document.activeElement === first) { e.preventDefault(); last.focus(); }
      else if (!e.shiftKey && document.activeElement === last) { e.preventDefault(); first.focus(); }
    }
  });

  window.addEventListener("resize", syncDrawerState);

  $("prompt").addEventListener("keydown", (e) => {
    if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
      e.preventDefault();
      $("chatForm").requestSubmit();
    }
  });

  boot().catch(() => {
    hide("app");
    show("login");
  });
});
