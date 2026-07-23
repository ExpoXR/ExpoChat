import { createApi } from "/js/api.mjs";
import { buildChatPayload, chatAgentStep, chatEventStatus, cloudEngineValue } from "/js/chat.mjs";
import { escapeHtml, formatBytes, formatLocalDateTime, formatLocalTime, renderMarkdown } from "/js/render.mjs";
import { artifactPresentation, explorerActivityState, fileActivity, runChoreography, runStatusLabel, subtaskCards, tokenCounts } from "/js/runs.mjs";
import { modelLabel, modelOptions, providerOptions } from "/js/settings.mjs";
import { buildSettingsPayload, usageMeter } from "/js/settings_api.mjs";
import { consumeSse } from "/js/sse.mjs";
import { readPreferences, writePreferences } from "/js/state.mjs";
import { splitCommand, updatePinnedPaths } from "/js/workspace.mjs";
import { baseName, duplicateName, joinPath, parentDir, pasteTarget } from "/js/fsops.mjs";

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

const $ = (id) => document.getElementById(id);

let models = [];
let chats = [];
let currentChat = null;
let currentFile = null;
let pinnedContextPaths = [];
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
let clipboard = null; // { mode: "copy" | "cut", path }

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

// Floating kebab / context menu. items: [{ label, handler, danger?, disabled? }]
function closeContextMenu() {
  const menu = $("contextMenu");
  menu.classList.add("hidden");
  menu.innerHTML = "";
}

function openContextMenu(anchorEl, items) {
  const menu = $("contextMenu");
  menu.innerHTML = "";
  items.filter(Boolean).forEach((item) => {
    const btn = document.createElement("button");
    btn.className = "context-item" + (item.danger ? " danger" : "");
    btn.textContent = item.label;
    btn.disabled = Boolean(item.disabled);
    btn.setAttribute("role", "menuitem");
    btn.onclick = (event) => {
      event.stopPropagation();
      closeContextMenu();
      item.handler();
    };
    menu.appendChild(btn);
  });
  menu.classList.remove("hidden");
  const rect = anchorEl.getBoundingClientRect();
  const width = menu.offsetWidth || 180;
  const left = Math.max(8, Math.min(rect.right - width, window.innerWidth - width - 8));
  const top = Math.min(rect.bottom + 4, window.innerHeight - menu.offsetHeight - 8);
  menu.style.left = `${left}px`;
  menu.style.top = `${Math.max(8, top)}px`;
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
  ["chatEditor", "runEditor", "fileEditor", "diffEditor", "artifactEditor"].forEach((pane) =>
    $(pane).classList.toggle("hidden", pane !== id)
  );
  document.querySelectorAll(".editor-tab").forEach((tab) =>
    tab.classList.toggle("active", tab.dataset.editor === id)
  );
}

// ---------------------------------------------------------------------------
// Message rendering
// ---------------------------------------------------------------------------

function addMessageNode(role, labelText) {
  // Remove stream cursor if it was floating
  const cursor = $("streamCursor");
  if (cursor && cursor.parentElement === $("messages")) {
    $("messages").removeChild(cursor);
  }

  const node = document.createElement("article");
  node.className = `msg ${role}`;
  const label = document.createElement("div");
  label.className = "role";
  label.textContent = labelText || (role === "assistant" ? "assistant" : role);
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

// Render/update an inline agent choreography card inside an assistant message.
function renderAgentStep(node, step) {
  if (!step) return;
  let strip = node.querySelector(".agent-strip");
  if (!strip) {
    strip = document.createElement("div");
    strip.className = "agent-strip";
    node.querySelector(".msg-body").before(strip);
  }
  let card = strip.querySelector(`[data-actor="${step.actor}"]`);
  if (!card) {
    card = document.createElement("div");
    card.className = "agent-card";
    card.dataset.actor = step.actor;
    strip.appendChild(card);
  }
  const active = step.state === "planning" || step.state === "working";
  card.classList.toggle("active", active);
  card.classList.toggle("done", step.state === "done");
  const planHtml = step.text ? `<details class="agent-plan"><summary>Plan</summary><div>${renderMarkdown(step.text)}</div></details>` : "";
  card.innerHTML =
    `<div class="agent-head"><span class="agent-dot"></span>` +
    `<span class="agent-name">${escapeHtml(step.title)}</span>` +
    `<span class="agent-state">${escapeHtml(step.label)}</span></div>${planHtml}`;
  $("messages").scrollTop = $("messages").scrollHeight;
}

// ---------------------------------------------------------------------------
// Chat — model list + history
// ---------------------------------------------------------------------------

function renderChats() {
  $("chatList").innerHTML = "";
  chats.forEach((chat) => {
    const row = document.createElement("div");
    row.className = "tree-row";
    const btn = document.createElement("button");
    btn.className = "item" + (chat.id === currentChat ? " active" : "");
    btn.textContent = (chat.pinned ? "📌 " : "") + (chat.title || "New chat");
    btn.onclick = () => loadChat(chat.id);
    const kebab = document.createElement("button");
    kebab.className = "kebab";
    kebab.textContent = "⋯";
    kebab.title = "Chat actions";
    kebab.setAttribute("aria-label", `Actions for ${chat.title || "chat"}`);
    kebab.onclick = (event) => { event.stopPropagation(); openChatMenu(kebab, chat); };
    row.append(btn, kebab);
    $("chatList").appendChild(row);
  });
}

function openChatMenu(anchorEl, chat) {
  openContextMenu(anchorEl, [
    { label: "Rename", handler: () => renameChat(chat) },
    { label: chat.pinned ? "Unpin" : "Pin", handler: () => pinChat(chat) },
    { label: "Duplicate", handler: () => duplicateChat(chat) },
    { label: "Delete", danger: true, handler: () => deleteChat(chat) },
  ]);
}

async function renameChat(chat) {
  const title = (prompt("Rename chat:", chat.title || "") || "").trim();
  if (!title || title === chat.title) return;
  try {
    await api(`/api/chats/${chat.id}`, { method: "PATCH", body: JSON.stringify({ title }) });
    await loadChats();
  } catch (err) { showToast(err.message, "error"); }
}

async function pinChat(chat) {
  try {
    await api(`/api/chats/${chat.id}`, { method: "PATCH", body: JSON.stringify({ pinned: !chat.pinned }) });
    await loadChats();
  } catch (err) { showToast(err.message, "error"); }
}

async function duplicateChat(chat) {
  try {
    const data = await api(`/api/chats/${chat.id}/duplicate`, { method: "POST", body: "{}" });
    await loadChats();
    if (data.chat) await loadChat(data.chat.id);
  } catch (err) { showToast(err.message, "error"); }
}

async function deleteChat(chat) {
  if (!confirm(`Delete chat "${chat.title || "New chat"}" and all its messages?`)) return;
  try {
    await api(`/api/chats/${chat.id}`, { method: "DELETE" });
    if (currentChat === chat.id) {
      currentChat = null;
      $("messages").innerHTML = '<span id="streamCursor" class="stream-cursor hidden">▋</span>';
    }
    await loadChats();
  } catch (err) { showToast(err.message, "error"); }
}

async function loadModels() {
  const data = await api("/api/models");
  models = data.models || [];
  renderChatEngines();
}

// Rebuild the chat engine picker: local Ollama models + linked cloud brains.
function renderChatEngines() {
  const sel = $("modelSelect");
  const prev = sel.value || loadPrefs().model;
  sel.innerHTML = "";
  const localGroup = document.createElement("optgroup");
  localGroup.label = "Local (Ollama)";
  models.forEach((m) => {
    const opt = document.createElement("option");
    opt.value = m.name || m.model;
    opt.textContent = m.name || m.model;
    localGroup.appendChild(opt);
  });
  sel.appendChild(localGroup);

  const cloud = providerOptions(brains || []);
  if (cloud.length) {
    const cloudGroup = document.createElement("optgroup");
    cloudGroup.label = "Cloud (API)";
    cloud.forEach((provider) => {
      const opt = document.createElement("option");
      opt.value = cloudEngineValue(provider.value);
      opt.textContent = provider.label;
      cloudGroup.appendChild(opt);
    });
    sel.appendChild(cloudGroup);
  }
  if (prev && [...sel.options].some((opt) => opt.value === prev)) sel.value = prev;
  $("ollamaStatus").textContent = (sel.value || "no model").replace(/^cloud:/, "");
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

  const engineLabel = $("agentModeToggle").checked
    ? "agents"
    : ($("modelSelect").selectedOptions[0]?.textContent || "assistant");
  const node = addMessageNode("assistant", engineLabel);
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
        { agentMode: $("agentModeToggle").checked, brainProvider: $("chatBrainSelect").value },
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
        if (frame.event === "agent") {
          const step = chatAgentStep(parsed);
          if (step) {
            renderAgentStep(node, step);
            setStatus(`${step.title} — ${step.label}`);
          }
          continue;
        }
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
    const row = document.createElement("div");
    row.className = "tree-row";
    const btn = document.createElement("button");
    btn.className = "item";
    btn.innerHTML = `<span class="file-icon">${item.is_dir ? "📁" : "📄"}</span>${escapeHtml(item.name)}`;
    btn.dataset.path = item.path;
    btn.dataset.directory = String(Boolean(item.is_dir));
    btn.onclick = () => (item.is_dir ? openPath(item.path) : openFile(item.path));
    const kebab = document.createElement("button");
    kebab.className = "kebab";
    kebab.textContent = "⋯";
    kebab.title = "Actions";
    kebab.setAttribute("aria-label", `Actions for ${item.name}`);
    kebab.onclick = (event) => { event.stopPropagation(); openEntryMenu(kebab, item.path, Boolean(item.is_dir)); };
    row.append(btn, kebab);
    $("fileList").appendChild(row);
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
// File / folder operations (Explorer)
// ---------------------------------------------------------------------------

function currentExplorerDir() {
  return $("filePath").value.trim();
}

async function fsRefresh() {
  await openPath(currentExplorerDir());
}

function syncPasteButton() {
  $("pasteHereBtn").classList.toggle("hidden", !clipboard);
}

async function fsCreateFolder(dir) {
  if (!dir) { showToast("Open a folder first.", "error"); return; }
  const name = (prompt("New folder name:") || "").trim();
  if (!name) return;
  try {
    await api("/api/fs/folder", { method: "POST", body: JSON.stringify({ path: joinPath(dir, name) }) });
    showToast("Folder created.", "success");
    await fsRefresh();
  } catch (err) { showToast(err.message, "error"); }
}

async function fsCreateFile(dir) {
  if (!dir) { showToast("Open a folder first.", "error"); return; }
  const name = (prompt("New file name:") || "").trim();
  if (!name) return;
  try {
    await api("/api/file", { method: "PUT", body: JSON.stringify({ path: joinPath(dir, name), content: "", chat_id: currentChat }) });
    showToast("File created.", "success");
    await fsRefresh();
    await openFile(joinPath(dir, name));
  } catch (err) { showToast(err.message, "error"); }
}

async function fsRename(path) {
  const next = (prompt("Rename to:", baseName(path)) || "").trim();
  if (!next || next === baseName(path)) return;
  try {
    await api("/api/fs/rename", { method: "POST", body: JSON.stringify({ path, new_path: joinPath(parentDir(path), next) }) });
    showToast("Renamed.", "success");
    await fsRefresh();
  } catch (err) { showToast(err.message, "error"); }
}

async function fsDelete(path, isDir) {
  if (!confirm(`Permanently delete ${isDir ? "folder" : "file"} "${baseName(path)}"? A snapshot is saved first for rollback.`)) return;
  try {
    await api("/api/fs/delete", { method: "POST", body: JSON.stringify({ path }) });
    showToast("Deleted (snapshot saved).", "success");
    if (currentFile === path) { currentFile = null; syncPinnedContext(); }
    await fsRefresh();
  } catch (err) { showToast(err.message, "error"); }
}

async function fsDuplicate(path) {
  try {
    await api("/api/fs/copy", { method: "POST", body: JSON.stringify({ src: path, dest: duplicateName(path) }) });
    showToast("Duplicated.", "success");
    await fsRefresh();
  } catch (err) { showToast(err.message, "error"); }
}

function fsSetClipboard(mode, path) {
  clipboard = { mode, path };
  syncPasteButton();
  showToast(`${mode === "cut" ? "Cut" : "Copied"}: ${baseName(path)}`, "info");
}

async function fsPasteInto(dir) {
  if (!clipboard || !dir) return;
  const dest = pasteTarget(dir, clipboard.path);
  try {
    if (clipboard.mode === "cut") {
      await api("/api/fs/rename", { method: "POST", body: JSON.stringify({ path: clipboard.path, new_path: dest }) });
      clipboard = null;
      syncPasteButton();
    } else {
      await api("/api/fs/copy", { method: "POST", body: JSON.stringify({ src: clipboard.path, dest }) });
    }
    showToast("Pasted.", "success");
    await fsRefresh();
  } catch (err) { showToast(err.message, "error"); }
}

function openEntryMenu(anchorEl, path, isDir) {
  openContextMenu(anchorEl, [
    { label: isDir ? "Open" : "Open file", handler: () => (isDir ? openPath(path) : openFile(path)) },
    { label: "Rename", handler: () => fsRename(path) },
    { label: "Copy", handler: () => fsSetClipboard("copy", path) },
    { label: "Cut", handler: () => fsSetClipboard("cut", path) },
    isDir && clipboard ? { label: `Paste into “${baseName(path)}”`, handler: () => fsPasteInto(path) } : null,
    { label: "Duplicate", handler: () => fsDuplicate(path) },
    { label: "Delete", danger: true, handler: () => fsDelete(path, isDir) },
  ]);
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
  switchEditor("runEditor");
  $("planStatus").textContent = "Creating staged research run…";
  hide("planResultArea");
  hide("planActions");
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
    hide("planActions");
    renderCurrentRun(data);
    subscribeRun(currentRun);
    await loadRuns();
    setStatus("Implementation running");
    switchEditor("runEditor");
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
    switchEditor("runEditor");
    hide("planResultArea");
    hide("planActions");
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
  hide("runEmptyHint");
  const queue = run.queue_position ? ` · queue #${run.queue_position}` : "";
  const chosen = (run.selected_agents || []).map((agent) => agent.name).join(" → ");
  $("currentRunBadge").textContent = `${run.id.slice(0, 8)} · ${runStatusLabel(run)}${queue}${chosen ? ` · ${chosen}` : ""}`;
  show("currentRunBadge");
  show("runProgress");
  show("runActions");
  $("planStatus").textContent = run.error ? `${runStatusLabel(run)} — ${run.error}` : runStatusLabel(run);
  const awaitingApproval = run.status === "awaiting_approval";
  if (!awaitingApproval) planEditing = false;
  const visiblePlan = awaitingApproval
    ? (run.draft_plan || run.approved_plan)
    : (run.approved_plan || run.draft_plan);
  $("planActions").classList.toggle("hidden", !(visiblePlan && awaitingApproval));
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

function renderRunChoreography(run, events) {
  const lane = $("runChoreography");
  if (!lane) return;
  const steps = runChoreography(run || {}, events || []);
  lane.innerHTML = steps.map((step, index) => {
    const cls = { working: "active", done: "done", error: "error" }[step.state] || "";
    const arrow = index < steps.length - 1 ? '<span class="agent-arrow">→</span>' : "";
    return (
      `<div class="agent-card ${cls}" data-actor="${step.role}">` +
      `<div class="agent-head"><span class="agent-dot"></span>` +
      `<span class="agent-name">${escapeHtml(step.title)}</span>` +
      `<span class="agent-state">${escapeHtml(step.label)}</span></div></div>${arrow}`
    );
  }).join("");
}

function renderSubtasks(run) {
  const section = $("subtaskSection");
  const container = $("subtaskCards");
  if (!section || !container) return;
  const subs = run?.subtasks || [];
  if (!subs.length) { section.classList.add("hidden"); return; }
  section.classList.remove("hidden");
  const cards = subtaskCards(subs);
  container.innerHTML = cards.map((card) => {
    const depLabel = card.deps.length ? `after ${card.deps.join(", ")}` : "";
    return (
      `<div class="subtask-card" data-status="${escapeHtml(card.status)}" data-node="${escapeHtml(card.node_id)}">` +
      `<div class="subtask-head"><span class="subtask-dot"></span>` +
      `<span class="subtask-title" title="${escapeHtml(card.title)}">${escapeHtml(card.title)}</span></div>` +
      `<div class="subtask-meta">` +
      `<span class="subtask-role">${escapeHtml(card.role)}</span>` +
      (card.agent_name ? `<span>${escapeHtml(card.agent_name)}</span>` : "") +
      `<span>${escapeHtml(card.status)}</span>` +
      (depLabel ? `<span class="subtask-deps">${escapeHtml(depLabel)}</span>` : "") +
      `</div></div>`
    );
  }).join("");
}

function renderRunEvents(events) {
  renderRunChoreography(currentRunData, events);
  renderSubtasks(currentRunData);
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
      switchEditor("runEditor");
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
  ["run.created", "research.started", "research.completed", "plan.ready", "plan.edited", "plan.redo", "plan.approved", "plan.decomposed", "scope.approved", "scope.approval_required", "implementation.started", "agent.activity", "subtask.started", "subtask.completed", "subtasks.merged", "subtasks.conflict", "verification.completed", "apply.completed", "rollback.completed", "run.completed", "run.failed", "run.cancelled", "plan.stale"].forEach((name) => {
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
    const prefix = brain.provider;
    if (!$(`${prefix}Model`)) return; // provider has no card in the current view
    populateModelSelect(brain.provider, brain.model);
    $(`${prefix}Model`).value = brain.model;
    const linked = Boolean(brain.enabled && brain.linked && !brain.last_error);
    $(`${prefix}State`).textContent = linked
      ? `Linked via ${brain.source}${brain.validated_at ? " · validated" : ""}${brain.last_error ? " · " + brain.last_error : ""}`
      : brain.last_error || "Not linked";
    $(`${prefix}State`).classList.toggle("linked", linked);
    const indicator = $(BRAIN_INDICATORS[brain.provider]);
    if (indicator) {
      indicator.textContent = `${modelLabel(brain.provider, brain.model)} ${linked ? "✓" : "!"}`;
      indicator.classList.toggle("linked", linked);
      indicator.classList.toggle("hidden", !brain.enabled && !brain.linked);
    }
  });
  syncProviderOptions();
}

const BRAIN_INDICATORS = { codex: "openaiIndicator", claude: "claudeIndicator", gemini: "geminiIndicator" };

function populateModelSelect(provider, current = "") {
  const select = $(`${provider}Model`);
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

  // Reflect linked providers into the chat engine picker + Agent Mode brain select.
  renderChatEngines();
  const brainSelect = $("chatBrainSelect");
  if (brainSelect) {
    const prev = brainSelect.value;
    brainSelect.innerHTML = "";
    enabled.forEach((provider) => {
      const option = document.createElement("option");
      option.value = provider.value;
      option.textContent = provider.label;
      brainSelect.appendChild(option);
    });
    if (enabled.some((provider) => provider.value === prev)) brainSelect.value = prev;
  }
}

async function saveBrain(provider) {
  const prefix = provider;
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
  const prefix = provider;
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

let appSettings = { theme: "dark", agent_mode_default: false };
let agentModeTouched = false;

function applyTheme(theme) {
  const resolved = theme === "auto"
    ? (window.matchMedia && window.matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark")
    : theme;
  document.documentElement.setAttribute("data-theme", resolved);
}

async function loadSettings() {
  try {
    appSettings = await api("/api/settings");
    $("budgetDaily").value = appSettings.token_budget_daily || 0;
    $("budgetRun").value = appSettings.token_budget_run || 0;
    $("maxOutputTokens").value = appSettings.max_output_tokens || 0;
    $("themeSelect").value = appSettings.theme || "dark";
    $("agentModeDefault").checked = Boolean(appSettings.agent_mode_default);
    applyTheme(appSettings.theme || "dark");
    if (!agentModeTouched) {
      $("agentModeToggle").checked = Boolean(appSettings.agent_mode_default);
      $("chatBrainSelect").classList.toggle("hidden", !$("agentModeToggle").checked);
    }
  } catch (_) {}
  await loadUsage();
}

async function loadUsage() {
  try {
    const usage = await api("/api/usage");
    const meter = usageMeter(usage);
    $("usageMeterText").textContent = meter.label;
    $("usageMeterFill").style.width = meter.unlimited ? "0%" : `${meter.pct}%`;
    $("usageMeterFill").classList.toggle("over", meter.over);
    const parts = (usage.by_provider || [])
      .filter((row) => row.total > 0)
      .map((row) => `${row.provider}: ${Number(row.total).toLocaleString()}`);
    $("usageMeterBreakdown").textContent = parts.length ? parts.join(" · ") : "No usage recorded today.";
  } catch (_) {}
}

async function saveSettings(values, message) {
  try {
    appSettings = await api("/api/settings", { method: "PUT", body: JSON.stringify(buildSettingsPayload(values)) });
    applyTheme(appSettings.theme || "dark");
    await loadUsage();
    showToast(message, "success");
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
    allowedRoots  = cfg.allowed_roots || [];

    // Populate plan provider selector based on what's enabled (loadBrains re-syncs later)
    const providerList = [
      { key: "codex", enabled: cfg.openai_enabled, label: "Codex (OpenAI)", indicator: "openaiIndicator" },
      { key: "claude", enabled: cfg.claude_enabled, label: "Claude (Anthropic)", indicator: "claudeIndicator" },
      { key: "gemini", enabled: cfg.gemini_enabled, label: "Gemini (Google)", indicator: "geminiIndicator" },
    ];
    const provSel = $("planProvider");
    provSel.innerHTML = "";
    let anyProvider = false;
    providerList.forEach((provider) => {
      if (!provider.enabled) return;
      anyProvider = true;
      show(provider.indicator);
      const option = document.createElement("option");
      option.value = provider.key;
      option.textContent = provider.label;
      provSel.appendChild(option);
    });
    if (!anyProvider) {
      const option = document.createElement("option");
      option.value = ""; option.textContent = "No AI provider configured";
      provSel.appendChild(option);
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
  await Promise.allSettled([loadRuns(), loadBrains(), loadAgents(), loadSettings()]);

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
  $("agentModeToggle").onchange = () => {
    agentModeTouched = true;
    $("chatBrainSelect").classList.toggle("hidden", !$("agentModeToggle").checked);
  };
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
  $("newFileBtn").onclick = () => fsCreateFile(currentExplorerDir());
  $("newFolderBtn").onclick = () => fsCreateFolder(currentExplorerDir());
  $("pasteHereBtn").onclick = () => fsPasteInto(currentExplorerDir());

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
    hide("planActions");
    hide("runProgress");
    hide("runActions");
    hide("currentRunBadge");
    show("runEmptyHint");
    planEditing = false;
    currentRun = null;
    currentRunData = null;
    $("planTask").focus();
  };

  $("newRunBtn").onclick = () => {
    currentRun = null;
    currentRunData = null;
    planEditing = false;
    $("planTask").value = "";
    $("planEditorContent").value = "";
    $("planStatus").textContent = "";
    hide("planResultArea");
    hide("planActions");
    hide("runProgress");
    hide("runActions");
    hide("currentRunBadge");
    show("runEmptyHint");
    switchEditor("runEditor");
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
  $("saveGeminiBtn").onclick = () => saveBrain("gemini");
  $("testGeminiBtn").onclick = () => testBrain("gemini");
  $("disconnectGeminiBtn").onclick = () => disconnectBrain("gemini");
  $("saveBudgetBtn").onclick = () => saveSettings({
    token_budget_daily: $("budgetDaily").value,
    token_budget_run: $("budgetRun").value,
    max_output_tokens: $("maxOutputTokens").value,
  }, "Limits saved");
  $("refreshUsageBtn").onclick = () => loadUsage();
  $("saveAppearanceBtn").onclick = () => saveSettings({
    theme: $("themeSelect").value,
    agent_mode_default: $("agentModeDefault").checked,
  }, "Appearance saved");
  $("discoverAgentsBtn").onclick = discoverAgentModels;

  // Activity bar — switches sidebar pane (plan button toggles the plan panel on mobile)
  $("activityChat").onclick     = () => switchSidePane("chat");
  $("activityRuns").onclick     = () => { switchSidePane("runs"); loadRuns(); };
  $("activityFiles").onclick    = () => switchSidePane("files");
  $("activityTools").onclick    = () => switchSidePane("tools");
  $("activitySnaps").onclick    = () => { switchSidePane("snaps");    loadSnaps();    };
  $("activityTimeline").onclick = () => { switchSidePane("timeline"); loadTimeline(); };
  $("activityPlan").onclick     = () => { document.querySelector(".panel-area").classList.toggle("open"); syncDrawerState(); };
  $("activitySettings").onclick = () => { switchSidePane("settings"); loadBrains(); loadAgents(); loadSettings(); };
  $("activityMore").onclick = () => switchSidePane("more");
  $("moreToolsBtn").onclick = () => switchSidePane("tools");
  $("moreSnapsBtn").onclick = () => { switchSidePane("snaps"); loadSnaps(); };
  $("moreTimelineBtn").onclick = () => { switchSidePane("timeline"); loadTimeline(); };
  $("moreSettingsBtn").onclick = () => { switchSidePane("settings"); loadBrains(); loadAgents(); loadSettings(); };

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
  document.addEventListener("click", (e) => {
    if (!e.target.closest("#contextMenu") && !e.target.closest(".kebab")) closeContextMenu();
  });

  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") {
      closeContextMenu();
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

  window.addEventListener("resize", () => { syncDrawerState(); closeContextMenu(); });

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
