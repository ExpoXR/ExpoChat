import { createApi } from "/js/api.mjs";
import { buildChatPayload, chatAgentStep, chatEventStatus, cloudEngineValue } from "/js/chat.mjs";
import { escapeHtml, formatBytes, formatLocalDateTime, formatLocalTime, renderMarkdown } from "/js/render.mjs";
import { artifactPresentation, explorerActivityState, fileActivity, runChoreography, runStatusLabel, subtaskCards, tokenCounts } from "/js/runs.mjs";
import { assignableAgents, buildGraphModel, computeLayout, edgeList, wouldCycle } from "/js/graph.mjs";
import { modelLabel, modelOptions, providerOptions } from "/js/settings.mjs";
import { buildSettingsPayload, clampPriority, hostStatusLabel, normalizeHostUrl, usageMeter } from "/js/settings_api.mjs";
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
let csrfToken = "";
let runs = [];
let currentRun = null;
let currentRunData = null;
let runEventSource = null;
let runStreamGen = 0;       // guards against stale overlapping loadRun() responses
let runRefreshTimer = null; // coalesces bursts of run events into one refetch
let runStreamErrors = 0;    // reconnect-failure cap for the run SSE stream
let brains = [];
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
let agentsCache = null; // [{id,name,roles,enabled,...}] for the task-graph LLM pickers
let ollamaOnline = null;
let ollamaPollTimer = null;

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

function renderOllamaStatus() {
  if (ollamaOnline === false) {
    $("ollamaStatus").textContent = "Offline";
    return;
  }
  const value = $("chatModelSelect")?.value || "";
  $("ollamaStatus").textContent = (value || (ollamaOnline ? "Online" : "connecting…")).replace(/^cloud:/, "");
}

async function refreshOllamaState() {
  try {
    const data = await api("/api/status");
    ollamaOnline = Boolean(data.ollama_available);
  } catch (_) {
    ollamaOnline = false;
  }
  renderOllamaStatus();
  const current = $("status").textContent;
  if (/^Ready(?: · Ollama offline)?$/.test(current)) {
    setStatus(ollamaOnline ? "Ready" : "Ready · Ollama offline");
  }
}

function setBusy(ids, busy) {
  ids.forEach((id) => { const el = $(id); if (el) el.disabled = busy; })
}

// One chat-context chip: workspace folder + pinned-file count in a single indicator.
function renderContextChip() {
  const tag = $("workspaceTag");
  if (!tag) return;
  const path = $("targetPath").value.trim();
  const folder = path ? (path.split("/").filter(Boolean).pop() || path) : "";
  const pins = pinnedContextPaths.length;
  if (!folder && !pins) { tag.classList.add("hidden"); return; }
  const parts = [];
  if (folder) parts.push(`📂 ${folder}`);
  if (pins) parts.push(`📌 ${pins}`);
  tag.textContent = parts.join(" · ");
  tag.title = (path ? `Workspace: ${path}` : "")
    + (pins ? `${path ? "\n" : ""}Pinned files:\n${pinnedContextPaths.join("\n")}` : "");
  tag.classList.remove("hidden");
}

function setWorkspaceTag() {
  renderContextChip();
}

function syncPinnedContext() {
  const button = $("pinFileBtn");
  const currentPinned = Boolean(currentFile && pinnedContextPaths.includes(currentFile));
  button.disabled = !currentFile;
  button.textContent = currentPinned ? "Unpin from Chat" : "Pin to Chat";
  renderContextChip();
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
  // Errors are announced assertively and linger longer so actionable failures aren't missed.
  const assertive = type === "error";
  t.setAttribute("role", assertive ? "alert" : "status");
  t.setAttribute("aria-live", assertive ? "assertive" : "polite");
  clearTimeout(t._tid);
  t._tid = setTimeout(() => t.classList.add("hidden"), assertive ? 6000 : 3500);
}

// Render an inline error placeholder into a list pane so a failed fetch leaves a
// visible, retryable message instead of a silently blank/stale container.
function showPaneError(containerId, message) {
  const container = $(containerId);
  if (!container) return;
  container.innerHTML = `<div class="pane-error" role="alert">${escapeHtml(message)}</div>`;
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

  // Keyboard access: move focus into the menu and support arrow/Home/End/Escape.
  const options = () => [...menu.querySelectorAll("button:not([disabled])")];
  const focusAt = (list, index) => { if (list.length) list[(index + list.length) % list.length].focus(); };
  const first = options();
  if (first.length) first[0].focus();
  menu.onkeydown = (event) => {
    const list = options();
    const idx = list.indexOf(document.activeElement);
    if (event.key === "ArrowDown") { event.preventDefault(); focusAt(list, idx + 1); }
    else if (event.key === "ArrowUp") { event.preventDefault(); focusAt(list, idx - 1); }
    else if (event.key === "Home") { event.preventDefault(); focusAt(list, 0); }
    else if (event.key === "End") { event.preventDefault(); focusAt(list, list.length - 1); }
    else if (event.key === "Escape") { event.preventDefault(); closeContextMenu(); anchorEl.focus(); }
  };
}

// ---------------------------------------------------------------------------
// Layout switches
// ---------------------------------------------------------------------------

function switchSidePane(name) {
  setSettingsMode(false);
  ["chat", "runs", "files", "tools", "snaps", "timeline", "more"].forEach((pane) => {
    const panEl = $(pane + "Pane");
    if (panEl) panEl.classList.toggle("hidden", pane !== name);
    const btn = $("activity" + pane[0].toUpperCase() + pane.slice(1));
    if (btn) {
      btn.classList.toggle("active", pane === name);
      btn.setAttribute("aria-current", pane === name ? "page" : "false");
    }
  });
  drawerReturnFocus = document.activeElement;
  const sidebar = document.querySelector(".sidebar");
  sidebar.classList.add("open");
  syncDrawerState();
  focusSheetInto(sidebar);
}

function setSettingsMode(active) {
  const grid = document.querySelector(".main-grid");
  const wasActive = grid.classList.contains("settings-active");
  grid.classList.toggle("settings-active", active);
  $("activitySettings").classList.toggle("active", active);
  $("activitySettings").setAttribute("aria-current", active ? "page" : "false");
  // Returning from settings restores the editor you were in, not always chat.
  if (!active && wasActive && !$("settingsEditor").classList.contains("hidden")) switchEditor(lastContentEditor || "chatEditor");
}

async function openSettingsPage() {
  setSettingsMode(true);
  switchEditor("settingsEditor");
  closeDrawers();
  await Promise.allSettled([loadBrains(), loadHosts(), loadAgents(), loadSettings(), loadSettingsStorage()]);
}

function syncDrawerState() {
  const mobile = window.matchMedia("(max-width: 767px)").matches;
  const desktop = window.matchMedia("(min-width: 1181px)").matches;
  const drawerOpen = document.querySelector(".panel-area").classList.contains("open");
  const open = document.querySelector(".sidebar").classList.contains("open") || drawerOpen;
  $("drawerBackdrop").classList.toggle("hidden", !(mobile && open));
  // On desktop the panel is a column toggled by .panel-collapsed; below that it's a drawer.
  const panelVisible = desktop
    ? !document.querySelector(".main-grid").classList.contains("panel-collapsed")
    : drawerOpen;
  $("activityPlan").setAttribute("aria-expanded", String(panelVisible));
}

// The Supervisor Run panel: a collapsible column on desktop, a drawer on smaller widths.
function toggleSupervisorPanel() {
  const panel = document.querySelector(".panel-area");
  if (window.matchMedia("(min-width: 1181px)").matches) {
    document.querySelector(".main-grid").classList.toggle("panel-collapsed");
  } else {
    panel.classList.toggle("open");
    if (panel.classList.contains("open")) focusSheetInto(panel);
  }
  syncDrawerState();
}

// From the mobile "More" menu: close the sidebar drawer and reveal the panel drawer.
function openSupervisorFromMore() {
  const panel = document.querySelector(".panel-area");
  document.querySelector(".sidebar").classList.remove("open");
  panel.classList.add("open");
  syncDrawerState();
  focusSheetInto(panel);
}

// Ensure the panel is showing (used when a plan/run needs it), regardless of layout.
function revealSupervisorPanel() {
  if (window.matchMedia("(min-width: 1181px)").matches) {
    document.querySelector(".main-grid").classList.remove("panel-collapsed");
  } else {
    document.querySelector(".panel-area").classList.add("open");
  }
  syncDrawerState();
}

function closeDrawers() {
  document.querySelector(".sidebar").classList.remove("open");
  document.querySelector(".panel-area").classList.remove("open");
  syncDrawerState();
  if (drawerReturnFocus && drawerReturnFocus.isConnected) drawerReturnFocus.focus();
  drawerReturnFocus = null;
}

// When a drawer/overlay opens (≤1180px), move focus into it so keyboard users land inside.
function focusSheetInto(el) {
  if (!el || !window.matchMedia("(max-width: 1180px)").matches) return;
  const focusable = el.querySelector(
    "button:not([disabled]), input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex='-1'])"
  );
  if (focusable) requestAnimationFrame(() => focusable.focus());
}

const PANEL_LAYOUT_KEY = "ollma_panel_layout";

function clampPanelWidth(value, min, max) {
  return Math.min(max, Math.max(min, Math.round(value)));
}

function setPanelWidth(kind, value, persist = true) {
  const grid = document.querySelector(".main-grid");
  const isSidebar = kind === "sidebar";
  const width = clampPanelWidth(value, isSidebar ? 220 : 260, isSidebar ? 520 : 600);
  grid.style.setProperty(isSidebar ? "--sidebar-width" : "--panel-width", `${width}px`);
  $(isSidebar ? "sidebarResize" : "panelResize").setAttribute("aria-valuenow", String(width));
  if (persist) {
    let layout = {};
    try { layout = JSON.parse(localStorage.getItem(PANEL_LAYOUT_KEY) || "{}"); } catch (_) {}
    layout[kind] = width;
    localStorage.setItem(PANEL_LAYOUT_KEY, JSON.stringify(layout));
  }
}

function setupPanelResizer(id, kind, fallback) {
  const handle = $(id);
  handle.setAttribute("aria-valuemin", kind === "sidebar" ? "220" : "260");
  handle.setAttribute("aria-valuemax", kind === "sidebar" ? "520" : "600");
  let layout = {};
  try { layout = JSON.parse(localStorage.getItem(PANEL_LAYOUT_KEY) || "{}"); } catch (_) {}
  setPanelWidth(kind, Number(layout[kind]) || fallback, false);

  handle.addEventListener("pointerdown", (event) => {
    if (window.matchMedia("(max-width: 767px)").matches) return;
    event.preventDefault();
    handle.classList.add("dragging");
    handle.setPointerCapture(event.pointerId);
  });
  handle.addEventListener("pointermove", (event) => {
    if (!handle.hasPointerCapture(event.pointerId)) return;
    setPanelWidth(kind, kind === "sidebar" ? event.clientX - 48 : window.innerWidth - event.clientX);
  });
  handle.addEventListener("pointerup", (event) => {
    handle.classList.remove("dragging");
    if (handle.hasPointerCapture(event.pointerId)) handle.releasePointerCapture(event.pointerId);
  });
  handle.addEventListener("dblclick", () => setPanelWidth(kind, fallback));
  handle.addEventListener("keydown", (event) => {
    if (!["ArrowLeft", "ArrowRight"].includes(event.key)) return;
    event.preventDefault();
    const current = Number(handle.getAttribute("aria-valuenow")) || fallback;
    const direction = event.key === "ArrowRight" ? 1 : -1;
    setPanelWidth(kind, current + direction * (kind === "sidebar" ? 16 : -16));
  });
}

let lastContentEditor = "chatEditor"; // last non-settings editor, for return-from-settings

function switchEditor(id) {
  // Leaving the run view: stop the run SSE stream so it doesn't keep refetching
  // in the background (it was previously only closed on terminal status).
  if (id !== "runEditor") closeRunStream();
  if (id !== "settingsEditor") lastContentEditor = id;
  ["chatEditor", "runEditor", "fileEditor", "diffEditor", "artifactEditor", "settingsEditor"].forEach((pane) =>
    $(pane).classList.toggle("hidden", pane !== id)
  );
  document.querySelectorAll(".editor-tab").forEach((tab) => {
    const active = tab.dataset.editor === id;
    tab.classList.toggle("active", active);
    tab.setAttribute("aria-selected", active ? "true" : "false");
    // Reveal a tab the first time its pane is shown (Chat stays always-visible); optional
    // tabs (run/file/diff/evidence) start hidden so you can't navigate into a blank pane.
    if (active && id !== "settingsEditor") tab.classList.remove("hidden");
  });
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

// Rebuild the chat engine picker (single control): local Ollama models + linked cloud brains.
function renderChatEngines() {
  const select = $("chatModelSelect");
  if (!select) return;
  const prev = select.value || loadPrefs().model;
  const cloud = providerOptions(brains || []);
  select.innerHTML = "";
  const localGroup = document.createElement("optgroup");
  localGroup.label = "Local (Ollama)";
  models.forEach((model) => {
    const opt = document.createElement("option");
    opt.value = model.name || model.model;
    opt.textContent = model.name || model.model;
    localGroup.appendChild(opt);
  });
  select.appendChild(localGroup);
  if (cloud.length) {
    const cloudGroup = document.createElement("optgroup");
    cloudGroup.label = "Brains (cloud)";
    cloud.forEach((provider) => {
      const opt = document.createElement("option");
      opt.value = cloudEngineValue(provider.value);
      opt.textContent = provider.label;
      cloudGroup.appendChild(opt);
    });
    select.appendChild(cloudGroup);
  }
  if (prev && [...select.options].some((opt) => opt.value === prev)) select.value = prev;
  syncChatModel(select.value || "");
}

function syncChatModel(value) {
  const select = $("chatModelSelect");
  if (select && [...select.options].some((option) => option.value === value)) select.value = value;
  renderOllamaStatus();
  savePrefs();
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
    if (chat.model && [...$("chatModelSelect").options].some((option) => option.value === chat.model)) {
      syncChatModel(chat.model);
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
  const model = $("chatModelSelect").value;
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
  if (!$("chatModelSelect").value) {
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
    : ($("chatModelSelect").selectedOptions[0]?.textContent || "assistant");
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
        $("chatModelSelect").value,
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
    const body = node.querySelector(".msg-body");
    if (!fullText) body.textContent = "Error: " + err.message;
    // Leave an inline retry so a failed send isn't a dead end after the toast fades.
    const retry = document.createElement("button");
    retry.className = "msg-retry link-hint";
    retry.textContent = "Retry";
    retry.onclick = () => {
      node.remove();
      $("prompt").value = content;
      sendPrompt({ preventDefault() {} });
    };
    body.appendChild(retry);
  } finally {
    hide("streamCursor");
    setBusy(["sendBtn", "prompt"], false);
  }
}

// ---------------------------------------------------------------------------
// File explorer
// ---------------------------------------------------------------------------

// Explicitly point the chat (and supervisor) at a folder. This is the ONE place browsing
// turns into "what the model sees", so the pin reset is intentional and announced.
function useFolderForContext(path) {
  const previous = $("targetPath").value.trim();
  $("targetPath").value = path;
  $("planPath").value = path;
  setWorkspaceTag(path);
  hide("targetHint");
  if (previous && previous !== path) {
    resetPinnedContext();
    showToast("Chat context set — pinned files cleared", "success");
  } else {
    showToast(`Chat context set to ${path}`, "success");
  }
  savePrefs();
  openPath(path);
}

async function openPath(path) {
  const data = await api(`/api/files?path=${encodeURIComponent(path || "/")}`);
  if (data.is_dir === false) {
    await openFile(data.path);
    return;
  }
  $("filePath").value = data.path === "/" ? "" : data.path;
  $("fileList").innerHTML = "";

  // Browsing is purely navigational: it no longer silently repoints the chat context or
  // drops pinned files. Setting the chat context is an explicit action below.
  if (data.path && data.path !== "/") {
    const ctx = $("explorerContext");
    ctx.innerHTML = "";
    const isContext = $("targetPath").value.trim() === data.path;
    const label = document.createElement("span");
    label.textContent = isContext ? `📂 Chat context: ${data.path}` : `📁 ${data.path}`;
    ctx.appendChild(label);
    if (!isContext) {
      const useBtn = document.createElement("button");
      useBtn.className = "link-hint link-hint-inline";
      useBtn.textContent = "Use for chat context";
      useBtn.onclick = () => useFolderForContext(data.path);
      ctx.appendChild(useBtn);
    }
    show("explorerContext");
  } else {
    hide("explorerContext");
  }

  if (data.path !== "/") {
    const up = document.createElement("button");
    up.className = "item";
    up.innerHTML = '<span class="file-icon">↑</span>..';
    const atAllowedRoot = (data.roots || []).includes(data.path);
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
    data.path.split("/").pop() || "File";
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
  let data, storage;
  try {
    [data, storage] = await Promise.all([api(`/api/snapshots?cursor=${cursor || 0}&limit=50`), api("/api/maintenance/storage")]);
  } catch (err) {
    if (!append) showPaneError("snapList", `Couldn't load snapshots: ${err.message}`);
    return;
  }
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
  let data;
  try {
    data = await api(`/api/timeline?cursor=${cursor || 0}&limit=50`);
  } catch (err) {
    if (!append) showPaneError("timelineList", `Couldn't load timeline: ${err.message}`);
    return;
  }
  if (!append) $("timelineList").innerHTML = "";
  (data.timeline || []).forEach((event) => {
    const item = document.createElement("button");
    item.className = "item timeline-item";
    const summary = document.createElement("span");
    summary.textContent = `${formatLocalDateTime(event.created_at)}  [${event.event_type}]  ${event.summary.slice(0, 90)}`;
    const path = document.createElement("small");
    path.className = "timeline-path";
    path.textContent = event.path || "No workspace path";
    item.append(summary, path);
    item.title = event.path || event.summary;
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
  // Single workspace folder: planPath and targetPath are bound; filePath is explorer-nav only.
  const path = $("planPath").value.trim() || $("targetPath").value.trim();
  if (path) { $("planPath").value = path; $("targetPath").value = path; }
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
    showToast("Link a Brain in Settings → Brains first.", "error");
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
  if (!confirm("Reject this plan and end the run? The generated plan will be discarded.")) return;
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
  const runDetail = run.error || run.wait_reason || "";
  $("planStatus").textContent = runDetail ? `${runStatusLabel(run)} — ${runDetail}` : runStatusLabel(run);
  const awaitingApproval = run.status === "awaiting_approval";
  const provisionalEditable = run.status === "waiting_for_ollama" && run.plan_state === "provisional";
  const planEditable = awaitingApproval || provisionalEditable;
  if (!planEditable) planEditing = false;
  const visiblePlan = awaitingApproval
    ? (run.draft_plan || run.approved_plan)
    : (run.approved_plan || run.draft_plan);
  $("planActions").classList.toggle("hidden", !(visiblePlan && planEditable));
  if (visiblePlan) {
    $("planProof").innerHTML = renderMarkdown(visiblePlan);
    if (!planEditing) $("planEditorContent").value = visiblePlan;
    $("approvePlanBtn").classList.toggle("hidden", !awaitingApproval);
    $("editPlanBtn").classList.toggle("hidden", !planEditable || planEditing);
    $("savePlanBtn").classList.toggle("hidden", !planEditable || !planEditing);
    $("cancelPlanEditBtn").classList.toggle("hidden", !planEditable || !planEditing);
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
  renderTaskGraph(run);
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

// ---------------------------------------------------------------------------
// Visual task graph (Cline-style linked boxes)
// ---------------------------------------------------------------------------

const TG_NODE_W = 170;
const TG_NODE_H = 64;

async function loadAgentsCache() {
  if (agentsCache) return agentsCache;
  try {
    const data = await api("/api/agents");
    agentsCache = data.agents || [];
  } catch (_) { agentsCache = agentsCache || []; }
  return agentsCache;
}

// Render the drafted DAG as linked boxes. Gated on subtasks existing — which the backend
// creates the moment a plan is drafted (awaiting_approval). Editable (LLM pickers +
// drag-to-rewire) only while awaiting approval; read-only + live status afterwards.
function renderTaskGraph(run) {
  const section = $("taskGraphSection");
  const host = $("taskGraph");
  if (!section || !host) return;
  const subs = run?.subtasks || [];
  if (!subs.length) { section.classList.add("hidden"); return; }
  section.classList.remove("hidden");
  const editable = run.status === "awaiting_approval";
  $("taskGraphHint").textContent = run.status === "waiting_for_ollama"
    ? " · Ollama offline · saved"
    : editable
      ? " · assign an LLM · drag ▸ to add dependency · click edge to remove"
      : " · live";
  drawTaskGraph(run, subs, agentsCache || [], editable);
  if (!agentsCache) {
    loadAgentsCache().then(() => {
      // Redraw with real agents if this run is still the one on screen and still shown.
      if (currentRun === run.id && !section.classList.contains("hidden")) {
        drawTaskGraph(run, currentRunData?.subtasks || subs, agentsCache || [], editable);
      }
    });
  }
}

function drawTaskGraph(run, subs, agents, editable) {
  const host = $("taskGraph");
  const model = buildGraphModel(subs, agents);
  const layout = computeLayout(model, { nodeW: TG_NODE_W, nodeH: TG_NODE_H });
  const edges = edgeList(model);
  const pos = new Map(layout.nodes.map((node) => [node.node_id, node]));
  const statusById = new Map(model.map((node) => [node.node_id, node.status]));
  // A pending node whose deps are all done is the "next task" the brain just released.
  const isReady = (node) =>
    run.status === "implementing" && node.status === "pending" &&
    (node.deps || []).every((dep) => statusById.get(dep) === "done");

  host.style.width = `${layout.width}px`;
  host.style.height = `${layout.height}px`;
  host.textContent = "";

  const svgNS = "http://www.w3.org/2000/svg";
  const svg = document.createElementNS(svgNS, "svg");
  svg.setAttribute("class", `tg-edges${editable ? " editable" : ""}`);
  svg.setAttribute("width", String(layout.width));
  svg.setAttribute("height", String(layout.height));
  svg.setAttribute("role", "img");
  const svgTitle = document.createElementNS(svgNS, "title");
  svgTitle.textContent = `Task dependency graph, ${layout.nodes.length} task(s)`;
  svg.appendChild(svgTitle);
  const defs = document.createElementNS(svgNS, "defs");
  const marker = document.createElementNS(svgNS, "marker");
  marker.setAttribute("id", "tg-arrow");
  marker.setAttribute("viewBox", "0 0 10 10");
  marker.setAttribute("refX", "9");
  marker.setAttribute("refY", "5");
  marker.setAttribute("markerWidth", "6");
  marker.setAttribute("markerHeight", "6");
  marker.setAttribute("orient", "auto-start-reverse");
  const arrow = document.createElementNS(svgNS, "path");
  arrow.setAttribute("d", "M 0 0 L 10 5 L 0 10 z");
  arrow.setAttribute("fill", "context-stroke");
  marker.appendChild(arrow);
  defs.appendChild(marker);
  svg.appendChild(defs);
  for (const edge of edges) {
    const a = pos.get(edge.from);
    const b = pos.get(edge.to);
    if (!a || !b) continue;
    const x1 = a.x + TG_NODE_W;
    const y1 = a.y + TG_NODE_H / 2;
    const x2 = b.x;
    const y2 = b.y + TG_NODE_H / 2;
    const midX = (x1 + x2) / 2;
    const path = document.createElementNS(svgNS, "path");
    path.setAttribute("d", `M ${x1} ${y1} C ${midX} ${y1}, ${midX} ${y2}, ${x2} ${y2}`);
    path.setAttribute("marker-end", "url(#tg-arrow)");
    let cls = "tg-edge";
    if (statusById.get(edge.from) === "done") cls += " done";
    if (isReady(pos.get(edge.to))) cls += " ready";
    path.setAttribute("class", cls);
    if (editable) {
      path.setAttribute("tabindex", "0");
      path.setAttribute("aria-label", `Remove dependency ${edge.from} before ${edge.to}`);
      const remove = () => removeTaskLink(run, model, edge.from, edge.to);
      path.addEventListener("click", remove);
      path.addEventListener("keydown", (event) => {
        if (["Enter", " ", "Delete", "Backspace"].includes(event.key)) {
          event.preventDefault();
          remove();
        }
      });
    }
    svg.appendChild(path);
  }
  host.appendChild(svg);

  for (const node of layout.nodes) {
    const box = document.createElement("div");
    box.className = "tg-node";
    box.dataset.status = node.status;
    box.dataset.complexity = node.complexity;
    box.dataset.node = node.node_id;
    box.title = node.blocked_reason || node.handoff?.summary || node.title;
    // Make each task focusable and announced (nodes were previously SR-invisible divs).
    box.setAttribute("tabindex", "0");
    box.setAttribute("role", "group");
    const deps = (node.depends_on || []).join(", ");
    box.setAttribute(
      "aria-label",
      `Task ${node.title}. Role ${node.role}. Status ${node.status}.`
      + (node.complexity === "complex" ? " Complex." : "")
      + (deps ? ` Depends on ${deps}.` : "")
      + (node.blocked_reason ? ` Blocked: ${node.blocked_reason}.` : ""),
    );
    if (isReady(node)) box.dataset.ready = "1";
    box.style.left = `${node.x}px`;
    box.style.top = `${node.y}px`;
    box.style.width = `${TG_NODE_W}px`;

    const head = document.createElement("div");
    head.className = "tg-node-head";
    head.innerHTML =
      `<span class="tg-dot"></span><span class="tg-title" title="${escapeHtml(node.title)}">${escapeHtml(node.title)}</span>` +
      (node.complexity === "complex" ? `<span class="tg-complex" title="complex task">🔴</span>` : "");
    box.appendChild(head);

    const meta = document.createElement("div");
    meta.className = "tg-node-meta";
    meta.innerHTML = `<span class="tg-role">${escapeHtml(node.role)}</span><span class="tg-state">${escapeHtml(node.status)}</span>`;
    box.appendChild(meta);

    if (editable) {
      const sel = document.createElement("select");
      sel.className = "tg-node-picker";
      const auto = document.createElement("option");
      auto.value = "";
      auto.textContent = "⚙ auto (scheduler)";
      sel.appendChild(auto);
      for (const agent of assignableAgents(agents, node.role)) {
        const opt = document.createElement("option");
        opt.value = agent.id;
        opt.textContent = agent.name;
        if (agent.id === node.assigned_agent_id) opt.selected = true;
        sel.appendChild(opt);
      }
      sel.onchange = () => saveGraphEdit(run.id, { node_id: node.node_id, assigned_agent_id: sel.value });
      box.appendChild(sel);

      const handle = document.createElement("div");
      handle.className = "tg-handle";
      handle.title = "drag onto another task to make this one run first";
      handle.textContent = "▸";
      attachLinkDrag(handle, node.node_id, run, model);
      box.appendChild(handle);
    } else {
      const who = document.createElement("div");
      who.className = "tg-node-agent";
      who.textContent = node.agent_name || "unassigned";
      box.appendChild(who);
    }
    host.appendChild(box);
  }
}

async function saveGraphEdit(runId, node) {
  try {
    await api(`/api/runs/${runId}/graph`, { method: "PUT", body: JSON.stringify({ nodes: [node] }) });
  } catch (err) {
    showToast(err.message, "error");
  }
  await loadRun(runId);
}

// Toggle a dependency edge: source must run before target (target depends on source).
// Dragging an existing link again removes it.
function linkTasks(run, model, sourceId, targetId) {
  const target = model.find((node) => node.node_id === targetId);
  if (!target) return;
  const deps = new Set(target.deps || []);
  if (deps.has(sourceId)) {
    showToast("Tasks are already linked", "info");
    return;
  }
  if (wouldCycle(model, sourceId, targetId)) {
    showToast("That link would create a cycle", "error");
    return;
  }
  deps.add(sourceId);
  saveGraphEdit(run.id, { node_id: targetId, depends_on: [...deps] });
}

function removeTaskLink(run, model, sourceId, targetId) {
  const target = model.find((node) => node.node_id === targetId);
  if (!target) return;
  const deps = new Set(target.deps || []);
  if (!deps.delete(sourceId)) return;
  saveGraphEdit(run.id, { node_id: targetId, depends_on: [...deps] });
}

// Pointer-based drag from a node's ▸ handle onto another node. Mirrors the pointer-capture
// pattern used by setupPanelResizer.
function attachLinkDrag(handle, sourceId, run, model) {
  handle.addEventListener("pointerdown", (event) => {
    event.preventDefault();
    event.stopPropagation();
    const host = $("taskGraph");
    const ghost = document.createElement("div");
    ghost.className = "tg-drag-ghost";
    ghost.textContent = "link →";
    document.body.appendChild(ghost);
    handle.setPointerCapture?.(event.pointerId);
    const clearDrop = () => host.querySelectorAll(".tg-node.tg-drop").forEach((el) => el.classList.remove("tg-drop"));
    const move = (ev) => {
      ghost.style.left = `${ev.clientX + 8}px`;
      ghost.style.top = `${ev.clientY + 8}px`;
      clearDrop();
      const over = document.elementFromPoint(ev.clientX, ev.clientY)?.closest?.(".tg-node");
      if (over && over.dataset.node !== sourceId) over.classList.add("tg-drop");
    };
    const cleanup = () => {
      document.removeEventListener("pointermove", move);
      document.removeEventListener("pointerup", up);
      document.removeEventListener("pointercancel", cancel);
      document.removeEventListener("keydown", keydown);
      window.removeEventListener("blur", cancel);
      ghost.remove();
      clearDrop();
    };
    const up = (ev) => {
      cleanup();
      const over = document.elementFromPoint(ev.clientX, ev.clientY)?.closest?.(".tg-node");
      if (over && over.dataset.node && over.dataset.node !== sourceId) {
        linkTasks(run, model, sourceId, over.dataset.node);
      }
    };
    const cancel = () => cleanup();
    const keydown = (ev) => {
      if (ev.key === "Escape") cleanup();
    };
    document.addEventListener("pointermove", move);
    document.addEventListener("pointerup", up);
    document.addEventListener("pointercancel", cancel);
    document.addEventListener("keydown", keydown);
    window.addEventListener("blur", cancel);
  });
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
  // Generation guard: only the most recent loadRun() render wins, so overlapping
  // responses from a burst of run events (or a fast run switch) can't clobber
  // newer state with stale data.
  const gen = ++runStreamGen;
  const run = await api(`/api/runs/${id}`);
  if (gen !== runStreamGen) return run;
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
  let data;
  try {
    const cursor = append ? runNext : 0;
    data = await api(`/api/runs?cursor=${cursor || 0}&limit=50`);
  } catch (err) {
    if (!append) showPaneError("runList", `Couldn't load runs: ${err.message}`);
    return;
  }
  runs = append ? [...runs, ...(data.runs || [])] : (data.runs || []);
  runNext = data.next_cursor;
  $("runList").innerHTML = "";
  runs.forEach((run) => {
    const btn = document.createElement("button");
    btn.className = "item" + (run.id === currentRun ? " active" : "");
    btn.innerHTML = `<strong>${escapeHtml(run.task.slice(0, 55))}</strong><small>${escapeHtml(runStatusLabel(run))} · ${escapeHtml(run.brain_provider)}</small>`;
    btn.onclick = async () => {
      try {
        await loadRun(run.id);
        switchEditor("runEditor");
        closeDrawers();
        drawerReturnFocus = btn;
        revealSupervisorPanel();
        subscribeRun(run.id);
      } catch (err) { showToast(err.message, "error"); }
    };
    $("runList").appendChild(btn);
  });
  $("loadMoreRunsBtn").classList.toggle("hidden", runNext === null || runNext === undefined);
}

const TERMINAL_RUN_STATUSES = ["completed", "failed", "cancelled", "rolled_back"];
const RUN_STREAM_MAX_ERRORS = 5;

function closeRunStream() {
  if (runEventSource) { runEventSource.close(); runEventSource = null; }
  if (runRefreshTimer) { clearTimeout(runRefreshTimer); runRefreshTimer = null; }
}

// Coalesce a burst of run events into a single refetch (~250ms) instead of one
// full loadRun per event across ~40 listeners.
function scheduleRunRefresh(id) {
  if (runRefreshTimer) return;
  runRefreshTimer = setTimeout(() => {
    runRefreshTimer = null;
    loadRun(id).then((run) => {
      loadRuns().catch(() => {});
      if (TERMINAL_RUN_STATUSES.includes(run.status)) closeRunStream();
    }).catch(() => {});
  }, 250);
}

function subscribeRun(id) {
  closeRunStream();
  runStreamErrors = 0;
  runEventSource = new EventSource(`/api/runs/${id}/events`);
  const onEvent = () => { runStreamErrors = 0; scheduleRunRefresh(id); };
  runEventSource.onmessage = onEvent;
  ["run.created", "ollama.waiting", "ollama.reconnected", "plan.provisional", "plan.refining", "research.started", "research.completed", "plan.ready", "plan.edited", "plan.graph_ready", "plan.redo", "plan.approved", "plan.decomposed", "plan.graph_edited", "scope.approved", "scope.approval_required", "implementation.started", "agent.activity", "subtask.started", "subtask.completed", "subtask.verified", "subtask.retry", "subtask.failed", "subtask.blocked", "subtasks.merged", "subtasks.conflict", "check.completed", "gate.evaluated", "verification.completed", "apply.completed", "rollback.completed", "run.completed", "run.failed", "run.cancelled", "plan.stale"].forEach((name) => {
    runEventSource.addEventListener(name, onEvent);
  });
  // The native EventSource auto-reconnects; cap repeated failures so a persistently
  // broken stream stops hammering the server, and reconcile state once per failure.
  runEventSource.onerror = () => {
    runStreamErrors += 1;
    if (runStreamErrors >= RUN_STREAM_MAX_ERRORS) { closeRunStream(); return; }
    scheduleRunRefresh(id);
  };
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
    option.textContent = "No Brain linked";
    select.appendChild(option);
  } else if (enabled.some((provider) => provider.value === selected)) {
    select.value = selected;
  }
  // Guide the user when no Brain is linked: disable generate + show a link to Settings.
  const hasProvider = enabled.length > 0;
  $("generatePlanBtn").disabled = !hasProvider;
  $("planProviderHint").classList.toggle("hidden", hasProvider);

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
const systemThemeMedia = window.matchMedia?.("(prefers-color-scheme: light)") || null;

function applyTheme(theme) {
  const resolved = theme === "auto"
    ? (systemThemeMedia?.matches ? "light" : "dark")
    : theme;
  document.documentElement.dataset.theme = resolved;
  document.documentElement.dataset.themePreference = theme === "auto" ? "system" : resolved;
  const themeColor = $("themeColor");
  if (themeColor) themeColor.content = resolved === "light" ? "#eeeae8" : "#0f0f0f";
}

systemThemeMedia?.addEventListener?.("change", () => {
  if (appSettings.theme === "auto") applyTheme("auto");
});

async function loadSettings() {
  try {
    appSettings = await api("/api/settings");
    $("budgetDaily").value = appSettings.token_budget_daily || 0;
    $("budgetRun").value = appSettings.token_budget_run || 0;
    $("maxOutputTokens").value = appSettings.max_output_tokens || 0;
    $("themeSelect").value = appSettings.theme || "dark";
    $("agentModeDefault").checked = Boolean(appSettings.agent_mode_default);
    $("snapshotRetentionDays").value = appSettings.snapshot_retention_days || 30;
    $("timelineCap").value = appSettings.timeline_cap || 5000;
    $("subtaskMaxAttempts").value = appSettings.subtask_max_attempts || 2;
    $("brainMemoryBudget").value = appSettings.brain_memory_budget || 4000;
    $("runEventsCap").value = appSettings.run_events_cap || 500;
    $("runArtifactsCap").value = appSettings.run_artifacts_cap || 200;
    applyTheme(appSettings.theme || "dark");
    if (!agentModeTouched) {
      $("agentModeToggle").checked = Boolean(appSettings.agent_mode_default);
      $("chatBrainSelect").classList.toggle("hidden", !$("agentModeToggle").checked);
    }
  } catch (_) {}
  await loadUsage();
}

async function loadSettingsStorage() {
  try {
    const [storage, config] = await Promise.all([api("/api/maintenance/storage"), api("/api/config")]);
    $("settingsStorageSummary").textContent =
      `${storage.tracked.count} active · ${formatBytes(storage.tracked.bytes)} used · ${formatBytes(storage.filesystem.free_bytes)} free`;
    const paths = config.paths || {};
    $("settingsBackupSummary").textContent =
      `Database: ${paths.database || "configured data volume"} · Snapshots: ${paths.snapshots || "configured snapshot volume"}`;
  } catch (err) {
    $("settingsStorageSummary").textContent = "Storage details unavailable.";
    $("settingsBackupSummary").textContent = err.message;
  }
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

let hostsCache = [];

const KNOWN_ROLES = ["research", "implementation", "verification"];

async function loadHosts() {
  let data;
  try { data = await api("/api/hosts"); }
  catch (err) { showPaneError("hostList", `Couldn't load hosts: ${err.message}`); return; }
  hostsCache = data.hosts || [];
  const list = $("hostList");
  if (!list) return;
  list.innerHTML = "";
  if (!hostsCache.length) {
    const empty = document.createElement("div");
    empty.className = "settings-empty";
    empty.textContent = "No hosts yet — add one above, or Scan network to detect a local server.";
    list.appendChild(empty);
    return;
  }
  hostsCache.forEach((host) => {
    const badge = hostStatusLabel(host.status, host.last_seen);
    const row = document.createElement("div");
    row.className = "host-row";
    const reason = host.status === "unreachable" && host.last_error ? ` · ${host.last_error}` : "";
    row.innerHTML = `<strong>${escapeHtml(host.name)}</strong>`
      + `<span class="host-badge host-badge-${badge.tone}">${escapeHtml(badge.text)}</span>`
      + `<small>${escapeHtml(host.base_url)} · ${escapeHtml(host.kind)}${escapeHtml(reason)}</small>`;
    const test = document.createElement("button");
    test.textContent = "Test";
    test.onclick = () => testHost(host);
    const refresh = document.createElement("button");
    refresh.textContent = "Refresh models";
    refresh.onclick = () => discoverAgentModels(host.id);
    const toggle = document.createElement("button");
    toggle.textContent = host.enabled ? "Enabled" : "Disabled";
    toggle.onclick = async () => {
      try {
        await api(`/api/hosts/${host.id}`, { method: "PATCH", body: JSON.stringify({ enabled: !host.enabled }) });
        showToast(host.enabled ? "Host disabled" : "Host enabled", "success");
        await Promise.allSettled([loadHosts(), loadAgents()]);
      } catch (err) { showToast(err.message, "error"); }
    };
    const remove = document.createElement("button");
    remove.className = "danger-btn";
    remove.textContent = "Remove";
    remove.onclick = async () => {
      if (!confirm(`Remove host "${host.name}" and its agents?`)) return;
      try {
        await api(`/api/hosts/${host.id}`, { method: "DELETE" });
        showToast("Host removed", "success");
        await Promise.allSettled([loadHosts(), loadAgents()]);
      } catch (err) { showToast(err.message, "error"); }
    };
    row.append(test, refresh, toggle, remove);
    list.appendChild(row);
  });
}

// Probe one host on demand and report reachability inline (reuses the scan endpoint).
async function testHost(host) {
  try {
    const data = await api("/api/hosts/scan", { method: "POST", body: JSON.stringify({ base_url: host.base_url }) });
    const hit = (data.results || []).find((r) => r.base_url === host.base_url);
    if (hit && hit.reachable) showToast(`${host.name} online — ${hit.models.length} model(s)`, "success");
    else showToast(`${host.name} unreachable at ${host.base_url}`, "error");
    await loadHosts();
  } catch (err) { showToast(err.message, "error"); }
}

async function addHost() {
  const url = normalizeHostUrl($("hostUrl").value);
  if (!url) { showToast("Enter a valid host URL like http://127.0.0.1:11434", "error"); return; }
  const name = ($("hostName").value || "").trim() || url;
  setBusy(["addHostBtn"], true);
  try {
    await api("/api/hosts", { method: "POST", body: JSON.stringify({ name, base_url: url, kind: $("hostKind").value }) });
    $("hostName").value = "";
    $("hostUrl").value = "";
    await Promise.allSettled([loadHosts(), loadAgents()]);
    showToast("Host added", "success");
  } catch (err) { showToast(err.message, "error"); }
  finally { setBusy(["addHostBtn"], false); }
}

async function scanHosts() {
  const entered = normalizeHostUrl($("hostUrl").value);
  const box = $("scanResults");
  setBusy(["scanHostsBtn"], true);
  try {
    const data = await api("/api/hosts/scan", { method: "POST", body: JSON.stringify(entered ? { base_url: entered } : {}) });
    const results = data.results || [];
    box.innerHTML = "";
    box.hidden = false;
    const title = document.createElement("div");
    title.className = "scan-results-title";
    title.textContent = "Detected servers";
    box.appendChild(title);
    results.forEach((r) => {
      const row = document.createElement("div");
      row.className = "scan-row";
      const label = r.reachable ? `${r.models.length} model(s)` : "unreachable";
      row.innerHTML = `<span class="host-badge host-badge-${r.reachable ? "ok" : "bad"}">${r.reachable ? "Online" : "Offline"}</span><small>${escapeHtml(r.base_url)} · ${escapeHtml(label)}</small>`;
      if (r.reachable && !r.registered) {
        const add = document.createElement("button");
        add.textContent = "Add";
        add.onclick = async () => {
          try {
            await api("/api/hosts", { method: "POST", body: JSON.stringify({ name: r.base_url, base_url: r.base_url, kind: "network" }) });
            await Promise.allSettled([loadHosts(), loadAgents()]);
            showToast("Host added", "success");
          } catch (err) { showToast(err.message, "error"); }
        };
        row.appendChild(add);
      } else if (r.registered) {
        const tag = document.createElement("small");
        tag.textContent = "already added";
        row.appendChild(tag);
      }
      box.appendChild(row);
    });
    if (!results.length) {
      const none = document.createElement("div");
      none.className = "settings-empty";
      none.textContent = "No servers detected. Enter a URL above and try again.";
      box.appendChild(none);
    }
  } catch (err) { showToast(err.message, "error"); }
  finally { setBusy(["scanHostsBtn"], false); }
}

async function loadAgents() {
  let data;
  try { data = await api("/api/agents"); }
  catch (err) { showPaneError("agentList", `Couldn't load models: ${err.message}`); return; }
  const agents = data.agents || [];
  agentsCache = agents;
  const hostName = (id) => (hostsCache.find((h) => h.id === id) || {}).name || "";
  const list = $("agentList");
  const head = document.querySelector(".agent-list-head");
  list.innerHTML = "";
  if (head) head.style.display = agents.length ? "grid" : "none";
  if (!agents.length) {
    const empty = document.createElement("div");
    empty.className = "settings-empty";
    empty.textContent = "No models yet — add a host above, then Refresh models.";
    list.appendChild(empty);
    return;
  }
  agents.forEach((agent) => {
    const row = document.createElement("div");
    row.className = "agent-row";
    const host = hostName(agent.host_id);
    row.innerHTML = `<strong>${escapeHtml(agent.name)}</strong><span>${escapeHtml((agent.roles || []).join(" · "))}</span><small>${escapeHtml((agent.capabilities || []).join(", "))} · priority ${agent.priority}${host ? ` · ${escapeHtml(host)}` : ""}</small>`;
    // Roles as checkboxes over the known roles, not free-text CSV. Implementation needs the
    // tools capability, so disable it when the model can't call tools.
    const canImpl = (agent.capabilities || []).includes("tools");
    const roleWrap = document.createElement("div");
    roleWrap.className = "agent-roles";
    const roleBoxes = KNOWN_ROLES.map((role) => {
      const lbl = document.createElement("label");
      lbl.className = "role-check";
      const cb = document.createElement("input");
      cb.type = "checkbox";
      cb.value = role;
      cb.checked = (agent.roles || []).includes(role);
      if (role === "implementation" && !canImpl) {
        cb.disabled = true;
        cb.checked = false;
        lbl.title = "Model lacks the tools capability required for implementation";
      }
      lbl.append(cb, document.createTextNode(role));
      roleWrap.appendChild(lbl);
      return cb;
    });
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
            roles: roleBoxes.filter((cb) => cb.checked).map((cb) => cb.value),
            priority: clampPriority(priority.value, { fallback: agent.priority }),
            system_prompt: prompt.value,
          }),
        });
        showToast("Agent saved", "success");
        await loadAgents();
      } catch (err) { showToast(err.message, "error"); }
    };
    const toggle = document.createElement("button");
    toggle.textContent = agent.enabled ? "Enabled" : "Disabled";
    toggle.onclick = async () => {
      try {
        await api(`/api/agents/${agent.id}`, { method: "PATCH", body: JSON.stringify({ enabled: !agent.enabled }) });
        showToast(agent.enabled ? "Agent disabled" : "Agent enabled", "success");
        await loadAgents();
      } catch (err) { showToast(err.message, "error"); }
    };
    row.append(roleWrap, priority, prompt, save, toggle);
    list.appendChild(row);
  });
}

async function discoverAgentModels(hostId = null) {
  setBusy(["discoverAgentsBtn"], true);
  try {
    const body = hostId ? JSON.stringify({ host_id: hostId }) : "{}";
    const data = await api("/api/agents/discover", { method: "POST", body });
    await Promise.allSettled([loadHosts(), loadAgents()]);
    const errors = data.errors || [];
    if (errors.length) {
      showToast(`Unreachable: ${errors.map((e) => e.name).join(", ")}`, "error");
    } else {
      showToast(`Discovered ${(data.agents || []).length} agent(s)`, "success");
    }
  } catch (err) { showToast(err.message, "error"); }
  finally { setBusy(["discoverAgentsBtn"], false); }
}

// ---------------------------------------------------------------------------
// Preferences (localStorage)
// ---------------------------------------------------------------------------

function savePrefs() {
  try {
    writePreferences(localStorage, {
      model: $("chatModelSelect").value,
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
  hide("app");
  clearInterval(ollamaPollTimer);
  setStatus("Loading…");

  // Config — know which AI providers are available
  try {
    const cfg = await api("/api/config");
    // Populate plan provider selector based on what's enabled (loadBrains re-syncs later)
    const providerList = [
      { key: "codex", enabled: cfg.openai_enabled, label: "ChatGPT", indicator: "openaiIndicator" },
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

  await refreshOllamaState();
  try {
    if (ollamaOnline) await loadModels();
  } catch (err) {
    ollamaOnline = false;
    renderOllamaStatus();
    showToast("Ollama models unavailable: " + err.message, "error");
  }
  // Guard each loader: a transient chat/FS/list error must not reject boot() and
  // bounce the user back to the login screen.
  try { await loadChats(); } catch (err) { showToast("Failed to load chats: " + err.message, "error"); }
  await Promise.allSettled([loadRuns(), loadBrains(), loadAgents(), loadSettings()]);

  if (prefs.chat && chats.find((c) => c.id === prefs.chat)) {
    try { await loadChat(prefs.chat); } catch (_) {}
  }

  try { await openPath(prefs.target || ""); } catch (err) { showToast(err.message, "error"); }

  if (prefs.file) {
    try { await openFile(prefs.file); } catch (_) {}
  }

  setStatus(ollamaOnline ? "Ready" : "Ready · Ollama offline");
  show("app");
  ollamaPollTimer = setInterval(refreshOllamaState, 15_000);
}

// ---------------------------------------------------------------------------
// DOMContentLoaded — wire up all events
// ---------------------------------------------------------------------------

document.addEventListener("DOMContentLoaded", () => {
  populateModelSelect("codex", "gpt-5.6-sol");
  populateModelSelect("claude", "claude-sonnet-5");
  populateModelSelect("gemini", "gemini-2.5-pro");
  setupPanelResizer("sidebarResize", "sidebar", 292);
  setupPanelResizer("panelResize", "panel", 340);
  // Auth
  $("loginForm").onsubmit = login;
  $("logoutBtn").onclick  = async () => {
    clearInterval(ollamaPollTimer);
    closeRunStream();
    try { await api("/api/auth/logout", { method: "POST", body: "{}" }); } finally { location.reload(); }
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
  $("chatModelSelect").onchange = () => syncChatModel($("chatModelSelect").value);
  // Chat target and supervisor path are one workspace folder — keep them in lock-step so the
  // run target can't silently diverge from the chat context.
  $("targetPath").onchange = () => {
    const v = $("targetPath").value.trim();
    hide("targetHint");
    setWorkspaceTag(v);
    $("planPath").value = v;
    resetPinnedContext();
  };
  $("targetPath").oninput  = () => { if ($("targetPath").value.trim()) { hide("targetHint"); setWorkspaceTag($("targetPath").value.trim()); } };
  $("planPath").onchange = () => {
    const v = $("planPath").value.trim();
    $("targetPath").value = v;
    setWorkspaceTag(v);
  };

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
  $("planProviderHint").onclick = openSettingsPage;
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
    revealSupervisorPanel();
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
  $("saveRunBehaviorBtn").onclick = () => saveSettings({
    brain_memory_budget: $("brainMemoryBudget").value,
    subtask_max_attempts: $("subtaskMaxAttempts").value,
  }, "Run behavior saved");
  $("saveHistoryBtn").onclick = async () => {
    await saveSettings({
      snapshot_retention_days: $("snapshotRetentionDays").value,
      timeline_cap: $("timelineCap").value,
    }, "Snapshots and history saved");
    await loadSettingsStorage();
  };
  $("saveRetentionBtn").onclick = () => saveSettings({
    run_events_cap: $("runEventsCap").value,
    run_artifacts_cap: $("runArtifactsCap").value,
  }, "Evidence retention saved");
  $("discoverAgentsBtn").onclick = () => discoverAgentModels();
  $("addHostBtn").onclick = addHost;
  $("scanHostsBtn").onclick = scanHosts;
  $("closeSettingsBtn").onclick = () => setSettingsMode(false);
  $("openSnapshotsSettingsBtn").onclick = () => { switchSidePane("snaps"); loadSnaps(); };
  $("openTimelineSettingsBtn").onclick = () => { switchSidePane("timeline"); loadTimeline(); };

  // Activity bar — switches sidebar pane (plan button toggles the plan panel on mobile)
  $("activityChat").onclick     = () => switchSidePane("chat");
  $("activityRuns").onclick     = () => { switchSidePane("runs"); loadRuns(); };
  $("activityFiles").onclick    = () => switchSidePane("files");
  $("activityTools").onclick    = () => switchSidePane("tools");
  $("activitySnaps").onclick    = () => { switchSidePane("snaps");    loadSnaps();    };
  $("activityTimeline").onclick = () => { switchSidePane("timeline"); loadTimeline(); };
  $("activityPlan").onclick     = toggleSupervisorPanel;
  $("activitySettings").onclick = openSettingsPage;
  $("activityMore").onclick = () => switchSidePane("more");
  $("moreToolsBtn").onclick = () => switchSidePane("tools");
  $("moreSnapsBtn").onclick = () => { switchSidePane("snaps"); loadSnaps(); };
  $("moreTimelineBtn").onclick = () => { switchSidePane("timeline"); loadTimeline(); };
  $("moreSettingsBtn").onclick = openSettingsPage;
  $("moreSupervisorBtn").onclick = openSupervisorFromMore;

  // Editor tabs
  document.querySelectorAll(".editor-tab").forEach((tab) => {
    tab.onclick = () => { setSettingsMode(false); switchEditor(tab.dataset.editor); };
  });

  // Status bar toggles
  $("menuBtn").onclick      = () => { document.querySelector(".sidebar").classList.toggle("open"); syncDrawerState(); };
  $("workspaceBtn").onclick = toggleSupervisorPanel;
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
    if (e.key === "Tab" && window.matchMedia("(max-width: 1180px)").matches) {
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
