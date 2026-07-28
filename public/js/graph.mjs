// Pure task-graph logic for the Supervisor Run visual graph. No DOM here so it can be
// unit-tested with `node --test` (see tests/js/graph.test.mjs). The DOM/SVG rendering and
// drag wiring live in app.js.
//
// A node's `deps` (depends_on) are the nodes that must finish before it runs. Edges are
// drawn dep -> node (left-to-right timetable): roots with no deps run first.

export function buildGraphModel(subtasks = [], agents = []) {
  const agentById = new Map((agents || []).map((agent) => [agent.id, agent]));
  return (subtasks || []).map((node) => {
    const assigned = node.assigned_agent_id ? agentById.get(node.assigned_agent_id) : null;
    return {
      node_id: node.node_id,
      title: node.title || node.node_id,
      status: node.status || "pending",
      role: node.role || "implementation",
      complexity: node.complexity === "complex" ? "complex" : "simple",
      deps: Array.isArray(node.depends_on) ? node.depends_on.slice() : [],
      assigned_agent_id: node.assigned_agent_id || "",
      // Prefer the user-pinned agent name; fall back to the scheduler-chosen one.
      agent_name: (assigned && assigned.name) || node.agent_name || "",
    };
  });
}

// Assign each node a layer = its longest dependency depth (roots = 0). This is the
// left-to-right "what runs first" timetable ordering. Returns a Map node_id -> layer.
export function layerize(nodes = []) {
  const byId = new Map(nodes.map((node) => [node.node_id, node]));
  const layer = new Map();
  const visiting = new Set();
  const depth = (id) => {
    if (layer.has(id)) return layer.get(id);
    const node = byId.get(id);
    if (!node || visiting.has(id)) return 0; // missing dep or cycle guard -> treat as root
    visiting.add(id);
    const deps = (node.deps || []).filter((dep) => byId.has(dep));
    const value = deps.length ? Math.max(...deps.map((dep) => depth(dep) + 1)) : 0;
    visiting.delete(id);
    layer.set(id, value);
    return value;
  };
  for (const node of nodes) depth(node.node_id);
  return layer;
}

// Position nodes into columns (by layer) and rows (order within a layer). Returns
// { nodes: [{...node, x, y, layer, row}], width, height } in px for absolute layout.
export function computeLayout(nodes = [], opts = {}) {
  const colGap = opts.colGap || 210;
  const rowGap = opts.rowGap || 96;
  const nodeW = opts.nodeW || 170;
  const nodeH = opts.nodeH || 64;
  const padX = opts.padX || 16;
  const padY = opts.padY || 16;
  const layer = layerize(nodes);
  const rows = new Map(); // layer -> next free row index
  const placed = nodes.map((node) => {
    const col = layer.get(node.node_id) || 0;
    const row = rows.get(col) || 0;
    rows.set(col, row + 1);
    return {
      ...node,
      layer: col,
      row,
      x: padX + col * colGap,
      y: padY + row * rowGap,
    };
  });
  const maxCol = placed.reduce((max, node) => Math.max(max, node.layer), 0);
  const maxRow = Math.max(0, ...Array.from(rows.values()));
  return {
    nodes: placed,
    width: padX * 2 + maxCol * colGap + nodeW,
    height: padY * 2 + maxRow * rowGap + (nodeH - rowGap > 0 ? nodeH - rowGap : 0),
  };
}

// Flatten depends_on into drawable edges { from: dep, to: dependent }.
export function edgeList(nodes = []) {
  const ids = new Set(nodes.map((node) => node.node_id));
  const edges = [];
  for (const node of nodes) {
    for (const dep of node.deps || []) {
      if (ids.has(dep)) edges.push({ from: dep, to: node.node_id });
    }
  }
  return edges;
}

// Client-side guard before persisting a new dependency (server re-validates
// authoritatively). Adding "dependent depends on dep" is a cycle if dep already
// (transitively) depends on dependent, or dep === dependent.
export function wouldCycle(nodes = [], dep, dependent) {
  if (!dep || !dependent || dep === dependent) return true;
  const byId = new Map(nodes.map((node) => [node.node_id, node]));
  const seen = new Set();
  const stack = [dep];
  while (stack.length) {
    const current = stack.pop();
    if (current === dependent) return true;
    if (seen.has(current)) continue;
    seen.add(current);
    const node = byId.get(current);
    if (node) stack.push(...(node.deps || []));
  }
  return false;
}

// Local Ollama agents eligible to run a subtask of the given role: enabled and their
// roles include the role. Mirrors backend _agent_is_eligible (role membership).
export function assignableAgents(agents = [], role = "implementation") {
  return (agents || []).filter((agent) => {
    if (!agent || !agent.enabled) return false;
    const roles = Array.isArray(agent.roles) ? agent.roles : [];
    return roles.includes(role);
  });
}
