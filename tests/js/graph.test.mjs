import assert from "node:assert/strict";
import test from "node:test";

import { assignableAgents, buildGraphModel, computeLayout, edgeList, layerize, wouldCycle } from "../../public/js/graph.mjs";

const chain = [
  { node_id: "a", title: "A", status: "done", role: "implementation", complexity: "simple", depends_on: [] },
  { node_id: "b", title: "B", status: "running", role: "implementation", complexity: "complex", depends_on: ["a"] },
  { node_id: "c", title: "C", status: "pending", role: "verification", complexity: "simple", depends_on: ["b"] },
];

test("buildGraphModel normalizes fields and resolves the pinned agent name", () => {
  const agents = [{ id: "agent-1", name: "qwen3-coder", roles: ["implementation"], enabled: 1 }];
  const model = buildGraphModel(
    [{ node_id: "a", title: "A", depends_on: ["x"], assigned_agent_id: "agent-1", agent_name: "auto" }],
    agents,
  );
  assert.equal(model[0].complexity, "simple");
  assert.deepEqual(model[0].deps, ["x"]);
  assert.equal(model[0].agent_name, "qwen3-coder"); // pinned wins over scheduler-chosen
  assert.equal(model[0].status, "pending");
});

test("layerize orders a chain into the left-to-right timetable", () => {
  const layer = layerize(buildGraphModel(chain));
  assert.equal(layer.get("a"), 0);
  assert.equal(layer.get("b"), 1);
  assert.equal(layer.get("c"), 2);
});

test("layerize uses longest path when a node has multiple deps", () => {
  const nodes = buildGraphModel([
    { node_id: "a", depends_on: [] },
    { node_id: "b", depends_on: ["a"] },
    { node_id: "d", depends_on: ["a", "b"] }, // longest path a->b->d = layer 2
  ]);
  const layer = layerize(nodes);
  assert.equal(layer.get("d"), 2);
});

test("edgeList emits dep->node edges and drops dangling deps", () => {
  const edges = edgeList([
    { node_id: "a", deps: [] },
    { node_id: "b", deps: ["a", "ghost"] },
  ]);
  assert.deepEqual(edges, [{ from: "a", to: "b" }]);
});

test("computeLayout places layers into columns and stacks rows", () => {
  const model = buildGraphModel(chain);
  const layout = computeLayout(model, { colGap: 200, padX: 10 });
  const byId = Object.fromEntries(layout.nodes.map((node) => [node.node_id, node]));
  assert.equal(byId.a.x, 10);
  assert.equal(byId.b.x, 210);
  assert.equal(byId.c.x, 410);
  assert.ok(layout.width >= 410);
});

test("wouldCycle blocks back-edges and self-links but allows forward links", () => {
  const model = buildGraphModel(chain);
  assert.equal(wouldCycle(model, "c", "a"), true); // c already depends on a -> a depends on c = cycle
  assert.equal(wouldCycle(model, "a", "a"), true);
  assert.equal(wouldCycle(model, "a", "c"), false); // c already depends on a (redundant but acyclic)
  assert.equal(wouldCycle(model, "b", "a"), true);
});

test("assignableAgents filters by enabled flag and role membership", () => {
  const agents = [
    { id: "1", name: "coder", roles: ["implementation"], enabled: 1 },
    { id: "2", name: "off", roles: ["implementation"], enabled: 0 },
    { id: "3", name: "researcher", roles: ["research"], enabled: 1 },
  ];
  const impl = assignableAgents(agents, "implementation").map((agent) => agent.id);
  assert.deepEqual(impl, ["1"]);
});
