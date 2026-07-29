"""Task-graph parsing and validation for multi-agent decomposition.

The supervisor brain decomposes an approved plan into a DAG of subtasks. This module
is pure (no DB, no network): it validates graph shape so the orchestrator can persist
and schedule it safely.  JSON extraction is delegated to brain_io.extract_json.

Graph shape (what the brain is asked to emit):
    {"subtasks": [
        {"node_id": "a", "title": "...", "spec": "...", "depends_on": [],
         "file_globs": ["src/x/**"], "acceptance_criteria": "...",
         "role": "implementation", "complexity": "simple", "suggested_model": null}
    ]}
"""
from __future__ import annotations

import fnmatch
import re
from typing import Any

from .brain_io import extract_json as _extract_json

VALID_ROLES = {"research", "implementation", "verification"}
MAX_SUBTASKS = 40
# node_id becomes an on-disk worktree directory name, so it must be a safe slug
# (no slashes, no '..', no spaces) to prevent path traversal.
NODE_ID_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._-]{0,63}$")


class GraphError(ValueError):
    """Raised when a brain-produced task graph is missing or malformed."""


def parse_task_graph(text: str) -> dict[str, Any]:
    """Extract a JSON task graph from brain text. Delegates to brain_io.extract_json.

    Kept as a named entry point for backward compatibility; raises GraphError on failure.
    """
    try:
        return _extract_json(text)
    except ValueError as exc:
        raise GraphError(str(exc)) from exc


def _as_str_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    result: list[str] = []
    seen: set[str] = set()
    for item in value:
        if not isinstance(item, (str, int, float)):
            continue
        text = str(item).strip()
        if text and text not in seen:
            seen.add(text)
            result.append(text)
    return result


def validate_graph(graph: dict[str, Any]) -> list[dict[str, Any]]:
    """Validate and normalize a task graph into an ordered list of subtask nodes.

    Enforces: non-empty, unique node_ids, dependency references resolve, and the graph
    is acyclic. Returns nodes in a topological order (dependencies before dependants),
    each normalized to: node_id, title, spec, depends_on, file_globs, acceptance_criteria,
    role, suggested_model, complexity. Raises GraphError on any violation.
    """
    raw = graph.get("subtasks") if isinstance(graph, dict) else None
    if not isinstance(raw, list) or not raw:
        raise GraphError("Task graph has no subtasks")
    if len(raw) > MAX_SUBTASKS:
        raise GraphError(f"Task graph exceeds {MAX_SUBTASKS} subtasks")

    nodes: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise GraphError(f"Subtask #{index} is not an object")
        node_id = str(item.get("node_id") or item.get("id") or "").strip()
        if not node_id:
            raise GraphError(f"Subtask #{index} is missing node_id")
        if not NODE_ID_RE.match(node_id):
            raise GraphError(f"Unsafe node_id (must be a slug): {node_id!r}")
        if node_id in nodes:
            raise GraphError(f"Duplicate node_id: {node_id}")
        title = str(item.get("title") or "").strip() or node_id
        spec = str(item.get("spec") or item.get("description") or "").strip()
        if not spec:
            raise GraphError(f"Subtask {node_id} is missing spec")
        role = str(item.get("role") or item.get("suggested_role") or "implementation").strip().lower()
        if role not in VALID_ROLES:
            role = "implementation"
        suggested_model = str(item["suggested_model"]).strip() if item.get("suggested_model") else None
        complexity = "complex" if str(item.get("complexity") or "").strip().lower() == "complex" else "simple"
        nodes[node_id] = {
            "node_id": node_id,
            "title": title[:200],
            "spec": spec,
            "depends_on": _as_str_list(item.get("depends_on")),
            "file_globs": _as_str_list(item.get("file_globs")),
            "acceptance_criteria": str(item.get("acceptance_criteria") or "").strip(),
            "role": role,
            "suggested_model": suggested_model,
            "complexity": complexity,
        }
        for pattern in nodes[node_id]["file_globs"]:
            if pattern.startswith(("/", "\\")) or ".." in pattern.replace("\\", "/").split("/"):
                raise GraphError(f"Unsafe file scope for subtask {node_id}: {pattern!r}")
        order.append(node_id)

    for node_id, node in nodes.items():
        for dep in node["depends_on"]:
            if dep == node_id:
                raise GraphError(f"Subtask {node_id} depends on itself")
            if dep not in nodes:
                raise GraphError(f"Subtask {node_id} depends on unknown node {dep}")

    return _topological_order(nodes, order)


def _topological_order(nodes: dict[str, dict[str, Any]], order: list[str]) -> list[dict[str, Any]]:
    """Kahn's algorithm; raises GraphError if a cycle is present."""
    remaining = {nid: set(nodes[nid]["depends_on"]) for nid in order}
    resolved: list[dict[str, Any]] = []
    while remaining:
        ready = [nid for nid in order if nid in remaining and not remaining[nid]]
        if not ready:
            raise GraphError("Task graph contains a dependency cycle")
        for nid in ready:
            resolved.append(nodes[nid])
            del remaining[nid]
            for deps in remaining.values():
                deps.discard(nid)
    return resolved


def independent_pairs_sharing_globs(nodes: list[dict[str, Any]]) -> list[tuple[str, str]]:
    """Return pairs of nodes with no dependency path between them that share a file glob.

    Such pairs can run in parallel yet touch the same files — the merge step must treat
    them as potential conflicts. The brain is instructed to avoid these by adding a
    depends_on edge; this surfaces any it missed. Advisory only (does not raise).
    """
    reachable = _reachability(nodes)
    conflicts: list[tuple[str, str]] = []
    for i, left in enumerate(nodes):
        for right in nodes[i + 1 :]:
            a, b = left["node_id"], right["node_id"]
            if b in reachable[a] or a in reachable[b]:
                continue  # one depends (transitively) on the other → serialized
            if any(
                _globs_overlap(left_glob, right_glob)
                for left_glob in left["file_globs"]
                for right_glob in right["file_globs"]
            ):
                conflicts.append((a, b))
    return conflicts


def _globs_overlap(left: str, right: str) -> bool:
    """Conservatively detect whether two file scopes can select the same path."""
    left = left.replace("\\", "/").removeprefix("./")
    right = right.replace("\\", "/").removeprefix("./")
    if left == right:
        return True
    left_wild = any(char in left for char in "*?[")
    right_wild = any(char in right for char in "*?[")
    if not left_wild:
        return fnmatch.fnmatchcase(left, right) if right_wild else False
    if not right_wild:
        return fnmatch.fnmatchcase(right, left)

    def literal_edges(pattern: str) -> tuple[str, str]:
        first = min((pattern.find(char) for char in "*?[" if char in pattern), default=len(pattern))
        last = max(pattern.rfind("*"), pattern.rfind("?"), pattern.rfind("]"))
        return pattern[:first], pattern[last + 1 :]

    left_prefix, left_suffix = literal_edges(left)
    right_prefix, right_suffix = literal_edges(right)
    if left_prefix and right_prefix and not (
        left_prefix.startswith(right_prefix) or right_prefix.startswith(left_prefix)
    ):
        return False
    if left_suffix and right_suffix and not (
        left_suffix.endswith(right_suffix) or right_suffix.endswith(left_suffix)
    ):
        return False
    return True


def _reachability(nodes: list[dict[str, Any]]) -> dict[str, set[str]]:
    deps = {node["node_id"]: set(node["depends_on"]) for node in nodes}
    reach: dict[str, set[str]] = {nid: set() for nid in deps}
    for nid in deps:
        stack = list(deps[nid])
        while stack:
            current = stack.pop()
            if current in reach[nid] or current not in deps:
                continue
            reach[nid].add(current)
            stack.extend(deps[current])
    return reach
