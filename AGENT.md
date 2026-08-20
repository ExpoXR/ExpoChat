# Ollma — Agent Guide

Guidance for any AI agent (Claude Code, Codex, or the in-app supervisor) working on
this repo. Companion to [CLAUDE.md](CLAUDE.md); keep the two in sync.

## What this is

TrueNAS-hosted supervisor: FastAPI backend (`backend/`) + no-build ES-module frontend
(`public/`). It orchestrates local **Ollama** workers and **Codex/Claude** planning.
Changes run in staged workspaces, pass independent verification, then apply behind a
verified snapshot.

## Key modules

- `backend/orchestrator.py`: run state machine, durable job queue, DAG subtask execution, brain calls.
- `backend/brain_io.py`: structured brain I/O — prompt builders (`build_plan_prompt`, `build_decompose_prompt`, `build_verdict_prompt`, `build_verification_prompt`), `extract_json`, `parse_verdict`, typed dataclasses.
- `backend/plan_graph.py`: task-graph validation, `suggested_model` pass-through.
- `backend/worker.py`: credential-isolated Ollama agent loop.
- `backend/workspace.py`: staging, worktrees, manifests, snapshots.
- `backend/migrations.py`: ordered, idempotent SQLite migrations.

## Multi-agent pipeline

DAG-decomposed runs use durable subtask jobs:

1. Brain decomposes plan → task graph (nodes with deps, roles, `suggested_model`).
2. `_enqueue_ready_subtasks` walks DAG, enqueues ready nodes.
3. Each subtask runs in isolated worktree (`_run_durable_subtask`).
4. Completion chains next-ready nodes or merge job.
5. `_merge_and_verify` → `_verify_and_apply` (shared with single-agent path).
6. Brain memory (`brain_memory` table) carries reasoning across plan/decompose/verdict/repair calls.
7. `choose_subtask_agent` honors node `role` + `suggested_model` hint with round-robin tiebreak.

## The three AI paths (all caveman-wrapped)

The app prepends the caveman output instruction from
[`backend/prompts.py`](backend/prompts.py) to **every** model call:

1. **Interactive chat** — Ollama with read-only tools (`backend/main.py`).
2. **Supervised worker** — Ollama agent loop, research/implementation/verification
   (`backend/worker.py`, via `agent_system_prompt()`).
3. **Supervisor brain** — Codex/Claude for plans and verdicts (`backend/orchestrator.py`,
   via `call_brain_with_memory()` → `call_brain_result()` → `with_caveman()`).
   Brain memory accumulates across calls within a run (budget-truncated, oldest dropped first).

When adding a new model call, route it through `with_caveman()` /
`CAVEMAN_OUTPUT_INSTRUCTIONS` so terseness stays enforced everywhere.

When adding a new brain call, use `call_brain_with_memory()` so the brain retains context
across steps. Prompt builders live in `brain_io.py` — add new ones there, not inline.

## Caveman skills & commands (Claude Code)

These live under `.claude/` (skill `caveman/SKILL.md` with levels lite / full / ultra,
plus slash commands `/caveman`, `/caveman-commit`, `/caveman-review`, `/caveman-help`,
`/caveman-compress`; stop with "normal mode"). These skill files **are checked into the
repo** under `.claude/skills/`, so they ship with a fresh clone. They only affect Claude
Code sessions on this repo. The app's own terseness does not depend on them; it comes from
`CAVEMAN_OUTPUT_INSTRUCTIONS` in [`backend/prompts.py`](backend/prompts.py).

## Working rules

- All workspace file access goes through `allowed_path()` (`backend/workspace.py`) — never
  touch paths outside `ALLOWED_ROOTS`. Destructive `/api/fs/*` ops snapshot first.
- Providers never receive original workspace mounts; workers edit staged `/jobs` copies.
- Gate before commit: `make lint && make test && make test-js`.
- SQLite migrations are ordered/idempotent in `backend/migrations.py` — add, never edit
  applied ones.

## Deploy (TrueNAS Scale · Docker · Portainer)

```bash
sudo docker compose build
sudo docker compose up -d
curl -fsS http://127.0.0.1:31001/livez && curl -fsS http://127.0.0.1:31001/readyz
```
