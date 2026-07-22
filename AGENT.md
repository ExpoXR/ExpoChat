# Ollma — Agent Guide

Guidance for any AI agent (Claude Code, Codex, or the in-app supervisor) working on
this repo. Companion to [CLAUDE.md](CLAUDE.md); keep the two in sync.

## What this is

TrueNAS-hosted supervisor: FastAPI backend (`backend/`) + no-build ES-module frontend
(`public/`). It orchestrates local **Ollama** workers and **Codex/Claude** planning.
Changes run in staged workspaces, pass independent verification, then apply behind a
verified snapshot.

## The three AI paths (all caveman-wrapped)

The app prepends the caveman output instruction from
[`backend/prompts.py`](backend/prompts.py) to **every** model call:

1. **Interactive chat** — Ollama with read-only tools (`backend/main.py`).
2. **Supervised worker** — Ollama agent loop, research/implementation/verification
   (`backend/worker.py`, via `agent_system_prompt()`).
3. **Supervisor brain** — Codex/Claude for plans and verdicts (`backend/orchestrator.py`,
   via `call_brain_result()` → `with_caveman()`).

When adding a new model call, route it through `with_caveman()` /
`CAVEMAN_OUTPUT_INSTRUCTIONS` so terseness stays enforced everywhere.

## Caveman skills & commands (Claude Code)

These live under `.claude/` (skill `caveman/SKILL.md` with levels lite / full / ultra,
plus slash commands `/caveman`, `/caveman-commit`, `/caveman-review`, `/caveman-help`,
`/caveman-compress`; stop with "normal mode"). **`.claude/` is git-ignored** (see
`.gitignore`), so these files are a local, per-developer Claude Code setup — they are
**not** checked into the repo and may be absent in a fresh clone. The app's own terseness
does not depend on them; it comes from `CAVEMAN_OUTPUT_INSTRUCTIONS` in
[`backend/prompts.py`](backend/prompts.py).

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
