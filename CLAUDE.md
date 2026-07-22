# Ollma Chat System

TrueNAS-hosted supervisor for local Ollama workers plus Codex/Claude planning.
FastAPI backend (`backend/`) + no-build ES-module frontend (`public/`).

## Architecture (where the AI runs)

```text
Browser -> FastAPI web -> SQLite / snapshots / staged jobs
                    |-> Brain service   -> Codex / Claude   (planning, verdicts)
                    |-> Worker API       -> Ollama agent     (research, edits, verification)
                    |-> /api/chats/.../message -> Ollama      (interactive chat)
```

- `backend/main.py`: HTTP/SSE API, static UI, interactive chat stream, file + `/api/fs/*` ops.
- `backend/orchestrator.py`: run state machine + all Brain (Codex/Claude) calls.
- `backend/worker.py`: credential-isolated Ollama agent loop.
- `backend/prompts.py`: shared caveman output instruction (see below).
- `public/`: workbench UI (chat, unified Supervisor Run center view, Explorer, Snapshots…).

## Caveman output mode — applied to ALL AI in the app

Every model call the app makes is prefixed with the compact caveman instruction from
[`backend/prompts.py`](backend/prompts.py) (`CAVEMAN_OUTPUT_INSTRUCTIONS` / `with_caveman()`),
so responses stay terse with full technical accuracy. The three entry points:

| AI path | Where it runs | Caveman applied at |
|---------|---------------|--------------------|
| Interactive chat | Ollama (read-only tools) | `backend/main.py` (chat system prompt) |
| Supervised worker | Ollama agent (research / implementation / verification) | `backend/worker.py` → `agent_system_prompt()` |
| Supervisor brain | Codex / Claude (plan + verdict) | `backend/orchestrator.py` → `call_brain_result()` → `with_caveman()` |

The app deliberately injects the **compact** instruction, not the full multi-level skill,
to keep token savings positive. The full skill and its intensity levels live in
`.claude/` (below) and drive Claude Code sessions on this repo.

## Caveman Skills (Claude Code, this repo)

Ultra-compressed communication mode with full technical accuracy.

| Command | What |
|---------|------|
| `/caveman` | Activate (full mode default) |
| `/caveman lite\|full\|ultra` | Switch intensity |
| `/caveman-commit` | Terse Conventional Commits message |
| `/caveman-review` | One-line PR comments: `L42: 🔴 bug: user null. Add guard.` |
| `/caveman-help` | Quick-reference card |
| `/caveman-compress <file>` | Compress .md file and preserve original backup |

Stop: "stop caveman" or "normal mode".
Skills live in `.claude/skills/` (canonical: `caveman/SKILL.md`) and expose matching slash
commands. **`.claude/` is git-ignored** (see `.gitignore`), so these are a
local, per-developer Claude Code setup — not checked into the repo and possibly absent in
a fresh clone. The app's own terseness comes from `CAVEMAN_OUTPUT_INSTRUCTIONS`
([`backend/prompts.py`](backend/prompts.py)), independent of these files.

## Developer commands

```bash
make lint       # Ruff
make test       # Python unit/integration tests (pytest)
make test-js    # Frontend unit tests (node --test)
make check      # fast local gate
make up         # docker compose up -d
make logs       # follow Compose logs
```
