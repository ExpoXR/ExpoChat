# Ollma UI

TrueNAS-hosted supervisor for local Ollama workers plus Codex or Claude planning. Changes run in staged workspaces, pass independent verification, then apply behind a verified snapshot.

## Architecture

```text
Browser -> FastAPI web -> SQLite / snapshots / staged jobs
                    |-> mount-free Brain service -> Codex / Claude (planning)
                    |-> worker API -> Ollama (research, edits, verification)
```

- `backend/main.py`: app factory, authenticated HTTP/SSE API, static UI.
- `backend/orchestrator.py`: durable job queue, run state machine, DAG subtask execution.
- `backend/brain_io.py`: structured brain I/O — prompt builders, JSON extraction, typed results.
- `backend/worker.py`: credential-isolated Ollama agent loop.
- `backend/workspace.py`: staging, manifests, atomic snapshots, restore, storage reporting.
- `backend/workspace_tools.py`: shared path-safe file/search/check tools.
- `backend/plan_graph.py`: task-graph validation, dependency resolution, `suggested_model` pass-through.
- `backend/migrations.py`: ordered, idempotent SQLite migrations (12 migrations through Series B).
- `public/`: native ES-module workbench; no frontend build step.

Providers never receive original workspace mounts. Ollama workers edit `/jobs`; web service applies verified results after approval. Interactive chat uses read-only tools or bounded pinned-file context.

## Setup

```bash
cp .env.example .env
chmod 600 .env
make setup
make check
make up
make smoke
```

Open `http://127.0.0.1:31001` or configured reverse-proxy URL.

Required production values:

- `ADMIN_PASSWORD_HASH`: Argon2id admin password hash.
- `SESSION_SECRET`: long random session signing secret.
- `CREDENTIAL_ENCRYPTION_KEY`: Fernet key for stored provider credentials.
- `WORKER_TOKEN`: long random internal service token.

Rotate any value previously committed or copied from an unsafe example. Use HTTPS with `SECURE_COOKIE=true`, explicit `ALLOWED_ORIGINS`, and trusted `FORWARDED_ALLOW_IPS`.

## Developer Commands

```bash
make lint       # Ruff
make test       # Python unit/integration tests
make test-js    # Frontend unit tests (node --test)
make e2e        # isolated fake services + containerized Playwright
make check      # fast local gate
make logs       # follow Compose logs
make backup     # SQLite online backup
```

`make e2e` starts a temporary app/database and fake Ollama service on port `31002`. Browser dependencies live in the pinned Playwright image, not the host.

## Configuration

| Variable | Default | Purpose |
|---|---:|---|
| `OLLAMA_BASE_URL` | LAN Ollama URL | Model API |
| `ALLOWED_ROOTS` | `/workspace` | Comma-separated workspace roots |
| `COMMAND_TIMEOUT` | `120` | Maximum safe-check seconds |
| `SNAPSHOT_RETENTION_DAYS` | `30` | Tracked archive retention |
| `SNAPSHOT_MAX_BYTES` | `21474836480` | Maximum uncompressed source bytes |
| `SNAPSHOT_RESERVE_BYTES` | `2147483648` | Free-space reserve before snapshot |
| `ORPHAN_GRACE_HOURS` | `24` | Minimum orphan age before manual cleanup |
| `RUNNER_CONCURRENCY` | `1` | Concurrent durable job drainers (separate runs) |
| `WORKER_POOL_SIZE` | `1` | Subtasks a single run runs in parallel (1 = serialized; raise only when Ollama serves models concurrently) |
| `CHAT_CONTEXT_BYTES` | `120000` | Pinned chat context budget |

Brain memory budget (per-run context window for brain continuity) is a DB setting
(`brain_memory_budget`, default 4000 tokens). Configurable at runtime via settings API.

Keep concurrency at `1` until Ollama host has capacity for parallel model requests.

## Operations

```bash
docker compose ps
docker compose logs --tail=200 ollma-ui ollma-brain ollma-worker
curl -fsS http://127.0.0.1:31001/livez
curl -fsS http://127.0.0.1:31001/readyz
```

Storage view reports tracked, missing, partial, and orphan snapshot archives. Incomplete `.part` files older than grace period are removed automatically. Complete orphan archives are never deleted automatically; UI requires explicit archive selection and confirmation.

Before upgrade:

```bash
make backup
docker compose build
docker compose up -d
make smoke
```

Migrations run transactionally at startup. Restore a DB backup only while services are stopped. Workspace rollback stays available through run History while its snapshot archive exists.

## Troubleshooting

- `Unsafe or missing required configuration`: replace placeholder secrets in `.env`.
- `Snapshot source exceeds limit`: narrow target or intentionally raise `SNAPSHOT_MAX_BYTES`.
- `Insufficient snapshot storage`: free space or lower target size; reserve is intentionally conservative.
- E2E browser library errors: use `make e2e`, not host Playwright.
- Run stuck after restart: pending jobs and running subtasks recover automatically to pending; failed runs expose Resume.
- Subtask stuck in "running": `init_db` resets running subtasks to pending on startup. Already-done subtasks skipped on re-execution.
- Cancellation: web marks run cancelled and asks worker to close active Ollama request; verify worker connectivity if response remains active.

## Release Checklist

1. `make check` and `make e2e` pass.
2. `make backup` completes.
3. Review migrations and `.env.example` changes.
4. Build both images without cached application layers.
5. Deploy, then verify `/livez`, `/readyz`, login, model discovery, storage report, and one staged test run.
6. Confirm no new orphan or partial archives appear.
