import asyncio
import contextlib
import difflib
import hmac
import json
import logging
import os
import secrets
import shutil
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
from fastapi import APIRouter, Body, Depends, FastAPI, HTTPException, Query, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import db
from .config import settings
from .logging_utils import configure_logging
from .orchestrator import (
    approve_run,
    call_brain,
    cancel_run,
    create_run,
    discover_agents,
    edit_plan,
    redo_plan,
    resume_run,
    rollback_run,
    start_job_queue,
)
from .prompts import CAVEMAN_OUTPUT_INSTRUCTIONS
from .security import (
    create_session,
    destroy_session,
    encrypt_secret,
    rate_limit,
    require_user,
    session_for,
    verify_password,
)
from .workspace import (
    EXCLUDED_DIRS,
    _copy_ignore,
    allowed_path,
    cleanup_orphan_snapshots,
    cleanup_partial_snapshots,
    cleanup_snapshots,
    create_snapshot,
    restore_snapshot,
    storage_report,
)
from .workspace_tools import TOOL_DEFS, execute_tool, search_text

configure_logging()
log = logging.getLogger("ollma.web")
TERMINAL_STATES = {"completed", "failed", "cancelled", "rolled_back"}
MAX_TIMELINE_ENTRIES = 5000


class LoginBody(BaseModel):
    username: str
    password: str


class BrainBody(BaseModel):
    provider: str = Field(pattern="^(codex|claude|gemini)$")
    model: str = Field(min_length=1, max_length=200)
    api_key: str = Field(default="", max_length=20_000)
    enabled: bool = True


class AgentBody(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=200)
    roles: list[str] | None = None
    system_prompt: str | None = Field(default=None, max_length=40_000)
    priority: int | None = Field(default=None, ge=0, le=1000)
    role_scores: dict[str, int] | None = None
    enabled: bool | None = None


class AgentCreateBody(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    model: str = Field(min_length=1, max_length=200)
    roles: list[str] = Field(default_factory=lambda: ["research", "verification"])
    system_prompt: str = Field(default="", max_length=40_000)
    capabilities: list[str] = Field(default_factory=list)
    context_size: int = Field(default=0, ge=0)
    priority: int = Field(default=50, ge=0, le=1000)
    role_scores: dict[str, int] = Field(default_factory=dict)
    enabled: bool = True


class RunBody(BaseModel):
    task: str = Field(min_length=3, max_length=40_000)
    brain_provider: str = Field(pattern="^(codex|claude|gemini)$")
    target_path: str
    web_research: bool = False


class ApprovalBody(BaseModel):
    plan: str = Field(min_length=3, max_length=200_000)


class ChatBody(BaseModel):
    title: str = Field(default="New chat", min_length=1, max_length=200)
    model: str = Field(default="", max_length=200)
    target_path: str = Field(default="", max_length=4096)
    snapshot: bool = False


class ChatUpdateBody(BaseModel):
    title: str | None = Field(default=None, min_length=1, max_length=200)
    pinned: bool | None = None


class FsPathBody(BaseModel):
    path: str = Field(min_length=1, max_length=4096)


class FsRenameBody(BaseModel):
    path: str = Field(min_length=1, max_length=4096)
    new_path: str = Field(min_length=1, max_length=4096)


class FsCopyBody(BaseModel):
    src: str = Field(min_length=1, max_length=4096)
    dest: str = Field(min_length=1, max_length=4096)
    overwrite: bool = False


class MessageBody(BaseModel):
    content: str = Field(min_length=1, max_length=200_000)
    model: str = Field(min_length=1, max_length=200)
    target_path: str = Field(default="", max_length=4096)
    context_paths: list[str] = Field(default_factory=list, max_length=20)


class FileBody(BaseModel):
    path: str = Field(min_length=1, max_length=4096)
    content: str
    chat_id: str | None = None


class ConsoleBody(BaseModel):
    command: str
    args: list[str] = Field(default_factory=list, max_length=40)
    timeout: int | None = Field(default=None, ge=1)


class SnapshotCleanupBody(BaseModel):
    days: int = Field(default_factory=lambda: settings.snapshot_retention_days, ge=1)
    dry_run: bool = False
    cleanup_tracked: bool = True
    orphan_refs: list[str] = Field(default_factory=list, max_length=100)


def parse_json_fields(row: dict[str, Any], fields: list[str]) -> dict[str, Any]:
    for field in fields:
        if field in row and isinstance(row[field], str):
            with contextlib.suppress(json.JSONDecodeError):
                row[field.removesuffix("_json")] = json.loads(row[field])
    return row


def add_timeline(event_type: str, summary: str, path: str | None = None, chat_id: str | None = None, before: str | None = None, after: str | None = None, diff: str | None = None) -> None:
    before = before[:200_000] if before else None
    after = after[:200_000] if after else None
    diff = diff[:200_000] if diff else None
    with db.transaction() as conn:
        conn.execute(
            "insert into timeline(chat_id,event_type,path,summary,before,after,diff,created_at) values(?,?,?,?,?,?,?,?)",
            (chat_id, event_type, path, summary[:1000], before, after, diff, db.utcnow()),
        )
        conn.execute(
            "delete from timeline where id not in (select id from timeline order by id desc limit ?)",
            (MAX_TIMELINE_ENTRIES,),
        )


def list_item(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "name": path.name, "path": str(path), "is_dir": path.is_dir(), "size": stat.st_size,
        "modified_at": datetime.fromtimestamp(stat.st_mtime, UTC).isoformat(),
    }


def _paths_overlap(left: str | Path, right: str | Path) -> bool:
    left_path = Path(left).resolve(strict=False)
    right_path = Path(right).resolve(strict=False)
    return left_path == right_path or left_path in right_path.parents or right_path in left_path.parents


def _snapshot_conflicts_with_active_run(snapshot: dict[str, Any], active_runs: list[dict[str, Any]] | None = None) -> bool:
    active_runs = active_runs if active_runs is not None else db.all_rows(
        "select snapshot_id,target_path from runs where status not in ('completed','failed','cancelled','rolled_back')"
    )
    for run in active_runs:
        if run.get("snapshot_id") == snapshot.get("id") or _paths_overlap(snapshot["path"], run["target_path"]):
            return True
    return False


def workspace_summary(path: Path, max_files: int = 100) -> str:
    rows = [f"Workspace: {path}", "Visible files:"]
    count = 0
    for base, dirs, files in os.walk(path):
        dirs[:] = [name for name in dirs if name not in EXCLUDED_DIRS]
        if len(Path(base).relative_to(path).parts) > 4:
            dirs[:] = []
        for name in sorted(files):
            item = Path(base) / name
            with contextlib.suppress(OSError):
                rows.append(f"- {item.relative_to(path)} ({item.stat().st_size} bytes)")
                count += 1
            if count >= max_files:
                return "\n".join(rows + ["..."])
    return "\n".join(rows)


def pinned_context(root: Path, paths: list[str], byte_limit: int | None = None) -> str:
    remaining = byte_limit or settings.chat_context_bytes
    chunks = []
    for value in paths:
        target = allowed_path(value)
        if target != root and root not in target.parents:
            raise HTTPException(400, "Context path is outside chat workspace")
        if not target.is_file() or target.is_symlink():
            continue
        size = target.stat().st_size
        if size > remaining:
            chunks.append(f"Pinned file omitted (context limit): {target.relative_to(root)}")
            continue
        content = target.read_text(errors="replace")
        chunks.append(f"\n--- PINNED FILE: {target.relative_to(root)} ---\n{content}")
        remaining -= len(content.encode())
    return "\n".join(chunks)


@asynccontextmanager
async def lifespan(_: FastAPI):
    unsafe = []
    if settings.session_secret in {"", "change-this-session-secret"}:
        unsafe.append("SESSION_SECRET")
    if settings.worker_token in {"", "change-worker-token", "change-this-worker-token"}:
        unsafe.append("WORKER_TOKEN")
    if not settings.credential_key:
        unsafe.append("CREDENTIAL_ENCRYPTION_KEY")
    if not settings.admin_password_hash and settings.admin_password in {"", "change-me-now"}:
        unsafe.append("ADMIN_PASSWORD_HASH")
    if unsafe:
        raise RuntimeError("Unsafe or missing required configuration: " + ", ".join(unsafe))
    if not settings.admin_password_hash:
        log.warning("legacy_plaintext_admin_password_configured migrate_to_argon2id=true")
    db.init_db()
    for interrupted in db.all_rows(
        "select id,snapshot_id from runs where status='failed' and error='Interrupted by service restart' and snapshot_id is not null"
    ):
        try:
            restore_snapshot(interrupted["snapshot_id"])
            db.execute(
                "update runs set status='rolled_back',error='Interrupted during apply; snapshot restored',completed_at=?,updated_at=? where id=?",
                (db.utcnow(), db.utcnow(), interrupted["id"]),
            )
            db.add_event(interrupted["id"], "rollback.completed", "Interrupted apply recovered from snapshot")
        except Exception as exc:
            db.execute("update runs set error=? where id=?", (f"Interrupted recovery failed: {exc}"[:4000], interrupted["id"]))
    with contextlib.suppress(Exception):
        cleanup_partial_snapshots()
        cleanup_snapshots()
    start_job_queue()
    yield


system_router = APIRouter()
auth_router = APIRouter()
config_router = APIRouter()
run_router = APIRouter()
workspace_router = APIRouter()


async def request_context(request: Request, call_next):
    request_id = request.headers.get("x-request-id") or secrets.token_hex(8)
    started = time.monotonic()
    try:
        response = await call_next(request)
    except Exception:
        log.exception("request_failed", extra={"request_id": request_id, "method": request.method, "path": request.url.path})
        raise
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "same-origin"
    response.headers["Content-Security-Policy"] = "default-src 'self'; style-src 'self' 'unsafe-inline'; script-src 'self'; img-src 'self' data:; connect-src 'self'"
    log.info("request", extra={
        "request_id": request_id, "method": request.method, "path": request.url.path,
        "status": response.status_code, "duration_ms": int((time.monotonic() - started) * 1000),
    })
    return response


async def http_error(_: Request, exc: HTTPException):
    return JSONResponse({"error": exc.detail}, status_code=exc.status_code)


@system_router.get("/livez")
@system_router.get("/api/health")
def livez() -> dict[str, Any]:
    return {"ok": True, "service": "ollma-web"}


@system_router.api_route("/internal/ollama/{ollama_path:path}", methods=["GET", "POST"])
async def internal_ollama_proxy(ollama_path: str, request: Request):
    token = request.headers.get("x-worker-token", "")
    if not token or not hmac.compare_digest(token, settings.worker_token):
        raise HTTPException(401, "Invalid worker token")
    if ollama_path not in {"api/chat", "api/tags", "api/show", "api/version"}:
        raise HTTPException(404, "Unsupported Ollama route")
    body = await request.body()
    if len(body) > 2_000_000:
        raise HTTPException(413, "Ollama request too large")
    async with httpx.AsyncClient(timeout=320) as client:
        upstream = await client.request(
            request.method,
            f"{settings.ollama_url}/{ollama_path}",
            content=body or None,
            headers={"Content-Type": "application/json"},
        )
    return Response(
        content=upstream.content,
        status_code=upstream.status_code,
        media_type=upstream.headers.get("content-type", "application/json"),
    )


@system_router.get("/readyz")
def readyz() -> JSONResponse:
    checks: dict[str, Any] = {}
    try:
        checks["database"] = db.one("select 1 as ok")["ok"] == 1
        checks["storage"] = all(
            path.exists() and os.access(path, os.W_OK)
            for path in (settings.data_dir, settings.snapshot_dir, settings.jobs_dir)
        )
        with httpx.Client(timeout=3) as client:
            worker_response = client.get(settings.worker_url + "/healthz")
            worker_response.raise_for_status()
            checks["worker"] = bool(worker_response.json().get("ok"))
            brain_response = client.get(settings.brain_url + "/healthz")
            brain_response.raise_for_status()
            checks["brain"] = bool(brain_response.json().get("ok"))
            ollama_response = client.get(settings.ollama_url + "/api/version")
            ollama_response.raise_for_status()
            checks["ollama"] = bool(ollama_response.json().get("version"))
    except Exception as exc:
        checks["error"] = str(exc)
    ready = all(checks.get(name) is True for name in ("database", "storage", "worker", "brain", "ollama"))
    return JSONResponse({"ok": ready, "checks": checks}, 200 if ready else 503)


@auth_router.post("/api/auth/login")
def login(body: LoginBody, request: Request, response: Response):
    client = request.client.host if request.client else "unknown"
    rate_limit(f"login:{client}", 5)
    if body.username != settings.admin_user or not verify_password(body.password):
        raise HTTPException(401, "Invalid login")
    return create_session(response, request)


@auth_router.post("/api/auth/logout")
def logout(request: Request, response: Response):
    require_user(request)
    destroy_session(request, response)
    return {"ok": True}


@auth_router.get("/api/auth/me")
def me(request: Request):
    session = session_for(request)
    return {"user": session["user"], "csrf": session["csrf"]} if session else {"user": None, "csrf": None}


def _public_brain(row: dict[str, Any]) -> dict[str, Any]:
    environment_key = settings.environment_key(row.get("provider", ""))
    has_key = bool(row.get("key_ciphertext")) if row.get("source") == "stored" else bool(environment_key)
    return {
        key: row.get(key)
        for key in ("provider", "model", "source", "enabled", "validated_at", "last_error", "updated_at")
    } | {"linked": bool(row.get("enabled") and has_key)}


@config_router.get("/api/config")
def config(_: dict = Depends(require_user)):
    brains = {row["provider"]: _public_brain(row) for row in db.all_rows("select * from brain_configs")}
    return {
        "claude_enabled": bool(brains.get("claude", {}).get("linked")),
        "claude_model": brains.get("claude", {}).get("model", settings.claude_model),
        "openai_enabled": bool(brains.get("codex", {}).get("linked")),
        "openai_model": brains.get("codex", {}).get("model", settings.openai_model),
        "gemini_enabled": bool(brains.get("gemini", {}).get("linked")),
        "gemini_model": brains.get("gemini", {}).get("model", settings.gemini_model),
        "ollama_url": settings.ollama_url,
        "allowed_roots": [str(root) for root in settings.allowed_roots],
        "credential_storage_enabled": bool(settings.credential_key),
    }


@config_router.get("/api/brains")
def brains(_: dict = Depends(require_user)):
    return {"brains": [_public_brain(row) for row in db.all_rows("select * from brain_configs order by provider")]}


@config_router.put("/api/brains")
def save_brain(body: BrainBody, _: dict = Depends(require_user)):
    current = db.one("select * from brain_configs where provider=?", (body.provider,))
    ciphertext = current.get("key_ciphertext") if current else None
    source = current.get("source", "environment") if current else "environment"
    if body.api_key:
        ciphertext = encrypt_secret(body.api_key)
        source = "stored"
    if not body.enabled:
        ciphertext = None
        source = "environment"
    environment_key = settings.environment_key(body.provider)
    if body.enabled and not (ciphertext if source == "stored" else environment_key):
        raise HTTPException(400, "API key required before linking provider")
    model = body.model.strip()
    if not model:
        raise HTTPException(400, "Model is required")
    db.execute(
        "insert into brain_configs(provider,model,key_ciphertext,source,enabled,updated_at) values(?,?,?,?,?,?) "
        "on conflict(provider) do update set model=excluded.model,key_ciphertext=excluded.key_ciphertext,source=excluded.source,enabled=excluded.enabled,validated_at=null,last_error=null,updated_at=excluded.updated_at",
        (body.provider, model, ciphertext, source, int(body.enabled), db.utcnow()),
    )
    return _public_brain(db.one("select * from brain_configs where provider=?", (body.provider,)) or {})


@config_router.post("/api/brains/{provider}/validate")
def validate_brain(provider: str, _: dict = Depends(require_user)):
    if provider not in {"codex", "claude"}:
        raise HTTPException(404, "Unknown provider")
    try:
        content = call_brain(provider, "Reply with exactly: OK", False, timeout=300)
        db.execute("update brain_configs set validated_at=?,last_error=null where provider=?", (db.utcnow(), provider))
        return {"ok": True, "reply": content[:100]}
    except Exception as exc:
        db.execute("update brain_configs set last_error=? where provider=?", (str(exc)[:1000], provider))
        raise HTTPException(502, str(exc)) from exc


@config_router.get("/api/agents")
def agents(_: dict = Depends(require_user)):
    rows = [parse_json_fields(row, ["roles_json", "capabilities_json", "role_scores_json"]) for row in db.all_rows("select * from agent_profiles order by priority desc,name")]
    return {"agents": rows}


@config_router.post("/api/agents")
def agent_create(body: AgentCreateBody, _: dict = Depends(require_user)):
    valid_roles = {"research", "implementation", "verification"}
    if not set(body.roles).issubset(valid_roles):
        raise HTTPException(400, "Invalid agent role")
    if "implementation" in body.roles and "tools" not in body.capabilities:
        raise HTTPException(400, "Implementation agents require tools capability")
    if not set(body.role_scores).issubset(valid_roles) or any(not 0 <= score <= 1000 for score in body.role_scores.values()):
        raise HTTPException(400, "Invalid agent role score")
    if not body.name.strip() or not body.model.strip():
        raise HTTPException(400, "Agent name and model are required")
    agent_id = "agent-" + secrets.token_hex(8)
    now = db.utcnow()
    try:
        db.execute(
            "insert into agent_profiles(id,name,model,roles_json,system_prompt,capabilities_json,context_size,priority,role_scores_json,enabled,discovered_at,updated_at) values(?,?,?,?,?,?,?,?,?,?,?,?)",
            (agent_id, body.name.strip(), body.model.strip(), json.dumps(body.roles), body.system_prompt, json.dumps(body.capabilities), body.context_size, body.priority, json.dumps(body.role_scores), int(body.enabled), now, now),
        )
    except Exception as exc:
        raise HTTPException(409, "Agent model already exists") from exc
    return parse_json_fields(db.one("select * from agent_profiles where id=?", (agent_id,)) or {}, ["roles_json", "capabilities_json", "role_scores_json"])


@config_router.post("/api/agents/discover")
def agents_discover(_: dict = Depends(require_user)):
    try:
        rows = discover_agents()
        return {"agents": [parse_json_fields(row, ["roles_json", "capabilities_json", "role_scores_json"]) for row in rows]}
    except Exception as exc:
        raise HTTPException(502, f"Ollama discovery failed: {exc}") from exc


@config_router.patch("/api/agents/{agent_id}")
def agent_update(agent_id: str, body: AgentBody, _: dict = Depends(require_user)):
    row = db.one("select * from agent_profiles where id=?", (agent_id,))
    if not row:
        raise HTTPException(404, "Agent not found")
    roles = body.roles if body.roles is not None else json.loads(row["roles_json"] or "[]")
    if not set(roles).issubset({"research", "implementation", "verification"}):
        raise HTTPException(400, "Invalid agent role")
    if "implementation" in roles and "tools" not in json.loads(row["capabilities_json"] or "[]"):
        raise HTTPException(400, "Implementation agents require tools capability")
    if body.role_scores is not None and (
        not set(body.role_scores).issubset({"research", "implementation", "verification"})
        or any(not 0 <= score <= 1000 for score in body.role_scores.values())
    ):
        raise HTTPException(400, "Invalid agent role score")
    if body.name is not None and not body.name.strip():
        raise HTTPException(400, "Agent name is required")
    values = {
        "name": body.name.strip() if body.name is not None else row["name"],
        "roles_json": json.dumps(roles),
        "system_prompt": body.system_prompt if body.system_prompt is not None else row["system_prompt"],
        "priority": body.priority if body.priority is not None else row["priority"],
        "role_scores_json": json.dumps(body.role_scores) if body.role_scores is not None else row["role_scores_json"],
        "enabled": int(body.enabled) if body.enabled is not None else row["enabled"],
    }
    db.execute(
        "update agent_profiles set name=?,roles_json=?,system_prompt=?,priority=?,role_scores_json=?,enabled=?,updated_at=? where id=?",
        (*values.values(), db.utcnow(), agent_id),
    )
    return parse_json_fields(db.one("select * from agent_profiles where id=?", (agent_id,)) or {}, ["roles_json", "capabilities_json", "role_scores_json"])


@run_router.post("/api/runs")
def run_create(body: RunBody, session: dict = Depends(require_user)):
    rate_limit(f"runs:{session['user']}", 5)
    target = allowed_path(body.target_path)
    if not target.is_dir():
        raise HTTPException(400, "Run target must be a directory")
    try:
        return create_run(body.task.strip(), body.brain_provider, target, body.web_research)
    except Exception as exc:
        raise HTTPException(400, str(exc)) from exc


@run_router.get("/api/runs")
def runs(cursor: int = Query(default=0, ge=0), limit: int = Query(default=50, ge=1, le=100), _: dict = Depends(require_user)):
    fields = (
        "id,task,brain_provider,brain_model,target_path,status,research_agent_id,implementation_agent_id,"
        "repair_count,error,created_at,updated_at,approved_at,completed_at,snapshot_id,usage_json"
    )
    rows = db.all_rows(f"select {fields} from runs order by updated_at desc limit ? offset ?", (limit + 1, cursor))
    has_more = len(rows) > limit
    return {"runs": rows[:limit], "next_cursor": cursor + limit if has_more else None}


@run_router.get("/api/runs/{run_id}")
def run_get(run_id: str, _: dict = Depends(require_user)):
    row = db.one("select * from runs where id=?", (run_id,))
    if not row:
        raise HTTPException(404, "Run not found")
    parse_json_fields(row, ["usage_json"])
    row["events"] = db.all_rows("select * from run_events where run_id=? order by id", (run_id,))
    row["artifacts"] = db.all_rows("select id,kind,name,created_at from run_artifacts where run_id=? order by id", (run_id,))
    row["history"] = db.one("select * from history_snippets where run_id=?", (run_id,))
    row["approvals"] = db.all_rows("select * from run_approvals where run_id=? order by id", (run_id,))
    row["verification_results"] = db.all_rows("select * from verification_results where run_id=? order by id", (run_id,))
    row["jobs"] = db.all_rows(
        "select id,job_type,status,attempts,error,started_at,completed_at,cancel_requested_at,created_at,updated_at "
        "from jobs where run_id=? order by id",
        (run_id,),
    )
    pending = db.one("select id from jobs where run_id=? and status='pending' order by id limit 1", (run_id,))
    row["queue_position"] = None
    if pending:
        position = db.one("select count(*) as value from jobs where status='pending' and id<=?", (pending["id"],))
        row["queue_position"] = position["value"] if position else None
    agent_ids = [value for value in (row.get("research_agent_id"), row.get("implementation_agent_id")) if value]
    row["selected_agents"] = [agent for agent_id in agent_ids if (agent := db.one("select id,name,model from agent_profiles where id=?", (agent_id,)))]
    return row


@run_router.get("/api/runs/{run_id}/artifacts/{artifact_id}")
def artifact_get(run_id: str, artifact_id: int, _: dict = Depends(require_user)):
    row = db.one("select * from run_artifacts where run_id=? and id=?", (run_id, artifact_id))
    if not row:
        raise HTTPException(404, "Artifact not found")
    return row


@run_router.get("/api/runs/{run_id}/events")
async def run_events(run_id: str, request: Request, after: int = Query(default=0), _: dict = Depends(require_user)):
    if not db.one("select id from runs where id=?", (run_id,)):
        raise HTTPException(404, "Run not found")
    last_header = request.headers.get("last-event-id")
    cursor = max(after, int(last_header) if last_header and last_header.isdigit() else 0)

    async def stream() -> AsyncIterator[str]:
        nonlocal cursor
        idle = 0
        while not await request.is_disconnected():
            rows = db.all_rows("select * from run_events where run_id=? and id>? order by id", (run_id, cursor))
            if rows:
                idle = 0
                for row in rows:
                    cursor = row["id"]
                    yield f"id: {cursor}\nevent: {row['event_type']}\ndata: {json.dumps(row, ensure_ascii=False)}\n\n"
            else:
                idle += 1
                if idle % 15 == 0:
                    yield ": keepalive\n\n"
            run = db.one("select status from runs where id=?", (run_id,))
            if run and run["status"] in TERMINAL_STATES and not rows:
                # Final drain: an event may have landed after the last query and
                # before this terminal-status read; deliver it before closing.
                for row in db.all_rows("select * from run_events where run_id=? and id>? order by id", (run_id, cursor)):
                    cursor = row["id"]
                    yield f"id: {cursor}\nevent: {row['event_type']}\ndata: {json.dumps(row, ensure_ascii=False)}\n\n"
                break
            await asyncio.sleep(1)

    return StreamingResponse(stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@run_router.post("/api/runs/{run_id}/approve")
def run_approve(run_id: str, body: ApprovalBody, _: dict = Depends(require_user)):
    try:
        return approve_run(run_id, body.plan.strip())
    except Exception as exc:
        message = str(exc)
        raise HTTPException(409 if "changed" in message else 400, message) from exc


@run_router.put("/api/runs/{run_id}/plan")
def run_plan_edit(run_id: str, body: ApprovalBody, _: dict = Depends(require_user)):
    try:
        return edit_plan(run_id, body.plan.strip())
    except Exception as exc:
        raise HTTPException(400, str(exc)) from exc


@run_router.post("/api/runs/{run_id}/redo")
def run_plan_redo(run_id: str, _: dict = Depends(require_user)):
    try:
        return redo_plan(run_id)
    except Exception as exc:
        raise HTTPException(400, str(exc)) from exc


@run_router.post("/api/runs/{run_id}/reject")
@run_router.post("/api/runs/{run_id}/cancel")
def run_cancel(run_id: str, _: dict = Depends(require_user)):
    try:
        cancel_run(run_id)
        return {"ok": True}
    except Exception as exc:
        raise HTTPException(400, str(exc)) from exc


@run_router.post("/api/runs/{run_id}/resume")
def run_resume(run_id: str, _: dict = Depends(require_user)):
    try:
        resume_run(run_id)
        return {"ok": True}
    except Exception as exc:
        raise HTTPException(400, str(exc)) from exc


@run_router.post("/api/runs/{run_id}/rollback")
def run_rollback(run_id: str, _: dict = Depends(require_user)):
    try:
        rollback_run(run_id)
        return {"ok": True}
    except Exception as exc:
        raise HTTPException(400, str(exc)) from exc


@run_router.post("/api/runs/{run_id}/console")
def run_console(run_id: str, payload: ConsoleBody, _: dict = Depends(require_user)):
    run = db.one("select * from runs where id=?", (run_id,))
    if not run:
        raise HTTPException(404, "Run not found")
    command = payload.command.strip()
    args = payload.args
    if not command:
        raise HTTPException(400, "command required")
    started = time.monotonic()
    try:
        with httpx.Client(timeout=(payload.timeout or settings.command_timeout) + 10) as client:
            response = client.post(
                settings.worker_url + "/check",
                headers={"X-Worker-Token": settings.worker_token},
                json={
                    "run_id": run_id,
                    "workspace": "workspace",
                    "command": command,
                    "args": args,
                    "timeout": payload.timeout or settings.command_timeout,
                },
            )
            response.raise_for_status()
            result = response.json()
    except httpx.HTTPError as exc:
        raise HTTPException(502, f"Worker console failed: {exc}") from exc
    result["duration_ms"] = int((time.monotonic() - started) * 1000)
    return result


@config_router.get("/api/models")
def models(_: dict = Depends(require_user)):
    try:
        with httpx.Client(timeout=10) as client:
            response = client.get(settings.ollama_url + "/api/tags")
            response.raise_for_status()
            return response.json()
    except Exception as exc:
        raise HTTPException(502, f"Ollama error: {exc}") from exc


@config_router.get("/api/status")
def status_api(_: dict = Depends(require_user)):
    try:
        with httpx.Client(timeout=5) as client:
            response = client.get(settings.ollama_url + "/api/version")
            response.raise_for_status()
            version = response.json()
        return {"ollama": version, "allowed_roots": [str(root) for root in settings.allowed_roots]}
    except Exception as exc:
        raise HTTPException(502, f"Ollama error: {exc}") from exc


@workspace_router.get("/api/chats")
def chats(cursor: int = Query(default=0, ge=0), limit: int = Query(default=50, ge=1, le=100), _: dict = Depends(require_user)):
    rows = db.all_rows("select * from chats order by pinned desc, updated_at desc limit ? offset ?", (limit + 1, cursor))
    has_more = len(rows) > limit
    return {"chats": rows[:limit], "next_cursor": cursor + limit if has_more else None}


@workspace_router.post("/api/chats")
def chat_create(body: ChatBody, _: dict = Depends(require_user)):
    target_path = allowed_path(body.target_path) if body.target_path else None
    if target_path and not target_path.is_dir():
        raise HTTPException(400, "Chat target must be a directory")
    target = str(target_path) if target_path else ""
    chat_id = secrets.token_hex(12)
    now = db.utcnow()
    db.execute("insert into chats(id,title,model,target_path,created_at,updated_at) values(?,?,?,?,?,?)", (chat_id, body.title, body.model, target, now, now))
    snap = create_snapshot(Path(target), chat_id) if target and body.snapshot else None
    add_timeline("chat", "New chat created", target or None, chat_id)
    return {"chat": db.one("select * from chats where id=?", (chat_id,)), "snapshot": snap}


@workspace_router.patch("/api/chats/{chat_id}")
def chat_update(chat_id: str, body: ChatUpdateBody, _: dict = Depends(require_user)):
    chat = db.one("select * from chats where id=?", (chat_id,))
    if not chat:
        raise HTTPException(404, "Chat not found")
    title = body.title.strip() if body.title is not None else chat["title"]
    if not title:
        raise HTTPException(400, "Title cannot be empty")
    pinned = int(body.pinned) if body.pinned is not None else chat["pinned"]
    db.execute(
        "update chats set title=?,pinned=?,updated_at=? where id=?",
        (title, pinned, db.utcnow(), chat_id),
    )
    return db.one("select * from chats where id=?", (chat_id,))


@workspace_router.delete("/api/chats/{chat_id}")
def chat_delete(chat_id: str, _: dict = Depends(require_user)):
    if not db.one("select id from chats where id=?", (chat_id,)):
        raise HTTPException(404, "Chat not found")
    db.execute("delete from chats where id=?", (chat_id,))
    return {"ok": True}


@workspace_router.post("/api/chats/{chat_id}/duplicate")
def chat_duplicate(chat_id: str, _: dict = Depends(require_user)):
    source = db.one("select * from chats where id=?", (chat_id,))
    if not source:
        raise HTTPException(404, "Chat not found")
    new_id = secrets.token_hex(12)
    now = db.utcnow()
    title = f"{source['title']} (copy)"[:200]
    db.execute(
        "insert into chats(id,title,model,target_path,created_at,updated_at) values(?,?,?,?,?,?)",
        (new_id, title, source["model"], source["target_path"], now, now),
    )
    return {"chat": db.one("select * from chats where id=?", (new_id,))}


@workspace_router.get("/api/chats/{chat_id}/messages")
def chat_messages(chat_id: str, _: dict = Depends(require_user)):
    chat = db.one("select * from chats where id=?", (chat_id,))
    if not chat:
        raise HTTPException(404, "Chat not found")
    return {"chat": chat, "messages": db.all_rows("select * from messages where chat_id=? order by id", (chat_id,))}


@workspace_router.post("/api/chats/{chat_id}/message")
def chat_message(chat_id: str, body: MessageBody, _: dict = Depends(require_user)):
    chat = db.one("select * from chats where id=?", (chat_id,))
    if not chat:
        raise HTTPException(404, "Chat not found")
    target = allowed_path(body.target_path or chat["target_path"]) if (body.target_path or chat["target_path"]) else None
    if target and not target.is_dir():
        raise HTTPException(400, "Chat target must be a directory")
    db.execute("insert into messages(chat_id,role,content,created_at) values(?,?,?,?)", (chat_id, "user", body.content, db.utcnow()))
    history = db.all_rows("select role,content from messages where chat_id=? order by id", (chat_id,))
    system = (
        "You are an expert software assistant. Inspect supplied workspace with read-only tools before answering. "
        "Do not claim direct edits.\n\n" + CAVEMAN_OUTPUT_INSTRUCTIONS
    )
    if target:
        system += "\n\n" + workspace_summary(target) + pinned_context(target, body.context_paths)
    messages = [{"role": "system", "content": system}, *history]

    def stream():
        answer: list[str] = []
        try:
            with httpx.Client(timeout=300) as client:
                yield f"event: phase\ndata: {json.dumps({'phase': 'context', 'message': 'Inspecting workspace'})}\n\n"
                supports_tools = False
                if target:
                    with contextlib.suppress(Exception):
                        info = client.post(settings.ollama_url + "/api/show", json={"model": body.model}).json()
                        supports_tools = "tools" in (info.get("capabilities") or [])
                if target and supports_tools:
                    read_tools = [tool for tool in TOOL_DEFS if tool["function"]["name"] in {"list_files", "read_file", "search_files"}]
                    for _turn in range(8):
                        response = client.post(
                            settings.ollama_url + "/api/chat",
                            json={"model": body.model, "messages": messages, "tools": read_tools, "stream": False},
                        )
                        response.raise_for_status()
                        message = response.json().get("message", {})
                        messages.append(message)
                        content = message.get("content", "")
                        if content:
                            answer.append(content)
                            yield f"data: {json.dumps({'token': content}, ensure_ascii=False)}\n\n"
                        calls = message.get("tool_calls") or []
                        if not calls:
                            break
                        for call in calls:
                            function = call.get("function", {})
                            name = function.get("name", "")
                            arguments = function.get("arguments") or {}
                            if isinstance(arguments, str):
                                with contextlib.suppress(json.JSONDecodeError):
                                    arguments = json.loads(arguments)
                            if not isinstance(arguments, dict):
                                arguments = {}
                            result = execute_tool(target, name, arguments, False)
                            yield f"event: tool\ndata: {json.dumps({'tool': name, 'phase': 'tool'}, ensure_ascii=False)}\n\n"
                            messages.append({"role": "tool", "tool_name": name, "content": result})
                    else:
                        raise RuntimeError("Chat tool turn limit reached")
                else:
                    with client.stream("POST", settings.ollama_url + "/api/chat", json={"model": body.model, "messages": messages, "stream": True}) as response:
                        response.raise_for_status()
                        for line in response.iter_lines():
                            if not line:
                                continue
                            chunk = json.loads(line)
                            token = chunk.get("message", {}).get("content", "")
                            if token:
                                answer.append(token)
                                yield f"data: {json.dumps({'token': token}, ensure_ascii=False)}\n\n"
                            if chunk.get("done"):
                                break
        except Exception as exc:
            yield f"data: {json.dumps({'error': str(exc)}, ensure_ascii=False)}\n\n"
        full = "".join(answer)
        if full:
            db.execute("insert into messages(chat_id,role,content,created_at) values(?,?,?,?)", (chat_id, "assistant", full, db.utcnow()))
            db.execute("update chats set model=?,target_path=?,updated_at=?,title=case when title='New chat' then ? else title end where id=?", (body.model, str(target or ""), db.utcnow(), body.content[:70], chat_id))
        yield f"data: {json.dumps({'done': True, 'workspace': str(target or '')})}\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream", headers={"X-Accel-Buffering": "no"})


@workspace_router.get("/api/files")
def files(path: str = Query(default="/"), _: dict = Depends(require_user)):
    if path == "/" or not path:
        items = [list_item(root) for root in settings.allowed_roots if root.exists()]
        return {"path": "/", "is_dir": True, "roots": [str(root) for root in settings.allowed_roots], "items": items}
    target = allowed_path(path)
    items = []
    if target.is_dir():
        for child in sorted(target.iterdir(), key=lambda item: (not item.is_dir(), item.name.lower())):
            if child.name in EXCLUDED_DIRS:
                continue
            with contextlib.suppress(OSError):
                items.append(list_item(child))
    return {"path": str(target), "is_dir": target.is_dir(), "roots": [str(root) for root in settings.allowed_roots], "items": items}


@workspace_router.get("/api/file")
def file_get(path: str, _: dict = Depends(require_user)):
    target = allowed_path(path)
    if not target.is_file():
        raise HTTPException(400, "Path is not a file")
    if target.stat().st_size > 5_000_000:
        raise HTTPException(413, "File exceeds editor limit")
    return {"path": str(target), "content": target.read_text(errors="replace")}


@workspace_router.put("/api/file")
def file_put(body: FileBody, _: dict = Depends(require_user)):
    if len(body.content.encode()) > 5_000_000:
        raise HTTPException(413, "File exceeds editor limit")
    target = allowed_path(body.path, must_exist=False)
    before = target.read_text(errors="replace") if target.exists() else ""
    snapshot = create_snapshot(target if target.exists() else target.parent, body.chat_id)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(body.content)
    diff = "".join(difflib.unified_diff(before.splitlines(True), body.content.splitlines(True), fromfile=f"{target} original", tofile=f"{target} changed"))
    add_timeline("file_change", f"Changed {target.name}", str(target), body.chat_id, before, body.content, diff)
    return {"ok": True, "snapshot": snapshot, "diff": diff}


def _is_allowed_root(path: Path) -> bool:
    return any(path == root for root in settings.allowed_roots)


def _path_bytes(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    total = 0
    for base, dirs, files in os.walk(path):
        dirs[:] = [name for name in dirs if name not in EXCLUDED_DIRS]
        for name in files:
            item = Path(base) / name
            with contextlib.suppress(OSError):
                if not item.is_symlink():
                    total += item.stat().st_size
    return total


@workspace_router.post("/api/fs/folder")
def fs_folder(body: FsPathBody, _: dict = Depends(require_user)):
    target = allowed_path(body.path, must_exist=False)
    if target.exists():
        raise HTTPException(409, "Path already exists")
    target.mkdir(parents=True, exist_ok=False)
    add_timeline("fs_mkdir", f"Created folder {target.name}", str(target))
    return {"ok": True, "path": str(target)}


@workspace_router.post("/api/fs/rename")
def fs_rename(body: FsRenameBody, _: dict = Depends(require_user)):
    source = allowed_path(body.path)
    dest = allowed_path(body.new_path, must_exist=False)
    if source.is_symlink():
        raise HTTPException(400, "Cannot move a symlink")
    if _is_allowed_root(source):
        raise HTTPException(400, "Cannot move a workspace root")
    if dest == source:
        raise HTTPException(400, "Source and destination are identical")
    if source == dest or source in dest.parents:
        raise HTTPException(400, "Destination is inside source")
    if dest.exists():
        raise HTTPException(409, "Destination already exists")
    snapshot = create_snapshot(source)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(source), str(dest))
    add_timeline("fs_rename", f"Moved {source.name} → {dest.name}", str(dest), diff=str(source))
    return {"ok": True, "path": str(dest), "snapshot": snapshot}


@workspace_router.post("/api/fs/copy")
def fs_copy(body: FsCopyBody, _: dict = Depends(require_user)):
    source = allowed_path(body.src)
    dest = allowed_path(body.dest, must_exist=False)
    if source.is_symlink():
        raise HTTPException(400, "Cannot copy a symlink")
    if dest == source or source in dest.parents:
        raise HTTPException(400, "Destination is inside source")
    if _path_bytes(source) > settings.snapshot_max_bytes:
        raise HTTPException(413, "Copy source exceeds size limit")
    snapshot = None
    if dest.exists():
        if not body.overwrite:
            raise HTTPException(409, "Destination already exists")
        if _is_allowed_root(dest):
            raise HTTPException(400, "Cannot overwrite a workspace root")
        snapshot = create_snapshot(dest)
        if dest.is_dir():
            shutil.rmtree(dest)
        else:
            dest.unlink()
    dest.parent.mkdir(parents=True, exist_ok=True)
    if source.is_dir():
        shutil.copytree(source, dest, ignore=_copy_ignore)
    else:
        shutil.copy2(source, dest)
    add_timeline("fs_copy", f"Copied {source.name} → {dest.name}", str(dest), diff=str(source))
    return {"ok": True, "path": str(dest), "snapshot": snapshot}


@workspace_router.post("/api/fs/delete")
def fs_delete(body: FsPathBody, _: dict = Depends(require_user)):
    target = allowed_path(body.path)
    if _is_allowed_root(target):
        raise HTTPException(400, "Cannot delete a workspace root")
    snapshot = create_snapshot(target)
    if target.is_dir() and not target.is_symlink():
        shutil.rmtree(target)
    else:
        target.unlink()
    add_timeline("fs_delete", f"Deleted {target.name}", str(target), diff=str(target))
    return {"ok": True, "path": str(target), "snapshot": snapshot}


@workspace_router.get("/api/search")
def search(root: str, q: str, cursor: int = Query(default=0, ge=0), limit: int = Query(default=50, ge=1, le=100), _: dict = Depends(require_user)):
    target = allowed_path(root)
    if not target.is_dir():
        raise HTTPException(400, "Search root must be a directory")
    return search_text(target, q, cursor, limit)


@workspace_router.get("/api/snapshots")
def snapshots(cursor: int = Query(default=0, ge=0), limit: int = Query(default=50, ge=1, le=100), _: dict = Depends(require_user)):
    rows = db.all_rows("select * from snapshots order by created_at desc limit ? offset ?", (limit + 1, cursor))
    has_more = len(rows) > limit
    visible = rows[:limit]
    active_runs = db.all_rows(
        "select snapshot_id,target_path from runs where status not in ('completed','failed','cancelled','rolled_back')"
    )
    for row in visible:
        row["protected"] = _snapshot_conflicts_with_active_run(row, active_runs)
    return {"snapshots": visible, "next_cursor": cursor + limit if has_more else None}


@workspace_router.post("/api/snapshots")
def snapshot_create(payload: dict[str, Any] = Body(...), _: dict = Depends(require_user)):
    snapshot = create_snapshot(allowed_path(str(payload.get("path", ""))), payload.get("chat_id"))
    add_timeline("snapshot", "Snapshot created", snapshot["path"], payload.get("chat_id"), diff=snapshot["ref"])
    return {"snapshot": snapshot}


@workspace_router.post("/api/snapshots/{snapshot_id}/restore")
def snapshot_restore(snapshot_id: str, _: dict = Depends(require_user)):
    row = db.one("select * from snapshots where id=?", (snapshot_id,))
    if not row:
        raise HTTPException(404, "Snapshot not found")
    if _snapshot_conflicts_with_active_run(row):
        raise HTTPException(409, "Snapshot is protected by an active run")
    restored = restore_snapshot(snapshot_id)
    add_timeline("snapshot_restore", "Snapshot restored", restored["path"], diff=restored["ref"])
    return {"snapshot": restored, "ok": True}


@workspace_router.delete("/api/snapshots/{snapshot_id}")
def snapshot_delete(snapshot_id: str, _: dict = Depends(require_user)):
    row = db.one("select * from snapshots where id=?", (snapshot_id,))
    if not row:
        raise HTTPException(404, "Snapshot not found")
    active = db.one(
        "select id from runs where snapshot_id=? and status not in ('completed','failed','cancelled','rolled_back') limit 1",
        (snapshot_id,),
    )
    if active:
        raise HTTPException(409, "Snapshot is protected by an active run")
    (settings.snapshot_dir / Path(row["ref"]).name).unlink(missing_ok=True)
    db.execute(
        "update snapshots set status='deleted',archive_deleted_at=? where id=?",
        (db.utcnow(), snapshot_id),
    )
    return {"ok": True}


@workspace_router.post("/api/snapshots/cleanup")
def snapshot_cleanup(payload: SnapshotCleanupBody | None = Body(default=None), _: dict = Depends(require_user)):
    payload = payload or SnapshotCleanupBody()
    dry_run = payload.dry_run
    tracked = 0
    if payload.cleanup_tracked:
        tracked = cleanup_snapshots(payload.days, dry_run=dry_run)
    orphans = cleanup_orphan_snapshots(payload.orphan_refs, dry_run=dry_run)
    return {"deleted": 0 if dry_run else tracked + len(orphans), "tracked": tracked, "orphans": orphans, "dry_run": dry_run}


@workspace_router.get("/api/maintenance/storage")
def maintenance_storage(_: dict = Depends(require_user)):
    return storage_report()


@workspace_router.get("/api/timeline")
def timeline(cursor: int = Query(default=0, ge=0), limit: int = Query(default=50, ge=1, le=100), _: dict = Depends(require_user)):
    rows = db.all_rows(
        "select id,chat_id,event_type,path,summary,diff,created_at from timeline order by created_at desc limit ? offset ?",
        (limit + 1, cursor),
    )
    has_more = len(rows) > limit
    return {"timeline": rows[:limit], "next_cursor": cursor + limit if has_more else None}


def create_app() -> FastAPI:
    created = FastAPI(title="Ollma UI", version="1.1.0", lifespan=lifespan, docs_url=None, redoc_url=None)
    if settings.allowed_origins:
        created.add_middleware(
            CORSMiddleware, allow_origins=list(settings.allowed_origins), allow_credentials=True,
            allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
            allow_headers=["Content-Type", "X-CSRF-Token", "Last-Event-ID"],
        )
    created.middleware("http")(request_context)
    created.add_exception_handler(HTTPException, http_error)
    for router in (system_router, auth_router, config_router, run_router, workspace_router):
        created.include_router(router)
    if not settings.public_dir.exists():
        raise RuntimeError(f"Public directory missing: {settings.public_dir}")
    created.mount("/", StaticFiles(directory=settings.public_dir, html=True), name="public")
    return created


app = create_app()
