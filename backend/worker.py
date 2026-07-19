import asyncio
import json
import threading
import time
from pathlib import Path
from typing import Any, Literal

import httpx
from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from .config import settings
from .logging_utils import configure_logging
from .workspace_tools import TOOL_DEFS, execute_tool

configure_logging()
app = FastAPI(title="Ollma Isolated Worker", docs_url=None, redoc_url=None)
JOBS_ROOT = settings.jobs_dir.resolve()
_cancel_events: dict[str, asyncio.Event] = {}
_active_tasks: dict[str, asyncio.Task[Any]] = {}
_cancel_lock = threading.Lock()


class WorkRequest(BaseModel):
    run_id: str = Field(pattern=r"^[a-f0-9-]{8,64}$")
    workspace: str = "workspace"
    model: str
    mode: Literal["research", "implementation", "verification", "console"]
    task: str
    max_turns: int = Field(default=24, ge=1, le=40)


def authorize(x_worker_token: str = Header(default="")) -> None:
    if not settings.worker_token or x_worker_token != settings.worker_token:
        raise HTTPException(401, "Invalid worker token")


def workspace_for(request: WorkRequest) -> Path:
    path = (JOBS_ROOT / request.run_id / request.workspace).resolve()
    if JOBS_ROOT not in path.parents or not path.exists() or not path.is_dir():
        raise HTTPException(404, "Staged workspace not found")
    return path


async def agent_loop(request: WorkRequest, root: Path, cancelled: asyncio.Event | None = None) -> dict[str, Any]:
    writable = request.mode == "implementation"
    tools = TOOL_DEFS if writable else [tool for tool in TOOL_DEFS if tool["function"]["name"] not in {"write_file", "replace_text", "delete_file"}]
    system = {
        "research": "Inspect workspace deeply. Use tools before answering. Return architecture, relevant files, risks, tests, and implementation advice. Never edit.",
        "implementation": "Implement approved task fully in staged workspace. Use precise file tools, run relevant checks, fix failures, then summarize changed files and evidence.",
        "verification": "Independently verify implementation. Inspect files and run safe checks. Return PASS or FAIL first, then concrete evidence and defects. Never edit.",
        "console": "Run requested safe check and return exact output. Never edit.",
    }[request.mode]
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system + " Workspace root is current staged directory."},
        {"role": "user", "content": request.task},
    ]
    events: list[dict[str, Any]] = []
    final = ""
    usage = {"prompt_eval_count": 0, "eval_count": 0, "model_duration_ns": 0, "tool_calls": 0}
    deadline = time.monotonic() + 900
    async with httpx.AsyncClient(timeout=300) as client:
        for turn in range(request.max_turns):
            if cancelled and cancelled.is_set():
                return {"ok": False, "cancelled": True, "content": final, "events": events, "turns": turn, "usage": usage}
            if time.monotonic() >= deadline:
                return {"ok": False, "content": final[:200_000], "events": events, "turns": turn, "usage": usage, "error": "Agent time limit reached"}
            try:
                response = await client.post(
                    settings.ollama_url + "/api/chat",
                    headers={"X-Worker-Token": settings.worker_token},
                    json={"model": request.model, "messages": messages, "tools": tools, "stream": False},
                )
            except httpx.HTTPError:
                if cancelled and cancelled.is_set():
                    return {"ok": False, "cancelled": True, "content": final, "events": events, "turns": turn, "usage": usage}
                raise
            response.raise_for_status()
            payload = response.json()
            message = payload.get("message", {})
            for key in ("prompt_eval_count", "eval_count"):
                usage[key] += int(payload.get(key) or 0)
            usage["model_duration_ns"] += int(payload.get("total_duration") or 0)
            messages.append(message)
            calls = message.get("tool_calls") or []
            usage["tool_calls"] += len(calls)
            content = message.get("content", "")
            if content:
                final = content
                events.append({"type": "message", "turn": turn + 1, "content": content[:8000]})
            if not calls:
                return {"ok": True, "content": final[:200_000], "events": events, "turns": turn + 1, "usage": usage}
            for call in calls:
                fn = call.get("function", {})
                name = fn.get("name", "")
                args = fn.get("arguments") or {}
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {}
                try:
                    result = await asyncio.to_thread(execute_tool, root, name, args, writable)
                except Exception as exc:
                    result = f"Tool error: {type(exc).__name__}: {exc}"
                events.append({"type": "tool", "turn": turn + 1, "name": name, "args": args, "result": result[:8000]})
                messages.append({"role": "tool", "tool_name": name, "content": result})
    return {"ok": False, "content": final[:200_000], "events": events, "turns": request.max_turns, "usage": usage, "error": "Agent turn limit reached"}


@app.get("/healthz")
def health() -> dict[str, Any]:
    return {"ok": JOBS_ROOT.exists(), "jobs_root": str(JOBS_ROOT)}


@app.post("/execute", dependencies=[Depends(authorize)])
async def execute(request: WorkRequest) -> dict[str, Any]:
    root = workspace_for(request)
    event = asyncio.Event()
    task = asyncio.current_task()
    with _cancel_lock:
        _cancel_events[request.run_id] = event
        if task:
            _active_tasks[request.run_id] = task
    try:
        return await agent_loop(request, root, event)
    except asyncio.CancelledError:
        return {"ok": False, "cancelled": True, "content": "", "events": [], "turns": 0, "usage": {}}
    except httpx.HTTPError as exc:
        raise HTTPException(502, f"Ollama request failed: {exc}") from exc
    finally:
        with _cancel_lock:
            _cancel_events.pop(request.run_id, None)
            _active_tasks.pop(request.run_id, None)


@app.post("/cancel/{run_id}", dependencies=[Depends(authorize)])
async def cancel(run_id: str) -> dict[str, Any]:
    with _cancel_lock:
        event = _cancel_events.get(run_id)
        task = _active_tasks.get(run_id)
        if event:
            event.set()
        if task:
            task.cancel()
    return {"ok": True, "active": event is not None}
