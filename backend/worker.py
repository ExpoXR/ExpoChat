import asyncio
import contextlib
import hmac
import json
import threading
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from pathlib import Path
from typing import Any, Literal

import httpx
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from .config import settings
from .logging_utils import configure_logging
from .prompts import CAVEMAN_OUTPUT_INSTRUCTIONS
from .workspace_tools import TOOL_DEFS, execute_tool

configure_logging()
app = FastAPI(title="Ollma Isolated Worker", docs_url=None, redoc_url=None)
JOBS_ROOT = settings.jobs_dir.resolve()
_cancel_events: dict[str, asyncio.Event] = {}
_active_tasks: dict[str, asyncio.Task[Any]] = {}
_cancel_lock = threading.Lock()


def _cancel_key(run_id: str, node_id: str | None = None) -> str:
    return f"{run_id}:{node_id}" if node_id else run_id
MODE_PROMPTS = {
    "research": "Inspect workspace deeply. Use tools before answering. Return architecture, relevant files, risks, tests, and implementation advice. Never edit.",
    "implementation": "Implement approved task fully in staged workspace. Use precise file tools, run relevant checks, fix failures, then summarize changed files and evidence.",
    "verification": "Independently verify implementation. Inspect files and run safe checks. Return PASS or FAIL first, then concrete evidence and defects. Never edit.",
    "console": "Run requested safe check and return exact output. Never edit.",
}


class WorkRequest(BaseModel):
    run_id: str = Field(pattern=r"^[a-f0-9-]{8,64}$")
    workspace: str = "workspace"
    model: str
    mode: Literal["research", "implementation", "verification", "console"]
    task: str
    max_turns: int = Field(default=24, ge=1, le=40)
    node_id: str | None = None
    # Which registered Ollama host to route this call to. Carried to the UI's internal proxy,
    # which resolves it to the host's base_url; absent = the default host.
    ollama_host_id: str | None = None


class CheckRequest(BaseModel):
    run_id: str = Field(pattern=r"^[a-f0-9-]{8,64}$")
    workspace: str = "workspace"
    command: str = Field(min_length=1, max_length=100)
    args: list[str] = Field(default_factory=list, max_length=40)
    timeout: int = Field(default=120, ge=1)


def authorize(x_worker_token: str = Header(default="")) -> None:
    if not settings.worker_token or not hmac.compare_digest(x_worker_token, settings.worker_token):
        raise HTTPException(401, "Invalid worker token")


def workspace_for(request: WorkRequest | CheckRequest) -> Path:
    path = (JOBS_ROOT / request.run_id / request.workspace).resolve()
    if JOBS_ROOT not in path.parents or not path.exists() or not path.is_dir():
        raise HTTPException(404, "Staged workspace not found")
    return path


def agent_system_prompt(mode: str) -> str:
    return MODE_PROMPTS[mode] + " Workspace root is current staged directory.\n\n" + CAVEMAN_OUTPUT_INSTRUCTIONS


async def agent_loop(
    request: WorkRequest,
    root: Path,
    cancelled: asyncio.Event | None = None,
    emit: Callable[[dict[str, Any]], Awaitable[None]] | None = None,
) -> dict[str, Any]:
    writable = request.mode == "implementation"
    tools = TOOL_DEFS if writable else [tool for tool in TOOL_DEFS if tool["function"]["name"] not in {"write_file", "replace_text", "delete_file"}]
    messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": agent_system_prompt(request.mode),
        },
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
                    params={"host_id": request.ollama_host_id} if request.ollama_host_id else None,
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
                item = {"type": "message", "turn": turn + 1, "content": content[:8000]}
                events.append(item)
                if emit:
                    await emit(item)
            if not calls:
                if not final.strip():
                    return {
                        "ok": False,
                        "content": "",
                        "events": events,
                        "turns": turn + 1,
                        "usage": usage,
                        "error": "Agent returned an empty response",
                    }
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
                started = {"type": "tool.started", "turn": turn + 1, "name": name, "args": args}
                if emit:
                    await emit(started)
                try:
                    result = await asyncio.to_thread(execute_tool, root, name, args, writable)
                except Exception as exc:
                    result = f"Tool error: {type(exc).__name__}: {exc}"
                item = {"type": "tool", "turn": turn + 1, "name": name, "args": args, "result": result[:8000]}
                events.append(item)
                if emit:
                    await emit({**item, "type": "tool.completed"})
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
    key = _cancel_key(request.run_id, request.node_id)
    with _cancel_lock:
        _cancel_events[key] = event
        if task:
            _active_tasks[key] = task
    try:
        return await agent_loop(request, root, event)
    except asyncio.CancelledError:
        return {"ok": False, "cancelled": True, "content": "", "events": [], "turns": 0, "usage": {}}
    except httpx.HTTPError as exc:
        raise HTTPException(502, f"Ollama request failed: {exc}") from exc
    finally:
        with _cancel_lock:
            _cancel_events.pop(key, None)
            _active_tasks.pop(key, None)


@app.post("/execute/stream", dependencies=[Depends(authorize)])
async def execute_stream(request: WorkRequest) -> StreamingResponse:
    root = workspace_for(request)

    async def generate() -> AsyncIterator[str]:
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        cancelled = asyncio.Event()

        async def emit(item: dict[str, Any]) -> None:
            await queue.put(item)

        task = asyncio.create_task(agent_loop(request, root, cancelled, emit))
        key = _cancel_key(request.run_id, request.node_id)
        with _cancel_lock:
            _cancel_events[key] = cancelled
            _active_tasks[key] = task
        try:
            while not task.done() or not queue.empty():
                try:
                    item = await asyncio.wait_for(queue.get(), timeout=0.25)
                except TimeoutError:
                    continue
                yield json.dumps(item, ensure_ascii=False) + "\n"
            result = await task
            yield json.dumps({"type": "result", "result": result}, ensure_ascii=False) + "\n"
        except asyncio.CancelledError:
            cancelled.set()
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
            raise
        except httpx.HTTPError as exc:
            yield json.dumps({"type": "error", "error": f"Ollama request failed: {exc}"}) + "\n"
        finally:
            with _cancel_lock:
                _cancel_events.pop(key, None)
                _active_tasks.pop(key, None)

    return StreamingResponse(generate(), media_type="application/x-ndjson")


@app.post("/check", dependencies=[Depends(authorize)])
def check(request: CheckRequest) -> dict[str, Any]:
    output = execute_tool(
        workspace_for(request),
        "run_check",
        {"command": request.command, "args": request.args, "timeout": request.timeout},
        False,
    )
    return {"ok": output.startswith("exit=0\n"), "content": output}


@app.post("/cancel/{run_id}", dependencies=[Depends(authorize)])
async def cancel(run_id: str, node_id: str | None = None) -> dict[str, Any]:
    cancelled = 0
    with _cancel_lock:
        if node_id:
            key = _cancel_key(run_id, node_id)
            event = _cancel_events.get(key)
            task = _active_tasks.get(key)
            if event:
                event.set()
                cancelled += 1
            if task:
                task.cancel()
        else:
            for key in list(_cancel_events):
                if key == run_id or key.startswith(f"{run_id}:"):
                    _cancel_events[key].set()
                    cancelled += 1
            for key in list(_active_tasks):
                if key == run_id or key.startswith(f"{run_id}:"):
                    _active_tasks[key].cancel()
    return {"ok": True, "cancelled": cancelled}
