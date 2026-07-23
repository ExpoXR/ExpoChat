import contextlib
import json
import logging
import secrets
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx

from . import db
from .brain_io import (
    build_decompose_prompt,
    build_plan_prompt,
    build_verdict_prompt,
    build_verification_prompt,
    extract_json,
    parse_verdict,
)
from .config import settings
from .plan_graph import GraphError, independent_pairs_sharing_globs, validate_graph
from .prompts import with_caveman
from .run_state import validate_transition
from .security import decrypt_secret
from .workspace import (
    MergeConflict,
    apply_stage,
    create_snapshot,
    discard_snapshot,
    manifest_hash,
    merge_worktrees,
    restore_snapshot,
    stage_subtask,
    stage_workspace,
)

log = logging.getLogger("ollma.orchestrator")
executor = ThreadPoolExecutor(max_workers=settings.runner_concurrency, thread_name_prefix="ollma-runner")
_run_lock = threading.Lock()
_queue_lock = threading.Lock()
_active_drainers = 0


def _lease_heartbeat(job_id: int, lease_owner: str, stopped: threading.Event) -> None:
    while not stopped.wait(30):
        expires = (datetime.now(UTC) + timedelta(minutes=20)).isoformat()
        db.execute(
            "update jobs set lease_expires_at=?,updated_at=? where id=? and status='running' and lease_owner=?",
            (expires, db.utcnow(), job_id, lease_owner),
        )


def enqueue_job(run_id: str, job_type: str) -> None:
    now = db.utcnow()
    with db.transaction() as conn:
        active = conn.execute(
            "select id from jobs where run_id=? and job_type=? and status in ('pending','running') limit 1",
            (run_id, job_type),
        ).fetchone()
        if not active:
            conn.execute(
                "insert into jobs(run_id,job_type,status,created_at,updated_at) values(?,?,'pending',?,?)",
                (run_id, job_type, now, now),
            )
    start_job_queue()


def start_job_queue() -> None:
    global _active_drainers
    with _queue_lock:
        count = settings.runner_concurrency - _active_drainers
        _active_drainers += count
    for _ in range(count):
        executor.submit(_drain_jobs)


def _drain_jobs() -> None:
    global _active_drainers
    lease_owner = "runner-" + secrets.token_hex(6)
    try:
        while True:
            with db.transaction() as conn:
                now = db.utcnow()
                conn.execute(
                    "update jobs set status='pending',error='Expired lease; queued for recovery',lease_owner=null,"
                    "lease_expires_at=null,updated_at=? where status='running' and lease_expires_at<? "
                    "and run_id in (select id from runs where status not in ('completed','failed','cancelled','rolled_back'))",
                    (now, now),
                )
                row = conn.execute("select * from jobs where status='pending' order by id limit 1").fetchone()
                if not row:
                    break
                job = dict(row)
                now = db.utcnow()
                lease_expires = (datetime.now(UTC) + timedelta(minutes=20)).isoformat()
                claimed = conn.execute(
                    "update jobs set status='running',attempts=attempts+1,started_at=?,lease_owner=?,"
                    "lease_expires_at=?,updated_at=? where id=? and status='pending'",
                    (now, lease_owner, lease_expires, now, job["id"]),
                )
                if claimed.rowcount != 1:
                    continue
            heartbeat_stop = threading.Event()
            heartbeat = threading.Thread(
                target=_lease_heartbeat,
                args=(job["id"], lease_owner, heartbeat_stop),
                name=f"ollma-lease-{job['id']}",
                daemon=True,
            )
            heartbeat.start()
            try:
                # implementation jobs also drive the subtask DAG + merge inside implement_run;
                # subtask/merge job_types are reserved for a future fully-durable per-node queue.
                if job["job_type"] == "research":
                    research_run(job["run_id"])
                else:
                    implement_run(job["run_id"])
                now = db.utcnow()
                final_job = db.one("select cancel_requested_at from jobs where id=?", (job["id"],)) or {}
                final_run = db.one("select status,error from runs where id=?", (job["run_id"],)) or {}
                if final_job.get("cancel_requested_at") or final_run.get("status") == "cancelled":
                    final_status = "cancelled"
                    final_error = "Cancellation requested"
                elif final_run.get("status") == "failed":
                    final_status = "failed"
                    final_error = final_run.get("error")
                else:
                    final_status = "done"
                    final_error = None
                db.execute(
                    "update jobs set status=?,error=?,completed_at=?,lease_owner=null,lease_expires_at=null,updated_at=? where id=?",
                    (final_status, final_error, now, now, job["id"]),
                )
            except Exception as exc:
                log.exception("job_failed", extra={"job_id": job["id"], "run_id": job["run_id"]})
                current = db.one("select cancel_requested_at from jobs where id=?", (job["id"],)) or {}
                final_status = "cancelled" if current.get("cancel_requested_at") else "failed"
                db.execute(
                    "update jobs set status=?,error=?,completed_at=?,lease_owner=null,lease_expires_at=null,updated_at=? where id=?",
                    (final_status, str(exc)[:4000], db.utcnow(), db.utcnow(), job["id"]),
                )
            finally:
                heartbeat_stop.set()
                heartbeat.join(timeout=1)
    finally:
        with _queue_lock:
            _active_drainers -= 1
        if db.one("select id from jobs where status='pending' limit 1"):
            start_job_queue()


def provider_config(provider: str) -> tuple[str, str]:
    row = db.one("select * from brain_configs where provider=?", (provider,))
    if not row or not row["enabled"]:
        raise RuntimeError(f"{provider} brain is not linked")
    if row["source"] == "stored":
        key = decrypt_secret(row["key_ciphertext"])
    else:
        key = settings.environment_key(provider)
    if not key:
        raise RuntimeError(f"{provider} API key is missing")
    return key, row["model"]


class BudgetExceeded(RuntimeError):
    """Raised when a configured token budget blocks a paid provider call."""


def check_budget(run_id: str | None = None) -> None:
    daily_cap = db.get_setting_int("token_budget_daily")
    if daily_cap > 0 and db.ledger_totals_today(paid_only=True)["total"] >= daily_cap:
        raise BudgetExceeded(
            f"Daily API token budget reached ({daily_cap:,}). Raise it in Settings to continue."
        )
    run_cap = db.get_setting_int("token_budget_run")
    if run_cap > 0 and run_id:
        used = json.loads((db.one("select usage_json from runs where id=?", (run_id,)) or {}).get("usage_json") or "{}")
        brain = used.get("brain", {})
        total = int(brain.get("total_tokens") or (int(brain.get("input_tokens", 0) or 0) + int(brain.get("output_tokens", 0) or 0)))
        if total >= run_cap:
            raise BudgetExceeded(
                f"This run reached its token budget ({run_cap:,}). Raise it in Settings to continue."
            )


def call_brain_result(
    provider: str, prompt: str, allow_web: bool = False, timeout: int = 900, run_id: str | None = None
) -> dict[str, Any]:
    check_budget(run_id)
    key, model = provider_config(provider)
    payload = {
        "provider": provider,
        "api_key": key,
        "model": model,
        "prompt": with_caveman(prompt),
        "allow_web": allow_web,
    }
    max_output = db.get_setting_int("max_output_tokens")
    if max_output > 0:
        payload["max_output_tokens"] = max_output
    payload["timeout"] = timeout
    try:
        with httpx.Client(timeout=timeout + 10) as client:
            response = client.post(
                settings.brain_url + "/execute",
                headers={"X-Worker-Token": settings.worker_token},
                json=payload,
            )
    except httpx.HTTPError as exc:
        raise RuntimeError(f"Brain service unavailable: {exc}") from exc
    if response.status_code != 200:
        try:
            detail = response.json().get("detail")
        except Exception:
            detail = None
        if detail:
            raise RuntimeError(str(detail))
        raise RuntimeError(f"Brain service failed with HTTP {response.status_code}")
    data = response.json()
    return {"content": str(data["content"]), "usage": data.get("usage") or {}}


def call_brain(provider: str, prompt: str, allow_web: bool = False, timeout: int = 900) -> str:
    return call_brain_result(provider, prompt, allow_web, timeout)["content"]


def _build_memory_digest(run_id: str) -> str:
    """Build a bounded digest of prior brain interactions for this run."""
    entries = db.brain_memory(run_id)
    if not entries:
        return ""
    budget = db.get_setting_int("brain_memory_budget") or 4000
    char_budget = budget * 4  # tokens_estimate = chars // 4
    parts: list[str] = []
    total_chars = 0
    # newest last — build in order, trim oldest if over budget
    for entry in entries:
        line = f"[{entry['step']}/{entry['role']}] {entry['content']}"
        total_chars += len(line)
        parts.append(line)
    # Drop oldest entries until within budget
    while total_chars > char_budget and len(parts) > 1:
        dropped = parts.pop(0)
        total_chars -= len(dropped)
    return "PRIOR CONTEXT (brain's earlier reasoning this run):\n" + "\n\n".join(parts) + "\n\n"


def call_brain_with_memory(
    run_id: str, provider: str, step: str, prompt: str,
    allow_web: bool = False, timeout: int = 900,
) -> dict[str, Any]:
    """Brain call with per-run memory continuity.

    Injects a digest of prior brain steps, calls the brain, and persists both
    a prompt summary and the response for future calls to reference.
    """
    digest = _build_memory_digest(run_id)
    full_prompt = digest + prompt if digest else prompt
    result = call_brain_result(provider, full_prompt, allow_web, timeout, run_id=run_id)
    # Persist prompt summary (first 2000 chars) and full response
    summary = prompt[:2000]
    db.add_brain_memory(run_id, step, "prompt", summary)
    db.add_brain_memory(run_id, step, "response", result["content"][:4000])
    return result


def worker_call(run_id: str, model: str, mode: str, task: str, workspace: str = "workspace", max_turns: int = 24) -> dict[str, Any]:
    attempts = 1 if mode == "implementation" else 3
    for attempt in range(attempts):
        try:
            with httpx.Client(timeout=1200) as client:
                with client.stream(
                    "POST",
                    settings.worker_url + "/execute/stream",
                    headers={"X-Worker-Token": settings.worker_token},
                    json={"run_id": run_id, "workspace": workspace, "model": model, "mode": mode, "task": task, "max_turns": max_turns},
                ) as response:
                    response.raise_for_status()
                    for line in response.iter_lines():
                        if not line:
                            continue
                        item = json.loads(line)
                        if item.get("type") == "result":
                            return item["result"]
                        if item.get("type") == "error":
                            raise RuntimeError(item.get("error") or "Worker failed")
                        record_worker_activity(run_id, mode, item)
        except (httpx.ConnectError, httpx.RemoteProtocolError, httpx.HTTPStatusError):
            if attempt + 1 >= attempts:
                raise
            time.sleep(1 << attempt)
    raise RuntimeError("Worker unavailable")


def record_worker_activity(run_id: str, mode: str, item: dict[str, Any]) -> None:
    kind = str(item.get("type") or "")
    if kind == "message":
        content = " ".join(str(item.get("content") or "").split())[:1000]
        if content:
            db.add_event(run_id, "agent.activity", content, {"phase": mode, "state": "message"})
        return
    if kind not in {"tool.started", "tool.completed"}:
        return
    tool = str(item.get("name") or "tool")
    args = item.get("args") if isinstance(item.get("args"), dict) else {}
    path = str(args.get("path") or "")[:2000]
    target = path or str(args.get("command") or "")[:200]
    state = "working" if kind == "tool.started" else "done"
    result = str(item.get("result") or "")
    mutation = tool in {"write_file", "replace_text", "delete_file"}
    failed = result.startswith(("Tool error:", "Rejected", "File not found", "Text not found", "Unknown tool"))
    if kind == "tool.completed" and mutation and not failed:
        state = "changed"
    verb = "Working" if kind == "tool.started" else "Finished"
    message = f"{verb}: {tool}{f' · {target}' if target else ''}"
    db.add_event(
        run_id,
        "agent.activity",
        message,
        {"phase": mode, "state": state, "tool": tool, "path": path, "turn": item.get("turn")},
    )


def discover_agents() -> list[dict[str, Any]]:
    now = db.utcnow()
    with httpx.Client(timeout=30) as client:
        tags_response = client.get(settings.ollama_url + "/api/tags")
        tags_response.raise_for_status()
        tags = tags_response.json().get("models", [])
        discovered: list[dict[str, Any]] = []
        for item in tags:
            model = item.get("name") or item.get("model")
            if not model:
                continue
            show_response = client.post(settings.ollama_url + "/api/show", json={"model": model})
            show_response.raise_for_status()
            show = show_response.json()
            capabilities = show.get("capabilities") or []
            context = max(
                [int(v) for k, v in (show.get("model_info") or {}).items() if k.endswith(".context_length") and isinstance(v, (int, float))]
                or [0]
            )
            lower = model.lower()
            roles = ["research", "verification"]
            if "tools" in capabilities:
                roles.append("implementation")
            scores = {
                "research": 90 if "thinking" in capabilities else 70,
                "implementation": 95 if "qwen3-coder" in lower else 80 if "tools" in capabilities else 0,
                "verification": 90 if "gemma" in lower else 75,
            }
            priority = 100 if "qwen3-coder" in lower else 90 if "gemma" in lower else 75
            agent_id = "agent-" + secrets.token_hex(8)
            existing = db.one("select id from agent_profiles where model=?", (model,))
            if existing:
                agent_id = existing["id"]
                db.execute(
                    "update agent_profiles set capabilities_json=?,context_size=?,discovered_at=?,updated_at=? where id=?",
                    (json.dumps(capabilities), context, now, now, agent_id),
                )
            else:
                db.execute(
                    "insert into agent_profiles(id,name,model,roles_json,capabilities_json,context_size,priority,role_scores_json,discovered_at,updated_at) values(?,?,?,?,?,?,?,?,?,?)",
                    (agent_id, model, model, json.dumps(roles), json.dumps(capabilities), context, priority, json.dumps(scores), now, now),
                )
            discovered.append(db.one("select * from agent_profiles where id=?", (agent_id,)) or {})
        return discovered


def _agent_is_eligible(agent: dict[str, Any] | None, role: str, exclude: set[str] | None = None) -> bool:
    if not agent or not agent.get("enabled") or agent["id"] in (exclude or set()):
        return False
    roles = json.loads(agent.get("roles_json") or "[]")
    capabilities = json.loads(agent.get("capabilities_json") or "[]")
    return role in roles and (role != "implementation" or "tools" in capabilities)


def choose_agent(role: str, exclude: set[str] | None = None, discover: bool = True) -> dict[str, Any]:
    exclude = exclude or set()
    candidates = []
    for row in db.all_rows("select * from agent_profiles where enabled=1"):
        if not _agent_is_eligible(row, role, exclude):
            continue
        scores = json.loads(row["role_scores_json"] or "{}")
        candidates.append((int(scores.get(role, 0)), int(row["priority"]), int(row["context_size"]), row["name"], row))
    if not candidates:
        if discover:
            discover_agents()
            return choose_agent(role, exclude, False)
        raise RuntimeError(
            f"No enabled {role} agent available. Open Brains & Agents, enable a compatible Ollama model, then retry."
        )
    candidates.sort(key=lambda item: (-item[0], -item[1], -item[2], item[3].lower()))
    return candidates[0][-1]


# In-memory round-robin counters per run for distributing work across equally-scored agents.
_subtask_assignment_counts: dict[str, dict[str, int]] = {}


def choose_subtask_agent(
    node: dict[str, Any], exclude: set[str] | None = None, run_id: str | None = None,
) -> dict[str, Any]:
    """Choose an agent for a specific subtask, honoring role and optional model hint.

    1. If node has suggested_model and that model is enabled+eligible -> use it.
    2. Else choose_agent(node.role, exclude) — now honoring the subtask's role.
    3. Round-robin tiebreak: when multiple agents have equal scores, rotate.
    """
    exclude = exclude or set()
    role = node.get("role") or "implementation"
    suggested = node.get("suggested_model") or ""
    if suggested:
        agent = db.one("select * from agent_profiles where model=? and enabled=1", (suggested,))
        if agent and _agent_is_eligible(agent, role, exclude):
            return agent
    # Collect all eligible candidates for round-robin
    candidates = []
    for row in db.all_rows("select * from agent_profiles where enabled=1"):
        if not _agent_is_eligible(row, role, exclude):
            continue
        scores = json.loads(row["role_scores_json"] or "{}")
        candidates.append((int(scores.get(role, 0)), int(row["priority"]), int(row["context_size"]), row["name"], row))
    if not candidates:
        # Fallback to standard choose_agent which can auto-discover
        return choose_agent(role, exclude)
    candidates.sort(key=lambda item: (-item[0], -item[1], -item[2], item[3].lower()))
    # Round-robin among top-tied candidates
    top_score = (candidates[0][0], candidates[0][1])
    tied = [c for c in candidates if (c[0], c[1]) == top_score]
    if len(tied) > 1 and run_id:
        counts = _subtask_assignment_counts.setdefault(run_id, {})
        # Pick the agent with the fewest assignments
        tied.sort(key=lambda c: (counts.get(c[-1]["id"], 0), c[3].lower()))
        chosen = tied[0][-1]
        counts[chosen["id"]] = counts.get(chosen["id"], 0) + 1
        return chosen
    return tied[0][-1]


def save_artifact(run_id: str, kind: str, name: str, content: Any) -> None:
    serialized = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
    with db.transaction() as conn:
        conn.execute(
            "insert into run_artifacts(run_id,kind,name,content,created_at) values(?,?,?,?,?)",
            (run_id, kind, name, serialized[:500_000], db.utcnow()),
        )
        conn.execute(
            "delete from run_artifacts where run_id=? and id not in "
            "(select id from run_artifacts where run_id=? order by id desc limit ?)",
            (run_id, run_id, db.retention_cap("run_artifacts_cap", db.MAX_RUN_ARTIFACTS)),
        )


def record_usage(run_id: str, result: dict[str, Any], source: str = "ollama") -> None:
    usage = result.get("usage") or {}
    if not usage:
        return
    run = db.one("select usage_json from runs where id=?", (run_id,)) or {}
    current = json.loads(run.get("usage_json") or "{}")
    if source == "ollama" and "ollama" not in current:
        legacy = {key: value for key, value in current.items() if isinstance(value, (int, float))}
        for key in legacy:
            current.pop(key, None)
        current["ollama"] = legacy
    bucket = current.setdefault(source, {})
    for key, value in usage.items():
        if isinstance(value, (int, float)):
            bucket[key] = bucket.get(key, 0) + value
    update_run(run_id, usage_json=json.dumps(current, ensure_ascii=False))

    provider = "ollama"
    if source == "brain":
        provider = (db.one("select brain_provider from runs where id=?", (run_id,)) or {}).get("brain_provider") or "brain"
    inp = int(usage.get("input_tokens", usage.get("prompt_eval_count", 0)) or 0)
    out = int(usage.get("output_tokens", usage.get("eval_count", 0)) or 0)
    total = int(usage.get("total_tokens") or (inp + out))
    db.record_ledger(source, provider, inp, out, total)


def record_chat_usage(source: str, provider: str, usage: dict[str, Any] | None) -> None:
    """Ledger-only usage recording for interactive chat (no run row)."""
    usage = usage or {}
    if not usage:
        return
    inp = int(usage.get("input_tokens", usage.get("prompt_eval_count", 0)) or 0)
    out = int(usage.get("output_tokens", usage.get("eval_count", 0)) or 0)
    total = int(usage.get("total_tokens") or (inp + out))
    db.record_ledger(source, provider, inp, out, total)


def agent_task(agent: dict[str, Any], task: str) -> str:
    prompt = str(agent.get("system_prompt") or "").strip()
    return ("AGENT PROFILE INSTRUCTIONS:\n" + prompt + "\n\nTASK:\n" + task) if prompt else task


def update_run(run_id: str, **values: Any) -> None:
    if "status" in values:
        current = db.one("select status from runs where id=?", (run_id,))
        if not current:
            raise RuntimeError("Run not found")
        validate_transition(current["status"], values["status"])
    values["updated_at"] = db.utcnow()
    columns = ",".join(f"{key}=?" for key in values)
    db.execute(f"update runs set {columns} where id=?", (*values.values(), run_id))


def create_run(task: str, provider: str, target: Path, web_research: bool) -> dict[str, Any]:
    run_id = secrets.token_hex(12)
    _, model = provider_config(provider)
    research_agent = choose_agent("research")
    implementation_agent = choose_agent("implementation")
    stage_workspace(run_id, target)
    baseline = manifest_hash(target)
    now = db.utcnow()
    with db.transaction() as conn:
        conn.execute(
            "insert into runs(id,task,brain_provider,brain_model,target_path,web_research,status,baseline_hash,"
            "research_agent_id,implementation_agent_id,created_at,updated_at) values(?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                run_id, task, provider, model, str(target), int(web_research), "researching", baseline,
                research_agent["id"], implementation_agent["id"], now, now,
            ),
        )
        conn.execute(
            "insert into jobs(run_id,job_type,status,created_at,updated_at) values(?,'research','pending',?,?)",
            (run_id, now, now),
        )
    db.add_event(
        run_id,
        "run.created",
        f"Research queued with {research_agent['name']}",
        {"provider": provider, "target": str(target), "research_agent": research_agent["id"]},
    )
    start_job_queue()
    return db.one("select * from runs where id=?", (run_id,)) or {}


def decompose_plan(run_id: str, plan: str, provider: str) -> list[dict[str, Any]]:
    """Ask the brain to turn a plan into a validated task graph and persist it.

    Best-effort and backward-compatible: any failure logs an event and leaves the run on
    the existing single-implementer path (no subtasks persisted). Returns the node list.
    """
    try:
        result = call_brain_with_memory(run_id, provider, "decompose", build_decompose_prompt(plan))
        record_usage(run_id, result, "brain")
        nodes = validate_graph(extract_json(result["content"]))
        db.insert_subtasks(run_id, nodes)
        save_artifact(run_id, "task_graph", "Task graph", {"subtasks": nodes})
        conflicts = independent_pairs_sharing_globs(nodes)
        db.add_event(
            run_id,
            "plan.decomposed",
            f"Plan decomposed into {len(nodes)} subtask(s)",
            {"count": len(nodes), "overlap_warnings": conflicts},
        )
        return nodes
    except (GraphError, KeyError, RuntimeError, ValueError) as exc:
        log.warning("decompose_failed", extra={"run_id": run_id, "error": str(exc)})
        db.add_event(run_id, "plan.decompose_skipped", "Plan kept as a single unit", {"reason": str(exc)[:400]})
        return []


def research_run(run_id: str) -> None:
    try:
        run = db.one("select * from runs where id=?", (run_id,))
        if not run or run["status"] == "cancelled":
            return
        agent = db.one("select * from agent_profiles where id=?", (run.get("research_agent_id"),))
        if not _agent_is_eligible(agent, "research"):
            agent = choose_agent("research")
        update_run(run_id, research_agent_id=agent["id"], status="researching")
        db.add_event(run_id, "research.started", f"Research started with {agent['name']}")
        result = worker_call(run_id, agent["model"], "research", agent_task(agent, run["task"]), max_turns=18)
        record_usage(run_id, result)
        if not result.get("ok"):
            raise RuntimeError(result.get("error") or "Research agent failed")
        if (db.one("select status from runs where id=?", (run_id,)) or {}).get("status") == "cancelled":
            return
        dossier = result.get("content", "")
        save_artifact(run_id, "research", "Ollama research dossier", {"summary": dossier, "events": result.get("events", [])})
        db.add_event(run_id, "research.completed", "Ollama research complete")
        prompt = build_plan_prompt(run["task"], dossier)
        brain_result = call_brain_with_memory(
            run_id, run["brain_provider"], "plan", prompt,
            allow_web=bool(run["web_research"]),
        )
        record_usage(run_id, brain_result, "brain")
        plan = brain_result["content"]
        if (db.one("select status from runs where id=?", (run_id,)) or {}).get("status") == "cancelled":
            return
        implementation = choose_agent("implementation")
        update_run(
            run_id,
            dossier=dossier,
            draft_plan=plan,
            implementation_agent_id=implementation["id"],
            status="awaiting_approval",
        )
        save_artifact(run_id, "plan", "Supervisor plan", plan)
        db.add_plan_version(run_id, "draft", plan, run["brain_provider"])
        decompose_plan(run_id, plan, run["brain_provider"])
        db.add_event(run_id, "plan.ready", "Plan ready for approval")
    except Exception as exc:
        log.exception("research_failed", extra={"run_id": run_id})
        if (db.one("select status from runs where id=?", (run_id,)) or {}).get("status") == "cancelled":
            return
        update_run(run_id, status="failed", error=str(exc)[:4000])
        db.add_event(run_id, "run.failed", "Research or planning failed", {"error": str(exc)[:1000]})


def approve_run(run_id: str, approved_plan: str) -> dict[str, Any]:
    with _run_lock:
        run = db.one("select * from runs where id=?", (run_id,))
        if not run or run["status"] != "awaiting_approval":
            raise RuntimeError("Run is not awaiting approval")
        target = Path(run["target_path"])
        current_hash = manifest_hash(target)
        if current_hash != run["baseline_hash"]:
            stage_workspace(run_id, target)
            update_run(run_id, baseline_hash=current_hash, status="researching", error="Workspace changed; plan refresh required")
            db.add_event(run_id, "plan.stale", "Workspace changed; research restarted")
            enqueue_job(run_id, "research")
            raise RuntimeError("Workspace changed; research restarted")
        implementation = db.one("select * from agent_profiles where id=?", (run.get("implementation_agent_id"),))
        if not _agent_is_eligible(implementation, "implementation"):
            implementation = choose_agent("implementation")
        if run.get("snapshot_id") and db.one("select id from history_snippets where run_id=?", (run_id,)):
            now = db.utcnow()
            with db.transaction() as conn:
                conn.execute(
                    "update runs set approved_plan=?,implementation_agent_id=?,status='implementing',repair_count=0,error=null,approved_at=?,updated_at=? where id=?",
                    (approved_plan, implementation["id"], now, now, run_id),
                )
                conn.execute(
                    "insert into run_approvals(run_id,approved_plan,snapshot_id,created_at) values(?,?,?,?)",
                    (run_id, approved_plan, run["snapshot_id"], now),
                )
                conn.execute(
                    "insert into jobs(run_id,job_type,status,created_at,updated_at) values(?,'implementation','pending',?,?)",
                    (run_id, now, now),
                )
            db.add_plan_version(run_id, "approved", approved_plan, run["brain_provider"])
            db.add_event(run_id, "scope.approved", "Expanded scope approved and implementation queued")
            start_job_queue()
            return db.one("select * from runs where id=?", (run_id,)) or {}
        snapshot = create_snapshot(target)
        workers = [run.get("research_agent_id"), implementation["id"]]
        now = db.utcnow()
        try:
            with db.transaction() as conn:
                conn.execute(
                    "update runs set approved_plan=?,snapshot_id=?,implementation_agent_id=?,status='implementing',approved_at=?,updated_at=? where id=?",
                    (approved_plan, snapshot["id"], implementation["id"], now, now, run_id),
                )
                conn.execute(
                    "insert into history_snippets(id,run_id,request,approved_plan,brain_provider,workers_json,target_path,snapshot_id,created_at) "
                    "values(?,?,?,?,?,?,?,?,?) on conflict(run_id) do update set approved_plan=excluded.approved_plan,"
                    "workers_json=excluded.workers_json,target_path=excluded.target_path,snapshot_id=excluded.snapshot_id,"
                    "created_at=excluded.created_at,completed_at=null,final_verdict=null",
                    ("history-" + secrets.token_hex(10), run_id, run["task"], approved_plan, run["brain_provider"], json.dumps(workers), run["target_path"], snapshot["id"], now),
                )
                conn.execute(
                    "insert into run_approvals(run_id,approved_plan,snapshot_id,created_at) values(?,?,?,?)",
                    (run_id, approved_plan, snapshot["id"], now),
                )
                conn.execute(
                    "insert into jobs(run_id,job_type,status,created_at,updated_at) values(?,'implementation','pending',?,?)",
                    (run_id, now, now),
                )
        except Exception:
            discard_snapshot(snapshot["id"])
            raise
        db.add_plan_version(run_id, "approved", approved_plan, run["brain_provider"])
        db.add_event(run_id, "plan.approved", "History and snapshot created", {"snapshot_id": snapshot["id"]})
        start_job_queue()
        return db.one("select * from runs where id=?", (run_id,)) or {}


def edit_plan(run_id: str, plan: str) -> dict[str, Any]:
    run = db.one("select * from runs where id=?", (run_id,))
    if not run or run["status"] != "awaiting_approval":
        raise RuntimeError("Run is not awaiting plan edits")
    update_run(run_id, draft_plan=plan, error=None)
    save_artifact(run_id, "plan_edit", "User-edited plan", plan)
    db.add_plan_version(run_id, "edit", plan, run["brain_provider"])
    db.add_event(run_id, "plan.edited", "Plan changes saved")
    return db.one("select * from runs where id=?", (run_id,)) or {}


def redo_plan(run_id: str) -> dict[str, Any]:
    with _run_lock:
        run = db.one("select * from runs where id=?", (run_id,))
        if not run or run["status"] != "awaiting_approval":
            raise RuntimeError("Only a plan awaiting approval can be redone")
        target = Path(run["target_path"])
        research_agent = choose_agent("research")
        stage_workspace(run_id, target)
        update_run(
            run_id,
            status="researching",
            baseline_hash=manifest_hash(target),
            research_agent_id=research_agent["id"],
            dossier=None,
            draft_plan=None,
            error=None,
        )
        db.add_event(run_id, "plan.redo", f"Plan redo queued with {research_agent['name']}")
        enqueue_job(run_id, "research")
        return db.one("select * from runs where id=?", (run_id,)) or {}


def verification_prompt(run: dict[str, Any], implementation_summary: str) -> str:
    return build_verification_prompt(run["approved_plan"] or "", implementation_summary)


def brain_verdict(run: dict[str, Any], reports: list[str]) -> tuple[bool, str]:
    prompt = build_verdict_prompt(run["approved_plan"] or "", reports)
    brain_result = call_brain_with_memory(run["id"], run["brain_provider"], "verdict", prompt)
    record_usage(run["id"], brain_result, "brain")
    verdict = parse_verdict(brain_result["content"])
    return verdict.passed, verdict.to_json()


def _run_subtask(run_id: str, node: dict[str, Any], base_stage: Path) -> dict[str, Any]:
    """Execute one subtask in its own isolated worktree. Runs on a pool thread."""
    subtask_id = node["id"]
    if (db.one("select status from runs where id=?", (run_id,)) or {}).get("status") == "cancelled":
        raise RuntimeError("Run cancelled")
    agent = choose_subtask_agent(node, run_id=run_id)
    worktree = stage_subtask(run_id, node["node_id"], base_stage)
    db.update_subtask(subtask_id, status="running", agent_id=agent["id"], worktree_ref=str(worktree), attempts=int(node.get("attempts") or 0) + 1)
    db.add_event(run_id, "subtask.started", f"{node['title']} started with {agent['name']}", {"node_id": node["node_id"]})
    globs = json.loads(node.get("file_globs_json") or "[]")
    task = (
        "Implement ONLY this subtask, completely, editing only files in its scope.\n\n"
        f"SUBTASK: {node['title']}\n\nSPEC:\n{node['spec']}\n\n"
        + (f"FILE SCOPE: {', '.join(globs)}\n\n" if globs else "")
        + (f"ACCEPTANCE CRITERIA:\n{node['acceptance_criteria']}\n" if node.get("acceptance_criteria") else "")
    )
    result = worker_call(
        run_id, agent["model"], "implementation", agent_task(agent, task),
        workspace=f"subtasks/{node['node_id']}/workspace", max_turns=32,
    )
    record_usage(run_id, result)
    db.add_subtask_result(subtask_id, run_id, "implementation", result.get("content", ""))
    if not result.get("ok"):
        db.update_subtask(subtask_id, status="failed", result_summary=(result.get("error") or "failed")[:2000])
        raise RuntimeError(f"Subtask {node['node_id']} failed: {result.get('error') or 'worker error'}")
    db.update_subtask(subtask_id, status="done", result_summary=result.get("content", "")[:4000])
    db.add_event(run_id, "subtask.completed", f"{node['title']} complete", {"node_id": node["node_id"]})
    return node


def execute_dag(run_id: str, base_stage: Path) -> tuple[str, set[str]]:
    """Run a run's subtask DAG with a bounded worker pool, then merge into base_stage.

    Dependency-gated: a subtask starts only when all its depends_on are done. Concurrency
    is capped at settings.worker_pool_size (default 1 = serialized on a single GPU). Already
    'done' subtasks are skipped so a restarted run resumes rather than repeats. Returns a
    combined summary and the set of implementer agent ids (to exclude from verification).
    """
    nodes = db.subtasks(run_id)
    status = {n["node_id"]: n["status"] for n in nodes}
    remaining = [n for n in nodes if status.get(n["node_id"]) != "done"]

    def deps_done(node: dict[str, Any]) -> bool:
        return all(status.get(dep) == "done" for dep in json.loads(node.get("depends_on_json") or "[]"))

    pool = ThreadPoolExecutor(max_workers=max(1, settings.worker_pool_size), thread_name_prefix=f"dag-{run_id[:8]}")
    futures: dict[Any, dict[str, Any]] = {}
    try:
        while remaining or futures:
            if (db.one("select status from runs where id=?", (run_id,)) or {}).get("status") == "cancelled":
                raise RuntimeError("Run cancelled")
            ready = [n for n in remaining if deps_done(n) and n["node_id"] not in {v["node_id"] for v in futures.values()}]
            for node in ready:
                remaining.remove(node)
                futures[pool.submit(_run_subtask, run_id, node, base_stage)] = node
            if not futures:
                raise RuntimeError("Subtask DAG deadlocked (unsatisfiable dependencies)")
            done, _ = wait(list(futures), return_when=FIRST_COMPLETED)
            for future in done:
                node = futures.pop(future)
                future.result()  # re-raise subtask failure
                status[node["node_id"]] = "done"
    except Exception:
        for future in futures:
            future.cancel()
        raise
    finally:
        pool.shutdown(wait=True)

    worktrees = [(n["node_id"], settings.jobs_dir / run_id / "subtasks" / n["node_id"] / "workspace") for n in nodes]
    merged = merge_worktrees(base_stage, worktrees)
    save_artifact(run_id, "merge", "Merged subtask manifest", merged)
    db.add_event(run_id, "subtasks.merged", f"Merged {merged['subtasks']} subtask worktree(s)", merged)
    completed = db.subtasks(run_id)
    implementer_ids = {n["agent_id"] for n in completed if n.get("agent_id")}
    summary = "\n\n".join(f"### {n['title']}\n{n.get('result_summary') or ''}" for n in completed)
    return summary, implementer_ids


def implement_run(run_id: str) -> None:
    try:
        run = db.one("select * from runs where id=?", (run_id,))
        if not run or run["status"] == "cancelled":
            return
        base_stage = settings.jobs_dir / run_id / "workspace"
        subtask_nodes = db.subtasks(run_id)
        if subtask_nodes:
            # Multi-agent path: execute the approved task DAG, then merge into base_stage.
            db.add_event(run_id, "implementation.started", f"Executing {len(subtask_nodes)} subtask(s) (pool {settings.worker_pool_size})")
            summary, implementer_ids = execute_dag(run_id, base_stage)
            repair_agent = choose_agent("implementation")
        else:
            # Single-agent path (backward compatible): one implementer runs the whole plan.
            agent = db.one("select * from agent_profiles where id=?", (run["implementation_agent_id"],))
            if not agent:
                raise RuntimeError("Implementation agent missing")
            db.add_event(run_id, "implementation.started", f"Implementation started with {agent['name']}")
            task = "Implement this approved plan completely.\n\n" + (run["approved_plan"] or "")
            implementation = worker_call(run_id, agent["model"], "implementation", agent_task(agent, task), max_turns=32)
            record_usage(run_id, implementation)
            summary = implementation.get("content", "")
            save_artifact(run_id, "implementation", "Implementation transcript", implementation)
            if not implementation.get("ok"):
                raise RuntimeError(implementation.get("error") or "Implementation agent failed")
            implementer_ids = {agent["id"]}
            repair_agent = agent
        if (db.one("select status from runs where id=?", (run_id,)) or {}).get("status") == "cancelled":
            return
        update_run(run_id, status="verifying")
        for repair in range(3):
            run = db.one("select * from runs where id=?", (run_id,)) or run
            first = choose_agent("verification", set(implementer_ids))
            try:
                second = choose_agent("verification", implementer_ids | {first["id"]})
            except RuntimeError:
                second = first
            reports = []
            verifier_ids = []
            for verifier in (first, second):
                result = worker_call(run_id, verifier["model"], "verification", agent_task(verifier, verification_prompt(run, summary)), max_turns=18)
                record_usage(run_id, result)
                reports.append(result.get("content", ""))
                verifier_ids.append(verifier["id"])
                save_artifact(run_id, "verification", verifier["name"], result)
                verifier_passed = bool(result.get("ok")) and result.get("content", "").lstrip().upper().startswith("PASS")
                db.execute(
                    "insert into verification_results(run_id,agent_id,cycle,report,passed,created_at) values(?,?,?,?,?,?)",
                    (run_id, verifier["id"], repair, result.get("content", ""), int(verifier_passed), db.utcnow()),
                )
                if not result.get("ok"):
                    raise RuntimeError(result.get("error") or f"Verifier {verifier['name']} failed")
            if (db.one("select status from runs where id=?", (run_id,)) or {}).get("status") == "cancelled":
                return
            update_run(run_id, verification_agent_ids_json=json.dumps(verifier_ids))
            passed, verdict = brain_verdict(run, reports)
            update_run(run_id, verdict=verdict)
            db.add_event(run_id, "verification.completed", "Supervisor verdict", {"passed": passed, "repair": repair})
            if passed:
                break
            try:
                parsed = json.loads(verdict)
            except json.JSONDecodeError:
                parsed = {}
            if parsed.get("scope_expansion"):
                expansion = parsed.get("repair_task") or parsed.get("verdict") or "Verifier requested expanded scope."
                revised_plan = (run["approved_plan"] or "") + "\n\n## Requested Scope Expansion\n\n" + expansion
                update_run(run_id, status="awaiting_approval", draft_plan=revised_plan, error="Scope expansion requires approval")
                db.add_event(run_id, "scope.approval_required", "Repair exceeds approved scope")
                return
            if repair >= 2:
                update_run(run_id, status="failed", error="Verification failed after two repair cycles")
                return
            repair_task = parsed.get("repair_task") or "Fix every defect in these verifier reports:\n" + "\n\n".join(reports)
            update_run(run_id, status="implementing", repair_count=repair + 1)
            repair_result = worker_call(run_id, repair_agent["model"], "implementation", agent_task(repair_agent, repair_task), max_turns=24)
            record_usage(run_id, repair_result)
            if not repair_result.get("ok"):
                raise RuntimeError(repair_result.get("error") or "Repair agent failed")
            summary = repair_result.get("content", "")
            save_artifact(run_id, "repair", f"Repair cycle {repair + 1}", repair_result)
            update_run(run_id, status="verifying")
        if (db.one("select status from runs where id=?", (run_id,)) or {}).get("status") == "cancelled":
            return
        target = Path(run["target_path"])
        if manifest_hash(target) != run["baseline_hash"]:
            raise RuntimeError("Workspace changed during run; verified stage was not applied")
        update_run(run_id, status="applying")
        stage = settings.jobs_dir / run_id / "workspace"
        changes = apply_stage(target, stage)
        save_artifact(run_id, "changes", "Applied file manifest", changes)
        db.add_event(run_id, "apply.completed", "Verified changes applied", changes)
        update_run(run_id, status="post_check")
        stage_workspace(run_id, target, "postcheck")
        verifier = choose_agent("verification", set(implementer_ids))
        post = worker_call(
            run_id, verifier["model"], "verification",
            agent_task(verifier, "Post-apply check. Verify copied final server state still satisfies approved plan. Start with PASS or FAIL.\n\n" + (run["approved_plan"] or "")),
            workspace="postcheck", max_turns=18,
        )
        record_usage(run_id, post)
        save_artifact(run_id, "post_check", "Post-apply verification", post)
        post_pass = post.get("ok") and post.get("content", "").lstrip().upper().startswith("PASS")
        if not post_pass:
            restore_snapshot(run["snapshot_id"])
            update_run(run_id, status="rolled_back", error="Post-apply verification failed; snapshot restored", completed_at=db.utcnow())
            db.add_event(run_id, "rollback.completed", "Post-check failed; snapshot restored")
            return
        completed = db.utcnow()
        update_run(run_id, status="completed", completed_at=completed)
        final_run = db.one("select verdict from runs where id=?", (run_id,)) or {}
        db.execute("update history_snippets set final_verdict=?,completed_at=? where run_id=?", (final_run.get("verdict"), completed, run_id))
        db.add_event(run_id, "run.completed", "Run completed successfully")
        _subtask_assignment_counts.pop(run_id, None)
    except MergeConflict as exc:
        log.warning("merge_conflict", extra={"run_id": run_id})
        update_run(run_id, status="failed", error=str(exc)[:4000])
        db.add_event(run_id, "subtasks.conflict", "Subtask worktrees conflict; re-plan with disjoint scopes", {"conflicts": exc.conflicts})
    except Exception as exc:
        log.exception("implementation_failed", extra={"run_id": run_id})
        run = db.one("select * from runs where id=?", (run_id,))
        if run and run.get("status") == "cancelled":
            return
        if run and run.get("snapshot_id") and run.get("status") in {"applying", "post_check"}:
            try:
                restore_snapshot(run["snapshot_id"])
                update_run(run_id, status="rolled_back", error=str(exc)[:4000], completed_at=db.utcnow())
            except Exception as rollback_error:
                update_run(run_id, status="failed", error=f"{exc}; rollback failed: {rollback_error}"[:4000])
        else:
            update_run(run_id, status="failed", error=str(exc)[:4000])
        db.add_event(run_id, "run.failed", "Implementation workflow failed", {"error": str(exc)[:1000]})


def cancel_run(run_id: str) -> None:
    run = db.one("select status from runs where id=?", (run_id,))
    if not run:
        raise RuntimeError("Run not found")
    if run["status"] in {"completed", "rolled_back", "applying", "post_check"}:
        raise RuntimeError("Run cannot be cancelled while applying or after completion")
    update_run(run_id, status="cancelled", completed_at=db.utcnow())
    now = db.utcnow()
    db.execute(
        "update jobs set status=case when status='pending' then 'cancelled' else status end,cancel_requested_at=?,updated_at=? "
        "where run_id=? and status in ('pending','running')",
        (now, now, run_id),
    )
    with contextlib.suppress(Exception):
        with httpx.Client(timeout=5) as client:
            client.post(
                f"{settings.worker_url}/cancel/{run_id}",
                headers={"X-Worker-Token": settings.worker_token},
            ).raise_for_status()
    db.add_event(run_id, "run.cancelled", "Cancellation requested")


def resume_run(run_id: str) -> None:
    run = db.one("select * from runs where id=?", (run_id,))
    if not run or run["status"] != "failed":
        raise RuntimeError("Only failed runs can resume")
    target = Path(run["target_path"])
    current_hash = manifest_hash(target)
    if current_hash != run.get("baseline_hash"):
        stage_workspace(run_id, target)
        update_run(
            run_id,
            status="researching",
            baseline_hash=current_hash,
            dossier=None,
            draft_plan=None,
            approved_plan=None,
            snapshot_id=None,
            error="Workspace changed; plan refresh required",
        )
        db.add_event(run_id, "plan.stale", "Workspace changed; research restarted")
        enqueue_job(run_id, "research")
        return
    if run["approved_plan"]:
        update_run(run_id, status="implementing", error=None)
        enqueue_job(run_id, "implementation")
    else:
        stage_workspace(run_id, target)
        update_run(run_id, status="researching", baseline_hash=current_hash, error=None)
        enqueue_job(run_id, "research")


def rollback_run(run_id: str) -> None:
    run = db.one("select * from runs where id=?", (run_id,))
    if not run or not run["snapshot_id"]:
        raise RuntimeError("Run has no snapshot")
    if run["status"] not in {"completed", "failed", "rolled_back"}:
        raise RuntimeError("Active run cannot be rolled back")
    restore_snapshot(run["snapshot_id"])
    update_run(run_id, status="rolled_back", completed_at=db.utcnow())
    db.add_event(run_id, "rollback.completed", "Snapshot restored by user")
