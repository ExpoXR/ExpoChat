import contextlib
import hashlib
import json
import logging
import secrets
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx

from . import db
from .brain_io import (
    build_decompose_prompt,
    build_plan_prompt,
    build_provisional_plan_prompt,
    build_refine_plan_prompt,
    build_subtask_verify_prompt,
    build_verdict_prompt,
    build_verification_prompt,
    extract_json,
    parse_subtask_verdict,
    parse_verdict,
)
from .config import settings
from .plan_graph import GraphError, independent_pairs_sharing_globs, validate_graph
from .prompts import with_caveman
from .run_state import validate_transition
from .security import decrypt_secret
from .verification_policy import (
    evaluate_apply_gate,
    record_check_evidence,
)
from .workspace import (
    MergeConflict,
    apply_stage,
    cleanup_run_jobs,
    create_snapshot,
    discard_snapshot,
    manifest_hash,
    merge_task_lineage,
    restore_snapshot,
    stage_subtask_with_ancestors,
    stage_workspace,
    workspace_delta,
    workspace_manifest,
)

log = logging.getLogger("ollma.orchestrator")
BRAIN_CALL_ATTEMPTS = 3
_queue_workers = max(settings.runner_concurrency, settings.worker_pool_size)
executor = ThreadPoolExecutor(max_workers=_queue_workers, thread_name_prefix="ollma-runner")
_run_lock = threading.RLock()
_queue_lock = threading.Lock()
_active_drainers = 0
_brain_locks: dict[str, threading.Lock] = {}
_brain_locks_guard = threading.Lock()
_ollama_monitor_stop = threading.Event()
_ollama_monitor_thread: threading.Thread | None = None
_jobs_sweeper_stop = threading.Event()
_jobs_sweeper_wake = threading.Event()
_jobs_sweeper_thread: threading.Thread | None = None

TERMINAL_RUN_STATES = frozenset({"completed", "failed", "cancelled", "rolled_back"})
# Only these terminal states have their working job tree reclaimed. 'failed' is
# excluded because it is resumable (resume_run reuses jobs/<run_id> without
# restaging), so deleting its tree would break resume.
SWEEPABLE_RUN_STATES = frozenset({"completed", "cancelled", "rolled_back"})
# A directory with no runs row is only a crash orphan once it is older than this;
# younger ones may be an in-flight create_run (which stages before inserting the row).
ORPHAN_JOB_GRACE_SECONDS = 900


class OllamaUnavailable(RuntimeError):
    pass


def host_reachable(base_url: str, timeout: float = 3.0) -> bool:
    """Probe one Ollama host's /api/version. Pure check — no DB writes."""
    try:
        with httpx.Client(timeout=timeout) as client:
            response = client.get(base_url + "/api/version")
            response.raise_for_status()
            return bool(response.json().get("version"))
    except Exception:
        return False


def refresh_host_statuses(timeout: float = 3.0) -> bool:
    """Probe every enabled host, persist status/last_seen, and report if any is reachable.

    Single writer of host health so the UI reads fresh status without extra probing.
    """
    any_ok = False
    for host in db.ollama_hosts(enabled_only=True):
        ok = host_reachable(host["base_url"], timeout)
        db.set_host_status(host["id"], "reachable" if ok else "unreachable", None if ok else "version probe failed")
        any_ok = any_ok or ok
    return any_ok


def ollama_available(timeout: float = 3.0) -> bool:
    """True if at least one enabled host answers. Stops at the first reachable host."""
    return any(host_reachable(h["base_url"], timeout) for h in db.ollama_hosts(enabled_only=True))


def _brain_lock(run_id: str) -> threading.Lock:
    with _brain_locks_guard:
        return _brain_locks.setdefault(run_id, threading.Lock())


def _pause_job_for_ollama(job: dict[str, Any], message: str) -> None:
    run = db.one("select * from runs where id=?", (job["run_id"],)) or {}
    prior = run.get("status") or "implementing"
    if prior == "post_check" and run.get("snapshot_id"):
        restore_snapshot(run["snapshot_id"])
        prior = "verifying"
        db.add_event(job["run_id"], "ollama.postcheck_paused", "Post-check paused; snapshot restored before retry")
    now = db.utcnow()
    db.execute(
        "update jobs set status='waiting_ollama',attempts=max(0,attempts-1),error=null,wait_reason=?,"
        "lease_owner=null,lease_expires_at=null,next_attempt_at=?,updated_at=? where id=?",
        (message[:1000], (datetime.now(UTC) + timedelta(seconds=30)).isoformat(), now, job["id"]),
    )
    db.execute(
        "update jobs set status='waiting_ollama',wait_reason=?,next_attempt_at=?,updated_at=? "
        "where run_id=? and status='pending' and job_type in ('research','implementation','subtask','merge')",
        (
            message[:1000],
            (datetime.now(UTC) + timedelta(seconds=30)).isoformat(),
            now,
            job["run_id"],
        ),
    )
    db.execute(
        "update subtasks set status='waiting_ollama',blocked_reason=?,updated_at=? "
        "where run_id=? and status='pending' and node_id in "
        "(select node_id from jobs where run_id=? and status='waiting_ollama' and node_id is not null)",
        (message[:1000], now, job["run_id"], job["run_id"]),
    )
    if job.get("node_id"):
        db.execute(
            "update subtasks set status='waiting_ollama',attempts=max(0,attempts-1),blocked_reason=?,updated_at=? "
            "where run_id=? and node_id=?",
            (message[:1000], now, job["run_id"], job["node_id"]),
        )
    if prior != "waiting_for_ollama":
        update_run(
            job["run_id"],
            status="waiting_for_ollama",
            resume_status=prior,
            wait_reason=message[:1000],
            next_retry_at=(datetime.now(UTC) + timedelta(seconds=30)).isoformat(),
        )
        db.add_event(job["run_id"], "ollama.waiting", "Ollama offline; work saved and paused")


def recover_waiting_ollama() -> int:
    if not ollama_available():
        return 0
    waiting = db.all_rows("select * from jobs where status='waiting_ollama' order by id")
    if waiting:
        try:
            discover_agents()
        except Exception:
            log.exception("ollama_recovery_discovery_failed")
            return 0
    now = db.utcnow()
    run_ids = {row["run_id"] for row in waiting}
    for run_id in run_ids:
        run = db.one("select * from runs where id=?", (run_id,)) or {}
        resume = run.get("resume_status") or "researching"
        update_run(
            run_id,
            status=resume,
            wait_reason=None,
            next_retry_at=None,
            resume_status=None,
            error=None,
        )
        db.add_event(run_id, "ollama.reconnected", "Ollama online; queued work resumed")
    if waiting:
        db.execute(
            "update jobs set status='pending',wait_reason=null,next_attempt_at=null,updated_at=? "
            "where status='waiting_ollama'",
            (now,),
        )
        db.execute(
            "update subtasks set status='pending',blocked_reason=null,updated_at=? where status='waiting_ollama'",
            (now,),
        )
        start_job_queue()
    return len(waiting)


def _ollama_monitor() -> None:
    delay = 5.0
    while not _ollama_monitor_stop.wait(delay):
        if refresh_host_statuses():
            recover_waiting_ollama()
            delay = 10.0
        else:
            delay = min(60.0, delay * 1.5)


def start_ollama_monitor() -> None:
    global _ollama_monitor_thread
    if _ollama_monitor_thread and _ollama_monitor_thread.is_alive():
        return
    _ollama_monitor_stop.clear()
    _ollama_monitor_thread = threading.Thread(target=_ollama_monitor, name="ollma-connectivity", daemon=True)
    _ollama_monitor_thread.start()


def stop_ollama_monitor() -> None:
    global _ollama_monitor_thread
    _ollama_monitor_stop.set()
    if _ollama_monitor_thread:
        _ollama_monitor_thread.join(timeout=2)
    _ollama_monitor_thread = None


def _has_active_jobs(run_id: str) -> bool:
    row = db.one(
        "select 1 from jobs where run_id=? and status in ('running','pending','waiting_ollama') limit 1",
        (run_id,),
    )
    return row is not None


def _forget_run_memory(run_id: str) -> None:
    """Drop per-run in-memory bookkeeping so failed runs don't leak it."""
    _subtask_assignment_counts.pop(run_id, None)
    with _brain_locks_guard:
        _brain_locks.pop(run_id, None)


def sweep_orphan_jobs() -> int:
    """Delete working job trees for runs that are terminal (and idle) or gone.

    Runs the correct safety guard — a directory is removed only when its run has
    no active jobs — so it never races a live worktree. Also reclaims dirs whose
    run row no longer exists (crash orphans). Returns the number removed.
    """
    jobs_dir = settings.jobs_dir
    if not jobs_dir.exists():
        return 0
    now = time.time()
    removed = 0
    for child in list(jobs_dir.iterdir()):
        if not child.is_dir():
            continue
        run_id = child.name
        run = db.one("select status from runs where id=?", (run_id,))
        if run is None:
            # No run row: a genuine crash orphan OR a create_run staging its tree
            # before inserting the row. Only reclaim once it is older than the grace
            # window so we never delete an in-flight run's directory.
            try:
                age = now - child.stat().st_mtime
            except OSError:
                continue
            if age < ORPHAN_JOB_GRACE_SECONDS:
                continue
            if cleanup_run_jobs(run_id):
                removed += 1
            _forget_run_memory(run_id)
            continue
        if run["status"] in SWEEPABLE_RUN_STATES and not _has_active_jobs(run_id):
            if cleanup_run_jobs(run_id):
                removed += 1
            _forget_run_memory(run_id)
    return removed


def _jobs_sweeper() -> None:
    while not _jobs_sweeper_stop.is_set():
        # Wake promptly on a terminal transition, else re-check periodically as a
        # backstop for jobs that were still 'running' at the last pass.
        _jobs_sweeper_wake.wait(timeout=120)
        _jobs_sweeper_wake.clear()
        if _jobs_sweeper_stop.is_set():
            break
        try:
            sweep_orphan_jobs()
        except Exception:
            log.exception("jobs sweeper failed")


def trigger_jobs_sweep() -> None:
    _jobs_sweeper_wake.set()


def start_jobs_sweeper() -> None:
    global _jobs_sweeper_thread
    if _jobs_sweeper_thread and _jobs_sweeper_thread.is_alive():
        return
    _jobs_sweeper_stop.clear()
    _jobs_sweeper_thread = threading.Thread(target=_jobs_sweeper, name="ollma-jobs-sweeper", daemon=True)
    _jobs_sweeper_thread.start()


def stop_jobs_sweeper() -> None:
    global _jobs_sweeper_thread
    _jobs_sweeper_stop.set()
    _jobs_sweeper_wake.set()
    if _jobs_sweeper_thread:
        _jobs_sweeper_thread.join(timeout=2)
    _jobs_sweeper_thread = None


def _lease_heartbeat(job_id: int, lease_owner: str, stopped: threading.Event) -> None:
    while not stopped.wait(30):
        expires = (datetime.now(UTC) + timedelta(minutes=20)).isoformat()
        db.execute(
            "update jobs set lease_expires_at=?,updated_at=? where id=? and status='running' and lease_owner=?",
            (expires, db.utcnow(), job_id, lease_owner),
        )


def enqueue_job(run_id: str, job_type: str, node_id: str | None = None) -> None:
    now = db.utcnow()
    with db.transaction() as conn:
        if node_id:
            active = conn.execute(
                "select id from jobs where run_id=? and node_id=? and status in ('pending','running','waiting_ollama') limit 1",
                (run_id, node_id),
            ).fetchone()
        else:
            active = conn.execute(
                "select id from jobs where run_id=? and job_type=? and status in ('pending','running','waiting_ollama') limit 1",
                (run_id, job_type),
            ).fetchone()
        if not active:
            conn.execute(
                "insert into jobs(run_id,job_type,node_id,status,created_at,updated_at) values(?,?,?,'pending',?,?)",
                (run_id, job_type, node_id, now, now),
            )
    start_job_queue()


def enqueue_waiting_job(run_id: str, job_type: str, node_id: str | None = None) -> None:
    now = db.utcnow()
    with db.transaction() as conn:
        conn.execute(
            "insert or ignore into jobs(run_id,job_type,node_id,status,wait_reason,next_attempt_at,created_at,updated_at) "
            "values(?,?,?,'waiting_ollama','Ollama offline',?,?,?)",
            (run_id, job_type, node_id, (datetime.now(UTC) + timedelta(seconds=30)).isoformat(), now, now),
        )


def start_job_queue() -> None:
    global _active_drainers
    with _queue_lock:
        count = _queue_workers - _active_drainers
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
                row = conn.execute(
                    "select j.* from jobs j where j.status='pending' and (j.job_type!='subtask' or "
                    "(select count(*) from jobs active where active.run_id=j.run_id and active.job_type='subtask' "
                    "and active.status='running')<?) order by j.id limit 1",
                    (max(1, settings.worker_pool_size),),
                ).fetchone()
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
                if job["job_type"] == "provisional":
                    provisional_plan_run(job["run_id"])
                elif job["job_type"] == "decompose":
                    regenerate_graph(job["run_id"])
                elif job["job_type"] == "research":
                    research_run(job["run_id"])
                elif job["job_type"] == "subtask":
                    _run_durable_subtask(job["run_id"], job)
                elif job["job_type"] == "merge":
                    _merge_and_verify(job["run_id"])
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
            except OllamaUnavailable as exc:
                _pause_job_for_ollama(job, str(exc) or "Ollama offline")
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
    # The brain call is the single most critical dependency (plan/decompose/verdict
    # /subtask-verify all funnel here); a lone transient blip must not kill a run.
    # Retry network errors, timeouts, 429 and 5xx with exponential backoff; fail
    # fast on 4xx (auth/bad request) and never mask budget errors (raised above).
    attempts = BRAIN_CALL_ATTEMPTS
    for attempt in range(attempts):
        last = attempt + 1 >= attempts
        try:
            with httpx.Client(timeout=timeout + 10) as client:
                response = client.post(
                    settings.brain_url + "/execute",
                    headers={"X-Worker-Token": settings.worker_token},
                    json=payload,
                )
        except httpx.HTTPError as exc:
            if last:
                raise RuntimeError(f"Brain service unavailable: {exc}") from exc
            time.sleep(1 << attempt)
            continue
        if response.status_code == 200:
            data = response.json()
            return {"content": str(data["content"]), "usage": data.get("usage") or {}}
        try:
            detail = response.json().get("detail")
        except Exception:
            detail = None
        retryable = response.status_code == 429 or response.status_code >= 500
        if retryable and not last:
            time.sleep(1 << attempt)
            continue
        if detail:
            raise RuntimeError(str(detail))
        raise RuntimeError(f"Brain service failed with HTTP {response.status_code}")
    raise RuntimeError("Brain service unavailable")


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
    with _brain_lock(run_id):
        digest = _build_memory_digest(run_id)
        full_prompt = digest + prompt if digest else prompt
        result = call_brain_result(provider, full_prompt, allow_web, timeout, run_id=run_id)
        summary = prompt[:2000]
        db.add_brain_memory(run_id, step, "prompt", summary)
        db.add_brain_memory(run_id, step, "response", result["content"][:4000])
        return result


def worker_call(run_id: str, model: str, mode: str, task: str, workspace: str = "workspace", max_turns: int = 24, node_id: str | None = None, host_id: str | None = None) -> dict[str, Any]:
    # Implementation restages an isolated worktree per attempt, so a transient network blip
    # is safe to retry once (was attempts=1, which failed a whole subtask on one hiccup).
    attempts = 2 if mode == "implementation" else 3
    for attempt in range(attempts):
        try:
            with httpx.Client(timeout=1200) as client:
                with client.stream(
                    "POST",
                    settings.worker_url + "/execute/stream",
                    headers={"X-Worker-Token": settings.worker_token},
                    json={"run_id": run_id, "workspace": workspace, "model": model, "mode": mode, "task": task, "max_turns": max_turns, **({"node_id": node_id} if node_id else {}), **({"ollama_host_id": host_id} if host_id else {})},
                ) as response:
                    response.raise_for_status()
                    for line in response.iter_lines():
                        if not line:
                            continue
                        item = json.loads(line)
                        if item.get("type") == "result":
                            return item["result"]
                        if item.get("type") == "error":
                            error = item.get("error") or "Worker failed"
                            if "Ollama request failed" in error and not ollama_available():
                                raise OllamaUnavailable("Ollama offline")
                            raise RuntimeError(error)
                        record_worker_activity(run_id, mode, item, node_id=node_id)
        except httpx.HTTPError as exc:
            if attempt + 1 >= attempts:
                if not ollama_available():
                    raise OllamaUnavailable("Ollama offline") from exc
                raise
            time.sleep(1 << attempt)
    raise RuntimeError("Worker unavailable")


def record_worker_activity(run_id: str, mode: str, item: dict[str, Any], *, node_id: str | None = None) -> None:
    kind = str(item.get("type") or "")
    if kind == "message":
        content = " ".join(str(item.get("content") or "").split())[:1000]
        if content:
            data: dict[str, Any] = {"phase": mode, "state": "message"}
            if node_id:
                data["node_id"] = node_id
            db.add_event(run_id, "agent.activity", content, data)
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
    data = {"phase": mode, "state": state, "tool": tool, "path": path, "turn": item.get("turn")}
    if node_id:
        data["node_id"] = node_id
    db.add_event(run_id, "agent.activity", message, data)


def _score_model(model: str, show: dict[str, Any]) -> dict[str, Any]:
    """Derive roles / role-scores / priority for a model from its /api/show payload."""
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
    return {"capabilities": capabilities, "context": context, "roles": roles, "scores": scores, "priority": priority}


def discover_agents(host_id: str | None = None) -> list[dict[str, Any]]:
    """Discover Ollama models across hosts and upsert them as agent_profiles.

    host_id=None sweeps every enabled host; a specific id scans just that host. Each host
    is probed independently: an unreachable host is flagged and skipped (never aborts the
    sweep), and a single model whose /api/show fails is skipped without losing the rest.
    Agents are keyed by (host_id, model), so the same model on two devices is two agents.
    """
    now = db.utcnow()
    hosts = [db.ollama_host(host_id)] if host_id else db.ollama_hosts(enabled_only=True)
    hosts = [h for h in hosts if h]
    discovered: list[dict[str, Any]] = []
    for host in hosts:
        hid, base = host["id"], host["base_url"]
        try:
            # Short timeout so a dead host doesn't stall a multi-host sweep; per-model /api/show
            # below keeps a longer budget for the reachable host's model metadata.
            with httpx.Client(timeout=10) as client:
                tags_response = client.get(base + "/api/tags")
                tags_response.raise_for_status()
                tags = tags_response.json().get("models", [])
        except Exception as exc:
            db.set_host_status(hid, "unreachable", str(exc))
            log.warning("discover_host_unreachable host=%s base=%s err=%s", hid, base, exc)
            continue
        db.set_host_status(hid, "reachable")
        for item in tags:
            model = item.get("name") or item.get("model")
            if not model:
                continue
            try:
                with httpx.Client(timeout=30) as client:
                    show_response = client.post(base + "/api/show", json={"model": model})
                    show_response.raise_for_status()
                    show = show_response.json()
            except Exception as exc:
                log.warning("discover_model_failed host=%s model=%s err=%s", hid, model, exc)
                continue
            prof = _score_model(model, show)
            existing = db.one("select id from agent_profiles where host_id=? and model=?", (hid, model))
            if existing:
                agent_id = existing["id"]
                db.execute(
                    "update agent_profiles set capabilities_json=?,context_size=?,host_base_url=?,discovered_at=?,updated_at=? where id=?",
                    (json.dumps(prof["capabilities"]), prof["context"], base, now, now, agent_id),
                )
            else:
                agent_id = "agent-" + secrets.token_hex(8)
                name = model if hid == db.DEFAULT_HOST_ID else f"{model} @ {host['name']}"
                db.execute(
                    "insert into agent_profiles(id,name,model,roles_json,capabilities_json,context_size,priority,"
                    "role_scores_json,host_id,host_base_url,discovered_at,updated_at) values(?,?,?,?,?,?,?,?,?,?,?,?)",
                    (agent_id, name, model, json.dumps(prof["roles"]), json.dumps(prof["capabilities"]),
                     prof["context"], prof["priority"], json.dumps(prof["scores"]), hid, base, now, now),
                )
            discovered.append(db.one("select * from agent_profiles where id=?", (agent_id,)) or {})
    return discovered


def _agent_is_eligible(agent: dict[str, Any] | None, role: str, exclude: set[str] | None = None) -> bool:
    if not agent or not agent.get("enabled") or agent["id"] in (exclude or set()):
        return False
    roles = json.loads(agent.get("roles_json") or "[]")
    capabilities = json.loads(agent.get("capabilities_json") or "[]")
    return role in roles and (role != "implementation" or "tools" in capabilities)


def _reachable_host_ids() -> set[str]:
    """Enabled hosts an agent may be dispatched to.

    'unknown' counts as eligible so a cold start (before the connectivity monitor has run
    a probe) isn't starved of every agent; a genuinely-down host is still caught at dispatch
    via the OllamaUnavailable pause/resume path.
    """
    return {h["id"] for h in db.ollama_hosts(enabled_only=True) if h["status"] in ("reachable", "unknown")}


def _candidate_agents(role: str, exclude: set[str] | None = None) -> list[tuple]:
    """Score + sort eligible agents for a role, preferring reachable hosts.

    Host reachability is a best-effort hint: agents on a reachable/unknown host are preferred,
    but if that would leave no candidates (e.g. a stale/transient 'unreachable' status), we
    fall back to all eligible agents rather than starving selection — a genuinely-down host is
    still caught at dispatch by the OllamaUnavailable pause/resume path.
    """
    exclude = exclude or set()
    reachable = _reachable_host_ids()
    eligible: list[tuple] = []
    for row in db.all_rows("select * from agent_profiles where enabled=1"):
        if not _agent_is_eligible(row, role, exclude):
            continue
        scores = json.loads(row["role_scores_json"] or "{}")
        eligible.append((int(scores.get(role, 0)), int(row["priority"]), int(row["context_size"]), row["name"], row))
    on_reachable = [c for c in eligible if not c[-1].get("host_id") or c[-1]["host_id"] in reachable]
    candidates = on_reachable or eligible
    candidates.sort(key=lambda item: (-item[0], -item[1], -item[2], item[3].lower()))
    return candidates


def choose_agent(role: str, exclude: set[str] | None = None, discover: bool = True) -> dict[str, Any]:
    exclude = exclude or set()
    candidates = _candidate_agents(role, exclude)
    if not candidates:
        if discover:
            discover_agents()
            return choose_agent(role, exclude, False)
        raise RuntimeError(
            f"No enabled {role} agent available. Open Brains & Agents, enable a compatible Ollama model, then retry."
        )
    return candidates[0][-1]


# In-memory round-robin counters per run for distributing work across equally-scored agents.
# Guarded by _assignment_lock: with worker_pool_size>1, multiple drainer threads may
# schedule subtasks for the same run concurrently. A dedicated lock (not _queue_lock)
# avoids any reentrancy with the enqueue path that calls choose_subtask_agent.
_subtask_assignment_counts: dict[str, dict[str, int]] = {}
_assignment_lock = threading.Lock()


def choose_subtask_agent(
    node: dict[str, Any], exclude: set[str] | None = None, run_id: str | None = None,
) -> dict[str, Any]:
    """Choose an agent for a specific subtask, honoring role and optional model hint.

    0. If the user pinned assigned_agent_id and that agent is enabled+eligible -> use it.
    1. If node has suggested_model and that model is enabled+eligible -> use it.
    2. Else choose_agent(node.role, exclude) — now honoring the subtask's role.
    3. Round-robin tiebreak: when multiple agents have equal scores, rotate.
    """
    exclude = exclude or set()
    role = node.get("role") or "implementation"
    assigned_id = node.get("assigned_agent_id") or ""
    if assigned_id:
        agent = db.one("select * from agent_profiles where id=? and enabled=1", (assigned_id,))
        if assigned_id in exclude or not agent or not _agent_is_eligible(agent, role, exclude):
            raise RuntimeError(f"Pinned agent {assigned_id} is unavailable for role '{role}'")
        return agent
    candidates = _candidate_agents(role, exclude)
    if not candidates:
        return choose_agent(role, exclude)
    if node.get("complexity") == "complex":
        return candidates[0][-1]
    suggested = node.get("suggested_model") or ""
    suggested_agent = next((item[-1] for item in candidates if item[-1]["model"] == suggested), None)
    if not run_id:
        return suggested_agent or candidates[0][-1]
    with _assignment_lock:
        counts = _subtask_assignment_counts.setdefault(run_id, {})
        for row in db.all_rows(
            "select agent_id,count(*) as assignments from subtasks "
            "where run_id=? and agent_id is not null group by agent_id",
            (run_id,),
        ):
            counts[row["agent_id"]] = max(counts.get(row["agent_id"], 0), int(row["assignments"]))
        chosen = min(
            enumerate(candidates),
            key=lambda item: (
                counts.get(item[1][-1]["id"], 0),
                0 if suggested_agent and item[1][-1]["id"] == suggested_agent["id"] else 1,
                item[0],
            ),
        )[1][-1]
        counts[chosen["id"]] = counts.get(chosen["id"], 0) + 1
        return chosen


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
    with _run_lock:
        run = db.one("select usage_json,brain_provider from runs where id=?", (run_id,)) or {}
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

        provider = "ollama" if source != "brain" else (run.get("brain_provider") or "brain")
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
    became_terminal = False
    if "status" in values:
        current = db.one("select status from runs where id=?", (run_id,))
        if not current:
            raise RuntimeError("Run not found")
        validate_transition(current["status"], values["status"])
        became_terminal = (
            values["status"] in TERMINAL_RUN_STATES and current["status"] not in TERMINAL_RUN_STATES
        )
    values["updated_at"] = db.utcnow()
    columns = ",".join(f"{key}=?" for key in values)
    db.execute(f"update runs set {columns} where id=?", (*values.values(), run_id))
    if became_terminal:
        # Reclaim the run's working job tree once no jobs are active. The sweeper
        # applies the idle guard so it never races an in-flight worktree.
        trigger_jobs_sweep()
        _forget_run_memory(run_id)


def _plan_hash(plan: str) -> str:
    return hashlib.sha256(plan.strip().encode()).hexdigest()


def _workspace_inventory(target: Path, max_files: int = 800, max_chars: int = 60_000) -> str:
    manifest = workspace_manifest(target)
    lines = [f"{path} ({value['size']} bytes)" for path, value in sorted(manifest.items())[:max_files]]
    text = "\n".join(lines)
    if len(manifest) > max_files:
        text += f"\n... {len(manifest) - max_files} more files"
    return text[:max_chars]


def create_run(task: str, provider: str, target: Path, web_research: bool) -> dict[str, Any]:
    run_id = secrets.token_hex(12)
    _, model = provider_config(provider)
    stage_workspace(run_id, target)
    baseline = manifest_hash(target)
    online = ollama_available()
    research_agent = choose_agent("research") if online else None
    implementation_agent = choose_agent("implementation") if online else None
    status = "researching" if online else "planning_provisional"
    job_type = "research" if online else "provisional"
    now = db.utcnow()
    with db.transaction() as conn:
        conn.execute(
            "insert into runs(id,task,brain_provider,brain_model,target_path,web_research,status,baseline_hash,"
            "research_agent_id,implementation_agent_id,plan_state,created_at,updated_at) values(?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                run_id, task, provider, model, str(target), int(web_research), status, baseline,
                research_agent["id"] if research_agent else None,
                implementation_agent["id"] if implementation_agent else None,
                "none", now, now,
            ),
        )
        conn.execute(
            "insert into jobs(run_id,job_type,status,created_at,updated_at) values(?,?,'pending',?,?)",
            (run_id, job_type, now, now),
        )
    db.add_event(
        run_id,
        "run.created",
        f"Research queued with {research_agent['name']}" if research_agent else "Ollama offline; provisional Brain plan queued",
        {
            "provider": provider,
            "target": str(target),
            "research_agent": research_agent["id"] if research_agent else None,
            "ollama_available": online,
        },
    )
    start_job_queue()
    return db.one("select * from runs where id=?", (run_id,)) or {}


def provisional_plan_run(run_id: str) -> None:
    try:
        run = db.one("select * from runs where id=?", (run_id,))
        if not run or run["status"] == "cancelled":
            return
        inventory = _workspace_inventory(Path(run["target_path"]))
        result = call_brain_with_memory(
            run_id,
            run["brain_provider"],
            "provisional_plan",
            build_provisional_plan_prompt(run["task"], inventory),
            allow_web=bool(run["web_research"]),
        )
        record_usage(run_id, result, "brain")
        plan = result["content"].strip()
        if not plan:
            raise RuntimeError("Brain returned an empty provisional plan")
        update_run(
            run_id,
            draft_plan=plan,
            plan_state="provisional",
            status="waiting_for_ollama",
            resume_status="researching",
            wait_reason="Ollama offline; provisional plan saved",
            next_retry_at=(datetime.now(UTC) + timedelta(seconds=30)).isoformat(),
        )
        save_artifact(run_id, "plan", "Provisional Brain plan", plan)
        db.add_plan_version(run_id, "provisional", plan, run["brain_provider"])
        enqueue_waiting_job(run_id, "research")
        db.add_event(run_id, "plan.provisional", "Provisional plan saved; waiting for Ollama")
    except Exception as exc:
        if (db.one("select status from runs where id=?", (run_id,)) or {}).get("status") == "cancelled":
            return
        update_run(run_id, status="failed", error=str(exc)[:4000])
        db.add_event(run_id, "run.failed", "Provisional planning failed", {"error": str(exc)[:1000]})
        raise


def decompose_plan(run_id: str, plan: str, provider: str) -> list[dict[str, Any]]:
    """Create strict task graph matching the current plan."""
    try:
        result = call_brain_with_memory(run_id, provider, "decompose", build_decompose_prompt(plan))
        record_usage(run_id, result, "brain")
        nodes = validate_graph(extract_json(result["content"]))
        missing_scopes = [
            node["node_id"]
            for node in nodes
            if node.get("role") == "implementation" and not node.get("file_globs")
        ]
        if missing_scopes:
            raise GraphError(f"Implementation subtasks missing file scopes: {missing_scopes}")
        conflicts = independent_pairs_sharing_globs(nodes)
        if conflicts:
            raise GraphError(f"Independent subtasks share file scopes: {conflicts}")
        db.insert_subtasks(run_id, nodes)
        update_run(run_id, graph_plan_hash=_plan_hash(plan))
        save_artifact(run_id, "task_graph", "Task graph", {"subtasks": nodes})
        db.add_event(
            run_id,
            "plan.decomposed",
            f"Plan decomposed into {len(nodes)} subtask(s)",
            {"count": len(nodes), "overlap_warnings": conflicts},
        )
        return nodes
    except (GraphError, KeyError, RuntimeError, ValueError) as exc:
        log.warning("decompose_failed", extra={"run_id": run_id, "error": str(exc)})
        update_run(run_id, graph_plan_hash=None)
        db.add_event(run_id, "plan.decompose_failed", "Task graph generation failed", {"reason": str(exc)[:400]})
        raise


def research_run(run_id: str) -> None:
    try:
        run = db.one("select * from runs where id=?", (run_id,))
        if not run or run["status"] == "cancelled":
            return
        if not ollama_available():
            raise OllamaUnavailable("Ollama offline")
        target = Path(run["target_path"])
        if run.get("plan_state") == "provisional":
            stage_workspace(run_id, target)
            update_run(run_id, baseline_hash=manifest_hash(target))
            db.add_event(run_id, "plan.refining", "Ollama online; refining provisional plan")
        agent = db.one("select * from agent_profiles where id=?", (run.get("research_agent_id"),))
        if not _agent_is_eligible(agent, "research"):
            agent = choose_agent("research")
        update_run(run_id, research_agent_id=agent["id"], status="researching")
        db.add_event(run_id, "research.started", f"Research started with {agent['name']}")
        result = worker_call(run_id, agent["model"], "research", agent_task(agent, run["task"]), max_turns=18, host_id=agent.get("host_id"))
        record_usage(run_id, result)
        if not result.get("ok"):
            raise RuntimeError(result.get("error") or "Research agent failed")
        if (db.one("select status from runs where id=?", (run_id,)) or {}).get("status") == "cancelled":
            return
        dossier = result.get("content", "")
        save_artifact(run_id, "research", "Ollama research dossier", {"summary": dossier, "events": result.get("events", [])})
        db.add_event(run_id, "research.completed", "Ollama research complete")
        prompt = (
            build_refine_plan_prompt(run["task"], run.get("draft_plan") or "", dossier)
            if run.get("plan_state") == "provisional"
            else build_plan_prompt(run["task"], dossier)
        )
        brain_result = call_brain_with_memory(
            run_id, run["brain_provider"], "plan", prompt,
            allow_web=bool(run["web_research"]),
        )
        record_usage(run_id, brain_result, "brain")
        plan = brain_result["content"]
        if (db.one("select status from runs where id=?", (run_id,)) or {}).get("status") == "cancelled":
            return
        implementation = choose_agent("implementation")
        nodes = decompose_plan(run_id, plan, run["brain_provider"])
        if not nodes:
            raise RuntimeError("Task graph generation returned no tasks")
        update_run(
            run_id,
            dossier=dossier,
            draft_plan=plan,
            implementation_agent_id=implementation["id"],
            plan_state="refined",
            wait_reason=None,
            next_retry_at=None,
            resume_status=None,
            status="awaiting_approval",
        )
        save_artifact(run_id, "plan", "Supervisor plan", plan)
        db.add_plan_version(
            run_id,
            "refined" if run.get("plan_state") == "provisional" else "draft",
            plan,
            run["brain_provider"],
        )
        db.add_event(run_id, "plan.ready", "Plan ready for approval")
    except OllamaUnavailable:
        raise
    except Exception as exc:
        if not ollama_available():
            raise OllamaUnavailable("Ollama offline") from exc
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
        if run.get("plan_state") != "refined":
            raise RuntimeError("Provisional plan must be refined after Ollama reconnects")
        if not run.get("graph_plan_hash") or run["graph_plan_hash"] != _plan_hash(approved_plan):
            raise RuntimeError("Task graph is missing or stale; regenerate it before approval")
        graph = db.subtasks(run_id)
        if not graph:
            raise RuntimeError("Task graph is required before approval")
        for node in graph:
            if node.get("assigned_agent_id"):
                _validated_agent_id(node["assigned_agent_id"], node.get("role") or "implementation")
            else:
                choose_subtask_agent(node)
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
    provisional_wait = bool(
        run and run["status"] == "waiting_for_ollama" and run.get("plan_state") == "provisional"
    )
    if not run or (run["status"] != "awaiting_approval" and not provisional_wait):
        raise RuntimeError("Run is not awaiting plan edits")
    if provisional_wait:
        update_run(run_id, draft_plan=plan, error=None)
        save_artifact(run_id, "plan_edit", "User-edited provisional plan", plan)
        db.add_plan_version(run_id, "provisional_edit", plan, run["brain_provider"])
        db.add_event(run_id, "plan.edited", "Provisional plan changes saved")
        return db.one("select * from runs where id=?", (run_id,)) or {}
    update_run(
        run_id,
        draft_plan=plan,
        graph_plan_hash=None,
        plan_state="editing",
        status="decomposing",
        error=None,
    )
    db.execute("delete from subtasks where run_id=?", (run_id,))
    save_artifact(run_id, "plan_edit", "User-edited plan", plan)
    db.add_plan_version(run_id, "edit", plan, run["brain_provider"])
    db.add_event(run_id, "plan.edited", "Plan changes saved; task graph regenerating")
    enqueue_job(run_id, "decompose")
    return db.one("select * from runs where id=?", (run_id,)) or {}


def regenerate_graph(run_id: str) -> None:
    run = db.one("select * from runs where id=?", (run_id,))
    if not run or run["status"] == "cancelled":
        return
    try:
        decompose_plan(run_id, run.get("draft_plan") or "", run["brain_provider"])
        update_run(run_id, status="awaiting_approval", plan_state="refined", error=None)
        db.add_event(run_id, "plan.graph_ready", "Task graph regenerated; plan ready for approval")
    except Exception as exc:
        update_run(run_id, status="failed", error=f"Task graph regeneration failed: {exc}"[:4000])
        db.add_event(run_id, "run.failed", "Task graph regeneration failed", {"error": str(exc)[:1000]})
        raise


def _validated_agent_id(agent_id: Any, role: str) -> str | None:
    """Resolve a user-picked agent id, or None to clear. Rejects a missing/disabled/
    role-ineligible agent so the graph never pins an agent that cannot run the task."""
    value = str(agent_id or "").strip()
    if not value:
        return None
    agent = db.one("select * from agent_profiles where id=? and enabled=1", (value,))
    if not agent or not _agent_is_eligible(agent, role):
        raise RuntimeError(f"Agent {value} is not available for role '{role}'")
    return value


def edit_task_graph(run_id: str, edits: list[dict[str, Any]]) -> dict[str, Any]:
    """Apply user edits (per-node agent assignment and/or dependency rewiring) to a
    drafted task graph, re-validating the DAG (cycles/unknown deps) before persisting.

    Only allowed while the run is awaiting approval — once implementation starts the graph
    is being executed. Returns the fresh subtasks plus any file-glob overlap warnings.
    """
    with _run_lock:
        run = db.one("select * from runs where id=?", (run_id,))
        if not run or run["status"] != "awaiting_approval":
            raise RuntimeError("Task graph can only be edited while awaiting approval")
        rows = db.subtasks(run_id)
        if not rows:
            raise RuntimeError("Run has no task graph to edit")
        # Reconstruct the brain-graph node shape from persisted rows.
        by_id: dict[str, dict[str, Any]] = {}
        for row in rows:
            by_id[row["node_id"]] = {
                "node_id": row["node_id"],
                "title": row["title"],
                "spec": row["spec"],
                "depends_on": json.loads(row.get("depends_on_json") or "[]"),
                "file_globs": json.loads(row.get("file_globs_json") or "[]"),
                "acceptance_criteria": row.get("acceptance_criteria") or "",
                "role": row.get("role") or "implementation",
                "suggested_model": row.get("suggested_model"),
                "complexity": row.get("complexity") or "simple",
                "assigned_agent_id": row.get("assigned_agent_id"),
            }
        for edit in edits or []:
            node_id = str(edit.get("node_id") or "").strip()
            node = by_id.get(node_id)
            if not node:
                raise RuntimeError(f"Unknown subtask node: {node_id}")
            if "depends_on" in edit:
                node["depends_on"] = [str(dep).strip() for dep in (edit.get("depends_on") or []) if str(dep).strip()]
            if "assigned_agent_id" in edit:
                node["assigned_agent_id"] = _validated_agent_id(edit.get("assigned_agent_id"), node["role"])
        # validate_graph rejects cycles/unknown deps and normalizes; it drops assigned_agent_id,
        # so overlay the user's assignment back onto each validated node before persisting.
        try:
            validated = validate_graph({"subtasks": list(by_id.values())})
        except GraphError as exc:
            raise RuntimeError(str(exc)) from exc
        for node in validated:
            node["assigned_agent_id"] = by_id[node["node_id"]].get("assigned_agent_id")
        warnings = [list(pair) for pair in independent_pairs_sharing_globs(validated)]
        if warnings:
            raise RuntimeError(f"Independent subtasks share file scopes: {warnings}")
        db.insert_subtasks(run_id, validated)
        db.add_event(
            run_id, "plan.graph_edited",
            f"Task graph updated ({len(validated)} subtask(s))",
            {"overlap_warnings": warnings},
        )
        return {"subtasks": db.subtasks(run_id), "overlap_warnings": warnings}


def redo_plan(run_id: str) -> dict[str, Any]:
    with _run_lock:
        run = db.one("select * from runs where id=?", (run_id,))
        if not run or run["status"] not in {"awaiting_approval", "decomposing"}:
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
            graph_plan_hash=None,
            plan_state="none",
            error=None,
        )
        db.execute("delete from subtasks where run_id=?", (run_id,))
        db.execute(
            "update jobs set status='cancelled',error='Superseded by plan redo',updated_at=? "
            "where run_id=? and job_type='decompose' and status in ('pending','waiting_ollama')",
            (db.utcnow(), run_id),
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


_CHECK_EXCLUDED_DIRS = {".git", "node_modules", "__pycache__", ".next", ".cache", "dist", "build", "vendor", ".venv"}


def _workspace_file_stats(stage: Path, cap: int = 5000) -> tuple[set[str], bool, int]:
    """Bounded walk of a staged workspace, skipping vendored/build dirs.

    Returns (root-level file+dir names, whether any *.py exists, total files seen up to cap).
    """
    root_names: set[str] = set()
    has_py = False
    total = 0
    if not stage.exists():
        return root_names, has_py, total
    root_names = {p.name for p in stage.iterdir()}
    for path in stage.rglob("*"):
        if any(part in _CHECK_EXCLUDED_DIRS for part in path.relative_to(stage).parts):
            continue
        if path.is_file():
            total += 1
            if path.suffix == ".py":
                has_py = True
            if total >= cap:
                break
    return root_names, has_py, total


def _qualifying_checks_for(stage: Path) -> list[dict[str, Any]]:
    """Pick deterministic checks matching the workspace's toolchain.

    A Python project runs ruff+pytest; a Node project runs its npm test script; a workspace
    with no recognized toolchain gets a single builtin ``workspace-nonempty`` check. This
    fixes the apply gate being hardcoded to Python (ruff+pytest), which permanently blocked
    apply on any non-Python workspace — while still requiring one genuine passing machine
    check so a run never applies on the brain verdict alone.
    """
    root_names, has_py, _ = _workspace_file_stats(stage)
    py_markers = {"pyproject.toml", "setup.cfg", "setup.py"} | {
        n for n in root_names if n.startswith("requirements") and n.endswith(".txt")
    }
    if (root_names & py_markers) or has_py:
        return [
            {"command": "ruff", "args": ["check", "."], "label": "ruff-lint"},
            {"command": "pytest", "args": ["--tb=short", "-q"], "label": "pytest"},
        ]
    if "package.json" in root_names:
        return [{"command": "npm", "args": ["test", "--silent"], "label": "npm-test"}]
    return [{"command": "workspace-nonempty", "args": [], "label": "workspace-nonempty", "builtin": True}]


def _run_qualifying_checks(
    run_id: str, cycle: int, workspace: str = "workspace", node_id: str | None = None,
) -> list[dict[str, Any]]:
    """Run deterministic qualifying checks on the staged workspace via the worker.

    Records each result as structured check_evidence. Returns the list of evidence
    records. Real checks use the /check endpoint which returns exit codes directly; a
    builtin check is evaluated locally against the staged files.
    """
    stage = settings.jobs_dir / run_id / workspace
    workspace_hash = manifest_hash(stage) if stage.exists() else ""
    results: list[dict[str, Any]] = []
    for check in _qualifying_checks_for(stage):
        command = check["command"]
        args = check["args"]
        started = time.monotonic()
        if check.get("builtin"):
            _, _, total = _workspace_file_stats(stage)
            ok = total > 0
            data = {"ok": ok, "content": f"exit={0 if ok else 1}\nworkspace files: {total}"}
        else:
            try:
                with httpx.Client(timeout=300) as client:
                    response = client.post(
                        settings.worker_url + "/check",
                        headers={"X-Worker-Token": settings.worker_token},
                        json={"run_id": run_id, "workspace": workspace, "command": command, "args": args, "timeout": 120},
                    )
                    response.raise_for_status()
                    data = response.json()
            except Exception as exc:
                data = {"ok": False, "content": f"Check unavailable: {exc}"}
        duration_ms = int((time.monotonic() - started) * 1000)
        output = data.get("content", "")
        exit_code = 0 if data.get("ok") else 1
        if output.startswith("exit="):
            try:
                exit_code = int(output.split("\n", 1)[0].split("=", 1)[1])
            except (ValueError, IndexError):
                pass
        row_id = record_check_evidence(
            run_id=run_id,
            cycle=cycle,
            command=command,
            args=args,
            exit_code=exit_code,
            output=output,
            duration_ms=duration_ms,
            workspace_hash=workspace_hash,
            node_id=node_id,
        )
        evidence = {"id": row_id, "command": command, "args": args, "exit_code": exit_code, "duration_ms": duration_ms, "label": check.get("label", command)}
        results.append(evidence)
        db.add_event(
            run_id,
            "check.completed",
            f"Check {check.get('label', command)}: exit={exit_code} ({duration_ms}ms)",
            {"check": evidence, "node_id": node_id},
        )
    return results


def _topological_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id = {row["node_id"]: row for row in rows}
    remaining = {
        node_id: set(json.loads(row.get("depends_on_json") or "[]"))
        for node_id, row in by_id.items()
    }
    ordered: list[dict[str, Any]] = []
    while remaining:
        ready = sorted(node_id for node_id, deps in remaining.items() if not deps)
        if not ready:
            raise RuntimeError("Persisted task graph contains a dependency cycle")
        for node_id in ready:
            ordered.append(by_id[node_id])
            del remaining[node_id]
            for deps in remaining.values():
                deps.discard(node_id)
    return ordered


def _ancestor_rows(node: dict[str, Any], rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    dep_map = {
        row["node_id"]: json.loads(row.get("depends_on_json") or "[]")
        for row in rows
    }
    wanted = _transitive_deps(node["node_id"], dep_map)
    return [row for row in _topological_rows(rows) if row["node_id"] in wanted]


def _dependency_context(node: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    direct = set(json.loads(node.get("depends_on_json") or "[]"))
    parts: list[str] = []
    for row in _topological_rows(rows):
        if row["node_id"] not in direct:
            continue
        handoff = json.loads(row.get("handoff_json") or "{}")
        parts.append(
            f"DEPENDENCY {row['node_id']} — {row['title']}\n"
            f"{json.dumps(handoff, ensure_ascii=False)}"
        )
    return "\n\n".join(parts)[:16_000]


def _run_subtask(run_id: str, node: dict[str, Any], base_stage: Path) -> dict[str, Any]:
    """Execute one subtask in its own isolated worktree. Runs on a pool thread."""
    subtask_id = node["id"]
    if (db.one("select status from runs where id=?", (run_id,)) or {}).get("status") == "cancelled":
        raise RuntimeError("Run cancelled")
    rows = db.subtasks(run_id)
    agent = db.one("select * from agent_profiles where id=? and enabled=1", (node.get("agent_id"),)) if node.get("agent_id") else None
    if not _agent_is_eligible(agent, node.get("role") or "implementation"):
        agent = choose_subtask_agent(node, run_id=run_id)
    ancestors = _ancestor_rows(node, rows)
    worktree, input_manifest = stage_subtask_with_ancestors(run_id, node["node_id"], base_stage, ancestors)
    db.update_subtask(
        subtask_id,
        status="running",
        agent_id=agent["id"],
        worktree_ref=str(worktree),
        input_manifest_json=json.dumps(input_manifest, ensure_ascii=False),
        delta_manifest_json=None,
        blocked_reason=None,
        attempts=int(node.get("attempts") or 0) + 1,
    )
    db.add_event(run_id, "subtask.started", f"{node['title']} started with {agent['name']}", {"node_id": node["node_id"]})
    globs = json.loads(node.get("file_globs_json") or "[]")
    dependency_context = _dependency_context(node, rows)
    task = (
        "Complete ONLY this subtask. Respect role and file scope.\n\n"
        f"SUBTASK: {node['title']}\n\nSPEC:\n{node['spec']}\n\n"
        + (f"FILE SCOPE: {', '.join(globs)}\n\n" if globs else "")
        + (f"ACCEPTANCE CRITERIA:\n{node['acceptance_criteria']}\n" if node.get("acceptance_criteria") else "")
        + (f"\nDEPENDENCY HANDOFFS:\n{dependency_context}\n" if dependency_context else "")
    )
    # Decouple execution-mode from the scoring-role. A node that owns a file scope, or is an
    # implementation node, is expected to mutate its worktree, so it must run with write tools
    # (write tools are stripped unless mode == "implementation"). A research/verification node
    # with no file scope is genuinely read-only (analysis/handoff) and stays read-only. This
    # fixes silent no-op subtasks where a mutating node was labeled research/verification and
    # ran without write tools yet was still marked done.
    role = node.get("role") or "implementation"
    writes_expected = role == "implementation" or bool(globs)
    mode = "implementation" if writes_expected else role
    result = worker_call(
        run_id, agent["model"], mode, agent_task(agent, task),
        workspace=f"subtasks/{node['node_id']}/workspace", max_turns=32,
        node_id=node["node_id"], host_id=agent.get("host_id"),
    )
    record_usage(run_id, result)
    db.add_subtask_result(subtask_id, run_id, "implementation", result.get("content", ""))
    if not result.get("ok"):
        db.update_subtask(subtask_id, status="failed", result_summary=(result.get("error") or "failed")[:2000])
        raise RuntimeError(f"Subtask {node['node_id']} failed: {result.get('error') or 'worker error'}")
    impl_summary = result.get("content", "")[:4000]
    output_manifest = workspace_manifest(worktree)
    delta = workspace_delta(input_manifest, output_manifest)
    changed_files = sorted(set(delta["changed"]) | set(delta["deleted"]))
    if writes_expected and not changed_files:
        db.update_subtask(subtask_id, status="failed", result_summary="No file changes produced (empty delta)")
        db.add_event(run_id, "subtask.noop", f"{node['title']} produced no file changes", {"node_id": node["node_id"]})
        raise RuntimeError(f"Subtask {node['node_id']} produced no file changes")
    checks = [
        {
            "command": event.get("args", {}).get("command"),
            "args": event.get("args", {}).get("args", []),
            "result": str(event.get("result") or "")[:4000],
        }
        for event in (result.get("events") or [])
        if event.get("type") == "tool" and event.get("name") == "run_check"
    ]
    db.add_subtask_result(subtask_id, run_id, "checks", json.dumps(checks, ensure_ascii=False))
    criteria = node.get("acceptance_criteria") or ""
    run = db.one("select * from runs where id=?", (run_id,)) or {}
    provider = run.get("brain_provider") or "codex"
    verify_prompt = build_subtask_verify_prompt(node["title"], criteria, impl_summary, changed_files, checks)
    verify_result = call_brain_with_memory(
        run_id, provider, f"subtask_complete:{node['node_id']}", verify_prompt,
    )
    record_usage(run_id, verify_result, "brain")
    sv = parse_subtask_verdict(verify_result["content"])
    handoff = sv.handoff or {}
    db.add_subtask_result(subtask_id, run_id, "brain_handoff", json.dumps(handoff, ensure_ascii=False))
    db.add_event(run_id, "subtask.verified", f"{node['title']}: {'passed' if sv.passed else 'FAILED'}", {"node_id": node["node_id"], "passed": sv.passed, "issues": sv.issues, "handoff": handoff})
    if not sv.passed:
        db.update_subtask(subtask_id, status="failed", result_summary=f"Verification failed: {sv.issues}"[:2000])
        raise RuntimeError(f"Subtask {node['node_id']} failed verification: {sv.issues}")
    db.update_subtask(
        subtask_id,
        status="done",
        result_summary=impl_summary,
        verdict=json.dumps({"passed": True, "issues": sv.issues}, ensure_ascii=False),
        handoff_json=json.dumps(handoff, ensure_ascii=False),
        output_manifest_json=json.dumps(output_manifest, ensure_ascii=False),
        delta_manifest_json=json.dumps(delta, ensure_ascii=False),
    )
    db.add_event(run_id, "subtask.completed", f"{node['title']} complete", {"node_id": node["node_id"]})
    return node


def _transitive_deps(node_id: str, dep_map: dict[str, list[str]], seen: set[str] | None = None) -> set[str]:
    """Return all transitive dependencies for a node."""
    if seen is None:
        seen = set()
    for dep in dep_map.get(node_id, []):
        if dep not in seen:
            seen.add(dep)
            _transitive_deps(dep, dep_map, seen)
    return seen


def _enqueue_ready_subtasks(run_id: str) -> None:
    with _run_lock:
        _enqueue_ready_subtasks_locked(run_id)


def _enqueue_ready_subtasks_locked(run_id: str) -> None:
    """Dispatch ready nodes with dependency, pool, and per-agent exclusivity."""
    nodes = db.subtasks(run_id)
    if not nodes:
        return
    status_map = {n["node_id"]: n["status"] for n in nodes}
    dep_map = {n["node_id"]: json.loads(n.get("depends_on_json") or "[]") for n in nodes}
    failed_nodes = {nid for nid, status in status_map.items() if status in {"failed", "blocked"}}
    for node in nodes:
        if node["status"] != "pending":
            continue
        failed_deps = _transitive_deps(node["node_id"], dep_map) & failed_nodes
        if failed_deps:
            reason = "Blocked by failed dependency: " + ", ".join(sorted(failed_deps))
            db.update_subtask(node["id"], status="blocked", blocked_reason=reason)
            db.add_event(
                run_id,
                "subtask.blocked",
                f"{node['title']} blocked",
                {"node_id": node["node_id"], "reason": reason},
            )
    nodes = db.subtasks(run_id)
    status_map = {n["node_id"]: n["status"] for n in nodes}
    if nodes and all(status == "done" for status in status_map.values()):
        enqueue_job(run_id, "merge")
        return
    if all(status in {"done", "failed", "blocked"} for status in status_map.values()):
        failures = [n for n in nodes if n["status"] in {"failed", "blocked"}]
        message = "; ".join(
            f"{n['node_id']}: {n.get('result_summary') or n.get('blocked_reason') or n['status']}"
            for n in failures
        )
        update_run(run_id, status="failed", error=f"Task graph failed closed: {message}"[:4000])
        db.add_event(run_id, "run.failed", "Task graph failed closed; no changes applied")
        return
    active = [n for n in nodes if n["status"] in {"running", "waiting_ollama"}]
    reserved_agents = {
        n["agent_id"]
        for n in nodes
        if n.get("agent_id") and n["status"] in {"pending", "running", "waiting_ollama"}
    }
    reserved_agents.update(
        row["agent_id"]
        for row in db.all_rows(
            "select distinct s.agent_id from subtasks s join jobs j "
            "on j.run_id=s.run_id and j.node_id=s.node_id "
            "where s.agent_id is not null and s.status in ('pending','running','waiting_ollama') "
            "and j.status in ('pending','running','waiting_ollama')"
        )
    )
    slots = max(0, max(1, settings.worker_pool_size) - len(active))
    for node in nodes:
        if slots <= 0 or node["status"] != "pending":
            continue
        deps = dep_map[node["node_id"]]
        if not all(status_map.get(dep) == "done" for dep in deps):
            continue
        pinned = node.get("assigned_agent_id")
        if pinned and pinned in reserved_agents:
            continue
        eligible_all = [
            row
            for row in db.all_rows("select * from agent_profiles where enabled=1")
            if _agent_is_eligible(row, node.get("role") or "implementation")
        ]
        if pinned and not any(row["id"] == pinned for row in eligible_all):
            reason = f"Pinned agent {pinned} is unavailable for role '{node.get('role') or 'implementation'}'"
            db.update_subtask(node["id"], status="failed", result_summary=reason)
            db.add_event(run_id, "subtask.failed", reason, {"node_id": node["node_id"]})
            _enqueue_ready_subtasks(run_id)
            return
        if not eligible_all:
            reason = f"No enabled {node.get('role') or 'implementation'} agent available"
            db.update_subtask(node["id"], status="failed", result_summary=reason)
            db.add_event(run_id, "subtask.failed", reason, {"node_id": node["node_id"]})
            _enqueue_ready_subtasks(run_id)
            return
        eligible = [row for row in eligible_all if row["id"] not in reserved_agents]
        if not eligible:
            continue
        agent = choose_subtask_agent(node, reserved_agents, run_id=run_id)
        db.update_subtask(node["id"], agent_id=agent["id"], blocked_reason=None)
        reserved_agents.add(agent["id"])
        slots -= 1
        enqueue_job(run_id, "subtask", node_id=node["node_id"])


def _run_durable_subtask(run_id: str, job: dict[str, Any]) -> None:
    """Execute one subtask node via the durable job queue, then chain next ready nodes."""
    run_status = (db.one("select status from runs where id=?", (run_id,)) or {}).get("status")
    if run_status in ("cancelled", "failed"):
        return
    node_id = job["node_id"]
    nodes = db.subtasks(run_id)
    node = next((n for n in nodes if n["node_id"] == node_id), None)
    if not node:
        raise RuntimeError(f"Subtask node {node_id} not found")
    if node["status"] == "done":
        _enqueue_ready_subtasks(run_id)
        return
    base_stage = settings.jobs_dir / run_id / "workspace"
    max_attempts = db.get_setting_int("subtask_max_attempts") or 2
    try:
        _run_subtask(run_id, node, base_stage)
        _enqueue_ready_subtasks(run_id)
    except OllamaUnavailable:
        raise
    except Exception as exc:
        current_attempts = int(node.get("attempts") or 0) + 1
        log.warning("subtask_failed", extra={"run_id": run_id, "node_id": node_id, "attempt": current_attempts, "max": max_attempts})
        if current_attempts < max_attempts:
            db.update_subtask(node["id"], status="pending")
            db.add_event(run_id, "subtask.retry", f"{node['title']} retry {current_attempts}/{max_attempts}", {"node_id": node_id, "attempt": current_attempts, "error": str(exc)[:500]})
            now = db.utcnow()
            db.execute(
                "insert into jobs(run_id,job_type,node_id,status,created_at,updated_at) values(?,?,?,'pending',?,?)",
                (run_id, "subtask", node_id, now, now),
            )
            start_job_queue()
            return
        db.update_subtask(node["id"], status="failed", result_summary=str(exc)[:2000])
        db.add_event(run_id, "subtask.failed", f"{node['title']} failed after {current_attempts} attempt(s)", {"node_id": node_id, "error": str(exc)[:1000]})
        _enqueue_ready_subtasks(run_id)


def _verify_and_apply(run_id: str, summary: str, implementer_ids: set[str], repair_agent: dict[str, Any]) -> None:
    """Verification loop + apply + post-check. Shared by single-agent and multi-agent paths."""
    update_run(run_id, status="verifying")
    for repair in range(3):
        run = db.one("select * from runs where id=?", (run_id,)) or {}
        first = choose_agent("verification", set(implementer_ids))
        verifiers = [first]
        try:
            # A genuinely independent second verifier — a different agent (now possibly on a
            # different host). If none exists, run one verifier; the brain verdict below is the
            # independent second signal. (Previously second=first ran the same agent twice.)
            verifiers.append(choose_agent("verification", implementer_ids | {first["id"]}))
        except RuntimeError:
            db.add_event(run_id, "verification.single", "Only one independent verifier available; brain verdict is the second signal", {"repair": repair})
        reports = []
        verifier_ids = []
        for verifier in verifiers:
            result = worker_call(run_id, verifier["model"], "verification", agent_task(verifier, verification_prompt(run, summary)), max_turns=18, host_id=verifier.get("host_id"))
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
            revised_plan = (run.get("approved_plan") or "") + "\n\n## Requested Scope Expansion\n\n" + expansion
            update_run(
                run_id,
                status="decomposing",
                draft_plan=revised_plan,
                graph_plan_hash=None,
                plan_state="editing",
                error="Scope expansion requires approval",
            )
            db.execute("delete from subtasks where run_id=?", (run_id,))
            db.add_event(run_id, "scope.approval_required", "Repair exceeds approved scope; graph regenerating")
            enqueue_job(run_id, "decompose")
            return
        if repair >= 2:
            update_run(run_id, status="failed", error="Verification failed after two repair cycles")
            return
        repair_task = parsed.get("repair_task") or "Fix every defect in these verifier reports:\n" + "\n\n".join(reports)
        update_run(run_id, status="implementing", repair_count=repair + 1)
        repair_result = worker_call(run_id, repair_agent["model"], "implementation", agent_task(repair_agent, repair_task), max_turns=24, host_id=repair_agent.get("host_id"))
        record_usage(run_id, repair_result)
        if not repair_result.get("ok"):
            raise RuntimeError(repair_result.get("error") or "Repair agent failed")
        summary = repair_result.get("content", "")
        save_artifact(run_id, "repair", f"Repair cycle {repair + 1}", repair_result)
        update_run(run_id, status="verifying")
    if (db.one("select status from runs where id=?", (run_id,)) or {}).get("status") == "cancelled":
        return
    # --- Deterministic evidence gate (Phase 0A) ---
    _run_qualifying_checks(run_id, cycle=repair)
    run = db.one("select * from runs where id=?", (run_id,)) or {}
    last_verdict = run.get("verdict") or "{}"
    try:
        brain_passed = json.loads(last_verdict).get("passed", False)
    except (json.JSONDecodeError, AttributeError):
        brain_passed = False
    gate = evaluate_apply_gate(run_id, brain_passed)
    save_artifact(run_id, "apply_gate", "Verification policy decision", gate.to_dict())
    db.add_event(run_id, "gate.evaluated", f"Apply gate: {'ALLOWED' if gate.allowed else 'BLOCKED'}", gate.to_dict())
    if not gate.allowed:
        reasons = ", ".join(gate.reasons)
        update_run(run_id, status="failed", error=f"Apply gate blocked: {reasons}")
        db.add_event(run_id, "run.failed", f"Deterministic verification gate blocked apply: {reasons}", gate.to_dict())
        return
    # --- End evidence gate ---
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
        agent_task(verifier, "Post-apply check. Verify copied final server state still satisfies approved plan. Start with PASS or FAIL.\n\n" + (run.get("approved_plan") or "")),
        workspace="postcheck", max_turns=18, host_id=verifier.get("host_id"),
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
    with _brain_locks_guard:
        _brain_locks.pop(run_id, None)


def _merge_and_verify(run_id: str) -> None:
    """Merge subtask worktrees then run verification+apply. Called by merge job."""
    try:
        run = db.one("select * from runs where id=?", (run_id,))
        if not run or run["status"] in ("cancelled", "failed"):
            return
        base_stage = settings.jobs_dir / run_id / "workspace"
        nodes = _topological_rows(db.subtasks(run_id))
        if not nodes or any(node["status"] != "done" for node in nodes):
            raise RuntimeError("Task graph is incomplete; fail-closed policy prevents merge")
        merged = merge_task_lineage(base_stage, run_id, nodes)
        save_artifact(run_id, "merge", "Merged subtask manifest", merged)
        db.add_event(run_id, "subtasks.merged", f"Merged {merged['subtasks']} task delta(s)", merged)
        implementer_ids = {n["agent_id"] for n in nodes if n.get("agent_id")}
        summary = "\n\n".join(f"### {n['title']}\n{n.get('result_summary') or ''}" for n in nodes)
        repair_agent = choose_agent("implementation")
        _verify_and_apply(run_id, summary, implementer_ids, repair_agent)
    except OllamaUnavailable:
        raise
    except MergeConflict as exc:
        log.warning("merge_conflict", extra={"run_id": run_id})
        update_run(run_id, status="failed", error=str(exc)[:4000])
        db.add_event(run_id, "subtasks.conflict", "Subtask worktrees conflict; re-plan with disjoint scopes", {"conflicts": exc.conflicts})
    except Exception as exc:
        log.exception("merge_verify_failed", extra={"run_id": run_id})
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
        db.add_event(run_id, "run.failed", "Merge/verification workflow failed", {"error": str(exc)[:1000]})


def implement_run(run_id: str) -> None:
    try:
        run = db.one("select * from runs where id=?", (run_id,))
        if not run or run["status"] == "cancelled":
            return
        if not ollama_available():
            raise OllamaUnavailable("Ollama offline")
        subtask_nodes = db.subtasks(run_id)
        if subtask_nodes:
            # Multi-agent path: dispatch durable subtask jobs and return.
            # Completion is driven by _run_durable_subtask chaining → merge job.
            pending = [n for n in subtask_nodes if n["status"] not in ("done", "running")]
            db.add_event(run_id, "implementation.started", f"Dispatching {len(pending)} subtask(s) via durable queue")
            _enqueue_ready_subtasks(run_id)
            return
        # Single-agent path (backward compatible): one implementer runs the whole plan.
        agent = db.one("select * from agent_profiles where id=?", (run["implementation_agent_id"],))
        if not agent:
            raise RuntimeError("Implementation agent missing")
        db.add_event(run_id, "implementation.started", f"Implementation started with {agent['name']}")
        task = "Implement this approved plan completely.\n\n" + (run["approved_plan"] or "")
        implementation = worker_call(run_id, agent["model"], "implementation", agent_task(agent, task), max_turns=32, host_id=agent.get("host_id"))
        record_usage(run_id, implementation)
        summary = implementation.get("content", "")
        save_artifact(run_id, "implementation", "Implementation transcript", implementation)
        if not implementation.get("ok"):
            raise RuntimeError(implementation.get("error") or "Implementation agent failed")
        if (db.one("select status from runs where id=?", (run_id,)) or {}).get("status") == "cancelled":
            return
        _verify_and_apply(run_id, summary, {agent["id"]}, agent)
    except OllamaUnavailable:
        raise
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
    _subtask_assignment_counts.pop(run_id, None)


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
