import contextlib
import json
import logging
import os
import secrets
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx

from . import db
from .config import settings
from .run_state import validate_transition
from .security import decrypt_secret
from .workspace import apply_stage, create_snapshot, discard_snapshot, manifest_hash, restore_snapshot, stage_workspace

log = logging.getLogger("ollma.orchestrator")
executor = ThreadPoolExecutor(max_workers=settings.runner_concurrency, thread_name_prefix="ollma-runner")
_run_lock = threading.Lock()
_queue_lock = threading.Lock()
_active_drainers = 0


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
            try:
                if job["job_type"] == "research":
                    research_run(job["run_id"])
                else:
                    implement_run(job["run_id"])
                now = db.utcnow()
                db.execute(
                    "update jobs set status='done',completed_at=?,lease_owner=null,lease_expires_at=null,updated_at=? where id=?",
                    (now, now, job["id"]),
                )
            except Exception as exc:
                log.exception("job_failed", extra={"job_id": job["id"], "run_id": job["run_id"]})
                db.execute(
                    "update jobs set status='failed',error=?,completed_at=?,lease_owner=null,lease_expires_at=null,updated_at=? where id=?",
                    (str(exc)[:4000], db.utcnow(), db.utcnow(), job["id"]),
                )
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
    elif provider == "codex":
        key = settings.openai_key
    else:
        key = settings.claude_key
    if not key:
        raise RuntimeError(f"{provider} API key is missing")
    return key, row["model"]


def call_brain(provider: str, prompt: str, allow_web: bool = False, timeout: int = 900) -> str:
    key, model = provider_config(provider)
    payload = {"provider": provider, "api_key": key, "model": model, "prompt": prompt, "allow_web": allow_web}
    env = {
        "PATH": os.getenv("PATH", "/usr/local/bin:/usr/bin:/bin"),
        "PYTHONPATH": str(Path(__file__).resolve().parents[1]),
        "HOME": str(settings.data_dir / "provider-home" / provider),
        "LANG": "C.UTF-8",
    }
    Path(env["HOME"]).mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [sys.executable, "-m", "backend.brain_runner"],
        input=json.dumps(payload), text=True, capture_output=True, timeout=timeout, env=env,
    )
    try:
        data = json.loads(result.stdout.strip().splitlines()[-1])
    except Exception as exc:
        raise RuntimeError((result.stderr or result.stdout or "Brain returned invalid output")[-4000:]) from exc
    if result.returncode != 0 or not data.get("ok"):
        raise RuntimeError(data.get("error") or "Brain failed")
    return str(data.get("content", ""))


def worker_call(run_id: str, model: str, mode: str, task: str, workspace: str = "workspace", max_turns: int = 24) -> dict[str, Any]:
    attempts = 1 if mode == "implementation" else 3
    for attempt in range(attempts):
        try:
            with httpx.Client(timeout=1200) as client:
                response = client.post(
                    settings.worker_url + "/execute",
                    headers={"X-Worker-Token": settings.worker_token},
                    json={"run_id": run_id, "workspace": workspace, "model": model, "mode": mode, "task": task, "max_turns": max_turns},
                )
                response.raise_for_status()
                return response.json()
        except (httpx.ConnectError, httpx.RemoteProtocolError, httpx.HTTPStatusError):
            if attempt + 1 >= attempts:
                raise
            time.sleep(1 << attempt)
    raise RuntimeError("Worker unavailable")


def discover_agents() -> list[dict[str, Any]]:
    now = db.utcnow()
    with httpx.Client(timeout=30) as client:
        tags = client.get(settings.ollama_url + "/api/tags").json().get("models", [])
        discovered: list[dict[str, Any]] = []
        for item in tags:
            model = item.get("name") or item.get("model")
            if not model:
                continue
            show = client.post(settings.ollama_url + "/api/show", json={"model": model}).json()
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


def choose_agent(role: str, exclude: set[str] | None = None, discover: bool = True) -> dict[str, Any]:
    exclude = exclude or set()
    candidates = []
    for row in db.all_rows("select * from agent_profiles where enabled=1"):
        roles = json.loads(row["roles_json"] or "[]")
        capabilities = json.loads(row["capabilities_json"] or "[]")
        if row["id"] in exclude or role not in roles:
            continue
        if role == "implementation" and "tools" not in capabilities:
            continue
        scores = json.loads(row["role_scores_json"] or "{}")
        candidates.append((int(scores.get(role, 0)), int(row["priority"]), int(row["context_size"]), row["name"], row))
    if not candidates:
        if discover:
            discover_agents()
            return choose_agent(role, exclude, False)
        raise RuntimeError(f"No eligible {role} agent available")
    candidates.sort(key=lambda item: (-item[0], -item[1], -item[2], item[3].lower()))
    return candidates[0][-1]


def save_artifact(run_id: str, kind: str, name: str, content: Any) -> None:
    serialized = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
    db.execute(
        "insert into run_artifacts(run_id,kind,name,content,created_at) values(?,?,?,?,?)",
        (run_id, kind, name, serialized[:500_000], db.utcnow()),
    )


def record_usage(run_id: str, result: dict[str, Any]) -> None:
    usage = result.get("usage") or {}
    if not usage:
        return
    run = db.one("select usage_json from runs where id=?", (run_id,)) or {}
    current = json.loads(run.get("usage_json") or "{}")
    for key, value in usage.items():
        if isinstance(value, (int, float)):
            current[key] = current.get(key, 0) + value
    update_run(run_id, usage_json=json.dumps(current, ensure_ascii=False))


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
    stage_workspace(run_id, target)
    baseline = manifest_hash(target)
    _, model = provider_config(provider)
    now = db.utcnow()
    with db.transaction() as conn:
        conn.execute(
            "insert into runs(id,task,brain_provider,brain_model,target_path,web_research,status,baseline_hash,created_at,updated_at) values(?,?,?,?,?,?,?,?,?,?)",
            (run_id, task, provider, model, str(target), int(web_research), "researching", baseline, now, now),
        )
        conn.execute(
            "insert into jobs(run_id,job_type,status,created_at,updated_at) values(?,'research','pending',?,?)",
            (run_id, now, now),
        )
    db.add_event(run_id, "run.created", "Research queued", {"provider": provider, "target": str(target)})
    start_job_queue()
    return db.one("select * from runs where id=?", (run_id,)) or {}


def research_run(run_id: str) -> None:
    try:
        run = db.one("select * from runs where id=?", (run_id,))
        if not run or run["status"] == "cancelled":
            return
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
        prompt = (
            "You are supervisor brain. Use worker dossier to write decision-complete Markdown implementation plan. "
            "Include architecture, exact behavior, interfaces, failure handling, tests, and acceptance criteria. "
            "Do not implement or edit files.\n\nUSER TASK:\n" + run["task"] + "\n\nWORKER DOSSIER:\n" + dossier
        )
        plan = call_brain(run["brain_provider"], prompt, bool(run["web_research"]))
        if (db.one("select status from runs where id=?", (run_id,)) or {}).get("status") == "cancelled":
            return
        implementation = choose_agent("implementation")
        update_run(
            run_id,
            dossier=dossier,
            draft_plan=plan,
            implementation_agent_id=implementation["id"],
            status="plan_ready",
        )
        save_artifact(run_id, "plan", "Supervisor plan", plan)
        db.add_event(run_id, "plan.ready", "Plan ready for approval")
        update_run(run_id, dossier=dossier, draft_plan=plan, status="awaiting_approval")
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
        implementation = db.one("select * from agent_profiles where id=? and enabled=1", (run.get("implementation_agent_id"),))
        if not implementation or "implementation" not in json.loads(implementation["roles_json"] or "[]"):
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
                    "insert into history_snippets(id,run_id,request,approved_plan,brain_provider,workers_json,target_path,snapshot_id,created_at) values(?,?,?,?,?,?,?,?,?)",
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
        db.add_event(run_id, "plan.approved", "History and snapshot created", {"snapshot_id": snapshot["id"]})
        start_job_queue()
        return db.one("select * from runs where id=?", (run_id,)) or {}


def verification_prompt(run: dict[str, Any], implementation_summary: str) -> str:
    return (
        "Verify approved implementation independently. Inspect staged code and run relevant safe checks. "
        "Start final answer with PASS or FAIL. Cite exact files and test output.\n\nAPPROVED PLAN:\n"
        + (run["approved_plan"] or "") + "\n\nIMPLEMENTER SUMMARY:\n" + implementation_summary
    )


def brain_verdict(run: dict[str, Any], reports: list[str]) -> tuple[bool, str]:
    prompt = (
        "Act as supervisor. Judge whether implementation satisfies approved plan using verifier reports. "
        "Return JSON only: {\"passed\":boolean,\"verdict\":string,\"repair_task\":string,\"scope_expansion\":boolean}. "
        "Set scope_expansion true when a required repair exceeds approved scope; otherwise repair_task must remain inside scope.\n\nPLAN:\n" + (run["approved_plan"] or "")
        + "\n\nREPORTS:\n" + "\n\n---\n\n".join(reports)
    )
    raw = call_brain(run["brain_provider"], prompt, False)
    cleaned = raw.strip().removeprefix("```json").removesuffix("```").strip()
    try:
        value = json.loads(cleaned)
        return bool(value.get("passed")), json.dumps(value, ensure_ascii=False)
    except json.JSONDecodeError:
        passed = "\"passed\":true" in raw.replace(" ", "").lower()
        return passed, raw


def implement_run(run_id: str) -> None:
    try:
        run = db.one("select * from runs where id=?", (run_id,))
        if not run or run["status"] == "cancelled":
            return
        agent = db.one("select * from agent_profiles where id=?", (run["implementation_agent_id"],))
        if not agent:
            raise RuntimeError("Implementation agent missing")
        db.add_event(run_id, "implementation.started", f"Implementation started with {agent['name']}")
        task = "Implement this approved plan completely.\n\n" + (run["approved_plan"] or "")
        implementation = worker_call(run_id, agent["model"], "implementation", agent_task(agent, task), max_turns=32)
        record_usage(run_id, implementation)
        if (db.one("select status from runs where id=?", (run_id,)) or {}).get("status") == "cancelled":
            return
        summary = implementation.get("content", "")
        save_artifact(run_id, "implementation", "Implementation transcript", implementation)
        if not implementation.get("ok"):
            raise RuntimeError(implementation.get("error") or "Implementation agent failed")
        update_run(run_id, status="verifying")
        for repair in range(3):
            run = db.one("select * from runs where id=?", (run_id,)) or run
            first = choose_agent("verification", {agent["id"]})
            try:
                second = choose_agent("verification", {agent["id"], first["id"]})
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
                db.execute(
                    "insert into verification_results(run_id,agent_id,cycle,report,passed,created_at) values(?,?,?,?,?,?)",
                    (run_id, verifier["id"], repair, result.get("content", ""), int(bool(result.get("ok"))), db.utcnow()),
                )
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
            repair_result = worker_call(run_id, agent["model"], "implementation", agent_task(agent, repair_task), max_turns=24)
            record_usage(run_id, repair_result)
            if not repair_result.get("ok"):
                raise RuntimeError(repair_result.get("error") or "Repair agent failed")
            summary = repair_result.get("content", "")
            save_artifact(run_id, "repair", f"Repair cycle {repair + 1}", repair_result)
            update_run(run_id, status="verifying")
        if (db.one("select status from runs where id=?", (run_id,)) or {}).get("status") == "cancelled":
            return
        update_run(run_id, status="applying")
        target = Path(run["target_path"])
        stage = settings.jobs_dir / run_id / "workspace"
        changes = apply_stage(target, stage)
        save_artifact(run_id, "changes", "Applied file manifest", changes)
        db.add_event(run_id, "apply.completed", "Verified changes applied", changes)
        update_run(run_id, status="post_check")
        stage_workspace(run_id, target, "postcheck")
        verifier = choose_agent("verification", {agent["id"]})
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
        raise RuntimeError("Finished run cannot be cancelled")
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
    if run["approved_plan"]:
        update_run(run_id, status="implementing", error=None)
        enqueue_job(run_id, "implementation")
    else:
        update_run(run_id, status="researching", error=None)
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
