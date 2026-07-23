import json
import sqlite3
from collections.abc import Iterator
from contextlib import closing, contextmanager
from datetime import UTC, datetime
from typing import Any

from .config import settings
from .migrations import apply_migrations

MAX_RUN_EVENTS = 500
MAX_RUN_ARTIFACTS = 200
MAX_TIMELINE_ENTRIES = 5000


def utcnow() -> str:
    return datetime.now(UTC).isoformat()


def connect() -> sqlite3.Connection:
    db = sqlite3.connect(settings.db_path, timeout=30, check_same_thread=False)
    db.row_factory = sqlite3.Row
    db.execute("pragma foreign_keys=on")
    db.execute("pragma busy_timeout=30000")
    return db


@contextmanager
def transaction() -> Iterator[sqlite3.Connection]:
    db = connect()
    try:
        db.execute("begin immediate")
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def execute(sql: str, args: tuple[Any, ...] = ()) -> None:
    with closing(connect()) as db:
        db.execute(sql, args)
        db.commit()


def one(sql: str, args: tuple[Any, ...] = ()) -> dict[str, Any] | None:
    with closing(connect()) as db:
        row = db.execute(sql, args).fetchone()
        return dict(row) if row else None


def all_rows(sql: str, args: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
    with closing(connect()) as db:
        return [dict(row) for row in db.execute(sql, args).fetchall()]


def retention_cap(key: str, fallback: int) -> int:
    """Configurable retention limit for the live SSE feed / prunable evidence tables.

    Reads an app_settings override; a non-positive/invalid value falls back to the
    historical default so a bad value can never wipe a table. These caps govern only
    prunable tables — durable history lives in append-only tables (plan_versions,
    subtask_results, verification_results) that are never pruned.
    """
    value = get_setting_int(key)
    return value if value > 0 else fallback


def add_event(run_id: str, event_type: str, message: str, data: Any = None) -> int:
    with closing(connect()) as db:
        cursor = db.execute(
            "insert into run_events(run_id,event_type,message,data_json,created_at) values(?,?,?,?,?)",
            (run_id, event_type, message, json.dumps(data, ensure_ascii=False) if data is not None else None, utcnow()),
        )
        db.execute(
            "delete from run_events where run_id=? and id not in "
            "(select id from run_events where run_id=? order by id desc limit ?)",
            (run_id, run_id, retention_cap("run_events_cap", MAX_RUN_EVENTS)),
        )
        db.commit()
        return int(cursor.lastrowid)


def add_plan_version(run_id: str, kind: str, content: str, brain_provider: str | None = None) -> int:
    """Append an immutable plan snapshot. kind: draft | edit | redo | approved.

    runs.draft_plan / approved_plan remain the 'current' pointer; this table is the
    never-pruned history that powers plan-diff views.
    """
    with closing(connect()) as db:
        version = int(
            (db.execute("select coalesce(max(version),0)+1 as v from plan_versions where run_id=?", (run_id,)).fetchone() or {"v": 1})["v"]
        )
        cursor = db.execute(
            "insert into plan_versions(run_id,version,kind,content,brain_provider,created_at) values(?,?,?,?,?,?)",
            (run_id, version, kind, content, brain_provider, utcnow()),
        )
        db.commit()
        return int(cursor.lastrowid)


def plan_versions(run_id: str) -> list[dict[str, Any]]:
    return all_rows("select * from plan_versions where run_id=? order by version", (run_id,))


def insert_subtasks(run_id: str, nodes: list[dict[str, Any]]) -> None:
    """Persist a validated task-graph (from plan_graph.validate_graph) for a run.

    Replaces any prior graph for the run so a re-decomposition is clean. node_id is
    unique per run; id is a stable surrogate key used by subtask_results/jobs.
    """
    now = utcnow()
    with transaction() as conn:
        conn.execute("delete from subtasks where run_id=?", (run_id,))
        for node in nodes:
            conn.execute(
                "insert into subtasks(id,run_id,node_id,title,spec,depends_on_json,file_globs_json,"
                "role,suggested_model,status,created_at,updated_at) values(?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    f"sub-{run_id}-{node['node_id']}",
                    run_id,
                    node["node_id"],
                    node["title"],
                    node["spec"],
                    json.dumps(node.get("depends_on") or [], ensure_ascii=False),
                    json.dumps(node.get("file_globs") or [], ensure_ascii=False),
                    node.get("role") or "implementation",
                    node.get("suggested_model"),
                    "pending",
                    now,
                    now,
                ),
            )


def subtasks(run_id: str) -> list[dict[str, Any]]:
    return all_rows("select * from subtasks where run_id=? order by id", (run_id,))


def update_subtask(subtask_id: str, **values: Any) -> None:
    if not values:
        return
    values["updated_at"] = utcnow()
    columns = ",".join(f"{key}=?" for key in values)
    execute(f"update subtasks set {columns} where id=?", (*values.values(), subtask_id))


def add_subtask_result(subtask_id: str, run_id: str, kind: str, content: str) -> None:
    """Append durable per-subtask evidence. Never pruned (unlike run_artifacts)."""
    execute(
        "insert into subtask_results(subtask_id,run_id,kind,content,created_at) values(?,?,?,?,?)",
        (subtask_id, run_id, kind, content, utcnow()),
    )


# ---------------------------------------------------------------------------
# Brain memory — per-run continuity across brain calls
# ---------------------------------------------------------------------------

def add_brain_memory(run_id: str, step: str, role: str, content: str) -> int:
    """Append a brain memory entry. seq auto-increments per run. Never pruned."""
    with closing(connect()) as db:
        seq = int(
            (db.execute("select coalesce(max(seq),0)+1 as s from brain_memory where run_id=?", (run_id,)).fetchone() or {"s": 1})["s"]
        )
        tokens_estimate = len(content) // 4
        cursor = db.execute(
            "insert into brain_memory(run_id,seq,step,role,content,tokens_estimate,created_at) values(?,?,?,?,?,?,?)",
            (run_id, seq, step, role, content, tokens_estimate, utcnow()),
        )
        db.commit()
        return int(cursor.lastrowid)


def brain_memory(run_id: str) -> list[dict[str, Any]]:
    """Retrieve all brain memory entries for a run, ordered by sequence."""
    return all_rows("select * from brain_memory where run_id=? order by seq", (run_id,))


DEFAULT_SETTINGS: dict[str, str] = {
    "token_budget_run": "0",      # per-run paid-token cap (0 = unlimited)
    "token_budget_daily": "0",    # per-day paid-token cap (0 = unlimited)
    "max_output_tokens": "0",     # cap on brain/API response length (0 = provider default)
    "theme": "dark",              # dark | light | auto
    "agent_mode_default": "0",    # chat Agent Mode default (0/1)
    "run_events_cap": "500",      # live run-event feed retention (default 500; raise to keep more)
    "run_artifacts_cap": "200",   # run-artifact retention (default 200)
    "timeline_cap": "5000",       # global timeline retention (default 5000)
    "brain_memory_budget": "4000", # per-run brain memory token budget (chars / 4)
}


def get_setting(key: str, default: str = "") -> str:
    row = one("select value from app_settings where key=?", (key,))
    if row:
        return str(row["value"])
    return DEFAULT_SETTINGS.get(key, default)


def get_setting_int(key: str) -> int:
    try:
        return int(get_setting(key, "0"))
    except (TypeError, ValueError):
        return 0


def all_settings() -> dict[str, str]:
    stored = {row["key"]: row["value"] for row in all_rows("select key,value from app_settings")}
    return {**DEFAULT_SETTINGS, **stored}


def set_setting(key: str, value: Any) -> None:
    execute(
        "insert into app_settings(key,value,updated_at) values(?,?,?) "
        "on conflict(key) do update set value=excluded.value,updated_at=excluded.updated_at",
        (key, str(value), utcnow()),
    )


def record_ledger(source: str, provider: str, input_tokens: int, output_tokens: int, total: int) -> None:
    execute(
        "insert into usage_ledger(day,source,provider,input,output,total,created_at) values(?,?,?,?,?,?,?)",
        (datetime.now(UTC).date().isoformat(), source, provider, int(input_tokens), int(output_tokens), int(total), utcnow()),
    )


def ledger_totals_today(paid_only: bool = True) -> dict[str, int]:
    day = datetime.now(UTC).date().isoformat()
    clause = " and source!='ollama'" if paid_only else ""
    row = one(
        "select coalesce(sum(input),0) as i, coalesce(sum(output),0) as o, coalesce(sum(total),0) as t "
        f"from usage_ledger where day=?{clause}",
        (day,),
    ) or {"i": 0, "o": 0, "t": 0}
    return {"input": int(row["i"]), "output": int(row["o"]), "total": int(row["t"])}


def ledger_by_provider_today() -> list[dict[str, Any]]:
    day = datetime.now(UTC).date().isoformat()
    return all_rows(
        "select provider, source, coalesce(sum(total),0) as total from usage_ledger "
        "where day=? group by provider, source order by total desc",
        (day,),
    )


def init_db() -> None:
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.snapshot_dir.mkdir(parents=True, exist_ok=True)
    settings.jobs_dir.mkdir(parents=True, exist_ok=True)
    with closing(connect()) as db:
        db.execute("pragma journal_mode=wal")
        db.executescript(
            """
            create table if not exists chats (
              id text primary key, title text not null, model text, target_path text,
              created_at text not null, updated_at text not null
            );
            create table if not exists messages (
              id integer primary key autoincrement, chat_id text not null, role text not null,
              content text not null, created_at text not null,
              foreign key(chat_id) references chats(id) on delete cascade
            );
            create table if not exists snapshots (
              id text primary key, chat_id text, path text not null, kind text not null,
              ref text not null, created_at text not null, archive_deleted_at text
            );
            create table if not exists timeline (
              id integer primary key autoincrement, chat_id text, event_type text not null,
              path text, summary text not null, before text, after text, diff text,
              created_at text not null
            );
            create table if not exists sessions (
              id text primary key, user text not null, csrf text not null,
              created_at real not null, last_seen real not null, expires_at real not null
            );
            create table if not exists brain_configs (
              provider text primary key check(provider in ('codex','claude','gemini')),
              model text not null, key_ciphertext text, source text not null default 'environment',
              enabled integer not null default 0, validated_at text, last_error text,
              updated_at text not null
            );
            create table if not exists agent_profiles (
              id text primary key, name text not null, model text not null unique,
              roles_json text not null, system_prompt text not null default '',
              capabilities_json text not null default '[]', context_size integer not null default 0,
              priority integer not null default 50, role_scores_json text not null default '{}',
              enabled integer not null default 1, discovered_at text not null, updated_at text not null
            );
            create table if not exists runs (
              id text primary key, task text not null, brain_provider text not null,
              brain_model text, target_path text not null, web_research integer not null default 0,
              status text not null, baseline_hash text, research_agent_id text,
              implementation_agent_id text, verification_agent_ids_json text not null default '[]',
              dossier text, draft_plan text, approved_plan text, snapshot_id text,
              repair_count integer not null default 0, verdict text, error text,
              created_at text not null, updated_at text not null, approved_at text, completed_at text
            );
            create table if not exists run_events (
              id integer primary key autoincrement, run_id text not null, event_type text not null,
              message text not null, data_json text, created_at text not null,
              foreign key(run_id) references runs(id) on delete cascade
            );
            create table if not exists run_artifacts (
              id integer primary key autoincrement, run_id text not null, kind text not null,
              name text not null, content text not null, created_at text not null,
              foreign key(run_id) references runs(id) on delete cascade
            );
            create table if not exists history_snippets (
              id text primary key, run_id text not null unique, request text not null,
              approved_plan text not null, brain_provider text not null,
              workers_json text not null, target_path text not null, snapshot_id text not null,
              final_verdict text, created_at text not null, completed_at text,
              foreign key(run_id) references runs(id) on delete cascade
            );
            create table if not exists run_approvals (
              id integer primary key autoincrement, run_id text not null,
              approved_plan text not null, snapshot_id text not null, created_at text not null,
              foreign key(run_id) references runs(id) on delete cascade
            );
            create table if not exists verification_results (
              id integer primary key autoincrement, run_id text not null,
              agent_id text, cycle integer not null, report text not null,
              passed integer, created_at text not null,
              foreign key(run_id) references runs(id) on delete cascade
            );
            create table if not exists jobs (
              id integer primary key autoincrement, run_id text not null,
              job_type text not null check(job_type in ('research','implementation')),
              status text not null default 'pending', attempts integer not null default 0,
              error text, created_at text not null, updated_at text not null,
              foreign key(run_id) references runs(id) on delete cascade
            );
            create index if not exists idx_run_events_run_id_id on run_events(run_id,id);
            create index if not exists idx_runs_updated_at on runs(updated_at desc);
            create index if not exists idx_jobs_status_id on jobs(status,id);
            """
        )
        apply_migrations(db)
        db.execute(
            "insert or ignore into brain_configs(provider,model,source,enabled,updated_at) values(?,?,?,?,?)",
            ("codex", settings.openai_model, "environment", int(bool(settings.openai_key)), utcnow()),
        )
        db.execute(
            "insert or ignore into brain_configs(provider,model,source,enabled,updated_at) values(?,?,?,?,?)",
            ("claude", settings.claude_model, "environment", int(bool(settings.claude_key)), utcnow()),
        )
        db.execute(
            "insert or ignore into brain_configs(provider,model,source,enabled,updated_at) values(?,?,?,?,?)",
            ("gemini", settings.gemini_model, "environment", int(bool(settings.gemini_key)), utcnow()),
        )
        db.execute(
            "update jobs set status='pending',error='Interrupted; queued for recovery',lease_owner=null,"
            "lease_expires_at=null,updated_at=? where status='running'",
            (utcnow(),),
        )
        db.execute(
            "update runs set status='failed', error=coalesce(error,'Interrupted by service restart'), updated_at=? "
            "where status in ('applying','post_check')",
            (utcnow(),),
        )
        db.execute(
            "update jobs set status='failed',error='Interrupted during apply; manual resume required',updated_at=? "
            "where status='pending' and run_id in (select id from runs where status='failed' and error='Interrupted by service restart')",
            (utcnow(),),
        )
        db.commit()
